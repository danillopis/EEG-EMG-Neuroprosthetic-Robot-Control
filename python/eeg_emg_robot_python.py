import time
import pickle
from pathlib import Path
from collections import deque, Counter

import numpy as np
import pandas as pd
import serial
import matplotlib.pyplot as plt

from pylsl import resolve_streams, StreamInlet
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# CONFIGURACIÓN GENERAL

ARDUINO_PORT = "COM17"
ARDUINO_BAUD = 115200

MODEL_FILE = Path("GR01-D3_eeg_csp_model.pkl")
DATA_DIR = Path("eeg_recordings")
PLOT_DIR = Path("eeg_plots")

DATA_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

FS = 256

EXPECTED_CHANNELS = ["FC3", "FC4", "C3", "CZ", "C4", "CP3", "PZ", "CP4"]
CHANNELS_CSP = ["FC3", "FC4", "C3", "CZ", "C4"]

CLASS_REST = 0
CLASS_LEFT = 1
CLASS_RIGHT = 2

CLASS_NAMES = {
    CLASS_REST: "REST",
    CLASS_LEFT: "LEFT",
    CLASS_RIGHT: "RIGHT",
}

COMMAND_LEFT = b"L\n"
COMMAND_RIGHT = b"R\n"

# Grabación
N_TRIALS_PER_CLASS = 30
CUE_SECONDS = 1.0
MI_SECONDS = 4.0
REST_SECONDS = 4.0

# Ventana online
ONLINE_WINDOW_SECONDS = 2.5
ONLINE_STEP_SECONDS = 0.5

# Decisión online
PROBA_THRESHOLD_MI = 0.55
PROBA_THRESHOLD_LR = 0.55
MIN_MARGIN_LR = 0.08
VOTE_HISTORY_SIZE = 5
MIN_VOTES_TO_SEND = 3

# Filtros
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0
HIGHPASS_CUTOFF = 1.0
BANDPASS_LOW = 8.0
BANDPASS_HIGH = 30.0

# FILTROS EEG

def highpass_filter(data, fs, cutoff=1.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="high")
    return filtfilt(b, a, data, axis=0)


def notch_filter(data, fs, freq=50.0, q=30.0):
    b, a = iirnotch(freq / (fs / 2), q)
    return filtfilt(b, a, data, axis=0)


def bandpass_filter(data, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data, axis=0)


def common_average_reference(data):
    return data - np.mean(data, axis=1, keepdims=True)


def preprocess_eeg(data):
    data = highpass_filter(data, FS, HIGHPASS_CUTOFF)
    data = notch_filter(data, FS, NOTCH_FREQ, NOTCH_Q)
    data = common_average_reference(data)
    data = bandpass_filter(data, FS, BANDPASS_LOW, BANDPASS_HIGH)
    return data

# LSL

def find_eeg_stream():
    print("Buscando stream EEG por LSL...")

    streams = resolve_streams(wait_time=5.0)
    eeg_streams = [s for s in streams if s.type().lower() == "eeg"]

    if not eeg_streams:
        raise RuntimeError("No se encontró stream EEG. Revisa SennsLite, Bluetooth y LSL.")

    print("\nStreams EEG encontrados:")
    for i, s in enumerate(eeg_streams):
        print(
            f"{i}: name={s.name()} | type={s.type()} | "
            f"channels={s.channel_count()} | fs={s.nominal_srate()}"
        )

    return eeg_streams[0]


def get_channel_names(info):
    try:
        desc = info.desc()
        ch = desc.child("channels").child("channel")
        names = []

        for _ in range(info.channel_count()):
            label = ch.child_value("label")
            if label:
                names.append(label.upper())
            ch = ch.next_sibling()

        if len(names) == info.channel_count():
            return names

    except Exception:
        pass

    return EXPECTED_CHANNELS[:info.channel_count()]


def record_eeg(inlet, seconds):
    samples = []
    timestamps = []

    t0 = time.time()

    while time.time() - t0 < seconds:
        sample, timestamp = inlet.pull_sample(timeout=1.0)

        if sample is not None:
            samples.append(sample)
            timestamps.append(timestamp)

    return np.asarray(samples, dtype=float), np.asarray(timestamps, dtype=float)

# SELECCIÓN DE CANALES

def get_channel_indices(channel_names, selected_channels):
    names_upper = [c.upper() for c in channel_names]
    indices = []

    for ch in selected_channels:
        if ch.upper() in names_upper:
            indices.append(names_upper.index(ch.upper()))

    if len(indices) < 3:
        raise RuntimeError(
            f"No hay suficientes canales para CSP. Detectados: {channel_names}. "
            f"Necesarios recomendados: {selected_channels}"
        )

    return indices


def select_channels(data, channel_names):
    indices = get_channel_indices(channel_names, CHANNELS_CSP)
    return data[:, indices]


# CSP

class CSPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None

    def _covariance(self, trial):
        cov = trial.T @ trial
        cov = cov / (np.trace(cov) + 1e-12)
        return cov

    def fit(self, X, y):
        # X: n_trials, n_samples, n_channels
        classes = np.unique(y)

        if len(classes) != 2:
            raise ValueError("CSPTransformer solo admite clasificación binaria.")

        covs = {}

        for cls in classes:
            cls_trials = X[y == cls]
            covs[cls] = np.mean([self._covariance(trial) for trial in cls_trials], axis=0)

        c0 = covs[classes[0]]
        c1 = covs[classes[1]]
        composite = c0 + c1

        eigvals, eigvecs = np.linalg.eigh(composite)
        eigvals = np.maximum(eigvals, 1e-12)

        whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        s0 = whitening.T @ c0 @ whitening

        eigvals_s0, eigvecs_s0 = np.linalg.eigh(s0)

        order = np.argsort(eigvals_s0)[::-1]
        eigvecs_s0 = eigvecs_s0[:, order]

        filters = whitening @ eigvecs_s0

        n = self.n_components // 2
        selected = np.r_[0:n, -n:0]

        self.filters_ = filters[:, selected]

        return self

    def transform(self, X):
        features = []

        for trial in X:
            projected = trial @ self.filters_
            var = np.var(projected, axis=0)
            feat = np.log(var / (np.sum(var) + 1e-12) + 1e-12)
            features.append(feat)

        return np.asarray(features)

# DATASET POR TRIALS

def save_trial_csv(data, timestamps, channel_names, class_name, trial_id, phase):
    df = pd.DataFrame(data, columns=channel_names)
    df.insert(0, "timestamp", timestamps)
    df["trial_id"] = trial_id
    df["phase"] = phase
    df["class"] = class_name

    filename = DATA_DIR / f"trial_{trial_id:03d}_{class_name}_{phase}.csv"
    df.to_csv(filename, index=False)


def plot_raw_trial(data, channel_names, class_name, trial_id):
    t = np.arange(data.shape[0]) / FS

    plt.figure(figsize=(14, 8))
    offset = 0.0

    for i, ch in enumerate(channel_names):
        signal = data[:, i]
        signal = signal - np.mean(signal)
        signal = signal / (np.std(signal) + 1e-9)

        plt.plot(t, signal + offset, label=ch)
        offset += 5.0

    plt.title(f"EEG raw normalizado - {class_name} - trial {trial_id}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Canales con offset")
    plt.legend(loc="upper right")
    plt.tight_layout()

    filename = PLOT_DIR / f"trial_{trial_id:03d}_{class_name}_raw.png"
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_psd_like_summary(trials, labels, channel_names):
    rows = []

    for trial, label in zip(trials, labels):
        data = preprocess_eeg(trial)
        selected = select_channels(data, channel_names)

        power = np.mean(selected ** 2, axis=0)
        log_power = np.log(power + 1e-9)

        selected_names = [channel_names[i] for i in get_channel_indices(channel_names, CHANNELS_CSP)]

        for ch, val in zip(selected_names, log_power):
            rows.append({
                "class": CLASS_NAMES[int(label)],
                "channel": ch,
                "log_power": val,
            })

    df = pd.DataFrame(rows)
    summary = df.groupby(["class", "channel"])["log_power"].mean().reset_index()

    plt.figure(figsize=(10, 5))

    for cls in summary["class"].unique():
        sub = summary[summary["class"] == cls]
        plt.plot(sub["channel"], sub["log_power"], marker="o", label=cls)

    plt.title("Resumen potencia logarítmica 8-30 Hz por clase")
    plt.xlabel("Canal")
    plt.ylabel("Log power")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = PLOT_DIR / "band_power_summary_all_classes.png"
    plt.savefig(filename, dpi=150)
    plt.close()


def record_trials(inlet, channel_names):
    all_trials = []
    all_labels = []
    all_groups = []

    trial_id = 0

    print("GRABACIÓN EEG POR TRIALS")
    print("Clases: REST, LEFT, RIGHT.")
    print("Durante LEFT/RIGHT NO muevas el brazo: solo imagina el movimiento.")
    print("Evita parpadear durante los segundos de imaginación.")
    input("Pulsa ENTER para empezar...")

    recording_plan = [
        (CLASS_REST, "REST"),
        (CLASS_LEFT, "LEFT"),
        (CLASS_RIGHT, "RIGHT"),
    ]

    for class_id, class_name in recording_plan:
        print(f"CLASE: {class_name}")
        input(f"Pulsa ENTER para empezar trials de {class_name}...")

        for k in range(N_TRIALS_PER_CLASS):
            trial_id += 1

            print(f"\nTrial {trial_id} / clase {class_name}")
            print("Preparado...")
            time.sleep(1.0)

            print("3")
            time.sleep(1.0)
            print("2")
            time.sleep(1.0)
            print("1")
            time.sleep(1.0)

            if class_name == "REST":
                print("REST: relájate y mira a un punto fijo.")
                data, ts = record_eeg(inlet, MI_SECONDS)
            else:
                print(f"CUE: IMAGINA {class_name}")
                time.sleep(CUE_SECONDS)
                data, ts = record_eeg(inlet, MI_SECONDS)

            if len(data) < int(0.8 * MI_SECONDS * FS):
                print("AVISO: trial demasiado corto. Se descarta.")
                continue

            save_trial_csv(data, ts, channel_names, class_name, trial_id, "MI")
            plot_raw_trial(data, channel_names, class_name, trial_id)

            all_trials.append(data)
            all_labels.append(class_id)
            all_groups.append(trial_id)

            print("Descanso...")
            rest_data, rest_ts = record_eeg(inlet, REST_SECONDS)
            save_trial_csv(rest_data, rest_ts, channel_names, class_name, trial_id, "REST_AFTER")

    return all_trials, np.asarray(all_labels), np.asarray(all_groups)


def equalize_trials_length(trials):
    min_len = min(len(t) for t in trials)
    return np.asarray([t[:min_len] for t in trials], dtype=float)


def preprocess_trials_for_csp(trials, channel_names):
    processed = []

    for trial in trials:
        data = preprocess_eeg(trial)
        data = select_channels(data, channel_names)
        processed.append(data)

    return equalize_trials_length(processed)

# ENTRENAMIENTO DOS ETAPAS

def train_models(inlet, channel_names):
    trials, labels, groups = record_trials(inlet, channel_names)

    plot_psd_like_summary(trials, labels, channel_names)

    X = preprocess_trials_for_csp(trials, channel_names)
    y = labels

    print("\nDataset:")
    print("X:", X.shape)
    print("y:", y.shape)
    print("Distribución:", Counter(y))

    # Etapa 1: REST vs MI
    y_mi = np.where(y == CLASS_REST, 0, 1)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_mi, groups=groups))

    rest_mi_model = Pipeline([
        ("csp", CSPTransformer(n_components=4)),
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis())
    ])

    rest_mi_model.fit(X[train_idx], y_mi[train_idx])
    pred_mi = rest_mi_model.predict(X[test_idx])

    print("\nRESULTADOS REST VS MI:")
    print(confusion_matrix(y_mi[test_idx], pred_mi))
    print(classification_report(y_mi[test_idx], pred_mi, target_names=["REST", "MI"]))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_mi[test_idx],
        pred_mi,
        display_labels=["REST", "MI"]
    )
    disp.figure_.savefig(PLOT_DIR / "confusion_rest_vs_mi.png", dpi=150)
    plt.close(disp.figure_)

    # Etapa 2: LEFT vs RIGHT
    lr_mask = y != CLASS_REST
    X_lr = X[lr_mask]
    y_lr_original = y[lr_mask]
    groups_lr = groups[lr_mask]

    y_lr = np.where(y_lr_original == CLASS_LEFT, 0, 1)

    gss_lr = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_lr, test_lr = next(gss_lr.split(X_lr, y_lr, groups=groups_lr))

    left_right_model = Pipeline([
        ("csp", CSPTransformer(n_components=4)),
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42))
    ])

    left_right_model.fit(X_lr[train_lr], y_lr[train_lr])
    pred_lr = left_right_model.predict(X_lr[test_lr])

    print("\nRESULTADOS LEFT VS RIGHT:")
    print(confusion_matrix(y_lr[test_lr], pred_lr))
    print(classification_report(y_lr[test_lr], pred_lr, target_names=["LEFT", "RIGHT"]))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_lr[test_lr],
        pred_lr,
        display_labels=["LEFT", "RIGHT"]
    )
    disp.figure_.savefig(PLOT_DIR / "confusion_left_vs_right.png", dpi=150)
    plt.close(disp.figure_)

    package = {
        "rest_mi_model": rest_mi_model,
        "left_right_model": left_right_model,
        "channel_names": channel_names,
        "channels_csp": CHANNELS_CSP,
        "fs": FS,
        "class_names": CLASS_NAMES,
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(package, f)

    print("\nModelo guardado:", MODEL_FILE)

    return package


def load_or_train_model(inlet, channel_names):
    if MODEL_FILE.exists():
        ans = input(f"\nExiste {MODEL_FILE}. ¿Cargar modelo? [s/n]: ").strip().lower()

        if ans == "s":
            with open(MODEL_FILE, "rb") as f:
                return pickle.load(f)

    return train_models(inlet, channel_names)

# ARDUINO

def read_arduino_lines(arduino, duration=1.0):
    t0 = time.time()

    while time.time() - t0 < duration:
        line = arduino.readline()

        if line:
            try:
                print("[ARDUINO]", line.decode(errors="ignore").strip())
            except Exception:
                pass


def wait_until_robot_finishes(arduino):
    print("\nEsperando a que termine el robot...")
    print("Usa EMG cuando Arduino lo pida.\n")

    while True:
        line = arduino.readline()

        if not line:
            continue

        text = line.decode(errors="ignore").strip()

        if text:
            print("[ARDUINO]", text)

        if (
            "=== RIGHT TRAJECTORY END ===" in text
            or "=== LEFT TRAJECTORY END ===" in text
            or "=== D2 + EMG END ===" in text
        ):
            print("Robot terminado.\n")
            break


def calibrate_emg_from_python(arduino):
    print("\nCALIBRACIÓN EMG")
    print("Primero relaja 5 s. Luego contrae fuerte 5 s.")
    input("Pulsa ENTER para calibrar EMG...")

    arduino.write(b"c\n")
    arduino.flush()

    read_arduino_lines(arduino, duration=12.0)


def test_robot_branch(arduino, command, name):
    input(f"Pulsa ENTER para probar trayectoria {name} + EMG...")

    arduino.write(command)
    arduino.flush()

    wait_until_robot_finishes(arduino)

# ONLINE

def prepare_online_trial(buffer, channel_names, target_len=None):
    data = np.asarray(buffer, dtype=float)

    data = preprocess_eeg(data)
    data = select_channels(data, channel_names)

    if target_len is not None:
        if len(data) > target_len:
            data = data[-target_len:]
        elif len(data) < target_len:
            pad = np.zeros((target_len - len(data), data.shape[1]))
            data = np.vstack([pad, data])

    return data.reshape(1, data.shape[0], data.shape[1])


def get_proba_binary(model, X):
    probs = model.predict_proba(X)[0]
    classes = list(model.named_steps["clf"].classes_)
    return probs, classes


def online_control(inlet, channel_names, package, arduino):
    rest_mi_model = package["rest_mi_model"]
    left_right_model = package["left_right_model"]

    window_samples = int(FS * ONLINE_WINDOW_SECONDS)
    buffer = deque(maxlen=window_samples)
    vote_history = deque(maxlen=VOTE_HISTORY_SIZE)

    print("ONLINE EEG CSP")
    print("Etapa 1: REST vs MI")
    print("Etapa 2: LEFT vs RIGHT")
    print("LEFT  -> envía L")
    print("RIGHT -> envía R")
    print("REST  -> no envía nada")
    print("Pulsa Ctrl+C para salir.")
    input("Pulsa ENTER para empezar online...")

    last_decision_time = time.time()

    while True:
        sample, timestamp = inlet.pull_sample(timeout=1.0)

        if sample is None:
            print("No llega EEG...")
            continue

        buffer.append(sample)

        if len(buffer) < window_samples:
            continue

        if time.time() - last_decision_time < ONLINE_STEP_SECONDS:
            continue

        last_decision_time = time.time()

        X_online = prepare_online_trial(buffer, channel_names)

        mi_probs, mi_classes = get_proba_binary(rest_mi_model, X_online)
        p_rest = float(mi_probs[mi_classes.index(0)])
        p_mi = float(mi_probs[mi_classes.index(1)])

        vote = None
        p_left = 0.0
        p_right = 0.0
        margin_lr = 0.0

        if p_mi >= PROBA_THRESHOLD_MI:
            lr_probs, lr_classes = get_proba_binary(left_right_model, X_online)

            p_left = float(lr_probs[lr_classes.index(0)])
            p_right = float(lr_probs[lr_classes.index(1)])

            margin_lr = abs(p_left - p_right)

            if margin_lr >= MIN_MARGIN_LR:
                if p_left >= PROBA_THRESHOLD_LR:
                    vote = "LEFT"
                elif p_right >= PROBA_THRESHOLD_LR:
                    vote = "RIGHT"
        else:
            vote = "REST"

        vote_history.append(vote)

        counts = Counter(vote_history)

        print(
            f"REST={p_rest:.2f} MI={p_mi:.2f} | "
            f"LEFT={p_left:.2f} RIGHT={p_right:.2f} margin={margin_lr:.2f} | "
            f"vote={vote if vote else 'dudoso'} | "
            f"hist={dict(counts)}"
        )

        if counts["LEFT"] >= MIN_VOTES_TO_SEND:
            print("\nDECISIÓN ESTABLE: LEFT -> mando L")
            arduino.write(COMMAND_LEFT)
            arduino.flush()

            wait_until_robot_finishes(arduino)

            buffer.clear()
            vote_history.clear()
            input("Pulsa ENTER para otro intento...")

        elif counts["RIGHT"] >= MIN_VOTES_TO_SEND:
            print("\nDECISIÓN ESTABLE: RIGHT -> mando R")
            arduino.write(COMMAND_RIGHT)
            arduino.flush()

            wait_until_robot_finishes(arduino)

            buffer.clear()
            vote_history.clear()
            input("Pulsa ENTER para otro intento...")

        elif counts["REST"] >= MIN_VOTES_TO_SEND:
            print("\nREST estable. No mando nada.")
            buffer.clear()
            vote_history.clear()
            time.sleep(1.0)

# MAIN

def main():
    global FS

    eeg_stream = find_eeg_stream()
    inlet = StreamInlet(eeg_stream)

    info = inlet.info()
    channel_names = get_channel_names(info)

    if info.nominal_srate() > 0:
        FS = int(info.nominal_srate())

    print("\nCanales detectados:")
    print(channel_names)
    print("FS usada:", FS)

    print("\nCanales CSP usados:")
    print(CHANNELS_CSP)

    missing = [ch for ch in CHANNELS_CSP if ch not in channel_names]
    if missing:
        print("\nAVISO: faltan canales CSP recomendados:")
        print(missing)

    print("\nConectando con Arduino...")
    arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
    time.sleep(2.0)

    read_arduino_lines(arduino, duration=2.0)

    while True:
        print("\nOpciones:")
        print("1) Calibrar EMG")
        print("2) Probar LEFT + EMG")
        print("3) Probar RIGHT + EMG")
        print("4) Entrenar/cargar EEG y empezar online")
        print("5) Salir")

        option = input("Elige [1/2/3/4/5]: ").strip()

        if option == "1":
            calibrate_emg_from_python(arduino)

        elif option == "2":
            test_robot_branch(arduino, COMMAND_LEFT, "LEFT")

        elif option == "3":
            test_robot_branch(arduino, COMMAND_RIGHT, "RIGHT")

        elif option == "4":
            package = load_or_train_model(inlet, channel_names)
            online_control(inlet, channel_names, package, arduino)

        elif option == "5":
            print("Saliendo.")
            break

        else:
            print("Opción no válida.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrograma detenido.")