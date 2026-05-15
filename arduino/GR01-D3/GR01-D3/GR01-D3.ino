#include <ax12.h>

#include "poses.h"
#include "robot.h"
#include <avr/interrupt.h>
#include <avr/io.h>
#include <math.h>

// GR01 - D3 EEG + EMG
//
// Python envía:
//   "R\n" -> RIGHT: q0 -> qtg CUBIC1, EMG, qtg -> qtv -> qtr CUBIC2, EMG
//   "L\n" -> LEFT:  q0 -> qtg CUBIC1, EMG, qtg -> qtr CUBIC1 directo, EMG
//
// RIGHT usa punto intermedio qtv para evitar obstáculo.
// LEFT va directo de qtg a qtr y representa trayectoria incorrecta/colisión.

// CONFIGURACIÓN ROBOT

static const uint8_t DOFS_LOCAL = DOFs;

static double q0[DOFs]  = {0.0,    2.89,   -2.89,   0.0,     0.0};
static double qtg[DOFs] = {0.3530, 1.7217, -2.6405, 0.9188,  1.5708};
static double qtr[DOFs] = {1.5458, 1.1833, -0.9807, -1.7734, 3.1166};
static double qtv[DOFs] = {0.7484, 2.0075, -1.3783, -0.6292, 1.5708};

static double qMin[DOFs] = {-2.62, -0.33, -2.89, -1.83, -1.05};
static double qMax[DOFs] = { 2.62,  2.97,  0.26,  1.86,  4.19};

static const uint16_t T_HOME_MS = 1500;
static const uint16_t T_1A_MS   = 2000;
static const uint16_t T_1B_1_MS = 1500;
static const uint16_t T_1B_2_MS = 1500;

static const unsigned long PLOT_DURATION_MS = 10000;

bool robotBusy = false;

// CONFIGURACIÓN EMG

#define EMG_PIN A0

static const unsigned long EMG_TS_US = 1000;

static const float BASELINE_ALPHA = 0.995f;
static const float ENV_ALPHA      = 0.75f;

static float emgThresholdHigh = 45.0f;
static float emgThresholdLow  = 30.0f;

static const unsigned long REFRACTORY_MS = 300;
static const unsigned long RELEASE_CONFIRM_MS = 300;
static const unsigned long RELEASE_PLOT_DT_MS = 20;

// ESTADO EMG

volatile int   emgRaw = 0;
volatile float emgBaseline = 512.0f;
volatile float emgHighPass = 0.0f;
volatile float emgRectified = 0.0f;
volatile float emgEnvelope = 0.0f;

bool emgStateHigh = false;

unsigned long lastEmgSampleUs = 0;
unsigned long lastTriggerMs = 0;
unsigned long lastPrintMs = 0;

// DECLARACIONES

void MenuOptions();

void UpdateEMG();
bool EMGContractionDetected();
bool IsEMGCurrentlyContracted();

void CalibrateEMGThreshold();
void WaitForEMGTrigger(const char* message);
void WaitForEMGRelease(const char* message);
void PlotEMG();
void MonitorEMG();

void Run_D2_EMG_Sequence();

void ExecuteEEG_LeftTrajectory();
void ExecuteEEG_RightTrajectory();
void GoHome();

bool ValidatePose(const char* name, double q[DOFs]);
bool ValidateAllPoses();

// SETUP

void setup()
{
  ROBOT_Init();

  Serial.begin(115200);
  Serial.setTimeout(20);

  delay(300);

  pinMode(EMG_PIN, INPUT);

  SERVOS_ServosOn();

  Serial.println("--------------------------------------------");
  Serial.println("GR01-D3 READY - EEG + EMG");
  Serial.println("--------------------------------------------");

  ValidateAllPoses();

  GoHome();

  Serial.println("Python EEG commands:");
  Serial.println("  R -> CUBIC1 + CUBIC2 with EMG, obstacle avoidance");
  Serial.println("  L -> CUBIC1 + CUBIC1 direct with EMG, collision path");
  Serial.println();

  MenuOptions();
}

// LOOP

void loop()
{
  UpdateEMG();

  if (Serial.available() <= 0) return;

  String input = Serial.readStringUntil('\n');
  input.trim();

  if (input.length() == 0) return;

  char inByte = input.charAt(0);

  if (robotBusy)
  {
    Serial.println("Robot busy. Command ignored.");
    return;
  }

  switch (inByte)
  {
    case 'L':
    case 'l':
      ExecuteEEG_LeftTrajectory();
      break;

    case 'R':
    case 'r':
      ExecuteEEG_RightTrajectory();
      break;

    case 'H':
    case 'h':
      GoHome();
      break;

    case '0':
      SERVOS_ServosOff();
      Serial.println("Servos OFF.");
      break;

    case '1':
      SERVOS_ServosOn();
      Serial.println("Servos ON.");
      break;

    case '3':
      ROBOT_GripperClose();
      Serial.println("Gripper CLOSE.");
      break;

    case '4':
      ROBOT_GripperOpen();
      Serial.println("Gripper OPEN.");
      break;

    case 'c':
    case 'C':
      CalibrateEMGThreshold();
      break;

    case 'm':
    case 'M':
      MonitorEMG();
      break;

    case 'p':
    case 'P':
      PlotEMG();
      break;

    case '8':
      Run_D2_EMG_Sequence();
      break;

    default:
      Serial.println("Unknown option.");
      break;
  }

  MenuOptions();
}

// MENÚ

void MenuOptions()
{
  Serial.println("\n----------------------------");
  Serial.println("EEG commands from Python:");
  Serial.println("L) EEG LEFT: CUBIC1 + CUBIC1 direct qtg -> qtr with EMG");
  Serial.println("R) EEG RIGHT: CUBIC1 + CUBIC2 qtg -> qtv -> qtr with EMG");
  Serial.println();
  Serial.println("Manual / debug:");
  Serial.println("H) Home");
  Serial.println("0) Relax Servos");
  Serial.println("1) Hold Servos");
  Serial.println("3) Gripper Close");
  Serial.println("4) Gripper Open");
  Serial.println("c) Calibrate EMG threshold");
  Serial.println("m) Monitor EMG for 10 s");
  Serial.println("p) Plot EMG signal");
  Serial.println("8) Run original D2 EMG sequence");
  Serial.println("----------------------------\n");
}

// VALIDACIÓN DE POSES

bool ValidatePose(const char* name, double q[DOFs])
{
  bool ok = true;

  Serial.print("Validating pose ");
  Serial.println(name);

  for (int i = 0; i < DOFs; i++)
  {
    if (q[i] < qMin[i] || q[i] > qMax[i])
    {
      Serial.print("  ERROR joint q");
      Serial.print(i + 1);
      Serial.print(" = ");
      Serial.print(q[i], 4);
      Serial.print(" outside [");
      Serial.print(qMin[i], 4);
      Serial.print(", ");
      Serial.print(qMax[i], 4);
      Serial.println("]");
      ok = false;
    }
  }

  if (ok)
  {
    Serial.print("  ");
    Serial.print(name);
    Serial.println(" OK");
  }

  return ok;
}


bool ValidateAllPoses()
{
  bool ok = true;

  ok &= ValidatePose("q0", q0);
  ok &= ValidatePose("qtg", qtg);
  ok &= ValidatePose("qtr", qtr);
  ok &= ValidatePose("qtv", qtv);

  if (ok)
  {
    Serial.println("All poses are inside mechanical limits.");
  }
  else
  {
    Serial.println("WARNING: some poses are outside mechanical limits.");
  }

  return ok;
}

// HOME

void GoHome()
{
  robotBusy = true;

  Serial.println("Going to q0...");
  SERVOS_ServosOn();

  ROBOT_SetSingleTrajectory(q0, T_HOME_MS, CUBIC1);
  delay(T_HOME_MS + 300);

  ROBOT_GripperOpen();
  delay(400);

  Serial.println("Home reached.");

  robotBusy = false;
}

// TRAYECTORIA LEFT

//
// LEFT:
//   q0 -> qtg con CUBIC1
//   EMG cierra pinza
//   qtg -> qtr directo con CUBIC1
//   EMG abre pinza
//
// Esta trayectoria NO usa qtv.

void ExecuteEEG_LeftTrajectory()
{
  robotBusy = true;

  Serial.println("\n=== EEG LEFT DETECTED ===");
  Serial.println("Executing LEFT trajectory.");
  Serial.println("CUBIC1 + CUBIC1 with EMG.");
  Serial.println("This trajectory does NOT use qtv.");
  Serial.println("It moves directly from qtg to qtr.");

  SERVOS_ServosOn();
  delay(200);

  Serial.println("Going to q0...");
  ROBOT_SetSingleTrajectory(q0, T_HOME_MS, CUBIC1);
  delay(T_HOME_MS + 300);

  Serial.println("Opening gripper...");
  ROBOT_GripperOpen();
  delay(600);

  Serial.println("LEFT trajectory part 1: q0 -> qtg with CUBIC1...");
  ROBOT_SetSingleTrajectory(qtg, T_1A_MS, CUBIC1);
  delay(T_1A_MS + 300);

  WaitForEMGTrigger("At grasp pose qtg. Contract forearm to close gripper.");

  Serial.println("Closing gripper...");
  ROBOT_GripperClose();
  emgStateHigh = true;
  delay(800);

  Serial.println("LEFT trajectory part 2: qtg -> qtr directly with CUBIC1...");
  ROBOT_SetSingleTrajectory(qtr, T_1B_1_MS + T_1B_2_MS, CUBIC1);

  unsigned long trajStart = millis();
  unsigned long trajDuration = (unsigned long)T_1B_1_MS + (unsigned long)T_1B_2_MS + 400UL;

  while (millis() - trajStart < trajDuration)
  {
    UpdateEMG();

    if (millis() - lastPrintMs >= 20)
    {
      lastPrintMs = millis();

      Serial.print(emgRaw);
      Serial.print(",");
      Serial.print(emgEnvelope, 2);
      Serial.print(",");
      Serial.println(emgThresholdHigh, 2);
    }
  }

  WaitForEMGRelease("At release pose qtr. Relax forearm to open gripper.");

  Serial.println("Opening gripper...");
  ROBOT_GripperOpen();
  delay(800);

  Serial.println("Returning to q0...");
  ROBOT_SetSingleTrajectory(q0, T_HOME_MS, CUBIC1);
  delay(T_HOME_MS + 300);

  Serial.println("=== LEFT TRAJECTORY END ===\n");

  robotBusy = false;
}

// TRAYECTORIA RIGHT
//
// RIGHT:
//   q0 -> qtg con CUBIC1
//   EMG cierra pinza
//   qtg -> qtv -> qtr con CUBIC2
//   EMG abre pinza
//
// Esta trayectoria usa qtv para evitar obstáculo.

void ExecuteEEG_RightTrajectory()
{
  robotBusy = true;

  Serial.println("\n=== EEG RIGHT DETECTED ===");
  Serial.println("Executing RIGHT trajectory.");
  Serial.println("CUBIC1 + CUBIC2 with EMG.");
  Serial.println("This trajectory uses qtv to avoid the obstacle.");

  SERVOS_ServosOn();
  delay(200);

  Serial.println("Going to q0...");
  ROBOT_SetSingleTrajectory(q0, T_HOME_MS, CUBIC1);
  delay(T_HOME_MS + 300);

  Serial.println("Opening gripper...");
  ROBOT_GripperOpen();
  delay(600);

  Serial.println("RIGHT trajectory part 1: q0 -> qtg with CUBIC1...");
  ROBOT_SetSingleTrajectory(qtg, T_1A_MS, CUBIC1);
  delay(T_1A_MS + 300);

  WaitForEMGTrigger("At grasp pose qtg. Contract forearm to close gripper.");

  Serial.println("Closing gripper...");
  ROBOT_GripperClose();
  emgStateHigh = true;
  delay(800);

  Serial.println("RIGHT trajectory part 2: qtg -> qtv -> qtr with CUBIC2...");
  ROBOT_SetDoubleTrajectory(qtv, qtr, T_1B_1_MS, T_1B_2_MS, CUBIC2);

  unsigned long trajStart = millis();
  unsigned long trajDuration = (unsigned long)T_1B_1_MS + (unsigned long)T_1B_2_MS + 400UL;

  while (millis() - trajStart < trajDuration)
  {
    UpdateEMG();

    if (millis() - lastPrintMs >= 20)
    {
      lastPrintMs = millis();

      Serial.print(emgRaw);
      Serial.print(",");
      Serial.print(emgEnvelope, 2);
      Serial.print(",");
      Serial.println(emgThresholdHigh, 2);
    }
  }

  WaitForEMGRelease("At release pose qtr. Relax forearm to open gripper.");

  Serial.println("Opening gripper...");
  ROBOT_GripperOpen();
  delay(800);

  Serial.println("Returning to q0...");
  ROBOT_SetSingleTrajectory(q0, T_HOME_MS, CUBIC1);
  delay(T_HOME_MS + 300);

  Serial.println("=== RIGHT TRAJECTORY END ===\n");

  robotBusy = false;
}

// EMG UPDATE

void UpdateEMG()
{
  unsigned long nowUs = micros();

  if (nowUs - lastEmgSampleUs < EMG_TS_US) return;

  lastEmgSampleUs = nowUs;

  int sample = analogRead(EMG_PIN);
  emgRaw = sample;

  emgBaseline = BASELINE_ALPHA * emgBaseline + (1.0f - BASELINE_ALPHA) * (float)sample;
  emgHighPass = (float)sample - emgBaseline;
  emgRectified = fabs(emgHighPass);
  emgEnvelope = ENV_ALPHA * emgEnvelope + (1.0f - ENV_ALPHA) * emgRectified;
}

// DETECTOR EMG

bool EMGContractionDetected()
{
  unsigned long nowMs = millis();

  if (!emgStateHigh && emgEnvelope >= emgThresholdHigh)
  {
    if (nowMs - lastTriggerMs > REFRACTORY_MS)
    {
      emgStateHigh = true;
      lastTriggerMs = nowMs;
      return true;
    }
  }

  if (emgStateHigh && emgEnvelope <= emgThresholdLow)
  {
    emgStateHigh = false;
  }

  return false;
}


bool IsEMGCurrentlyContracted()
{
  if (emgEnvelope >= emgThresholdHigh)
  {
    emgStateHigh = true;
  }
  else if (emgEnvelope <= emgThresholdLow)
  {
    emgStateHigh = false;
  }

  return emgStateHigh;
}

// CALIBRACIÓN EMG

void CalibrateEMGThreshold()
{
  Serial.println("\n=== EMG CALIBRATION START ===");
  Serial.println("Phase 1: RELAX for 5 seconds...");

  float restMean = 0.0f;
  float restMax  = 0.0f;
  unsigned long count = 0;

  unsigned long t0 = millis();

  while (millis() - t0 < 5000UL)
  {
    UpdateEMG();

    restMean += emgEnvelope;

    if (emgEnvelope > restMax)
    {
      restMax = emgEnvelope;
    }

    count++;
  }

  restMean /= (float)count;

  Serial.println("Phase 2: make STRONG contractions for 5 seconds...");

  float actMean = 0.0f;
  float actMax  = 0.0f;
  count = 0;
  t0 = millis();

  while (millis() - t0 < 5000UL)
  {
    UpdateEMG();

    actMean += emgEnvelope;

    if (emgEnvelope > actMax)
    {
      actMax = emgEnvelope;
    }

    count++;
  }

  actMean /= (float)count;

  emgThresholdHigh = restMax + 0.20f * (actMax - restMax);
  emgThresholdLow  = restMax + 0.10f * (actMax - restMax);

  if (emgThresholdLow >= emgThresholdHigh)
  {
    emgThresholdLow = 0.7f * emgThresholdHigh;
  }

  Serial.print("restMean = "); Serial.println(restMean, 2);
  Serial.print("restMax  = "); Serial.println(restMax, 2);
  Serial.print("actMean  = "); Serial.println(actMean, 2);
  Serial.print("actMax   = "); Serial.println(actMax, 2);
  Serial.print("thrHigh  = "); Serial.println(emgThresholdHigh, 2);
  Serial.print("thrLow   = "); Serial.println(emgThresholdLow, 2);

  Serial.println("=== EMG CALIBRATION END ===\n");
}

// ESPERA CONTRACCIÓN EMG

void WaitForEMGTrigger(const char* message)
{
  Serial.println(message);
  Serial.println("Perform one clear muscle contraction...");

  while (true)
  {
    UpdateEMG();

    if (millis() - lastPrintMs >= 20)
    {
      lastPrintMs = millis();

      Serial.print(emgRaw);
      Serial.print(",");
      Serial.print(emgEnvelope, 2);
      Serial.print(",");
      Serial.println(emgThresholdHigh, 2);
    }

    if (EMGContractionDetected())
    {
      Serial.println("EMG trigger detected.");
      delay(200);
      return;
    }
  }
}

// ESPERA RELAJACIÓN EMG

void WaitForEMGRelease(const char* message)
{
  Serial.println(message);
  Serial.println("Relax the forearm to open the gripper...");

  unsigned long belowSinceMs = 0;

  while (true)
  {
    UpdateEMG();

    if (millis() - lastPrintMs >= RELEASE_PLOT_DT_MS)
    {
      lastPrintMs = millis();

      Serial.print(emgRaw);
      Serial.print(",");
      Serial.print(emgEnvelope, 2);
      Serial.print(",");
      Serial.println(emgThresholdLow, 2);
    }

    if (emgEnvelope <= emgThresholdLow)
    {
      if (belowSinceMs == 0)
      {
        belowSinceMs = millis();
      }

      if (millis() - belowSinceMs >= RELEASE_CONFIRM_MS)
      {
        emgStateHigh = false;
        Serial.println("EMG release detected.");
        delay(200);
        return;
      }
    }
    else
    {
      belowSinceMs = 0;
      emgStateHigh = true;
    }
  }
}

// MONITOR EMG

void MonitorEMG()
{
  Serial.println("EMG monitor for 10 s: raw,envelope,thrHigh");

  unsigned long t0 = millis();

  while (millis() - t0 < 10000UL)
  {
    UpdateEMG();

    if (millis() - lastPrintMs >= 10)
    {
      lastPrintMs = millis();

      Serial.print(emgRaw);
      Serial.print(",");
      Serial.print(emgEnvelope, 2);
      Serial.print(",");
      Serial.println(emgThresholdHigh, 2);
    }
  }

  Serial.println("End monitor.");
}

// PLOT EMG

void PlotEMG()
{
  unsigned long t0 = millis();

  while (millis() - t0 < PLOT_DURATION_MS)
  {
    UpdateEMG();

    bool contracted = IsEMGCurrentlyContracted();
    int trigger = (emgEnvelope >= emgThresholdHigh) ? 1 : 0;

    Serial.print("raw:");
    Serial.print(emgRaw);
    Serial.print(",");

    Serial.print("envelope:");
    Serial.print(emgEnvelope, 2);
    Serial.print(",");

    Serial.print("thrHigh:");
    Serial.print(emgThresholdHigh, 2);
    Serial.print(",");

    Serial.print("thrLow:");
    Serial.print(emgThresholdLow, 2);
    Serial.print(",");

    Serial.print("contracted:");
    Serial.print(contracted ? 1 : 0);
    Serial.print(",");

    Serial.print("trigger:");
    Serial.println(trigger);

    if (Serial.available() > 0)
    {
      char c = Serial.read();

      if (c == 'q' || c == 'Q')
      {
        break;
      }
    }

    delay(10);
  }

  Serial.println("End plot mode.");
}

// SECUENCIA D2 ORIGINAL

void Run_D2_EMG_Sequence()
{
  robotBusy = true;

  Serial.println("\n=== D2 + EMG START ===");

  SERVOS_ServosOn();
  delay(200);

  Serial.println("Going to q0...");
  ROBOT_SetSingleTrajectory(q0, T_HOME_MS, CUBIC1);
  delay(T_HOME_MS + 300);

  Serial.println("Opening gripper...");
  ROBOT_GripperOpen();
  delay(600);

  Serial.println("Trajectory 1a: q0 -> qtg...");
  ROBOT_SetSingleTrajectory(qtg, T_1A_MS, CUBIC1);
  delay(T_1A_MS + 300);

  WaitForEMGTrigger("At grasp pose qtg. Contract forearm to grasp.");

  Serial.println("Closing gripper...");
  ROBOT_GripperClose();
  emgStateHigh = true;
  delay(800);

  Serial.println("Trajectory 1b: qtg -> qtv -> qtr...");
  ROBOT_SetDoubleTrajectory(qtv, qtr, T_1B_1_MS, T_1B_2_MS, CUBIC2);

  unsigned long trajStart = millis();
  unsigned long trajDuration = (unsigned long)T_1B_1_MS + (unsigned long)T_1B_2_MS + 400UL;

  while (millis() - trajStart < trajDuration)
  {
    UpdateEMG();

    if (millis() - lastPrintMs >= 20)
    {
      lastPrintMs = millis();

      Serial.print(emgRaw);
      Serial.print(",");
      Serial.print(emgEnvelope, 2);
      Serial.print(",");
      Serial.println(emgThresholdHigh, 2);
    }
  }

  WaitForEMGRelease("At release pose qtr. Relax forearm to release.");

  Serial.println("Opening gripper...");
  ROBOT_GripperOpen();
  delay(800);

  Serial.println("Returning to q0...");
  ROBOT_SetSingleTrajectory(q0, T_HOME_MS, CUBIC1);
  delay(T_HOME_MS + 300);

  Serial.println("=== D2 + EMG END ===\n");

  robotBusy = false;
}