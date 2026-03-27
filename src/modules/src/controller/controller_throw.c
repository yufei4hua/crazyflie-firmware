/**
 * ,---------,       ____  _ __
 * |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 * | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * | / ,--´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * Crazyflie control firmware
 *
 * Copyright (C) 2024 Bitcraze AB
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, in version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * out_of_tree_controller.c - App layer application of an out of tree controller.
 */

#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "app.h"

#include "FreeRTOS.h"
#include "task.h"

// The new controller goes here --------------------------------------------
// Move the includes to the the top of the file if you want to
#include "param.h"
#include "log.h"
#include "controller.h"
#include "controller_throw.h"
#include "rl_throw_policy_params.h"


// =========================
// Drone parameters (const)
// =========================

// Controller Parameters
static int ctrl_freq_hz = 500;  // controller frequency (matches rotor_vel training: freq=250)

// Action scaling: policy output in [-1, 1] -> rotor RPM in [ROTOR_RPM_MIN, ROTOR_RPM_MAX]
static const float ROTOR_RPM_MIN = 6962.07f;
static const float ROTOR_RPM_MAX = 21302.27f;
static const float ROTOR_RPM_SCALE = (ROTOR_RPM_MAX - ROTOR_RPM_MIN) * 0.5f;
static const float ROTOR_RPM_MEAN  = (ROTOR_RPM_MAX + ROTOR_RPM_MIN) * 0.5f;

// thrust = a0 + a1 * rpm + a2 * rpm^2
static const float RPM2THRUST_A0 = 0.0f;
static const float RPM2THRUST_A1 = -3.133427287299859e-7f;
static const float RPM2THRUST_A2 =  4.407354891648379e-10f;

// Max thrust used for normalization: pwm_norm = force / thrust_max
static const float THRUST_MAX = 0.2f;

// Unit conversion
static const float DEG2RAD = 0.01745329251994329577f;  // pi/180


// =========================
// Internal state
// =========================
// Stored as normalized policy output in [-1, 1]; initialized to hover (~0.25)
static float g_last_actions[4] = {0.25f, 0.25f, 0.25f, 0.25f};


// =========================
// Small utilities
// =========================
static inline float clampf(float x, float lo, float hi) {
  return (x < lo) ? lo : (x > hi) ? hi : x;
}

static inline float relu(float x) {
  return (x > 0.0f) ? x : 0.0f;
}

// NOTE:
// - For Cortex-M4F, tanhf() is acceptable for 4 outputs.
// - If later you need speed, replace with LUT/approx.
static inline float tanh_act(float x) {
  return tanhf(x);
}


// Dense: y = x @ W + b
// W stored as row-major flattened with shape (in_dim, out_dim):
//   W[i*out_dim + j] == kernel[i, j]
// W and b are stored as float16 (__fp16) to save memory (472 bytes vs 944 bytes)
// They are automatically promoted to float32 during arithmetic operations
static void dense_forward(const float* x, int in_dim,
                          const __fp16* W, const __fp16* b, int out_dim,
                          float* y) {
  for (int j = 0; j < out_dim; ++j) {
    float s = b[j];  // __fp16 -> float promotion
    for (int i = 0; i < in_dim; ++i) {
      s += x[i] * W[i * out_dim + j];  // __fp16 -> float promotion
    }
    y[j] = s;
  }
}

// =========================
// Observation construction
// =========================
// Throw env obs dict (rotor_vel policy, from _obs()):
//   "ang_vel" (3), "last_actions" (4), "quat" (4), "vel" (3), "z" (1)
// After FlattenJaxObservation with sorted keys => 15 dims
//
// Notes:
//   - "z" is raw z position (no goal subtraction)
//   - "vel" is raw world-frame velocity (no goal subtraction)
//   - "last_actions" stores previous normalized policy outputs in [-1, 1]
static void build_obs(float obs[15],
                      const sensorData_t* sensors,
                      const state_t* state) {
  // ang_vel from gyro, deg/s -> rad/s
  const float wx = sensors->gyro.x * DEG2RAD;
  const float wy = sensors->gyro.y * DEG2RAD;
  const float wz = sensors->gyro.z * DEG2RAD;

  // quat from state->attitudeQuaternion, order xyzw
  const float qx = state->attitudeQuaternion.x;
  const float qy = state->attitudeQuaternion.y;
  const float qz = state->attitudeQuaternion.z;
  const float qw = state->attitudeQuaternion.w;

  // raw world-frame velocity
  const float vx = state->velocity.x;
  const float vy = state->velocity.y;
  const float vz = state->velocity.z;

  // raw z position
  const float z = state->position.z;

  // Fill in strict alphabetical order of Python obs dict keys:
  // 0..2   "ang_vel" (3)
  obs[0] = wx;
  obs[1] = wy;
  obs[2] = wz;

  // 3..6   "last_actions" (4) - previous normalized policy output in [-1, 1]
  obs[3] = g_last_actions[0];
  obs[4] = g_last_actions[1];
  obs[5] = g_last_actions[2];
  obs[6] = g_last_actions[3];

  // 7..10  "quat" (4) - xyzw
  obs[7]  = qx;
  obs[8]  = qy;
  obs[9]  = qz;
  obs[10] = qw;

  // 11..13 "vel" (3)
  obs[11] = vx;
  obs[12] = vy;
  obs[13] = vz;

  // 14     "z" (1)
  obs[14] = z;
}

// =========================
// Policy forward
// =========================
static void policy_forward(const float obs[15], float action_out[4]) {
  // 3 Dense layers: (15->48), (48->48), (48->4)
  float h0[ACTOR_L0_OUT];
  float h1[ACTOR_L1_OUT];

  // Layer 0: ReLU
  dense_forward(obs, ACTOR_L0_IN, actor_W0, actor_b0, ACTOR_L0_OUT, h0);
  for (int i = 0; i < ACTOR_L0_OUT; ++i) {
    h0[i] = relu(h0[i]);
  }

  // Layer 1: ReLU
  dense_forward(h0, ACTOR_L1_IN, actor_W1, actor_b1, ACTOR_L1_OUT, h1);
  for (int i = 0; i < ACTOR_L1_OUT; ++i) {
    h1[i] = relu(h1[i]);
  }

  // Output layer: tanh
  dense_forward(h1, ACTOR_L2_IN, actor_W2, actor_b2, ACTOR_L2_OUT, action_out);
  for (int i = 0; i < ACTOR_L2_OUT; ++i) {
    action_out[i] = tanh_act(action_out[i]);
  }
}

// =========================
// Public API
// =========================
void controllerThrowInit(void) {
  // Reset last_actions on each controller switch
  g_last_actions[0] = 0.25f;
  g_last_actions[1] = 0.25f;
  g_last_actions[2] = 0.25f;
  g_last_actions[3] = 0.25f;

  // Policy params (actor_W0, actor_b0, etc.) are loaded at boot with defaults
  // and can be modified via CRTP params. They persist across controller switches.
}

bool controllerThrowTest(void) {
  return true;
}

void controllerThrow(control_t *control, const setpoint_t *setpoint, const sensorData_t *sensors, const state_t *state, const stabilizerStep_t stabilizerStep) {
  control->controlMode = controlModePWM;

  if (!RATE_DO_EXECUTE(ctrl_freq_hz, stabilizerStep)) {
    return;
  }

  // 1. Build observation vector (15-dim)
  float obs[15];
  build_obs(obs, sensors, state);

  // 2. Policy forward -> normalized rotor velocities in [-1, 1]
  float normalized_rotor_vel[4];
  policy_forward(obs, normalized_rotor_vel);

  // 3. Update last_actions (normalized policy output, used in next obs)
  g_last_actions[0] = normalized_rotor_vel[0];
  g_last_actions[1] = normalized_rotor_vel[1];
  g_last_actions[2] = normalized_rotor_vel[2];
  g_last_actions[3] = normalized_rotor_vel[3];

  // 4. Map policy output [-1,1] -> rotor RPM -> thrust -> normalized forces in [0,1]
  for (int i = 0; i < 4; ++i) {
    // Scale to RPM: rpm = clip(a, -1, 1) * ((max-min)/2) + ((max+min)/2)
    const float rpm = clampf(normalized_rotor_vel[i], -1.0f, 1.0f) * ROTOR_RPM_SCALE + ROTOR_RPM_MEAN;

    // Convert RPM to thrust force via quadratic model
    const float force = RPM2THRUST_A0
                      + RPM2THRUST_A1 * rpm
                      + RPM2THRUST_A2 * (rpm * rpm);

    // Normalize to [0,1] and clip
    float pwm_norm = force / THRUST_MAX;
    pwm_norm = clampf(pwm_norm, 0.0f, 1.0f);

    control->normalizedForces[i] = pwm_norm;
  }
}


/**
 * Tuning variables for the throw RL controller
 */
PARAM_GROUP_START(ctrlThrow)
PARAM_ADD_CORE(PARAM_INT16 | PARAM_PERSISTENT, freq, &ctrl_freq_hz)

// Layer 0 weights: 120 params (15x8 matrix stored row-major)
PARAM_ADD(PARAM_FLOAT16, W0_000, &actor_W0[0])
PARAM_ADD(PARAM_FLOAT16, W0_001, &actor_W0[1])
PARAM_ADD(PARAM_FLOAT16, W0_002, &actor_W0[2])
PARAM_ADD(PARAM_FLOAT16, W0_003, &actor_W0[3])
PARAM_ADD(PARAM_FLOAT16, W0_004, &actor_W0[4])
PARAM_ADD(PARAM_FLOAT16, W0_005, &actor_W0[5])
PARAM_ADD(PARAM_FLOAT16, W0_006, &actor_W0[6])
PARAM_ADD(PARAM_FLOAT16, W0_007, &actor_W0[7])
PARAM_ADD(PARAM_FLOAT16, W0_008, &actor_W0[8])
PARAM_ADD(PARAM_FLOAT16, W0_009, &actor_W0[9])
PARAM_ADD(PARAM_FLOAT16, W0_010, &actor_W0[10])
PARAM_ADD(PARAM_FLOAT16, W0_011, &actor_W0[11])
PARAM_ADD(PARAM_FLOAT16, W0_012, &actor_W0[12])
PARAM_ADD(PARAM_FLOAT16, W0_013, &actor_W0[13])
PARAM_ADD(PARAM_FLOAT16, W0_014, &actor_W0[14])
PARAM_ADD(PARAM_FLOAT16, W0_015, &actor_W0[15])
PARAM_ADD(PARAM_FLOAT16, W0_016, &actor_W0[16])
PARAM_ADD(PARAM_FLOAT16, W0_017, &actor_W0[17])
PARAM_ADD(PARAM_FLOAT16, W0_018, &actor_W0[18])
PARAM_ADD(PARAM_FLOAT16, W0_019, &actor_W0[19])
PARAM_ADD(PARAM_FLOAT16, W0_020, &actor_W0[20])
PARAM_ADD(PARAM_FLOAT16, W0_021, &actor_W0[21])
PARAM_ADD(PARAM_FLOAT16, W0_022, &actor_W0[22])
PARAM_ADD(PARAM_FLOAT16, W0_023, &actor_W0[23])
PARAM_ADD(PARAM_FLOAT16, W0_024, &actor_W0[24])
PARAM_ADD(PARAM_FLOAT16, W0_025, &actor_W0[25])
PARAM_ADD(PARAM_FLOAT16, W0_026, &actor_W0[26])
PARAM_ADD(PARAM_FLOAT16, W0_027, &actor_W0[27])
PARAM_ADD(PARAM_FLOAT16, W0_028, &actor_W0[28])
PARAM_ADD(PARAM_FLOAT16, W0_029, &actor_W0[29])
PARAM_ADD(PARAM_FLOAT16, W0_030, &actor_W0[30])
PARAM_ADD(PARAM_FLOAT16, W0_031, &actor_W0[31])
PARAM_ADD(PARAM_FLOAT16, W0_032, &actor_W0[32])
PARAM_ADD(PARAM_FLOAT16, W0_033, &actor_W0[33])
PARAM_ADD(PARAM_FLOAT16, W0_034, &actor_W0[34])
PARAM_ADD(PARAM_FLOAT16, W0_035, &actor_W0[35])
PARAM_ADD(PARAM_FLOAT16, W0_036, &actor_W0[36])
PARAM_ADD(PARAM_FLOAT16, W0_037, &actor_W0[37])
PARAM_ADD(PARAM_FLOAT16, W0_038, &actor_W0[38])
PARAM_ADD(PARAM_FLOAT16, W0_039, &actor_W0[39])
PARAM_ADD(PARAM_FLOAT16, W0_040, &actor_W0[40])
PARAM_ADD(PARAM_FLOAT16, W0_041, &actor_W0[41])
PARAM_ADD(PARAM_FLOAT16, W0_042, &actor_W0[42])
PARAM_ADD(PARAM_FLOAT16, W0_043, &actor_W0[43])
PARAM_ADD(PARAM_FLOAT16, W0_044, &actor_W0[44])
PARAM_ADD(PARAM_FLOAT16, W0_045, &actor_W0[45])
PARAM_ADD(PARAM_FLOAT16, W0_046, &actor_W0[46])
PARAM_ADD(PARAM_FLOAT16, W0_047, &actor_W0[47])
PARAM_ADD(PARAM_FLOAT16, W0_048, &actor_W0[48])
PARAM_ADD(PARAM_FLOAT16, W0_049, &actor_W0[49])
PARAM_ADD(PARAM_FLOAT16, W0_050, &actor_W0[50])
PARAM_ADD(PARAM_FLOAT16, W0_051, &actor_W0[51])
PARAM_ADD(PARAM_FLOAT16, W0_052, &actor_W0[52])
PARAM_ADD(PARAM_FLOAT16, W0_053, &actor_W0[53])
PARAM_ADD(PARAM_FLOAT16, W0_054, &actor_W0[54])
PARAM_ADD(PARAM_FLOAT16, W0_055, &actor_W0[55])
PARAM_ADD(PARAM_FLOAT16, W0_056, &actor_W0[56])
PARAM_ADD(PARAM_FLOAT16, W0_057, &actor_W0[57])
PARAM_ADD(PARAM_FLOAT16, W0_058, &actor_W0[58])
PARAM_ADD(PARAM_FLOAT16, W0_059, &actor_W0[59])
PARAM_ADD(PARAM_FLOAT16, W0_060, &actor_W0[60])
PARAM_ADD(PARAM_FLOAT16, W0_061, &actor_W0[61])
PARAM_ADD(PARAM_FLOAT16, W0_062, &actor_W0[62])
PARAM_ADD(PARAM_FLOAT16, W0_063, &actor_W0[63])
PARAM_ADD(PARAM_FLOAT16, W0_064, &actor_W0[64])
PARAM_ADD(PARAM_FLOAT16, W0_065, &actor_W0[65])
PARAM_ADD(PARAM_FLOAT16, W0_066, &actor_W0[66])
PARAM_ADD(PARAM_FLOAT16, W0_067, &actor_W0[67])
PARAM_ADD(PARAM_FLOAT16, W0_068, &actor_W0[68])
PARAM_ADD(PARAM_FLOAT16, W0_069, &actor_W0[69])
PARAM_ADD(PARAM_FLOAT16, W0_070, &actor_W0[70])
PARAM_ADD(PARAM_FLOAT16, W0_071, &actor_W0[71])
PARAM_ADD(PARAM_FLOAT16, W0_072, &actor_W0[72])
PARAM_ADD(PARAM_FLOAT16, W0_073, &actor_W0[73])
PARAM_ADD(PARAM_FLOAT16, W0_074, &actor_W0[74])
PARAM_ADD(PARAM_FLOAT16, W0_075, &actor_W0[75])
PARAM_ADD(PARAM_FLOAT16, W0_076, &actor_W0[76])
PARAM_ADD(PARAM_FLOAT16, W0_077, &actor_W0[77])
PARAM_ADD(PARAM_FLOAT16, W0_078, &actor_W0[78])
PARAM_ADD(PARAM_FLOAT16, W0_079, &actor_W0[79])
PARAM_ADD(PARAM_FLOAT16, W0_080, &actor_W0[80])
PARAM_ADD(PARAM_FLOAT16, W0_081, &actor_W0[81])
PARAM_ADD(PARAM_FLOAT16, W0_082, &actor_W0[82])
PARAM_ADD(PARAM_FLOAT16, W0_083, &actor_W0[83])
PARAM_ADD(PARAM_FLOAT16, W0_084, &actor_W0[84])
PARAM_ADD(PARAM_FLOAT16, W0_085, &actor_W0[85])
PARAM_ADD(PARAM_FLOAT16, W0_086, &actor_W0[86])
PARAM_ADD(PARAM_FLOAT16, W0_087, &actor_W0[87])
PARAM_ADD(PARAM_FLOAT16, W0_088, &actor_W0[88])
PARAM_ADD(PARAM_FLOAT16, W0_089, &actor_W0[89])
PARAM_ADD(PARAM_FLOAT16, W0_090, &actor_W0[90])
PARAM_ADD(PARAM_FLOAT16, W0_091, &actor_W0[91])
PARAM_ADD(PARAM_FLOAT16, W0_092, &actor_W0[92])
PARAM_ADD(PARAM_FLOAT16, W0_093, &actor_W0[93])
PARAM_ADD(PARAM_FLOAT16, W0_094, &actor_W0[94])
PARAM_ADD(PARAM_FLOAT16, W0_095, &actor_W0[95])
PARAM_ADD(PARAM_FLOAT16, W0_096, &actor_W0[96])
PARAM_ADD(PARAM_FLOAT16, W0_097, &actor_W0[97])
PARAM_ADD(PARAM_FLOAT16, W0_098, &actor_W0[98])
PARAM_ADD(PARAM_FLOAT16, W0_099, &actor_W0[99])
PARAM_ADD(PARAM_FLOAT16, W0_100, &actor_W0[100])
PARAM_ADD(PARAM_FLOAT16, W0_101, &actor_W0[101])
PARAM_ADD(PARAM_FLOAT16, W0_102, &actor_W0[102])
PARAM_ADD(PARAM_FLOAT16, W0_103, &actor_W0[103])
PARAM_ADD(PARAM_FLOAT16, W0_104, &actor_W0[104])
PARAM_ADD(PARAM_FLOAT16, W0_105, &actor_W0[105])
PARAM_ADD(PARAM_FLOAT16, W0_106, &actor_W0[106])
PARAM_ADD(PARAM_FLOAT16, W0_107, &actor_W0[107])
PARAM_ADD(PARAM_FLOAT16, W0_108, &actor_W0[108])
PARAM_ADD(PARAM_FLOAT16, W0_109, &actor_W0[109])
PARAM_ADD(PARAM_FLOAT16, W0_110, &actor_W0[110])
PARAM_ADD(PARAM_FLOAT16, W0_111, &actor_W0[111])
PARAM_ADD(PARAM_FLOAT16, W0_112, &actor_W0[112])
PARAM_ADD(PARAM_FLOAT16, W0_113, &actor_W0[113])
PARAM_ADD(PARAM_FLOAT16, W0_114, &actor_W0[114])
PARAM_ADD(PARAM_FLOAT16, W0_115, &actor_W0[115])
PARAM_ADD(PARAM_FLOAT16, W0_116, &actor_W0[116])
PARAM_ADD(PARAM_FLOAT16, W0_117, &actor_W0[117])
PARAM_ADD(PARAM_FLOAT16, W0_118, &actor_W0[118])
PARAM_ADD(PARAM_FLOAT16, W0_119, &actor_W0[119])

// Layer 0 biases: 8 params
PARAM_ADD(PARAM_FLOAT16, b0_0, &actor_b0[0])
PARAM_ADD(PARAM_FLOAT16, b0_1, &actor_b0[1])
PARAM_ADD(PARAM_FLOAT16, b0_2, &actor_b0[2])
PARAM_ADD(PARAM_FLOAT16, b0_3, &actor_b0[3])
PARAM_ADD(PARAM_FLOAT16, b0_4, &actor_b0[4])
PARAM_ADD(PARAM_FLOAT16, b0_5, &actor_b0[5])
PARAM_ADD(PARAM_FLOAT16, b0_6, &actor_b0[6])
PARAM_ADD(PARAM_FLOAT16, b0_7, &actor_b0[7])

// Layer 1 weights: 64 params (8x8 matrix stored row-major)
PARAM_ADD(PARAM_FLOAT16, W1_00, &actor_W1[0])
PARAM_ADD(PARAM_FLOAT16, W1_01, &actor_W1[1])
PARAM_ADD(PARAM_FLOAT16, W1_02, &actor_W1[2])
PARAM_ADD(PARAM_FLOAT16, W1_03, &actor_W1[3])
PARAM_ADD(PARAM_FLOAT16, W1_04, &actor_W1[4])
PARAM_ADD(PARAM_FLOAT16, W1_05, &actor_W1[5])
PARAM_ADD(PARAM_FLOAT16, W1_06, &actor_W1[6])
PARAM_ADD(PARAM_FLOAT16, W1_07, &actor_W1[7])
PARAM_ADD(PARAM_FLOAT16, W1_08, &actor_W1[8])
PARAM_ADD(PARAM_FLOAT16, W1_09, &actor_W1[9])
PARAM_ADD(PARAM_FLOAT16, W1_10, &actor_W1[10])
PARAM_ADD(PARAM_FLOAT16, W1_11, &actor_W1[11])
PARAM_ADD(PARAM_FLOAT16, W1_12, &actor_W1[12])
PARAM_ADD(PARAM_FLOAT16, W1_13, &actor_W1[13])
PARAM_ADD(PARAM_FLOAT16, W1_14, &actor_W1[14])
PARAM_ADD(PARAM_FLOAT16, W1_15, &actor_W1[15])
PARAM_ADD(PARAM_FLOAT16, W1_16, &actor_W1[16])
PARAM_ADD(PARAM_FLOAT16, W1_17, &actor_W1[17])
PARAM_ADD(PARAM_FLOAT16, W1_18, &actor_W1[18])
PARAM_ADD(PARAM_FLOAT16, W1_19, &actor_W1[19])
PARAM_ADD(PARAM_FLOAT16, W1_20, &actor_W1[20])
PARAM_ADD(PARAM_FLOAT16, W1_21, &actor_W1[21])
PARAM_ADD(PARAM_FLOAT16, W1_22, &actor_W1[22])
PARAM_ADD(PARAM_FLOAT16, W1_23, &actor_W1[23])
PARAM_ADD(PARAM_FLOAT16, W1_24, &actor_W1[24])
PARAM_ADD(PARAM_FLOAT16, W1_25, &actor_W1[25])
PARAM_ADD(PARAM_FLOAT16, W1_26, &actor_W1[26])
PARAM_ADD(PARAM_FLOAT16, W1_27, &actor_W1[27])
PARAM_ADD(PARAM_FLOAT16, W1_28, &actor_W1[28])
PARAM_ADD(PARAM_FLOAT16, W1_29, &actor_W1[29])
PARAM_ADD(PARAM_FLOAT16, W1_30, &actor_W1[30])
PARAM_ADD(PARAM_FLOAT16, W1_31, &actor_W1[31])
PARAM_ADD(PARAM_FLOAT16, W1_32, &actor_W1[32])
PARAM_ADD(PARAM_FLOAT16, W1_33, &actor_W1[33])
PARAM_ADD(PARAM_FLOAT16, W1_34, &actor_W1[34])
PARAM_ADD(PARAM_FLOAT16, W1_35, &actor_W1[35])
PARAM_ADD(PARAM_FLOAT16, W1_36, &actor_W1[36])
PARAM_ADD(PARAM_FLOAT16, W1_37, &actor_W1[37])
PARAM_ADD(PARAM_FLOAT16, W1_38, &actor_W1[38])
PARAM_ADD(PARAM_FLOAT16, W1_39, &actor_W1[39])
PARAM_ADD(PARAM_FLOAT16, W1_40, &actor_W1[40])
PARAM_ADD(PARAM_FLOAT16, W1_41, &actor_W1[41])
PARAM_ADD(PARAM_FLOAT16, W1_42, &actor_W1[42])
PARAM_ADD(PARAM_FLOAT16, W1_43, &actor_W1[43])
PARAM_ADD(PARAM_FLOAT16, W1_44, &actor_W1[44])
PARAM_ADD(PARAM_FLOAT16, W1_45, &actor_W1[45])
PARAM_ADD(PARAM_FLOAT16, W1_46, &actor_W1[46])
PARAM_ADD(PARAM_FLOAT16, W1_47, &actor_W1[47])
PARAM_ADD(PARAM_FLOAT16, W1_48, &actor_W1[48])
PARAM_ADD(PARAM_FLOAT16, W1_49, &actor_W1[49])
PARAM_ADD(PARAM_FLOAT16, W1_50, &actor_W1[50])
PARAM_ADD(PARAM_FLOAT16, W1_51, &actor_W1[51])
PARAM_ADD(PARAM_FLOAT16, W1_52, &actor_W1[52])
PARAM_ADD(PARAM_FLOAT16, W1_53, &actor_W1[53])
PARAM_ADD(PARAM_FLOAT16, W1_54, &actor_W1[54])
PARAM_ADD(PARAM_FLOAT16, W1_55, &actor_W1[55])
PARAM_ADD(PARAM_FLOAT16, W1_56, &actor_W1[56])
PARAM_ADD(PARAM_FLOAT16, W1_57, &actor_W1[57])
PARAM_ADD(PARAM_FLOAT16, W1_58, &actor_W1[58])
PARAM_ADD(PARAM_FLOAT16, W1_59, &actor_W1[59])
PARAM_ADD(PARAM_FLOAT16, W1_60, &actor_W1[60])
PARAM_ADD(PARAM_FLOAT16, W1_61, &actor_W1[61])
PARAM_ADD(PARAM_FLOAT16, W1_62, &actor_W1[62])
PARAM_ADD(PARAM_FLOAT16, W1_63, &actor_W1[63])

// Layer 1 biases: 8 params
PARAM_ADD(PARAM_FLOAT16, b1_0, &actor_b1[0])
PARAM_ADD(PARAM_FLOAT16, b1_1, &actor_b1[1])
PARAM_ADD(PARAM_FLOAT16, b1_2, &actor_b1[2])
PARAM_ADD(PARAM_FLOAT16, b1_3, &actor_b1[3])
PARAM_ADD(PARAM_FLOAT16, b1_4, &actor_b1[4])
PARAM_ADD(PARAM_FLOAT16, b1_5, &actor_b1[5])
PARAM_ADD(PARAM_FLOAT16, b1_6, &actor_b1[6])
PARAM_ADD(PARAM_FLOAT16, b1_7, &actor_b1[7])

// Layer 2 weights: 32 params (8x4 matrix stored row-major)
PARAM_ADD(PARAM_FLOAT16, W2_00, &actor_W2[0])
PARAM_ADD(PARAM_FLOAT16, W2_01, &actor_W2[1])
PARAM_ADD(PARAM_FLOAT16, W2_02, &actor_W2[2])
PARAM_ADD(PARAM_FLOAT16, W2_03, &actor_W2[3])
PARAM_ADD(PARAM_FLOAT16, W2_04, &actor_W2[4])
PARAM_ADD(PARAM_FLOAT16, W2_05, &actor_W2[5])
PARAM_ADD(PARAM_FLOAT16, W2_06, &actor_W2[6])
PARAM_ADD(PARAM_FLOAT16, W2_07, &actor_W2[7])
PARAM_ADD(PARAM_FLOAT16, W2_08, &actor_W2[8])
PARAM_ADD(PARAM_FLOAT16, W2_09, &actor_W2[9])
PARAM_ADD(PARAM_FLOAT16, W2_10, &actor_W2[10])
PARAM_ADD(PARAM_FLOAT16, W2_11, &actor_W2[11])
PARAM_ADD(PARAM_FLOAT16, W2_12, &actor_W2[12])
PARAM_ADD(PARAM_FLOAT16, W2_13, &actor_W2[13])
PARAM_ADD(PARAM_FLOAT16, W2_14, &actor_W2[14])
PARAM_ADD(PARAM_FLOAT16, W2_15, &actor_W2[15])
PARAM_ADD(PARAM_FLOAT16, W2_16, &actor_W2[16])
PARAM_ADD(PARAM_FLOAT16, W2_17, &actor_W2[17])
PARAM_ADD(PARAM_FLOAT16, W2_18, &actor_W2[18])
PARAM_ADD(PARAM_FLOAT16, W2_19, &actor_W2[19])
PARAM_ADD(PARAM_FLOAT16, W2_20, &actor_W2[20])
PARAM_ADD(PARAM_FLOAT16, W2_21, &actor_W2[21])
PARAM_ADD(PARAM_FLOAT16, W2_22, &actor_W2[22])
PARAM_ADD(PARAM_FLOAT16, W2_23, &actor_W2[23])
PARAM_ADD(PARAM_FLOAT16, W2_24, &actor_W2[24])
PARAM_ADD(PARAM_FLOAT16, W2_25, &actor_W2[25])
PARAM_ADD(PARAM_FLOAT16, W2_26, &actor_W2[26])
PARAM_ADD(PARAM_FLOAT16, W2_27, &actor_W2[27])
PARAM_ADD(PARAM_FLOAT16, W2_28, &actor_W2[28])
PARAM_ADD(PARAM_FLOAT16, W2_29, &actor_W2[29])
PARAM_ADD(PARAM_FLOAT16, W2_30, &actor_W2[30])
PARAM_ADD(PARAM_FLOAT16, W2_31, &actor_W2[31])

// Layer 2 biases: 4 params
PARAM_ADD(PARAM_FLOAT16, b2_0, &actor_b2[0])
PARAM_ADD(PARAM_FLOAT16, b2_1, &actor_b2[1])
PARAM_ADD(PARAM_FLOAT16, b2_2, &actor_b2[2])
PARAM_ADD(PARAM_FLOAT16, b2_3, &actor_b2[3])


PARAM_GROUP_STOP(ctrlThrow)

/**
 * Logging variables for the throw RL controller
 */
LOG_GROUP_START(ctrlThrow)
LOG_ADD(LOG_FLOAT, action_1, &g_last_actions[0])
LOG_ADD(LOG_FLOAT, action_2, &g_last_actions[1])
LOG_ADD(LOG_FLOAT, action_3, &g_last_actions[2])
LOG_ADD(LOG_FLOAT, action_4, &g_last_actions[3])
LOG_GROUP_STOP(ctrlThrow)