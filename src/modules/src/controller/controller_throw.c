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
#include "platform_defaults.h"
#include "math3d.h"
#include "rl_throw_policy_params.h"

// =========================
// Drone parameters (const)
// =========================

// Controller Parameters
static int ctrl_freq_hz = 500;  // controller frequency (matches rotor_vel training: freq=250)

// thrust = a0 + a1 * rpm + a2 * rpm^2
// Values are valid for the cf21B_500
static const float RPM2THRUST_A0 = 0.0f;
static const float RPM2THRUST_A1 = -3.133427287299859e-7f;
static const float RPM2THRUST_A2 =  4.407354891648379e-10f;

// Action scaling: policy output in [-1, 1] -> rotor RPM in [ROTOR_RPM_MIN, ROTOR_RPM_MAX]
// Values are valid for the cf21B_500
// TODO: Compute from known min/max thrust and the thrust curve instead of hardcoding
static const float ROTOR_RPM_MIN = 6962.07f;
static const float ROTOR_RPM_MAX = 21302.27f;
static const float ROTOR_RPM_SCALE = (ROTOR_RPM_MAX - ROTOR_RPM_MIN) * 0.5f;
static const float ROTOR_RPM_MEAN  = (ROTOR_RPM_MAX + ROTOR_RPM_MIN) * 0.5f;

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
// - USE_QUAT=1: "ang_vel" (3), "last_actions" (4), "quat" (4), "vel" (3), "z" (1) => 15 dims
// - USE_QUAT=0: "ang_vel" (3), "last_actions" (4), "rot_mat" (9), "vel" (3), "z" (1) => 20 dims
//
// Notes:
//   - "z" is raw z position (no goal subtraction)
//   - "vel" is raw world-frame velocity (no goal subtraction)
//   - "last_actions" stores previous normalized policy outputs in [-1, 1]
static void build_obs(float obs[OBS_DIM],
                      const sensorData_t* sensors,
                      const state_t* state) {
  // ang_vel from gyro, deg/s -> rad/s
  const float wx = sensors->gyro.x * DEG2RAD;
  const float wy = sensors->gyro.y * DEG2RAD;
  const float wz = sensors->gyro.z * DEG2RAD;

  // attitude quaternion from state, order xyzw
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

#if USE_QUAT
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
#else
  // 7..15  "rot_mat" (9), row-major from quaternion
  const struct quat q = mkquat(qx, qy, qz, qw);
  const struct mat33 R = quat2rotmat(q);
  obs[7]  = R.m[0][0];
  obs[8]  = R.m[0][1];
  obs[9]  = R.m[0][2];
  obs[10] = R.m[1][0];
  obs[11] = R.m[1][1];
  obs[12] = R.m[1][2];
  obs[13] = R.m[2][0];
  obs[14] = R.m[2][1];
  obs[15] = R.m[2][2];

  // 16..18 "vel" (3)
  obs[16] = vx;
  obs[17] = vy;
  obs[18] = vz;

  // 19     "z" (1)
  obs[19] = z;
#endif
}

// =========================
// Policy forward
// =========================
static void policy_forward(const float obs[OBS_DIM], float action_out[ACTOR_OUTPUT_SIZE]) {
  // 3 Dense layers: (OBS_DIM->ACTOR_HIDDEN_SIZE), (ACTOR_HIDDEN_SIZE->ACTOR_HIDDEN_SIZE), (ACTOR_HIDDEN_SIZE->ACTOR_OUTPUT_SIZE)
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

  // 1. Build observation vector (15- or 20-dim)
  float obs[OBS_DIM];
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

// Layer 0 weights: 320 params (20x16 matrix stored row-major)
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
PARAM_ADD(PARAM_FLOAT16, W0_120, &actor_W0[120])
PARAM_ADD(PARAM_FLOAT16, W0_121, &actor_W0[121])
PARAM_ADD(PARAM_FLOAT16, W0_122, &actor_W0[122])
PARAM_ADD(PARAM_FLOAT16, W0_123, &actor_W0[123])
PARAM_ADD(PARAM_FLOAT16, W0_124, &actor_W0[124])
PARAM_ADD(PARAM_FLOAT16, W0_125, &actor_W0[125])
PARAM_ADD(PARAM_FLOAT16, W0_126, &actor_W0[126])
PARAM_ADD(PARAM_FLOAT16, W0_127, &actor_W0[127])
PARAM_ADD(PARAM_FLOAT16, W0_128, &actor_W0[128])
PARAM_ADD(PARAM_FLOAT16, W0_129, &actor_W0[129])
PARAM_ADD(PARAM_FLOAT16, W0_130, &actor_W0[130])
PARAM_ADD(PARAM_FLOAT16, W0_131, &actor_W0[131])
PARAM_ADD(PARAM_FLOAT16, W0_132, &actor_W0[132])
PARAM_ADD(PARAM_FLOAT16, W0_133, &actor_W0[133])
PARAM_ADD(PARAM_FLOAT16, W0_134, &actor_W0[134])
PARAM_ADD(PARAM_FLOAT16, W0_135, &actor_W0[135])
PARAM_ADD(PARAM_FLOAT16, W0_136, &actor_W0[136])
PARAM_ADD(PARAM_FLOAT16, W0_137, &actor_W0[137])
PARAM_ADD(PARAM_FLOAT16, W0_138, &actor_W0[138])
PARAM_ADD(PARAM_FLOAT16, W0_139, &actor_W0[139])
PARAM_ADD(PARAM_FLOAT16, W0_140, &actor_W0[140])
PARAM_ADD(PARAM_FLOAT16, W0_141, &actor_W0[141])
PARAM_ADD(PARAM_FLOAT16, W0_142, &actor_W0[142])
PARAM_ADD(PARAM_FLOAT16, W0_143, &actor_W0[143])
PARAM_ADD(PARAM_FLOAT16, W0_144, &actor_W0[144])
PARAM_ADD(PARAM_FLOAT16, W0_145, &actor_W0[145])
PARAM_ADD(PARAM_FLOAT16, W0_146, &actor_W0[146])
PARAM_ADD(PARAM_FLOAT16, W0_147, &actor_W0[147])
PARAM_ADD(PARAM_FLOAT16, W0_148, &actor_W0[148])
PARAM_ADD(PARAM_FLOAT16, W0_149, &actor_W0[149])
PARAM_ADD(PARAM_FLOAT16, W0_150, &actor_W0[150])
PARAM_ADD(PARAM_FLOAT16, W0_151, &actor_W0[151])
PARAM_ADD(PARAM_FLOAT16, W0_152, &actor_W0[152])
PARAM_ADD(PARAM_FLOAT16, W0_153, &actor_W0[153])
PARAM_ADD(PARAM_FLOAT16, W0_154, &actor_W0[154])
PARAM_ADD(PARAM_FLOAT16, W0_155, &actor_W0[155])
PARAM_ADD(PARAM_FLOAT16, W0_156, &actor_W0[156])
PARAM_ADD(PARAM_FLOAT16, W0_157, &actor_W0[157])
PARAM_ADD(PARAM_FLOAT16, W0_158, &actor_W0[158])
PARAM_ADD(PARAM_FLOAT16, W0_159, &actor_W0[159])
PARAM_ADD(PARAM_FLOAT16, W0_160, &actor_W0[160])
PARAM_ADD(PARAM_FLOAT16, W0_161, &actor_W0[161])
PARAM_ADD(PARAM_FLOAT16, W0_162, &actor_W0[162])
PARAM_ADD(PARAM_FLOAT16, W0_163, &actor_W0[163])
PARAM_ADD(PARAM_FLOAT16, W0_164, &actor_W0[164])
PARAM_ADD(PARAM_FLOAT16, W0_165, &actor_W0[165])
PARAM_ADD(PARAM_FLOAT16, W0_166, &actor_W0[166])
PARAM_ADD(PARAM_FLOAT16, W0_167, &actor_W0[167])
PARAM_ADD(PARAM_FLOAT16, W0_168, &actor_W0[168])
PARAM_ADD(PARAM_FLOAT16, W0_169, &actor_W0[169])
PARAM_ADD(PARAM_FLOAT16, W0_170, &actor_W0[170])
PARAM_ADD(PARAM_FLOAT16, W0_171, &actor_W0[171])
PARAM_ADD(PARAM_FLOAT16, W0_172, &actor_W0[172])
PARAM_ADD(PARAM_FLOAT16, W0_173, &actor_W0[173])
PARAM_ADD(PARAM_FLOAT16, W0_174, &actor_W0[174])
PARAM_ADD(PARAM_FLOAT16, W0_175, &actor_W0[175])
PARAM_ADD(PARAM_FLOAT16, W0_176, &actor_W0[176])
PARAM_ADD(PARAM_FLOAT16, W0_177, &actor_W0[177])
PARAM_ADD(PARAM_FLOAT16, W0_178, &actor_W0[178])
PARAM_ADD(PARAM_FLOAT16, W0_179, &actor_W0[179])
PARAM_ADD(PARAM_FLOAT16, W0_180, &actor_W0[180])
PARAM_ADD(PARAM_FLOAT16, W0_181, &actor_W0[181])
PARAM_ADD(PARAM_FLOAT16, W0_182, &actor_W0[182])
PARAM_ADD(PARAM_FLOAT16, W0_183, &actor_W0[183])
PARAM_ADD(PARAM_FLOAT16, W0_184, &actor_W0[184])
PARAM_ADD(PARAM_FLOAT16, W0_185, &actor_W0[185])
PARAM_ADD(PARAM_FLOAT16, W0_186, &actor_W0[186])
PARAM_ADD(PARAM_FLOAT16, W0_187, &actor_W0[187])
PARAM_ADD(PARAM_FLOAT16, W0_188, &actor_W0[188])
PARAM_ADD(PARAM_FLOAT16, W0_189, &actor_W0[189])
PARAM_ADD(PARAM_FLOAT16, W0_190, &actor_W0[190])
PARAM_ADD(PARAM_FLOAT16, W0_191, &actor_W0[191])
PARAM_ADD(PARAM_FLOAT16, W0_192, &actor_W0[192])
PARAM_ADD(PARAM_FLOAT16, W0_193, &actor_W0[193])
PARAM_ADD(PARAM_FLOAT16, W0_194, &actor_W0[194])
PARAM_ADD(PARAM_FLOAT16, W0_195, &actor_W0[195])
PARAM_ADD(PARAM_FLOAT16, W0_196, &actor_W0[196])
PARAM_ADD(PARAM_FLOAT16, W0_197, &actor_W0[197])
PARAM_ADD(PARAM_FLOAT16, W0_198, &actor_W0[198])
PARAM_ADD(PARAM_FLOAT16, W0_199, &actor_W0[199])
PARAM_ADD(PARAM_FLOAT16, W0_200, &actor_W0[200])
PARAM_ADD(PARAM_FLOAT16, W0_201, &actor_W0[201])
PARAM_ADD(PARAM_FLOAT16, W0_202, &actor_W0[202])
PARAM_ADD(PARAM_FLOAT16, W0_203, &actor_W0[203])
PARAM_ADD(PARAM_FLOAT16, W0_204, &actor_W0[204])
PARAM_ADD(PARAM_FLOAT16, W0_205, &actor_W0[205])
PARAM_ADD(PARAM_FLOAT16, W0_206, &actor_W0[206])
PARAM_ADD(PARAM_FLOAT16, W0_207, &actor_W0[207])
PARAM_ADD(PARAM_FLOAT16, W0_208, &actor_W0[208])
PARAM_ADD(PARAM_FLOAT16, W0_209, &actor_W0[209])
PARAM_ADD(PARAM_FLOAT16, W0_210, &actor_W0[210])
PARAM_ADD(PARAM_FLOAT16, W0_211, &actor_W0[211])
PARAM_ADD(PARAM_FLOAT16, W0_212, &actor_W0[212])
PARAM_ADD(PARAM_FLOAT16, W0_213, &actor_W0[213])
PARAM_ADD(PARAM_FLOAT16, W0_214, &actor_W0[214])
PARAM_ADD(PARAM_FLOAT16, W0_215, &actor_W0[215])
PARAM_ADD(PARAM_FLOAT16, W0_216, &actor_W0[216])
PARAM_ADD(PARAM_FLOAT16, W0_217, &actor_W0[217])
PARAM_ADD(PARAM_FLOAT16, W0_218, &actor_W0[218])
PARAM_ADD(PARAM_FLOAT16, W0_219, &actor_W0[219])
PARAM_ADD(PARAM_FLOAT16, W0_220, &actor_W0[220])
PARAM_ADD(PARAM_FLOAT16, W0_221, &actor_W0[221])
PARAM_ADD(PARAM_FLOAT16, W0_222, &actor_W0[222])
PARAM_ADD(PARAM_FLOAT16, W0_223, &actor_W0[223])
PARAM_ADD(PARAM_FLOAT16, W0_224, &actor_W0[224])
PARAM_ADD(PARAM_FLOAT16, W0_225, &actor_W0[225])
PARAM_ADD(PARAM_FLOAT16, W0_226, &actor_W0[226])
PARAM_ADD(PARAM_FLOAT16, W0_227, &actor_W0[227])
PARAM_ADD(PARAM_FLOAT16, W0_228, &actor_W0[228])
PARAM_ADD(PARAM_FLOAT16, W0_229, &actor_W0[229])
PARAM_ADD(PARAM_FLOAT16, W0_230, &actor_W0[230])
PARAM_ADD(PARAM_FLOAT16, W0_231, &actor_W0[231])
PARAM_ADD(PARAM_FLOAT16, W0_232, &actor_W0[232])
PARAM_ADD(PARAM_FLOAT16, W0_233, &actor_W0[233])
PARAM_ADD(PARAM_FLOAT16, W0_234, &actor_W0[234])
PARAM_ADD(PARAM_FLOAT16, W0_235, &actor_W0[235])
PARAM_ADD(PARAM_FLOAT16, W0_236, &actor_W0[236])
PARAM_ADD(PARAM_FLOAT16, W0_237, &actor_W0[237])
PARAM_ADD(PARAM_FLOAT16, W0_238, &actor_W0[238])
PARAM_ADD(PARAM_FLOAT16, W0_239, &actor_W0[239])
PARAM_ADD(PARAM_FLOAT16, W0_240, &actor_W0[240])
PARAM_ADD(PARAM_FLOAT16, W0_241, &actor_W0[241])
PARAM_ADD(PARAM_FLOAT16, W0_242, &actor_W0[242])
PARAM_ADD(PARAM_FLOAT16, W0_243, &actor_W0[243])
PARAM_ADD(PARAM_FLOAT16, W0_244, &actor_W0[244])
PARAM_ADD(PARAM_FLOAT16, W0_245, &actor_W0[245])
PARAM_ADD(PARAM_FLOAT16, W0_246, &actor_W0[246])
PARAM_ADD(PARAM_FLOAT16, W0_247, &actor_W0[247])
PARAM_ADD(PARAM_FLOAT16, W0_248, &actor_W0[248])
PARAM_ADD(PARAM_FLOAT16, W0_249, &actor_W0[249])
PARAM_ADD(PARAM_FLOAT16, W0_250, &actor_W0[250])
PARAM_ADD(PARAM_FLOAT16, W0_251, &actor_W0[251])
PARAM_ADD(PARAM_FLOAT16, W0_252, &actor_W0[252])
PARAM_ADD(PARAM_FLOAT16, W0_253, &actor_W0[253])
PARAM_ADD(PARAM_FLOAT16, W0_254, &actor_W0[254])
PARAM_ADD(PARAM_FLOAT16, W0_255, &actor_W0[255])
PARAM_ADD(PARAM_FLOAT16, W0_256, &actor_W0[256])
PARAM_ADD(PARAM_FLOAT16, W0_257, &actor_W0[257])
PARAM_ADD(PARAM_FLOAT16, W0_258, &actor_W0[258])
PARAM_ADD(PARAM_FLOAT16, W0_259, &actor_W0[259])
PARAM_ADD(PARAM_FLOAT16, W0_260, &actor_W0[260])
PARAM_ADD(PARAM_FLOAT16, W0_261, &actor_W0[261])
PARAM_ADD(PARAM_FLOAT16, W0_262, &actor_W0[262])
PARAM_ADD(PARAM_FLOAT16, W0_263, &actor_W0[263])
PARAM_ADD(PARAM_FLOAT16, W0_264, &actor_W0[264])
PARAM_ADD(PARAM_FLOAT16, W0_265, &actor_W0[265])
PARAM_ADD(PARAM_FLOAT16, W0_266, &actor_W0[266])
PARAM_ADD(PARAM_FLOAT16, W0_267, &actor_W0[267])
PARAM_ADD(PARAM_FLOAT16, W0_268, &actor_W0[268])
PARAM_ADD(PARAM_FLOAT16, W0_269, &actor_W0[269])
PARAM_ADD(PARAM_FLOAT16, W0_270, &actor_W0[270])
PARAM_ADD(PARAM_FLOAT16, W0_271, &actor_W0[271])
PARAM_ADD(PARAM_FLOAT16, W0_272, &actor_W0[272])
PARAM_ADD(PARAM_FLOAT16, W0_273, &actor_W0[273])
PARAM_ADD(PARAM_FLOAT16, W0_274, &actor_W0[274])
PARAM_ADD(PARAM_FLOAT16, W0_275, &actor_W0[275])
PARAM_ADD(PARAM_FLOAT16, W0_276, &actor_W0[276])
PARAM_ADD(PARAM_FLOAT16, W0_277, &actor_W0[277])
PARAM_ADD(PARAM_FLOAT16, W0_278, &actor_W0[278])
PARAM_ADD(PARAM_FLOAT16, W0_279, &actor_W0[279])
PARAM_ADD(PARAM_FLOAT16, W0_280, &actor_W0[280])
PARAM_ADD(PARAM_FLOAT16, W0_281, &actor_W0[281])
PARAM_ADD(PARAM_FLOAT16, W0_282, &actor_W0[282])
PARAM_ADD(PARAM_FLOAT16, W0_283, &actor_W0[283])
PARAM_ADD(PARAM_FLOAT16, W0_284, &actor_W0[284])
PARAM_ADD(PARAM_FLOAT16, W0_285, &actor_W0[285])
PARAM_ADD(PARAM_FLOAT16, W0_286, &actor_W0[286])
PARAM_ADD(PARAM_FLOAT16, W0_287, &actor_W0[287])
PARAM_ADD(PARAM_FLOAT16, W0_288, &actor_W0[288])
PARAM_ADD(PARAM_FLOAT16, W0_289, &actor_W0[289])
PARAM_ADD(PARAM_FLOAT16, W0_290, &actor_W0[290])
PARAM_ADD(PARAM_FLOAT16, W0_291, &actor_W0[291])
PARAM_ADD(PARAM_FLOAT16, W0_292, &actor_W0[292])
PARAM_ADD(PARAM_FLOAT16, W0_293, &actor_W0[293])
PARAM_ADD(PARAM_FLOAT16, W0_294, &actor_W0[294])
PARAM_ADD(PARAM_FLOAT16, W0_295, &actor_W0[295])
PARAM_ADD(PARAM_FLOAT16, W0_296, &actor_W0[296])
PARAM_ADD(PARAM_FLOAT16, W0_297, &actor_W0[297])
PARAM_ADD(PARAM_FLOAT16, W0_298, &actor_W0[298])
PARAM_ADD(PARAM_FLOAT16, W0_299, &actor_W0[299])
PARAM_ADD(PARAM_FLOAT16, W0_300, &actor_W0[300])
PARAM_ADD(PARAM_FLOAT16, W0_301, &actor_W0[301])
PARAM_ADD(PARAM_FLOAT16, W0_302, &actor_W0[302])
PARAM_ADD(PARAM_FLOAT16, W0_303, &actor_W0[303])
PARAM_ADD(PARAM_FLOAT16, W0_304, &actor_W0[304])
PARAM_ADD(PARAM_FLOAT16, W0_305, &actor_W0[305])
PARAM_ADD(PARAM_FLOAT16, W0_306, &actor_W0[306])
PARAM_ADD(PARAM_FLOAT16, W0_307, &actor_W0[307])
PARAM_ADD(PARAM_FLOAT16, W0_308, &actor_W0[308])
PARAM_ADD(PARAM_FLOAT16, W0_309, &actor_W0[309])
PARAM_ADD(PARAM_FLOAT16, W0_310, &actor_W0[310])
PARAM_ADD(PARAM_FLOAT16, W0_311, &actor_W0[311])
PARAM_ADD(PARAM_FLOAT16, W0_312, &actor_W0[312])
PARAM_ADD(PARAM_FLOAT16, W0_313, &actor_W0[313])
PARAM_ADD(PARAM_FLOAT16, W0_314, &actor_W0[314])
PARAM_ADD(PARAM_FLOAT16, W0_315, &actor_W0[315])
PARAM_ADD(PARAM_FLOAT16, W0_316, &actor_W0[316])
PARAM_ADD(PARAM_FLOAT16, W0_317, &actor_W0[317])
PARAM_ADD(PARAM_FLOAT16, W0_318, &actor_W0[318])
PARAM_ADD(PARAM_FLOAT16, W0_319, &actor_W0[319])
// Layer 0 biases: 16 params
PARAM_ADD(PARAM_FLOAT16, b0_00, &actor_b0[0])
PARAM_ADD(PARAM_FLOAT16, b0_01, &actor_b0[1])
PARAM_ADD(PARAM_FLOAT16, b0_02, &actor_b0[2])
PARAM_ADD(PARAM_FLOAT16, b0_03, &actor_b0[3])
PARAM_ADD(PARAM_FLOAT16, b0_04, &actor_b0[4])
PARAM_ADD(PARAM_FLOAT16, b0_05, &actor_b0[5])
PARAM_ADD(PARAM_FLOAT16, b0_06, &actor_b0[6])
PARAM_ADD(PARAM_FLOAT16, b0_07, &actor_b0[7])
PARAM_ADD(PARAM_FLOAT16, b0_08, &actor_b0[8])
PARAM_ADD(PARAM_FLOAT16, b0_09, &actor_b0[9])
PARAM_ADD(PARAM_FLOAT16, b0_10, &actor_b0[10])
PARAM_ADD(PARAM_FLOAT16, b0_11, &actor_b0[11])
PARAM_ADD(PARAM_FLOAT16, b0_12, &actor_b0[12])
PARAM_ADD(PARAM_FLOAT16, b0_13, &actor_b0[13])
PARAM_ADD(PARAM_FLOAT16, b0_14, &actor_b0[14])
PARAM_ADD(PARAM_FLOAT16, b0_15, &actor_b0[15])
// Layer 1 weights: 256 params (16x16 matrix stored row-major)
PARAM_ADD(PARAM_FLOAT16, W1_000, &actor_W1[0])
PARAM_ADD(PARAM_FLOAT16, W1_001, &actor_W1[1])
PARAM_ADD(PARAM_FLOAT16, W1_002, &actor_W1[2])
PARAM_ADD(PARAM_FLOAT16, W1_003, &actor_W1[3])
PARAM_ADD(PARAM_FLOAT16, W1_004, &actor_W1[4])
PARAM_ADD(PARAM_FLOAT16, W1_005, &actor_W1[5])
PARAM_ADD(PARAM_FLOAT16, W1_006, &actor_W1[6])
PARAM_ADD(PARAM_FLOAT16, W1_007, &actor_W1[7])
PARAM_ADD(PARAM_FLOAT16, W1_008, &actor_W1[8])
PARAM_ADD(PARAM_FLOAT16, W1_009, &actor_W1[9])
PARAM_ADD(PARAM_FLOAT16, W1_010, &actor_W1[10])
PARAM_ADD(PARAM_FLOAT16, W1_011, &actor_W1[11])
PARAM_ADD(PARAM_FLOAT16, W1_012, &actor_W1[12])
PARAM_ADD(PARAM_FLOAT16, W1_013, &actor_W1[13])
PARAM_ADD(PARAM_FLOAT16, W1_014, &actor_W1[14])
PARAM_ADD(PARAM_FLOAT16, W1_015, &actor_W1[15])
PARAM_ADD(PARAM_FLOAT16, W1_016, &actor_W1[16])
PARAM_ADD(PARAM_FLOAT16, W1_017, &actor_W1[17])
PARAM_ADD(PARAM_FLOAT16, W1_018, &actor_W1[18])
PARAM_ADD(PARAM_FLOAT16, W1_019, &actor_W1[19])
PARAM_ADD(PARAM_FLOAT16, W1_020, &actor_W1[20])
PARAM_ADD(PARAM_FLOAT16, W1_021, &actor_W1[21])
PARAM_ADD(PARAM_FLOAT16, W1_022, &actor_W1[22])
PARAM_ADD(PARAM_FLOAT16, W1_023, &actor_W1[23])
PARAM_ADD(PARAM_FLOAT16, W1_024, &actor_W1[24])
PARAM_ADD(PARAM_FLOAT16, W1_025, &actor_W1[25])
PARAM_ADD(PARAM_FLOAT16, W1_026, &actor_W1[26])
PARAM_ADD(PARAM_FLOAT16, W1_027, &actor_W1[27])
PARAM_ADD(PARAM_FLOAT16, W1_028, &actor_W1[28])
PARAM_ADD(PARAM_FLOAT16, W1_029, &actor_W1[29])
PARAM_ADD(PARAM_FLOAT16, W1_030, &actor_W1[30])
PARAM_ADD(PARAM_FLOAT16, W1_031, &actor_W1[31])
PARAM_ADD(PARAM_FLOAT16, W1_032, &actor_W1[32])
PARAM_ADD(PARAM_FLOAT16, W1_033, &actor_W1[33])
PARAM_ADD(PARAM_FLOAT16, W1_034, &actor_W1[34])
PARAM_ADD(PARAM_FLOAT16, W1_035, &actor_W1[35])
PARAM_ADD(PARAM_FLOAT16, W1_036, &actor_W1[36])
PARAM_ADD(PARAM_FLOAT16, W1_037, &actor_W1[37])
PARAM_ADD(PARAM_FLOAT16, W1_038, &actor_W1[38])
PARAM_ADD(PARAM_FLOAT16, W1_039, &actor_W1[39])
PARAM_ADD(PARAM_FLOAT16, W1_040, &actor_W1[40])
PARAM_ADD(PARAM_FLOAT16, W1_041, &actor_W1[41])
PARAM_ADD(PARAM_FLOAT16, W1_042, &actor_W1[42])
PARAM_ADD(PARAM_FLOAT16, W1_043, &actor_W1[43])
PARAM_ADD(PARAM_FLOAT16, W1_044, &actor_W1[44])
PARAM_ADD(PARAM_FLOAT16, W1_045, &actor_W1[45])
PARAM_ADD(PARAM_FLOAT16, W1_046, &actor_W1[46])
PARAM_ADD(PARAM_FLOAT16, W1_047, &actor_W1[47])
PARAM_ADD(PARAM_FLOAT16, W1_048, &actor_W1[48])
PARAM_ADD(PARAM_FLOAT16, W1_049, &actor_W1[49])
PARAM_ADD(PARAM_FLOAT16, W1_050, &actor_W1[50])
PARAM_ADD(PARAM_FLOAT16, W1_051, &actor_W1[51])
PARAM_ADD(PARAM_FLOAT16, W1_052, &actor_W1[52])
PARAM_ADD(PARAM_FLOAT16, W1_053, &actor_W1[53])
PARAM_ADD(PARAM_FLOAT16, W1_054, &actor_W1[54])
PARAM_ADD(PARAM_FLOAT16, W1_055, &actor_W1[55])
PARAM_ADD(PARAM_FLOAT16, W1_056, &actor_W1[56])
PARAM_ADD(PARAM_FLOAT16, W1_057, &actor_W1[57])
PARAM_ADD(PARAM_FLOAT16, W1_058, &actor_W1[58])
PARAM_ADD(PARAM_FLOAT16, W1_059, &actor_W1[59])
PARAM_ADD(PARAM_FLOAT16, W1_060, &actor_W1[60])
PARAM_ADD(PARAM_FLOAT16, W1_061, &actor_W1[61])
PARAM_ADD(PARAM_FLOAT16, W1_062, &actor_W1[62])
PARAM_ADD(PARAM_FLOAT16, W1_063, &actor_W1[63])
PARAM_ADD(PARAM_FLOAT16, W1_064, &actor_W1[64])
PARAM_ADD(PARAM_FLOAT16, W1_065, &actor_W1[65])
PARAM_ADD(PARAM_FLOAT16, W1_066, &actor_W1[66])
PARAM_ADD(PARAM_FLOAT16, W1_067, &actor_W1[67])
PARAM_ADD(PARAM_FLOAT16, W1_068, &actor_W1[68])
PARAM_ADD(PARAM_FLOAT16, W1_069, &actor_W1[69])
PARAM_ADD(PARAM_FLOAT16, W1_070, &actor_W1[70])
PARAM_ADD(PARAM_FLOAT16, W1_071, &actor_W1[71])
PARAM_ADD(PARAM_FLOAT16, W1_072, &actor_W1[72])
PARAM_ADD(PARAM_FLOAT16, W1_073, &actor_W1[73])
PARAM_ADD(PARAM_FLOAT16, W1_074, &actor_W1[74])
PARAM_ADD(PARAM_FLOAT16, W1_075, &actor_W1[75])
PARAM_ADD(PARAM_FLOAT16, W1_076, &actor_W1[76])
PARAM_ADD(PARAM_FLOAT16, W1_077, &actor_W1[77])
PARAM_ADD(PARAM_FLOAT16, W1_078, &actor_W1[78])
PARAM_ADD(PARAM_FLOAT16, W1_079, &actor_W1[79])
PARAM_ADD(PARAM_FLOAT16, W1_080, &actor_W1[80])
PARAM_ADD(PARAM_FLOAT16, W1_081, &actor_W1[81])
PARAM_ADD(PARAM_FLOAT16, W1_082, &actor_W1[82])
PARAM_ADD(PARAM_FLOAT16, W1_083, &actor_W1[83])
PARAM_ADD(PARAM_FLOAT16, W1_084, &actor_W1[84])
PARAM_ADD(PARAM_FLOAT16, W1_085, &actor_W1[85])
PARAM_ADD(PARAM_FLOAT16, W1_086, &actor_W1[86])
PARAM_ADD(PARAM_FLOAT16, W1_087, &actor_W1[87])
PARAM_ADD(PARAM_FLOAT16, W1_088, &actor_W1[88])
PARAM_ADD(PARAM_FLOAT16, W1_089, &actor_W1[89])
PARAM_ADD(PARAM_FLOAT16, W1_090, &actor_W1[90])
PARAM_ADD(PARAM_FLOAT16, W1_091, &actor_W1[91])
PARAM_ADD(PARAM_FLOAT16, W1_092, &actor_W1[92])
PARAM_ADD(PARAM_FLOAT16, W1_093, &actor_W1[93])
PARAM_ADD(PARAM_FLOAT16, W1_094, &actor_W1[94])
PARAM_ADD(PARAM_FLOAT16, W1_095, &actor_W1[95])
PARAM_ADD(PARAM_FLOAT16, W1_096, &actor_W1[96])
PARAM_ADD(PARAM_FLOAT16, W1_097, &actor_W1[97])
PARAM_ADD(PARAM_FLOAT16, W1_098, &actor_W1[98])
PARAM_ADD(PARAM_FLOAT16, W1_099, &actor_W1[99])
PARAM_ADD(PARAM_FLOAT16, W1_100, &actor_W1[100])
PARAM_ADD(PARAM_FLOAT16, W1_101, &actor_W1[101])
PARAM_ADD(PARAM_FLOAT16, W1_102, &actor_W1[102])
PARAM_ADD(PARAM_FLOAT16, W1_103, &actor_W1[103])
PARAM_ADD(PARAM_FLOAT16, W1_104, &actor_W1[104])
PARAM_ADD(PARAM_FLOAT16, W1_105, &actor_W1[105])
PARAM_ADD(PARAM_FLOAT16, W1_106, &actor_W1[106])
PARAM_ADD(PARAM_FLOAT16, W1_107, &actor_W1[107])
PARAM_ADD(PARAM_FLOAT16, W1_108, &actor_W1[108])
PARAM_ADD(PARAM_FLOAT16, W1_109, &actor_W1[109])
PARAM_ADD(PARAM_FLOAT16, W1_110, &actor_W1[110])
PARAM_ADD(PARAM_FLOAT16, W1_111, &actor_W1[111])
PARAM_ADD(PARAM_FLOAT16, W1_112, &actor_W1[112])
PARAM_ADD(PARAM_FLOAT16, W1_113, &actor_W1[113])
PARAM_ADD(PARAM_FLOAT16, W1_114, &actor_W1[114])
PARAM_ADD(PARAM_FLOAT16, W1_115, &actor_W1[115])
PARAM_ADD(PARAM_FLOAT16, W1_116, &actor_W1[116])
PARAM_ADD(PARAM_FLOAT16, W1_117, &actor_W1[117])
PARAM_ADD(PARAM_FLOAT16, W1_118, &actor_W1[118])
PARAM_ADD(PARAM_FLOAT16, W1_119, &actor_W1[119])
PARAM_ADD(PARAM_FLOAT16, W1_120, &actor_W1[120])
PARAM_ADD(PARAM_FLOAT16, W1_121, &actor_W1[121])
PARAM_ADD(PARAM_FLOAT16, W1_122, &actor_W1[122])
PARAM_ADD(PARAM_FLOAT16, W1_123, &actor_W1[123])
PARAM_ADD(PARAM_FLOAT16, W1_124, &actor_W1[124])
PARAM_ADD(PARAM_FLOAT16, W1_125, &actor_W1[125])
PARAM_ADD(PARAM_FLOAT16, W1_126, &actor_W1[126])
PARAM_ADD(PARAM_FLOAT16, W1_127, &actor_W1[127])
PARAM_ADD(PARAM_FLOAT16, W1_128, &actor_W1[128])
PARAM_ADD(PARAM_FLOAT16, W1_129, &actor_W1[129])
PARAM_ADD(PARAM_FLOAT16, W1_130, &actor_W1[130])
PARAM_ADD(PARAM_FLOAT16, W1_131, &actor_W1[131])
PARAM_ADD(PARAM_FLOAT16, W1_132, &actor_W1[132])
PARAM_ADD(PARAM_FLOAT16, W1_133, &actor_W1[133])
PARAM_ADD(PARAM_FLOAT16, W1_134, &actor_W1[134])
PARAM_ADD(PARAM_FLOAT16, W1_135, &actor_W1[135])
PARAM_ADD(PARAM_FLOAT16, W1_136, &actor_W1[136])
PARAM_ADD(PARAM_FLOAT16, W1_137, &actor_W1[137])
PARAM_ADD(PARAM_FLOAT16, W1_138, &actor_W1[138])
PARAM_ADD(PARAM_FLOAT16, W1_139, &actor_W1[139])
PARAM_ADD(PARAM_FLOAT16, W1_140, &actor_W1[140])
PARAM_ADD(PARAM_FLOAT16, W1_141, &actor_W1[141])
PARAM_ADD(PARAM_FLOAT16, W1_142, &actor_W1[142])
PARAM_ADD(PARAM_FLOAT16, W1_143, &actor_W1[143])
PARAM_ADD(PARAM_FLOAT16, W1_144, &actor_W1[144])
PARAM_ADD(PARAM_FLOAT16, W1_145, &actor_W1[145])
PARAM_ADD(PARAM_FLOAT16, W1_146, &actor_W1[146])
PARAM_ADD(PARAM_FLOAT16, W1_147, &actor_W1[147])
PARAM_ADD(PARAM_FLOAT16, W1_148, &actor_W1[148])
PARAM_ADD(PARAM_FLOAT16, W1_149, &actor_W1[149])
PARAM_ADD(PARAM_FLOAT16, W1_150, &actor_W1[150])
PARAM_ADD(PARAM_FLOAT16, W1_151, &actor_W1[151])
PARAM_ADD(PARAM_FLOAT16, W1_152, &actor_W1[152])
PARAM_ADD(PARAM_FLOAT16, W1_153, &actor_W1[153])
PARAM_ADD(PARAM_FLOAT16, W1_154, &actor_W1[154])
PARAM_ADD(PARAM_FLOAT16, W1_155, &actor_W1[155])
PARAM_ADD(PARAM_FLOAT16, W1_156, &actor_W1[156])
PARAM_ADD(PARAM_FLOAT16, W1_157, &actor_W1[157])
PARAM_ADD(PARAM_FLOAT16, W1_158, &actor_W1[158])
PARAM_ADD(PARAM_FLOAT16, W1_159, &actor_W1[159])
PARAM_ADD(PARAM_FLOAT16, W1_160, &actor_W1[160])
PARAM_ADD(PARAM_FLOAT16, W1_161, &actor_W1[161])
PARAM_ADD(PARAM_FLOAT16, W1_162, &actor_W1[162])
PARAM_ADD(PARAM_FLOAT16, W1_163, &actor_W1[163])
PARAM_ADD(PARAM_FLOAT16, W1_164, &actor_W1[164])
PARAM_ADD(PARAM_FLOAT16, W1_165, &actor_W1[165])
PARAM_ADD(PARAM_FLOAT16, W1_166, &actor_W1[166])
PARAM_ADD(PARAM_FLOAT16, W1_167, &actor_W1[167])
PARAM_ADD(PARAM_FLOAT16, W1_168, &actor_W1[168])
PARAM_ADD(PARAM_FLOAT16, W1_169, &actor_W1[169])
PARAM_ADD(PARAM_FLOAT16, W1_170, &actor_W1[170])
PARAM_ADD(PARAM_FLOAT16, W1_171, &actor_W1[171])
PARAM_ADD(PARAM_FLOAT16, W1_172, &actor_W1[172])
PARAM_ADD(PARAM_FLOAT16, W1_173, &actor_W1[173])
PARAM_ADD(PARAM_FLOAT16, W1_174, &actor_W1[174])
PARAM_ADD(PARAM_FLOAT16, W1_175, &actor_W1[175])
PARAM_ADD(PARAM_FLOAT16, W1_176, &actor_W1[176])
PARAM_ADD(PARAM_FLOAT16, W1_177, &actor_W1[177])
PARAM_ADD(PARAM_FLOAT16, W1_178, &actor_W1[178])
PARAM_ADD(PARAM_FLOAT16, W1_179, &actor_W1[179])
PARAM_ADD(PARAM_FLOAT16, W1_180, &actor_W1[180])
PARAM_ADD(PARAM_FLOAT16, W1_181, &actor_W1[181])
PARAM_ADD(PARAM_FLOAT16, W1_182, &actor_W1[182])
PARAM_ADD(PARAM_FLOAT16, W1_183, &actor_W1[183])
PARAM_ADD(PARAM_FLOAT16, W1_184, &actor_W1[184])
PARAM_ADD(PARAM_FLOAT16, W1_185, &actor_W1[185])
PARAM_ADD(PARAM_FLOAT16, W1_186, &actor_W1[186])
PARAM_ADD(PARAM_FLOAT16, W1_187, &actor_W1[187])
PARAM_ADD(PARAM_FLOAT16, W1_188, &actor_W1[188])
PARAM_ADD(PARAM_FLOAT16, W1_189, &actor_W1[189])
PARAM_ADD(PARAM_FLOAT16, W1_190, &actor_W1[190])
PARAM_ADD(PARAM_FLOAT16, W1_191, &actor_W1[191])
PARAM_ADD(PARAM_FLOAT16, W1_192, &actor_W1[192])
PARAM_ADD(PARAM_FLOAT16, W1_193, &actor_W1[193])
PARAM_ADD(PARAM_FLOAT16, W1_194, &actor_W1[194])
PARAM_ADD(PARAM_FLOAT16, W1_195, &actor_W1[195])
PARAM_ADD(PARAM_FLOAT16, W1_196, &actor_W1[196])
PARAM_ADD(PARAM_FLOAT16, W1_197, &actor_W1[197])
PARAM_ADD(PARAM_FLOAT16, W1_198, &actor_W1[198])
PARAM_ADD(PARAM_FLOAT16, W1_199, &actor_W1[199])
PARAM_ADD(PARAM_FLOAT16, W1_200, &actor_W1[200])
PARAM_ADD(PARAM_FLOAT16, W1_201, &actor_W1[201])
PARAM_ADD(PARAM_FLOAT16, W1_202, &actor_W1[202])
PARAM_ADD(PARAM_FLOAT16, W1_203, &actor_W1[203])
PARAM_ADD(PARAM_FLOAT16, W1_204, &actor_W1[204])
PARAM_ADD(PARAM_FLOAT16, W1_205, &actor_W1[205])
PARAM_ADD(PARAM_FLOAT16, W1_206, &actor_W1[206])
PARAM_ADD(PARAM_FLOAT16, W1_207, &actor_W1[207])
PARAM_ADD(PARAM_FLOAT16, W1_208, &actor_W1[208])
PARAM_ADD(PARAM_FLOAT16, W1_209, &actor_W1[209])
PARAM_ADD(PARAM_FLOAT16, W1_210, &actor_W1[210])
PARAM_ADD(PARAM_FLOAT16, W1_211, &actor_W1[211])
PARAM_ADD(PARAM_FLOAT16, W1_212, &actor_W1[212])
PARAM_ADD(PARAM_FLOAT16, W1_213, &actor_W1[213])
PARAM_ADD(PARAM_FLOAT16, W1_214, &actor_W1[214])
PARAM_ADD(PARAM_FLOAT16, W1_215, &actor_W1[215])
PARAM_ADD(PARAM_FLOAT16, W1_216, &actor_W1[216])
PARAM_ADD(PARAM_FLOAT16, W1_217, &actor_W1[217])
PARAM_ADD(PARAM_FLOAT16, W1_218, &actor_W1[218])
PARAM_ADD(PARAM_FLOAT16, W1_219, &actor_W1[219])
PARAM_ADD(PARAM_FLOAT16, W1_220, &actor_W1[220])
PARAM_ADD(PARAM_FLOAT16, W1_221, &actor_W1[221])
PARAM_ADD(PARAM_FLOAT16, W1_222, &actor_W1[222])
PARAM_ADD(PARAM_FLOAT16, W1_223, &actor_W1[223])
PARAM_ADD(PARAM_FLOAT16, W1_224, &actor_W1[224])
PARAM_ADD(PARAM_FLOAT16, W1_225, &actor_W1[225])
PARAM_ADD(PARAM_FLOAT16, W1_226, &actor_W1[226])
PARAM_ADD(PARAM_FLOAT16, W1_227, &actor_W1[227])
PARAM_ADD(PARAM_FLOAT16, W1_228, &actor_W1[228])
PARAM_ADD(PARAM_FLOAT16, W1_229, &actor_W1[229])
PARAM_ADD(PARAM_FLOAT16, W1_230, &actor_W1[230])
PARAM_ADD(PARAM_FLOAT16, W1_231, &actor_W1[231])
PARAM_ADD(PARAM_FLOAT16, W1_232, &actor_W1[232])
PARAM_ADD(PARAM_FLOAT16, W1_233, &actor_W1[233])
PARAM_ADD(PARAM_FLOAT16, W1_234, &actor_W1[234])
PARAM_ADD(PARAM_FLOAT16, W1_235, &actor_W1[235])
PARAM_ADD(PARAM_FLOAT16, W1_236, &actor_W1[236])
PARAM_ADD(PARAM_FLOAT16, W1_237, &actor_W1[237])
PARAM_ADD(PARAM_FLOAT16, W1_238, &actor_W1[238])
PARAM_ADD(PARAM_FLOAT16, W1_239, &actor_W1[239])
PARAM_ADD(PARAM_FLOAT16, W1_240, &actor_W1[240])
PARAM_ADD(PARAM_FLOAT16, W1_241, &actor_W1[241])
PARAM_ADD(PARAM_FLOAT16, W1_242, &actor_W1[242])
PARAM_ADD(PARAM_FLOAT16, W1_243, &actor_W1[243])
PARAM_ADD(PARAM_FLOAT16, W1_244, &actor_W1[244])
PARAM_ADD(PARAM_FLOAT16, W1_245, &actor_W1[245])
PARAM_ADD(PARAM_FLOAT16, W1_246, &actor_W1[246])
PARAM_ADD(PARAM_FLOAT16, W1_247, &actor_W1[247])
PARAM_ADD(PARAM_FLOAT16, W1_248, &actor_W1[248])
PARAM_ADD(PARAM_FLOAT16, W1_249, &actor_W1[249])
PARAM_ADD(PARAM_FLOAT16, W1_250, &actor_W1[250])
PARAM_ADD(PARAM_FLOAT16, W1_251, &actor_W1[251])
PARAM_ADD(PARAM_FLOAT16, W1_252, &actor_W1[252])
PARAM_ADD(PARAM_FLOAT16, W1_253, &actor_W1[253])
PARAM_ADD(PARAM_FLOAT16, W1_254, &actor_W1[254])
PARAM_ADD(PARAM_FLOAT16, W1_255, &actor_W1[255])
// Layer 1 biases: 16 params
PARAM_ADD(PARAM_FLOAT16, b1_00, &actor_b1[0])
PARAM_ADD(PARAM_FLOAT16, b1_01, &actor_b1[1])
PARAM_ADD(PARAM_FLOAT16, b1_02, &actor_b1[2])
PARAM_ADD(PARAM_FLOAT16, b1_03, &actor_b1[3])
PARAM_ADD(PARAM_FLOAT16, b1_04, &actor_b1[4])
PARAM_ADD(PARAM_FLOAT16, b1_05, &actor_b1[5])
PARAM_ADD(PARAM_FLOAT16, b1_06, &actor_b1[6])
PARAM_ADD(PARAM_FLOAT16, b1_07, &actor_b1[7])
PARAM_ADD(PARAM_FLOAT16, b1_08, &actor_b1[8])
PARAM_ADD(PARAM_FLOAT16, b1_09, &actor_b1[9])
PARAM_ADD(PARAM_FLOAT16, b1_10, &actor_b1[10])
PARAM_ADD(PARAM_FLOAT16, b1_11, &actor_b1[11])
PARAM_ADD(PARAM_FLOAT16, b1_12, &actor_b1[12])
PARAM_ADD(PARAM_FLOAT16, b1_13, &actor_b1[13])
PARAM_ADD(PARAM_FLOAT16, b1_14, &actor_b1[14])
PARAM_ADD(PARAM_FLOAT16, b1_15, &actor_b1[15])
// Layer 2 weights: 64 params (16x4 matrix stored row-major)
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
PARAM_ADD(PARAM_FLOAT16, W2_32, &actor_W2[32])
PARAM_ADD(PARAM_FLOAT16, W2_33, &actor_W2[33])
PARAM_ADD(PARAM_FLOAT16, W2_34, &actor_W2[34])
PARAM_ADD(PARAM_FLOAT16, W2_35, &actor_W2[35])
PARAM_ADD(PARAM_FLOAT16, W2_36, &actor_W2[36])
PARAM_ADD(PARAM_FLOAT16, W2_37, &actor_W2[37])
PARAM_ADD(PARAM_FLOAT16, W2_38, &actor_W2[38])
PARAM_ADD(PARAM_FLOAT16, W2_39, &actor_W2[39])
PARAM_ADD(PARAM_FLOAT16, W2_40, &actor_W2[40])
PARAM_ADD(PARAM_FLOAT16, W2_41, &actor_W2[41])
PARAM_ADD(PARAM_FLOAT16, W2_42, &actor_W2[42])
PARAM_ADD(PARAM_FLOAT16, W2_43, &actor_W2[43])
PARAM_ADD(PARAM_FLOAT16, W2_44, &actor_W2[44])
PARAM_ADD(PARAM_FLOAT16, W2_45, &actor_W2[45])
PARAM_ADD(PARAM_FLOAT16, W2_46, &actor_W2[46])
PARAM_ADD(PARAM_FLOAT16, W2_47, &actor_W2[47])
PARAM_ADD(PARAM_FLOAT16, W2_48, &actor_W2[48])
PARAM_ADD(PARAM_FLOAT16, W2_49, &actor_W2[49])
PARAM_ADD(PARAM_FLOAT16, W2_50, &actor_W2[50])
PARAM_ADD(PARAM_FLOAT16, W2_51, &actor_W2[51])
PARAM_ADD(PARAM_FLOAT16, W2_52, &actor_W2[52])
PARAM_ADD(PARAM_FLOAT16, W2_53, &actor_W2[53])
PARAM_ADD(PARAM_FLOAT16, W2_54, &actor_W2[54])
PARAM_ADD(PARAM_FLOAT16, W2_55, &actor_W2[55])
PARAM_ADD(PARAM_FLOAT16, W2_56, &actor_W2[56])
PARAM_ADD(PARAM_FLOAT16, W2_57, &actor_W2[57])
PARAM_ADD(PARAM_FLOAT16, W2_58, &actor_W2[58])
PARAM_ADD(PARAM_FLOAT16, W2_59, &actor_W2[59])
PARAM_ADD(PARAM_FLOAT16, W2_60, &actor_W2[60])
PARAM_ADD(PARAM_FLOAT16, W2_61, &actor_W2[61])
PARAM_ADD(PARAM_FLOAT16, W2_62, &actor_W2[62])
PARAM_ADD(PARAM_FLOAT16, W2_63, &actor_W2[63])
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
