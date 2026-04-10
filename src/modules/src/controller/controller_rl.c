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
#include "platform_defaults.h"
#include "math3d.h"
#include "rl_policy_params.h"

// =========================
// Deployment/training config
// =========================
// TRAIN_Hz: policy training/control rate used to define action-history spacing.
// DEPLOY_Hz: runtime controller execution rate in firmware.
// N_LAST_ACTIONS: number of historical actions to include in observation.
// ACT_DIM: action vector dimension.
#define TRAIN_Hz 100
#define DEPLOY_Hz 500
#define N_LAST_ACTIONS 4
#define ACT_DIM 4
#define RATIO (DEPLOY_Hz / TRAIN_Hz)

#define LAST_ACTION_BUFFER_LEN (RATIO * N_LAST_ACTIONS)
#define LAST_ACTIONS_OBS_DIM (N_LAST_ACTIONS * ACT_DIM)

#define USE_QUAT 1

// Observation layout indices
#define OBS_IDX_ANG_VEL 0
#define OBS_IDX_LAST_ACTIONS (OBS_IDX_ANG_VEL + 3)
#define OBS_IDX_POS (OBS_IDX_LAST_ACTIONS + LAST_ACTIONS_OBS_DIM)

#if USE_QUAT
#define OBS_IDX_QUAT (OBS_IDX_POS + 3)
#define OBS_IDX_VEL (OBS_IDX_QUAT + 4)
#define OBS_DIM (OBS_IDX_VEL + 3)
#else
#define OBS_IDX_ROT_MAT (OBS_IDX_POS + 3)
#define OBS_IDX_VEL (OBS_IDX_ROT_MAT + 9)
#define OBS_DIM (OBS_IDX_VEL + 3)
#endif


// =========================
// Drone parameters (const)
// =========================

// Controller Parameters
static int ctrl_freq_hz = DEPLOY_Hz;  // controller frequency

// thrust = a0 + a1 * rpm + a2 * rpm^2
// Values are selected at compile time by platform macro
#if defined(CONFIG_PLATFORM_CF21BL)
static const float RPM2THRUST_A0 = 0.0f;
static const float RPM2THRUST_A1 = -3.133427287299859e-7f;
static const float RPM2THRUST_A2 =  4.407354891648379e-10f;
#elif defined(CONFIG_PLATFORM_CF2)
static const float RPM2THRUST_A0 = 0.0f;
static const float RPM2THRUST_A1 = -5.382196214637237e-7f;
static const float RPM2THRUST_A2 =  2.4582929831265485e-10f;
#else
static const float RPM2THRUST_A0 = 0.0f;
static const float RPM2THRUST_A1 = -3.133427287299859e-7f;
static const float RPM2THRUST_A2 =  4.407354891648379e-10f;
#endif

// Action scaling: policy output in [-1, 1] -> rotor RPM in [ROTOR_RPM_MIN, ROTOR_RPM_MAX]
// Values are selected at compile time by platform macro
// TODO: Compute from known min/max thrust and the thrust curve instead of hardcoding
#if defined(CONFIG_PLATFORM_CF21BL)
static const float ROTOR_RPM_MIN = 6962.07f;
static const float ROTOR_RPM_MAX = 21302.27f;
#elif defined(CONFIG_PLATFORM_CF2)
static const float ROTOR_RPM_MIN = 7220.81f;
static const float ROTOR_RPM_MAX = 22093.97f;
#else
static const float ROTOR_RPM_MIN = 6962.07f;
static const float ROTOR_RPM_MAX = 21302.27f;
#endif

// static const float ROTOR_RPM_K0  = (ROTOR_RPM_MAX + ROTOR_RPM_MIN) * 0.5f;
// static const float ROTOR_RPM_K1 = (ROTOR_RPM_MAX - ROTOR_RPM_MIN) * 0.5f;
// static const float ROTOR_RPM_K2  = 0.0f;
static const float ROTOR_RPM_K0 = 18967.0f;
static const float ROTOR_RPM_K1 = 3126.9727f;
static const float ROTOR_RPM_K2 = -1563.4863f;

// Unit conversion
static const float DEG2RAD = 0.01745329251994329577f;  // pi/180


// =========================
// Internal state
// =========================
// [0] is most recent action, increasing index is older actions.
static float g_last_actions_buffer[LAST_ACTION_BUFFER_LEN][ACT_DIM];


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
// Python obs dict:
//   pos (3), quat (4), vel (3), ang_vel (3), last_actions (N_LAST_ACTIONS*ACT_DIM)
// After FlattenObs with alphabetical keys:
//   "ang_vel"(3), "last_actions"(...), "pos"(3), "quat"(4), "vel"(3)
static void build_obs(float obs[OBS_DIM],
                         const setpoint_t* setpoint,
                         const sensorData_t* sensors,
                         const state_t* state) {
  // goal_pos from setpoint
  const float goal_x = setpoint->position.x;
  const float goal_y = setpoint->position.y;
  const float goal_z = setpoint->position.z;
  const float goal_vx = setpoint->velocity.x;
  const float goal_vy = setpoint->velocity.y;
  const float goal_vz = setpoint->velocity.z;

  // pos, vel from state (apply commands as offsets)
  const float px = state->position.x - goal_x;
  const float py = state->position.y - goal_y;
  const float pz = state->position.z - goal_z;

  const float vx = state->velocity.x - goal_vx;
  const float vy = state->velocity.y - goal_vy;
  const float vz = state->velocity.z - goal_vz;

  // quat from state->attitudeQuaternion, sequence xyzw (as you confirmed)
  const float qx = state->attitudeQuaternion.x;
  const float qy = state->attitudeQuaternion.y;
  const float qz = state->attitudeQuaternion.z;
  const float qw = state->attitudeQuaternion.w;

  // ang_vel from gyro, deg/s -> rad/s
  const float wx = sensors->gyro.x * DEG2RAD;
  const float wy = sensors->gyro.y * DEG2RAD;
  const float wz = sensors->gyro.z * DEG2RAD;

  // Fill in strict alphabetical order:
  // ang_vel
  obs[OBS_IDX_ANG_VEL + 0] = wx;
  obs[OBS_IDX_ANG_VEL + 1] = wy;
  obs[OBS_IDX_ANG_VEL + 2] = wz;

  // last_actions sampled at indices [0, RATIO, ..., (N_LAST_ACTIONS-1)*RATIO]
  int obs_idx = OBS_IDX_LAST_ACTIONS;
  for (int k = 0; k < N_LAST_ACTIONS; ++k) {
    const int sample_idx = k * RATIO;
    for (int a = 0; a < ACT_DIM; ++a) {
      obs[obs_idx++] = g_last_actions_buffer[sample_idx][a];
    }
  }

  // pos
  obs[OBS_IDX_POS + 0] = px;
  obs[OBS_IDX_POS + 1] = py;
  obs[OBS_IDX_POS + 2] = pz;

#if USE_QUAT
  // quat (xyzw)
  obs[OBS_IDX_QUAT + 0] = qx;
  obs[OBS_IDX_QUAT + 1] = qy;
  obs[OBS_IDX_QUAT + 2] = qz;
  obs[OBS_IDX_QUAT + 3] = qw;

  // vel
  obs[OBS_IDX_VEL + 0] = vx;
  obs[OBS_IDX_VEL + 1] = vy;
  obs[OBS_IDX_VEL + 2] = vz;
#else
  // rot_mat (9), row-major from quaternion
  const struct quat q = mkquat(qx, qy, qz, qw);
  const struct mat33 R = quat2rotmat(q);
  obs[OBS_IDX_ROT_MAT + 0] = R.m[0][0];
  obs[OBS_IDX_ROT_MAT + 1] = R.m[0][1];
  obs[OBS_IDX_ROT_MAT + 2] = R.m[0][2];
  obs[OBS_IDX_ROT_MAT + 3] = R.m[1][0];
  obs[OBS_IDX_ROT_MAT + 4] = R.m[1][1];
  obs[OBS_IDX_ROT_MAT + 5] = R.m[1][2];
  obs[OBS_IDX_ROT_MAT + 6] = R.m[2][0];
  obs[OBS_IDX_ROT_MAT + 7] = R.m[2][1];
  obs[OBS_IDX_ROT_MAT + 8] = R.m[2][2];

  // vel
  obs[OBS_IDX_VEL + 0] = vx;
  obs[OBS_IDX_VEL + 1] = vy;
  obs[OBS_IDX_VEL + 2] = vz;
#endif
}

// =========================
// Policy forward
// =========================
static void policy_forward(const float obs[OBS_DIM], float action_out[ACTOR_OUTPUT_SIZE]) {
  // Expect 3 Dense layers: (20->H), (H->H), (H->4)
  // NOTE: No dynamic memory; use fixed-size buffers.
  // Hidden size read from header macros (ACTOR_L0_OUT), but we need compile-time max.
  float h0[ACTOR_L0_OUT];
  float h1[ACTOR_L1_OUT];

  // Layer 0
  dense_forward(obs, ACTOR_L0_IN, actor_W0, actor_b0, ACTOR_L0_OUT, h0);
  for (int i = 0; i < ACTOR_L0_OUT; ++i) {
    h0[i] = relu(h0[i]);
  }

  // Layer 1
  dense_forward(h0, ACTOR_L1_IN, actor_W1, actor_b1, ACTOR_L1_OUT, h1);
  for (int i = 0; i < ACTOR_L1_OUT; ++i) {
    h1[i] = relu(h1[i]);
  }

  // Output layer
  dense_forward(h1, ACTOR_L2_IN, actor_W2, actor_b2, ACTOR_L2_OUT, action_out);
  for (int i = 0; i < ACTOR_L2_OUT; ++i) {
    action_out[i] = tanh_act(action_out[i]);  // policy outputs rotor_vel directly after tanh
  }
}

// =========================
// Public API
// =========================
void controllerRLInit(void) {
  for (int i = 0; i < LAST_ACTION_BUFFER_LEN; ++i) {
    for (int a = 0; a < ACT_DIM; ++a) {
      g_last_actions_buffer[i][a] = 0.0f;
    }
  }
}

bool controllerRLTest(void) {
  return true;
}

void controllerRL(control_t *control, const setpoint_t *setpoint, const sensorData_t *sensors, const state_t *state, const stabilizerStep_t stabilizerStep) {
  control->controlMode = controlModePWM;

  // Run at ctrl_freq_hz Hz
  if (!RATE_DO_EXECUTE(ctrl_freq_hz, stabilizerStep)) {
    return;
  }

  // 1. Build observation vector with sampled action history
  float obs[OBS_DIM];
  build_obs(obs, setpoint, sensors, state);

  // 2. Policy forward -> rotor_vel actions in [-1, 1]
  float normalized_rotor_vel[ACTOR_OUTPUT_SIZE];
  policy_forward(obs, normalized_rotor_vel);

  // 3. Push newest action into history buffer.
  memmove(&g_last_actions_buffer[1][0],
          &g_last_actions_buffer[0][0],
          sizeof(float) * (LAST_ACTION_BUFFER_LEN - 1) * ACT_DIM);
  for (int a = 0; a < ACT_DIM; ++a) {
    g_last_actions_buffer[0][a] = normalized_rotor_vel[a];
  }

  // 4. Map policy output [-1,1] -> rotor RPM -> thrust -> normalized forces in [0,1]
  for (int i = 0; i < 4; ++i) {
    // Scale to RPM (same as Python):
    //   rpm = clip(a, -1, 1) * k0 + k1 * a + k2 * (a**2)
    const float rpm = ROTOR_RPM_K0 + normalized_rotor_vel[i] * ROTOR_RPM_K1 + ROTOR_RPM_K2 * normalized_rotor_vel[i] * normalized_rotor_vel[i];

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
 * Tunning variables for the full state RL Controller
 */
PARAM_GROUP_START(ctrlRL)
PARAM_ADD_CORE(PARAM_INT16 | PARAM_PERSISTENT, freq, &ctrl_freq_hz)
PARAM_GROUP_STOP(ctrlRL)

/**
 * Logging variables for the command and reference signals for the
 * RL controller
 */
LOG_GROUP_START(ctrlRL)
LOG_ADD(LOG_FLOAT, action_1, &g_last_actions_buffer[0][0])
LOG_ADD(LOG_FLOAT, action_2, &g_last_actions_buffer[0][1])
LOG_ADD(LOG_FLOAT, action_3, &g_last_actions_buffer[0][2])
LOG_ADD(LOG_FLOAT, action_4, &g_last_actions_buffer[0][3])
LOG_GROUP_STOP(ctrlRL)