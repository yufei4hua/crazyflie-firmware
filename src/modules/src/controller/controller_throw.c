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
static const int ctrl_freq_hz = 500;  // controller frequency (matches rotor_vel training: freq=250)

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
static void dense_forward(const float* x, int in_dim,
                          const float* W, const float* b, int out_dim,
                          float* y) {
  for (int j = 0; j < out_dim; ++j) {
    float s = b[j];
    for (int i = 0; i < in_dim; ++i) {
      s += x[i] * W[i * out_dim + j];
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
  g_last_actions[0] = 0.25f;
  g_last_actions[1] = 0.25f;
  g_last_actions[2] = 0.25f;
  g_last_actions[3] = 0.25f;
}

bool controllerThrowTest(void) {
  return true;
}

void controllerThrow(control_t *control, const setpoint_t *setpoint, const sensorData_t *sensors, const state_t *state, const stabilizerStep_t stabilizerStep) {
  control->controlMode = controlModePWM;

  if (!RATE_DO_EXECUTE(250, stabilizerStep)) {
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