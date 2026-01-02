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
#include "controller.h"
#include "rl_policy_params.h"


// =========================
// Drone parameters (const)
// =========================
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
static float g_last_actions[4] = {0.25f, 0.25f, 0.25f, 0.25f};  // rotor_vel (policy output), before scaling


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
// Python obs dict:
//   pos (3), quat (4), vel (3), ang_vel (3), last_actions (4)
// After FlattenObs with alphabetical keys:
//   "ang_vel"(3), "last_actions"(4), "pos"(3), "quat"(4), "vel"(3)  => 17 dims
static void build_obs(float obs[17],
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
  // 0..2   ang_vel
  obs[0] = wx;
  obs[1] = wy;
  obs[2] = wz;

  // 3..6   last_actions (policy output before scaling)
  obs[3] = g_last_actions[0];
  obs[4] = g_last_actions[1];
  obs[5] = g_last_actions[2];
  obs[6] = g_last_actions[3];

  // 7..9 pos
  obs[7] = px;
  obs[8] = py;
  obs[9] = pz;

  // 10..13 quat (xyzw)
  obs[10] = qx;
  obs[11] = qy;
  obs[12] = qz;
  obs[13] = qw;

  // 14..16 vel
  obs[14] = vx;
  obs[15] = vy;
  obs[16] = vz;
}

// =========================
// Policy forward
// =========================
static void policy_forward(const float obs[20], float action_out[4]) {
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
  return;
}

bool controllerRLTest(void) {
  return true;
}

void controllerRL(control_t *control, const setpoint_t *setpoint, const sensorData_t *sensors, const state_t *state, const uint32_t tick) {
  control->controlMode = controlModePWM;

  // Run at 500 Hz
  if (!RATE_DO_EXECUTE(100, tick)) {
    return;
  }

  // 1. Build observation vector (17-dim)
  float obs[17];
  build_obs(obs, setpoint, sensors, state);

  // 2. Policy forward -> rotor_vel actions in [-1, 1]
  float normalized_rotor_vel[4];
  policy_forward(obs, normalized_rotor_vel);

  // 3. Update last_actions (before scaling)
  g_last_actions[0] = normalized_rotor_vel[0];
  g_last_actions[1] = normalized_rotor_vel[1];
  g_last_actions[2] = normalized_rotor_vel[2];
  g_last_actions[3] = normalized_rotor_vel[3];

  // 4. Map policy output [-1,1] -> rotor RPM -> thrust -> normalized forces in [0,1]
  for (int i = 0; i < 4; ++i) {
    // Scale to RPM (same as Python):
    //   rpm = clip(a, -1, 1) * ((max-min)/2) + ((max+min)/2)
    const float rpm = normalized_rotor_vel[i] * ROTOR_RPM_SCALE + ROTOR_RPM_MEAN;

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