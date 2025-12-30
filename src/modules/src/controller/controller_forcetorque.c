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

#include "app.h"

#include "FreeRTOS.h"
#include "task.h"

// The new controller goes here --------------------------------------------
// Move the includes to the the top of the file if you want to
#include "controller.h"

void controllerForceTorqueInit() {
  // Initialize your controller data here...
  return;
}

bool controllerForceTorqueTest() {
  // Always return true
  return true;
}

void controllerForceTorque(control_t *control, const setpoint_t *setpoint, const sensorData_t *sensors, const state_t *state, const uint32_t tick) {
  // Implement your controller here...
  
  // temporarily borrow attitude interface for force_torque
  control->controlMode = controlModeLegacy;
  // thrust in PWM (done convertion in python side)
  control->thrust = setpoint->attitude.yaw;
  // roll, pitch, yaw torque in PWM (done convertion in python side)
  control->roll = (uint16_t)(setpoint->position.x);
  control->pitch = (uint16_t)(setpoint->position.y);
  control->yaw = (uint16_t)(setpoint->position.z);
}
