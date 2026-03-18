# Adding / Removing a Controller

## Files to touch (in order)

1. **`src/modules/interface/controller/controller_<name>.h`**
   - Declare `controllerNameInit(void)`, `controllerNameTest(void)`, `controllerName(..., stabilizerStep_t stabilizerStep)`

2. **`src/modules/src/controller/controller_<name>.c`**
   - Implement the three functions above
   - Use `RATE_DO_EXECUTE(freq, stabilizerStep)` (not `tick`)
   - Register params/logs under a unique group name (e.g. `ctrlName`)

3. **`src/modules/interface/controller/controller.h`**
   - Add `ControllerTypeName` to the `ControllerType` enum, **before** `ControllerTypeOot` and `ControllerType_COUNT`
   - The enum value = the index used in `stabilizer.controller` param (0-based, AutoSelect=0, PID=1, …)

4. **`src/modules/src/controller/controller.c`**
   - `#include "controller_<name>.h"`
   - Add an entry to `controllerFunctions[]` at the matching index:
     ```c
     {.init = controllerNameInit, .test = controllerNameTest, .update = controllerName, .name = "Name"},
     ```

## Current controller ID map

| ID | ControllerType            | Name          |
|----|---------------------------|---------------|
| 0  | ControllerTypeAutoSelect  | (auto)        |
| 1  | ControllerTypePID         | PID           |
| 2  | ControllerTypeMellinger   | Mellinger     |
| 3  | ControllerTypeINDI        | INDI          |
| 4  | ControllerTypeBrescianini | Brescianini   |
| 5  | ControllerTypeLee         | Lee           |
| 6  | ControllerTypeRotorVelocity | RotorVelocity |
| 7  | ControllerTypeForceTorque | ForceTorque   |
| 8  | ControllerTypeRL          | RL            |
| 9  | ControllerTypeThrow       | Throw         |

Select via: `drone.param.set_value("stabilizer.controller", <ID>)`

## Notes
- The `ControllerFcns.update` function pointer in `controller.c` uses `stabilizerStep_t` — match this in your controller signature
- The `controllerFunctions[]` array index must align exactly with the enum value
- Missing the enum entry causes `controllerInit` to reject the ID (it checks `controller < ControllerType_COUNT`)
