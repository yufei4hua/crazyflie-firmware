// Auto-generated from: /home/yufei/DiffSim/crazyflow_experiments/crazyflow_experiments/rl/saves/bptt_throw_model.ckpt
// Generated at: 2026-03-15T20:52:01
// ActorNet params for deterministic inference (mean only).
//
// Actor network metadata (derived from Dense params, excluding actor_logstd):
// - input_size   : 15
// - output_size  : 4
// - num_layers   : 2 (hidden Dense layers)
// - hidden_size  : 8
// - dense_layers : 3 (including output Dense)
// - total_params : 236 (kernel+bias only)
// - param_bytes  : 472 bytes (float16)
// - param_size   : 0.461 KiB (float16)
//
// Storage convention:
// - Flax Dense kernel shape is (in_features, out_features)
// - We store actor_Wk as row-major flattened kernel:
//     actor_Wk[i*out + j] == kernel[i, j]
// - Bias stored as actor_bk[j]
// - Parameters stored as float16 (__fp16) to save bandwidth over CRTP
//

#ifndef ACTOR_PARAMS_H
#define ACTOR_PARAMS_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef USE_QUAT
#define USE_QUAT 0
#endif

#if USE_QUAT
#define OBS_DIM 15
#else
#define OBS_DIM 20
#endif

// We need to determine the sizes of the layers at compile time from the OBS_DIM, ACTOR_HIDDEN_SIZE and ACTOR_OUTPUT_SIZE.
// Since C does not have constexpr, we use enums as a workaround to compute these and use them in the array declarations.
enum { ACTOR_OUTPUT_SIZE = 4 };
enum { ACTOR_HIDDEN_SIZE = 8 };
static const int ACTOR_NUM_LAYERS  = 2;
static const int ACTOR_NUM_DENSE   = 3;
// static const int ACTOR_TOTAL_PARAMS = 236;
// static const int ACTOR_PARAM_BYTES  = 944;

// Layer 0
static const int ACTOR_L0_IN  = OBS_DIM;
static const int ACTOR_L0_OUT = ACTOR_HIDDEN_SIZE;
enum { DIM_W0 = OBS_DIM * ACTOR_HIDDEN_SIZE };
static const float actor_W0[DIM_W0] = {0.0f};
static const float actor_b0[ACTOR_HIDDEN_SIZE] = {0.0f};

// Layer 1
static const int ACTOR_L1_IN  = ACTOR_HIDDEN_SIZE;
static const int ACTOR_L1_OUT = ACTOR_HIDDEN_SIZE;
enum { DIM_W1 = ACTOR_HIDDEN_SIZE * ACTOR_HIDDEN_SIZE };
static const float actor_W1[DIM_W1] = {0.0f};
static const float actor_b1[ACTOR_HIDDEN_SIZE] = {0.0f};

// Layer 2
static const int ACTOR_L2_IN  = ACTOR_HIDDEN_SIZE;
static const int ACTOR_L2_OUT = ACTOR_OUTPUT_SIZE;
enum { DIM_W2 = ACTOR_HIDDEN_SIZE * ACTOR_OUTPUT_SIZE };
static const float actor_W2[DIM_W2] = {0.0f};
static const float actor_b2[ACTOR_OUTPUT_SIZE] = {0.0f};

#ifdef __cplusplus
}
#endif

#endif  // ACTOR_PARAMS_H
