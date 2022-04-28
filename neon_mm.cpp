#include <arm_neon.h>
#define USE_NEON
#if !defined(USE_NEON)
void add_arrays(int* restrict a, int* restrict b, int* restrict target, int size) {
    for (int i = 0; i < size; i++) {
        target[i] = a[i] + b[i];
    }
}
#else
void add_vectors(int* a, int* b, int* target, int size) {
    for (int i = 0; i < size; i += 4) {
        /* Load data into NEON register */
        int32x4_t av = vld1q_s32(&(a[i]));
        int32x4_t bv = vld1q_s32(&(b[i]));

        /* Perform the addition */
        int32x4_t targetv = vaddq_s32(av, bv);

        /* Store the result */
        vst1q_s32(&(target[i]), targetv);
    }
}
#endif
