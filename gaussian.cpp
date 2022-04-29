#include<arm_neon.h>
#include<stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <iomanip>
#include "Timer.h"

using namespace std;

const int maxN = 1024;
const int rand_range = 100;

typedef float m_size;
m_size A[maxN][maxN];
m_size B[maxN][maxN];


void print_matrix(int n, m_size mat[][maxN]) {
    cout.precision(4);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << setw(8) << mat[i][j] << ' ';
        }
        cout << endl;
    }
}
void gen_matrix(int n) {
    //srand((unsigned)time(NULL));
    srand(1);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (int)((m_size)rand() / RAND_MAX * rand_range);
        }
    }
}
void dup_matrix(int n, m_size src[][maxN], m_size dst[][maxN]) {
    //srand((unsigned)time(NULL));
    srand(1);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dst[i][j] = src[i][j];
        }
    }
}

void LU(int n, m_size src[][maxN]) {
    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            src[k][j] = src[k][j] / src[k][k];
        }
        src[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                src[i][j] = src[i][j] - src[k][j] * src[i][k];
            }
            src[i][k] = 0.0;
        }
    }
}
void add_vec_float(float* dst, float* src1, float* src2, int size)
{
    int frag_len = 4;
    int main_loop = size / frag_len;
    for (int i = 0; i < main_loop; i++)
    {
        float32x4_t in1, in2, out;
        in1 = vld1q_f32(src1);
        in2 = vld1q_f32(src2);
        out = vaddq_f32(in1, in2);
        vst1q_f32(dst, out);
        src1 += frag_len; src2 += frag_len; dst += frag_len;
#if defined (__aarch64__)
#endif
    }
    int residual = size - main_loop * frag_len;
    for (int j = 0; j < residual; j++)
    {
        *dst = *src1 + *src2;
        dst++;
        src1++;
        src2++;
    }
}

void scaler_multiply_vec_float(float* dst, float* src, float factor, int size)
{
    int frag_len = 4;
    int main_loop = size / frag_len;                 //一个循环处理4个数据，则需要 size / 4个循环
    float32x4_t factor_vct = vdupq_n_f32(factor); //将系数factor装载入neon寄存器 
    for (int i = 0; i < main_loop; i++)
    {
        float32x4_t src_vct = vld1q_f32(src);    //将源数据装载入neon寄存器
        float32x4_t dst_vct = vmulq_f32(src_vct, factor_vct);  //执行乘法操作，且将结果放入dst_vct寄存器中
        vst1q_f32(dst, dst_vct);              //将dst_vct寄存器中的结果放回内存中
        src += frag_len, dst += frag_len;                 //改变地址，指向下个循环要处理的数据。
    }
    int aa = size - main_loop * frag_len;          //考虑到size不能被4整除的情况
    for (int j = 0; j < aa; j++)
    {
        *dst = *src * factor;
        dst++;
        src++;
    }
}
void scaler_multiply_vec_subtract_vec_float(float* dst, float* src1, float* src2, float factor, int size)
{
    int frag_len = 4;
    int main_loop = size / frag_len;
    float32x4_t factor_vec = vdupq_n_f32(factor);
    for (int i = 0; i < main_loop; i++)
    {
        float32x4_t in1, in2, out;
        in1 = vld1q_f32(src1);
        in2 = vld1q_f32(src2);
        out = vmlsq_f32(in1, in2, factor_vec);
        vst1q_f32(dst, out);
        src1 += frag_len; src2 += frag_len; dst += frag_len;
#if defined (__aarch64__)
#endif
    }
    int residual = size - main_loop * frag_len;
    for (int j = 0; j < residual; j++)
    {
        *dst = *src1 - *src2 * factor;
        dst++;
        src1++;
        src2++;
    }
}
void LU_neon(int n, m_size src[][maxN]) {
    for (int k = 0; k < n; k++) {
        scaler_multiply_vec_float(src[k] + k + 1, src[k] + k + 1, 1.0 / src[k][k], n - k);
        src[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            scaler_multiply_vec_subtract_vec_float(src[i] + k + 1, src[i] + k + 1, src[k] + k + 1, src[i][k], n - k);
            src[i][k] = 0.0;
        }
    }
}
void scaler_multiply_vec_aligned_float(float* dst, float* src, float factor, int k, int size)
{
    int frag_len = 4;
    int initial_pos = k / frag_len;
    int aa = k - initial_pos * frag_len;
    for (int j = 0; j < aa; j++)
    {
        *dst = *src * factor;
        dst++;
        src++;
    }
    int main_loop = (size - aa) / frag_len;
    float32x4_t factor_vct = vdupq_n_f32(factor);
    for (int i = 0; i < main_loop; i++)
    {
        float32x4_t src_vct = vld1q_f32(src);
        float32x4_t dst_vct = vmulq_f32(src_vct, factor_vct);
        vst1q_f32(dst, dst_vct);
        src += frag_len, dst += frag_len;
    }
    int bb = size - aa - main_loop * frag_len;
    for (int j = 0; j < bb; j++)
    {
        *dst = *src * factor;
        dst++;
        src++;
    }
}
void scaler_multiply_vec_subtract_vec_aligned_float(float* dst, float* src1, float* src2, float factor,int k, int size)
{
    int frag_len = 4;
    int initial_pos = k / frag_len;
    int aa = k - initial_pos * frag_len;
    for (int j = 0; j < aa; j++)
    {
        *dst = *src1 - *src2 * factor;
        dst++;
        src1++;
        src2++;
    }
    int main_loop = (size-aa) / frag_len;
    float32x4_t factor_vec = vdupq_n_f32(factor);
    for (int i = 0; i < main_loop; i++)
    {
        float32x4_t in1, in2, out;
        in1 = vld1q_f32(src1);
        in2 = vld1q_f32(src2);
        out = vmlsq_f32(in1, in2, factor_vec);
        vst1q_f32(dst, out);
        src1 += frag_len; src2 += frag_len; dst += frag_len;
#if defined (__aarch64__)
#endif
    }
    int residual = size - aa - main_loop * frag_len;
    for (int j = 0; j < residual; j++)
    {
        *dst = *src1 - *src2 * factor;
        dst++;
        src1++;
        src2++;
    }
}
void LU_neon_aligned(int n, m_size src[][maxN]) {
    for (int k = 0; k < n; k++) {
        scaler_multiply_vec_aligned_float(src[k] + k + 1, src[k] + k + 1, 1.0 / src[k][k], k + 1, n - k);
        src[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            scaler_multiply_vec_subtract_vec_aligned_float(src[i] + k + 1, src[i] + k + 1, src[k] + k + 1, src[i][k],k+1, n - k);
            src[i][k] = 0.0;
        }
    }
}
int main()
{
    int num;
    //num = 1024;
    num = 512;
    Timer timer;
    gen_matrix(num);
    dup_matrix(num, A, B);
    //print_matrix(num,A);
    timer.Start();
    LU(num, A);
    timer.Stop("time:");
    //print_matrix(num,A);
    timer.Start();
    //LU_neon(num, B);
    LU_neon(num, B);
    timer.Stop("neon time:");
    //print_matrix(num,B);
    return 0;
}