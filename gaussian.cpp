#include<arm_neon.h>
#include<stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <iomanip>

using namespace std;

const int maxN = 64;
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

void LU(int n, m_size src[][maxN], m_size dst[][maxN]) {
    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            dst[k][j] = src[k][j] / src[k][k];
        }
        dst[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                dst[i][j] = src[i][j] - src[k][j] * src[i][k];
            }
            dst[i][k] = 0.0;
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
        printf("\n");
#endif
    }
}
void scaler_multiply_vec_float(float* dst, float* src, float factor, int size)
{
    int frag_len = 4;
    int main_loop = size / frag_len;                 //一个循环处理8个数据，则需要 size / 4个循环
    float32x4_t factor_vct = vdupq_n_f32(factor); //将系数factor装载入neon寄存器 
    for (int i = 0; i < main_loop; i++)
    {
        float32x4_t src_vct = vld1q_f32(src);    //将源数据装载入neon寄存器
        float32x4_t dst_vct = vmulq_f32(src_vct, factor_vct);  //执行乘法操作，且将结果放入dst_vct寄存器中
        vst1q_f32(dst, dst_vct);              //将dst_vct寄存器中的结果放回内存中
        src += frag_len, dst += frag_len;                 //改变地址，指向下个循环要处理的数据。
    }
    int aa = size - main_loop * frag_len;          //考虑到size不能被8整除的情况
    for (int j = 0; j < aa; j++)
    {
        *dst = *src * factor;
        dst++;
        src++;
    }
}




void LU_neon(int n, m_size src[][maxN], m_size dst[][maxN]) {
    for (int k = 0; k < n; k++) {
        scaler_multiply_vec_float(dst[k] + k + 1, src[k] + k + 1, 1.0 / src[k][k], n - k);
        dst[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                dst[i][j] = src[i][j] - src[k][j] * src[i][k];
            }
            dst[i][k] = 0.0;
        }
    }
}
int main()
{
    int num;
    num = 8;
    gen_matrix(num);
    print_matrix(num,A);
    LU(num,A,B);
    print_matrix(num,B);
    LU_neon(num, A, B);
    print_matrix(num,B);
    return 0;
}