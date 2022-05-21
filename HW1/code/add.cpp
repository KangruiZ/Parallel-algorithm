#include<arm_neon.h>
#include<stdio.h>

const int maxN = 1024;

float A[maxN][maxN];
float B[maxN][maxN];
float C[maxN][maxN];

void scaler_multiply_vec_float(float* dst, float* src, float factor, int size)
{
    int frag_len = 4;
    int main_loop = size / frag_len;                 //һ��ѭ������8�����ݣ�����Ҫ size / 8��ѭ��
    float32x4_t factor_vct = vdupq_n_f32(factor); //��ϵ��factorװ����neon�Ĵ��� 
    for (int i = 0; i < main_loop; i++)
    {
        float32x4_t src_vct = vld1q_f32(src);    //��Դ����װ����neon�Ĵ���
        float32x4_t dst_vct = vmulq_f32(src_vct, factor_vct);  //ִ�г˷��������ҽ��������dst_vct�Ĵ�����
        vst1q_f32(dst, dst_vct);              //��dst_vct�Ĵ����еĽ���Ż��ڴ���
        src += frag_len, dst += frag_len;                 //�ı��ַ��ָ���¸�ѭ��Ҫ��������ݡ�
    }
    int aa = size - main_loop * frag_len;          //���ǵ�size���ܱ�8���������
    for (int j = 0; j < aa; j++)
    {
        *dst = *src * factor;
        dst++;
        src++;
    }
}
void add_float_neon1(float* dst, float* src1, float* src2, int size)
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
void vec_multiply_vec_plus_vec_float(float* dst, float* src1, float* src2, float* src3, int size)
{
    int frag_len = 4;
    int main_loop = size / frag_len;
    for (int i = 0; i < main_loop; i++)
    {
        float32x4_t in1, in2, in3, out;
        in1 = vld1q_f32(src1);
        in2 = vld1q_f32(src2);
        in3 = vld1q_f32(src3);
        out = vmlsq_f32(in1, in2, in3);
        vst1q_f32(dst, out);
        src1 += frag_len; src2 += frag_len; src3 += frag_len; dst += frag_len;
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
    int main_loop = (size-aa) / frag_len;
    float32x4_t factor_vct = vdupq_n_f32(factor);
    for (int i = 0; i < main_loop; i++)
    {
        float32x4_t src_vct = vld1q_f32(src);
        float32x4_t dst_vct = vmulq_f32(src_vct, factor_vct);
        vst1q_f32(dst, dst_vct);
        src += frag_len, dst += frag_len;
    }
    int bb = size-aa - main_loop * frag_len;
    for (int j = 0; j < bb; j++)
    {
        *dst = *src * factor;
        dst++;
        src++;
    }
}
int main()
{
    float src1[9] = { 1,2,3,4,5,6,7,8,9 };
    float src2[9] = { 2,3,4,4,5,6,7,8,9 };
    float src3[9] = { 2,3,4,4,5,6,7,8,9 };
    float dst[9];
    int size = 9;
    float factor = 2;
    int k = 1;
    scaler_multiply_vec_aligned_float(src1 + k +1, src1 + k + 1, 1.0 / 2, k + 1, size - k);
    for (int i = 0; i < 9; i++)
    {
        printf("%f\n", src1[i]);
    }
    return 0;
}