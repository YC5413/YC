#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 假設最大通道數與特徵圖尺寸（Atari 預處理後灰度堆疊 4 幀，尺寸 84×84）
#define MAX_CHANNELS  4    // 輸入通道數（4 幀）
#define MAX_HEIGHT   84    // 影像高度
#define MAX_WIDTH    84    // 影像寬度

// 第一層卷積：輸入通道 -> 32 個輸出通道，卷積核大小 8×8，步距 4
#define CONV1_OUT   32
#define CONV1_K     8
#define CONV1_S     4

// 第二層卷積：32 -> 64，核 4×4，步距 2
#define CONV2_OUT   64
#define CONV2_K2    4
#define CONV2_S2    2

// 第三層卷積：64 -> 64，核 3×3，步距 1
#define CONV3_OUT   64
#define CONV3_K3    3
#define CONV3_S3    1

// 全連接層隱藏單元數
#define FC_HIDDEN   512

// 動作數目，由外部定義傳入
extern const int N_ACTIONS;

// 定義 DQN 網路結構體：包含卷積層與全連接層的權重和偏置
typedef struct {
    // 第一層卷積 (32 個核)
    float conv1_w[CONV1_OUT][MAX_CHANNELS][CONV1_K][CONV1_K];  // 權重
    float conv1_b[CONV1_OUT];                                  // 偏置
    // 第二層卷積 (64 個核)
    float conv2_w[CONV2_OUT][CONV1_OUT][CONV2_K2][CONV2_K2];
    float conv2_b[CONV2_OUT];                                  
    // 第三層卷積 (64 個核)
    float conv3_w[CONV3_OUT][CONV2_OUT][CONV3_K3][CONV3_K3];
    float conv3_b[CONV3_OUT];                                  

    // 全連接層權重與偏置
    int flatten_size;  // 展平後維度，動態計算得到
    float fc1_w[FC_HIDDEN][1];   // 實際應為 [FC_HIDDEN][flatten_size]
    float fc1_b[FC_HIDDEN];      // 第一層偏置
    // 第二層全連接 (輸出動作 Q 值)
    float fc2_w[/*N_ACTIONS*/][FC_HIDDEN]; // [n_actions][FC_HIDDEN]
    float fc2_b[/*N_ACTIONS*/];            // 偏置
} QNetwork;

// ReLU 激活函數
static float relu(float x) {
    return x > 0 ? x : 0;
}

// ------------------------------------------------------------------
// conv2d_layer: 單步卷積運算 (無填充)，並加上 ReLU 激活
// 參數:
//   in[in_c][in_h][in_w]    - 輸入影像 (灰度、堆疊 4 幀)
//   in_c, in_h, in_w        - 輸入通道、輸入高寬
//   out[out_c][out_h][out_w] - 輸出特徵圖
//   out_c                   - 輸出通道數
//   k                       - 卷積核大小
//   s                       - 步距
//   w[out_c][in_c][k][k]    - 權重
//   b[out_c]                - 偏置
// ------------------------------------------------------------------
void conv2d_layer(
    float in[MAX_CHANNELS][MAX_HEIGHT][MAX_WIDTH],
    int in_c, int in_h, int in_w,
    float out[][MAX_HEIGHT][MAX_WIDTH],
    int out_c, int k, int s,
    float w[][MAX_CHANNELS][k][k], float b[]
) {
    // 計算輸出尺寸
    int out_h = (in_h - k) / s + 1;
    int out_w = (in_w - k) / s + 1;
    // 遍歷每個輸出通道與空間位置
    for(int oc = 0; oc < out_c; oc++){
        for(int i = 0; i < out_h; i++){
            for(int j = 0; j < out_w; j++){
                float sum = b[oc];  // 初始化為該通道偏置
                // 計算卷積
                for(int ic = 0; ic < in_c; ic++){
                    for(int ki = 0; ki < k; ki++){
                        for(int kj = 0; kj < k; kj++){
                            // 累加權重與對應輸入像素乘積
                            sum += w[oc][ic][ki][kj] * in[ic][i*s+ki][j*s+kj];
                        }
                    }
                }
                // 激活並存入輸出
                out[oc][i][j] = relu(sum);
            }
        }
    }
}

// ------------------------------------------------------------------
// linear_layer: 全連接層運算，並加上 ReLU 激活
// 參數:
//   in[in_size]    - 輸入向量
//   in_size        - 輸入維度
//   out[out_size]  - 輸出向量
//   out_size       - 輸出維度
//   w[out_size][in_size] - 權重矩陣
//   b[out_size]          - 偏置
// ------------------------------------------------------------------
void linear_layer(
    const float *in, int in_size,
    float *out, int out_size,
    const float w[][in_size], const float b[]
) {
    for(int o = 0; o < out_size; o++){
        float sum = b[o];
        for(int i = 0; i < in_size; i++){
            sum += w[o][i] * in[i];
        }
        out[o] = relu(sum);
    }
}

// ------------------------------------------------------------------
// qnetwork_forward: 對 QNetwork 進行前向推理
// 參數:
//   net         - 指向 QNetwork 結構體指標，包含所有權重
//   input       - 預處理後並歸一化的輸入 (4,84,84)
//   q_values[o] - 返回每個動作的 Q 值
// ------------------------------------------------------------------
void qnetwork_forward(
    QNetwork *net,
    float input[MAX_CHANNELS][MAX_HEIGHT][MAX_WIDTH],
    float q_values[]                 // 長度為 N_ACTIONS
) {
    // 1. 準備中間緩衝區
    static float out1[CONV1_OUT][MAX_HEIGHT][MAX_WIDTH];
    static float out2[CONV2_OUT][MAX_HEIGHT][MAX_WIDTH];
    static float out3[CONV3_OUT][MAX_HEIGHT][MAX_WIDTH];

    // 第一層卷積
    conv2d_layer(
        input, MAX_CHANNELS, MAX_HEIGHT, MAX_WIDTH,
        out1, CONV1_OUT, CONV1_K, CONV1_S,
        net->conv1_w, net->conv1_b
    );

    // 第二層卷積：尺寸依據前一層輸出動態計算
    int h1 = (MAX_HEIGHT - CONV1_K)/CONV1_S + 1;
    int w1 = (MAX_WIDTH  - CONV1_K)/CONV1_S + 1;
    conv2d_layer(
        out1, CONV1_OUT, h1, w1,
        out2, CONV2_OUT, CONV2_K2, CONV2_S2,
        net->conv2_w, net->conv2_b
    );

    // 第三層卷積
    int h2 = (h1 - CONV2_K2)/CONV2_S2 + 1;
    int w2 = (w1 - CONV2_K2)/CONV2_S2 + 1;
    conv2d_layer(
        out2, CONV2_OUT, h2, w2,
        out3, CONV3_OUT, CONV3_K3, CONV3_S3,
        net->conv3_w, net->conv3_b
    );

    // 2. 展平 conv3 輸出為一維向量
    int h3 = (h2 - CONV3_K3)/CONV3_S3 + 1;
    int w3 = (w2 - CONV3_K3)/CONV3_S3 + 1;
    int flatten_size = CONV3_OUT * h3 * w3;
    float *flatten = (float*)malloc(sizeof(float) * flatten_size);
    int idx = 0;
    for(int c = 0; c < CONV3_OUT; c++){
        for(int i = 0; i < h3; i++){
            for(int j = 0; j < w3; j++){
                flatten[idx++] = out3[c][i][j];
            }
        }
    }

    // 3. 第一層全連接 (flatten -> hidden)
    float hidden[FC_HIDDEN];
    linear_layer(
        flatten, flatten_size,
        hidden, FC_HIDDEN,
        (const float (*)[flatten_size])net->fc1_w, net->fc1_b
    );
    free(flatten);

    // 4. 第二層全連接 (hidden -> q_values)，不加激活
    for(int o = 0; o < N_ACTIONS; o++){
        float sum = net->fc2_b[o];
        for(int i = 0; i < FC_HIDDEN; i++){
            sum += net->fc2_w[o][i] * hidden[i];
        }
        q_values[o] = sum;  // 最終 Q 值
    }
}
