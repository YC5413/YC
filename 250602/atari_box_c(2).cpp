#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

///==================
/// 共用定義與工具
///==================

// 激活函式：ReLU
static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// 申請一維浮點陣列
static float* malloc_f(int n) {
    float* p = (float*)malloc(sizeof(float) * n);
    if (!p) { fprintf(stderr, "記憶體不足\n"); exit(1); }
    return p;
}

///==================
/// 捲積層 (Conv2D)
///==================

typedef struct {
    int in_c, out_c;   // 輸入、輸出通道數
    int k;             // 卷積核尺寸
    int s;             // 步距 (stride)
    float* w;          // 權重：維度 out_c × in_c × k × k
    float* b;          // 偏置：維度 out_c
} Conv2D;

// 建立捲積層，呼叫者需自行填充 w、b
Conv2D* Conv2D_new(int in_c, int out_c, int k, int s) {
    Conv2D* layer = (Conv2D*)malloc(sizeof(Conv2D));
    layer->in_c = in_c;
    layer->out_c = out_c;
    layer->k = k;
    layer->s = s;
    layer->w = malloc_f(out_c * in_c * k * k);
    layer->b = malloc_f(out_c);
    return layer;
}

// 捲積層前向運算：輸入大小 (in_c, H, W)，輸出 (out_c, OH, OW)
void Conv2D_forward(
    const Conv2D* L,
    const float* in, int H, int W,
    float* out
) {
    int in_c = L->in_c, out_c = L->out_c;
    int k = L->k, s = L->s;
    int OH = (H - k) / s + 1;
    int OW = (W - k) / s + 1;

    // 對每個輸出通道與位置進行卷積
    for (int oc = 0; oc < out_c; oc++) {
        for (int oy = 0; oy < OH; oy++) {
            for (int ox = 0; ox < OW; ox++) {
                float sum = L->b[oc];  // 初始化為偏置
                // 累加所有輸入通道與卷積核的乘積
                for (int ic = 0; ic < in_c; ic++) {
                    for (int ky = 0; ky < k; ky++) {
                        for (int kx = 0; kx < k; kx++) {
                            int iy = oy * s + ky;
                            int ix = ox * s + kx;
                            int wi = ((oc * in_c + ic) * k + ky) * k + kx;
                            int ii = (ic * H + iy) * W + ix;
                            sum += L->w[wi] * in[ii];
                        }
                    }
                }
                // ReLU 激活後存入輸出
                int oi = (oc * OH + oy) * OW + ox;
                out[oi] = relu(sum);
            }
        }
    }
}

///==================
/// 全連接層 (Dense)
///==================

typedef struct {
    int in_n, out_n;  // 輸入、輸出維度
    float* w;         // 權重：out_n × in_n
    float* b;         // 偏置：out_n
} Dense;

// 建立全連接層，呼叫者需自行填充 w、b
Dense* Dense_new(int in_n, int out_n) {
    Dense* L = (Dense*)malloc(sizeof(Dense));
    L->in_n = in_n;
    L->out_n = out_n;
    L->w = malloc_f(out_n * in_n);
    L->b = malloc_f(out_n);
    return L;
}

// 全連接前向：in[in_n] -> out[out_n]
void Dense_forward(
    const Dense* L,
    const float* in,
    float* out
) {
    for (int o = 0; o < L->out_n; o++) {
        float sum = L->b[o];
        for (int i = 0; i < L->in_n; i++) {
            sum += L->w[o * L->in_n + i] * in[i];
        }
        out[o] = relu(sum);
    }
}

///==================
/// 網路主體：QNetwork
///==================

typedef struct {
    Conv2D* conv1;
    Conv2D* conv2;
    Conv2D* conv3;
    Dense*  fc1;
    Dense*  fc2;      // 最後一層直接輸出 Q 值 (無激活)
    int     flatten_n; // 展平後維度
} QNetwork;

// 初始化 QNetwork 使用經典 DQN 架構
QNetwork* QNetwork_new(int in_c, int H, int W, int n_actions) {
    QNetwork* net = (QNetwork*)malloc(sizeof(QNetwork));
    // 定義三層卷積
    net->conv1 = Conv2D_new(in_c, 32, 8, 4);
    net->conv2 = Conv2D_new(32, 64, 4, 2);
    net->conv3 = Conv2D_new(64, 64, 3, 1);
    // 計算每層輸出尺寸
    int H1 = (H - 8) / 4 + 1;
    int W1 = (W - 8) / 4 + 1;
    int H2 = (H1 - 4) / 2 + 1;
    int W2 = (W1 - 4) / 2 + 1;
    int H3 = (H2 - 3) / 1 + 1;
    int W3 = (W2 - 3) / 1 + 1;
    net->flatten_n = 64 * H3 * W3;
    // 定義兩層全連接
    net->fc1 = Dense_new(net->flatten_n, 512);
    net->fc2 = Dense_new(512, n_actions);
    return net;
}

// 前向推理：input[in_c][H][W] -> q_value[n_actions]
void QNetwork_forward(
    const QNetwork* net,
    const float* input,
    int H,
    int W,
    float* q_value
) {
    // 計算中間維度
    int H1 = (H - 8) / 4 + 1, W1 = (W - 8) / 4 + 1;
    int H2 = (H1 - 4) / 2 + 1, W2 = (W1 - 4) / 2 + 1;
    int H3 = (H2 - 3) / 1 + 1, W3 = (W2 - 3) / 1 + 1;

    // 申請中間緩衝區
    float* out1 = malloc_f(32 * H1 * W1);
    float* out2 = malloc_f(64 * H2 * W2);
    float* out3 = malloc_f(64 * H3 * W3);

    // 三層卷積前向
    Conv2D_forward(net->conv1, input,  H,  W,  out1);
    Conv2D_forward(net->conv2, out1,  H1, W1, out2);
    Conv2D_forward(net->conv3, out2,  H2, W2, out3);

    // 展平
    float* flat = malloc_f(net->flatten_n);
    memcpy(flat, out3, sizeof(float) * net->flatten_n);
    free(out1); free(out2); free(out3);

    // 第一層全連接
    float* h1 = malloc_f(512);
    Dense_forward(net->fc1, flat, h1);
    free(flat);

    // 第二層全連接輸出 Q 值
    for (int a = 0; a < net->fc2->out_n; a++) {
        float sum = net->fc2->b[a];
        for (int i = 0; i < net->fc2->in_n; i++) {
            sum += net->fc2->w[a * net->fc2->in_n + i] * h1[i];
        }
        q_value[a] = sum;  // 最終 Q 值
    }
    free(h1);
}
