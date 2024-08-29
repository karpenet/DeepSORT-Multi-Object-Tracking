#include <cmath>
#include <cuda_runtime.h>

// CUDA kernel to normalize vectors
__device__ void normalize_vectors(float* vec, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float norm = 0.0f;
        for (int i = 0; i < n; ++i) {
            norm += vec[i] * vec[i];
        }
        norm = sqrtf(norm);
        if (norm > 0.0f) {
            for (int i = 0; i < n; ++i) {
                vec[i] /= norm;
            }
        }
    }
}

// CUDA kernel to compute dot product
__device__ float cosine_similarity(float* A, float* B, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float dot_product = 0.0f;
    if (idx < n) {
        for (int i = 0; i < n; ++i) {
            dot_product += A[i] * B[i];
        }
    }
    return dot_product;
}

__device__ float check_division_by_0(float value, float epsilon = 0.01) {
    return value < epsilon ? epsilon : value;
}

__device__ float compute_iou(float* box1, float* box2) {
    float xA = fmaxf(box1[0], box2[0]);
    float yA = fmaxf(box1[1], box2[1]);
    float xB = fminf(box1[2], box2[2]);
    float yB = fminf(box1[3], box2[3]);

    float inter_area = fmaxf(0, xB - xA + 1) * fmaxf(0, yB - yA + 1);
    float box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1);
    float box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1);
    float union_area = (box1_area + box2_area) - inter_area;
    return inter_area / union_area;
}

__device__ float sanchez_matilla(float* box1, float* box2, int w, int h) {
    float Q_dist = sqrtf(w * w + h * h);
    float Q_shape = w * h;
    float distance_term = Q_dist / check_division_by_0(sqrtf(powf(box1[0] - box2[0], 2) + powf(box1[1] - box2[1], 2)));
    float shape_term = Q_shape / check_division_by_0(sqrtf(powf(box1[2] - box2[2], 2) + powf(box1[3] - box2[3], 2)));
    return distance_term * shape_term;
}

__device__ float yu(float* box1, float* box2) {
    float w1 = 0.5;
    float w2 = 1.5;
    float a = (box1[0] - box2[0]) / check_division_by_0(box1[2]);
    float a_2 = powf(a, 2);
    float b = (box1[1] - box2[1]) / check_division_by_0(box1[3]);
    float b_2 = powf(b, 2);
    float ab = (a_2 + b_2) * w1 * (-1);
    float c = fabsf(box1[3] - box2[3]) / (box1[3] + box2[3]);
    float d = fabsf(box1[2] - box2[2]) / (box1[2] + box2[2]);
    float cd = (c + d) * w2 * (-1);
    return expf(ab) * expf(cd);
}

__global__ void compute_cost_kernel(float* old_boxes, float* new_boxes, float* old_features, float* new_features, float* costs, int num_boxes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < num_boxes && j < num_boxes) {
        float* old_box = &old_boxes[i * 4];
        float* new_box = &new_boxes[j * 4];
        float* old_feature = &old_features[i * 1024];
        float* new_feature = &new_features[j * 1024];

        normalize_vectors(old_feature, 1024);
        normalize_vectors(new_feature, 1024);

        float iou_cost = compute_iou(old_box, new_box);
        float linear_cost = sanchez_matilla(old_box, new_box, 1920, 1080);
        float exponential_cost = yu(old_box, new_box);
        float feature_cost = cosine_similarity(old_feature, new_feature, 1024);

        costs[i * num_boxes + j] = iou_cost + linear_cost + exponential_cost + feature_cost;
    }
    
}