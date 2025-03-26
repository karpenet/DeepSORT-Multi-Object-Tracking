#include <cmath>
#include <cuda_runtime.h>

#define TILE_SIZE 16 // Calibrate to GPU
#define FEATURE_DIM 1024

// CUDA kernel to normalize vectors
__device__ void normalize_vectors(float* vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < FEATURE_DIM) {
        float norm = 0.0f;
        for (int i = 0; i < FEATURE_DIM; ++i) {
            norm += vec[i] * vec[i];
        }
        norm = sqrtf(norm);
        if (norm > 0.0f) {
            for (int i = 0; i < FEATURE_DIM; ++i) {
                vec[i] /= norm;
            }
        }
    }
}

// CUDA kernel to compute dot product
__device__ float cosine_similarity(float* A, float* B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float dot_product = 0.0f;
    if (idx < FEATURE_DIM) {
        for (int i = 0; i < FEATURE_DIM; ++i) {
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
        float* old_feature = &old_features[i * FEATURE_DIM];
        float* new_feature = &new_features[j * FEATURE_DIM];

        normalize_vectors(old_feature);
        normalize_vectors(new_feature);

        float iou_cost = compute_iou(old_box, new_box);
        float linear_cost = sanchez_matilla(old_box, new_box, 1920, 1080);
        float exponential_cost = yu(old_box, new_box);
        float feature_cost = cosine_similarity(old_feature, new_feature, 1024);

        costs[i * num_boxes + j] = iou_cost + linear_cost + exponential_cost + feature_cost;
    }
    
}

__global__ void compute_cost_kernel_fast(float* old_boxes, float* new_boxes, float* old_features, float* new_features, float* costs, int num_old_boxes, int num_new_boxes) {
    __shared__ float shared_old_boxes[TILE_SIZE][4];
    __shared__ float shared_new_boxes[TILE_SIZE][4];  

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.x * TILE_SIZE + tx;
    int col = blockIdx.y * TILE_SIZE + ty;

    if (row >= num_boxes || col >= num_boxes) return;
    float cost = 0.0f;

    for (int i = 0; i < (num_old_boxes + TILE_SIZE - 1) / TILE_SIZE; i++) {
        if (row < num_old_boxes && (i * TILE_SIZE + ty) < num_new_boxes) {
            int index = row * 4;
            shared_old_boxes[tx][0] = old_boxes[index + 0];
            shared_old_boxes[tx][1] = old_boxes[index + 1];
            shared_old_boxes[tx][2] = old_boxes[index + 2];
            shared_old_boxes[tx][3] = old_boxes[index + 3];
        }

        if (col < num_new_boxes && (i * TILE_SIZE + tx) < num_old_boxes) {
            int index = col * 4;
            shared_new_boxes[ty][0] = new_boxes[index + 0];
            shared_new_boxes[ty][1] = new_boxes[index + 1];
            shared_new_boxes[ty][2] = new_boxes[index + 2];
            shared_new_boxes[ty][3] = new_boxes[index + 3];
        }

        __syncthreads();

        if (row < num_old_boxes && col < num_new_boxes) {
            float* old_box = shared_old_boxes[tx];
            float* new_box = shared_new_boxes[ty];

            float* old_feature = &old_features[row * FEATURE_DIM];
            float* new_feature = &new_features[col * FEATURE_DIM];

            normalize_vectors(old_feature, FEATURE_DIM);
            normalize_vectors(new_feature, FEATURE_DIM);

            float iou_cost = compute_iou(shared_old_boxes[tx], shared_new_boxes[ty]);
            float linear_cost = sanchez_matilla(shared_old_boxes[tx], shared_new_boxes[ty], 1920, 1080);
            float exponential_cost = yu(shared_old_boxes[tx], shared_new_boxes[ty]);
            float feature_cost = cosine_similarity(old_feature, new_feature);

            cost += iou_cost + linear_cost + exponential_cost + feature_cost;
        }

        __syncthreads();
    }
    
    if (row < num_old_boxes && col < num_new_boxes) {

        costs[row * num_new_boxes + col] = cost;
    }
}