#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void image_blur_cuda_kernel(
    const float* __restrict__ input, // [Channels, Height, Width]
    float* __restrict__ output,
    int channels, int height, int width, int radius) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // RGB Channels
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        int count = 0;

        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
                int px = x + kx;
                int py = y + ky;

                // Boundary check
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    // PyTorch CHW format (Channels, Height, Width)
                    // Index: c * (H * W) + py * W + px
                    int input_idx = c * (height * width) + py * width + px;
                    sum += input[input_idx];
                    count++;
                }
            }
        }

        int output_idx = c * (height * width) + y * width + x;
        output[output_idx] = sum / count;
    }
}

constexpr int TILE_WIDTH = 16;
constexpr int TILE_HEIGHT = 16;

__global__ void image_blur_horizontal_cuda_kernel(
    const float* __restrict__ input,
    float* __restrict__ intermediate,
    int height,
    int width,
    int radius) {
    extern __shared__ float tile[];

    const int channel = blockIdx.z;
    const int block_x = blockIdx.x * blockDim.x;
    const int block_y = blockIdx.y * blockDim.y;
    const int tile_width = blockDim.x + 2 * radius;
    const int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    const int thread_count = blockDim.x * blockDim.y;
    const int tile_size = blockDim.y * tile_width;
    const int channel_offset = channel * height * width;

    for (int index = thread_index; index < tile_size; index += thread_count) {
        const int local_y = index / tile_width;
        const int local_x = index % tile_width;
        const int global_y = block_y + local_y;
        const int global_x = block_x + local_x - radius;
        tile[index] = global_y < height && global_x >= 0 && global_x < width
            ? input[channel_offset + global_y * width + global_x]
            : 0.0f;
    }
    __syncthreads();

    const int x = block_x + threadIdx.x;
    const int y = block_y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int offset = 0; offset <= 2 * radius; ++offset) {
        sum += tile[threadIdx.y * tile_width + threadIdx.x + offset];
    }

    const int count = min(width - 1, x + radius) - max(0, x - radius) + 1;
    intermediate[channel_offset + y * width + x] = sum / count;
}

__global__ void image_blur_vertical_cuda_kernel(
    const float* __restrict__ intermediate,
    float* __restrict__ output,
    int height,
    int width,
    int radius) {
    extern __shared__ float tile[];

    const int channel = blockIdx.z;
    const int block_x = blockIdx.x * blockDim.x;
    const int block_y = blockIdx.y * blockDim.y;
    const int tile_height = blockDim.y + 2 * radius;
    const int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    const int thread_count = blockDim.x * blockDim.y;
    const int tile_size = tile_height * blockDim.x;
    const int channel_offset = channel * height * width;

    for (int index = thread_index; index < tile_size; index += thread_count) {
        const int local_y = index / blockDim.x;
        const int local_x = index % blockDim.x;
        const int global_y = block_y + local_y - radius;
        const int global_x = block_x + local_x;
        tile[index] = global_y >= 0 && global_y < height && global_x < width
            ? intermediate[channel_offset + global_y * width + global_x]
            : 0.0f;
    }
    __syncthreads();

    const int x = block_x + threadIdx.x;
    const int y = block_y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int offset = 0; offset <= 2 * radius; ++offset) {
        sum += tile[(threadIdx.y + offset) * blockDim.x + threadIdx.x];
    }

    const int count = min(height - 1, y + radius) - max(0, y - radius) + 1;
    output[channel_offset + y * width + x] = sum / count;
}

// C++ wrapper to kernel
torch::Tensor image_blur_cuda(torch::Tensor input, int radius) {
    auto output = torch::zeros_like(input);

    int channels = input.size(0);
    int height = input.size(1);
    int width = input.size(2);

    // 2D-s blocks (16x16)
    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks(
        (width + threads_per_block.x - 1) / threads_per_block.x,
        (height + threads_per_block.y - 1) / threads_per_block.y
    );

    image_blur_cuda_kernel<<<number_of_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        channels, height, width, radius
    );

    return output;
}

torch::Tensor image_blur_cuda_opt(torch::Tensor input, int radius) {
    auto intermediate = torch::empty_like(input);
    auto output = torch::empty_like(input);

    const int channels = input.size(0);
    const int height = input.size(1);
    const int width = input.size(2);
    const dim3 threads_per_block(TILE_WIDTH, TILE_HEIGHT);
    const dim3 number_of_blocks(
        (width + TILE_WIDTH - 1) / TILE_WIDTH,
        (height + TILE_HEIGHT - 1) / TILE_HEIGHT,
        channels
    );
    const size_t horizontal_shared_memory = TILE_HEIGHT * (TILE_WIDTH + 2 * radius) * sizeof(float);
    const size_t vertical_shared_memory = (TILE_HEIGHT + 2 * radius) * TILE_WIDTH * sizeof(float);

    image_blur_horizontal_cuda_kernel<<<number_of_blocks, threads_per_block, horizontal_shared_memory>>>(
        input.data_ptr<float>(), intermediate.data_ptr<float>(), height, width, radius
    );
    image_blur_vertical_cuda_kernel<<<number_of_blocks, threads_per_block, vertical_shared_memory>>>(
        intermediate.data_ptr<float>(), output.data_ptr<float>(), height, width, radius
    );

    return output;
}
