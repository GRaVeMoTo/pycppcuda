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