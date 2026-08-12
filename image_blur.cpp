#include <torch/extension.h>

torch::Tensor image_blur_cuda(torch::Tensor input, int radius);
torch::Tensor image_blur_cuda_wrapper(torch::Tensor input, int radius) {
    TORCH_CHECK(input.device().is_cuda(), "Image in GPU");
    TORCH_CHECK(input.dim() == 3, "Dimensions: [Channels, Height, Width]");
    TORCH_CHECK(radius > 0, "radius grater than 0!");

    return image_blur_cuda(input, radius);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blur", &image_blur_cuda_wrapper, "Blur with CUDA");
}