#include <torch/extension.h>

torch::Tensor image_blur_cuda(torch::Tensor input, int radius);
torch::Tensor image_blur_cuda_opt(torch::Tensor input, int radius);

void validate_input(torch::Tensor input, int radius) {
    TORCH_CHECK(input.device().is_cuda(), "Image must be on CUDA");
    TORCH_CHECK(input.dim() == 3, "Dimensions must be [Channels, Height, Width]");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Image must use float32 values");
    TORCH_CHECK(input.is_contiguous(), "Image must be contiguous");
    TORCH_CHECK(radius > 0, "radius must be greater than 0");
}

torch::Tensor image_blur_cuda_wrapper(torch::Tensor input, int radius) {
    validate_input(input, radius);

    return image_blur_cuda(input, radius);
}

torch::Tensor image_blur_cuda_opt_wrapper(torch::Tensor input, int radius) {
    validate_input(input, radius);

    return image_blur_cuda_opt(input, radius);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blur", &image_blur_cuda_wrapper, "Blur with CUDA");
    m.def("blur_opt", &image_blur_cuda_opt_wrapper, "Optimized blur with CUDA");
}
