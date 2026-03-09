#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void render_cuda(
    torch::Tensor canvas,
    torch::Tensor flow,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor R_old,
    torch::Tensor pos,
    torch::Tensor pos_old,
    float drone_radius,
    int n_drones_per_group,
    float fov_x_half_tan);

void rerender_backward_cuda(
    torch::Tensor depth,
    torch::Tensor dddp,
    float fov_x_half_tan);

void find_nearest_pt_cuda(
    torch::Tensor nearest_pt,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor pos,
    float drone_radius,
    int n_drones_per_group);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render", &render_cuda, "render (CUDA)");
  m.def("find_nearest_pt", &find_nearest_pt_cuda, "find_nearest_pt (CUDA)");
  m.def("rerender_backward", &rerender_backward_cuda, "rerender_backward_cuda (CUDA)");
}
