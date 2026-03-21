# **GPU Warp Perspective — Branch gpu-warp**

GSOC PROPOSAL: [gpu\_BEV\_proposal](https://docs.google.com/document/d/1FyUEpVq5QxWn2hFEUPUe9RgBU4c9i0Z7GXF-9KUIUMA/edit?usp=sharing)

## **What was added**

### **crates/kornia-imgproc/src/warp/perspective.rs**

* Added WarpBackend enum — selects CPU or GPU path for warp\_perspective  
* Added GpuWarpContext struct — persistent GPU context (allocates VRAM once, reused across frames)  
  * new() — allocates destination buffer on GPU  
  * upload\_src() — uploads image data to GPU  
  * upload\_inv\_m() — uploads inverse perspective matrix to GPU  
  * dispatch() — runs the kernel, no host transfers  
  * read\_back() — downloads result from GPU  
* Added warp\_perspective\_kernel (CubeCL kernel) — bilinear interpolation on GPU, one thread per output pixel  
* Updated warp\_perspective() signature to accept backend: \&WarpBackend  
* All GPU code is gated behind \#\[cfg(feature \= "gpu")\]

### **crates/kornia-imgproc/src/warp/mod.rs**

* Re-exported WarpBackend and GpuWarpContext (gpu-gated)

### **crates/kornia-imgproc/Cargo.toml**

* Added cubecl, cubecl-wgpu, and bytemuck dependencies under the gpu feature flag

### **crates/kornia-imgproc/benches/bench\_warp.rs**

* Updated existing CPU benchmark to pass \&WarpBackend::Cpu  
* Added bench\_warp\_perspective\_gpu — benchmarks GPU-only dispatch path (H2D and D2H outside the loop)  
* criterion\_group\! is conditioned on the gpu feature

### **kornia-py/src/warp.rs**

* Updated Python binding to pass \&WarpBackend::Cpu to the updated warp\_perspective signature

---

## **Benchmark results**

| Resolution | CPU (ms) | GPU (µs) | Speedup |
| :---- | :---- | :---- | :---- |
| 256×224 | 0.198 ms | 7.3 µs | \~27× |
| 512×448 | 0.677 ms | 22.9 µs | \~30× |
| 1024×896 | 2.72 ms | 98.5 µs | \~28× |

GPU times measure kernel dispatch only (no data transfers).

Designed for video/multi-frame use: upload once, warp many frames, download once.

For single-image use, results will differ.

**Built a Bubbaloop node that takes a live V4L camera feed and runs it through the GPU perspective warp added to kornia-rs.**

[**DEMO**](https://drive.google.com/file/d/1lHT9Z_hKT8lNZagvvL1HbKSjnwzt00xZ/view?usp=drive_link)

