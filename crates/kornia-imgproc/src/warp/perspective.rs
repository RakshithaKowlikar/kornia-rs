use crate::{
    interpolation::{interpolate_pixel_fast, validate_interpolation, InterpolationMode},
    parallel,
};

use kornia_image::{allocator::ImageAllocator, Image, ImageError};
/// CPU or GPU backend selector for [`warp_perspective`].
pub enum WarpBackend {
    /// CPU backend
    Cpu,
    /// GPU backend
    #[cfg(feature = "gpu")]
    Gpu(std::sync::Arc<GpuWarpContext>),
}

/// Persistent GPU context for [`warp_perspective`].
#[cfg(feature = "gpu")]
pub struct GpuWarpContext {
    client: cubecl::prelude::ComputeClient<
    <cubecl::wgpu::WgpuRuntime as cubecl::Runtime>::Server,
    <cubecl::wgpu::WgpuRuntime as cubecl::Runtime>::Channel,>,
    dst_handle: cubecl::server::Handle,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: u32,
}

#[cfg(feature = "gpu")]
impl GpuWarpContext {
    /// Creates a new [`GpuWarpContext`] and allocates GPU buffers for the given dimensions.
        pub fn new(
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        channels: u32,
    ) -> Self {
        use cubecl::Runtime;
        let device = cubecl::wgpu::WgpuDevice::DefaultDevice;
        let client = cubecl::wgpu::WgpuRuntime::client(&device);
        let dst_handle =
            client.empty((dst_width * dst_height * channels) as usize * std::mem::size_of::<f32>());
        Self { client, dst_handle, src_width, src_height, dst_width, dst_height, channels }
    }
    /// Uploads image data to GPU memory, returning a handle.
    pub fn upload_src(&self, data: &[f32]) -> cubecl::server::Handle {
        self.client.create(bytemuck::cast_slice(data))
    }

    /// Uploads the inverse perspective matrix to GPU memory, returning a handle.
    pub fn upload_inv_m(&self, inv_m: &[f32; 9]) -> cubecl::server::Handle {
        self.client.create(bytemuck::cast_slice(inv_m.as_ref()))
    }

    /// Runs the warp perspective kernel on the GPU without any host transfers.
    pub fn dispatch(&self, src_handle: &cubecl::server::Handle, inv_m_handle: &cubecl::server::Handle) {
        use cubecl::prelude::*;
        use cubecl::wgpu::WgpuRuntime;
        let n_pixels = self.dst_width * self.dst_height;
        let block = 256u32;
        let grid = n_pixels.div_ceil(block);
        unsafe {
            kernel::warp_perspective_kernel::launch::<f32, WgpuRuntime>(
                &self.client,
                CubeCount::Static(grid, 1, 1),
                CubeDim::new(block, 1, 1),
                ArrayArg::from_raw_parts::<f32>(src_handle, (self.src_width * self.src_height * self.channels) as usize, 1),
                ArrayArg::from_raw_parts::<f32>(&self.dst_handle, (self.dst_width * self.dst_height * self.channels) as usize, 1),
                ArrayArg::from_raw_parts::<f32>(&inv_m_handle, 9, 1),
                self.src_width,
                self.src_height,
                self.dst_width,
                self.dst_height,
                self.channels,
            );
        }
    }

    /// Downloads the result from GPU memory.
    pub fn read_back(&self) -> Vec<Vec<u8>> {
        self.client.read(vec![self.dst_handle.clone().binding()])
    }
}

#[cfg(feature = "gpu")]
mod kernel {
    use cubecl::prelude::*;

    #[cube(launch)]
    pub fn warp_perspective_kernel<F: Float>(
        src: &Array<F>,
        dst: &mut Array<F>,
        inv_m: &Array<F>,
        #[comptime] src_w: u32,
        #[comptime] src_h: u32,
        #[comptime] dst_w: u32,
        #[comptime] dst_h: u32,
        #[comptime] ch: u32,
    ) {
        let idx = ABSOLUTE_POS;
        if idx >= dst_w * dst_h { 
            return;
        }

        let px = idx % dst_w;
        let py = idx / dst_w;
        let fx = F::cast_from(px);
        let fy = F::cast_from(py);

        let w  = inv_m[6] * fx + inv_m[7] * fy + inv_m[8];
        let sx = (inv_m[0] * fx + inv_m[1] * fy + inv_m[2]) / w;
        let sy = (inv_m[3] * fx + inv_m[4] * fy + inv_m[5]) / w;

        let base = idx * ch;
        let zero = F::new(0.0);

        if sx < zero || sx >= F::cast_from(src_w) || sy < zero || sy >= F::cast_from(src_h) {
            for k in 0..ch {
                dst[base + k] = zero;
            }
            return;
        }

        let x0 = u32::cast_from(sx);
        let y0 = u32::cast_from(sy);
        let x1 = cubecl::frontend::Min::min(x0 + 1, src_w - 1);
        let y1 = cubecl::frontend::Min::min(y0 + 1, src_h - 1);
        let dx = sx - F::cast_from(x0);
        let dy = sy - F::cast_from(y0);
        let one = F::new(1.0);

        for k in 0..ch {
            let v00 = src[(y0 * src_w + x0) * ch + k];
            let v10 = src[(y0 * src_w + x1) * ch + k];
            let v01 = src[(y1 * src_w + x0) * ch + k];
            let v11 = src[(y1 * src_w + x1) * ch + k];
            dst[base + k] =
                (v00 * (one - dx) + v10 * dx) * (one - dy)
              + (v01 * (one - dx) + v11 * dx) * dy;
        }
    }
}

#[rustfmt::skip]
fn determinant3x3(m: &[f32; 9]) -> f32 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6])
}

#[rustfmt::skip]
fn adjugate3x3(m: &[f32; 9]) -> [f32; 9] {
    [
        m[4] * m[8] - m[5] * m[7],  // [0, 0]
        m[2] * m[7] - m[1] * m[8],  // [0, 1]
        m[1] * m[5] - m[2] * m[4],  // [0, 2]
        m[5] * m[6] - m[3] * m[8],  // [1, 0]
        m[0] * m[8] - m[2] * m[6],  // [1, 1]
        m[2] * m[3] - m[0] * m[5],  // [1, 2]
        m[3] * m[7] - m[4] * m[6],  // [2, 0]
        m[1] * m[6] - m[0] * m[7],  // [2, 1]
        m[0] * m[4] - m[1] * m[3],  // [2, 2]
    ]
}

fn inverse_perspective_matrix(m: &[f32; 9]) -> Result<[f32; 9], ImageError> {
    let det = determinant3x3(m);

    if det == 0.0 {
        return Err(ImageError::CannotComputeDeterminant);
    }

    let adj = adjugate3x3(m);
    let inv_det = 1.0 / det;

    let mut inv_m = [0.0; 9];
    for i in 0..9 {
        inv_m[i] = adj[i] * inv_det;
    }

    Ok(inv_m)
}

// implement later as batched operation
fn transform_point(x: f32, y: f32, m: &[f32; 9]) -> (f32, f32) {
    let w = m[6] * x + m[7] * y + m[8];
    let x_out = (m[0] * x + m[1] * y + m[2]) / w;
    let y_out = (m[3] * x + m[4] * y + m[5]) / w;
    (x_out, y_out)
}

/// Applies a perspective transformation to an image.
///
/// * `src` - The input image with shape (height, width, channels).
/// * `dst` - The output image with shape (height, width, channels).
/// * `m` - The 3x3 perspective transformation matrix src -> dst.
/// * `interpolation` - The interpolation mode to use.
///
/// # Returns
///
/// The output image with shape (new_height, new_width, channels).
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::interpolation::InterpolationMode;
/// use kornia_imgproc::warp::warp_perspective;
///
/// let src = Image::<f32, 1, _>::new(
///   ImageSize {
///     width: 4,
///     height: 5,
///   },
///   vec![0.0f32; 4 * 5],
///   CpuAllocator
/// ).unwrap();
///
/// let m = [1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
///
/// let mut dst = Image::<f32, 1, _>::from_size_val(
///   ImageSize {
///     width: 2,
///     height: 3,
///   },
///   0.0,
///   CpuAllocator
/// ).unwrap();
///
/// use kornia_imgproc::warp::WarpBackend;
/// warp_perspective(&src, &mut dst, &m, InterpolationMode::Bilinear, &WarpBackend::Cpu).unwrap();
///
/// assert_eq!(dst.size().width, 2);
/// assert_eq!(dst.size().height, 3);
/// ```
pub fn warp_perspective<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, C, A1>,
    dst: &mut Image<f32, C, A2>,
    m: &[f32; 9],
    interpolation: InterpolationMode,
    backend: &WarpBackend,
) -> Result<(), ImageError> {
    validate_interpolation(interpolation)?;
    match backend {
        WarpBackend::Cpu => {
            let inv_m = inverse_perspective_matrix(m)?;
            parallel::par_iter_rows_spatial_mapping(
                dst,
                |x, y| transform_point(x as f32, y as f32, &inv_m),
                |x, y, dst_pixel| {
                    if x >= 0.0f32 && x < src.cols() as f32 && y >= 0.0f32 && y < src.rows() as f32 {
                        dst_pixel.iter_mut().enumerate().for_each(|(k, pixel)| {
                            *pixel = interpolate_pixel_fast(src, x, y, k, interpolation);
                        });
                    }
                },
            );
            Ok(())
        }

        #[cfg(feature = "gpu")]
        WarpBackend::Gpu(ctx) => {
            let inv_m = inverse_perspective_matrix(m)?;

            let src_handle = ctx.upload_src(src.as_slice());
            let inv_m_handle = ctx.upload_inv_m(&inv_m);

            ctx.dispatch(&src_handle, &inv_m_handle);

            let result = ctx.read_back();
            let bytes = result.into_iter().next().ok_or(ImageError::InvalidImageSize(0, 0, 0, 0))?;
            dst.as_slice_mut().copy_from_slice(bytemuck::cast_slice(&bytes));

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn inverse_perspective_matrix() -> Result<(), ImageError> {
        let m = [1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let expected = [1.0, 0.0, 1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0];
        let inv_m = super::inverse_perspective_matrix(&m)?;
        assert_eq!(inv_m, expected);
        Ok(())
    }

    #[test]
    fn transform_point() {
        let m = [1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let (x, y) = super::transform_point(1.0, 1.0, &m);
        let (x_expected, y_expected) = (0.0, 2.0);
        assert_eq!(x, x_expected);
        assert_eq!(y, y_expected);
    }

    #[test]
    fn warp_perspective_identity() -> Result<(), ImageError> {
        let image: Image<f32, 3, _> = Image::from_size_val(
            ImageSize {
                width: 4,
                height: 5,
            },
            0.0f32,
            CpuAllocator,
        )?;

        // identity matrix
        let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_transformed = Image::from_size_val(new_size, 0.0, CpuAllocator)?;

        super::warp_perspective(
            &image,
            &mut image_transformed,
            &m,
            super::InterpolationMode::Bilinear,
            &super::WarpBackend::Cpu
        )?;

        assert_eq!(image_transformed.num_channels(), 3);
        assert_eq!(image_transformed.size().width, 2);
        assert_eq!(image_transformed.size().height, 3);

        Ok(())
    }

    #[test]
    fn warp_perspective_unsupported_interpolation() -> Result<(), ImageError> {
        let src = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0,
            CpuAllocator,
        )?;
        let mut dst = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0,
            CpuAllocator,
        )?;
        let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let err = super::warp_perspective(&src, &mut dst, &m, super::InterpolationMode::Lanczos, &super::WarpBackend::Cpu);
        assert!(err.is_err());
        Ok(())
    }

    #[test]
    fn warp_perspective_hflip() -> Result<(), ImageError> {
        let image = Image::<_, 1, _>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0],
            CpuAllocator,
        )?;

        let image_expected = vec![1.0, 0.0, 3.0, 2.0, 5.0, 4.0];

        // flip matrix
        let m = [-1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_transformed = Image::<_, 1, _>::from_size_val(new_size, 0.0, CpuAllocator)?;

        super::warp_perspective(
            &image,
            &mut image_transformed,
            &m,
            super::InterpolationMode::Bilinear,
            &super::WarpBackend::Cpu
        )?;

        assert_eq!(image_transformed.num_channels(), 1);
        assert_eq!(image_transformed.size().width, 2);
        assert_eq!(image_transformed.size().height, 3);

        assert_eq!(image_transformed.as_slice(), image_expected);

        Ok(())
    }

    #[test]
    fn test_warp_perspective_resize() -> Result<(), ImageError> {
        let image = Image::<_, 1, _>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![
                0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
            CpuAllocator,
        )?;

        // resize matrix (from get_perspective_transform)
        let m = [0.3333, 0.0, 0.0, 0.0, 0.3333, 0.0, 0.0, 0.0, 1.0];

        let image_expected = vec![0.0, 3.0, 12.0, 15.0];

        let new_size = ImageSize {
            width: 2,
            height: 2,
        };

        let mut image_transformed = Image::<_, 1, _>::from_size_val(new_size, 0.0, CpuAllocator)?;

        super::warp_perspective(
            &image,
            &mut image_transformed,
            &m,
            super::InterpolationMode::Bilinear,
            &super::WarpBackend::Cpu
        )?;

        let mut image_resized = Image::<_, 1, _>::from_size_val(new_size, 0.0, CpuAllocator)?;

        crate::resize::resize_native(
            &image,
            &mut image_resized,
            super::InterpolationMode::Bilinear,
        )?;

        assert_eq!(image_transformed.num_channels(), 1);
        assert_eq!(image_transformed.size().width, 2);
        assert_eq!(image_transformed.size().height, 2);

        assert_eq!(image_transformed.as_slice(), image_expected);
        assert_eq!(image_transformed.as_slice(), image_resized.as_slice());

        Ok(())
    }

    #[test]
    fn test_warp_perspective_shift() -> Result<(), ImageError> {
        let image = Image::<_, 1, _>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![
                0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
            CpuAllocator,
        )?;

        // shift left by 1 pixel
        let shift_right = -1;
        let m = [1.0, 0.0, shift_right as f32, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let image_expected = vec![
            1.0f32, 2.0, 3.0, 0.0, 5.0, 6.0, 7.0, 0.0, 9.0, 10.0, 11.0, 0.0, 13.0, 14.0, 15.0, 0.0,
        ];

        let new_size = ImageSize {
            width: image.rows(),
            height: image.cols(),
        };

        let mut image_transformed = Image::<_, 1, _>::from_size_val(new_size, 0.0, CpuAllocator)?;

        super::warp_perspective(
            &image,
            &mut image_transformed,
            &m,
            super::InterpolationMode::Bilinear,
            &super::WarpBackend::Cpu
        )?;

        assert_eq!(image_transformed.num_channels(), 1);
        assert_eq!(image_transformed.size().width, 4);
        assert_eq!(image_transformed.size().height, 4);

        assert_eq!(image_transformed.as_slice(), image_expected);

        Ok(())
    }
}
