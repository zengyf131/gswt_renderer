use std::{io::Cursor, sync::mpsc};
use wgpu::BufferAddress;
use wgpu::DynamicOffset;
use wgpu::util::DeviceExt;

use crate::camera::Camera;
use crate::log;
use crate::structure::*;
use crate::texture::Texture;
use crate::utils::*;

pub struct Proxy {
    render_pipeline: wgpu::RenderPipeline,

    full_vertex_buffer: wgpu::Buffer,
    map_vertex_buffer: Option<wgpu::Buffer>,

    uniforms_buffer: wgpu::Buffer,
    uniforms_bind_group: wgpu::BindGroup,

    texture_bind_group_layout: wgpu::BindGroupLayout,
    height_map: Option<Texture>,
    proxy_texture: Option<Texture>,
    texture_bind_group: Option<wgpu::BindGroup>,

    user_data: UserData,
}
impl Proxy {
    const GRID_DIM: i32 = 2048;

    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let uniforms_block_size = ((std::mem::size_of::<Uniforms>() as u64 - 1) / 256 + 1) * 256;
        let uniforms_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(uniforms_block_size),
                    },
                    count: None,
                }],
                label: Some("Proxy uniform_bind_group_layout"),
            });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("Proxy texture_bind_group_layout"),
            });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Proxy Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("proxy.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Proxy Pipeline Layout"),
                bind_group_layouts: &[&uniforms_bind_group_layout, &texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Proxy Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Vertex buffer
        let mut triangle_vertices: Vec<Vertex> =
            Vec::with_capacity((6 * Self::GRID_DIM * Self::GRID_DIM) as usize);
        for i in 0..Self::GRID_DIM {
            for j in 0..Self::GRID_DIM {
                let p_x = (i - Self::GRID_DIM / 2) as f32;
                let p_y = (j - Self::GRID_DIM / 2) as f32;
                let mut grid_verts = vec![
                    Vertex {
                        position: [p_x, p_y],
                    },
                    Vertex {
                        position: [p_x + 1.0, p_y],
                    },
                    Vertex {
                        position: [p_x, p_y + 1.0],
                    },
                    Vertex {
                        position: [p_x + 1.0, p_y],
                    },
                    Vertex {
                        position: [p_x + 1.0, p_y + 1.0],
                    },
                    Vertex {
                        position: [p_x, p_y + 1.0],
                    },
                ];
                triangle_vertices.append(&mut grid_verts);
            }
        }
        let full_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Proxy Vertex Buffer"),
            contents: bytemuck::cast_slice(triangle_vertices.as_slice()),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Proxy Uniforms Buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size: uniforms_block_size * 2,
            mapped_at_creation: false,
        });

        let uniforms_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniforms_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniforms_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(uniforms_block_size),
                }),
            }],
            label: Some("Proxy uniforms_bind_group"),
        });

        Self {
            render_pipeline,

            full_vertex_buffer,
            map_vertex_buffer: None,
            uniforms_buffer,
            uniforms_bind_group,

            texture_bind_group_layout,
            height_map: None,
            proxy_texture: None,
            texture_bind_group: None,

            user_data: UserData::new(),
        }
    }

    pub fn configure(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        user_data: &UserData,
        render_data: &RenderData,
        proxy_tex: &(Vec<Vec<f32>>, Vector2<usize>),
    ) {
        self.user_data = user_data.clone();

        // Vertex buffer for map proxy
        let mut triangle_vertices: Vec<Vertex> =
            Vec::with_capacity(12 * user_data.tile_map_wh.x * user_data.tile_map_wh.y);
        for i in 0..user_data.tile_map_wh.x as i32 {
            for j in 0..user_data.tile_map_wh.y as i32 {
                let p_x = (i - user_data.tile_map_half_wh.x as i32) as f32;
                let p_y = (j - user_data.tile_map_half_wh.y as i32) as f32;
                let mut grid_verts = vec![
                    Vertex {
                        position: [p_x, p_y],
                    },
                    Vertex {
                        position: [p_x + 1.0, p_y],
                    },
                    Vertex {
                        position: [p_x, p_y + 1.0],
                    },
                    Vertex {
                        position: [p_x + 1.0, p_y],
                    },
                    Vertex {
                        position: [p_x + 1.0, p_y + 1.0],
                    },
                    Vertex {
                        position: [p_x, p_y + 1.0],
                    },
                ];
                triangle_vertices.append(&mut grid_verts);
            }
        }
        triangle_vertices.iter_mut().for_each(|vert| {
            vert.position[0] *= user_data.tile_width;
            vert.position[1] *= user_data.tile_width;
        });
        let map_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Proxy Vertex Buffer"),
            contents: bytemuck::cast_slice(triangle_vertices.as_slice()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.map_vertex_buffer = Some(map_vertex_buffer);

        // Texture bind group
        let mut group_entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(6);

        let height_map: Texture;
        height_map = Texture::from_bytes(
            device,
            queue,
            bytemuck::cast_slice(self.user_data.height_map.as_slice()),
            self.user_data.height_map_wh.x as u32,
            self.user_data.height_map_wh.y as u32,
            4,
            wgpu::TextureFormat::R32Float,
            wgpu::FilterMode::Linear,
            wgpu::AddressMode::Repeat,
            Some("Dummy Height Map Texture"),
        )
        .unwrap();
        self.height_map = Some(height_map);
        group_entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&self.height_map.as_ref().unwrap().view),
        });
        group_entries.push(wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::Sampler(
                self.height_map.as_ref().unwrap().sampler.as_ref().unwrap(),
            ),
        });

        // Proxy texture
        let mut size = wgpu::Extent3d {
            width: proxy_tex.1.x as u32,
            height: proxy_tex.1.y as u32,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Proxy Texture"),
            size,
            mip_level_count: proxy_tex.0.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for i in 0..proxy_tex.0.len() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    aspect: wgpu::TextureAspect::All,
                    texture: &texture,
                    mip_level: i as u32,
                    origin: wgpu::Origin3d::ZERO,
                },
                bytemuck::cast_slice(proxy_tex.0[i].as_slice()),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(16 * size.width as u32),
                    rows_per_image: Some(size.height as u32),
                },
                size,
            );
            size.width /= 2;
            size.height /= 2;
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        self.proxy_texture = Some(Texture {
            texture,
            view,
            sampler: Some(sampler),
        });

        group_entries.push(wgpu::BindGroupEntry {
            binding: 2,
            resource: wgpu::BindingResource::TextureView(
                &self.proxy_texture.as_ref().unwrap().view,
            ),
        });
        group_entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::Sampler(
                self.proxy_texture
                    .as_ref()
                    .unwrap()
                    .sampler
                    .as_ref()
                    .unwrap(),
            ),
        });

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.texture_bind_group_layout,
            entries: &group_entries,
            label: Some("texture_bind_group"),
        });

        self.texture_bind_group = Some(texture_bind_group);
    }

    pub fn render(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        camera: &Camera,
        render_data: &RenderData,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &render_data.depth_texture.as_ref().unwrap().view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        if render_data.render_config.proxy_full {
            queue.write_buffer(
                &self.uniforms_buffer,
                0,
                bytemuck::bytes_of(&Uniforms::new(
                    camera,
                    &self.user_data,
                    &render_data,
                    render_data.cur_scene_data.as_ref().unwrap(),
                    false,
                )),
            );

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.uniforms_bind_group, &[0]);
            render_pass.set_bind_group(1, &self.texture_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.full_vertex_buffer.slice(..));
            render_pass.draw(0..Self::GRID_DIM as u32 * Self::GRID_DIM as u32 * 6, 0..1);
        }

        if render_data.render_config.proxy_map {
            let uniforms_block_size =
                ((std::mem::size_of::<Uniforms>() as u64 - 1) / 256 + 1) * 256;
            queue.write_buffer(
                &self.uniforms_buffer,
                uniforms_block_size as BufferAddress,
                bytemuck::bytes_of(&Uniforms::new(
                    camera,
                    &self.user_data,
                    &render_data,
                    render_data.cur_scene_data.as_ref().unwrap(),
                    true,
                )),
            );

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(
                0,
                &self.uniforms_bind_group,
                &[uniforms_block_size as DynamicOffset],
            );
            render_pass.set_bind_group(1, &self.texture_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.map_vertex_buffer.as_ref().unwrap().slice(..));
            render_pass.draw(
                0..self.user_data.tile_map_wh.x as u32 * self.user_data.tile_map_wh.y as u32 * 6,
                0..1,
            );
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
}
impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x2];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;

        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    height_offset: f32,
    tile_width: f32,
    surface_type: u32,
    width_scale: f32,
    map_proxy: u32,
    use_clip: u32,
    clip_height: f32,
    brightness: f32,
    black_background: u32,
    _pad0: [u32; 3],

    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
    map_half_wh: [u32; 2],
    center_coord: [i32; 2],
    height_map_scale: [f32; 4],
    cam_pos: [f32; 4],
}
impl Uniforms {
    fn new(cam: &Camera, ud: &UserData, rd: &RenderData, sd: &SceneData, proxy_map: bool) -> Self {
        let rc = &rd.render_config;
        Self {
            height_offset: rc.proxy_height,
            tile_width: ud.tile_width,
            surface_type: ud.surface_type as u32,
            width_scale: rc.proxy_width_scale,
            map_proxy: proxy_map as u32,
            use_clip: rc.use_clip as u32,
            clip_height: rc.clip_height,
            brightness: rc.proxy_brightness,
            black_background: rc.proxy_black_background as u32,
            _pad0: [0; 3],

            view: (*cam.view()).into(),
            projection: (*cam.projection()).into(),
            map_half_wh: [ud.tile_map_half_wh.x as u32, ud.tile_map_half_wh.y as u32],
            center_coord: sd.center_coord.into(),
            height_map_scale: ud.height_map_scale.extend(0.0).into(),
            cam_pos: (*cam.position()).extend(0.0).into(),
        }
    }
}

pub fn upload_proxy_texture() -> mpsc::Receiver<(Vec<Vec<f32>>, Vector2<usize>)> {
    let (tx, rx) = mpsc::channel();
    let task = rfd::AsyncFileDialog::new()
        .add_filter("Proxy Texture", &["png", "jpg"])
        .pick_file();
    execute_future(async move {
        let file = task.await;
        if let Some(file) = file {
            let buffer = file.read().await;

            let img = image::ImageReader::new(Cursor::new(buffer))
                .with_guessed_format()
                .expect("Failed to guess format.")
                .decode()
                .expect("Failed to open image.");

            let max_size = (img.width() as f32).log(2.0).floor().exp2() as usize;
            let mut level_size = max_size as u32;
            let mut texture_vec: Vec<Vec<f32>> = Vec::new();
            while level_size >= 1 {
                let level_img = img.resize_exact(
                    level_size,
                    level_size,
                    image::imageops::FilterType::Lanczos3,
                );
                texture_vec.push(level_img.to_rgba32f().into_raw());
                level_size /= 2;
            }

            log!(
                "Load proxy texture: width {}, height {}, mip_level {}",
                max_size,
                max_size,
                texture_vec.len()
            );
            tx.send((texture_vec, vec2(max_size, max_size)))
                .expect("Error sending proxy texture to main thread.");
        }
    });

    rx
}
