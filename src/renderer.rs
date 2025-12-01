use wgpu::BufferAddress;
use wgpu::util::DeviceExt;

use crate::camera::{Camera, CameraUniforms};
use crate::log;
use crate::structure::*;
use crate::texture::Texture;
use crate::utils::*;

pub struct GSWTRenderer {
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,

    camera_uniforms_buffer: wgpu::Buffer,
    scene_uniforms_buffer: wgpu::Buffer,
    gaussian_texture: Texture,
    scene_bind_group_layout: wgpu::BindGroupLayout,
    scene_bind_group: Option<wgpu::BindGroup>,

    tile_uniforms_buffer: wgpu::Buffer,
    tile_bind_group: wgpu::BindGroup,

    gs_index_buffer: wgpu::Buffer,
    map_id_buffer: wgpu::Buffer,
    lod_id_buffer: wgpu::Buffer,
    buffer_base_data: Vec<Vec<Vec<BufferDataValue>>>,

    user_data: UserData,
}
impl GSWTRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        preload_data: PreloadData,
    ) -> Self {
        let scene_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Uint,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("scene_bind_group_layout"),
            });

        // TODO: change back to uniform
        let tile_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(256),
                    },
                    count: None,
                }],
                label: Some("tile_bind_group_layout"),
            });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gswt.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&scene_bind_group_layout, &tile_bind_group_layout],
                push_constant_ranges: &[],
            });

        let alpha_blend = Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    Vertex2D::desc(),
                    wgpu::VertexBufferLayout {
                        array_stride: 4 as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![1 => Uint32],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: 4 as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![2 => Uint32],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: 4 as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![3 => Uint32],
                    },
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: alpha_blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                cull_mode: None,
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: false,
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
        let vertices = &mut [
            // quad
            Vertex2D {
                position: [-2.0, -2.0],
            },
            Vertex2D {
                position: [2.0, -2.0],
            },
            Vertex2D {
                position: [2.0, 2.0],
            },
            Vertex2D {
                position: [2.0, 2.0],
            },
            Vertex2D {
                position: [-2.0, 2.0],
            },
            Vertex2D {
                position: [-2.0, -2.0],
            },
        ];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Scene global data
        let camera_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniforms Buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size: std::mem::size_of::<CameraUniforms>() as u64,
            mapped_at_creation: false,
        });
        let scene_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scene Uniforms Buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size: std::mem::size_of::<SceneUniforms>() as u64,
            mapped_at_creation: false,
        });
        let gaussian_texture = Texture::from_bytes(
            device,
            queue,
            transmute_slice::<_, u8>(preload_data.tile_splats_merged.tex_data.as_slice()),
            preload_data.tile_splats_merged.tex_width as u32,
            preload_data.tile_splats_merged.tex_height as u32,
            16,
            wgpu::TextureFormat::Rgba32Uint,
            wgpu::FilterMode::Nearest,
            wgpu::AddressMode::ClampToEdge,
            Some("Gaussian Texture"),
        )
        .unwrap();

        let tile_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Per-instance Tile Uniforms Storage Buffer"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            size: 20_000 * u64::max(256, std::mem::size_of::<TileUniforms>() as u64),
            mapped_at_creation: false,
        });
        let tile_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &tile_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &tile_uniforms_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(256),
                }),
            }],
            label: Some("tile_bind_group"),
        });

        // Instance buffers
        let gs_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("gs_index_buffer").as_str()),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            size: 10_000_000 * 4,
            mapped_at_creation: false,
        });
        let map_id_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("map_id_buffer").as_str()),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            size: 10_000_000 * 4,
            mapped_at_creation: false,
        });
        let lod_id_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("lod_id_buffer").as_str()),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            size: 10_000_000 * 4,
            mapped_at_creation: false,
        });

        // Preloaded instance buffers
        let mut buffer_base_data = Vec::with_capacity(preload_data.tile_base_data.len());
        for i in 0..preload_data.tile_base_data.len() {
            let tile_data_vec = &preload_data.tile_base_data[i];
            let mut tile_buf_vec: Vec<Vec<BufferDataValue>> =
                Vec::with_capacity(tile_data_vec.len());
            for j in 0..tile_data_vec.len() {
                let view_data_vec = &tile_data_vec[j];
                let mut view_buf_vec: Vec<BufferDataValue> =
                    Vec::with_capacity(view_data_vec.len());
                for k in 0..view_data_vec.len() {
                    let base_data = &view_data_vec[k];

                    let gs_index_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(format!("gs_index_buffer [{i}.{j}.{k}]").as_str()),
                            contents: bytemuck::cast_slice(base_data.gs_index.as_slice()),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                    let lod_id_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(format!("lod_id_buffer [{i}.{j}.{k}]").as_str()),
                            contents: bytemuck::cast_slice(base_data.gs_lod_id.as_slice()),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                    let buffer_data = BufferDataValue {
                        splat_count: base_data.splat_count as u32,
                        gs_index_buffer,
                        map_id_buffer: None,
                        lod_id_buffer: Some(lod_id_buffer),
                    };
                    view_buf_vec.push(buffer_data);
                }
                tile_buf_vec.push(view_buf_vec);
            }
            buffer_base_data.push(tile_buf_vec);
        }

        Self {
            render_pipeline,
            vertex_buffer,

            camera_uniforms_buffer,
            scene_uniforms_buffer,
            gaussian_texture,
            scene_bind_group_layout,
            scene_bind_group: None,

            tile_uniforms_buffer,
            tile_bind_group,

            gs_index_buffer,
            map_id_buffer,
            lod_id_buffer,
            buffer_base_data,

            user_data: UserData::new(),
        }
    }

    pub fn configure(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        user_data: &UserData,
        render_data: &RenderData,
    ) {
        self.user_data = user_data.clone();

        let mut group_entries: Vec<wgpu::BindGroupEntry> = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self.camera_uniforms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: self.scene_uniforms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&self.gaussian_texture.view),
            },
        ];

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
        group_entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::TextureView(&height_map.view),
        });
        group_entries.push(wgpu::BindGroupEntry {
            binding: 4,
            resource: wgpu::BindingResource::Sampler(height_map.sampler.as_ref().unwrap()),
        });

        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.scene_bind_group_layout,
            entries: &group_entries,
            label: Some("scene_bind_group"),
        });

        self.scene_bind_group = Some(scene_bind_group);
    }

    pub fn render(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        camera: &Camera,
        render_data: &RenderData,
    ) {
        let scene_data = render_data.cur_scene_data.as_ref().unwrap();
        let sort_data = render_data.cur_sort_data.as_ref().unwrap();
        let render_config = &render_data.render_config;

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
                    load: if render_data.use_proxy {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(1.0)
                    },
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        queue.write_buffer(
            &self.camera_uniforms_buffer,
            0,
            bytemuck::bytes_of(&CameraUniforms::from_camera(camera)),
        );

        queue.write_buffer(
            &self.scene_uniforms_buffer,
            0,
            bytemuck::bytes_of(&SceneUniforms::from_data(
                &self.user_data,
                scene_data,
                render_data,
            )),
        );

        let tile_uniforms_block_size =
            usize::max(256, std::mem::size_of::<TileUniforms>() as usize);
        let view_proj = camera.view_proj();
        let mut buffer_offset: BufferAddress = 0;
        for i in 0..sort_data.render_data_vec.len() {
            let tile_instance = &sort_data.tile_instance_vec[i];
            let (render_data_key, option_render_data_value) = &sort_data.render_data_vec[i];
            let tid = tile_instance.tid;

            // viewport culling (only for non-merged tiles)
            if render_data_key.tid.len() == 1 {
                let mut pos2d = vec3(f32::MAX, f32::MAX, -f32::MAX);
                for ci in 0..4 {
                    let corner = view_proj
                        * tile_instance.corner_data.as_ref().unwrap()[ci]
                            .0
                            .extend(1.0);
                    let corner = corner.truncate() / corner.w;
                    if corner.x.abs() < pos2d.x {
                        pos2d.x = corner.x.abs();
                    }
                    if corner.y.abs() < pos2d.y {
                        pos2d.y = corner.y.abs();
                    }
                    if corner.z > pos2d.z {
                        pos2d.z = corner.z;
                    }
                }
                let clip = render_config.culling_dist;
                if pos2d.z < -clip || pos2d.x > clip || pos2d.y > clip {
                    continue;
                }
            }
            if !render_config.lod_enable[tid.0] {
                continue;
            }

            let mut tile_uniforms =
                TileUniforms::from_tile(tile_instance, option_render_data_value);
            if let Some(render_data_value) = option_render_data_value {
                tile_uniforms.single_draw = 1;
                tile_uniforms.single_lod_id = render_data_value.single_lod_id;
            }
            if render_config.debug_log && render_data_key.tid.len() >= 9 {
                log! {"{:?}", tile_instance};
                log! {"{:?}", render_data_key};
                log! {"{:?}", option_render_data_value};
            }

            queue.write_buffer(
                &self.tile_uniforms_buffer,
                (i * tile_uniforms_block_size) as BufferAddress,
                bytemuck::bytes_of(&tile_uniforms),
            );

            let splat_count: u32;
            if let Some(render_data_value) = option_render_data_value {
                splat_count = render_data_value.splat_count as u32;
                let size_byte = splat_count as u64 * 4;

                queue.write_buffer(
                    &self.gs_index_buffer,
                    buffer_offset,
                    bytemuck::cast_slice(render_data_value.gs_index.as_slice()),
                );
                render_pass.set_vertex_buffer(
                    1,
                    self.gs_index_buffer
                        .slice(buffer_offset..buffer_offset + size_byte),
                );

                queue.write_buffer(
                    &self.map_id_buffer,
                    buffer_offset,
                    bytemuck::cast_slice(render_data_value.gs_map_id.as_slice()),
                );
                render_pass.set_vertex_buffer(
                    2,
                    self.map_id_buffer
                        .slice(buffer_offset..buffer_offset + size_byte),
                );

                if render_data_value.single_lod_id == -1 {
                    queue.write_buffer(
                        &self.lod_id_buffer,
                        buffer_offset,
                        bytemuck::cast_slice(
                            render_data_value.gs_lod_id.as_ref().unwrap().as_slice(),
                        ),
                    );
                    render_pass.set_vertex_buffer(
                        3,
                        self.lod_id_buffer
                            .slice(buffer_offset..buffer_offset + size_byte),
                    );
                } else {
                    render_pass.set_vertex_buffer(3, self.lod_id_buffer.slice(..));
                }

                buffer_offset += size_byte;
            } else {
                let base_data: &BufferDataValue;
                if let TileTransitionStatus::Changing(to_lower) = tile_instance.transition_status {
                    if to_lower {
                        base_data = &self.buffer_base_data[tid.0][tid.1][tile_instance.view_id];
                    } else {
                        base_data = &self.buffer_base_data[tid.0 - 1][tid.1][tile_instance.view_id];
                    }
                } else {
                    base_data = &self.buffer_base_data[tid.0][tid.1][tile_instance.view_id];
                }
                splat_count = base_data.splat_count;

                render_pass.set_vertex_buffer(1, base_data.gs_index_buffer.slice(..));
                render_pass.set_vertex_buffer(2, self.map_id_buffer.slice(..));
                render_pass
                    .set_vertex_buffer(3, base_data.lod_id_buffer.as_ref().unwrap().slice(..));
            }

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.scene_bind_group, &[]);
            render_pass.set_bind_group(
                1,
                &self.tile_bind_group,
                &[(i * tile_uniforms_block_size) as u32],
            );
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

            render_pass.draw(0..6, 0..splat_count);
        }
    }
}

struct BufferDataValue {
    splat_count: u32,
    gs_index_buffer: wgpu::Buffer,
    map_id_buffer: Option<wgpu::Buffer>,
    lod_id_buffer: Option<wgpu::Buffer>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniforms {
    splat_scale: f32,
    tile_width: f32,
    use_clip: u32,
    clip_height: f32,
    surface_type: u32,
    sphere_radius: f32,
    point_cloud_radius: f32,
    transition_width_ratio: f32,
    num_lod: u32,
    draw_mode: u32,

    map_half_wh: [u32; 2],
    center_coord: [i32; 2],
    _pad0: [u32; 2],
    transition_dist_vec: [f32; 16],
    height_map_scale: [f32; 4],
    scene_scale: [f32; 4],
}
impl SceneUniforms {
    fn expand_to_array<const N: usize, T: Copy>(slice: &[T], pad_value: T) -> [T; N] {
        let mut arr = [pad_value; N];
        let len = slice.len().min(N); // avoid overflow
        arr[..len].copy_from_slice(&slice[..len]);
        arr
    }

    fn from_data(user_data: &UserData, scene_data: &SceneData, render_data: &RenderData) -> Self {
        let render_config = &render_data.render_config;
        Self {
            splat_scale: render_config.splat_scale,
            tile_width: user_data.tile_width,
            use_clip: render_config.use_clip as u32,
            clip_height: render_config.clip_height,
            surface_type: user_data.surface_type as u32,
            sphere_radius: user_data.sphere_radius,
            point_cloud_radius: if render_config.draw_point_cloud {
                render_config.point_cloud_radius
            } else {
                0.0
            },
            transition_width_ratio: user_data.lod_transition_width_ratio,
            num_lod: user_data.n_tiles.1 as u32,
            draw_mode: render_config.draw_mode as u32,

            map_half_wh: [
                user_data.tile_map_half_wh.x as u32,
                user_data.tile_map_half_wh.y as u32,
            ],
            center_coord: [scene_data.center_coord.x, scene_data.center_coord.y],
            _pad0: [0; 2],
            transition_dist_vec: Self::expand_to_array::<16, f32>(
                &user_data.lod_transition_dist,
                0.0,
            ),
            scene_scale: [
                render_config.scene_scale.x,
                render_config.scene_scale.y,
                render_config.scene_scale.z,
                0.0,
            ],
            height_map_scale: [
                user_data.height_map_scale.x,
                user_data.height_map_scale.y,
                user_data.height_map_scale.z * render_config.height_map_scale_v,
                0.0,
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TileUniforms {
    single_draw: u32,
    map_index: u32,
    single_lod_id: i32,
    valid_lod_id: i32,
    changing: u32,
    changing_to_lower: i32,
    _pad0: [u32; 2],

    tile_id: [u32; 4],
    offset: [f32; 4],
    map_coord: [u32; 4],
}
impl TileUniforms {
    fn from_tile(tile: &TileInstance, render_data_value: &Option<RenderDataValue>) -> Self {
        let mut uniforms = Self {
            single_draw: 0,
            map_index: tile.map_index as u32,
            single_lod_id: -1,
            valid_lod_id: -1,
            changing: 0,
            changing_to_lower: -1,
            _pad0: [0; 2],

            tile_id: [tile.tid.0 as u32, tile.tid.1 as u32, tile.view_id as u32, 0],
            offset: [
                tile.tile_offset.x,
                tile.tile_offset.y,
                tile.tile_offset.z,
                0.0,
            ],
            map_coord: [tile.map_coord.x as u32, tile.map_coord.y as u32, 0, 0],
        };

        if let Some(data_value) = render_data_value {
            uniforms.single_draw = 1;
            uniforms.single_lod_id = data_value.single_lod_id;
            uniforms.changing = (uniforms.single_lod_id == -1) as u32;
        } else {
            if let TileTransitionStatus::Changing(to_lower) = tile.transition_status {
                uniforms.changing = 1;
                uniforms.changing_to_lower = to_lower as i32;
            } else {
                uniforms.valid_lod_id = tile.tid.0 as i32;
            }
        }

        uniforms
    }
}
