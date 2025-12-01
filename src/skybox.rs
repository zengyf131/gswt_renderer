use std::{io::Cursor, sync::mpsc};

use wgpu::util::DeviceExt;

use crate::camera::Camera;
use crate::log; // macro import
use crate::texture::Texture;
use crate::utils::*;

pub enum SkyboxTexture {
    Cubemap(Vec<Vec<f32>>),
    HDRI(Vec<f32>),
}

// https://learnopengl.com/PBR/IBL/Diffuse-irradiance
// ChatGPT
pub struct Skybox {
    render_pipeline: wgpu::RenderPipeline,
    bake_pipeline: wgpu::RenderPipeline,

    vertex_buffer: wgpu::Buffer,

    uniforms_buffer: wgpu::Buffer,
    uniforms_bind_group: wgpu::BindGroup,

    skybox_texture_bind_group_layout: wgpu::BindGroupLayout,
    skybox_texture: Option<Texture>,
    skybox_texture_bind_group: Option<wgpu::BindGroup>,

    equi_bind_group_layout: wgpu::BindGroupLayout,

    is_equi: bool,
}
impl Skybox {
    const CUBEMAP_RESO: u32 = 2048; // Choose a resolution for each face
    const SKYBOX_VERTICES: &'static [Vertex; 36] = &[
        // Front face
        Vertex {
            position: [-1.0, -1.0, 1.0],
        },
        Vertex {
            position: [1.0, -1.0, 1.0],
        },
        Vertex {
            position: [1.0, 1.0, 1.0],
        },
        Vertex {
            position: [-1.0, -1.0, 1.0],
        },
        Vertex {
            position: [-1.0, 1.0, 1.0],
        },
        Vertex {
            position: [1.0, 1.0, 1.0],
        },
        // Back face
        Vertex {
            position: [-1.0, -1.0, -1.0],
        },
        Vertex {
            position: [1.0, -1.0, -1.0],
        },
        Vertex {
            position: [1.0, 1.0, -1.0],
        },
        Vertex {
            position: [-1.0, -1.0, -1.0],
        },
        Vertex {
            position: [-1.0, 1.0, -1.0],
        },
        Vertex {
            position: [1.0, 1.0, -1.0],
        },
        // Left face
        Vertex {
            position: [-1.0, -1.0, -1.0],
        },
        Vertex {
            position: [-1.0, -1.0, 1.0],
        },
        Vertex {
            position: [-1.0, 1.0, 1.0],
        },
        Vertex {
            position: [-1.0, -1.0, -1.0],
        },
        Vertex {
            position: [-1.0, 1.0, 1.0],
        },
        Vertex {
            position: [-1.0, 1.0, -1.0],
        },
        // Right face
        Vertex {
            position: [1.0, -1.0, -1.0],
        },
        Vertex {
            position: [1.0, -1.0, 1.0],
        },
        Vertex {
            position: [1.0, 1.0, 1.0],
        },
        Vertex {
            position: [1.0, -1.0, -1.0],
        },
        Vertex {
            position: [1.0, 1.0, -1.0],
        },
        Vertex {
            position: [1.0, 1.0, 1.0],
        },
        // Top face
        Vertex {
            position: [-1.0, 1.0, -1.0],
        },
        Vertex {
            position: [1.0, 1.0, -1.0],
        },
        Vertex {
            position: [1.0, 1.0, 1.0],
        },
        Vertex {
            position: [-1.0, 1.0, -1.0],
        },
        Vertex {
            position: [-1.0, 1.0, 1.0],
        },
        Vertex {
            position: [1.0, 1.0, 1.0],
        },
        // Bottom face
        Vertex {
            position: [-1.0, -1.0, -1.0],
        },
        Vertex {
            position: [1.0, -1.0, -1.0],
        },
        Vertex {
            position: [1.0, -1.0, 1.0],
        },
        Vertex {
            position: [-1.0, -1.0, -1.0],
        },
        Vertex {
            position: [-1.0, -1.0, 1.0],
        },
        Vertex {
            position: [1.0, -1.0, 1.0],
        },
    ];

    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let uniforms_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("Skybox uniform_bind_group_layout"),
            });

        let skybox_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("Skybox skybox_texture_bind_group_layout"),
            });

        let equi_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("Skybox equi_bind_group_layout"),
            });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Skybox Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("skybox.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Skybox Pipeline Layout"),
                bind_group_layouts: &[
                    &uniforms_bind_group_layout,
                    &skybox_texture_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let bake_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Skybox Bake Pipeline Layout"),
            bind_group_layouts: &[
                &uniforms_bind_group_layout,
                &skybox_texture_bind_group_layout,
                &equi_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Skybox Pipeline"),
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let bake_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Skybox Baking Pipeline"),
            layout: Some(&bake_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_bake"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_bake"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba32Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Vertex buffer
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Skybox Vertex Buffer"),
            contents: bytemuck::cast_slice(Self::SKYBOX_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Skybox Uniforms Buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size: std::mem::size_of::<Uniforms>() as u64,
            mapped_at_creation: false,
        });

        let uniforms_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniforms_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms_buffer.as_entire_binding(),
            }],
            label: Some("Skybox uniforms_bind_group"),
        });

        Self {
            render_pipeline,
            bake_pipeline,

            vertex_buffer,
            uniforms_buffer,
            uniforms_bind_group,

            skybox_texture_bind_group_layout,
            skybox_texture: None,
            skybox_texture_bind_group: None,

            equi_bind_group_layout,

            is_equi: false,
        }
    }

    pub fn configure(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &(SkyboxTexture, Vector2<usize>),
    ) {
        let tex_wh = vec2(texture.1.x as u32, texture.1.y as u32);

        let skybox_tex: wgpu::Texture;
        match &texture.0 {
            SkyboxTexture::Cubemap(tex) => {
                self.is_equi = false;

                skybox_tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Skybox Texture"),
                    size: wgpu::Extent3d {
                        width: tex_wh.x,
                        height: tex_wh.y,
                        depth_or_array_layers: 6,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba32Float,
                    usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });

                for i in 0..6 {
                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            aspect: wgpu::TextureAspect::All,
                            texture: &skybox_tex,
                            mip_level: 0,
                            origin: wgpu::Origin3d {
                                x: 0,
                                y: 0,
                                z: i as u32,
                            },
                        },
                        transmute_slice(tex[i].as_slice()),
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(16 * tex_wh.x),
                            rows_per_image: Some(tex_wh.y),
                        },
                        wgpu::Extent3d {
                            width: tex_wh.x,
                            height: tex_wh.y,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
            SkyboxTexture::HDRI(tex) => {
                self.is_equi = true;
                skybox_tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Skybox Texture"),
                    size: wgpu::Extent3d {
                        width: Self::CUBEMAP_RESO,
                        height: Self::CUBEMAP_RESO,
                        depth_or_array_layers: 6,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                self.bake_skybox(device, queue, &skybox_tex, &tex, tex_wh);
            }
        }

        let skybox_view = skybox_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let skybox_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let skybox_texture = Texture {
            texture: skybox_tex,
            view: skybox_view,
            sampler: Some(skybox_sampler),
        };

        let skybox_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.skybox_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&skybox_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(
                        skybox_texture.sampler.as_ref().unwrap(),
                    ),
                },
            ],
            label: Some("Skybox skybox_bind_group"),
        });

        self.skybox_texture = Some(skybox_texture);
        self.skybox_texture_bind_group = Some(skybox_texture_bind_group);
    }

    pub fn render(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        camera: &Camera,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            ..Default::default()
        });

        let mut uniforms = Uniforms::from_camera(camera);
        uniforms.equirectangular = self.is_equi as u32;

        queue.write_buffer(&self.uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.uniforms_bind_group, &[]);
        render_pass.set_bind_group(1, &self.skybox_texture_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..36, 0..1);
    }

    fn bake_skybox(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        skybox_texture: &wgpu::Texture,
        equi_texture: &Vec<f32>,
        tex_wh: Vector2<u32>,
    ) {
        // Bake equirectangular_texture
        let dummy_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Skybox Bake dummy_tex"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let dummy_view = dummy_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        let dummy_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let dummy_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.skybox_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&dummy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&dummy_sampler),
                },
            ],
            label: Some("Skybox Bake dummy_texture_bind_group"),
        });

        let equi = Texture::from_bytes(
            device,
            queue,
            transmute_slice::<_, u8>(equi_texture.as_slice()),
            tex_wh.x,
            tex_wh.y,
            16,
            wgpu::TextureFormat::Rgba32Float,
            wgpu::FilterMode::Linear,
            wgpu::AddressMode::Repeat,
            Some("Skybox Equirectangular Texture"),
        )
        .unwrap();

        let equi_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.equi_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&equi.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(equi.sampler.as_ref().unwrap()),
                },
            ],
            label: Some("Skybox equi_bind_group"),
        });

        let skybox_views: Vec<wgpu::TextureView> = (0..6)
            .map(|i| {
                skybox_texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        let bake_projection = perspective(degrees(90.0), 1.0, 0.1, 10.0);
        let bake_views = [
            Mat4::look_at_rh(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                vec3(0.0, 1.0, 0.0),
            ), // +X
            Mat4::look_at_rh(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(-1.0, 0.0, 0.0),
                vec3(0.0, 1.0, 0.0),
            ), // -X
            Mat4::look_at_rh(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                vec3(0.0, 0.0, 1.0),
            ), // +Y
            Mat4::look_at_rh(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, -1.0, 0.0),
                vec3(0.0, 0.0, -1.0),
            ), // -Y
            Mat4::look_at_rh(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 0.0, -1.0),
                vec3(0.0, 1.0, 0.0),
            ), // +Z
            Mat4::look_at_rh(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 0.0, 1.0),
                vec3(0.0, 1.0, 0.0),
            ), // -Z
        ];

        // Start baking
        for i in 0..6 {
            log!("Baking skybox face {i}");

            let uniforms = Uniforms {
                equirectangular: 0,
                _padding: [0; 3],
                view: bake_views[i].into(),
                projection: bake_projection.into(),
            };
            queue.write_buffer(&self.uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Skybox bake_encoder"),
            });

            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Skybox bake_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &skybox_views[i],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });

                rpass.set_pipeline(&self.bake_pipeline);
                rpass.set_bind_group(0, &self.uniforms_bind_group, &[]);
                rpass.set_bind_group(1, &dummy_texture_bind_group, &[]);
                rpass.set_bind_group(2, &equi_bind_group, &[]);
                rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                rpass.draw(0..36, 0..1);
            }

            queue.submit(Some(encoder.finish()));
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
}
impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];

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
    equirectangular: u32,
    _padding: [u32; 3],

    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
}
impl Uniforms {
    fn from_camera(cam: &Camera) -> Self {
        Self {
            equirectangular: 0,
            _padding: [0; 3],

            view: (*cam.view()).into(),
            projection: (*cam.projection()).into(),
        }
    }
}

pub fn upload_skybox() -> mpsc::Receiver<(SkyboxTexture, Vector2<usize>)> {
    // https://matheowis.github.io/HDRI-to-CubeMap/
    const FILE_STR: [&str; 6] = ["px.png", "nx.png", "py.png", "ny.png", "pz.png", "nz.png"];

    let (tx, rx) = mpsc::channel();
    let task = rfd::AsyncFileDialog::new()
        .add_filter("Skybox images", &["png", "exr"])
        .pick_files();

    execute_future(async move {
        let file_vec = task.await;

        if file_vec.is_none() {
            return;
        }

        let mut cubemap: [Option<Vec<f32>>; 6] = [None, None, None, None, None, None];
        let mut hdri: Option<Vec<f32>> = None;
        let mut width: usize = 0;
        let mut height: usize = 0;
        let mut is_hdri = false;
        for f in file_vec.expect("Invalid file!") {
            // hdri
            if f.file_name().contains(".exr") {
                is_hdri = true;

                let buffer = f.read().await;

                let img = image::ImageReader::new(Cursor::new(buffer))
                    .with_guessed_format()
                    .expect("Failed to guess format.")
                    .decode()
                    .expect("Failed to open image.");
                width = img.width() as usize;
                height = img.height() as usize;

                let rgba = img.to_rgba32f();
                let raw: Vec<f32> = rgba.into_raw();
                hdri = Some(raw);

                break;
            }

            // Cubemap
            for i in 0..6 {
                if !f.file_name().contains(FILE_STR[i]) {
                    continue;
                }
                let buffer = f.read().await;

                /* png crate
                let decoder = png::Decoder::new(Cursor::new(buffer));
                let mut reader = decoder.read_info().unwrap();
                let mut buf = vec![0; reader.output_buffer_size()];
                let info = reader.next_frame(&mut buf).unwrap();
                let raw = buf[..info.buffer_size()].to_vec();
                log!("skybox {} buffer size: {}", i, info.buffer_size());
                let w = info.width;
                let h = info.height;
                log!("skybox wh: {}, {}", w, h);
                log!("skybox first pixel: {}, {}, {}, {}", raw[0], raw[1], raw[2], raw[3]);
                */

                let img = image::ImageReader::new(Cursor::new(buffer))
                    .with_guessed_format()
                    .expect("Failed to guess format.")
                    .decode()
                    .expect("Failed to open image.");
                let w = img.width() as usize;
                let h = img.height() as usize;

                // Ensure all images have the same dimensions
                if width == 0 && height == 0 {
                    width = w;
                    height = h;
                } else if width != w || height != h {
                    panic!("Skybox images must have the same dimensions!");
                }

                let rgba = img.to_rgba32f();
                let raw = rgba.into_raw();
                cubemap[i] = Some(raw);
            }
        }

        if is_hdri {
            let texture: SkyboxTexture = SkyboxTexture::HDRI(hdri.unwrap());
            tx.send((texture, vec2(width, height)))
                .expect("Error sending skybox texture to main thread.");
        } else {
            let mut cubemap_vec: Vec<Vec<f32>> = Vec::new();
            for i in 0..6 {
                cubemap_vec.push(cubemap[i].clone().expect("Missing some cubemap images."));
            }
            let texture: SkyboxTexture = SkyboxTexture::Cubemap(cubemap_vec);
            tx.send((texture, vec2(width, height)))
                .expect("Error sending skybox texture to main thread.");
        }
    });

    rx
}
