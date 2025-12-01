use std::{f32, sync::Arc, sync::mpsc};

use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::*,
    event_loop::ActiveEventLoop,
    keyboard::KeyCode,
    window::Window,
};

use crate::camera::Camera;
use crate::control::FlyPathControl;
use crate::control::{CameraControl, KeyboardFlyControl};
use crate::gui::GUI;
use crate::log;
use crate::proxy::Proxy;
use crate::renderer::GSWTRenderer;
use crate::scene::load_scene_zip;
use crate::skybox::Skybox;
use crate::structure::*;
use crate::texture::Texture;
use crate::utils::*;
use crate::wangtile::WangTile;

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    pub window: Arc<Window>,

    gui: GUI,
    gswt_renderer: GSWTRenderer,
    skybox: Skybox,
    proxy: Proxy,

    camera: Camera,
    keyboard_fly_control: KeyboardFlyControl,
    fly_path_control: FlyPathControl,
    channels: MainChannels,
    worker_thread_handle: Option<wasm_thread::JoinHandle<()>>,
    render_data: RenderData,
    input_status: InputStatus,
}
impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let now = get_time_milliseconds();

        // let image_width = 1920;
        // let image_height = 1080;
        // let viewport_size = PhysicalSize::<u32>::new(image_width, image_height);
        // let _ = window.request_inner_size(viewport_size);
        let viewport_size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features {
                    features_wgpu: wgpu::FeaturesWGPU::empty(),
                    features_webgpu: wgpu::FeaturesWebGPU::FLOAT32_FILTERABLE,
                },
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web we'll have to disable some.
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::default()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        log!("{:?}", surface_caps);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| !f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        log!("{:?}", surface_format);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: viewport_size.width,
            height: viewport_size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let camera = Camera::new_perspective(
            viewport_size,
            vec3(0.0, 0.0, 5.0),
            vec3(0.0, 1.0, 5.0),
            vec3(0.0, 0.0, 1.0),
            degrees(45.0),
            0.1,    //0.2,
            2400.0, //200.0,
        );

        let keyboard_fly_control = KeyboardFlyControl::new();
        let fly_path_control = FlyPathControl::new();

        let scene_vec = load_scene_zip().await;

        let max_lod_count = scene_vec.len();
        log!("max lod count: {}", max_lod_count);

        let mut wang = WangTile::new(scene_vec);

        let gui = GUI::new(&device, config.format, window.clone());
        let gswt_renderer = GSWTRenderer::new(&device, &queue, &config, wang.preload());
        let skybox = Skybox::new(&device, &config);
        let proxy = Proxy::new(&device, &config);

        let (channels, worker_thread_handle) = launch_worker_thread(wang);

        log!("Init completed in {}ms", get_time_milliseconds() - now);
        let mut timer = Timer::new();
        timer.start();

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            window,

            gui,
            gswt_renderer,
            skybox,
            proxy,

            camera,
            keyboard_fly_control,
            fly_path_control,
            channels,
            worker_thread_handle: Some(worker_thread_handle),
            render_data: RenderData::new(max_lod_count),
            input_status: InputStatus::new(),
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            // self.config.width = width;
            // self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
            self.camera.set_viewport(width, height);
            self.render_data.depth_texture = Some(Texture::create_depth_texture(
                &self.device,
                &self.config,
                "depth_texture",
            ));
        }
    }

    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, key: KeyCode, pressed: bool) {
        if key == KeyCode::Escape && pressed {
            event_loop.exit();
            if let Some(handle) = self.worker_thread_handle.take() {
                let _ = handle.join();
            }
        }
        self.input_status.update(key, pressed);

        let rd = &mut self.render_data;
        match &mut rd.camera_control_type {
            CameraControl::KeyboardFly => {
                self.keyboard_fly_control.handle_key(key, pressed);
            }
            CameraControl::FlyPath => {}
        }

        if pressed {
            match key {
                KeyCode::KeyM => {
                    rd.show_main_menu = !rd.show_main_menu;
                }
                KeyCode::KeyP => {
                    rd.show_perf_menu = !rd.show_perf_menu;
                }
                _ => {}
            }
        }
    }

    pub fn handle_mouse_input(&mut self, mouse_state: ElementState, button: MouseButton) {

    }

    pub fn handle_mouse_moved(&mut self, position: PhysicalPosition<f64>) {

    }

    pub fn handle_gui(&mut self, event: &WindowEvent) {
        self.gui.handle_input(self.window.clone(), event);
    }

    pub fn update(&mut self) {
        let rd = &mut self.render_data;
        match &mut rd.camera_control_type {
            CameraControl::KeyboardFly => {
                rd.update_worker = self.keyboard_fly_control.update(
                    &mut self.camera,
                    rd.frame_time_ma.calc().0 as f32,
                    rd.lockon_center,
                );
            }
            CameraControl::FlyPath => {
                rd.update_worker = self.fly_path_control.handle_events(&mut self.camera);
            }
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        match self.gui.gui_status {
            GUIStatus::Config => {}
            GUIStatus::PostConfig => {
                // Clear previous build / sort data
                while let Ok(_) = self.channels.rx_scene_data.try_recv() {}
                while let Ok(_) = self.channels.rx_sort_data.try_recv() {}

                if let Ok(wang_user_data) = self.channels.rx_user_data.try_recv() {
                    if wang_user_data.config_id == self.gui.config_user_data.config_id {
                        self.gswt_renderer.configure(
                            &self.device,
                            &self.queue,
                            &wang_user_data,
                            &self.render_data,
                        );
                        if let Some(skybox_rawtex) = &self.render_data.skybox_rawtex {
                            self.render_data.use_skybox = true;
                            self.skybox.configure(
                                &self.device,
                                &self.queue,
                                skybox_rawtex,
                            );
                        }
                        if let Some(proxy_rawtex) = &self.render_data.proxy_rawtex {
                            self.render_data.use_proxy = true;
                            self.proxy.configure(
                                &self.device,
                                &self.queue,
                                &wang_user_data,
                                &self.render_data,
                                proxy_rawtex,
                            );
                        }

                        self.gui.gui_status = GUIStatus::Render;
                        self.gui.config_user_data = wang_user_data;

                        log!("Config {} ready.", self.gui.config_user_data.config_id);
                    }
                }
            }
            GUIStatus::Render => {
                let now = get_time_milliseconds();
                let rd = &mut self.render_data;
                rd.frame_time_ma.add(now - rd.frame_prev);
                rd.frame_prev = now;

                if rd.cur_scene_data_id.is_some() && rd.cur_sort_data_id.is_some() {
                    if let Ok(f) = self.channels.rx_sort_time.try_recv() {
                        rd.sort_time_ma.add(f);
                        rd.sort_trigger_ma.add(1.0);
                    } else {
                        rd.sort_trigger_ma.add(0.0);
                    }

                    if let Ok(f) = self.channels.rx_build_time.try_recv() {
                        rd.build_time_ma.add(f);
                        rd.build_trigger_ma.add(1.0);
                    } else {
                        rd.build_trigger_ma.add(0.0);
                    }

                    if rd.set_cam_clicked {
                        self.camera.set_view(
                            rd.set_cam_pos,
                            rd.set_cam_dir + rd.set_cam_pos,
                            rd.set_cam_up,
                        );
                        rd.set_cam_clicked = false;
                    }
                }

                if rd.update_worker {
                    // Send cam pos to worker thread
                    let _ = self
                        .channels
                        .tx_build_info
                        .send((!rd.lock_tile, *self.camera.position()));

                    // Send view_proj to worker thread
                    if !rd.lock_sort {
                        let _ = self.channels.tx_vp.send(self.camera.view_proj());
                    }
                }

                // Recv scene data from worker thread
                if let Ok(scene) = self.channels.rx_scene_data.try_recv() {
                    if rd.cur_scene_data_id.is_some()
                        && scene.scene_id == rd.cur_scene_data_id.unwrap()
                    {
                        // Second condition impossible for now
                        rd.cur_scene_data = Some(scene);
                    } else {
                        rd.next_scene_data_id = Some(scene.scene_id);
                        rd.next_scene_data = Some(scene);
                    }
                }

                // Recv sort data from worker thread
                if let Ok(sort_data) = self.channels.rx_sort_data.try_recv() {
                    if rd.cur_sort_data_id.is_some()
                        && sort_data.scene_id == rd.cur_sort_data_id.unwrap()
                    {
                        rd.cur_sort_data = Some(sort_data);
                    } else {
                        rd.next_sort_data_id = Some(sort_data.scene_id);
                        rd.next_sort_data = Some(sort_data);
                    }
                }

                // Update scene & sort data only when both are ready and synchronized
                if rd.next_scene_data_id.is_some()
                    && rd.next_sort_data_id.is_some()
                    && rd.next_sort_data_id.unwrap() == rd.next_scene_data_id.unwrap()
                {
                    // log!("main(): Update scene");
                    rd.cur_scene_data = rd.next_scene_data.clone();
                    rd.cur_sort_data = rd.next_sort_data.clone();
                    rd.cur_scene_data_id = rd.next_scene_data_id.clone();
                    rd.cur_sort_data_id = rd.next_sort_data_id.clone();
                    rd.next_scene_data = None;
                    rd.next_sort_data = None;
                    rd.next_scene_data_id = None;
                    rd.next_sort_data_id = None;
                    // log!("main(): Update scene finished");
                }

                if rd.cur_scene_data_id.is_some()
                    && rd.cur_sort_data_id.is_some()
                    && (!rd.freeze_frame || rd.step_frame)
                {
                    rd.step_frame = false;

                    if rd.use_skybox {
                        self.skybox
                            .render(&self.queue, &mut encoder, &view, &self.camera);
                    }

                    if rd.use_proxy {
                        self.proxy
                            .render(&self.queue, &mut encoder, &view, &self.camera, rd);
                    }

                    if rd.render_gs {
                        self.gswt_renderer.render(
                            &self.queue,
                            &mut encoder,
                            &view,
                            &self.camera,
                            rd,
                        );
                    }
                }
            }
        }

        // GUI render
        {
            let screen_descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [self.config.width, self.config.height],
                pixels_per_point: self.window.scale_factor() as f32,
            };

            self.gui.begin_frame(self.window.clone());

            self.gui.render(
                &mut self.channels,
                &self.camera,
                &mut self.render_data,
                &mut self.fly_path_control,
            );

            self.gui.end_frame_and_draw(
                &self.device,
                &self.queue,
                &mut encoder,
                &self.window,
                &view,
                screen_descriptor,
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub fn launch_worker_thread(mut wang: WangTile) -> (MainChannels, wasm_thread::JoinHandle<()>) {
    let (tx_vp, rx_vp) = mpsc::channel::<Mat4>();
    let (tx_build_info, rx_build_info) = mpsc::channel::<(bool, Vec3)>(); // (do_build, camera_pos)
    let (tx_main_user_data, rx_worker_user_data) = mpsc::channel::<UserData>();

    let (tx_worker_user_data, rx_main_user_data) = mpsc::channel::<UserData>(); // Post config user data
    let (tx_sort_data, rx_sort_data) = mpsc::channel::<SortData>();
    let (tx_scene_data, rx_scene_data) = mpsc::channel::<SceneData>();
    let (tx_sort_time, rx_sort_time) = mpsc::channel::<f64>();
    let (tx_build_time, rx_build_time) = mpsc::channel::<f64>();

    let main_channels = MainChannels {
        tx_vp,
        tx_build_info,
        tx_user_data: tx_main_user_data,
        rx_user_data: rx_main_user_data,
        rx_sort_data,
        rx_scene_data,
        rx_sort_time,
        rx_build_time,
        rx_fly_path_control: None,
        rx_height_tex: None,
        rx_skybox_tex: None,
        rx_proxy_tex: None,
    };

    let worker_channels = WorkerChannels {
        rx_vp,
        rx_build_info,
        rx_user_data: rx_worker_user_data,
        tx_user_data: tx_worker_user_data,
        tx_sort_data,
        tx_scene_data,
        tx_sort_time,
        tx_build_time,
    };

    // launch another thread for view-dependent splat sorting
    let thread_handle = wasm_thread::spawn({
        let mut cur_camera_pos: Option<Vector3<f32>> = None;
        let mut prev_vp: Option<Mat4> = None;
        let mut next_scene_id: u32 = 0;

        move || loop {
            if let Ok(user_data) = worker_channels.rx_user_data.try_recv() {
                let wang_user_data = wang.configure(user_data);
                worker_channels
                    .tx_user_data
                    .send(wang_user_data)
                    .expect("Error sending wang user data");
                cur_camera_pos = None;
                prev_vp = None;
            }

            let mut recv_build = false;
            let mut do_build = false;
            let mut camera_pos = Vec3::zero();
            while let Ok((a, b)) = worker_channels.rx_build_info.try_recv() {
                recv_build = true;
                do_build = a;
                camera_pos = b;
            }
            if recv_build {
                cur_camera_pos = Some(camera_pos);

                if do_build && wang.check_update(&camera_pos) {
                    let start = get_time_milliseconds();
                    let mut scene_data = wang.build_tiles(camera_pos);
                    scene_data.scene_id = next_scene_id;
                    let build_time = get_time_milliseconds() - start;

                    let _ = worker_channels.tx_scene_data.send(scene_data);
                    let _ = worker_channels.tx_build_time.send(build_time);
                    next_scene_id += 1;
                }
            }

            let mut recv_vp = false;
            let mut view_proj = Mat4::identity();
            while let Ok(a) = worker_channels.rx_vp.try_recv() {
                recv_vp = true;
                view_proj = a;
            }
            if recv_vp {
                if cur_camera_pos.is_none() {
                    continue;
                }
                if !wang.user_data.always_sort && prev_vp.is_some() {
                    let diff = prev_vp.unwrap() - view_proj;
                    let diff = diff[0][0].abs()
                        + diff[0][1].abs()
                        + diff[0][2].abs()
                        + diff[0][3].abs()
                        + diff[1][0].abs()
                        + diff[1][1].abs()
                        + diff[1][2].abs()
                        + diff[1][3].abs()
                        + diff[2][0].abs()
                        + diff[2][1].abs()
                        + diff[2][2].abs()
                        + diff[2][3].abs()
                        + diff[3][0].abs()
                        + diff[3][1].abs()
                        + diff[3][2].abs()
                        + diff[3][3].abs();
                    if diff < 0.01 {
                        continue;
                    }
                }
                prev_vp = Some(view_proj);

                let start = get_time_milliseconds();
                // TODO: fix cur_camera_pos when lock tile
                let mut sort_data = wang.sort_tiles(cur_camera_pos.unwrap(), view_proj);
                sort_data.scene_id = next_scene_id - 1;
                let sort_time = get_time_milliseconds() - start;

                let _ = worker_channels.tx_sort_data.send(sort_data);
                let _ = worker_channels.tx_sort_time.send(sort_time);
            }
        }
    });

    (main_channels, thread_handle)
}
