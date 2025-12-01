use std::{
    collections::VecDeque,
    sync::mpsc::{Receiver, Sender},
};
use winit::keyboard::KeyCode;

use crate::control::{CameraControl, FlyPathControl};
use crate::scene::Scene;
use crate::skybox::SkyboxTexture;
use crate::texture::Texture;
use crate::utils::*;

/// All user data from Config stage
#[derive(Clone)]
pub struct UserData {
    /// ID for this config
    pub config_id: u32,
    /// Half of width/height for tile map (in number of tiles)
    /// 
    /// Actual width/height computed based on surface type (2n or 2n+1)
    pub tile_map_half_wh: Vector2<usize>,
    /// Number of center options for each tile (less equal than that provided during upload)
    pub center_option: usize,
    /// The distance (squared) that the camera needs to travel until a tile map update is triggered
    pub update_distance2: f32,
    /// Width of a tile
    pub tile_width: f32,

    pub tile_sort_type: TileSortType,

    // Surface
    pub surface_type: SurfaceType,
    pub height_map_wh: Vector2<usize>,
    pub height_map_type: HeightMapType,
    pub height_map_scale: Vec3,
    pub height_tex: Option<(Vec<f32>, Vector2<usize>)>,
    pub sphere_radius: f32,

    // LOD
    pub lod_max_dist: f32,
    pub lod_blending: bool,
    pub lod_transition_width_ratio: f32,
    pub lod_bbox_check: bool,
    pub lod_dist_tolerance: f32,

    // Selective merging
    pub merge_type: SelectiveMergeType,
    pub merge_tile_dist: (i32, i32),
    pub merge_dot_threshold: f32,
    pub merge_topk: usize,
    pub use_cache: bool,
    pub cache_size: usize,

    pub reset_rng: bool,
    pub always_sort: bool,

    // From wang thread
    /// Actual width/height for the tile map (in number of tiles)
    pub tile_map_wh: Vector2<usize>,
    pub height_map: Vec<f32>,
    /// A list of transition distances of each lod
    pub lod_transition_dist: Vec<f32>,
    /// n_lod, n_tile, n_view
    pub n_tiles: (usize, usize, usize),
}
impl UserData {
    pub fn new() -> Self {
        Self {
            config_id: 0,
            tile_map_half_wh: Vector2::new(48, 48),
            center_option: 1,
            update_distance2: 1.0,
            tile_width: 4.0,
            tile_sort_type: TileSortType::Graph,
            surface_type: SurfaceType::HeightMap,
            height_map_wh: Vector2::new(0, 0),
            height_map_type: HeightMapType::Random,
            height_map_scale: vec3(1.0, 1.0, 0.0),
            height_tex: None,
            sphere_radius: 0.0,
            lod_max_dist: 0.0,
            lod_blending: true,
            lod_transition_width_ratio: 0.0,
            lod_bbox_check: true,
            lod_dist_tolerance: 0.0,
            merge_type: SelectiveMergeType::Edge,
            merge_tile_dist: (-1, -1),
            merge_dot_threshold: 3.0,
            merge_topk: 100,
            use_cache: true,
            cache_size: 1024,
            reset_rng: true,
            always_sort: false,

            tile_map_wh: Vector2::new(0, 0),
            height_map: Vec::new(),
            lod_transition_dist: Vec::new(),
            n_tiles: (0, 0, 0),
        }
    }
}

/// String version of [UserData] to help with config parsing
pub struct UserDataString {
    pub tile_map_half_wh_s: Vector2<String>,
    pub center_option_s: String,
    pub update_dist_s: String,
    pub tile_width_s: String,
    pub height_map_wh_s: Vector2<String>,
    pub height_map_scale_s: Vector2<String>,
    pub sphere_radius_s: String,
    pub merge_tile_dist_s: Vector2<String>,
    pub merge_dot_threshold_s: String,
    pub merge_topk_s: String,
    pub lod_max_dist_s: String,
    pub lod_transition_width_ratio_s: String,
    pub lod_dist_tolerance_s: String,
    pub cache_size_s: String,
}
impl UserDataString {
    pub fn new() -> Self {
        Self {
            tile_map_half_wh_s: vec2(48.to_string(), 48.to_string()),
            center_option_s: 1.to_string(),
            update_dist_s: 1.to_string(),
            tile_width_s: 4.to_string(),
            height_map_wh_s: vec2(10.to_string(), 10.to_string()),
            height_map_scale_s: vec2(1.to_string(), 1.to_string()),
            sphere_radius_s: 20.to_string(),
            merge_tile_dist_s: vec2(3.to_string(), 10.to_string()),
            merge_dot_threshold_s: 0.2.to_string(),
            merge_topk_s: 100.to_string(),
            lod_max_dist_s: 96.to_string(),
            lod_transition_width_ratio_s: String::from("0.05"),
            lod_dist_tolerance_s: 0.to_string(),
            cache_size_s: 1024.to_string(),
        }
    }

    pub fn to_raw(&self, user_data: &mut UserData, err_msg: &mut Option<String>) {
        parse_num(
            &self.tile_map_half_wh_s.x,
            &mut user_data.tile_map_half_wh.x,
            err_msg,
        );
        parse_num(
            &self.tile_map_half_wh_s.y,
            &mut user_data.tile_map_half_wh.y,
            err_msg,
        );
        parse_num(&self.center_option_s, &mut user_data.center_option, err_msg);
        parse_num(
            &self.update_dist_s,
            &mut user_data.update_distance2,
            err_msg,
        );
        user_data.update_distance2 = user_data.update_distance2.powi(2);
        parse_num(&self.tile_width_s, &mut user_data.tile_width, err_msg);
        parse_num(
            &self.height_map_wh_s.x,
            &mut user_data.height_map_wh.x,
            err_msg,
        );
        parse_num(
            &self.height_map_wh_s.y,
            &mut user_data.height_map_wh.y,
            err_msg,
        );
        parse_num(
            &self.height_map_scale_s.x,
            &mut user_data.height_map_scale.x,
            err_msg,
        );
        user_data.height_map_scale.y = user_data.height_map_scale.x;
        parse_num(
            &self.height_map_scale_s.y,
            &mut user_data.height_map_scale.z,
            err_msg,
        );
        parse_num(&self.sphere_radius_s, &mut user_data.sphere_radius, err_msg);
        parse_num(
            &self.merge_tile_dist_s.x,
            &mut user_data.merge_tile_dist.0,
            err_msg,
        );
        parse_num(
            &self.merge_tile_dist_s.y,
            &mut user_data.merge_tile_dist.1,
            err_msg,
        );
        parse_num(
            &self.merge_dot_threshold_s,
            &mut user_data.merge_dot_threshold,
            err_msg,
        );
        parse_num(&self.merge_topk_s, &mut user_data.merge_topk, err_msg);

        parse_num(&self.lod_max_dist_s, &mut user_data.lod_max_dist, err_msg);
        user_data.lod_max_dist *= user_data.tile_width;
        parse_num(
            &self.lod_transition_width_ratio_s,
            &mut user_data.lod_transition_width_ratio,
            err_msg,
        );
        parse_num(
            &self.lod_dist_tolerance_s,
            &mut user_data.lod_dist_tolerance,
            err_msg,
        );
        parse_num(&self.cache_size_s, &mut user_data.cache_size, err_msg);
    }
}

pub struct RenderData {
    pub cur_scene_data: Option<SceneData>,
    pub next_scene_data: Option<SceneData>,
    pub cur_sort_data: Option<SortData>,
    pub next_sort_data: Option<SortData>,
    pub cur_scene_data_id: Option<u32>,
    pub next_scene_data_id: Option<u32>,
    pub cur_sort_data_id: Option<u32>,
    pub next_sort_data_id: Option<u32>,

    pub frame_prev: f64,
    pub time_ma_window: usize,
    pub frame_time_ma: IncrementalMA,
    pub sort_time_ma: IncrementalMA,
    pub build_time_ma: IncrementalMA,
    pub sort_trigger_ma: IncrementalMA,
    pub build_trigger_ma: IncrementalMA,

    pub show_main_menu: bool,
    pub show_perf_menu: bool,
    pub show_fly_path_menu: bool,
    pub hide_menu_when_start: bool,

    pub camera_control_type: CameraControl,
    pub set_cam_clicked: bool,
    pub cam_pos_s: Vector3<String>,
    pub cam_dir_s: Vector3<String>,
    pub cam_up_s: Vector3<String>,
    pub set_cam_pos: Vec3,
    pub set_cam_dir: Vec3,
    pub set_cam_up: Vec3,
    pub lockon_center: bool,

    pub lock_tile: bool,
    pub lock_sort: bool,
    pub freeze_frame: bool,
    pub step_frame: bool,
    pub update_worker: bool,

    pub render_config: RenderConfig,
    pub render_gs: bool,
    pub use_skybox: bool,
    pub use_proxy: bool,

    pub max_lod_count: usize,

    pub fly_path_error_msg: Option<String>,
    pub fly_path_benchmark: bool,

    pub skybox_rawtex: Option<(SkyboxTexture, Vector2<usize>)>,
    pub proxy_rawtex: Option<(Vec<Vec<f32>>, Vector2<usize>)>,

    pub depth_texture: Option<Texture>,
}
impl RenderData {
    pub fn new(max_lod_count: usize) -> Self {
        let default_ma_window: usize = 200;

        Self {
            cur_scene_data: None,
            next_scene_data: None,
            cur_sort_data: None,
            next_sort_data: None,
            cur_scene_data_id: None,
            next_scene_data_id: None,
            cur_sort_data_id: None,
            next_sort_data_id: None,

            frame_prev: get_time_milliseconds(),
            time_ma_window: default_ma_window,
            frame_time_ma: IncrementalMA::new(default_ma_window),
            sort_time_ma: IncrementalMA::new(default_ma_window),
            build_time_ma: IncrementalMA::new(default_ma_window),
            sort_trigger_ma: IncrementalMA::new(default_ma_window),
            build_trigger_ma: IncrementalMA::new(default_ma_window),

            show_main_menu: true,
            show_perf_menu: false,
            show_fly_path_menu: false,
            hide_menu_when_start: false,

            camera_control_type: CameraControl::KeyboardFly,
            set_cam_clicked: false,
            cam_pos_s: vec3(0.to_string(), 0.to_string(), 0.to_string()),
            cam_dir_s: vec3(0.to_string(), 1.to_string(), 0.to_string()),
            cam_up_s: vec3(0.to_string(), 0.to_string(), 1.to_string()),
            set_cam_pos: Vec3::zero(),
            set_cam_dir: Vec3::zero(),
            set_cam_up: Vec3::zero(),
            lockon_center: false,

            lock_tile: false,
            lock_sort: false,
            freeze_frame: false,
            step_frame: false,
            update_worker: false,

            render_config: RenderConfig::new(max_lod_count),
            render_gs: true,
            use_skybox: false,
            use_proxy: false,

            max_lod_count,

            fly_path_error_msg: None,
            fly_path_benchmark: false,

            skybox_rawtex: None,
            proxy_rawtex: None,

            depth_texture: None,
        }
    }

    pub fn parse_camera_config(&mut self) {
        let mut err: Option<String> = None;
        parse_num(&self.cam_pos_s.x, &mut self.set_cam_pos.x, &mut err);
        parse_num(&self.cam_pos_s.y, &mut self.set_cam_pos.y, &mut err);
        parse_num(&self.cam_pos_s.z, &mut self.set_cam_pos.z, &mut err);
        parse_num(&self.cam_dir_s.x, &mut self.set_cam_dir.x, &mut err);
        parse_num(&self.cam_dir_s.y, &mut self.set_cam_dir.y, &mut err);
        parse_num(&self.cam_dir_s.z, &mut self.set_cam_dir.z, &mut err);
        parse_num(&self.cam_up_s.x, &mut self.set_cam_up.x, &mut err);
        parse_num(&self.cam_up_s.y, &mut self.set_cam_up.y, &mut err);
        parse_num(&self.cam_up_s.z, &mut self.set_cam_up.z, &mut err);

        if err.is_none() {
            self.set_cam_clicked = true;
        }
    }
}

#[derive(Clone)]
pub struct RenderConfig {
    pub draw_mode: DrawMode,
    pub height_map_scale_v: f32,
    pub scene_scale: Vec3,
    pub use_clip: bool,
    pub clip_height: f32,
    pub draw_point_cloud: bool,
    pub point_cloud_radius: f32,
    pub culling_dist: f32,
    pub proxy_full: bool,
    pub proxy_map: bool,
    pub proxy_height: f32,
    pub proxy_width_scale: f32,
    pub proxy_brightness: f32,
    pub proxy_black_background: bool,
    pub lod_enable: Vec<bool>,
    pub debug_log: bool,
    pub splat_scale: f32,
}
impl RenderConfig {
    pub fn new(max_lod_count: usize) -> Self {
        Self {
            draw_mode: DrawMode::Normal,
            height_map_scale_v: 1.0,
            scene_scale: vec3(1.0, 1.0, 1.0),
            use_clip: false,
            clip_height: 0.0,
            draw_point_cloud: false,
            point_cloud_radius: 0.01,
            culling_dist: 1.0,
            proxy_full: false,
            proxy_map: true,
            proxy_height: -0.5,
            proxy_width_scale: 4.0,
            proxy_brightness: 1.0,
            proxy_black_background: false,
            lod_enable: vec![true; max_lod_count],
            debug_log: false,
            splat_scale: 1.0,
        }
    }
}

#[derive(PartialEq, Clone, Copy)]
pub enum DrawMode {
    Normal,
    TileID,
    TileLOD,
    LOD,
    View,
}

pub struct MainChannels {
    pub tx_vp: Sender<Mat4>,
    pub tx_build_info: Sender<(bool, Vec3)>,
    pub tx_user_data: Sender<UserData>,

    pub rx_user_data: Receiver<UserData>,
    pub rx_sort_data: Receiver<SortData>,
    pub rx_scene_data: Receiver<SceneData>,
    pub rx_sort_time: Receiver<f64>,
    pub rx_build_time: Receiver<f64>,

    pub rx_fly_path_control: Option<Receiver<FlyPathControl>>,
    pub rx_height_tex: Option<Receiver<(Vec<f32>, Vector2<usize>)>>,
    pub rx_skybox_tex: Option<Receiver<(SkyboxTexture, Vector2<usize>)>>,
    pub rx_proxy_tex: Option<Receiver<(Vec<Vec<f32>>, Vector2<usize>)>>,
}

pub struct WorkerChannels {
    pub rx_vp: Receiver<Mat4>,
    pub rx_build_info: Receiver<(bool, Vec3)>,
    pub rx_user_data: Receiver<UserData>,

    pub tx_user_data: Sender<UserData>,
    pub tx_sort_data: Sender<SortData>,
    pub tx_scene_data: Sender<SceneData>,
    pub tx_sort_time: Sender<f64>,
    pub tx_build_time: Sender<f64>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum GUIStatus {
    Config,
    PostConfig,
    Render,
}

#[derive(PartialEq, Clone, Copy)]
pub enum SurfaceType {
    None,
    HeightMap,
    Sphere,
}

#[derive(PartialEq, Clone)]
pub enum HeightMapType {
    Texture,
    Random,
    SlopeX,
    SlopeY,
    DualSlope,
}

#[derive(PartialEq, Clone)]
pub enum TileSortType {
    Distance,
    Viewport,
    Object,
    Graph,
}

#[derive(PartialEq, Clone)]
pub enum SelectiveMergeType {
    None,
    Axis,
    Edge,
}

#[derive(Clone)]
pub struct SceneData {
    pub scene_id: u32,
    pub splat_count: usize,
    pub blending_splat_count: usize,
    pub center_coord: Vector2<i32>,
    pub lod_splat_count: Vec<usize>,
    pub lod_instance_count: Vec<usize>,
}
impl SceneData {
    pub fn new() -> Self {
        Self {
            scene_id: 0,
            splat_count: 0,
            blending_splat_count: 0,
            center_coord: vec2(0, 0),
            lod_splat_count: Vec::new(),
            lod_instance_count: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct SortData {
    pub scene_id: u32,
    pub tile_instance_vec: Vec<TileInstance>,
    pub render_data_vec: Vec<(RenderDataKey, Option<RenderDataValue>)>,
}

#[derive(Clone, Debug)]
pub struct TileInstance {
    pub tid: (usize, usize), // (lod_id, tile_id)
    pub view_id: usize,
    pub tile_offset: Vec3,
    pub map_index: usize,
    pub map_coord: Vector2<usize>,
    pub tile_center: Vec3,
    pub merge_status: TileMergeStatus,
    pub transition_status: TileTransitionStatus,
    pub to_local: Mat3,

    pub corner_data: Option<TileCornerData>,
    pub edge_data: Option<TileEdgeData>,
}
impl TileInstance {
    pub fn new() -> Self {
        Self {
            tid: (0, 0),
            view_id: 0,
            tile_offset: Vec3::zero(),
            map_index: 0,
            map_coord: Vector2::zero(),
            tile_center: Vec3::zero(),
            merge_status: TileMergeStatus::None,
            transition_status: TileTransitionStatus::None,
            to_local: Mat3::zero(),

            corner_data: None,
            edge_data: None,
        }
    }

    pub fn from_metadata(tile_inst: &Self) -> Self {
        Self {
            tid: tile_inst.tid.clone(),
            view_id: tile_inst.view_id,
            tile_offset: tile_inst.tile_offset,
            map_index: tile_inst.map_index,
            map_coord: tile_inst.map_coord,
            tile_center: tile_inst.tile_center,
            merge_status: tile_inst.merge_status.clone(),
            transition_status: tile_inst.transition_status.clone(),
            to_local: tile_inst.to_local,
            corner_data: None,
            edge_data: None,
        }
    }
}

#[derive(Clone)]
pub struct TileBaseData {
    pub splat_count: usize,
    pub tile_center: Vec3,
    pub aabb: (Vec3, Vec3),

    pub raw_depth: Vec<i32>,
    pub gs_index: Vec<u32>,
    pub gs_lod_id: Vec<u32>,
}

#[derive(PartialEq, Clone, Debug)]
pub enum TileMergeStatus {
    None,
    MergedFrom(Vec<usize>),
    MergedTo(usize),
}

#[derive(PartialEq, Clone, Debug)]
pub enum TileTransitionStatus {
    None,
    Spawning(f32),  // blending_factor
    Changing(bool), // to_lower
}

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub enum TileTransitionStatusHash {
    None,
    Spawning,
    Changing(bool),
}
impl TileTransitionStatusHash {
    pub fn from_status(status: &TileTransitionStatus) -> Self {
        match status {
            &TileTransitionStatus::None => Self::None,
            &TileTransitionStatus::Spawning(blend_f) => Self::Spawning,
            &TileTransitionStatus::Changing(to_lower) => Self::Changing(to_lower),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TileCornerData {
    pub southwest: (Vec3, Mat3), // (corner pos, to_world)
    pub northwest: (Vec3, Mat3),
    pub northeast: (Vec3, Mat3),
    pub southeast: (Vec3, Mat3),
}
impl TileCornerData {
    pub fn new() -> Self {
        Self {
            southwest: (Vec3::zero(), Mat3::identity()),
            northwest: (Vec3::zero(), Mat3::identity()),
            northeast: (Vec3::zero(), Mat3::identity()),
            southeast: (Vec3::zero(), Mat3::identity()),
        }
    }
}
impl std::ops::Index<usize> for TileCornerData {
    type Output = (Vec3, Mat3);

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.southwest,
            1 => &self.northwest,
            2 => &self.northeast,
            3 => &self.southeast,
            _ => panic!("Index out of range: {}", index),
        }
    }
}
impl std::ops::IndexMut<usize> for TileCornerData {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.southwest,
            1 => &mut self.northwest,
            2 => &mut self.northeast,
            3 => &mut self.southeast,
            _ => panic!("Index out of range: {}", index),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TileEdgeData {
    pub west: (Vec3, Vec3), // (edge pos, edge normal)
    pub north: (Vec3, Vec3),
    pub east: (Vec3, Vec3),
    pub south: (Vec3, Vec3),
}
impl TileEdgeData {
    pub fn new() -> Self {
        Self {
            west: (Vec3::zero(), Vec3::zero()),
            north: (Vec3::zero(), Vec3::zero()),
            east: (Vec3::zero(), Vec3::zero()),
            south: (Vec3::zero(), Vec3::zero()),
        }
    }
}
impl std::ops::Index<usize> for TileEdgeData {
    type Output = (Vec3, Vec3);

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.west,
            1 => &self.north,
            2 => &self.east,
            3 => &self.south,
            _ => panic!("Index out of range: {}", index),
        }
    }
}
impl std::ops::IndexMut<usize> for TileEdgeData {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.west,
            1 => &mut self.north,
            2 => &mut self.east,
            3 => &mut self.south,
            _ => panic!("Index out of range: {}", index),
        }
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct RenderDataKey {
    pub view_id: usize,
    pub tid: Vec<(usize, usize)>,
    pub transition_status: Vec<TileTransitionStatusHash>,
}
impl RenderDataKey {
    pub fn new() -> Self {
        Self {
            view_id: 0,
            tid: Vec::new(),
            transition_status: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RenderDataValue {
    pub splat_count: usize,
    pub gs_index: Vec<u32>,
    pub gs_map_id: Vec<u32>,
    pub merge_from_vec: Vec<usize>,
    pub single_lod_id: i32,
    pub gs_lod_id: Option<Vec<u32>>,
}

#[derive(Clone)]
pub struct MapNeighbor {
    pub west: Option<(Vector2<usize>, usize)>, // (map_coord, which neighbor this is for that)
    pub east: Option<(Vector2<usize>, usize)>,
    pub north: Option<(Vector2<usize>, usize)>,
    pub south: Option<(Vector2<usize>, usize)>,
}
impl MapNeighbor {
    pub fn new() -> Self {
        Self {
            west: None,
            east: None,
            north: None,
            south: None,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Option<(Vector2<usize>, usize)>> {
        [&self.west, &self.north, &self.east, &self.south].into_iter()
    }
}
impl std::ops::Index<usize> for MapNeighbor {
    type Output = Option<(Vector2<usize>, usize)>;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.west,
            1 => &self.north,
            2 => &self.east,
            3 => &self.south,
            _ => panic!("Index out of range: {}", index),
        }
    }
}

pub struct PreloadData<'a> {
    pub tile_splats_merged: &'a Scene,
    pub tile_base_data: &'a mut Vec<Vec<Vec<TileBaseData>>>,
    // pub tile_spawning_data: &'a mut Vec<Vec<Vec<TileTransitionData>>>,
    // pub tile_changing_data: &'a mut Vec<Vec<Vec<TileTransitionData>>>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex2D {
    pub position: [f32; 2],
}
impl Vertex2D {
    const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x2];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;

        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub struct InputStatus {
    pub control_left: bool,
    pub shift_left: bool,
}
impl InputStatus {
    pub fn new() -> Self {
        Self {
            control_left: false,
            shift_left: false,
        }
    }

    pub fn update(&mut self, key: KeyCode, pressed: bool) {
        match key {
            KeyCode::ControlLeft => {
                self.control_left = pressed;
            }
            KeyCode::ShiftLeft => {
                self.shift_left = pressed;
            }
            _ => {}
        }
    }
}
