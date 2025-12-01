use std::{collections::HashSet, f32, io::Cursor, sync::mpsc, usize};

use cgmath::{Point3, Vector2, Vector4, perspective, prelude::*, vec2, vec3};
use lru::LruCache;
use ndarray::Array2;
use petgraph::{
    algo::toposort,
    graph::{DiGraph, NodeIndex},
};

use crate::log; // macro import
use crate::scene::*;
use crate::structure::*;
use crate::utils::*;

use rand::{Rng, SeedableRng, rngs::StdRng};

pub struct WangTile {
    pub user_data: UserData,
    pub tile_splats_vec: Vec<Vec<Scene>>,
    pub n_tiles: (usize, usize, usize), // n_lod, n_tile, n_view
    pub initialized: bool,

    tile_map: Array2<Option<TileInstance>>,
    neighbor_map: Array2<MapNeighbor>,

    center_coord: Vector2<i32>,
    camera_pos: Vec3,

    presort_dirs: Vec<Vec3>,
    rng: StdRng,

    tile_splats_merged: Scene,
    splats_merge_offset: Vec<Vec<u32>>, // lod, tile
    lod_avg_scale: Vec<f32>,
    tile_base_data: Vec<Vec<Vec<TileBaseData>>>, // lod, tile, view
    sort_lru_cache: LruCache<RenderDataKey, RenderDataValue>,
    // sort_lru_cache: LRUCache<RenderDataKey, RenderDataValue, caches::DefaultHashBuilder>,
}
impl WangTile {
    pub fn new(tile_splats_vec: Vec<Vec<Scene>>) -> Self {
        let mut wang = Self {
            user_data: UserData::new(),
            tile_splats_vec,
            n_tiles: (0, 0, 0),
            initialized: false,

            tile_map: Array2::from_elem((1, 1), None),
            neighbor_map: Array2::from_elem((1, 1), MapNeighbor::new()),

            center_coord: Vector2::<i32>::new(0, 0),
            camera_pos: vec3(0.0, 0.0, 0.0),

            presort_dirs: Vec::new(),
            rng: StdRng::seed_from_u64(0),

            tile_splats_merged: Scene::new(),
            splats_merge_offset: Vec::new(),
            lod_avg_scale: Vec::new(),
            tile_base_data: Vec::new(),
            sort_lru_cache: LruCache::new(std::num::NonZeroUsize::new(1).unwrap()),
            // sort_lru_cache: LRUCache::new(1).unwrap(),
        };
        let now = get_time_milliseconds();
        wang.preprocess();
        log!("Wangtile preprocess: {}ms.", get_time_milliseconds() - now);

        wang
    }

    fn preprocess(&mut self) {
        self.n_tiles = (
            self.tile_splats_vec.len(),
            self.tile_splats_vec[0].len(),
            0,
        );

        // Compute aabb & avg center
        let mut aabb_vec: Vec<(Vec3, Vec3)> = Vec::with_capacity(self.n_tiles.1);
        let mut avg_center_vec: Vec<Vec3> = Vec::with_capacity(self.n_tiles.1);
        for tile_id in 0..self.n_tiles.1 {
            let mut tile_aabb: Option<(Vec3, Vec3)> = None;
            let mut tile_avg_center = Vec3::zero();
            let scene = &mut self.tile_splats_vec[0][tile_id];
            let (mut aabb, mut avg_center) = scene.compute_aabb_and_center();

            // Height normalization
            for lod_id in 0..self.n_tiles.0 {
                let scene = &mut self.tile_splats_vec[lod_id][tile_id];
                scene.translate(vec3(0.0, 0.0, -avg_center.z));
            }
            aabb.0.z -= avg_center.z;
            aabb.1.z -= avg_center.z;
            avg_center.z = 0.0;

            if let Some(tile_aabb_ref) = tile_aabb.as_mut() {
                tile_aabb_ref.0 = vec3(
                    tile_aabb_ref.0.x.min(aabb.0.x),
                    tile_aabb_ref.0.y.min(aabb.0.y),
                    tile_aabb_ref.0.z.min(aabb.0.z),
                );
                tile_aabb_ref.1 = vec3(
                    tile_aabb_ref.1.x.max(aabb.1.x),
                    tile_aabb_ref.1.y.max(aabb.1.y),
                    tile_aabb_ref.1.z.max(aabb.1.z),
                );
            } else {
                tile_aabb = Some(aabb);
            }
            tile_avg_center += avg_center;
            tile_avg_center /= self.n_tiles.0 as f32;

            aabb_vec.push(tile_aabb.unwrap());
            avg_center_vec.push(tile_avg_center);
        }

        // Merge tiles
        let mut new_scene = Scene::new();
        self.splats_merge_offset = Vec::with_capacity(self.n_tiles.0);
        for tile_vec in &self.tile_splats_vec {
            let mut tile_offset_vec: Vec<u32> = Vec::with_capacity(self.n_tiles.1);
            for spl in tile_vec {
                tile_offset_vec.push(new_scene.splat_count as u32);
                new_scene.merge(spl);
            }
            self.splats_merge_offset.push(tile_offset_vec);
        }
        new_scene.generate_texture();
        self.tile_splats_merged = new_scene;

        // Compute avg scale
        for l in 0..self.n_tiles.0 {
            let mut lod_scale_sum: f32 = 0.0;
            let mut lod_scale_num: usize = 0;
            for t in 0..self.n_tiles.1 {
                let spl = &self.tile_splats_vec[l][t];
                lod_scale_sum += spl.compute_scale_sum();
                lod_scale_num += spl.splat_count * 3;
            }
            let avg_scale = lod_scale_sum / lod_scale_num as f32;
            log!("Lod {} avg scale: {}", l, avg_scale);
            if l > 0 {
                assert!(avg_scale > self.lod_avg_scale[l - 1]);
            }
            self.lod_avg_scale.push(avg_scale);
        }

        // Pre-sort
        let sort_projection = perspective(degrees(90.0), 1.0, 0.1, 10.0);
        let sort_dirs = Vec::from([
            vec3(1.0, 0.0, 0.0).normalize(),
            vec3(-1.0, 0.0, 0.0).normalize(),
            vec3(0.0, 1.0, 0.0).normalize(),
            vec3(0.0, -1.0, 0.0).normalize(),
            vec3(1.0, 0.0, -1.0).normalize(),
            vec3(-1.0, 0.0, -1.0).normalize(),
            vec3(0.0, 1.0, -1.0).normalize(),
            vec3(0.0, -1.0, -1.0).normalize(),
            vec3(0.0, 0.0, -1.0).normalize(),
        ]);
        self.n_tiles.2 = sort_dirs.len();
        self.presort_dirs = sort_dirs;
        let mut sort_views: Vec<Mat4> = Vec::new();
        for sort_dir in &self.presort_dirs {
            if sort_dir.x != 0.0 || sort_dir.y != 0.0 {
                sort_views.push(Mat4::look_at_rh(
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::from_vec(*sort_dir),
                    vec3(0.0, 0.0, 1.0),
                ));
            } else {
                sort_views.push(Mat4::look_at_rh(
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::from_vec(*sort_dir),
                    vec3(0.0, 1.0, 0.0),
                ));
            }
        }

        // Base data
        // Raw depth load first (need reference later)
        for i in 0..self.n_tiles.0 {
            let mut base_data_tile_vec: Vec<Vec<TileBaseData>> = Vec::new();
            for j in 0..self.n_tiles.1 {
                let mut base_data_view_vec: Vec<TileBaseData> = Vec::new();
                for (k, view) in sort_views.iter().enumerate() {
                    let view_proj = sort_projection * view;
                    let view_proj_slice = &[
                        view_proj[0][0],
                        view_proj[0][1],
                        view_proj[0][2],
                        view_proj[0][3],
                        view_proj[1][0],
                        view_proj[1][1],
                        view_proj[1][2],
                        view_proj[1][3],
                        view_proj[2][0],
                        view_proj[2][1],
                        view_proj[2][2],
                        view_proj[2][3],
                        view_proj[3][0],
                        view_proj[3][1],
                        view_proj[3][2],
                        view_proj[3][3],
                    ];

                    let scene = &self.tile_splats_vec[i][j];
                    let (_, raw_depth) = scene.sort_self(view_proj_slice);

                    let t_data = TileBaseData {
                        splat_count: 0,
                        tile_center: avg_center_vec[j],
                        aabb: aabb_vec[j],
                        raw_depth,
                        gs_index: Vec::new(),
                        gs_lod_id: Vec::new(),
                    };
                    base_data_view_vec.push(t_data);
                }
                base_data_tile_vec.push(base_data_view_vec);
            }
            self.tile_base_data.push(base_data_tile_vec);
        }
        // Other info later
        for i in 0..self.n_tiles.0 {
            for j in 0..self.n_tiles.1 {
                for (k, view) in sort_views.iter().enumerate() {
                    let mut tile_raw_depth: Vec<&Vec<i32>> = Vec::new();
                    let mut tile_lod_id: Vec<u32> = Vec::new();
                    let mut tile_merge_offset: Vec<u32> = Vec::new();

                    tile_raw_depth.push(&self.tile_base_data[i][j][k].raw_depth);
                    tile_lod_id.push(i as u32);
                    tile_merge_offset.push(self.splats_merge_offset[i][j]);

                    if i < self.n_tiles.0 - 1 {
                        // Transition to lower lod
                        tile_raw_depth.push(&self.tile_base_data[i + 1][j][k].raw_depth);
                        tile_lod_id.push(i as u32 + 1);
                        tile_merge_offset.push(self.splats_merge_offset[i + 1][j]);
                    }

                    let sort_result: Vec<(usize, usize)> =
                        Scene::sort_raw_depth_vec(tile_raw_depth);
                    let splat_count = sort_result.len();
                    let mut gs_index: Vec<u32> = Vec::with_capacity(splat_count);
                    let mut gs_lod_id: Vec<u32> = Vec::with_capacity(splat_count);
                    for (tile_idx, gs_idx) in sort_result {
                        gs_index.push(gs_idx as u32 + tile_merge_offset[tile_idx]);
                        gs_lod_id.push(tile_lod_id[tile_idx]);
                    }

                    self.tile_base_data[i][j][k].splat_count = splat_count;
                    self.tile_base_data[i][j][k].gs_index = gs_index;
                    self.tile_base_data[i][j][k].gs_lod_id = gs_lod_id;
                }
            }
        }
    }

    fn compute_map_neighbors(&self, map_coord: Vector2<usize>) -> MapNeighbor {
        let mut neighbor = MapNeighbor::new();
        match self.user_data.surface_type {
            SurfaceType::Sphere => {
                let map_w = self.user_data.tile_map_wh.x;
                let map_h = self.user_data.tile_map_wh.y;
                let block_w = self.user_data.tile_map_wh.x / 5;
                let block_id_x = 5 * map_coord.x / map_w;
                let block_id_y = 2 * map_coord.y / map_h;
                let block_x = map_coord.x - block_id_x * block_w;
                let block_y = map_coord.y - block_id_y * block_w;
                // West
                if block_x > 0 {
                    neighbor.west = Some((vec2(map_coord.x - 1, map_coord.y), 2));
                } else {
                    if block_id_y == 0 {
                        neighbor.west = Some((
                            vec2((map_w + map_coord.x - 1) % map_w, map_coord.y + block_w),
                            2,
                        ));
                    } else {
                        neighbor.west = Some((
                            vec2((map_w + map_coord.x - block_y - 1) % map_w, map_h - 1),
                            1,
                        ));
                    }
                }
                // East
                if block_x < block_w - 1 {
                    neighbor.east = Some((vec2(map_coord.x + 1, map_coord.y), 0));
                } else {
                    if block_id_y == 0 {
                        neighbor.east =
                            Some((vec2((map_coord.x + block_w - block_y) % map_w, 0), 3));
                    } else {
                        neighbor.east =
                            Some((vec2((map_coord.x + 1) % map_w, map_coord.y - block_w), 0));
                    }
                }
                // South
                if map_coord.y > 0 {
                    neighbor.south = Some((vec2(map_coord.x, map_coord.y - 1), 1));
                } else {
                    neighbor.south = Some((
                        vec2(
                            (map_w + block_id_x * block_w - 1) % map_w,
                            block_w - 1 - block_x,
                        ),
                        2,
                    ));
                }
                // North
                if map_coord.y < map_h - 1 {
                    neighbor.north = Some((vec2(map_coord.x, map_coord.y + 1), 3));
                } else {
                    neighbor.north = Some((
                        vec2(
                            (block_id_x * block_w + block_w) % map_w,
                            2 * block_w - 1 - block_x,
                        ),
                        0,
                    ));
                }
            }
            _ => {
                if map_coord.x > 0 {
                    neighbor.west = Some((vec2(map_coord.x - 1, map_coord.y), 2));
                }
                if map_coord.x < self.user_data.tile_map_wh.x - 1 {
                    neighbor.east = Some((vec2(map_coord.x + 1, map_coord.y), 0));
                }
                if map_coord.y > 0 {
                    neighbor.south = Some((vec2(map_coord.x, map_coord.y - 1), 1));
                }
                if map_coord.y < self.user_data.tile_map_wh.y - 1 {
                    neighbor.north = Some((vec2(map_coord.x, map_coord.y + 1), 3));
                }
            }
        }

        neighbor
    }

    pub fn preload(&mut self) -> PreloadData {
        PreloadData {
            tile_splats_merged: &mut self.tile_splats_merged,
            tile_base_data: &mut self.tile_base_data,
            // tile_spawning_data: &mut self.tile_spawning_data,
            // tile_changing_data: &mut self.tile_changing_data,
        }
    }

    pub fn configure(&mut self, user_data: UserData) -> UserData {
        self.initialized = false;
        self.user_data = user_data;
        if self.user_data.reset_rng {
            self.rng = StdRng::seed_from_u64(0);
        }

        if self.user_data.surface_type == SurfaceType::Sphere {
            self.user_data.tile_map_wh = self.user_data.tile_map_half_wh * 2;
            assert!(self.user_data.tile_map_wh.x * 2 == self.user_data.tile_map_wh.y * 5);
        } else {
            self.user_data.tile_map_wh = self.user_data.tile_map_half_wh * 2 + vec2(1, 1);
        }

        let map_w = self.user_data.tile_map_wh.x;
        let map_h = self.user_data.tile_map_wh.y;
        self.tile_map = Array2::from_elem((map_w, map_h), None);
        assert!(self.n_tiles.1 / 16 >= self.user_data.center_option);

        // Neighbor map
        self.neighbor_map = Array2::from_elem((map_w, map_h), MapNeighbor::new());
        for i in 0..map_w {
            for j in 0..map_h {
                self.neighbor_map[[i, j]] = self.compute_map_neighbors(vec2(i, j));
            }
        }

        // Height map
        const MAP_RESO: usize = 1024; // Internal map resolution
        let h_map_width = self.user_data.height_map_wh.x;
        let h_map_height = self.user_data.height_map_wh.y;
        let mut height_map = Vec::new();
        for i in 0..h_map_height {
            for j in 0..h_map_width {
                let h: f32 = match self.user_data.height_map_type {
                    HeightMapType::Texture => 0.0, // Placeholder, handled below
                    HeightMapType::Random => self.rng.random_range(-1.0..=1.0),
                    HeightMapType::SlopeX => j as f32 / h_map_height as f32 * 2.0 - 1.0,
                    HeightMapType::SlopeY => i as f32 / h_map_height as f32 * 2.0 - 1.0,
                    HeightMapType::DualSlope => {
                        i as f32 / h_map_width as f32 + j as f32 / h_map_height as f32 - 1.0
                    }
                };
                height_map.push(h);
            }
        }
        if self.user_data.height_map_type == HeightMapType::Texture
            && self.user_data.height_tex.is_some()
        {
            self.user_data.height_map_wh = self.user_data.height_tex.as_ref().unwrap().1;
            height_map = self.user_data.height_tex.as_ref().unwrap().0.clone();
        }
        height_map
            .iter_mut()
            .for_each(|v| *v *= self.user_data.tile_width * self.user_data.height_map_scale.z);
        // Map resize
        if self.user_data.height_map_type == HeightMapType::Random {
            height_map = self.map_resize(
                height_map.as_slice(),
                self.user_data.height_map_wh,
                vec2(MAP_RESO, MAP_RESO),
            );
            self.user_data.height_map_wh = vec2(MAP_RESO, MAP_RESO);
        }
        self.user_data.height_map = height_map;

        // Lod transition dist
        self.user_data.lod_transition_dist.clear();
        let s_n = self.lod_avg_scale.last().unwrap();
        for (lod_lv, &lod_scale) in self.lod_avg_scale.iter().enumerate() {
            self.user_data
                .lod_transition_dist
                .push(self.user_data.lod_max_dist * lod_scale / s_n);
        }
        log! {"LOD trans dist: {:?}", self.user_data.lod_transition_dist};

        // Cache reset
        self.sort_lru_cache =
            LruCache::new(std::num::NonZeroUsize::new(self.user_data.cache_size).unwrap());
        // self.sort_lru_cache = LRUCache::new(self.user_data.cache_size).unwrap();

        self.user_data.n_tiles = self.n_tiles.clone();
        self.user_data.clone()
    }

    pub fn build_tiles(&mut self, camera_pos: Vec3) -> SceneData {
        // log!{"Wang thread: build_tiles() start"};

        if !self.initialized {
            self.initialized = true;
        }

        self.update_tile_map(camera_pos);

        let mut scene_data = SceneData::new();
        scene_data.center_coord = self.center_coord;
        scene_data.lod_splat_count = vec![0; self.n_tiles.0];
        scene_data.lod_instance_count = vec![0; self.n_tiles.0];
        for i in 0..self.user_data.tile_map_wh.x {
            for j in 0..self.user_data.tile_map_wh.y {
                let tile_instance = self.tile_map[[i, j]].as_ref().unwrap();
                let tid = tile_instance.tid;
                let tile_base = &self.tile_base_data[tid.0][tid.1][0];
                scene_data.splat_count += tile_base.splat_count;
                scene_data.blending_splat_count += tile_base.splat_count;
                scene_data.lod_splat_count[tid.0] += tile_base.splat_count;
                scene_data.lod_instance_count[tid.0] += 1;
                let mut blend_lower = tid.0 < self.n_tiles.0 - 1;
                if let TileTransitionStatus::Changing(to_lower) = tile_instance.transition_status {
                    if !to_lower {
                        let higher_tile = &self.tile_base_data[tid.0 - 1][tid.1][0];
                        scene_data.blending_splat_count += higher_tile.splat_count;
                        blend_lower = false;
                    }
                }
                if blend_lower {
                    // Lower lod is always included if possible
                    let lower_tile = &self.tile_base_data[tid.0 + 1][tid.1][0];
                    scene_data.blending_splat_count += lower_tile.splat_count;
                }
            }
        }

        // log!{"Wang thread: build_tiles() finish"};
        scene_data
    }

    pub fn sort_tiles(&mut self, camera_pos: Vec3, view_proj: Mat4) -> SortData {
        // log!{"Wang thread: sort_tiles() start"};

        match self.user_data.merge_type {
            SelectiveMergeType::Axis => {
                self.selective_merge_axis(camera_pos, view_proj);
            }
            SelectiveMergeType::Edge => {
                self.selective_merge_edge(camera_pos, view_proj);
            }
            SelectiveMergeType::None => {}
        }

        let tile_object_sorted = match self.user_data.tile_sort_type {
            TileSortType::Distance => self.sort_tiles_object_pos(camera_pos),
            TileSortType::Viewport => self.sort_tiles_object_vp(view_proj),
            TileSortType::Object => self.sort_tiles_object_bfs(camera_pos),
            TileSortType::Graph => self.sort_tiles_object_graph(camera_pos),
        };

        let mut render_data_vec: Vec<(RenderDataKey, Option<RenderDataValue>)> =
            Vec::with_capacity(tile_object_sorted.len());
        let mut tile_instance_vec: Vec<TileInstance> = Vec::with_capacity(tile_object_sorted.len());
        for mi in tile_object_sorted {
            let map_coord = self.index_to_map(mi);
            // log!{"Process {}, {:?} start", i, map_coord};
            let tile_instance = self.tile_map[[map_coord.x, map_coord.y]].as_ref().unwrap();

            // Build cache key
            let view_id: usize;
            let cache_key: RenderDataKey;
            if let TileMergeStatus::MergedFrom(from_vec) = &tile_instance.merge_status {
                let merge_len = from_vec.len();
                let mut merge_x = true;
                let mut merge_y = true;
                let mut avg_center = Vec3::zero();
                let mut avg_quat: Quaternion<f32> = Quaternion::zero();
                let mut tid: Vec<(usize, usize)> = Vec::with_capacity(merge_len);
                let mut transition_status: Vec<TileTransitionStatusHash> =
                    Vec::with_capacity(merge_len);
                for &m_mi in from_vec.iter() {
                    let m_mc = self.index_to_map(m_mi);
                    if m_mc.x != map_coord.x {
                        merge_x = false;
                    }
                    if m_mc.y != map_coord.y {
                        merge_y = false;
                    }

                    let m_tile_instance = self.tile_map[[m_mc.x, m_mc.y]].as_ref().unwrap();
                    tid.push(m_tile_instance.tid.clone());
                    transition_status.push(TileTransitionStatusHash::from_status(
                        &m_tile_instance.transition_status,
                    ));
                    avg_center += m_tile_instance.tile_center;
                    avg_quat += Quaternion::from(m_tile_instance.to_local);
                }
                // Force top-down view if not merging a line
                if !merge_x && !merge_y {
                    view_id = self.presort_dirs.len() - 1;
                } else {
                    let from_len = from_vec.len() as f32;
                    avg_center /= from_len;
                    avg_quat /= from_len;
                    view_id =
                        self.choose_presort_view(Mat3::from(avg_quat), avg_center, camera_pos);
                }

                cache_key = RenderDataKey {
                    view_id,
                    tid,
                    transition_status,
                };
            } else {
                view_id = self.choose_presort_view(
                    tile_instance.to_local,
                    tile_instance.tile_center,
                    camera_pos,
                );
                cache_key = RenderDataKey {
                    view_id,
                    tid: vec![tile_instance.tid.clone()],
                    transition_status: vec![TileTransitionStatusHash::from_status(
                        &tile_instance.transition_status,
                    )],
                };
            }

            // Push metadata
            // let mut tile_instance_meta = TileInstance::from_metadata(tile_instance);
            // tile_instance_meta.view_id = view_id;
            // tile_instance_vec.push(tile_instance_meta);
            let mut new_tile_instance = tile_instance.clone();
            new_tile_instance.view_id = view_id;
            tile_instance_vec.push(new_tile_instance);

            let mut cache_value: Option<RenderDataValue> = None;
            if let TileMergeStatus::MergedFrom(from_vec) = &tile_instance.merge_status {
                // Use cache if exist
                if self.user_data.use_cache {
                    if let Some(cache_value) = self.sort_lru_cache.get(&cache_key) {
                        let mut new_cache_value = cache_value.clone();
                        // Update map index
                        let mut gs_map_id = new_cache_value.gs_map_id;
                        let merge_from_vec: &Vec<usize> = new_cache_value.merge_from_vec.as_ref();
                        for i in 0..new_cache_value.splat_count {
                            for j in 0..merge_from_vec.len() {
                                if gs_map_id[i] == merge_from_vec[j] as u32 {
                                    gs_map_id[i] = from_vec[j] as u32;
                                    break;
                                }
                            }
                        }
                        new_cache_value.gs_map_id = gs_map_id;
                        render_data_vec.push((cache_key, Some(new_cache_value)));
                        continue;
                    }
                }

                // Quick check for meta data
                let merge_len = from_vec.len();
                let mut m_tile_instance_vec: Vec<&TileInstance> =
                    Vec::with_capacity(from_vec.len());
                let mut do_transition = false;
                for &m_mi in from_vec {
                    let m_mc = self.index_to_map(m_mi);
                    let tile_instance = self.tile_map[[m_mc.x, m_mc.y]].as_ref().unwrap();
                    if !do_transition {
                        if tile_instance.transition_status != TileTransitionStatus::None {
                            do_transition = true;
                        }
                    }
                    m_tile_instance_vec.push(tile_instance);
                }

                let mut tile_raw_depth: Vec<&Vec<i32>> = Vec::new();
                let mut tile_lod_id: Vec<u32> = Vec::new();
                let mut tile_map_index: Vec<u32> = Vec::new();
                let mut tile_merge_offset: Vec<u32> = Vec::new();
                for i in 0..merge_len {
                    let m_mi = from_vec[i];
                    let m_tile_instance = m_tile_instance_vec[i];
                    let m_tid = m_tile_instance.tid;
                    let base_data = &self.tile_base_data[m_tid.0][m_tid.1][view_id];

                    tile_raw_depth.push(&base_data.raw_depth);
                    if do_transition {
                        tile_lod_id.push(m_tid.0 as u32);
                    }
                    tile_map_index.push(m_mi as u32);
                    tile_merge_offset.push(self.splats_merge_offset[m_tid.0][m_tid.1]);

                    // LOD
                    if let TileTransitionStatus::Changing(to_lower) =
                        m_tile_instance.transition_status
                    {
                        let other_lod = if to_lower { m_tid.0 + 1 } else { m_tid.0 - 1 };
                        let other_base_data = &self.tile_base_data[other_lod][m_tid.1][view_id];
                        tile_raw_depth.push(&other_base_data.raw_depth);
                        if do_transition {
                            tile_lod_id.push(other_lod as u32);
                        }
                        tile_map_index.push(m_mi as u32);
                        tile_merge_offset
                            .push(self.splats_merge_offset[other_lod][m_tid.1]);
                    }
                }
                let sort_result = Scene::sort_raw_depth_vec(tile_raw_depth);
                let splat_count = sort_result.len();
                let mut gs_index: Vec<u32> = Vec::with_capacity(splat_count);
                let mut gs_lod_id: Vec<u32> = if do_transition {
                    Vec::with_capacity(splat_count)
                } else {
                    Vec::new()
                };
                let mut gs_map_id: Vec<u32> = Vec::with_capacity(splat_count);
                for (tile_idx, gs_idx) in sort_result {
                    gs_index.push(gs_idx as u32 + tile_merge_offset[tile_idx]);
                    if do_transition {
                        gs_lod_id.push(tile_lod_id[tile_idx]);
                    }
                    gs_map_id.push(tile_map_index[tile_idx]);
                }
                cache_value = Some(RenderDataValue {
                    splat_count,
                    gs_index,
                    gs_map_id,
                    merge_from_vec: from_vec.clone(),
                    single_lod_id: if do_transition {
                        -1
                    } else {
                        // Assumption: all tiles must have same lod if no tile has transition
                        tile_instance.tid.0 as i32
                    },
                    gs_lod_id: if do_transition { Some(gs_lod_id) } else { None },
                });

                if self.user_data.use_cache {
                    self.sort_lru_cache
                        .put(cache_key.clone(), cache_value.clone().unwrap());
                }
            }

            render_data_vec.push((cache_key, cache_value));
            // log!{"Process {}, {:?} finish", i, map_coord};
        }

        let sort_data = SortData {
            scene_id: 0,
            tile_instance_vec,
            render_data_vec,
        };

        // log!{"Wang thread: sort_tiles() finish"};
        sort_data
    }

    pub fn check_update(&self, camera_pos: &Vec3) -> bool {
        if !self.initialized {
            return true;
        }

        let dist = camera_pos.distance2(self.camera_pos);
        dist >= self.user_data.update_distance2
    }

    fn choose_presort_view(&self, transform: Mat3, pos: Vec3, cam_pos: Vec3) -> usize {
        let dir_global = (pos - cam_pos).normalize();
        let dir_local = transform * dir_global;

        let mut best_view: usize = 0;
        let mut best_err: f32 = 1000.0;
        for (i, presort_dir) in self.presort_dirs.iter().enumerate() {
            let err = (dir_local.x - presort_dir.x).powi(2)
                + (dir_local.y - presort_dir.y).powi(2)
                + (dir_local.z - presort_dir.z).powi(2);
            if err < best_err {
                best_view = i;
                best_err = err;
            }
        }

        best_view
    }

    // For plane / height map only, sphere has bug (after switching block, merge direction is wrong)
    // TODO: sort by map index
    fn selective_merge_axis(&mut self, camera_pos: Vec3, view_proj: Mat4) {
        // Find map center
        let mut sort_center_mc: Vector2<usize> = vec2(0, 0);
        if self.user_data.surface_type == SurfaceType::Sphere {
            let mut min_dist = -1.0;
            let n_instance = self.user_data.tile_map_wh.x * self.user_data.tile_map_wh.y;
            for index in 0..n_instance {
                let map_coord = self.index_to_map(index);
                let tile_instance = self.tile_map[[map_coord.x, map_coord.y]].as_ref().unwrap();
                if let TileMergeStatus::MergedTo(_) = tile_instance.merge_status {
                    continue;
                }
                let dist = camera_pos.distance2(tile_instance.tile_center);
                if (min_dist < 0.0) || (dist < min_dist) {
                    min_dist = dist;
                    sort_center_mc = map_coord;
                }
            }
        } else {
            sort_center_mc = self.coord_to_map(self.center_coord);
        }

        // Find merge direction
        let neighbors = &self.neighbor_map[[sort_center_mc.x, sort_center_mc.y]];
        let mut best_dir_proj: f32 = 0.0;
        let mut merge_dir: i32 = -1;
        let mut dir_proj_vec: [f32; 4] = [0.0; 4];
        let cam_dir = vec3(view_proj[0][2], view_proj[1][2], view_proj[2][2]).normalize();
        for check_i in 0..4 {
            if let Some((map_coord, _)) = neighbors[check_i] {
                let tile_pos = self.tile_map[[map_coord.x, map_coord.y]]
                    .as_ref()
                    .unwrap()
                    .tile_center;
                let dir_proj = (tile_pos - camera_pos).normalize().dot(cam_dir);
                dir_proj_vec[check_i] = dir_proj;
                if best_dir_proj < dir_proj {
                    best_dir_proj = dir_proj;
                    merge_dir = check_i as i32;
                }
            }
        }
        if merge_dir < 0 {
            return;
        }
        let merge_dir = merge_dir as usize;
        // log!("Merge tile: dir_proj_vec: {:?}", dir_proj_vec);

        // Update merge status
        let mut merge_center_mc = sort_center_mc;
        let merge_neighbors: Vec<(usize, usize)> = vec![(3, 1), (0, 2), (1, 3), (2, 0)];
        for i in 0..self.user_data.merge_tile_dist.0 {
            merge_center_mc = self.neighbor_map[[merge_center_mc.x, merge_center_mc.y]][merge_dir]
                .unwrap()
                .0;
        }
        for i in self.user_data.merge_tile_dist.0..self.user_data.merge_tile_dist.1 {
            let merge_center_index = self.map_to_index(merge_center_mc);
            let neighbors = self.neighbor_map[[merge_center_mc.x, merge_center_mc.y]].clone();
            let merge_vec = vec![
                self.map_to_index(neighbors[merge_neighbors[merge_dir].0].unwrap().0),
                merge_center_index,
                self.map_to_index(neighbors[merge_neighbors[merge_dir].1].unwrap().0),
            ];
            let neighbor1_mc = neighbors[merge_neighbors[merge_dir].0].unwrap().0;
            let neighbor2_mc = neighbors[merge_neighbors[merge_dir].1].unwrap().0;
            if self.tile_map[[merge_center_mc.x, merge_center_mc.y]]
                .as_ref()
                .unwrap()
                .merge_status
                != TileMergeStatus::None
                || self.tile_map[[neighbor1_mc.x, neighbor1_mc.y]]
                    .as_ref()
                    .unwrap()
                    .merge_status
                    != TileMergeStatus::None
                || self.tile_map[[neighbor2_mc.x, neighbor2_mc.y]]
                    .as_ref()
                    .unwrap()
                    .merge_status
                    != TileMergeStatus::None
            {
                log!(
                    "Encountered already merged tiles at distance {}. Merge max distance too far.",
                    i
                );
                break;
            }
            self.tile_map[[merge_center_mc.x, merge_center_mc.y]]
                .as_mut()
                .unwrap()
                .merge_status = TileMergeStatus::MergedFrom(merge_vec);
            self.tile_map[[neighbor1_mc.x, neighbor1_mc.y]]
                .as_mut()
                .unwrap()
                .merge_status = TileMergeStatus::MergedTo(merge_center_index);
            self.tile_map[[neighbor2_mc.x, neighbor2_mc.y]]
                .as_mut()
                .unwrap()
                .merge_status = TileMergeStatus::MergedTo(merge_center_index);

            merge_center_mc = neighbors[merge_dir].unwrap().0;
        }
    }

    fn selective_merge_edge(&mut self, camera_pos: Vec3, view_proj: Mat4) {
        let xmax = self.user_data.tile_map_wh.x;
        let ymax = self.user_data.tile_map_wh.y;

        // Init Edge info
        let mut edge_index_vec: Vec<(usize, usize, f32, f32)> = Vec::with_capacity(2 * xmax * ymax); // (map_index, edge_index, dot_result_abs, normalized_dot)
        let mut check_map: Array2<bool> = Array2::from_elem((xmax, ymax), false);
        for i in 0..xmax {
            for j in 0..ymax {
                let map_coord = vec2(i, j);
                let map_index = self.map_to_index(map_coord);
                check_map[[map_coord.x, map_coord.y]] = true;
                self.tile_map[[map_coord.x, map_coord.y]]
                    .as_mut()
                    .unwrap()
                    .merge_status = TileMergeStatus::None;
                for n_i in 0..4 {
                    if let Some((neighbor, _)) = self.neighbor_map[[map_coord.x, map_coord.y]][n_i]
                    {
                        if check_map[[neighbor.x, neighbor.y]] {
                            continue;
                        }
                        let (edge_pos, edge_normal) = self.tile_map[[map_coord.x, map_coord.y]]
                            .as_ref()
                            .unwrap()
                            .edge_data
                            .as_ref()
                            .unwrap()[n_i];
                        let (corner_pos_1, corner_to_world_1) = self.tile_map
                            [[map_coord.x, map_coord.y]]
                        .as_ref()
                        .unwrap()
                        .corner_data
                        .as_ref()
                        .unwrap()[n_i];
                        let (corner_pos_2, corner_to_world_2) = self.tile_map
                            [[map_coord.x, map_coord.y]]
                        .as_ref()
                        .unwrap()
                        .corner_data
                        .as_ref()
                        .unwrap()[(n_i + 1) % 4];
                        let view_dir = edge_pos - camera_pos;
                        let view_dir_length = view_dir.magnitude();

                        // Discard out of view edge
                        if view_dir == Vec3::zero() {
                            continue;
                        }
                        if view_dir.dot(corner_to_world_1.z) > 0.0
                            || view_dir.dot(corner_to_world_2.z) > 0.0
                        {
                            continue;
                        }
                        let pos2d_1 = view_proj * corner_pos_1.extend(1.0);
                        let pos2d_1 = pos2d_1.truncate() / pos2d_1.w;
                        let pos2d_2 = view_proj * corner_pos_2.extend(1.0);
                        let pos2d_2 = pos2d_2.truncate() / pos2d_2.w;
                        let clip: f32 = 1.0;
                        if (pos2d_1.z < -clip
                            || pos2d_1.x < -clip
                            || pos2d_1.x > clip
                            || pos2d_1.y < -clip
                            || pos2d_1.y > clip)
                            && (pos2d_2.z < -clip
                                || pos2d_2.x < -clip
                                || pos2d_2.x > clip
                                || pos2d_2.y < -clip
                                || pos2d_2.y > clip)
                        {
                            continue;
                        }

                        let dot_result_abs = edge_normal.dot(view_dir).abs();
                        let normalized_dot = dot_result_abs / view_dir_length;
                        edge_index_vec.push((map_index, n_i, dot_result_abs, normalized_dot));
                    }
                }
            }
        }

        edge_index_vec.sort_by(|(_, _, a, _), (_, _, b, _)| a.partial_cmp(b).unwrap());

        // Build groups
        let mut topk = 0;
        let mut merge_map: Array2<Option<usize>> = Array2::from_elem((xmax, ymax), None);
        let mut merge_groups: Vec<Vec<usize>> = Vec::new();
        for (map_index, edge_index, _, normalized_dot) in edge_index_vec {
            if topk >= self.user_data.merge_topk {
                break;
            }
            if normalized_dot > self.user_data.merge_dot_threshold {
                continue;
            } // Prevent close-to-camera edge sneaking in
            let map_coord = self.index_to_map(map_index);
            let (neighbor, _) = self.neighbor_map[[map_coord.x, map_coord.y]][edge_index].unwrap();
            let neighbor_index = self.map_to_index(neighbor);

            match (
                merge_map[[map_coord.x, map_coord.y]],
                merge_map[[neighbor.x, neighbor.y]],
            ) {
                (None, None) => {
                    let new_group = vec![map_index, neighbor_index];
                    let group_index = merge_groups.len();
                    merge_groups.push(new_group);
                    merge_map[[map_coord.x, map_coord.y]] = Some(group_index);
                    merge_map[[neighbor.x, neighbor.y]] = Some(group_index);
                }
                (Some(gi), None) => {
                    merge_groups[gi].push(neighbor_index);
                    merge_map[[neighbor.x, neighbor.y]] = Some(gi);
                }
                (None, Some(gi)) => {
                    merge_groups[gi].push(map_index);
                    merge_map[[map_coord.x, map_coord.y]] = Some(gi);
                }
                (Some(this_gi), Some(neighbor_gi)) => {
                    if this_gi != neighbor_gi {
                        for &g_mi in merge_groups[neighbor_gi].as_slice() {
                            let g_mc = self.index_to_map(g_mi);
                            merge_map[[g_mc.x, g_mc.y]] = Some(this_gi);
                        }
                        let temp = std::mem::take(&mut merge_groups[neighbor_gi]);
                        merge_groups[this_gi].extend(temp);
                    }
                }
            }
            topk += 1;
        }
        // log!("{:?}", merge_groups.clone());

        // Fix non-convex groups
        for i in 0..merge_groups.len() {
            let mut neighbors_hash: HashSet<usize> = HashSet::new();
            let mut j = 0;
            while j < merge_groups[i].len() {
                let t_mc = self.index_to_map(merge_groups[i][j]);
                let t_neighbors = &self.neighbor_map[[t_mc.x, t_mc.y]];
                for neighbor in t_neighbors.iter() {
                    if let &Some((neighbor_mc, _)) = neighbor {
                        let neighbor_index = self.map_to_index(neighbor_mc);
                        if !merge_groups[i].contains(&neighbor_index) {
                            let insert_result = neighbors_hash.insert(neighbor_index);
                            if !insert_result {
                                if let Some(other_i) = merge_map[[neighbor_mc.x, neighbor_mc.y]] {
                                    // Merge group if neighbor already belongs to another group
                                    for &g_mi in merge_groups[other_i].as_slice() {
                                        let g_mc = self.index_to_map(g_mi);
                                        merge_map[[g_mc.x, g_mc.y]] = Some(i);
                                    }
                                    let temp = std::mem::take(&mut merge_groups[other_i]);
                                    merge_groups[i].extend(temp);
                                } else {
                                    merge_groups[i].push(neighbor_index);
                                    merge_map[[neighbor_mc.x, neighbor_mc.y]] = Some(i);
                                }
                            }
                        }
                    }
                }
                j += 1;
            }
        }

        for mut group in merge_groups {
            if !group.is_empty() {
                group.sort(); // sort by map index

                let mut min_dist2 = f32::MAX;
                let mut min_index: usize = 0;
                for i in 0..group.len() {
                    let mc = self.index_to_map(group[i]);
                    let dist2 = self.tile_map[[mc.x, mc.y]]
                        .as_mut()
                        .unwrap()
                        .tile_center
                        .distance2(camera_pos);
                    if min_dist2 > dist2 {
                        min_dist2 = dist2;
                        min_index = i;
                    }
                }

                for i in 0..group.len() {
                    if i != min_index {
                        let map_coord = self.index_to_map(group[i]);
                        self.tile_map[[map_coord.x, map_coord.y]]
                            .as_mut()
                            .unwrap()
                            .merge_status = TileMergeStatus::MergedTo(group[min_index]);
                    }
                }
                let map_coord = self.index_to_map(group[min_index]);
                self.tile_map[[map_coord.x, map_coord.y]]
                    .as_mut()
                    .unwrap()
                    .merge_status = TileMergeStatus::MergedFrom(group);
            }
        }
    }

    fn sort_tiles_object_pos(&self, camera_pos: Vec3) -> Vec<usize> {
        let mut sort_vec: Vec<(usize, f32)> = Vec::new();
        let n_instance = self.user_data.tile_map_wh.x * self.user_data.tile_map_wh.y;
        for index in 0..n_instance {
            let map_coord = self.index_to_map(index);
            let tile_instance = self.tile_map[[map_coord.x, map_coord.y]].as_ref().unwrap();
            if let TileMergeStatus::MergedTo(_) = tile_instance.merge_status {
                continue;
            }
            let dist = camera_pos.distance2(tile_instance.tile_center);
            sort_vec.push((index, dist));
        }

        sort_vec.sort_by(|&i, &j| i.1.partial_cmp(&j.1).unwrap());
        sort_vec.reverse();

        let index_vec: Vec<usize> = sort_vec.iter().map(|&i| i.0).collect();
        index_vec
    }

    fn sort_tiles_object_vp(&self, view_proj: Mat4) -> Vec<usize> {
        let mut sort_vec: Vec<(usize, f32)> = Vec::new();
        let n_instance = self.user_data.tile_map_wh.x * self.user_data.tile_map_wh.y;
        for index in 0..n_instance {
            let map_coord = self.index_to_map(index);
            let tile_instance = self.tile_map[[map_coord.x, map_coord.y]].as_ref().unwrap();
            if let TileMergeStatus::MergedTo(_) = tile_instance.merge_status {
                continue;
            }
            let tile_pos = tile_instance.tile_center;
            let dist = view_proj[0][2] * tile_pos.x
                + view_proj[1][2] * tile_pos.y
                + view_proj[2][2] * tile_pos.z;
            sort_vec.push((index, dist));
        }

        sort_vec.sort_by(|&i, &j| i.1.partial_cmp(&j.1).unwrap());
        sort_vec.reverse();

        let index_vec: Vec<usize> = sort_vec.iter().map(|&i| i.0).collect();
        index_vec
    }

    fn sort_tiles_object_bfs(&self, camera_pos: Vec3) -> Vec<usize> {
        let mut min_dist_mc: Vector2<usize> = vec2(0, 0);
        let mut min_dist = -1.0;
        let n_instance = self.user_data.tile_map_wh.x * self.user_data.tile_map_wh.y;
        for index in 0..n_instance {
            let map_coord = self.index_to_map(index);
            let tile_instance = self.tile_map[[map_coord.x, map_coord.y]].as_ref().unwrap();
            if let TileMergeStatus::MergedTo(_) = tile_instance.merge_status {
                continue;
            }
            let dist = camera_pos.distance2(tile_instance.tile_center);
            if (min_dist < 0.0) || (dist < min_dist) {
                min_dist = dist;
                min_dist_mc = map_coord;
            }
        }

        // log!("Sort: min dist at {:?}", self.map_to_coord(min_dist_mc));
        let xmax = self.user_data.tile_map_wh.x;
        let ymax = self.user_data.tile_map_wh.y;
        let mut tile_trans_vec: Vec<usize> = Vec::with_capacity(n_instance); // (cur/prev tile, tile index)
        let mut check_map: Array2<bool> = Array2::from_elem((xmax, ymax), false);
        let mut queue = std::collections::VecDeque::<Vector2<usize>>::new();
        queue.push_back(min_dist_mc);
        check_map[[min_dist_mc.x, min_dist_mc.y]] = true;
        while !queue.is_empty() {
            let map_coord = queue.pop_front().unwrap();
            let map_index = self.map_to_index(map_coord);
            tile_trans_vec.push(map_index);
            for n_i in 0..4 {
                if let Some((neighbor, _)) = self.neighbor_map[[map_coord.x, map_coord.y]][n_i] {
                    if !check_map[[neighbor.x, neighbor.y]] {
                        queue.push_back(neighbor);
                        check_map[[neighbor.x, neighbor.y]] = true;
                    }
                }
            }
        }

        tile_trans_vec.reverse();
        tile_trans_vec
    }

    fn sort_tiles_object_graph(&mut self, camera_pos: Vec3) -> Vec<usize> {
        let xmax = self.user_data.tile_map_wh.x;
        let ymax = self.user_data.tile_map_wh.y;

        let mut graph = DiGraph::<usize, ()>::with_capacity(xmax * ymax, 2 * xmax * ymax);
        let node_map: Array2<Option<NodeIndex>> = Array2::from_shape_fn((xmax, ymax), |(i, j)| {
            let tile_instance = self.tile_map[[i, j]].as_ref().unwrap();
            if let TileMergeStatus::MergedTo(to_index) = tile_instance.merge_status {
                None
            } else {
                Some(graph.add_node(self.map_to_index(vec2(i, j))))
            }
        });
        let mut check_map: Array2<bool> = Array2::from_elem((xmax, ymax), false);

        // log!("Graph sort start");
        for i in 0..xmax {
            for j in 0..ymax {
                let map_coord = vec2(i, j);
                let this_instance = self.tile_map[[map_coord.x, map_coord.y]].as_ref().unwrap();
                let this_node =
                    if let TileMergeStatus::MergedTo(to_tile) = this_instance.merge_status {
                        let to_tile_mc = self.index_to_map(to_tile);
                        node_map[[to_tile_mc.x, to_tile_mc.y]].unwrap()
                    } else {
                        node_map[[map_coord.x, map_coord.y]].unwrap()
                    };
                check_map[[map_coord.x, map_coord.y]] = true;
                // log!("Check {:?}", map_coord);
                for n_i in 0..4 {
                    if let Some((neighbor, neighbor_edge_idx)) =
                        self.neighbor_map[[map_coord.x, map_coord.y]][n_i]
                    {
                        if check_map[[neighbor.x, neighbor.y]] {
                            continue;
                        }
                        let neighbor_instance =
                            self.tile_map[[neighbor.x, neighbor.y]].as_ref().unwrap();

                        let neighbor_node = if let TileMergeStatus::MergedTo(to_tile) =
                            neighbor_instance.merge_status
                        {
                            let to_tile_mc = self.index_to_map(to_tile);
                            node_map[[to_tile_mc.x, to_tile_mc.y]].unwrap()
                        } else {
                            node_map[[neighbor.x, neighbor.y]].unwrap()
                        };
                        if this_node == neighbor_node {
                            continue;
                        }

                        let (edge_pos, edge_normal) =
                            this_instance.edge_data.as_ref().unwrap()[n_i];
                        let view_dir = edge_pos - camera_pos;
                        if view_dir == Vec3::zero() {
                            continue;
                        }
                        let dot_result = edge_normal.dot(view_dir);
                        if dot_result > 0.0 {
                            // log!("Edge: {:?} to {:?}", map_coord, neighbor);
                            // log!("Node: {:?} to {:?}", this_node, neighbor_node);
                            graph.add_edge(this_node, neighbor_node, ());
                        } else if dot_result < 0.0 {
                            // log!("Edge: {:?} to {:?}", neighbor, map_coord);
                            // log!("Node: {:?} to {:?}", neighbor_node, this_node);
                            graph.add_edge(neighbor_node, this_node, ());
                        }
                    }
                }
            }
        }

        let mut tile_trans_vec: Vec<usize> = Vec::with_capacity(xmax * ymax); // (cur/prev tile, tile index)
        let mut removed_vec: Vec<usize> = Vec::new();
        loop {
            match toposort(&graph, None) {
                Ok(sorted) => {
                    for node in sorted {
                        if graph
                            .neighbors_directed(node, petgraph::Direction::Incoming)
                            .next()
                            .is_some()
                            || graph
                                .neighbors_directed(node, petgraph::Direction::Outgoing)
                                .next()
                                .is_some()
                        {
                            tile_trans_vec.push(graph[node]);
                        }
                    }
                    break;
                }
                Err(cycle) => {
                    // log!("Cycle detected at node: {:?}", cycle.node_id());
                    removed_vec.push(graph[cycle.node_id()]);
                    graph.remove_node(cycle.node_id());
                }
            }
        }

        tile_trans_vec.append(&mut removed_vec);
        tile_trans_vec.reverse();
        tile_trans_vec
    }

    fn map_fetch_bilinear_with_auxiliary(
        &self,
        map: &[f32],
        map_wh: Vector2<usize>,
        uv: Vec2,
        dt: f32,
    ) -> Vec<f32> {
        let width = map_wh.x;
        let height = map_wh.y;

        // Prevent negative x or y. In rust, -1 % 10 == -1
        let get_texel = |tex: &[f32], x: isize, y: isize| -> f32 {
            let xi = ((x % width as isize + width as isize) % width as isize) as usize;
            let yi = ((y % height as isize + height as isize) % height as isize) as usize;
            tex[yi * width + xi]
        };

        // Map uv to texel space
        let x = uv.x * width as f32 - 0.5;
        let y = uv.y * height as f32 - 0.5;
        let dx = dt * width as f32;
        let dy = dt * height as f32;

        let x0 = x.floor() as isize;
        let y0 = y.floor() as isize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let tx = x - (x0 as f32);
        let ty = y - (y0 as f32);

        // Fetch texel values
        let i00 = get_texel(map, x0, y0);
        let i10 = get_texel(map, x1, y0);
        let i01 = get_texel(map, x0, y1);
        let i11 = get_texel(map, x1, y1);

        // Bilinear interpolation
        let mut results: Vec<f32> = Vec::with_capacity(5);
        // Center
        let i0 = i00 * (1.0 - tx) + i10 * tx;
        let i1 = i01 * (1.0 - tx) + i11 * tx;
        let r = i0 * (1.0 - ty) + i1 * ty;
        results.push(r);
        // Right
        let n_tx = tx + dx;
        let i0 = i00 * (1.0 - n_tx) + i10 * n_tx;
        let i1 = i01 * (1.0 - n_tx) + i11 * n_tx;
        let r = i0 * (1.0 - ty) + i1 * ty;
        results.push(r);
        // Left
        let n_tx = tx - dx;
        let i0 = i00 * (1.0 - n_tx) + i10 * n_tx;
        let i1 = i01 * (1.0 - n_tx) + i11 * n_tx;
        let r = i0 * (1.0 - ty) + i1 * ty;
        results.push(r);
        // Up
        let n_ty = ty + dy;
        let i0 = i00 * (1.0 - tx) + i10 * tx;
        let i1 = i01 * (1.0 - tx) + i11 * tx;
        let r = i0 * (1.0 - n_ty) + i1 * n_ty;
        results.push(r);
        // Down
        let n_ty = ty - dy;
        let i0 = i00 * (1.0 - tx) + i10 * tx;
        let i1 = i01 * (1.0 - tx) + i11 * tx;
        let r = i0 * (1.0 - n_ty) + i1 * n_ty;
        results.push(r);

        results
    }

    fn map_fetch_bicubic(&self, map: &[f32], map_wh: Vector2<usize>, uv: Vec2) -> f32 {
        fn cubic_weight(t: f32) -> [f32; 4] {
            [
                ((-0.5 * t + 1.0) * t - 0.5) * t,
                ((1.5 * t - 2.5) * t) * t + 1.0,
                ((-1.5 * t + 2.0) * t + 0.5) * t,
                ((0.5 * t - 0.5) * t) * t,
            ]
        }

        let width = map_wh.x;
        let height = map_wh.y;

        let get_texel = |tex: &[f32], x: isize, y: isize| -> f32 {
            let xi = ((x % width as isize + width as isize) % width as isize) as usize;
            let yi = ((y % height as isize + height as isize) % height as isize) as usize;
            tex[yi * width + xi]
        };

        let x = uv.x * width as f32 - 0.5;
        let y = uv.y * height as f32 - 0.5;

        let x0 = x.floor() as isize;
        let y0 = y.floor() as isize;
        let dx = x - x0 as f32;
        let dy = y - y0 as f32;

        let wx = cubic_weight(dx);
        let wy = cubic_weight(dy);

        let mut result = 0.0;
        for j in 0..4 {
            for i in 0..4 {
                let val = get_texel(map, x0 + i - 1, y0 + j - 1);
                result += val * wx[i as usize] * wy[j as usize];
            }
        }

        result
    }

    fn map_resize(
        &self,
        map: &[f32],
        from_size: Vector2<usize>,
        to_size: Vector2<usize>,
    ) -> Vec<f32> {
        let mut new_map: Vec<f32> = Vec::new();
        for j in 0..to_size.y {
            for i in 0..to_size.x {
                let uv = vec2(i as f32 / to_size.x as f32, j as f32 / to_size.y as f32);
                let new_value = self.map_fetch_bicubic(map, from_size, uv);
                new_map.push(new_value);
            }
        }

        new_map
    }

    // Out: mapped position, world to local tranform
    fn surface_mapping(
        &self,
        map_coord: Vector2<usize>,
        pos: Vec3,
        to_world: bool,
    ) -> (Vec3, Mat3) {
        const PI: f32 = f32::consts::PI;
        const DELTA: f32 = 0.001;

        let mut new_pos = pos;
        let mut transform = Mat3::identity();
        match self.user_data.surface_type {
            SurfaceType::HeightMap => {
                let hmap_xrange = self.user_data.tile_map_wh.x as f32
                    * self.user_data.tile_width
                    * self.user_data.height_map_scale.x;
                let hmap_yrange = self.user_data.tile_map_wh.y as f32
                    * self.user_data.tile_width
                    * self.user_data.height_map_scale.y;
                let u = (pos.x
                    + (self.user_data.tile_map_half_wh.x) as f32 * self.user_data.tile_width)
                    / hmap_xrange;
                let v = (pos.y
                    + (self.user_data.tile_map_half_wh.y) as f32 * self.user_data.tile_width)
                    / hmap_yrange;
                let uv = vec2(u, v);
                let dt: f32 = DELTA;
                let height_vec = self.map_fetch_bilinear_with_auxiliary(
                    self.user_data.height_map.as_slice(),
                    self.user_data.height_map_wh,
                    uv,
                    dt,
                );
                let height = height_vec[0] * self.user_data.height_map_scale.z;
                new_pos.z = height;

                let height_r = height_vec[1] * self.user_data.height_map_scale.z;
                let height_l = height_vec[2] * self.user_data.height_map_scale.z;
                let height_u = height_vec[3] * self.user_data.height_map_scale.z;
                let height_d = height_vec[4] * self.user_data.height_map_scale.z;

                let local_x = vec3(1.0, 0.0, (height_r - height_l) / (2.0 * dt * hmap_xrange));
                let local_y = vec3(0.0, 1.0, (height_u - height_d) / (2.0 * dt * hmap_yrange));
                let local_z = local_x.cross(local_y).normalize();

                let local_to_world = Mat3::from_cols(local_x, local_y, local_z);
                let local_offset = local_to_world * vec3(0.0, 0.0, pos.z);
                new_pos += local_offset;
                if to_world {
                    transform = local_to_world;
                } else {
                    transform = local_to_world.invert().unwrap();
                }
            }
            SurfaceType::Sphere => {
                let xmax = self.user_data.tile_map_wh.x as f32 * self.user_data.tile_width;
                let ymax = self.user_data.tile_map_wh.y as f32 * self.user_data.tile_width;
                let block_w = xmax / 5.0;

                let get_uv =
                    |block_id_x: f32, block_id_y: f32, block_x: f32, block_y: f32| -> Vec2 {
                        let mut u: f32;
                        let mut v: f32;
                        if block_id_y == 0.0 {
                            if block_y < block_x {
                                if block_x - block_y == block_w {
                                    u = 0.0;
                                } else {
                                    u = (block_y / (block_w - (block_x - block_y)) + block_id_x)
                                        / 5.0;
                                }
                                v = (block_w - (block_x - block_y)) / block_w / 3.0;
                            } else {
                                u = (block_x / block_w + block_id_x) / 5.0
                                    + (block_y - block_x) / block_w * 0.1;
                                v = (block_y - block_x) / block_w / 3.0 + 1.0 / 3.0;
                            }
                        } else {
                            if block_y < block_x {
                                u = (block_x / block_w + block_id_x) / 5.0
                                    + (block_w - (block_x - block_y)) / block_w * 0.1;
                                v = (block_w - (block_x - block_y)) / block_w / 3.0 + 1.0 / 3.0;
                            } else {
                                if block_y - block_x == block_w {
                                    u = 0.0;
                                } else {
                                    u = (block_x / (block_w - (block_y - block_x)) + block_id_x)
                                        / 5.0
                                        + 0.1;
                                }
                                v = (block_y - block_x) / block_w / 3.0 + 2.0 / 3.0;
                            }
                        }

                        u += 0.5 * v.floor(); // In case v is out of [0, 1]
                        u *= 2.0 * PI;
                        v = (v - 0.5) * PI;

                        vec2(u, v)
                    };

                let uv_to_pos = |uv: Vec2| -> Vec3 {
                    vec3(
                        f32::cos(uv.y) * f32::cos(uv.x),
                        f32::cos(uv.y) * f32::sin(uv.x),
                        f32::sin(uv.y),
                    )
                };

                new_pos -= self.coord_to_pos(self.map_to_coord(vec2(0, 0)));
                let block_id_x = (5 * map_coord.x / self.user_data.tile_map_wh.x as usize) as f32;
                let block_id_y = (2 * map_coord.y / self.user_data.tile_map_wh.y as usize) as f32;
                let block_x = new_pos.x - block_id_x * block_w;
                let block_y = new_pos.y - block_id_y * block_w;

                let uv = get_uv(block_id_x, block_id_y, block_x, block_y);
                let local_z = uv_to_pos(uv);
                let r = self.user_data.sphere_radius;
                new_pos = local_z * r;

                let dt: f32 = DELTA * ymax;
                let pos_r = uv_to_pos(get_uv(block_id_x, block_id_y, block_x + dt, block_y)) * r;
                let pos_l = uv_to_pos(get_uv(block_id_x, block_id_y, block_x - dt, block_y)) * r;
                let pos_u = uv_to_pos(get_uv(block_id_x, block_id_y, block_x, block_y + dt)) * r;
                let pos_d = uv_to_pos(get_uv(block_id_x, block_id_y, block_x, block_y - dt)) * r;

                let local_x = (pos_r - pos_l) / (2.0 * dt);
                let local_y = (pos_u - pos_d) / (2.0 * dt);

                let local_to_world = Mat3::from_cols(local_x, local_y, local_z);
                let local_offset = local_to_world * vec3(0.0, 0.0, pos.z);
                new_pos += local_offset;
                if to_world {
                    transform = local_to_world;
                } else {
                    transform = local_to_world.invert().unwrap();
                }
            }
            SurfaceType::None => {}
        }

        (new_pos, transform)
    }

    fn lod_select_spatial(
        &self,
        map_coord: Vector2<usize>,
        cam_pos: Vec3,
    ) -> (usize, TileTransitionStatus) {
        let pos_offset = self.coord_to_pos(self.map_to_coord(map_coord));
        let tile_instance = self.tile_map[[map_coord.x, map_coord.y]].as_ref().unwrap();
        let tid = tile_instance.tid;
        let tile_base = &self.tile_base_data[0][tid.1][0];

        let lod_transition_dist = &self.user_data.lod_transition_dist;
        let center_dist = tile_instance.tile_center.distance(cam_pos);
        let mut selected_lod: usize = lod_transition_dist.len() - 1;
        for (lod_lv, &transition_dist) in lod_transition_dist.iter().enumerate() {
            if center_dist <= transition_dist {
                selected_lod = lod_lv;
                break;
            }
        }

        let mut trans_status = TileTransitionStatus::None;
        if self.user_data.lod_blending {
            let aabb = tile_base.aabb;
            let check_pos_vec = if self.user_data.lod_bbox_check {
                vec![
                    aabb.0,
                    vec3(aabb.0.x, aabb.0.y, aabb.1.z),
                    vec3(aabb.0.x, aabb.1.y, aabb.0.z),
                    vec3(aabb.0.x, aabb.1.y, aabb.1.z),
                    vec3(aabb.1.x, aabb.0.y, aabb.0.z),
                    vec3(aabb.1.x, aabb.0.y, aabb.1.z),
                    vec3(aabb.1.x, aabb.1.y, aabb.0.z),
                    aabb.1,
                ]
            } else {
                vec![tile_base.tile_center]
            };
            let mut min_dist: f32 = -1.0;
            let mut max_dist: f32 = -1.0;
            check_pos_vec.iter().for_each(|&pos| {
                let (p, _) = self.surface_mapping(map_coord, pos + pos_offset, true);
                let dist = p.distance(cam_pos);
                if (min_dist < 0.0) || (dist < min_dist) {
                    min_dist = dist;
                }
                if (max_dist < 0.0) || (dist > max_dist) {
                    max_dist = dist;
                }
            });

            // Blend with higher lod
            if selected_lod > 0 {
                let prev_transition_dist = lod_transition_dist[selected_lod - 1];
                if min_dist
                    < prev_transition_dist * (1.0 + self.user_data.lod_transition_width_ratio)
                        + self.user_data.lod_dist_tolerance
                {
                    trans_status = TileTransitionStatus::Changing(false);
                }
            }
            // Blend with lower lod
            if selected_lod < lod_transition_dist.len() - 1 {
                let transition_dist = lod_transition_dist[selected_lod];
                if max_dist
                    > transition_dist * (1.0 - self.user_data.lod_transition_width_ratio)
                        - self.user_data.lod_dist_tolerance
                {
                    trans_status = TileTransitionStatus::Changing(true);
                }
            }
        }

        (selected_lod, trans_status)
    }

    fn update_lod(&mut self, camera_pos: Vec3) {
        let xmax = self.user_data.tile_map_wh.x;
        let ymax = self.user_data.tile_map_wh.y;
        let cam_pos_u =
            (camera_pos.x - self.coord_to_pos(self.center_coord).x) / self.user_data.tile_width;
        let cam_pos_v =
            (camera_pos.y - self.coord_to_pos(self.center_coord).y) / self.user_data.tile_width;
        for i in 0..xmax {
            for j in 0..ymax {
                let map_coord = vec2(i, j);
                let (lod_lv, trans_status) = self.lod_select_spatial(map_coord, camera_pos);
                let tile_instance = self.tile_map[[i, j]].as_mut().unwrap();
                tile_instance.tid.0 = lod_lv;
                tile_instance.transition_status = trans_status;

                // Border tiles fade in/out, use tile spawning
                if self.user_data.lod_blending && self.user_data.surface_type != SurfaceType::Sphere
                {
                    let mut blend_f: f32 = 1.0;
                    if i == 0 {
                        blend_f *= 1.0 - cam_pos_u;
                    } else if i == xmax - 1 {
                        blend_f *= cam_pos_u;
                    }
                    if j == 0 {
                        blend_f *= 1.0 - cam_pos_v;
                    } else if j == ymax - 1 {
                        blend_f *= cam_pos_v;
                    }
                    if blend_f != 1.0 {
                        let tile_instance = self.tile_map[[i, j]].as_mut().unwrap();
                        tile_instance.transition_status = TileTransitionStatus::Spawning(blend_f);
                    }
                }
            }
        }
    }

    fn compute_corner_edge(
        &self,
        map_coord: Vector2<usize>,
        tile_base: &TileBaseData,
    ) -> (Option<TileCornerData>, Option<TileEdgeData>) {
        if self.user_data.tile_sort_type != TileSortType::Graph
            && self.user_data.merge_type != SelectiveMergeType::Edge
        {
            return (None, None);
        }

        let d_coords = [vec2(0, 0), vec2(0, 1), vec2(1, 1), vec2(1, 0)];
        let mut corner_data = TileCornerData::new();
        let mut edge_data = TileEdgeData::new();
        for corner_i in 0..4_usize {
            let mut copy_from_neighbor = false;
            if let Some((n_mc, n_edge_idx)) =
                self.neighbor_map[[map_coord.x, map_coord.y]][corner_i]
            {
                if let Some(n_instance) = self.tile_map[[n_mc.x, n_mc.y]].as_ref() {
                    if let Some(n_corner) = &n_instance.corner_data {
                        corner_data[corner_i] = n_corner[(n_edge_idx + 1) % 4].clone();
                        copy_from_neighbor = true;
                    }
                }
            }
            if !copy_from_neighbor {
                if let Some((n_mc, n_edge_idx)) =
                    self.neighbor_map[[map_coord.x, map_coord.y]][(corner_i + 3) % 4]
                {
                    if let Some(n_instance) = self.tile_map[[n_mc.x, n_mc.y]].as_ref() {
                        if let Some(n_corner) = &n_instance.corner_data {
                            corner_data[corner_i] = n_corner[n_edge_idx].clone();
                            copy_from_neighbor = true;
                        }
                    }
                }
            }
            if !copy_from_neighbor {
                let corner_mc = map_coord + d_coords[corner_i];
                let corner_pos = self.coord_to_pos(self.map_to_coord(corner_mc))
                    + Vec3::unit_z() * tile_base.tile_center.z;
                corner_data[corner_i] = self.surface_mapping(map_coord, corner_pos, true).clone();
            }
        }
        for edge_i in 0..4_usize {
            let (corner1_pos, corner1_to_world) = corner_data[edge_i];
            let (corner2_pos, corner2_to_world) = corner_data[(edge_i + 1) % 4];
            let edge_pos = (corner1_pos + corner2_pos) / 2.0;
            let corner_dir = corner2_pos - corner1_pos;
            let normal_1 = corner1_to_world * Vec3::unit_z();
            let normal_2 = corner2_to_world * Vec3::unit_z();
            let normal = (normal_1 + normal_2) / 2.0;
            let edge_normal = normal.cross(corner_dir).normalize();
            // log!("{:?}, {:?}", corner1_pos, corner2_pos);
            // log!("{:?}, {:?}, {:?}, {:?}", edge_normal, normal, normal_1, normal_2);
            edge_data[edge_i] = (edge_pos, edge_normal);
        }

        (Some(corner_data), Some(edge_data))
    }

    fn update_tile_map(&mut self, camera_pos: Vec3) {
        // TODO: assume 2^4 = 16 edge combinations for now
        const NUM_P: usize = 2;

        // log!{"Wang thread: update_tile_map()"};

        // Copy existing tiles
        let xmax = self.user_data.tile_map_wh.x;
        let ymax = self.user_data.tile_map_wh.y;
        self.camera_pos = camera_pos;

        if self.user_data.surface_type != SurfaceType::Sphere {
            let prev_center_coord = self.center_coord;
            self.center_coord = self.pos_to_coord(camera_pos);
            let mut new_tile_map = Array2::from_elem((xmax, ymax), None);
            for i in 0..xmax {
                for j in 0..ymax {
                    let new_mc = vec2(i, j);
                    let prev_mc = vec2(
                        i as i32 + self.center_coord.x - prev_center_coord.x,
                        j as i32 + self.center_coord.y - prev_center_coord.y,
                    );

                    if prev_mc.x >= 0
                        && prev_mc.x < xmax as i32
                        && prev_mc.y >= 0
                        && prev_mc.y < ymax as i32
                    {
                        if let Some(prev_tile_instance) =
                            self.tile_map[[prev_mc.x as usize, prev_mc.y as usize]].as_ref()
                        {
                            let tile_instance = TileInstance {
                                tid: (0, prev_tile_instance.tid.1), // lod_id initialize later
                                view_id: 0, // initialize during sort
                                tile_offset: prev_tile_instance.tile_offset,
                                map_index: self.map_to_index(new_mc),
                                map_coord: new_mc,
                                tile_center: prev_tile_instance.tile_center,
                                merge_status: TileMergeStatus::None,
                                transition_status: TileTransitionStatus::None,
                                to_local: prev_tile_instance.to_local,
                                corner_data: prev_tile_instance.corner_data.clone(),
                                edge_data: prev_tile_instance.edge_data.clone(),
                            };
                            new_tile_map[[i, j]] = Some(tile_instance);
                        }
                    }
                }
            }
            self.tile_map = new_tile_map;
        } else {
            self.center_coord = Vector2::zero();
        }
        // log!("center: ({}, {}) -> ({}, {})", self.center_coord.x, self.center_coord.y, new_center_coord.x, new_center_coord.y);

        // Spawn new tiles
        for i in 0..xmax {
            for j in 0..ymax {
                if self.tile_map[[i, j]].is_some() {
                    continue;
                }

                let map_coord = vec2(i, j);
                let tile_offset = self.coord_to_pos(self.map_to_coord(map_coord));

                // Tile id
                let mut color: Vector4<usize> = Vector4::zero(); // west, north, east, south
                for idx in 0..4 {
                    if let Some((neighbor_mc, n_idx)) = self.neighbor_map[[i, j]][idx] {
                        if let Some(neighbor_tile) = &self.tile_map[[neighbor_mc.x, neighbor_mc.y]]
                        {
                            let n_color = self.tile_id_to_color(neighbor_tile.tid.1);
                            color[idx] = n_color[n_idx];
                            // color[idx] = n_color[(idx + 2) % 4];    // west <-> east, north <-> south
                        } else {
                            color[idx] = self.rng.random_range(0..NUM_P);
                        }
                    } else {
                        color[idx] = self.rng.random_range(0..NUM_P);
                    }
                }
                let center_option = self.rng.random_range(0..self.user_data.center_option);
                let tile_id = self.color_to_tile_id(color, center_option);

                // Tile center
                let tile_base = &self.tile_base_data[0][tile_id][0];
                let tile_center = tile_base.tile_center + tile_offset;
                let (tile_center, to_local) = self.surface_mapping(map_coord, tile_center, false);

                // Corner & edge data
                let (corner_data, edge_data) = self.compute_corner_edge(map_coord, tile_base);

                self.tile_map[[i, j]] = Some(TileInstance {
                    tid: (0, tile_id),
                    view_id: 0,
                    tile_offset,
                    map_index: self.map_to_index(map_coord),
                    map_coord,
                    tile_center,
                    merge_status: TileMergeStatus::None,
                    transition_status: TileTransitionStatus::None,
                    to_local,
                    corner_data,
                    edge_data,
                });
                // log!("update {:?}", map_coord);
            }
        }

        self.update_lod(camera_pos);
    }

    // World coordinate to world position
    fn coord_to_pos(&self, c: Vector2<i32>) -> Vec3 {
        let position_offset = c.cast::<f32>().unwrap() * self.user_data.tile_width;

        vec3(position_offset.x, position_offset.y, 0.0)
    }

    // World position to world coordinate
    fn pos_to_coord(&self, p: Vec3) -> Vector2<i32> {
        let pos = p / self.user_data.tile_width;

        Vector2::<i32>::new(pos.x.floor() as i32, pos.y.floor() as i32)
    }

    // Map index to map coordinate
    // (0, 0) -> (0, 1) -> ... -> (0, h) -> (1, 0) -> ... -> (w, h)
    fn index_to_map(&self, index: usize) -> Vector2<usize> {
        let map_height = self.user_data.tile_map_wh.y;
        let map_coord_x = index / map_height;
        let map_coord_y = index % map_height;

        vec2(map_coord_x, map_coord_y)
    }

    // Map coordinate to map index
    fn map_to_index(&self, map_coord: Vector2<usize>) -> usize {
        let map_height = self.user_data.tile_map_wh.y;

        map_coord.x * map_height + map_coord.y
    }

    // Map coordinate to world coordinate
    fn map_to_coord(&self, map_coord: Vector2<usize>) -> Vector2<i32> {
        Vector2::<i32>::new(
            map_coord.x as i32 + self.center_coord.x - self.user_data.tile_map_half_wh.x as i32,
            map_coord.y as i32 + self.center_coord.y - self.user_data.tile_map_half_wh.y as i32,
        )
    }

    // World coordinate to map coordinate 
    fn coord_to_map(&self, coord: Vector2<i32>) -> Vector2<usize> {
        Vector2::<usize>::new(
            (coord.x - self.center_coord.x + self.user_data.tile_map_half_wh.x as i32) as usize,
            (coord.y - self.center_coord.y + self.user_data.tile_map_half_wh.y as i32) as usize,
        )
    }

    fn tile_id_to_color(&self, tile_id: usize) -> Vector4<usize> {
        // TODO: assume 2^4 = 16 edge combinations for now
        // West, North, East, South
        Vector4::<usize>::new(
            tile_id % 16 / 8 % 2,
            tile_id % 16 / 4 % 2,
            tile_id % 16 / 2 % 2,
            tile_id % 16 % 2,
        )
    }

    fn color_to_tile_id(&self, color: Vector4<usize>, center_idx: usize) -> usize {
        // TODO: assume 2^4 = 16 edge combinations for now
        let edge_id = color.x * 8 + color.y * 4 + color.z * 2 + color.w;

        edge_id + 16 * center_idx
    }
}

pub fn upload_height_map() -> mpsc::Receiver<(Vec<f32>, Vector2<usize>)> {
    let (tx, rx) = mpsc::channel();
    let task = rfd::AsyncFileDialog::new()
        .add_filter("Height Texture", &["png", "jpg"])
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
            let width = img.width() as usize;
            let height = img.height() as usize;

            let raw_map: Vec<f32> = img.to_rgba32f().into_raw();
            let mut height_map: Vec<f32> = Vec::new();
            for i in 0..height {
                for j in 0..width {
                    let raw_index = ((height - 1 - i) * width + j) * 4;
                    height_map.push(raw_map[raw_index]);
                }
            }

            let mut h_max = f32::MIN;
            let mut h_min = f32::MAX;
            height_map.iter().for_each(|&v| {
                if v > h_max {
                    h_max = v;
                }
                if v < h_min {
                    h_min = v;
                }
            });
            height_map = height_map
                .iter()
                .map(|&v| {
                    let n = (v - h_min) / (h_max - h_min);
                    n * 2.0 - 1.0
                })
                .collect();

            log!("Load height map: max {}, min {}", h_max, h_min);
            tx.send((height_map, vec2(width, height)))
                .expect("Error sending fly path json to main thread.");
        }
    });

    rx
}