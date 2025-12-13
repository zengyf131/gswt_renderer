// Derived from https://github.com/BladeTransformerLLC/gauzilla

// Vertex shader
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) v_color: vec4<f32>,
    @location(1) v_position: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> u_camera: CameraUniforms;

@group(0) @binding(1)
var<uniform> u_scene: SceneUniforms;

@group(0) @binding(2)
var t_gaussian_tex: texture_2d<u32>;

@group(0) @binding(3)
var t_height_map: texture_2d<f32>;
@group(0) @binding(4)
var s_height_map: sampler;

@group(1) @binding(0)
var<storage, read> u_tile_array: array<TileUniforms>;

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
    @location(1) gs_index: u32,
    @location(2) map_id: u32,
    @location(3) lod_id: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let u_tile = u_tile_array[0];

    // Early discard: false lod id
    if u_tile.valid_lod_id >= 0 && u_tile.valid_lod_id != i32(lod_id) {
        out.clip_position = vec4(0.0, 0.0, 2.0, 0.0);
        return out;
    }

    // 0x3ffu (1023 in decimal) masks the lower 10 bits of index
    let u = (gs_index & 0x3ffu) << 1u;
    let v = gs_index >> 10u;

    let pos = textureLoad(t_gaussian_tex, vec2(u, v), 0).rgb;
    var center = bitcast<vec3<f32>>(pos); // splat pos in world space

    // Offset
    var offset = u_tile.offset;
    var map_wh = 2u * u_scene.map_half_wh;
    if u_scene.surface_type != 2u {
        map_wh = map_wh + 1u;
    }
    if u_tile.single_draw == 1u {
        offset = vec3(
            f32(i32(map_id / map_wh.y - u_scene.map_half_wh.x) + u_scene.center_coord.x) * u_scene.tile_width,
            f32(i32(map_id % map_wh.y - u_scene.map_half_wh.y) + u_scene.center_coord.y) * u_scene.tile_width,
            0.0
        );
    }
    center = center + offset;
    center *= u_scene.scene_scale;
    let ori_center = center;

    let map_xrange = (2.0 * f32(u_scene.map_half_wh.x) + 1.0) * u_scene.tile_width * u_scene.height_map_scale.x;
    let map_yrange = (2.0 * f32(u_scene.map_half_wh.y) + 1.0) * u_scene.tile_width * u_scene.height_map_scale.y;
    let map_u = (ori_center.x + f32(u_scene.map_half_wh.x) * u_scene.tile_width) / map_xrange;
    let map_v = (ori_center.y + f32(u_scene.map_half_wh.y) * u_scene.tile_width) / map_yrange;
    let map_uv = vec2(map_u, map_v);

    // Surface mapping
    var mapped_center = vec3(center.xy, 0.0);
    var transform = mat3x3(vec3(1.0), vec3(1.0), vec3(1.0));
    var surface_normal = vec3(0.0, 0.0, 1.0);
    if u_scene.surface_type > 0u {
        surface_mapping(center.xy, map_id, &mapped_center, &transform);
        center = mapped_center + transform * vec3(0.0, 0.0, center.z);
        surface_normal = transform * surface_normal;
    }

    if u_scene.use_clip == 1u && mapped_center.z < u_scene.clip_height {
        out.clip_position = vec4(0.0, 0.0, 2.0, 0.0);
        return out;
    }

    // Lod transition early discard
    // let t_ratio = transition_ratio;
    var t_ratio = -1.0;
    var spawning = 0;
    var higher_lod = 0u;
    if u_tile.changing == 1u {
        if t_ratio < 0.0 {
            // Changing
            let cam_dist = distance(center, u_camera.cam_pos);
            if u_tile.single_draw == 1u {
                // Find current lod, cannot trust tid here, because could be merged tiles
                // for (var i = 0u; i < u_scene.num_lod - 1; i += 1u) {
                //     let dist = u_scene.transition_dist_vec[i / 4u][i % 4u];
                //     if dist >= cam_dist {
                //         this_lod = i;
                //         break;
                //     }
                // }
                // if transition_dist < 0.0 {
                //     this_lod = u_scene.num_lod - 1;
                // }

                // Find higher lod
                if lod_id == 0u {
                    higher_lod = 0u;
                } else if lod_id == u_scene.num_lod - 1 {
                    higher_lod = lod_id - 1u;
                } else {
                    // Test 2 distances
                    let dist_1 = u_scene.transition_dist_vec[(lod_id - 1u)/4u][(lod_id - 1u)%4u];
                    let dist_2 = u_scene.transition_dist_vec[lod_id/4u][lod_id%4u];
                    if cam_dist - dist_1 < dist_2 - cam_dist {
                        higher_lod = lod_id - 1u;
                    } else {
                        higher_lod = lod_id;
                    }
                }
            } else {
                if u_tile.changing_to_lower == 1 {
                    higher_lod = u_tile.tile_id.x;
                } else {
                    higher_lod = u_tile.tile_id.x - 1u;
                }
            }

            let transition_dist = u_scene.transition_dist_vec[higher_lod/4u][higher_lod%4u];
            let transition_half_width = u_scene.transition_width_ratio * transition_dist;
            t_ratio = clamp((cam_dist - transition_dist) / transition_half_width + 0.5, 0.0, 1.0);

            if ((lod_id == higher_lod + 1u) && (t_ratio == 0.0)) || ((lod_id == higher_lod) && (t_ratio == 1.0)) {
                out.clip_position = vec4(0.0, 0.0, 2.0, 0.0);
                return out;
            }
        } else {
            // Spawning
            spawning = 1;
            if t_ratio == 0.0 {
                out.clip_position = vec4(0.0, 0.0, 2.0, 0.0);
                return out;
            }
        }
    }

    let opengl_to_wgpu = mat4x4(
        vec4(1.0, 0.0, 0.0, 0.0),
        vec4(0.0, 1.0, 0.0, 0.0),
        vec4(0.0, 0.0, 0.5, 0.0),
        vec4(0.0, 0.0, 0.5, 1.0),
    );

    let cam = u_camera.view * vec4(center, 1.0);
    let pos2d = opengl_to_wgpu * u_camera.projection * cam;
    // let pos2d = u_camera.projection * cam;

    let clip = 1.2 * pos2d.w;
    if pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip {
        out.clip_position = vec4(0.0, 0.0, 2.0, 0.0);
        return out;
    }

    let cov = textureLoad(t_gaussian_tex, vec2(u | 1u, v), 0);
    // cf. Eq.29 of https://www.cs.umd.edu/~zwicker/publications/EWASplatting-TVCG02.pdf
    let u1 = unpackHalf2x16(cov.x); // a, b
    let u2 = unpackHalf2x16(cov.y); // c, d
    let u3 = unpackHalf2x16(cov.z); // e, f
    // eq.24, symmetric matrix, R * S * S^T * R^T
    var Vrk = mat3x3(
        u1.x, u1.y, u2.x,
        u1.y, u2.y, u3.x,
        u2.x, u3.x, u3.y
    );

    // Point cloud
    if u_scene.point_cloud_radius > 0.0 {
        var p_r = u_scene.point_cloud_radius;
        if u_scene.draw_mode > 0u {
            p_r *= pow(2.0, f32(u_tile.tile_id.x));
        }
        Vrk = mat3x3(
            p_r, 0.0, 0.0,
            0.0, p_r, 0.0,
            0.0, 0.0, p_r
        );
    }

    // Surface mapping
    if u_scene.surface_type > 0u {
        Vrk = transform * Vrk * transpose(transform);
    }

    // Global scene scale
    let scene_scale_mat = mat3x3(
        u_scene.scene_scale.x, 0.0, 0.0,
        0.0, u_scene.scene_scale.y, 0.0,
        0.0, 0.0, u_scene.scene_scale.z
    );
    Vrk = scene_scale_mat * Vrk * transpose(scene_scale_mat);

    let view3 = mat3x3(
        u_camera.view[0].xyz,
        u_camera.view[1].xyz,
        u_camera.view[2].xyz
    );

    // splat pos in camera space
    var t = view3 * (center - u_camera.cam_pos);

    // 3D camera space -> 2D screen space
    let txtz = t.x / t.z;
    let tytz = t.y / t.z;

    let limx = 1.3 * u_camera.htan_fov.x;
    let limy = 1.3 * u_camera.htan_fov.y;

    t.x = clamp(txtz, -limx, limx)*t.z;
    t.y = clamp(tytz, -limy, limy)*t.z;

    // Jacobian for the Taylor approximation of the nonlinear camera->ray transformation (eq.29)
    let tz2 = t.z*t.z;
    let J_T = mat3x3(
        u_camera.focal.x/t.z, 0., -u_camera.focal.x*t.x/tz2,
        0., u_camera.focal.y/t.z , -u_camera.focal.y*t.y/tz2,
        0., 0., 0.
    );
    /*
        f32 cam_z_2 = cam.z * cam.z;
        mat3 J_T = mat3(
            focal.x/cam.z, 0., -(focal.x*cam.x)/cam_z_2,
            0., focal.y/cam.z, -(focal.y*cam.y)/cam_z_2,
            0., 0., 0.
        );
    */

    let T = transpose(view3) * J_T;

    // covariance matrix in ray space
    let cov2d = transpose(T) * Vrk * T;

    let mid = 0.5*(cov2d[0][0] + cov2d[1][1]);
    let radius = length(vec2(0.5*(cov2d[0][0] - cov2d[1][1]), cov2d[0][1]));
    let lambda1 = mid + radius;
    let lambda2 = mid - radius;

    if lambda2 < 0.0 {
        out.clip_position = vec4(0.0, 0.0, 2.0, 0.0);
        return out;
    }
    let diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
    let majorAxis = min(sqrt(2.0*lambda1), 1024.0) * diagonalVector;
    let minorAxis = min(sqrt(2.0*lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    var vColor = vec4(
        f32((cov.w) & 0xffu), // 0xffu == 255 in decimal, masks the lowest 8 bits (value in [0, 255])
        f32((cov.w >> 8u) & 0xffu),
        f32((cov.w >> 16u) & 0xffu),
        f32((cov.w >> 24u) & 0xffu)
    ) / 255.0;

    // Debug draw
    var debug_draw_color = vColor;
    switch u_scene.draw_mode {
        case 0u: {} // Normal
        case 1u: { // TileID
            let gray_scale = clamp((vColor.r + vColor.g + vColor.b) / 0.6, 0.0, 1.0);
            debug_draw_color = vec4(gray_scale, gray_scale, gray_scale, debug_draw_color.a);
            let vpos = bitcast<vec3<f32>>(pos);
            let margin = 0.05 * u_scene.tile_width;
            if u_tile.single_draw == 1u {
                debug_draw_color *= vec4(randomVec3(u_tile.offset.xy), 1.0);
            } else if (vpos.x < margin) {
                if ((vpos.y < margin) || (vpos.y > u_scene.tile_width - margin)) {
                    debug_draw_color = vec4(0.5, 0.5, 0.5, debug_draw_color.a);
                } else {
                    if (u_tile.tile_id.y / 8u % 2u == 0u) {
                        debug_draw_color = vec4(1.0, 0.0, 0.0, debug_draw_color.a);
                    } else {
                        debug_draw_color = vec4(0.0, 1.0, 0.13, debug_draw_color.a);
                    }
                }
            } else if (vpos.x > u_scene.tile_width - margin) {
                if ((vpos.y < margin) || (vpos.y > u_scene.tile_width - margin)) {
                    debug_draw_color = vec4(0.5, 0.5, 0.5, debug_draw_color.a);
                } else {
                    if (u_tile.tile_id.y / 2u % 2u == 0u) {
                        debug_draw_color = vec4(1.0, 0.0, 0.0, debug_draw_color.a);
                    } else {
                        debug_draw_color = vec4(0.0, 1.0, 0.13, debug_draw_color.a);
                    }
                }
            } else if (vpos.y < margin) {
                if (u_tile.tile_id.y % 2u == 0u) {
                    if (u_scene.surface_type == 2u) {
                        debug_draw_color = vec4(1.0, 0.0, 0.0, debug_draw_color.a);
                    } else {
                        debug_draw_color = vec4(1.0, 0.85, 0.0, debug_draw_color.a);
                    }
                } else {
                    if (u_scene.surface_type == 2u) {
                        debug_draw_color = vec4(0.0, 1.0, 0.13, debug_draw_color.a);
                    } else {
                        debug_draw_color = vec4(0.0, 0.58, 1.0, debug_draw_color.a);
                    }
                }
            } else if (vpos.y > u_scene.tile_width - margin) {
                if (u_tile.tile_id.y / 4u % 2u == 0u) {
                    if (u_scene.surface_type == 2u) {
                        debug_draw_color = vec4(1.0, 0.0, 0.0, debug_draw_color.a);
                    } else {
                        debug_draw_color = vec4(1.0, 0.85, 0.0, debug_draw_color.a);
                    }
                } else {
                    if (u_scene.surface_type == 2u) {
                        debug_draw_color = vec4(0.0, 1.0, 0.13, debug_draw_color.a);
                    } else {
                        debug_draw_color = vec4(0.0, 0.58, 1.0, debug_draw_color.a);
                    }
                }
            }
        }
        case 2u: { // TileLOD
            if (t_ratio > 0.0 && t_ratio < 1.0) {
                debug_draw_color = vec4(0.0, 0.0, 0.0, debug_draw_color.a);
            } else if u_tile.changing == 1 {
                debug_draw_color = vec4(0.0, 1.0, 0.0, debug_draw_color.a);
            } else {
                var color_x: f32 = 0.0;
                if u_tile.tile_id[0] < 3u {
                    color_x = (3.0 - f32(u_tile.tile_id[0])) / 3.0;
                }
                var color_y: f32 = 1.0;
                if u_tile.tile_id[0] >= 3u {
                    color_y = (6.0 - f32(u_tile.tile_id[0])) / 3.0;
                }
                debug_draw_color = vec4(
                    0.5,
                    color_x,
                    color_y,
                    debug_draw_color.a,
                );
            }
        }
        case 3u: { // LOD
            if (t_ratio > 0.0 && t_ratio < 1.0) {
                debug_draw_color = vec4(0.0, 0.0, 0.0, debug_draw_color.a);
            } else {
                var color_x: f32 = 0.0;
                var color_y: f32 = 1.0;
                if u_tile.single_lod_id >= 0 {
                    if u_tile.single_lod_id < 3 {
                        color_x = (3.0 - f32(u_tile.single_lod_id)) / 3.0;
                    } else {
                        color_y = (6.0 - f32(u_tile.single_lod_id)) / 3.0;
                    }
                } else {
                    if lod_id < 3u {
                        color_x = (3.0 - f32(lod_id)) / 3.0;
                    } else {
                        color_y = (6.0 - f32(lod_id)) / 3.0;
                    }
                }
                debug_draw_color = vec4(
                    0.5,
                    color_x,
                    color_y,
                    debug_draw_color.a,
                );
            }
        }
        case 4u: { // View
            var color_x: f32 = 0.0;
            if u_tile.tile_id[2] < 4u {
                color_x = (4.0 - f32(u_tile.tile_id[2])) / 4.0;
            }
            var color_y: f32 = 0.0;
            if u_tile.tile_id[2] >= 4u {
                color_y = (8.0 - f32(u_tile.tile_id[2])) / 4.0;
            }
            if u_tile.tile_id[2] >= 8u {
                color_x = 1.0;
                color_y = 1.0;
            }
            debug_draw_color = vec4(
                0.5,
                color_x,
                color_y,
                debug_draw_color.a,
            );
        }
        default: {}
    }
    vColor = debug_draw_color;

    // Lod transition
    if u_tile.changing == 1u {
        if (lod_id != higher_lod || spawning == 1) {
            vColor.a = vColor.a * t_ratio;
        } else {
            vColor.a = vColor.a * (1.0 - t_ratio);
        }
    }

    vColor = vColor * clamp(pos2d.z/pos2d.w+1.0, 0.0, 1.0);

    out.v_color = vColor;
    out.v_position = position;

    let vCenter = pos2d.xyz / pos2d.w;

    let major = (position.x*majorAxis) / u_camera.viewport;
    let minor = (position.y*minorAxis) / u_camera.viewport;
    out.clip_position = vec4(vCenter.xy + u_scene.splat_scale*(major + minor), vCenter.z, 1.0);

    return out;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let A = -dot(in.v_position, in.v_position);
    if A < -4.0 {
        discard;
    }
    let B = exp(A) * in.v_color.a;
    let frag_color = vec4(B * in.v_color.rgb, B);

    return frag_color;
}

struct CameraUniforms {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
    focal: vec2<f32>,
    viewport: vec2<f32>,
    htan_fov: vec2<f32>,
    cam_pos: vec3<f32>,
}

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

    map_half_wh: vec2<u32>,
    center_coord: vec2<i32>,
    transition_dist_vec: array<vec4<f32>, 4>,
    height_map_scale: vec3<f32>,
    scene_scale: vec3<f32>,
}

struct TileUniforms {
    single_draw: u32,
    map_index: u32,
    single_lod_id: i32,
    valid_lod_id: i32,
    changing: u32,
    changing_to_lower: i32,

    tile_id: vec3<u32>,
    offset: vec3<f32>,
    map_coord: vec2<u32>,
}

fn halfToFloat(h: u32) -> f32 {
    let s = f32((h >> 15u) & 0x1u);
    let e = f32((h >> 10u) & 0x1Fu);
    let f = f32(h & 0x3FFu);

    if (e == 0.0) {
        // Subnormal
        return pow(2.0, -15.0) * (f / pow(2.0, 10.0)) * (1.0 - 2.0 * s);
    } else if (e == 31.0) {
        // Inf or NaN
        // return (1.0 - 2.0 * s) * (f == 0.0 ? f32(INFINITY) : f32(NAN));
        return 0.0;
    } else {
        // Normalized
        return pow(2.0, e - 15.0) * (1.0 + f / pow(2.0, 10.0)) * (1.0 - 2.0 * s);
    }
}

fn unpackHalf2x16(val: u32) -> vec2<f32> {
    let lo = val & 0xFFFFu;
    let hi = (val >> 16u) & 0xFFFFu;
    return vec2<f32>(halfToFloat(lo), halfToFloat(hi));
}

fn rand(co: vec2<f32>) -> f32 {
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

fn randomVec3(seed: vec2<f32>) -> vec3<f32> {
    return vec3(
        rand(seed),
        rand(seed + 23.45),
        rand(seed + 67.89)
    );
}

// Sphere surface
fn sphere_get_uv(block_id_x: f32, block_id_y: f32, block_x: f32, block_y: f32) -> vec2<f32> {
    let PI = 3.1415926535897932384626433832795;
    let xmax = f32(u_scene.map_half_wh.x) * 2.0 * u_scene.tile_width;
    let block_w = xmax / 5.0;

    var u = 0.0;
    var v = 0.0;
    if block_id_y == 0.0 {
        if block_y < block_x {
            if block_x - block_y == block_w {
                u = 0.0;
            } else {
                u = (block_y / (block_w - (block_x - block_y)) + block_id_x) / 5.0;
            }
            v = (block_w - (block_x - block_y)) / block_w / 3.0;
        } else {
            u = (block_x / block_w + block_id_x) / 5.0 + (block_y - block_x) / block_w * 0.1;
            v = (block_y - block_x) / block_w / 3.0 + 1.0 / 3.0;
        }
    } else {
        if block_y < block_x {
            u = (block_x / block_w + block_id_x) / 5.0 + (block_w - (block_x - block_y)) / block_w * 0.1;
            v = (block_w - (block_x - block_y)) / block_w / 3.0 + 1.0 / 3.0;
        } else {
            if (block_y - block_x == block_w) {
                u = 0.0;
            } else {
                u = (block_x / (block_w - (block_y - block_x)) + block_id_x) / 5.0 + 0.1;
            }
            v = (block_y - block_x) / block_w / 3.0 + 2.0 / 3.0;
        }
    }

    u += 0.5 * floor(v);
    u *= 2.0 * PI;
    v = (v - 0.5) * PI;

    return vec2(u, v);
}

// Sphere surface
fn sphere_uv_to_pos(uv: vec2<f32>) -> vec3<f32> {
    return vec3(
        cos(uv.y) * cos(uv.x),
        cos(uv.y) * sin(uv.x),
        sin(uv.y)
    );
}

// Out: new position, to_world transform
fn surface_mapping(pos: vec2<f32>, map_id: u32, new_pos: ptr<function, vec3<f32>>, transform: ptr<function, mat3x3<f32>>) {
    let u_tile = u_tile_array[0];
    let DELTA = 0.001;

    (*new_pos) = vec3(pos, 0.0);
    (*transform) = mat3x3(vec3(1.0), vec3(1.0), vec3(1.0));
    if u_scene.surface_type == 1u {
        let hmap_xrange = (2.0 * f32(u_scene.map_half_wh.x) + 1.0) * u_scene.tile_width * u_scene.height_map_scale.x;
        let hmap_yrange = (2.0 * f32(u_scene.map_half_wh.y) + 1.0) * u_scene.tile_width * u_scene.height_map_scale.y;
        let h_u = (pos.x + f32(u_scene.map_half_wh.x) * u_scene.tile_width) / hmap_xrange;
        let h_v = (pos.y + f32(u_scene.map_half_wh.y) * u_scene.tile_width) / hmap_yrange;
        let raw_height = textureSampleLevel(t_height_map, s_height_map, vec2(h_u, h_v), 0.0).r;
        (*new_pos).z = raw_height * u_scene.height_map_scale.z;

        let dt = DELTA;
        let height_r = textureSampleLevel(t_height_map, s_height_map, vec2(h_u + dt, h_v), 0.0).r * u_scene.height_map_scale.z;
        let height_l = textureSampleLevel(t_height_map, s_height_map, vec2(h_u - dt, h_v), 0.0).r * u_scene.height_map_scale.z;
        let height_u = textureSampleLevel(t_height_map, s_height_map, vec2(h_u, h_v + dt), 0.0).r * u_scene.height_map_scale.z;
        let height_d = textureSampleLevel(t_height_map, s_height_map, vec2(h_u, h_v - dt), 0.0).r * u_scene.height_map_scale.z;

        let local_x = vec3(1.0, 0.0, (height_r - height_l) / (2.0 * dt * hmap_xrange));
        let local_y = vec3(0.0, 1.0, (height_u - height_d) / (2.0 * dt * hmap_yrange));
        let local_z = normalize(cross(local_x, local_y));

        (*transform) = mat3x3(local_x, local_y, local_z);
    } else if u_scene.surface_type == 2u {
        let xmax = f32(u_scene.map_half_wh.x) * 2.0 * u_scene.tile_width;
        let ymax = f32(u_scene.map_half_wh.y) * 2.0 * u_scene.tile_width;
        let block_w = xmax / 5.0;

        // new_pos -= self.coord_to_pos(self.map_to_coord(vec2(0, 0)));
        (*new_pos).x -= f32(u_scene.center_coord.x - i32(u_scene.map_half_wh.x)) * u_scene.tile_width;
        (*new_pos).y -= f32(u_scene.center_coord.y - i32(u_scene.map_half_wh.y)) * u_scene.tile_width;
        var block_id_x = f32(5 * u_tile.map_coord.x / (u_scene.map_half_wh.x * 2));
        var block_id_y = f32(2 * u_tile.map_coord.y / (u_scene.map_half_wh.y * 2));
        if u_tile.single_draw == 1u {
            let map_height = 2u * u_scene.map_half_wh.y;
            let this_mc = vec2<u32>(map_id / map_height, map_id % map_height);
            block_id_x = f32(5 * this_mc.x / (u_scene.map_half_wh.x * 2u));
            block_id_y = f32(2 * this_mc.y / (u_scene.map_half_wh.y * 2u));
        }
        let block_x = (*new_pos).x - block_id_x * block_w;
        let block_y = (*new_pos).y - block_id_y * block_w;

        let uv = sphere_get_uv(block_id_x, block_id_y, block_x, block_y);
        let local_z = sphere_uv_to_pos(uv);
        (*new_pos) = local_z * u_scene.sphere_radius;

        let dt = DELTA * ymax;
        let pos_r = sphere_uv_to_pos(sphere_get_uv(block_id_x, block_id_y, block_x + dt, block_y)) * u_scene.sphere_radius;
        let pos_l = sphere_uv_to_pos(sphere_get_uv(block_id_x, block_id_y, block_x - dt, block_y)) * u_scene.sphere_radius;
        let pos_u = sphere_uv_to_pos(sphere_get_uv(block_id_x, block_id_y, block_x, block_y + dt)) * u_scene.sphere_radius;
        let pos_d = sphere_uv_to_pos(sphere_get_uv(block_id_x, block_id_y, block_x, block_y - dt)) * u_scene.sphere_radius;

        let local_x = (pos_r - pos_l) / (2.0 * dt);
        let local_y = (pos_u - pos_d) / (2.0 * dt);

        (*transform) = mat3x3(local_x, local_y, local_z);
    }
}