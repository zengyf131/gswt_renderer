struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) mapped_height: f32,
    @location(2) vertex_alpha: f32,
}

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

    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    map_half_wh: vec2<u32>,
    center_coord: vec2<i32>,
    height_map_scale: vec3<f32>,
    cam_pos: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(0)
var t_height_map: texture_2d<f32>;
@group(1) @binding(1)
var s_height_map: sampler;

@group(1) @binding(2)
var t_proxy_texture: texture_2d<f32>;
@group(1) @binding(3)
var s_proxy_texture: sampler;

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;

    var real_position = position;
    var height = uniforms.height_offset;
    out.vertex_alpha = 1.0;
    if uniforms.map_proxy == 1u {
        real_position = real_position + vec2<f32>(uniforms.center_coord) * uniforms.tile_width;
        // if border_tile > 0 {
        //     float cam_pos_u = cam_pos.x / tile_width - float(center_coord.x);
        //     float cam_pos_v = cam_pos.y / tile_width - float(center_coord.y);
        //     if (border_tile % 10 == 1) { // South
        //         vertex_alpha *= 1.0 - cam_pos_v;
        //     } else if (border_tile % 10 == 2) { // North
        //         vertex_alpha *= cam_pos_v;
        //     }
        //     if (border_tile / 10 == 1) { // West
        //         vertex_alpha *= 1.0 - cam_pos_u;
        //     } else if (border_tile / 10 == 2) { // East
        //         vertex_alpha *= cam_pos_u;
        //     }
        // }
    } else {
        real_position = position * uniforms.width_scale;
        real_position = real_position + floor(vec2<f32>(uniforms.center_coord) * uniforms.tile_width / uniforms.width_scale) * uniforms.width_scale; // always move by a multiple of width_scale
        // height = height - 1.0; // prevent clipping due to different resolution
    }

    out.mapped_height = 0.0;
    var local_z = vec3(0.0, 0.0, 1.0);
    if uniforms.surface_type == 1u {
        let DELTA = 0.001;

        let hmap_xrange = (2.0 * f32(uniforms.map_half_wh.x) + 1.0) * uniforms.tile_width * uniforms.height_map_scale.x;
        let hmap_yrange = (2.0 * f32(uniforms.map_half_wh.y) + 1.0) * uniforms.tile_width * uniforms.height_map_scale.y;
        let h_u = (real_position.x + f32(uniforms.map_half_wh.x) * uniforms.tile_width) / hmap_xrange;
        let h_v = (real_position.y + f32(uniforms.map_half_wh.y) * uniforms.tile_width) / hmap_yrange;
        out.mapped_height = textureSampleLevel(t_height_map, s_height_map, vec2(h_u, h_v), 0.0).r * uniforms.height_map_scale.z;
        height += out.mapped_height;
    }

    let opengl_to_wgpu = mat4x4(
        vec4(1.0, 0.0, 0.0, 0.0),
        vec4(0.0, 1.0, 0.0, 0.0),
        vec4(0.0, 0.0, 0.5, 0.0),
        vec4(0.0, 0.0, 0.5, 1.0),
    );

    out.clip_position = opengl_to_wgpu * uniforms.projection * uniforms.view * vec4(real_position, height, 1.0);
    out.tex_coords = real_position / uniforms.tile_width / 4.0;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var frag_color = vec4(0.0);
    if uniforms.use_clip == 1u && in.mapped_height < uniforms.clip_height {
        discard;
    }
    if uniforms.black_background == 1u {
        frag_color = vec4(0.0, 0.0, 0.0, in.vertex_alpha);
    } else {
        var color = vec3(textureSample(t_proxy_texture, s_proxy_texture, in.tex_coords).rgb);
        frag_color = vec4(color * uniforms.brightness, 1.0);
    }

    return frag_color;
}