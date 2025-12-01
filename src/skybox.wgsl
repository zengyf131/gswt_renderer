struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec3<f32>,
}

struct Uniforms {
    equirectangular: u32,

    view: mat4x4<f32>,
    projection: mat4x4<f32>,    
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(0)
var t_skybox: texture_cube<f32>;
@group(1) @binding(1)
var s_skybox: sampler;

@group(2) @binding(0)
var t_equi: texture_2d<f32>;
@group(2) @binding(1)
var s_equi: sampler;

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;

    var tex_coords = position;
    // Change back to Z foward, Y down, X right (X rot -90)
    tex_coords = vec3(tex_coords.x, -tex_coords.z, tex_coords.y);
    // Flip Y if using cubemap
    if uniforms.equirectangular == 0u {
        tex_coords.y = -tex_coords.y;
    }
    out.tex_coords = tex_coords;

    // Remove translation
    let view = mat4x4<f32>(
        vec4<f32>(uniforms.view[0].xyz, 0.0),
        vec4<f32>(uniforms.view[1].xyz, 0.0),
        vec4<f32>(uniforms.view[2].xyz, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    );

    let pos = uniforms.projection * view * vec4(position, 1.0);
    out.clip_position = pos.xyww; // Force depth to 1.0

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let frag_color = vec4(textureSampleLevel(t_skybox, s_skybox, in.tex_coords, 0.0).rgb, 1.0);
    return frag_color;
}

@vertex
fn vs_bake(
    @location(0) position: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;

    out.tex_coords = position;
    out.clip_position = uniforms.projection * uniforms.view * vec4(position, 1.0);

    return out;
}

@fragment
fn fs_bake(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = SampleSphericalMap(normalize(in.tex_coords));
    var color = textureSampleLevel(t_equi, s_equi, uv, 0.0).rgb;

    // gamma correction
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2)); 

    let frag_color = vec4(color, 1.0);
    return frag_color;
}

// https://learnopengl.com/PBR/IBL/Diffuse-irradiance
// const float pi = 3.1415926535897932384626433832795;
// const float half_pi = 1.5707963267948966;
fn SampleSphericalMap(dir: vec3<f32>) -> vec2<f32> {
    let phi = atan2(dir.z, dir.x); // Longitude
    let theta = asin(dir.y);    // Latitude
    // float u = (phi + pi) / (2.0 * pi);
    // float v = (theta + half_pi) / pi;
    let u = phi * 0.1591 + 0.5;
    let v = theta * 0.3183 + 0.5;
    return vec2(u, v);
}