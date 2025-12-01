use winit::{
    dpi::PhysicalSize,
};

use crate::utils::*;

// From three_d
pub struct Camera {
    viewport: PhysicalSize<u32>,
    position: Vec3,
    target: Vec3,
    up: Vec3,
    fovy: Radians,
    z_near: f32,
    z_far: f32,

    view: Mat4,
    projection: Mat4,
}
impl Camera {
    pub fn new(viewport: PhysicalSize<u32>) -> Self {
        Self {
            viewport,
            position: Vec3::zero(),
            target: Vec3::zero(),
            up: Vec3::zero(),
            fovy: radians(0.0),
            z_near: 0.0,
            z_far: 0.0,

            view: Mat4::zero(),
            projection: Mat4::zero(),
        }
    }

    pub fn new_perspective(
        viewport: PhysicalSize<u32>,
        position: Vec3,
        target: Vec3,
        up: Vec3,
        fovy: impl Into<Radians>,
        z_near: f32,
        z_far: f32,
    ) -> Self {
        let mut cam = Self::new(viewport);

        cam.set_view(position, target, up);
        cam.set_perspective_projection(fovy, z_near, z_far);
        cam
    }

    pub fn viewport(&self) -> &PhysicalSize<u32> {
        &self.viewport
    }

    pub fn position(&self) -> &Vec3 {
        &self.position
    }

    pub fn target(&self) -> &Vec3 {
        &self.target
    }

    pub fn up(&self) -> &Vec3 {
        &self.up
    }

    pub fn fovy(&self) -> Radians {
        self.fovy
    }

    pub fn view_direction(&self) -> Vec3 {
        (self.target - self.position).normalize()
    }

    pub fn right_direction(&self) -> Vec3 {
        self.view_direction().cross(self.up)
    }

    pub fn view(&self) -> &Mat4 {
        &self.view
    }

    pub fn projection(&self) -> &Mat4 {
        &self.projection
    }

    pub fn view_proj(&self) -> Mat4 {
        self.projection * self.view
    }

    pub fn set_view(&mut self, position: Vec3, target: Vec3, up: Vec3) {
        self.position = position;
        self.target = target;
        self.up = up;
        self.view = Mat4::look_at_rh(
            Point3::from_vec(self.position),
            Point3::from_vec(self.target),
            self.up,
        );
    }

    pub fn set_perspective_projection(
        &mut self,
        fovy: impl Into<Radians>,
        z_near: f32,
        z_far: f32,
    ) {
        assert!(
            z_near >= 0.0 || z_near < z_far,
            "Wrong perspective camera parameters"
        );
        self.fovy = fovy.into();
        self.z_near = z_near;
        self.z_far = z_far;

        self.projection = cgmath::perspective(
            self.fovy,
            self.viewport.width as f32 / self.viewport.height as f32,
            z_near,
            z_far,
        );
    }

    pub fn set_viewport(&mut self, width: u32, height: u32) {
        self.viewport = PhysicalSize { width, height };
    }

    pub fn translate(&mut self, change: &Vec3) {
        self.set_view(self.position + change, self.target + change, self.up);
    }

    pub fn pitch(&mut self, delta: impl Into<Radians>) {
        let target = (self.view.invert().unwrap()
            * Mat4::from_angle_x(delta)
            * self.view
            * self.target.extend(1.0))
        .truncate();
        if (target - self.position).normalize().dot(self.up).abs() < 0.999 {
            self.set_view(self.position, target, self.up);
        }
    }

    pub fn yaw(&mut self, delta: impl Into<Radians>) {
        let target = (self.view.invert().unwrap()
            * Mat4::from_angle_y(delta)
            * self.view
            * self.target.extend(1.0))
        .truncate();
        self.set_view(self.position, target, self.up);
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniforms {
    pub projection: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub focal: [f32; 2],
    pub viewport: [f32; 2],
    pub htan_fov: [f32; 4],
    pub cam_pos: [f32; 4],
}
impl CameraUniforms {
    pub fn from_camera(cam: &Camera) -> Self {
        let view_matrix: &Mat4 = cam.view();
        let projection_matrix: &Mat4 = cam.projection();
        let w = cam.viewport().width as f32;
        let h = cam.viewport().height as f32;
        let cam_pos = cam.position();
        let fx = 0.5 * projection_matrix[0][0] * w;
        let fy = -0.5 * projection_matrix[1][1] * h;
        let htany = (cam.fovy() / 2.0).tan() as f32;
        let htanx = (htany / h) * w;

        Self {
            projection: (*projection_matrix).into(),
            view: (*view_matrix).into(),
            focal: [fx.abs(), fy.abs()],
            viewport: [w, h],
            htan_fov: [htanx, htany, 0.0, 0.0],
            cam_pos: [cam_pos.x, cam_pos.y, cam_pos.z, 0.0],
        }
    }
}
