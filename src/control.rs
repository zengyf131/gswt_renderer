use serde;
use serde_json;
use std::sync::mpsc;
use winit::keyboard::KeyCode;

use crate::camera::Camera;
use crate::utils::*;

#[derive(PartialEq)]
pub enum CameraControl {
    KeyboardFly,
    FlyPath,
}

pub struct KeyboardFlyControl {
    max_speed: f32,
    max_speed_sprint: f32,
    acc: f32,
    acc_sprint: f32,
    rot_speed: f32,
    speed: [f32; 6],
    moving: [bool; 6],
    rotating: [bool; 4],
    sprinting: bool,
}
impl KeyboardFlyControl {
    /// Creates a new orbit control with the given target and minimum and maximum distance to the target.
    pub fn new() -> Self {
        Self {
            // max_speed: 0.02,
            // max_speed_sprint: 0.1,
            // acc: 0.0002,
            // acc_sprint: 0.001,
            // rot_speed: 0.01,
            max_speed: 0.002,
            max_speed_sprint: 0.005,
            acc: 0.00001,
            acc_sprint: 0.0001,
            rot_speed: 0.001,
            speed: [0.0; 6],
            moving: [false; 6],
            rotating: [false; 4],
            sprinting: false,
        }
    }

    fn handle_speed(&mut self, index: usize, moving: bool, frame_time: f32) {
        let frame_time = frame_time.max(10.0); // set min time
        let mut new_speed = self.speed[index];
        if moving {
            if self.sprinting {
                if new_speed < self.max_speed_sprint {
                    new_speed += self.acc_sprint * frame_time;
                } else {
                    new_speed = self.max_speed_sprint;
                }
            } else {
                if new_speed < self.max_speed {
                    new_speed += self.acc * frame_time;
                } else {
                    new_speed = self.max_speed;
                }
            }
        } else {
            if new_speed > self.max_speed {
                new_speed -= self.acc_sprint * frame_time;
            } else if new_speed > 0.0 {
                new_speed -= self.acc * frame_time;
            } else {
                new_speed = 0.0;
            }
        }

        self.speed[index] = new_speed;
    }

    pub fn rotate_around(&self, cam: &mut Camera, center: Vec3, d_forward: f32, d_right: f32) {
        let up = (cam.position() - center).normalize();
        let right = cam.view_direction().cross(up).normalize();
        let dir = up.cross(right).normalize();

        let new_up = (cam.position() - center + dir * d_forward + right * d_right).normalize();
        let rotation = rotation_matrix_from_dir_to_dir(up, new_up);
        let new_position = (rotation * (cam.position() - center).extend(1.0)).truncate() + center;
        let new_target = (rotation * (cam.target() - center).extend(1.0)).truncate() + center;
        cam.set_view(new_position, new_target, new_up);
    }

    /// Handles the events. Must be called each frame.
    pub fn handle_key(&mut self, key: KeyCode, pressed: bool) {
        if pressed {
            match key {
                KeyCode::KeyW => {
                    self.moving[0] = true;
                } // Forward
                KeyCode::KeyS => {
                    self.moving[1] = true;
                } // Backward
                KeyCode::KeyA => {
                    self.moving[2] = true;
                } // Left
                KeyCode::KeyD => {
                    self.moving[3] = true;
                } // Right
                KeyCode::KeyR => {
                    self.moving[4] = true;
                } // Up
                KeyCode::KeyF => {
                    self.moving[5] = true;
                } // Down
                KeyCode::KeyI => {
                    self.rotating[0] = true;
                } // Look up
                KeyCode::KeyK => {
                    self.rotating[1] = true;
                } // Look down
                KeyCode::KeyJ => {
                    self.rotating[2] = true;
                } // Look left
                KeyCode::KeyL => {
                    self.rotating[3] = true;
                } // Look right
                KeyCode::Space => {
                    self.sprinting = true;
                }
                _ => {}
            }
        } else {
            match key {
                KeyCode::KeyW => {
                    self.moving[0] = false;
                }
                KeyCode::KeyS => {
                    self.moving[1] = false;
                }
                KeyCode::KeyA => {
                    self.moving[2] = false;
                }
                KeyCode::KeyD => {
                    self.moving[3] = false;
                }
                KeyCode::KeyR => {
                    self.moving[4] = false;
                }
                KeyCode::KeyF => {
                    self.moving[5] = false;
                }
                KeyCode::KeyI => {
                    self.rotating[0] = false;
                }
                KeyCode::KeyK => {
                    self.rotating[1] = false;
                }
                KeyCode::KeyJ => {
                    self.rotating[2] = false;
                }
                KeyCode::KeyL => {
                    self.rotating[3] = false;
                }
                KeyCode::Space => {
                    self.sprinting = false;
                }
                _ => {}
            }
        }
    }

    pub fn update(&mut self, camera: &mut Camera, frame_time: f32, lock_center: bool) -> bool {
        for i in 0..6 {
            if self.moving[i] {
                self.handle_speed(i, true, frame_time);
            } else {
                self.handle_speed(i, false, frame_time);
            }
        }

        /*
        if self.speed[0] > 0.0 {    // Forward
            let change = camera.view_direction() * self.speed[0];
            camera.translate(&change);

        }
        if self.speed[1] > 0.0 {    // Backward
            let change = -camera.view_direction() * self.speed[1];
            camera.translate(&change);
        }
        if self.speed[2] > 0.0 {    // Left
            let change = -camera.right_direction() * self.speed[2];
            camera.translate(&change);
        }
        if self.speed[3] > 0.0 {    // Right
            let change = camera.right_direction() * self.speed[3];
            camera.translate(&change);
        }
        if self.speed[4] > 0.0 {    // Up
            let right = camera.right_direction();
            let up = right.cross(camera.view_direction());
            let change = up * self.speed[4];
            camera.translate(&change);
        }
        if self.speed[5] > 0.0 {    // Down
            let right = camera.right_direction();
            let up = right.cross(camera.view_direction());
            let change = -up * self.speed[5];
            camera.translate(&change);
        }
        */

        // Lock Z
        if self.speed[0] > 0.0 {
            // Forward
            let delta = self.speed[0] * frame_time;
            if lock_center {
                self.rotate_around(camera, vec3(0.0, 0.0, 0.0), delta, 0.0);
            } else {
                let mut change = camera.view_direction();
                change.z = 0.0;
                change = change.normalize();
                change *= delta;
                camera.translate(&change);
            }
        }
        if self.speed[1] > 0.0 {
            // Backward
            let delta = -self.speed[1] * frame_time;
            if lock_center {
                self.rotate_around(camera, vec3(0.0, 0.0, 0.0), delta, 0.0);
            } else {
                let mut change = camera.view_direction();
                change.z = 0.0;
                change = change.normalize();
                change *= delta;
                camera.translate(&change);
            }
        }
        if self.speed[2] > 0.0 {
            // Left
            let delta = -self.speed[2] * frame_time;
            if lock_center {
                self.rotate_around(camera, vec3(0.0, 0.0, 0.0), 0.0, delta);
            } else {
                let mut change = camera.right_direction();
                change.z = 0.0;
                change = change.normalize();
                change *= delta;
                camera.translate(&change);
            }
        }
        if self.speed[3] > 0.0 {
            // Right
            let delta = self.speed[3] * frame_time;
            if lock_center {
                self.rotate_around(camera, vec3(0.0, 0.0, 0.0), 0.0, delta);
            } else {
                let mut change = camera.right_direction();
                change.z = 0.0;
                change = change.normalize();
                change *= delta;
                camera.translate(&change);
            }
        }
        if self.speed[4] > 0.0 {
            // Up
            let change = camera.up() * self.speed[4] * frame_time;
            camera.translate(&change);
        }
        if self.speed[5] > 0.0 {
            // Down
            let change = camera.up() * -self.speed[5] * frame_time;
            camera.translate(&change);
        }

        if self.rotating[0] {
            // Look up
            camera.pitch(radians(self.rot_speed * frame_time));
        }
        if self.rotating[1] {
            // Look down
            camera.pitch(radians(-self.rot_speed * frame_time));
        }
        if self.rotating[2] {
            // Look left
            camera.yaw(radians(self.rot_speed * frame_time));
        }
        if self.rotating[3] {
            // Look right
            camera.yaw(radians(-self.rot_speed * frame_time));
        }

        return true;
    }
}

#[derive(Clone)]
pub struct FlyPathFrame {
    pub timestamp: f32,
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,

    pub timestamp_s: String,
    pub position_s: Vector3<String>,
    pub target_s: Vector3<String>,
    pub up_s: Vector3<String>,
}
impl FlyPathFrame {
    pub fn new() -> Self {
        Self {
            timestamp: 0.0,
            position: Vec3::zero(),
            target: Vec3::zero(),
            up: Vec3::unit_z(),
            timestamp_s: 0.to_string(),
            position_s: Vector3 {
                x: 0.to_string(),
                y: 0.to_string(),
                z: 0.to_string(),
            },
            target_s: Vector3 {
                x: 0.to_string(),
                y: 0.to_string(),
                z: 0.to_string(),
            },
            up_s: Vector3 {
                x: 0.to_string(),
                y: 0.to_string(),
                z: 0.to_string(),
            },
        }
    }

    pub fn from_json(frame_json: &FlyPathFrameJSON) -> Self {
        let mut this_frame = Self::new();
        this_frame.timestamp = frame_json.timestamp;
        this_frame.position = vec3(
            frame_json.position_x,
            frame_json.position_y,
            frame_json.position_z,
        );
        this_frame.target = vec3(
            frame_json.target_x,
            frame_json.target_y,
            frame_json.target_z,
        );
        this_frame.update_string();

        this_frame
    }

    pub fn from_camera(camera: &Camera) -> Self {
        let mut frame = Self::new();
        frame.position = *camera.position();
        frame.target = *camera.target();
        frame.update_string();

        frame
    }

    pub fn parse_string(&mut self) -> Option<String> {
        let mut err: Option<String> = None;
        parse_num(&self.timestamp_s, &mut self.timestamp, &mut err);
        parse_num(&self.position_s.x, &mut self.position.x, &mut err);
        parse_num(&self.position_s.y, &mut self.position.y, &mut err);
        parse_num(&self.position_s.z, &mut self.position.z, &mut err);
        parse_num(&self.target_s.x, &mut self.target.x, &mut err);
        parse_num(&self.target_s.y, &mut self.target.y, &mut err);
        parse_num(&self.target_s.z, &mut self.target.z, &mut err);

        err
    }

    pub fn update_string(&mut self) {
        self.timestamp_s = self.timestamp.to_string();
        self.position_s.x = self.position.x.to_string();
        self.position_s.y = self.position.y.to_string();
        self.position_s.z = self.position.z.to_string();
        self.target_s.x = self.target.x.to_string();
        self.target_s.y = self.target.y.to_string();
        self.target_s.z = self.target.z.to_string();
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct FlyPathFrameJSON {
    pub timestamp: f32,
    pub position_x: f32,
    pub position_y: f32,
    pub position_z: f32,
    pub target_x: f32,
    pub target_y: f32,
    pub target_z: f32,
}
impl FlyPathFrameJSON {
    pub fn from_frame(frame: &FlyPathFrame) -> Self {
        Self {
            timestamp: frame.timestamp,
            position_x: frame.position.x,
            position_y: frame.position.y,
            position_z: frame.position.z,
            target_x: frame.target.x,
            target_y: frame.target.y,
            target_z: frame.target.z,
        }
    }
}

pub struct FlyPathControl {
    pub keyframes: Vec<FlyPathFrame>,
    pub timer: Timer,
    pub ready: bool,
    pub finished: bool,
    cur_keyframe_index: usize,
    refresh: bool,
}
impl FlyPathControl {
    /// Creates a new orbit control with the given target and minimum and maximum distance to the target.
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
            timer: Timer::new(),
            ready: false,
            finished: false,
            cur_keyframe_index: 0,
            refresh: false,
        }
    }

    pub fn reset_path(&mut self) -> Option<String> {
        let mut err: Option<String> = None;
        for keyframe in &mut self.keyframes {
            let this_err = keyframe.parse_string();
            if this_err.is_some() {
                err = this_err;
            }
        }
        self.timer.reset();
        self.cur_keyframe_index = 0;
        if self.keyframes.len() >= 2 && err.is_none() {
            self.ready = true;
            self.refresh = true;
        } else {
            self.ready = false;
            self.refresh = false;
        }
        self.finished = false;

        err
    }

    pub fn start_path(&mut self) {
        self.timer.start();
    }

    pub fn pause_path(&mut self) {
        self.timer.pause();
    }

    /// Handles the events. Must be called each frame.
    pub fn handle_events(&mut self, camera: &mut Camera) -> bool {
        if !self.ready {
            return false;
        }
        if self.refresh {
            self.refresh = false;
        } else if self.timer.is_paused() {
            return false;
        }

        let ela_time = self.timer.elapsed() as f32 / 1000.0;
        if ela_time >= self.keyframes.last().unwrap().timestamp {
            self.pause_path();
            self.finished = true;
            return false;
        }

        let mut cur_frame = FlyPathFrame::new();
        cur_frame.timestamp = ela_time;
        if ela_time >= self.keyframes[self.cur_keyframe_index + 1].timestamp {
            self.cur_keyframe_index += 1;
        }
        let fi = self.cur_keyframe_index;
        let t = (ela_time - self.keyframes[fi].timestamp)
            / (self.keyframes[fi + 1].timestamp - self.keyframes[fi].timestamp);
        let t2 = t * t;
        let t3 = t2 * t;

        // Interpolate position (catmull_rom)
        let p0 = if fi == 0 {
            // Extrapolate p0 = p1 - (p2 - p1)
            self.keyframes[0].position * 2.0 - self.keyframes[1].position
        } else {
            self.keyframes[fi - 1].position
        };
        let p1 = self.keyframes[fi].position;
        let p2 = self.keyframes[fi + 1].position;
        let p3 = if fi + 2 >= self.keyframes.len() {
            // Extrapolate p3 = p2 + (p2 - p1)
            self.keyframes[fi + 1].position * 2.0 - self.keyframes[fi].position
        } else {
            self.keyframes[fi + 2].position
        };
        cur_frame.position = 0.5
            * ((2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);

        // Interpolate target (catmull_rom)
        let p0 = if fi == 0 {
            // Extrapolate p0 = p1 - (p2 - p1)
            self.keyframes[0].target * 2.0 - self.keyframes[1].target
        } else {
            self.keyframes[fi - 1].target
        };
        let p1 = self.keyframes[fi].target;
        let p2 = self.keyframes[fi + 1].target;
        let p3 = if fi + 2 >= self.keyframes.len() {
            // Extrapolate p3 = p2 + (p2 - p1)
            self.keyframes[fi + 1].target * 2.0 - self.keyframes[fi].target
        } else {
            self.keyframes[fi + 2].target
        };
        cur_frame.target = 0.5
            * ((2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);

        camera.set_view(cur_frame.position, cur_frame.target, cur_frame.up);

        return true;
    }

    // Ref: https://github.com/PolyMeilex/rfd/blob/master/examples/async.rs
    pub fn upload() -> mpsc::Receiver<FlyPathControl> {
        let (tx, rx) = mpsc::channel();
        let task = rfd::AsyncFileDialog::new()
            .add_filter("Fly Path", &["json"])
            .pick_file();
        execute_future(async move {
            let file = task.await;
            if let Some(file) = file {
                let raw_data = file.read().await;
                let json_data: Vec<FlyPathFrameJSON> =
                    serde_json::from_slice(raw_data.as_slice()).unwrap();
                let mut new_control = Self::new();
                for frame_json in json_data {
                    new_control
                        .keyframes
                        .push(FlyPathFrame::from_json(&frame_json));
                }
                tx.send(new_control)
                    .expect("Error sending fly path json to main thread.");
            }
        });

        rx
    }

    pub fn download(keyframes: &Vec<FlyPathFrame>) {
        let mut frames_json: Vec<FlyPathFrameJSON> = Vec::with_capacity(keyframes.len());
        for frame in keyframes {
            frames_json.push(FlyPathFrameJSON::from_frame(frame));
        }
        let raw: String = serde_json::to_string_pretty(&frames_json).unwrap();

        let task = rfd::AsyncFileDialog::new()
            .set_file_name("fly_path.json")
            .save_file();
        execute_future(async move {
            let file = task.await;
            if let Some(file) = file {
                file.write(raw.as_bytes())
                    .await
                    .expect("Error saving fly path json.");
            }
        });
    }
}

// From three_d
pub fn rotation_matrix_from_dir_to_dir(source_dir: Vec3, target_dir: Vec3) -> Mat4 {
    Mat4::from(Mat3::from(cgmath::Basis3::between_vectors(
        source_dir, target_dir,
    )))
}
