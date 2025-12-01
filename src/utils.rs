// Partially copied from https://github.com/BladeTransformerLLC/gauzilla

use std::collections::VecDeque;
use wasm_bindgen::prelude::*;

pub use cgmath::{
    Angle, EuclideanSpace, InnerSpace, Matrix, MetricSpace, One, Rotation, Rotation2, Rotation3,
    SquareMatrix, Transform, Transform2, Transform3, VectorSpace, Zero,
};
pub use cgmath::{
    Deg, Matrix2, Matrix3, Matrix4, Point2, Point3, Quaternion, Rad, Vector2, Vector3, Vector4,
    dot, frustum, ortho, perspective, vec2, vec3, vec4,
};
use half::f16;

pub type Vec3 = Vector3<f32>;
pub type Vec2 = Vector2<f32>;
pub type Mat3 = Matrix3<f32>;
pub type Mat4 = Matrix4<f32>;

pub type Degrees = Deg<f32>;
pub type Radians = Rad<f32>;

pub const fn degrees<T>(v: T) -> Deg<T> {
    cgmath::Deg(v)
}

pub const fn radians<T>(v: T) -> Rad<T> {
    cgmath::Rad(v)
}

#[wasm_bindgen(module = "/src/helper.js")]
extern "C" {
    pub fn get_time_milliseconds() -> f64;
}

#[macro_export]
macro_rules! log {
    ( $( $t:tt )* ) => {
        web_sys::console::log_1(&format!( $( $t )* ).into());
    }
}

/// Executes an async Future on the current thread
#[inline(always)]
pub fn execute_future<F: Future<Output = ()> + 'static>(f: F) {
    wasm_bindgen_futures::spawn_local(f);
}

/// Transmutes a slice
#[inline(always)]
pub fn transmute_slice<S, T>(slice: &[S]) -> &[T] {
    let ptr = slice.as_ptr() as *const T;
    let len = std::mem::size_of_val(slice) / std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// Transmutes a mutable slice
#[inline(always)]
pub fn transmute_slice_mut<S, T>(slice: &mut [S]) -> &mut [T] {
    let ptr = slice.as_mut_ptr() as *mut T;
    let len = std::mem::size_of_val(slice) / std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

/// Packs two f32s as two f16s combined together
#[inline(always)]
pub fn pack_half_2x16(x: f32, y: f32) -> u32 {
    let x_half = f16::from_f32(x);
    let y_half = f16::from_f32(y);
    let result = u32::from(x_half.to_bits()) | (u32::from(y_half.to_bits()) << 16);
    result // & 0xFFFFFFFF
}

/// Incremental Moving Average
pub struct IncrementalMA {
    v: VecDeque<f64>,
    v_sum: f64,
    v_sum_sq: f64,
    v_avg: f64,
    v_stddev: f64,
}
impl IncrementalMA {
    pub fn new(window_size: usize) -> Self {
        IncrementalMA {
            v: VecDeque::with_capacity(window_size),
            v_sum: 0_f64,
            v_sum_sq: 0_f64,
            v_avg: 0_f64,
            v_stddev: 0_f64,
        }
    }

    pub fn add(&mut self, value: f64) -> (f64, f64) {
        if self.v.len() == self.v.capacity() {
            let old = self.v.pop_front().unwrap();
            self.v_sum -= old;
            self.v_sum_sq -= old * old;
        }

        self.v.push_back(value);
        self.v_sum += value;
        self.v_sum_sq += value * value;

        let len_f = self.v.len() as f64;
        let avg = self.v_sum / len_f;
        let variance = (self.v_sum_sq / len_f) - avg * avg;
        let stddev = variance.sqrt();

        self.v_avg = avg;
        self.v_stddev = stddev;
        (avg, stddev)
    }

    pub fn calc(&self) -> (f64, f64) {
        if self.v.is_empty() {
            (0_f64, 0_f64)
        } else {
            (self.v_avg, self.v_stddev)
        }
    }

    pub fn clear(&mut self) {
        self.v.clear();
        self.v_sum = 0_f64;
        self.v_sum_sq = 0_f64;
        self.v_avg = 0_f64;
        self.v_stddev = 0_f64;
    }
}

pub struct Timer {
    start_time: Option<f64>,
    accumulated: f64,
    paused: bool,
}
impl Timer {
    pub fn new() -> Self {
        Self {
            start_time: None,
            accumulated: 0.0,
            paused: true,
        }
    }

    pub fn start(&mut self) {
        if self.paused {
            self.start_time = Some(get_time_milliseconds());
            self.paused = false;
        }
    }

    pub fn pause(&mut self) {
        if !self.paused {
            if let Some(start) = self.start_time {
                self.accumulated += get_time_milliseconds() - start;
            }
            self.start_time = None;
            self.paused = true;
        }
    }

    pub fn reset(&mut self) {
        self.start_time = None;
        self.accumulated = 0.0;
        self.paused = true;
    }

    pub fn elapsed(&self) -> f64 {
        if self.paused {
            self.accumulated
        } else {
            if let Some(start) = self.start_time {
                self.accumulated + (get_time_milliseconds() - start)
            } else {
                self.accumulated
            }
        }
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }
}

pub fn parse_num<T>(s: &String, num: &mut T, err: &mut Option<String>)
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    let parsed: Result<T, _> = s.parse();
    match parsed {
        Ok(count) => *num = count,
        Err(e) => *err = Some(e.to_string()),
    }
}
