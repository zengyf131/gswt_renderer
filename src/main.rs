#[cfg(target_arch = "wasm32")]
use gswt::run_web;

#[cfg(not(target_arch = "wasm32"))]
use gswt::run;

fn main() {
    #[cfg(target_arch = "wasm32")]
    run_web().unwrap();
    #[cfg(not(target_arch = "wasm32"))]
    run().unwrap();
}
