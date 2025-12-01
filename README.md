# GSWT Renderer

This is the official codebase for SIGGRAPH Asia 2025 paper: _GSWT: Gaussian Splatting Wang Tiles_ ([Project page](https://yunfan.zone/gswt_webpage/)). It only contains the GWST renderer. For the GSWT constructor part, please visit the other repository: TBD.

## Introduction

This renderer is a _Gaussian Splatting Wang Tiles_ renderer that runs on the web, using [wgpu](https://github.com/gfx-rs/wgpu) and WebGPU as the rendering backend. It contains specific functionalities and optimizations tailored to GSWT, including _procedural tiling_, _selective merging_ and _LOD blending_. It takes a set of tiles produced by the GSWT constructor as input, and generates an infinitely expanding 3DGS terrian on-the-fly. The user can interact with the renderer and navigate the terrian in real time.

The original renderer discussed in the paper is developed with WebGL. This renderer is an improved version of the original one and might perform differently in certain cases (usually faster).

## Getting Started

The renderer is available online: TBD. It requires a web browser that supports WebGPU (e.g., latest Chrome).

A set of official datasets can be found here: TBD

To start the renderer:
1. Find a dataset and upload the zip file containing the set of tiles. After a moment of preprocessing (usually a few seconds), the config menu should show up.
2. Play with the config and click "Confirm". The renderer will switch to rendering stage and show the rendering menu.
3. Navigate the scene using **WASD** and hold **Space** to sprint. Use **IJKL** to look around.
4. There are some hotkeys to hide/unhide menus: **M** for main rendering menu and **P** for performance menu.
5. Click "Reconfig" to go back to config menu.
6. Reload the webpage to switch to another scene.

There are quite a few config options in this renderer. The default config is used in most experiments in the paper (usually with a skybox texture and a proxy texture). Below are some commonly used options: 

TBD

## Building locally

The renderer is written in Rust and targets WebAssembly (WASM). It is currently built with `rustc 1.92.0-nightly` and `wasm-pack 0.13.1`. They are required for building this project.

To build the project, run the following command:

```
wasm-pack build --target web
```

Then the package for release should be available in `./pkg`. It can be deployed on a web server together with `./index.html`. 

To test it locally, [sfz](https://github.com/weihanglo/sfz) is recommended. Run the following command to start a local server:

```
sfz -r --coi
```

## Credits

This renderer is derived from and heavily inspired by [Gauzilla](https://github.com/BladeTransformerLLC/gauzilla), under its [MIT License](https://github.com/BladeTransformerLLC/gauzilla?tab=MIT-1-ov-file). The egui rendering logic is derived from [this repository](https://github.com/kaphula/winit-egui-wgpu-template), under its [MIT License](https://github.com/kaphula/winit-egui-wgpu-template?tab=MIT-1-ov-file).

## BibTex

```
@inproceedings{Zeng:2025:gswt,
  author = {Zeng, Yunfan and Ma, Li and Sander, Pedro V.},
  title = {GSWT: Gaussian Splatting Wang Tiles},
  year = {2025},
  publisher = {Association for Computing Machinery},
  booktitle = {SIGGRAPH Asia 2025 Conference Papers},
  location = {Hong Kong, China},
  series = {SA '25}
}
```