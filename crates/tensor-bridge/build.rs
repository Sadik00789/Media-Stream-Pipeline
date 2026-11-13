use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn write_fallback_bindings(out_path: &PathBuf) {
    let fallback_bindings = r#"
pub const FRAME_SAMPLES: u32 = 512;
pub const SPECTROGRAM_BINS: u32 = 256;
pub const SHM_RING_CAPACITY: u32 = 64;
pub const SHM_MAGIC: u32 = 1297306994; /* 0x4D535452 */
pub const SHM_VERSION: u32 = 1;

pub const VIDEO_WIDTH: u32 = 640;
pub const VIDEO_HEIGHT: u32 = 480;
pub const VIDEO_CHANNELS: u32 = 3;
pub const VIDEO_FRAME_SIZE: u32 = 921600;
pub const VIDEO_RING_CAPACITY: u32 = 8;
pub const VIDEO_SHM_MAGIC: u32 = 1447380037; /* 0x56494445 */
pub const VIDEO_SHM_VERSION: u32 = 1;

pub const VIDEO_FORMAT_RGB24: u32 = 0;
pub const VIDEO_FORMAT_YUV420P: u32 = 1;
pub const MAX_DETECTIONS: u32 = 16;

pub const FLAG_INGESTED: u32 = 1;
pub const FLAG_DSP_DONE: u32 = 2;
pub const FLAG_ML_DONE: u32 = 4;
pub const FLAG_BROADCAST: u32 = 8;

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct DetectionBox {
    pub ymin: f32,
    pub xmin: f32,
    pub ymax: f32,
    pub xmax: f32,
    pub confidence: f32,
    pub class_id: u32,
}

impl Default for DetectionBox {
    fn default() -> Self {
        unsafe { ::std::mem::zeroed() }
    }
}

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct AudioFramePacket {
    pub sequence_number: u64,
    pub timestamp_ns: u64,
    pub sample_rate: u32,
    pub channels: u32,
    pub sample_count: u32,
    pub vad_active: u32,
    pub vad_confidence: f32,
    pub ml_confidence: f32,
    pub flags: u32,
    pub reserved: u32,
    pub pcm_data: [f32; 512usize],
    pub spectrogram: [f32; 256usize],
    pub ml_transcript: [::std::os::raw::c_char; 128usize],
}

impl Default for AudioFramePacket {
    fn default() -> Self {
        unsafe { ::std::mem::zeroed() }
    }
}

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct VideoFramePacket {
    pub sequence_number: u64,
    pub timestamp_ns: u64,
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub detection_count: u32,
    pub flags: u32,
    pub reserved: u32,
    pub detections: [DetectionBox; 16usize],
    pub pixel_data: [u8; 921600usize],
}

impl Default for VideoFramePacket {
    fn default() -> Self {
        unsafe { ::std::mem::zeroed() }
    }
}

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct SharedMemoryRingBuffer {
    pub magic: u32,
    pub version: u32,
    pub buffer_capacity: u32,
    pub packet_size: u32,
    pub write_head: u64,
    pub read_head: u64,
    pub dropped_frames: u64,
    pub overrun_count: u64,
    pub packets: [AudioFramePacket; 64usize],
}

impl Default for SharedMemoryRingBuffer {
    fn default() -> Self {
        unsafe { ::std::mem::zeroed() }
    }
}

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct SharedMemoryVideoRingBuffer {
    pub magic: u32,
    pub version: u32,
    pub buffer_capacity: u32,
    pub packet_size: u32,
    pub write_head: u64,
    pub read_head: u64,
    pub dropped_frames: u64,
    pub overrun_count: u64,
    pub packets: [VideoFramePacket; 8usize],
}

impl Default for SharedMemoryVideoRingBuffer {
    fn default() -> Self {
        unsafe { ::std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DspWindowType {
    DSP_WINDOW_RECTANGULAR = 0,
    DSP_WINDOW_HAMMING = 1,
    DSP_WINDOW_HANNING = 2,
    DSP_WINDOW_BLACKMAN = 3,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DspConfig {
    pub vad_threshold: f32,
    pub window_type: DspWindowType,
    pub sample_rate: u32,
}

unsafe extern "C" {
    pub fn dsp_init() -> ::std::os::raw::c_int;
    pub fn dsp_configure(config: *const DspConfig);
    pub fn dsp_process_frame(packet: *mut AudioFramePacket);
    pub fn dsp_destroy();
    pub fn simd_rgb24_to_planar_f32(rgb_in: *const u8, planar_out: *mut f32, width: ::std::os::raw::c_int, height: ::std::os::raw::c_int);
    pub fn simd_fast_box_nms(boxes_in: *const DetectionBox, num_boxes: u32, iou_threshold: f32, boxes_out: *mut DetectionBox, max_out: u32) -> u32;
}
"#;
    fs::write(out_path, fallback_bindings).expect("Failed to write bindings");
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir.parent().unwrap().parent().unwrap();
    let dsp_dir = repo_root.join("dsp");
    let dsp_build_dir = dsp_dir.join("build");
    let dsp_release_dir = dsp_build_dir.join("Release");
    let include_dir = repo_root.join("include");
    let dsp_include_dir = dsp_dir.join("include");

    let has_dsp_lib = dsp_build_dir.join("libdsp_core.a").exists()
        || dsp_build_dir.join("dsp_core.lib").exists()
        || dsp_release_dir.join("dsp_core.lib").exists();

    // Automatically build C DSP library if not present
    if !has_dsp_lib {
        let status = Command::new("cmake")
            .args(["-B", "build", "-DCMAKE_BUILD_TYPE=Release"])
            .current_dir(&dsp_dir)
            .status()
            .expect("Failed to run cmake for dsp");
        assert!(status.success(), "CMake configuration failed");

        let build_status = Command::new("cmake")
            .args(["--build", "build", "--config", "Release"])
            .current_dir(&dsp_dir)
            .status()
            .expect("Failed to build dsp_core");
        assert!(build_status.success(), "CMake build failed");
    }

    if dsp_release_dir.exists() {
        println!(
            "cargo:rustc-link-search=native={}",
            dsp_release_dir.display()
        );
    }
    println!("cargo:rustc-link-search=native={}", dsp_build_dir.display());
    println!("cargo:rustc-link-lib=static=dsp_core");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        println!("cargo:rustc-link-lib=m");
        // Search common Linux python3 shared object directories
        println!("cargo:rustc-link-search=native=/usr/lib/python3.14/config-3.14-x86_64-linux-gnu");
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-search=native=/usr/local/lib");
    } else if let Ok(output) = Command::new("python")
        .args([
            "-c",
            "import sys, os; print(os.path.join(sys.base_prefix, 'libs'))",
        ])
        .output()
        && output.status.success()
    {
        let py_libs = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !py_libs.is_empty() {
            println!("cargo:rustc-link-search=native={}", py_libs);
        }
    }

    println!(
        "cargo:rerun-if-changed={}",
        include_dir.join("ipc_protocol.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        dsp_include_dir.join("dsp_core.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        dsp_include_dir.join("simd_vision.h").display()
    );

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");

    let result = std::panic::catch_unwind(|| {
        bindgen::Builder::default()
            .header(include_dir.join("ipc_protocol.h").to_str().unwrap())
            .header(dsp_include_dir.join("dsp_core.h").to_str().unwrap())
            .header(dsp_include_dir.join("simd_vision.h").to_str().unwrap())
            .clang_arg(format!("-I{}", include_dir.display()))
            .clang_arg(format!("-I{}", dsp_include_dir.display()))
            .rustified_enum("DspWindowType")
            .derive_default(true)
            .derive_copy(true)
            .derive_debug(true)
            .generate()
    });

    match result {
        Ok(Ok(bindings)) => {
            let code = bindings.to_string();
            let patched = code.replace("extern \"C\" {", "unsafe extern \"C\" {");
            fs::write(&out_path, patched).expect("Couldn't write bindings!");
        }
        _ => {
            write_fallback_bindings(&out_path);
        }
    }
}
