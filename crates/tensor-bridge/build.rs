use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn write_fallback_bindings(out_path: &PathBuf) {
    let fallback_bindings = r#"
pub const FRAME_SAMPLES: usize = 512;
pub const SPECTROGRAM_BINS: usize = 256;
pub const SHM_RING_CAPACITY: usize = 64;
pub const SHM_MAGIC: u32 = 1297306994; /* 0x4D535452 */
pub const SHM_VERSION: u32 = 1;

pub const FLAG_INGESTED: u32 = 1;
pub const FLAG_DSP_DONE: u32 = 2;
pub const FLAG_ML_DONE: u32 = 4;
pub const FLAG_BROADCAST: u32 = 8;

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

extern "C" {
    pub fn dsp_init() -> ::std::os::raw::c_int;
    pub fn dsp_configure(config: *const DspConfig);
    pub fn dsp_process_frame(packet: *mut AudioFramePacket);
    pub fn dsp_destroy();
}
"#;
    fs::write(out_path, fallback_bindings).expect("Failed to write bindings");
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir.parent().unwrap().parent().unwrap();
    let dsp_dir = repo_root.join("dsp");
    let dsp_build_dir = dsp_dir.join("build");
    let include_dir = repo_root.join("include");
    let dsp_include_dir = dsp_dir.join("include");

    // Automatically build C DSP library if not present
    if !dsp_build_dir.join("libdsp_core.a").exists() {
        let status = Command::new("cmake")
            .args(["-B", "build", "-DCMAKE_BUILD_TYPE=Release"])
            .current_dir(&dsp_dir)
            .status()
            .expect("Failed to run cmake for dsp");
        assert!(status.success(), "CMake configuration failed");

        let build_status = Command::new("cmake")
            .args(["--build", "build"])
            .current_dir(&dsp_dir)
            .status()
            .expect("Failed to build dsp_core");
        assert!(build_status.success(), "CMake build failed");
    }

    println!("cargo:rustc-link-search=native={}", dsp_build_dir.display());
    println!("cargo:rustc-link-lib=static=dsp_core");
    println!("cargo:rustc-link-lib=m");

    // Search common Linux python3 shared object directories
    println!("cargo:rustc-link-search=native=/usr/lib/python3.14/config-3.14-x86_64-linux-gnu");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-search=native=/usr/local/lib");

    println!("cargo:rerun-if-changed={}", include_dir.join("ipc_protocol.h").display());
    println!("cargo:rerun-if-changed={}", dsp_include_dir.join("dsp_core.h").display());

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");

    let result = std::panic::catch_unwind(|| {
        bindgen::Builder::default()
            .header(include_dir.join("ipc_protocol.h").to_str().unwrap())
            .header(dsp_include_dir.join("dsp_core.h").to_str().unwrap())
            .clang_arg(format!("-I{}", include_dir.display()))
            .clang_arg(format!("-I{}", dsp_include_dir.display()))
            .derive_default(true)
            .derive_copy(true)
            .derive_debug(true)
            .generate()
    });

    match result {
        Ok(Ok(bindings)) => {
            bindings.write_to_file(&out_path).expect("Couldn't write bindings!");
        }
        _ => {
            write_fallback_bindings(&out_path);
        }
    }
}
