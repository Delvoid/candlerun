use anyhow::Result;
use candle_core::{Device, Tensor};
use log::debug;
use rand::Rng;
use serde::Serialize;
use std::{fmt, time::Instant};

const WARMUP_ITERATIONS: usize = 5;
pub(crate) trait BenchDevice {
    fn sync(&self) -> Result<()>;
    fn name(&self) -> DeviceName;
}

#[derive(Serialize)]
pub enum DeviceName {
    CPU,
    GPU,
}

impl fmt::Display for DeviceName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            DeviceName::CPU => write!(f, "CPU"),
            DeviceName::GPU => write!(f, "GPU"),
        }
    }
}

impl BenchDevice for Device {
    fn sync(&self) -> Result<()> {
        match self {
            Device::Cpu => Ok(()),
            Device::Cuda(device) => {
                #[cfg(feature = "cuda")]
                return Ok(device.synchronize()?);
                #[cfg(not(feature = "cuda"))]
                panic!("Cuda device without cuda feature enabled: {:?}", device)
            }
            Device::Metal(device) => {
                #[cfg(feature = "metal")]
                return Ok(device.wait_until_completed()?);
                #[cfg(not(feature = "metal"))]
                panic!("Metal device without metal feature enabled: {:?}", device)
            }
        }
    }

    fn name(&self) -> DeviceName {
        match self {
            Device::Cpu => DeviceName::CPU,
            Device::Cuda(_) | Device::Metal(_) => DeviceName::GPU,
        }
    }
}

#[derive(Debug)]
pub struct BenchDeviceHandler {
    pub devices: Vec<Device>,
}

impl BenchDeviceHandler {
    pub fn new() -> Result<Self> {
        let mut devices = Vec::new();
        if cfg!(all(
            target_os = "macos",
            target_arch = "aarch64",
            feature = "metal"
        )) {
            devices.push(Device::new_metal(0)?);
        } else if cfg!(feature = "cuda") {
            devices.push(Device::new_cuda(0)?);
        }
        devices.push(Device::Cpu);
        Ok(Self { devices })
    }
}

pub fn run_benchmark(device: Device) -> Result<f64> {
    debug!("Running benchmark...");

    let mut rng = rand::thread_rng();

    let m = rng.gen_range(3000..=5000);
    let n = rng.gen_range(3000..=5000);
    let p = rng.gen_range(3000..=5000);

    let a = Tensor::randn(0f32, 1., (m, n), &device)?;
    let b = Tensor::randn(0f32, 1., (n, p), &device)?;

    // Warm-up
    for _ in 0..WARMUP_ITERATIONS {
        let _ = a.matmul(&b)?;
    }

    device.sync()?;

    let start = Instant::now();
    // Perform matrix multiplication
    let _c = a.matmul(&b)?;
    device.sync()?;
    let duration = start.elapsed();

    debug!("Time taken for matrix multiplication: {:?}", duration);

    // Estimate the number of operations: 2 * m * n * p
    let ops = 2 * m * n * p;

    // Convert time to seconds and calculate FLOPS
    let time_in_seconds = duration.as_secs_f64();
    let flops = (ops as f64) / time_in_seconds;

    debug!("flops: {:.2}", flops);

    // Convert to TFLOPS
    let tflops = flops / 1e12;
    let formated_tflops = format!("{:.2}", tflops);

    debug!("Estimated performance: {:.2} TFLOPS", tflops);

    Ok(formated_tflops.parse().unwrap())
}

pub struct DevicePerformance {
    pub device: DeviceName,
    pub tflops: f64,
}

pub fn estimate_tflops() -> Result<Vec<(DeviceName, f64)>> {
    let handler = BenchDeviceHandler::new()?;
    let mut tflops_results: Vec<(DeviceName, f64)> = Vec::new();

    for device in handler.devices {
        let device_name = device.name();
        let tflops = run_benchmark(device)?;

        tflops_results.push((device_name, tflops));
    }

    Ok(tflops_results)
}
