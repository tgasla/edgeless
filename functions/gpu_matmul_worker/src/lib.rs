// SPDX-FileCopyrightText: © 2024 Edgeless Project
// SPDX-License-Identifier: MIT
use edgeless_function::*;
use candle_core::{Tensor, Device};

struct GpuWorkerFun;

impl EdgeFunction for GpuWorkerFun {
    fn handle_cast(_src: InstanceId, encoded_message: &[u8]) {
        let str_message = core::str::from_utf8(encoded_message).unwrap_or("");
        log::info!("GpuWorker: 'Cast' called, MSG: {}", str_message);
        
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        log::info!("GpuWorker: Using device {:?}", device);
        
        match Tensor::randn(0f32, 1f32, (1000, 1000), &device) {
            Ok(a) => {
                if let Ok(b) = Tensor::randn(0f32, 1f32, (1000, 1000), &device) {
                    let start = std::time::Instant::now();
                    if let Ok(_c) = a.matmul(&b) {
                        let elapsed = start.elapsed();
                        log::info!("GpuWorker: Matrix multiplication completed on {:?} in {:?}", device, elapsed);
                    } else {
                        log::error!("GpuWorker: Matmul failed");
                    }
                }
            },
            Err(e) => {
                log::error!("GpuWorker: Failed to create tensor: {:?}", e);
            }
        }
    }

    fn handle_call(_src: InstanceId, encoded_message: &[u8]) -> CallRet {
        let str_message = core::str::from_utf8(encoded_message).unwrap_or("");
        log::info!("GpuWorker: 'Call' called, MSG: {}", str_message);
        
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                log::warn!("GpuWorker: CUDA device not found, falling back to CPU");
                Device::Cpu
            }
        };
        log::info!("GpuWorker: Using device {:?}", device);
        
        let a = match Tensor::randn(0f32, 1f32, (2000, 2000), &device) {
            Ok(t) => t,
            Err(e) => {
                let err_msg = format!("Failed to create tensor: {:?}", e);
                return CallRet::Reply(OwnedByteBuff::new_from_slice(err_msg.as_bytes()));
            }
        };
        let b = match Tensor::randn(0f32, 1f32, (2000, 2000), &device) {
            Ok(t) => t,
            Err(e) => {
                let err_msg = format!("Failed to create tensor b: {:?}", e);
                return CallRet::Reply(OwnedByteBuff::new_from_slice(err_msg.as_bytes()));
            }
        };
        
        let start = std::time::Instant::now();
        match a.matmul(&b) {
            Ok(_c) => {
                let elapsed = start.elapsed();
                let result_msg = format!("GPU Matrix multiplication (2000x2000) completed on {:?} in {:?}", device, elapsed);
                log::info!("{}", result_msg);
                CallRet::Reply(OwnedByteBuff::new_from_slice(result_msg.as_bytes()))
            },
            Err(e) => {
                let err_msg = format!("Failed matmul: {:?}", e);
                CallRet::Reply(OwnedByteBuff::new_from_slice(err_msg.as_bytes()))
            }
        }
    }

    fn handle_init(_payload: Option<&[u8]>, _serialized_state: Option<&[u8]>) {
        edgeless_function::init_logger();
        log::info!("GpuWorker: 'Init' called");
    }

    fn handle_stop() {
        log::info!("GpuWorker: 'Stop' called");
    }
}

edgeless_function::export!(GpuWorkerFun);
