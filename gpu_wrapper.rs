use anyhow::Result;
use cust::{context::{Context, ContextFlags}, device::Device, function::Function, module::Module, prelude::*, stream::Stream, memory::DeviceBuffer};
use std::{sync::Arc, ffi::CString};

pub struct GpuWorker<'m> {
    function: Function<'m>,     // Kernel function
    stream: Stream,
    wordlist_len: i32,
    known_indices: Vec<i32>,
    known_count: i32,
    target_address: [u8; 20],
    match_mode: i32,
    match_prefix_len: i32,
    worker_id: u32,
    total_workers: u32,
    resume_from: u64,
}

/// Initialize CUDA context and load the PTX module on specified GPU device.
pub fn init_gpu_context(device_id: u32) -> Result<(Context, Module)> {
    // Select device and create context
    let device = Device::get_device(device_id)?;
    let context = Context::create_and_push(ContextFlags::MAP_HOST, device)?;

    // Load PTX
    let ptx = include_str!("../src/gpu_kernel.ptx");
    let ptx_cstr = CString::new(ptx)?;
    let module = Module::load_from_string(ptx_cstr.as_c_str())?;

    Ok((context, module))
}

impl<'m> GpuWorker<'m> {
    pub fn new(
        _context: &Context,
        module: &'m Module,
        wordlist: Arc<Vec<String>>,
        known_words: Arc<Vec<String>>,
        address: Arc<String>,
        match_mode: i32,
        match_prefix_len: i32,
        worker_id: u32,
        total_workers: u32,
        resume_from: u64,
    ) -> Result<Self> {
        // Convert known words to indices
        let known_indices: Vec<i32> = known_words
            .iter()
            .map(|w| wordlist.iter().position(|x| x == w).unwrap() as i32)
            .collect();
        let known_count = known_indices.len() as i32;

        // Parse hex address into 20 bytes
        let mut target_address = [0u8; 20];
        let addr_str = address.trim();
        for i in 0..(addr_str.len() / 2) {
            target_address[i] = u8::from_str_radix(&addr_str[2*i..2*i+2], 16)?;
        }

        // Retrieve the GPU function and create a non-blocking stream
        let function = module.get_function("search_seeds")?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(GpuWorker {
            function,
            stream,
            wordlist_len: wordlist.len() as i32,
            known_indices,
            known_count,
            target_address,
            match_mode,
            match_prefix_len,
            worker_id,
            total_workers,
            resume_from,
        })
    }

    pub fn run(&mut self) -> Result<()> {
        // Compute search space and batch size
        let unknown_count = 12 - self.known_count as u32;
        let total_candidates = 2048u128.pow(unknown_count) as u64;
        let batch_size = total_candidates / self.total_workers as u64;

        // Determine starting index including resume offset
        let start_index = batch_size * self.worker_id as u64 + self.resume_from;

        // Allocate device buffers
        let d_known = DeviceBuffer::from_slice(&self.known_indices)?;
        let d_address = DeviceBuffer::from_slice(&self.target_address)?;

        // Launch the kernel
let threads_per_block = 256;
let blocks = ((batch_size + threads_per_block - 1) / threads_per_block) as u32;
unsafe {
    launch!(
        self.function<<<blocks, threads_per_block, 0, self.stream>>>(
            start_index,
            batch_size,
            self.wordlist_len,
            d_known.as_device_ptr(),
            self.known_count,
            self.target_address,
            self.match_mode,
            self.match_prefix_len,
        )
    )?;
}
self.stream.synchronize()?;

        // Write new offset for resume
        let new_offset = start_index + batch_size;
        std::fs::write(format!("worker{}.offset", self.worker_id), new_offset.to_string())?;

        Ok(())
    }
}
