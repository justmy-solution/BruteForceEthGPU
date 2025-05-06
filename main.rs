use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use std::thread;

mod gpu_wrapper;
use gpu_wrapper::{init_gpu_context, GpuWorker};

/// Command-line arguments
#[derive(Parser)]
struct Args {
    /// Known seed words separated by commas
    #[clap(long, default_value = "")]
    known: String,

    /// Target Ethereum address (hex, no 0x prefix)
    #[clap(long, default_value = "")]
    address: String,

    /// Number of worker threads (and GPUs)
    #[clap(long, default_value = "1")]
    workers: u32,

    /// Resume offset per worker
    #[clap(long, default_value = "0")]
    resume_from: u64,

    /// Match mode: 0=full, 1=prefix, 2=zero
    #[clap(long, default_value = "0")]
    match_mode: i32,

    /// Prefix length (for match_mode=1)
    #[clap(long, default_value = "0")]
    match_prefix_len: i32,
}

fn main() -> Result<()> {
    // Parse CLI arguments
    let args = Args::parse();

    // Prepare shared data
    let wordlist: Arc<Vec<String>> = Arc::new(
        std::fs::read_to_string("words.txt")? 
            .lines() 
            .map(|s| s.trim().to_string()) 
            .collect()
    );
    let known_words: Arc<Vec<String>> = Arc::new(
        if args.known.is_empty() {
            Vec::new()
        } else {
            args.known.split(',').map(|s| s.trim().to_string()).collect()
        }
    );
    let address = Arc::new(args.address.clone());

    // Spawn one thread per worker/GPU
    let mut handles = Vec::new();
    for worker_id in 0..args.workers {
        let wordlist = Arc::clone(&wordlist);
        let known_words = Arc::clone(&known_words);
        let address = Arc::clone(&address);
        let match_mode = args.match_mode;
        let match_prefix_len = args.match_prefix_len;
        let total_workers = args.workers;
        let resume_from = args.resume_from;

        handles.push(thread::spawn(move || {
            // Each thread initializes its own GPU context
            let (ctx, module) = init_gpu_context(worker_id).expect("Failed to init GPU");

            // Create worker and run
            let mut worker = GpuWorker::new(
                &ctx,
                module,
                wordlist,
                known_words,
                address,
                match_mode,
                match_prefix_len,
                worker_id,
                total_workers,
                resume_from,
            ).expect("Failed to create GPU worker");

            worker.run().expect("Worker run failed");
        }));
    }

    // Wait for all threads to complete
    for handle in handles {
        if let Err(err) = handle.join() {
            eprintln!("Worker thread panicked: {:?}", err);
        }
    }

    Ok(())
}
