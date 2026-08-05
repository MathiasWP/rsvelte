//! Phase split of `compile()`, taken from inside the production pipeline.
//!
//! # What this can and cannot support
//!
//! The shares here are the shipped compiler's shares. The timers sit on the
//! production call sites, so there is no orchestration to get wrong: the
//! retained scripts are threaded because production threads them, every step
//! runs because production runs it, and every argument has the value production
//! passes.
//!
//! The older splits could not say that. `corpus_profile` and `compile_profile`
//! call the phase functions themselves and diverge from production four ways at
//! once — no retained scripts, no TypeScript removal, no `<svelte:options>`
//! merge, and `compute_line_offsets(_, false)`. Two of those make a bucket too
//! large and one makes the denominator too small, so their ratios are not
//! bounded in a single direction and cannot be read as a split.
//!
//! `unattributed` is printed on every run. It is `total` minus the sum of the
//! buckets, measured independently, so a phase nobody instrumented lands there
//! rather than inflating the buckets that exist.
//!
//! # Entry point
//!
//! Drives `compile_with_external_sourcemap_content`, which is what the napi
//! binding calls (`rsvelte_napi/src/lib.rs`), i.e. the entry whose timings the
//! gap benchmark reports. `rsvelte_core::compile` differs from it only in
//! passing `include_sourcemap_content = true`; if the benchmark's entry ever
//! changes, change the call below.
//!
//! ```text
//! cargo run --release -p rsvelte_devtools --bin pipeline_split -- <dir>...
//! ```

use std::path::{Path, PathBuf};
use std::time::Duration;

use rsvelte_core::compiler::compile_with_external_sourcemap_content;
use rsvelte_core::compiler::phases::phase3_transform::profile;
use rsvelte_core::{CompileOptions, GenerateMode};

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n == "node_modules") {
                continue;
            }
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "svelte") {
            out.push(path);
        }
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn main() {
    let roots: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if roots.is_empty() {
        eprintln!("usage: pipeline_split <dir>...");
        std::process::exit(1);
    }

    let mut files = Vec::new();
    for root in &roots {
        collect(root, &mut files);
    }
    files.sort();
    let sources: Vec<String> = files
        .iter()
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .collect();
    if sources.is_empty() {
        eprintln!("no .svelte files under the given roots");
        std::process::exit(1);
    }

    // Warm up on the same entry point that is measured, so the timed pass is
    // not paying for first-touch costs the shipped compiler amortizes.
    for source in sources.iter().take(100) {
        let _ = compile_with_external_sourcemap_content(source, CompileOptions::default());
    }
    let _ = profile::take_pipeline_breakdown();

    for dev in [false, true] {
        for source in &sources {
            let _ = compile_with_external_sourcemap_content(
                source,
                CompileOptions {
                    generate: GenerateMode::Client,
                    dev,
                    ..Default::default()
                },
            );
        }
    }

    let b = profile::take_pipeline_breakdown();
    let total = ms(b.total);
    let pct = |d: Duration| {
        if total == 0.0 {
            f64::NAN
        } else {
            ms(d) / total * 100.0
        }
    };

    println!("{} files, {} compiles", sources.len(), b.compiles);
    println!("{:<16}{:>12}{:>10}", "bucket", "ms", "share");
    for (name, d) in [
        ("parse", b.parse),
        ("line_offsets", b.line_offsets),
        ("ensure_script", b.ensure_script),
        ("ts_removal", b.ts_removal),
        ("options_merge", b.options_merge),
        ("analyze", b.analyze),
        ("transform", b.transform),
    ] {
        println!("{name:<16}{:>12.1}{:>9.1}%", ms(d), pct(d));
    }
    // Printed unconditionally: a large residual is the signal that a phase is
    // missing from the buckets, which is exactly what went unnoticed before.
    println!(
        "{:<16}{:>12.1}{:>9.1}%",
        "unattributed",
        ms(b.unattributed()),
        pct(b.unattributed())
    );
    println!("{:<16}{total:>12.1}{:>9.1}%", "total", 100.0);
}
