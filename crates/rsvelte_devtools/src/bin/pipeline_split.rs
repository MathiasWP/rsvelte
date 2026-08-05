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

/// Order-dependent digest of every byte the compiler produced.
///
/// The instrumentation-cost A/B needs proof that the no-timer arm removed only
/// measurement work. Comparing the two arms' timings cannot show that; comparing
/// what they compiled can. Kept out of the timed mode so it never enters the
/// wall clock the A/B divides.
fn digest(acc: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *acc = (*acc ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3);
    }
}

fn main() {
    // The timers are off in the shipped compiler, so a profiler has to ask.
    // The digest mode below deliberately leaves them on: it compares compiler
    // output, and turning them off there would compare a different build state
    // than the one the split was measured in.
    profile::set_timers_enabled(true);
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let hash_only = args.iter().any(|a| a == "--hash");
    args.retain(|a| a != "--hash");
    let roots: Vec<PathBuf> = args.into_iter().map(PathBuf::from).collect();
    if roots.is_empty() {
        eprintln!("usage: pipeline_split [--hash] <dir>...");
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

    if hash_only {
        let mut acc = 0xcbf2_9ce4_8422_2325u64;
        for dev in [false, true] {
            for source in &sources {
                match compile_with_external_sourcemap_content(
                    source,
                    CompileOptions {
                        generate: GenerateMode::Client,
                        dev,
                        ..Default::default()
                    },
                ) {
                    Ok(result) => {
                        digest(&mut acc, result.js.code.as_bytes());
                        if let Some(css) = result.css.as_ref() {
                            digest(&mut acc, css.code.as_bytes());
                        }
                    }
                    // Failures have to contribute too: an arm that stopped
                    // compiling would otherwise agree with one that did not.
                    Err(_) => digest(&mut acc, b"<err>"),
                }
            }
        }
        println!("{} files, digest {acc:016x}", sources.len());
        return;
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
        ("resolve_lazy", b.resolve_lazy),
        ("ensure_script", b.ensure_script),
        ("finalize", b.finalize),
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

    // The divisor for the instrumentation's own cost. `timer_start` is the same
    // work at every site; the recorders are not, since they take between zero
    // and two `Duration`s, so dividing by recorder calls would mix sites that
    // read the clock twice with sites that never read it at all.
    #[cfg(feature = "measure-timer-calls")]
    {
        let pairs = profile::take_timer_starts();
        println!(
            "\ntimer_start calls {pairs} ({:.1} per compile)",
            if b.compiles == 0 {
                f64::NAN
            } else {
                pairs as f64 / b.compiles as f64
            }
        );
    }
}
