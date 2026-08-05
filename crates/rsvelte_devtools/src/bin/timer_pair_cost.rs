//! Direct cost of one phase-timer pair, measured on the production functions.
//!
//! The corpus-wide A/B between an instrumented and a timer-free build could not
//! answer this: its run-to-run spread was tens of percent against a target of
//! 0.3%, because a whole-compile wall clock also carries every scheduling and
//! layout difference between the two binaries. This binary drops that ambition
//! and measures only what a runtime toggle could actually remove -- the clock
//! reads and the recorder call -- by running the very functions the production
//! path runs, in a loop, in one process.
//!
//! What it therefore does NOT measure: the indirect cost of the instrumentation
//! existing at all (instruction-cache pressure, inlining decisions elsewhere).
//! That component survives a runtime toggle, since the code stays in the binary,
//! so it is not an input to the toggle-or-not decision. Its sign is unknown.
//!
//! Both estimates below are reported because they are independent:
//!
//!   outer  (loop wall clock - empty-loop wall clock) / iterations
//!          = now() + elapsed() + recorder
//!   inner  accumulated Duration / iterations
//!          = the interval the timer itself observes between now() and
//!            elapsed(), i.e. roughly one clock read
//!
//! `outer` should land near twice `inner` plus the recorder. If it does not,
//! one of the two is wrong, and the disagreement is visible without appeal to
//! any figure from outside this run.
//!
//! Minimum across batches is the estimator, not the median. For a fixed
//! instruction sequence, contention can only add time, so the minimum is the
//! least-contaminated sample. That is the opposite of the rule for A/B ratios
//! of two workloads, where taking minima on both sides inflates the difference.

use std::time::{Duration, Instant};

use rsvelte_core::compiler::phases::phase3_transform::profile;

/// Iterations per batch. Large enough that the two clock reads bounding the
/// batch are below the noise of a single iteration's measurement.
const ITERS: u64 = 2_000_000;
/// Batches per loop. The minimum over these is the reported figure.
const BATCHES: usize = 25;

/// Timer pairs one legacy-corpus compile pays for, measured with
/// `measure-timer-calls` over 633 files.
const PAIRS_PER_COMPILE: f64 = 101.8;
/// Wall clock of one legacy-corpus compile, same run as the count above.
const COMPILE_US: f64 = 2292.0;

/// One batch of the production timer trio. Nothing is `black_box`ed: the
/// recorder's thread-local write is an observable side effect, so the chain
/// cannot be folded away, and forcing the `Instant` through memory would add a
/// spill the production path does not pay.
fn timed_batch() -> f64 {
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let start = profile::timer_start();
        let d = profile::timer_elapsed(start);
        profile::record_pipeline_parse(d);
    }
    t0.elapsed().as_nanos() as f64 / ITERS as f64
}

/// The same loop with the trio removed, so the reported cost excludes the
/// counter and branch that any loop pays.
fn control_batch() -> f64 {
    let t0 = Instant::now();
    for i in 0..ITERS {
        std::hint::black_box(i);
    }
    t0.elapsed().as_nanos() as f64 / ITERS as f64
}

fn summarize(label: &str, mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    let min = samples[0];
    let median = samples[samples.len() / 2];
    let max = samples[samples.len() - 1];
    println!("{label:10} min {min:8.3} ns  median {median:8.3} ns  max {max:8.3} ns");
    min
}

fn main() {
    // Alternate the two loops so a drifting machine cannot land entirely on one
    // of them, and discard the first round as warm-up.
    timed_batch();
    control_batch();
    profile::take_pipeline_breakdown();

    let mut timed = Vec::with_capacity(BATCHES);
    let mut control = Vec::with_capacity(BATCHES);
    for i in 0..BATCHES {
        if i % 2 == 0 {
            timed.push(timed_batch());
            control.push(control_batch());
        } else {
            control.push(control_batch());
            timed.push(timed_batch());
        }
    }

    let accumulated: Duration = profile::take_pipeline_breakdown().parse;
    let iterations = ITERS as f64 * BATCHES as f64;

    let timed_min = summarize("timed", timed);
    let control_min = summarize("control", control);

    let outer = timed_min - control_min;
    let inner = accumulated.as_nanos() as f64 / iterations;

    println!();
    println!("outer  {outer:.3} ns/pair   now() + elapsed() + recorder");
    println!("inner  {inner:.3} ns/pair   interval the timer observes");
    if inner > 0.0 {
        println!("ratio  {:.2}   expected near 2 plus the recorder", outer / inner);
    } else {
        // A zero here means the accumulator never ran, not that the clock is
        // free; the outer figure would then be measuring an empty loop.
        println!("ratio  n/a   accumulator stayed at zero -- outer is not trustworthy");
    }

    println!();
    println!("per compile  {:.3} us", outer * PAIRS_PER_COMPILE / 1000.0);
    println!(
        "share        {:.4} %  of {COMPILE_US:.0} us",
        outer * PAIRS_PER_COMPILE / 10.0 / COMPILE_US
    );
}
