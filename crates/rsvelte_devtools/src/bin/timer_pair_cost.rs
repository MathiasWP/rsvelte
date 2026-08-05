//! Direct cost of one phase-timer pair, measured on the production functions.
//!
//! The corpus-wide A/B between an instrumented and a timer-free build could not
//! answer this. Its spread was 8.09% in the best of two attempts and 219% in the
//! worst, against a target of 0.3%: a whole-compile wall clock also carries
//! every layout and scheduling difference between two separately compiled
//! binaries, and `measure-no-timers` changes `TimerStart` from `Instant` to
//! `()`, so the two arms do not merely differ by the clock reads. This binary
//! drops that ambition and measures only what a runtime toggle could actually
//! remove -- the clock reads and the recorder call -- by running the production
//! functions themselves, in a loop, in one process. Unlike a two-binary A/B its
//! resolution is a parameter: raising `ITERS` buys precision.
//!
//! What it therefore does NOT measure: the indirect cost of the instrumentation
//! existing at all (instruction-cache pressure, inlining decisions elsewhere,
//! frame size). That component survives a runtime toggle, since the code stays
//! in the binary, so it cannot decide toggle-or-not. Its sign is unknown.
//!
//! ## Why three shapes
//!
//! The 101.8 pairs a compile pays are spread over recorders of three different
//! costs, and the mix is not known:
//!
//!   simple   one `Cell<Duration>` add
//!   pair     two separate cells, a `Duration` and a call count
//!   tuple    one `Cell<(Duration, u64)>`, read and written as a unit
//!
//! All three are reported. The decision uses the most expensive, because
//! choosing the cheapest would assume the mix is favourable.
//!
//! ## Why the result is a lower bound
//!
//! In a loop the branch predictor is perfect and the cache is hot, neither of
//! which holds at a real call site. A synthetic figure can therefore come out
//! below the in-situ cost, so it may only be used to conclude "cheap" with
//! margin, never "cheap" at the edge.
//!
//! ## Why minimum, not median
//!
//! For a fixed instruction sequence contention can only add time, so the
//! minimum is the least contaminated sample. This is the opposite of the rule
//! for A/B ratios of two workloads, where taking minima on both sides inflates
//! the difference; the statistics differ because that question is a ratio of
//! two changing quantities and this one is one instruction sequence's absolute
//! cost.

use std::time::Instant;

use rsvelte_core::compiler::phases::phase3_transform::profile;

/// Iterations per batch. Large enough that the two clock reads bounding a batch
/// are far below one iteration's measured cost.
const ITERS: u64 = 2_000_000;
/// Batches per shape. The minimum over these is the reported figure.
const BATCHES: usize = 21;

/// Timer pairs one legacy-corpus compile pays for, from the `measure-timer-calls`
/// build over 633 files.
const PAIRS_PER_COMPILE: f64 = 101.8;
/// Wall clock of one legacy-corpus compile, same run as the count above.
const COMPILE_US: f64 = 2292.0;
/// The registered cut: below this the instrumentation ships unconditionally,
/// at or above it the recorders go behind a runtime toggle.
const DECISION_PCT: f64 = 0.15;

/// One batch of the production timer trio.
///
/// `$bb` selects whether the `Duration` is forced through `black_box` on its way
/// to the recorder. The instrumented path does not do that, so the black-boxed
/// figure is an overestimate; it exists as the arm that cannot be optimised away
/// under any assumption, and its agreement with the plain arm is what shows the
/// plain arm was not optimised away either.
macro_rules! batch {
    ($rec:path, $bb:expr) => {{
        let t0 = Instant::now();
        for _ in 0..ITERS {
            let start = profile::timer_start();
            let d = profile::timer_elapsed(start);
            $rec(if $bb { std::hint::black_box(d) } else { d });
        }
        t0.elapsed().as_nanos() as f64 / ITERS as f64
    }};
}

/// The same loop with the trio removed, so the reported cost excludes the
/// counter and branch every loop pays.
fn control_batch() -> f64 {
    let t0 = Instant::now();
    for i in 0..ITERS {
        std::hint::black_box(i);
    }
    t0.elapsed().as_nanos() as f64 / ITERS as f64
}

fn min_median_max(samples: &[f64]) -> (f64, f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    (sorted[0], sorted[sorted.len() / 2], sorted[sorted.len() - 1])
}

fn report(label: &str, samples: &[f64], control: f64) -> f64 {
    let (min, median, max) = min_median_max(samples);
    let net = min - control;
    println!("{label:14} min {min:7.2}  median {median:7.2}  max {max:7.2}  net {net:7.2} ns/pair");
    net
}

fn decision_share(net_ns: f64) -> f64 {
    net_ns * PAIRS_PER_COMPILE / 10.0 / COMPILE_US
}

fn main() {
    // Warm every path once, then discard what the warm-up accumulated so the
    // execution proofs below count only the measured batches.
    let _ = batch!(profile::record_pipeline_parse, true);
    let _ = batch!(profile::record_st_prenormalize, true);
    let _ = batch!(profile::record_esrap_client_split, true);
    control_batch();
    profile::take_pipeline_breakdown();
    profile::take_esrap_breakdown();

    let mut simple = Vec::with_capacity(BATCHES);
    let mut pair = Vec::with_capacity(BATCHES);
    let mut tuple = Vec::with_capacity(BATCHES);
    let mut plain = Vec::with_capacity(BATCHES);
    let mut control = Vec::with_capacity(BATCHES);

    for i in 0..BATCHES {
        // Rotate the order so no shape sits permanently at the position where a
        // drifting machine is slowest.
        if i % 2 == 0 {
            simple.push(batch!(profile::record_pipeline_parse, true));
            pair.push(batch!(profile::record_st_prenormalize, true));
            tuple.push(batch!(profile::record_esrap_client_split, true));
            plain.push(batch!(profile::record_pipeline_parse, false));
            control.push(control_batch());
        } else {
            control.push(control_batch());
            plain.push(batch!(profile::record_pipeline_parse, false));
            tuple.push(batch!(profile::record_esrap_client_split, true));
            pair.push(batch!(profile::record_st_prenormalize, true));
            simple.push(batch!(profile::record_pipeline_parse, true));
        }
    }

    let pipeline = profile::take_pipeline_breakdown();
    let esrap = profile::take_esrap_breakdown();
    let iterations = ITERS as f64 * BATCHES as f64;

    let (control_min, _, _) = min_median_max(&control);
    println!("iterations per shape  {iterations:.0}");
    println!("control        min {control_min:7.2} ns/iter  (empty loop)");
    println!();

    let simple_net = report("simple cell", &simple, control_min);
    let pair_net = report("two cells", &pair, control_min);
    let tuple_net = report("tuple cell", &tuple, control_min);
    report("simple, no bb", &plain, control_min);

    // Execution proof. The tuple recorder increments a call count once per
    // iteration, so a mismatch here means the loop did not run the number of
    // times the wall clock was divided by -- which no amount of reasoning about
    // `black_box` could otherwise rule out.
    println!();
    let expected = ITERS * BATCHES as u64;
    let calls = esrap.client_split_calls;
    println!(
        "tuple recorder calls  {calls}  expected {expected}  {}",
        if calls == expected {
            "match"
        } else {
            "MISMATCH -- the loop did not run as many times as assumed"
        }
    );

    // Independent estimator: the interval the timer observes between `now()` and
    // `elapsed()`, which is roughly one clock read. Two of those plus the
    // recorder should land near the net figures above. The two come from
    // different measurements, so disagreement is visible without appealing to
    // any figure from outside this run. `parse` collects both simple-cell
    // shapes, black-boxed and not, hence the factor of two.
    let inner = pipeline.parse.as_nanos() as f64 / (iterations * 2.0);
    if inner > 0.0 {
        println!("inner estimator       {inner:7.2} ns  one clock read");
        println!(
            "simple / inner        {:7.2}  expected near 2",
            simple_net / inner
        );
    } else {
        println!("inner estimator       zero -- the accumulator never ran, net figures unusable");
    }

    println!();
    let worst = simple_net.max(pair_net).max(tuple_net);
    for (label, net) in [
        ("simple cell", simple_net),
        ("two cells", pair_net),
        ("tuple cell", tuple_net),
    ] {
        println!(
            "{label:14} {:6.3} us/compile   {:6.4} % of {COMPILE_US:.0} us",
            net * PAIRS_PER_COMPILE / 1000.0,
            decision_share(net)
        );
    }
    println!();
    println!(
        "decision uses the most expensive shape: {:.4} %  cut {DECISION_PCT} %  -> {}",
        decision_share(worst),
        if decision_share(worst) < DECISION_PCT {
            "ship unconditionally"
        } else {
            "runtime toggle"
        }
    );
}
