//! Direct cost of one phase-timer pair, measured on the production functions.
//!
//! The corpus-wide A/B between an instrumented and a timer-free build could not
//! answer this. Its spread was 8.09% in the best of two attempts and 219% in the
//! worst, against a target of 0.3%: a whole-compile wall clock also carries
//! every layout and scheduling difference between two separately compiled
//! binaries, and the timer-free arm changed `TimerStart` from `Instant` to
//! `()`, so the two arms did not merely differ by the clock reads. This binary
//! drops that ambition and measures only what the runtime gate actually removes
//! -- the clock reads and the recorder call -- by running the production
//! functions themselves, in a loop, in one process. Unlike a two-binary A/B its
//! resolution is a parameter: raising `ITERS` buys precision.
//!
//! It also runs one sweep with the gate shut, which is the state a shipped
//! compile is in, and checks that no recorder accumulated while it was shut.
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

/// Band the two independent estimators must agree within. A pair is two clock
/// reads plus a recorder, so their ratio sits a little above two.
///
/// This catches contamination that hits the two estimators unequally, and only
/// that. A machine slow enough to stretch the clock read itself stretches both,
/// and the ratio is scale-free, so it stays in band while both figures are
/// wrong -- observed, not hypothesised. [`MAX_MEDIAN_OVER_MIN`] is the check
/// that moves in that case.
const RATIO_LOW: f64 = 1.8;
const RATIO_HIGH: f64 = 3.0;

/// How far a shape's median may sit above its minimum.
///
/// On an uncontended machine a fixed instruction sequence repeats to within a
/// few percent, so a median well above the minimum means most batches were
/// disturbed and the minimum is the only clean sample left -- if it is clean at
/// all. Unlike the ratio this is an absolute statement about one series, so a
/// uniform slowdown cannot hide inside it.
const MAX_MEDIAN_OVER_MIN: f64 = 1.2;

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
    (
        sorted[0],
        sorted[sorted.len() / 2],
        sorted[sorted.len() - 1],
    )
}

/// Minimum, median and maximum of one shape's batches. The spread is printed
/// because it is the evidence that the minimum was worth taking: a machine busy
/// enough to push the maximum well above the minimum is a machine whose median
/// would have carried that contention into the answer.
type Spread = (f64, f64, f64);

/// One sweep's five series.
struct Sweep {
    simple: Spread,
    pair: Spread,
    tuple: Spread,
    plain: Spread,
    control: f64,
}

fn report(label: &str, spread: Spread, control: f64) -> f64 {
    let (min, median, max) = spread;
    let net = min - control;
    println!("{label:14} min {min:7.2}  median {median:7.2}  max {max:7.2}  net {net:7.2} ns/pair");
    net
}

fn decision_share(net_ns: f64) -> f64 {
    net_ns * PAIRS_PER_COMPILE / 10.0 / COMPILE_US
}

/// One sweep of the three shapes plus the empty-loop control, returned as
/// per-shape minima. Run once with the gate open and once with it shut, so the
/// residual a shipped compile pays comes from the same code and the same
/// machine minute as the cost the gate removes.
fn sweep() -> Sweep {
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

    Sweep {
        simple: min_median_max(&simple),
        pair: min_median_max(&pair),
        tuple: min_median_max(&tuple),
        plain: min_median_max(&plain),
        control: min_median_max(&control).0,
    }
}

fn main() {
    profile::set_timers_enabled(true);

    // Warm every path once, then discard what the warm-up accumulated so the
    // execution proofs below count only the measured batches.
    let _ = batch!(profile::record_pipeline_parse, true);
    let _ = batch!(profile::record_st_prenormalize, true);
    let _ = batch!(profile::record_esrap_client_split, true);
    control_batch();
    profile::take_pipeline_breakdown();
    profile::take_esrap_breakdown();

    let on = sweep();

    let pipeline = profile::take_pipeline_breakdown();
    let esrap = profile::take_esrap_breakdown();
    let iterations = ITERS as f64 * BATCHES as f64;

    println!("iterations per shape  {iterations:.0}");
    println!(
        "control        min {:7.2} ns/iter  (empty loop)",
        on.control
    );
    println!();

    let simple_net = report("simple cell", on.simple, on.control);
    let pair_net = report("two cells", on.pair, on.control);
    let tuple_net = report("tuple cell", on.tuple, on.control);
    report("simple, no bb", on.plain, on.control);

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
    let ratio = simple_net / inner;
    // The band is wide because the recorder's share of the pair is not fixed
    // across shapes, and it is calibrated on a handful of runs, so treat a value
    // just outside it as a reason to look rather than a verdict.
    let steady = [on.simple, on.pair, on.tuple]
        .iter()
        .all(|&(min, median, _)| median / min <= MAX_MEDIAN_OVER_MIN);
    let trustworthy = (RATIO_LOW..=RATIO_HIGH).contains(&ratio) && steady;
    let worst_spread = [on.simple, on.pair, on.tuple]
        .iter()
        .map(|&(min, median, _)| median / min)
        .fold(0.0_f64, f64::max);
    println!(
        "median / min          {worst_spread:7.2}  limit {MAX_MEDIAN_OVER_MIN}  {}",
        if steady {
            "ok"
        } else {
            "DISTURBED -- most batches were contended, not just a few"
        }
    );
    if inner > 0.0 {
        println!("inner estimator       {inner:7.2} ns  one clock read");
        println!(
            "simple / inner        {ratio:7.2}  expected {RATIO_LOW} to {RATIO_HIGH}  {}",
            if trustworthy {
                "ok"
            } else {
                "OUT OF BAND -- the open-gate figures below are contaminated"
            }
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
    if trustworthy {
        println!(
            "decision uses the most expensive shape: {:.4} %  cut {DECISION_PCT} %  -> {}",
            decision_share(worst),
            if decision_share(worst) < DECISION_PCT {
                "ship unconditionally"
            } else {
                "runtime toggle"
            }
        );
    } else {
        println!("no decision: the estimators disagree, so this run cannot price the open gate");
    }

    // What a shipped compile still pays: the gate's own load and branch, times
    // the same 101.8 sites. Measured here rather than reasoned about, because
    // "a relaxed load is free" is the kind of claim that turns out to be a
    // factor of two off.
    profile::set_timers_enabled(false);
    let off = sweep();
    let residual = off.tuple.0.max(off.simple.0).max(off.pair.0) - off.control;
    println!();
    println!("--- gate closed, the state a shipped compile runs in ---");
    report("simple cell", off.simple, off.control);
    report("two cells", off.pair, off.control);
    report("tuple cell", off.tuple, off.control);
    // The closed-gate figure is the one that describes shipped builds, so it
    // gets the same steadiness check rather than inheriting the open sweep's
    // verdict. In practice it passes on machines where the open sweep does not:
    // a load and a branch are too short to be stretched by contention.
    let closed_spread = [off.simple, off.pair, off.tuple]
        .iter()
        .map(|&(min, median, _)| median / min)
        .fold(0.0_f64, f64::max);
    println!(
        "residual      {:6.3} us/compile   {:6.4} % of {COMPILE_US:.0} us   median/min {closed_spread:.2} {}",
        residual * PAIRS_PER_COMPILE / 1000.0,
        decision_share(residual),
        if closed_spread <= MAX_MEDIAN_OVER_MIN {
            "ok"
        } else {
            "DISTURBED"
        }
    );

    // The gate is only worth its complexity if it removes most of the cost.
    let removed = 1.0 - residual / worst;
    println!("gate removes  {:.1} % of the direct cost", removed * 100.0);

    // A recorder that still accumulated with the gate shut would make every
    // profile silently wrong in the other direction, so check rather than
    // assume the early returns are reached.
    let leaked = profile::take_esrap_breakdown().client_split_calls;
    println!(
        "leaked recorder calls  {leaked}  {}",
        if leaked == 0 {
            "none -- the gate reaches every recorder"
        } else {
            "LEAK -- a recorder ran with the gate shut"
        }
    );
}
