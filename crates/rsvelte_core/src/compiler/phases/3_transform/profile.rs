//! Lightweight thread-local timers for splitting Phase 3 (Transform) into
//! sub-phases (template fragment walk, instance-script text transform, CSS
//! render, JS codegen).
//!
//! ## What the instrumentation costs, and why it is off by default
//!
//! Measured 2026-08-05 by `rsvelte_devtools/bin/timer_pair_cost.rs`, which runs
//! these very functions in a loop rather than comparing two builds:
//!
//! ```text
//! one clock read                      18.7 ns
//! recorder (thread-local + Cell)       6.7 ns
//! one timer pair                      44.2 ns   (three recorder shapes agree to 0.3%)
//! pairs per compile                  101.8      legacy corpus, 633 files
//! per compile                          4.5 µs = 0.196% of a 2292 µs compile
//! ```
//!
//! An earlier version of this note claimed ~100–200ns per file. That was off by
//! a factor of ~25, because it counted a handful of sites rather than the 101.8
//! pairs a compile actually reaches. The rune corpus pays 24.8 pairs, or 0.117%.
//!
//! 0.196% is above the cut we set for shipping timers unconditionally, so
//! [`set_timers_enabled`] gates them and they start off. The gate is a relaxed
//! atomic load, so both states stay in one binary and the profiled build is the
//! shipped build.
//!
//! What a shipped compile still pays is the gate itself, measured the same way:
//!
//! ```text
//! gate load and branch                 0.33 ns per site
//! per compile                          0.034 µs = 0.0015%
//! ```
//!
//! so the gate removes 99.5% of the direct cost. The same run checks that no
//! recorder accumulated while the gate was shut, since a recorder the gate did
//! not reach would corrupt every later profile rather than merely cost time.
//!
//! What this does NOT cover: the cost of the instrumentation merely existing
//! (instruction-cache pressure, inlining decisions elsewhere). A runtime gate
//! cannot remove that, since the code stays in the binary, so it is unmeasured
//! and its sign is unknown. The wall-clock A/B that would have measured it had a
//! best-case spread of 8.09% against a 0.3% question and was retired.
//!
//! Consumed by the `rsvelte_devtools` profiling binaries, which enable the gate
//! at startup.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

// `std::time::Instant::now()` traps on `wasm32-unknown-unknown` (no system
// clock — see std::sys::time::unsupported). The profile instrumentation
// below is consumed only by native devtools (`rsvelte_devtools/bin/compile_profile.rs`), but
// the call sites live in shared compile paths, so the Instant calls would
// fire from the WASM playground and crash the page. Provide a WASM-safe
// shim that returns a unit "instant" with a zero-cost elapsed so the
// instrumented sites stay compile-target-portable without #[cfg] noise.

/// Whether the phase timers read the clock. See the module note for the 0.196%
/// that makes this a gate rather than an unconditional cost.
///
/// Relaxed is enough: nothing is published through this flag, and a profiling
/// run that missed the first few compiles would still be a valid sample of the
/// rest. Making it `Acquire`/`Release` would buy ordering no reader needs.
static TIMERS_ENABLED: AtomicBool = AtomicBool::new(false);

/// Turn the phase timers on or off for the whole process.
///
/// Profiling binaries call this at startup. Leaving it off is what the shipped
/// compiler does, and the two states differ only in a branch, so there is no
/// build in which the timers exist and no build in which they are free.
pub fn set_timers_enabled(on: bool) {
    TIMERS_ENABLED.store(on, Ordering::Relaxed);
}

#[inline]
pub fn timers_enabled() -> bool {
    TIMERS_ENABLED.load(Ordering::Relaxed)
}

#[cfg(all(not(target_arch = "wasm32"), not(feature = "measure-no-timers")))]
pub type TimerStart = Option<std::time::Instant>;

#[cfg(any(target_arch = "wasm32", feature = "measure-no-timers"))]
pub type TimerStart = ();

#[cfg(all(not(target_arch = "wasm32"), not(feature = "measure-no-timers")))]
#[inline]
pub fn timer_start() -> TimerStart {
    if !timers_enabled() {
        return None;
    }
    #[cfg(feature = "measure-timer-calls")]
    TIMER_STARTS.with(|c| c.set(c.get() + 1));
    Some(std::time::Instant::now())
}

#[cfg(any(target_arch = "wasm32", feature = "measure-no-timers"))]
#[inline]
pub fn timer_start() -> TimerStart {}

#[cfg(all(not(target_arch = "wasm32"), not(feature = "measure-no-timers")))]
#[inline]
pub fn timer_elapsed(start: TimerStart) -> Duration {
    // `None` means the gate was off when the timer started. Returning the
    // elapsed time of a clock read that never happened is not an option, and
    // zero is what every recorder already treats as "no contribution".
    start.map_or(Duration::ZERO, |start| start.elapsed())
}

#[cfg(any(target_arch = "wasm32", feature = "measure-no-timers"))]
#[inline]
pub fn timer_elapsed(_start: TimerStart) -> Duration {
    Duration::ZERO
}

/// How many `Instant` pairs one compile pays for.
///
/// The unit is deliberately `timer_start`, not `record_*`: the recorders take
/// between zero and two `Duration`s, so counting them would mix sites that read
/// the clock twice with sites that never read it. Every `timer_start` is one
/// `now()` plus the `elapsed()` that consumes it, so dividing the measured
/// instrumentation cost by this count gives a per-pair figure without assuming
/// the sites are alike.
#[cfg(feature = "measure-timer-calls")]
thread_local! {
    static TIMER_STARTS: Cell<u64> = const { Cell::new(0) };
}

#[cfg(feature = "measure-timer-calls")]
pub fn take_timer_starts() -> u64 {
    TIMER_STARTS.with(|c| c.replace(0))
}

/// Per-call-site cost of the `rsvelte_esrap` printer inside one compile run.
///
/// Split out from [`Phase3Breakdown`] because the printer is reached from six
/// independent sites and the question "would a faster printer speed up
/// `compile()`" needs each site's share separately, not their sum. The three
/// client branches stay apart because they take different printer entry points
/// and only one of them is reachable per compile.
#[derive(Default, Debug, Clone, Copy)]
pub struct EsrapBreakdown {
    /// Client final print, comment-bearing branch (`print_split`).
    pub client_split: Duration,
    pub client_split_calls: u64,
    /// Client final print, sourcemap branch (`print_with_map`).
    pub client_map: Duration,
    pub client_map_calls: u64,
    /// Client final print, plain branch (`print_with`).
    pub client_plain: Duration,
    pub client_plain_calls: u64,
    /// Server final print (the single whole-program `print`).
    pub server_print: Duration,
    pub server_print_calls: u64,
    /// Server async-body round-trip: the print half.
    pub server_pipe_print: Duration,
    /// Server async-body round-trip: the re-parse half of the same round-trip.
    pub server_pipe_reparse: Duration,
    pub server_pipe_calls: u64,
    /// `normalize_js_with_oxc` slow path (parse + print), print half only.
    pub normalize_print: Duration,
    pub normalize_calls: u64,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Phase3Breakdown {
    pub visit_program: Duration,
    pub script_text_transform: Duration,
    pub template_fragment: Duration,
    pub assembly_after_fragment: Duration,
    pub css_render: Duration,
    pub codegen: Duration,
}

/// One level below [`Phase3Breakdown::script_text_transform`], which is the
/// largest Phase 3 bucket. The five stages are sequential and disjoint, so the
/// difference between their sum and `script_text_transform` is the prologue
/// plus the early-out paths.
///
/// Do not expect `Σ stages == script_text_transform`. The residual is small
/// against the parent but is neither reproducible in magnitude nor stable in
/// sign -- measured between -0.05% and +2.5% of the parent on an idle machine,
/// and it flips sign and grows to double digits under load. The cause is
/// unexplained; read the stage shares as good to roughly a point, and read the
/// residual as an instrument reading rather than as prologue time.
#[derive(Default, Debug, Clone, Copy)]
pub struct ScriptTextBreakdown {
    /// Comment strip, class fields, comma split, arrow-paren strip.
    pub prenormalize: Duration,
    /// Gathering reactive / proxy / prop variable sets from the script text.
    pub collect_vars: Duration,
    /// The line-by-line accumulation loop, `process_accumulated` included.
    pub line_loop: Duration,
    /// The part of `line_loop` spent transforming completed statements; the
    /// remainder is the loop's own per-line scanning.
    pub process_accumulated: Duration,
    /// Completed statements handed to the processor.
    pub statements: u64,
    /// `transform_client_runes_with_skip_and_state`, the per-statement rune rewrite.
    pub runes: Duration,
    /// The legacy `$:` reactive-statement branch.
    pub reactive_stmt: Duration,
    pub reactive_calls: u64,
    /// Reactive-statement append plus the runes-mode AST transforms.
    pub ast_transforms: Duration,
    /// Shadowed-local post-pass and dev-mode instrumentation.
    pub post_passes: Duration,
    /// Calls that reached the staged region (i.e. did not take an early out).
    pub calls: u64,
    /// Every entry into the staged function, early outs included.
    pub entries: u64,
    /// Every `record_script_text`. Must equal `entries`, or some staged work
    /// ran outside the parent timer and the stage sum cannot be compared to it.
    pub parent_calls: u64,
    /// Entries that happened while an outer entry was still on the stack.
    ///
    /// `entries == parent_calls` does not prove the two are paired: a missing
    /// one and a spare one cancel. Re-entry is the case that actually breaks
    /// the arithmetic, because the inner call's stage timers accumulate into
    /// the same totals the outer call's parent timer already covers, letting
    /// the stage sum exceed its parent. Any non-zero value here invalidates
    /// every share taken from `script_text` and below.
    pub nested_entries: u64,
    /// `record_script_text` split by call site, so a parent call with no entry
    /// (or the reverse) cannot hide inside an equal total.
    pub parent_site_main: u64,
    pub parent_site_pub: u64,
    /// Wall time inside the staged function, measured by the entry guard.
    ///
    /// Bounds the stage sum from above and the parent timer from below, so
    /// when the two disagree this says which of them is wrong.
    pub in_function: Duration,
    /// Entries that ran while no parent interval was open.
    ///
    /// This is the case equal totals cannot rule out: a parent interval that
    /// wraps no call, paired with a call that no parent wraps.
    pub entries_outside_parent: u64,
}

/// The `*_ast` rewrite passes all reach the parser through one choke point,
/// [`super::shared::ast_rewrite::with_program`], so counting there covers every
/// pass at once.
///
/// `bytes` is the load-independent quantity: it is the total source length
/// handed to the parser across a compile, so `bytes / file_len` says how many
/// times over the pipeline re-reads the same script. A ratio that grows with
/// file size is superlinear re-parsing; a flat ratio is a constant factor.
#[derive(Default, Debug, Clone, Copy)]
pub struct ReparseBreakdown {
    /// Time inside `Parser::parse` only.
    pub parse: Duration,
    /// Time inside the visitor closure, i.e. everything the pass does with the
    /// program once it exists.
    pub visit: Duration,
    pub calls: u64,
    /// Summed `source.len()` over every call.
    pub bytes: u64,
    /// The same three numbers for passes that build a `Parser` themselves
    /// instead of going through the shared driver. Kept apart so the driver's
    /// count is never mistaken for the whole re-parse cost.
    pub direct_parse: Duration,
    pub direct_calls: u64,
    pub direct_bytes: u64,
}

/// The whole of `compile()`, split at the boundaries the production pipeline
/// actually has.
///
/// Recorded from inside the pipeline rather than by a driver that calls the
/// phase functions itself. A driver has to thread the retained scripts, call
/// every step, pick the same implementation, and pass the same argument
/// values — four things it can get wrong, and the older profilers got all four
/// wrong at once. Timing the production call sites removes the notion of
/// "calling it correctly": whatever `compile()` does is what is measured.
///
/// `total` is taken around the entire compile, so `total - Σ(buckets)` is time
/// no bucket claims. Report that residual. A bucket nobody added shows up
/// there instead of inflating the shares of the buckets that do exist, which
/// is how TypeScript removal stayed missing from the older split unnoticed.
#[derive(Default, Debug, Clone, Copy)]
pub struct PipelineBreakdown {
    /// Template parse, deferred script split included.
    pub parse: Duration,
    /// The one full-source scan that produces line offsets.
    pub line_offsets: Duration,
    /// OXC over the instance and module scripts, retaining the programs.
    pub ensure_script: Duration,
    /// TypeScript node removal. Absent from every older profiler.
    pub ts_removal: Duration,
    /// Merging `<svelte:options>` into the compile options.
    pub options_merge: Duration,
    /// Phase 2. Its own sub-split is not covered here.
    pub analyze: Duration,
    /// Phase 3, whose sub-split is [`Phase3Breakdown`].
    pub transform: Duration,
    /// The entire compile, measured independently of the buckets above.
    pub total: Duration,
    /// Compiles that reached the pipeline, for per-file figures.
    pub compiles: u64,
}

impl PipelineBreakdown {
    /// Time inside `total` that no bucket claims. A missing bucket lands here.
    pub fn unattributed(&self) -> Duration {
        self.total.saturating_sub(
            self.parse
                + self.line_offsets
                + self.ensure_script
                + self.ts_removal
                + self.options_merge
                + self.analyze
                + self.transform,
        )
    }
}

thread_local! {
    static PL_PARSE: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static PL_LINE_OFFSETS: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static PL_ENSURE_SCRIPT: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static PL_TS_REMOVAL: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static PL_OPTIONS_MERGE: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static PL_ANALYZE: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static PL_TRANSFORM: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static PL_TOTAL: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static PL_COMPILES: Cell<u64> = const { Cell::new(0) };

    static REPARSE: Cell<(Duration, Duration, u64, u64)> =
        const { Cell::new((Duration::ZERO, Duration::ZERO, 0, 0)) };
    static REPARSE_DIRECT: Cell<(Duration, u64, u64)> =
        const { Cell::new((Duration::ZERO, 0, 0)) };

    static VISIT_PROGRAM: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static SCRIPT_TEXT: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static TEMPLATE_FRAGMENT: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static ASSEMBLY_AFTER_FRAGMENT: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static CSS_RENDER: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static CODEGEN: Cell<Duration> = const { Cell::new(Duration::ZERO) };

    static ST_PRENORMALIZE: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static ST_COLLECT_VARS: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static ST_LINE_LOOP: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static ST_AST_TRANSFORMS: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static ST_POST_PASSES: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static ST_CALLS: Cell<u64> = const { Cell::new(0) };
    static ST_ENTRIES: Cell<u64> = const { Cell::new(0) };
    static ST_PROCESS_ACCUMULATED: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static ST_STATEMENTS: Cell<u64> = const { Cell::new(0) };
    static ST_RUNES: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static ST_REACTIVE_STMT: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static ST_REACTIVE_CALLS: Cell<u64> = const { Cell::new(0) };
    static ST_PARENT_CALLS: Cell<u64> = const { Cell::new(0) };
    static ST_DEPTH: Cell<u64> = const { Cell::new(0) };
    static ST_NESTED_ENTRIES: Cell<u64> = const { Cell::new(0) };
    static ST_IN_FUNCTION: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    static ST_PARENT_OPEN: Cell<u64> = const { Cell::new(0) };
    static ST_ENTRIES_OUTSIDE: Cell<u64> = const { Cell::new(0) };
    static ST_PARENT_SITE_MAIN: Cell<u64> = const { Cell::new(0) };
    static ST_PARENT_SITE_PUB: Cell<u64> = const { Cell::new(0) };

    static ESRAP_CLIENT_SPLIT: Cell<(Duration, u64)> = const { Cell::new((Duration::ZERO, 0)) };
    static ESRAP_CLIENT_MAP: Cell<(Duration, u64)> = const { Cell::new((Duration::ZERO, 0)) };
    static ESRAP_CLIENT_PLAIN: Cell<(Duration, u64)> = const { Cell::new((Duration::ZERO, 0)) };
    static ESRAP_SERVER: Cell<(Duration, u64)> = const { Cell::new((Duration::ZERO, 0)) };
    static ESRAP_PIPE: Cell<(Duration, Duration, u64)> =
        const { Cell::new((Duration::ZERO, Duration::ZERO, 0)) };
    static ESRAP_NORMALIZE: Cell<(Duration, u64)> = const { Cell::new((Duration::ZERO, 0)) };
}

#[inline]
pub fn record_reparse(parse: Duration, visit: Duration, bytes: usize) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    REPARSE.with(|c| {
        let (p, v, n, b) = c.get();
        c.set((p + parse, v + visit, n + 1, b + bytes as u64));
    });
}

#[inline]
pub fn record_direct_parse(parse: Duration, bytes: usize) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    REPARSE_DIRECT.with(|c| {
        let (p, n, b) = c.get();
        c.set((p + parse, n + 1, b + bytes as u64));
    });
}

pub fn take_reparse_breakdown() -> ReparseBreakdown {
    let (parse, visit, calls, bytes) = REPARSE.replace((Duration::ZERO, Duration::ZERO, 0, 0));
    let (direct_parse, direct_calls, direct_bytes) = REPARSE_DIRECT.replace((Duration::ZERO, 0, 0));
    ReparseBreakdown {
        parse,
        visit,
        calls,
        bytes,
        direct_parse,
        direct_calls,
        direct_bytes,
    }
}

#[inline]
pub fn record_esrap_client_split(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    ESRAP_CLIENT_SPLIT.with(|c| {
        let (t, n) = c.get();
        c.set((t + d, n + 1));
    });
}

#[inline]
pub fn record_esrap_client_map(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    ESRAP_CLIENT_MAP.with(|c| {
        let (t, n) = c.get();
        c.set((t + d, n + 1));
    });
}

#[inline]
pub fn record_esrap_client_plain(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    ESRAP_CLIENT_PLAIN.with(|c| {
        let (t, n) = c.get();
        c.set((t + d, n + 1));
    });
}

#[inline]
pub fn record_esrap_server(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    ESRAP_SERVER.with(|c| {
        let (t, n) = c.get();
        c.set((t + d, n + 1));
    });
}

#[inline]
pub fn record_esrap_pipe(print: Duration, reparse: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    ESRAP_PIPE.with(|c| {
        let (p, r, n) = c.get();
        c.set((p + print, r + reparse, n + 1));
    });
}

#[inline]
pub fn record_esrap_normalize(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    ESRAP_NORMALIZE.with(|c| {
        let (t, n) = c.get();
        c.set((t + d, n + 1));
    });
}

pub fn take_esrap_breakdown() -> EsrapBreakdown {
    let (client_split, client_split_calls) =
        ESRAP_CLIENT_SPLIT.with(|c| c.replace((Duration::ZERO, 0)));
    let (client_map, client_map_calls) = ESRAP_CLIENT_MAP.with(|c| c.replace((Duration::ZERO, 0)));
    let (client_plain, client_plain_calls) =
        ESRAP_CLIENT_PLAIN.with(|c| c.replace((Duration::ZERO, 0)));
    let (server_print, server_print_calls) = ESRAP_SERVER.with(|c| c.replace((Duration::ZERO, 0)));
    let (server_pipe_print, server_pipe_reparse, server_pipe_calls) =
        ESRAP_PIPE.with(|c| c.replace((Duration::ZERO, Duration::ZERO, 0)));
    let (normalize_print, normalize_calls) =
        ESRAP_NORMALIZE.with(|c| c.replace((Duration::ZERO, 0)));
    EsrapBreakdown {
        client_split,
        client_split_calls,
        client_map,
        client_map_calls,
        client_plain,
        client_plain_calls,
        server_print,
        server_print_calls,
        server_pipe_print,
        server_pipe_reparse,
        server_pipe_calls,
        normalize_print,
        normalize_calls,
    }
}

#[inline]
pub fn record_visit_program(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    VISIT_PROGRAM.with(|c| c.set(c.get() + d));
}

#[inline]
pub fn record_script_text(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    SCRIPT_TEXT.with(|c| c.set(c.get() + d));
    ST_PARENT_CALLS.with(|c| c.set(c.get() + 1));
}

#[inline]
pub fn record_parent_site(is_pub: bool) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    if is_pub {
        ST_PARENT_SITE_PUB.with(|c| c.set(c.get() + 1));
    } else {
        ST_PARENT_SITE_MAIN.with(|c| c.set(c.get() + 1));
    }
}

/// Tracks how deep the staged function is on the stack, so a re-entrant call
/// is counted rather than inferred.
pub struct EntryGuard(TimerStart);

impl EntryGuard {
    #[expect(clippy::new_without_default, reason = "a guard is never defaulted")]
    pub fn new() -> Self {
        ST_DEPTH.with(|d| {
            let depth = d.get() + 1;
            d.set(depth);
            if depth > 1 {
                ST_NESTED_ENTRIES.with(|c| c.set(c.get() + 1));
            }
        });
        if ST_PARENT_OPEN.with(Cell::get) == 0 {
            ST_ENTRIES_OUTSIDE.with(|c| c.set(c.get() + 1));
        }
        Self(timer_start())
    }
}

/// Marks the span a parent timer covers, so an entry can tell whether it is
/// inside one.
pub struct ParentScope;

impl ParentScope {
    #[expect(clippy::new_without_default, reason = "a guard is never defaulted")]
    pub fn new() -> Self {
        ST_PARENT_OPEN.with(|c| c.set(c.get() + 1));
        Self
    }
}

impl Drop for ParentScope {
    fn drop(&mut self) {
        // Arm A of the instrumentation-cost A/B: the whole body folds away, so
        // the measured difference is the timers plus their recorders, not a subset.
        if cfg!(feature = "measure-no-timers") || !timers_enabled() {
            return;
        }
        ST_PARENT_OPEN.with(|c| c.set(c.get().saturating_sub(1)));
    }
}

impl Drop for EntryGuard {
    fn drop(&mut self) {
        // Arm A of the instrumentation-cost A/B: the whole body folds away, so
        // the measured difference is the timers plus their recorders, not a subset.
        if cfg!(feature = "measure-no-timers") || !timers_enabled() {
            return;
        }
        ST_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
        let elapsed = timer_elapsed(self.0);
        ST_IN_FUNCTION.with(|c| c.set(c.get() + elapsed));
    }
}

#[inline]
pub fn record_st_entry() {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    ST_ENTRIES.with(|c| c.set(c.get() + 1));
}

#[inline]
pub fn record_template_fragment(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    TEMPLATE_FRAGMENT.with(|c| c.set(c.get() + d));
}

#[inline]
pub fn record_assembly_after_fragment(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    ASSEMBLY_AFTER_FRAGMENT.with(|c| c.set(c.get() + d));
}

#[inline]
pub fn record_css_render(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    CSS_RENDER.with(|c| c.set(c.get() + d));
}

#[inline]
pub fn record_codegen(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    CODEGEN.with(|c| c.set(c.get() + d));
}

/// Records into [`ScriptTextBreakdown::process_accumulated`] on drop, so the
/// statement processor's many early returns all get counted.
pub struct ProcessAccumulatedGuard(pub TimerStart);

impl Drop for ProcessAccumulatedGuard {
    fn drop(&mut self) {
        // Arm A of the instrumentation-cost A/B: the whole body folds away, so
        // the measured difference is the timers plus their recorders, not a subset.
        if cfg!(feature = "measure-no-timers") || !timers_enabled() {
            return;
        }
        ST_PROCESS_ACCUMULATED.with(|c| c.set(c.get() + timer_elapsed(self.0)));
        ST_STATEMENTS.with(|c| c.set(c.get() + 1));
    }
}

#[inline]
pub fn record_st_runes(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    ST_RUNES.with(|c| c.set(c.get() + d));
}

/// Records the legacy `$:` branch on drop, which returns from several points.
pub struct ReactiveStmtGuard(pub TimerStart);

impl Drop for ReactiveStmtGuard {
    fn drop(&mut self) {
        // Arm A of the instrumentation-cost A/B: the whole body folds away, so
        // the measured difference is the timers plus their recorders, not a subset.
        if cfg!(feature = "measure-no-timers") || !timers_enabled() {
            return;
        }
        ST_REACTIVE_STMT.with(|c| c.set(c.get() + timer_elapsed(self.0)));
        ST_REACTIVE_CALLS.with(|c| c.set(c.get() + 1));
    }
}

#[inline]
pub fn record_st_prenormalize(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    ST_PRENORMALIZE.with(|c| c.set(c.get() + d));
    ST_CALLS.with(|c| c.set(c.get() + 1));
}

#[inline]
pub fn record_st_collect_vars(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    ST_COLLECT_VARS.with(|c| c.set(c.get() + d));
}

#[inline]
pub fn record_st_line_loop(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    ST_LINE_LOOP.with(|c| c.set(c.get() + d));
}

#[inline]
pub fn record_st_ast_transforms(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    ST_AST_TRANSFORMS.with(|c| c.set(c.get() + d));
}

#[inline]
pub fn record_st_post_passes(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    ST_POST_PASSES.with(|c| c.set(c.get() + d));
}

pub fn take_script_text_breakdown() -> ScriptTextBreakdown {
    ScriptTextBreakdown {
        prenormalize: ST_PRENORMALIZE.with(|c| c.replace(Duration::ZERO)),
        collect_vars: ST_COLLECT_VARS.with(|c| c.replace(Duration::ZERO)),
        line_loop: ST_LINE_LOOP.with(|c| c.replace(Duration::ZERO)),
        ast_transforms: ST_AST_TRANSFORMS.with(|c| c.replace(Duration::ZERO)),
        post_passes: ST_POST_PASSES.with(|c| c.replace(Duration::ZERO)),
        calls: ST_CALLS.with(|c| c.replace(0)),
        process_accumulated: ST_PROCESS_ACCUMULATED.with(|c| c.replace(Duration::ZERO)),
        statements: ST_STATEMENTS.with(|c| c.replace(0)),
        runes: ST_RUNES.with(|c| c.replace(Duration::ZERO)),
        reactive_stmt: ST_REACTIVE_STMT.with(|c| c.replace(Duration::ZERO)),
        reactive_calls: ST_REACTIVE_CALLS.with(|c| c.replace(0)),
        entries: ST_ENTRIES.with(|c| c.replace(0)),
        parent_calls: ST_PARENT_CALLS.with(|c| c.replace(0)),
        nested_entries: ST_NESTED_ENTRIES.with(|c| c.replace(0)),
        parent_site_main: ST_PARENT_SITE_MAIN.with(|c| c.replace(0)),
        parent_site_pub: ST_PARENT_SITE_PUB.with(|c| c.replace(0)),
        in_function: ST_IN_FUNCTION.with(|c| c.replace(Duration::ZERO)),
        entries_outside_parent: ST_ENTRIES_OUTSIDE.with(|c| c.replace(0)),
    }
}

#[inline]
pub fn record_pipeline_parse(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    PL_PARSE.with(|c| c.set(c.get() + d));
}

#[inline]
pub fn record_pipeline_line_offsets(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    PL_LINE_OFFSETS.with(|c| c.set(c.get() + d));
}

#[inline]
pub fn record_pipeline_ensure_script(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    PL_ENSURE_SCRIPT.with(|c| c.set(c.get() + d));
}

#[inline]
pub fn record_pipeline_ts_removal(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    PL_TS_REMOVAL.with(|c| c.set(c.get() + d));
}

#[inline]
pub fn record_pipeline_options_merge(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    PL_OPTIONS_MERGE.with(|c| c.set(c.get() + d));
}

#[inline]
pub fn record_pipeline_analyze(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    PL_ANALYZE.with(|c| c.set(c.get() + d));
}

#[inline]
pub fn record_pipeline_transform(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    PL_TRANSFORM.with(|c| c.set(c.get() + d));
}

/// One compile, timed end to end. Separate from the buckets on purpose: the
/// two are compared, not derived from each other.
#[inline]
pub fn record_pipeline_total(d: Duration) {
    // Arm A of the instrumentation-cost A/B: the whole body folds away, so
    // the measured difference is the timers plus their recorders, not a subset.
    if cfg!(feature = "measure-no-timers") || !timers_enabled() {
        return;
    }
    PL_TOTAL.with(|c| c.set(c.get() + d));
    PL_COMPILES.with(|c| c.set(c.get() + 1));
}

pub fn take_pipeline_breakdown() -> PipelineBreakdown {
    PipelineBreakdown {
        parse: PL_PARSE.with(|c| c.replace(Duration::ZERO)),
        line_offsets: PL_LINE_OFFSETS.with(|c| c.replace(Duration::ZERO)),
        ensure_script: PL_ENSURE_SCRIPT.with(|c| c.replace(Duration::ZERO)),
        ts_removal: PL_TS_REMOVAL.with(|c| c.replace(Duration::ZERO)),
        options_merge: PL_OPTIONS_MERGE.with(|c| c.replace(Duration::ZERO)),
        analyze: PL_ANALYZE.with(|c| c.replace(Duration::ZERO)),
        transform: PL_TRANSFORM.with(|c| c.replace(Duration::ZERO)),
        total: PL_TOTAL.with(|c| c.replace(Duration::ZERO)),
        compiles: PL_COMPILES.with(|c| c.replace(0)),
    }
}

pub fn take_breakdown() -> Phase3Breakdown {
    Phase3Breakdown {
        visit_program: VISIT_PROGRAM.with(|c| c.replace(Duration::ZERO)),
        script_text_transform: SCRIPT_TEXT.with(|c| c.replace(Duration::ZERO)),
        template_fragment: TEMPLATE_FRAGMENT.with(|c| c.replace(Duration::ZERO)),
        assembly_after_fragment: ASSEMBLY_AFTER_FRAGMENT.with(|c| c.replace(Duration::ZERO)),
        css_render: CSS_RENDER.with(|c| c.replace(Duration::ZERO)),
        codegen: CODEGEN.with(|c| c.replace(Duration::ZERO)),
    }
}
