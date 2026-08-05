//! Repository-local instrumentation for the runes state transform's reuse of
//! the already-parsed instance script.
//!
//! `transform_instance_script_for_visitors` is handed the `RetainedProgram` the
//! prepare step parsed. It can only reuse it when the retained source and the
//! text the pipeline has reached still agree, so passing one is necessary but
//! not sufficient — the two counters here separate "reused" from "re-parsed"
//! instead of assuming the first.
//!
//! This measures the production path, not a counterfactual: `reparses` is the
//! number of extra full parses of the instance script the shipped compiler
//! performs. Compiled out unless `measure-ast-state` is enabled.

use std::cell::Cell;

thread_local! {
    static RETAINED_USES: Cell<u64> = const { Cell::new(0) };
    static REPARSES: Cell<u64> = const { Cell::new(0) };
}

/// The retained program was usable: the state transform read it in place.
pub fn record_retained_use() {
    RETAINED_USES.with(|c| c.set(c.get() + 1));
}

/// The retained program was absent or unusable: the state transform parsed the
/// script again.
pub fn record_reparse() {
    REPARSES.with(|c| c.set(c.get() + 1));
}

/// `(retained uses, re-parses)` accumulated on this thread.
pub fn snapshot() -> (u64, u64) {
    (RETAINED_USES.with(Cell::get), REPARSES.with(Cell::get))
}

pub fn reset() {
    RETAINED_USES.with(|c| c.set(0));
    REPARSES.with(|c| c.set(0));
}
