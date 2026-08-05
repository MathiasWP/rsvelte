//! Counts how often the shipped compiler reuses the retained instance-script
//! program in the runes state transform, and how often it parses the script
//! again instead.
//!
//! The question this answers is whether passing a `RetainedProgram` is enough:
//! the transform also requires the retained source and the pipeline's current
//! text to still agree, so "a program was passed" and "the program was used"
//! are different claims. Only the second one makes the profiler's extra parse
//! pure contamination.
//!
//! Drives `compile()`, i.e. the production entry point — not the phase
//! functions a profiler stages by hand. Requires the instrumentation feature:
//!
//! ```text
//! cargo run --release -p rsvelte_devtools --bin ast_state_reuse_count \
//!   --features measure-ast-state -- <dir>...
//! ```

#[cfg(not(feature = "measure-ast-state"))]
fn main() {
    eprintln!("build with --features measure-ast-state");
    std::process::exit(2);
}

#[cfg(feature = "measure-ast-state")]
fn main() {
    use std::path::{Path, PathBuf};

    use rsvelte_core::{CompileOptions, GenerateMode, compile, measure_ast_state};

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

    let roots: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if roots.is_empty() {
        eprintln!("usage: ast_state_reuse_count <dir>...");
        std::process::exit(1);
    }

    println!(
        "{:<34} {:>7} {:>9} {:>9} {:>8}",
        "root", "files", "reused", "reparsed", "reuse %"
    );
    let (mut all_uses, mut all_reparses, mut all_files) = (0u64, 0u64, 0usize);
    for root in &roots {
        let mut files = Vec::new();
        collect(root, &mut files);
        files.sort();
        measure_ast_state::reset();
        for path in &files {
            let Ok(source) = std::fs::read_to_string(path) else {
                continue;
            };
            for dev in [false, true] {
                let _ = compile(
                    &source,
                    CompileOptions {
                        generate: GenerateMode::Client,
                        dev,
                        filename: Some(path.display().to_string()),
                        ..Default::default()
                    },
                );
            }
        }
        let (uses, reparses) = measure_ast_state::snapshot();
        let total = uses + reparses;
        let pct = if total == 0 {
            f64::NAN
        } else {
            uses as f64 / total as f64 * 100.0
        };
        println!(
            "{:<34} {:>7} {:>9} {:>9} {:>7.1}%",
            root.file_name().map_or_else(
                || root.display().to_string(),
                |n| n.to_string_lossy().into_owned()
            ),
            files.len(),
            uses,
            reparses,
            pct
        );
        all_uses += uses;
        all_reparses += reparses;
        all_files += files.len();
    }

    let total = all_uses + all_reparses;
    println!(
        "\n{all_files} files | reused {all_uses} | reparsed {all_reparses} | reuse {:.1}%",
        if total == 0 {
            f64::NAN
        } else {
            all_uses as f64 / total as f64 * 100.0
        }
    );
    // A zero denominator would make the ratio unreadable rather than perfect.
    if total == 0 {
        println!("nothing reached the site — the ratio above says nothing");
    }
}
