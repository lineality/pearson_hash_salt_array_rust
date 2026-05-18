// src/main.rs

//! # Demo binary for `pearson_hash_salt_array_rust`
//!
//! ## Project-Level Context
//!
//! This is the demonstration entry point for the crate. It is **not**
//! production hashing code, and it is **not** a test suite. Its job
//! is to show, on stdout, what the crate does and how its two
//! permutation tables behave, so a user evaluating the crate or
//! choosing a table for their use case can read concrete numbers and
//! decide for themselves.
//!
//! What this binary demonstrates, in order:
//!
//! 1. The base Pearson hash (`pearson_hash_base`) on a sample input,
//!    run separately with both the 1990 table and the generated
//!    table, so the user can see the two tables produce different
//!    output bytes on the same input.
//!
//! 2. The salt-array Pearson hash (`pearson_hash_salt_array`) on the
//!    same input with a fixed list of four salts, again on both
//!    tables. This is the headline production feature: an N-byte
//!    hash built from one input plus N salts, with no concatenation
//!    and no heap.
//!
//! 3. A side-by-side **measurement report** for both tables,
//!    produced by `pearson_hash_tools::print_comparative_report`.
//!    This is the "measurement, not verdict" surface — the reader
//!    interprets the numbers.
//!
//! 4. A small **seed sweep** that generates several Fisher-Yates
//!    tables with different seeds and prints a one-line summary per
//!    seed, illustrating how a downstream user would shop for a
//!    seed that best fits their criteria.
//!
//! 5. Full per-table reports for those who want to see the complete
//!    `TableQualityReport` rather than the side-by-side delta table.
//!
//! 6. **The table finder** (`table_finder`): builds a corpus of
//!    pseudo-random 8×8 chess boards, sweeps 10,000 Fisher-Yates
//!    seeds against that corpus, refines the top 20 leaders with
//!    single-bit-flip perturbation, and prints the leaderboard.
//!    The leaderboard is also saved to a timestamped file so the
//!    user can come back to it later. The intended workflow is
//!    "run this, read the leaderboard, copy a winning hex seed,
//!    hard-code it in production."
//!
//! ## CLI Flag
//!
//! `--reproducible` (optional, anywhere in `argv`): tells the table
//! finder to use a fixed meta-seed instead of one derived from the
//! system clock. With the flag, the same search trajectory is
//! repeated every run; without it, each run explores a different
//! search region, which is what you want when shopping seeds.
//!
//! ## Heap Usage in This File
//!
//! This is demo / sample-print code, not production. Per project
//! rules, heap (`println!`, `String`, `Vec`) is acceptable here. The
//! production hashing functions called from this file do not
//! themselves allocate; only the demo scaffolding and the
//! tools/finder modules do.
//!
//! ## Error Handling
//!
//! `main` returns `Result<(), std::io::Error>` so that any error
//! from the hashing or finder functions propagates cleanly. None of
//! the calls here should error in practice (inputs are non-empty,
//! salt array is non-empty, corpus is non-empty), but we handle
//! the `Result` explicitly rather than using `unwrap`, per project
//! rules.

// Module declarations. All three modules are compiled into this binary.
mod pearson_hash_salt_array_rust;
mod pearson_hash_tools;
mod table_finder;

use std::env;
use std::io::{self, BufRead, Error, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use pearson_hash_salt_array_rust::{
    GENERATED_TABLE, PEARSON_1990_TABLE, pearson_hash_base, pearson_hash_salt_array,
};

use pearson_hash_tools::{
    evaluate_table, generate_table_fisher_yates, print_comparative_report,
    print_table_evaluation_report,
};

use table_finder::{
    RankBy, SaltArrayConfig, SweepMode, build_chess_corpus_sample, perturb_seed_search,
    print_and_save_search_report, search_seeds,
};

// =============================================================================
// Constants — existing demo (sections 1-5)
// =============================================================================

/// The sample input used throughout sections 1-2.
///
/// Project-level note: chosen to be a short ASCII string that is
/// long enough to exercise the Pearson loop meaningfully (more than
/// one byte) and short enough to keep stdout output readable.
const DEMO_INPUT: &[u8] = b"Hello, World is the first onasei!";

/// Four salts used for the salt-array demo in section 2.
///
/// Project-level note: the values are arbitrary but deliberately
/// well-separated bit patterns so the four output bytes are very
/// unlikely to coincide. In a real application, salts are typically
/// fixed compile-time constants chosen once per use case (e.g.
/// per Bloom-filter instance).
const DEMO_SALTS: [u128; 4] = [
    0x0000_0000_0000_0000_0000_0000_0000_0001,
    0x0000_0000_0000_0000_0000_0000_0000_00FF,
    0xFFFF_FFFF_FFFF_FFFF_0000_0000_0000_0000,
    0xDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0,
];

/// Seeds used in the closing seed-sweep demonstration (section 4).
///
/// Project-level note: these are illustrative only. A real user
/// shopping for a seed would sweep hundreds or thousands of seeds
/// and select by their own criterion (most commonly: lowest
/// XOR-uniformity worst-case chi-square).
const SWEEP_SEEDS: [u64; 8] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_0042,
    0x9E37_79B9_7F4A_7C15, // the crate's default seed
    0xDEAD_BEEF_CAFE_BABE,
    0x0123_4567_89AB_CDEF,
    0xFFFF_FFFF_FFFF_FFFF,
    0x1234_5678_9ABC_DEF0,
    0xA5A5_A5A5_5A5A_5A5A,
];

// =============================================================================
// Constants — table-finder demo (section 6)
// =============================================================================

/// Number of chess-board samples in the finder corpus.
///
/// Project-level note: 5,000 is large enough to produce stable
/// rankings and small enough to keep the phase-1 sweep fast
/// (under a second per thousand seeds on a modern CPU).
const FINDER_CORPUS_SIZE: usize = 5_000;

/// Seed used to build the deterministic chess corpus.
///
/// Project-level note: distinct from the search meta-seed. Fixed
/// so the corpus itself is the same across runs; only the search
/// trajectory varies. This makes results comparable across runs.
const FINDER_CORPUS_SEED: u64 = 0xC4E5_5C0F_FEEC_0DE5;

/// Number of Fisher-Yates seeds the finder evaluates in phase 1.
///
/// Project-level note: 10,000 is a sensible default. Typical
/// runtime is 10–30 seconds. Increase for broader exploration.
const FINDER_SEED_COUNT: u64 = 10_000;

/// Meta-seed used when `--reproducible` is passed.
///
/// Project-level note: without the flag, the meta-seed is derived
/// from `SystemTime` so each run explores a different region of
/// the seed space. With the flag, the meta-seed is fixed so the
/// search trajectory is identical across runs — useful when the
/// user wants to reproduce a previously-seen result.
const FINDER_FIXED_META_SEED: u64 = 0xFEED_FACE_DEAD_BEEF;

/// Top-K leaderboard size for the finder.
const FINDER_TOP_K: usize = 20;

/// Number of worst-performing seeds to also report, for contrast
/// against the top-K. Set to 0 to omit the worst section entirely.
///
/// Project-level note: seeing the worst alongside the best gives the
/// user a concrete sense of the spread across the seed space, and
/// of what a structurally poor table looks like.
const FINDER_BOTTOM_K: usize = 5;

/// Salts used for the salt-array part of the finder.
///
/// Project-level note: four salts is sufficient to demonstrate
/// the salt-array collision metric. Fewer than `MAX_SALTS = 8`,
/// so the remaining internal slots are zero-padded; this
/// padding is documented in `table_finder::SaltArrayConfig`.
const FINDER_SALTS: [u128; 4] = [
    0x0000_0000_0000_0000_0000_0000_0000_0001,
    0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF,
    0xAAAA_AAAA_AAAA_AAAA_5555_5555_5555_5555,
    0xDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0,
];

// =============================================================================
// Helper: format a byte array of any length for printing
// =============================================================================

/// Format a `&[u8]` as a comma-separated, zero-padded hex string.
///
/// Project-level note: heap usage is acceptable in this demo file.
/// This helper exists solely to make the demo output readable.
fn format_hash_bytes(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 6);
    out.push('[');
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("0x{:02X}", b));
    }
    out.push(']');
    out
}

// =============================================================================
// Demo sections 1-5 (the original demos)
// =============================================================================

/// Section 1: base Pearson hash on both tables.
///
/// Project-level note: the same input produces different hash bytes
/// under the two different permutation tables. Either is a valid
/// Pearson hash; they are simply hashes under different (but
/// equally valid) permutations.
fn demo_section_base_hash() -> Result<(), Error> {
    println!("================================================================");
    println!(" Section 1: base Pearson hash (single byte output)");
    println!("================================================================");
    println!("Input bytes: {:?}", DEMO_INPUT);
    println!(
        "Input as text: {:?}",
        std::str::from_utf8(DEMO_INPUT).unwrap_or("<non-UTF-8>")
    );
    println!();

    let hash_1990 = pearson_hash_base(DEMO_INPUT, &PEARSON_1990_TABLE)?;
    let hash_gen = pearson_hash_base(DEMO_INPUT, &GENERATED_TABLE)?;

    println!(
        "  using PEARSON_1990_TABLE   -> 0x{:02X}  ({})",
        hash_1990, hash_1990
    );
    println!(
        "  using GENERATED_TABLE      -> 0x{:02X}  ({})",
        hash_gen, hash_gen
    );
    println!();
    println!("Note: both values are valid 8-bit Pearson hashes; the");
    println!("tables are different permutations so the outputs differ.");
    println!();

    Ok(())
}

/// Section 2: salt-array Pearson hash on both tables.
///
/// Project-level note: this is the headline production function.
/// The output is `[u8; 4]` on the stack — no concatenation, no heap.
fn demo_section_salt_array_hash() -> Result<(), Error> {
    println!("================================================================");
    println!(" Section 2: salt-array Pearson hash (N-byte output)");
    println!("================================================================");
    println!("Input bytes: {:?}", DEMO_INPUT);
    println!("Number of salts: {}", DEMO_SALTS.len());
    println!("Salts:");
    for (i, s) in DEMO_SALTS.iter().enumerate() {
        println!("  [{}] = 0x{:032X}", i, s);
    }
    println!();

    let hashes_1990: [u8; 4] =
        pearson_hash_salt_array(DEMO_INPUT, &DEMO_SALTS, &PEARSON_1990_TABLE)?;
    let hashes_gen: [u8; 4] = pearson_hash_salt_array(DEMO_INPUT, &DEMO_SALTS, &GENERATED_TABLE)?;

    println!(
        "  using PEARSON_1990_TABLE   -> {}",
        format_hash_bytes(&hashes_1990)
    );
    println!(
        "  using GENERATED_TABLE      -> {}",
        format_hash_bytes(&hashes_gen)
    );
    println!();
    println!("Each output byte is one Pearson hash of");
    println!("    (input bytes) followed by (salt[i] as big-endian 16 bytes).");
    println!("No concatenation buffer is allocated; the input is hashed once");
    println!("and each salt continues the running hash state.");
    println!();

    Ok(())
}

/// Section 3: comparative measurement report on both tables.
///
/// Project-level note: this is a **report**, not a verdict. The
/// reader interprets the numbers. The tools module's docstrings
/// document the preferred direction for each metric.
fn demo_section_comparative_report() {
    println!("================================================================");
    println!(" Section 3: comparative measurement report");
    println!("================================================================");
    println!();
    print_comparative_report(
        "PEARSON_1990_TABLE",
        &PEARSON_1990_TABLE,
        "GENERATED_TABLE (default seed)",
        &GENERATED_TABLE,
    );
}

/// Section 4: seed sweep illustrating how to shop for a seed.
///
/// Project-level note: for each seed, we generate a Fisher-Yates
/// table, evaluate it, and print one summary line. A real seed
/// sweep would iterate over thousands of seeds and pick the
/// winner on a chosen criterion; this is illustrative. Section 6
/// performs a real large-scale sweep against a chess-board corpus.
fn demo_section_seed_sweep() {
    println!("================================================================");
    println!(" Section 4: seed sweep (illustrative, 8 seeds)");
    println!("================================================================");
    println!();
    println!(
        "{:>20}  {:>3}  {:>4}  {:>8}  {:>11}  {:>10}",
        "seed", "fix", "long", "mean disp", "XOR worst", "collisions"
    );
    println!(
        "{}",
        "-".repeat(20 + 2 + 3 + 2 + 4 + 2 + 8 + 2 + 11 + 2 + 10)
    );

    let baseline = evaluate_table(&PEARSON_1990_TABLE);
    println!(
        "{:>20}  {:>3}  {:>4}  {:>8.3}  {:>11.2}  {:>10}",
        "[1990 baseline]",
        baseline.fixed_point_count,
        baseline.longest_cycle,
        baseline.displacement.mean_displacement,
        baseline.xor_uniformity.worst_chi_square,
        baseline.empirical_collisions,
    );

    for &seed in SWEEP_SEEDS.iter() {
        let table = generate_table_fisher_yates(seed);
        let r = evaluate_table(&table);
        println!(
            "  0x{:016X}  {:>3}  {:>4}  {:>8.3}  {:>11.2}  {:>10}",
            seed,
            r.fixed_point_count,
            r.longest_cycle,
            r.displacement.mean_displacement,
            r.xor_uniformity.worst_chi_square,
            r.empirical_collisions,
        );
    }

    println!();
    println!("Legend:");
    println!("  fix         = fixed-point count (lower is better; ~1 for random)");
    println!("  long        = longest cycle length");
    println!("  mean disp   = mean |T[i] - i|  (higher is better)");
    println!("  XOR worst   = worst-case chi-square over all XOR differences");
    println!("                (lower is better; most directly relevant to");
    println!("                 Pearson hash behavior)");
    println!("  collisions  = collisions on the fixed corpus (high variance;");
    println!("                 use only as a rough indicator)");
    println!();
    println!("To shop for a seed for your production use, sweep many seeds,");
    println!("call evaluate_table() on each, and select by the metric that");
    println!("matters most for your data. Then hard-code the winning seed.");
    println!("Section 6 demonstrates this at scale against a chess corpus.");
    println!();
}

/// Section 5: full per-table report for both shipped tables.
fn demo_section_full_reports() {
    println!("================================================================");
    println!(" Section 5: full single-table reports");
    println!("================================================================");
    println!();
    print_table_evaluation_report("PEARSON_1990_TABLE", &PEARSON_1990_TABLE);
    print_table_evaluation_report("GENERATED_TABLE (default seed)", &GENERATED_TABLE);
}

// =============================================================================
// Demo section 6: chess-board table finder
// =============================================================================

/// Derive a meta-seed from the system clock.
///
/// Project-level note: when the user does not pass `--reproducible`,
/// we want each run of the finder to explore a different region of
/// the seed space. The seconds-since-epoch value is mixed through a
/// splitmix64 step so neighbouring runtimes give well-separated
/// seeds, not adjacent ones.
fn time_derived_meta_seed() -> u64 {
    let secs: u64 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut s: u64 = secs ^ 0x9E37_79B9_7F4A_7C15;
    s = (s ^ (s >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    s = (s ^ (s >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    s ^ (s >> 31)
}

/// User's chosen mode for the section-6 table-finder run.
///
/// # Project-Level Context
///
/// Section 6 is the most expensive part of the demo (a 10,000-seed
/// sweep, typically 10-30 seconds). Different users want different
/// things from it, and forcing every user through the same code path
/// every time is wasteful. This enum is the result of the section-6
/// prompt and tells `demo_section_table_finder` which branch to take.
enum SectionSixMode {
    /// Run the full random sweep against the chess-board corpus.
    /// This is the default and what every previous version of this
    /// demo did unconditionally.
    FullRandomSweep,

    /// Skip the full sweep. Instead, score the user-supplied 64-bit
    /// seed and its 64 single-bit-flip neighbors, then print the
    /// same report layout. Useful for refining a seed copied out of
    /// a previous run's leaderboard.
    PerturbSpecificSeed(u64),

    /// Skip section 6 entirely. The rest of the demo (sections 1–5)
    /// still runs.
    Skip,
}

/// Interactively prompt the user for their section-6 choice.
///
/// # Project-Level Context
///
/// Section 6 has three sensible modes (see `SectionSixMode`). This
/// function asks one stdin question, optionally a follow-up for the
/// hex seed, and returns the chosen mode. It never panics: any
/// stdin read error is returned as an `io::Error`; any unparseable
/// input falls back to the default mode with a printed notice.
///
/// # Bypass for Non-Interactive Runs
///
/// If `auto_mode` is `true` (typically set by the `--auto` CLI
/// flag), the prompt is skipped entirely and `FullRandomSweep` is
/// returned. This lets the demo run unattended in CI or with stdin
/// redirected to `/dev/null`.
///
/// # Input Format
///
/// Menu choice: one of `1`, `2`, `3`, or empty (= `1`).
/// Hex seed: any 64-bit hex value, with or without `0x`/`0X` prefix,
/// case-insensitive. Leading and trailing whitespace are trimmed.
///
/// # Error Handling
///
/// Returns `io::Result<SectionSixMode>` so a broken stdin pipe
/// propagates as an error rather than crashing the demo. The user's
/// choices themselves never cause an error return; unparseable input
/// falls back to `FullRandomSweep` with a notice printed to stdout.
///
/// # Arguments
///
/// * `auto_mode` — when `true`, bypass the prompt and return
///   `FullRandomSweep`.
///
/// # Returns
///
/// `Ok(SectionSixMode)` on success; `Err(io::Error)` only on stdin
/// I/O failure.
fn prompt_for_section_six_mode(auto_mode: bool) -> io::Result<SectionSixMode> {
    if auto_mode {
        return Ok(SectionSixMode::FullRandomSweep);
    }

    // Render the menu. The default is option 1 so that pressing
    // Enter (the most likely "I just want to see it work" action)
    // produces the historical behavior of this demo.
    println!("----------------------------------------------------------------");
    println!(" Section 6 — table finder. Choose a mode:");
    println!(
        "   [1] (default)  Full random sweep (~{} seeds)",
        FINDER_SEED_COUNT
    );
    println!("   [2]            Perturb a specific seed (you supply 64-bit hex)");
    println!("   [3]            Skip section 6");
    println!("----------------------------------------------------------------");
    print!("Choice [1/2/3] (Enter = 1): ");
    io::stdout().flush()?;

    // Read the menu line. Stdin errors propagate to the caller.
    let mut menu_line = String::new();
    let stdin_handle = io::stdin();
    stdin_handle.lock().read_line(&mut menu_line)?;
    let menu_choice = menu_line.trim();

    match menu_choice {
        // Empty input or "1" -> default behavior.
        "" | "1" => Ok(SectionSixMode::FullRandomSweep),

        // Explicit skip.
        "3" => Ok(SectionSixMode::Skip),

        // Perturb-specific-seed path: read a second line for the hex.
        "2" => {
            print!(
                "Enter 64-bit seed in hex (with or without 0x prefix, \
                 e.g. 0xDEADBEEFCAFEBABE): "
            );
            io::stdout().flush()?;

            let mut hex_line = String::new();
            stdin_handle.lock().read_line(&mut hex_line)?;

            // Normalize: trim whitespace, strip optional 0x/0X prefix.
            let trimmed = hex_line
                .trim()
                .trim_start_matches("0x")
                .trim_start_matches("0X");

            match u64::from_str_radix(trimmed, 16) {
                Ok(parsed_seed) => {
                    println!("Using seed 0x{:016X} for perturbation.", parsed_seed);
                    Ok(SectionSixMode::PerturbSpecificSeed(parsed_seed))
                }
                Err(_) => {
                    // Per project rules: handle and move on. Do not
                    // panic on bad user input; tell the user what
                    // happened and fall back to the default.
                    println!(
                        "Could not parse '{}' as 64-bit hex. \
                         Falling back to full random sweep.",
                        trimmed
                    );
                    Ok(SectionSixMode::FullRandomSweep)
                }
            }
        }

        // Anything else: notify and fall back.
        other => {
            println!(
                "Unrecognized menu choice '{}'. Falling back to full random sweep.",
                other
            );
            Ok(SectionSixMode::FullRandomSweep)
        }
    }
}

/// Section 6: search for, or refine, a Pearson permutation table
/// against an 8×8 chess-board byte-array corpus.
///
/// # Project-Level Context
///
/// This section is the demo's headline use case: given a corpus
/// representative of the user's real data, find a Fisher-Yates seed
/// whose generated permutation table performs well on that corpus.
///
/// The function dispatches on `mode`:
///   - `FullRandomSweep`        → call `search_seeds` (the original
///                                three-phase pipeline).
///   - `PerturbSpecificSeed(s)` → call `perturb_seed_search` (only
///                                Phase 3, against the user's seed).
///   - `Skip`                   → print one line and return.
///
/// Both real branches print the same report layout, because both
/// produce the same `SearchReport` struct.
///
/// # Arguments
///
/// * `reproducible` — if `true`, the full sweep uses a fixed
///   meta-seed so two runs produce identical results. Ignored in
///   the `PerturbSpecificSeed` branch (the user's seed is the
///   determinism source there).
/// * `mode` — what the user asked for at the section-6 prompt.
///
/// # Returns
///
/// `Ok(())` on success; `Err(io::Error)` if corpus construction,
/// scoring, or file-saving fails.
fn demo_section_table_finder(reproducible: bool, mode: SectionSixMode) -> Result<(), Error> {
    println!("================================================================");
    println!(" Section 6: Pearson table search for chess-board corpus");
    println!("================================================================");
    println!();

    // Early exit for "skip" — printed message keeps the section's
    // overall narrative consistent.
    if matches!(mode, SectionSixMode::Skip) {
        println!("Section 6 skipped by user request.");
        println!();
        return Ok(());
    }

    // Build the shared chess-board corpus. Both real modes need it.
    println!("Building chess-board corpus...");
    let boards = build_chess_corpus_sample(FINDER_CORPUS_SEED, FINDER_CORPUS_SIZE);
    let corpus: Vec<&[u8]> = boards.iter().map(|board| board.as_slice()).collect();
    println!(
        "  {} boards, {} bytes each (8x8 squares, chess alphabet)",
        corpus.len(),
        corpus[0].len()
    );

    println!(
        "  Leaderboard: top {} best, bottom {} worst",
        FINDER_TOP_K, FINDER_BOTTOM_K
    );

    // Salt-array configuration is the same for both real modes so
    // the report's `salt coll` column has the same meaning.
    let salt_config = SaltArrayConfig {
        salts: FINDER_SALTS.to_vec(),
    };

    // Dispatch on the user's mode.
    let report = match mode {
        SectionSixMode::FullRandomSweep => {
            // Meta-seed selection: fixed if --reproducible, otherwise
            // derived from the system clock so each run explores a
            // different region of the seed space.
            let meta_seed: u64 = if reproducible {
                FINDER_FIXED_META_SEED
            } else {
                time_derived_meta_seed()
            };
            println!(
                "Full random sweep: {} seeds, meta_seed = 0x{:016X}{}",
                FINDER_SEED_COUNT,
                meta_seed,
                if reproducible {
                    " (reproducible)"
                } else {
                    " (time-derived; different each run)"
                },
            );
            println!(
                "This typically takes 10–30 seconds. Pass --reproducible to make\n\
                 it deterministic; pass --auto to skip the menu in CI."
            );
            println!();

            search_seeds(
                &corpus,
                Some(&salt_config),
                SweepMode::PseudoRandom {
                    meta_seed,
                    count: FINDER_SEED_COUNT,
                },
                RankBy::BaseCollisions,
                FINDER_TOP_K,
                FINDER_BOTTOM_K,
            )?
        }

        SectionSixMode::PerturbSpecificSeed(base_seed) => {
            println!(
                "Perturbation-only mode: base seed 0x{:016X}, 64 single-bit flips.",
                base_seed
            );
            println!(
                "Total seeds evaluated this run: 65 (base + 64 neighbors).\n\
                 Typical runtime: well under one second."
            );
            println!();

            perturb_seed_search(
                base_seed,
                &corpus,
                Some(&salt_config),
                RankBy::BaseCollisions,
                FINDER_TOP_K,
                FINDER_BOTTOM_K,
            )?
        }

        // Already returned above; the compiler requires the arm.
        SectionSixMode::Skip => unreachable!("Skip handled with early return"),
    };

    print_and_save_search_report(&report)?;
    Ok(())
}

// =============================================================================
// CLI parsing
// =============================================================================

/// Test whether a given flag string appears anywhere in `argv`.
///
/// # Project-Level Context
///
/// The demo binary supports two independent boolean flags:
///   `--reproducible` — finder uses a fixed meta-seed, so two runs
///                      produce identical results.
///   `--auto`         — skip the interactive section-6 prompt and
///                      go straight to a full random sweep. Intended
///                      for non-interactive runs (CI, scripted
///                      benchmarking, redirected stdin).
///
/// This helper exists so we can ask "is `--foo` present?" without
/// pulling in a third-party CLI crate. No argument values or
/// positional parsing is needed for this demo.
///
/// # Arguments
///
/// * `flag_name` — the flag to look for, including its leading `--`.
///
/// # Returns
///
/// `true` if any `argv` element equals `flag_name`, otherwise `false`.
fn cli_flag_is_set(flag_name: &str) -> bool {
    env::args().any(|argument| argument == flag_name)
}

// =============================================================================
// main
// =============================================================================

/// Run all demo sections in sequence.
///
/// Project-level note: each section is independent and prints its
/// own header, so the output reads top-to-bottom as a self-explaining
/// transcript. Any error from a hashing or finder call is propagated;
/// the process exits non-zero on error rather than panicking.
fn main() -> Result<(), Error> {
    let reproducible = cli_flag_is_set("--reproducible");
    let auto_mode = cli_flag_is_set("--auto");

    println!();
    println!("################################################################");
    println!("#                                                              #");
    println!("#   pearson_hash_salt_array_rust  --  demo binary              #");
    println!("#                                                              #");
    println!("#   This is a demonstration of the crate's production          #");
    println!("#   functions and its table-quality measurement tools.         #");
    println!("#                                                              #");
    if reproducible {
        println!("#   Flag detected: --reproducible (fixed finder meta-seed)     #");
    } else {
        println!("#   Tip: pass --reproducible to use a fixed finder meta-seed   #");
    }
    if auto_mode {
        println!("#   Flag detected: --auto (skip section-6 prompt)              #");
    } else {
        println!("#   Tip: pass --auto to skip the section-6 prompt              #");
    }
    println!("#                                                              #");
    println!("################################################################");
    println!();

    demo_section_base_hash()?;
    demo_section_salt_array_hash()?;
    demo_section_comparative_report();
    demo_section_seed_sweep();
    demo_section_full_reports();

    // Ask the user what to do for section 6 (or auto-default).
    let section_six_mode = prompt_for_section_six_mode(auto_mode)?;
    demo_section_table_finder(reproducible, section_six_mode)?;

    println!("================================================================");
    println!(" Demo complete.");
    println!("================================================================");
    println!();

    Ok(())
}
