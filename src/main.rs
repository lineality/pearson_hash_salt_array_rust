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
//! ## Heap Usage in This File
//!
//! This is demo / sample-print code, not production. Per project
//! rules, heap (`println!`, `String`, `Vec`) is acceptable here. The
//! production hashing functions called from this file do not
//! themselves allocate; only the demo scaffolding does.
//!
//! ## Error Handling
//!
//! `main` returns `Result<(), std::io::Error>` so that any error
//! from the hashing functions propagates cleanly. None of the calls
//! here should error in practice (inputs are non-empty, salt array
//! is non-empty), but we handle the `Result` explicitly rather than
//! using `unwrap`, per project rules.

// Module declarations. The production module and the tools module
// are both compiled into this binary.

mod pearson_hash_salt_array_rust;
mod pearson_hash_tools;

use std::io::Error;

use pearson_hash_salt_array_rust::{
    GENERATED_TABLE, PEARSON_1990_TABLE, pearson_hash_base, pearson_hash_salt_array,
};

use pearson_hash_tools::{
    evaluate_table, generate_table_fisher_yates, print_comparative_report,
    print_table_evaluation_report,
};

// =============================================================================
// Constants for the demo
// =============================================================================

/// The sample input used throughout the demo.
///
/// Project-level note: chosen to be a short ASCII string that is
/// long enough to exercise the Pearson loop meaningfully (more than
/// one byte) and short enough to keep stdout output readable.
const DEMO_INPUT: &[u8] = b"Hello, World is the first onasei!";

/// Four salts used for the salt-array demo.
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

/// Seeds used in the closing seed-sweep demonstration.
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
// Demo sections
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
/// winner on a chosen criterion; this is illustrative.
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

    // Baseline row: the 1990 table, for comparison.
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
    println!();
}

/// Section 5 (optional): full per-table report for those who want
/// to see the complete `TableQualityReport` rather than the
/// side-by-side delta table.
fn demo_section_full_reports() {
    println!("================================================================");
    println!(" Section 5: full single-table reports");
    println!("================================================================");
    println!();
    print_table_evaluation_report("PEARSON_1990_TABLE", &PEARSON_1990_TABLE);
    print_table_evaluation_report("GENERATED_TABLE (default seed)", &GENERATED_TABLE);
}

// =============================================================================
// main
// =============================================================================

/// Run all demo sections in sequence.
///
/// Project-level note: each section is independent and prints its
/// own header, so the output reads top-to-bottom as a self-explaining
/// transcript. Any error from a hashing call is propagated; the
/// process exits non-zero on error rather than panicking.
fn main() -> Result<(), Error> {
    println!();
    println!("################################################################");
    println!("#                                                              #");
    println!("#   pearson_hash_salt_array_rust  --  demo binary              #");
    println!("#                                                              #");
    println!("#   This is a demonstration of the crate's production          #");
    println!("#   functions and its table-quality measurement tools.         #");
    println!("#                                                              #");
    println!("################################################################");
    println!();

    demo_section_base_hash()?;
    demo_section_salt_array_hash()?;
    demo_section_comparative_report();
    demo_section_seed_sweep();
    demo_section_full_reports();

    println!("================================================================");
    println!(" Demo complete.");
    println!("================================================================");
    println!();

    Ok(())
}
