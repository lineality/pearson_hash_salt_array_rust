// src/pearson_hash_tools.rs

//! # `pearson_hash_tools` — Permutation-Table Generation & Quality Measurement
//!
//! ## Project-Level Context
//!
//! This module is the **measurement and tooling companion** to
//! `pearson_hash_salt_array_rust.rs`. The production module ships
//! two fixed permutation tables and the hashing functions that use
//! them. This module provides:
//!
//! 1. Independent (intentionally redundant) generators for
//!    permutation tables.
//! 2. Six statistical metrics describing the quality of any
//!    256-byte permutation when used as a Pearson hash table.
//! 3. Human-readable reporting and side-by-side comparison
//!    utilities that **print** measurements for the operator to
//!    interpret.
//!
//! ## Design Posture: Measurement, Not Verdict
//!
//! This is the single most important point about this module:
//!
//! > **Metrics produce measurements. Verdicts are a policy
//! > decision that belongs to the caller.**
//!
//! The metrics here measure structural properties of a permutation
//! (fixed points, cycle structure, displacement, sequential
//! correlation, XOR uniformity) and one empirical property
//! (collision count on a fixed corpus). None of these numbers, in
//! isolation, declares a table "good" or "bad." Pearson himself
//! observed in the 1990 paper that random permutations score
//! similarly on aggregate metrics but vary on any specific small
//! corpus.
//!
//! Accordingly:
//!
//! - **Cargo tests in this module verify code correctness only.**
//!   Every test uses inputs whose correct outputs are known a
//!   priori (identity, reverse, single-swap, etc.). No test passes
//!   or fails based on whether one random table happens to beat
//!   another on a noisy metric.
//!
//! - **Reports are printed for human consumption** via the
//!   `print_*` functions in this module. The reader interprets the
//!   numbers, ideally against a baseline such as the 1990 table.
//!
//! - **`compare_against` and `ComparisonVerdict` remain in the
//!   public API** as a convenience for users who want to write
//!   their own thresholded gates with their own policy. This
//!   crate's own tests do not call them as gates.
//!
//! ## Choosing a Seed for Production
//!
//! A typical downstream use is "find a permutation table that
//! suits my use case." The recommended workflow:
//!
//! 1. Identify which metric matters most for your data
//!    (XOR uniformity is the most directly relevant to Pearson
//!    hash behavior; empirical collisions on **your own** corpus
//!    is the most directly relevant to your application).
//! 2. Write a small program that sweeps seeds, calls
//!    `evaluate_table` on each generated table, and selects by
//!    your chosen metric.
//! 3. Hard-code the winning seed (and the resulting table) into
//!    your build.
//!
//! ## Deliberate Redundancy with the Production Module
//!
//! Both `generate_table_fisher_yates_const` and the 1990 table are
//! also defined in `pearson_hash_salt_array_rust.rs`. That
//! redundancy is intentional and required: each file must be
//! self-contained so either can be lifted out independently.
//!
//! ## Heap Usage
//!
//! This is **tools / development code**. Heap (`Vec`, `String`,
//! `format!`, `println!`) is used freely here for reporting and
//! for the empirical-collision corpus. None of this code is on
//! the production hash path.

use std::fmt;
use std::io::{Error, ErrorKind};

// =============================================================================
// SECTION 1: Local copies — deliberate redundancy with the production module
// =============================================================================

/// Local copy of the 1990 Pearson table.
///
/// Identical to `PEARSON_1990_TABLE` in
/// `pearson_hash_salt_array_rust.rs`. Tools must not import the
/// production table — the two files must be independently liftable.
/// A cargo test (`test_local_1990_table_copy_is_valid_permutation`)
/// verifies this copy is a valid permutation.
/// human-hand-copied from Pearson's paper: dl.acm.org/doi/epdf/10.1145/78973.78978
pub const PEARSON_1990_TABLE_COPY: [u8; 256] = [
    1, 87, 49, 12, 176, 178, 102, 166, 121, 193, 6, 84, 249, 230, 44, 163, 14, 197, 213, 181, 161,
    85, 218, 80, 64, 239, 24, 226, 236, 142, 38, 200, 110, 177, 104, 103, 141, 253, 255, 50, 77,
    101, 81, 18, 45, 96, 31, 222, 25, 107, 190, 70, 86, 237, 240, 34, 72, 242, 20, 214, 244, 227,
    149, 235, 97, 234, 57, 22, 60, 250, 82, 175, 208, 5, 127, 199, 111, 62, 135, 248, 174, 169,
    211, 58, 66, 154, 106, 195, 245, 171, 17, 187, 182, 179, 0, 243, 132, 56, 148, 75, 128, 133,
    158, 100, 130, 126, 91, 13, 153, 246, 216, 219, 119, 68, 223, 78, 83, 88, 201, 99, 122, 11, 92,
    32, 136, 114, 52, 10, 138, 30, 48, 183, 156, 35, 61, 26, 143, 74, 251, 94, 129, 162, 63, 152,
    170, 7, 115, 167, 241, 206, 3, 150, 55, 59, 151, 220, 90, 53, 23, 131, 125, 173, 15, 238, 79,
    95, 89, 16, 105, 137, 225, 224, 217, 160, 37, 123, 118, 73, 2, 157, 46, 116, 9, 145, 134, 228,
    207, 212, 202, 215, 69, 229, 27, 188, 67, 124, 168, 252, 42, 4, 29, 108, 21, 247, 19, 205, 39,
    203, 233, 40, 186, 147, 198, 192, 155, 33, 164, 191, 98, 204, 165, 180, 117, 76, 140, 36, 210,
    172, 41, 54, 159, 8, 185, 232, 113, 196, 231, 47, 146, 120, 51, 65, 28, 144, 254, 221, 93, 189,
    194, 139, 112, 43, 71, 109, 184, 209,
];

// =============================================================================
// SECTION 2: Table generators (compile-time and runtime)
// =============================================================================

/// Compile-time Fisher-Yates table generator.
///
/// ## Project-Level Context
///
/// Identical algorithm and PRNG (`splitmix64`) to the production
/// module's generator. The function name differs to avoid any
/// possibility of glob-import collision when both modules are in
/// scope from a binary.
///
/// See `generate_table_fisher_yates` (the runtime variant) for the
/// full algorithmic and PRNG discussion. This is a `const fn` copy
/// of the same logic so tables can be embedded at compile time.
pub const fn generate_table_fisher_yates_const(seed: u64) -> [u8; 256] {
    let mut table: [u8; 256] = [0u8; 256];
    let mut init_index: usize = 0;
    while init_index < 256 {
        table[init_index] = init_index as u8;
        init_index += 1;
    }

    let mut prng_state: u64 = seed;
    let mut high_index: usize = 255;
    while high_index > 0 {
        prng_state = prng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z: u64 = prng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z = z ^ (z >> 31);

        let swap_target: usize = (z % ((high_index as u64) + 1)) as usize;

        let temp_value: u8 = table[high_index];
        table[high_index] = table[swap_target];
        table[swap_target] = temp_value;

        high_index -= 1;
    }

    table
}

/// Runtime Fisher-Yates table generator.
///
/// ## Project-Level Context
///
/// Same algorithm as `generate_table_fisher_yates_const`, but as a
/// regular function. Useful for sweeping many seeds at runtime to
/// shop for a table whose measurements suit a specific use case.
///
/// ## Algorithm: Fisher-Yates / Knuth Shuffle
///
/// Fisher-Yates and Knuth shuffle are the same algorithm — Knuth
/// rediscovered and popularized Fisher and Yates's 1938 method:
///
/// ```text
///     for i from n-1 downto 1:
///         j = uniform_random_integer_in(0, i)   // inclusive
///         swap(a[i], a[j])
/// ```
///
/// Properties:
///
/// - Produces every permutation with equal probability when the
///   PRNG is uniform.
/// - Is a permutation by construction. We start from the identity
///   (a permutation) and every swap preserves the multiset of
///   values, so duplicates are impossible. No extra "ensure no
///   repeats" step is required.
/// - `O(n)` time, `O(1)` extra space.
///
/// ## PRNG: splitmix64
///
/// ```text
///     state = state + 0x9E3779B97F4A7C15
///     z = state
///     z = (z XOR (z >> 30)) * 0xBF58476D1CE4E5B9
///     z = (z XOR (z >> 27)) * 0x94D049BB133111EB
///     z = z XOR (z >> 31)
/// ```
///
/// `splitmix64` is small, well-studied, passes BigCrush, and is
/// trivial to implement without dependencies. It is not
/// cryptographic; that is fine because the resulting table is
/// public.
///
/// ## Arguments
///
/// * `seed` — 64-bit PRNG seed. Same seed yields same table.
///
/// ## Returns
///
/// A `[u8; 256]` permutation, guaranteed valid by construction.
pub fn generate_table_fisher_yates(seed: u64) -> [u8; 256] {
    let mut table: [u8; 256] = [0u8; 256];
    for init_index in 0..256usize {
        table[init_index] = init_index as u8;
    }

    let mut prng_state: u64 = seed;
    // Bounded loop: exactly 255 iterations.
    for high_index in (1usize..=255).rev() {
        prng_state = prng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z: u64 = prng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z = z ^ (z >> 31);

        let swap_target: usize = (z % ((high_index as u64) + 1)) as usize;
        table.swap(high_index, swap_target);
    }

    table
}

/// Heap-free check that `table` is a permutation of `0..=255`.
///
/// ## Project-Level Context
///
/// Local copy of the production module's `is_valid_permutation`
/// (renamed to avoid glob-import collision). Uses a 32-byte stack
/// bitmap; allocates nothing. Safe to call from any context.
pub fn is_valid_permutation_tools(table: &[u8; 256]) -> bool {
    let mut presence_bitmap: [u8; 32] = [0u8; 32];
    for &value in table.iter() {
        let byte_index: usize = (value as usize) >> 3;
        let bit_mask: u8 = 1u8 << ((value as usize) & 7);
        if (presence_bitmap[byte_index] & bit_mask) != 0 {
            return false;
        }
        presence_bitmap[byte_index] |= bit_mask;
    }
    true
}

// =============================================================================
// SECTION 3: Local Pearson hash (for the collision-count metric only)
// =============================================================================

/// Local Pearson hash used inside `empirical_collision_count`.
///
/// Tools must not import the production hash. This is a minimal
/// implementation used only by the empirical metric.
fn local_pearson_hash(input: &[u8], table: &[u8; 256]) -> Result<u8, Error> {
    if input.is_empty() {
        return Err(Error::new(ErrorKind::InvalidInput, "TOOL-PH: empty input"));
    }
    let mut running_hash: u8 = 0;
    for &byte in input {
        running_hash = table[(running_hash ^ byte) as usize];
    }
    Ok(running_hash)
}

// =============================================================================
// SECTION 4: Individual metric functions
// =============================================================================

/// Metric 1: count of fixed points (`T[i] == i`).
///
/// ## What It Measures
///
/// A fixed point is a position where the table maps a value to
/// itself. Identity = 256 fixed points. A random permutation of
/// 256 elements has expected fixed-point count of exactly 1
/// (derangement formula in the large-n limit).
///
/// ## Interpretation
///
/// Lower is generally better. Identity-like tables (many fixed
/// points) collapse the Pearson update step toward a plain XOR
/// checksum on the affected positions.
pub fn count_fixed_points(table: &[u8; 256]) -> usize {
    let mut fixed_count: usize = 0;
    for i in 0..256usize {
        if (table[i] as usize) == i {
            fixed_count += 1;
        }
    }
    fixed_count
}

/// Metric 2: cycle decomposition.
///
/// ## What It Measures
///
/// Every permutation decomposes uniquely into disjoint cycles.
/// E.g. `T[3]=7, T[7]=12, T[12]=3` is a 3-cycle. The cycle lengths
/// sum to 256.
///
/// ## Interpretation
///
/// - Identity: 256 one-cycles. Worst case.
/// - Random permutation: mix of cycle lengths, longest cycle
///   averaging ~62% of n (Golomb-Dickman constant).
/// - Many short cycles → local patterns → poor Pearson dispersal.
///
/// ## Returns
///
/// `Vec<usize>` of cycle lengths, sorted descending. Sum is 256.
pub fn cycle_structure(table: &[u8; 256]) -> Vec<usize> {
    let mut visited_bitmap: [u8; 32] = [0u8; 32];
    let mut cycle_lengths: Vec<usize> = Vec::new();

    for start_index in 0..256usize {
        let byte_index: usize = start_index >> 3;
        let bit_mask: u8 = 1u8 << (start_index & 7);
        if (visited_bitmap[byte_index] & bit_mask) != 0 {
            continue;
        }

        let mut current_index: usize = start_index;
        let mut current_length: usize = 0;
        // Bounded: cycle length cannot exceed 256.
        for _step in 0..=256usize {
            let cur_byte: usize = current_index >> 3;
            let cur_mask: u8 = 1u8 << (current_index & 7);
            if (visited_bitmap[cur_byte] & cur_mask) != 0 {
                break;
            }
            visited_bitmap[cur_byte] |= cur_mask;
            current_index = table[current_index] as usize;
            current_length += 1;
        }
        cycle_lengths.push(current_length);
    }

    cycle_lengths.sort_unstable_by(|a, b| b.cmp(a));
    cycle_lengths
}

/// Summary of displacement statistics for a table.
#[derive(Debug, Clone)]
pub struct DisplacementStats {
    /// Minimum `|T[i] - i|`.
    pub min_displacement: u32,
    /// Maximum `|T[i] - i|`.
    pub max_displacement: u32,
    /// Mean `|T[i] - i|`.
    pub mean_displacement: f64,
    /// Count of positions where `T[i] == i` (same as fixed-point
    /// count; reported here for completeness of the displacement
    /// view).
    pub zero_displacement_count: usize,
}

/// Metric 3: displacement statistics (`|T[i] - i|`).
///
/// ## What It Measures
///
/// How far each value is moved from its starting position. Large
/// mean displacement indicates the permutation scatters values
/// widely. Identity → all zeros. Reverse table → mean 127.5.
///
/// Absolute (not modular) distance, matching the project scope.
pub fn displacement_stats(table: &[u8; 256]) -> DisplacementStats {
    let mut min_d: u32 = u32::MAX;
    let mut max_d: u32 = 0;
    let mut sum_d: u64 = 0;
    let mut zero_count: usize = 0;

    for i in 0..256usize {
        let t_val: i32 = table[i] as i32;
        let i_val: i32 = i as i32;
        let displacement: u32 = (t_val - i_val).unsigned_abs();

        if displacement < min_d {
            min_d = displacement;
        }
        if displacement > max_d {
            max_d = displacement;
        }
        sum_d += displacement as u64;
        if displacement == 0 {
            zero_count += 1;
        }
    }

    DisplacementStats {
        min_displacement: min_d,
        max_displacement: max_d,
        mean_displacement: (sum_d as f64) / 256.0,
        zero_displacement_count: zero_count,
    }
}

/// Metric 4: absolute Pearson product-moment correlation between
/// `T[i]` and `T[i+1]`.
///
/// ## What It Measures
///
/// If a permutation is well-randomized, knowing `T[i]` should
/// give no information about `T[i+1]`. The Pearson correlation
/// coefficient over the 255 pairs `(T[i], T[i+1])` quantifies
/// linear dependence:
///
/// - `+1.0`: perfect positive correlation (e.g. identity).
/// - `-1.0`: perfect negative correlation (e.g. reverse).
/// - `~0.0`: no linear correlation (good).
///
/// We report the absolute value; smaller is better.
///
/// ## Naming
///
/// "Pearson correlation" here refers to Karl Pearson the
/// statistician, unrelated to Peter Pearson the hash author.
/// Coincidence of names.
pub fn sequential_correlation_abs(table: &[u8; 256]) -> f64 {
    let pair_count: f64 = 255.0;
    let mut sum_x: f64 = 0.0;
    let mut sum_y: f64 = 0.0;
    for i in 0..255usize {
        sum_x += table[i] as f64;
        sum_y += table[i + 1] as f64;
    }
    let mean_x: f64 = sum_x / pair_count;
    let mean_y: f64 = sum_y / pair_count;

    let mut covariance: f64 = 0.0;
    let mut variance_x: f64 = 0.0;
    let mut variance_y: f64 = 0.0;
    for i in 0..255usize {
        let dx: f64 = (table[i] as f64) - mean_x;
        let dy: f64 = (table[i + 1] as f64) - mean_y;
        covariance += dx * dy;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }

    let denominator: f64 = (variance_x * variance_y).sqrt();
    if denominator == 0.0 {
        // Degenerate (constant) sequence — would not be a valid
        // permutation; return worst-case magnitude.
        return 1.0;
    }

    (covariance / denominator).abs()
}

/// Summary of XOR-uniformity (differential-distribution) results.
#[derive(Debug, Clone)]
pub struct XorUniformityStats {
    /// Worst-case chi-square statistic across all nonzero
    /// differences `d`. Lower is better. With 256 samples and 256
    /// buckets, expected count per bucket is 1.0.
    pub worst_chi_square: f64,
    /// The difference `d` that produced the worst-case chi-square.
    pub worst_difference: u8,
    /// Mean chi-square across all 255 nonzero differences.
    pub mean_chi_square: f64,
}

/// Metric 5: XOR uniformity (differential distribution).
///
/// ## What It Measures
///
/// For each nonzero `d in 1..=255`, examine the multiset:
///
/// ```text
///     S_d = { T[i] XOR T[i XOR d] : i in 0..256 }
/// ```
///
/// If `T` were ideal for Pearson hashing, `S_d` would cover
/// `0..=255` uniformly. We measure non-uniformity via chi-square
/// against the uniform distribution (expected count 1.0 per
/// bucket): `chi_d = sum_b (observed_b - 1)^2`.
///
/// We return the worst-case `chi_d`, the `d` that produced it, and
/// the mean over all nonzero `d`. This is the metric most directly
/// relevant to Pearson-hash behavior, since the algorithm's
/// avalanche behavior is governed entirely by XOR-uniformity of
/// the table.
///
/// ## Identity Special Case (used in tests)
///
/// For the identity, `T[i] ^ T[i ^ d] = i ^ (i ^ d) = d` for all
/// `i`. So the histogram for difference `d` has count 256 at
/// position `d` and 0 elsewhere. Per-bucket chi-square contribution:
/// 255 buckets contribute `(0-1)^2 = 1` each, and one bucket
/// contributes `(256-1)^2 = 65025`. Total per `d`: `255 + 65025 =
/// 65280`. Same for every `d`, so worst = mean = 65280.
pub fn xor_uniformity(table: &[u8; 256]) -> XorUniformityStats {
    let mut worst_chi: f64 = 0.0;
    let mut worst_d: u8 = 1;
    let mut sum_chi: f64 = 0.0;

    for d in 1u16..=255u16 {
        let d_u8: u8 = d as u8;

        let mut histogram: [u16; 256] = [0u16; 256];
        for i in 0u16..=255u16 {
            let i_u8: u8 = i as u8;
            let other: u8 = i_u8 ^ d_u8;
            let output_diff: u8 = table[i_u8 as usize] ^ table[other as usize];
            histogram[output_diff as usize] += 1;
        }

        let mut chi_square: f64 = 0.0;
        for bucket in 0..256usize {
            let observed: f64 = histogram[bucket] as f64;
            let delta: f64 = observed - 1.0;
            chi_square += delta * delta;
        }

        sum_chi += chi_square;
        if chi_square > worst_chi {
            worst_chi = chi_square;
            worst_d = d_u8;
        }
    }

    XorUniformityStats {
        worst_chi_square: worst_chi,
        worst_difference: worst_d,
        mean_chi_square: sum_chi / 255.0,
    }
}

/// Metric 6: empirical collision count over a fixed structured
/// corpus.
///
/// ## What It Measures
///
/// The five preceding metrics measure structural properties of the
/// table. This one measures actual Pearson-hash collisions when
/// the table is used to hash a fixed corpus of inputs.
///
/// ## Statistical Caveat
///
/// On a fixed corpus of a few hundred inputs into 256 buckets,
/// the collision count is high-variance birthday-paradox noise.
/// Different "equally good" permutations will produce collision
/// counts that differ by tens. A small difference here is not
/// meaningful evidence that one table is better than another.
/// For an apples-to-apples comparison, run this metric on
/// **your own** representative corpus.
///
/// ## Fixed Corpus
///
/// Fixed in code so different tables can be compared on identical
/// data. Contains:
///
/// 1. All 256 single-byte inputs.
/// 2. 2-byte inputs from a 16×16 grid (256 inputs).
/// 3. A list of short English words.
/// 4. Single-bit-flipped variants of a few base strings.
///
/// ## Returns
///
/// Total number of unordered colliding pairs across the corpus.
pub fn empirical_collision_count(table: &[u8; 256]) -> usize {
    let mut corpus: Vec<Vec<u8>> = Vec::with_capacity(800);

    // (1) All single-byte inputs.
    for b in 0u16..=255u16 {
        corpus.push(vec![b as u8]);
    }

    // (2) 16x16 grid of 2-byte inputs.
    for i in 0u8..16u8 {
        for j in 0u8..16u8 {
            corpus.push(vec![i, j]);
        }
    }

    // (3) Short English words.
    let word_list: [&[u8]; 24] = [
        b"the",
        b"and",
        b"of",
        b"to",
        b"in",
        b"is",
        b"it",
        b"that",
        b"hash",
        b"hashing",
        b"pearson",
        b"table",
        b"permutation",
        b"collision",
        b"input",
        b"output",
        b"random",
        b"function",
        b"byte",
        b"bytes",
        b"value",
        b"values",
        b"index",
        b"key",
    ];
    for w in word_list.iter() {
        corpus.push(w.to_vec());
    }

    // (4) Single-bit-flipped variants of a few base strings.
    let base_strings: [&[u8]; 4] = [b"alpha", b"bravo", b"charlie", b"delta"];
    for base in base_strings.iter() {
        corpus.push(base.to_vec());
        let last_index: usize = base.len() - 1;
        for bit in 0u8..8u8 {
            let mut variant: Vec<u8> = base.to_vec();
            variant[last_index] ^= 1u8 << bit;
            corpus.push(variant);
        }
    }

    let mut hashes: Vec<u8> = Vec::with_capacity(corpus.len());
    for entry in corpus.iter() {
        match local_pearson_hash(entry, table) {
            Ok(h) => hashes.push(h),
            Err(_) => continue, // Empty entry: never occurs in fixed corpus.
        }
    }

    let mut count_per_bucket: [u32; 256] = [0u32; 256];
    for &h in hashes.iter() {
        count_per_bucket[h as usize] += 1;
    }

    let mut collision_pairs: usize = 0;
    for &c in count_per_bucket.iter() {
        if c >= 2 {
            let c_usize: usize = c as usize;
            collision_pairs += c_usize * (c_usize - 1) / 2;
        }
    }

    collision_pairs
}

// =============================================================================
// SECTION 5: TableQualityReport — aggregation of all six metrics
// =============================================================================

/// Aggregate measurement report for a single permutation table.
///
/// ## Project-Level Context
///
/// One call to `evaluate_table` produces one of these. The struct
/// is a **measurement record**, not a verdict.
#[derive(Debug, Clone)]
pub struct TableQualityReport {
    pub is_valid_permutation: bool,
    pub fixed_point_count: usize,
    pub cycle_lengths: Vec<usize>,
    pub one_cycle_count: usize,
    pub two_cycle_count: usize,
    pub longest_cycle: usize,
    pub displacement: DisplacementStats,
    pub sequential_correlation_abs: f64,
    pub xor_uniformity: XorUniformityStats,
    pub empirical_collisions: usize,
}

/// Verdict from `compare_against`.
///
/// ## Project-Level Context
///
/// This type and `compare_against` are **convenience for callers**
/// who want automated gating with the policy implemented here.
/// This crate's own cargo tests do not use them as gates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComparisonVerdict {
    /// Candidate matches or exceeds baseline on every metric
    /// (within the documented tolerances of `compare_against`).
    AtLeastAsGood,
    /// Candidate falls short on one or more metrics. Strings list
    /// the offending metrics with values.
    Worse(Vec<String>),
}

impl TableQualityReport {
    /// Compare this report against a baseline using the policy
    /// described below.
    ///
    /// ## Project-Level Context
    ///
    /// This is a **policy decision** offered as a convenience. The
    /// policy:
    ///
    /// - Permutation validity is non-negotiable.
    /// - Fixed points: candidate ≤ baseline.
    /// - Mean displacement: candidate ≥ baseline − 1.0.
    /// - Sequential correlation magnitude: candidate ≤ baseline + 0.02.
    /// - XOR worst-case chi-square: candidate ≤ 1.10 × baseline.
    /// - Empirical collisions: candidate ≤ 2 × baseline (loose;
    ///   small-corpus collision count is high-variance noise).
    ///
    /// Callers who want a different policy should not use this
    /// method — they should read the fields of `TableQualityReport`
    /// directly and apply their own thresholds.
    pub fn compare_against(&self, baseline: &TableQualityReport) -> ComparisonVerdict {
        let mut complaints: Vec<String> = Vec::new();

        if !self.is_valid_permutation {
            complaints.push(String::from("candidate is not a valid permutation"));
        }

        if self.fixed_point_count > baseline.fixed_point_count {
            complaints.push(format!(
                "fixed points: candidate {} > baseline {}",
                self.fixed_point_count, baseline.fixed_point_count
            ));
        }

        if self.displacement.mean_displacement < baseline.displacement.mean_displacement - 1.0 {
            complaints.push(format!(
                "mean displacement: candidate {:.3} < baseline {:.3}",
                self.displacement.mean_displacement, baseline.displacement.mean_displacement
            ));
        }

        if self.sequential_correlation_abs > baseline.sequential_correlation_abs + 0.02 {
            complaints.push(format!(
                "sequential correlation: candidate {:.4} > baseline {:.4}",
                self.sequential_correlation_abs, baseline.sequential_correlation_abs
            ));
        }

        if self.xor_uniformity.worst_chi_square > baseline.xor_uniformity.worst_chi_square * 1.10 {
            complaints.push(format!(
                "XOR worst chi-square: candidate {:.2} > 1.10 * baseline {:.2}",
                self.xor_uniformity.worst_chi_square, baseline.xor_uniformity.worst_chi_square
            ));
        }

        // Empirical collisions on a small fixed corpus are
        // high-variance noise. We only flag truly egregious cases.
        if self.empirical_collisions > baseline.empirical_collisions * 2 {
            complaints.push(format!(
                "empirical collisions: candidate {} > 2 * baseline {}",
                self.empirical_collisions, baseline.empirical_collisions
            ));
        }

        if complaints.is_empty() {
            ComparisonVerdict::AtLeastAsGood
        } else {
            ComparisonVerdict::Worse(complaints)
        }
    }
}

impl fmt::Display for TableQualityReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TableQualityReport {{")?;
        writeln!(
            f,
            "    valid permutation:        {}",
            self.is_valid_permutation
        )?;
        writeln!(
            f,
            "    fixed points:             {}",
            self.fixed_point_count
        )?;
        writeln!(f, "    one-cycles:               {}", self.one_cycle_count)?;
        writeln!(f, "    two-cycles:               {}", self.two_cycle_count)?;
        writeln!(f, "    longest cycle:            {}", self.longest_cycle)?;
        writeln!(
            f,
            "    total cycles:             {}",
            self.cycle_lengths.len()
        )?;
        writeln!(
            f,
            "    displacement min:         {}",
            self.displacement.min_displacement
        )?;
        writeln!(
            f,
            "    displacement max:         {}",
            self.displacement.max_displacement
        )?;
        writeln!(
            f,
            "    displacement mean:        {:.3}",
            self.displacement.mean_displacement
        )?;
        writeln!(
            f,
            "    displacement zero count:  {}",
            self.displacement.zero_displacement_count
        )?;
        writeln!(
            f,
            "    seq correlation |r|:      {:.6}",
            self.sequential_correlation_abs
        )?;
        writeln!(
            f,
            "    XOR worst chi-square:     {:.3} (at d = 0x{:02X})",
            self.xor_uniformity.worst_chi_square, self.xor_uniformity.worst_difference
        )?;
        writeln!(
            f,
            "    XOR mean chi-square:      {:.3}",
            self.xor_uniformity.mean_chi_square
        )?;
        writeln!(
            f,
            "    empirical collisions:     {}",
            self.empirical_collisions
        )?;
        write!(f, "}}")
    }
}

/// Run all six metrics on `table` and return a single report.
///
/// ## Project-Level Context
///
/// This is the headline entry point for measurement. The result
/// is a record of facts; callers interpret it.
pub fn evaluate_table(table: &[u8; 256]) -> TableQualityReport {
    let is_perm: bool = is_valid_permutation_tools(table);
    let fixed_count: usize = count_fixed_points(table);
    let cycles: Vec<usize> = cycle_structure(table);

    let longest: usize = cycles.first().copied().unwrap_or(0);
    let one_cycles: usize = cycles.iter().filter(|&&c| c == 1).count();
    let two_cycles: usize = cycles.iter().filter(|&&c| c == 2).count();

    let disp: DisplacementStats = displacement_stats(table);
    let corr_abs: f64 = sequential_correlation_abs(table);
    let xor_stats: XorUniformityStats = xor_uniformity(table);
    let collisions: usize = empirical_collision_count(table);

    TableQualityReport {
        is_valid_permutation: is_perm,
        fixed_point_count: fixed_count,
        cycle_lengths: cycles,
        one_cycle_count: one_cycles,
        two_cycle_count: two_cycles,
        longest_cycle: longest,
        displacement: disp,
        sequential_correlation_abs: corr_abs,
        xor_uniformity: xor_stats,
        empirical_collisions: collisions,
    }
}

// =============================================================================
// SECTION 6: Reporting — human-readable output
// =============================================================================

/// Print a single labeled evaluation report to stdout.
///
/// ## Project-Level Context
///
/// This is the recommended way for `main.rs` and downstream
/// tooling to display measurements. It produces no verdict; the
/// reader interprets the numbers.
pub fn print_table_evaluation_report(label: &str, table: &[u8; 256]) {
    let report = evaluate_table(table);
    println!("===== {} =====", label);
    println!("{}", report);
    println!();
}

/// Print two reports side by side with per-metric delta.
///
/// ## Project-Level Context
///
/// The intended use is "candidate vs. baseline" comparison. The
/// `delta` column shows `candidate - baseline` for each numeric
/// metric. Sign convention is **not** "good vs. bad" — it is
/// straight arithmetic. The reader interprets each metric's
/// preferred direction (which is documented per-metric in the
/// docstrings above).
///
/// ## Arguments
///
/// * `label_a`, `table_a` — first table (typically baseline).
/// * `label_b`, `table_b` — second table (typically candidate).
pub fn print_comparative_report(
    label_a: &str,
    table_a: &[u8; 256],
    label_b: &str,
    table_b: &[u8; 256],
) {
    let report_a = evaluate_table(table_a);
    let report_b = evaluate_table(table_b);

    println!("===== Comparative report =====");
    println!("  A = {}", label_a);
    println!("  B = {}", label_b);
    println!();
    println!("{:<28} {:>14} {:>14} {:>14}", "metric", "A", "B", "B - A");
    println!("{}", "-".repeat(28 + 1 + 14 + 1 + 14 + 1 + 14));

    // Row helpers. Each row prints a metric name, value-A, value-B,
    // and (B - A). Integer and float variants.

    fn row_int(name: &str, a: i64, b: i64) {
        println!("{:<28} {:>14} {:>14} {:>+14}", name, a, b, b - a);
    }
    fn row_uint(name: &str, a: usize, b: usize) {
        // Signed delta — use i64 to preserve sign.
        let a_i: i64 = a as i64;
        let b_i: i64 = b as i64;
        println!("{:<28} {:>14} {:>14} {:>+14}", name, a, b, b_i - a_i);
    }
    fn row_float(name: &str, a: f64, b: f64, precision: usize) {
        println!(
            "{:<28} {:>14.*} {:>14.*} {:>+14.*}",
            name,
            precision,
            a,
            precision,
            b,
            precision,
            b - a
        );
    }
    fn row_bool(name: &str, a: bool, b: bool) {
        println!(
            "{:<28} {:>14} {:>14} {:>14}",
            name,
            a,
            b,
            if a == b { "same" } else { "differ" }
        );
    }

    row_bool(
        "valid permutation",
        report_a.is_valid_permutation,
        report_b.is_valid_permutation,
    );
    row_uint(
        "fixed points",
        report_a.fixed_point_count,
        report_b.fixed_point_count,
    );
    row_uint(
        "one-cycles",
        report_a.one_cycle_count,
        report_b.one_cycle_count,
    );
    row_uint(
        "two-cycles",
        report_a.two_cycle_count,
        report_b.two_cycle_count,
    );
    row_uint(
        "longest cycle",
        report_a.longest_cycle,
        report_b.longest_cycle,
    );
    row_uint(
        "total cycles",
        report_a.cycle_lengths.len(),
        report_b.cycle_lengths.len(),
    );
    row_int(
        "displacement min",
        report_a.displacement.min_displacement as i64,
        report_b.displacement.min_displacement as i64,
    );
    row_int(
        "displacement max",
        report_a.displacement.max_displacement as i64,
        report_b.displacement.max_displacement as i64,
    );
    row_float(
        "displacement mean",
        report_a.displacement.mean_displacement,
        report_b.displacement.mean_displacement,
        3,
    );
    row_uint(
        "displacement zero count",
        report_a.displacement.zero_displacement_count,
        report_b.displacement.zero_displacement_count,
    );
    row_float(
        "seq correlation |r|",
        report_a.sequential_correlation_abs,
        report_b.sequential_correlation_abs,
        6,
    );
    row_float(
        "XOR worst chi-square",
        report_a.xor_uniformity.worst_chi_square,
        report_b.xor_uniformity.worst_chi_square,
        3,
    );
    row_float(
        "XOR mean chi-square",
        report_a.xor_uniformity.mean_chi_square,
        report_b.xor_uniformity.mean_chi_square,
        3,
    );
    row_uint(
        "empirical collisions",
        report_a.empirical_collisions,
        report_b.empirical_collisions,
    );

    println!();
    println!("Note: 'B - A' is straight arithmetic. Preferred");
    println!("direction depends on the metric (see module docs).");
    println!();
}

// =============================================================================
// SECTION 7: Cargo tests — code correctness only
// =============================================================================
//
// Every test in this module verifies a metric's behavior against
// an input whose correct answer is known a priori. No test passes
// or fails based on whether one random table happens to outscore
// another on a noisy metric. The intent is: if these tests pass,
// the metric functions compute what they claim to compute. The
// quality of any particular table is a question for the reports
// printed by Section 6, not for cargo test.

#[cfg(test)]
mod tests {
    use super::*;

    // ---- known test fixtures (each is a valid permutation) ----

    /// Identity: `T[i] = i`. The pathological case for Pearson.
    fn identity_table() -> [u8; 256] {
        let mut t: [u8; 256] = [0u8; 256];
        for i in 0..256usize {
            t[i] = i as u8;
        }
        t
    }

    /// Reverse: `T[i] = 255 - i`. A valid permutation with
    /// zero fixed points and known displacement statistics.
    fn reverse_table() -> [u8; 256] {
        let mut t: [u8; 256] = [0u8; 256];
        for i in 0..256usize {
            t[i] = (255 - i) as u8;
        }
        t
    }

    /// Identity with positions 3 and 7 swapped — produces exactly
    /// one 2-cycle and 254 one-cycles.
    fn single_swap_table() -> [u8; 256] {
        let mut t = identity_table();
        t.swap(3, 7);
        t
    }

    // ===================================================================
    // Local 1990 copy: valid permutation
    // ===================================================================

    #[test]
    fn test_local_1990_table_copy_is_valid_permutation() {
        assert!(is_valid_permutation_tools(&PEARSON_1990_TABLE_COPY));
    }

    // ===================================================================
    // is_valid_permutation_tools
    // ===================================================================

    #[test]
    fn test_is_valid_permutation_accepts_identity() {
        assert!(is_valid_permutation_tools(&identity_table()));
    }

    #[test]
    fn test_is_valid_permutation_accepts_reverse() {
        assert!(is_valid_permutation_tools(&reverse_table()));
    }

    #[test]
    fn test_is_valid_permutation_rejects_duplicate() {
        let mut bad = identity_table();
        bad[5] = 7; // duplicate of position 7
        assert!(!is_valid_permutation_tools(&bad));
    }

    // ===================================================================
    // Fisher-Yates generators
    // ===================================================================

    #[test]
    fn test_fisher_yates_runtime_produces_valid_permutation_for_many_seeds() {
        let seeds: [u64; 8] = [
            0,
            1,
            2,
            42,
            0xDEAD_BEEF,
            0x9E37_79B9_7F4A_7C15,
            u64::MAX,
            0x1234_5678_9ABC_DEF0,
        ];
        for &seed in seeds.iter() {
            let t = generate_table_fisher_yates(seed);
            assert!(
                is_valid_permutation_tools(&t),
                "seed {:#x} produced non-permutation",
                seed
            );
        }
    }

    #[test]
    fn test_fisher_yates_const_matches_runtime_for_many_seeds() {
        let seeds: [u64; 8] = [
            0,
            1,
            2,
            42,
            0xDEAD_BEEF,
            0x9E37_79B9_7F4A_7C15,
            u64::MAX,
            0x1234_5678_9ABC_DEF0,
        ];
        for &seed in seeds.iter() {
            let runtime = generate_table_fisher_yates(seed);
            let constv = generate_table_fisher_yates_const(seed);
            assert_eq!(runtime, constv, "seed {:#x}: const != runtime", seed);
        }
    }

    #[test]
    fn test_fisher_yates_is_deterministic_in_seed() {
        let a = generate_table_fisher_yates(0xCAFEBABE);
        let b = generate_table_fisher_yates(0xCAFEBABE);
        assert_eq!(a, b);
    }

    #[test]
    fn test_fisher_yates_different_seeds_produce_different_tables() {
        let a = generate_table_fisher_yates(1);
        let b = generate_table_fisher_yates(2);
        assert_ne!(a, b);
    }

    // ===================================================================
    // count_fixed_points  (known-answer)
    // ===================================================================

    #[test]
    fn test_count_fixed_points_identity_is_256() {
        assert_eq!(count_fixed_points(&identity_table()), 256);
    }

    #[test]
    fn test_count_fixed_points_reverse_is_zero() {
        // T[i] = 255 - i; T[i] == i iff 2i == 255, no integer i.
        assert_eq!(count_fixed_points(&reverse_table()), 0);
    }

    #[test]
    fn test_count_fixed_points_single_swap_is_254() {
        // Swapped 3<->7, all others fixed.
        assert_eq!(count_fixed_points(&single_swap_table()), 254);
    }

    // ===================================================================
    // cycle_structure  (known-answer)
    // ===================================================================

    #[test]
    fn test_cycle_structure_identity_is_256_ones() {
        let cycles = cycle_structure(&identity_table());
        assert_eq!(cycles.len(), 256);
        assert!(cycles.iter().all(|&c| c == 1));
    }

    #[test]
    fn test_cycle_structure_reverse_is_128_twos() {
        // T(T(i)) = 255 - (255 - i) = i, so every element is in
        // a 2-cycle with its mirror. 256 elements -> 128 two-cycles.
        let cycles = cycle_structure(&reverse_table());
        assert_eq!(cycles.len(), 128);
        assert!(cycles.iter().all(|&c| c == 2));
    }

    #[test]
    fn test_cycle_structure_single_swap_is_one_two_and_254_ones() {
        let cycles = cycle_structure(&single_swap_table());
        // 254 fixed points (one-cycles) + 1 two-cycle = 255 cycles
        let one_count = cycles.iter().filter(|&&c| c == 1).count();
        let two_count = cycles.iter().filter(|&&c| c == 2).count();
        assert_eq!(one_count, 254);
        assert_eq!(two_count, 1);
        assert_eq!(cycles.len(), 255);
    }

    #[test]
    fn test_cycle_structure_sums_to_256_for_random_table() {
        let t = generate_table_fisher_yates(0xABCDEF);
        let cycles = cycle_structure(&t);
        let total: usize = cycles.iter().sum();
        assert_eq!(total, 256);
    }

    // ===================================================================
    // displacement_stats  (known-answer)
    // ===================================================================

    #[test]
    fn test_displacement_identity_is_all_zero() {
        let s = displacement_stats(&identity_table());
        assert_eq!(s.min_displacement, 0);
        assert_eq!(s.max_displacement, 0);
        assert_eq!(s.mean_displacement, 0.0);
        assert_eq!(s.zero_displacement_count, 256);
    }

    #[test]
    fn test_displacement_reverse_has_known_mean() {
        // T[i] = 255 - i, so |T[i] - i| = |255 - 2i|.
        // Sum_{i=0..256} |255 - 2i| = 2 * Sum_{k=0..128} (2k + 1)
        //                           = 2 * 128^2 = 32768.
        // Mean = 32768 / 256 = 128.0.
        let s = displacement_stats(&reverse_table());
        assert_eq!(s.mean_displacement, 128.0);
        assert_eq!(s.zero_displacement_count, 0);
        assert_eq!(s.min_displacement, 1); // i=127 -> |255-254|=1
        assert_eq!(s.max_displacement, 255); // i=0   -> |255-0|=255
    }

    // ===================================================================
    // sequential_correlation_abs  (known-answer)
    // ===================================================================

    #[test]
    fn test_sequential_correlation_identity_is_one() {
        // T[i+1] = T[i] + 1 — perfect positive linear relation.
        let c = sequential_correlation_abs(&identity_table());
        assert!((c - 1.0).abs() < 1e-9, "got {}", c);
    }

    #[test]
    fn test_sequential_correlation_reverse_is_one() {
        // T[i+1] = T[i] - 1 — perfect negative linear relation,
        // abs value = 1.
        let c = sequential_correlation_abs(&reverse_table());
        assert!((c - 1.0).abs() < 1e-9, "got {}", c);
    }

    // ===================================================================
    // xor_uniformity  (known-answer)
    // ===================================================================

    #[test]
    fn test_xor_uniformity_identity_known_values() {
        // For identity, T[i] XOR T[i^d] = i XOR (i^d) = d, for all i.
        // Histogram for each d: 256 in bucket `d`, 0 elsewhere.
        // Chi-square per d: 255 buckets contribute (0-1)^2 = 1 each,
        // and one bucket contributes (256-1)^2 = 65025.
        // Total per d: 255 + 65025 = 65280. Same for every d, so
        // worst == mean == 65280 exactly.
        let s = xor_uniformity(&identity_table());
        assert_eq!(s.worst_chi_square, 65280.0);
        assert_eq!(s.mean_chi_square, 65280.0);
        // Worst d is just whatever 1..=255 we encounter first; the
        // value is implementation-detail. Just check it is in range.
        assert!(s.worst_difference >= 1);
    }

    #[test]
    fn test_xor_uniformity_returns_finite_for_random_table() {
        let t = generate_table_fisher_yates(0x4242_4242);
        let s = xor_uniformity(&t);
        assert!(s.worst_chi_square.is_finite());
        assert!(s.mean_chi_square.is_finite());
        assert!(s.worst_chi_square >= s.mean_chi_square);
        assert!(s.worst_difference >= 1);
    }

    // ===================================================================
    // empirical_collision_count  (determinism)
    // ===================================================================

    #[test]
    fn test_empirical_collision_count_is_deterministic() {
        let t = generate_table_fisher_yates(0x1357_9BDF);
        let a = empirical_collision_count(&t);
        let b = empirical_collision_count(&t);
        assert_eq!(a, b);
    }

    #[test]
    fn test_empirical_collision_count_same_for_same_table() {
        // Calling on the 1990 table twice must produce identical
        // results — the corpus is fixed, the table is fixed.
        let a = empirical_collision_count(&PEARSON_1990_TABLE_COPY);
        let b = empirical_collision_count(&PEARSON_1990_TABLE_COPY);
        assert_eq!(a, b);
    }

    // ===================================================================
    // evaluate_table  (runs and reports correctly)
    // ===================================================================

    #[test]
    fn test_evaluate_table_marks_invalid_table_false() {
        let mut bad = identity_table();
        bad[5] = 7;
        let r = evaluate_table(&bad);
        assert!(!r.is_valid_permutation);
    }

    #[test]
    fn test_evaluate_table_marks_valid_table_true() {
        let r = evaluate_table(&PEARSON_1990_TABLE_COPY);
        assert!(r.is_valid_permutation);
        let total: usize = r.cycle_lengths.iter().sum();
        assert_eq!(total, 256);
    }

    #[test]
    fn test_evaluate_table_fields_consistent_with_individual_metrics() {
        // The aggregate report must agree with calling each metric
        // function directly on the same table.
        let table = generate_table_fisher_yates(0x9876_5432);
        let r = evaluate_table(&table);
        assert_eq!(r.is_valid_permutation, is_valid_permutation_tools(&table));
        assert_eq!(r.fixed_point_count, count_fixed_points(&table));
        let cycles_direct = cycle_structure(&table);
        assert_eq!(r.cycle_lengths, cycles_direct);
        let disp_direct = displacement_stats(&table);
        assert_eq!(
            r.displacement.min_displacement,
            disp_direct.min_displacement
        );
        assert_eq!(
            r.displacement.max_displacement,
            disp_direct.max_displacement
        );
        assert_eq!(
            r.displacement.mean_displacement,
            disp_direct.mean_displacement
        );
        let corr_direct = sequential_correlation_abs(&table);
        assert_eq!(r.sequential_correlation_abs, corr_direct);
        let xor_direct = xor_uniformity(&table);
        assert_eq!(
            r.xor_uniformity.worst_chi_square,
            xor_direct.worst_chi_square
        );
        assert_eq!(r.xor_uniformity.mean_chi_square, xor_direct.mean_chi_square);
        assert_eq!(r.empirical_collisions, empirical_collision_count(&table));
    }

    // ===================================================================
    // compare_against  (verifies the comparison logic itself works,
    //                   using inputs with known-correct verdicts —
    //                   NOT a quality gate on any specific table)
    // ===================================================================

    #[test]
    fn test_compare_against_identical_reports_returns_at_least_as_good() {
        let r = evaluate_table(&PEARSON_1990_TABLE_COPY);
        let verdict = r.compare_against(&r);
        assert_eq!(verdict, ComparisonVerdict::AtLeastAsGood);
    }

    #[test]
    fn test_compare_against_invalid_candidate_is_flagged_worse() {
        // Build an invalid table by hand, evaluate it, compare to
        // a valid baseline. The verdict must be Worse, and the
        // complaints must mention the permutation failure.
        let mut bad = identity_table();
        bad[5] = 7;
        let bad_report = evaluate_table(&bad);
        let baseline = evaluate_table(&PEARSON_1990_TABLE_COPY);
        match bad_report.compare_against(&baseline) {
            ComparisonVerdict::AtLeastAsGood => {
                panic!("invalid candidate should never compare as at-least-as-good");
            }
            ComparisonVerdict::Worse(complaints) => {
                let any_perm = complaints.iter().any(|c| c.contains("permutation"));
                assert!(
                    any_perm,
                    "complaints did not mention permutation: {:?}",
                    complaints
                );
            }
        }
    }

    #[test]
    fn test_compare_against_identity_versus_1990_is_flagged_worse() {
        // Identity table is structurally pathological: 256 fixed
        // points, perfect sequential correlation, zero displacement.
        // The comparison policy must flag at least one of these.
        // This is a known-answer test of the comparison logic, not
        // a quality verdict on any generated table.
        let r_id = evaluate_table(&identity_table());
        let r_1990 = evaluate_table(&PEARSON_1990_TABLE_COPY);
        match r_id.compare_against(&r_1990) {
            ComparisonVerdict::AtLeastAsGood => {
                panic!("identity must not compare as at-least-as-good");
            }
            ComparisonVerdict::Worse(_complaints) => {
                // Pass: identity flagged as worse, as required.
            }
        }
    }
}
