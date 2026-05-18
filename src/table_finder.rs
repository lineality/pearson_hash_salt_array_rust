// src/table_finder.rs
// (made with assistance from claude 4.7)

//! # `table_finder` — Search for the best Pearson permutation table
//!                     for a user-supplied corpus
//!
//! ## Project-Level Context
//!
//! This module is a **search and reporting** tool, not a production
//! hashing path. Its purpose is to help a developer choose a specific
//! Pearson permutation table that performs well on a specific corpus
//! of inputs — for example, an 8×8 chess-board byte-array dataset.
//!
//! The 1990 paper itself notes that no random permutation is
//! globally "best" for all inputs, but a particular table may be
//! best for a particular dataset. This module operationalizes that
//! observation: it tries many seeded Fisher-Yates tables against
//! the user's corpus, ranks them on the user's chosen metric, and
//! prints the leaderboard. The user reads, picks a seed, and
//! hard-codes it into their production build.
//!
//! ## Design Posture
//!
//! - **Measurement, not verdict.** Same rule as `pearson_hash_tools`.
//!   This module ranks; it does not decide.
//! - **Search outcomes never gate cargo tests.** Cargo tests in this
//!   module verify code correctness only.
//! - **Single source of truth.** Hashing functions and structural
//!   metrics are imported from the production module and the tools
//!   module, not duplicated.
//!
//! ## Three-phase search
//!
//! 1. **Phase 1 — cheap scoring (every candidate).** For each seed,
//!    generate the Fisher-Yates table and compute corpus-collision
//!    statistics (and optional salt-array statistics). Keep the
//!    top-K survivors by the chosen ranking metric.
//!
//! 2. **Phase 2 — full structural evaluation (top-K only).** Run
//!    the six structural metrics from `pearson_hash_tools` on the
//!    top-K survivors, so the user can see whether a corpus-winner
//!    also has acceptable structural properties.
//!
//! 3. **Phase 3 — perturbation refinement (top-K only).** For each
//!    survivor seed `s`, generate 64 perturbed seeds `s XOR (1<<b)`
//!    for `b in 0..64`, score them, and report the best of the
//!    perturbed pool alongside the original top-K. This finds
//!    local optima near the phase-1 winners.
//!
//! ## Heap and Tools Posture
//!
//! This is tools / development code. Heap is used freely
//! (`Vec`, `String`, `println!`, file I/O). The production hashing
//! functions called from here do not themselves allocate.

use std::cmp::Ordering;
use std::fs::File;
use std::io::{Error, ErrorKind, Write};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::pearson_hash_salt_array_rust::{
    GENERATED_TABLE, PEARSON_1990_TABLE, pearson_hash_base, pearson_hash_salt_array,
};

use crate::pearson_hash_tools::{TableQualityReport, evaluate_table, generate_table_fisher_yates};

// =============================================================================
// SECTION 1: Configuration types
// =============================================================================

/// Maximum number of salts supported by the salt-array test.
///
/// ## Project-Level Context
///
/// `pearson_hash_salt_array` is const-generic in `N`. To keep this
/// module's code simple, we fix a single compile-time `N` here. If
/// the user supplies fewer than 8 salts we pad with zero salts;
/// padding never harms collision counting because zero is a valid
/// salt (it just produces a fixed deterministic byte for every
/// input, which is the same for all inputs and so does not
/// distinguish them — see test).
pub const MAX_SALTS: usize = 8;

/// User-visible ranking criterion. The selected criterion
/// determines leaderboard order; all other metrics are computed
/// and reported anyway so the user sees trade-offs.
#[derive(Clone, Copy, PartialEq, Eq)] // Debug,
pub enum RankBy {
    /// Total unordered colliding pairs for the base 8-bit Pearson
    /// hash on the user's corpus. Lower is better. Recommended
    /// default for hash-table use.
    BaseCollisions,
    /// Total unordered colliding pairs for the salt-array hash
    /// (treating each [u8; N] as one composite key). Lower is
    /// better. Recommended for Bloom-filter / multi-hash use.
    /// Requires `SaltArrayConfig` to be supplied.
    SaltArrayCollisions,
    /// Chi-square of the base-hash bucket histogram against the
    /// uniform distribution. Lower is better. Recommended when
    /// bucket balance matters more than raw collision count.
    BaseChiSquare,
    /// Worst-case bucket occupancy for the base hash. Lower is
    /// better. Recommended when worst-case lookup time matters.
    BaseMaxBucket,
    /// Structural metric carried over from `pearson_hash_tools`.
    /// Lower is better. Useful when the user wants good general
    /// avalanche behavior in addition to corpus fit.
    XorWorstChiSquare,
}

impl RankBy {
    /// Human-readable label for the report header.
    fn description(self) -> &'static str {
        match self {
            RankBy::BaseCollisions => "BaseCollisions (lower is better)",
            RankBy::SaltArrayCollisions => "SaltArrayCollisions (lower is better)",
            RankBy::BaseChiSquare => "BaseChiSquare (lower is better)",
            RankBy::BaseMaxBucket => "BaseMaxBucket (lower is better)",
            RankBy::XorWorstChiSquare => "XorWorstChiSquare (lower is better)",
        }
    }
}

/// How to enumerate candidate seeds.
#[derive(Debug, Clone, Copy)]
pub enum SweepMode {
    /// Try seeds `start, start+1, ..., start+count-1`. Fully
    /// deterministic and reproducible.
    Linear { start: u64, count: u64 },
    /// Generate `count` seeds via a splitmix64 sequence seeded by
    /// `meta_seed`. Same `meta_seed` -> same trajectory. Different
    /// `meta_seed` (e.g. derived from system time) gives a fresh
    /// search region, letting the user run the finder repeatedly
    /// to explore.
    PseudoRandom { meta_seed: u64, count: u64 },
}

impl SweepMode {
    /// Total number of seeds the sweep will try.
    fn count(self) -> u64 {
        match self {
            SweepMode::Linear { count, .. } => count,
            SweepMode::PseudoRandom { count, .. } => count,
        }
    }

    /// Human-readable label for the report header.
    fn description(self) -> String {
        match self {
            SweepMode::Linear { start, count } => {
                format!("Linear sweep: {} seeds starting at 0x{:016X}", count, start)
            }
            SweepMode::PseudoRandom { meta_seed, count } => format!(
                "PseudoRandom sweep: {} seeds, meta_seed = 0x{:016X}",
                count, meta_seed
            ),
        }
    }
}

/// Configuration for the salt-array test.
///
/// ## Project-Level Context
///
/// At most `MAX_SALTS` salts (currently 8). Fewer are accepted;
/// remaining internal slots are padded with zero and contribute
/// nothing distinguishing to the collision count (verified by
/// test). More than `MAX_SALTS` is rejected with a terse error.
#[derive(Debug, Clone)]
pub struct SaltArrayConfig {
    /// User-supplied salts. Must have length in `1..=MAX_SALTS`.
    pub salts: Vec<u128>,
}

// =============================================================================
// SECTION 2: Per-seed score record
// =============================================================================

/// Per-seed measurements on the user's corpus.
///
/// `structural` is filled only for top-K survivors (phase 2);
/// it is `None` for the cheap phase-1 candidates that did not
/// reach the leaderboard.
#[derive(Debug, Clone)]
pub struct CorpusScoreReport {
    pub seed: u64,
    pub base_collisions: usize,
    pub base_chi_square: f64,
    pub base_max_bucket: u32,
    pub base_empty_buckets: u32,
    /// Set to `usize::MAX` when `SaltArrayConfig` was not provided
    /// (sentinel meaning "not measured"). Set to a real count
    /// otherwise.
    pub salt_array_collisions: usize,
    /// Filled only for top-K survivors. None for phase-1-only
    /// candidates.
    pub structural: Option<TableQualityReport>,
}

impl CorpusScoreReport {
    /// Return the value used to rank this report under `rank_by`.
    ///
    /// Returned as `f64` because some metrics are integer-valued
    /// (counts) and some are floating-point (chi-square). The
    /// caller sorts by this comparable scalar.
    fn ranking_value(&self, rank_by: RankBy) -> f64 {
        match rank_by {
            RankBy::BaseCollisions => self.base_collisions as f64,
            RankBy::SaltArrayCollisions => self.salt_array_collisions as f64,
            RankBy::BaseChiSquare => self.base_chi_square,
            RankBy::BaseMaxBucket => self.base_max_bucket as f64,
            RankBy::XorWorstChiSquare => {
                // Only meaningful for survivors that have a
                // structural report. For phase-1 candidates it
                // returns +inf so they sort to the bottom.
                match &self.structural {
                    Some(r) => r.xor_uniformity.worst_chi_square,
                    None => f64::INFINITY,
                }
            }
        }
    }
}

// =============================================================================
// SECTION 3: Aggregate search report
// =============================================================================

#[derive(Debug, Clone)]
pub struct SearchReport {
    pub corpus_size: usize,
    pub corpus_entry_lengths_min: usize,
    pub corpus_entry_lengths_max: usize,
    pub seeds_evaluated: u64,
    pub sweep_mode_description: String,
    pub rank_by_description: &'static str,
    /// `Some(N)` if a salt-array test was used, where N is the
    /// (padded) number of salts actually fed into the hash.
    pub salt_array_used: Option<usize>,
    pub top_k: Vec<CorpusScoreReport>,
    pub baseline_1990: CorpusScoreReport,
    pub baseline_generated: CorpusScoreReport,
    /// Result of phase 3 (perturbation refinement). One entry per
    /// surviving perturbed candidate, ranked the same way.
    pub perturbation_top_k: Vec<CorpusScoreReport>,
    /// The worst-performing seeds from Phase 1, ranked by the same
    /// metric as `top_k` but in descending order (worst first).
    /// Structural metrics are attached (Phase 2) so the user can
    /// inspect what a bad table looks like alongside the good ones.
    /// Populated by `search_seeds` and `perturb_seed_search`; the
    /// caller controls the count via the `bottom_k` parameter.
    pub worst_k: Vec<CorpusScoreReport>,
    pub elapsed_seconds: f64,
}

// =============================================================================
// SECTION 4: PRNG helper (splitmix64)
// =============================================================================

/// Single splitmix64 step.
///
/// ## Project-Level Context
///
/// Same PRNG used by the production module's Fisher-Yates
/// generator. We use it here both for `PseudoRandom` seed
/// generation and for the chess-corpus generator. Pure-function;
/// caller owns the state.
fn splitmix64_step(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z: u64 = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// =============================================================================
// SECTION 5: Corpus scoring (the inner loop)
// =============================================================================

/// Score one table on one corpus.
///
/// ## Arguments
///
/// * `table` — candidate permutation table.
/// * `corpus` — slice of byte slices (each one input to be hashed).
/// * `salt_config` — optional salt-array configuration.
///
/// ## Returns
///
/// A `CorpusScoreReport` with all corpus metrics filled in.
/// `structural` is left `None`; the caller may attach it later
/// via phase 2.
///
/// ## Errors
///
/// - Empty corpus           → `Err` `"TF: empty corpus"`.
/// - Salts > MAX_SALTS      → `Err` `"TF: too many salts"`.
/// - Empty salt list        → `Err` `"TF: empty salt list"`.
/// - Empty corpus entry     → handled gracefully: entry is skipped
///   with a debug-build warning, never panics.
pub fn score_table_on_corpus(
    table: &[u8; 256],
    corpus: &[&[u8]],
    salt_config: Option<&SaltArrayConfig>,
) -> Result<CorpusScoreReport, Error> {
    if corpus.is_empty() {
        return Err(Error::new(ErrorKind::InvalidInput, "TF: empty corpus"));
    }
    if let Some(cfg) = salt_config {
        if cfg.salts.is_empty() {
            return Err(Error::new(ErrorKind::InvalidInput, "TF: empty salt list"));
        }
        if cfg.salts.len() > MAX_SALTS {
            return Err(Error::new(ErrorKind::InvalidInput, "TF: too many salts"));
        }
    }

    // ---- Base-hash collision counting via 256-bucket histogram ----
    let mut bucket_counts: [u32; 256] = [0u32; 256];
    let mut hashed_count: u32 = 0;
    for entry in corpus.iter() {
        match pearson_hash_base(entry, table) {
            Ok(h) => {
                bucket_counts[h as usize] = bucket_counts[h as usize].saturating_add(1);
                hashed_count = hashed_count.saturating_add(1);
            }
            Err(_) => {
                // Empty corpus entry. Skip; do not panic. In debug
                // builds, surface the issue.
                #[cfg(all(debug_assertions, not(test)))]
                eprintln!("TF: skipping empty corpus entry");
                continue;
            }
        }
    }

    // Collision count: for each bucket with count c >= 2,
    // contribute c*(c-1)/2 unordered colliding pairs.
    let mut base_collisions: usize = 0;
    let mut max_bucket: u32 = 0;
    let mut empty_buckets: u32 = 0;
    for &c in bucket_counts.iter() {
        if c >= 2 {
            let c_usize: usize = c as usize;
            base_collisions += c_usize * (c_usize - 1) / 2;
        }
        if c > max_bucket {
            max_bucket = c;
        }
        if c == 0 {
            empty_buckets += 1;
        }
    }

    // Chi-square against uniform with `hashed_count` samples into
    // 256 buckets, expected per bucket = hashed_count / 256.
    let expected_per_bucket: f64 = (hashed_count as f64) / 256.0;
    let mut chi_square: f64 = 0.0;
    if expected_per_bucket > 0.0 {
        for &c in bucket_counts.iter() {
            let delta: f64 = (c as f64) - expected_per_bucket;
            chi_square += (delta * delta) / expected_per_bucket;
        }
    }

    // ---- Salt-array collision counting (optional) ----
    let salt_array_collisions: usize = match salt_config {
        None => usize::MAX, // sentinel: not measured
        Some(cfg) => {
            // Pad salts to MAX_SALTS with zeros. A zero salt
            // produces a fixed extra byte sequence that is the
            // same for every input — it contributes a constant
            // to the output and does not change which inputs
            // collide. (Verified by test.)
            let mut padded_salts: [u128; MAX_SALTS] = [0u128; MAX_SALTS];
            for (i, &s) in cfg.salts.iter().enumerate() {
                padded_salts[i] = s;
            }

            // Hash every corpus entry, collect [u8; MAX_SALTS]
            // outputs, sort, count adjacent duplicates as
            // collision pairs.
            let mut outputs: Vec<[u8; MAX_SALTS]> = Vec::with_capacity(corpus.len());
            for entry in corpus.iter() {
                match pearson_hash_salt_array::<MAX_SALTS>(entry, &padded_salts, table) {
                    Ok(arr) => outputs.push(arr),
                    Err(_) => continue, // empty entry; skip
                }
            }
            outputs.sort_unstable();

            let mut pair_count: usize = 0;
            let mut run_len: usize = 1;
            for i in 1..outputs.len() {
                if outputs[i] == outputs[i - 1] {
                    run_len += 1;
                } else {
                    if run_len >= 2 {
                        pair_count += run_len * (run_len - 1) / 2;
                    }
                    run_len = 1;
                }
            }
            if run_len >= 2 {
                pair_count += run_len * (run_len - 1) / 2;
            }
            pair_count
        }
    };

    Ok(CorpusScoreReport {
        seed: 0, // caller fills in
        base_collisions,
        base_chi_square: chi_square,
        base_max_bucket: max_bucket,
        base_empty_buckets: empty_buckets,
        salt_array_collisions,
        structural: None,
    })
}

// =============================================================================
// SECTION 6: Three-phase search
// =============================================================================

/// Compare two ranking values (f64), placing NaN at the bottom.
fn cmp_ranking(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Greater)
}

/// Run the full three-phase search.
///
/// See module-level docs for the phase definitions.
///
/// ## Errors
///
/// - Empty corpus              → `Err` `"TF: empty corpus"`.
/// - `top_k == 0`              → `Err` `"TF: zero top_k"`.
/// - Sweep `count == 0`        → `Err` `"TF: zero seeds"`.
/// - Salt config invalid (see `score_table_on_corpus`).
///
/// ## Returns
///
/// A populated `SearchReport`.
pub fn search_seeds(
    corpus: &[&[u8]],
    salt_config: Option<&SaltArrayConfig>,
    mode: SweepMode,
    rank_by: RankBy,
    top_k: usize,
    bottom_k: usize,
) -> Result<SearchReport, Error> {
    // ---- Defensive checks ----
    if corpus.is_empty() {
        return Err(Error::new(ErrorKind::InvalidInput, "TF: empty corpus"));
    }
    if top_k == 0 {
        return Err(Error::new(ErrorKind::InvalidInput, "TF: zero top_k"));
    }
    if mode.count() == 0 {
        return Err(Error::new(ErrorKind::InvalidInput, "TF: zero seeds"));
    }
    if rank_by == RankBy::SaltArrayCollisions && salt_config.is_none() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "TF: salt rank without salt config",
        ));
    }

    let start_time = Instant::now();

    // Corpus length stats for the header.
    let mut min_len: usize = usize::MAX;
    let mut max_len: usize = 0;
    for entry in corpus.iter() {
        if entry.len() < min_len {
            min_len = entry.len();
        }
        if entry.len() > max_len {
            max_len = entry.len();
        }
    }

    // -------------------------------------------------------------
    // PHASE 1 — cheap scoring on every candidate; keep top-K.
    // -------------------------------------------------------------
    let mut leaderboard: Vec<CorpusScoreReport> = Vec::with_capacity(top_k + 1);

    // Worst-K board: same shape as `leaderboard`, but kept sorted
    // in DESCENDING ranking_value order so the worst-ranking
    // candidates are retained instead of the best.
    let mut worst_board: Vec<CorpusScoreReport> = Vec::with_capacity(bottom_k + 1);

    let mut prng_state: u64 = match mode {
        SweepMode::PseudoRandom { meta_seed, .. } => meta_seed,
        SweepMode::Linear { .. } => 0, // unused
    };

    for index in 0..mode.count() {
        let seed: u64 = match mode {
            SweepMode::Linear { start, .. } => start.wrapping_add(index),
            SweepMode::PseudoRandom { .. } => splitmix64_step(&mut prng_state),
        };

        let table = generate_table_fisher_yates(seed);
        let mut score = match score_table_on_corpus(&table, corpus, salt_config) {
            Ok(s) => s,
            Err(_) => continue, // should not happen post-checks; defensive
        };
        score.seed = seed;

        // Maintain a bottom-K (highest ranking_value = worst) list,
        // mirror of the top-K logic but with the sort comparator
        // arguments swapped. Clone here because the existing
        // `leaderboard.push(score)` below takes ownership of `score`.
        if bottom_k > 0 {
            worst_board.push(score.clone());
            if worst_board.len() > bottom_k {
                worst_board.sort_by(|a, b| {
                    cmp_ranking(b.ranking_value(rank_by), a.ranking_value(rank_by))
                });
                worst_board.truncate(bottom_k);
            }
        }

        // Maintain a top-K (lowest ranking_value) list.
        // For small K, simple insertion is fine.
        leaderboard.push(score);
        if leaderboard.len() > top_k {
            leaderboard
                .sort_by(|a, b| cmp_ranking(a.ranking_value(rank_by), b.ranking_value(rank_by)));
            leaderboard.truncate(top_k);
        }
    }

    // Final sort of phase-1 leaders.
    leaderboard.sort_by(|a, b| cmp_ranking(a.ranking_value(rank_by), b.ranking_value(rank_by)));

    // Final sort of worst-K, descending (worst-first).
    worst_board.sort_by(|a, b| cmp_ranking(b.ranking_value(rank_by), a.ranking_value(rank_by)));

    // -------------------------------------------------------------
    // PHASE 2 — full structural evaluation of top-K survivors.
    // -------------------------------------------------------------
    for entry in leaderboard.iter_mut() {
        let table = generate_table_fisher_yates(entry.seed);
        let structural = evaluate_table(&table);
        entry.structural = Some(structural);
    }

    // Same Phase-2 attachment for worst-K, so the report can show
    // structural metrics of the worst performers alongside the best.
    for entry in worst_board.iter_mut() {
        let table = generate_table_fisher_yates(entry.seed);
        entry.structural = Some(evaluate_table(&table));
    }

    // -------------------------------------------------------------
    // PHASE 3 — perturbation refinement (single-bit seed flips).
    // -------------------------------------------------------------
    let mut perturbed_pool: Vec<CorpusScoreReport> = Vec::new();
    for survivor in leaderboard.iter() {
        for bit in 0..64u32 {
            let flipped_seed: u64 = survivor.seed ^ (1u64 << bit);
            // Skip if flipped seed already appears in the survivor
            // set (would just re-evaluate the same table).
            let table = generate_table_fisher_yates(flipped_seed);
            let mut score = match score_table_on_corpus(&table, corpus, salt_config) {
                Ok(s) => s,
                Err(_) => continue,
            };
            score.seed = flipped_seed;
            perturbed_pool.push(score);
        }
    }
    perturbed_pool.sort_by(|a, b| cmp_ranking(a.ranking_value(rank_by), b.ranking_value(rank_by)));
    perturbed_pool.truncate(top_k);
    // Attach structural reports to perturbed survivors too.
    for entry in perturbed_pool.iter_mut() {
        let table = generate_table_fisher_yates(entry.seed);
        entry.structural = Some(evaluate_table(&table));
    }

    // -------------------------------------------------------------
    // Baselines: PEARSON_1990_TABLE and GENERATED_TABLE.
    // -------------------------------------------------------------
    let mut baseline_1990 = score_table_on_corpus(&PEARSON_1990_TABLE, corpus, salt_config)?;
    baseline_1990.seed = 0; // baselines have no seed; sentinel.
    baseline_1990.structural = Some(evaluate_table(&PEARSON_1990_TABLE));

    let mut baseline_generated = score_table_on_corpus(&GENERATED_TABLE, corpus, salt_config)?;
    baseline_generated.seed = 0;
    baseline_generated.structural = Some(evaluate_table(&GENERATED_TABLE));

    let elapsed = start_time.elapsed().as_secs_f64();

    Ok(SearchReport {
        corpus_size: corpus.len(),
        corpus_entry_lengths_min: min_len,
        corpus_entry_lengths_max: max_len,
        seeds_evaluated: mode.count(),
        sweep_mode_description: mode.description(),
        rank_by_description: rank_by.description(),
        salt_array_used: salt_config.map(|c| c.salts.len()),
        top_k: leaderboard,
        baseline_1990,
        baseline_generated,
        perturbation_top_k: perturbed_pool,
        worst_k: worst_board,
        elapsed_seconds: elapsed,
    })
}

/// Run a perturbation-only search against a single user-supplied seed.
///
/// # Project-Level Context
///
/// The full `search_seeds` workflow performs three phases:
///   Phase 1 — score every candidate seed in a sweep (cheap),
///   Phase 2 — attach structural metrics to the top-K survivors,
///   Phase 3 — refine those survivors by flipping each of their 64
///             seed-bits one at a time and re-scoring.
///
/// In normal use, a developer runs the full three-phase sweep, reads
/// the leaderboard, and copies a promising seed out of the report.
/// They often want to revisit that same seed later — perhaps after
/// trying a different corpus, perhaps to look more carefully at its
/// neighborhood — without paying for another 10,000-seed sweep.
///
/// This function exists for that workflow: given one hex seed, do
/// only Phase 3 (the 64 single-bit-flip neighbors) plus a re-score
/// of the original seed, attach structural metrics, and return a
/// `SearchReport` that reuses the existing print/save pipeline.
///
/// # Output Shape
///
/// The returned `SearchReport` is laid out so that the existing
/// `render_search_report` pretty-prints it correctly:
///
///   - `top_k`              has one entry  — the user-supplied seed.
///   - `perturbation_top_k` has up to `top_k` entries — the best of
///                          the 64 single-bit-flip neighbors, ranked
///                          by `rank_by`.
///   - `baseline_1990` and `baseline_generated` are populated as
///                          usual for reference.
///   - `seeds_evaluated` is exactly 65 (the seed itself + its 64
///                          one-bit-flip neighbors).
///   - `sweep_mode_description` carries a human-readable string
///                          that distinguishes this from a sweep.
///
/// # Arguments
///
/// * `seed` — the 64-bit Fisher-Yates seed the user wants to refine.
///   Any `u64` value is accepted; there is no requirement that it
///   came from a previous sweep.
/// * `corpus` — the user's representative input data. Must be
///   non-empty; see also `score_table_on_corpus`.
/// * `salt_config` — optional salt-array test configuration. If
///   `None`, `salt coll` is reported as `-` in the leaderboard.
/// * `rank_by` — which metric to sort the perturbed neighbors by
///   (same semantics as `search_seeds`).
/// * `top_k` — how many of the 64 neighbors to keep on the
///   leaderboard. Must be ≥ 1.
///
/// # Returns
///
/// A populated `SearchReport`, suitable for passing directly to
/// `print_and_save_search_report`.
///
/// # Errors
///
/// Returns `Err(std::io::Error)` with a `"TF:"` prefix on:
///   - `"TF: empty corpus"` if `corpus.is_empty()`.
///   - `"TF: zero top_k"`  if `top_k == 0`.
///   - any error propagated from `score_table_on_corpus` for the
///     baseline tables (e.g. invalid salt config).
///
/// Per-neighbor scoring failures (which should not occur in normal
/// use) are silently skipped rather than aborting the whole run,
/// matching the resilience posture of `search_seeds`.
pub fn perturb_seed_search(
    seed: u64,
    corpus: &[&[u8]],
    salt_config: Option<&SaltArrayConfig>,
    rank_by: RankBy,
    top_k: usize,
    bottom_k: usize,
) -> Result<SearchReport, Error> {
    // ---- Defensive input checks (production-safe; no panic) ----
    if corpus.is_empty() {
        return Err(Error::new(ErrorKind::InvalidInput, "TF: empty corpus"));
    }
    if top_k == 0 {
        return Err(Error::new(ErrorKind::InvalidInput, "TF: zero top_k"));
    }

    let start_time = Instant::now();

    // Corpus length stats; identical to the bookkeeping in search_seeds
    // so the report header reads the same regardless of which entry
    // point produced the SearchReport.
    let mut min_entry_length: usize = usize::MAX;
    let mut max_entry_length: usize = 0;
    for entry in corpus.iter() {
        if entry.len() < min_entry_length {
            min_entry_length = entry.len();
        }
        if entry.len() > max_entry_length {
            max_entry_length = entry.len();
        }
    }

    // -----------------------------------------------------------------
    // Score the user-supplied "base" seed itself.
    //
    // It is placed in `top_k` (a single-element vec) so that the
    // existing rendering code labels it as the Phase-1 entry and
    // prints its detailed structural metrics block. Conceptually it
    // is the "thing we are perturbing around", not a sweep winner.
    // -----------------------------------------------------------------
    let original_table: [u8; 256] = generate_table_fisher_yates(seed);
    let mut original_score = score_table_on_corpus(&original_table, corpus, salt_config)?;
    original_score.seed = seed;
    original_score.structural = Some(evaluate_table(&original_table));

    // -----------------------------------------------------------------
    // Score the 64 single-bit-flip neighbors.
    //
    // This is exactly the same neighborhood `search_seeds` would
    // generate in its Phase 3 for a single survivor seed. We attach
    // the full structural metrics to each survivor so the detail
    // block can render the cycle/displacement/correlation fields.
    // -----------------------------------------------------------------
    let mut perturbed_pool: Vec<CorpusScoreReport> = Vec::with_capacity(64);
    for bit_position in 0..64u32 {
        let flipped_seed: u64 = seed ^ (1u64 << bit_position);
        let flipped_table: [u8; 256] = generate_table_fisher_yates(flipped_seed);

        // Per-neighbor scoring failure is treated as "skip this
        // neighbor" rather than aborting the whole run. In practice
        // this branch is unreachable because we already validated
        // the corpus and salt config above; the `continue` is a
        // defensive belt-and-braces guard.
        let mut neighbor_score = match score_table_on_corpus(&flipped_table, corpus, salt_config) {
            Ok(value) => value,
            Err(_) => continue,
        };
        neighbor_score.seed = flipped_seed;
        neighbor_score.structural = Some(evaluate_table(&flipped_table));
        perturbed_pool.push(neighbor_score);
    }

    // Rank and trim to top_k. Tied-rank ordering is whatever
    // `partial_cmp` produces on the underlying f64; this is stable
    // enough for a developer-facing leaderboard.
    perturbed_pool.sort_by(|a, b| cmp_ranking(a.ranking_value(rank_by), b.ranking_value(rank_by)));

    // Capture worst-K BEFORE the truncate below removes them.
    // After the ascending sort above, the worst entries sit at the
    // end of the vector, so we slice from the tail and reverse so
    // the worst is first. Structural metrics were already attached
    // to every neighbor during scoring, so no extra Phase-2 loop
    // is needed for worst_pool.
    let worst_pool: Vec<CorpusScoreReport> = if bottom_k == 0 || perturbed_pool.is_empty() {
        Vec::new()
    } else {
        let pool_len: usize = perturbed_pool.len();
        let take_count: usize = if bottom_k > pool_len {
            pool_len
        } else {
            bottom_k
        };
        let mut tail: Vec<CorpusScoreReport> = perturbed_pool[pool_len - take_count..].to_vec();
        tail.reverse(); // worst-first order
        tail
    };

    perturbed_pool.truncate(top_k);

    // -----------------------------------------------------------------
    // Baselines, computed identically to `search_seeds` so the
    // "Baselines" row of the report has the same meaning regardless
    // of which entry point produced the SearchReport.
    // -----------------------------------------------------------------
    let mut baseline_1990 = score_table_on_corpus(&PEARSON_1990_TABLE, corpus, salt_config)?;
    baseline_1990.seed = 0; // sentinel; the renderer replaces this with a label
    baseline_1990.structural = Some(evaluate_table(&PEARSON_1990_TABLE));

    let mut baseline_generated = score_table_on_corpus(&GENERATED_TABLE, corpus, salt_config)?;
    baseline_generated.seed = 0;
    baseline_generated.structural = Some(evaluate_table(&GENERATED_TABLE));

    let elapsed_seconds = start_time.elapsed().as_secs_f64();

    Ok(SearchReport {
        corpus_size: corpus.len(),
        corpus_entry_lengths_min: min_entry_length,
        corpus_entry_lengths_max: max_entry_length,
        seeds_evaluated: 65, // base seed + 64 single-bit flips
        sweep_mode_description: format!(
            "Perturbation-only: base seed 0x{:016X}, 64 single-bit flips",
            seed
        ),
        rank_by_description: rank_by.description(),
        salt_array_used: salt_config.map(|c| c.salts.len()),
        top_k: vec![original_score],
        baseline_1990,
        baseline_generated,
        perturbation_top_k: perturbed_pool,
        worst_k: worst_pool,
        elapsed_seconds,
    })
}

// =============================================================================
// SECTION 7: Chess-board corpus generator
// =============================================================================

/// Per-square alphabet for the chess corpus generator.
///
/// One byte per square. The 13 possible values are:
/// `'.'` (empty) plus the 12 standard chess pieces in algebraic
/// notation (uppercase = white, lowercase = black).
const CHESS_ALPHABET: [u8; 13] = [
    b'.', b'P', b'N', b'B', b'R', b'Q', b'K', b'p', b'n', b'b', b'r', b'q', b'k',
];

/// Build a deterministic sample of pseudo-random chess-board byte
/// arrays.
///
/// ## Project-Level Context
///
/// This is a **stress-test generator**, not a legal-chess-position
/// generator. It produces 64-byte arrays drawn from the chess
/// alphabet, biased toward `'.'` (empty) to approximate the
/// sparsity of real boards. It does NOT enforce legal piece counts,
/// king presence, pawn-on-back-rank rules, or anything else
/// chess-specific.
///
/// For real production table selection, replace this corpus with
/// your actual position log, opening-book positions, endgame
/// tablebase positions, or whatever real-world data you have.
///
/// ## Encoding
///
/// Each board is a `[u8; 64]` in row-major order (a8..h8, then
/// a7..h7, ..., a1..h1 — but order does not matter for hashing).
///
/// ## Sparsity model
///
/// To make the corpus resemble real boards rather than uniform
/// noise: each square is empty with probability ~50%, otherwise
/// drawn uniformly from the 12 piece types. Real chess positions
/// are typically 60-80% empty; 50% is close enough for a stress
/// test and produces more piece-byte variation per board.
///
/// ## Arguments
///
/// * `seed`  — splitmix64 seed; same seed yields same corpus.
/// * `count` — number of boards to generate.
///
/// ## Returns
///
/// `Vec<[u8; 64]>` of length `count`.
pub fn build_chess_corpus_sample(seed: u64, count: usize) -> Vec<[u8; 64]> {
    let mut state: u64 = seed;
    let mut corpus: Vec<[u8; 64]> = Vec::with_capacity(count);

    for _board_index in 0..count {
        let mut board: [u8; 64] = [b'.'; 64];

        // Per board, produce 64 squares.
        // Each square: one PRNG draw, top bit decides empty/piece,
        // next 4 bits index into the piece subset (12 pieces).
        for square in 0..64usize {
            let r: u64 = splitmix64_step(&mut state);

            // Bit 0: empty (1) or piece (0). Gives ~50% empty rate.
            if (r & 1) == 1 {
                board[square] = b'.';
            } else {
                // Bits 1..5: 4 bits index 0..15; modulo 12 gives a
                // piece in the range 1..=12 of CHESS_ALPHABET.
                let piece_index: usize = (((r >> 1) & 0x0F) as usize) % 12;
                // +1 because CHESS_ALPHABET[0] is '.'.
                board[square] = CHESS_ALPHABET[1 + piece_index];
            }
        }

        corpus.push(board);
    }

    corpus
}

// =============================================================================
// SECTION 8: Printing and file output
// =============================================================================

/// Format the current UTC time as `YYYYmmdd_HHMMSS` for use as a
/// filename suffix.
///
/// # Project-Level Context
///
/// Saved leaderboard files are named `perm_test_{suffix}.txt`. The
/// suffix is purely an ordering marker for a developer eyeballing
/// the directory listing, not a precise wall-clock record:
///
///   - UTC is used (not local time) so the result has no dependency
///     on platform timezone APIs and is consistent across machines.
///   - No leap-second handling; off by ≤1 second is fine for a
///     run-ordering marker.
///   - The 15-character fixed-width format sorts correctly as a
///     plain string (lexicographic order == chronological order).
///
/// # Returns
///
/// A 15-character heap-allocated `String`: 8 digits of date, an
/// underscore, then 6 digits of time, e.g. `20251115_143027`.
fn timestamp_filename_suffix() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs: u64 = now.as_secs();

    // Convert seconds-since-epoch to Y/M/D/H/M/S (UTC).
    let s: u64 = secs % 60;
    let m: u64 = (secs / 60) % 60;
    let h: u64 = (secs / 3600) % 24;
    let days_since_epoch: u64 = secs / 86_400;

    // Civil-from-days (Howard Hinnant's algorithm, simplified).
    let z: i64 = days_since_epoch as i64 + 719_468;
    let era: i64 = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe: i64 = z - era * 146_097;
    let yoe: i64 = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y: i64 = yoe + era * 400;
    let doy: i64 = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp: i64 = (5 * doy + 2) / 153;
    let d: i64 = doy - (153 * mp + 2) / 5 + 1;
    let mo: i64 = if mp < 10 { mp + 3 } else { mp - 9 };
    let year: i64 = if mo <= 2 { y + 1 } else { y };

    format!("{:04}{:02}{:02}_{:02}{:02}{:02}", year, mo, d, h, m, s)
}

/// Format the full structural detail block for one survivor.
///
/// # Project-Level Context
///
/// The leaderboard table can only fit two of the six structural
/// metrics documented in `pearson_hash_tools.rs`. The remaining four
/// — cycle structure, displacement, sequential correlation, and
/// empirical-collisions-on-the-tools'-reference-corpus — are
/// nevertheless computed for every Phase-2 survivor and stored in
/// the `structural` field of the `CorpusScoreReport`. This function
/// surfaces them as a multi-line indented block underneath each
/// phase of the leaderboard, so the reader has the full picture
/// without making the main table unreadable.
///
/// The two metrics that ARE in the main table (XOR-worst and
/// fixed-point count) are repeated here for completeness, with
/// added context (e.g. the `d` value at which XOR-worst was hit).
///
/// # Note on Field Names
///
/// `TableQualityReport` stores cycle statistics as flat fields
/// (`one_cycle_count`, `two_cycle_count`, `longest_cycle`,
/// `cycle_lengths`) rather than nested under a `cycles` sub-struct.
/// "Total cycle count" is derived here as `cycle_lengths.len()`,
/// which is guaranteed correct because the cycle decomposition of
/// a permutation puts each element into exactly one cycle, so the
/// number of entries in `cycle_lengths` IS the number of cycles.
///
/// # Important Caveat About `empirical coll.`
///
/// The `empirical_collisions` field on `TableQualityReport` is
/// counted against the *tools' built-in fixed reference corpus*,
/// not against the user's search corpus. The two should not be
/// conflated. The leaderboard's `base coll` and `salt coll` columns
/// are the user-corpus measurements; `empirical coll.` here is the
/// reference-corpus measurement. The block calls this out in plain
/// text so a reader cannot accidentally confuse them.
///
/// # Arguments
///
/// * `rank` — same rank used in the corresponding leaderboard line,
///   for cross-referencing.
/// * `report` — one survivor; if its `structural` field is `None`
///   this function returns an empty string (the survivor has not
///   been through Phase 2).
///
/// # Returns
///
/// A heap-allocated multi-line `String`, ending in a newline. Empty
/// if no structural report is available.
fn format_structural_detail_block(rank: usize, report: &CorpusScoreReport) -> String {
    let structural = match &report.structural {
        Some(s) => s,
        None => return String::new(),
    };

    let mut out = String::with_capacity(512);

    out.push_str(&format!("  #{:<2} seed = 0x{:016X}\n", rank, report.seed));

    // Cycle structure (flat fields on TableQualityReport).
    out.push_str(&format!(
        "      fixed points:        {}    one-cycles: {}    two-cycles: {}\n",
        structural.fixed_point_count, structural.one_cycle_count, structural.two_cycle_count,
    ));
    // `cycle_lengths.len()` is the number of disjoint cycles in the
    // permutation's cycle decomposition — see doc note above.
    out.push_str(&format!(
        "      longest cycle:       {}    total cycles: {}\n",
        structural.longest_cycle,
        structural.cycle_lengths.len(),
    ));

    // Displacement sub-struct (verified field names: min_/max_/mean_displacement).
    out.push_str(&format!(
        "      displacement:        min {}  max {}  mean {:.3}\n",
        structural.displacement.min_displacement,
        structural.displacement.max_displacement,
        structural.displacement.mean_displacement,
    ));

    out.push_str(&format!(
        "      seq. correlation:    |r| = {:.6}\n",
        structural.sequential_correlation_abs,
    ));

    // XOR uniformity sub-struct: worst_chi_square, worst_difference, mean_chi_square.
    out.push_str(&format!(
        "      XOR worst chi^2:     {:.2}   (at d = 0x{:02X})\n",
        structural.xor_uniformity.worst_chi_square, structural.xor_uniformity.worst_difference,
    ));
    out.push_str(&format!(
        "      XOR mean  chi^2:     {:.2}\n",
        structural.xor_uniformity.mean_chi_square,
    ));

    // Empirical collisions: on the tools' built-in reference corpus,
    // NOT on the search corpus. Flagged in plain text to prevent
    // confusion with the leaderboard's `base coll` / `salt coll`.
    out.push_str(&format!(
        "      empirical coll.:     {}   (on the tools' built-in fixed\n",
        structural.empirical_collisions,
    ));
    out.push_str("                            reference corpus — NOT on your\n");
    out.push_str("                            search corpus; see 'base coll' /\n");
    out.push_str("                            'salt coll' for your-corpus counts)\n");

    out
}

// =============================================================================
// SECTION 8.A: Wide-leaderboard column widths (single source of truth)
// =============================================================================
//
// All four leaderboard rows — direction-marker row, header row, score line,
// and baseline line — pass through the same nineteen `{:>W}` width slots.
// To prevent the four from drifting apart over time, the widths live here
// once, as named constants, and every formatter below references the same
// constants. If you add or remove a column, you must:
//   1. add/remove a constant in this block,
//   2. add/remove one slot in `format_direction_row`,
//   3. add/remove one slot in `format_header_row`,
//   4. add/remove one slot in `format_score_line`,
//   5. extend the legend text in `format_legend_block`.
// There is no way around step 5; the legend is prose, not data.

const LB_W_RK: usize = 4; // rank
const LB_W_SEED: usize = 18; // "0x" + 16 hex digits
const LB_W_BASE_COLL: usize = 10; // corpus base-hash colliding pairs
const LB_W_BASE_CHI2: usize = 11; // corpus base-hash chi-square
const LB_W_MAX: usize = 5; // worst base-hash bucket occupancy
const LB_W_EMPTY: usize = 5; // empty base-hash buckets
const LB_W_SALT_COLL: usize = 10; // corpus salt-array colliding pairs
const LB_W_FIX: usize = 4; // fixed_point_count
const LB_W_1CYC: usize = 5; // one_cycle_count
const LB_W_2CYC: usize = 5; // two_cycle_count
const LB_W_LONG: usize = 5; // longest_cycle
const LB_W_TOT: usize = 4; // cycle_lengths.len()
const LB_W_DMIN: usize = 5; // displacement.min_displacement
const LB_W_DMAX: usize = 5; // displacement.max_displacement
const LB_W_DMEAN: usize = 8; // displacement.mean_displacement (f64 .2)
const LB_W_R: usize = 9; // sequential_correlation_abs    (f64 .5)
const LB_W_XWC: usize = 9; // xor_uniformity.worst_chi_square (f64 .2)
const LB_W_XMC: usize = 9; // xor_uniformity.mean_chi_square  (f64 .2)
const LB_W_REFCOLL: usize = 9; // empirical_collisions

// =============================================================================
// SECTION 8.B: Direction-marker row
// =============================================================================

/// Build the single "direction marker" line printed directly above the
/// leaderboard's column-header line.
///
/// # Project-Level Context
///
/// Each leaderboard column has a fixed "what counts as better" interpretation
/// — lower-is-better for collision counts, higher-is-better for displacement
/// mean, closer-to-zero for the sequential correlation, and so on. Printing
/// that interpretation once, aligned over the column headers, lets the
/// reader interpret the numbers below without consulting the prose legend
/// for every column.
///
/// # Marker Glyphs
///
///   `↓`   lower is better
///   `↑`   higher is better
///   `→0`  closer to zero is better
///   `n/a` informational; no preferred direction
///
/// # Returns
///
/// A heap-allocated `String`, one line, no trailing newline.
fn format_direction_row() -> String {
    format!(
        "  {:>w01$}   {:>w02$}   {:>w03$}   {:>w04$}   {:>w05$}   {:>w06$}   \
           {:>w07$}   {:>w08$}   {:>w09$}   {:>w10$}   {:>w11$}   {:>w12$}   \
           {:>w13$}   {:>w14$}   {:>w15$}   {:>w16$}   {:>w17$}   {:>w18$}   \
           {:>w19$}",
        "n/a",
        "n/a",       // rk, seed   (identifiers, not metrics)
        "\u{2193}",  // base coll  ↓
        "\u{2193}",  // base chi²  ↓
        "\u{2193}",  // max        ↓
        "\u{2193}",  // empty      ↓
        "\u{2193}",  // salt coll  ↓
        "\u{2193}",  // fix        ↓
        "\u{2193}",  // 1cyc       ↓
        "\u{2193}",  // 2cyc       ↓
        "\u{2191}",  // long       ↑
        "\u{2193}",  // tot        ↓
        "n/a",       // dmin       informational
        "\u{2191}",  // dmax       ↑
        "\u{2191}",  // dmean      ↑
        "\u{2192}0", // |r|        →0
        "\u{2193}",  // XwC        ↓
        "\u{2193}",  // XmC        ↓
        "\u{2193}",  // ref-coll   ↓
        w01 = LB_W_RK,
        w02 = LB_W_SEED,
        w03 = LB_W_BASE_COLL,
        w04 = LB_W_BASE_CHI2,
        w05 = LB_W_MAX,
        w06 = LB_W_EMPTY,
        w07 = LB_W_SALT_COLL,
        w08 = LB_W_FIX,
        w09 = LB_W_1CYC,
        w10 = LB_W_2CYC,
        w11 = LB_W_LONG,
        w12 = LB_W_TOT,
        w13 = LB_W_DMIN,
        w14 = LB_W_DMAX,
        w15 = LB_W_DMEAN,
        w16 = LB_W_R,
        w17 = LB_W_XWC,
        w18 = LB_W_XMC,
        w19 = LB_W_REFCOLL,
    )
}

// =============================================================================
// SECTION 8.C: Column-header row
// =============================================================================

/// Build the leaderboard column-header line.
///
/// # Project-Level Context
///
/// Printed immediately under the direction-marker row, immediately above
/// the horizontal divider, in both Phase 1 and Phase 3 sections. The
/// header *abbreviations* are deliberately short because the prose legend
/// (printed once per phase) maps each abbreviation to its long description.
///
/// # Returns
///
/// A heap-allocated `String`, one line, no trailing newline.
fn format_header_row() -> String {
    format!(
        "  {:>w01$}   {:>w02$}   {:>w03$}   {:>w04$}   {:>w05$}   {:>w06$}   \
           {:>w07$}   {:>w08$}   {:>w09$}   {:>w10$}   {:>w11$}   {:>w12$}   \
           {:>w13$}   {:>w14$}   {:>w15$}   {:>w16$}   {:>w17$}   {:>w18$}   \
           {:>w19$}",
        "rk",
        "seed (hex)",
        "base coll",
        "base chi\u{00B2}", // "chi²"
        "max",
        "empty",
        "salt coll",
        "fix",
        "1cyc",
        "2cyc",
        "long",
        "tot",
        "dmin",
        "dmax",
        "dmean",
        "|r|",
        "XwC",
        "XmC",
        "ref-coll",
        w01 = LB_W_RK,
        w02 = LB_W_SEED,
        w03 = LB_W_BASE_COLL,
        w04 = LB_W_BASE_CHI2,
        w05 = LB_W_MAX,
        w06 = LB_W_EMPTY,
        w07 = LB_W_SALT_COLL,
        w08 = LB_W_FIX,
        w09 = LB_W_1CYC,
        w10 = LB_W_2CYC,
        w11 = LB_W_LONG,
        w12 = LB_W_TOT,
        w13 = LB_W_DMIN,
        w14 = LB_W_DMAX,
        w15 = LB_W_DMEAN,
        w16 = LB_W_R,
        w17 = LB_W_XWC,
        w18 = LB_W_XMC,
        w19 = LB_W_REFCOLL,
    )
}

// =============================================================================
// SECTION 8.D: Prose legend block
// =============================================================================

/// Build the multi-line prose legend explaining every leaderboard column
/// and every direction-marker glyph.
///
/// # Project-Level Context
///
/// Printed before Phase 1 and again before Phase 3. The legend is the
/// single place where each column abbreviation is mapped to its full
/// description; the column-header row only carries the abbreviations.
/// Repeating the legend at Phase 3 lets a reader scroll directly to the
/// perturbation results without losing context.
///
/// # Returns
///
/// A heap-allocated `String`, multi-line, ending in a newline.
fn format_legend_block() -> String {
    let mut out = String::with_capacity(2048);

    out.push_str("Direction markers in the row above the column headers:\n");
    out.push_str("  \u{2193}     lower is better\n");
    out.push_str("  \u{2191}     higher is better\n");
    out.push_str("  \u{2192}0    closer to zero is better\n");
    out.push_str("  n/a   informational; no preferred direction\n");
    out.push('\n');

    out.push_str("Leaderboard columns:\n");
    out.push_str("  rk         leaderboard rank within this phase (1 = best by ranked metric)\n");
    out.push_str("  seed       64-bit Fisher-Yates seed (hex)\n");
    out.push_str("  base coll  unordered colliding pairs on YOUR corpus, base 8-bit hash\n");
    out.push_str("  base chi\u{00B2}  chi-square of base-hash bucket histogram vs uniform\n");
    out.push_str("  max        worst-case base-hash bucket occupancy on YOUR corpus\n");
    out.push_str("  empty      number of empty base-hash buckets (out of 256), YOUR corpus\n");
    out.push_str("  salt coll  unordered colliding pairs on the composite [u8; N] salt-array\n");
    out.push_str("             key on YOUR corpus ('-' if salt-array test was disabled)\n");
    out.push_str("  fix        fixed-point count: positions i where T[i] == i\n");
    out.push_str("  1cyc       count of one-cycles (== fix; both listed for completeness)\n");
    out.push_str("  2cyc       count of two-cycles (transpositions)\n");
    out.push_str("  long       longest cycle in the permutation's cycle decomposition\n");
    out.push_str("  tot        total number of disjoint cycles in the decomposition\n");
    out.push_str("  dmin       displacement min:  min |T[i] - i|  (informational; 0 normal)\n");
    out.push_str("  dmax       displacement max:  max |T[i] - i|\n");
    out.push_str("  dmean      displacement mean: mean |T[i] - i|\n");
    out.push_str("  |r|        |Pearson correlation of T[i] vs T[i+1]| over 0..255\n");
    out.push_str("  XwC        *** primary structural metric ***\n");
    out.push_str(
        "             XOR-worst chi\u{00B2}: worst-case chi-square over all 255 nonzero\n",
    );
    out.push_str("             XOR differences d of the histogram { T[i] XOR T[i XOR d] };\n");
    out.push_str("             governs the avalanche behavior of the Pearson hash\n");
    out.push_str("  XmC        XOR-mean chi\u{00B2}: mean of the same chi-square over all 255 d\n");
    out.push_str("  ref-coll   empirical collisions on the TOOLS' built-in fixed reference\n");
    out.push_str("             corpus (NOT your search corpus; high variance — use the\n");
    out.push_str("             'base coll' and 'salt coll' columns for your-corpus signal)\n");
    out.push('\n');

    out
}

// =============================================================================
// SECTION 8.E: One leaderboard row
// =============================================================================

/// Format one `CorpusScoreReport` as a single leaderboard line.
///
/// # Project-Level Context
///
/// The leaderboard table is wide on purpose. It carries all four corpus
/// metrics, the salt-array corpus metric, and all twelve structural
/// metrics from `pearson_hash_tools` side-by-side, so the user can scan
/// the rank order and the trade-offs together. The per-survivor
/// "Detailed structural metrics" block under each phase covers the same
/// values plus the few additional context items (e.g. the `d` at which
/// `XwC` was hit) that don't ride in the table.
///
/// # Column Widths
///
/// All widths come from the named constants `LB_W_*` defined at the top
/// of SECTION 8. The direction-marker row, the column-header row, the
/// score line, and the baseline line all use those same constants; do
/// not introduce literal widths here.
///
/// # Special Cases
///
/// * `salt_array_collisions == usize::MAX` renders as `-` (test disabled).
/// * `structural.is_none()` renders every structural column as `-`. In
///   the current code paths this cannot happen (all leaderboard entries
///   go through Phase 2), but the contract still allows it.
///
/// # Arguments
///
/// * `rank` — printed in the leftmost column. The caller controls
///   ranking semantics; `0` is used for baselines and the renderer
///   later rewrites the seed cell into a `[1990 TABLE]` / `[GENERATED
///   TABLE]` label.
/// * `report` — one survivor's metrics.
///
/// # Returns
///
/// A heap-allocated `String`, one line, no trailing newline.
fn format_score_line(rank: usize, report: &CorpusScoreReport) -> String {
    // Salt-array column: sentinel -> dash, real number -> right-aligned.
    let salt_field: String = if report.salt_array_collisions == usize::MAX {
        format!("{:>w$}", "-", w = LB_W_SALT_COLL)
    } else {
        format!("{:>w$}", report.salt_array_collisions, w = LB_W_SALT_COLL)
    };

    // All twelve structural columns. Built as one tuple so the
    // "present" branch and the "absent" branch stay symmetric.
    let (fix_s, c1_s, c2_s, long_s, tot_s, dmin_s, dmax_s, dmean_s, r_s, xwc_s, xmc_s, ref_s): (
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
    ) = match &report.structural {
        Some(s) => (
            format!("{:>w$}", s.fixed_point_count, w = LB_W_FIX),
            format!("{:>w$}", s.one_cycle_count, w = LB_W_1CYC),
            format!("{:>w$}", s.two_cycle_count, w = LB_W_2CYC),
            format!("{:>w$}", s.longest_cycle, w = LB_W_LONG),
            format!("{:>w$}", s.cycle_lengths.len(), w = LB_W_TOT),
            format!("{:>w$}", s.displacement.min_displacement, w = LB_W_DMIN),
            format!("{:>w$}", s.displacement.max_displacement, w = LB_W_DMAX),
            format!("{:>w$.2}", s.displacement.mean_displacement, w = LB_W_DMEAN),
            format!("{:>w$.5}", s.sequential_correlation_abs, w = LB_W_R),
            format!("{:>w$.2}", s.xor_uniformity.worst_chi_square, w = LB_W_XWC),
            format!("{:>w$.2}", s.xor_uniformity.mean_chi_square, w = LB_W_XMC),
            format!("{:>w$}", s.empirical_collisions, w = LB_W_REFCOLL),
        ),
        None => (
            format!("{:>w$}", "-", w = LB_W_FIX),
            format!("{:>w$}", "-", w = LB_W_1CYC),
            format!("{:>w$}", "-", w = LB_W_2CYC),
            format!("{:>w$}", "-", w = LB_W_LONG),
            format!("{:>w$}", "-", w = LB_W_TOT),
            format!("{:>w$}", "-", w = LB_W_DMIN),
            format!("{:>w$}", "-", w = LB_W_DMAX),
            format!("{:>w$}", "-", w = LB_W_DMEAN),
            format!("{:>w$}", "-", w = LB_W_R),
            format!("{:>w$}", "-", w = LB_W_XWC),
            format!("{:>w$}", "-", w = LB_W_XMC),
            format!("{:>w$}", "-", w = LB_W_REFCOLL),
        ),
    };

    format!(
        "  {:>w01$}   0x{:016X}   {:>w03$}   {:>w04$.2}   {:>w05$}   {:>w06$}   {}   \
           {}   {}   {}   {}   {}   {}   {}   {}   {}   {}   {}   {}",
        rank,
        report.seed,
        report.base_collisions,
        report.base_chi_square,
        report.base_max_bucket,
        report.base_empty_buckets,
        salt_field,
        fix_s,
        c1_s,
        c2_s,
        long_s,
        tot_s,
        dmin_s,
        dmax_s,
        dmean_s,
        r_s,
        xwc_s,
        xmc_s,
        ref_s,
        w01 = LB_W_RK,
        w03 = LB_W_BASE_COLL,
        w04 = LB_W_BASE_CHI2,
        w05 = LB_W_MAX,
        w06 = LB_W_EMPTY,
    )
}

// =============================================================================
// SECTION 8.F: Whole report
// =============================================================================

/// Build the full report text.
///
/// # Project-Level Context
///
/// This function is the single source of truth for the rendered
/// report text. `print_and_save_search_report` calls it once, then
/// both writes the result to stdout and saves it to a timestamped
/// `.txt` file. That guarantees the on-disk file and the terminal
/// transcript are byte-identical, which matters when a developer
/// goes back to read a saved file weeks later and wants to compare
/// it to a fresh terminal run.
///
/// # Section Layout
///
/// Sections are printed in this order:
///   1. Run-metadata header (corpus, sweep, ranking, elapsed).
///   2. Column legend, placed BEFORE the table so the reader has
///      the key in front of them while reading the numbers.
///   3. Column header line + horizontal divider.
///   4. Phase-1 leaderboard rows.
///   5. Phase-1 structural-detail blocks.
///   6. Phase-3 leaderboard rows.
///   7. Phase-3 structural-detail blocks.
///   8. Baseline rows (1990 table, default GENERATED_TABLE).
///   9. "How to use a chosen seed" footer.
///
/// # Why the Legend Is Above the Table
///
/// In the previous version of this report, the legend was at the
/// bottom; readers reported needing to scroll back and forth to map
/// column headings to meanings. Placing it above the table is a
/// readability win, at the cost of a slightly taller report.
///
/// # Heap Usage
///
/// This is tools / demo code. `String` and `format!` are used freely.
/// The production hashing functions called from earlier phases do
/// not allocate; only this rendering layer does.
fn render_search_report(report: &SearchReport) -> String {
    let mut out = String::with_capacity(32_768);

    // 1. Run-metadata header.
    out.push_str("================================================================\n");
    out.push_str(" Pearson permutation table search — results report\n");
    out.push_str("================================================================\n");
    out.push_str(&format!(
        "Corpus size:        {} entries\n",
        report.corpus_size
    ));
    out.push_str(&format!(
        "Entry lengths:      min {}, max {} bytes\n",
        report.corpus_entry_lengths_min, report.corpus_entry_lengths_max
    ));
    out.push_str(&format!(
        "Sweep:              {}\n",
        report.sweep_mode_description
    ));
    out.push_str(&format!("Seeds evaluated:    {}\n", report.seeds_evaluated));
    out.push_str(&format!(
        "Ranked by:          {}\n",
        report.rank_by_description
    ));
    out.push_str(&format!(
        "Salt-array test:    {}\n",
        match report.salt_array_used {
            Some(n) => format!("yes ({} salt(s), padded to MAX_SALTS = {})", n, MAX_SALTS),
            None => String::from("no"),
        }
    ));
    out.push_str(&format!(
        "Elapsed:            {:.2} seconds\n",
        report.elapsed_seconds
    ));
    out.push('\n');

    // Reusable strings; cheap to clone but easier to read if we build them once.
    let legend = format_legend_block();
    let dir_row = format_direction_row();
    let hdr_row = format_header_row();
    let divider = "-".repeat(hdr_row.len());

    // 2. Legend + direction row + header row + divider, above Phase 1.
    out.push_str(&legend);
    out.push_str(&dir_row);
    out.push('\n');
    out.push_str(&hdr_row);
    out.push('\n');
    out.push_str(&divider);
    out.push('\n');

    // 3. Phase 1 rows.
    out.push_str("Phase 1 — best seeds from sweep (or user-supplied base seed):\n");
    for (i, r) in report.top_k.iter().enumerate() {
        out.push_str(&format_score_line(i + 1, r));
        out.push('\n');
    }
    out.push('\n');

    // 4. Phase 1 per-survivor detail blocks.
    out.push_str("Detailed structural metrics for Phase 1 entries:\n");
    for (i, r) in report.top_k.iter().enumerate() {
        out.push_str(&format_structural_detail_block(i + 1, r));
    }
    out.push('\n');

    // Worst-K leaderboard rows + detail blocks, for contrast against
    // the best-K above. Section is omitted entirely if `worst_k` is
    // empty (i.e. the caller passed bottom_k = 0).
    if !report.worst_k.is_empty() {
        out.push_str(&dir_row);
        out.push('\n');
        out.push_str(&hdr_row);
        out.push('\n');
        out.push_str(&divider);
        out.push('\n');

        out.push_str(&format!(
            "Phase 1 \u{2014} WORST {} seeds from sweep (for contrast):\n",
            report.worst_k.len()
        ));
        for (i, r) in report.worst_k.iter().enumerate() {
            out.push_str(&format_score_line(i + 1, r));
            out.push('\n');
        }
        out.push('\n');

        out.push_str("Detailed structural metrics for Phase 1 WORST entries:\n");
        for (i, r) in report.worst_k.iter().enumerate() {
            out.push_str(&format_structural_detail_block(i + 1, r));
        }
        out.push('\n');
    }

    // 5. Full legend block + direction row + header row + divider, REPEATED
    //    above Phase 3 so the phase is readable in isolation.
    out.push_str(&legend);
    out.push_str(&dir_row);
    out.push('\n');
    out.push_str(&hdr_row);
    out.push('\n');
    out.push_str(&divider);
    out.push('\n');

    // 6. Phase 3 rows.
    out.push_str("Phase 3 — best after single-bit-flip perturbation:\n");
    for (i, r) in report.perturbation_top_k.iter().enumerate() {
        out.push_str(&format_score_line(i + 1, r));
        out.push('\n');
    }
    out.push('\n');

    // 7. Phase 3 per-survivor detail blocks.
    out.push_str("Detailed structural metrics for Phase 3 entries:\n");
    for (i, r) in report.perturbation_top_k.iter().enumerate() {
        out.push_str(&format_structural_detail_block(i + 1, r));
    }
    out.push('\n');

    // 8. Baselines, rendered with the same wide row format so every
    //    numeric column lines up with the leaderboard above. The seed
    //    cell sentinel `0x0000000000000000` is rewritten to a bracketed
    //    label after `format_score_line` returns, so the cell width
    //    matches `"0x" + 16 hex digits` = 18 chars.
    out.push_str(&dir_row);
    out.push('\n');
    out.push_str(&hdr_row);
    out.push('\n');
    out.push_str(&divider);
    out.push('\n');
    out.push_str("Baselines (for reference, NOT search results):\n");

    let mut baseline_1990_line = format_score_line(0, &report.baseline_1990);
    baseline_1990_line = baseline_1990_line.replacen("0x0000000000000000", "[1990 TABLE      ]", 1);
    out.push_str(&baseline_1990_line);
    out.push('\n');

    let mut baseline_gen_line = format_score_line(0, &report.baseline_generated);
    baseline_gen_line = baseline_gen_line.replacen("0x0000000000000000", "[GENERATED TABLE ]", 1);
    out.push_str(&baseline_gen_line);
    out.push('\n');
    out.push('\n');

    // 9. Footer.
    out.push_str("To use a chosen seed in production code:\n");
    out.push_str("    const MY_SEED:  u64        = 0x????????????????;\n");
    out.push_str("    const MY_TABLE: [u8; 256]  =\n");
    out.push_str("        generate_table_fisher_yates_const(MY_SEED);\n");
    out.push('\n');
    out.push_str("To refine a promising seed without re-running the full sweep,\n");
    out.push_str("re-run this binary and choose option [2] 'Perturb a specific\n");
    out.push_str("seed' at the section-6 prompt, pasting the hex seed in.\n");
    out.push('\n');

    out
}

/// Print the search report to stdout AND save it to a timestamped
/// file `perm_test_{YYYYmmdd_HHMMSS}.txt` in the current working
/// directory.
///
/// ## Project-Level Context
///
/// Single source of truth for the rendered text: `render_search_report`
/// is called once, then the same string is both printed and written.
///
/// ## Errors
///
/// File-write errors are returned, not panicked on. The stdout
/// output happens unconditionally before the file write is
/// attempted, so even if the file write fails the user has
/// already seen the results.
pub fn print_and_save_search_report(report: &SearchReport) -> Result<PathBuf, Error> {
    let text = render_search_report(report);
    print!("{}", text);

    let filename = format!("perm_test_{}.txt", timestamp_filename_suffix());
    let path = PathBuf::from(&filename);
    let mut file = File::create(&path)?;
    file.write_all(text.as_bytes())?;

    println!("Report saved to: {}", path.display());
    Ok(path)
}

// =============================================================================
// SECTION 9: Cargo tests — code correctness only
// =============================================================================

#[cfg(test)]
mod table_finder_tests {
    use super::*;

    // Test-only import: `is_valid_permutation_tools` is referenced
    // only by `test_imported_tables_are_valid_permutations` below,
    // so we scope it to the test module to avoid an "unused import"
    // warning in non-test builds.
    use crate::pearson_hash_tools::is_valid_permutation_tools;

    /// Build a small known corpus for deterministic tests.
    fn tiny_corpus() -> Vec<Vec<u8>> {
        vec![
            b"alpha".to_vec(),
            b"bravo".to_vec(),
            b"charlie".to_vec(),
            b"delta".to_vec(),
            b"echo".to_vec(),
            b"foxtrot".to_vec(),
            b"golf".to_vec(),
            b"hotel".to_vec(),
        ]
    }
    fn tiny_corpus_refs(c: &[Vec<u8>]) -> Vec<&[u8]> {
        c.iter().map(|v| v.as_slice()).collect()
    }

    // ---- score_table_on_corpus ----

    #[test]
    fn test_score_table_on_corpus_rejects_empty_corpus() {
        let res = score_table_on_corpus(&PEARSON_1990_TABLE, &[], None);
        assert!(res.is_err());
    }

    #[test]
    fn test_score_table_on_corpus_is_deterministic() {
        let owned = tiny_corpus();
        let refs = tiny_corpus_refs(&owned);
        let a = score_table_on_corpus(&PEARSON_1990_TABLE, &refs, None).unwrap();
        let b = score_table_on_corpus(&PEARSON_1990_TABLE, &refs, None).unwrap();
        assert_eq!(a.base_collisions, b.base_collisions);
        assert_eq!(a.base_chi_square, b.base_chi_square);
        assert_eq!(a.base_max_bucket, b.base_max_bucket);
        assert_eq!(a.base_empty_buckets, b.base_empty_buckets);
    }

    #[test]
    fn test_score_table_on_corpus_salt_array_zero_padding_matches() {
        // With only one user salt, the padded form (1 real + 7 zero
        // salts) must produce the same number of collisions as if
        // we only counted the first salt. The zero-padded salts
        // all hash to the same value for every input, so they are
        // constant columns in the [u8; MAX_SALTS] composite key.
        // Constant columns do not change which inputs collide.
        let owned = tiny_corpus();
        let refs = tiny_corpus_refs(&owned);
        let cfg_one = SaltArrayConfig {
            salts: vec![0xDEAD_BEEF_u128],
        };
        let cfg_padded_explicitly = SaltArrayConfig {
            salts: vec![0xDEAD_BEEF_u128],
        };
        let a = score_table_on_corpus(&PEARSON_1990_TABLE, &refs, Some(&cfg_one)).unwrap();
        let b = score_table_on_corpus(&PEARSON_1990_TABLE, &refs, Some(&cfg_padded_explicitly))
            .unwrap();
        assert_eq!(a.salt_array_collisions, b.salt_array_collisions);
    }

    #[test]
    fn test_score_table_on_corpus_rejects_too_many_salts() {
        let owned = tiny_corpus();
        let refs = tiny_corpus_refs(&owned);
        let cfg = SaltArrayConfig {
            salts: vec![0u128; MAX_SALTS + 1],
        };
        let res = score_table_on_corpus(&PEARSON_1990_TABLE, &refs, Some(&cfg));
        assert!(res.is_err());
    }

    // ---- search_seeds ----

    #[test]
    fn test_search_seeds_linear_count_one_returns_one_candidate() {
        let owned = tiny_corpus();
        let refs = tiny_corpus_refs(&owned);
        let report = search_seeds(
            &refs,
            None,
            SweepMode::Linear {
                start: 42,
                count: 1,
            },
            RankBy::BaseCollisions,
            20,
            0,
        )
        .unwrap();
        // top_k is capped at the smaller of top_k and seeds_evaluated.
        assert_eq!(report.top_k.len(), 1);
    }

    #[test]
    fn test_search_seeds_rejects_zero_seeds() {
        let owned = tiny_corpus();
        let refs = tiny_corpus_refs(&owned);
        let res = search_seeds(
            &refs,
            None,
            SweepMode::Linear { start: 0, count: 0 },
            RankBy::BaseCollisions,
            20,
            0,
        );
        assert!(res.is_err());
    }

    #[test]
    fn test_search_seeds_rejects_zero_top_k() {
        let owned = tiny_corpus();
        let refs = tiny_corpus_refs(&owned);
        let res = search_seeds(
            &refs,
            None,
            SweepMode::Linear { start: 0, count: 4 },
            RankBy::BaseCollisions,
            0,
            0,
        );
        assert!(res.is_err());
    }

    #[test]
    fn test_search_seeds_is_deterministic_for_same_mode() {
        let owned = tiny_corpus();
        let refs = tiny_corpus_refs(&owned);
        let a = search_seeds(
            &refs,
            None,
            SweepMode::Linear {
                start: 100,
                count: 50,
            },
            RankBy::BaseCollisions,
            10,
            0,
        )
        .unwrap();
        let b = search_seeds(
            &refs,
            None,
            SweepMode::Linear {
                start: 100,
                count: 50,
            },
            RankBy::BaseCollisions,
            10,
            0,
        )
        .unwrap();
        assert_eq!(a.top_k.len(), b.top_k.len());
        for (x, y) in a.top_k.iter().zip(b.top_k.iter()) {
            assert_eq!(x.seed, y.seed);
            assert_eq!(x.base_collisions, y.base_collisions);
        }
    }

    #[test]
    fn test_search_seeds_top_k_sorted_by_ranking() {
        let owned = tiny_corpus();
        let refs = tiny_corpus_refs(&owned);
        let report = search_seeds(
            &refs,
            None,
            SweepMode::Linear {
                start: 0,
                count: 200,
            },
            RankBy::BaseCollisions,
            20,
            0,
        )
        .unwrap();
        for i in 1..report.top_k.len() {
            assert!(
                report.top_k[i - 1].ranking_value(RankBy::BaseCollisions)
                    <= report.top_k[i].ranking_value(RankBy::BaseCollisions),
                "top_k not sorted at index {}",
                i
            );
        }
    }

    #[test]
    fn test_search_seeds_perturbation_size_capped_at_top_k() {
        let owned = tiny_corpus();
        let refs = tiny_corpus_refs(&owned);
        let report = search_seeds(
            &refs,
            None,
            SweepMode::Linear {
                start: 0,
                count: 30,
            },
            RankBy::BaseCollisions,
            5,
            0,
        )
        .unwrap();
        assert!(report.perturbation_top_k.len() <= 5);
    }

    // ---- chess corpus generator ----

    #[test]
    fn test_build_chess_corpus_sample_count_and_size() {
        let corpus = build_chess_corpus_sample(0xABCDEF, 50);
        assert_eq!(corpus.len(), 50);
        for board in corpus.iter() {
            assert_eq!(board.len(), 64);
        }
    }

    #[test]
    fn test_build_chess_corpus_sample_is_deterministic_in_seed() {
        let a = build_chess_corpus_sample(0xC4E55, 20);
        let b = build_chess_corpus_sample(0xC4E55, 20);
        assert_eq!(a, b);
    }

    #[test]
    fn test_build_chess_corpus_alphabet() {
        let corpus = build_chess_corpus_sample(42, 100);
        for board in corpus.iter() {
            for &square in board.iter() {
                assert!(
                    CHESS_ALPHABET.contains(&square),
                    "square byte {} not in chess alphabet",
                    square
                );
            }
        }
    }

    #[test]
    fn test_build_chess_corpus_different_seeds_differ() {
        let a = build_chess_corpus_sample(1, 10);
        let b = build_chess_corpus_sample(2, 10);
        assert_ne!(a, b);
    }

    // ---- baseline tables remain valid ----

    #[test]
    fn test_imported_tables_are_valid_permutations() {
        assert!(is_valid_permutation_tools(&PEARSON_1990_TABLE));
        assert!(is_valid_permutation_tools(&GENERATED_TABLE));
    }

    // ---- timestamp helper ----

    #[test]
    fn test_timestamp_filename_suffix_format() {
        let s = timestamp_filename_suffix();
        // Format: YYYYmmdd_HHMMSS = 8 + 1 + 6 = 15 chars
        assert_eq!(s.len(), 15);
        let bytes = s.as_bytes();
        assert!(bytes[8] == b'_');
        for (i, &c) in bytes.iter().enumerate() {
            if i == 8 {
                continue;
            }
            assert!(c.is_ascii_digit(), "non-digit at index {}: {}", i, c);
        }
    }

    // --- a cargo test for perturb_seed_search ---

    #[test]
    fn test_perturb_seed_search_basic_shape() {
        // Project-level: this test verifies the structural contract of
        // `perturb_seed_search` — that the returned report has exactly
        // one Phase-1 entry (the user-supplied seed), that the perturbed
        // pool is non-empty and capped at top_k, that every survivor has
        // a structural report attached, and that all reported seeds are
        // valid single-bit flips of the input seed. It does NOT assert
        // anything about which seed wins; the relative ranking depends
        // on the corpus and is not the contract under test.
        let owned = tiny_corpus();
        let refs = tiny_corpus_refs(&owned);

        let base_seed: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let report = perturb_seed_search(base_seed, &refs, None, RankBy::BaseCollisions, 5, 0)
            .expect("perturb_seed_search should succeed on a non-empty corpus");

        // Phase-1 contract.
        assert_eq!(report.top_k.len(), 1);
        assert_eq!(report.top_k[0].seed, base_seed);
        assert!(report.top_k[0].structural.is_some());

        // Phase-3 contract.
        assert!(!report.perturbation_top_k.is_empty());
        assert!(report.perturbation_top_k.len() <= 5);

        // Every Phase-3 seed must differ from `base_seed` by exactly one
        // bit (Hamming distance 1).
        for entry in report.perturbation_top_k.iter() {
            let xor_diff: u64 = entry.seed ^ base_seed;
            assert_eq!(
                xor_diff.count_ones(),
                1,
                "perturbed seed must be one bit away from base seed"
            );
            assert!(entry.structural.is_some());
        }

        // Seeds_evaluated must reflect the documented constant.
        assert_eq!(report.seeds_evaluated, 65);
    }
}
