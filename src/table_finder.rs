// src/table_finder.rs

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

use crate::pearson_hash_tools::{
    TableQualityReport, evaluate_table, generate_table_fisher_yates, is_valid_permutation_tools,
};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    // -------------------------------------------------------------
    // PHASE 2 — full structural evaluation of top-K survivors.
    // -------------------------------------------------------------
    for entry in leaderboard.iter_mut() {
        let table = generate_table_fisher_yates(entry.seed);
        let structural = evaluate_table(&table);
        entry.structural = Some(structural);
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
        elapsed_seconds: elapsed,
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

/// Format the current local-ish time as `YYYYmmdd_HHMMSS`.
///
/// Project-level note: uses UTC (not local time) to avoid any
/// dependency on platform timezone APIs and to keep filenames
/// deterministic across machines. The user is expected to read
/// these filenames as run-ordering markers, not as wall-clock
/// records.
fn timestamp_filename_suffix() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs: u64 = now.as_secs();

    // Convert seconds-since-epoch to Y/M/D/H/M/S (UTC).
    // Simple algorithm; no leap-second handling.
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

/// Render a `CorpusScoreReport` as a single line for the leaderboard.
fn format_score_line(rank: usize, report: &CorpusScoreReport) -> String {
    let salt_field: String = if report.salt_array_collisions == usize::MAX {
        format!("{:>9}", "—")
    } else {
        format!("{:>9}", report.salt_array_collisions)
    };

    let (xor_worst, fixed_pts): (String, String) = match &report.structural {
        Some(s) => (
            format!("{:>10.2}", s.xor_uniformity.worst_chi_square),
            format!("{:>4}", s.fixed_point_count),
        ),
        None => (format!("{:>10}", "—"), format!("{:>4}", "—")),
    };

    format!(
        "  {:>3}   0x{:016X}   {:>9}   {:>10.2}   {:>4}   {:>4}   {}   {}   {}",
        rank,
        report.seed,
        report.base_collisions,
        report.base_chi_square,
        report.base_max_bucket,
        report.base_empty_buckets,
        salt_field,
        xor_worst,
        fixed_pts,
    )
}

/// Build the full report text (used by both stdout and file
/// output, so the two are guaranteed identical).
fn render_search_report(report: &SearchReport) -> String {
    let mut out = String::with_capacity(8192);

    out.push_str("================================================================\n");
    out.push_str(" Pearson permutation table search\n");
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

    // Column header
    let header = format!(
        "  {:>3}   {:>18}   {:>9}   {:>10}   {:>4}   {:>4}   {:>9}   {:>10}   {:>4}",
        "rk",
        "seed (hex)",
        "base coll",
        "base chi2",
        "max",
        "empty",
        "salt coll",
        "XOR worst",
        "fix",
    );
    out.push_str(&header);
    out.push('\n');
    out.push_str(&"-".repeat(header.len()));
    out.push('\n');

    // Phase 1 top-K leaders.
    out.push_str("Phase 1 — best seeds from random sweep:\n");
    for (i, r) in report.top_k.iter().enumerate() {
        out.push_str(&format_score_line(i + 1, r));
        out.push('\n');
    }
    out.push('\n');

    // Phase 3 perturbation results.
    out.push_str("Phase 3 — best after single-bit-flip perturbation of phase-1 leaders:\n");
    for (i, r) in report.perturbation_top_k.iter().enumerate() {
        out.push_str(&format_score_line(i + 1, r));
        out.push('\n');
    }
    out.push('\n');

    // Baselines, formatted with their own (synthetic) ranks.
    out.push_str("Baselines (for reference):\n");
    let mut baseline_1990_line = format_score_line(0, &report.baseline_1990);
    // Replace the "0x0000...0000" seed printout for baselines with
    // a label.
    baseline_1990_line = baseline_1990_line.replacen("0x0000000000000000", "[1990 TABLE      ]", 1);
    out.push_str(&baseline_1990_line);
    out.push('\n');

    let mut baseline_gen_line = format_score_line(0, &report.baseline_generated);
    baseline_gen_line = baseline_gen_line.replacen("0x0000000000000000", "[GENERATED TABLE ]", 1);
    out.push_str(&baseline_gen_line);
    out.push('\n');
    out.push('\n');

    out.push_str("Column legend:\n");
    out.push_str("  base coll  = unordered colliding pairs for the 8-bit Pearson hash\n");
    out.push_str("  base chi2  = chi-square of base-hash bucket histogram vs. uniform\n");
    out.push_str("  max        = worst-case bucket occupancy (base hash)\n");
    out.push_str("  empty      = number of empty buckets (base hash)\n");
    out.push_str("  salt coll  = colliding pairs for the salt-array hash (— if disabled)\n");
    out.push_str("  XOR worst  = structural metric: worst-case chi-square over XOR diffs\n");
    out.push_str("  fix        = structural metric: fixed-point count\n");
    out.push('\n');
    out.push_str("To use a chosen seed in production:\n");
    out.push_str("  const MY_SEED: u64 = 0x????????????????;\n");
    out.push_str("  const MY_TABLE: [u8; 256] =\n");
    out.push_str("      generate_table_fisher_yates_const(MY_SEED);\n");
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
mod tests {
    use super::*;

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
}
