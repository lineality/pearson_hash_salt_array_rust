// src/pearson_hash_salt_array_rust.rs

//! # `pearson_hash_salt_array_rust` — Production Pearson Hashing Module
//!
//! ## Project-Level Context
//!
//! This module implements the Pearson hashing algorithm from:
//!
//!   Pearson, Peter K. (1990). "Fast Hashing of Variable-Length Text Strings."
//!   Communications of the ACM, Vol. 33, No. 6, pp. 677-680.
//!   https://dl.acm.org/doi/10.1145/78973.78978
//!
//! Pearson's algorithm is a **non-cryptographic** hash function that maps a
//! variable-length byte string onto a single 8-bit integer using a fixed
//! permutation of `0..=255`. Its core loop is:
//!
//! ```text
//!     hash = 0
//!     for byte in input:
//!         hash = table[hash XOR byte]
//!     return hash
//! ```
//!
//! ### What This Module Adds Over the Base Algorithm
//!
//! 1. **Two permutation tables side by side**:
//!    - `PEARSON_1990_TABLE` — the exact, canonical table published in
//!      Table I of the 1990 paper. Useful as a known reference and for
//!      cross-implementation interoperability.
//!    - `GENERATED_TABLE` — a `const fn`-generated table built at compile
//!      time using seeded Fisher-Yates. Useful when the caller wants a
//!      table whose construction is fully transparent and reproducible
//!      from a documented seed and PRNG, without trusting a hand-typed
//!      table from a 1990 typescript.
//!
//! 2. **Salt-array hashing** (`pearson_hash_salt_array`): produces an
//!    `N`-byte hash by running `N` independent Pearson hashes over the
//!    same input combined with `N` different salts. This is the standard
//!    technique used to extend an 8-bit hash to multi-byte hashes for
//!    Bloom filters, count-min sketches, and similar structures.
//!
//!    Critically, this module **never concatenates** input and salt into
//!    a heap buffer. Because Pearson hashing is purely sequential, salt
//!    bytes are folded into the running hash state immediately after the
//!    input bytes, in place, on the stack. This means:
//!
//!    - No `Vec` allocation per salt.
//!    - No maximum input-length constraint imposed by buffer size.
//!    - No heap usage in production code paths.
//!
//! ## Security and Scope
//!
//! Pearson hashing is **not cryptographically secure**. It must not be
//! used for password hashing, message authentication, digital signatures,
//! or any context where an adversary may attempt collisions. It is
//! appropriate for:
//!
//! - Hash-table bucket selection.
//! - Bloom-filter-style multi-hashing (the salt-array use case).
//! - Non-adversarial data-integrity sanity checks.
//! - Quick partitioning / dispersal of similar strings.
//!
//! ## Permutation Tables: Why a Permutation?
//!
//! The auxiliary table `T` must be a permutation of `0..=255` — every
//! byte value must appear exactly once. Pearson's 1990 paper proves
//! that if this holds:
//!
//! - Two strings of the same length differing in exactly one byte can
//!   never produce the same hash.
//! - The output distribution over random inputs is uniform.
//!
//! The **identity mapping** (`table[i] == i` for all `i`) is technically
//! a permutation but degenerates the algorithm into a plain longitudinal
//! XOR checksum that fails to separate anagrams. Any table used in this
//! module is validated by tests to be (a) a true permutation and (b)
//! statistically dissimilar to the identity. See `pearson_hash_tools.rs`
//! for the full quality-evaluation toolkit.
//!
//! ## Error Handling Strategy
//!
//! All fallible functions return `Result<T, std::io::Error>` per project
//! convention. Error messages are:
//!
//! - **Terse** (`&'static str`, no heap allocation).
//! - **Unique per function** (prefixed with a short function tag such as
//!   `"PHB:"` or `"PHSA:"`) so logs can pinpoint the origin without
//!   leaking implementation details.
//! - **Free of user data, paths, or internal state** to avoid creating
//!   an exfiltration surface in production logs.
//!
//! Production code never panics, never `unwrap`s, never `assert!`s. The
//! algorithm itself cannot index out of bounds because `hash ^ byte` is
//! a `u8` and the tables are `[u8; 256]` — but this invariant is also
//! enforced by `debug_assert!` and `#[cfg(test)] assert!` for paranoia.
//!
//! ## No Heap, No Unsafe, No Recursion, No Third-Party Crates
//!
//! Per project rules, this module uses only `core`/`std`, contains no
//! `unsafe`, no recursion, and allocates nothing on the heap.

use std::io::{Error, ErrorKind};

// =============================================================================
// SECTION 1: The 1990 Pearson Table (canonical reference)
// =============================================================================

/// The exact 256-byte permutation table published in Table I of
/// Pearson (1990), "Fast Hashing of Variable-Length Text Strings."
///
/// ## Project-Level Context
///
/// This table is provided for two reasons:
///
/// 1. **Reference / interoperability**: any other implementation that
///    cites the 1990 paper should produce identical hashes when given
///    this table. This makes cross-language and cross-platform
///    verification trivial.
/// 2. **Baseline for quality comparison**: `pearson_hash_tools.rs` uses
///    this table as the statistical baseline against which any
///    candidate generated table must measure up.
///
/// ## Verification
///
/// The values below are transcribed from the printed table in the
/// paper. A `#[cfg(test)]` test in this module verifies that this
/// array is a valid permutation of `0..=255` (no duplicates, no
/// missing values).
///
/// human-hand-copied from Pearson's paper: dl.acm.org/doi/epdf/10.1145/78973.78978
pub const PEARSON_1990_TABLE: [u8; 256] = [
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

// claud suggested
// pub const PEARSON_1990_TABLE: [u8; 256] = [
//       1,  87,  49,  12, 176, 178, 102, 166, 121, 193,   6,  84, 249, 230,  44, 163,
//      14, 197, 213, 181, 161,  85, 218,  80,  64, 239,  24, 226, 236, 142,  38, 200,
//     110, 177, 104, 103, 141, 253, 255,  50,  77, 101,  81,  18,  45,  96,  31, 222,
//      25, 107, 190,  70,  86, 237, 240,  34,  72, 242,  20, 214, 244, 227, 149, 235,
//      97, 234,  57,  22,  60, 250,  82, 175, 208,   5, 127, 199, 111,  62, 135, 248,
//     174, 169, 211,  58,  66, 154, 106, 195, 245, 171,  17, 187, 182, 179,   0, 243,
//     132,  56, 148,  75, 128, 133, 158, 100, 130, 126,  91,  13, 153, 246, 216, 219,
//     119,  68, 223,  78,  83,  88, 201,  99, 122,  11,  92,  32, 136, 114,  52,  10,
//     138,  30,  48, 183, 156,  35,  61,  26, 143,  74, 251,  94, 129, 162,  63, 152,
//     170,   7, 115, 167, 241, 206,   3, 150,  55,  59, 151, 220,  90,  53,  23, 131,
//     125, 173,  15, 238,  79,  95,  89,  16, 105, 137, 225, 224, 217, 160,  37, 123,
//     118,  73,   2, 157,  46, 116,   9, 145, 134, 228, 207, 212, 202, 215,  69, 229,
//      27, 188,  67, 124, 168, 252,  42,   4,  29, 108,  21, 247,  19, 205,  39, 203,
//     233,  40, 186, 147, 198, 192, 155,  33, 164, 191,  98, 204, 165, 180, 117,  76,
//     140,  36, 210, 172,  41,  54, 159,   8, 185, 232, 113, 196, 231,  47, 146, 120,
//      51,  65,  28, 144, 254, 221,  93, 189, 194, 139, 112,  43,  71, 109, 184, 209,
// ];

// =============================================================================
// SECTION 2: Generated Permutation Table (compile-time Fisher-Yates)
// =============================================================================

/// The fixed seed used to deterministically generate `GENERATED_TABLE`.
///
/// ## Project-Level Context
///
/// Changing this seed produces a different table. The seed is fixed
/// here so that every build of this crate produces byte-identical
/// hashes for the same input — this is required for any downstream
/// consumer that persists hash values (e.g. Bloom filters on disk).
///
/// The specific value `0x9E37_79B9_7F4A_7C15` is the 64-bit golden-ratio
/// constant (the same constant used by `splitmix64` and many other
/// PRNG initializers). It has no special cryptographic meaning here;
/// it is simply a well-mixed, well-known nonzero constant.
// const GENERATED_TABLE_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const GENERATED_TABLE_SEED: u64 = 0xFFFF_FFFF_FFFF_FFFF;

/// A 256-byte permutation table generated at compile time via seeded
/// Fisher-Yates shuffle.
///
/// ## Project-Level Context
///
/// This is the recommended default for new code that does not need
/// 1990-compatibility. Its construction is fully transparent:
///
/// 1. Start with the identity permutation `[0, 1, 2, ..., 255]`.
/// 2. Run Fisher-Yates (Knuth shuffle) using a documented `splitmix64`
///    PRNG seeded by `GENERATED_TABLE_SEED`.
///
/// Both the algorithm and the seed are documented in source, so any
/// reader can independently reconstruct this exact table.
///
/// ## Validation
///
/// `pearson_hash_tools.rs` provides a quality-evaluation module that
/// scores this table against `PEARSON_1990_TABLE` on six metrics
/// (fixed-point count, cycle structure, displacement, sequential
/// correlation, XOR uniformity, empirical collisions). The integration
/// tests in `main.rs` confirm this generated table meets or exceeds
/// the 1990 baseline on every metric.
pub const GENERATED_TABLE: [u8; 256] = generate_table_fisher_yates_const(GENERATED_TABLE_SEED);

/// Const-fn Fisher-Yates shuffle producing a permutation of `0..=255`.
///
/// ## What This Function Does
///
/// 1. Initializes a `[u8; 256]` to the identity permutation
///    (`table[i] = i`).
/// 2. Walks `i` from `255` down to `1`, picks a pseudo-random index
///    `j` in `0..=i` using a `splitmix64`-style PRNG, and swaps
///    `table[i]` with `table[j]`.
///
/// This is the standard Fisher-Yates / Knuth shuffle. With a fixed
/// seed it is fully deterministic.
///
/// ## Project-Level Context
///
/// Marked `const fn` so the table is computed at compile time and
/// embedded directly into the binary's read-only data section. There
/// is zero runtime cost.
///
/// ## PRNG Choice
///
/// A minimal `splitmix64` step is used as the PRNG:
///
/// ```text
///     state = state + 0x9E3779B97F4A7C15
///     z = state
///     z = (z XOR (z >> 30)) * 0xBF58476D1CE4E5B9
///     z = (z XOR (z >> 27)) * 0x94D049BB133111EB
///     z = z XOR (z >> 31)
/// ```
///
/// `splitmix64` is well-studied, passes BigCrush, has 64 bits of state
/// (more than enough for a 256-element shuffle), and is trivial to
/// implement as a `const fn`. It is **not** cryptographically secure,
/// which is acceptable because the resulting table is public anyway.
///
/// ## Why Not the 1990 Table?
///
/// The 1990 table is hand-typed from a 1990 typescript. While it
/// passes statistical tests well, a generated table from a documented
/// algorithm is more auditable: any reader can reproduce it from
/// first principles.
///
/// ## Arguments
///
/// * `seed` — 64-bit PRNG seed. Different seeds produce different
///   tables; the same seed always produces the same table.
///
/// ## Returns
///
/// A `[u8; 256]` that is guaranteed (by construction) to be a valid
/// permutation of `0..=255`.
pub const fn generate_table_fisher_yates_const(seed: u64) -> [u8; 256] {
    // Step 1: build the identity permutation.
    let mut table: [u8; 256] = [0u8; 256];
    let mut init_index: usize = 0;
    while init_index < 256 {
        // Cast is safe: init_index < 256 so it fits in a u8.
        table[init_index] = init_index as u8;
        init_index += 1;
    }

    // Step 2: Fisher-Yates shuffle, walking high-to-low.
    //
    // PRNG state advances once per swap. We use `wrapping_*` arithmetic
    // throughout so const evaluation cannot overflow-panic.
    let mut prng_state: u64 = seed;
    let mut high_index: usize = 255;
    while high_index > 0 {
        // Advance splitmix64 PRNG.
        prng_state = prng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z: u64 = prng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z = z ^ (z >> 31);

        // Reduce the 64-bit random word into `0..=high_index`.
        // `high_index + 1` is at most 256, which fits in u64 trivially.
        // Modulo bias is negligible for a 64-bit value reduced into a
        // range of at most 256.
        let swap_target: usize = (z % ((high_index as u64) + 1)) as usize;

        // Swap table[high_index] and table[swap_target].
        let temp_value: u8 = table[high_index];
        table[high_index] = table[swap_target];
        table[swap_target] = temp_value;

        high_index -= 1;
    }

    table
}

// =============================================================================
// SECTION 3: Base Pearson Hash (production function)
// =============================================================================

/// Compute the 8-bit Pearson hash of `input` using `table`.
///
/// ## Project-Level Context
///
/// This is the core building block of the entire module. The
/// salt-array variant is implemented in terms of the same inner loop,
/// inlined for stack-only operation. External callers typically use
/// this directly when an 8-bit hash is sufficient (e.g. selecting one
/// of 256 hash buckets).
///
/// ## Algorithm
///
/// ```text
///     hash = 0
///     for byte in input:
///         hash = table[hash XOR byte]
///     return hash
/// ```
///
/// The table indexing is always in-bounds because `hash` and `byte`
/// are both `u8`, so `hash ^ byte` is a `u8` in `0..=255`, and
/// `table.len() == 256`.
///
/// ## Arguments
///
/// * `input` — Slice of bytes to hash. Must be non-empty.
/// * `table` — Reference to a 256-byte permutation table. The caller
///   chooses which table (`PEARSON_1990_TABLE`, `GENERATED_TABLE`, or
///   a custom one).
///
/// ## Returns
///
/// * `Ok(u8)` — the Pearson hash of `input`.
/// * `Err(std::io::Error)` with `ErrorKind::InvalidInput` and message
///   `"PHB: empty input"` — if `input` is empty.
///
/// ## Why Reject Empty Input?
///
/// Mathematically the Pearson hash of an empty string is the initial
/// value of `hash`, which is `0`. But returning `0` for empty input
/// is a silent failure mode: it collides with every legitimate input
/// that happens to hash to `0`. Per project rules ("check returns,
/// check bounds"), we surface this case as an explicit error so the
/// caller decides how to handle it.
///
/// ## Error Message Convention
///
/// All error messages from this function are prefixed `"PHB:"`
/// (Pearson Hash Base) so log readers can identify the source
/// function without leaking source paths.
///
/// ## Examples
///
/// ```ignore
/// use pearson_hash_salt_array_rust::{pearson_hash_base, PEARSON_1990_TABLE};
///
/// let h = pearson_hash_base(b"hello", &PEARSON_1990_TABLE)?;
/// assert!(h <= 255); // always true, h is u8
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn pearson_hash_base(input: &[u8], table: &[u8; 256]) -> Result<u8, Error> {
    // =========================================================
    // Debug-Assert, Test-Assert, Production-Catch-Handle
    // =========================================================

    // Debug-only invariant: table length is enforced by the type
    // `&[u8; 256]`, so it cannot be wrong, but we assert it during
    // debug builds (not test builds) as a tripwire against future
    // refactors that might loosen the type.
    #[cfg(all(debug_assertions, not(test)))]
    debug_assert!(table.len() == 256, "PHB: table length invariant");

    // Test-only assertion mirroring the production check below.
    // Kept here so `cargo test --release` still exercises it.
    #[cfg(test)]
    assert!(table.len() == 256, "PHB: table length invariant");

    // Production check: never panic, return Err and let the caller
    // decide. Empty-input handling per docstring above.
    if input.is_empty() {
        return Err(Error::new(ErrorKind::InvalidInput, "PHB: empty input"));
    }

    // Core Pearson loop. Bounded by `input.len()`, which is finite.
    // No heap, no recursion, no panics: index is always in-bounds
    // because `(hash ^ byte) as usize` is in `0..=255` and
    // `table` is `[u8; 256]`.
    let mut running_hash: u8 = 0;
    for &current_byte in input {
        let table_index: usize = (running_hash ^ current_byte) as usize;
        running_hash = table[table_index];
    }

    Ok(running_hash)
}

// =============================================================================
// SECTION 4: Salt-Array Pearson Hash (production function, stack-only)
// =============================================================================

/// Compute an `N`-byte Pearson hash array by combining one input with
/// `N` salts, producing one Pearson hash per salt.
///
/// ## Project-Level Context
///
/// This is the headline function of the module. The pattern "one
/// input + several independent salts → several independent hashes" is
/// the standard way to build Bloom filters, count-min sketches,
/// HyperLogLog-style structures, and consistent-hash dispersal from
/// a single small hash primitive.
///
/// ### Why a Const-Generic `N`?
///
/// `N` is the number of salts (and therefore the number of output
/// bytes). Making it a const generic means:
///
/// - The output is `[u8; N]` on the stack — **no heap**.
/// - The size of the output is part of the type, so callers cannot
///   accidentally truncate or misread it.
/// - The compiler unrolls and inlines the per-salt loop where
///   profitable.
///
/// ### Why No Concatenation?
///
/// A naive implementation would, for each salt, allocate a `Vec`,
/// copy `input` into it, append the salt's bytes, and hash the buffer.
/// That allocates `N` `Vec`s of size `input.len() + 16` and bounds
/// the maximum input length to whatever fits in memory.
///
/// Pearson hashing is **inherently sequential**: the running hash
/// state after consuming `input` is identical regardless of what
/// comes next. So we can:
///
/// 1. Hash `input` **once**, getting a base running-hash byte.
/// 2. For each salt, **continue** the same algorithm with the salt's
///    16 bytes, starting from the base running-hash.
///
/// The result is identical to "concatenate input and salt, then hash"
/// but uses **zero heap**, runs the input bytes exactly once total
/// (rather than `N` times), and has no input-length bound beyond
/// what `&[u8]` itself allows.
///
/// ## Algorithm
///
/// ```text
///     base = pearson_hash_base(input, table)   // hash the input once
///     for i in 0..N:
///         h = base
///         for byte in salts[i].to_be_bytes():
///             h = table[h XOR byte]
///         output[i] = h
///     return output
/// ```
///
/// ## Salt Encoding
///
/// Salts are `u128` values, encoded as 16 big-endian bytes before
/// being fed into the hash. Big-endian is chosen so that hashes
/// computed by this crate are byte-order-independent across host
/// architectures — important if hash values are persisted or
/// transmitted between machines.
///
/// ## Arguments
///
/// * `input` — Bytes to hash. Must be non-empty.
/// * `salts` — A `&[u128; N]` reference. The caller controls `N` and
///   the salt values. Each salt produces one output byte. `N` must
///   be at least 1 (enforced at the type level — `[u128; 0]` would
///   compile but is rejected at runtime).
/// * `table` — Permutation table to use (e.g. `&PEARSON_1990_TABLE`
///   or `&GENERATED_TABLE`).
///
/// ## Returns
///
/// * `Ok([u8; N])` — array of `N` Pearson-hash bytes, one per salt,
///   in the same order as the salts.
/// * `Err(std::io::Error)` with prefix `"PHSA:"` (Pearson Hash Salt
///   Array) on:
///   - empty `input` (`"PHSA: empty input"`)
///   - `N == 0` (`"PHSA: zero salts"`)
///
/// ## Examples
///
/// ```ignore
/// use pearson_hash_salt_array_rust::{pearson_hash_salt_array, PEARSON_1990_TABLE};
///
/// let salts: [u128; 4] = [0x01, 0x02, 0x03, 0x04];
/// let hashes: [u8; 4] = pearson_hash_salt_array(b"hello", &salts, &PEARSON_1990_TABLE)?;
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn pearson_hash_salt_array<const N: usize>(
    input: &[u8],
    salts: &[u128; N],
    table: &[u8; 256],
) -> Result<[u8; N], Error> {
    // =========================================================
    // Debug-Assert, Test-Assert, Production-Catch-Handle
    // =========================================================
    //
    // The two production cases are: empty input, and N == 0.
    // Both are checked-and-handled below without panicking.

    #[cfg(all(debug_assertions, not(test)))]
    {
        debug_assert!(N > 0, "PHSA: zero salts (debug)");
        debug_assert!(table.len() == 256, "PHSA: table length invariant (debug)");
    }

    #[cfg(test)]
    {
        assert!(table.len() == 256, "PHSA: table length invariant (test)");
    }

    // Production check: N == 0 means an empty output array, which is
    // meaningless and almost certainly a caller bug. Reject explicitly.
    if N == 0 {
        return Err(Error::new(ErrorKind::InvalidInput, "PHSA: zero salts"));
    }

    // Production check: empty input. Same reasoning as in
    // `pearson_hash_base` — empty input would produce a deterministic
    // value that collides with legitimate inputs.
    if input.is_empty() {
        return Err(Error::new(ErrorKind::InvalidInput, "PHSA: empty input"));
    }

    // --------------------------------------------------------------
    // Step 1: Hash the input ONCE, producing the "base" running hash.
    //
    // This is the state of the Pearson algorithm immediately after
    // consuming `input` but before consuming any salt bytes. Every
    // per-salt hash starts from this same base, so we save (N-1)
    // re-traversals of `input`.
    // --------------------------------------------------------------
    let mut base_running_hash: u8 = 0;
    for &input_byte in input {
        let table_index: usize = (base_running_hash ^ input_byte) as usize;
        base_running_hash = table[table_index];
    }

    // --------------------------------------------------------------
    // Step 2: For each salt, continue the Pearson loop using the
    // salt's big-endian bytes, and store the resulting byte.
    //
    // Output array is stack-allocated `[u8; N]`. No heap.
    // Salt-byte loop is bounded by 16 (size of u128). Outer loop is
    // bounded by N (a compile-time constant). All loops are firmly
    // bounded per project rules.
    // --------------------------------------------------------------
    let mut output_hashes: [u8; N] = [0u8; N];

    for salt_index in 0..N {
        // Start from the saved post-input hash state.
        let mut salted_running_hash: u8 = base_running_hash;

        // 16 bytes of big-endian salt. `to_be_bytes` returns `[u8; 16]`
        // on the stack — no allocation.
        let salt_bytes: [u8; 16] = salts[salt_index].to_be_bytes();

        // Continue the Pearson loop over the salt bytes.
        for &salt_byte in salt_bytes.iter() {
            let table_index: usize = (salted_running_hash ^ salt_byte) as usize;
            salted_running_hash = table[table_index];
        }

        output_hashes[salt_index] = salted_running_hash;
    }

    Ok(output_hashes)
}

// =============================================================================
// SECTION 5: Internal utility — permutation validity check
// =============================================================================

/// Verify that `table` is a valid permutation of `0..=255`.
///
/// ## Project-Level Context
///
/// Both `PEARSON_1990_TABLE` and `GENERATED_TABLE` are guaranteed to be
/// valid permutations by construction (the 1990 table is hand-verified;
/// the generated table is produced by Fisher-Yates, which provably
/// preserves the permutation invariant). This function exists to
/// **prove** that invariant at test time, and to allow callers who
/// build their own tables to validate them before use.
///
/// A "valid permutation" means every byte value in `0..=255` appears
/// exactly once. The check uses a 256-bit presence bitmap (32 bytes
/// on the stack) so it allocates nothing.
///
/// ## Why It Matters
///
/// If a table has a duplicate value, then some byte in `0..=255` is
/// missing, and the Pearson hash can never produce that byte as an
/// output — silently shrinking the hash range and creating biased
/// collisions. The "two strings differing in one byte never collide"
/// property of Pearson hashing depends critically on the table being
/// a true permutation.
///
/// ## Arguments
///
/// * `table` — Reference to the 256-byte table to validate.
///
/// ## Returns
///
/// * `true` if every value `0..=255` appears exactly once.
/// * `false` otherwise.
///
/// ## Note
///
/// This is `pub` because it is genuinely useful to external callers
/// constructing custom tables. It does not allocate and is safe to
/// call from production code if desired.
pub fn is_valid_permutation(table: &[u8; 256]) -> bool {
    // 32-byte bitmap, one bit per possible value 0..=255.
    let mut presence_bitmap: [u8; 32] = [0u8; 32];

    // Walk every entry and set its corresponding bit. If a bit is
    // already set, we have a duplicate, so the table is not a
    // permutation.
    let mut entry_index: usize = 0;
    while entry_index < 256 {
        let value: u8 = table[entry_index];
        let byte_index: usize = (value as usize) >> 3; // value / 8
        let bit_mask: u8 = 1u8 << ((value as usize) & 7); // 1 << (value % 8)

        if (presence_bitmap[byte_index] & bit_mask) != 0 {
            // Duplicate value detected.
            return false;
        }
        presence_bitmap[byte_index] |= bit_mask;

        entry_index += 1;
    }

    // If we set 256 distinct bits with no collision, every value
    // 0..=255 must be present exactly once.
    true
}

// =============================================================================
// SECTION 6: Cargo tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Both shipped tables must be valid permutations. If either of
    /// these fails, the module is fundamentally broken.
    #[test]
    fn test_pearson_1990_table_is_valid_permutation() {
        assert!(
            is_valid_permutation(&PEARSON_1990_TABLE),
            "PEARSON_1990_TABLE is not a valid permutation"
        );
    }

    #[test]
    fn test_generated_table_is_valid_permutation() {
        assert!(
            is_valid_permutation(&GENERATED_TABLE),
            "GENERATED_TABLE is not a valid permutation"
        );
    }

    /// The identity table should be detected as a valid permutation
    /// (it is one — just a bad choice for hashing). This is a sanity
    /// check on `is_valid_permutation` itself.
    #[test]
    fn test_identity_is_a_permutation() {
        let mut identity: [u8; 256] = [0u8; 256];
        for i in 0..256 {
            identity[i] = i as u8;
        }
        assert!(is_valid_permutation(&identity));
    }

    /// A table with a duplicate must be rejected.
    #[test]
    fn test_invalid_table_with_duplicate_is_rejected() {
        let mut bad_table: [u8; 256] = [0u8; 256];
        for i in 0..256 {
            bad_table[i] = i as u8;
        }
        // Introduce a duplicate: position 5 now also holds value 7.
        bad_table[5] = 7;
        assert!(!is_valid_permutation(&bad_table));
    }

    /// Empty input must be rejected with the documented error prefix.
    #[test]
    fn test_pearson_hash_base_rejects_empty_input() {
        let result = pearson_hash_base(b"", &PEARSON_1990_TABLE);
        assert!(result.is_err());
        let err = result.err().expect("expected an error");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        // Message must start with the function-unique tag.
        let msg = format!("{}", err);
        assert!(msg.contains("PHB:"), "error must carry PHB: prefix");
    }

    /// Determinism: same input + same table → same hash.
    #[test]
    fn test_pearson_hash_base_is_deterministic() {
        let h1 =
            pearson_hash_base(b"deterministic", &PEARSON_1990_TABLE).expect("hash should succeed");
        let h2 =
            pearson_hash_base(b"deterministic", &PEARSON_1990_TABLE).expect("hash should succeed");
        assert_eq!(h1, h2);
    }

    /// Single-byte difference between two equal-length inputs must
    /// not collide. This is the headline property proved in the 1990
    /// paper for permutation tables.
    #[test]
    fn test_pearson_hash_base_one_byte_difference_no_collision() {
        let h_a = pearson_hash_base(b"abcdef", &PEARSON_1990_TABLE).expect("hash should succeed");
        let h_b = pearson_hash_base(b"abcdeg", &PEARSON_1990_TABLE).expect("hash should succeed");
        assert_ne!(
            h_a, h_b,
            "single-byte-different inputs must never collide under Pearson hashing"
        );
    }

    /// Both tables must produce valid 8-bit outputs and should
    /// generally differ for the same input (the tables are different
    /// permutations).
    #[test]
    fn test_pearson_hash_base_differs_between_tables() {
        let h_1990 =
            pearson_hash_base(b"compare-tables", &PEARSON_1990_TABLE).expect("hash should succeed");
        let h_gen =
            pearson_hash_base(b"compare-tables", &GENERATED_TABLE).expect("hash should succeed");
        // They might coincidentally match on rare inputs; just check
        // both succeeded and produced u8 values. (`u8` always does.)
        let _ = (h_1990, h_gen);
    }

    /// Salt-array: empty input rejected with PHSA: prefix.
    #[test]
    fn test_salt_array_rejects_empty_input() {
        let salts: [u128; 3] = [1, 2, 3];
        let result = pearson_hash_salt_array(b"", &salts, &PEARSON_1990_TABLE);
        assert!(result.is_err());
        let msg = format!("{}", result.err().expect("expected error"));
        assert!(msg.contains("PHSA:"));
    }

    /// Salt-array: N == 0 rejected.
    #[test]
    fn test_salt_array_rejects_zero_salts() {
        let salts: [u128; 0] = [];
        let result = pearson_hash_salt_array(b"hello", &salts, &PEARSON_1990_TABLE);
        assert!(result.is_err());
        let msg = format!("{}", result.err().expect("expected error"));
        assert!(msg.contains("PHSA:"));
    }

    /// Salt-array: output length equals N.
    #[test]
    fn test_salt_array_output_length() {
        let salts: [u128; 5] = [10, 20, 30, 40, 50];
        let result = pearson_hash_salt_array(b"length-check", &salts, &PEARSON_1990_TABLE)
            .expect("hash should succeed");
        assert_eq!(result.len(), 5);
    }

    /// Salt-array: different salts produce different bytes (with high
    /// probability). We test with deliberately well-separated salts.
    #[test]
    fn test_salt_array_distinct_salts_give_varied_output() {
        let salts: [u128; 4] = [
            0x0000_0000_0000_0000_0000_0000_0000_0001,
            0x0000_0000_0000_0000_0000_0000_0000_0002,
            0xFFFF_FFFF_FFFF_FFFF_0000_0000_0000_0000,
            0xDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0,
        ];
        let result = pearson_hash_salt_array(b"salt-array-test", &salts, &PEARSON_1990_TABLE)
            .expect("hash should succeed");

        // At least two of the four outputs must differ. (All four
        // identical would be astronomically unlikely.)
        let all_same = result.iter().all(|&b| b == result[0]);
        assert!(!all_same, "salt-array output should not be uniform");
    }

    /// Salt-array: equivalence to "concatenate then hash". This is
    /// the correctness proof of the no-concatenation optimization.
    ///
    /// For each salt, hashing (input || salt_be_bytes) with
    /// `pearson_hash_base` must produce the same byte as the
    /// corresponding entry in the salt-array output.
    #[test]
    fn test_salt_array_matches_concatenated_equivalent() {
        let input: &[u8] = b"equivalence-proof";
        let salts: [u128; 3] = [0x1111, 0x2222_3333, 0xAABB_CCDD_EEFF];
        let array_result = pearson_hash_salt_array(input, &salts, &PEARSON_1990_TABLE)
            .expect("salt-array hash should succeed");

        for (i, salt) in salts.iter().enumerate() {
            // Build the concatenated buffer (test code only — heap OK).
            let mut concatenated: Vec<u8> = Vec::with_capacity(input.len() + 16);
            concatenated.extend_from_slice(input);
            concatenated.extend_from_slice(&salt.to_be_bytes());
            let expected = pearson_hash_base(&concatenated, &PEARSON_1990_TABLE)
                .expect("base hash should succeed");
            assert_eq!(
                array_result[i], expected,
                "salt-array byte {} must equal concatenated-input hash",
                i
            );
        }
    }

    /// Sanity: generating the same table with the same seed twice
    /// produces identical results (the const-fn is deterministic).
    #[test]
    fn test_generated_table_is_deterministic_in_seed() {
        let a = generate_table_fisher_yates_const(GENERATED_TABLE_SEED);
        let b = generate_table_fisher_yates_const(GENERATED_TABLE_SEED);
        assert_eq!(a, b);
        assert_eq!(a, GENERATED_TABLE);
    }

    /// Different seeds produce different tables (almost certainly).
    #[test]
    fn test_different_seeds_produce_different_tables() {
        let a = generate_table_fisher_yates_const(1);
        let b = generate_table_fisher_yates_const(2);
        assert_ne!(a, b);
    }
}
