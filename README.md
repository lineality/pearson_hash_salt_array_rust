# pearson_hash_salt_array_rust

A Rust implementation of the Pearson hashing algorithm (Pearson 1990),
extended with salt-array hashing for
multi-hash use cases such as Bloom filters and count-min sketches.

Note: a hash-array, or hash-list, can have advantages over a single-blob hash
such as:
- incremental (and parallelizable) creation
- incremental (and parallelizable) checking/validation
- granular scale: 3 byte or 5 byte are no issue
- potentially larger effective scale: the size of a single-hash from 128, to 256, to 512, to 1024, to 2048, etc. becomes problematic for real-world memory. But an array can easily be made and used at affectively longer lengths. E.g. How many pearson-hashes could you generate in the amount of time needed to make one sha256 hash? Quite a few.

#### Note:
- For uses where the salt does not need to be very unique, a u8 hash will work as well as u128 to make a different salted result

## Contents

- [What Pearson hashing is](#what-pearson-hashing-is)
- [What this crate adds](#what-this-crate-adds)
- [Security scope](#security-scope)
- [Files](#files)
- [Quick start](#quick-start)
- [Permutation tables](#permutation-tables)
  - [The 1990 table](#the-1990-table)
  - [The generated table](#the-generated-table)
  - [Which table should I use?](#which-table-should-i-use)
  - [Why a permutation?](#why-a-permutation)
  - [The identity-table pitfall](#the-identity-table-pitfall)
- [Salt-array hashing](#salt-array-hashing)
  - [Why no concatenation?](#why-no-concatenation)
  - [Salt encoding](#salt-encoding)
- [Table quality measurement](#table-quality-measurement)
  - [The six metrics](#the-six-metrics)
  - [Measured results for both tables](#measured-results-for-both-tables)
  - [Shopping for a seed](#shopping-for-a-seed)
- [API reference](#api-reference)
- [Running the demo](#running-the-demo)
- [Running the tests](#running-the-tests)
- [Error handling](#error-handling)
- [Design rules](#design-rules)
- [References](#references)

---

## Metric interpretation cheat-sheet

```
METRIC                    BETTER    WORSE     NOTES
─────────────────────────────────────────────────────────────────────────────
Fixed-point count         Lower     Higher    Identity = 256 (worst).
                                              Random permutation expects ~1.
                                              0 is fine. >10 is suspicious.

One-cycle count           Lower     Higher    Same thing as fixed points.
                                              Listed separately in the cycle
                                              structure view.

Two-cycle count           Lower     Higher    Transpositions. A few are fine.
                                              Many indicate poor shuffling.

Longest cycle             Longer    Shorter   A single very long cycle is not
                                              a problem. Very many short
                                              cycles is the problem to avoid.
                                              Random 256-permutation expects
                                              longest cycle ~159 (Golomb-
                                              Dickman constant × 256 ≈ 0.62).

Total cycle count         Lower     Higher    Fewer, longer cycles is better
                                              than many short ones. Identity
                                              = 256 cycles (all length 1).

Displacement min          N/A       Zero      A single zero means one fixed
                                              point. Min = 0 is expected and
                                              fine for ~1 fixed point.

Displacement max          Higher    Lower     Higher max indicates at least
                                              some values are scattered far.
                                              Not the primary metric but low
                                              max is a weak warning sign.

Displacement mean         Higher    Lower     Identity = 0.0 (worst).
                                              Reverse table = 128.0.
                                              Random permutation expects ~85
                                              (converges to n/3 for large n).
                                              Below ~70 is suspicious.

Displacement zero count   Lower     Higher    Same as fixed-point count.
                                              Redundant column; ~1 expected.

Seq correlation |r|       Closer    Farther   Target: near 0.0.
                          to 0.0    from 0.0  Identity = 1.0 (worst).
                                              Below 0.05 is good.
                                              Above 0.15 is a warning sign.

XOR worst chi-square      Lower     Higher    *** PRIMARY METRIC ***
                                              Most directly governs Pearson
                                              hash avalanche behavior.
                                              Identity = 65,280 (worst).
                                              1990 table and default seed
                                              both = 640.
                                              Seeds 0xDEADBEEFCAFEBABE and
                                              0xA5A5A5A55A5A5A5A achieve
                                              632 and 624 respectively
                                              (from the 8-seed demo sweep).
                                              No 8-bit permutation can
                                              reach 0 perfectly.

XOR mean chi-square       Lower     Higher    Average behavior across all
                                              255 nonzero XOR differences.
                                              Complements the worst-case.
                                              1990 table ≈ 507.
                                              Less decisive than worst-case
                                              but useful for comparison.

Empirical collisions      Lower     Higher    HIGH VARIANCE — treat as a
                                              rough indicator only.
                                              Differences of ±50–100 on
                                              the fixed demo corpus are
                                              birthday-paradox noise, not
                                              meaningful quality differences.
                                              For meaningful results,
                                              replace the fixed corpus with
                                              your own representative data.
─────────────────────────────────────────────────────────────────────────────
```

---

### One-sentence summary per metric

| Metric | One sentence |
|---|---|
| Fixed-point count | How many values map to themselves; lower is better, ~1 is normal. |
| One-cycle count | Same as fixed-point count; listed separately in the cycle view. |
| Two-cycle count | How many values swap with one other value; lower is better. |
| Longest cycle | Should be long; very short longest cycle means poor shuffling. |
| Total cycle count | Fewer, longer cycles beats many short cycles; identity has 256. |
| Displacement min | Informational; 0 expected when fixed-point count is ~1. |
| Displacement max | Higher indicates some values are scattered far; low max is a mild warning. |
| Displacement mean | Higher is better; identity = 0, reverse = 128, random ≈ 85. |
| Displacement zero count | Same as fixed-point count; ~1 expected. |
| Seq correlation \|r\| | Closer to 0.0 is better; identity = 1.0; target < 0.05. |
| **XOR worst chi-square** | **Lower is better; primary metric; governs Pearson avalanche behavior.** |
| XOR mean chi-square | Lower is better; complements worst-case; less decisive. |
| Empirical collisions | Lower is better but high-variance; use your own corpus for real signal. |

---


## What Pearson hashing is

Pearson hashing is a non-cryptographic hash function published in
1990 by Peter K. Pearson:

> Pearson, Peter K. (1990). "Fast Hashing of Variable-Length Text
> Strings." *Communications of the ACM*, Vol. 33, No. 6, pp. 677–680.
> https://dl.acm.org/doi/10.1145/78973.78978

The algorithm maps a variable-length byte string onto a single
8-bit integer using a fixed 256-byte permutation table `T`:

```text
hash = 0
for byte in input:
    hash = T[hash XOR byte]
return hash
```

Properties proven in the paper:

- Two strings of the same length that differ in exactly one byte
  **never** produce the same hash (given a valid permutation table).
- Over random inputs, the output distribution is uniform across
  all 256 possible values.
- The length of the input does not need to be known in advance.
- Each byte requires only one XOR and one table lookup — no
  multiplication, no division.

---

## What this crate adds

1. **Two permutation tables** available side-by-side:
   - `PEARSON_1990_TABLE` — the exact canonical table from Table I
     of the 1990 paper.
   - `GENERATED_TABLE` — a compile-time Fisher-Yates shuffle with a
     documented seed and PRNG (splitmix64), fully transparent and
     reproducible from source.

2. **Salt-array hashing** (`pearson_hash_salt_array`) — produces an
   `N`-byte hash by running `N` independent Pearson hashes over the
   same input combined with `N` different `u128` salts. Stack-only,
   no heap, no concatenation, const-generic output `[u8; N]`.

3. **Table quality measurement** (`pearson_hash_tools`) — six
   statistical metrics for evaluating any `[u8; 256]` permutation,
   a comparative report printer, and a seed-sweep example so users
   can choose a permutation that suits their specific data.

---

## Security scope

Pearson hashing is **not cryptographically secure**. Do not use it
for:

- Password hashing or storage.
- Message authentication codes (MACs).
- Digital signatures.
- Any context where an adversary may choose inputs to cause
  collisions.

It is appropriate for:

- Hash-table bucket selection.
- Bloom filters and count-min sketches (the salt-array use case).
- Non-adversarial data-integrity checks.
- Fast partitioning or dispersal of similar strings on constrained
  hardware.

---

## Files

```
src/
├── main.rs                          demo binary
├── pearson_hash_salt_array_rust.rs  production hashing module
└── pearson_hash_tools.rs            table generation and measurement tools
```

The production module (`pearson_hash_salt_array_rust.rs`) is
self-contained. The tools module (`pearson_hash_tools.rs`) is
self-contained. Neither imports from the other. Common definitions
(the 1990 table, the Fisher-Yates generator) are duplicated by
design so each file can be lifted out independently.

---

## Quick start

### Single hash

```rust
use pearson_hash_salt_array_rust::{pearson_hash_base, PEARSON_1990_TABLE};

let hash: u8 = pearson_hash_base(b"hello", &PEARSON_1990_TABLE)?;
```

### Salt-array hash (N independent bytes)

```rust
use pearson_hash_salt_array_rust::{pearson_hash_salt_array, PEARSON_1990_TABLE};

let salts: [u128; 4] = [0x01, 0x02, 0x03, 0x04];
let hashes: [u8; 4] =
    pearson_hash_salt_array(b"hello", &salts, &PEARSON_1990_TABLE)?;
```

Both functions return `Result<T, std::io::Error>` and reject empty
input with `ErrorKind::InvalidInput`.

---

## Permutation tables

### The 1990 table

`PEARSON_1990_TABLE` is the exact 256-byte table from Table I of the
1990 paper, transcribed verbatim. It is verified by test to be a
valid permutation of `0..=255`.

Use this table when:

- You need byte-for-byte compatibility with other implementations
  that cite the 1990 paper.
- You want the result most directly validated by Pearson's own
  chi-square tests on a 26,662-word dictionary.

Measured quality (from `cargo run`):

```
fixed points:             1
longest cycle:            99
total cycles:             7
displacement mean:        83.477
seq correlation |r|:      0.012330
XOR worst chi-square:     640.000  (at d = 0x24)
XOR mean chi-square:      506.729
empirical collisions:     490  (on the fixed demo corpus)
```

### The generated table

`GENERATED_TABLE` is built at **compile time** via `const fn`
Fisher-Yates shuffle using seed `0x9E3779B97F4A7C15` and a
splitmix64 PRNG. Any reader can reproduce it from the source
without trusting a hand-typed 1990 typescript.

Measured quality:

```
fixed points:             1
longest cycle:            225
total cycles:             5
displacement mean:        84.703
seq correlation |r|:      0.022330
XOR worst chi-square:     640.000  (at d = 0x84)
XOR mean chi-square:      505.412
empirical collisions:     556  (on the fixed demo corpus; see note below)
```

Note on empirical collisions: the 66-collision difference between
the two tables on the demo corpus is high-variance birthday-paradox
noise, not a meaningful quality difference. Both tables produce
identical XOR worst-case chi-square (640.0), which is the metric
most directly relevant to Pearson hash behavior. See
[Shopping for a seed](#shopping-for-a-seed) for how to find a seed
with XOR worst-case chi-square below 640.

### Which table should I use?

| Situation | Recommendation |
|---|---|
| Need compatibility with other Pearson implementations | `PEARSON_1990_TABLE` |
| Want fully auditable, seed-reproducible generation | `GENERATED_TABLE` |
| Have domain-specific data and want to optimize | Sweep seeds, pick your own (see below) |
| Any of the above | Either is fine; they are statistically comparable |

Both tables are valid permutations. Both score identically on the
most important metric (XOR uniformity worst-case chi-square). Pick
either. If you need to choose one for long-term stability, prefer
`PEARSON_1990_TABLE` — its behavior is documented by a peer-reviewed
paper and it will never change.

### Why a permutation?

The table `T` must be a permutation of `0..=255` — every byte value
appears exactly once. If this holds, Pearson's 1990 paper proves:

- The "one-byte-difference no-collision" property.
- Uniform output distribution over random inputs.

If the table has a duplicate, some byte value is unreachable as a
hash output, silently shrinking the hash range and creating biased
collisions. `is_valid_permutation` (production module) and
`is_valid_permutation_tools` (tools module) check this property in
O(256) time using a 32-byte stack bitmap — no heap.

Both shipped tables are verified by `cargo test` to be valid
permutations.

### The identity-table pitfall

The identity mapping (`T[i] = i` for all `i`) is technically a valid
permutation but the worst possible choice for Pearson hashing.
Under the identity:

```text
hash = T[hash XOR byte] = hash XOR byte
```

The algorithm degenerates to a plain longitudinal XOR checksum,
which maps all anagrams to identical hash values (XOR is
commutative). Pearson explicitly warns against this in the paper.

Any table generated by Fisher-Yates from a non-pathological seed
is astronomically unlikely to resemble the identity. The tools
module includes a test (`test_identity_table_fails_comparison_against_1990`)
that confirms the identity is correctly identified as structurally
pathological by the metrics.

---

## Salt-array hashing

The `pearson_hash_salt_array<const N: usize>` function produces an
`N`-byte hash from one input and `N` `u128` salts. This pattern is
the standard construction for:

- **Bloom filters**: `N` hash functions, one per bit position.
- **Count-min sketches**: `N` hash functions, one per row.
- **Consistent hashing**: multiple independent hash values for
  ring-placement.

The naive approach would concatenate `input + salt_bytes` into a
buffer for each salt, then hash each buffer. This crate does not do
that.

### Why no concatenation?

Pearson hashing is inherently sequential:

```text
hash = T[hash XOR byte]   for each byte in sequence
```

The running `hash` value after consuming all of `input` is a
complete intermediate state. We can continue the algorithm from
that state by feeding in the salt bytes, without ever forming a
contiguous `input || salt` buffer.

Concretely, for each salt `i`:

```text
h = base_hash   (= result of hashing input once)
for byte in salts[i].to_be_bytes():
    h = T[h XOR byte]
output[i] = h
```

This means:

- Input bytes are hashed exactly **once**, regardless of `N`.
- No `Vec` is allocated per salt.
- No maximum input-length constraint from a buffer.
- Output is a stack-allocated `[u8; N]`.
- The result is **identical** to "concatenate input and salt, then
  hash" — this is verified by `test_salt_array_matches_concatenated_equivalent`
  in the production module's tests.

### Salt encoding

Salts are `u128` values encoded as 16 big-endian bytes before being
fed into the hash. Big-endian is chosen for cross-platform
consistency: the same salt value produces the same salt-byte
sequence on any host architecture. This matters if hash values are
persisted to disk or transmitted between machines.

---

## Table quality measurement

`pearson_hash_tools` provides six metrics for evaluating any
`[u8; 256]` candidate permutation table. The design posture:

> **Metrics produce measurements. Verdicts are a policy decision
> that belongs to the caller.**

None of the metric functions declare a table "good" or "bad."
`evaluate_table` returns a `TableQualityReport` struct. The
caller — whether a human reading `cargo run` output, or a program
sweeping seeds — interprets the numbers.

### The six metrics

#### 1. Fixed-point count

A fixed point is a position `i` where `T[i] == i`. The identity
table has 256 fixed points (worst case). A random permutation of
256 elements has an expected fixed-point count of exactly 1
(derangement formula in the large-n limit).

At a fixed point, the Pearson update `hash = T[hash XOR byte]`
reduces to `hash = hash XOR byte` — a plain XOR with no mixing.
Fewer fixed points is generally better.

#### 2. Cycle structure

Every permutation decomposes uniquely into disjoint cycles. For
example, `T[3]=7, T[7]=12, T[12]=3` is a 3-cycle.

- Identity: 256 one-cycles. Worst case.
- Random permutation: mix of cycle lengths; longest cycle averages
  approximately 62% of n (Golomb-Dickman constant ~0.6243).
- Many short cycles (1-cycles and 2-cycles) create local patterns
  that reduce dispersal of similar inputs.

Reported as a sorted list of cycle lengths plus summary counts
(one-cycles, two-cycles, longest cycle, total cycle count).

#### 3. Displacement

For each index `i`, displacement is `|T[i] - i|` — how far the
permutation moves value `i` from its starting position.

- Identity: all displacements 0.
- Reverse table (`T[i] = 255 - i`): mean displacement exactly 128.0.
- Higher mean displacement generally indicates better mixing.

#### 4. Sequential correlation

The Pearson product-moment correlation coefficient (note: Karl
Pearson the statistician, unrelated to Peter Pearson the hash
author) between `T[i]` and `T[i+1]` over all 255 adjacent pairs.

- Identity: correlation +1.0.
- Reverse table: correlation -1.0.
- Well-shuffled permutation: near 0.0.

Reported as the absolute value; smaller is better.

#### 5. XOR uniformity

**This is the most directly relevant metric for Pearson hashing.**

For each nonzero XOR difference `d` in `1..=255`, the metric
examines the 256-value multiset:

```text
S_d = { T[i] XOR T[i XOR d] : i in 0..256 }
```

If `T` were ideal, `S_d` would cover `0..=255` exactly uniformly
(each output difference appearing once for each input difference —
the perfect-APN property of differential cryptanalysis). No 8-bit
permutation achieves this perfectly, but tables closer to uniform
here have better Pearson hash avalanche behavior.

Non-uniformity is measured by chi-square against the uniform
distribution (256 samples into 256 buckets, expected count 1.0
per bucket).

Identity special case: for the identity, `T[i] XOR T[i XOR d] = d`
for all `i`, so the histogram has all 256 counts in one bucket and
0 in all others. Chi-square per `d` = 65,280 (computed analytically,
verified by test).

Reported: worst-case chi-square across all 255 nonzero `d`,
the `d` value that produced it, and the mean chi-square.

#### 6. Empirical collision count

Hashes a fixed structured corpus (all 256 single-byte inputs, a
16×16 grid of 2-byte inputs, 24 common English words, and
single-bit-flipped variants of four base strings) and counts
actual collisions.

**Statistical caveat:** on a corpus of a few hundred inputs into
256 buckets, this metric is high-variance birthday-paradox noise.
Two permutations of equal structural quality will routinely differ
by tens of collisions on this corpus. Use it only as a rough
indicator. For meaningful collision testing, replace the fixed
corpus with your own representative data.

### Measured results for both tables

From `cargo run`:

```
metric                          1990 table    generated (default seed)   B - A
--------------------------------------------------------------------------------
valid permutation                     true                        true    same
fixed points                             1                           1      +0
one-cycles                               1                           1      +0
two-cycles                               0                           0      +0
longest cycle                           99                         225    +126
total cycles                             7                           5      -2
displacement min                         0                           0      +0
displacement max                       237                         242      +5
displacement mean                   83.477                      84.703   +1.227
seq correlation |r|               0.012330                    0.022375  +0.010
XOR worst chi-square               640.000                     640.000   +0.000
XOR mean chi-square                506.729                     505.412   -1.318
empirical collisions                   490                         556     +66
```

Interpretation:

- XOR worst chi-square (the key metric): **tied** at 640.0.
- XOR mean chi-square: essentially tied (difference < 0.3%).
- Empirical collisions: 66-pair difference on a small corpus —
  high-variance noise, not a meaningful quality difference.
- The two tables are statistically comparable on every metric
  that matters.

### Shopping for a seed

If you want a Fisher-Yates table whose XOR worst-case chi-square
is lower than 640 (the 1990 baseline), the seed sweep in `cargo run`
shows it is possible with a small search. From the 8-seed
illustrative sweep:

```
seed                    fix  long  mean disp  XOR worst  collisions
-------------------------------------------------------------------
[1990 baseline]           1    99     83.477     640.00         490
0x0000000000000001        0   100     83.328     656.00         427
0x0000000000000042        3   115     87.727     664.00         511
0x9E3779B97F4A7C15        1   225     84.703     640.00         556  <- default 1
0xDEADBEEFCAFEBABE        1   130     84.062     632.00         456
0x0123456789ABCDEF        0   162     84.023     680.00         543
0xFFFFFFFFFFFFFFFF        2   117     84.180     640.00         436  <- default 2
0x123456789ABCDEF0        1   104     82.617     656.00         484
0xA5A5A5A55A5A5A5A        0   175     82.945     624.00         530
```

`0xDEADBEEFCAFEBABE` achieves XOR worst 632 and
`0xA5A5A5A55A5A5A5A` achieves 624 — both below the 1990 baseline —
at the cost of different trade-offs on other metrics.

Recommended workflow for choosing a production seed:

1. Identify which metric is most important for your data (usually
   XOR worst chi-square, unless you have a specific corpus in mind).
2. Write a loop that calls `generate_table_fisher_yates(seed)` and
   `evaluate_table(&table)` for each seed in a range, collecting
   the metric you care about.
3. Select the seed with the best score on your chosen metric.
4. Hard-code that seed (and optionally the resulting table) into
   your build. Document the seed and the selection criterion.

---

## API reference

### `pearson_hash_salt_array_rust`

#### `pearson_hash_base(input: &[u8], table: &[u8; 256]) -> Result<u8, std::io::Error>`

Computes the 8-bit Pearson hash of `input` using `table`.

- Returns `Err(ErrorKind::InvalidInput)` with message prefix `"PHB:"`
  if `input` is empty.
- Caller chooses the table; both `PEARSON_1990_TABLE` and
  `GENERATED_TABLE` are valid choices.

#### `pearson_hash_salt_array<const N: usize>(input: &[u8], salts: &[u128; N], table: &[u8; 256]) -> Result<[u8; N], std::io::Error>`

Produces an `N`-byte hash using `N` salts. No heap, no
concatenation, stack-allocated output.

- Returns `Err` with prefix `"PHSA:"` if `input` is empty or `N == 0`.

#### `PEARSON_1990_TABLE: [u8; 256]`

The exact table from Pearson (1990), Table I.

#### `GENERATED_TABLE: [u8; 256]`

Compile-time Fisher-Yates table, seed `0x9E3779B97F4A7C15`.

#### `generate_table_fisher_yates_const(seed: u64) -> [u8; 256]`

`const fn`. Same table for same seed, every build.

#### `is_valid_permutation(table: &[u8; 256]) -> bool`

Heap-free permutation check. Stack bitmap, O(256).

---

### `pearson_hash_tools`

#### `evaluate_table(table: &[u8; 256]) -> TableQualityReport`

Runs all six metrics and returns a `TableQualityReport`.

#### `print_table_evaluation_report(label: &str, table: &[u8; 256])`

Prints a single labeled report to stdout.

#### `print_comparative_report(label_a, table_a, label_b, table_b)`

Prints a side-by-side delta table of two reports.

#### `generate_table_fisher_yates(seed: u64) -> [u8; 256]`

Runtime Fisher-Yates. Same algorithm as the `const fn` variant.
Useful for seed sweeps.

#### `generate_table_fisher_yates_const(seed: u64) -> [u8; 256]`

`const fn` copy, independent of the production module.

#### `TableQualityReport`

Struct holding all six metric results. Implements `Display` for
human-readable output and `Debug` for diagnostic use.

#### `TableQualityReport::compare_against(&self, baseline) -> ComparisonVerdict`

Convenience comparison with the documented policy. Returns
`AtLeastAsGood` or `Worse(Vec<String>)`. Callers who want a
different policy should read the struct fields directly.

---

## Running the demo

```bash
cargo run
```

Produces the five-section demo transcript shown in the project
output above. No arguments required.

---

## Running the tests

```bash
cargo test
```

All cargo tests in this crate verify **code correctness only**.
Every test uses inputs whose correct answers are known a priori
(identity table, reverse table, single-swap table, etc.). No test
passes or fails based on whether one random permutation happens to
score higher than another on a noisy metric.

```bash
cargo test --release
```

Runs the same tests against the release binary, exercising
`#[cfg(test)] assert!` guards in production code paths.

---

## Error handling

All fallible functions return `Result<T, std::io::Error>`.

Error messages are:

- **Terse** — `&'static str`, no heap allocation in production.
- **Unique per function** — prefixed with a short tag (`"PHB:"`,
  `"PHSA:"`, `"TOOL-PH:"`) so log readers can identify the source
  without leaking implementation details or internal state.
- **Free of user data, file paths, and internal details** — safe
  for production log output.

Production code paths never `panic!`, never `unwrap`, and never
`assert!`. All error conditions are returned as `Err(...)` for the
caller to handle. Debug and test builds additionally use
`debug_assert!` and `#[cfg(test)] assert!` respectively.

---

## Design rules

This crate follows a set of explicit production rules:

- **No heap in production hashing functions.** All production hash
  functions use stack memory only.
- **No unsafe code.** No raw pointers, no `unsafe` blocks.
- **No recursion.** All algorithms are iterative.
- **No third-party crates.** Only `core` and `std`.
- **No `unwrap` in production paths.** All fallible operations return
  `Result`.
- **No `panic!` in production paths.** Errors are returned, not
  thrown.
- **Bounded loops.** Every loop is bounded by a compile-time constant
  or a finite slice length.
- **Separation of debug, test, and production code.** Debug
  diagnostics are gated with `#[cfg(debug_assertions)]`; test
  assertions with `#[cfg(test)]`; production error messages are
  terse and contain no user data.

---

## References

1. Pearson, Peter K. (1990). "Fast Hashing of Variable-Length Text
   Strings." *Communications of the ACM*, Vol. 33, No. 6, pp. 677–680.
   https://dl.acm.org/doi/10.1145/78973.78978

2. Wikipedia: Pearson hashing.
   https://en.wikipedia.org/wiki/Pearson_hashing

3. Vigna, Sebastiano. "splitmix64."
   https://xorshift.di.unimi.it/splitmix64.c
   (PRNG used in the Fisher-Yates generator.)

4. Knuth, Donald E. (1998). *The Art of Computer Programming,
   Vol. 2: Seminumerical Algorithms*, 3rd ed., §3.4.2.
   (Fisher-Yates / Knuth shuffle.)

5. Bos, Mara (2023). *Rust Atomics and Locks*. O'Reilly.
   (Background on Rust production-code discipline.)
