# YYYY-MM-DD · Short title of the change

| | |
|---|---|
| PLAN step | Step number and name from `docs/PLAN.md` section 4, or "none" |
| Branch | `stage0/NN-slug` |
| SRS requirements | IDs from `docs/SRS.md` that this change implements or affects |
| Status | Planned, Implemented, or Verified (say which, and on what) |

Copy this file to `docs/implementations/YYYY-MM-DD-<slug>.md`. Write the plan sections (Problem, What Changed, Approach, Design Decisions) before any code; complete Verification after. Project prose style: no em or en dashes.

## Problem / Motivation

Why this change is needed. What risk or gap it addresses. Which requirement or decision it serves.

## What Changed

| File | Description |
|---|---|
| `path/to/file` | One sentence per file. |

## Implementation Approach

The "how": the pattern, algorithm or strategy chosen. For each public function or class, state its contract: inputs, outputs, side effects, and the invariants it relies on.

## Mathematical / Statistical Details

Every formula, statistical test or numeric algorithm involved, in plain English with notation, so that a reader can audit the math without reading the code. Omit only for purely structural changes, and say so.

## Design Decisions

The alternatives considered, why this one was chosen, and who decided (owner decision with date, or engineering default stated for review).

## Verification

Concrete, runnable steps that confirm the change works, and their observed results. For every result, state what was run, on what hardware, model size, dataset and number of seeds (SRS S0-N-07).

**Not tested:** what this verification does not cover.

## Related Docs

Links to the SRS, PLAN, other implementation docs and `RESULTS.md` entries.
