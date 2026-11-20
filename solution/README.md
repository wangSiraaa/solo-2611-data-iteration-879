# Route chainage catalog + inspection coverage facts

Python 3.11 / SQLite. Synthetic data only — no real railway standard is
involved. The route is a physical axis `[0, L)` tiled by positive-length
half-open segments; labels within a segment are `label = physical -
physical_start + label_start` (slope fixed at 1). Labels are **not** a
global coordinate: they may repeat and may have gaps.

## Why labels cannot be a continuous coordinate

The materials' r1 route:

| section | physical `[start, end)` | label `[start, end)` |
|---------|-------------------------|----------------------|
| P1      | `[0, 400)`              | `[0, 400)`           |
| P2      | `[400, 700)`            | `[300, 600)`         |
| P3      | `[700, 1000)`           | `[650, 950)`         |

* label `350` has **two** physical candidates: P1@350 and P2@450. Treating
  `350` as one coordinate would silently swap the inspected object.
* label `620` is a **gap** (P2 ends at 600, P3 starts at 650): unresolved.
* `620` resolves only after the r2 base correction (P2 label base 300 → 330
  widens P2's labels to `[330, 630)`).

The point reverse lookup therefore returns **all** candidates and never
picks a nearest value:

| candidates | status       |
|------------|--------------|
| 0          | `unresolved` |
| 1          | `resolved`   |
| >1         | `ambiguous`  |

A caller may pass an explicit `section_id` to disambiguate; an unknown
section is reported `unresolved`, never guessed.

## Half-open semantics (the subtle part)

Point membership and interval endpoints are treated differently, on purpose:

* A physical **point** exactly on a segment's right end does **not** belong
  to that segment (`start <= p < end`).
* An interval's **upper bound** may equal a segment's exclusive end label.
  Converting a span never re-tests point membership for its upper bound, so
  a legal end-of-section span is accepted. The resulting physical interval
  is still half-open, which correctly excludes a facility sitting exactly
  at the boundary.

The `merge_intervals` union merges overlapping **and touching** intervals,
so `sum(length)` is the true covered measure — repeated or overlapping
spans are never double counted.

## Facilities

Each facility has a stable `asset_id` and a fixed **physical** position that
does not move when a label base changes. A facility counts as covered only
when its physical point lies in the merged physical coverage
(`start <= p < end`). Facility hits are **never** derived from a label
point lookup, so an ambiguous or unresolved label can never manufacture a
hit.

## Independent catalog vs. inspection facts

`chainage_catalog.py` knows nothing about jobs or reports: it can resolve
points and convert intervals on its own. `inspection_library.py` depends on
it, not the reverse. Storage is two separate schemas:

* **catalog**: `route_revision`, `route_section` (exact rationals stored as
  `n/d` text plus a sortable key for indexed range prefiltering).
* **inspection facts**: `asset`, `report`, `report_span`,
  `report_interval`, `report_facility`, `applied_op`.

Reports record the original label request, the selected section, the actual
physical coverage, the covered facility identities, and the catalog
revision. `report_id` is content-addressed: identical content is idempotent,
differing content under the same id is rejected.

## Revision behaviour

`revise_chainage` changes **only** the named section's `label_start`
(honouring the material's explicit base correction). Physical extents,
section order, and every other section are copied verbatim — this is not a
route editor, and no other edits are inferred.

After `r2` (P2 base 300 → 330), the *same* request `P2 [350,390)`:

* r1: physical `[450, 490)` → covers **B**@470 (not C@440).
* r2: physical `[420, 460)` → covers **C**@440 (not B@470).

The facility identities and physical positions are unchanged; only the
label→physical mapping moved. Old report R1 keeps explaining the original
labels and positions (`stale=true` once r2 is current).

## Running

```bash
python3 solution/run_demo.py                    # in-memory
python3 solution/run_demo.py --db route.db      # durable (resumable)
python3 -m unittest discover -s solution/tests -v
python3 solution/bench_scale.py                 # 20 sections / 500 assets / 200 jobs
```

`run_demo.py` reads `materials/730` operation by operation, applies the P2
base correction, and prints the real facility change (B in r1 → C in r2).
A file-backed DB records an op-log so re-running resumes (mutating ops are
skipped, read-only ops still render).

## Scale / cost (20 sections, 500 facilities, 200 jobs)

Measured by `bench_scale.py`:

* **Point reverse lookup** — per-section scan, ~39 µs/query at 20 sections
  (~11 % ambiguous, ~88 % unresolvable in the stress route). A per-revision
  label-range index would make this `O(log n + k)` for the k duplicate
  candidates. Duplicate labels mean the result is a *set*, so no index can
  legitimately collapse it to one value.
* **Batch conversion** — ~0.18 ms/job: convert spans `O(#spans)`, merge
  `O(spans log spans)`, facility scan `O(#assets)`. 500 assets × 200 jobs =
  100k point checks; a physical-position index would reduce the facility
  side to an interval query per job.
* **Revision invalidation** — building the new revision is ~0.05 ms (copy
  the section rows). Marking old reports stale is metadata-only. The real
  work is re-resolving jobs that must be republished, `O(jobs × (spans +
  assets))`; in the stress route moving one section's base pushed 14/200
  spans outside the moved section's label range, and those jobs are
  correctly rejected rather than remapped.

## Explicit non-goals

No path optimisation, no measurement adjustment/least-squares, no inference
of missing positions. Nothing beyond the one explicit base correction is
changed.

---

# Patrol photo-band interpretation (`patrol_interpretation.py`)

The catalogue resolves *one* label to a set of candidate physical readings.
A patrol band is an **ordered sequence** of photographed mileage labels, and
the question is whether the *whole band* can be explained — not whether each
photo could individually be somewhere.  The new module layers this on top of
an existing `RouteCatalog`; it never changes how labels resolve.

## What a legal interpretation is

For a **designated base-marker revision** `catalog_revision` and a
**designated step-credential revision** `credential_revision`:

* every photo picks exactly one real catalogue candidate
  `(section_id, physical)` — no invented positions;
* every adjacent pair satisfies its step window `[min_step, max_step]`,
  **inclusive of both bounds**, and the displacement is `>= 0` (forward-only;
  standing still is allowed when `min_step == 0`).

The feasible relation is built as a layered graph by a forward pass from the
head and a backward pass from the tail.  A candidate/transition survives only
if it lies on **some** complete interpretation.  This is a per-photo query
(`photo_view`) and a head-to-tail splice test for a named local selection
(`check_selection`).

## Zero candidates vs. a broken chain

These are reported distinctly, never collapsed:

| situation | `band.reason` | per-photo `status` |
|-----------|---------------|--------------------|
| some photo has no real candidate | `no_candidates` | `no_candidates` |
| every photo has candidates but no full path exists | `disconnected` | `unreachable` / `dead_end` |
| a full path exists | `None` | `viable` |

`check_selection` reasons: `ok`, `no_complete_path`, `photo_out_of_range`,
`no_candidates`, `not_a_candidate`, `unreachable_from_head`,
`cannot_reach_tail`, `cannot_connect`.

## Zones, contracts and outcomes

A `Zone` is a union of **non-overlapping physical half-open intervals**
`[start, end)`; overlapping registration is rejected, touching intervals are
merged.  A contract pins `(band_id, catalog_revision, credential_revision)`,
a zone, and a threshold.

`evaluate_zone` returns, over **all** legal interpretations:

* `min_hits` / `max_hits` — the fewest / most photos whose chosen point lands
  in the zone;
* one witness `min_trace` and one `max_trace` (full per-photo candidate
  choices), each verified legal by `verify_trace`.

`classify_outcome` then decides:

| condition | outcome |
|-----------|---------|
| no complete interpretation | `contradiction` |
| `min_hits >= threshold` | `confirmed` |
| `max_hits >= threshold` (only) | `manual_review` |
| neither | `not_met` |

Crucially, min/max are computed by **dynamic programming** over the layered
DAG (`O(photos * candidates^2)`), so the number of complete traces — which
grows exponentially — is never enumerated.  A single photo's possible region
therefore can never stand in for the band's association.

## Worked example (the task's case)

Route: two physical segments `[0,10)` and `[10,20)`, both labelled from 0.
Photos 2 then 4; step exactly 2.

* candidates: photo 2 → `{S1@2, S2@12}`, photo 4 → `{S1@4, S2@14}`.
* exact displacement 2 leaves only `(2,4)` or `(12,14)`.
* zone `[2,3) ∪ [14,15)`: every path hits exactly once → `min=max=1` →
  threshold 1 **confirmed**.
* zone `[2,5)`: `(12,14)` hits 0, `(2,4)` hits 2 → range `0..2` → threshold 1
  **manual_review**.
* revise the credential so the step is exactly 12: only `(2,14)` remains, and
  zone `[2,3) ∪ [14,15)` hits twice → threshold 2 **confirmed**.

Demoed in `materials/patrol` (`python3 solution/run_demo.py --materials
materials/patrol`), including a late mid-band credential revision and a
reopen of a file-backed DB.

## Versioning, persistence, and why old contracts survive

Step credentials live in their own `StepCredentialRegistry`, mirroring
`CatalogRegistry`: `revise_transition` copies every other band/boundary and
changes exactly one.  An interpretation is keyed by
`band|catalog_revision|credential_revision`; a late revision of an
intermediate credential or of a base marker yields a **new** interpretation
while the old one — and any contract pinned to it — keeps explaining its
original photos, versions and feasible relation.  `contract_view` reports
`catalog_stale` / `credential_stale` as metadata only.

Everything is persisted in a separate patrol schema (`patrol_band`,
`step_credential_*`, `patrol_interpretation`, `patrol_interp_*`,
`patrol_contract*`), so a reopened DB can answer `check_selection`,
`evaluate_contract`, and ship a fresh witness without replaying operations.
Several contracts read the **same** interpretation independently.

## Scale / cost

`bench_scale_patrol.py` — a **200-photo** band with **20 candidates/photo**
and a capped count already `>= 1e18` complete interpretations:

* build the relation once: ~0.13 s (layered forward/backward reachability);
* zone min/max: ~2.5 ms/zone (DP, independent of the trace count);
* local head-to-tail splice check: ~2 µs (reuses the built relation);
* contract zone re-evaluation (rotating zones): DP only, no rebuild.

Memory is `O(photos * candidates^2)` edges, **not** `O(#traces)` — so the
field terminal can hammer local checks and the contract side can keep
swapping zones without an exponential blow-up.

```bash
python3 solution/bench_scale_patrol.py
```

