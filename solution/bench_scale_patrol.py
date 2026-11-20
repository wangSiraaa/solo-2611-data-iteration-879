"""Scale demonstration for patrol band interpretation.

Stresses the stated limits: a band of up to 200 photos with up to 20 real
candidates per photo, where the number of complete interpretations is
astronomically large.  Shows that:

* the feasible relation is built once by layer-wise reachability, not by
  enumerating traces;
* min/max zone hits come from dynamic programming (``O(photos * cand^2)``),
  independent of the number of complete traces;
* repeated local selection checks reuse the built relation and stay fast;
* rotating the zone for a contract re-runs only the DP, not the whole band.

Run: python3 solution/bench_scale_patrol.py
"""

from __future__ import annotations

import os
import random
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chainage_catalog import CatalogRegistry, RouteCatalog, Section  # noqa: E402
from inspection_library import InspectionLibrary  # noqa: E402
from patrol_interpretation import (  # noqa: E402
    Band,
    Interpretation,
    PatrolContract,
    StepBound,
    Zone,
    interpret_band,
)

N_PHOTOS = 200
CAND_PER_PHOTO = 20
SEGMENT_LEN = 100


def build_stress(seed: int = 11):
    """A route whose labels overlap so every photo has many candidates.

    ``CAND_PER_PHOTO`` segments each carry the label window ``[0, 100)`` and
    are laid out contiguously on the physical axis.  A label ``l`` therefore
    has one physical reading per segment: ``segment_start + l``.
    """
    rnd = random.Random(seed)
    sections = []
    phys = Fraction(0)
    for i in range(CAND_PER_PHOTO):
        # Every segment uses label base 0 -> label l -> physical start + l.
        sections.append(
            Section(f"S{i:02d}", phys, phys + SEGMENT_LEN, Fraction(0))
        )
        phys += SEGMENT_LEN
    cat = RouteCatalog("d1", "D", phys, sections)

    # A monotonically non-decreasing label walk keeps at least the
    # same-segment reading advancing within the step window, while the 20
    # overlapping segments add many additional cross-segment readings.  Every
    # label lies in [0, 100), so each photo really has CAND_PER_PHOTO
    # candidates and the band has a huge (not enumerable) number of traces.
    labels = []
    cur = 0
    for _ in range(N_PHOTOS):
        cur = min(SEGMENT_LEN - 1, cur + rnd.randint(0, 3))
        labels.append(cur)
    steps = [StepBound(Fraction(0), Fraction(150)) for _ in range(N_PHOTOS - 1)]
    band = Band("STRESS", "D", tuple(labels))
    return cat, band, steps


def count_paths_estimate(interp: Interpretation) -> int:
    """Count complete interpretations exactly (for honesty about explosion).

    With 20 candidates/layer and dense edges this overflows fast; we cap the
    count so the numbers stay printable and prove it is not enumerated.
    """
    cap = 10**18
    counts = [1] * len(interp.candidates[0])
    for i in range(1, len(interp.candidates)):
        new = [0] * len(interp.candidates[i])
        inc = interp._incoming[i]
        for b in interp.viable[i]:
            total = 0
            for a in inc.get(b, ()):
                if a in interp.viable[i - 1]:
                    total += counts[a]
            new[b] = min(total, cap)
        counts = new
    return min(sum(counts[b] for b in interp.viable[-1]), cap)


def main() -> int:
    cat, band, steps = build_stress()
    print("=" * 72)
    print(f"Patrol scale: {N_PHOTOS} photos x {CAND_PER_PHOTO} candidates/photo")
    print("=" * 72)

    t0 = time.perf_counter()
    interp = interpret_band(cat, band, steps, "c1")
    build_s = time.perf_counter() - t0
    viable = [len(v) for v in interp.viable]
    edges = sum(len(e) for e in interp.edges)
    print(f"\n[1] build feasible relation")
    print(f"    {build_s*1000:.2f} ms; complete={interp.complete}")
    print(f"    photos with candidates={sum(1 for c in interp.candidates if c)}/"
          f"{interp.photo_count}; viable/photo: min={min(viable)} max={max(viable)}"
          f" mean={sum(viable)/len(viable):.1f}")
    print(f"    viable adjacent transitions: {edges}")
    paths = count_paths_estimate(interp)
    print(f"    complete interpretations: {paths:,}"
          + ("  (capped)" if paths >= 10**18 else ""))
    print("    -> the relation is a layered DAG; traces are never enumerated.")

    # 2. Zone evaluation (DP) -- repeated across many zones.
    rnd = random.Random(3)
    zones = []
    for _ in range(200):
        a = rnd.randint(0, CAND_PER_PHOTO * SEGMENT_LEN - 20)
        zones.append(Zone([(a, a + 15)]))
    t0 = time.perf_counter()
    results = [interp.evaluate_zone(z) for z in zones]
    dp_s = time.perf_counter() - t0
    print(f"\n[2] min/max zone evaluation (200 rotating zones)")
    print(f"    {dp_s*1000:.1f} ms total, {dp_s/200*1e3:.3f} ms/zone")
    print(f"    sample min/max: "
          + ", ".join(f"({r.min_hits},{r.max_hits})" for r in results[:5]))
    print("    cost: O(photos * cand^2) per zone, independent of #traces;"
          " witness traces rebuilt from stored predecessors.")

    # 3. Repeated local checks (field terminal) reuse the built relation.
    checks = []
    for _ in range(2000):
        i = rnd.randrange(0, N_PHOTOS - 1)
        ii = interp.candidates[i]
        jj = interp.candidates[i + 1]
        checks.append({i: ii[rnd.randrange(len(ii))],
                       i + 1: jj[rnd.randrange(len(jj))]})
    t0 = time.perf_counter()
    ok = sum(1 for c in checks if interp.check_selection(c).ok)
    check_s = time.perf_counter() - t0
    print(f"\n[3] local selection checks (2000 head-to-tail splices)")
    print(f"    {check_s*1000:.2f} ms total, {check_s/2000*1e6:.1f} us/check;"
          f" spliceable={ok}")

    # 4. Same-band contract zone swap is a cheap re-evaluation.
    lib = library_holding(interp)
    lib.register_contract(
        PatrolContract("Z0", band.band_id, "d1", "c1", zones[0], 1)
    )
    t0 = time.perf_counter()
    for _ in range(200):
        lib.evaluate_contract("Z0")
    swap_s = time.perf_counter() - t0
    print(f"\n[4] contract zone re-check (200 evaluations of the same relation)")
    print(f"    {swap_s*1000:.1f} ms total, {swap_s/200*1e3:.3f} ms/eval")
    print("    multiple contracts read one interpretation; a zone change"
          " re-runs only the DP.")

    print("\nNote: no trace enumeration, no per-photo-region substitution."
          " Memory is O(photos * cand^2) edges, not O(#traces).")
    return 0


def library_holding(interp):
    """Build a library that already holds ``interp`` so contracts can pin it."""
    reg = CatalogRegistry()
    cat, _band, _steps = build_stress()
    reg.add(cat)
    lib = InspectionLibrary(reg, [])
    lib.step_registry.add(
        interp.credential_revision, {interp.band.band_id: list(interp.steps)}
    )
    lib.register_band(interp.band)
    lib._interpretations[interp.interpretation_key] = interp
    return lib


if __name__ == "__main__":
    sys.exit(main())
