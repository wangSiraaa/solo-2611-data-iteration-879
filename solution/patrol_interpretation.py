"""Patrol photo-band interpretation on top of a designated catalogue revision.

A *band* is an ordered sequence of photographed mileage labels.  A single
label is not a physical coordinate: when the route repeats labels, one photo
has several physical readings.  The catalogue (:mod:`chainage_catalog`) is and
stays the only thing that resolves a label to its candidate
``(section_id, physical)`` pairs for a *designated base-marker revision*.

This module explains **the whole band** at once:

* a legal interpretation picks, for every photo, one real catalogue
  candidate, so that every adjacent displacement lies inside that boundary's
  inclusive step window ``[min_step, max_step]``.  Only forward motion is
  permitted (``physical`` is non-decreasing, standing still is allowed when
  the lower bound is 0).
* candidates / transitions that cannot appear in *any* complete interpretation
  are pruned (a layered forward/backward reachability pass).  Zero candidate
  photos are reported differently from photos that merely cannot connect to
  the head or the tail.
* a zone is a set of non-overlapping physical half-open intervals.  For a
  zone, the minimum and maximum number of photos whose chosen point lands in
  the zone over *all* legal interpretations, together with one witness trace
  for each, is computed by dynamic programming over the layered graph -- never
  by enumerating the (potentially exponential) set of traces.  This is why a
  single photo's possible region can never stand in for the whole band's
  association.

Everything here is exact rational arithmetic and immutable; versioning of
step credentials mirrors :class:`chainage_catalog.CatalogRegistry`, so a late
revision of one intermediate credential or of a base marker yields *new*
interpretations while old interpretations -- and the contracts pinned to
them -- keep explaining their original versions.
"""

from __future__ import annotations

import bisect
import sqlite3
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping, Optional, Sequence

from chainage_catalog import (
    Interval,
    RouteCatalog,
    as_fraction,
    fraction_to_db,
    merge_intervals,
)


class PatrolError(ValueError):
    """Base error for patrol interpretation."""


class BandDefinitionError(PatrolError):
    """A band / step credential / zone is structurally invalid."""


# ---------------------------------------------------------------------------
# Step windows and bands
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepBound:
    """Inclusive forward displacement window between two adjacent photos."""

    min_step: Fraction
    max_step: Fraction

    def __post_init__(self):
        lo = as_fraction(self.min_step)
        hi = as_fraction(self.max_step)
        object.__setattr__(self, "min_step", lo)
        object.__setattr__(self, "max_step", hi)
        if lo < 0:
            raise BandDefinitionError(
                f"minimum step {lo} is negative; only forward motion is allowed"
            )
        if hi < lo:
            raise BandDefinitionError(
                f"step window [{lo}, {hi}] has max below min"
            )

    def admits(self, delta: Fraction) -> bool:
        """True when ``delta`` lies in the inclusive window and is forward."""
        return delta >= 0 and self.min_step <= delta <= self.max_step

    @classmethod
    def from_pair(cls, pair: Sequence) -> "StepBound":
        lo, hi = pair
        return cls(as_fraction(lo), as_fraction(hi))

    def to_canonical(self) -> dict:
        return {
            "min_step": fraction_to_db(self.min_step),
            "max_step": fraction_to_db(self.max_step),
        }


@dataclass(frozen=True)
class Band:
    """Ordered photographed labels plus the route they belong to."""

    band_id: str
    route_id: str
    labels: tuple[Fraction, ...]

    def __post_init__(self):
        labels = tuple(as_fraction(x) for x in self.labels)
        object.__setattr__(self, "labels", labels)
        if not labels:
            raise BandDefinitionError("a band needs at least one photo")

    @property
    def photo_count(self) -> int:
        return len(self.labels)

    @classmethod
    def from_dict(cls, data: dict) -> "Band":
        return cls(
            band_id=str(data["band_id"]),
            route_id=str(data["route_id"]),
            labels=tuple(data["labels"]),
        )

    def to_dict(self) -> dict:
        return {
            "band_id": self.band_id,
            "route_id": self.route_id,
            "labels": [fraction_to_db(x) for x in self.labels],
        }


# ---------------------------------------------------------------------------
# Zones (non-overlapping physical half-open intervals)
# ---------------------------------------------------------------------------


class Zone:
    """A union of non-overlapping physical half-open intervals ``[start, end)``.

    Registered intervals may touch (``[0, 5)`` and ``[5, 9)`` do not overlap);
    touching intervals are merged into their equivalent union so membership
    and the canonical fingerprint are stable.  Overlapping intervals are a
    registration error, never silently reconciled.
    """

    def __init__(self, intervals: Iterable[Sequence]):
        parsed: list[tuple[Fraction, Fraction]] = []
        for item in intervals:
            start = as_fraction(item[0])
            end = as_fraction(item[1])
            if not start < end:
                raise BandDefinitionError(
                    f"zone interval [{start}, {end}) is not positive length"
                )
            parsed.append((start, end))
        parsed.sort()
        for (s0, e0), (s1, _e1) in zip(parsed, parsed[1:]):
            if s1 < e0:
                raise BandDefinitionError(
                    f"zone intervals [{s0}, {e0}) and "
                    f"[{s1}, ...) overlap"
                )
        merged = merge_intervals(Interval(s, e) for s, e in parsed)
        self._intervals = tuple((iv.start, iv.end) for iv in merged)
        self._starts = tuple(iv.start for iv in merged)

    @property
    def intervals(self) -> tuple[tuple[Fraction, Fraction], ...]:
        return self._intervals

    def contains(self, physical) -> bool:
        point = as_fraction(physical)
        idx = bisect.bisect_right(self._starts, point) - 1
        if idx < 0:
            return False
        return point < self._intervals[idx][1]

    def to_list(self) -> list[dict]:
        return [
            {"start": fraction_to_db(s), "end": fraction_to_db(e)}
            for s, e in self._intervals
        ]

    def fingerprint(self) -> tuple:
        return tuple((fraction_to_db(s), fraction_to_db(e))
                     for s, e in self._intervals)

    @classmethod
    def from_list(cls, items: Iterable[Mapping]) -> "Zone":
        return cls([(it["start"], it["end"]) for it in items])

    def __eq__(self, other):
        if not isinstance(other, Zone):
            return NotImplemented
        return self.fingerprint() == other.fingerprint()

    def __hash__(self):
        return hash(self.fingerprint())

    def __repr__(self):
        return f"Zone({self.to_list()})"


# ---------------------------------------------------------------------------
# Candidates and checked selections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhotoCandidate:
    section_id: str
    physical: Fraction

    def __post_init__(self):
        object.__setattr__(self, "physical", as_fraction(self.physical))

    def to_canonical(self) -> dict:
        return {
            "section_id": self.section_id,
            "physical": fraction_to_db(self.physical),
        }


def _normalise_candidate_spec(spec) -> PhotoCandidate:
    if isinstance(spec, PhotoCandidate):
        return spec
    if isinstance(spec, Mapping):
        section = spec.get("section_id")
        physical = spec.get("physical", spec.get("physical_position"))
        if section is None or physical is None:
            raise PatrolError(f"bad candidate spec {spec!r}")
        return PhotoCandidate(str(section), as_fraction(physical))
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        return PhotoCandidate(str(spec[0]), as_fraction(spec[1]))
    raise PatrolError(f"bad candidate spec {spec!r}")


@dataclass(frozen=True)
class SelectionCheck:
    """Result of asking whether a local choice can be spliced head-to-tail."""

    ok: bool
    # ok | no_complete_path | photo_out_of_range | no_candidates |
    # not_a_candidate | unreachable_from_head | cannot_reach_tail |
    # cannot_connect
    reason: str
    blocked_photo: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "blocked_photo": self.blocked_photo,
        }


@dataclass(frozen=True)
class ZoneEvaluation:
    """Min/max zone-hit count over all legal interpretations + witnesses."""

    complete: bool
    min_hits: Optional[int]
    max_hits: Optional[int]
    min_trace: Optional[tuple[PhotoCandidate, ...]]
    max_trace: Optional[tuple[PhotoCandidate, ...]]

    def to_dict(self) -> dict:
        def trace(t):
            return None if t is None else [c.to_canonical() for c in t]

        return {
            "complete": self.complete,
            "min_hits": self.min_hits,
            "max_hits": self.max_hits,
            "min_trace": trace(self.min_trace),
            "max_trace": trace(self.max_trace),
        }


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------


def make_interp_key(band_id: str, catalog_revision: str, credential_revision: str) -> str:
    return f"{band_id}|{catalog_revision}|{credential_revision}"


class Interpretation:
    """Immutable feasible relation for one (band, base, credential) triple.

    Holds the per-photo real candidates, the forward/backward reachability
    that prunes them to the ones appearing in *some* complete interpretation,
    and the viable adjacent transitions.  Query it per photo, ask whether a
    local choice can connect head-to-tail, or evaluate a zone.
    """

    def __init__(
        self,
        band: Band,
        catalog_revision: str,
        credential_revision: str,
        steps: Sequence[StepBound],
        candidates: Sequence[Sequence[PhotoCandidate]],
        forward: Sequence[Iterable[int]],
        backward: Sequence[Iterable[int]],
        edges: Sequence[Iterable[tuple[int, int]]],
        complete: bool,
        reason: Optional[str],
    ):
        self.band = band
        self.catalog_revision = catalog_revision
        self.credential_revision = credential_revision
        self.steps = tuple(steps)
        self.candidates = tuple(tuple(c) for c in candidates)
        if len(self.candidates) != band.photo_count:
            raise PatrolError("candidate layers do not match band length")
        self.forward = tuple(frozenset(x) for x in forward)
        self.backward = tuple(frozenset(x) for x in backward)
        self.edges = tuple(tuple(e) for e in edges)
        self.complete = bool(complete)
        self.reason = reason
        self.viable = tuple(
            sorted(self.forward[i] & self.backward[i])
            for i in range(len(self.candidates))
        )
        # adjacency, built once so repeated local checks stay cheap
        outgoing: list[dict[int, tuple[int, ...]]] = [
            {} for _ in range(max(len(self.candidates) - 1, 0))
        ]
        incoming: list[dict[int, tuple[int, ...]]] = [
            {} for _ in range(len(self.candidates))
        ]
        for i, layer_edges in enumerate(self.edges):
            out: dict[int, list[int]] = {}
            inc: dict[int, list[int]] = {}
            for a, b in layer_edges:
                out.setdefault(a, []).append(b)
                inc.setdefault(b, []).append(a)
            outgoing[i] = {k: tuple(v) for k, v in out.items()}
            incoming[i + 1] = {k: tuple(v) for k, v in inc.items()}
        self._outgoing = tuple(outgoing)
        self._incoming = tuple(incoming)

    # -- identity -----------------------------------------------------
    @property
    def interpretation_key(self) -> str:
        return make_interp_key(
            self.band.band_id, self.catalog_revision, self.credential_revision
        )

    @property
    def photo_count(self) -> int:
        return len(self.candidates)

    # -- per-photo query ---------------------------------------------
    def _photo_status(self, i: int) -> str:
        cands = self.candidates[i]
        if not cands:
            return "no_candidates"
        fwd = self.forward[i]
        bwd = self.backward[i]
        if not fwd and not bwd:
            return "isolated"
        if not fwd:
            return "unreachable"       # cannot connect to the head
        if not bwd:
            return "dead_end"          # cannot connect to the tail
        return "viable"

    def photo_view(self, i: int) -> dict:
        if not 0 <= i < self.photo_count:
            raise PatrolError(f"photo index {i} out of range")
        layer = self.candidates[i]
        incoming = self._incoming[i]
        outgoing = self._outgoing[i] if i < self.photo_count - 1 else {}
        cand_view = []
        for idx, c in enumerate(layer):
            cand_view.append(
                {
                    **c.to_canonical(),
                    "candidate_index": idx,
                    "forward": idx in self.forward[i],
                    "backward": idx in self.backward[i],
                    "viable": idx in self.forward[i] and idx in self.backward[i],
                    "predecessors": list(incoming.get(idx, ())),
                    "successors": list(outgoing.get(idx, ())),
                }
            )
        return {
            "photo_index": i,
            "label": fraction_to_db(self.band.labels[i]),
            "status": self._photo_status(i),
            "viable_candidates": list(self.viable[i]),
            "candidates": cand_view,
        }

    def band_view(self) -> dict:
        return {
            "band_id": self.band.band_id,
            "route_id": self.band.route_id,
            "catalog_revision": self.catalog_revision,
            "credential_revision": self.credential_revision,
            "interpretation_key": self.interpretation_key,
            "photo_count": self.photo_count,
            "complete": self.complete,
            "reason": self.reason,
            "photos_without_candidates": [
                i for i, c in enumerate(self.candidates) if not c
            ],
            "photo_status": [self._photo_status(i) for i in range(self.photo_count)],
            "viable_candidate_counts": [len(self.viable[i]) for i in range(self.photo_count)],
        }

    # -- local selection feasibility ---------------------------------
    def _find_candidate(self, i: int, spec) -> Optional[int]:
        want = _normalise_candidate_spec(spec)
        for idx, c in enumerate(self.candidates[i]):
            if c.section_id == want.section_id and c.physical == want.physical:
                return idx
        return None

    def _can_reach(self, i: int, a: int, j: int, b: int) -> bool:
        """Is there a viable path from candidate ``a``@i to ``b``@j?"""
        if j < i:
            return False
        if j == i:
            return a == b
        frontier = {a}
        for layer in range(i, j):
            nxt: set[int] = set()
            for x in frontier:
                nxt.update(self._outgoing[layer].get(x, ()))
            frontier = nxt
            if not frontier:
                return False
        return b in frontier

    def check_selection(
        self, selection: Mapping[int, object]
    ) -> SelectionCheck:
        """Can the specified per-photo choices be spliced head-to-tail?

        ``selection`` maps photo index to a candidate spec (a
        :class:`PhotoCandidate`, a ``(section_id, physical)`` pair, or a mapping
        with those keys).  Unspecified photos are free.  The answer is True
        only when every named candidate is real, on some complete
        interpretation, and the named choices lie in order on one common full
        path.
        """
        if not self.complete:
            # Still distinguish a truly bad named candidate from a band that
            # simply has no complete interpretation.
            for i, spec in selection.items():
                if not 0 <= int(i) < self.photo_count:
                    return SelectionCheck(False, "photo_out_of_range", int(i))
                if not self.candidates[int(i)]:
                    return SelectionCheck(False, "no_candidates", int(i))
                if self._find_candidate(int(i), spec) is None:
                    return SelectionCheck(False, "not_a_candidate", int(i))
            return SelectionCheck(False, "no_complete_path", None)

        chosen: dict[int, int] = {}
        for raw_i, spec in selection.items():
            i = int(raw_i)
            if not 0 <= i < self.photo_count:
                return SelectionCheck(False, "photo_out_of_range", i)
            cand_idx = self._find_candidate(i, spec)
            if cand_idx is None:
                reason = (
                    "no_candidates" if not self.candidates[i] else "not_a_candidate"
                )
                return SelectionCheck(False, reason, i)
            if cand_idx not in self.forward[i]:
                return SelectionCheck(False, "unreachable_from_head", i)
            if cand_idx not in self.backward[i]:
                return SelectionCheck(False, "cannot_reach_tail", i)
            chosen[i] = cand_idx

        order = sorted(chosen)
        for i, j in zip(order, order[1:]):
            if not self._can_reach(i, chosen[i], j, chosen[j]):
                return SelectionCheck(False, "cannot_connect", j)
        return SelectionCheck(True, "ok", None)

    # -- zone evaluation ---------------------------------------------
    def evaluate_zone(self, zone: Zone) -> ZoneEvaluation:
        """Min/max zone-hit count over all legal interpretations.

        Dynamic programming over the layered DAG: ``O(photos * candidates^2)``
        time, independent of the number of complete traces.  Witness traces
        are reconstructed from stored predecessors.
        """
        n = self.photo_count
        if not self.complete:
            return ZoneEvaluation(False, None, None, None, None)

        weights = [
            [1 if zone.contains(c.physical) else 0 for c in layer]
            for layer in self.candidates
        ]
        # Per-layer value maps restricted to viable candidates.
        min_v: list[dict[int, int]] = [dict() for _ in range(n)]
        max_v: list[dict[int, int]] = [dict() for _ in range(n)]
        min_par: list[dict[int, Optional[int]]] = [dict() for _ in range(n)]
        max_par: list[dict[int, Optional[int]]] = [dict() for _ in range(n)]

        for a in self.viable[0]:
            min_v[0][a] = max_v[0][a] = weights[0][a]
            min_par[0][a] = max_par[0][a] = None

        for i in range(1, n):
            for b in self.viable[i]:
                preds = self._incoming[i].get(b, ())
                best_min = best_max = None
                pmin = pmax = None
                for a in preds:
                    if a not in min_v[i - 1]:
                        continue
                    vm = min_v[i - 1][a]
                    vx = max_v[i - 1][a]
                    if best_min is None or vm < best_min:
                        best_min, pmin = vm, a
                    if best_max is None or vx > best_max:
                        best_max, pmax = vx, a
                if best_min is None:
                    continue
                min_v[i][b] = weights[i][b] + best_min
                max_v[i][b] = weights[i][b] + best_max
                min_par[i][b] = pmin
                max_par[i][b] = pmax

        last = n - 1
        finals = [b for b in self.viable[last] if b in min_v[last]]
        if not finals:  # cannot happen when complete, but stay honest
            return ZoneEvaluation(False, None, None, None, None)
        min_end = min(finals, key=lambda b: (min_v[last][b], b))
        max_end = min(finals, key=lambda b: (-max_v[last][b], b))
        min_hits = min_v[last][min_end]
        max_hits = max_v[last][max_end]

        def build(end: int, par: list[dict[int, Optional[int]]]):
            trace: list[PhotoCandidate] = []
            cur: Optional[int] = end
            for i in range(last, -1, -1):
                assert cur is not None
                trace.append(self.candidates[i][cur])
                if i == 0:
                    break
                cur = par[i][cur]
            trace.reverse()
            return tuple(trace)

        return ZoneEvaluation(
            complete=True,
            min_hits=min_hits,
            max_hits=max_hits,
            min_trace=build(min_end, min_par),
            max_trace=build(max_end, max_par),
        )

    # -- validation helper -------------------------------------------
    def verify_trace(self, trace: Sequence[PhotoCandidate]) -> tuple[bool, Optional[str]]:
        """Check that ``trace`` is one legal complete interpretation."""
        if len(trace) != self.photo_count:
            return False, f"trace has {len(trace)} photos, band has {self.photo_count}"
        for i, cand in enumerate(trace):
            try:
                idx = self._find_candidate(i, cand)
            except PatrolError:
                return False, f"photo {i}: bad candidate"
            if idx is None:
                return False, f"photo {i}: {cand} is not a real candidate"
        for i in range(self.photo_count - 1):
            delta = trace[i + 1].physical - trace[i].physical
            if not self.steps[i].admits(delta):
                return False, (
                    f"photo {i}->{i + 1}: displacement {delta} outside "
                    f"[{self.steps[i].min_step}, {self.steps[i].max_step}]"
                )
        return True, None


def interpret_band(
    catalog: RouteCatalog,
    band: Band,
    steps: Sequence[StepBound],
    credential_revision: str = "<inline>",
) -> Interpretation:
    """Explain the whole band against a *designated* catalogue revision.

    ``credential_revision`` is only a label used to key the resulting
    :class:`Interpretation`; the step windows themselves are passed in.
    """
    if catalog.route_id != band.route_id:
        raise PatrolError(
            f"band {band.band_id!r} is route {band.route_id!r}, catalogue "
            f"revision {catalog.revision_id!r} is route {catalog.route_id!r}"
        )
    steps = tuple(steps)
    n = band.photo_count
    if len(steps) != n - 1:
        raise BandDefinitionError(
            f"band {band.band_id!r} has {n} photos and needs {n - 1} step "
            f"windows, got {len(steps)}"
        )
    for s in steps:
        if not isinstance(s, StepBound):
            raise BandDefinitionError("steps must be StepBound instances")

    candidates: list[tuple[PhotoCandidate, ...]] = []
    for label in band.labels:
        raw = catalog.resolve_label_point(label)
        seen = sorted(
            {(c["section_id"], c["physical"]) for c in raw["candidates"]},
            key=lambda t: (t[1], t[0]),
        )
        candidates.append(
            tuple(PhotoCandidate(s, p) for s, p in seen)
        )

    # forward reachability from the first photo
    forward: list[set[int]] = [set() for _ in range(n)]
    forward[0] = set(range(len(candidates[0])))
    for i in range(1, n):
        bound = steps[i - 1]
        prev = forward[i - 1]
        for b, cb in enumerate(candidates[i]):
            for a in prev:
                if bound.admits(cb.physical - candidates[i - 1][a].physical):
                    forward[i].add(b)
                    break

    # backward reachability to the last photo
    backward: list[set[int]] = [set() for _ in range(n)]
    backward[n - 1] = set(range(len(candidates[n - 1])))
    for i in range(n - 2, -1, -1):
        bound = steps[i]
        nxt = backward[i + 1]
        for a, ca in enumerate(candidates[i]):
            for b in nxt:
                if bound.admits(candidates[i + 1][b].physical - ca.physical):
                    backward[i].add(a)
                    break

    # viable adjacent transitions (edges lying on some complete path)
    edges: list[tuple[tuple[int, int], ...]] = []
    for i in range(n - 1):
        bound = steps[i]
        layer_edges = []
        for a, ca in enumerate(candidates[i]):
            if a not in forward[i]:
                continue
            for b, cb in enumerate(candidates[i + 1]):
                if b in backward[i + 1] and bound.admits(cb.physical - ca.physical):
                    layer_edges.append((a, b))
        edges.append(tuple(layer_edges))

    complete = bool(forward[n - 1])
    zero = [i for i, c in enumerate(candidates) if not c]
    if zero:
        reason = "no_candidates"
    elif not complete:
        reason = "disconnected"
    else:
        reason = None

    return Interpretation(
        band=band,
        catalog_revision=catalog.revision_id,
        credential_revision=credential_revision,
        steps=steps,
        candidates=candidates,
        forward=forward,
        backward=backward,
        edges=edges,
        complete=complete,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Step-credential revisions (mirror of CatalogRegistry)
# ---------------------------------------------------------------------------


class StepCredentialRegistry:
    """Ordered set of step-credential revisions, newest is ``current()``.

    A revision maps ``band_id -> tuple[StepBound, ...]`` (one window per
    adjacent photo boundary).  ``revise_transition`` copies every other band
    and boundary verbatim and changes exactly one intermediate credential, so
    a late change yields a *new* revision instead of mutating history.
    """

    def __init__(self):
        self._revs: list[str] = []
        self._by: dict[str, dict[str, tuple[StepBound, ...]]] = {}

    def add(self, revision_id: str, bounds_by_band: Mapping[str, Sequence]) -> None:
        if revision_id in self._by:
            raise BandDefinitionError(f"credential revision {revision_id!r} exists")
        clean: dict[str, tuple[StepBound, ...]] = {}
        for band_id, bounds in bounds_by_band.items():
            clean[band_id] = tuple(
                b if isinstance(b, StepBound) else StepBound.from_pair(b)
                for b in bounds
            )
        self._revs.append(revision_id)
        self._by[revision_id] = clean

    def get(self, revision_id: str) -> dict[str, tuple[StepBound, ...]]:
        try:
            return self._by[revision_id]
        except KeyError as exc:
            raise PatrolError(f"unknown credential revision {revision_id!r}") from exc

    def current(self) -> str:
        if not self._revs:
            raise PatrolError("no credential revisions registered")
        return self._revs[-1]

    def all_revisions(self) -> list[str]:
        return list(self._revs)

    def bounds_for(self, revision_id: str, band_id: str) -> tuple[StepBound, ...]:
        rev = self.get(revision_id)
        if band_id not in rev:
            raise PatrolError(
                f"credential revision {revision_id!r} has no bounds for band "
                f"{band_id!r}"
            )
        return rev[band_id]

    def revise_transition(
        self,
        revision_id: str,
        band_id: str,
        index: int,
        min_step,
        max_step,
        base_revision: Optional[str] = None,
    ) -> None:
        base = self._by[base_revision or self.current()]
        if band_id not in base:
            raise PatrolError(
                f"base revision has no bounds for band {band_id!r}"
            )
        bounds = list(base[band_id])
        if not 0 <= index < len(bounds):
            raise PatrolError(
                f"transition {index} out of range for band {band_id!r} "
                f"({len(bounds)} transitions)"
            )
        bounds[index] = StepBound(as_fraction(min_step), as_fraction(max_step))
        new_map = dict(base)
        new_map[band_id] = tuple(bounds)
        self.add(revision_id, new_map)


# ---------------------------------------------------------------------------
# Contract outcome classification
# ---------------------------------------------------------------------------


#: ``min_hits`` cleared the threshold -> confirm.
CONFIRMED = "confirmed"
#: only ``max_hits`` clears the threshold -> a human must look.
MANUAL_REVIEW = "manual_review"
#: neither bound clears the threshold -> not met.
NOT_MET = "not_met"
#: no complete interpretation exists -> the material contradicts itself.
CONTRADICTION = "contradiction"


def classify_outcome(
    complete: bool, min_hits: Optional[int], max_hits: Optional[int], threshold: int
) -> str:
    if not complete:
        return CONTRADICTION
    if min_hits >= threshold:
        return CONFIRMED
    if max_hits >= threshold:
        return MANUAL_REVIEW
    return NOT_MET


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatrolContract:
    """A zone/threshold contract pinned to one interpretation key.

    Pinning ``(band_id, catalog_revision, credential_revision)`` is what keeps
    an old contract valid: a later revision of a step credential or a base
    marker produces a *new* interpretation, but this contract keeps naming the
    photos, version and feasible relation it was registered against.
    """

    contract_id: str
    band_id: str
    catalog_revision: str
    credential_revision: str
    zone: Zone
    threshold: int

    def __post_init__(self):
        if not self.contract_id:
            raise BandDefinitionError("contract_id must be non-empty")
        if not isinstance(self.zone, Zone):
            raise BandDefinitionError("zone must be a Zone")
        if self.threshold < 0:
            raise BandDefinitionError("threshold must be non-negative")

    @property
    def interpretation_key(self) -> str:
        return make_interp_key(
            self.band_id, self.catalog_revision, self.credential_revision
        )

    def canonical_content(self) -> dict:
        return {
            "band_id": self.band_id,
            "catalog_revision": self.catalog_revision,
            "credential_revision": self.credential_revision,
            "zone": self.zone.to_list(),
            "threshold": self.threshold,
        }

    def fingerprint(self) -> str:
        import hashlib
        import json

        payload = json.dumps(
            self.canonical_content(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        data = self.canonical_content()
        data["contract_id"] = self.contract_id
        data["interpretation_key"] = self.interpretation_key
        data["fingerprint"] = self.fingerprint()
        return data


@dataclass(frozen=True)
class ContractEvaluation:
    contract_id: str
    interpretation_key: str
    band_id: str
    complete: bool
    min_hits: Optional[int]
    max_hits: Optional[int]
    threshold: int
    outcome: str
    min_trace: Optional[tuple[PhotoCandidate, ...]]
    max_trace: Optional[tuple[PhotoCandidate, ...]]

    def to_dict(self) -> dict:
        def trace(t):
            return None if t is None else [c.to_canonical() for c in t]

        return {
            "contract_id": self.contract_id,
            "interpretation_key": self.interpretation_key,
            "band_id": self.band_id,
            "complete": self.complete,
            "min_hits": self.min_hits,
            "max_hits": self.max_hits,
            "threshold": self.threshold,
            "outcome": self.outcome,
            "min_trace": trace(self.min_trace),
            "max_trace": trace(self.max_trace),
        }


class ContractConflictError(PatrolError):
    """A contract_id was reused with different content."""


# ---------------------------------------------------------------------------
# SQLite persistence for patrol material (bands / credentials / relations)
# ---------------------------------------------------------------------------


PATROL_SCHEMA = """
CREATE TABLE IF NOT EXISTS patrol_band (
    band_id    TEXT PRIMARY KEY,
    route_id   TEXT NOT NULL,
    n_photos   INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS patrol_band_photo (
    band_id    TEXT NOT NULL REFERENCES patrol_band(band_id),
    idx        INTEGER NOT NULL,
    label_num  TEXT NOT NULL,
    label_den  TEXT NOT NULL,
    PRIMARY KEY (band_id, idx)
);
CREATE TABLE IF NOT EXISTS step_credential_revision (
    revision_id TEXT PRIMARY KEY,
    created_seq INTEGER NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS step_credential_band (
    revision_id TEXT NOT NULL REFERENCES step_credential_revision(revision_id),
    band_id     TEXT NOT NULL,
    PRIMARY KEY (revision_id, band_id)
);
CREATE TABLE IF NOT EXISTS step_credential_bound (
    revision_id TEXT NOT NULL REFERENCES step_credential_revision(revision_id),
    band_id     TEXT NOT NULL,
    idx         INTEGER NOT NULL,
    min_num TEXT NOT NULL, min_den TEXT NOT NULL,
    max_num TEXT NOT NULL, max_den TEXT NOT NULL,
    PRIMARY KEY (revision_id, band_id, idx)
);
CREATE TABLE IF NOT EXISTS patrol_interpretation (
    interp_key          TEXT PRIMARY KEY,
    band_id             TEXT NOT NULL,
    catalog_revision    TEXT NOT NULL,
    credential_revision TEXT NOT NULL,
    complete            INTEGER NOT NULL,
    reason              TEXT,
    n_photos            INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS patrol_interp_step (
    interp_key TEXT NOT NULL REFERENCES patrol_interpretation(interp_key),
    idx        INTEGER NOT NULL,
    min_num TEXT NOT NULL, min_den TEXT NOT NULL,
    max_num TEXT NOT NULL, max_den TEXT NOT NULL,
    PRIMARY KEY (interp_key, idx)
);
CREATE TABLE IF NOT EXISTS patrol_interp_candidate (
    interp_key TEXT NOT NULL REFERENCES patrol_interpretation(interp_key),
    photo_idx  INTEGER NOT NULL,
    cand_idx   INTEGER NOT NULL,
    section_id TEXT NOT NULL,
    phys_num   TEXT NOT NULL,
    phys_den   TEXT NOT NULL,
    fwd        INTEGER NOT NULL,
    bwd        INTEGER NOT NULL,
    PRIMARY KEY (interp_key, photo_idx, cand_idx)
);
CREATE TABLE IF NOT EXISTS patrol_interp_edge (
    interp_key   TEXT NOT NULL REFERENCES patrol_interpretation(interp_key),
    boundary_idx INTEGER NOT NULL,
    from_idx     INTEGER NOT NULL,
    to_idx       INTEGER NOT NULL,
    PRIMARY KEY (interp_key, boundary_idx, from_idx, to_idx)
);
CREATE TABLE IF NOT EXISTS patrol_contract (
    contract_id         TEXT PRIMARY KEY,
    band_id             TEXT NOT NULL,
    catalog_revision    TEXT NOT NULL,
    credential_revision TEXT NOT NULL,
    threshold           INTEGER NOT NULL,
    fingerprint         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS patrol_contract_zone (
    contract_id TEXT NOT NULL REFERENCES patrol_contract(contract_id),
    idx         INTEGER NOT NULL,
    start_num TEXT NOT NULL, start_den TEXT NOT NULL,
    end_num   TEXT NOT NULL, end_den   TEXT NOT NULL,
    PRIMARY KEY (contract_id, idx)
);
"""


def ensure_patrol_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(PATROL_SCHEMA)


def persist_band(conn: sqlite3.Connection, band: Band) -> None:
    with conn:
        try:
            conn.execute(
                "INSERT INTO patrol_band (band_id, route_id, n_photos) "
                "VALUES (?, ?, ?)",
                (band.band_id, band.route_id, band.photo_count),
            )
        except sqlite3.IntegrityError:
            existing = load_bands(conn).get(band.band_id)
            if existing is None or existing.to_dict() != band.to_dict():
                raise BandDefinitionError(
                    f"band {band.band_id!r} stored with different content"
                )
            return
        for i, label in enumerate(band.labels):
            conn.execute(
                "INSERT INTO patrol_band_photo (band_id, idx, label_num, label_den) "
                "VALUES (?, ?, ?, ?)",
                (band.band_id, i, str(label.numerator), str(label.denominator)),
            )


def load_bands(conn: sqlite3.Connection) -> dict[str, Band]:
    bands: dict[str, Band] = {}
    for row in conn.execute("SELECT * FROM patrol_band").fetchall():
        labels = [
            Fraction(int(r["label_num"]), int(r["label_den"]))
            for r in conn.execute(
                "SELECT * FROM patrol_band_photo WHERE band_id = ? ORDER BY idx",
                (row["band_id"],),
            ).fetchall()
        ]
        bands[row["band_id"]] = Band(row["band_id"], row["route_id"], tuple(labels))
    return bands


def persist_step_credential_revision(
    conn: sqlite3.Connection, revision_id: str, bounds_by_band: Mapping[str, Sequence]
) -> None:
    clean = {
        band_id: tuple(
            b if isinstance(b, StepBound) else StepBound.from_pair(b) for b in bounds
        )
        for band_id, bounds in bounds_by_band.items()
    }
    with conn:
        try:
            conn.execute(
                "INSERT INTO step_credential_revision (revision_id, created_seq) "
                "VALUES (?, (SELECT COALESCE(MAX(created_seq), 0) + 1 "
                "FROM step_credential_revision))",
                (revision_id,),
            )
        except sqlite3.IntegrityError:
            existing = load_step_credential_registry(conn).get(revision_id)
            if existing != clean:
                raise BandDefinitionError(
                    f"credential revision {revision_id!r} stored differently"
                )
            return
        for band_id, bounds in clean.items():
            conn.execute(
                "INSERT INTO step_credential_band (revision_id, band_id) VALUES (?, ?)",
                (revision_id, band_id),
            )
            for i, b in enumerate(bounds):
                conn.execute(
                    "INSERT INTO step_credential_bound "
                    "(revision_id, band_id, idx, min_num, min_den, max_num, max_den) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        revision_id, band_id, i,
                        str(b.min_step.numerator), str(b.min_step.denominator),
                        str(b.max_step.numerator), str(b.max_step.denominator),
                    ),
                )


def load_step_credential_registry(conn: sqlite3.Connection) -> StepCredentialRegistry:
    reg = StepCredentialRegistry()
    revs = conn.execute(
        "SELECT revision_id FROM step_credential_revision ORDER BY created_seq"
    ).fetchall()
    for r in revs:
        rid = r["revision_id"]
        bands = conn.execute(
            "SELECT band_id FROM step_credential_band WHERE revision_id = ?",
            (rid,),
        ).fetchall()
        mapping: dict[str, tuple[StepBound, ...]] = {}
        for b in bands:
            bid = b["band_id"]
            bounds = [
                StepBound(
                    Fraction(int(x["min_num"]), int(x["min_den"])),
                    Fraction(int(x["max_num"]), int(x["max_den"])),
                )
                for x in conn.execute(
                    "SELECT * FROM step_credential_bound "
                    "WHERE revision_id = ? AND band_id = ? ORDER BY idx",
                    (rid, bid),
                ).fetchall()
            ]
            mapping[bid] = tuple(bounds)
        reg.add(rid, mapping)
    return reg


def persist_interpretation(conn: sqlite3.Connection, interp: Interpretation) -> None:
    key = interp.interpretation_key
    with conn:
        try:
            conn.execute(
                "INSERT INTO patrol_interpretation (interp_key, band_id, "
                "catalog_revision, credential_revision, complete, reason, n_photos) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    key, interp.band.band_id, interp.catalog_revision,
                    interp.credential_revision, int(interp.complete),
                    interp.reason, interp.photo_count,
                ),
            )
        except sqlite3.IntegrityError:
            existing = load_interpretation(conn, key)
            if _interpretation_fingerprint(existing) != _interpretation_fingerprint(interp):
                raise PatrolError(
                    f"interpretation {key!r} stored with different content"
                )
            return
        for i, b in enumerate(interp.steps):
            conn.execute(
                "INSERT INTO patrol_interp_step (interp_key, idx, min_num, min_den, "
                "max_num, max_den) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    key, i,
                    str(b.min_step.numerator), str(b.min_step.denominator),
                    str(b.max_step.numerator), str(b.max_step.denominator),
                ),
            )
        for i, layer in enumerate(interp.candidates):
            for idx, c in enumerate(layer):
                conn.execute(
                    "INSERT INTO patrol_interp_candidate (interp_key, photo_idx, "
                    "cand_idx, section_id, phys_num, phys_den, fwd, bwd) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        key, i, idx, c.section_id,
                        str(c.physical.numerator), str(c.physical.denominator),
                        int(idx in interp.forward[i]), int(idx in interp.backward[i]),
                    ),
                )
        for i, layer_edges in enumerate(interp.edges):
            for a, b in layer_edges:
                conn.execute(
                    "INSERT INTO patrol_interp_edge (interp_key, boundary_idx, "
                    "from_idx, to_idx) VALUES (?, ?, ?, ?)",
                    (key, i, a, b),
                )


def _interpretation_fingerprint(interp: Interpretation) -> tuple:
    return (
        interp.band.band_id,
        interp.catalog_revision,
        interp.credential_revision,
        interp.complete,
        interp.reason,
        tuple(
            tuple((c.section_id, c.physical) for c in layer)
            for layer in interp.candidates
        ),
        tuple(tuple(sorted(x)) for x in interp.forward),
        tuple(tuple(sorted(x)) for x in interp.backward),
        tuple(tuple(e) for e in interp.edges),
    )


def load_interpretation(conn: sqlite3.Connection, interp_key: str) -> Interpretation:
    row = conn.execute(
        "SELECT * FROM patrol_interpretation WHERE interp_key = ?", (interp_key,)
    ).fetchone()
    if row is None:
        raise PatrolError(f"interpretation {interp_key!r} not stored")
    bands = load_bands(conn)
    band = bands[row["band_id"]]
    n = int(row["n_photos"])

    steps = [
        StepBound(
            Fraction(int(r["min_num"]), int(r["min_den"])),
            Fraction(int(r["max_num"]), int(r["max_den"])),
        )
        for r in conn.execute(
            "SELECT * FROM patrol_interp_step WHERE interp_key = ? ORDER BY idx",
            (interp_key,),
        ).fetchall()
    ]

    cand_rows = conn.execute(
        "SELECT * FROM patrol_interp_candidate WHERE interp_key = ? "
        "ORDER BY photo_idx, cand_idx",
        (interp_key,),
    ).fetchall()
    grouped: list[dict[int, PhotoCandidate]] = [dict() for _ in range(n)]
    forward: list[set[int]] = [set() for _ in range(n)]
    backward: list[set[int]] = [set() for _ in range(n)]
    for r in cand_rows:
        pi = int(r["photo_idx"])
        ci = int(r["cand_idx"])
        grouped[pi][ci] = PhotoCandidate(
            r["section_id"],
            Fraction(int(r["phys_num"]), int(r["phys_den"])),
        )
        if int(r["fwd"]):
            forward[pi].add(ci)
        if int(r["bwd"]):
            backward[pi].add(ci)
    # cand_idx is dense (0..k-1), so sorting the keys recovers layer order.
    candidates = [
        tuple(grouped[pi][ci] for ci in sorted(grouped[pi])) for pi in range(n)
    ]

    edges_by_boundary: list[list[tuple[int, int]]] = [[] for _ in range(max(n - 1, 0))]
    for r in conn.execute(
        "SELECT * FROM patrol_interp_edge WHERE interp_key = ? "
        "ORDER BY boundary_idx, from_idx, to_idx",
        (interp_key,),
    ).fetchall():
        edges_by_boundary[int(r["boundary_idx"])].append(
            (int(r["from_idx"]), int(r["to_idx"]))
        )

    return Interpretation(
        band=band,
        catalog_revision=row["catalog_revision"],
        credential_revision=row["credential_revision"],
        steps=steps,
        candidates=candidates,
        forward=forward,
        backward=backward,
        edges=[tuple(e) for e in edges_by_boundary],
        complete=bool(row["complete"]),
        reason=row["reason"],
    )


def load_all_interpretations(conn: sqlite3.Connection) -> dict[str, Interpretation]:
    keys = [
        r["interp_key"]
        for r in conn.execute("SELECT interp_key FROM patrol_interpretation").fetchall()
    ]
    return {k: load_interpretation(conn, k) for k in keys}


def persist_contract(conn: sqlite3.Connection, contract: PatrolContract) -> None:
    with conn:
        existing = conn.execute(
            "SELECT fingerprint FROM patrol_contract WHERE contract_id = ?",
            (contract.contract_id,),
        ).fetchone()
        if existing is not None:
            if existing["fingerprint"] != contract.fingerprint():
                raise ContractConflictError(
                    f"contract_id {contract.contract_id!r} stored with different content"
                )
            return
        conn.execute(
            "INSERT INTO patrol_contract (contract_id, band_id, catalog_revision, "
            "credential_revision, threshold, fingerprint) VALUES (?, ?, ?, ?, ?, ?)",
            (
                contract.contract_id, contract.band_id, contract.catalog_revision,
                contract.credential_revision, contract.threshold,
                contract.fingerprint(),
            ),
        )
        for i, (s, e) in enumerate(contract.zone.intervals):
            conn.execute(
                "INSERT INTO patrol_contract_zone (contract_id, idx, start_num, "
                "start_den, end_num, end_den) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    contract.contract_id, i,
                    str(s.numerator), str(s.denominator),
                    str(e.numerator), str(e.denominator),
                ),
            )


def load_contract(conn: sqlite3.Connection, contract_id: str) -> PatrolContract:
    row = conn.execute(
        "SELECT * FROM patrol_contract WHERE contract_id = ?", (contract_id,)
    ).fetchone()
    if row is None:
        raise PatrolError(f"no contract {contract_id!r}")
    zone_items = [
        {
            "start": Fraction(int(r["start_num"]), int(r["start_den"])),
            "end": Fraction(int(r["end_num"]), int(r["end_den"])),
        }
        for r in conn.execute(
            "SELECT * FROM patrol_contract_zone WHERE contract_id = ? ORDER BY idx",
            (contract_id,),
        ).fetchall()
    ]
    return PatrolContract(
        contract_id=row["contract_id"],
        band_id=row["band_id"],
        catalog_revision=row["catalog_revision"],
        credential_revision=row["credential_revision"],
        zone=Zone.from_list(zone_items),
        threshold=int(row["threshold"]),
    )


def load_all_contracts(conn: sqlite3.Connection) -> dict[str, PatrolContract]:
    ids = [
        r["contract_id"]
        for r in conn.execute("SELECT contract_id FROM patrol_contract").fetchall()
    ]
    return {cid: load_contract(conn, cid) for cid in ids}
