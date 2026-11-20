"""Tests for patrol photo-band interpretation and contracts.

Run:  python3 -m unittest discover -s solution/tests -v
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chainage_catalog import (  # noqa: E402
    CatalogRegistry,
    RouteCatalog,
    Section,
    open_db,
    persist_catalog,
)
from inspection_library import (  # noqa: E402
    Asset,
    InspectionLibrary,
    ensure_inspection_schema,
)
from patrol_interpretation import (  # noqa: E402
    CONFIRMED,
    CONTRADICTION,
    MANUAL_REVIEW,
    NOT_MET,
    Band,
    BandDefinitionError,
    ContractConflictError,
    PatrolContract,
    PatrolError,
    PhotoCandidate,
    StepBound,
    StepCredentialRegistry,
    Zone,
    classify_outcome,
    ensure_patrol_schema,
    interpret_band,
    load_all_contracts,
    load_all_interpretations,
    load_bands,
    load_step_credential_registry,
    persist_band,
    persist_contract,
    persist_interpretation,
    persist_step_credential_revision,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def two_segment_route(revision: str = "d1", s1_base=0, s2_base=0) -> RouteCatalog:
    """Two physical segments [0,10) and [10,20), both labels from a base."""
    return RouteCatalog(
        revision, "D", 20,
        [
            Section("S1", F(0), F(10), F(s1_base)),
            Section("S2", F(10), F(20), F(s2_base)),
        ],
    )


def band_2_4() -> Band:
    return Band("B", "D", (2, 4))


def lib_with_band(route=None) -> InspectionLibrary:
    route = route or two_segment_route()
    reg = CatalogRegistry()
    reg.add(route)
    return InspectionLibrary(reg, [])


# ---------------------------------------------------------------------------
# Step windows and band validation
# ---------------------------------------------------------------------------


class TestStepBound(unittest.TestCase):
    def test_inclusive_bounds(self):
        b = StepBound(F(2), F(4))
        self.assertTrue(b.admits(F(2)))
        self.assertTrue(b.admits(F(4)))
        self.assertTrue(b.admits(F(3)))
        self.assertFalse(b.admits(F(1)))
        self.assertFalse(b.admits(F(5)))

    def test_negative_delta_never_admitted(self):
        # Forward-only: even a wide window rejects backward motion.
        b = StepBound(F(0), F(10))
        self.assertFalse(b.admits(F(-1)))

    def test_zero_lower_bound_allows_standing_still(self):
        b = StepBound(F(0), F(0))
        self.assertTrue(b.admits(F(0)))

    def test_negative_min_rejected(self):
        with self.assertRaises(BandDefinitionError):
            StepBound(F(-1), F(5))

    def test_max_below_min_rejected(self):
        with self.assertRaises(BandDefinitionError):
            StepBound(F(5), F(4))


class TestZoneRegistration(unittest.TestCase):
    def test_overlap_rejected(self):
        with self.assertRaises(BandDefinitionError):
            Zone([(0, 10), (5, 15)])

    def test_touching_is_allowed_and_merged(self):
        z = Zone([(0, 5), (5, 10)])
        self.assertEqual(z.intervals, ((F(0), F(10)),))
        self.assertTrue(z.contains(F(0)))
        self.assertTrue(z.contains(F(9)))
        self.assertFalse(z.contains(F(10)))

    def test_half_open_membership(self):
        z = Zone([(2, 3)])
        self.assertTrue(z.contains(F(2)))
        self.assertFalse(z.contains(F(3)))

    def test_non_positive_interval_rejected(self):
        with self.assertRaises(BandDefinitionError):
            Zone([(5, 5)])

    def test_order_independent_canonical(self):
        self.assertEqual(Zone([(10, 15), (2, 3)]), Zone([(2, 3), (10, 15)]))


# ---------------------------------------------------------------------------
# Example from the task
# ---------------------------------------------------------------------------


class TestWorkedExample(unittest.TestCase):
    def setUp(self):
        self.cat = two_segment_route()
        self.band = band_2_4()

    def test_boundary_inclusive_displacement_2(self):
        interp = interpret_band(self.cat, self.band, (StepBound(F(2), F(2)),), "c1")
        self.assertTrue(interp.complete)
        # two physical readings per photo, ordered by physical
        self.assertEqual(
            [(c.section_id, c.physical) for c in interp.candidates[0]],
            [("S1", F(2)), ("S2", F(12))],
        )
        self.assertEqual(
            [(c.section_id, c.physical) for c in interp.candidates[1]],
            [("S1", F(4)), ("S2", F(14))],
        )
        # exactly two interpretations: (2,4) and (12,14)
        self.assertEqual(sorted(interp.edges[0]), [(0, 0), (1, 1)])

    def test_zone_hits_both_paths_one_each(self):
        interp = interpret_band(self.cat, self.band, (StepBound(F(2), F(2)),), "c1")
        ev = interp.evaluate_zone(Zone([(2, 3), (14, 15)]))
        self.assertEqual((ev.min_hits, ev.max_hits), (1, 1))
        self.assertEqual(
            classify_outcome(ev.complete, ev.min_hits, ev.max_hits, 1), CONFIRMED
        )
        # min trace hits a zone, max trace too
        self.assertTrue(
            any(Zone([(2, 3), (14, 15)]).contains(c.physical) for c in ev.min_trace)
        )

    def test_wide_zone_forces_manual_review(self):
        interp = interpret_band(self.cat, self.band, (StepBound(F(2), F(2)),), "c1")
        ev = interp.evaluate_zone(Zone([(2, 5)]))
        self.assertEqual((ev.min_hits, ev.max_hits), (0, 2))
        self.assertEqual(
            classify_outcome(ev.complete, ev.min_hits, ev.max_hits, 1), MANUAL_REVIEW
        )
        # the min witness avoids the zone, the max witness hits twice
        self.assertEqual([c.physical for c in ev.min_trace], [F(12), F(14)])
        self.assertEqual([c.physical for c in ev.max_trace], [F(2), F(4)])

    def test_witness_traces_are_legal_and_hit_counts_match(self):
        interp = interpret_band(self.cat, self.band, (StepBound(F(2), F(2)),), "c1")
        zone = Zone([(2, 5)])
        ev = interp.evaluate_zone(zone)
        for trace, expected in ((ev.min_trace, ev.min_hits),
                                (ev.max_trace, ev.max_hits)):
            ok, why = interp.verify_trace(trace)
            self.assertTrue(ok, why)
            hits = sum(1 for c in trace if zone.contains(c.physical))
            self.assertEqual(hits, expected)

    def test_exact_step_12_leaves_one_interpretation(self):
        interp = interpret_band(self.cat, self.band, (StepBound(F(12), F(12)),), "c2")
        self.assertTrue(interp.complete)
        self.assertEqual(sorted(interp.edges[0]), [(0, 1)])  # only (2,14)
        ev = interp.evaluate_zone(Zone([(2, 3), (14, 15)]))
        self.assertEqual((ev.min_hits, ev.max_hits), (2, 2))
        self.assertEqual(
            classify_outcome(ev.complete, ev.min_hits, ev.max_hits, 2), CONFIRMED
        )


# ---------------------------------------------------------------------------
# Zero candidates vs. connectivity failures
# ---------------------------------------------------------------------------


class TestDisconnectionKinds(unittest.TestCase):
    def test_zero_candidate_photo_flagged(self):
        # label 99 is not carried by either segment -> no candidate at photo 1
        cat = two_segment_route()
        band = Band("B", "D", (2, 99))
        interp = interpret_band(cat, band, (StepBound(F(0), F(100)),), "c1")
        self.assertFalse(interp.complete)
        self.assertEqual(interp.reason, "no_candidates")
        self.assertEqual(interp.band_view()["photos_without_candidates"], [1])
        self.assertEqual(interp.photo_view(1)["status"], "no_candidates")

    def test_unreachable_head_vs_dead_end(self):
        # photo 0 at label 2 -> physical 2 or 12; photo 1 at 4 -> 4 or 14.
        # window [0,3]: 2->4 ok, 12->14 ok, but 2->14 too far, 12->4 backward.
        cat = two_segment_route()
        band = Band("B", "D", (2, 4))
        interp = interpret_band(cat, band, (StepBound(F(0), F(3)),), "c1")
        self.assertTrue(interp.complete)
        # all four candidates still connect to *something*, but the pair
        # (S1@2 -> S1@4) and (S2@12 -> S2@14) work; cross pairs don't.
        self.assertEqual(sorted(interp.edges[0]), [(0, 0), (1, 1)])

    def test_no_path_between_any_pair(self):
        # window requiring delta in [5,5]: 2->4 no, 12->14 no, 2->14=12 no.
        cat = two_segment_route()
        band = Band("B", "D", (2, 4))
        interp = interpret_band(cat, band, (StepBound(F(5), F(5)),), "c1")
        self.assertFalse(interp.complete)
        self.assertEqual(interp.reason, "disconnected")
        # Candidates exist everywhere, so it is NOT a zero-candidate failure.
        self.assertEqual(interp.band_view()["photos_without_candidates"], [])
        self.assertEqual(
            classify_outcome(interp.complete, None, None, 1), CONTRADICTION
        )

    def test_incomplete_contract_reports_contradiction(self):
        lib = lib_with_band()
        lib.register_band(Band("B", "D", (2, 4)))
        lib.step_registry.add("c1", {"B": [(5, 5)]})
        interp = lib.build_interpretation("B", "d1", "c1")
        contract = PatrolContract(
            "X", "B", "d1", "c1", Zone([(2, 3)]), 1
        )
        lib.register_contract(contract)
        ev = lib.evaluate_contract("X")
        self.assertEqual(ev.outcome, CONTRADICTION)
        self.assertIsNone(ev.min_trace)
        self.assertFalse(ev.complete)


# ---------------------------------------------------------------------------
# Local selection feasibility
# ---------------------------------------------------------------------------


class TestSelectionCheck(unittest.TestCase):
    def setUp(self):
        self.cat = two_segment_route()
        self.interp = interpret_band(
            self.cat, band_2_4(), (StepBound(F(2), F(2)),), "c1"
        )

    def test_valid_pair_splices(self):
        r = self.interp.check_selection(
            {0: ("S1", F(2)), 1: ("S2", F(14))} if False else
            {0: ("S1", 2), 1: ("S1", 4)}
        )
        self.assertTrue(r.ok)

    def test_cross_pair_cannot_connect(self):
        r = self.interp.check_selection({0: ("S2", 12), 1: ("S1", 4)})
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, "cannot_connect")

    def test_non_candidate_rejected(self):
        r = self.interp.check_selection({0: ("S1", 7)})
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, "not_a_candidate")

    def test_out_of_range_rejected(self):
        r = self.interp.check_selection({9: ("S1", 2)})
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, "photo_out_of_range")

    def test_partial_selection(self):
        # naming only photo 0 is enough when it can reach a tail
        r = self.interp.check_selection({0: ("S2", 12)})
        self.assertTrue(r.ok)

    def test_verify_trace_rejects_bad_displacement(self):
        bad = (PhotoCandidate("S1", F(2)), PhotoCandidate("S2", F(14)))
        ok, why = self.interp.verify_trace(bad)
        self.assertFalse(ok)
        self.assertIn("outside", why)

    def test_verify_trace_accepts_legal(self):
        good = (PhotoCandidate("S2", F(12)), PhotoCandidate("S2", F(14)))
        ok, why = self.interp.verify_trace(good)
        self.assertTrue(ok, why)


class TestSinglePhoto(unittest.TestCase):
    def test_one_photo_band(self):
        cat = two_segment_route()
        band = Band("One", "D", (2,))
        interp = interpret_band(cat, band, (), "c1")
        self.assertTrue(interp.complete)
        self.assertEqual(interp.photo_count, 1)
        ev = interp.evaluate_zone(Zone([(2, 3)]))
        self.assertEqual((ev.min_hits, ev.max_hits), (0, 1))
        self.assertEqual(ev.min_trace, (PhotoCandidate("S2", F(12)),))
        self.assertEqual(ev.max_trace, (PhotoCandidate("S1", F(2)),))
        self.assertTrue(interp.check_selection({0: ("S1", 2)}).ok)

    def test_band_step_count_must_match(self):
        cat = two_segment_route()
        with self.assertRaises(BandDefinitionError):
            interpret_band(cat, Band("B", "D", (2, 4)), (), "c1")

    def test_isolated_candidate_unreachable(self):
        # Third segment carries only far labels, so the tail photo has a
        # single high candidate.  One middle candidate can reach it, the other
        # cannot -> a dead end; a candidate with no incoming edge at all is a
        # separate "unreachable from head" case.
        cat = RouteCatalog(
            "d1", "D", 30,
            [Section("S1", F(0), F(10), F(0)),
             Section("S2", F(10), F(20), F(0)),
             Section("S3", F(20), F(30), F(50))],
        )
        # label 2 -> S1@2, S2@12 ; label 4 -> S1@4, S2@14 ; label 52 -> S3@22
        band = Band("B3", "D", (2, 4, 52))
        interp = interpret_band(
            cat, band, (StepBound(F(0), F(3)), StepBound(F(8), F(8))), "c1"
        )
        self.assertTrue(interp.complete)
        pv = interp.photo_view(1)
        by_phys = {c["physical"]: c for c in pv["candidates"]}
        # 14 -> 22 (delta 8) reaches the tail; 4 -> 22 (delta 18) does not.
        self.assertTrue(by_phys["14/1"]["viable"])
        self.assertFalse(by_phys["4/1"]["viable"])
        self.assertFalse(by_phys["4/1"]["backward"])
        self.assertEqual(pv["viable_candidates"], [1])


# ---------------------------------------------------------------------------
# Versioning: late intermediate credential + reopened verification
# ---------------------------------------------------------------------------


class TestLateCredentialRevision(unittest.TestCase):
    def test_revising_intermediate_credential_yields_new_interpretation(self):
        reg = StepCredentialRegistry()
        reg.add("c1", {"B": [(2, 2)]})
        lib = lib_with_band()
        lib.step_registry = reg
        lib.register_band(band_2_4())
        old = lib.build_interpretation("B", "d1", "c1")
        self.assertEqual(sorted(old.edges[0]), [(0, 0), (1, 1)])

        reg.revise_transition("c2", "B", 0, 12, 12)
        new = lib.build_interpretation("B", "d1", "c2")
        self.assertEqual(sorted(new.edges[0]), [(0, 1)])
        # the old relation is untouched and still queryable
        self.assertEqual(sorted(old.edges[0]), [(0, 0), (1, 1)])

    def test_contract_pinned_to_old_credential_stays_valid(self):
        reg = StepCredentialRegistry()
        reg.add("c1", {"B": [(2, 2)]})
        lib = lib_with_band()
        lib.step_registry = reg
        lib.register_band(band_2_4())
        lib.build_interpretation("B", "d1", "c1")
        contract = PatrolContract("C1", "B", "d1", "c1", Zone([(2, 3), (14, 15)]), 1)
        lib.register_contract(contract)

        reg.revise_transition("c2", "B", 0, 12, 12)
        lib.build_interpretation("B", "d1", "c2")
        # Contract still evaluates against c1 and still confirms.
        ev = lib.evaluate_contract("C1")
        self.assertEqual(ev.interpretation_key, "B|d1|c1")
        self.assertEqual(ev.outcome, CONFIRMED)
        view = lib.contract_view("C1")
        self.assertTrue(view["credential_stale"])

    def test_contract_conflict_on_same_id(self):
        lib = lib_with_band()
        lib.register_band(band_2_4())
        lib.step_registry.add("c1", {"B": [(2, 2)]})
        lib.build_interpretation("B", "d1", "c1")
        lib.register_contract(PatrolContract("C", "B", "d1", "c1", Zone([(2, 3)]), 1))
        with self.assertRaises(ContractConflictError):
            lib.register_contract(
                PatrolContract("C", "B", "d1", "c1", Zone([(2, 4)]), 1)
            )

    def test_contract_requires_built_interpretation(self):
        lib = lib_with_band()
        lib.register_band(band_2_4())
        lib.step_registry.add("c1", {"B": [(2, 2)]})
        with self.assertRaises(PatrolError):
            lib.register_contract(
                PatrolContract("C", "B", "d1", "c1", Zone([(2, 3)]), 1)
            )


class TestCatalogRevisionInteraction(unittest.TestCase):
    def test_base_marker_correction_changes_candidates(self):
        band = Band("B", "D", (7, 9))
        cat1 = two_segment_route("d1", s2_base=0)   # label 7 -> S1@7, S2@17
        interp1 = interpret_band(cat1, band, (StepBound(F(2), F(2)),), "c1")
        self.assertEqual(
            [c.physical for c in interp1.candidates[0]], [F(7), F(17)]
        )
        cat2 = two_segment_route("d2", s2_base=5)   # label 7 -> S1@7, S2@12
        interp2 = interpret_band(cat2, band, (StepBound(F(2), F(2)),), "c1")
        self.assertEqual(
            [c.physical for c in interp2.candidates[0]], [F(7), F(12)]
        )
        # The far-segment reading moved; both graphs stay independently
        # explainable, with the same topology but different physicals.
        self.assertNotEqual(
            [c.physical for c in interp1.candidates[1]],
            [c.physical for c in interp2.candidates[1]],
        )

    def test_old_contract_survives_new_base_marker_revision(self):
        # Register a contract against d1, then revise the base marker (new
        # catalogue revision d2).  The contract keeps its pinned d1 photos,
        # version and feasible relation; it is merely marked stale.
        reg = CatalogRegistry()
        reg.add(two_segment_route("d1", s2_base=0))
        lib = InspectionLibrary(reg, [])
        lib.step_registry.add("c1", {"B": [(2, 2)]})
        lib.register_band(Band("B", "D", (2, 4)))
        lib.build_interpretation("B", "d1", "c1")
        lib.register_contract(
            PatrolContract("C1", "B", "d1", "c1", Zone([(2, 3), (14, 15)]), 1)
        )
        before = lib.evaluate_contract("C1")

        # A base-marker correction creates d2 (and would move candidate
        # physicals for the same labels).
        reg.revise_section_label_start("d2", "D", "S2", 5)
        new = lib.build_interpretation("B", "d2", "c1")
        self.assertNotEqual(new.interpretation_key, before.interpretation_key)

        after = lib.evaluate_contract("C1")
        self.assertEqual(after.interpretation_key, "B|d1|c1")
        self.assertEqual((after.min_hits, after.max_hits),
                         (before.min_hits, before.max_hits))
        self.assertEqual(after.min_trace, before.min_trace)
        view = lib.contract_view("C1")
        self.assertTrue(view["catalog_stale"])
        self.assertEqual(view["catalog_revision"], "d1")


# ---------------------------------------------------------------------------
# Persistence + reopen
# ---------------------------------------------------------------------------


class TestPatrolPersistence(unittest.TestCase):
    def setUp(self):
        self.conn = open_db(":memory:")
        ensure_inspection_schema(self.conn)
        ensure_patrol_schema(self.conn)

    def tearDown(self):
        self.conn.close()

    def test_band_and_credential_roundtrip(self):
        persist_band(self.conn, band_2_4())
        persist_step_credential_revision(self.conn, "c1", {"B": [(2, 2)]})
        bands = load_bands(self.conn)
        self.assertEqual(bands["B"], band_2_4())
        reg = load_step_credential_registry(self.conn)
        self.assertEqual(reg.current(), "c1")
        b = reg.bounds_for("c1", "B")[0]
        self.assertEqual((b.min_step, b.max_step), (F(2), F(2)))

    def test_interpretation_roundtrip(self):
        cat = two_segment_route()
        interp = interpret_band(cat, band_2_4(), (StepBound(F(2), F(2)),), "c1")
        persist_band(self.conn, interp.band)
        persist_interpretation(self.conn, interp)
        back = load_all_interpretations(self.conn)[interp.interpretation_key]
        self.assertEqual(back.complete, interp.complete)
        self.assertEqual(
            [[(c.section_id, c.physical) for c in layer] for layer in back.candidates],
            [[(c.section_id, c.physical) for c in layer] for layer in interp.candidates],
        )
        self.assertEqual(back.edges, interp.edges)
        self.assertEqual(
            [sorted(x) for x in back.forward], [sorted(x) for x in interp.forward]
        )
        # zone evaluation equals the freshly-built one
        z = Zone([(2, 3), (14, 15)])
        self.assertEqual(
            (back.evaluate_zone(z).min_hits, back.evaluate_zone(z).max_hits),
            (interp.evaluate_zone(z).min_hits, interp.evaluate_zone(z).max_hits),
        )

    def test_contract_roundtrip(self):
        contract = PatrolContract("C1", "B", "d1", "c1", Zone([(2, 3), (14, 15)]), 1)
        persist_contract(self.conn, contract)
        back = load_all_contracts(self.conn)["C1"]
        self.assertEqual(back.fingerprint(), contract.fingerprint())
        self.assertEqual(back.zone, contract.zone)
        self.assertEqual(back.interpretation_key, "B|d1|c1")

    def test_contract_conflict_in_sqlite(self):
        persist_contract(self.conn, PatrolContract("C", "B", "d1", "c1", Zone([(2, 3)]), 1))
        with self.assertRaises(ContractConflictError):
            persist_contract(
                self.conn, PatrolContract("C", "B", "d1", "c1", Zone([(2, 4)]), 1)
            )

    def test_reopen_file_db_verifies_contract(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "patrol.db")
            conn = open_db(path)
            ensure_inspection_schema(conn)
            ensure_patrol_schema(conn)
            cat = two_segment_route()
            persist_catalog(conn, cat)
            persist_band(conn, band_2_4())
            persist_step_credential_revision(conn, "c1", {"B": [(2, 2)]})
            interp = interpret_band(cat, band_2_4(), (StepBound(F(2), F(2)),), "c1")
            persist_interpretation(conn, interp)
            contract = PatrolContract(
                "C1", "B", "d1", "c1", Zone([(2, 3), (14, 15)]), 1
            )
            persist_contract(conn, contract)
            conn.close()

            # reopen and evaluate purely from storage
            conn2 = open_db(path)
            ensure_inspection_schema(conn2)
            ensure_patrol_schema(conn2)
            reg = CatalogRegistry()
            reg.add(cat)
            step_reg = load_step_credential_registry(conn2)
            lib = InspectionLibrary(reg, [], step_registry=step_reg)
            for b in load_bands(conn2).values():
                lib.register_band(b)
            for k, it in load_all_interpretations(conn2).items():
                lib._interpretations[k] = it
            for c in load_all_contracts(conn2).values():
                lib._contracts[c.contract_id] = c
            ev = lib.evaluate_contract("C1")
            self.assertEqual(ev.outcome, CONFIRMED)
            self.assertEqual(ev.min_hits, 1)
            conn2.close()


# ---------------------------------------------------------------------------
# Independent contracts read the same interpretation
# ---------------------------------------------------------------------------


class TestIndependentContracts(unittest.TestCase):
    def test_two_contracts_share_interpretation_different_zones(self):
        lib = lib_with_band()
        lib.register_band(band_2_4())
        lib.step_registry.add("c1", {"B": [(2, 2)]})
        interp = lib.build_interpretation("B", "d1", "c1")
        lib.register_contract(PatrolContract("A", "B", "d1", "c1", Zone([(2, 3), (14, 15)]), 1))
        lib.register_contract(PatrolContract("W", "B", "d1", "c1", Zone([(2, 5)]), 1))
        ea = lib.evaluate_contract("A")
        ew = lib.evaluate_contract("W")
        self.assertEqual((ea.min_hits, ea.max_hits), (1, 1))
        self.assertEqual((ew.min_hits, ew.max_hits), (0, 2))
        # same feasible relation, different zones -> independent reads
        self.assertEqual(ea.interpretation_key, ew.interpretation_key)
        self.assertIs(lib.get_interpretation(interp.interpretation_key), interp)

    def test_single_photo_region_does_not_imply_band_association(self):
        # Every photo individually *could* be in [2,3) OR [14,15) under c1,
        # but no complete path puts both photos in the same segment; a contract
        # counts per-photo lands, and min/max reflect whole-band correlation.
        lib = lib_with_band()
        lib.register_band(band_2_4())
        lib.step_registry.add("c1", {"B": [(2, 2)]})
        lib.build_interpretation("B", "d1", "c1")
        # zone [12,20) (the far segment): both photos or neither, never one.
        lib.register_contract(PatrolContract("Z", "B", "d1", "c1", Zone([(12, 20)]), 1))
        ev = lib.evaluate_contract("Z")
        self.assertEqual((ev.min_hits, ev.max_hits), (0, 2))


if __name__ == "__main__":
    unittest.main()
