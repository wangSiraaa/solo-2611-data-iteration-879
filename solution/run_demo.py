"""CLI / orchestrator for the independent chainage catalog + inspection facts.

Reads the materials in ``materials/730`` (route, assets, inspections,
revision, operations), applies operations in order, persists everything in
SQLite (two separate schemas: catalogue vs. inspection facts), and prints
the P2 revision's real facility changes.

Usage:
    python3 solution/run_demo.py [--materials DIR] [--db PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from fractions import Fraction

from chainage_catalog import (
    CatalogRegistry,
    RouteCatalog,
    load_catalog_registry,
    open_db,
    persist_catalog,
)
from inspection_library import (
    Asset,
    InspectionLibrary,
    ensure_inspection_schema,
    load_all_reports,
    load_assets,
    persist_assets,
    persist_report,
)
from patrol_interpretation import (
    Band,
    PatrolContract,
    StepBound,
    Zone,
    ensure_patrol_schema,
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
# Material loading
# ---------------------------------------------------------------------------


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_jsonl(path: str) -> list[dict]:
    ops = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                ops.append(json.loads(line))
    return ops


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _frac(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _fmt_interval(iv) -> str:
    return f"[{_frac(iv.start)}, {_frac(iv.end)})"


def _trace_str(trace) -> str:
    return " -> ".join(
        f"{c['section_id']}@{_dbfrac(c['physical'])}" for c in trace
    )


def _dbfrac(text: str) -> str:
    """Render a stored 'n/d' string compactly (drops '/1')."""
    return _frac(Fraction(text))


def print_point_query(catalog: RouteCatalog, label) -> None:
    result = catalog.resolve_label_point(label)
    if result["status"] == "resolved":
        c = result["candidates"][0]
        print(
            f"  point label {_frac(result['label'])}: resolved -> "
            f"{c['section_id']} @ physical {_frac(c['physical'])}"
        )
    elif result["status"] == "ambiguous":
        parts = ", ".join(
            f"{c['section_id']} @ physical {_frac(c['physical'])}"
            for c in result["candidates"]
        )
        print(
            f"  point label {_frac(result['label'])}: AMBIGUOUS "
            f"({len(result['candidates'])} candidates): {parts}"
        )
    else:
        print(
            f"  point label {_frac(result['label'])}: UNRESOLVED "
            f"(no section carries this label)"
        )


def print_report(lib: InspectionLibrary, report_id: str) -> None:
    view = lib.report_view(report_id)
    print(f"  report {report_id}  (revision {view['catalog_revision']}, "
          f"job {view['job_id']})")
    print(f"    requested spans (labels -> selected section -> physical):")
    for s in view["requested_spans"]:
        print(
            f"      {s['section_id']} label [{_dbfrac(s['label_start'])}, "
            f"{_dbfrac(s['label_end'])}) -> physical "
            f"[{_dbfrac(s['physical_start'])}, {_dbfrac(s['physical_end'])})"
        )
    cov = ", ".join(
        f"[{_dbfrac(iv['start'])}, {_dbfrac(iv['end'])})"
        for iv in view["merged_coverage"]
    )
    print(f"    merged coverage: {cov}  (length {view['covered_length']})")
    facs = ", ".join(
        f"{f['asset_id']}@{_dbfrac(f['physical_position'])}" for f in view["facilities"]
    ) or "(none)"
    print(f"    facilities covered: {facs}")
    print(f"    stale={view['stale']} (current revision {view['current_revision']})")


# ---------------------------------------------------------------------------
# Operation application
# ---------------------------------------------------------------------------


class Application:
    """Holds in-memory state mirrored to SQLite."""

    def __init__(
        self,
        registry: CatalogRegistry,
        lib: InspectionLibrary,
        conn: sqlite3.Connection,
        materials_dir: str,
    ):
        self.registry = registry
        self.lib = lib
        self.conn = conn
        self.materials_dir = materials_dir
        self._persisted_revisions = {
            r["revision_id"]
            for r in conn.execute("SELECT revision_id FROM route_revision")
        }
        self._persisted_reports = set(
            r["report_id"] for r in conn.execute("SELECT report_id FROM report")
        )

    def _path(self, name: str) -> str:
        return os.path.join(self.materials_dir, name)

    def _persist_new_revision(self) -> None:
        for cat in self.registry.all_revisions():
            if cat.revision_id not in self._persisted_revisions:
                persist_catalog(self.conn, cat)
                self._persisted_revisions.add(cat.revision_id)

    def _persist_new_report(self, report) -> None:
        if report.report_id not in self._persisted_reports:
            persist_report(self.conn, report)
            self._persisted_reports.add(report.report_id)

    def _op_done(self, op: dict) -> bool:
        """True if this exact op was already applied (durable resume)."""
        h = hashlib.sha256(
            json.dumps(op, sort_keys=True).encode("utf-8")
        ).hexdigest()
        row = self.conn.execute(
            "SELECT 1 FROM applied_op WHERE op_hash = ?", (h,)
        ).fetchone()
        return row is not None

    def _mark_op_done(self, op: dict) -> None:
        h = hashlib.sha256(
            json.dumps(op, sort_keys=True).encode("utf-8")
        ).hexdigest()
        with self.conn:
            conn_row = self.conn.execute(
                "SELECT COALESCE(MAX(op_seq), 0) + 1 FROM applied_op"
            ).fetchone()[0]
            self.conn.execute(
                "INSERT INTO applied_op (op_seq, op_hash) VALUES (?, ?)",
                (conn_row, h),
            )

    # Ops that mutate durable state; these are logged for resume. Read-only
    # ops are always executed so a reopened DB can still render them.
    MUTATING = {
        "resolve_and_inspect",
        "revise_chainage",
        "define_band",
        "define_step_credentials",
        "revise_step_credential",
        "open_contract",
    }

    def apply(self, op: dict) -> None:
        kind = op.get("op")
        if kind in self.MUTATING and self._op_done(op):
            print(f"\n(skipping already-applied op: {kind})")
            return
        if kind == "resolve_and_inspect":
            self._op_resolve_and_inspect(op)
        elif kind == "revise_chainage":
            self._op_revise_chainage(op)
        elif kind == "read_report":
            self._op_read_report(op)
        elif kind == "resolve_points":
            self._op_resolve_points(op)
        elif kind == "define_band":
            self._op_define_band(op)
        elif kind == "define_step_credentials":
            self._op_define_step_credentials(op)
        elif kind == "revise_step_credential":
            self._op_revise_step_credential(op)
        elif kind == "open_contract":
            self._op_open_contract(op)
        elif kind == "evaluate_contract":
            self._op_evaluate_contract(op)
        elif kind == "query_band":
            self._op_query_band(op)
        else:
            raise ValueError(f"unknown operation {kind!r}")
        if kind in self.MUTATING:
            self._mark_op_done(op)

    # -- patrol operations -------------------------------------------
    def _interpret(self, band_id: str, cat_rev, cred_rev):
        interp = self.lib.build_interpretation(band_id, cat_rev, cred_rev)
        persist_interpretation(self.conn, interp)
        return interp

    def _op_define_band(self, op: dict) -> None:
        payload = _read_json(self._path(op["file"]))
        band = Band.from_dict(payload)
        print(f"\n[define_band] {band.band_id}: {band.photo_count} photos "
              f"labels {[_frac(x) for x in band.labels]}")
        self.lib.register_band(band)
        persist_band(self.conn, band)

    def _op_define_step_credentials(self, op: dict) -> None:
        payload = _read_json(self._path(op["file"]))
        revision_id = str(payload["revision"])
        by_band = {
            str(band_id): [tuple(pair) for pair in pairs]
            for band_id, pairs in payload["bands"].items()
        }
        print(f"\n[define_step_credentials] revision {revision_id}")
        self.lib.step_registry.add(revision_id, by_band)
        persist_step_credential_revision(self.conn, revision_id, by_band)
        for band_id, bounds in by_band.items():
            shown = ", ".join(f"[{_frac(Fraction(lo))}, {_frac(Fraction(hi))}]"
                              for lo, hi in bounds)
            print(f"  {band_id}: step windows {shown}")

    def _op_revise_step_credential(self, op: dict) -> None:
        payload = _read_json(self._path(op["file"]))
        revision_id = str(payload["revision"])
        band_id = str(payload["band_id"])
        index = int(payload["index"])
        print(f"\n[revise_step_credential] revision {revision_id} "
              f"{band_id} transition {index} -> "
              f"[{payload['min_step']}, {payload['max_step']}]")
        base = str(payload.get("base_revision") or self.lib.step_registry.current())
        self.lib.step_registry.revise_transition(
            revision_id, band_id, index,
            payload["min_step"], payload["max_step"], base_revision=base,
        )
        persist_step_credential_revision(
            self.conn, revision_id,
            self.lib.step_registry.get(revision_id),
        )
        print(f"  now {len(self.lib.step_registry.all_revisions())} step "
              f"credential revisions; current is {revision_id}")

    def _op_open_contract(self, op: dict) -> None:
        payload = _read_json(self._path(op["file"]))
        contract = self._contract_from_payload(payload)
        interp = self._interpret(
            contract.band_id, contract.catalog_revision,
            contract.credential_revision,
        )
        if not interp.complete:
            print(f"\n[open_contract] {contract.contract_id}: interpretation "
                  f"{interp.interpretation_key} has no complete trace "
                  f"({interp.reason}) -- still registered for auditing")
        else:
            print(f"\n[open_contract] {contract.contract_id} band {contract.band_id} "
                  f"@ catalog {contract.catalog_revision} / credential "
                  f"{contract.credential_revision}, threshold {contract.threshold}")
        self.lib.register_contract(contract)
        persist_contract(self.conn, contract)
        self._print_contract(contract.contract_id)

    def _contract_from_payload(self, payload: dict) -> PatrolContract:
        band_id = str(payload["band_id"])
        cat_rev = str(payload.get("catalog_revision")
                      or self.registry.current().revision_id)
        cred_rev = str(payload.get("credential_revision")
                       or self.lib.step_registry.current())
        zone = Zone([(iv["start"], iv["end"]) for iv in payload["zone"]])
        return PatrolContract(
            contract_id=str(payload["contract_id"]),
            band_id=band_id,
            catalog_revision=cat_rev,
            credential_revision=cred_rev,
            zone=zone,
            threshold=int(payload["threshold"]),
        )

    def _op_evaluate_contract(self, op: dict) -> None:
        contract_id = str(op["contract_id"])
        print(f"\n[evaluate_contract] {contract_id}")
        self._print_contract(contract_id)

    def _op_query_band(self, op: dict) -> None:
        band_id = str(op["band_id"])
        interp = self._interpret(
            band_id, op.get("catalog_revision"), op.get("credential_revision")
        )
        print(f"\n[query_band] {band_id} @ {interp.interpretation_key} "
              f"complete={interp.complete}")
        for i in range(interp.photo_count):
            view = interp.photo_view(i)
            viable = view["viable_candidates"]
            print(f"  photo {i} label {_dbfrac(view['label'])}: "
                  f"{view['status']} ({len(viable)}/{len(view['candidates'])} viable)")
        for group in op.get("selections", []):
            choice = {int(item["photo"]): tuple(item["candidate"])
                      for item in group}
            result = interp.check_selection(choice).to_dict()
            shown = ", ".join(
                f"p{int(item['photo'])}={item['candidate']}" for item in group
            )
            print(f"  splice [{shown}]: ok={result['ok']} "
                  f"reason={result['reason']}")

    def _print_contract(self, contract_id: str) -> None:
        view = self.lib.contract_view(contract_id)
        zone = ", ".join(
            f"[{_dbfrac(iv['start'])}, {_dbfrac(iv['end'])})"
            for iv in self.lib.get_contract(contract_id).zone.to_list()
        )
        print(f"  zone: {zone}   threshold: {view['threshold']}")
        if not view["complete"]:
            print("  outcome: CONTRADICTION (no complete trace)")
        else:
            print(f"  min_hits={view['min_hits']} max_hits={view['max_hits']} "
                  f"-> {view['outcome']}")
            min_trace = view["min_trace"]
            max_trace = view["max_trace"]
            if min_trace is not None:
                print("  min trace: " + _trace_str(min_trace))
            if max_trace is not None and max_trace != min_trace:
                print("  max trace: " + _trace_str(max_trace))
        print(f"  pinned catalog={view['catalog_revision']} "
              f"(stale={view['catalog_stale']}), credential="
              f"{view['credential_revision']} (stale={view['credential_stale']})")

    # -- operations ---------------------------------------------------
    def _op_resolve_and_inspect(self, op: dict) -> None:
        payload = _read_json(self._path(op["file"]))
        report_id = str(op["report_id"])
        current = self.registry.current()
        print(f"\n[resolve_and_inspect] report_id={report_id} "
              f"(revision {current.revision_id})")

        # Point queries are always shown; they help expose ambiguity.
        for label in payload.get("point_queries", []):
            print_point_query(current, label)

        for job in payload.get("jobs", []):
            try:
                report = self.lib.resolve_and_inspect(job, report_id)
            except Exception as exc:  # invalid span aborts whole job
                print(f"  job {job.get('job_id')} NOT PUBLISHED: {exc}")
                continue
            self._persist_new_report(report)
            print_report(self.lib, report_id)

    def _op_revise_chainage(self, op: dict) -> None:
        payload = _read_json(self._path(op["file"]))
        print(f"\n[revise_chainage] {payload['route_id']} "
              f"{payload['section_id']} label_start -> {payload['label_start']} "
              f"(revision {payload['revision']})")
        cat = self.registry.revise_section_label_start(
            revision_id=str(payload["revision"]),
            route_id=str(payload["route_id"]),
            section_id=str(payload["section_id"]),
            label_start=payload["label_start"],
        )
        self._persist_new_revision()
        for s in cat.sections:
            print(
                f"  {s.section_id}: physical [{_frac(s.physical_start)}, "
                f"{_frac(s.physical_end)}) labels [{_frac(s.label_at(s.physical_start))}, "
                f"{_frac(s.label_end)})"
            )
        affected = self.lib.reports_affected_by(cat.revision_id)
        if affected:
            print(f"  reports now stale (kept as-is): {', '.join(affected)}")

    def _op_read_report(self, op: dict) -> None:
        report_id = str(op["report_id"])
        print(f"\n[read_report] {report_id}")
        # Prefer in-memory; fall back to SQLite for durability across runs.
        if report_id not in self.lib._reports:
            from inspection_library import load_report

            try:
                rep = load_report(self.conn, report_id)
            except KeyError:
                print(f"  no such report {report_id}")
                return
            # Register for consistent rendering without re-publishing.
            self.lib._reports[report_id] = rep
        print_report(self.lib, report_id)

    def _op_resolve_points(self, op: dict) -> None:
        payload = _read_json(self._path(op["file"]))
        cat = self.registry.current()
        print(f"\n[resolve_points] revision {cat.revision_id}")
        for label in payload.get("point_queries", []):
            print_point_query(cat, label)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap(materials_dir: str, db_path: str):
    conn = open_db(db_path)
    ensure_inspection_schema(conn)
    ensure_patrol_schema(conn)

    route = _read_json(os.path.join(materials_dir, "route.json"))
    catalog = RouteCatalog.from_dict(route)
    persist_catalog(conn, catalog)

    registry = load_catalog_registry(conn)
    if not registry.all_revisions():
        registry.add(catalog)

    assets_data = _read_json(os.path.join(materials_dir, "assets.json"))
    assets = [Asset.from_dict(a) for a in assets_data["assets"]]
    persist_assets(conn, assets)
    stored_assets = load_assets(conn)
    if not stored_assets:
        stored_assets = assets

    step_registry = load_step_credential_registry(conn)
    lib = InspectionLibrary(registry, stored_assets, step_registry=step_registry)
    # Reload any previously persisted reports (durability across runs).
    for rid, rep in load_all_reports(conn).items():
        lib._reports[rid] = rep
    # Reload patrol material so a reopened DB can answer without replaying.
    for band in load_bands(conn).values():
        lib._bands[band.band_id] = band
    for key, interp in load_all_interpretations(conn).items():
        lib._interpretations[key] = interp
    for contract in load_all_contracts(conn).values():
        lib._contracts[contract.contract_id] = contract

    return conn, registry, lib


def run(materials_dir: str, db_path: str) -> None:
    conn, registry, lib = bootstrap(materials_dir, db_path)
    app = Application(registry, lib, conn, materials_dir)
    ops = _read_jsonl(os.path.join(materials_dir, "operations.jsonl"))

    print("=" * 78)
    print("Route chainage catalog + inspection coverage facts")
    print("=" * 78)
    cat = registry.current()
    print(f"\nInitial revision {cat.revision_id} of route {cat.route_id} "
          f"(length {_frac(cat.length)}):")
    for s in cat.sections:
        print(
            f"  {s.section_id}: physical [{_frac(s.physical_start)}, "
            f"{_frac(s.physical_end)}) labels [{_frac(s.label_at(s.physical_start))}, "
            f"{_frac(s.label_end)})"
        )

    for op in ops:
        app.apply(op)

    if lib._reports:
        print("\n" + "=" * 78)
        print("Summary: P2 base-label correction and real facility change")
        print("=" * 78)
        _summarize_revision_effect(lib, registry)
    if lib._contracts:
        print("\n" + "=" * 78)
        print("Summary: patrol contracts")
        print("=" * 78)
        _summarize_contracts(lib)


def _summarize_revision_effect(lib: InspectionLibrary, registry: CatalogRegistry) -> None:
    # Compare reports across revisions by job.
    reports = sorted(
        lib._reports.values(), key=lambda r: (r.job_id, r.catalog_revision, r.report_id)
    )
    by_job: dict[str, list] = {}
    for rep in reports:
        by_job.setdefault(rep.job_id, []).append(rep)
    for job_id, reps in by_job.items():
        print(f"\n  job {job_id}:")
        for rep in reps:
            facs = ", ".join(f.asset_id for f in rep.facilities) or "(none)"
            cov = ", ".join(_fmt_interval(iv) for iv in rep.merged)
            labels = ", ".join(
                f"{s.section_id} [{_frac(s.label_start)}, {_frac(s.label_end)})"
                for s in rep.requested_spans
            )
            print(
                f"    {rep.report_id} @ {rep.catalog_revision}: labels {labels} "
                f"-> coverage {cov} -> facilities [{facs}]"
            )


def _summarize_contracts(lib: InspectionLibrary) -> None:
    for contract_id in sorted(lib._contracts):
        view = lib.contract_view(contract_id)
        if not view["complete"]:
            result = "CONTRADICTION (no complete trace)"
        else:
            result = (
                f"min={view['min_hits']} max={view['max_hits']} "
                f"threshold={view['threshold']} -> {view['outcome']}"
            )
        print(
            f"  {contract_id} band {view['band_id']} @ "
            f"{view['catalog_revision']}/{view['credential_revision']}: {result}"
        )
        if view["min_trace"] is not None:
            print(f"    min path: {_trace_str(view['min_trace'])}")
        if view["max_trace"] is not None:
            print(f"    max path: {_trace_str(view['max_trace'])}")


def main(argv=None) -> int:
    default_materials = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "materials", "730"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", default=default_materials)
    parser.add_argument("--db", default=":memory:")
    args = parser.parse_args(argv)
    run(args.materials, args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
