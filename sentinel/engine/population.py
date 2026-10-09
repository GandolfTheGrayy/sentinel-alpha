"""Variant population: seeding, loading from the DB and instantiating strategies."""
from __future__ import annotations

from typing import Any

from sentinel.config import Settings
from sentinel.store.db import Database, now_iso
from sentinel.strategies import REGISTRY, Strategy, get_family, load_evolved
from sentinel.strategies.seeds import SEEDS


def load_evolved_families(settings: Settings, log=print) -> list[str]:
    return load_evolved(settings.lab_dir.parent / "sentinel" / "strategies" / "evolved", log=log)


def seed_population(db: Database, settings: Settings, log=print) -> int:
    """Insert the seed variants if the population is empty. Returns the number created."""
    if db.variants():
        return 0
    n = 0
    non_control = [s for s in SEEDS if not s.get("is_control")]
    control = [s for s in SEEDS if s.get("is_control")]
    base_w = (1.0 - settings.population.control_weight * len(control)) / max(1, len(non_control))
    base_w = min(base_w, settings.population.allocation_cap)
    for s in SEEDS:
        cls = get_family(s["family"])
        params = cls.validate_params({**cls.DEFAULTS, **s["params"]})
        vid = db.next_variant_id(s["family"])
        db.insert_variant({
            "id": vid, "family": s["family"], "name": cls.display_name(params), "params": params,
            "status": "active", "origin": "seed", "markets": s.get("markets", cls.markets), "timeframe": params.get("tf", cls.timeframe),
            "is_control": bool(s.get("is_control")), "allocation": settings.population.control_weight if s.get("is_control") else base_w,
            "notes": cls.description, "created_at": now_iso(),
        })
        n += 1
    log(f"seeded {n} variants")
    db.add_event("tournament", f"seeded population with {n} variants", "info")
    return n


def create_variant(db: Database, settings: Settings, family: str, params: dict[str, Any], *, origin: str, parent_id: str | None = None,
                   markets: list[str] | None = None, status: str = "incubating", notes: str | None = None, allocation: float | None = None) -> dict[str, Any]:
    cls = get_family(family)
    params = cls.validate_params({**cls.DEFAULTS, **params})
    vid = db.next_variant_id(family)
    if allocation is None:
        allocation = settings.population.exploration_floor if status == "incubating" else 0.05
    db.insert_variant({
        "id": vid, "family": family, "name": cls.display_name(params), "params": params, "status": status, "origin": origin, "parent_id": parent_id,
        "markets": markets or cls.markets, "timeframe": str(params.get("tf", cls.timeframe)), "is_control": cls.is_control, "allocation": allocation,
        "notes": notes or cls.description, "created_at": now_iso(),
    })
    return db.variant(vid)


def instantiate(variants: list[dict[str, Any]], log=print) -> dict[str, Strategy]:
    out: dict[str, Strategy] = {}
    for v in variants:
        if v["family"] not in REGISTRY:
            log(f"variant {v['id']}: family {v['family']} not available (evolved module missing?)")
            continue
        try:
            out[v["id"]] = REGISTRY[v["family"]]({**v["params"], "markets": v.get("markets") or REGISTRY[v["family"]].markets})
        except Exception as exc:  # noqa: BLE001
            log(f"variant {v['id']} failed to instantiate: {exc}")
    return out


def population_counts(db: Database) -> dict[str, int]:
    counts = {"active": 0, "incubating": 0, "probation": 0, "paused": 0, "retired": 0}
    for row in db.query("SELECT status, COUNT(*) AS n FROM variants GROUP BY status"):
        counts[row["status"]] = int(row["n"])
    return counts
