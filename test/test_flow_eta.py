# -*- coding: utf-8 -*-
"""主流程预计剩余时间：连拍/识别张数与速度分开估算，并在运行中校正。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flow_eta import (  # noqa: E402
    SEC_SPECIES_LOCAL,
    FlowEtaEstimator,
    blend_rate,
    build_eta_phase_estimates,
    species_count_expected,
)


def _cfg(**kwargs):
    d = {
        "enable_burst_detection": True,
        "enable_species_detection": True,
        "use_fast_mode": True,
        "use_bird_detection": True,
        "use_local_model": True,
        "burst_keep_ratio": 0.1,
        "burst_keep_min": 1,
        "enable_gps_write": False,
        "enable_watermark_generation": False,
    }
    d.update(kwargs)
    return d


def _est_from_cfg(config, n):
    phases, meta = build_eta_phase_estimates(config, n)
    return FlowEtaEstimator.from_start(
        [{"name": a, "est": b} for a, b in phases],
        n_images=n,
        n_species_expected=int(meta["n_species_expected"]),
        burst_sec_per=meta.get("burst_sec_per"),
        species_sec_per=meta.get("species_sec_per"),
        keep_ratio=float(meta.get("burst_keep_ratio") or 0.1),
        keep_min=int(meta.get("burst_keep_min") or 1),
    )


def test_species_count_uses_keep_ratio_not_full_folder():
    n = 1000
    assert species_count_expected(_cfg(burst_keep_ratio=0.1, burst_keep_min=1), n) == 100
    phases, meta = build_eta_phase_estimates(_cfg(), n)
    by_name = dict(phases)
    assert meta["n_species_expected"] == 100
    assert by_name["species"] == max(8.0, 100 * SEC_SPECIES_LOCAL)
    assert by_name["species"] < n * 2.0
    assert by_name["burst"] <= n * 0.55 + 1e-6


def test_old_priors_were_several_times_larger():
    n = 1000
    phases, _ = build_eta_phase_estimates(_cfg(), n)
    total = sum(e for _, e in phases)
    old = n * 2.8 + n * 5.0
    assert total * 3 < old


def test_burst_result_revises_future_species_and_not_full_n():
    est = _est_from_cfg(_cfg(), 1000)
    before = next(p["est"] for p in est.phases if p["name"] == "species")
    est.burst_result(1000, 80)
    after = next(p["est"] for p in est.phases if p["name"] == "species")
    assert after == max(6.0, 80 * SEC_SPECIES_LOCAL)
    assert after < before
    assert after < 1000 * 1.0


def test_species_begin_does_not_reset_to_five_seconds_each():
    est = _est_from_cfg(_cfg(), 1000)
    est.burst_result(1000, 80)
    est.species_begin(80)
    sp = next(p["est"] for p in est.phases if p["name"] == "species")
    assert sp == max(5.0, 80 * SEC_SPECIES_LOCAL)
    assert sp < 80 * 4.0


def test_slow_first_item_does_not_explode_remaining():
    est = _est_from_cfg(_cfg(), 1000)
    t0 = 1_000.0
    est.phase_begin("burst", now=t0)
    est.phase_tick("burst", 1, 1000, now=t0 + 25.0)
    rem = est.remaining_sec(now=t0 + 25.0)
    # 旧算法会把 25 秒/张外推到剩余 999 张（约 7 小时）再加全量识别。
    assert rem < 30 * 60
    burst_left = 999 * 0.50
    species = 100 * SEC_SPECIES_LOCAL
    assert abs(rem - (burst_left + species)) < 5.0


def test_steady_burst_rate_excludes_startup_and_keeps_species_prior():
    est = _est_from_cfg(_cfg(), 1000)
    t0 = 0.0
    est.phase_begin("burst", now=t0)
    est.phase_tick("burst", 1, 1000, now=t0 + 20.0)
    # 随后 99 张按 0.20 秒/张
    est.phase_tick("burst", 100, 1000, now=t0 + 20.0 + 99 * 0.20)
    rem = est.remaining_sec(now=t0 + 20.0 + 99 * 0.20)
    species = next(p["est"] for p in est.phases if p["name"] == "species")
    assert abs(species - 100 * SEC_SPECIES_LOCAL) < 1e-6
    # 连拍剩余应接近 900*0.20，而不是把识别也按 0.20 秒去乘。
    assert 100 < rem - species < 400
    assert rem > species


def test_blend_rate_early_samples_stay_near_prior():
    prior = 0.50
    measured = 8.0
    blended = blend_rate(measured, prior, 1, phase="burst")
    assert abs(blended - prior) < abs(blended - measured)
    later = blend_rate(0.20, prior, 80, phase="burst")
    assert later < 0.40


def test_learned_blends_into_next_run_prior():
    cfg = _cfg(_eta_learned={"burst_fast": 0.20, "species_local": 0.90})
    phases, meta = build_eta_phase_estimates(cfg, 100)
    assert meta["burst_sec_per"] < 0.50
    assert meta["species_sec_per"] < SEC_SPECIES_LOCAL
    by_name = dict(phases)
    assert by_name["burst"] < 100 * 0.50
