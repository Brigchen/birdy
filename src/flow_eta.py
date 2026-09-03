# -*- coding: utf-8 -*-
"""主流程预计剩余时间：分阶段先验 + 实测速度收缩，识别张数用筛选后的量。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

# 先验：秒/张。旧值连拍 2.8、本地识别 5、云端 14，整体偏慢数倍。
SEC_BURST_FAST_BIRD = 0.50
SEC_BURST_FULL_BIRD = 1.50
SEC_BURST_NO_BIRD = 0.12
SEC_SPECIES_LOCAL = 1.50
SEC_SPECIES_CLOUD = 6.50
SEC_GPS = 0.05
SEC_WATERMARK = 0.40

# 早期样本少时，把实测往先验拉；样本多了以实测为主。
PSEUDO_COUNT = {
    "burst": 16,
    "species": 5,
    "watermark": 6,
    "gps": 8,
}

RATE_CLIP = {
    "burst": (0.04, 8.0),
    "species": (0.15, 45.0),
    "watermark": (0.05, 10.0),
    "gps": (0.01, 3.0),
}


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def expected_kept_count(n: int, ratio: float, min_keep: int) -> int:
    if n <= 0:
        return 0
    r = max(0.01, min(1.0, float(ratio)))
    mk = max(1, int(min_keep))
    by_ratio = max(1, int(round(n * r)))
    return min(n, max(mk, by_ratio))


def burst_sec_per_image(config: Dict[str, Any]) -> float:
    learned = (config.get("_eta_learned") or {}) if isinstance(config, dict) else {}
    fast = bool(config.get("use_fast_mode", True))
    bird = bool(config.get("use_bird_detection", True))
    key = "burst_fast" if fast else "burst_full"
    if not bird:
        key = "burst_no_bird"
        default = SEC_BURST_NO_BIRD
    else:
        default = SEC_BURST_FAST_BIRD if fast else SEC_BURST_FULL_BIRD
    prev = learned.get(key)
    try:
        if prev is not None and float(prev) > 0:
            return 0.55 * default + 0.45 * float(prev)
    except (TypeError, ValueError):
        pass
    return default


def species_sec_per_image(config: Dict[str, Any]) -> float:
    learned = (config.get("_eta_learned") or {}) if isinstance(config, dict) else {}
    local = bool(config.get("use_local_model", True))
    key = "species_local" if local else "species_cloud"
    default = SEC_SPECIES_LOCAL if local else SEC_SPECIES_CLOUD
    prev = learned.get(key)
    try:
        if prev is not None and float(prev) > 0:
            return 0.55 * default + 0.45 * float(prev)
    except (TypeError, ValueError):
        pass
    return default


def species_count_expected(config: Dict[str, Any], n_images: int) -> int:
    n = max(0, int(n_images))
    if not config.get("enable_burst_detection", True):
        return n
    ratio = float(config.get("burst_keep_ratio", 0.2) or 0.2)
    min_keep = int(config.get("burst_keep_min", config.get("keep_top_n", 2)) or 2)
    return expected_kept_count(n, ratio, min_keep)


def build_eta_phase_estimates(
    config: Dict[str, Any], n_images: int
) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """
    各阶段耗时先验（秒）。
    物种阶段按「预计保留张数」而不是输入目录全量；连拍按是否快速模式给不同单张耗时。
    """
    n = max(0, int(n_images))
    n_sp = species_count_expected(config, n)
    burst_on = bool(config.get("enable_burst_detection", True))
    do_species = bool(config.get("enable_species_detection", True))
    phases: List[Tuple[str, float]] = []
    if burst_on:
        phases.append(("burst", max(8.0, n * burst_sec_per_image(config))))
    if config.get("enable_gps_write"):
        n_gps = n_sp if burst_on else n
        phases.append(("gps", max(3.0, min(90.0, 4.0 + n_gps * SEC_GPS))))
    if do_species:
        phases.append(
            ("species", max(8.0, n_sp * species_sec_per_image(config)))
        )
    if config.get("enable_watermark_generation", False):
        phases.append(("watermark", max(8.0, n_sp * SEC_WATERMARK)))
    if config.get("enable_record_export_auto", False):
        phases.append(("record_export", 8.0))
    if config.get("enable_track_map_auto", False):
        phases.append(("track_map", 12.0))
    meta = {
        "n_images": n,
        "n_species_expected": n_sp,
        "burst_sec_per": burst_sec_per_image(config) if burst_on else None,
        "species_sec_per": species_sec_per_image(config) if do_species else None,
        "burst_keep_ratio": float(config.get("burst_keep_ratio", 0.2) or 0.2),
        "burst_keep_min": int(
            config.get("burst_keep_min", config.get("keep_top_n", 2)) or 2
        ),
    }
    return phases, meta


def blend_rate(
    measured: Optional[float],
    prior_per: float,
    done: int,
    *,
    phase: str,
) -> float:
    """用伪计数把早期实测往先验收缩，避免首张加载把剩余时间拉爆。"""
    lo, hi = RATE_CLIP.get(phase, (0.02, 60.0))
    prior_per = _clip(prior_per, lo, hi)
    if measured is None or done <= 0:
        return prior_per
    k = float(PSEUDO_COUNT.get(phase, 10))
    m = _clip(float(measured), lo, hi)
    return (float(done) * m + k * prior_per) / (float(done) + k)


@dataclass
class FlowEtaEstimator:
    phases: List[Dict[str, Any]] = field(default_factory=list)
    n_images: int = 0
    n_species_expected: int = 0
    species_sec_per: float = SEC_SPECIES_LOCAL
    burst_sec_per: float = SEC_BURST_FAST_BIRD
    t0: Dict[str, Optional[float]] = field(default_factory=dict)
    done: Dict[str, int] = field(default_factory=dict)
    total: Dict[str, int] = field(default_factory=dict)
    first_elapsed: Dict[str, float] = field(default_factory=dict)
    species_done: int = 0
    species_total: int = 0
    learned: Dict[str, float] = field(default_factory=dict)
    keep_ratio: float = 0.2
    keep_min: int = 2
    kept_known: bool = False

    @classmethod
    def from_start(
        cls,
        phases: List[Dict[str, Any]],
        *,
        n_images: int,
        n_species_expected: int,
        burst_sec_per: Optional[float] = None,
        species_sec_per: Optional[float] = None,
        keep_ratio: float = 0.2,
        keep_min: int = 2,
    ) -> "FlowEtaEstimator":
        est = cls(
            n_images=max(0, int(n_images)),
            n_species_expected=max(0, int(n_species_expected)),
            burst_sec_per=float(burst_sec_per or SEC_BURST_FAST_BIRD),
            species_sec_per=float(species_sec_per or SEC_SPECIES_LOCAL),
            keep_ratio=max(0.01, min(1.0, float(keep_ratio))),
            keep_min=max(1, int(keep_min)),
        )
        for p in phases:
            nm = str(p.get("name") or "")
            if not nm:
                continue
            est.phases.append(
                {
                    "name": nm,
                    "est": float(p.get("est", 1.0)),
                    "done": False,
                }
            )
            est.t0[nm] = None
            est.done[nm] = 0
            est.total[nm] = 0
        return est

    def phase_begin(self, name: str, now: Optional[float] = None) -> None:
        t = float(now if now is not None else time.monotonic())
        self.t0[name] = t
        self.done[name] = 0
        self.first_elapsed.pop(name, None)
        if name == "species":
            self.species_done = 0

    def phase_tick(
        self, name: str, done: int, total: int, now: Optional[float] = None
    ) -> None:
        now_t = float(now if now is not None else time.monotonic())
        self.done[name] = max(0, int(done))
        self.total[name] = max(1, int(total))
        t0 = self.t0.get(name)
        if t0 is not None and self.done[name] == 1 and name not in self.first_elapsed:
            self.first_elapsed[name] = max(0.0, now_t - t0)
        if name == "species":
            self.species_done = self.done[name]
            self.species_total = self.total[name]
        if name == "burst" and not self.kept_known:
            tot = int(self.total[name])
            if tot > 1 and tot != self.n_images:
                self.n_images = tot
                n_sp = expected_kept_count(tot, self.keep_ratio, self.keep_min)
                self.n_species_expected = n_sp
                self._revise_post_burst_estimates(n_sp)

    def phase_done(self, name: str, now: Optional[float] = None) -> None:
        t1 = float(now if now is not None else time.monotonic())
        for p in self.phases:
            if p["name"] == name:
                p["done"] = True
                t0 = self.t0.get(name)
                tot = self.total.get(name) or 0
                if name == "species":
                    tot = self.species_total or tot
                if t0 is not None and tot > 0:
                    elapsed = max(0.05, t1 - t0)
                    first = self.first_elapsed.get(name)
                    if first is not None and tot >= 2 and elapsed > first:
                        sec = max(0.05, (elapsed - first) / float(tot - 1))
                    else:
                        sec = max(0.05, elapsed / float(tot))
                    lo, hi = RATE_CLIP.get(name, (0.02, 60.0))
                    self.learned[name] = _clip(sec, lo, hi)
                break

    def _revise_post_burst_estimates(self, kept: int) -> None:
        kept = max(0, int(kept))
        for p in self.phases:
            if p.get("done"):
                continue
            if p["name"] == "species":
                p["est"] = max(6.0, kept * self.species_sec_per)
            elif p["name"] == "watermark":
                p["est"] = max(6.0, kept * SEC_WATERMARK)
            elif p["name"] == "gps":
                p["est"] = max(3.0, min(90.0, 4.0 + kept * SEC_GPS))

    def burst_result(self, total: int, kept: int) -> None:
        kept = max(0, int(kept))
        total = max(0, int(total))
        if kept <= 0 and total > 0:
            kept = expected_kept_count(total, self.keep_ratio, self.keep_min)
        self.kept_known = True
        if total > 0:
            self.n_images = total
        self.n_species_expected = kept
        self._revise_post_burst_estimates(kept)

    def species_begin(self, n: int) -> None:
        n = max(0, int(n))
        self.species_total = n
        self.species_done = 0
        self.n_species_expected = n
        self.total["species"] = max(1, n) if n else 1
        self.done["species"] = 0
        for p in self.phases:
            if p["name"] == "species":
                p["est"] = max(5.0, n * self.species_sec_per) if n > 0 else 3.0
                break

    def _prior_per(self, name: str) -> float:
        est = 1.0
        for p in self.phases:
            if p["name"] == name:
                est = float(p.get("est", 1.0))
                break
        if name == "species":
            denom = max(1, self.species_total or self.n_species_expected or 1)
            return est / float(denom)
        if name == "burst":
            denom = max(1, self.total.get("burst") or self.n_images or 1)
            return est / float(denom)
        denom = max(1, self.total.get(name) or self.n_species_expected or self.n_images or 1)
        return est / float(denom)

    def _current_remaining(self, name: str, now: float) -> float:
        t0 = self.t0.get(name)
        elapsed = (now - t0) if t0 is not None else 0.0
        done = int(self.done.get(name, 0) or 0)
        total = int(self.total.get(name, 0) or 0)
        if name == "species":
            done = int(self.species_done)
            total = int(self.species_total or total)
        prior_est = next(
            (float(p["est"]) for p in self.phases if p["name"] == name), 1.0
        )
        prior_per = self._prior_per(name)
        if total <= 0:
            return max(0.0, prior_est - elapsed)
        left = max(0, total - done)
        if left <= 0:
            return 0.0
        if done <= 0:
            return max(0.0, prior_est - elapsed)
        # 首张常含模型加载，不能当稳态速度；未完成第 2 张前仍用先验。
        if done == 1:
            return float(left) * prior_per
        first = self.first_elapsed.get(name)
        if first is not None and elapsed > first:
            measured = (elapsed - first) / float(done - 1)
            rate = blend_rate(measured, prior_per, done - 1, phase=name)
        else:
            measured = elapsed / float(done) if elapsed > 0 else None
            rate = blend_rate(measured, prior_per, done, phase=name)
        return float(left) * rate

    def remaining_sec(self, now: Optional[float] = None) -> float:
        if not self.phases:
            return 0.0
        now_t = float(now if now is not None else time.monotonic())
        rem = 0.0
        seen_current = False
        for p in self.phases:
            if p.get("done"):
                continue
            name = p["name"]
            if not seen_current:
                seen_current = True
                rem += self._current_remaining(name, now_t)
            else:
                # 未开始阶段只用本阶段先验。连拍快不等于识别快，不按当前速度去乘。
                rem += max(0.0, float(p.get("est", 0.0)))
        return max(0.0, rem)

    def learned_for_config(self, config: Dict[str, Any]) -> Dict[str, float]:
        out = dict(config.get("_eta_learned") or {})
        if "burst" in self.learned:
            if not config.get("use_bird_detection", True):
                out["burst_no_bird"] = self.learned["burst"]
            elif config.get("use_fast_mode", True):
                out["burst_fast"] = self.learned["burst"]
            else:
                out["burst_full"] = self.learned["burst"]
        if "species" in self.learned:
            if config.get("use_local_model", True):
                out["species_local"] = self.learned["species"]
            else:
                out["species_cloud"] = self.learned["species"]
        return out
