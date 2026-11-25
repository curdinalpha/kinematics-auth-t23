#!/usr/bin/env python3
"""Operator-continuity verifier from 2D trajectory ticks + channel-tagged
symbol bursts. Python 3.12, standard library only, offline.

One command runs the whole benchmark (generate -> enroll -> calibrate ->
freeze -> holdout -> FAR/FRR table):

    python3.12 kinematics_auth.py

Other entry points:
    python3.12 kinematics_auth.py bench --data traces/     # use synth_traces.py output
    python3.12 kinematics_auth.py bench --sweep 10          # 10 seeds, aggregate table
    python3.12 kinematics_auth.py enroll --data enroll.jsonl --out model.json
    python3.12 kinematics_auth.py score --model model.json --data sessions.jsonl

Input session (JSON object):
    ticks : [[ts_ms, x, y, trigger_kind], ...]   x,y in [0,1]; move | press | release
    events: [[ts_ms, channel, op, n], ...]       DIRECT_KEY | VIRTUAL_GRID | BATCH_CHUNK
                                                 (BATCH_INSERT alias); INSERT | BACKSPACE | REPLACE
Nothing in the features depends on the sensor rate or on absolute speed:
trajectory features are ratios measured on interpolated crossing times,
durations only enter through log-log slopes, and burst features are counted
in symbols, with gaps normalised by the session's own per-symbol interval and
by the number of symbols each event carries (so a 4-symbol BATCH_CHUNK and 4
DIRECT_KEY presses look the same).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
import sys
import time
from typing import Dict, List, Optional, Sequence

import synth_traces as synth

KINDS = {"move", "press", "release"}
CHANNEL_ALIASES = {"DIRECT_KEY": "DIRECT_KEY", "VIRTUAL_GRID": "VIRTUAL_GRID",
                   "BATCH_CHUNK": "BATCH_CHUNK", "BATCH_INSERT": "BATCH_CHUNK"}
OPS = {"INSERT", "BACKSPACE", "REPLACE"}
FEATURES = ("curv", "half", "over", "home", "dwell", "fitts_b", "aniso",
            "blen", "bcv", "pause", "edit", "bs_len", "repl", "epause")
N_SEGMENTS = 4
LLR_CLIP = (-6.0, 3.0)
DRIFT = 0.25         # profile spread = hypot(own MAD spread, DRIFT x population spread)
SCORING = "segment"  # "segment": running mean of per-segment LLRs (default); "prefix": LLR of cumulative features
PAUSE_K = 2.5
RATE_Q = 0.25        # quantile of dt/n used as the per-symbol interval
MIN_REACH = 0.08


class InputError(ValueError):
    pass


# ------------------------------------------------------------------ validation
def validate(session: Dict) -> Dict:
    ticks, events = session.get("ticks"), session.get("events")
    if not isinstance(ticks, list) or not isinstance(events, list):
        raise InputError("session needs 'ticks' and 'events' lists")
    last = -math.inf
    out_t = []
    for i, tk in enumerate(ticks):
        if not (isinstance(tk, (list, tuple)) and len(tk) == 4):
            raise InputError(f"tick {i}: expected [ts_ms, x, y, kind]")
        ts, x, y, kind = tk
        if kind not in KINDS:
            raise InputError(f"tick {i}: unknown trigger_kind {kind!r}")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (ts, x, y)):
            raise InputError(f"tick {i}: non-numeric value")
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise InputError(f"tick {i}: x/y outside [0,1]")
        if ts < last:
            raise InputError(f"tick {i}: timestamps must be non-decreasing")
        last = ts
        out_t.append((float(ts), float(x), float(y), kind))
    last = -math.inf
    out_e = []
    for i, ev in enumerate(events):
        if not (isinstance(ev, (list, tuple)) and len(ev) == 4):
            raise InputError(f"event {i}: expected [ts_ms, channel, op, n]")
        ts, ch, op, n = ev
        if ch not in CHANNEL_ALIASES:
            raise InputError(f"event {i}: unknown channel {ch!r}")
        if op not in OPS:
            raise InputError(f"event {i}: unknown op {op!r}")
        if not isinstance(n, int) or n < 1:
            raise InputError(f"event {i}: n must be an int >= 1")
        if not (isinstance(ts, (int, float)) and math.isfinite(ts)) or ts < last:
            raise InputError(f"event {i}: bad or decreasing timestamp")
        last = ts
        out_e.append((float(ts), CHANNEL_ALIASES[ch], op, n))
    return {"ticks": out_t, "events": out_e}


# -------------------------------------------------------------- trajectories
def _cross_time(samples, value):
    """First time the projection crosses `value`, linearly interpolated."""
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        if p0 < value <= p1:
            return t0 + (t1 - t0) * (value - p0) / (p1 - p0) if p1 != p0 else t1
    return None


def reaches(ticks) -> List[Dict]:
    """Split ticks into reaches (release/start -> press) and measure each."""
    out = []
    start = None
    path = []
    for ts, x, y, kind in ticks:
        if kind == "release" or start is None:
            start = (ts, x, y)
            path = []
            if kind == "release":
                continue
        if kind == "move":
            path.append((ts, x, y))
        elif kind == "press":
            r = _measure(start, path, (ts, x, y))
            if r is not None:
                out.append(r)
            start, path = None, []
    return out


def _measure(start, path, press):
    _, sx, sy = start
    tp, ex, ey = press
    dist = math.hypot(ex - sx, ey - sy)
    if dist < MIN_REACH or len(path) < 3:
        return None
    ux, uy = (ex - sx) / dist, (ey - sy) / dist
    pts = [start] + path + [press]
    # onset: midpoint between last sample near the start and first sample that left it
    on = None
    for a, b in zip(pts, pts[1:]):
        if math.hypot(b[1] - sx, b[2] - sy) > 0.01:
            on = (a[0] + b[0]) / 2
            break
    # arrival: midpoint after the last sample that was still away from the target
    arr = None
    for a, b in zip(pts, pts[1:]):
        if math.hypot(a[1] - ex, a[2] - ey) > 0.01:
            arr = (a[0] + b[0]) / 2
    if on is None or arr is None or arr <= on:
        return None
    dur = arr - on
    prj = [(t, ((x - sx) * ux + (y - sy) * uy) / dist) for t, x, y in pts]
    lat = [(ux * (y - sy) - uy * (x - sx)) / dist for (t, x, y), (_, p) in zip(pts, prj) if 0.2 <= p <= 0.8]
    t_half = _cross_time(prj, 0.5)
    t_90 = _cross_time(prj, 0.9)
    if t_half is None or t_90 is None or not lat:
        return None
    return {
        "curv": sum(lat) / len(lat),
        "half": (t_half - on) / dur,
        "over": max(0.0, max(p for _, p in prj) - 1.0),
        "home": max(0.0, arr - t_90) / dur,
        "dwell": math.log(max(1e-3, tp - arr) / dur),
        "logd": math.log(dist),
        "logt": math.log(dur),
        "horiz": ux * ux - 0.5,
    }


def _ols2(rows):
    """log T = c + b log D + a horiz  ->  (b, a) by least squares."""
    n = len(rows)
    if n < 4:
        return None, None
    mx1 = sum(r["logd"] for r in rows) / n
    mx2 = sum(r["horiz"] for r in rows) / n
    my = sum(r["logt"] for r in rows) / n
    s11 = sum((r["logd"] - mx1) ** 2 for r in rows)
    s22 = sum((r["horiz"] - mx2) ** 2 for r in rows)
    s12 = sum((r["logd"] - mx1) * (r["horiz"] - mx2) for r in rows)
    s1y = sum((r["logd"] - mx1) * (r["logt"] - my) for r in rows)
    s2y = sum((r["horiz"] - mx2) * (r["logt"] - my) for r in rows)
    det = s11 * s22 - s12 * s12
    if det <= 1e-9:
        return None, None
    return (s22 * s1y - s12 * s2y) / det, -(s11 * s2y - s12 * s1y) / det


def traj_features(rs: List[Dict]) -> Dict[str, Optional[float]]:
    if len(rs) < 2:
        return {k: None for k in FEATURES[:7]}
    b, a = _ols2(rs)
    return {
        "curv": st.median(r["curv"] for r in rs),
        "half": st.median(r["half"] for r in rs),
        "over": st.fmean(r["over"] for r in rs),
        "home": st.median(r["home"] for r in rs),
        "dwell": st.median(r["dwell"] for r in rs),
        "fitts_b": b,
        "aniso": a,
    }


# --------------------------------------------------------------------- bursts
def symbol_rate(events) -> Optional[float]:
    """Robust per-symbol interval m over INSERT events: median of the dt/n
    values that are <= 3x their lower quartile. Plain median breaks on chunked
    delivery, where most chunks can be the first of a burst and their dt/n
    still contains the pause; trimming at 3x q25 drops those."""
    v = sorted((b[0] - a[0]) / b[3] for a, b in zip(events, events[1:]) if b[2] == "INSERT" and b[0] > a[0])
    if not v:
        return None
    q = v[int((len(v) - 1) * RATE_Q)]
    return st.median([x for x in v if x <= 3 * q])


def annotate(events) -> List[Dict]:
    """Per event: residual gap in units of m and whether it starts after a pause."""
    m = symbol_rate(events)
    out = []
    prev = None
    for ev in events:
        ts, ch, op, n = ev
        if prev is None or m is None:
            r = math.inf
        else:
            r = (ts - prev - (n - 1) * m) / m if op == "INSERT" else (ts - prev) / m
        out.append({"ts": ts, "op": op, "n": n, "pause": r > PAUSE_K, "r": r})
        prev = ts
    return out


def burst_features(ann: List[Dict]) -> Dict[str, Optional[float]]:
    bursts, cur = [], 0
    pauses, corr = [], []
    i = 0
    inserted = 0
    while i < len(ann):
        e = ann[i]
        if e["pause"] and e["op"] == "INSERT":
            if cur:
                bursts.append(cur)
            cur = 0
            if math.isfinite(e["r"]):
                pauses.append(e["r"])
        if e["op"] == "INSERT":
            cur += e["n"]
            inserted += e["n"]
            i += 1
        elif e["op"] == "REPLACE":
            corr.append(("R", e["n"], e["pause"]))
            i += 1
        else:  # a correction = maximal run of BACKSPACE events
            j, n = i, 0
            while j < len(ann) and ann[j]["op"] == "BACKSPACE":
                n += ann[j]["n"]
                j += 1
            corr.append(("B", n, e["pause"]))
            i = j
    if cur:
        bursts.append(cur)
    f = {k: None for k in FEATURES[7:]}
    if len(bursts) >= 3:
        mu = st.fmean(bursts)
        f["blen"] = math.log(st.median(bursts))
        f["bcv"] = st.pstdev(bursts) / mu
    if pauses:
        f["pause"] = math.log(st.median(pauses))
    if inserted:
        f["edit"] = len(corr) / inserted
    bs = [n for k, n, _ in corr if k == "B"]
    if bs:
        f["bs_len"] = st.fmean(bs)
    if corr:
        f["repl"] = sum(1 for k, _, _ in corr if k == "R") / len(corr)
        f["epause"] = sum(1 for _, _, p in corr if p) / len(corr)
    return f


# ------------------------------------------------------------------- segments
def _split(seq, k, weight=lambda x: 1):
    total = sum(weight(x) for x in seq)
    out, cur, acc, idx = [[] for _ in range(k)], 0, 0, 0
    for x in seq:
        out[min(k - 1, int(k * acc / total) if total else 0)].append(x)
        acc += weight(x)
    return out


def _segments(session: Dict, k: int):
    s = validate(session)
    rs = reaches(s["ticks"])
    ann = annotate(s["events"])
    return _split(rs, k), _split(ann, k, weight=lambda e: e["n"] if e["op"] == "INSERT" else 0)


def segment_vectors(session: Dict, k: int = N_SEGMENTS) -> List[Dict[str, Optional[float]]]:
    """One feature vector per segment (used to fit profiles)."""
    rsplit, asplit = _segments(session, k)
    vecs = []
    for i in range(k):
        v = traj_features(rsplit[i])
        v.update(burst_features(asplit[i]))
        vecs.append(v)
    return vecs


def prefix_vectors(session: Dict, k: int = N_SEGMENTS) -> List[Dict[str, Optional[float]]]:
    """Feature vector of segments 0..i for each i (what a streaming verifier has seen so far)."""
    rsplit, asplit = _segments(session, k)
    vecs = []
    for i in range(k):
        v = traj_features([r for part in rsplit[:i + 1] for r in part])
        v.update(burst_features([e for part in asplit[:i + 1] for e in part]))
        vecs.append(v)
    return vecs


# ---------------------------------------------------------------------- model
def _robust(vals):
    med = st.median(vals)
    mad = st.median(abs(v - med) for v in vals) * 1.4826
    return med, mad


def fit_background(vectors: Sequence[Dict]) -> Dict[str, tuple]:
    bg = {}
    for f in FEATURES:
        vals = [v[f] for v in vectors if v[f] is not None]
        med, mad = _robust(vals)
        bg[f] = (med, max(mad, 1e-4))
    return bg


def fit_profile(vectors: Sequence[Dict], bg: Dict[str, tuple]) -> Dict[str, tuple]:
    prof = {}
    for f in FEATURES:
        vals = [v[f] for v in vectors if v[f] is not None]
        if len(vals) < 5:
            prof[f] = None
            continue
        med, mad = _robust(vals)
        # session-to-session drift allowance: never tighter than DRIFT x population spread
        prof[f] = (med, max(math.hypot(mad, DRIFT * bg[f][1]), 1e-4))
    return prof


def _logpdf(x, mu, sd):
    z = (x - mu) / sd
    return -0.5 * z * z - math.log(sd)


def segment_llr(v: Dict, prof: Dict, bg: Dict) -> Optional[float]:
    """Mean clipped per-feature log-likelihood ratio, profile vs population.
    Missing features (e.g. no corrections in this stretch) are skipped."""
    tot, n = 0.0, 0
    for f in FEATURES:
        if v[f] is None or prof[f] is None:
            continue
        l = _logpdf(v[f], *prof[f]) - _logpdf(v[f], *bg[f])
        tot += min(LLR_CLIP[1], max(LLR_CLIP[0], l))
        n += 1
    return tot / n if n else None


def sequential_scores(vecs: Sequence[Dict], prof: Dict, bg: Dict) -> List[float]:
    """Cumulative mean segment LLR after each segment (the decision uses the last)."""
    out, acc, n = [], 0.0, 0
    for v in vecs:
        s = segment_llr(v, prof, bg)
        if s is not None:
            acc += s
            n += 1
        out.append(acc / n if n else -math.inf)
    return out


def prefix_scores(pvecs: Sequence[Dict], prof: Dict, bg: Dict) -> List[float]:
    """Sequential score after each segment: LLR of the features of everything
    seen so far (segments 0..i). The decision uses the last value."""
    out = []
    for v in pvecs:
        s = segment_llr(v, prof, bg)
        out.append(-math.inf if s is None else s)
    return out


def enroll(sessions_by_profile: Dict[str, List[Dict]], background_sessions: List[Dict]) -> Dict:
    groups: Dict[str, List[Dict]] = {}
    for s in background_sessions:
        groups.setdefault("bg:" + str(s.get("source")), []).extend(segment_vectors(s))
    pv = {p: [v for s in ss for v in segment_vectors(s)] for p, ss in sorted(sessions_by_profile.items())}
    groups.update(pv)
    bg = fit_background([v for vs in groups.values() for v in vs])
    profiles = {p: fit_profile(vs, bg) for p, vs in pv.items()}
    return {"features": list(FEATURES), "background": bg, "profiles": profiles,
            "n_segments": N_SEGMENTS, "llr_clip": list(LLR_CLIP), "drift": DRIFT}


def model_hash(model: Dict) -> str:
    return hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()


def score(model: Dict, claimed: str, session: Dict) -> float:
    return scores_for(model, claimed, session_features(session))[-1]


def session_features(session: Dict):
    return prefix_vectors(session) if SCORING == "prefix" else segment_vectors(session)


def scores_for(model: Dict, claimed: str, feats) -> List[float]:
    fn = prefix_scores if SCORING == "prefix" else sequential_scores
    return fn(feats, model["profiles"][claimed], model["background"])


# ---------------------------------------------------------------- calibration
def calibrate(genuine: Sequence[float], impostor: Sequence[float]) -> Dict:
    """Threshold minimising FAR+FRR on calibration scores only; ties broken by
    the widest gap (midpoint between neighbouring scores)."""
    pts = sorted(set(genuine) | set(impostor))
    cands = [pts[0] - 1.0] + [(a + b) / 2 for a, b in zip(pts, pts[1:])] + [pts[-1] + 1.0]
    best = None
    for c in cands:
        far = sum(1 for s in impostor if s >= c) / len(impostor)
        frr = sum(1 for s in genuine if s < c) / len(genuine)
        lo = max([s for s in pts if s < c], default=c - 1)
        hi = min([s for s in pts if s >= c], default=c + 1)
        key = (far + frr, -(hi - lo))
        if best is None or key < best[0]:
            best = (key, c, far, frr)
    return {"threshold": best[1], "cal_far": best[2], "cal_frr": best[3]}


class Frozen:
    """Token proving the threshold was fixed before any holdout data exists."""

    def __init__(self, threshold, mhash, cal_digest):
        self.threshold, self.model_hash, self.cal_digest = threshold, mhash, cal_digest
        self.frozen_at_step = "pre-holdout"
        self.digest = hashlib.sha256(f"{threshold!r}|{mhash}|{cal_digest}".encode()).hexdigest()


def wilson_upper(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c + r) / den


# ------------------------------------------------------------------ benchmark
def run_benchmark(seed: int = 2323, data: Dict[str, List[Dict]] = None, alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
                  holdout_seed: Optional[int] = None, log=print) -> Dict:
    t_start = time.perf_counter()
    ids = synth.build_population(seed) if data is None else None
    # 1) enrollment
    enroll_rows = data["enroll"] if data else synth.enrollment_split(ids, seed)
    by_p: Dict[str, List[Dict]] = {}
    bg_rows = []
    for s in enroll_rows:
        (by_p.setdefault(s["source"], []) if s["source"].startswith("P") else bg_rows).append(s)
    model = enroll(by_p, bg_rows)
    mh = model_hash(model)
    # 2) calibration (disjoint sessions, disjoint stranger and mimic identities)
    cal_rows = data["calibration"] if data else synth.calibration_split(ids, seed)
    cal_scored = []
    for s in cal_rows:
        targets = sorted(by_p) if s["condition"] == "stranger" else [s["claimed"]]
        for p in targets:
            cal_scored.append((s["condition"], s["regime"], p, score(model, p, s)))
    gen = [x[3] for x in cal_scored if x[0] in ("normal", "fatigued")]
    imp = [x[3] for x in cal_scored if x[0] in ("stranger", "mimic")]
    cal = calibrate(gen, imp)
    cal_digest = hashlib.sha256(json.dumps(cal_scored).encode()).hexdigest()
    frozen = Frozen(cal["threshold"], mh, cal_digest)
    log(f"[freeze] threshold={frozen.threshold:.6f} model={mh[:16]} cal={cal_digest[:16]} "
        f"(cal FAR {cal['cal_far']:.4f} FRR {cal['cal_frr']:.4f}; {len(gen)} genuine / {len(imp)} impostor)")
    # 3) holdout: generated only now, after the freeze
    #    (holdout_seed only changes the holdout sessions; identities stay those of `seed`)
    hs = seed if holdout_seed is None else holdout_seed
    hold_rows = data["holdout"] if data else synth.holdout_split(ids, hs, alphas=alphas)
    return evaluate(model, frozen, cal, hold_rows, sorted(by_p), alphas, t_start, seed)


def evaluate(model, frozen: Frozen, cal, hold_rows, profiles, alphas, t_start, seed):
    if not isinstance(frozen, Frozen) or frozen.model_hash != model_hash(model):
        raise RuntimeError("holdout evaluation requires a threshold frozen on this exact model")
    thr = frozen.threshold
    rows = []
    swap = {reg: {a: {b: [0, 0] for b in profiles} for a in profiles} for reg in synth.REGIMES}
    seq_hits = {}
    for s in hold_rows:
        cond, reg = s["condition"], s["regime"]
        vecs = session_features(s)
        if cond == "normal":
            for p in profiles:  # full N x N swap matrix: every model vs every profile's sessions
                sc = scores_for(model, p, vecs)
                swap[reg][p][s["source"]][0] += sc[-1] >= thr
                swap[reg][p][s["source"]][1] += 1
                if p == s["source"]:
                    rows.append((reg, "genuine_normal", sc))
        elif cond == "fatigued":
            rows.append((reg, "fatigued_self", scores_for(model, s["claimed"], vecs)))
        elif cond == "stranger":
            for p in profiles:
                rows.append((reg, "random_stranger", scores_for(model, p, vecs)))
        elif cond == "mimic":
            a = float(s.get("alpha", 0.0))
            key = "param_mimic" if a == 0.0 else f"mimic_alpha_{a:.2f}"
            rows.append((reg, key, scores_for(model, s["claimed"], vecs)))
    table = []
    classes = ["genuine_normal", "fatigued_self", "random_stranger", "param_mimic"]
    for reg in synth.REGIMES:
        for c in classes:
            sc = [r[2][-1] for r in rows if r[0] == reg and r[1] == c]
            genuine = c in ("genuine_normal", "fatigued_self")
            err = sum(1 for x in sc if (x < thr if genuine else x >= thr))
            table.append({"regime": reg, "class": c, "metric": "FRR" if genuine else "FAR",
                          "errors": err, "n": len(sc), "rate": err / len(sc) if sc else None,
                          "wilson95_upper": wilson_upper(err, len(sc)),
                          "min_margin": min((x - thr) if genuine else (thr - x) for x in sc) if sc else None})
        off = [(a, b) for a in profiles for b in profiles if a != b]
        k = sum(swap[reg][a][b][0] for a, b in off)
        n = sum(swap[reg][a][b][1] for a, b in off)
        table.append({"regime": reg, "class": "swap_offdiag", "metric": "FAR", "errors": k, "n": n,
                      "rate": k / n, "wilson95_upper": wilson_upper(k, n), "min_margin": None})
    # pooled
    pooled = {}
    for metric, cls in (("FRR", ("genuine_normal", "fatigued_self")),
                        ("FAR", ("random_stranger", "param_mimic", "swap_offdiag"))):
        k = sum(r["errors"] for r in table if r["class"] in cls)
        n = sum(r["n"] for r in table if r["class"] in cls)
        pooled[metric] = {"errors": k, "n": n, "rate": k / n, "wilson95_upper": wilson_upper(k, n)}
    sweep = []
    for a in alphas:
        key = "param_mimic" if a == 0.0 else f"mimic_alpha_{a:.2f}"
        sc = [r[2][-1] for r in rows if r[1] == key]
        if sc:
            sweep.append({"alpha": a, "accepted": sum(1 for x in sc if x >= thr), "n": len(sc)})
    seq = []
    for i in range(N_SEGMENTS):
        g = [r[2][i] for r in rows if r[1] in ("genuine_normal", "fatigued_self")]
        im = [r[2][i] for r in rows if r[1] in ("random_stranger", "param_mimic")]
        seq.append({"segments_seen": i + 1, "frr": sum(1 for x in g if x < thr) / len(g),
                    "far": sum(1 for x in im if x >= thr) / len(im)})
    # holdout EER: diagnostic only, computed after the fact, never used for the decision
    g = [r[2][-1] for r in rows if r[1] in ("genuine_normal", "fatigued_self")]
    im = [r[2][-1] for r in rows if r[1] in ("random_stranger", "param_mimic")]
    diag = calibrate(g, im)
    report = {
        "seed": seed,
        "frozen": {"threshold": round(frozen.threshold, 6), "model_sha256": frozen.model_hash,
                   "calibration_scores_sha256": frozen.cal_digest, "freeze_digest": frozen.digest,
                   "cal_far": cal["cal_far"], "cal_frr": cal["cal_frr"]},
        "table": table,
        "pooled": pooled,
        "swap_matrix": {reg: {a: {b: f"{v[0]}/{v[1]}" for b, v in row.items()} for a, row in m.items()}
                        for reg, m in swap.items()},
        "mimic_strength_sweep": sweep,
        "sequential": seq,
        "diagnostic_holdout_best_threshold": round(diag["threshold"], 6),
        "elapsed_s": round(time.perf_counter() - t_start, 2),
    }
    return report


def format_report(rep: Dict) -> str:
    L = []
    fz = rep["frozen"]
    L.append(f"seed={rep['seed']}  threshold={fz['threshold']} (frozen before holdout; freeze digest {fz['freeze_digest'][:16]})")
    L.append(f"{'regime':6} {'class':16} {'metric':6} {'errors/n':>10} {'rate':>7} {'95%up':>7} {'min margin':>10}")
    for r in rep["table"]:
        mm = "" if r["min_margin"] is None else f"{r['min_margin']:+.3f}"
        L.append(f"{r['regime']:6} {r['class']:16} {r['metric']:6} {str(r['errors']) + '/' + str(r['n']):>10} "
                 f"{r['rate']:7.4f} {r['wilson95_upper']:7.4f} {mm:>10}")
    for m, v in rep["pooled"].items():
        L.append(f"POOLED {m}: {v['errors']}/{v['n']} = {v['rate']:.4f} (95% upper {v['wilson95_upper']:.4f})")
    for reg, m in rep["swap_matrix"].items():
        L.append(f"swap matrix [{reg}] rows = enrolled model, cols = true source (accepted/attempts):")
        cols = list(m)
        L.append("       " + " ".join(f"{c:>5}" for c in cols))
        for a in cols:
            L.append(f"  {a:4} " + " ".join(f"{m[a][b]:>5}" for b in cols))
    L.append("mimic strength sweep (alpha=0: visible stats copied; 1: full clone): "
             + ", ".join(f"a={s['alpha']}: {s['accepted']}/{s['n']}" for s in rep["mimic_strength_sweep"]))
    L.append("sequential (same frozen threshold, after k of 4 segments): "
             + ", ".join(f"k={s['segments_seen']} FRR {s['frr']:.3f} FAR {s['far']:.3f}" for s in rep["sequential"]))
    L.append(f"elapsed {rep['elapsed_s']} s")
    return "\n".join(L)


# ------------------------------------------------------------------------ CLI
def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Operator-continuity verifier (kinematics + burst rhythm)")
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("bench", help="full benchmark (default)")
    b.add_argument("--seed", type=int, default=2323)
    b.add_argument("--data", help="directory with enroll/calibration/holdout.jsonl from synth_traces.py")
    b.add_argument("--json", help="write the full report as JSON here")
    b.add_argument("--sweep", type=int, default=0, help="also run N extra seeds and aggregate")
    e = sub.add_parser("enroll")
    e.add_argument("--data", required=True)
    e.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--model", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--claimed", help="profile id (default: the session's 'claimed' field)")
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv = ["bench"] + argv
    a = ap.parse_args(argv)
    if a.cmd == "bench":
        seed = getattr(a, "seed", 2323)
        data = None
        if getattr(a, "data", None):
            data = {k: _read_jsonl(f"{a.data}/{k}.jsonl") for k in ("enroll", "calibration", "holdout")}
        rep = run_benchmark(seed, data, alphas=(0.0,) if data else (0.0, 0.25, 0.5, 0.75, 1.0))
        print(format_report(rep))
        if getattr(a, "json", None):
            with open(a.json, "w") as f:
                json.dump(rep, f, indent=1, sort_keys=True)
        n = getattr(a, "sweep", 0)
        if n:
            agg = {"FAR": [0, 0], "FRR": [0, 0]}
            for sd in range(1, n + 1):
                r = run_benchmark(seed + 1000 * sd, alphas=(0.0,), log=lambda *_: None)
                for m in agg:
                    agg[m][0] += r["pooled"][m]["errors"]
                    agg[m][1] += r["pooled"][m]["n"]
                print(f"sweep seed={seed + 1000 * sd} threshold={r['frozen']['threshold']} "
                      f"FAR {r['pooled']['FAR']['errors']}/{r['pooled']['FAR']['n']} "
                      f"FRR {r['pooled']['FRR']['errors']}/{r['pooled']['FRR']['n']}")
            print(f"SWEEP ({n} seeds) FAR {agg['FAR'][0]}/{agg['FAR'][1]} FRR {agg['FRR'][0]}/{agg['FRR'][1]}")
        return 0
    if a.cmd == "enroll":
        rows = _read_jsonl(a.data)
        by_p, bg = {}, []
        for r in rows:
            (by_p.setdefault(r["source"], []) if r["source"].startswith("P") else bg).append(r)
        model = enroll(by_p, bg or [x for v in by_p.values() for x in v])
        with open(a.out, "w") as f:
            json.dump(model, f, sort_keys=True)
        print(f"enrolled {len(by_p)} profiles, model sha256 {model_hash(model)}")
        return 0
    if a.cmd == "score":
        with open(a.model) as f:
            model = json.load(f)
        for p, prof in model["profiles"].items():
            model["profiles"][p] = {k: (tuple(v) if v else None) for k, v in prof.items()}
        model["background"] = {k: tuple(v) for k, v in model["background"].items()}
        for r in _read_jsonl(a.data):
            claimed = a.claimed or r.get("claimed")
            print(json.dumps({"claimed": claimed, "score": round(score(model, claimed, r), 6)}))
        return 0
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except InputError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        sys.exit(2)
