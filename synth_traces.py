#!/usr/bin/env python3
"""Synthetic operator traces for kinematics_auth.py (Python 3.12, stdlib only).

A session is two streams:
  ticks  : [ts_ms, x, y, trigger_kind]   x, y in [0, 1]; kind in move/press/release
  events : [ts_ms, channel, op, n]       channel DIRECT_KEY | VIRTUAL_GRID | BATCH_CHUNK
                                         op INSERT | BACKSPACE | REPLACE, n = symbols
(BATCH_INSERT is accepted as an alias of BATCH_CHUNK.)

Two regimes:
  fast : ~8 ms sensor ticks, quick continuous reaches, DIRECT_KEY bursts
  slow : ~45-90 ms jittery ticks with dropouts, reaches 2.5-3.5x slower,
         VIRTUAL_GRID (one symbol per grid hop) and BATCH_CHUNK (2-6 symbols
         flushed at once) bursts

Each identity has hidden motor/rhythm parameters. Conditions:
  normal    the identity itself
  fatigued  same identity: slower, noisier, more pauses and edits, later notice
  stranger  an unrelated identity drawn from the same population prior
  mimic     a different identity that copies the target's *visible aggregate*
            parameters (speed, symbol interval, burst length, pause ratio,
            edit rate, dwell) but keeps its own hidden structure. ``alpha``
            blends ALL hidden parameters toward the target (alpha=1: clone),
            used for the mimic-strength sweep.

Everything is derived from sha256(seed, split, role, index), so any split
can be regenerated alone and runs are byte-identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from typing import Dict, List

CHANNELS = ("DIRECT_KEY", "VIRTUAL_GRID", "BATCH_CHUNK")
VISIBLE = ("speed", "iki_ms", "burst_mu", "pause_ratio", "edit_rate", "dwell_ratio")


def rng_for(*parts) -> random.Random:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def draw_identity(r: random.Random) -> Dict[str, float]:
    """Population prior over hidden parameters."""
    return {
        # visible aggregates (what a mimic can measure and copy)
        "speed": math.exp(r.gauss(0.0, 0.25)),        # global duration multiplier
        "iki_ms": r.uniform(110, 240),                 # median inter-symbol interval
        "burst_mu": r.uniform(3.0, 9.0),               # mean symbols per burst
        "pause_ratio": r.uniform(4.0, 12.0),           # inter-burst pause / interval
        "edit_rate": r.uniform(0.03, 0.12),            # corrections per inserted symbol
        "dwell_ratio": r.uniform(0.10, 0.55),          # settle time before press / reach time
        # hidden structure
        "curv": r.gauss(0.0, 0.07),                    # signed lateral bow (fraction of distance)
        "asym": r.uniform(0.36, 0.60),                 # time fraction to cover half the distance
        "over_p": r.uniform(0.0, 0.6),                 # overshoot probability
        "over_mag": r.uniform(0.03, 0.12),             # overshoot size (fraction of distance)
        "fitts_b": r.uniform(0.25, 0.85),              # duration-distance exponent
        "aniso": r.gauss(0.0, 0.22),                   # log speed ratio horizontal vs vertical
        "burst_cv": r.uniform(0.25, 0.75),             # burst length dispersion
        "notice": r.uniform(0.3, 3.5),                 # symbols typed before noticing an error
        "replace_p": r.uniform(0.0, 0.6),              # corrections done as REPLACE
        "over_del": r.uniform(0.0, 0.4),               # extra symbols deleted
        "edit_pause_p": r.uniform(0.1, 0.9),           # corrections that start after a pause
    }


def make_mimic(target: Dict[str, float], own: Dict[str, float], alpha: float = 0.0) -> Dict[str, float]:
    p = {}
    for k in target:
        if k in VISIBLE:
            p[k] = target[k]
        else:
            p[k] = (1 - alpha) * own[k] + alpha * target[k]
    return p


def fatigue(p: Dict[str, float], r: random.Random) -> Dict[str, float]:
    q = dict(p)
    q["speed"] *= r.uniform(1.25, 1.6)
    q["iki_ms"] *= r.uniform(1.2, 1.5)
    q["pause_ratio"] *= r.uniform(1.1, 1.4)
    q["edit_rate"] *= r.uniform(1.15, 1.5)
    q["notice"] += r.uniform(0.2, 0.8)
    q["dwell_ratio"] *= r.uniform(1.05, 1.3)
    q["_noise"] = 1.6
    return q


def _minjerk(u: float) -> float:
    return u * u * u * (10 - 15 * u + 6 * u * u)


class _Clock:
    def __init__(self, r: random.Random, regime: str):
        self.r, self.regime, self.t = r, regime, 0.0
        self.period = 8.0 if regime == "fast" else r.uniform(45, 90)

    def ticks_until(self, t_end: float):
        out = []
        while True:
            jitter = self.r.uniform(-1, 1) if self.regime == "fast" else self.r.uniform(-0.35, 0.35) * self.period
            nxt = self.t + self.period + jitter
            if self.regime == "slow" and self.r.random() < 0.08:
                nxt += self.period  # dropped frame
            if nxt > t_end:
                return out
            self.t = nxt
            out.append(nxt)


def gen_session(p: Dict[str, float], regime: str, r: random.Random, n_moves: int = None,
                n_symbols: int = None) -> Dict:
    noise = p.get("_noise", 1.0)
    slow = regime == "slow"
    reg_scale = r.uniform(2.5, 3.5) if slow else 1.0
    n_moves = n_moves or (36 if not slow else 28)
    n_symbols = n_symbols or (260 if not slow else 200)
    clock = _Clock(r, regime)
    ticks: List[list] = []
    t, x, y = 0.0, r.uniform(0.2, 0.8), r.uniform(0.2, 0.8)
    ticks.append([0.0, x, y, "move"])

    def pos_noise():
        return r.gauss(0, 0.0015 * noise)

    for _ in range(n_moves):
        while True:
            tx, ty = r.uniform(0.05, 0.95), r.uniform(0.05, 0.95)
            dist = math.hypot(tx - x, ty - y)
            if dist > 0.12:
                break
        ux, uy = (tx - x) / dist, (ty - y) / dist
        horiz = abs(ux) ** 2
        base = 520.0 * (dist / 0.4) ** p["fitts_b"]
        dur = base * p["speed"] * reg_scale * math.exp(-p["aniso"] * (horiz - 0.5)) * math.exp(r.gauss(0, 0.08 * noise))
        over = r.random() < p["over_p"]
        aim = 1.0 + (p["over_mag"] * r.uniform(0.6, 1.4) if over else 0.0)
        curv = p["curv"] * r.uniform(0.6, 1.4) + r.gauss(0, 0.015 * noise)
        hp = math.log(0.5) / math.log(min(0.95, max(0.05, p["asym"] + r.gauss(0, 0.02 * noise))))
        prim = dur * (0.82 if over else 1.0)
        t0 = t
        t_end = t0 + dur

        def at(tt):
            if tt <= t0 + prim:
                u = (tt - t0) / prim
                s = _minjerk(u ** hp) * aim
                lat = curv * math.sin(math.pi * min(1.0, s)) * dist
            else:  # corrective sub-movement back onto the target
                u = (tt - t0 - prim) / (dur - prim)
                s = aim + (1.0 - aim) * _minjerk(u)
                lat = 0.0
            px = x + ux * s * dist - uy * lat
            py = y + uy * s * dist + ux * lat
            return px, py

        for tt in clock.ticks_until(t_end):
            if tt <= t0:
                continue
            px, py = at(tt)
            ticks.append([tt, px + pos_noise(), py + pos_noise(), "move"])
        x, y = tx, ty
        dwell = dur * p["dwell_ratio"] * math.exp(r.gauss(0, 0.15 * noise))
        t_press = t_end + dwell
        for tt in clock.ticks_until(t_press):
            ticks.append([tt, x + pos_noise() * 0.3, y + pos_noise() * 0.3, "move"])
        ticks.append([t_press, x, y, "press"])
        t_rel = t_press + r.uniform(60, 120) * (1.6 if slow else 1.0)
        clock.t = max(clock.t, t_rel)
        ticks.append([t_rel, x, y, "release"])
        t = t_rel + r.uniform(80, 400) * reg_scale
        clock.t = max(clock.t, t)
    for tk in ticks:
        tk[0] = round(tk[0], 3)
        tk[1] = round(min(1.0, max(0.0, tk[1])), 5)
        tk[2] = round(min(1.0, max(0.0, tk[2])), 5)

    events = gen_events(p, regime, r, n_symbols, noise)
    return {"regime": regime, "ticks": ticks, "events": events}


def gen_events(p, regime, r, n_symbols, noise):
    slow = regime == "slow"
    iki = p["iki_ms"] * (r.uniform(2.0, 3.0) if slow else 1.0)
    events: List[list] = []
    t = 0.0
    typed = 0
    pending_err = None  # symbols remaining until the error is noticed
    sigma = 0.32 * noise
    while typed < n_symbols:
        mu = p["burst_mu"]
        blen = max(1, int(round(r.lognormvariate(math.log(mu), p["burst_cv"]))))
        chan = "DIRECT_KEY" if not slow else ("VIRTUAL_GRID" if r.random() < 0.5 else "BATCH_CHUNK")
        t += iki * p["pause_ratio"] * math.exp(r.gauss(0, 0.35))
        buf = 0
        buf_t = None
        chunk = r.randint(2, 6)
        for i in range(blen):
            t += iki * math.exp(r.gauss(0, sigma))
            typed += 1
            if pending_err is None and r.random() < p["edit_rate"]:
                pending_err = max(0, int(round(r.expovariate(1 / max(0.05, p["notice"])))))
            if chan == "BATCH_CHUNK":
                buf += 1
                if buf >= chunk or i == blen - 1:
                    events.append([t, chan, "INSERT", buf])
                    buf, chunk = 0, r.randint(2, 6)
            else:
                events.append([t, chan, "INSERT", 1])
            if pending_err is not None:
                if pending_err == 0:
                    if buf:
                        events.append([t, chan, "INSERT", buf])
                        buf = 0
                    if r.random() < p["edit_pause_p"]:
                        t += iki * p["pause_ratio"] * 0.6 * math.exp(r.gauss(0, 0.3))
                    else:
                        t += iki * math.exp(r.gauss(0, sigma))
                    if r.random() < p["replace_p"]:
                        events.append([t, chan, "REPLACE", r.randint(2, 6)])
                    else:
                        back = 1 + int(round(p["notice"])) + (1 if r.random() < p["over_del"] else 0)
                        if chan == "DIRECT_KEY" or chan == "VIRTUAL_GRID":
                            for _ in range(back):
                                events.append([t, chan, "BACKSPACE", 1])
                                t += iki * 0.45 * math.exp(r.gauss(0, sigma))
                        else:
                            events.append([t, chan, "BACKSPACE", back])
                    pending_err = None
                else:
                    pending_err -= 1
    for e in events:
        e[0] = round(e[0], 3)
    return events


# --------------------------------------------------------------------- splits
def build_population(seed: int, n_profiles=8, n_background=12, n_cal_strangers=10,
                     n_hold_strangers=20, n_mimic_pool=24):
    ids = {}
    for i in range(n_profiles):
        ids[f"P{i:02d}"] = draw_identity(rng_for(seed, "id", "P", i))
    for i in range(n_background):
        ids[f"B{i:02d}"] = draw_identity(rng_for(seed, "id", "B", i))
    for i in range(n_cal_strangers):
        ids[f"C{i:02d}"] = draw_identity(rng_for(seed, "id", "C", i))
    for i in range(n_hold_strangers):
        ids[f"H{i:02d}"] = draw_identity(rng_for(seed, "id", "H", i))
    for i in range(n_mimic_pool):
        ids[f"M{i:02d}"] = draw_identity(rng_for(seed, "id", "M", i))
    return ids


def session(ids, seed, split, who, regime, idx, cond="normal", target=None, alpha=0.0):
    r = rng_for(seed, split, who, regime, idx, cond, target, alpha)
    if cond == "mimic":
        p = make_mimic(ids[target], ids[who], alpha)
    else:
        p = dict(ids[who])
    if cond == "fatigued":
        p = fatigue(p, r)
    s = gen_session(p, regime, r)
    s.update({"split": split, "source": who, "condition": cond, "claimed": target or who, "idx": idx,
              "alpha": alpha})
    return s


REGIMES = ("fast", "slow")


def enrollment_split(ids, seed, n_per_regime=5):
    out = []
    for who in sorted(ids):
        if who[0] in "PB":
            for reg in REGIMES:
                for i in range(n_per_regime):
                    out.append(session(ids, seed, "enroll", who, reg, i))
    return out


def calibration_split(ids, seed, n_gen=4, n_mimic=4):
    profiles = sorted(k for k in ids if k[0] == "P")
    out = []
    for p in profiles:
        for reg in REGIMES:
            for i in range(n_gen):
                out.append(session(ids, seed, "cal", p, reg, i, "normal"))
                out.append(session(ids, seed, "cal", p, reg, i, "fatigued"))
            for i in range(n_mimic):
                m = f"M{(profiles.index(p) * n_mimic + i) % 12:02d}"  # M00..M11: calibration mimics
                out.append(session(ids, seed, "cal", m, reg, i, "mimic", target=p))
    for c in sorted(k for k in ids if k[0] == "C"):
        for reg in REGIMES:
            out.append(session(ids, seed, "cal", c, reg, 0, "stranger"))
    return out


def holdout_split(ids, seed, n_gen=8, n_mimic=8, alphas=(0.0,)):
    profiles = sorted(k for k in ids if k[0] == "P")
    out = []
    for p in profiles:
        for reg in REGIMES:
            for i in range(n_gen):
                out.append(session(ids, seed, "hold", p, reg, i, "normal"))
                out.append(session(ids, seed, "hold", p, reg, i, "fatigued"))
            for a in alphas:
                for i in range(n_mimic):
                    m = f"M{12 + (profiles.index(p) * n_mimic + i) % 12:02d}"  # M12..M23: fresh
                    out.append(session(ids, seed, "hold", m, reg, i, "mimic", target=p, alpha=a))
    for h in sorted(k for k in ids if k[0] == "H"):
        for reg in REGIMES:
            out.append(session(ids, seed, "hold", h, reg, 0, "stranger"))
    return out


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=2323)
    ap.add_argument("--out", default="traces")
    a = ap.parse_args(argv)
    ids = build_population(a.seed)
    os.makedirs(a.out, exist_ok=True)
    splits = {"enroll": enrollment_split(ids, a.seed), "calibration": calibration_split(ids, a.seed),
              "holdout": holdout_split(ids, a.seed)}
    for name, rows in splits.items():
        write_jsonl(os.path.join(a.out, f"{name}.jsonl"), rows)
        print(f"{name}: {len(rows)} sessions -> {os.path.join(a.out, name + '.jsonl')}")


if __name__ == "__main__":
    main()
