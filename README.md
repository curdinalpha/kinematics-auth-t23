# kinematics-auth-t23

Operator-continuity verifier from normalised 2D trajectory ticks plus
channel-tagged symbol bursts, for Stellarator Yard **t=23** (DISPATCH-06,
incidere/solana-agent-bounties#8).

Python 3.12 (also 3.13), standard library only, offline, deterministic.

```bash
python3.12 kinematics_auth.py      # one command: generate -> enroll -> calibrate -> freeze -> holdout -> table
```

## Input

```
ticks : [ts_ms, x, y, trigger_kind]   x, y in [0,1]; move | press | release
events: [ts_ms, channel, op, n]       DIRECT_KEY | VIRTUAL_GRID | BATCH_CHUNK (BATCH_INSERT alias)
                                      INSERT | BACKSPACE | REPLACE, n = symbols in the event
```

Malformed input (x outside [0,1], unknown kind/channel/op, time going
backwards, n < 1) is rejected with a clear error.

## Features (14, none of them depend on sensor rate, absolute speed or channel)

Trajectory, per reach (release -> press), aggregated per segment:
`curv` signed lateral bow, `half` time fraction to cover half the distance
(interpolated crossing), `over` overshoot, `home` homing fraction,
`dwell` log settle/reach time, `fitts_b` log-duration vs log-distance slope,
`aniso` horizontal vs vertical speed term. Durations only appear as ratios or
log-log slopes, so a uniformly faster device or operator gives identical
values.

Bursts, counted in symbols: `blen`, `bcv` burst length and spread, `pause`
pause in units of the session's own per-symbol interval, `edit` corrections
per symbol, `bs_len` symbols deleted per backspace run, `repl` REPLACE
share, `epause` corrections that start after a pause. Gaps are measured as
`dt - (n-1)*m` where `m` is the robust per-symbol interval, so one 4-symbol
BATCH_CHUNK and 4 DIRECT_KEY presses produce the same bursts.

Tested invariances: timestamps x3.7 and +1e7 ms give exactly the same
features, relabelling the channel changes nothing, re-chunking DIRECT_KEY into
3-symbol BATCH_CHUNKs moves burst features by < 0.6 population spreads, and
dropping 6 of every 7 ticks (8 ms -> 56 ms) moves trajectory features by
< 0.75 population spreads on average.

## Scoring

Each session is split into 4 segments. Per feature, the log-likelihood ratio
of the claimed profile (robust median/MAD, widened by a drift allowance) vs
the population is clipped to [-6, 3] and averaged. The sequential score is
the running mean over segments, and the decision uses the final value.

## Protocol (no leakage)

1. **Enroll**: 5 normal sessions per regime for each of 8 profiles, plus 12
   background identities for the population model.
2. **Calibrate**: separate sessions: genuine normal + fatigued, 10 calibration
   strangers, mimics drawn from mimic pool M00-M11. The threshold minimises
   FAR+FRR here only.
3. **Freeze**: the threshold, model sha256 and calibration-score sha256 go
   into a `Frozen` token. `evaluate()` refuses to run without a token that
   matches the model.
4. **Holdout**: generated only after the freeze, with 20 fresh strangers and
   fresh mimics (M12-M23). Tests check that `calibrate` runs before the holdout
   generator, and that a different holdout seed leaves the threshold
   byte-identical.

Attackers in both regimes (fast: 8 ms ticks + DIRECT_KEY; slow: 45-90 ms
jittery ticks with dropouts, 2.5-3.5x slower reaches, VIRTUAL_GRID +
BATCH_CHUNK):

* fatigued self: the same identity, slower and noisier, with more pauses and
  edits and later error notice. It must be accepted.
* random stranger: tested against all 8 models.
* parameter-matched mimic: copies the target's visible aggregates (speed,
  symbol interval, burst length, pause ratio, edit rate, dwell) but keeps
  its own motor and rhythm structure.
* full 8x8 swap matrix per regime: every model against every profile's
  holdout sessions.
* mimic-strength sweep: alpha blends all hidden parameters toward the target
  (alpha=1 is a clone).

## Results (seed 2323, `evidence/bench-seed2323-sweep10.txt`)

Frozen threshold -0.114291 (calibration FAR 0.0268, FRR 0.0156).

| regime | class | metric | errors/n | rate | 95% upper |
|---|---|---|---|---|---|
| fast | genuine normal | FRR | 1/64 | 0.0156 | 0.0833 |
| fast | fatigued self | FRR | 1/64 | 0.0156 | 0.0833 |
| fast | random stranger | FAR | 4/160 | 0.0250 | 0.0625 |
| fast | param mimic | FAR | 4/64 | 0.0625 | 0.1500 |
| fast | swap off-diagonal | FAR | 11/448 | 0.0246 | 0.0434 |
| slow | genuine normal | FRR | 1/64 | 0.0156 | 0.0833 |
| slow | fatigued self | FRR | 6/64 | 0.0938 | 0.1898 |
| slow | random stranger | FAR | 0/160 | 0.0000 | 0.0234 |
| slow | param mimic | FAR | 4/64 | 0.0625 | 0.1500 |
| slow | swap off-diagonal | FAR | 8/448 | 0.0179 | 0.0348 |
| **pooled** | | **FRR** | **9/256** | **0.0352** | 0.0655 |
| **pooled** | | **FAR** | **31/1344** | **0.0231** | 0.0326 |

* Swap matrix diagonal: 126/128 accepted. Off-diagonal: 19/896, and 17 of
  those come from one near-twin pair (P01/P02). The other 54 off-diagonal
  pairs have 2 accepts in total.
* Mimic strength (accepted/128): alpha 0: 8, 0.25: 31, 0.5: 68, 0.75: 118,
  1.0: 128. Copying visible stats is not enough, and a clone is
  indistinguishable, which is the expected behaviour.
* Sequential after 1/2/3/4 segments: FRR 0.156/0.102/0.051/0.035, FAR
  0.060/0.042/0.029/0.027.
* **10 more seeds** (3323...12323, each with fresh identities and its own
  calibration): FAR 298/13440 = 0.0222, FRR 111/2560 = 0.0434.
* ~13 s per full run. The report is byte-identical across runs
  (`evidence/determinism-3runs.txt`) and runs without network (`unshare -rn`).

These numbers are not zero, and that is expected: the mimic copies
everything a casual observer can measure, and the population has
near-twins. All numbers come from the frozen threshold on unseen data.

## Tests: 39 unittest cases, all passing (py3.12.16 and 3.13.5)

| file | what |
|---|---|
| `test_features.py` (18) | validation, reach geometry, exact edit accounting, time scale/shift invariance, channel relabel, re-chunking, tick downsampling, cross-regime identity, synth determinism, disjoint identity pools, mimic construction |
| `test_pipeline.py` (17) | calibrate-before-holdout order, threshold independent of holdout seed, `Frozen` guard, determinism, spec coverage, exact table arithmetic, swap matrix consistency, mimic sweep, calibrate/Wilson/LLR/drift unit tests |
| `test_cli.py` (4) | no network modules imported, byte-identical JSONL regeneration, `bench --data` equal to the in-memory run, enroll/score round trip, bad input exits non-zero |

```bash
python3.12 -m unittest discover -s tests -t . -v           # 39 tests, ~80 s
python3.12 synth_traces.py --seed 2323 --out traces        # JSONL splits
python3.12 kinematics_auth.py bench --data traces --seed 2323
python3.12 kinematics_auth.py bench --sweep 10             # multi-seed table
python3.12 kinematics_auth.py enroll --data traces/enroll.jsonl --out model.json
python3.12 kinematics_auth.py score --model model.json --data traces/holdout.jsonl
```

MIT licensed.
