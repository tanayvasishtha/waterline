# Waterline: technical report

Camera-side verification of river telemetry. Waterline reads the staff gauge a
USGS camera already points at, compares every reading to the official gage, and
asks a person to look when the two disagree or the picture cannot be trusted.
This report covers the method, the agent policy, the fault detector, the
evaluation protocol and its amendments, the development history, and the final
held-out results. Every number below comes from a committed file; where the
result lives only in git history, the commit is named.

## Method

### Registration

Each frame is registered to a site reference by SIFT (4000 features, Lowe ratio
0.7, RANSAC reprojection 4 px) and accepted when it keeps at least 60 inliers at
an inlier ratio of 0.35 or better (`src/waterline/register.py`, thresholds in
`src/waterline/settings.py`). Registration pins the frame into reference
coordinates, so the plate quad and calibration stay valid as the camera shifts.
A shift of more than 8 px in the registered plate corners means the camera moved
and the reference pose no longer applies.

### Per-reference quads and wet references

Two of the three evaluated sites fail registration on wet frames against a
dry-only reference bank, because high water submerges the keypoint texture. Site
configuration therefore carries a bank of references, including a wet/high-water
pose, and each reference can carry its own plate quad (`ref_quads` overrides
`plate_quad` per reference name). Registration returns which reference matched,
and rectification uses that reference's quad.

### Rectification and detection

A homography warps the frame into reference space, and a second homography
rectifies the plate quad to a tall, narrow strip (48 px wide, about 120 px per
foot of plate). The waterline row is found by changepoint detection down the
strip, requiring a minimum contrast of 0.8 and a minimum segment of 40 px. A
second, independent read (method B, lowest visible tick) is computed and
recorded but does not gate acceptance; see Amendment 4 below.

### Quality gate

A frame passes the gate only when all of: Laplacian blur variance >= 40, glare
fraction <= 0.10, not night (mean saturation above 12 separates IR-night frames),
plate tick visibility >= 3.0, and registration inliers >= 60. The gate output
drives the agent: a numeric reading can be accepted only if the frame it came
from passed the gate.

### Calibration and sigma

Stage is an affine fit `stage = a + b * detected` per site, least squares on
calibration-window (detected, gage) pairs, n >= 4. Hancock is the offset case
`[-0.396, 1.0]`; in_overland needed slope 1.414 because its ruler is
perspective-compressed and the configured marks were approximate (a ~0.63x
scale error that grows with stage; `eval/PROTOCOL.md` Amendment 2). Reading
sigma combines the detector's spread with the calibration uncertainty, scaled
by |b|, floored at 0.05 ft; reads wider than 0.25 ft escalate inside the agent.

## Agent policy

The agent is a deterministic policy over the vision result, not a scripted
pipeline and not an LLM: a chooser maps the last observation to the next tool
call, with a 6-step budget (`agent_step_budget`). The tools are enhance
(CLAHE, gamma), glare mask/inpaint, re-register on an alternate reference,
fetch a neighbour frame, and a daylight substitute for night frames. Repairs
are applied speculatively and reconciled: a repair is rejected when it loses
the reading or gains nothing, and the rejection, not the repair, is recorded.

Invariants (v3d, frozen):

- A numeric read is accepted only if the frame it came from passed the quality
  gate; a gate-failing accept becomes an abstain naming the failed checks.
- A gate-failing frame counts as "no usable reading" even if a stage value was
  produced; it takes the substitution path, and the substitute is accepted only
  if the substitute itself passes the gate.
- Neighbour fetches on an uncertain-but-readable frame confirm the reading
  rather than replacing it; sigma-uncertain frames (sigma > 0.25 ft) cannot be
  confirmed this way; confirmation is bounded by the site's p99 plausible rate.
- A substituted read is scored once, against the gage at the neighbour's time,
  and deduplicated by source frame (`agent_frame_file` in the CSV).

Bedrock is an optional chooser. The final evaluation ran the deterministic
policy only; all 471 committed traces say `policy: deterministic`.

## Fault detector

Four rules run per accepted-read frame, in fixed order: spike, stuck, step,
drift (`src/waterline/agent/fault.py`, thresholds in `settings.py`).

- **Spike**: a one-frame gage reversal larger than `spike_min_ft` (set to twice
  the largest one-frame reversal in the calibration records of the two fault
  sites) that recovers by the next read.
- **Stuck**: four identical gage values while the camera's first-to-last change
  over the window exceeds 3 combined sigmas.
- **Step**: the gage moves and the camera disagrees, by more than 3 combined
  sigmas.
- **Drift**: a z-scored CUSUM over camera-minus-gage residuals (residual divided
  by each reading's sigma), textbook k = 0.5, h = 5.

Flags carry persistence (consecutive frames seen), evidence frames, and a draft
review note. The injected-fault scorer replays the real record with a seeded
synthetic fault (20 per type per window, seed 20261026); a flag counts as a
detection only when it does not also occur on the clean series at the same
index and type (Amendment 3b), and recall is scored within 24 rows of onset as
well as at any later frame.

## Protocol and amendments

`eval/PROTOCOL.md` fixes the split before the final run. Calibration windows
are the earliest 30 days in the 12 months before the test windows; test windows
are all remaining frames in the same months, subsampled. The headline changes:

- Amendment 2: affine stage calibration; in_overland's only wet stretch is the
  one the affine was fit on, so its accuracy numbers are partially in-sample
  and hancock's disjoint split is the headline claim.
- Amendment 3: the detector was rewritten after a clean replay produced false
  flags (unstandardised CUSUM, a step rule that ignored the camera); the spike
  threshold was set from calibration reversals.
- Amendment 3b: clean-run flags no longer count as injected-fault detections;
  the stuck rule moved from a 2-sigma spread test to a 3-sigma first-to-last
  change test.
- Amendment 4: method B stopped gating acceptance (it carried no information
  about method A's error on the calibration windows), and agent metrics count
  accepted reads only; the v1 results were re-run and both are reported.
- Amendment 5: the previously examined windows became calibration data; fresh
  test windows were picked from the gage record alone and are enforced by a
  protocol hash check inside `eval --final`.
- Amendment 6: freeze before the final run at tag `freeze-final`
  (`b370871`); any later change needs a new amendment and a full repeat.

## Development history

Agent versions were developed on the calibration windows only.

- v1 reported a hancock dev MAE of **0.47 ft** (0.4717, commit `06aea81`,
  `summary_eval_agent_hancock_2026-07-27_2026-08-01.json`). It was wrong: the
  error metric counted abstained readings that still carried a stage value, so
  the easy reads were scored and the hard ones skipped. After Amendment 4 the
  same window under accepted-only scoring reads 0.6456 (v2, commit `05aa1a4`).
  The 0.47 number stays in the history and is retracted here.
- v3 added repair self-consistency (a repair that loses the reading or gains
  nothing is discarded) and camera-history plausibility.
- v3b made neighbour fetches confirm rather than replace; dev hancock MAE
  0.5749 (commit `2bab1af`) because night frames confirmed each other.
- v3c stopped sigma-uncertain reads from being neighbour-confirmed; dev
  hancock 0.5691 (commit `260b4c2`) still beaten by baseline 0.3262.
- v3d added the terminal gate invariant and sent gate-failing frames down the
  substitution path; dev hancock 0.3086, back under baseline. The v3c/v3b
  regression was night frames with a numeric stage being "confirmed" by
  neighbours that were also night frames.

## Final results

Held-out windows, frozen at `freeze-final`; pooled numbers are in
`results/final_summary_{site}.json` and `results/final_faults_{site}.json`.

Accuracy (baseline | agent, MAE ft):

| site | frames | baseline | agent | base reads | agent reads | pending | within 2 sigma | coverage |
|---|---|---|---|---|---|---|---|---|
| hancock | 295 | 0.307 | 0.290 | 170 | 172 | 29 | 98.2% | 57.6% |
| in_overland | 1356 | 0.032 | 0.032 | 218 | 222 | 2 | 95.4% | 16.1% |

Day/night at hancock: by day the agent reads 169 frames at 0.294 MAE (baseline
170 at 0.307). At night the baseline gate rejects all 66 frames; the agent
abstains on 34, escalates 27, and accepts 3, all via daylight substitution
(MAE 0.029 on those three). Night is where the agent earns its keep, at a cost
of sending 27 frames to a person.

in_overland by year (coverage declines with frame age; references are newer):

| year | frames | gate pass | baseline MAE | agent MAE | coverage |
|---|---|---|---|---|---|
| 2023 | 401 | 185 | 0.031 | 0.032 | 10.7% |
| 2024 | 566 | 269 | 0.032 | 0.032 | 9.4% |
| 2025 | 213 | 123 | 0.029 | 0.029 | 22.1% |
| 2026 | 176 | 114 | 0.035 | 0.035 | 42.6% |

Injected faults (pooled; per type: caught within 24 rows, median delay h,
type named right):

| site | drift | stuck | step | spike |
|---|---|---|---|---|
| hancock | 95%, 27.0 h, 100% | 58%, 44.0 h, 0% | 100%, 5.0 h, 0% | 100%, 0.0 h, 25% |
| in_overland | 13%, 4.7 h, 96% | 20%, 2.4 h, 10% | 48%, 0.0 h, 80% | 100%, 0.3 h, 77% |

Clean-record flags: hancock 8 (all drift, all negative residual), in_overland
4 (3 drift, 1 step) in `results/final_flags_{site}.json`.

Determinism: `scripts/final_traces.py` replayed every hancock frame and the
2026 in_overland windows through the frozen agent; 471 traces, 0 mismatches on
status, stage (1e-9) and source frame.

## Failure analysis

The gallery (`results/gallery/`, rule-picked by `scripts/failure_gallery.py`)
covers the six failure modes plus largest error. Where the system hurts:

- in_overland is nearly dry; most windows are below plate or fail the gate, so
  coverage is 16% pooled and the fault detector starves (drift recall 13%).
- Hancock's 8 clean flags are all drift with negative residuals, camera reading
  0.33 to 0.86 ft low, a persistent bias rather than a one-off error.
- 29 hancock frames (9.8% of the window) went to a person, mostly at night.
- Two frames listed by NIMS are absent from S3 and were skipped; the disclosure
  is in `results/FINAL_RUN_NOTES.md`.

## Threats to validity

- Hancock's test windows span 10 to 12 ft stage; they say nothing about flood
  accuracy. The July flood frames are calibration data now (Amendment 5) and
  any flood-stage number would be in-sample.
- in_overland's affine fit shares its wet period with the test windows
  (Amendment 2 disclosure).
- The USGS gage is the assumed truth; clean-record flags could be gage errors
  the camera caught, and the scorer cannot tell them apart.
- Small n: 2 hancock windows, 21 short in_overland episodes, 3 sites screened.
- The detector's 24-row recall window understates delay at hancock, where rows
  are hourly but episodes are long (stuck median delay 44 h).

## Reproduce and verify

```powershell
uv sync
uv run python scripts/gate.py
uv run waterline eval --final
uv run waterline faults --final
uv run python scripts/final_traces.py
```

The protocol hash (`results/protocol_hash.txt`) is checked inside
`eval --final`; a change to `eval/PROTOCOL.md` refuses the run.
