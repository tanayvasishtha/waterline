# Waterline evaluation protocol

Pre-registered before any test-set run. This file may only be extended, never rewritten:
anything tuned on test windows would invalidate the honesty claim. The git history is the
audit trail; `results/protocol_hash.txt` records the sha256 of this file, and
`waterline eval --final` refuses to run if the file has changed since.

## Sites

| site_id | camera | nwis id | role |
|---|---|---|---|
| hancock | NY_East_Branch_Delaware_River_at_Hancock_Staff | 01421500 | flagship |
| oh_wetland | OH_Inflow_to_wetland_near_Winn_Road_near_Defiance_STAFF | 411607084241200 | generalization (flume) |
| pa_youghiogheny | PA_Youghiogheny_River_at_Youghiogheny_River_Dam | 03077500 | generalization (mid-river board) |

Site choice rationale (recorded 2026-10-11): results/site_screen.json ranked all
staff cameras by range, event count and cadence; docs/media/site_screen_day.jpg is the
daylight contact sheet used for the final readability check.

## Ground truth

Primary: USGS continuous gage height, parameter 00065, OGC collection `continuous`,
aligned to each frame within 10 minutes. Approved values are scored separately from
Provisional where sample size allows; all are labeled.

Secondary: USGS `field-measurements` (technician wire-weight readings) matched within
30 minutes.

## Calibration versus test split

- Calibration window, per site: the earliest 30 consecutive days in the 12 months before
  evaluation whose stage range is at least 1.0 ft. Used only for: datum offset fit,
  quality-threshold fit (blur, glare), sigma calibration check.
- Test windows: all remaining frames in the same 12 months, subsampled to at most 1500
  frames per site, stratified across hours of day and stage quintiles, plus every frame
  within plus or minus 48 hours of each of the 3 largest peaks.
- Held-out event (never calibrated on, never browsed for tuning): Hancock,
  2026-07-27T00:00Z to 2026-07-31T23:59Z.

## Conditions

- The agent and every threshold are tuned only on calibration windows.
- The single-pass baseline is the identical pipeline with the agent disabled (no re-read,
  no alternate reference, no adjacent frame, forced accept-or-abstain by the same triage
  quality gate).
- Fault detection is evaluated on injected faults only: drift (0.005 ft per 15 min),
  stuck value, step (+0.8 ft), spike (+3 ft for 1 frame), all labeled "injected" in
  reports, with seed 20261026.

## Calibration checkpoint notes (recorded 2026-10-11)

- The camera-to-sensor bias is regime-dependent: fitted +0.40 ft on the August
  low-flow window, but the held-out July flood frames read about 0.6 ft low after
  that same offset. Water piles up against the plate differently at different
  stages; this is physical, not a calibration bug. We therefore report a single
  site offset and quote honest residual spread, and the agent flags frames whose
  camera-sensor disagreement exceeds the residual band.
- Night frames are cleanly separable by tick-pattern visibility (SNR ~0 vs ~45).
  The gate abstains on all of them. Readings at night fall to the agent's
  fetch-adjacent action or abstention, never a guess.
- Glare is not a gate: the pale plate itself trips absolute-whiteness metrics,
  so glare is an agent signal (choose mask_glare re-read), not a rejection.

## Metrics

Per site and pooled: MAE, median AE, P90 AE (ft), coverage (read frames / all frames,
day and night separately), abstain rate, 2-sigma calibration, agent versus single-pass
delta, fault recall, false flags per site-week, median detection delay, p50 and p95
latency per frame, Bedrock calls and cost per 1000 frames.

## Reporting rule

Every number in the README, report, video, and submission comes from `results/*.json`
written by scripts in this repo, with the command recorded beside it.

## Amendments (recorded 2026-10-12)

These notes record decisions made during M3; they extend the protocol above and
do not rewrite it. All fits below used calibration windows only.

### Site set update

- `in_overland` (IN_Overland_Flow_East_of_Bull_Rapids_Road_near_Harlan_STAFF,
  nwis 411229084541101) replaces `pa_youghiogheny`: the Youghiogheny board's
  marks were not legible at archive resolution, so it cannot support an honest
  evaluation. `oh_wetland` is retained as a coverage-limited site: registration
  works with a wet reference, but its tape lacks a clean row-wise boundary
  signal when submerged, so it contributes to coverage/abstention metrics, not
  accuracy tables.
- High-water reference frames are part of site configuration. Two of three
  sites fail registration on wet frames against a dry-only bank because the
  water submerges the keypoint texture; a "wet_high" reference fixes it.
  Reference quads may differ per reference pose (`ref_quads` overrides
  `plate_quad` per reference name).

### Affine stage calibration

- `stage_calib = [a, b]` replaces the flat `datum_offset_ft`: stage =
  a + b * detected. Hancock's fit is [-0.396, 1.0]
  (slope 1, the offset case). in_overland needed slope 1.414: its ruler is
  perspective-compressed and the config top/bottom marks were approximate,
  which produced a ~0.63x scale error growing with stage - the affine fit
  corrects it. Sigma is scaled by |b|.
- Fit procedure unchanged: calibration window rows only, least squares on
  (detected, gage) pairs, n >= 4.

### in_overland window-overlap disclosure

in_overland's only wet stretch in the archive (Sep 19-23, peak 1.11 ft) is the
same stretch the affine was fit on; no disjoint wet period exists for that
site. Its accuracy numbers are therefore partially in-sample for the affine
parameters and are reported as supporting evidence, while hancock's
calibration/test split remains fully disjoint and is the headline claim.

## Amendment 3 (2026-10-11): fault detector redesign

A replay of the held-out Hancock event with no injected fault produced three flags
(one drift, two step) while the USGS gage was correct. Cause: the step rule compared
consecutive gage values without checking the camera, and the drift CUSUM used
unstandardised residuals. The detector was rewritten before any injected-fault run.
The step rule now requires the gage change to disagree with the camera change by more
than 3 combined sigmas. CUSUM runs on residuals divided by each reading's sigma, with
textbook k = 0.5 and h = 5. The stuck rule requires the camera to move by more than
2 sigma while the gage holds still. A spike rule was added, with its threshold set to
twice the largest one-frame reversal in the calibration-window gage records of the
two sites under fault evaluation (hancock and in_overland). oh_wetland was left out
of that pool because its September record contains a single gage reading of 0.0 ft
between 1.31 and 0.97 ft, a dropout of exactly the kind the rule exists to catch.
None of these values was tuned on a test window. Because the trigger for this change was seen
on a test window, the clean-replay false-flag count on that window is reported, but it
is not an independent estimate. Agent results are now written to their own files
(eval_agent_*) because an earlier agent run had been overwritten by a baseline rerun.

## Amendment 3b (2026-10-11): scoring and stuck-rule corrections

Two defects were found in review before any fault result was reported. First, the
fault harness counted a flag that also appears on the clean, uninjected series as a
detection of the injected fault. Detections now exclude any flag whose frame index and
type match a clean-run flag, and recall is reported within 24 frames of onset as well
as at any later frame. Second, the stuck rule confirmed a frozen gage when the camera's
spread over four frames exceeded 2 sigma, which ordinary noise does about half the
time. It now requires the camera's first-to-last change over the window to exceed
3 combined sigmas, the same test the step rule uses.

## Amendment 4 (2026-10-11): method B no longer gates acceptance

Method B (lowest visible tick) was used as a cross-check: when it differed from method
A by more than 2 sigma, the agent treated the reading as uncertain and abstained or
asked for review. On the calibration windows it carries no information about method A's
error. On the nine wet in_overland calibration frames, B minus A ranged from -0.69 to
+0.05 ft while A's error held between -0.22 and -0.14 ft, because clear shallow water
leaves ticks visible below the surface. On the two near-dry frames A was wrong by about
+0.36 ft and B did disagree with it; those cases now rely on the below-plate test alone.
At hancock, B sat 0.6 to 1.0 ft above A on every gated frame regardless of A's error.
The cross-check was removed from the acceptance rule; method B is still computed and
recorded. This amendment was committed before the agent was re-run. The re-run (agent
v2) is the primary agent result whichever way it comes out. The v1 results remain in
the history at commit 06aea81, and both are reported. Agent read counts and error
metrics now include accepted readings only; v1 had also counted abstained readings
that carried a stage value.

## Amendment 5 (2026-10-11): fresh test windows; old test windows become calibration

The July to August hancock windows and the September in_overland window have been
examined repeatedly during development, so they can no longer serve as test data. From
this amendment on they are calibration data. The new test windows are listed in
results/test_windows.json. They were chosen from the gage record alone, before any frame
in them was processed: for hancock, 2026-04-14 to 2026-05-28 and 2026-09-01 to
2026-09-16; for in_overland, every on-plate episode since the camera came online that
has at least 50 frames and does not overlap a calibration window, padded by 6 hours on
each side. Reference frames are excluded from test windows. oh_wetland has no fresh
on-plate episode with frames, so it gets no new test window. The evaluation command
refuses these windows unless run with --final, and --final refuses to run if this
file's SHA-256 differs from results/protocol_hash.txt. The final run happens once,
after the agent and calibration are frozen. Its results are reported whatever they
show, together with all earlier results. The hancock test windows stay between 10 and
12 ft, so the fresh results say nothing about flood-stage accuracy. Flood-stage numbers
come only from the July event, which is now calibration data, and are labelled
in-sample.

## Amendment 6 (2026-10-13): freeze before the final run

The agent and the calibrations are frozen at the commit tagged freeze-final. Agent v3d
(repair self-consistency, camera-history plausibility, neighbour confirmation only for
frames that are self-inconsistent or implausible, and an accept gate that requires the
source frame of a reading to pass the quality gate) was developed on the calibration
windows only, through versions v3a to v3d, and each version's calibration-window
results are committed. Site calibrations are the affine fits in results/calibfit_*.json.
When a neighbour frame's reading replaces a frame's reading, it is scored once, against
the gage at the neighbour's time, even if that neighbour is also in the frame list. The
primary results are the --final run over results/test_windows.json: baseline and agent,
with injected-fault scoring on the agent series, pooled per site as described in
results/final_summary_*.json and results/final_faults_*.json. They are reported
whatever they show. Any change after this point needs a new amendment and a full repeat
of the final run, and both runs are reported.
