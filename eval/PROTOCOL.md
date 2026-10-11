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
