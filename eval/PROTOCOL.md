# Waterline evaluation protocol

Pre-registered before any test-set run. This file may only be extended, never rewritten:
anything tuned on test windows would invalidate the honesty claim. The git history is the
audit trail; `results/protocol_hash.txt` records the sha256 of this file, and
`waterline eval --final` refuses to run if the file has changed since.

## Sites

| site_id | camera | nwis id | role |
|---|---|---|---|
| hancock | NY_East_Branch_Delaware_River_at_Hancock_Staff | 01421500 | flagship |
| site_b | TBD after daylight contact sheet review | | generalization |
| site_c | TBD after daylight contact sheet review | | generalization |

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

## Metrics

Per site and pooled: MAE, median AE, P90 AE (ft), coverage (read frames / all frames,
day and night separately), abstain rate, 2-sigma calibration, agent versus single-pass
delta, fault recall, false flags per site-week, median detection delay, p50 and p95
latency per frame, Bedrock calls and cost per 1000 frames.

## Reporting rule

Every number in the README, report, video, and submission comes from `results/*.json`
written by scripts in this repo, with the command recorded beside it.
