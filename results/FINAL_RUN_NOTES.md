# Notes on the final run

The final run used the code and configuration at the commit tagged freeze-final. Three
things happened during it that a reader should know about.

1. The first attempt stopped after about 1.2 hours when the machine ran out of memory
   inside SIFT feature extraction. Nothing in the frozen code was changed. The remaining
   41 of 46 runs (one per site, window and mode) were run one at a time in fresh
   processes by scripts/resume_final.py. The 5 finished runs were kept, not repeated.
   Peak memory per run was about 1.1 GB.
2. Two in_overland frames are listed by the USGS imagery API but are missing from its
   storage (HTTP 404): 2024-04-02T15:20:26Z and 2024-04-12T16:25:23Z. The frozen
   evaluation loop crashed on a missing image instead of skipping it. The resume script
   skips them, and they are listed in results/final_missing_frames.txt. Both the
   baseline and the agent skip the same two frames.
3. The pooled fault command also treated the exclude_files entry in
   results/test_windows.json as a site and wrote an empty file for it. That file was
   deleted. It did not affect any reported number.

Fixing the last two in the package would need a protocol amendment and a full repeat
of the final run, so they are left as they are and disclosed here.
