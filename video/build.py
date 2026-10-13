"""Assemble video/waterline.mp4 from video/edl.json.

edl.json is an ordered list of segments:
  {"clip": "clips/shot01_hancock_all.mp4", "duration": 6,
   "caption": "on-screen caption", "vo": "spoken sentence"}

Per segment the clip is looped or trimmed to max(clip length, voiceover + 0.5 s),
the caption is burned in with a Consola face in the dashboard palette, and a
System.Speech (SAPI) wav is mixed in as the voiceover track. Segments are
concatenated into video/waterline.mp4. Voice clips land in video/wav/.

Run from the repo root: .venv/Scripts/python.exe video/build.py
Requires ffmpeg/ffprobe on PATH and Windows SAPI (Microsoft Zira Desktop).
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO = Path(__file__).parent
EDL = VIDEO / "edl.json"
OUT = VIDEO / "waterline.mp4"
WAV = VIDEO / "wav"
FONT = r"'C\:/Windows/Fonts/consola.ttf'"
VOICE = "Microsoft Zira Desktop"
FPS = 30
VO_PAD = 0.5


def _sh(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return r.stdout.strip()


def _dur(path: Path) -> float:
    return float(
        _sh(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(path),
            ]
        )
    )


def _tts(text: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    safe = text.replace("'", "''").replace('"', '""')
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Rate = 0; "
        f"$s.SelectVoice('{VOICE}'); "
        f"$s.SetOutputToWaveFile('{out.as_posix()}'); "
        f"$s.Speak('{safe}'); "
        "$s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, check=True)
    return out


def _escape_drawtext(text: str) -> str:
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace("%", "\\%")


def _wrap_caption(text: str) -> list[str]:
    """Long captions become two lines; short ones stay one."""
    if len(text) <= 64:
        return [text]
    mid = len(text) // 2
    cuts = [i for i, c in enumerate(text) if c == " "]
    cut = min(cuts, key=lambda i: abs(i - mid)) if cuts else mid
    return [text[:cut].strip(), text[cut:].strip()]


def _segment(idx: int, seg: dict, work: Path) -> Path:
    clip = VIDEO / str(seg["clip"])
    vo_text = str(seg.get("vo") or "").strip()
    caption = str(seg.get("caption") or "").strip()

    clip_dur = _dur(clip)
    wav = _tts(vo_text, WAV / f"seg{idx:02d}.wav") if vo_text else None
    vo_dur = _dur(wav) if wav else 0.0
    # segment lasts the requested duration or long enough for the voiceover;
    # footage plays through that whole span, holding the last frame only if
    # the clip is shorter than the segment
    dur = max(float(seg.get("duration") or 0), vo_dur + VO_PAD)

    vclip = min(clip_dur, dur)
    hold = dur - vclip  # voiceover overrun: clone the last frame, never loop
    vchain = f"[0:v]trim=0:{vclip:.3f},setpts=PTS-STARTPTS"
    if hold > 0.05:
        vchain += f",tpad=stop_mode=clone:stop_duration={hold:.3f}"
    vchain += (
        ",scale=1920:1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0xf6f4ee,"
        f"fps={FPS}"
    )
    cap_lines = _wrap_caption(caption) if caption else []
    for i, line in enumerate(cap_lines):
        n = len(cap_lines)
        y = "h-150" if n == 2 and i == 0 else ("h-95" if n == 2 else "h-110")
        cap_txt = _escape_drawtext(line)
        vchain += (
            f",drawtext=fontfile={FONT}:text='{cap_txt}':fontsize=34:"
            "fontcolor=0x15171a:box=1:boxcolor=0xf6f4ee@0.92:boxborderw=16:"
            f"x=(w-text_w)/2:y={y}"
        )
    filters = [vchain + ",format=yuv420p[v]"]
    if wav:
        filters.append(f"[1:a]aresample=48000,apad=whole_dur={dur:.3f}[a]")
        inputs = ["-i", str(clip), "-i", str(wav)]
    else:
        filters.append(f"anullsrc=r=48000:cl=stereo:d={dur:.3f}[a]")
        inputs = ["-i", str(clip)]

    out = work / f"seg{idx:02d}.mp4"
    _sh(
        [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-t",
            f"{dur:.3f}",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(out),
        ]
    )
    print(f"seg{idx:02d} clip={clip_dur:.1f}s vo={vo_dur:.1f}s -> {dur:.1f}s")
    return out


def main() -> int:
    if not EDL.exists():
        print(f"missing {EDL}")
        return 1
    edl = json.loads(EDL.read_text())
    work = VIDEO / "build"
    work.mkdir(parents=True, exist_ok=True)
    parts = []
    for idx, seg in enumerate(edl):
        if not (VIDEO / str(seg["clip"])).exists():
            print(f"missing clip {seg['clip']}")
            return 1
        parts.append(_segment(idx, seg, work))
    lst = work / "concat.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
    _sh(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(lst),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(OUT),
        ]
    )
    total = _dur(OUT)
    print(f"{OUT} {total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
