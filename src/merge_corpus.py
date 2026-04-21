#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from pathlib import Path

CORPUS = Path(__file__).parent / "miamiCorpus"
OUT_MP3 = Path(__file__).parent / "miamiCorpus_merged.mp3"
OUT_VTT = Path(__file__).parent / "miamiCorpus_merged.vtt"

TS_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{3})|(\d{1,2}):(\d{2})[.,](\d{3})")


def parse_ts(s: str) -> float:
    m = TS_RE.fullmatch(s.strip())
    if not m:
        raise ValueError(f"bad timestamp: {s!r}")
    if m.group(1) is not None:
        h, mi, se, ms = m.group(1, 2, 3, 4)
        return int(h) * 3600 + int(mi) * 60 + int(se) + int(ms) / 1000.0
    mi, se, ms = m.group(5, 6, 7)
    return int(mi) * 60 + int(se) + int(ms) / 1000.0


def fmt_ts(t: float) -> str:
    if t < 0:
        t = 0.0
    ms_total = int(round(t * 1000))
    h, rem = divmod(ms_total, 3600_000)
    mi, rem = divmod(rem, 60_000)
    se, ms = divmod(rem, 1000)
    return f"{h:02d}:{mi:02d}:{se:02d}.{ms:03d}"


def ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    )
    return float(out.strip())


def shift_vtt(text: str, offset: float) -> list[str]:
    cues: list[str] = []
    block: list[str] = []
    # Drop the WEBVTT header and any NOTE blocks by scanning cue by cue.
    lines = text.splitlines()
    i = 0
    # Skip header: WEBVTT and any metadata until the first blank line.
    if lines and lines[0].lstrip().startswith("WEBVTT"):
        i = 1
        while i < len(lines) and lines[i].strip() != "":
            i += 1
        # skip the blank separator
        while i < len(lines) and lines[i].strip() == "":
            i += 1

    def flush(block: list[str]):
        if not block:
            return
        # Find the timing line (contains "-->").
        timing_idx = next((k for k, l in enumerate(block) if "-->" in l), None)
        if timing_idx is None:
            return
        timing = block[timing_idx]
        left, _, right = timing.partition("-->")
        # Right side may have settings after the end timestamp.
        right = right.strip()
        right_parts = right.split(None, 1)
        end_ts = right_parts[0]
        settings = (" " + right_parts[1]) if len(right_parts) > 1 else ""
        new_start = fmt_ts(parse_ts(left.strip()) + offset)
        new_end = fmt_ts(parse_ts(end_ts) + offset)
        new_timing = f"{new_start} --> {new_end}{settings}"
        new_block = list(block)
        new_block[timing_idx] = new_timing
        cues.append("\n".join(new_block).rstrip() + "\n")

    while i < len(lines):
        line = lines[i]
        if line.strip() == "":
            flush(block)
            block = []
        else:
            block.append(line)
        i += 1
    flush(block)
    return cues


def main():
    dirs = sorted(p for p in CORPUS.iterdir() if p.is_dir())
    entries = []
    for d in dirs:
        mp3s = list(d.glob("*.mp3"))
        vtts = list(d.glob("*.vtt"))
        if len(mp3s) != 1 or len(vtts) != 1:
            print(f"skip {d}: mp3s={mp3s} vtts={vtts}", file=sys.stderr)
            sys.exit(1)
        entries.append((d.name, mp3s[0], vtts[0]))

    # Build concat list file.
    concat_list = CORPUS.parent / "_concat_list.txt"
    with concat_list.open("w") as f:
        for _, mp3, _ in entries:
            # Escape single quotes for ffmpeg concat demuxer.
            esc = str(mp3).replace("'", "'\\''")
            f.write(f"file '{esc}'\n")

    # Concatenate: re-encode so the output has a clean continuous frame table
    # and the final duration equals the sum of input durations (no gaps).
    print("running ffmpeg concat...", file=sys.stderr)
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c:a", "libmp3lame", "-b:a", "192k", "-ar", "44100", "-ac", "1",
            str(OUT_MP3),
        ]
    )
    concat_list.unlink()

    # Now walk each VTT, shift by cumulative input-mp3 durations, and write out.
    offset = 0.0
    out_lines = ["WEBVTT", ""]
    for name, mp3, vtt in entries:
        dur = ffprobe_duration(mp3)
        text = vtt.read_text(encoding="utf-8")
        cues = shift_vtt(text, offset)
        out_lines.append(f"NOTE {name} offset={fmt_ts(offset)} duration={dur:.3f}s")
        out_lines.append("")
        out_lines.extend(cues)
        offset += dur

    OUT_VTT.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")

    final_dur = ffprobe_duration(OUT_MP3)
    print(f"total input duration: {offset:.3f}s", file=sys.stderr)
    print(f"output mp3 duration:  {final_dur:.3f}s", file=sys.stderr)
    print(f"wrote {OUT_MP3}", file=sys.stderr)
    print(f"wrote {OUT_VTT}", file=sys.stderr)


if __name__ == "__main__":
    main()
