"""
Convert an LJSpeech-style dataset into a VoxCPM training manifest (jsonl).

Expected layout:
    <root>/metadata.csv     lines like:  <wav path relative to wavs/>|<text>[|<normalized text>]
    <root>/wavs/...         audio files (sub-directories allowed)

The wav field may include or omit the ".wav" suffix (classic LJSpeech omits it).
If a third column exists it is used as the text (LJSpeech's normalized text).

Example:
    python scripts/ljspeech_to_jsonl.py --root D:/data/my_ljspeech --output new_data.jsonl
    python scripts/ljspeech_to_jsonl.py --root D:/data/my_ljspeech --output merged.jsonl \
        --merge-with train.jsonl --dataset-id 1
"""

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf


def parse_args():
    p = argparse.ArgumentParser(description="LJSpeech metadata -> VoxCPM jsonl manifest")
    p.add_argument("--root", required=True, help="Dataset root containing metadata file and wavs/ folder")
    p.add_argument("--metadata", default="metadata.csv", help="Metadata filename (relative to --root) or absolute path")
    p.add_argument("--wavs-dir", default="wavs", help="Audio folder (relative to --root) or absolute path")
    p.add_argument("--output", required=True, help="Output jsonl path")
    p.add_argument("--merge-with", default="", help="Existing jsonl to prepend to the output (original data kept as-is)")
    p.add_argument("--dataset-id", type=int, default=None, help="Optional dataset_id to tag the new samples with")
    p.add_argument("--min-duration", type=float, default=0.0, help="Drop clips shorter than this (seconds)")
    p.add_argument("--max-duration", type=float, default=0.0, help="Drop clips longer than this (seconds, 0 = no limit)")
    return p.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    metadata = resolve(root, args.metadata)
    wavs_dir = resolve(root, args.wavs_dir)

    stats = {"ok": 0, "bad_line": 0, "empty_text": 0, "missing": 0, "unreadable": 0, "too_short": 0, "too_long": 0}
    total_sec = 0.0
    records = []

    with open(metadata, "r", encoding="utf-8-sig") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                stats["bad_line"] += 1
                print(f"[skip] line {lineno}: no '|' separator", file=sys.stderr)
                continue

            rel = parts[0].strip()
            text = (parts[2] if len(parts) >= 3 and parts[2].strip() else parts[1]).strip()
            if not text:
                stats["empty_text"] += 1
                print(f"[skip] line {lineno}: empty text", file=sys.stderr)
                continue

            if not rel.lower().endswith(".wav"):
                rel += ".wav"
            wav_path = wavs_dir / rel
            if not wav_path.is_file():
                stats["missing"] += 1
                print(f"[skip] line {lineno}: file not found: {wav_path}", file=sys.stderr)
                continue

            try:
                info = sf.info(str(wav_path))
                duration = info.frames / float(info.samplerate)
            except Exception as e:  # noqa: BLE001
                stats["unreadable"] += 1
                print(f"[skip] line {lineno}: cannot read {wav_path}: {e}", file=sys.stderr)
                continue

            if duration < args.min_duration:
                stats["too_short"] += 1
                continue
            if args.max_duration > 0 and duration > args.max_duration:
                stats["too_long"] += 1
                continue

            rec = {"audio": wav_path.as_posix(), "text": text, "duration": round(duration, 3)}
            if args.dataset_id is not None:
                rec["dataset_id"] = args.dataset_id
            records.append(rec)
            stats["ok"] += 1
            total_sec += duration

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        if args.merge_with:
            with open(args.merge_with, "r", encoding="utf-8-sig") as old:
                for line in old:
                    line = line.strip()
                    if not line:
                        continue
                    if args.dataset_id is not None:
                        # Mixed rows with/without dataset_id become None after HF json loading,
                        # which crashes the training script, so give old rows an explicit id.
                        rec = json.loads(line)
                        rec.setdefault("dataset_id", 0)
                        line = json.dumps(rec, ensure_ascii=False)
                    out.write(line + "\n")
                    merged += 1
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"New samples written: {stats['ok']} ({total_sec / 3600:.2f} h)")
    if merged:
        print(f"Existing samples kept from {args.merge_with}: {merged}")
    skipped = {k: v for k, v in stats.items() if k != "ok" and v}
    if skipped:
        print(f"Skipped: {skipped}")
    print(f"Output: {out_path.resolve()}")


if __name__ == "__main__":
    main()
