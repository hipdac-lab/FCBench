#!/usr/bin/env python3
"""
FastLanes benchmark – serial version (whole-file FastLanes load).

1. Download two Google-Drive folders (if not present).
2. Convert every *_f32|*_f64 mem-map → CSV + schema.
3. Compress each CSV with FastLanes (whole file) and validate.
4. Merge compression_stats.csv files and print averages.

All reporting lines are prefixed with "--" and printed in green.
Final summary (ratios) is printed in blue, with ratios to 2 decimal places.
"""

from __future__ import annotations
import argparse, csv, glob, inspect, json, os, sys
from typing import List, Tuple

import duckdb, gdown, numpy as np, pandas as pd, pyfastlanes
from validate_csvs import validate_csvs  # local helper

# ANSI color codes
GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"


# ───────────────────────── CLI ─────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serial FastLanes benchmark")
    p.add_argument("--dl-threads", type=int, default=8,
                   help="threads per Drive folder if supported (default 8)")
    return p.parse_args()


# ───────── Google-Drive download ─────────
def download_folder(url: str, out_dir: str, threads: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"{GREEN}-- Downloading {url} → {out_dir}{RESET}")
    sig = inspect.signature(gdown.download_folder)
    kw = dict(output=out_dir, quiet=False, use_cookies=False)
    if "threads" in sig.parameters:
        kw["threads"] = threads
        print(f"{GREEN}--   using threads={threads}{RESET}")
    else:
        print(f"{GREEN}--   threads unsupported → serial{RESET}")
    for opt in ("skip_existing", "remaining_ok"):
        if opt in sig.parameters:
            kw[opt] = True
    try:
        gdown.download_folder(url, **kw)
        print(f"{GREEN}-- ✓ Folder downloaded: {out_dir}{RESET}")
    except Exception as e:
        if any(os.scandir(out_dir)):
            print(f"{GREEN}-- ⚠️  Partial download for {out_dir}: {e} – continuing.{RESET}")
        else:
            sys.exit(f"{GREEN}-- Download failed for {out_dir}: {e}{RESET}")


# ───────── Helpers ─────────
def float_fmt(dtype) -> str:
    return "%.9g" if dtype is np.float32 else "%.17g"


def process_and_save(fname: str, src: str, dst_root: str, prefix: str) -> Tuple[str, str]:
    # Determine NumPy dtype vs. schema‐string
    if "_f32" in fname:
        dtype_np, duck = np.float32, "FLOAT"
    elif "_f64" in fname:
        dtype_np, duck = np.float64, "DOUBLE"
    else:
        return fname, ""

    base, _ = os.path.splitext(fname)
    col = f"{prefix}_{base}"
    col_dir = os.path.join(dst_root, col)
    csv_path, schema_path = os.path.join(col_dir, "data.csv"), os.path.join(col_dir, "schema.json")

    if os.path.exists(csv_path) and os.path.exists(schema_path):
        print(f"{GREEN}--   ↻ {fname}: CSV + schema exist.{RESET}")
        return fname, col_dir

    os.makedirs(col_dir, exist_ok=True)
    arr = np.memmap(os.path.join(src, fname), dtype=dtype_np, mode="r")
    pd.DataFrame({col: arr}).to_csv(
        csv_path,
        index=False,
        header=False,
        float_format=float_fmt(dtype_np),
        chunksize=100_000
    )
    with open(schema_path, "w") as f:
        json.dump({"columns": [{"name": f"{col} : CONSTANT", "type": duck}]}, f, indent=2)
    print(f"{GREEN}--   ✓ Converted {fname}{RESET}")
    return fname, col_dir


# ───────── compression + validation ─────────
def compress_and_validate(fname: str, col_dir: str, src_dir: str) -> bool:
    csv_path = os.path.join(col_dir, "data.csv")
    fls_path, dec_path = os.path.join(col_dir, "data.fls"), os.path.join(col_dir, "decoded.csv")

    if not os.path.exists(csv_path):
        print(f"{GREEN}--   ‼️  {fname}: data.csv missing — skip{RESET}")
        return False

    # Remove any preexisting outputs
    for p in (fls_path, dec_path):
        if os.path.exists(p):
            os.remove(p)

    try:
        conn = pyfastlanes.connect()
        conn.inline_footer().read_csv(col_dir).to_fls(fls_path)
        conn.read_fls(fls_path).to_csv(dec_path)
    except Exception as e:
        print(f"{GREEN}--   ❌ FastLanes error on {fname}: {e}{RESET}")
        return False

    prec = "float32" if fname.endswith("_f32") else "float64"
    if not validate_csvs(csv_path, dec_path, precision=prec):
        print(f"{GREEN}--   ❌ Validation FAILED for {fname}{RESET}")
        return False

    uncomp = os.path.getsize(os.path.join(src_dir, fname))
    comp = os.path.getsize(fls_path)
    if comp == 0:
        print(f"{GREEN}--   ❌ {fname}: empty .fls{RESET}")
        return False

    ratio = uncomp / comp
    with open(os.path.join(col_dir, "compression_stats.csv"), "w", newline="") as f:
        csv.writer(f).writerows([
            ["file", "uncompressed_size", "compressed_size", "ratio", "validated"],
            [fname, uncomp, comp, f"{ratio:.2f}", True]
        ])
    print(f"{GREEN}--   ✓ {fname}: PASSED (ratio {ratio:.2f}×){RESET}")
    return True


def already_done(col_dir: str) -> bool:
    return os.path.exists(os.path.join(col_dir, "data.fls")) and \
        os.path.exists(os.path.join(col_dir, "compression_stats.csv"))


# ───────── Stats aggregation ─────────
def combine_and_summarize(out_csv: str, root: str = ".") -> None:
    files = glob.glob(os.path.join(root, "**", "compression_stats.csv"), recursive=True)
    if not files:
        print(f"{GREEN}-- No stats found.{RESET}")
        return

    parts = []
    for p in files:
        ds = os.path.basename(os.path.dirname(p))
        header = open(p).readline().strip().split(",")
        cols = "*" if "validated" in header else \
            "file,uncompressed_size,compressed_size,ratio, TRUE AS validated"
        parts.append(
            f"SELECT '{ds}' AS dataset, {cols} "
            f"FROM read_csv_auto('{p}', header=TRUE)"
        )

    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE TABLE stats AS {' UNION ALL '.join(parts)}")
    con.execute(f"COPY stats TO '{out_csv}' (HEADER TRUE)")

    overall = con.execute("SELECT AVG(ratio) FROM stats").fetchone()[0]
    by_ds = con.execute(
        "SELECT dataset, AVG(ratio) AS avg_ratio "
        "FROM stats "
        "GROUP BY dataset "
        "ORDER BY dataset"
    ).fetchdf()

    # Print the summary in blue, with two decimal places
    print(f"{BLUE}-- Combined stats → {out_csv}{RESET}\n")
    print(f"{BLUE}-- Average ratio by dataset:{RESET}")
    # Format each avg_ratio to two decimals
    for row in by_ds.itertuples(index=False):
        ds_name = row.dataset
        avg_r = row.avg_ratio
        print(f"{BLUE}--   {ds_name:30s}  {avg_r:.2f}×{RESET}")
    print(f"{BLUE}-- Overall average compression ratio: {overall:.2f}×{RESET}")


# ───────────────────────── main ─────────────────────────
if __name__ == "__main__":
    args = parse_args()
    if args.dl_threads < 1:
        sys.exit(f"{GREEN}-- --dl-threads must be ≥1{RESET}")
    print(f"{GREEN}-- Download threads (if supported): {args.dl_threads}{RESET}\n")

    FOLDERS = [
        ("https://drive.google.com/drive/folders/1jdnzwvT1hya8XYdEJ7QuqUw3ALbQozc7", "HPC_TS_OBS"),
        ("https://drive.google.com/drive/folders/1WKvzMErKfhqAGRkJhqXZScH15kPHUxnG", "DB"),
    ]

    # 1. download
    for url, folder in FOLDERS:
        if os.path.exists(folder):
            print(f"{GREEN}-- {folder} exists; skip download.{RESET}")
        else:
            download_folder(url, folder, args.dl_threads)

    # 2. convert
    maps: List[Tuple[str, str]] = []
    for _, folder in FOLDERS:
        dst = f"{folder}_csvs"
        pref = "HPC_TS_OBS" if folder.startswith("HPC") else "DB"
        print(f"{GREEN}-- Converting binaries in {folder} → {dst}{RESET}")
        for fn in sorted(os.listdir(folder)):
            if "_f32" in fn or "_f64" in fn:
                maps.append(process_and_save(fn, folder, dst, pref))

    print(f"{GREEN}-- FastLanes {pyfastlanes.get_version()}{RESET}")

    # 3. compress & validate
    for fname, col_dir in maps:
        if not col_dir:
            continue
        if already_done(col_dir):
            print(f"{GREEN}-- {fname}: compression already done; skipping.{RESET}")
            continue

        src_root = "HPC_TS_OBS" if col_dir.startswith("HPC_TS_OBS_csvs") else "DB"
        print(f"{GREEN}-- === Compress & validate {fname} ==={RESET}")
        if not compress_and_validate(fname, col_dir, src_root):
            sys.exit(f"{GREEN}-- Stopping due to validation failure.{RESET}")

    print(f"{GREEN}-- ✔ All compression & validation tasks finished.{RESET}")

    # 4. stats
    combine_and_summarize("combined_compression_stats.csv")
    # Final line—no leading newline this time:
    print(f"{GREEN}-- All done successfully.{RESET}")
