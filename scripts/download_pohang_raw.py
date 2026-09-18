"""Fetch the raw Pohang Canal LiDAR recordings so the corpus can be converted again.

Why this is needed. The pretraining corpus on disk was produced by an earlier
conversion whose code we no longer have, and its fourth channel carries 2.8 to 3.9
percent negative values in every variant. Reflectance cannot be negative, so the wrong
source column was reduced from the raw eight-column format. Only converted copies
survive locally, so the correct column can only be recovered from the original release.

The dataset is in the AWS Open Data Registry and the bucket is world readable over
plain HTTPS, so no credentials and no AWS client are required.

Scope. Only the front LiDAR of sequences 00 and 01, which is what the corpus
`pohang_canal_seq_00_01_*` was built from, plus the calibration. About 61 GB.

Parallel by range request. A single connection to this bucket is throttled to roughly
300 kB/s, which would take 55 hours. Measured with three simultaneous connections, each
still reached 275 to 425 kB/s, so the limit is per connection and not our uplink.
Eight workers bring it to a few hours. Each worker owns a byte range and writes its own
part file, so an interrupted run resumes per part rather than from the start.
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "https://pohang-canal-dataset.s3.amazonaws.com"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "PohangCanal_raw"
KEYS = [
    "pohang00/calibration.zip",
    "pohang01/calibration.zip",
    "pohang00/lidar/lidar_front.zip",
    "pohang01/lidar/lidar_front.zip",
]
WORKERS = 8


def size_of(key):
    """Content length of one archive, or None if the server does not report it."""
    out = subprocess.run(["curl", "-sI", f"{BUCKET}/{key}"], capture_output=True,
                         text=True).stdout
    for line in out.splitlines():
        if line.lower().startswith("content-length"):
            return int(line.split()[1])
    return None


def fetch_part(key, part, start, end):
    """Fetch one byte range into its own part file, resuming if it is incomplete."""
    want = end - start + 1
    have = part.stat().st_size if part.exists() else 0
    if have >= want:
        return True
    subprocess.run(["curl", "-sL", "--retry", "10", "--retry-delay", "5",
                    "-r", f"{start + have}-{end}", "-o", str(part),
                    "--create-dirs"] + (["-C", "-"] if have else []) +
                   [f"{BUCKET}/{key}"], check=False)
    return (part.stat().st_size if part.exists() else 0) >= want


def get(key):
    """Fetch one archive, resuming a partial download rather than restarting it.

    The four archives are about 54 GB together, so an interrupted transfer continues
    from the byte it stopped at.
    """
    target = DEST / key.replace("/", "_")
    total = size_of(key)
    if total is None:
        print(f"[skip] {key}: no content length", flush=True)
        return
    if target.exists() and target.stat().st_size == total:
        print(f"[ok]   {key} already complete ({total / 1e9:.1f} GB)", flush=True)
        return

    n = 1 if total < 50_000_000 else WORKERS
    step = (total + n - 1) // n
    parts = [(DEST / "parts" / f"{key.replace('/', '_')}.{i:02d}",
              i * step, min((i + 1) * step - 1, total - 1)) for i in range(n)]
    print(f"[get]  {key}  {total / 1e9:.1f} GB in {n} parts", flush=True)

    with ThreadPoolExecutor(max_workers=n) as ex:
        ok = list(ex.map(lambda p: fetch_part(key, *p), parts))
    if not all(ok):
        print(f"[warn] {key}: {ok.count(False)} parts incomplete, rerun to resume",
              flush=True)
        return

    with target.open("wb") as out:
        for p, _, _ in parts:
            out.write(p.read_bytes())
    got = target.stat().st_size
    if got == total:
        for p, _, _ in parts:
            p.unlink()
        print(f"[done] {key}  {got / 1e9:.1f} GB  ok", flush=True)
    else:
        print(f"[warn] {key}: assembled {got} of {total}, parts kept", flush=True)


def main():
    """Fetch the four Pohang Canal archives into data/raw."""
    (DEST / "parts").mkdir(parents=True, exist_ok=True)
    for key in KEYS:
        get(key)
    print("all requested files processed", flush=True)


if __name__ == "__main__":
    sys.exit(main())
