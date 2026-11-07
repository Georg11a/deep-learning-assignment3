"""Download the full CREMA-D AudioWAV corpus (7,442 clips, ~600 MB) into data/crema/.

The corpus is NOT committed to git (data/ is git-ignored). File names come from
scripts/crema_files.txt (listing of CheyneyComputerScience/CREMA-D/AudioWAV).
Re-running skips files that are already present and valid.

Usage:  python3 scripts/download_crema.py [--workers 16]
CREMA-D is released under the Open Database License (ODbL); see DATA_ATTRIBUTION.md.
"""
import argparse
import concurrent.futures
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/'
DEST = ROOT / 'data' / 'crema'


def fetch(name, retries=4):
    dest = DEST / name
    if dest.exists() and dest.stat().st_size > 44 and dest.read_bytes()[:4] == b'RIFF':
        return name, 'skip'
    last = None
    for attempt in range(retries):
        try:
            with urlopen(BASE + name, timeout=60) as response:
                body = response.read()
            if body[:4] != b'RIFF':
                raise ValueError('not a WAV (LFS pointer or error page)')
            tmp = dest.with_suffix('.part')
            tmp.write_bytes(body)
            tmp.replace(dest)
            return name, 'ok'
        except Exception as exc:  # network hiccups: back off and retry
            last = exc
            time.sleep(1.5 * (attempt + 1))
    return name, f'FAIL {last}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in (ROOT / 'scripts' / 'crema_files.txt').read_text().splitlines() if n.strip()]
    counts = {'ok': 0, 'skip': 0, 'fail': 0}
    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (name, status) in enumerate(pool.map(fetch, names), 1):
            key = 'fail' if status.startswith('FAIL') else status
            counts[key] += 1
            if key == 'fail':
                failed.append(f'{name}: {status}')
            if i % 500 == 0 or i == len(names):
                print(f'{i}/{len(names)}  {counts}', flush=True)
    if failed:
        (DEST / 'FAILED.txt').write_text('\n'.join(failed))
        print(f'{len(failed)} files failed; listed in {DEST / "FAILED.txt"}. Re-run to retry.')


if __name__ == '__main__':
    main()
