"""Build a GitHub-ready source/results ZIP without raw audio or local caches."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def files():
    excluded = {'.git', '.venv', '__pycache__', 'node_modules', '.DS_Store'}
    result = []
    for path in ROOT.rglob('*'):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(p in excluded or p.startswith('reproduction') for p in rel.parts):
            continue
        if path.suffix in ('.pyc', '.zip', '.wav') or path.name in ('gain-features.npz', 'hw3_perturbed.npz', 'MANIFEST.sha256'):
            continue
        if path.stat().st_size >= 25_000_000:
            raise ValueError(f'File exceeds release per-file size limit: {rel}')
        result.append(path)
    return sorted(result)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, default=ROOT.parent / 'hw3-github-ready.zip')
    args = ap.parse_args()
    included = files()
    manifest = ROOT / 'MANIFEST.sha256'
    manifest.write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n' for p in included))
    with zipfile.ZipFile(args.out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in included + [manifest]:
            archive.write(p, Path(ROOT.name) / p.relative_to(ROOT))
    with zipfile.ZipFile(args.out) as archive:
        assert archive.testzip() is None
    print(json.dumps({'zip': str(args.out.resolve()), 'files': len(included) + 1,
                      'bytes': args.out.stat().st_size,
                      'sha256': hashlib.sha256(args.out.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
