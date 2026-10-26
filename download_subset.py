#!/usr/bin/env python3
"""Download a bounded Neophytes subset over SSH (Python 3 required on both ends)."""
import argparse
import csv
import io
import json
import random
import shlex
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

REPO = Path(globals().get("__file__", ".")).resolve().parent


def relative_path(value, root_name):
    parts = PurePosixPath(value).parts
    if root_name in parts:
        parts = parts[len(parts) - 1 - list(reversed(parts)).index(root_name) + 1:]
    elif parts and parts[0].isdigit():
        pass
    else:
        raise ValueError(f'Cannot locate dataset root {root_name!r} in {value!r}')
    p = PurePosixPath(*parts)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError(f'Unsafe path: {value}')
    return str(p)


def make_plan(root, count, seed):
    """Runs remotely: read metadata and filenames, never image contents."""
    root = Path(root).resolve()
    with (root / 'stats_imagewise.csv').open(newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or not {'img_path', 'mask_path'} <= set(fields):
        raise ValueError('stats_imagewise.csv needs img_path and mask_path columns')
    # One supplied course CSV has a blank header for this otherwise valid column.
    if '' in fields and 'S_inaequidens_px' not in fields:
        fields = ['S_inaequidens_px' if x == '' else x for x in fields]
        for row in rows:
            row['S_inaequidens_px'] = row.pop('')
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        img = relative_path(row['img_path'], root.name)
        mask = relative_path(row['mask_path'], root.name)
        if img in seen:
            raise ValueError(f'Duplicate image in CSV: {img}')
        seen.add(img)
        if 'images' not in PurePosixPath(img).parts:
            raise ValueError(f'Image outside images folder: {img}')
        groups[(row['split'], row['year'], row['site'])].append((row, img, mask))
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    # Round robin over splits, then sites: a single global limit, never per site.
    splits = defaultdict(list)
    for key in sorted(groups):
        splits[key[0]].append(groups[key])
    for buckets in splits.values():
        rng.shuffle(buckets)
    selected = []
    while len(selected) < count and any(splits.values()):
        for split in sorted(splits):
            buckets = splits[split]
            if buckets and len(selected) < count:
                bucket = buckets.pop(0)
                selected.append(bucket.pop())
                if bucket:
                    buckets.append(bucket)
    files, output_rows, cache = set(), [], {}
    for row, img, mask in selected:
        for rel in (img, mask):
            if not (root / rel).is_file():
                raise FileNotFoundError(root / rel)
            files.add(rel)
        image_path = PurePosixPath(img)
        base = PurePosixPath(*image_path.parts[:image_path.parts.index('images')])
        # Use precisely the loader's recursive sorted ordering, including differing
        # image/mask/elevation filenames, rather than guessing naming conventions.
        if str(base) not in cache:
            def listing(folder, extensions):
                import os
                return [str(Path(d, name).relative_to(root))
                        for d, _, names in sorted(os.walk(root / base / folder))
                        for name in sorted(names) if name.lower().endswith(extensions)]
            images = listing('images', ('.tif', '.png', '.jpg'))
            mask_folder = 'masks_prep' if (root / base / 'masks_prep').is_dir() else 'labels'
            masks = listing(mask_folder, ('.tif', '.png'))
            if len(images) != len(masks):
                raise ValueError(f'Image/mask counts differ in {base}')
            layers = {name: listing(name, ('.tif',)) for name in ('dsm', 'dtm')}
            for name, paths in layers.items():
                if (root / base / name).exists() and len(paths) != len(images):
                    raise ValueError(f'Incomplete {name} layer in {base}')
            cache[str(base)] = {im: (masks[i], [p[i] for p in layers.values() if p])
                                for i, im in enumerate(images)}
        paired_mask, elevation = cache[str(base)][img]
        if paired_mask != mask:
            raise ValueError(f'CSV mask disagrees with loader ordering for {img}')
        files.update(elevation)
        output_rows.append(dict(row, img_path=img, mask_path=mask))
    # Preserve every other CSV, at its original relative location. Only the
    # imagewise stats table needs subsetting and local path rewriting.
    files.update(str(p.relative_to(root)) for p in root.rglob('*.csv')
                 if p != root / 'stats_imagewise.csv')
    return dict(fields=fields, rows=output_rows, files=sorted(files),
                bytes=sum((root / p).stat().st_size for p in files))


def ssh(host, code, *args, jump_host=None, **kwargs):
    options = ["-J", jump_host] if jump_host else []
    return subprocess.Popen(['ssh', *options, host, shlex.join(['python3', '-c', code, *map(str, args)])], **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True, help='SSH config alias or user@host (uses your SSH keys/config)')
    parser.add_argument('--jump-host', help='Optional SSH jump host, e.g. lemtru@ssh.math.uzh.ch')
    parser.add_argument('--remote-root', required=True, help='Absolute remote path to Neophytes')
    parser.add_argument('--destination', type=Path, default=REPO.parent / 'data' / 'Neophytes')
    parser.add_argument('--count', type=int, default=100, help='TOTAL RGB tiles; hard maximum 100')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true', help='Read remote metadata only; download no tiles')
    args = parser.parse_args()
    if not 1 <= args.count <= 100:
        parser.error('--count must be between 1 and 100 (global safety limit)')
    if args.host.startswith('-') or not args.remote_root.startswith('/'):
        parser.error('Use a valid SSH host and an absolute --remote-root')
    dest = args.destination.resolve()
    if dest.name != 'Neophytes':
        parser.error('--destination must end in Neophytes for value_mapping compatibility')
    if not args.dry_run and dest.exists() and any(dest.iterdir()):
        parser.error('Destination must be empty; refusing to mix subsets or overwrite existing data')
    source = Path(__file__).read_text()
    worker = source.rsplit("if __name__ == '__main__':", 1)[0]
    proc = ssh(args.host, worker + '\nprint(json.dumps(make_plan(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))))',
               args.remote_root, args.count, args.seed, jump_host=args.jump_host, stdout=subprocess.PIPE, text=True)
    output, _ = proc.communicate()
    if proc.returncode:
        raise RuntimeError('Remote planning failed; no tiles downloaded')
    plan = json.loads(output)
    print(f"Selected {len(plan['rows'])} RGB images TOTAL, plus masks/elevation and CSVs: "
          f"{len(plan['files'])} files, {plan['bytes'] / 1024**3:.2f} GiB")
    if not plan['rows']:
        raise ValueError('No samples found')
    if args.dry_run:
        return
    dest.mkdir(parents=True, exist_ok=True)
    archive_code = '''import json,sys,tarfile,pathlib
root=pathlib.Path(sys.argv[1]).resolve()
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as tar:
 for name in json.load(sys.stdin):
  path=root/name
  if not path.resolve().is_relative_to(root) or not path.is_file():
   raise ValueError(name)
  tar.add(path, arcname=name, recursive=False)
'''
    proc = ssh(args.host, archive_code, args.remote_root, jump_host=args.jump_host, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    proc.stdin.write(json.dumps(plan['files']).encode())
    proc.stdin.close()
    expected = set(plan['files'])
    received = set()
    try:
        with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
            for member in archive:
                if member.name not in expected or not member.isfile() or member.name in received:
                    raise ValueError(f'Unexpected archive entry: {member.name}')
                target = dest / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, target.open('wb') as out:
                    import shutil
                    shutil.copyfileobj(src, out)
                received.add(member.name)
        if proc.wait() or received != expected:
            raise RuntimeError('Transfer incomplete; stats CSV was not written')
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
    for row in plan['rows']:
        for column in ('img_path', 'mask_path'):
            row[column] = str(dest / row[column])
    with (dest / 'stats_imagewise.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=plan['fields'])
        writer.writeheader()
        writer.writerows(plan['rows'])
    # Absolute directory overrides keep CSV matching correct from any working dir.
    config = {'dataset_stats': str(dest / 'stats_imagewise.csv'), 'image_dir': {}}
    for split in ('train', 'val', 'test'):
        config['image_dir'][split] = sorted({str(Path(r['img_path']).parent.parent)
                                            for r in plan['rows'] if r['split'] == split})
    (dest / 'subset_paths.yaml').write_text(json.dumps(config, indent=2) + '\n')
    print(f'Complete: {dest}\nLocal loader paths: {dest / "subset_paths.yaml"}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, tarfile.TarError) as exc:
        sys.exit(f'Error: {exc}')
