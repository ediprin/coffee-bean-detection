"""Validation-only dataset staging and persistent artifacts for the Kaggle pilot."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time
import zipfile


PROTOCOL = 'auliya-bc-shape-coeff-v1'
ARMS = ('RGB', 'SHAPECOEF1')
DATA_ZIP = 'auliya-robusta-sni-b-train-c-val.zip'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def stage_data(input_root, destination, files, override=None):
    if destination.is_dir():
        if any(not (destination / p).is_file() or sha(destination / p) != h for p, h in files.items()):
            raise ValueError(f'Dataset staging berubah: {destination}. Gunakan sesi bersih.')
        return destination
    if override:
        sources = [Path(override)]
    else:
        sources = [p.parent for p in sorted(input_root.rglob('development_contract.json'))
                   if sha(p) == files['development_contract.json']]
        sources = [p for p in sources if (p / 'train/images').is_dir() and (p / 'val/images').is_dir()]
        if not sources:
            sources = sorted(input_root.rglob(DATA_ZIP))
    if len(sources) != 1:
        raise FileNotFoundError(f'Pasang satu input B/C atau isi DATA_OVERRIDE. Kandidat: {sources}')
    source = sources[0]
    with TemporaryDirectory(dir=destination.parent, prefix='auliya-bc-') as tmp:
        payload = Path(tmp) / 'development'
        for name, digest in files.items():
            # File list is embedded in the SHA-pinned code bundle. No source test path is opened.
            p = Path(name)
            if not (name in ('data.yaml', 'development_contract.json') or (
                len(p.parts) == 3 and p.parts[0] in ('train', 'val')
                and '..' not in p.parts
                and ((p.parts[1] == 'images' and p.suffix == '.jpg') or
                     (p.parts[1] == 'labels' and p.suffix == '.txt'))
            )):
                raise ValueError(f'File di luar protokol B/C: {name}')
        if source.is_dir():
            for name in files:
                dst = payload / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, dst)
        else:
            with zipfile.ZipFile(source) as z:
                if sorted(z.namelist()) != sorted(files):
                    raise ValueError('Inventaris arsip berbeda dari B/C yang dibekukan')
                for name in files:
                    dst = payload / name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(name) as a, dst.open('wb') as b:
                        shutil.copyfileobj(a, b)
        if any(sha(payload / p) != h for p, h in files.items()):
            raise ValueError(f'SHA256 gambar/label atau kontrak B/C berbeda: {source}')
        os.replace(payload, destination)
    return destination


def restore_arms(input_root, output, contracts, previous_output=None):
    for arm, expected in contracts.items():
        dst = output / arm
        if dst.exists():  # The runner owns in-session resume, locks, and contract checking.
            continue
        markers = ([Path(previous_output) / arm / 'run_contract.json'] if previous_output else
                   sorted(p for p in input_root.rglob('run_contract.json')
                          if p.parent.name == arm and p.parent.parent.name == PROTOCOL))
        markers = [p for p in markers if p.is_file() and (p.parent / 'weights/last.pt').is_file()]
        if not markers:
            continue
        matches = [p for p in markers if read_json(p) == expected]
        if len(matches) != 1:
            raise RuntimeError(f'{arm}: ditemukan {len(matches)} snapshot cocok dari {len(markers)}. Cek runtime atau isi PREVIOUS_OUTPUT.')
        source = matches[0].parent
        progress = read_json(source / 'progress.json')
        last = source / 'weights/last.pt'
        if (progress.get('checkpoint_sha256') != sha(last) or progress.get('epochs') != expected['epochs']
                or not 1 <= progress.get('epoch', -1) <= expected['epochs']):
            raise ValueError(f'Snapshot checkpoint {arm} tidak konsisten: {source}')
        with TemporaryDirectory(dir=output, prefix='restore-') as tmp:
            staged = Path(tmp) / arm
            (staged / 'weights').mkdir(parents=True)
            for name in ('weights/last.pt', 'run_contract.json', 'progress.json', 'results.csv', 'validation_result.json'):
                if (source / name).is_file():
                    shutil.copyfile(source / name, staged / name)
            os.replace(staged, dst)
        print(f'RESTORE {arm}: {progress["epoch"]}/{expected["epochs"]}', flush=True)


def run_logged(command, env, output, arm=None):
    output.mkdir(parents=True, exist_ok=True)
    log = output / (f'{arm}_run.log' if arm else 'static_audit.log')
    last_epoch = None
    print(f'{"START/RESUME " + arm if arm else "STATIC AUDIT"} | {log}', flush=True)
    with log.open('ab') as stream:
        with subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT) as proc:
            try:
                while proc.poll() is None:
                    progress = output / arm / 'progress.json' if arm else None
                    epoch = read_json(progress)['epoch'] if progress and progress.is_file() else 0
                    if arm and epoch != last_epoch:
                        print(f'{arm}: {epoch}/50 epoch tersimpan', flush=True)
                        last_epoch = epoch
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        pass
            except KeyboardInterrupt:
                proc.terminate()
                raise
    if proc.returncode:
        print('\n'.join(log.read_text(encoding='utf-8', errors='replace').splitlines()[-40:]))
        raise RuntimeError(f'Proses gagal ({proc.returncode}); log: {log}')


def summarize(output):
    reports = {arm: output / arm / 'validation_result.json' for arm in ARMS}
    if not all(p.is_file() for p in reports.values()):
        print('Ringkasan berpasangan tersedia setelah kedua arm selesai.')
        return
    payloads = {arm: read_json(p) for arm, p in reports.items()}
    common = []
    for arm, row in payloads.items():
        contract = row['run_contract']
        if (row['test_images_accessed'] is not False or row['evaluation_split'] != 'val'
                or contract['arm'] != arm or contract['protocol'] != PROTOCOL
                or row['checkpoint_sha256'] != sha(output / arm / 'weights/last.pt')):
            raise ValueError(f'Invalid validation report: {arm}')
        common.append({k: v for k, v in contract.items() if k != 'arm'})
    if common[0] != common[1]:
        raise ValueError('Kedua arm bukan pasangan dengan kontrak identik')
    table = {}
    for arm, row in payloads.items():
        metrics = {k: v for section in ('box', 'mask') for k, v in row[section].items() if isinstance(v, (int, float))}
        box_classes = row['box']['box_ap_per_class']
        focus, = [k for k in box_classes if 'biji hitam sebagian' in k.lower().replace('_', ' ')]
        metrics[f'{focus}_box_ap50_95'] = box_classes[focus]
        metrics.update({k: row['focus_at_conf025_iou050'].get(k, 0)
                        for k in ('targets', 'target_matched', 'target_correct', 'target_fp', 'all_fp')})
        table[arm] = metrics
    left, right = (table[arm] for arm in ARMS)
    result = {'protocol': PROTOCOL, 'split': 'val', 'test_images_accessed': False,
              'seed': common[0]['seed'], 'arms': table,
              'shape_minus_rgb': {k: right.get(k, 0) - left.get(k, 0) for k in sorted(left.keys() | right.keys())}}
    (output / 'paired_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


def report_zip(output):
    dest = output.parent / f'{PROTOCOL}-reports.zip'
    with zipfile.ZipFile(dest, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file() and p.suffix in {'.log', '.json', '.csv'}:
                z.write(p, p.relative_to(output))
    print(f'REPORTS: {dest}\nRESUME: simpan Notebook Output lengkap dengan {output.name}/*/weights/last.pt')
