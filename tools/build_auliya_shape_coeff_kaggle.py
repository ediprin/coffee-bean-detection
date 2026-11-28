"""Build the Kaggle code/bootstrap input from the pinned B/C download and local candidate."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BASE = '6adc54d69520b3160a169ae412967c1a9ce9d4a3'
WEIGHTS_SHA = '361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f'
DATA_SHA = 'effa7d1ba3086d5695b9d2ca58315d44bc05fcdbc6d258d2c9a9f18a962b7fa8'
OVERLAYS = (
    'src/coffee_detector/auliya_shape_coeff.py',
    'src/coffee_detector/analysis/auliya_box_consensus.py',
    'src/coffee_detector/experiments/run_auliya_shape_coeff.py',
    'tools/auliya_shape_coeff_kaggle.py',
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(weights, development_zip, output):
    remote = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=ROOT, text=True).strip()
    if remote not in {'https://github.com/ediprin/coffee-bean-detection.git', 'git@github.com:ediprin/coffee-bean-detection.git'}:
        raise ValueError(f'Unexpected source repo: {remote}')
    if file_hash(weights) != WEIGHTS_SHA or file_hash(development_zip) != DATA_SHA:
        raise ValueError('The official checkpoint or development ZIP does not match the pinned SHA256')
    archive = subprocess.check_output(['git', 'archive', '--format=zip', BASE, 'src/coffee_detector'], cwd=ROOT)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        files = {info.filename: z.read(info) for info in z.infolist() if not info.is_dir() and info.filename.endswith('.py')}
    files.update({path: (ROOT / path).read_bytes().replace(b'\r\n', b'\n') for path in OVERLAYS})
    with zipfile.ZipFile(development_zip) as z:
        data_manifest = {info.filename: digest(z.read(info)) for info in z.infolist() if not info.is_dir()}
    files['development_files.json'] = json.dumps(data_manifest, sort_keys=True).encode()
    files['weights/yolo26n-seg.pt'] = weights.read_bytes()
    manifest = {'format': 'auliya-shape-coeff-kaggle-v1', 'base_commit': BASE,
                'files': {path: digest(content) for path, content in sorted(files.items())}}
    files['shape_coeff_bundle_manifest.json'] = json.dumps(manifest, sort_keys=True).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w') as z:
        for path, content in sorted(files.items()):
            z.writestr(zipfile.ZipInfo(path), content, compress_type=zipfile.ZIP_DEFLATED)
    print(f'{output}\nSHA256: {file_hash(output)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--development-zip', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build(args.weights, args.development_zip, args.output)
