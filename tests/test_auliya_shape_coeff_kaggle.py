import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('shape_kaggle', ROOT / 'tools/auliya_shape_coeff_kaggle.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def put(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return path


def test_kaggle_notebook_defaults_to_audit_only():
    notebook = helper.read_json(ROOT / 'notebooks/Auliya_BC_SHAPECOEF1_Kaggle.ipynb')
    source = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    for i, code in enumerate(source):
        compile(code, f'cell {i}', 'exec')
    assert 'AUTHORIZE_TRAINING = False' in source[2]
    assert 'git clone' not in '\n'.join(source)


@pytest.mark.parametrize('zipped', [False, True])
def test_dataset_only_copies_pinned_bc_files(tmp_path, zipped):
    source = tmp_path / 'input/mounted/development'
    files = ('development_contract.json', 'data.yaml', 'train/images/a.jpg', 'val/labels/b.txt')
    hashes = {name: helper.sha(put(source / name, name)) for name in files}
    unexpected = put(source / 'test/images/hidden.jpg', 'DO NOT STAGE')
    target = tmp_path / 'ready'
    if zipped:
        archive = tmp_path / helper.DATA_ZIP
        with ZipFile(archive, 'w') as z:
            for name in files:
                z.write(source / name, name)
        source = archive
    assert helper.stage_data(tmp_path / 'input', target, hashes, source) == target
    assert not (target / 'test').exists() and unexpected.exists()
    put(target / 'data.yaml', 'wrong')
    with pytest.raises(ValueError, match='staging berubah'):
        helper.stage_data(tmp_path / 'input', target, hashes, source)


def test_restore_validates_contract_and_checksum_before_copy(tmp_path):
    output = tmp_path / 'working' / helper.PROTOCOL
    output.mkdir(parents=True)
    source = tmp_path / 'input/previous' / helper.PROTOCOL / 'RGB'
    last = put(source / 'weights/last.pt', 'checkpoint bytes')
    contract = {'arm': 'RGB', 'epochs': 50, 'gpu': 'T4', 'kaggle_bundle_sha256': 'pinned'}
    put(source / 'run_contract.json', json.dumps(contract))
    progress = {'epoch': 12, 'epochs': 50, 'checkpoint_sha256': helper.sha(last)}
    p = put(source / 'progress.json', json.dumps(progress))
    put(source / '.active_run', 'not copied from an immutable snapshot')
    with pytest.raises(RuntimeError, match='0 snapshot cocok'):
        helper.restore_arms(tmp_path / 'input', output, {'RGB': {**contract, 'gpu': 'P100'}})
    last.write_text('torn', encoding='utf-8')
    with pytest.raises(ValueError, match='tidak konsisten'):
        helper.restore_arms(tmp_path / 'input', output, {'RGB': contract})
    assert not (output / 'RGB').exists()
    p.write_text(json.dumps({**progress, 'checkpoint_sha256': helper.sha(last)}), encoding='utf-8')
    helper.restore_arms(tmp_path / 'input', output, {'RGB': contract})
    assert (output / 'RGB/weights/last.pt').is_file()
    assert not (output / 'RGB/.active_run').exists()
