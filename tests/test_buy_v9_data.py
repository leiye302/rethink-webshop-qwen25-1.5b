"""Active buy-v9 data; preserve the training recipe and reject stale rows."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
VERSION=ROOT/'data/buy_v9'


def test_buy_v9_archive_and_unchanged_validation():
    lock=json.loads((VERSION/'data.json').read_text())
    assert lock['dataset_revision']=='buy-v9'
    assert hashlib.sha256((VERSION/'frozen_sft.zip').read_bytes()).hexdigest()==lock['bundle_sha256']
    with zipfile.ZipFile(VERSION/'frozen_sft.zip') as bundle:
        assert set(bundle.namelist())==set(lock['files'])|{'manifest.json'}
        manifest=json.loads(bundle.read('manifest.json'))
        assert manifest['files']==lock['files'] and manifest['user_authorized_repartition'] is True
        loaded={}
        for name,spec in lock['files'].items():
            content=bundle.read(name)
            assert len(content)==spec['bytes']
            assert hashlib.sha256(content).hexdigest()==spec['sha256']
            loaded[name]=[json.loads(line) for line in content.splitlines() if line]
        assert lock['files']['train.jsonl']['sha256']=='860a7b0ac73f91396089365947570c2f3052b2318d553ec1ac204964e78b9e59'
        with zipfile.ZipFile(ROOT/'data/frozen_sft.zip') as original:
            for name in ['val.jsonl','generation_val200.jsonl']:
                assert original.read(name)==bundle.read(name)
    train=loaded['train.jsonl']
    assert len(train)==12000 and Counter(row['route'] for row in train)=={'good':7500,'bad':4500}
    purchases=[row for row in train if row['supervision_action'].strip().lower()=='click[buy now]']
    assert Counter(row['route'] for row in purchases)=={'good':176,'bad':92}
    for row in train:
        if row['route']=='good':assert row['source_executed_action']==row['supervision_action']


def test_buy_v9_uses_existing_extractor_and_real_training_validator(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    from rethink_webshop import setup,common
    import train
    from route_bce import core
    (tmp_path/'data').mkdir();(tmp_path/'configs').mkdir()
    shutil.copyfile(VERSION/'frozen_sft.zip',tmp_path/'data/frozen_sft.zip')
    shutil.copyfile(VERSION/'data.json',tmp_path/'configs/data.json')
    monkeypatch.setattr(common,'ROOT',tmp_path)
    monkeypatch.setattr(setup,'ROOT',tmp_path)
    setup.data()
    config=train.load_config(ROOT/'configs/sft.json')
    training,validation,order,identity=train.validate_dataset(tmp_path,config,core)
    assert (len(training),len(validation),len(order))==(12000,902,1500)
    assert identity['generation_rows']==200
    assert all(sum(training[index]['route']=='good' for index in batch)==5 for batch in order)
    assert sorted(index for batch in order for index in batch)==list(range(12000))
    # A self-consistent old extraction must not bypass the active data lock.
    active=tmp_path/'configs/data.json'
    selected=json.loads(active.read_text());selected['files']['train.jsonl']['sha256']='0'*64
    active.write_text(json.dumps(selected))
    with pytest.raises(ValueError,match='selected database'):
        train.validate_dataset(tmp_path,config,core)
    shutil.copyfile(VERSION/'data.json',active)
    # A second identical preparation is safe; a partially replaced archive is not.
    setup.data()
    content=tmp_path/'datasets/frozen_sft/train.jsonl'
    content.write_bytes(content.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='Refuse to replace'):
        setup.data()


def test_active_bundle_is_a_complete_matching_released_profile():
    active=json.loads((ROOT/'configs/data.json').read_text())
    assert active['dataset_revision']=='buy-v9'
    assert active==json.loads((VERSION/'data.json').read_text())
    assert hashlib.sha256((ROOT/'data/frozen_sft.zip').read_bytes()).hexdigest()==active['bundle_sha256']
