"""Check unchanged evaluation tensors and train epoch traversal with real images."""
import json
import sys
from pathlib import Path
sys.path[:0] = ['/data1/haksoo/ReSCENE-RDT', '/home/haksoo/ReSCENE-RDT']
import torch
from torch.utils.data import DataLoader
from plan_b.mvimgnet.full_bin15_dataset import FullBin15Dataset
from plan_b.mvimgnet.bin15_dataset import MVImgNetBin15Stage1Dataset, mvimgnet_bin15_collate, _load_rgb
from plan_b.mvimgnet.epoch_sampling import EpochPermutation

torch.set_num_threads(2)
root = Path('/data1/haksoo/data/MVImgNet2.0')
catalog = Path('/data1/haksoo/ReSCENE-RDT/runs/codex_full_preparation_tests_20260930/catalog-test/catalog.sqlite')
checked = 0
for split in ['train', 'val', 'test']:
    old = MVImgNetBin15Stage1Dataset(root / 'processed_plan_b/manifests' / (split + '_instances.jsonl'),
        root / 'reports_plan_b/mvimgnet2_audit_followup/fixed_camera_views_followup.csv', split, deterministic=True)
    new = FullBin15Dataset(catalog, split, deterministic=True)
    for index in range(len(new.rows)):
        a, b = old[index], new[index]
        for group in ['model_inputs', 'supervision']:
            for key in a[group]:
                if torch.is_tensor(a[group][key]):
                    assert torch.equal(a[group][key], b[group][key]), (split, index, key)
        checked += 1

dataset = FullBin15Dataset(catalog, 'train', samples_per_epoch=40)
expected = []
for index in range(40):
    sample = dataset[index]
    meta = sample['metadata']
    expected.append(meta)
    assert meta['instance_uid'] == dataset.rows[dataset.object_order[index]]['instance_uid']
    row = dataset.rows[meta['instance_uid']]
    views = {v['image_id']: v for v in row['views']}
    ids = [i for i in meta['context_image_ids'] if i is not None]
    assert 2 <= len(ids) <= 5 and len(set(ids + [meta['target_image_id']])) == len(ids) + 1
    inputs = sample['model_inputs']
    assert int(inputs['context_present'].sum()) == len(ids)
    assert not inputs['context_present'][len(ids):].any()
    assert not inputs['context_images'][len(ids):].any()
    assert not inputs['context_camera_features'][len(ids):].any()
    for slot, image_id in enumerate(ids):
        assert torch.equal(inputs['context_camera_features'][slot], torch.tensor(views[image_id]['features']))
        assert tuple((views[image_id]['az_bin'], views[image_id]['el_bin'])) != tuple(meta['target_bin'])
    target = views[meta['target_image_id']]
    assert torch.equal(sample['supervision']['target_image'], _load_rgb(Path(target['path']))[0])
for start in range(0, 40, 10):
    assert len({m['instance_uid'] for m in expected[start:start+10]}) == 10
loader = DataLoader(FullBin15Dataset(catalog, 'train', samples_per_epoch=40), batch_size=8,
                    num_workers=4, collate_fn=mvimgnet_bin15_collate)
actual = [meta for batch in loader for meta in batch['metadata']]
assert actual == expected
result = {'state': 'passed', 'deterministic_evaluation_exact_samples': checked,
          'real_train_samples': 40, 'complete_object_epochs': 4,
          'workers4_identical_metadata': True, 'padding_and_camera_target_contract': True}
print(json.dumps(result))
