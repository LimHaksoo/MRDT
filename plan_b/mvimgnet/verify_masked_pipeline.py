"""Geometry/padding/filter checks and a real-image fixture for DDP and resume."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import numpy as np
import torch
from PIL import Image, ImageDraw
from .audit_full_masks import file_digest
from .build_masked_catalog import add_masks
from .full_bin15_dataset import FullBin15Dataset
from .masked_bin15_dataset import MaskedFullBin15Dataset, load_masked_rgb, PREPROCESS_POLICY
from .bin15_dataset import mvimgnet_bin15_collate
from .prepare_full_bin15 import dump

ROOT=Path('/data1/haksoo/ReSCENE-RDT')
CONTROL=ROOT/'runs/masked_full_training_20261001'
DATA=Path('/data1/haksoo/data/MVImgNet2.0/processed_plan_b_full')


def main():
    fixture=CONTROL/'fixture';fixture.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory(dir=CONTROL) as temp:
        temp=Path(temp);rgb=Image.new('RGB',(8,4),(240,20,10));rgb.save(temp/'rgb.png')
        mask=np.zeros((4,8),dtype=np.uint8);mask[:,2:6]=255;mask[1:3,3:5]=0
        Image.fromarray(mask).save(temp/'mask.png')
        result,foreground,valid=load_masked_rgb(temp/'rgb.png',temp/'mask.png',size=8)
        assert result.shape==(3,8,8) and foreground.shape==(1,8,8)
        assert torch.allclose(result[:,0,0],torch.full((3,),128/255))
        assert torch.allclose(result[:,2,2],torch.tensor([240,20,10])/255)
        assert torch.allclose(result[:,3,3],torch.full((3,),128/255)), 'Hole was filled incorrectly'
        assert valid.sum()==32 and foreground.sum()==12
        Image.new('L',(8,4),0).save(temp/'zero.png')
        try:load_masked_rgb(temp/'rgb.png',temp/'zero.png')
        except ValueError:pass
        else:raise AssertionError('Empty mask was accepted')
        Image.new('L',(4,4),255).save(temp/'small.png')
        try:load_masked_rgb(temp/'rgb.png',temp/'small.png')
        except ValueError:pass
        else:raise AssertionError('Misaligned mask was accepted')
    # Use completed camera metadata; all fixture masks are checked synchronously.
    selected={s:[] for s in ['train','val','test']};sampled_masks=0
    states=sorted((DATA/'bin15_catalog_v3/archive_status').glob('*.json'))
    for state_path in states:
        state=json.loads(state_path.read_text())
        if state['state']!='completed':continue
        with (DATA/'bin15_catalog_v3/archive_manifests'/(state_path.stem+'.jsonl')).open() as source:
            for line in source:
                row=json.loads(line);split=row['split']
                if len(selected[split])>=4:continue
                audited={}
                for view in row['views']:
                    path=Path(view['path']).parent.parent/'masks'/Path(view['path']).name
                    with Image.open(path) as mask,Image.open(view['path']) as rgb:
                        assert mask.mode=='L' and mask.size==rgb.size
                        histogram=mask.histogram();area=sum(histogram[128:]);pixels=mask.width*mask.height
                    sampled_masks+=1
                    audited[row['instance_uid']+'/'+path.name]={'mask_path':str(path),'mask_sha256':file_digest(path),
                        'valid':bool(area),'reason':'empty_foreground_mask','foreground_fraction':area/pixels}
                updated,reason=add_masks(row,audited)
                if reason:continue
                # No surviving view may change its original camera fields.
                original_views={v['image_id']:v for v in row['views']}
                for view in updated['views']:
                    assert all(view[k]==v for k,v in original_views[view['image_id']].items())
                assert updated['reference_image_id']==row['reference_image_id']
                # A completely empty segmentation must reject the object, visibly.
                invalid={key:{**value,'valid':False} for key,value in audited.items()}
                assert add_masks(row,invalid)[1]=='mask_too_few_valid_views'
                selected[split].append(updated)
                if all(len(v)==4 for v in selected.values()):break
        if all(len(v)==4 for v in selected.values()):break
    assert all(len(v)==4 for v in selected.values())
    db=sqlite3.connect(fixture/'catalog.sqlite')
    db.execute('CREATE TABLE instances(uid TEXT PRIMARY KEY,split TEXT,ordinal INTEGER,row_json TEXT)')
    db.execute('CREATE UNIQUE INDEX split_ordinal ON instances(split,ordinal)')
    for split,rows in selected.items():
        for index,row in enumerate(rows):db.execute('INSERT INTO instances VALUES(?,?,?,?)',(row['instance_uid'],split,index,json.dumps(row)))
    db.commit();assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok';db.close()
    (fixture/'mvimgnet2_category.json').write_bytes((DATA/'mvimgnet2_category.json').read_bytes())
    dump(fixture/'readiness.json',{'state':'ready','test_fixture':True,'archive_count':1,'all_archives_crc_verified':True,
        'all_masks_audited':True,'preprocessing_policy':PREPROCESS_POLICY,'masked_dataset_sha256':file_digest(Path(__file__).with_name('masked_bin15_dataset.py')),
        'instances_by_split':{s:len(v) for s,v in selected.items()},'checked_masks':sampled_masks})
    raw=FullBin15Dataset(fixture/'catalog.sqlite','train',seed=0)
    masked=MaskedFullBin15Dataset(fixture/'catalog.sqlite','train',seed=0)
    used_padding=False
    for index in range(12):
        original=raw[index];new=masked[index]
        for key in ['context_present','context_camera_features','target_camera_features']:
            assert torch.equal(original['model_inputs'][key],new['model_inputs'][key]),key
        for key in ['instance_uid','target_image_id','context_image_ids','target_bin']:
            assert original['metadata'][key]==new['metadata'][key],key
        assert set(original['model_inputs'])==set(new['model_inputs']), 'GT mask leaked into conditioning'
        absent=~new['model_inputs']['context_present'];used_padding|=bool(absent.any())
        assert not new['model_inputs']['context_images'][absent].any()
        assert new['supervision']['target_image'].isfinite().all()
    assert used_padding
    batch=mvimgnet_bin15_collate([masked[0],masked[1]])
    assert batch['model_inputs']['context_images'].shape==(2,5,3,256,256)
    # Export actual tensor preprocessing, not an illustrative replacement.
    panels=[]
    for index in range(3):
        sample=masked[index];image=sample['supervision']['target_image']
        rgb=Image.fromarray((image.permute(1,2,0).numpy()*255).round().astype('uint8'))
        mask=Image.fromarray((sample['supervision']['target_foreground_mask'][0].numpy()*255).round().astype('uint8')).convert('RGB')
        panel=Image.new('RGB',(512,278),'white');panel.paste(rgb,(0,22));panel.paste(mask,(256,22))
        ImageDraw.Draw(panel).text((4,3),sample['metadata']['instance_uid']+' : model RGB / foreground',fill='black')
        panels.append(panel)
    sheet=Image.new('RGB',(512,834),'white')
    for i,panel in enumerate(panels):sheet.paste(panel,(0,i*278))
    sheet.save(CONTROL/'masked_tensor_examples.png')
    sources={p.name:file_digest(p) for p in Path(__file__).parent.glob('*masked*.py')}
    dump(CONTROL/'cpu_checks_passed.json',dict(state='passed',finished_at=time.time(),fixture=str(fixture),
        examples=str(CONTROL/'masked_tensor_examples.png'),fixture_objects=12,fixture_masks=sampled_masks,
        checked_samples=12,checks=['gray_rgb','hole_retention','letterbox_geometry','empty_and_misaligned_rejection',
        'original_camera_features','fixed_camera_reference','unchanged_sample_identity','padding','no_gt_mask_conditioning','collate'],sources=sources))
    print((CONTROL/'cpu_checks_passed.json').read_text(),flush=True)

if __name__=='__main__':main()
