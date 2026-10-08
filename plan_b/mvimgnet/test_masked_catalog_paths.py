"""Regress COLMAP prefix matching against the actual independent mask audit."""
import copy
import json
from pathlib import Path
import time
from .build_masked_catalog import add_masks
from .audit_full_masks import file_digest
from .prepare_full_bin15 import dump


def main():
    data=Path('/data1/haksoo/data/MVImgNet2.0/processed_plan_b_full')
    control=Path('/data1/haksoo/ReSCENE-RDT/runs/masked_full_training_20261001')
    source=data/'mask_audit_v1/views/mvi2_00.jsonl'
    state=json.loads((data/'mask_audit_v1/archives/mvi2_00.json').read_text())
    assert state['state']=='completed' and file_digest(source)==state['audit_sha256']
    with source.open() as f:audited={r['key']:r for r in map(json.loads,f)}
    rows=views=rejected=0;regression_found=False
    with (data/'bin15_catalog_v3/archive_manifests/mvi2_00.jsonl').open() as f:
        for line in f:
            row=json.loads(line);updated,reason=add_masks(row,audited)
            rows+=1;views+=len(row['views']);rejected+=bool(reason)
            assert row['reference_image_id']==updated['reference_image_id']
            originals={v['image_id']:v for v in row['views']}
            for v in updated['views']:assert all(v[key]==value for key,value in originals[v['image_id']].items())
            if row['instance_uid']=='1/5702e292':
                regression_found=True
                assert any(v['frame_name'].startswith('images/') for v in row['views'])
                plain=copy.deepcopy(row)
                for v in plain['views']:v['frame_name']=Path(v['frame_name']).name
                plain_result,plain_reason=add_masks(plain,audited)
                assert plain_reason==reason
                assert [v['mask_path'] for v in plain_result['views']]==[v['mask_path'] for v in updated['views']]
                wrong=copy.deepcopy(row);wrong['instance_uid']='wrong/'+row['instance_id']
                try:add_masks(wrong,audited)
                except AssertionError:pass
                else:raise AssertionError('Wrong object UID matched masks')
    assert regression_found and rows>4000 and views>100000
    result=dict(state='passed',objects=rows,views=views,mask_rejected_objects=rejected,
        regression_uid='1/5702e292',builder_sha256=file_digest(Path(__file__).with_name('build_masked_catalog.py')),
        real_audit_sha256=state['audit_sha256'],finished_at=time.time())
    dump(control/'catalog_path_regression_passed.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
