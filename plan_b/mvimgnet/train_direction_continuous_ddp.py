"""Continuous direction-only full training with preserved bin-based view selection."""
import argparse
import datetime
import hashlib
import json
import math
import os
import signal
import time
from unittest.mock import patch
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from .direction_bin15_model import DirectionBin15RDT
from .continuous_direction_dataset import (ContinuousDirectionDataset as DirectionFullBin15Dataset,
    INVALID_VIEW_POLICY, PREPROCESS_POLICY, CONDITIONING_POLICY, ANGLE_SOURCE, SELECTION_BIN_DEG)
from .bin15_dataset import mvimgnet_bin15_collate
from .epoch_sampling import RankSamples, SAMPLING_POLICY
from .train_bin15_pilot import _pilot_rdt_config, make_train_batch, save_sample_panel
from .prepare_full_bin15 import dump
from rescene.models.encoders import build_image_encoder
from rescene.codec.vae_latent_delta import load_vae

FEATURE_MODELS={'siglip_model': '/data1/MFM_models/rescene/ckpt/siglip-so400m-patch14-384', 'vae_id': '/data1/MFM_models/rescene/ckpt/sd-vae-ft-mse'}
SOURCE_NAMES=['train_direction_continuous_ddp.py','continuous_direction_dataset.py','roll_canonical_object.py','roll_direction_dataset.py','colmap_io.py','direction_bin15_dataset.py','direction_bin15_model.py',
              'canonical_object.py','bin15_model.py','bin15_dataset.py','bin15_camera.py',
              'full_bin15_dataset.py','epoch_sampling.py','train_bin15_pilot.py']


class LossModule(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,batch):return self.model.compute_loss(batch)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def state_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        t=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(tuple(t.shape)).encode());h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def learning_rate(step,total):
    if step<50:return 5e-5*(step+1)/50
    return 2.5e-5*(1+math.cos(math.pi*min(1,(step-50)/(total-50))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--scratch',action='store_true');source.add_argument('--resume',type=Path)
    p.add_argument('--steps',type=int,default=100000);p.add_argument('--smoke-steps',type=int,default=0)
    p.add_argument('--memory-cap-gib',type=float,default=14)
    args=p.parse_args()
    if args.steps<=50:raise ValueError('Total schedule must exceed warmup')
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    assert world==2 and 12<=args.memory_cap_gib<=16
    torch.set_num_threads(2);torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib*1024**3/torch.cuda.get_device_properties(rank).total_memory,rank)
    device=torch.device('cuda',rank)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30))
    readiness=json.loads((args.data_root/'readiness.json').read_text())
    assert readiness['state']=='ready' and readiness['archive_count']==41
    assert readiness['all_masks_audited'] and readiness['all_archives_crc_verified']
    # Reuse the verified all-view catalog; its original RGB policy is provenance,
    # while this run records its own preprocessing policy separately.
    assert readiness['masked_dataset_sha256']==digest(Path(__file__).with_name('masked_bin15_dataset.py'))
    preparation=json.loads((args.data_root/'preparation_config.json').read_text())
    assert preparation['scope']=='all_41_archives' and preparation['instance_limit'] is None
    for archive in preparation['archives']:
        stat=Path(archive['path']).stat()
        assert stat.st_size==archive['bytes'] and stat.st_mtime_ns==archive['mtime_ns']
    sources={name:digest(Path(__file__).with_name(name)) for name in SOURCE_NAMES}
    identity=digest(args.data_root/'readiness.json')
    output=args.output
    if rank==0:
        output.mkdir(parents=True,exist_ok=bool(args.resume))
        (output/'checkpoints').mkdir(exist_ok=True);(output/'samples').mkdir(exist_ok=True)
    dist.barrier()
    if (output/'STOP').exists():raise RuntimeError('STOP request still present; inspect before resuming')
    torch.manual_seed(0)
    def construct():
        return DirectionBin15RDT(rdt_config=_pilot_rdt_config(),img_token_dim=1152,lang_token_dim=4096,
                                tokens_per_image=64,k_max=5,dtype=torch.float32).to(device)
    if args.scratch:
        # Fail immediately if any generator initialization path tries to load weights.
        with patch.object(torch,'load',side_effect=RuntimeError('Scratch generator must not load checkpoint')), \
             patch.object(torch.nn.Module,'load_state_dict',side_effect=RuntimeError('Scratch generator must not import weights')):
            model=construct()
        checkpoint=None
    else:
        model=construct()
        checkpoint=torch.load(str(args.resume),map_location='cpu',weights_only=True,mmap=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=0)
    config={'preprocessing_policy':PREPROCESS_POLICY,'conditioning_policy':CONDITIONING_POLICY,
        'conditioning_angle_source':ANGLE_SOURCE,'selection_bin_deg':SELECTION_BIN_DEG,
        'camera_feature_dim':4,'camera_fields':['sin_az','cos_az','sin_el','cos_el'],
        'sources':sources,'data_identity':identity,'data_root':str(args.data_root),
        'dataset_counts':readiness['instances_by_split'],'classes':readiness['classes'],
        'world_size':2,'batch_per_gpu':16,'global_batch':32,'dtype':'fp32','workers_per_gpu':4,
        'memory_cap_gib_per_rank':args.memory_cap_gib,'steps':args.steps,'lr':5e-5,'warmup':50,
        'sampling_policy':SAMPLING_POLICY,'object_seed':0,'prediction_type':'sample',
        'latest_every':1000,'milestone_every':5000,'sample_seed':1234,'sample_every':1000,
        'fill':.8,'background_rgb':[128,128,128],'loss':'unchanged full latent MSE',
        'schedule':'scratch direction-only updates0to100000, cosine, fresh optimizer',
        'generator_initialization':'scratch_seed0_no_checkpoint', 'feature_models':FEATURE_MODELS,
        'invalid_view_policy':INVALID_VIEW_POLICY}
    if args.resume:
        for key,value in config.items():
            assert checkpoint['config'][key]==value, 'Resume configuration mismatch: '+key
        model.load_state_dict(checkpoint['model'],strict=True);optimizer.load_state_dict(checkpoint['optimizer'])
        config=checkpoint['config'];start=int(checkpoint['step']);saved_rng=checkpoint['rng_by_rank']
    else:
        initial_hash=state_digest(model.state_dict())
        config.update(initial_checkpoint=None,base_checkpoint_step=0,initial_checkpoint_sha256=None,
            generator_initial_state_sha256=initial_hash,optimizer_initial_state_entries=len(optimizer.state),
            generator_weight_loading_blocked_during_construction=True,
            generator_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            frozen_pretrained_components=['SigLIP','VAE'])
        assert not optimizer.state
        start=0;saved_rng=None
    del checkpoint
    wrapped=DDP(LossModule(model),device_ids=[rank],find_unused_parameters=True)
    encoder=build_image_encoder('siglip',device=device,siglip_model=FEATURE_MODELS['siglip_model'],tokens_per_image=64,dim=1152).eval()
    vae=load_vae(FEATURE_MODELS['vae_id'],device=device).eval()
    if saved_rng:
        torch.set_rng_state(saved_rng[rank]['cpu']);torch.cuda.set_rng_state(saved_rng[rank]['cuda'],device)
    else:torch.manual_seed(1234+rank)
    stop_step=min(args.steps,start+args.smoke_steps) if args.smoke_steps else args.steps
    if start>=stop_step:raise ValueError('Checkpoint has reached the requested final step')
    dataset=DirectionFullBin15Dataset(args.data_root/'catalog.sqlite','train',seed=0,samples_per_epoch=args.steps*32)
    loader=DataLoader(dataset,batch_size=16,sampler=RankSamples(start,stop_step,16,rank,world),num_workers=4,
        pin_memory=True,prefetch_factor=2,collate_fn=mvimgnet_bin15_collate,generator=torch.Generator().manual_seed(20261004+rank))
    config['objects_per_epoch']=len(dataset.rows)
    panels={}
    if rank==0:
        for split in ['train','val']:
            selected=DirectionFullBin15Dataset(args.data_root/'catalog.sqlite',split,deterministic=True,seed=0)
            panels[split]=mvimgnet_bin15_collate([selected[0]])
        # Keep evaluation input choices identical to the baseline extra panels.
        candidates=[]
        for index in sorted(set(int(i*(len(selected.rows)-1)/511) for i in range(512))):
            row,_,tid,_=selected._choose(index);view=next(v for v in row['views'] if v['image_id']==tid)
            candidates.append((view['foreground_fraction'],row['class_id'],index,row['instance_uid']))
        candidates.sort();used=set()
        for number in range(min(8,len(candidates))):
            pivot=int((number+.5)*len(candidates)/8)
            options=sorted(candidates,key=lambda v:abs(candidates.index(v)-pivot))
            choice=next((v for v in options if v[1] not in used),options[0]);used.add(choice[1])
            panels[f'val_q{number}_{choice[1]}']=mvimgnet_bin15_collate([selected[choice[2]]])
        dump(output/'fixed_evaluation_inputs.json',{name:batch['metadata'] for name,batch in panels.items()})
        dump(output/'config_used.json',config)
    # Rank0's data-only fixed panel setup must not change the model RNG stream.
    dist.barrier();model.train();started=time.time();stop_signal=[False]
    def request_stop(*_):stop_signal[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)

    def save(step):
        rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device).cpu()}
        gathered=[None]*world;dist.all_gather_object(gathered,rng)
        if rank==0:
            state={'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,'rng_by_rank':gathered}
            targets=[output/'checkpoints/latest.pt']
            if step%5000==0 or step==args.steps:targets.append(output/'checkpoints'/f'step_{step:06d}.pt')
            for target in targets:
                temporary=target.with_suffix('.tmp');torch.save(state,temporary);os.replace(temporary,target)
        dist.barrier()

    def sample(step):
        if rank==0:
            with torch.random.fork_rng(devices=[rank]):
                for name,batch in panels.items():
                    if '_q' in name and step%5000:continue
                    save_sample_panel(model,encoder,vae,batch,device,output/'samples'/f'step_{step:06d}_{name}.png',seed=1234)
        dist.barrier()

    # Save baseline examples for the new input distribution before optimizing.
    if start==0:
        save(0)
        sample(0)
    elif config.get('conditioning_transition_step')==start:
        # Preserve a same-weight baseline under the new conditioning before updates.
        sample(start)
    stopped=False;iterator=iter(loader)
    for step in range(start+1,stop_step+1):
        data_error=None
        try:batch=next(iterator)
        except Exception as error:
            batch=None;data_error=f'{type(error).__name__}: {error}'
        loaded=torch.tensor(int(data_error is None),device=device)
        dist.all_reduce(loaded,op=dist.ReduceOp.MIN)
        if not loaded.item():
            errors=[None]*world;dist.all_gather_object(errors,data_error)
            save(step-1)
            if rank==0:dump(output/'training_status.json',{'state':'failed_data','step':step-1,
                'next_step':step,'errors_by_rank':errors,'checkpoint_saved':True,'camera_feature_dim':4})
            raise RuntimeError('Data loading failed; last completed update saved: '+repr(errors))
        fallback_examples=[]
        for meta in batch['metadata']:
            if meta.get('rejected_views') or meta.get('view_replacements'):
                with (output/f'rejected_views_rank{rank}.jsonl').open('a') as log:
                    log.write(json.dumps({'step':step,'instance_uid':meta['instance_uid'],
                        'policy':INVALID_VIEW_POLICY,'rejected':meta['rejected_views'],
                        'replacements':meta['view_replacements']})+'\n')
            for frame,transform in zip(meta['context_frame_ids']+[meta['target_frame_id']],
                                       meta['context_transforms']+[meta['target_transform']]):
                if transform and transform.get('distortion_status','').startswith('identity_fallback'):
                    fallback_examples.append({'instance_uid':meta['instance_uid'],'frame':frame,
                        'source_k':transform['source_distortion_k']})
        if fallback_examples:
            with (output/f'preprocess_fallback_rank{rank}.jsonl').open('a') as log:
                log.write(json.dumps({'step':step,'count':len(fallback_examples),'views':fallback_examples})+'\n')
        lr=learning_rate(step-1,args.steps)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        loss=wrapped(make_train_batch(batch,encoder,vae,device))
        finite=torch.isfinite(loss).to(torch.int32);dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite.item():raise RuntimeError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        stop=torch.tensor(int(stop_signal[0] or (rank==0 and (output/'STOP').exists())),device=device)
        dist.all_reduce(stop,op=dist.ReduceOp.MAX);stopped=bool(stop.item())
        if step%25==0 or step==start+1 or step==stop_step or stopped:
            average=loss.detach().clone();dist.all_reduce(average)
            if rank==0:
                record={'state':'training','step':step,'target_step':args.steps,'base_checkpoint_step':config['base_checkpoint_step'],
                    'loss':float(average/world),'lr':lr,'elapsed_seconds':time.time()-started,'full_data_samples':step*32,
                    'object_epochs':step*32/len(dataset.rows),'camera_feature_dim':4,
                    'cuda_peak_reserved_gib_rank0':torch.cuda.max_memory_reserved(device)/1024**3}
                dump(output/'training_status.json',record);print(json.dumps(record),flush=True)
        if step%1000==0 or step==stop_step or stopped:
            save(step)
            if not stopped:sample(step)
        if stopped:break
    if rank==0:
        dump(output/'training_status.json',{'state':'stopped' if stopped else ('smoke_completed' if args.smoke_steps else 'completed'),
            'step':step,'target_step':args.steps,'base_checkpoint_step':config['base_checkpoint_step'],'camera_feature_dim':4})
    dist.destroy_process_group()


if __name__=='__main__':main()
