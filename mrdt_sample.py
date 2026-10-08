"""Generate a target view from catalog context images and continuous angles."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='Required unless --prepare-only')
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--context-ids', type=int, nargs='+', help='Optional 1–5 catalog image IDs')
    parser.add_argument('--azimuth', type=float, required=True, help='Degrees in the object reference basis')
    parser.add_argument('--elevation', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--siglip-model')
    parser.add_argument('--vae-model')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--prepare-only', action='store_true', help='Save context inputs without loading weights or using GPU')
    args = parser.parse_args()
    if not args.prepare_only and args.checkpoint is None:
        parser.error('--checkpoint is required for generation')
    if args.steps < 1 or args.index < 0:
        parser.error('--steps must be positive and --index nonnegative')
    import models
    import rescene
    import torch
    from plan_b.mvimgnet.continuous_direction_dataset import (
        ContinuousDirectionDataset, context_inputs, CONDITIONING_POLICY, PREPROCESS_POLICY)
    from plan_b.mvimgnet.roll_canonical_object import reference_basis
    from plan_b.mvimgnet.direction_bin15_model import DirectionBin15RDT
    from plan_b.mvimgnet.train_bin15_pilot import _pilot_rdt_config, _tensor_to_pil
    from plan_b.mvimgnet.bin15_model import patch_tokens_to_latent
    from rescene.models.encoders import build_image_encoder
    from rescene.codec.vae_latent_delta import load_vae

    dataset = ContinuousDirectionDataset(args.catalog, args.split, deterministic=True, seed=0)
    if args.index >= len(dataset.rows):
        parser.error('--index exceeds the selected split')
    if args.context_ids:
        row = dataset.rows[args.index]
        ids = args.context_ids
    else:
        row, ids, _, _ = dataset._choose(args.index)
    if len(ids) != len(set(ids)):
        parser.error('--context-ids must be distinct')
    views = {view['image_id']: view for view in row['views']}
    if any(i not in views for i in ids):
        parser.error('A context image ID is absent from this object')
    # Only the selected contexts are opened; target RGB/mask are never loaded.
    mi, transforms = context_inputs([views[i] for i in ids], args.azimuth, args.elevation,
                                    basis=reference_basis(row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for slot in range(len(ids)):
        _tensor_to_pil(mi['context_images'][slot]).save(args.output.with_name(args.output.stem + f'_context{slot}.png'))
    metadata = {'instance_uid': row['instance_uid'], 'context_image_ids': ids,
                'context_angles_deg': [[views[i]['azimuth_deg'], views[i]['elevation_deg']] for i in ids],
                'target_angles_deg': [args.azimuth, args.elevation],
                'conditioning_policy': CONDITIONING_POLICY, 'preprocessing_policy': PREPROCESS_POLICY,
                'seed': args.seed, 'steps': args.steps, 'prepare_only': args.prepare_only,
                'context_transforms': transforms}
    if not args.prepare_only:
        checkpoint = torch.load(str(args.checkpoint), map_location='cpu', weights_only=True, mmap=True)
        config = checkpoint['config']
        if config.get('conditioning_policy') != CONDITIONING_POLICY or config.get('preprocessing_policy') != PREPROCESS_POLICY:
            raise ValueError('Checkpoint is not the current continuous-direction/roll-aligned model')
        if config.get('camera_feature_dim') != 4 or config.get('prediction_type') != 'sample':
            raise ValueError('Checkpoint has an incompatible camera/output contract')
        feature = config['feature_models']
        device = torch.device(args.device)
        model = DirectionBin15RDT(rdt_config=_pilot_rdt_config(), img_token_dim=1152,
            lang_token_dim=4096, tokens_per_image=64, k_max=5, dtype=torch.float32).to(device).eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        metadata['checkpoint_step'] = int(checkpoint['step'])
        del checkpoint
        encoder = build_image_encoder('siglip', device=device,
            siglip_model=args.siglip_model or feature['siglip_model'], tokens_per_image=64, dim=1152).eval()
        vae = load_vae(args.vae_model or feature['vae_id'], device=device).eval()
        with torch.no_grad():
            features = encoder.encode(mi['context_images'].to(device)).reshape(1, 5, 64, 1152)
            batch = {key: mi[key].unsqueeze(0).to(device) for key in
                     ['context_camera_features', 'target_camera_features', 'context_present']}
            batch['slot_feats'] = features.float()
            tokens = model.predict(batch, num_steps=args.steps, seed=args.seed)
            generated = vae.decode(patch_tokens_to_latent(tokens.float()))[0]
            _tensor_to_pil(generated).save(args.output)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'prepared_contexts': len(ids), 'generated': not args.prepare_only, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
