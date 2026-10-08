"""Run the unchanged continuous-direction trainer with explicit encoder paths."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--scratch', action='store_true')
    source.add_argument('--resume')
    parser.add_argument('--siglip-model', help='Local SigLIP directory; default: original server path')
    parser.add_argument('--vae-model', help='Local VAE directory; default: original server path')
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--memory-cap-gib', type=float, default=14)
    parser.add_argument('--smoke-steps', type=int, default=0)
    args = parser.parse_args()
    # Bind bundled packages before legacy modules add their original server path.
    import models
    import rescene
    from plan_b.mvimgnet import train_direction_continuous_ddp as trainer
    if args.siglip_model:
        trainer.FEATURE_MODELS['siglip_model'] = args.siglip_model
    if args.vae_model:
        trainer.FEATURE_MODELS['vae_id'] = args.vae_model
    sys.argv = [sys.argv[0], '--data-root', args.data_root, '--output', args.output,
                '--steps', str(args.steps), '--memory-cap-gib', str(args.memory_cap_gib),
                '--smoke-steps', str(args.smoke_steps)]
    sys.argv += ['--scratch'] if args.scratch else ['--resume', args.resume]
    trainer.main()


if __name__ == '__main__':
    main()
