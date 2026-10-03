import argparse
import sys
import pycolmap
from .storage import Run
from . import stages


def main():
    parser = argparse.ArgumentParser(description='Theta Z1 — local CPU diagnostic pipeline')
    parser.add_argument('action', choices=['audit', 'auto-mask', 'prepare', 'seg-faces', 'seg-trial',
                                           'sfm', 'auto-gates', 'partition', 'report', 'diagnostic'])
    parser.add_argument('--config', default='configs/salon.json')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        if pycolmap.__version__ != '4.2.1':
            raise RuntimeError('This adapter requires official pycolmap==4.2.1')
        with Run(args.config, args.run_id).locked() as run:
            pycolmap.set_random_seed(run.config['seed'])
            sam3 = run.config.get('auto_mask_backend') == 'sam3'
            masked = bool(run.config.get('semantic_run'))
            sfm = ['features', 'matching', 'mapping', 'diagnose']
            if masked:
                # New SfM experiment fed by the geometry masks of an ACCEPTED segmentation run.
                pipeline = ['audit', 'auto_mask', 'prepare', *sfm, 'auto_gates', 'partition', 'report']
            elif sam3:
                # The GPU runner segments between seg_faces and seg_trial, and before auto_mask.
                pipeline = ['audit', 'prepare', 'seg_faces', 'seg_trial', 'auto_mask', *sfm,
                            'auto_gates', 'partition', 'report']
            else:
                pipeline = ['audit', 'auto_mask', 'prepare', *sfm, 'auto_gates', 'partition', 'report']
            dependencies = {'auto_mask': ('seg_trial',) if sam3 and not masked else ('audit',),
                            'prepare': ('audit',), 'seg_faces': ('prepare',),
                            'seg_trial': ('seg_faces',), 'features': ('prepare',),
                            'matching': ('features',),
                            'mapping': ('matching',), 'diagnose': ('mapping',),
                            'auto_gates': ('auto_mask', 'diagnose'),
                            'partition': ('diagnose', 'auto_gates'), 'report': ('diagnose', 'auto_gates')}
            if args.action == 'diagnostic':
                requested = pipeline
            elif args.action == 'sfm':
                requested = [stage for stage in pipeline if stage != 'audit']
            else:
                requested = [args.action.replace('-', '_')]
            for stage in requested:
                run.stage(stage, getattr(stages, stage), dependencies.get(stage, ()))
            print(f'Artifacts: {run.path}')
    except (Exception, KeyboardInterrupt) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
