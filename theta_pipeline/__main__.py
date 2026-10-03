import argparse
import sys
import pycolmap
from .storage import Run
from . import stages


def main():
    parser = argparse.ArgumentParser(description='Theta Z1 — local CPU diagnostic pipeline')
    parser.add_argument('action', choices=['audit', 'auto-mask', 'prepare', 'sfm', 'auto-gates', 'partition', 'report', 'diagnostic'])
    parser.add_argument('--config', default='configs/salon.json')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        if pycolmap.__version__ != '4.2.1':
            raise RuntimeError('This adapter requires official pycolmap==4.2.1')
        with Run(args.config, args.run_id).locked() as run:
            pycolmap.set_random_seed(run.config['seed'])
            pipeline = ['audit', 'auto_mask', 'prepare', 'features', 'matching', 'mapping', 'diagnose', 'auto_gates', 'partition', 'report']
            dependencies = {'auto_mask': ('audit',), 'prepare': ('audit',), 'features': ('prepare',),
                            'matching': ('features',),
                            'mapping': ('matching',), 'diagnose': ('mapping',),
                            'auto_gates': ('auto_mask', 'diagnose'),
                            'partition': ('diagnose', 'auto_gates'), 'report': ('diagnose', 'auto_gates')}
            if args.action == 'diagnostic':
                requested = pipeline
            elif args.action == 'sfm':
                requested = ['auto_mask', 'prepare', 'features', 'matching', 'mapping', 'diagnose', 'auto_gates', 'partition', 'report']
            else:
                requested = [{'auto-mask': 'auto_mask', 'auto-gates': 'auto_gates'}.get(args.action, args.action)]
            for stage in requested:
                run.stage(stage, getattr(stages, stage), dependencies.get(stage, ()))
            print(f'Artifacts: {run.path}')
    except (Exception, KeyboardInterrupt) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
