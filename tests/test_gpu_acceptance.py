"""Real GPU acceptance, run on the GPU VM after the R0010004 trial:

    THETA_GPU_RUN=Output/runs/salon-sam3-001 .venv-gpu/bin/python -m unittest -v tests.test_gpu_acceptance

Skipped entirely without THETA_GPU_RUN. With it, every requirement must hold: nothing is
skipped and no fallback is accepted. Run this module alone: the other test modules need
the CPU environment. Optional THETA_SAM3_CHECKPOINT points to a local sam3.pt.
"""
import os
import re
import subprocess
import unittest
from pathlib import Path
from theta_pipeline.segmentation.environment import qualify
from theta_pipeline.segmentation.provenance import PACKAGE
from theta_pipeline.storage import read

RUN = os.environ.get('THETA_GPU_RUN')


@unittest.skipUnless(RUN, 'GPU acceptance runs only on the GPU VM with THETA_GPU_RUN set')
class GpuAcceptance(unittest.TestCase):
    run_dir = Path(RUN or '.').resolve()
    trial = run_dir / 'segmentation/trial'

    def scopes(self):
        """Trial artifacts always; all-panorama artifacts once --all has run."""
        return [self.trial] + ([self.run_dir] if (self.run_dir / 'semantic_masks.json').is_file() else [])

    def test_cuda_available(self):
        import torch
        self.assertTrue(torch.cuda.is_available())

    def test_triton_importable(self):
        import triton
        self.assertTrue(triton.__version__)

    def test_environment_matches_lock(self):
        result = qualify()
        self.assertTrue(result['qualified'], result['problems'])

    def test_checkpoint_accessible(self):
        path = os.environ.get('THETA_SAM3_CHECKPOINT')
        if not path:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt')
        self.assertTrue(Path(path).is_file())
        self.assertGreater(Path(path).stat().st_size, 0)

    def test_trial_pass(self):
        gate = read(self.trial / 'trial_gate.json')
        self.assertEqual(gate['panorama_id'], 'R0010004')
        self.assertEqual(gate['decision'], 'PASS', gate.get('gates'))

    def test_semantic_masks_present(self):
        for folder in self.scopes():
            masks = read(folder / 'semantic_masks.json')
            self.assertEqual(masks['backend'], 'sam3')
            self.assertTrue(masks['masks'])

    def test_provenance_records_commit_and_checkpoint(self):
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PACKAGE.parent, check=True,
                              capture_output=True, text=True).stdout.strip()
        for folder in self.scopes():
            provenance = read(folder / 'mask_provenance.json')
            self.assertEqual(provenance['git_commit'], head)
            self.assertRegex(provenance['checkpoint_sha256'], '^[0-9a-f]{64}$')
            self.assertTrue(provenance['model_and_code_hashed'], provenance['requirement'])

    def test_no_default_masks(self):
        for folder in self.scopes():
            masks = read(folder / 'semantic_masks.json')
            consistency = read(folder / 'mask_consistency.json')['panoramas']
            mask_root = folder if folder == self.trial else self.run_dir / 'segmentation/fused'
            for panorama in masks['panoramas']:
                pano = panorama['panorama_id']
                if panorama['status'] in ('UNKNOWN', 'FAILED'):
                    self.assertIsNone(panorama['artifacts'])
                    self.assertFalse((mask_root / pano).exists(), pano)
                    continue
                result = {c['name']: c['result'] for c in consistency[pano]['checks']}
                self.assertEqual(result['no_default_mask'], 'PASS', pano)
                self.assertTrue(re.fullmatch(r'OK|PARTIAL', panorama['status']))


if __name__ == '__main__':
    unittest.main()
