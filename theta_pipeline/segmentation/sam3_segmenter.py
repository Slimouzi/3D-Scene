"""SAM 3 image adapter. GPU environment only (requirements/sam3-*.lock.txt)."""
from pathlib import Path
from ..storage import digest
from . import MODEL_ID, POLICY


class ModelUnavailable(RuntimeError):
    """Model, checkpoint or CUDA missing: the caller must report UNKNOWN, never a mask."""


class SAM3Segmenter:
    model_id = MODEL_ID

    def __init__(self, checkpoint_path=None, confidence_threshold=POLICY['confidence_threshold']):
        try:
            import torch
            from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as error:
            raise ModelUnavailable(f'SAM 3 import failed: {error}') from error
        if not torch.cuda.is_available():
            raise ModelUnavailable('CUDA unavailable: SAM3Segmenter runs only in the GPU environment')
        try:
            path = Path(checkpoint_path or download_ckpt_from_hf())
        except Exception as error:
            raise ModelUnavailable(f'Checkpoint download failed: {error}') from error
        if not path.is_file():
            raise ModelUnavailable(f'Checkpoint not found: {path}')
        self.torch = torch
        self.checkpoint_path = path
        self.checkpoint_sha256 = digest(path)
        model = build_sam3_image_model(device='cuda', checkpoint_path=str(path), load_from_HF=False)
        self.processor = Sam3Processor(model, device='cuda', confidence_threshold=confidence_threshold)
        self.environment = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                            'device': torch.cuda.get_device_name(0),
                            'autocast_dtype': 'bfloat16'}

    def segment(self, image, prompts):
        """Return {prompt: [(bool mask HxW, score), ...]} for one perspective face."""
        if image.width != image.height:
            raise ValueError('SAM3Segmenter accepts square perspective faces only, never a full ERP')
        torch = self.torch
        with torch.autocast('cuda', dtype=torch.bfloat16):
            state = self.processor.set_image(image)
        results = {}
        for prompt in prompts:
            self.processor.reset_all_prompts(state)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                state = self.processor.set_text_prompt(prompt, state)
            masks = state['masks'][:, 0].cpu().numpy()
            scores = state['scores'].float().cpu().numpy()
            results[prompt] = [(m, float(s)) for m, s in zip(masks, scores)]
        return results
