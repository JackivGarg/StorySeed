"""CPU inference for declared local model releases; no training path."""
from dataclasses import dataclass
import math
import json
from pathlib import Path
from time import perf_counter

import torch

from storyseed.context_generation import sample_context
from storyseed.corpus import ROOT, sha256
from storyseed.decoder_io import load_inference
from storyseed.story_tokenizer import TOKENIZER_PATH

MODEL_SHA256 = 'f0f7241af4770372738c3c9b057e205cc60e66cc7fa1d21397a6281f383c0eed'
TOKENIZER_SHA256 = '3540aba1c244ed063b9fb317f0d62280a63d00725cac980e16c3b583cdf549cb'


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str = 'Once upon a time'
    seed: int = 17
    temperature: float = 0.8
    top_k: int = 40
    max_new_tokens: int = 192

    def validate(self):
        if not isinstance(self.prompt, str) or len(self.prompt) > 4000:
            raise ValueError('Use a prompt of at most 4,000 characters. Empty prompts are allowed.')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('Seed must be an integer between 0 and 4,294,967,295.')
        if type(self.temperature) not in (int, float) or not math.isfinite(self.temperature) or not 0.05 <= self.temperature <= 2:
            raise ValueError('Temperature must be a finite number between 0.05 and 2.')
        if type(self.top_k) is not int or not 1 <= self.top_k <= 4094:
            raise ValueError('Top-k must be an integer between 1 and 4094.')
        if type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= 512:
            raise ValueError('New-token budget must be an integer between 1 and 512.')


class GenerationService:
    def __init__(self, checkpoint=None, tokenizer_path=None, *, release=None):
        if release is None and checkpoint is None and tokenizer_path is None:
            release_path = ROOT / 'artifacts/application_release.json'
            if release_path.is_file():
                release = json.loads(release_path.read_text(encoding='utf-8'))
        if release is not None:
            if release.get('format') != 'storyseed.application-release.v1':
                raise ValueError('Unknown application release format.')
            checkpoint = checkpoint if checkpoint is not None else ROOT / release['checkpoint']
            tokenizer_path = tokenizer_path if tokenizer_path is not None else ROOT / release['tokenizer']
        expected_model = release['model_sha256'] if release else MODEL_SHA256
        expected_tokenizer = release['tokenizer_sha256'] if release else TOKENIZER_SHA256
        expected_context = release['context'] if release else 128
        checkpoint = Path(checkpoint) if checkpoint is not None else ROOT / 'checkpoints/phase07/main128-inference.pt'
        tokenizer_path = Path(tokenizer_path) if tokenizer_path is not None else TOKENIZER_PATH
        for label, path, expected in [('model', checkpoint, expected_model), ('tokenizer', tokenizer_path, expected_tokenizer)]:
            if not path.is_file():
                raise FileNotFoundError(f'Missing {label}: {path}. See docs/LOCAL_APP.md for bundle/setup instructions.')
            if sha256(path.read_bytes()) != expected:
                raise ValueError(f'The {label} is not the declared StorySeed application release.')
        torch.set_num_threads(1)
        self.model, self.tokenizer = load_inference(checkpoint, tokenizer_path=tokenizer_path)
        if self.model.config.context != expected_context:
            raise ValueError('The model context differs from its declared application policy.')
        self.metadata = {'model': release['model'] if release else 'StorySeed-main128', 'model_sha256': expected_model,
                         'tokenizer_sha256': expected_tokenizer, 'parameters': sum(p.numel() for p in self.model.parameters()),
                         'context': expected_context, 'device': 'cpu', 'precision': 'float32', 'training_performed': False}

    @torch.inference_mode()
    def generate(self, request: GenerationRequest):
        request.validate()
        if request.top_k > self.tokenizer.vocab_size - 2:
            raise ValueError('Top-k exceeds the loaded model vocabulary after PAD/BOS exclusion.')
        prompt_ids = self.tokenizer.encode(request.prompt, add_bos=True)
        started = perf_counter()
        result = sample_context(self.model, self.tokenizer, request.prompt, seed=request.seed,
                                temperature=request.temperature, top_k=request.top_k,
                                max_new_tokens=request.max_new_tokens,
                                compact=self.model.config.context == 256)
        return result | {'continuation': self.tokenizer.decode(result['generated_ids']), 'release': dict(self.metadata), 'prompt_ids_including_bos': len(prompt_ids),
                         'initial_visible_ids': min(len(prompt_ids), self.model.config.context),
                         'initial_cropped_ids': max(0, len(prompt_ids) - self.model.config.context),
                         'compact_inference': self.model.config.context == 256, 'elapsed_seconds': perf_counter() - started}
