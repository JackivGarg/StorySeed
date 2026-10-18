"""Inference-only bundles; full training state is a separate Phase 6 contract."""
import torch
from storyseed.corpus import sha256
from storyseed.story_tokenizer import StoryTokenizer,TOKENIZER_PATH
from storyseed.decoder import DecoderConfig,DecoderLanguageModel


def save_inference(model,path,purpose):
    path.parent.mkdir(parents=True,exist_ok=True)
    payload={'format':'storyseed.decoder.inference.v1','configuration':model.config.to_dict(),
        'tokenizer_sha256':sha256(TOKENIZER_PATH.read_bytes()),'purpose':purpose,
        'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
    temporary=path.with_suffix('.pt.tmp');torch.save(payload,temporary);temporary.replace(path)
    return {'path':str(path),'sha256':sha256(path.read_bytes()),'format':payload['format'],
            'purpose':purpose,'optimizer_resume':False}


def load_inference(path, *, tokenizer_path=TOKENIZER_PATH):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('format')!='storyseed.decoder.inference.v1':raise ValueError('Expected a decoder inference bundle.')
    tokenizer=StoryTokenizer.load(tokenizer_path, expected_sha256=payload['tokenizer_sha256'])
    config=DecoderConfig(**payload['configuration'])
    if config.vocabulary!=tokenizer.vocab_size:raise ValueError('Vocabulary mismatch.')
    if any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in payload['state_dict'].values()):
        raise ValueError('Expected finite float32 decoder weights.')
    model=DecoderLanguageModel(config);model.load_state_dict(payload['state_dict'],strict=True)
    if model.features.token.lookup.weight[0].count_nonzero():raise ValueError('PAD row must remain zero.')
    return model.eval(),tokenizer
