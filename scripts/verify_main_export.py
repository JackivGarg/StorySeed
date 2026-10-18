"""Load the same final inference export in a fresh public CLI and compare its seeded result."""
import json
import subprocess
import sys
from pathlib import Path
import torch
from storyseed.corpus import ROOT
from storyseed.experiment_data import write_json
from storyseed.decoder_io import load_inference
from storyseed.context_generation import sample_context


def main():
    torch.set_num_threads(1)
    completed=json.loads((ROOT/'reports/phase07/main128.json').read_text(encoding='utf-8'))
    artifact=completed['inference']
    output=subprocess.run([sys.executable,str(ROOT/'scripts/generate_decoder.py'),'--checkpoint',artifact['path'],'--prompt','Once upon a time','--max-new-tokens','48','--json'],cwd=ROOT,check=True,capture_output=True,text=True,encoding='utf-8')
    model,tokenizer=load_inference(Path(artifact['path']))
    expected=sample_context(model,tokenizer,prompt='Once upon a time',max_new_tokens=48)
    actual=json.loads(output.stdout)
    if actual!=expected:raise RuntimeError('Fresh-process sample differs from the same exported artifact.')
    write_json(ROOT/'reports/phase07/04_fresh_inference.json',{'fresh_public_cli_matches':True,'inference':artifact,'sample':actual,'training_performed_by_generation':False})
    print(json.dumps({'fresh_public_cli_matches':True,'checkpoint_sha256':artifact['sha256']}))


if __name__=='__main__':main()
