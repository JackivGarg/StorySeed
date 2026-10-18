import json,subprocess,sys
import torch
from storyseed.corpus import ROOT,write_json
from storyseed.decoder_io import load_inference
from storyseed.context_generation import sample_context
from storyseed.build_evidence import record


def main():
    torch.set_num_threads(1)
    run=json.loads((ROOT/'runs/phase06/pilot-run.json').read_text(encoding='utf-8'))
    assert run['completed_requested_updates'] and run['actual_final_update']==10
    assert run['development_evaluations'][-1]['metrics']['prediction_positions']==39711
    path=ROOT/'checkpoints/phase06/pilot-inference.pt'
    model,tokenizer=load_inference(path);expected=sample_context(model,tokenizer)
    child=subprocess.run([sys.executable,str(ROOT/'scripts/generate_decoder.py'),'--checkpoint',str(path),'--json'],
         cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=30,check=True)
    assert json.loads(child.stdout)==expected
    write_json(ROOT/'data/manifests/phase06/pilot-artifacts.json',{'training':run['checkpoint'],'inference':run['inference_export']})
    record(6,'07_pipeline',{'command':'.venv/Scripts/python scripts/phase06/07_record_pipeline.py',
       'training_command':'.venv/Scripts/python scripts/train_decoder.py --config configs/phase06_teaching.json --updates 10 --checkpoint checkpoints/phase06/pilot-training.pt --export checkpoints/phase06/pilot-inference.pt --report runs/phase06/pilot-run.json --checkpoint-every 5 --evaluate-every 10 --max-seconds 25',
       'successful_updates':run['actual_final_update'],'full_development':run['development_evaluations'][-1],
       'total_sampled_scored_targets':sum(item['true_targets'] for item in run['history']),
       'first_batch_nll':run['history'][0]['nll'],'last_batch_nll':run['history'][-1]['nll'],
       'fresh_process_inference_matches':True,'sample':expected,'training_checkpoint':run['checkpoint'],
       'inference_export':run['inference_export'],'scope':'short CPU end-to-end machinery pilot, not the main SLM training or a quality comparison',
       'final_holdout_accessed':False})

if __name__=='__main__':main()
