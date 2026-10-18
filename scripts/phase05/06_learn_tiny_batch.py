import json,subprocess,sys,time
import torch
from storyseed.corpus import ROOT,write_json
from storyseed.decoder import DecoderConfig,DecoderLanguageModel,parameter_ledger
from storyseed.decoder_io import save_inference,load_inference
from storyseed.story_batches import DocumentBatches,masked_next_token_loss
from storyseed.story_tokenizer import StoryTokenizer
from storyseed.context_generation import sample_context
from storyseed.build_evidence import record


def main():
    torch.set_num_threads(1);torch.manual_seed(17)
    config=DecoderConfig(context=32,dropout=0.0)
    model=DecoderLanguageModel(config)
    source=DocumentBatches('train',context=32)
    rows=[0,1,17,len(source)-1];batch=source.batch(rows)
    x,y,mask=batch['inputs'],batch['targets'],batch['loss_mask']
    initial=masked_next_token_loss(model(x),y,mask).item()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
    history=[];started=time.monotonic()
    for step in range(1,201):
        if time.monotonic()-started>40:raise RuntimeError('Tiny-batch compute deadline reached.')
        optimizer.zero_grad(set_to_none=True)
        loss=masked_next_token_loss(model(x),y,mask)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite tiny-batch objective.')
        loss.backward();optimizer.step()
        if step==1 or step%50==0:history.append({'step':step,'pre_update_loss':loss.item()})
    training_seconds=time.monotonic()-started
    model.eval()
    with torch.no_grad():
        final=masked_next_token_loss(model(x),y,mask).item()
        probe=x[:1,:12].clone();other=probe.clone();other[:,7:]=23
        earlier_delta=(model(probe)[:,:7]-model(other)[:,:7]).abs().max().item()
        assert earlier_delta<1e-6
        assert final<initial*0.25
    captured={}
    def capture(module,args,output):
        output.retain_grad();captured['states']=output
    handle=model.features.register_forward_hook(capture)
    model.zero_grad(set_to_none=True);model(probe)[:,5].square().mean().backward();handle.remove()
    future_gradient=captured['states'].grad[:,6:].abs().max().item()
    assert future_gradient==0
    path=ROOT/'checkpoints/phase05/tiny-decoder.pt'
    artifact=save_inference(model,path,'four fixed training chunks: debugging overfit, not corpus quality')
    restored,tokenizer=load_inference(path)
    with torch.no_grad():assert torch.equal(model(x),restored(x))
    expected=sample_context(restored,tokenizer)
    completed=subprocess.run([sys.executable,str(ROOT/'scripts/generate_decoder.py'),'--json'],cwd=ROOT,
        capture_output=True,text=True,encoding='utf-8',check=True,timeout=45)
    assert json.loads(completed.stdout)==expected
    write_json(ROOT/'data/manifests/phase05/tiny-decoder.json',artifact)
    record(5,'06_tiny_learning',{'command':'.venv/Scripts/python scripts/phase05/06_learn_tiny_batch.py',
       'configuration':config.to_dict(),'parameter_ledger':parameter_ledger(model),'selected_training_rows':rows,
       'true_targets_per_row':mask.sum(-1).tolist(),'initial_tiny_batch_nll':initial,'final_tiny_batch_nll':final,
       'updates':200,'learning_rate':0.01,'minibatch_history':history,'training_compute_seconds':training_seconds,
       'future_input_change_max_earlier_logit_difference':earlier_delta,'earlier_objective_future_state_gradient':future_gradient,
       'saved_weights_restore_same_logits':True,'fresh_process_sample_matches':True,'sample':expected,
       'checkpoint':artifact,'scope':'deliberate memorization of four training chunks, not development/final story quality',
       'development_scored':False,'final_holdout_accessed':False})

if __name__=='__main__':main()
