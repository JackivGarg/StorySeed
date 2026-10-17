"""Trace trained attention, information boundaries and compatible development scores."""
import torch
from storyseed.corpus import ROOT
from storyseed.attention_training import load_attention_model
from storyseed.context_training import load_context_model, evaluate_scores
from storyseed.context_generation import predict_contexts, history_context, sample_context, context_width
from storyseed.fixed_context import CausalWindows
from storyseed.story_baseline import load_baseline
from storyseed.phase04_evidence import record


def main():
    torch.set_num_threads(1)
    model,tokenizer = load_attention_model()
    prompt = 'Mia found a small red box.'
    ids = torch.tensor([tokenizer.encode(prompt,add_bos=True)],dtype=torch.long)
    assert ids.shape[1] > 7
    with torch.no_grad():
        states = model.features(ids)
        q,k,v = model.attention.projection(states)
        retrieved,weights = model.attention(states,ids != 0,return_weights=True)
        logits = model(ids)
        final = model.forward_contexts(ids)
        final_delta = (logits[:,-1]-final).abs().max().item()
        assert torch.allclose(logits[:,-1],final,atol=1e-5,rtol=1e-6)
        altered_ids = ids.clone()
        altered_ids[:,7:] = 23
        altered_logits = model(altered_ids)
        future_delta = (logits[:,:7]-altered_logits[:,:7]).abs().max().item()
        assert torch.allclose(logits[:,:7],altered_logits[:,:7],atol=1e-6,rtol=0)
        suffix = [350,387,377,407,14,282,314,85]
        history_a,history_b = [1,17]+suffix,[1,19]+suffix
        za = model.forward_contexts(history_context(model,history_a))
        zb = model.forward_contexts(history_context(model,history_b))
        beyond_ann_delta = (za-zb).abs().max().item()
        train = CausalWindows('train',64)
        suffix64 = train.batch([64])[0][0].tolist()
        outside_a = model.forward_contexts(history_context(model,[1,17]+suffix64))
        outside_b = model.forward_contexts(history_context(model,[1,19]+suffix64))
        assert torch.equal(outside_a,outside_b)
        padded_contexts,_ = train.batch([0,64])
        _,padded_weights = model.attention(model.features(padded_contexts),padded_contexts != 0,
                                         last_only=True,return_weights=True)
        padding_mass = (padded_weights*(padded_contexts == 0)[:,None,:]).sum().item()
        assert padding_mass == 0
    model.zero_grad(set_to_none=True)
    differentiable_states = model.features(ids)
    differentiable_states.retain_grad()
    predicted = model.output(model.attention(differentiable_states,ids != 0))
    predicted[:,5].square().mean().backward()
    future_state_gradient = differentiable_states.grad[:,6:].abs().max().item()
    assert future_state_gradient == 0
    gradient_norms = {name: layer.weight.grad.norm().item() for name,layer in [
        ('query',model.attention.projection.query),('key',model.attention.projection.key),
        ('value',model.attention.projection.value),('token',model.features.token.lookup),
        ('position',model.features.position),('vocabulary_head',model.output)]}
    bigram,_ = load_baseline()
    k1,_ = load_context_model(ROOT/'checkpoints/phase03/context_k1.pt')
    k8,_ = load_context_model(ROOT/'checkpoints/phase03/context_k8.pt')
    models = {'phase02_bigram':bigram,'phase03_k1':k1,'phase03_k8':k8,'phase04_attention':model}
    metrics,samples = {},{}
    for name,predictor in models.items():
        data = CausalWindows('development',context_width(predictor))
        metrics[name] = evaluate_scores(lambda x: predict_contexts(predictor,x),data,batch_size=512)
        samples[name] = sample_context(predictor,tokenizer)
    record('06_integration',{
        'command':'.venv/Scripts/python scripts/phase04/06_inspect_model.py',
        'trace_prompt':prompt,'trace_ids':ids.tolist(),
        'shapes':{name:list(value.shape) for name,value in [
            ('ids',ids),('states',states),('query',q),('key',k),('value',v),
            ('attention_weights',weights),('retrieved',retrieved),('logits',logits),('final_logits',final)]},
        'last_query_weights':weights[:,-1].tolist(),
        'full_vs_final_query_logits_max_difference':final_delta,
        'future_input_change_max_earlier_logit_difference':future_delta,
        'position_5_objective_future_state_gradient_max':future_state_gradient,
        'gradient_norms_for_position_5_diagnostic_objective':gradient_norms,
        'earlier_change_outside_ann8_but_inside_attention64_max_logit_difference':beyond_ann_delta,
        'outside_attention64_prefix_change_has_identical_logits':True,
        'padding_source_weight_mass':padding_mass,
        'development_metrics':metrics,'same_prediction_population':39711,
        'samples_common_prompt_seed_temperature_top_k_budget':samples,
        'sample_scope':'unselected illustrations; not final held-out or blinded quality scores',
        'final_holdout_accessed':False})


if __name__ == '__main__':
    main()
