import torch
from storyseed.decoder import DecoderConfig,DecoderLanguageModel,parameter_ledger
from storyseed.build_evidence import record


def main():
    torch.set_num_threads(1);torch.manual_seed(17)
    model=DecoderLanguageModel(DecoderConfig()).eval()
    ids=torch.tensor([[1,47,75,67,274,388]],dtype=torch.long)
    shapes=[list(model.features(ids).shape)]
    handles=[block.register_forward_hook(lambda module,args,output:shapes.append(list(output.shape))) for block in model.blocks]
    with torch.no_grad():scores=model(ids)
    for handle in handles:handle.remove()
    ledger=parameter_ledger(model)
    assert ledger['total']==60032
    record(5,'05_stack',{'command':'.venv/Scripts/python scripts/phase05/05_inspect_stack.py',
        'configuration':model.config.to_dict(),'ids':ids.tolist(),'feature_and_block_shapes':shapes,'logits_shape':list(scores.shape),
        'parameter_ledger':ledger,'token_and_output_weights_are_separate':model.features.token.lookup.weight is not model.output.weight,
        'initialization':'normal std 0.02 for linear/embedding weights, zero linear biases, fixed zero PAD row, default LayerNorm affine',
        'next_need':'demonstrate tiny-batch learning and full-stack causal information boundaries','final_holdout_accessed':False})

if __name__=='__main__':main()
