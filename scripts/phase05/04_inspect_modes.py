import torch
from storyseed.decoder_components import DecoderBlock
from storyseed.build_evidence import record


def main():
    torch.set_num_threads(1);torch.manual_seed(17)
    block=DecoderBlock(8,2,dropout=0.25)
    x=torch.randn(2,4,8);valid=torch.ones(2,4,dtype=torch.bool)
    with torch.no_grad():
        block.train();a=block(x,valid);b=block(x,valid)
        block.eval();c=block(x,valid);d=block(x,valid)
        dropout=torch.nn.Dropout(0.25)
        sample=dropout(torch.ones(10000))
    assert not torch.equal(a,b)
    assert torch.equal(c,d)
    record(5,'04_modes',{'command':'.venv/Scripts/python scripts/phase05/04_inspect_modes.py',
       'drop_probability':0.25,'train_outputs_differ_under_no_grad':True,'eval_outputs_identical':True,
       'inverted_dropout_observed_mean':sample.mean().item(),'nonzero_multiplier':1/0.75,
       'returned_attention_weights':'normalized pre-dropout weights; used weights receive dropout during training',
       'layernorm_has_running_statistics':False,'scope':'mode/randomness mechanism, not evidence of reduced overfitting','final_holdout_accessed':False})

if __name__=='__main__':main()
