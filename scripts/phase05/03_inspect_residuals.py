import torch
from storyseed.decoder_components import DecoderBlock
from storyseed.build_evidence import record


def main():
    torch.set_num_threads(1);torch.manual_seed(17)
    norm=torch.nn.LayerNorm(2,eps=1e-5)
    illustrative=norm(torch.tensor([[1.,3.]]))
    block=DecoderBlock(8,2)
    states=torch.randn(2,4,8,requires_grad=True)
    valid=torch.ones(2,4,dtype=torch.bool)
    original=block(states,valid).detach()
    with torch.no_grad():
        block.attention.output.weight.zero_()
        block.feedforward.contract.weight.zero_();block.feedforward.contract.bias.zero_()
    identity=block(states,valid)
    assert torch.equal(identity,states)
    identity.sum().backward()
    assert torch.equal(states.grad,torch.ones_like(states))
    record(5,'03_residuals',{'command':'.venv/Scripts/python scripts/phase05/03_inspect_residuals.py',
       'architecture':'pre-LN: x+Attention(LN(x)), then x+FFN(LN(x)); padded states zeroed',
       'layernorm_example_input':[[1.,3.]],'layernorm_output':illustrative.detach().tolist(),
       'layernorm_population_variance':1.0,'layernorm_epsilon':1e-5,'ordinary_output_shape':list(original.shape),
       'zero_branches_preserve_state_exactly':True,'identity_path_input_gradient_is_one':True,
       'scope':'controlled zero-branch mechanism, not a claim that all deep gradients are stable','final_holdout_accessed':False})

if __name__=='__main__':main()
