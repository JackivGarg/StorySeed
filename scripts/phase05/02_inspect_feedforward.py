import torch
from storyseed.decoder_components import PositionwiseFeedForward
from storyseed.build_evidence import record


def main():
    torch.set_num_threads(1);torch.manual_seed(17)
    layer=PositionwiseFeedForward(8)
    states=torch.randn(2,4,8)
    output=layer(states)
    changed=states.clone();changed[:,1]+=2
    altered=layer(changed)
    unchanged=[0,2,3]
    assert torch.equal(output[:,unchanged],altered[:,unchanged])
    a,b=torch.randn(2,4,8),torch.randn(2,4,8)
    nonlinearity=(layer(a+b)-layer(a)-layer(b)+layer(torch.zeros_like(a))).norm().item()
    output.square().mean().backward()
    record(5,'02_feedforward',{'command':'.venv/Scripts/python scripts/phase05/02_inspect_feedforward.py',
       'input_shape':list(states.shape),'expanded_shape':list(layer.expand(states).shape),'output_shape':list(output.shape),
       'parameters':sum(p.numel() for p in layer.parameters()),'other_time_positions_unchanged':True,
       'affine_cancellation_residual_norm':nonlinearity,'expand_gradient_norm':layer.expand.weight.grad.norm().item(),
       'gelu_at_minus_one_zero_one':layer.activation(torch.tensor([-1.,0.,1.])).tolist(),
       'scope':'random nonlinear feature-processing illustration, not language training','final_holdout_accessed':False})

if __name__=='__main__':main()
