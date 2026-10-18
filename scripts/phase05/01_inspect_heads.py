import torch
from storyseed.attention import causal_attention
from storyseed.decoder_components import MultiHeadCausalAttention
from storyseed.build_evidence import record


def main():
    torch.set_num_threads(1);torch.manual_seed(17)
    layer=MultiHeadCausalAttention(8,2)
    x=torch.randn(2,4,8)
    valid=torch.tensor([[True]*4,[False,False,True,True]])
    output,weights=layer(x,valid,return_weights=True)
    q,k,v=[layer.split_heads(projection(x)) for projection in (layer.query,layer.key,layer.value)]
    pieces=[causal_attention(q[:,h],k[:,h],v[:,h],valid)[0] for h in range(2)]
    reference=layer.output(torch.cat(pieces,dim=-1))
    difference=(reference-output).abs().max().item()
    assert torch.allclose(reference,output,atol=1e-6,rtol=0)
    output.square().mean().backward()
    record(5,'01_heads',{'command':'.venv/Scripts/python scripts/phase05/01_inspect_heads.py',
       'input_shape':list(x.shape),'split_query_shape':list(q.shape),'weight_shape':list(weights.shape),
       'output_shape':list(output.shape),'parameters':sum(p.numel() for p in layer.parameters()),
       'head_loop_reference_max_difference':difference,'head_weight_difference_norm':(weights[:,0]-weights[:,1]).norm().item(),
       'future_weight_mass':weights.triu(1).sum().item(),'query_gradient_norm':layer.query.weight.grad.norm().item(),
       'scope':'seeded random mechanism illustration, not learned semantic head specialization','final_holdout_accessed':False})

if __name__=='__main__':main()
