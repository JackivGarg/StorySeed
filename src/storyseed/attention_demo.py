"""Hand-set numerical illustrations; these are not learned language features."""
import torch
from storyseed.attention import QKVProjection


def projection_fixture():
    features = torch.tensor([[[1., 0., 1., 0.],
                              [0., 1., 0., 1.],
                              [1., 1., 1., 0.]]])
    projection = QKVProjection(4, 2)
    with torch.no_grad():
        projection.query.weight.copy_(torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.]]))
        projection.key.weight.copy_(torch.tensor([[0.,0.,1.,0.],[0.,0.,0.,1.]]))
        projection.value.weight.copy_(torch.tensor([[1.,0.,1.,0.],[0.,1.,0.,1.]]))
    return features, projection
