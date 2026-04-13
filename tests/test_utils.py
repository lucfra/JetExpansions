import torch

from jex.utils import CachedF


def test_cached_f():
    """Just to make sure lru cache behaves as expected with tensor inputs"""

    def f(z):
        return z + torch.rand_like(z)

    wrapped = CachedF(f)

    z = torch.randn(3)
    one = wrapped(z)
    two = wrapped(z)
    three = f(z)
    assert torch.allclose(one, two)
    assert not torch.allclose(one, three)
