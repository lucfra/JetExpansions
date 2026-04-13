import torch
from jex.jet import jet


# --- basic polynomial tests ---


def test_jet_base():
    torch.random.manual_seed(1)
    dims = (10, 20)
    x = torch.randn(*dims, requires_grad=True)
    f = lambda _x: _x + _x**2 + _x**3
    fx = f(x)

    y = torch.randn(*dims)

    jet0 = jet(f, x, y, 0)

    print(jet0)

    assert torch.allclose(fx, jet0)  # this is just the function evaluated at x

    # now let's compute jet3, this should be equal to the function evaluated at f(y)

    jet1 = jet(f, x, y, 1, recenter=False)
    print(jet1)

    jet2 = jet(f, x, y, 2, recenter=False)
    jet3 = jet(f, x, y, 3, recenter=False)
    jet4 = jet(f, x, y, 4, recenter=False)

    print(jet3)

    fy = (x + y) + (x + y) ** 2 + (x + y) ** 3

    assert torch.allclose(fy, jet3, atol=1.0e-5)
    assert torch.allclose(
        fy, jet4, atol=1.0e-5
    )  # making order 4 does not change for this function!
    assert not torch.allclose(fy, jet2, atol=1.0e-5)  # this should not be the same!

    # make sure subtract works
    jet3_s = jet(f, x, y, 3, recenter=True)
    fy_s = y + y**2 + y**3

    assert torch.allclose(fy_s, jet3_s, atol=1.0e-5)


def test_quadratic():
    dim = d = 10
    W = torch.randn(dim, dim, dim // 2)
    f = lambda x: torch.einsum("i,j,ijk->k", x, x, W)
    other_f = lambda x: torch.stack([x @ W[..., i] @ x for i in range(dim // 2)])
    x1 = torch.randn(d, requires_grad=True)
    x2 = torch.randn(d, requires_grad=True)
    fx1 = f(x1)
    fx2 = f(x2)
    fx1x2 = f(x1 + x2)
    print(fx1 + fx2, "\n", fx1x2, "\n", fx2)
    assert not torch.allclose(fx1 + fx2, fx1x2)
    assert torch.allclose(fx1, other_f(x1))

    jet1 = jet(f, x1, x2, 1)
    jet2 = jet(f, x1, x2, 2)
    jet3 = jet(f, x1, x2, 3)

    print(jet1, jet2, jet3, sep="\n")
    assert torch.allclose(jet2, fx2, atol=1e-5, rtol=1e-5)
    assert torch.allclose(jet3, jet2, atol=1e-5, rtol=1e-5)  # this will be the same
    assert not torch.allclose(jet1, fx2, atol=1e-3, rtol=1e-3)


def test_cubic():
    dim = d = 10
    W = torch.randn(dim, dim, dim, dim // 2)
    f = lambda x: torch.einsum("i,j,r,ijrk->k", x, x, x, W)
    x1 = torch.randn(d, requires_grad=True)
    x2 = torch.randn(d, requires_grad=True)
    fx1 = f(x1)
    fx2 = f(x2)
    fx1x2 = f(x1 + x2)
    print(fx1 + fx2, "\n", fx1x2, "\n", fx2)
    assert not torch.allclose(fx1 + fx2, fx1x2)

    jet2 = jet(f, x1, x2, 2)
    jet3 = jet(f, x1, x2, 3)
    jet4 = jet(f, x1, x2, 4)

    print(jet2, jet3, jet4, sep="\n")
    assert torch.allclose(jet3, fx2, atol=1.0e-4, rtol=1.0e-4)
    assert torch.allclose(jet4, jet3, atol=1.0e-4, rtol=1.0e-4)  # this will be the same
    assert not torch.allclose(jet2, fx2, atol=1.0e-4, rtol=1.0e-4)


def test_nonlinear():
    """For nonlinear case, checking that as x2 gets close to x1, so does the jet"""
    d = 10

    linear1 = torch.nn.Linear(d, d)
    linear1.bias.data = torch.randn_like(linear1.bias)
    linear1.weight.data = torch.randn_like(linear1.weight)

    x1 = torch.randn(d, requires_grad=True)
    eps = torch.randn(d, requires_grad=True)

    sigmoid = torch.nn.ELU()
    f = lambda _x: sigmoid(linear1(_x))

    out_l = f(x1 + eps)

    print(out_l)

    intervals = [0, 0.001, 0.1, 0.2, 0.5]
    jets2 = [jet(f, x1, x1 + t * eps, 2) for t in intervals]
    jets3 = [jet(f, x1, x1 + t * eps, 3) for t in intervals]
    jets4 = [jet(f, x1, x1 + t * eps, 4) for t in intervals]

    norms = {}
    for kk, jets in enumerate([jets2, jets3, jets4]):
        norms[kk] = []
        for k, t in enumerate(intervals):
            out = f(x1 + t * eps)

            norms[kk].append(torch.norm(out - jets[k]).item())
            if t == 0:
                assert torch.allclose(out, jets[0], atol=1.0e-4, rtol=1.0e-4)
        print(norms)


# --- multivariate tests with analytical derivatives ---


def f_mv(x):
    return torch.sin(x[0] + 3 * x[1] + x[0] ** 3 * x[1])


def grad_f_mv(x):
    inner = x[0] + 3 * x[1] + x[0] ** 3 * x[1]
    df_dx0 = (1 + 3 * x[0] ** 2 * x[1]) * torch.cos(inner)
    df_dx1 = (3 + x[0] ** 3) * torch.cos(inner)
    return torch.stack([df_dx0, df_dx1])


def hess_f_mv(x):
    inner = x[0] + 3 * x[1] + x[0] ** 3 * x[1]
    c = torch.cos(inner)
    s = torch.sin(inner)
    d2f_dx0x0 = 6 * x[0] * x[1] * c - (1 + 3 * x[0] ** 2 * x[1]) ** 2 * s
    d2f_dx1x1 = -((3 + x[0] ** 3) ** 2) * s
    d2f_dx0x1 = 3 * x[0] ** 2 * c - (3 + x[0] ** 3) * (1 + 3 * x[0] ** 2 * x[1]) * s
    return torch.stack(
        [
            torch.stack([d2f_dx0x0, d2f_dx0x1]),
            torch.stack([d2f_dx0x1, d2f_dx1x1]),
        ]
    )


def jet1_mv(x, y):
    return f_mv(x) + grad_f_mv(x) @ (y - x)


def jet2_mv(x, y):
    return jet1_mv(x, y) + 1 / 2 * (hess_f_mv(x) @ (y - x)) @ (y - x)


def test_hessian():
    x = torch.randn(2, requires_grad=True)
    auto_hess = torch.autograd.functional.hessian(f_mv, x)
    anal_hess = hess_f_mv(x.detach())
    assert torch.allclose(auto_hess, anal_hess), (
        f"\nautograd:\n{auto_hess}\nanalytical:\n{anal_hess}"
    )


def test_jet_k1():
    x = torch.randn(2)
    y = torch.randn(2)
    assert f_mv(x).ndim == 0

    j1 = jet(f_mv, x, y, recenter=True, k=1)
    assert torch.allclose(jet1_mv(x, y), j1)


def test_jet_k2():
    x = torch.randn(2)
    y = torch.randn(2)

    j2 = jet(f_mv, x, y, recenter=True, k=2)
    assert torch.allclose(jet2_mv(x, y), j2), (
        f"\nanalytical: {jet2_mv(x, y)}\njet: {j2}"
    )
