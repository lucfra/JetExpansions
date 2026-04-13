import re
from typing import Callable, Literal, overload

import torch
from torch import Tensor
from torch.autograd.functional import jvp


@overload
def jet(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    y: Tensor | None,
    k: int,
    *,
    recenter: bool = True,
    return_coefficients: Literal[False] = ...,
) -> Tensor: ...


@overload
def jet(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    y: Tensor | None,
    k: int,
    *,
    recenter: bool = True,
    return_coefficients: Literal[True],
) -> tuple[Tensor, list[Tensor], list[Tensor]]: ...


def jet(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    y: Tensor | None,
    k: int,
    *,
    recenter: bool = True,
    return_coefficients: bool = False,
) -> Tensor | tuple[Tensor, list[Tensor], list[Tensor]]:
    """Compute the k-th order jet expansion of f centred at x, evaluated at y.

    The jet is the truncated Taylor polynomial:
        jet^k f(x)(y) = f(x) + '(x)(y-x) + f''(x)(y-x)²/2! + ... + f^(k)(x)(y-x)^k/k!

    Args:
        f: Callable to expand. Must map a single Tensor to a Tensor (no batch dim).
        x: Center of the expansion.
        y: Point at which to evaluate the expansion.
        k: Order of the expansion.
        recenter: If True, expands around x (standard Taylor). If False, expands around
            the origin, i.e. f(x + y) ≈ f(x) + f'(x)·y + ...
        return_coefficients: If True, also returns the list of per-order coefficients
            (each divided by j!) and their cumulative partial sums.

    Returns:
        The jet value jet^k f(x)(y). If return_coefficients is True, returns a tuple
        (jet_value, coefficients, partial_sums) where coefficients[j] = f^(j)(x)·(y-x)^j / j!
        and partial_sums[j] = sum of coefficients up to order j.

    Note:
        Does not support batched inputs.
    """
    assert callable(f), "need a callable function"
    res = [f(x)]
    if k == 0:
        # quicker path for 0th order, since it's just function evaluation at the center
        if return_coefficients:
            return res[0], res, res
        else:
            return res[0]
    if y is None:
        raise ValueError("y cannot be None if k>=0")
    functions: list[Callable[[Tensor], Tensor]] = [f]
    yp = y.detach()
    if recenter:
        yp = yp - x.detach()

    def make_functional(i: int) -> Callable[[Tensor], Tensor]:
        def _f(_x: Tensor) -> Tensor:
            return jvp(functions[i], _x, yp, create_graph=True, strict=False)[1]  # type: ignore

        return _f

    for j in range(1, k + 1):
        functions.append(make_functional(j - 1))
        res.append(functions[j](x) / torch.math.factorial(j))

    sum_all: Tensor = sum(res)  # type: ignore[assignment]
    if return_coefficients:
        partial_sums = [sum(res[1 : j + 1], res[0]) for j in range(k + 1)]
        return sum_all, res, partial_sums
    return sum_all
