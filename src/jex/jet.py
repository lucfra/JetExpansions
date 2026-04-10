import torch  # needs last version for one kwarg in autograd
from torch.autograd.functional import jvp


def jet(f, x, y, k, recenter=True, return_coefficients=False):
    """
    Does not work for batched input!  # TODO batchify (probably not)

    :param f: a callable (to be evaluated on x).
    :param x: center, in the domain of f, just a tensor for the time being
    :param y: variate in the domain of f, just a tensor for the time being
    :param k: order
    :param recenter: if True, f(y) = f(x) + f'(x) * (y - x) + ... If False, f(x+y) = f(x) + f'(x) * y + ...
    :param return_coefficients: if True, returns also all the list of coefficients of the polynomial (useful for
                                    including all orders before k for computing) and their cumulative sum

    :return: jet^k f(x)(y) .    If return_coefficients is true returns also the list of coefficients of the polynomial
                            and the partial sums
    """
    assert callable(f), "need a callable function"
    res, funcs = [f(x)], [f] + [None]*k
    yp = y.detach()
    if recenter:
        yp = yp.detach() - x.detach()  # detach here, otherwise will compute derivative also wrt x...

    def make_functional(i):
        def _f(_x):
            # noinspection PyTypeChecker
            return jvp(funcs[i], _x, yp, create_graph=True, strict=False)[1] # jvp returns f(x), D f(x) v
        return _f

    for j in range(1, k + 1):  # todo change this part to do all in once using the tuple signature of jvp
        # construct the next functional
        funcs[j] = make_functional(j - 1)
        # compute its value
        d_prev = funcs[j](x)
        res.append(d_prev / torch.math.factorial(j))  # divide by j!
    sum_all = sum(res)  # this is the numerical value of the full jet
    if return_coefficients:
        return sum_all, res, [sum(res[1:j + 1], res[0]) for j in range(k + 1)]
    return sum_all