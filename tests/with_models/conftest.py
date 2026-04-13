from pytest import fixture
import torch

from jex.models import LM, from_pretrained


@fixture(scope="module")
def gpt2() -> LM:
    return from_pretrained("gpt2")


@fixture(scope="module")
def pythia() -> LM:
    return from_pretrained("EleutherAI/pythia-70m")


@fixture(params=["gpt2", "pythia"])
def lm(request) -> LM:
    return request.getfixturevalue(request.param)


@fixture
def tokens(lm):
    return torch.tensor(lm.tokenizer.encode("Jet expansions?")).unsqueeze(0)
