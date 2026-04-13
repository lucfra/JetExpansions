from torch import Tensor


from functools import lru_cache
from typing import Callable


class CachedF:
    def __init__(self, variate: Callable[[Tensor], Tensor]) -> None:
        self._variate = variate

    @lru_cache(maxsize=1)
    def __call__(self, z: Tensor) -> Tensor:
        return self._variate(z)
