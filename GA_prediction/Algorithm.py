from abc import ABC, abstractmethod
from typing import Generic, TypeVar

R = TypeVar("R")

class Solution(ABC, Generic[R]):
    @abstractmethod
    def solve(self) -> R:
        raise NotImplementedError

class Algorithm(Solution[R], ABC):
    def solve(self) -> R:
        return self.run()

    @abstractmethod
    def run(self) -> R:
        raise NotImplementedError
    