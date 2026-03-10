from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, Sequence, Tuple, TypeVar
import random
import time

from Algorithm import Algorithm  # assumed to exist in your project

G = TypeVar("G")  # genome type (e.g., List[int], List[float], custom class)


@dataclass(slots=True)
class Individual(Generic[G]):
    genome: G
    fitness: Optional[float] = None


class GeneticAlgorithm(Algorithm, Generic[G]):
    """
    Fitness is assumed to be MAXIMIZED by default.
    Provide your own:
      - create_genome(): -> G
      - fitness_fn(genome): -> float
      - crossover_fn(a, b): -> (child_a, child_b)
      - mutate_fn(genome, rng): -> genome
    """

    def __init__(
        self,
        *,
        pop_size: int,
        create_genome: Callable[[random.Random], G],
        fitness_fn: Callable[[G], float],
        crossover_fn: Callable[[G, G, random.Random], Tuple[G, G]],
        mutate_fn: Callable[[G, random.Random], G],
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.9,
        elitism: int = 1,
        tournament_k: int = 3,
        maximize: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if pop_size <= 0:
            raise ValueError("pop_size must be > 0")
        if elitism < 0 or elitism > pop_size:
            raise ValueError("elitism must be in [0, pop_size]")
        if tournament_k < 2:
            raise ValueError("tournament_k must be >= 2")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0, 1]")
        if not (0.0 <= crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0, 1]")

        self.rng = random.Random(seed)

        self.pop_size = pop_size
        self.create_genome = create_genome
        self.fitness_fn = fitness_fn
        self.crossover_fn = crossover_fn
        self.mutate_fn = mutate_fn

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.tournament_k = tournament_k
        self.maximize = maximize

        self.population: List[Individual[G]] = self._init_population()

    def run(self, generations: int) -> Individual[G]:
        """Evolve for N generations and return the best individual found."""
        if generations < 0:
            raise ValueError("generations must be >= 0")

        self._evaluate_population(self.population)
        best = self._best_of(self.population)

        for _ in range(generations):
            self.population = self._next_generation(self.population)
            self._evaluate_population(self.population)

            cur_best = self._best_of(self.population)
            if self._better(cur_best.fitness, best.fitness):
                best = cur_best

        return best

    def _init_population(self) -> List[Individual[G]]:
        return [Individual(self.create_genome(self.rng)) for _ in range(self.pop_size)]

    def _evaluate_population(self, pop: List[Individual[G]]) -> None:
        for ind in pop:
            if ind.fitness is None:
                ind.fitness = float(self.fitness_fn(ind.genome))

    def _next_generation(self, pop: List[Individual[G]]) -> List[Individual[G]]:
        elites = self._sorted(pop)[: self.elitism]
        next_pop: List[Individual[G]] = [Individual(e.genome, e.fitness) for e in elites]

        while len(next_pop) < self.pop_size:
            p1 = self._tournament_select(pop)
            p2 = self._tournament_select(pop)

            c1g, c2g = self._maybe_crossover(p1.genome, p2.genome)
            c1g = self._maybe_mutate(c1g)
            c2g = self._maybe_mutate(c2g)

            next_pop.append(Individual(c1g))
            if len(next_pop) < self.pop_size:
                next_pop.append(Individual(c2g))

        return next_pop

    def _tournament_select(self, pop: Sequence[Individual[G]]) -> Individual[G]:
        contenders = self.rng.sample(list(pop), k=min(self.tournament_k, len(pop)))
        return self._best_of(contenders)

    def _maybe_crossover(self, a: G, b: G) -> Tuple[G, G]:
        if self.rng.random() < self.crossover_rate:
            return self.crossover_fn(a, b, self.rng)
        return a, b

    def _maybe_mutate(self, genome: G) -> G:
        if self.rng.random() < self.mutation_rate:
            return self.mutate_fn(genome, self.rng)
        return genome

    def _better(self, fa: Optional[float], fb: Optional[float]) -> bool:
        if fa is None:
            return False
        if fb is None:
            return True
        return fa > fb if self.maximize else fa < fb

    def _best_of(self, pop: Sequence[Individual[G]]) -> Individual[G]:
        # assumes fitness already computed
        best = pop[0]
        for ind in pop[1:]:
            if self._better(ind.fitness, best.fitness):
                best = ind
        return best

    def _sorted(self, pop: Sequence[Individual[G]]) -> List[Individual[G]]:
        # best-first
        return sorted(
            pop,
            key=lambda ind: float("-inf") if ind.fitness is None else ind.fitness,
            reverse=self.maximize,
        )


    def run_for_seconds(self, seconds: float):
        deadline = time.perf_counter() + seconds

        self._evaluate_population(self.population)
        best = self._best_of(self.population)

        while time.perf_counter() < deadline:
            print(time.perf_counter() , " deadline: ",  deadline,": ", time.perf_counter() < deadline)
            self.population = self._next_generation(self.population)

            self._evaluate_population(self.population)
            cur_best = self._best_of(self.population)
            if self._better(cur_best.fitness, best.fitness):
                best = cur_best

        return best
