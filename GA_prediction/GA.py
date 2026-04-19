from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from GA_prediction.Problem import MarketPredictionProblem
from GA_prediction.Representation import GARepresentation
from GA_prediction.Solution import GASolution


@dataclass
class GeneticAlgorithmConfig:
    population_size: int = 60
    generations: int = 40
    elite_fraction: float = 0.15
    mutation_rate: float = 0.15
    mutation_scale: float = 0.20
    tournament_size: int = 3
    random_seed: Optional[int] = 42


class GeneticAlgorithm:
    def __init__(
        self,
        problem: MarketPredictionProblem,
        config: GeneticAlgorithmConfig | None = None,
    ):
        self.problem = problem
        self.config = config or GeneticAlgorithmConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        self.history: list[dict] = []

    def _initial_population(self) -> list[GARepresentation]:
        return [
            GARepresentation.random(self.problem.feature_count, self.rng)
            for _ in range(self.config.population_size)
        ]

    def _evaluate_population(self, population: list[GARepresentation]) -> list[GASolution]:
        solutions = [self.problem.evaluate(member) for member in population]
        solutions.sort(reverse=True)
        return solutions

    def _select_parent(self, solutions: list[GASolution]) -> GARepresentation:
        tournament_size = min(self.config.tournament_size, len(solutions))
        indexes = self.rng.choice(len(solutions), size=tournament_size, replace=False)
        competitors = [solutions[int(index)] for index in indexes]
        competitors.sort(reverse=True)
        return competitors[0].representation

    def run(self) -> GASolution:
        if self.config.population_size < 4:
            raise ValueError("population_size must be at least 4")
        if self.config.generations < 1:
            raise ValueError("generations must be at least 1")

        population = self._initial_population()
        elite_count = max(1, int(self.config.population_size * self.config.elite_fraction))

        best_solution: GASolution | None = None

        for generation in range(self.config.generations):
            solutions = self._evaluate_population(population)
            generation_best = solutions[0]
            if best_solution is None or generation_best.fitness > best_solution.fitness:
                best_solution = generation_best

            self.history.append(
                {
                    "generation": generation,
                    "best_fitness": generation_best.fitness,
                    "best_metrics": generation_best.metrics,
                }
            )

            next_population = [
                solution.representation.copy()
                for solution in solutions[:elite_count]
            ]

            while len(next_population) < self.config.population_size:
                parent_a = self._select_parent(solutions)
                parent_b = self._select_parent(solutions)
                child = parent_a.crossover(parent_b, self.rng).mutate(
                    self.rng,
                    mutation_rate=self.config.mutation_rate,
                    mutation_scale=self.config.mutation_scale,
                )
                next_population.append(child)

            population = next_population

        if best_solution is None:
            raise RuntimeError("Genetic algorithm did not produce a solution")

        validation_solution = self.problem.evaluate(
            best_solution.representation,
            data=self.problem.validation_data,
        )
        validation_solution.metrics["train_fitness"] = best_solution.fitness
        validation_solution.metrics["train_metrics"] = best_solution.metrics
        return validation_solution
