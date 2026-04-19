from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GARepresentation:
    """
    A compact trading rule evolved by the genetic algorithm.

    The rule scores a normalized feature row with a linear model:
        score = dot(features, weights)

    If score is above threshold it predicts up, below -threshold it predicts down,
    otherwise it stays flat.
    """

    weights: np.ndarray
    threshold: float

    @classmethod
    def random(cls, feature_count: int, rng: np.random.Generator) -> "GARepresentation":
        weights = rng.normal(loc=0.0, scale=1.0, size=feature_count)
        threshold = float(rng.uniform(0.0, 1.0))
        return cls(weights=weights, threshold=threshold)

    def copy(self) -> "GARepresentation":
        return GARepresentation(weights=self.weights.copy(), threshold=float(self.threshold))

    def score(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=float) @ self.weights

    def predict(self, features: np.ndarray) -> np.ndarray:
        scores = self.score(features)
        predictions = np.zeros_like(scores, dtype=int)
        predictions[scores > self.threshold] = 1
        predictions[scores < -self.threshold] = -1
        return predictions

    def crossover(
        self,
        other: "GARepresentation",
        rng: np.random.Generator,
    ) -> "GARepresentation":
        mask = rng.random(self.weights.shape[0]) < 0.5
        child_weights = np.where(mask, self.weights, other.weights)
        child_threshold = float((self.threshold + other.threshold) / 2.0)
        return GARepresentation(weights=child_weights, threshold=child_threshold)

    def mutate(
        self,
        rng: np.random.Generator,
        mutation_rate: float,
        mutation_scale: float,
    ) -> "GARepresentation":
        child = self.copy()
        weight_mask = rng.random(child.weights.shape[0]) < mutation_rate
        child.weights[weight_mask] += rng.normal(
            loc=0.0,
            scale=mutation_scale,
            size=int(weight_mask.sum()),
        )

        if rng.random() < mutation_rate:
            child.threshold = max(
                0.0,
                float(child.threshold + rng.normal(loc=0.0, scale=mutation_scale)),
            )

        return child
