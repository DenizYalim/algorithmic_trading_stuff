from __future__ import annotations

from dataclasses import dataclass, field

from GA_prediction.Representation import GARepresentation


@dataclass(order=True)
class GASolution:
    fitness: float
    representation: GARepresentation = field(compare=False)
    metrics: dict = field(default_factory=dict, compare=False)
