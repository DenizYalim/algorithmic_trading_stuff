from dataclasses import dataclass
from typing import List, Optional


@dataclass
class NewsClassification:
    topics: List[str]
    about_tickers: List[str]
    relevance: List[float]


@dataclass
class MarketNews:
    title: Optional[str] = None
    content: Optional[str] = None
    source: Optional[str] = None
    date: Optional[str] = None
    classification: Optional[NewsClassification] = None

    def set_classification(self, classifier):
        self.classification = classifier.classify(self)
