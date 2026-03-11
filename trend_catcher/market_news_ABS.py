from dataclasses import dataclass
from typing import List, Optional
from to_add_later.news_classifier_ABS import News_Classifier


@dataclass
class News_Classification:
    topics: List[str]
    about_tickers: List[str]
    relevance: List[float]


@dataclass
class MarketNew:
    title: Optional[str] = None
    content: Optional[str] = None
    source: Optional[str] = None
    date: Optional[str] = None
    classification: Optional[News_Classification] = None

    def set_classification(self, classifier: News_Classifier):
        self.classification = classifier.classify(self)
