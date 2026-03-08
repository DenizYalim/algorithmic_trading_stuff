from abc import ABC, abstractmethod
from market_news_ABS import News_Classification


class News_Classifier(ABC): # This is pretty useless, i may deprecate this later 
    @abstractmethod
    def classify(self, title:str, content:str) -> News_Classification:
        pass

class REGEX_classifier(News_Classifier):
    def __init__(self):
        super().__init__()
    def classify(self, title:str, content:str) -> News_Classification:
        # regex ile haber başlığı ve içeriği üzerinden sınıflandırma yapacak
        pass
