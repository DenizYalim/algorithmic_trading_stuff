from abc import ABC, abstractmethod


class News_Classifier(ABC):  # This is pretty useless, i may deprecate this later
    @abstractmethod
    def classify(self, title: str, content: str):
        pass


class REGEX_classifier(News_Classifier):
    def __init__(self):
        super().__init__()

    def classify(self, title: str, content: str):
        # regex ile haber başlığı ve içeriği üzerinden sınıflandırma yapacak
        pass
