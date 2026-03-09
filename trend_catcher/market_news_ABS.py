from news_classifier_ABS import News_Classifier


class News_Classification:  # this will be its on its own file later; maybe
    def __init__(self, topics: list, about_tickers: list, relevance: list):
        self.topics = topics
        self.about_tickers = about_tickers
        self.relevance = relevance


class MarketNew:
    def __init__(self):
        self.title = None
        self.content = None
        self.source = None
        self.date = None
        self.classification: News_Classification = None

    def set_classification(self, classifier: News_Classifier):
        self.classification = classifier.classify(self)
        # return self.classification
