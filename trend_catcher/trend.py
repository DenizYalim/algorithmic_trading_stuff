# bizim capitilize etmek için market haberlerinde aradığımız trend. Bulabiliresek ona göre al sat yapacağız


# we should save trends maybe
class Trend:
    def __init__(self, affected_symbols, trend_direction, confidence):
        self.affected_symbols = []
        self.trend_direction = None
        self.confidence = None # confidence in trends existence
