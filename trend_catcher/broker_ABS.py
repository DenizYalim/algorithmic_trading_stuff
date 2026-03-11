from abc import ABC, abstractmethod
import logging
from txt_file_logging import my_logger


class _owned_assets:  # i aint writing this rn bro ffs
    pass


class TradeInfo:
    def __init__(self, symbol, entry_price, stop_loss=None, take_profit=None, date=None):  # TODO: adeti napcam
        self.symbol = symbol
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.date = date


class Broker(ABC):
    @abstractmethod
    def place_trade(self, trade_info: TradeInfo):
        pass

    def place_trade_requests(self):
        pass


class SimulatedBroker(Broker):
    def place_trade(self, trade_info: TradeInfo):
        logging.info(
            f"Placing trade for {trade_info.symbol} at entry price {trade_info.entry_price} with stop loss {trade_info.stop_loss} and take profit {trade_info.take_profit} on date {trade_info.date}"
        )
        my_log = my_logger("simulation_trader.txt")
        my_log.log(
            f"Placing trade for {trade_info.symbol} at entry price {trade_info.entry_price} with stop loss {trade_info.stop_loss} and take profit {trade_info.take_profit} on date {trade_info.date}"
        )

    def place_trade_requests(self):
        pass
