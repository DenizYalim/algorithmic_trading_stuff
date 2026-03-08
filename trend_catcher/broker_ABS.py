from abc import ABC, abstractmethod
import logging
import txt_file_logging

class _owned_assets(): # i aint writing this rn bro ffs
    pass

class trade_info():
    def __init__(self, symbol, entry_price, stop_loss, take_profit, date): # TODO: adeti napcam
        self.symbol = symbol
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.date = date

class Broker(ABC):
    @abstractmethod
    def place_trade(trade_info:trade_info):
        pass

class simulation_trader(Broker):
    def place_trade(trade_info:trade_info):
        logging.info(f"Placing trade for {trade_info.symbol} at entry price {trade_info.entry_price} with stop loss {trade_info.stop_loss} and take profit {trade_info.take_profit} on date {trade_info.date}")
        my_log = txt_file_logging.txt_file_logging("simulation_trader.txt")
        my_log.my_logger("simulation_trader.txt").log(f"Placing trade for {trade_info.symbol} at entry price {trade_info.entry_price} with stop loss {trade_info.stop_loss} and take profit {trade_info.take_profit} on date {trade_info.date}")
