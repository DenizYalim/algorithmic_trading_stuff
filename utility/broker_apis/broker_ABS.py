from abc import ABC, abstractmethod
import logging
from utility.txt_file_logging import my_logger
from dataclasses import dataclass


@dataclass
class _owned_assets:  # i aint writing this rn bro ffs
    pass


class TradeInfo:
    def __init__(self, symbol, entry_price, action="buy", stop_loss=None, take_profit=None, date=None):  # TODO: adeti napcam
        self.symbol = symbol
        self.entry_price = entry_price
        self.action = action
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
    def __init__(self, initial_cash=10000):
        self.cash = initial_cash
        self.positions = {}  # symbol: quantity
        self.trade_history = []

    def place_trade(self, trade_info: TradeInfo):
        quantity = 1  # Fixed quantity for simplicity
        if trade_info.action == "buy":
            if trade_info.entry_price > 0 and self.cash >= trade_info.entry_price * quantity:
                self.positions[trade_info.symbol] = self.positions.get(trade_info.symbol, 0) + quantity
                self.cash -= quantity * trade_info.entry_price
                self.trade_history.append({"symbol": trade_info.symbol, "quantity": quantity, "price": trade_info.entry_price, "date": trade_info.date, "action": "buy"})
                logging.info(f"Bought {quantity} shares of {trade_info.symbol} at {trade_info.entry_price} on {trade_info.date}")
            else:
                logging.info(f"Insufficient cash or invalid price for {trade_info.symbol}")
        elif trade_info.action == "sell":
            # Short selling: sell without owning, increase cash, negative position
            self.positions[trade_info.symbol] = self.positions.get(trade_info.symbol, 0) - quantity
            self.cash += quantity * trade_info.entry_price
            self.trade_history.append({"symbol": trade_info.symbol, "quantity": -quantity, "price": trade_info.entry_price, "date": trade_info.date, "action": "sell"})
            logging.info(f"Shorted {quantity} shares of {trade_info.symbol} at {trade_info.entry_price} on {trade_info.date}")
        else:
            logging.info(f"Unknown action {trade_info.action} for {trade_info.symbol}")

    def get_portfolio_value(self, current_prices):
        value = self.cash
        for symbol, qty in self.positions.items():
            if symbol in current_prices:
                value += qty * current_prices[symbol]
        return value

    def place_trade_requests(self):
        pass
