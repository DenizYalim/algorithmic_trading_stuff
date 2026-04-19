from abc import ABC, abstractmethod
import logging
from utility.txt_file_logging import my_logger
from dataclasses import dataclass


@dataclass
class _owned_assets:  # i aint writing this rn bro ffs
    pass


class TradeInfo:
    def __init__(self, symbol, entry_price, action="buy", quantity=1, stop_loss=None, take_profit=None, date=None):  # TODO: adeti napcam
        self.symbol = symbol
        self.entry_price = entry_price
        self.option = action
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.date = date


class Broker(ABC):
    @abstractmethod
    def place_trade(self, trade_info: TradeInfo):
        pass

    def place_trade_requests(self, trade_requests: list[TradeInfo]):
        pass


class SimulatedBroker(Broker):
    def __init__(self, initial_cash=10000):
        self.cash = initial_cash
        self.positions = {}  # symbol: quantity
        self.trade_history = []

    def place_trade(self, trade_info: TradeInfo):
        quantity = int(getattr(trade_info, "quantity", 1) or 1)
        price = getattr(trade_info, "entry_price", None)
        if price is None:
            price = getattr(trade_info, "price", None)
        option = getattr(trade_info, "option", getattr(trade_info, "action", None))

        if option == "hold":
            logging.info(f"Holding {trade_info.symbol}; no trade placed.")
            return None

        if price is None:
            logging.info(f"Missing trade price for {trade_info.symbol}")
            return None

        if option == "buy":
            if price > 0 and self.cash >= price * quantity:
                self.positions[trade_info.symbol] = self.positions.get(trade_info.symbol, 0) + quantity
                self.cash -= quantity * price
                execution = {"symbol": trade_info.symbol, "quantity": quantity, "price": price, "date": getattr(trade_info, "date", None), "action": "buy"}
                self.trade_history.append(execution)
                logging.info(f"Bought {quantity} shares of {trade_info.symbol} at {price} on {getattr(trade_info, 'date', None)}")
                return execution
            else:
                logging.info(f"Insufficient cash or invalid price for {trade_info.symbol}")
                return None
        elif option == "sell":
            # Short selling: sell without owning, increase cash, negative position
            self.positions[trade_info.symbol] = self.positions.get(trade_info.symbol, 0) - quantity
            self.cash += quantity * price
            execution = {"symbol": trade_info.symbol, "quantity": -quantity, "price": price, "date": getattr(trade_info, "date", None), "action": "sell"}
            self.trade_history.append(execution)
            logging.info(f"Shorted {quantity} shares of {trade_info.symbol} at {price} on {getattr(trade_info, 'date', None)}")
            return execution
        else:
            logging.info(f"Unknown action {option} for {trade_info.symbol}")
            return None

    def get_portfolio_value(self, current_prices):
        value = self.cash
        for symbol, qty in self.positions.items():
            if symbol in current_prices:
                value += qty * current_prices[symbol]
        return value

    def place_trade_requests(self, trade_requests: list[TradeInfo]):
        executions = []
        for trade_request in trade_requests:
            execution = self.place_trade(trade_request)
            if execution is not None:
                executions.append(execution)
        return executions
