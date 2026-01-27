from dataclasses import dataclass, field, asdict
from typing import Optional, List, Literal
import json


@dataclass
class TakeProfit:
    p: float  # price level
    pct: float = 100.0  # % of position to sell at this level (0-100)

    def to_dict(self) -> dict:
        return {"p": self.p, "pct": self.pct}


@dataclass
class Stock:
    symbol: str

    # Portfolio context
    quantity: float = 0.0
    value: Optional[float] = (
        None  # market value in portfolio currency (or asset currency if you prefer)
    )

    # Price snapshot (optional)
    current_price: Optional[float] = None
    current_price_timestamp: Optional[str] = None  # ISO timestamp

    # Entry
    avg_buy_in_price: Optional[float] = None
    buy_in_timestamp: Optional[str] = None  # ISO timestamp

    # Plan (sell-only management)
    stop_loss: Optional[float] = None
    take_profits: List[TakeProfit] = field(
        default_factory=list
    )  # replaces level1/level2

    # Minimal metadata (tiny tokens, big benefit)
    currency: Optional[str] = None  # e.g. "SEK", "USD"
    isin: Optional[str] = None  # optional, helpful in Sweden/EU
    state: Literal["watch", "owned", "sold"] = "owned"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "value": self.value,
            "current_price": self.current_price,
            "current_price_timestamp": self.current_price_timestamp,
            "avg_buy_in_price": self.avg_buy_in_price,
            "buy_in_timestamp": self.buy_in_timestamp,
            "stop_loss": self.stop_loss,
            "take_profits": [tp.to_dict() for tp in self.take_profits],
            "currency": self.currency,
            "isin": self.isin,
            "state": self.state,
        }


@dataclass
class Fund:
    symbol: str

    quantity: float = 0.0
    value: Optional[float] = None

    buy_in_price: Optional[float] = None
    buy_in_timestamp: Optional[str] = None

    # Small but useful
    currency: Optional[str] = None
    isin: Optional[str] = None
    kind: Optional[Literal["mutual", "etf"]] = (
        None  # optional; helps AI treat fund vs ETF correctly
    )
    state: Literal["owned", "sold"] = "owned"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "value": self.value,
            "buy_in_price": self.buy_in_price,
            "buy_in_timestamp": self.buy_in_timestamp,
            "currency": self.currency,
            "isin": self.isin,
            "kind": self.kind,
            "state": self.state,
        }


@dataclass
class Portfolio:
    value: Optional[float] = None  # total NAV
    cash: Optional[float] = None  # available cash
    currency: str = "SEK"

    stocks: List[Stock] = field(default_factory=list)
    funds: List[Fund] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "cash": self.cash,
            "currency": self.currency,
            "timestamp": self.timestamp,
            "stocks": [s.to_dict() for s in self.stocks],
            "funds": [f.to_dict() for f in self.funds],
        }


def save_portfolio_to_json(portfolio: Portfolio, path: str = "data.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(portfolio), f, indent=2)
