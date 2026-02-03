from dataclasses import dataclass, field, asdict
from typing import Optional, List, Literal
import ast
import json
from typing import Any, Dict, Optional
import os
import tempfile


@dataclass
class AlpacaPortfolio:
    pass


@dataclass
class TakeProfit:
    p: float  # price level
    pct: float = 100.0  # % of position to sell at this level (0-100)

    def to_dict(self) -> dict:
        return {"p": self.p, "pct": self.pct}


@dataclass
class TradingPlan:
    # Plan (sell-only management)
    stop_loss: Optional[float] = None
    take_profits: List[TakeProfit] = field(
        default_factory=list
    )  # replaces level1/level2
    last_updated_timestamp: str = None


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

    # trading plan
    trading_plan: TradingPlan = None

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

    buy_orders: list[Stock] = field(default_factory=list)
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
    data = asdict(portfolio)
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".data.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def save_string_to_json(data: str, path: str = "data.json") -> None:
    """
    Save a string to a JSON file. If the string is valid JSON, store it as JSON;
    otherwise store it under a "text" key for readability.
    """

    def _parse_scalar(val: str) -> Any:
        if val == "":
            return ""
        lowered = val.lower()
        if lowered in {"none", "null"}:
            return None
        if lowered in {"true", "false"}:
            return lowered == "true"
        # int first, then float
        try:
            if val.lstrip("+-").isdigit():
                return int(val)
        except Exception:
            pass
        try:
            return float(val)
        except Exception:
            pass
        # try JSON for objects/arrays
        if (val.startswith("{") and val.endswith("}")) or (
            val.startswith("[") and val.endswith("]")
        ):
            try:
                return json.loads(val)
            except Exception:
                pass
            try:
                return ast.literal_eval(val)
            except Exception:
                pass
        return val

    def _parse_ai_text_to_json(text: str) -> Optional[Dict[str, Any]]:
        lines = text.splitlines()
        result: Dict[str, Any] = {}
        current: Dict[str, Any] = result
        has_any = False

        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip()
                if not section:
                    continue
                result.setdefault(section, {})
                current = result[section]
                has_any = True
                continue

            if line.startswith("- "):
                line = line[2:]
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                if not key:
                    continue
                current[key] = _parse_scalar(val)
                has_any = True

        return result if has_any else None

    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".data.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            payload = data
            if isinstance(data, str):
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    parsed = _parse_ai_text_to_json(data)
                    payload = parsed if parsed is not None else {"text": data}
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
