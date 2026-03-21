"""
Market data tool — fetches live / historical prices and fundamentals.
"""


def get_price(ticker: str) -> dict:
    raise NotImplementedError


def get_fundamentals(ticker: str) -> dict:
    raise NotImplementedError
