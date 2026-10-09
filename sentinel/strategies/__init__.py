"""Strategy families. Importing this package registers every built-in family."""
from sentinel.strategies import (  # noqa: F401
    breakout,
    controls,
    crypto_mtf,
    meanrev,
    rotation,
    squeeze,
    trend,
    vwap,
)
from sentinel.strategies.base import REGISTRY, Context, Signal, Strategy, get_family, load_evolved, register  # noqa: F401
