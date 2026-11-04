"""
PEACE - Protomer Enumeration And Charge Engine
A package for molecular protomer and tautomer enumeration.
"""

__version__ = "0.1.0"
__all__ = [
    "protomer",
    "engine",
    "common",
    "main",
    "solvation",
    "pka",
    "api",
    "run",
    "enumerate_protomers",
    "solvate",
    "cli_defaults",
    "PeaceResult",
]

_API_NAMES = {"run", "enumerate_protomers", "solvate", "cli_defaults", "PeaceResult"}


def __getattr__(name):
    # Lazy so that `import peace` stays cheap and peace.main can import __version__.
    if name in _API_NAMES:
        from peace import api

        return getattr(api, name)
    raise AttributeError(f"module 'peace' has no attribute {name!r}")
