"""
This type stub file was generated by pyright.
"""

"""
Table plotter
"""
from __future__ import annotations

__all__ = ["table_plot"]

def table_plot(data, form=..., label=..., title=...):  # -> None:
    """
    Plot periodic table data using element symbol vs. value.

    :Parameters:
        *data* : { Element: float }
            Data values to plot

        *form* = "line" : "line|grid"
            Table layout to use

    :Returns: None
    """
