import pandas as pd
from pandas.io.formats.style import Styler


def to_percentage(df: pd.DataFrame) -> Styler:
    total = df.sum().sum()
    return df.div(total).mul(100).round(1).style.format("{}%")


def highlight_small_p(styler: Styler) -> Styler:
    return styler.highlight_between("p", left=0, right=0.05)


def format_small_values(styler: Styler) -> Styler:
    """
    Format small values to '< 0.05' or '< 0.01'
    """
    cols = [
        col
        for col, dtype in zip(styler.data.columns, styler.data.dtypes)
        if dtype == "float"
    ]
    for col in cols:
        subset = pd.IndexSlice[styler.data[col].between(0, 0.05, inclusive="left"), col]
        styler = styler.format("< 0.05", subset)
        subset = pd.IndexSlice[styler.data[col].between(0, 0.01, inclusive="left"), col]
        styler = styler.format("< 0.01", subset)
    return styler
