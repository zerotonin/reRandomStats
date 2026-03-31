"""
┌──────────────────────────────────────────────────────────────────────┐
│       pretty_table.py « Console-Friendly Table Formatter »           │
│                                                                      │
│  Renders a pandas DataFrame as a PrettyTable for quick terminal     │
│  inspection or plain-text file export.                              │
│                                                                      │
│  Author : Bart R.H. Geurten                                         │
│  Licence: MIT                                                        │
└──────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd
from prettytable import PrettyTable


def write_pretty_table(
    df: pd.DataFrame,
    file_path: Union[str, Path] = "",
    write: bool = False,
    show: bool = True,
) -> PrettyTable:
    """Render a DataFrame as an ASCII table.

    Args:
        df: The DataFrame to render.
        file_path: Destination file when *write* is ``True``.
        write: If ``True``, write the table to *file_path*.
        show: If ``True``, print the table to stdout.

    Returns:
        The :class:`~prettytable.PrettyTable` object.
    """
    table = PrettyTable()
    table.field_names = df.columns.tolist()

    for _, row in df.iterrows():
        table.add_row(row.tolist())

    if write and file_path:
        Path(file_path).write_text(table.get_string(), encoding="utf-8")

    if show:
        print(table)

    return table
