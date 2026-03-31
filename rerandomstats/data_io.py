"""
┌──────────────────────────────────────────────────────────────────────┐
│            data_io.py « CSV Data Ingestion Utilities »                │
│                                                                      │
│  Reads wide-format CSV files (including German-locale semicolon-    │
│  delimited variants), converts them to NumPy matrices, and          │
│  reshapes wide tables into long (value, id) format suitable for     │
│  statistical analysis.                                              │
│                                                                      │
│  Author : Bart R.H. Geurten                                         │
│  Licence: MIT                                                        │
└──────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


class DataIO:
    """CSV reader and wide→long table converter.

    Handles both standard comma-separated and German-locale
    semicolon-separated CSV files.

    Args:
        german_csv: If ``True``, treat semicolons as delimiters
            (common in German Excel exports).

    Attributes:
        raw_data: Raw rows as read from the CSV.

    Example:
        >>> dio = DataIO()
        >>> ids, vals = dio.wide_table_csv_to_long_table('data.csv')
    """

    def __init__(self, german_csv: bool = False) -> None:
        self.raw_data: Optional[List] = None
        self.german_csv = german_csv

    # ── CSV reading ──────────────────────────────────────────────────

    def read_csv(self, file_path: Union[str, Path]) -> None:
        """Read a CSV file into :attr:`raw_data`.

        Args:
            file_path: Path to the CSV file.
        """
        with open(file_path, "r", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            if self.german_csv:
                self.raw_data = [
                    row[:].replace(";", ",") for row in reader
                ]
            else:
                self.raw_data = [row for row in reader]

    # ── matrix construction ──────────────────────────────────────────

    def make_square_np_matrix(self, values: List) -> np.ndarray:
        """Build a rectangular NumPy float matrix from row data.

        Empty cells are filled with ``np.nan``.

        Args:
            values: List of rows (each row a list of strings or a
                semicolon-separated string when *german_csv* is set).

        Returns:
            2-D :class:`numpy.ndarray` of shape ``(nrows, ncols)``.
        """
        if self.german_csv:
            nrows = len(values)
            ncols = max(row.count(",") for row in values) + 1
            values = [row.split(",") for row in values]
        else:
            nrows = len(values)
            ncols = len(values[0])

        matrix = np.full((nrows, ncols), np.nan)
        for i, row in enumerate(values):
            for j, val in enumerate(row):
                if val:
                    matrix[i, j] = float(val)
        return matrix

    # ── header extraction ────────────────────────────────────────────

    def split_csv_headers(self) -> List[str]:
        """Return the column headers from the first CSV row.

        Returns:
            List of header strings.
        """
        assert self.raw_data is not None, "Call read_csv() first."
        if self.german_csv:
            return self.raw_data[0].split(",")
        return self.raw_data[0]

    # ── wide → long conversion ───────────────────────────────────────

    def wide_table_to_value_id_list(
        self,
        values: np.ndarray,
        col_header: Sequence[str],
    ) -> Tuple[List[str], List[float]]:
        """Convert a wide matrix to parallel id / value lists.

        Non-NaN cells are unpacked column-wise, tagging each value
        with its column header.

        Args:
            values: 2-D NumPy array.
            col_header: Column names (length must match
                ``values.shape[1]``).

        Returns:
            Tuple ``(id_list, value_list)``.
        """
        id_list: List[str] = []
        value_list: List[float] = []
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if not np.isnan(values[i, j]):
                    value_list.append(values[i, j])
                    id_list.append(col_header[j])
        return id_list, value_list

    def wide_table_csv_to_long_table(
        self, file_path: Union[str, Path]
    ) -> Tuple[List[str], List[float]]:
        """Read a wide-format CSV and return long-format id/value lists.

        Convenience wrapper that chains :meth:`read_csv`,
        :meth:`make_square_np_matrix`, :meth:`split_csv_headers`, and
        :meth:`wide_table_to_value_id_list`.

        Args:
            file_path: Path to the wide-format CSV.

        Returns:
            Tuple ``(id_list, value_list)``.
        """
        self.read_csv(file_path)
        assert self.raw_data is not None
        data = self.make_square_np_matrix(self.raw_data[1:])
        headers = self.split_csv_headers()
        return self.wide_table_to_value_id_list(data, headers)

    # ── subsetting ───────────────────────────────────────────────────

    @staticmethod
    def get_subset_of_data(
        id_list: Sequence[str],
        value_list: Sequence[float],
        id_subset: Sequence[str],
    ) -> Tuple[List[str], List[float]]:
        """Filter parallel id/value lists to a subset of ids.

        Args:
            id_list: Full list of identifiers.
            value_list: Full list of values.
            id_subset: Identifiers to keep.

        Returns:
            Filtered ``(id_list, value_list)`` tuple.

        Example:
            >>> DataIO.get_subset_of_data(
            ...     ['a', 'b', 'c'], [1, 2, 3], ['a', 'c']
            ... )
            (['a', 'c'], [1, 3])
        """
        subset_ids: List[str] = []
        subset_vals: List[float] = []
        for ident, val in zip(id_list, value_list):
            if ident in id_subset:
                subset_ids.append(ident)
                subset_vals.append(val)
        return subset_ids, subset_vals
