"""
┌──────────────────────────────────────────────────────────────────────┐
│         resample_n_of_k.py « Combinatorial Index Sampler »           │
│                                                                      │
│  Generates index tuples for Fisher-style resampling.  Supports       │
│  exhaustive enumeration of all C(N, k) combinations as well as       │
│  random sampling with or without uniqueness constraints.             │
│                                                                      │
│  Author : Bart R.H. Geurten                                         │
│  Licence: MIT                                                        │
└──────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import List, Literal, Sequence, Tuple, Union


class GetNofK:
    """Generate index-pair tuples for resampling two data sets.

    Given two data sets **A** and **B** the class combines them, then
    produces pairs of index tuples ``(indices_A, indices_B)`` where
    ``indices_A`` has the same length as the shorter set and
    ``indices_B`` is its complement.

    Args:
        data_set_a: First data set.
        data_set_b: Second data set.
        combination_n: ``'all'`` for exhaustive enumeration or an
            integer specifying the number of random draws.
        mode: ``'combinations'`` for exhaustive, ``'resampling'``
            for random (non-unique), ``'resample_unique'`` for
            random with uniqueness guarantee.
        max_len_possible_4_perms: If the shorter set exceeds this
            length, exhaustive mode falls back to resampling.
        resampling_n: Default number of random draws when
            *combination_n* triggers resampling mode.
        seed: Seed for this instance's private random generator.
            ``None`` (the default) seeds from OS entropy, so results
            vary between runs exactly as before.  Pass an integer when
            a reported p-value has to be reproducible — resampling
            draws a finite sample of the permutation space, so two
            unseeded runs of the same comparison differ slightly.

    Attributes:
        combined_data: Concatenation of both data sets.
        data_indices: List of ``(indices_A, indices_B)`` tuples
            produced by :meth:`main`.
        combination_n: Actual number of combinations generated.
    """

    def __init__(
        self,
        data_set_a: Sequence,
        data_set_b: Sequence,
        combination_n: Union[int, str],
        mode: Literal["combinations", "resampling", "resample_unique"] = "resampling",
        max_len_possible_4_perms: int = 10,
        resampling_n: int = 100_000,
        seed: int | None = None,
    ) -> None:
        # A private generator rather than the global `random` module, so a
        # seeded test cannot be perturbed by unrelated code drawing from the
        # shared stream, and seeding here cannot disturb the caller's.
        self.seed = seed
        self._rng = random.Random(seed)
        self.data_set_a: List = list(data_set_a)
        self.data_set_b: List = list(data_set_b)
        self.combination_n = combination_n
        self.max_len_possible_4_perms = max_len_possible_4_perms
        self.resampling_n = resampling_n
        self.mode = mode

        self.combined_data: List = self.data_set_a + self.data_set_b
        self.combined_len: int = self._get_combined_len()
        self.short_len: int = self._get_smaller_set_n()

        # ── resolve mode from combination_n ──────────────────────────
        if self.combination_n == "all":
            self.mode = "combinations"
        else:
            self.mode = "resampling"
            self.resampling_n = int(combination_n)

        if self.mode == "combinations" and self.max_len_possible_4_perms < self.short_len:
            self.mode = "resampling"
            self.combination_n = self.resampling_n

        self.combinations: List[Tuple[int, ...]] = []
        self.data_indices: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []

    # ── internal helpers ─────────────────────────────────────────────

    def _get_combined_len(self) -> int:
        """Return the total length of both data sets."""
        self.len_a = len(self.data_set_a)
        self.len_b = len(self.data_set_b)
        return self.len_a + self.len_b

    def _get_smaller_set_n(self) -> int:
        """Return the length of the smaller data set."""
        return min(self.len_a, self.len_b)

    # ── combination generators ───────────────────────────────────────

    def get_all_combinations(self) -> List[Tuple[int, ...]]:
        """Return all C(N, k) unique index tuples.

        Returns:
            Exhaustive list of index tuples of length *short_len*
            drawn from ``range(combined_len)``.
        """
        return list(combinations(range(self.combined_len), self.short_len))

    def get_unique_random_combinations(self) -> List[Tuple[int, ...]]:
        """Return *resampling_n* unique random index tuples.

        If the target count cannot be reached after
        ``10 × resampling_n`` attempts the method returns whatever
        unique tuples it managed to collect.

        Returns:
            List of unique random index tuples.
        """
        combis: set[Tuple[int, ...]] = set()
        all_indices = list(range(self.combined_len))
        tries = 0
        desperation = False
        while len(combis) < self.resampling_n and not desperation:
            combis.add(tuple(sorted(self._rng.sample(all_indices, self.short_len))))
            tries += 1
            if tries > self.resampling_n * 10:
                desperation = True
                print(
                    f"GetNofK: could not produce more than {len(combis)} "
                    f"unique combinations in {tries} tries — using those."
                )
        return [tuple(x) for x in combis]

    def get_random_combinations(self) -> List[Tuple[int, ...]]:
        """Return *resampling_n* random index tuples (may contain duplicates).

        Returns:
            List of random index tuples.
        """
        all_indices = list(range(self.combined_len))
        return [
            tuple(sorted(self._rng.sample(all_indices, self.short_len)))
            for _ in range(self.resampling_n)
        ]

    # ── complement construction ──────────────────────────────────────

    def complement_indices(self) -> None:
        """Build ``(indices_A, complement)`` pairs from *combinations*.

        Populates :attr:`data_indices` so that each entry contains
        the sampled indices and their complement within the combined
        data.
        """
        all_idx = set(range(self.combined_len))
        self.data_indices = [
            (idx_a, tuple(all_idx - set(idx_a)))
            for idx_a in self.combinations
        ]

    # ── main entry point ─────────────────────────────────────────────

    def main(self) -> None:
        """Generate combinations and their complements.

        Dispatches to the appropriate combination generator based on
        :attr:`mode`, then builds complement index pairs.
        """
        if self.mode == "combinations":
            self.combinations = self.get_all_combinations()
        elif self.mode == "resampling":
            self.combinations = self.get_random_combinations()
        elif self.mode == "resample_unique":
            self.combinations = self.get_unique_random_combinations()
        else:
            raise ValueError(
                f"GetNofK.main: unknown mode '{self.mode}'"
            )
        self.complement_indices()
        self.combination_n = len(self.combinations)

    # ── data retrieval ───────────────────────────────────────────────

    def get_shuffled_set(self, combi_i: int) -> Tuple[List, List]:
        """Retrieve the *combi_i*-th shuffled split of the data.

        Args:
            combi_i: Index into :attr:`data_indices`.

        Returns:
            A tuple ``(shuffled_A, shuffled_B)`` of data subsets.
        """
        idx_a, idx_b = self.data_indices[combi_i]
        shuffle_a = [self.combined_data[i] for i in idx_a]
        shuffle_b = [self.combined_data[i] for i in idx_b]
        return shuffle_a, shuffle_b
