"""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║   ██████╗ ███████╗██████╗  █████╗ ███╗   ██╗██████╗  ██████╗ ███╗  ██╗║
║   ██╔══██╗██╔════╝██╔══██╗██╔══██╗████╗  ██║██╔══██╗██╔═══██╗████╗██║║
║   ██████╔╝█████╗  ██████╔╝███████║██╔██╗ ██║██║  ██║██║   ██║██╔████║║
║   ██╔══██╗██╔══╝  ██╔══██╗██╔══██║██║╚██╗██║██║  ██║██║   ██║██║╚███║║
║   ██║  ██║███████╗██║  ██║██║  ██║██║ ╚████║██████╔╝╚██████╔╝██║ ╚██║║
║   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝  ╚═════╝ ╚═╝ ╚═╝║
║                                                                        ║
║          « Re-randomisation statistics in the spirit of Fisher »       ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝

reRandomStats — A comprehensive re-randomisation statistics toolkit.

Implements Fisher's resampling test, multiple hypothesis testing with
FDR correction, binomial proportion tests, and a suite of parametric
and non-parametric hypothesis tests.

Maintained by Bart R.H. Geurten
Department of Zoology, University of Otago, Dunedin, New Zealand
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("rerandomstats")
except PackageNotFoundError:
    __version__ = "0.1.0"

# ┌──────────────────────────────────────────────────────────────────────┐
# │                        « Public API »                               │
# └──────────────────────────────────────────────────────────────────────┘

from rerandomstats.fisher_exact import FisherExactTest
from rerandomstats.fisher_resampling import FisherResamplingTest
from rerandomstats.hypothesis_tests import HypothesisTests
from rerandomstats.binomial_stats import BinomialStats, MultipleBinomialTests
from rerandomstats.multi_group_test import MultiGroupTest
from rerandomstats.resample_n_of_k import GetNofK
from rerandomstats.data_io import DataIO
from rerandomstats.pretty_table import write_pretty_table

__all__ = [
    "FisherExactTest",
    "FisherResamplingTest",
    "HypothesisTests",
    "BinomialStats",
    "MultipleBinomialTests",
    "MultiGroupTest",
    "GetNofK",
    "DataIO",
    "write_pretty_table",
]
