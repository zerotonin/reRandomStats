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

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rerandomstats")
except PackageNotFoundError:
    __version__ = "0.1.0"

# ┌──────────────────────────────────────────────────────────────────────┐
# │                        « Public API »                               │
# └──────────────────────────────────────────────────────────────────────┘

# Case-crossover + model-comparison + dose-response submodules (added in v0.2.0)
from rerandomstats import case_crossover, dose_response, model_comparison
from rerandomstats.binomial_stats import BinomialStats, MultipleBinomialTests
from rerandomstats.case_crossover import (
    build_case_crossover_frame,
    case_crossover_conditional_logit,
    daylight_hours,
    event_anomaly_profile,
    h3_within_event_contrast,
    hsiang_sigma_rescaled,
    stratified_permutation,
    stratify_case_crossover,
)
from rerandomstats.data_io import DataIO
from rerandomstats.dose_response import (
    broken_stick_fit,
    davies_test,
    hill_fit,
    per_subject_segmented,
    pscore_test,
)
from rerandomstats.fisher_exact import FisherExactTest
from rerandomstats.fisher_resampling import FisherResamplingTest
from rerandomstats.hypothesis_tests import HypothesisTests
from rerandomstats.model_comparison import (
    benjamini_hochberg,
    correct_pvalues,
    correct_pvalues_array,
    likelihood_ratio_test,
    wald_two_sample_beta,
)
from rerandomstats.multi_group_test import MultiGroupTest
from rerandomstats.pretty_table import write_pretty_table
from rerandomstats.resample_n_of_k import GetNofK

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
    # case_crossover submodule
    "case_crossover",
    "build_case_crossover_frame",
    "case_crossover_conditional_logit",
    "daylight_hours",
    "event_anomaly_profile",
    "h3_within_event_contrast",
    "hsiang_sigma_rescaled",
    "stratified_permutation",
    "stratify_case_crossover",
    # model_comparison submodule
    "model_comparison",
    "benjamini_hochberg",
    "correct_pvalues",
    "correct_pvalues_array",
    "likelihood_ratio_test",
    "wald_two_sample_beta",
    # dose_response submodule
    "dose_response",
    "broken_stick_fit",
    "davies_test",
    "pscore_test",
    "hill_fit",
    "per_subject_segmented",
]
