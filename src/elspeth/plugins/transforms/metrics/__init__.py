"""Metrics and statistical experiment plugins."""

from .aggregation import (
    ScoreAgreementAggregator,
    ScoreAgreementAnalyzer,
    ScoreRecommendationAggregator,
    ScoreRecommendationAnalyzer,
    ScoreStatsAggregator,
    ScoreStatsAnalyzer,
    ScoreVariantRankingAggregator,
    ScoreVariantRankingAnalyzer,
)
from .baseline_comparison import (
    ScoreAssumptionsBaselinePlugin,
    ScoreCliffsDeltaPlugin,
    ScoreDeltaBaselinePlugin,
)
from .distribution import ScoreDistributionAggregator, ScoreDistributionAnalyzer
from .extractors import ScoreExtractorPlugin
from .power_analysis import ScorePowerAggregator, ScorePowerAnalyzer
from .practical import ScorePracticalBaselinePlugin
from .significance import ScoreBayesianBaselinePlugin, ScoreSignificanceBaselinePlugin

__all__ = [
    "ScoreAgreementAggregator",
    "ScoreAgreementAnalyzer",
    "ScoreAssumptionsBaselinePlugin",
    "ScoreBayesianBaselinePlugin",
    "ScoreCliffsDeltaPlugin",
    # Baseline comparison
    "ScoreDeltaBaselinePlugin",
    "ScoreDistributionAggregator",
    # Distribution
    "ScoreDistributionAnalyzer",
    # Extractors
    "ScoreExtractorPlugin",
    "ScorePowerAggregator",
    # Power analysis
    "ScorePowerAnalyzer",
    # Practical
    "ScorePracticalBaselinePlugin",
    "ScoreRecommendationAggregator",
    "ScoreRecommendationAnalyzer",
    # Significance
    "ScoreSignificanceBaselinePlugin",
    # Backward compatibility aliases (deprecated)
    "ScoreStatsAggregator",
    # Aggregation analyzers
    "ScoreStatsAnalyzer",
    "ScoreVariantRankingAggregator",
    "ScoreVariantRankingAnalyzer",
]
