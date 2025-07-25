module PortfolioOptimisers

using AverageShiftedHistograms, Clustering, Distances, Distributions, FLoops, GLM, Impute,
      InteractiveUtils, JuMP, LinearAlgebra, LogExpFunctions, MultivariateStats,
      NearestCorrelationMatrix, Optim, Graphs, SimpleWeightedGraphs, PythonCall, Random,
      Roots, SmartAsserts, SparseArrays, Statistics, StatsBase, DataFrames, TimeSeries

# Turn readme into PortfolioOptimisers' docs.
@doc let
    path = joinpath(dirname(@__DIR__), "docs/src/index.md")
    include_dependency(path)
    read(path, String)
end PortfolioOptimisers

include("./1_Base.jl")
include("./2_Tools.jl")
include("./3_PosdefMatrix.jl")
include("./4_Denoise.jl")
include("./5_Detone.jl")
include("./6_MatrixProcessing.jl")

include("./7_Moments/1_Base_Moments.jl")
include("./7_Moments/2_SimpleExpectedReturns.jl")
include("./7_Moments/3_Covariance.jl")
include("./7_Moments/4_SimpleVariance.jl")
include("./7_Moments/5_GerberCovariances.jl")
include("./7_Moments/6_SmythBrobyCovariance.jl")
include("./7_Moments/7_DistanceCovariance.jl")
include("./7_Moments/8_LTDCovariance.jl")
include("./7_Moments/9_RankCovariance.jl")
include("./7_Moments/10_Histogram.jl")
include("./7_Moments/11_MutualInfoCovariance.jl")
include("./7_Moments/12_PortfolioOptimisersCovariance.jl")
include("./7_Moments/13_ShrunkExpectedReturns.jl")
include("./7_Moments/14_EquilibriumExpectedReturns.jl")
include("./7_Moments/15_ExcessExpectedReturns.jl")
include("./7_Moments/16_Coskewness.jl")
include("./7_Moments/17_Cokurtosis.jl")
include("./7_Moments/18_Base_Regression.jl")
include("./7_Moments/19_StepwiseRegression.jl")
include("./7_Moments/20_DimensionReductionRegression.jl")
include("./7_Moments/21_ImpliedVolatility.jl")

include("./8_Distance/1_Base_Distance.jl")
include("./8_Distance/2_Distance.jl")
include("./8_Distance/3_DistanceDistance.jl")
include("./8_Distance/4_GeneralDistance.jl")
include("./8_Distance/5_GeneralDistanceDistance.jl")

include("./9_JuMPModelOptimisation.jl")

include("./10_OWA.jl")

include("./11_Philogeny/1_Base_Philogeny.jl")
include("./11_Philogeny/2_Clustering.jl")
include("./11_Philogeny/3_Hierarchical.jl")
include("./11_Philogeny/4_DBHT.jl")
include("./11_Philogeny/5_Philogeny.jl")

include("./12_ConstraintGeneration/1_Base_ConstraintGeneration.jl")
include("./12_ConstraintGeneration/2_LinearConstraintGeneration.jl")
include("./12_ConstraintGeneration/_2_LinearConstraintGeneration.jl")
include("./12_ConstraintGeneration/_3_CardinalityConstraintGeneration.jl")
include("./12_ConstraintGeneration/_5_RiskBudgetConstraintGeneration.jl")
include("./12_ConstraintGeneration/3_PhilogenyConstraintGeneration.jl")
include("./12_ConstraintGeneration/4_WeightBoundsConstraintGeneration.jl")

include("./13_Prior/1_Base_Prior.jl")
include("./13_Prior/2_EmpiricalPrior.jl")
include("./13_Prior/3_FactorPrior.jl")
include("./13_Prior/4_HighOrderPrior.jl")
include("./13_Prior/5_BlackLittermanViewsGeneration.jl")
include("./13_Prior/6_BlackLittermanPrior.jl")
include("./13_Prior/7_BayesianBlackLittermanPrior.jl")
include("./13_Prior/8_FactorBlackLittermanPrior.jl")
include("./13_Prior/9_AugmentedBlackLittermanPrior.jl")
include("./13_Prior/10_EntropyPoolingPrior.jl")
include("./13_Prior/11_OpinionPoolingPrior.jl")
include("./13_Prior/_10_EntropyPoolingViewsGeneration.jl")
include("./13_Prior/_11_EntropyPoolingPrior.jl")

include("./14_UncertaintySets/1_Base_UncertaintySets.jl")
include("./14_UncertaintySets/2_DeltaUncertaintySets.jl")
include("./14_UncertaintySets/3_NormalUncertaintySets.jl")
include("./14_UncertaintySets/4_BootstrapUncertaintySets.jl")
include("./14_UncertaintySets/5_UncertaintySetFactories.jl")

include("./15_Turnover.jl")
include("./16_Fees.jl")
include("./17_Tracking.jl")

include("./18_RiskMeasures/1_Base_RiskMeasures.jl")
include("./18_RiskMeasures/2_Base_RiskMeasureFactories.jl")
include("./18_RiskMeasures/3_Variance.jl")
include("./18_RiskMeasures/4_MomentRiskMeasures.jl")
include("./18_RiskMeasures/5_SquareRootKurtosis.jl")
include("./18_RiskMeasures/6_NegativeSkewness.jl")
include("./18_RiskMeasures/7_XatRisk.jl")
include("./18_RiskMeasures/8_ConditionalXatRisk.jl")
include("./18_RiskMeasures/9_EntropicXatRisk.jl")
include("./18_RiskMeasures/10_RelativisticXatRisk.jl")
include("./18_RiskMeasures/11_OWARiskMeasures.jl")
include("./18_RiskMeasures/12_AverageDrawdown.jl")
include("./18_RiskMeasures/13_UlcerIndex.jl")
include("./18_RiskMeasures/14_MaximumDrawdown.jl")
include("./18_RiskMeasures/15_BrownianDistanceVariance.jl")
include("./18_RiskMeasures/16_WorstRealisation.jl")
include("./18_RiskMeasures/17_Range.jl")
include("./18_RiskMeasures/18_EqualRiskMeasure.jl")
include("./18_RiskMeasures/19_TurnoverRiskMeasure.jl")
include("./18_RiskMeasures/20_TrackingRiskMeasure.jl")
include("./18_RiskMeasures/21_RatioRiskMeasure.jl")
include("./18_RiskMeasures/22_NoOptimisationRiskMeasures.jl")
include("./18_RiskMeasures/23_AdjustRiskContributions.jl")
include("./18_RiskMeasures/24_ExpectedRisk.jl")
include("./18_RiskMeasures/25_RiskMeasureTools.jl")

include("./19_Optimisation/1_Base_Optimisation.jl")
include("./19_Optimisation/2_Base_ClusteringOptimisation.jl")
include("./19_Optimisation/3_HierarchicalOptimiser.jl")
include("./19_Optimisation/4_HierarchicalRiskParity.jl")
include("./19_Optimisation/5_HierarchicalEqualRiskContribution.jl")
include("./19_Optimisation/6_Base_JuMPOptimisation.jl")
include("./19_Optimisation/7_Returns_and_ObjectiveFunctions.jl")
include("./19_Optimisation/8_JuMPConstraints.jl")
include("./19_Optimisation/9_JuMPOptimiser.jl")
include("./19_Optimisation/10_MeanRisk.jl")
include("./19_Optimisation/11_FactorRiskContribution.jl")
include("./19_Optimisation/12_NearOptimalCentering.jl")
include("./19_Optimisation/13_RiskBudgetting.jl")
include("./19_Optimisation/14_RelaxedRiskBudgetting.jl")
include("./19_Optimisation/15_NestedClustering.jl")
include("./19_Optimisation/16_Stacking.jl")
include("./19_Optimisation/17_RiskConstraints.jl")
include("./19_Optimisation/18_Base_FiniteAllocation.jl")
include("./19_Optimisation/19_GreedyFiniteAllocation.jl")
include("./19_Optimisation/20_DiscreteFiniteAllocation.jl")
include("./19_Optimisation/21_NaiveOptimisation.jl")
include("./19_Optimisation/22_SchurHierarchicalRiskParity.jl")

include("./20_Expected_Returns.jl")
include("./21_Plotting.jl")

end
