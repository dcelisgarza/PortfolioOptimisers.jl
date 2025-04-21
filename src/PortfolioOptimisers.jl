module PortfolioOptimisers

using AverageShiftedHistograms, Clustering, Distances, Distributions, GLM, Impute, JuMP,
      LinearAlgebra, MultivariateStats, NearestCorrelationMatrix, Optim, Graphs,
      SimpleWeightedGraphs, PythonCall, Random, ShortStrings, SmartAsserts, SparseArrays,
      Statistics, StatsBase, DataFrames, TimeSeries

# Turn readme into PortfolioOptimisers' docs.
@doc let
    path = joinpath(dirname(@__DIR__), "docs/src/index.md")
    include_dependency(path)
    read(path, String)
end PortfolioOptimisers

include("./1_Base.jl")
include("./2_Tools.jl")
include("./3_PosDefMatrix.jl")
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

include("./8_Distance/1_Base_Distance.jl")
include("./8_Distance/2_Distance.jl")
include("./8_Distance/3_DistanceDistance.jl")
include("./8_Distance/4_GeneralDistance.jl")
include("./8_Distance/5_GeneralDistanceDistance.jl")

include("./9_JuMPModelOptimisation.jl")

include("./10_OWA.jl")

include("./11_Regression/1_Base_Regression.jl")
include("./11_Regression/2_StepwiseRegression.jl")
include("./11_Regression/3_DimensionReductionRegression.jl")

include("./12_Philogeny/1_Base_Philogeny.jl")
include("./12_Philogeny/2_Clustering.jl")
include("./12_Philogeny/3_Hierarchical.jl")
include("./12_Philogeny/4_DBHT.jl")
include("./12_Philogeny/5_Philogeny.jl")

include("./13_ConstraintGeneration/1_Base_ConstraintGeneration.jl")
include("./13_ConstraintGeneration/2_LinearConstraintGeneration.jl")
include("./13_ConstraintGeneration/3_CardinalityConstraintGeneration.jl")
include("./13_ConstraintGeneration/4_WeightBoundsConstraintGeneration.jl")
include("./13_ConstraintGeneration/5_RiskBudgetConstraintGeneration.jl")
include("./13_ConstraintGeneration/6_PhilogenyConstraintGeneration.jl")

include("./14_Prior/1_Base_Prior.jl")
include("./14_Prior/2_EmpiricalPrior.jl")
include("./14_Prior/3_FactorPrior.jl")
include("./14_Prior/4_HighOrderPrior.jl")
include("./14_Prior/5_BlackLittermanViewsGeneration.jl")
include("./14_Prior/6_BlackLittermanPrior.jl")
include("./14_Prior/7_EmpiricalPartialFactorPrior.jl")
include("./14_Prior/8_BayesianBlackLittermanPrior.jl")
include("./14_Prior/9_FactorBlackLittermanPrior.jl")
include("./14_Prior/10_AugmentedBlackLittermanPrior.jl")
include("./14_Prior/11_EntropyPoolingViewsGeneration.jl")
include("./14_Prior/12_EntropyPoolingPrior.jl")

include("./15_UncertaintySets/1_Base_UncertaintySets.jl")
include("./15_UncertaintySets/2_DeltaUncertaintySets.jl")
include("./15_UncertaintySets/3_NormalUncertaintySets.jl")
include("./15_UncertaintySets/4_BootstrapUncertaintySets.jl")
include("./15_UncertaintySets/5_UncertaintySetFactories.jl")

include("./16_Turnover.jl")
include("./17_Fees.jl")
include("./18_Tracking.jl")

include("./19_RiskMeasures/1_Base_RiskMeasures.jl")
include("./19_RiskMeasures/2_Base_RiskMeasureFactories.jl")
include("./19_RiskMeasures/3_Variance.jl")
include("./19_RiskMeasures/4_MomentRiskMeasures.jl")
include("./19_RiskMeasures/5_SquareRootKurtosis.jl")
include("./19_RiskMeasures/6_NegativeSkewness.jl")
include("./19_RiskMeasures/7_XatRisk.jl")
include("./19_RiskMeasures/8_ConditionalXatRisk.jl")
include("./19_RiskMeasures/9_EntropicXatRisk.jl")
include("./19_RiskMeasures/10_RelativisticXatRisk.jl")
include("./19_RiskMeasures/11_OWARiskMeasures.jl")
include("./19_RiskMeasures/12_AverageDrawdown.jl")
include("./19_RiskMeasures/13_UlcerIndex.jl")
include("./19_RiskMeasures/14_MaximumDrawdown.jl")
include("./19_RiskMeasures/15_BrownianDistanceVariance.jl")
include("./19_RiskMeasures/16_WorstRealisation.jl")
include("./19_RiskMeasures/17_Range.jl")
include("./19_RiskMeasures/18_EqualRiskMeasure.jl")
include("./19_RiskMeasures/19_TurnoverRiskMeasure.jl")
include("./19_RiskMeasures/20_TrackingRiskMeasure.jl")
include("./19_RiskMeasures/21_NoOptimisationRiskMeasures.jl")
include("./19_RiskMeasures/22_AdjustRiskContributions.jl")
include("./19_RiskMeasures/23_ExpectedRisk.jl")

include("./20_Optimisation/1_Base_Optimisation.jl")
include("./20_Optimisation/2_Base_ClusteringOptimisation.jl")
include("./20_Optimisation/3_HierarchicalOptimiser.jl")
include("./20_Optimisation/4_HierarchicalRiskParity.jl")
include("./20_Optimisation/5_HierarchicalEqualRiskContribution.jl")
include("./20_Optimisation/6_Base_JuMPOptimisation.jl")
include("./20_Optimisation/7_Returns_and_ObjectiveFunctions.jl")
include("./20_Optimisation/8_JuMPConstraints.jl")

end
