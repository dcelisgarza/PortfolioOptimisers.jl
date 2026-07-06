"""
$(DocStringExtensions.README)
"""
module PortfolioOptimisers

using Accessors: Accessors
using ArgCheck: @argcheck
using Base.ScopedValues: ScopedValue
using AverageShiftedHistograms: AverageShiftedHistograms
using Clustering: Clustering, assignments
using Combinatorics: Combinatorics
using ConcreteStructs: @concrete
using DataFrames: DataFrames
using DataStructures: DataStructures
using Dates: Dates
using Distances: Distances
using Distributions: Distributions
using DocStringExtensions: DocStringExtensions
using FLoops: FLoops
using GLM: GLM
using Graphs: Graphs
using Impute: Impute
using InteractiveUtils: InteractiveUtils
using JuMP: JuMP
using LinearAlgebra: LinearAlgebra
using LogExpFunctions: LogExpFunctions
using MultivariateStats: MultivariateStats
using NearestCorrelationMatrix: NearestCorrelationMatrix
using Optim: Optim
using PrecompileTools: PrecompileTools
using Preferences: Preferences
using Random: Random
using Roots: Roots
using SimpleWeightedGraphs: SimpleWeightedGraphs
using SparseArrays: SparseArrays
using SpecialFunctions: SpecialFunctions
using Statistics: Statistics, mean, std, var, cor, cov
using StatsAPI: StatsAPI, predict
using StatsBase: StatsBase
using StringDistances: StringDistances
using TimeSeries: TimeSeries

#=
# Turn readme into PortfolioOptimisers' docs.
@doc let
    path = joinpath(dirname(@__DIR__), "docs/src/index.md")
    include_dependency(path)
    read(path, String)
end PortfolioOptimisers
=#

#=
# Programmatically include source files
src_files = String[]
sizehint!(src_files, 149)
for (root, dirs, files) in walkdir(@__DIR__)
    for file in files
        if file == "PortfolioOptimisers.jl"
            continue
        end
        push!(src_files, joinpath(root, file))
    end
end
sort!(src_files)
include.(src_files)
=#

include("01_Base.jl")
include("02_Tools.jl")
include("03_Preprocessing.jl")
include("04_PosdefMatrix.jl")
include("05_Denoise.jl")
include("06_Detone.jl")
include("07_MatrixProcessing.jl")
include("08_Moments/01_Base_Moments.jl")
include("08_Moments/02_SimpleExpectedReturns.jl")
include("08_Moments/03_Covariance.jl")
include("08_Moments/04_SimpleVariance.jl")
include("08_Moments/05_GerberCovariance.jl")
include("08_Moments/06_SmythBrobyCovariance.jl")
include("08_Moments/07_DistanceCovariance.jl")
include("08_Moments/08_LowerTailDependenceCovariance.jl")
include("08_Moments/09_RankCovariance.jl")
include("08_Moments/10_Histogram.jl")
include("08_Moments/11_MutualInfoCovariance.jl")
include("08_Moments/12_DenoiseCovariance.jl")
include("08_Moments/13_DetoneCovariance.jl")
include("08_Moments/14_ProcessedCovariance.jl")
include("08_Moments/15_PortfolioOptimisersCovariance.jl")
include("08_Moments/16_ShrunkExpectedReturns.jl")
include("08_Moments/17_EquilibriumExpectedReturns.jl")
include("08_Moments/18_ExcessExpectedReturns.jl")
include("08_Moments/19_Coskewness.jl")
include("08_Moments/20_Cokurtosis.jl")
include("08_Moments/21_Base_Regression.jl")
include("08_Moments/22_StepwiseRegression.jl")
include("08_Moments/23_DimensionReductionRegression.jl")
include("08_Moments/24_ImpliedVolatility.jl")
include("08_Moments/25_CorrelationCovariance.jl")
include("08_Moments/26_AbstractCovarianceVariance.jl")
include("08_Moments/27_StandardDeviationExpectedReturns.jl")
include("08_Moments/28_WindowedExpectedReturns.jl")
include("08_Moments/29_WindowedCovariance.jl")
include("08_Moments/30_WindowedVariance.jl")
include("08_Moments/31_WindowedCoskewness.jl")
include("08_Moments/32_WindowedCokurtosis.jl")
include("08_Moments/33_MedianExpectedReturns.jl")
include("08_Moments/34_CustomValueExpectedReturns.jl")
include("08_Moments/35_GerberIQCovariance.jl")
include("08_Moments/36_RegimeAdjustedExpWeightedVariance.jl")
include("08_Moments/37_RegimeAdjustedExpWeightedCovariance.jl")
include("09_Distance/01_Base_Distance.jl")
include("09_Distance/02_Distance.jl")
include("09_Distance/03_DistanceDistance.jl")
include("10_JuMPModelOptimisation.jl")
include("11_Phylogeny/01_Base_Phylogeny.jl")
include("11_Phylogeny/02_Clusters.jl")
include("11_Phylogeny/03_Hierarchical.jl")
include("11_Phylogeny/04_DBHT.jl")
include("11_Phylogeny/05_NonHierarchicalClustering.jl")
include("11_Phylogeny/06_Phylogeny.jl")
include("12_ConstraintGeneration/01_Base_ConstraintGeneration.jl")
include("12_ConstraintGeneration/02_LinearConstraintGeneration.jl")
include("12_ConstraintGeneration/03_RiskBudgetConstraintGeneration.jl")
include("12_ConstraintGeneration/04_PhylogenyConstraintGeneration.jl")
include("12_ConstraintGeneration/05_WeightBoundsConstraintGeneration.jl")
include("12_ConstraintGeneration/06_AssetSetsMatrix.jl")
include("12_ConstraintGeneration/07_ThresholdConstraintGeneration.jl")
include("13_Prior/01_Base_Prior.jl")
include("13_Prior/02_EmpiricalPrior.jl")
include("13_Prior/03_FactorPrior.jl")
include("13_Prior/04_HighOrderPrior.jl")
include("13_Prior/05_BlackLittermanViewsGeneration.jl")
include("13_Prior/06_BlackLittermanPrior.jl")
include("13_Prior/07_BayesianBlackLittermanPrior.jl")
include("13_Prior/08_FactorBlackLittermanPrior.jl")
include("13_Prior/09_AugmentedBlackLittermanPrior.jl")
include("13_Prior/10_EntropyPoolingPrior.jl")
include("13_Prior/11_OpinionPoolingPrior.jl")
include("13_Prior/12_HighOrderFactorPriorEstimator.jl")
include("14_UncertaintySets/01_Base_UncertaintySets.jl")
include("14_UncertaintySets/02_DeltaUncertaintySets.jl")
include("14_UncertaintySets/03_NormalUncertaintySets.jl")
include("14_UncertaintySets/04_BootstrapUncertaintySets.jl")
include("15_Turnover.jl")
include("16_Fees.jl")
include("17_NetReturnsDrawdowns.jl")
include("18_Tracking.jl")
include("19_RiskMeasures/01_Base_RiskMeasures.jl")
include("19_RiskMeasures/02_Variance.jl")
include("19_RiskMeasures/03_MomentRiskMeasures.jl")
include("19_RiskMeasures/04_Kurtosis.jl")
include("19_RiskMeasures/05_NegativeSkewness.jl")
include("19_RiskMeasures/06_XatRisk.jl")
include("19_RiskMeasures/07_ConditionalXatRisk.jl")
include("19_RiskMeasures/08_EntropicXatRisk.jl")
include("19_RiskMeasures/09_RelativisticXatRisk.jl")
include("19_RiskMeasures/10_OWARiskMeasures.jl")
include("19_RiskMeasures/11_AverageDrawdown.jl")
include("19_RiskMeasures/12_UlcerIndex.jl")
include("19_RiskMeasures/13_MaximumDrawdown.jl")
include("19_RiskMeasures/14_BrownianDistanceVariance.jl")
include("19_RiskMeasures/15_WorstRealisation.jl")
include("19_RiskMeasures/16_Range.jl")
include("19_RiskMeasures/17_TurnoverRiskMeasure.jl")
include("19_RiskMeasures/18_TrackingRiskMeasure.jl")
include("19_RiskMeasures/19_PowerNormXatRisk.jl")
include("19_RiskMeasures/20_VarianceSkewKurtosis.jl")
include("19_RiskMeasures/21_GenericValueatRiskRange.jl")
include("19_RiskMeasures/22_RatioRiskMeasure.jl")
include("19_RiskMeasures/23_EqualRiskMeasure.jl")
include("19_RiskMeasures/24_MedianAbsoluteDeviationRisk.jl")
include("19_RiskMeasures/25_NonOptimisationRiskMeasures.jl")
include("19_RiskMeasures/26_AdjustRiskContributions.jl")
include("19_RiskMeasures/27_ExpectedRisk.jl")
include("19_RiskMeasures/28_RiskMeasureTools.jl")
include("20_Optimisation/01_Base_Optimisation.jl")
include("20_Optimisation/02_CrossValidation/01_Base_CrossValidation.jl")
include("20_Optimisation/02_CrossValidation/02_KFold.jl")
include("20_Optimisation/02_CrossValidation/03_Combinatorial.jl")
include("20_Optimisation/02_CrossValidation/04_WalkForward.jl")
include("20_Optimisation/02_CrossValidation/05_MultipleRandomised.jl")
include("20_Optimisation/02_CrossValidation/06_Validation.jl")
include("20_Optimisation/02_CrossValidation/07_Scoring.jl")
include("20_Optimisation/02_CrossValidation/08_OptimisationCrossValidation.jl")
include("20_Optimisation/02_CrossValidation/09_Base_SearchCrossValidation.jl")
include("20_Optimisation/02_CrossValidation/10_GridSearchCrossValidation.jl")
include("20_Optimisation/02_CrossValidation/11_RandomisedSearchCrossValidation.jl")
include("20_Optimisation/03_NaiveOptimisation.jl")
include("20_Optimisation/04_Base_ClusteringOptimisation.jl")
include("20_Optimisation/05_HierarchicalRiskParity.jl")
include("20_Optimisation/06_SchurComplementHierarchicalRiskParity.jl")
include("20_Optimisation/07_HierarchicalEqualRiskContribution.jl")
include("20_Optimisation/08_Base_JuMPOptimisation.jl")
include("20_Optimisation/09_JuMPConstraints/01_Returns_and_ObjectiveFunctions.jl")
include("20_Optimisation/09_JuMPConstraints/02_BudgetConstraints.jl")
include("20_Optimisation/09_JuMPConstraints/03_WeightConstraints.jl")
include("20_Optimisation/09_JuMPConstraints/04_SDPConstraints.jl")
include("20_Optimisation/09_JuMPConstraints/05_MIPConstraints.jl")
include("20_Optimisation/09_JuMPConstraints/06_TurnoverConstraints.jl")
include("20_Optimisation/09_JuMPConstraints/07_FeesConstraints.jl")
include("20_Optimisation/09_JuMPConstraints/08_TrackingErrorConstraints.jl")
include("20_Optimisation/09_JuMPConstraints/09_EffectiveNumberAssetsConstraints.jl")
include("20_Optimisation/09_JuMPConstraints/10_RegularisationConstraints.jl")
include("20_Optimisation/10_JuMPOptimiser.jl")
include("20_Optimisation/11_MeanRisk.jl")
include("20_Optimisation/12_FactorRiskContribution.jl")
include("20_Optimisation/13_NearOptimalCentering.jl")
include("20_Optimisation/14_RiskBudgeting.jl")
include("20_Optimisation/15_RelaxedRiskBudgeting.jl")
include("20_Optimisation/16_Base_MetaOptimisation.jl")
include("20_Optimisation/17_NestedClustered.jl")
include("20_Optimisation/18_Stacking.jl")
include("20_Optimisation/19_SubsetResampling.jl")
include("20_Optimisation/20_RiskMeasureConstraints/01_BaseRiskConstraints.jl")
include("20_Optimisation/20_RiskMeasureConstraints/02_VarianceConstraints.jl")
include("20_Optimisation/20_RiskMeasureConstraints/03_MomentRiskMeasureConstraints.jl")
include("20_Optimisation/20_RiskMeasureConstraints/04_KurtosisConstraints.jl")
include("20_Optimisation/20_RiskMeasureConstraints/05_NegativeSkewnessConstraints.jl")
include("20_Optimisation/20_RiskMeasureConstraints/06_XatRiskConstraints.jl")
include("20_Optimisation/20_RiskMeasureConstraints/07_ConditionalXatRiskConstraints.jl")
include("20_Optimisation/20_RiskMeasureConstraints/08_EntropicXatRiskConstraints.jl")
include("20_Optimisation/20_RiskMeasureConstraints/09_RelativisticXatRiskConstraints.jl")
include("20_Optimisation/20_RiskMeasureConstraints/10_OWARiskMeasuresConstraints.jl")
include("20_Optimisation/20_RiskMeasureConstraints/11_AverageDrawdownConstraints.jl")
include("20_Optimisation/20_RiskMeasureConstraints/12_UlcerIndexConstraints.jl")
include("20_Optimisation/20_RiskMeasureConstraints/13_MaximumDrawdownConstraints.jl")
include("20_Optimisation/20_RiskMeasureConstraints/14_BrownianDistanceVarianceConstraints.jl")
include("20_Optimisation/20_RiskMeasureConstraints/15_WorstRealisationConstraints.jl")
include("20_Optimisation/20_RiskMeasureConstraints/16_RangeConstraints.jl")
include("20_Optimisation/20_RiskMeasureConstraints/17_TurnoverRiskMeasureConstraints.jl")
include("20_Optimisation/20_RiskMeasureConstraints/18_TrackingRiskMeasureConstraints.jl")
include("20_Optimisation/20_RiskMeasureConstraints/19_PowerNormXatRiskConstraints.jl")
include("20_Optimisation/20_RiskMeasureConstraints/20_VarianceSkewKurtosisConstraints.jl")
include("20_Optimisation/20_RiskMeasureConstraints/21_GenericValueatRiskRangeConstraints.jl")
include("20_Optimisation/21_Base_FiniteAllocation.jl")
include("20_Optimisation/22_DiscreteFiniteAllocation.jl")
include("20_Optimisation/23_GreedyFiniteAllocation.jl")
include("20_Optimisation/24_RiskMeasureCompatibility.jl")
include("21_ExpectedReturns.jl")
include("22_Plotting.jl")
include("23_Precompilation.jl")
include("24_Aliases.jl")

end
