module PortfolioOptimisers

using AverageShiftedHistograms, Clustering, Distances, Distributions, GLM, JuMP,
      LinearAlgebra, MultivariateStats, NearestCorrelationMatrix, Optim, Graphs,
      SimpleWeightedGraphs, PythonCall, Random, SmartAsserts, SparseArrays, Statistics,
      StatsBase, DataFrames, TimeSeries

# Turn readme into PortfolioOptimisers' docs.
@doc let
    path = joinpath(dirname(@__DIR__), "docs/src/index.md")
    include_dependency(path)
    read(path, String)
end PortfolioOptimisers

# Linear Algebra
include("./LinearAlgebra/Operators.jl")

# Constraints
include("./Constraints/Constraint_AbstractTypes.jl")
include("./Constraints/LinearConstraints.jl")
include("./Constraints/BlackLittermanViews.jl")
include("./Constraints/HierarchicalConstraints.jl")

# Solver
include("./ModelOptimisation/Solver.jl")
include("./ModelOptimisation/OptimiseModel.jl")

# Ordered Weight Arrays
include("./OrderedWeightsArray/OrderedWeightsArray_AbstractTypes.jl")
include("./OrderedWeightsArray/OWA_NCRRA.jl")
include("./OrderedWeightsArray/JuMP_OrderedWeightsArray/JuMP_OWA_AbstractTypes.jl")
include("./OrderedWeightsArray/JuMP_OrderedWeightsArray/OWA_MaximumEntropy.jl")
include("./OrderedWeightsArray/JuMP_OrderedWeightsArray/OWA_MinimumSumSquares.jl")
include("./OrderedWeightsArray/JuMP_OrderedWeightsArray/OWA_MinimumSquareDistance.jl")
include("./OrderedWeightsArray/OWA.jl")

# Fix Non Positive Definite
include("./LinearAlgebra/FixNonPositiveDefiniteMatrices/FNPDM_AbstractTypes.jl")
include("./LinearAlgebra/FixNonPositiveDefiniteMatrices/FNPDM_NoFix.jl")
include("./LinearAlgebra/FixNonPositiveDefiniteMatrices/FNPDM_NearestCorrelationMatrix.jl")

# Detone
include("./LinearAlgebra/Detone/Detone_AbstractTypes.jl")
include("./LinearAlgebra/Detone/NoDetone.jl")
include("./LinearAlgebra/Detone/Detone.jl")

# Denoise
include("./LinearAlgebra/Denoise/Denoise_AbstractTypes.jl")
include("./LinearAlgebra/Denoise/NoDenoise.jl")
include("./LinearAlgebra/Denoise/ShrunkDenoise.jl")
include("./LinearAlgebra/Denoise/SpectralDenoise.jl")
include("./LinearAlgebra/Denoise/FixedDenoise.jl")
include("./LinearAlgebra/Denoise/Denoise.jl")

# Histogram
include("./LinearAlgebra/Histogram/Histogram_AbstractTypes.jl")
include("./LinearAlgebra/Histogram/HistogramAstroPyBins.jl")
include("./LinearAlgebra/Histogram/HistogramHGRBins.jl")
include("./LinearAlgebra/Histogram/HistogramIntegerBins.jl")
include("./LinearAlgebra/Histogram/HistogramInformation.jl")

# Simple Expected Returns
include("./Moments/ExpectedReturns/ExpectedReturns_AbstractTypes.jl")
include("./Moments/ExpectedReturns/SimpleExpectedReturns.jl")

# Pearson Covariance
include("./Moments/Covariance/Covariance_AbstractTypes.jl")
include("./Moments/Covariance/PearsonCovariance/PearsonCovariance_AbstractTypes.jl")
include("./Moments/Covariance/PearsonCovariance/GeneralWeightedCovariance.jl")
include("./Moments/Covariance/PearsonCovariance/FullCovariance.jl")
include("./Moments/Covariance/PearsonCovariance/SemiCovariance.jl")

# Simple Variance
include("./Moments/Covariance/VarianceStd/VarianceStd_AbstractTypes.jl")
include("./Moments/Covariance/VarianceStd/SimpleVarianceStd.jl")

# Rank Covariance
include("./Moments/Covariance/RankCovariance/RankCovariance_AbstractTypes.jl")
include("./Moments/Covariance/RankCovariance/KendallCovariance.jl")
include("./Moments/Covariance/RankCovariance/SpearmanCovariance.jl")

# Distance Covariance
include("./Moments/Covariance/DistanceCovariance.jl")

# LTD Covariance
include("./Moments/Covariance/LTDCovariance.jl")

# Mutual Information Covariance
include("./Moments/Covariance/MutualInfoCovariance.jl")

# Shrunk Expected Returns
include("./Moments/ExpectedReturns/ShrunkExpectedReturns/ShrunkExpectedReturns_Types.jl")
include("./Moments/ExpectedReturns/ShrunkExpectedReturns/JamesSteinExpectedReturns.jl")
include("./Moments/ExpectedReturns/ShrunkExpectedReturns/BayesSteinExpectedReturns.jl")
include("./Moments/ExpectedReturns/ShrunkExpectedReturns/BodnarOkhrinParolyaExpectedReturns.jl")

# Equilibrium Expected Returns
include("./Moments/ExpectedReturns/EquilibriumExpectedReturns.jl")
include("./Moments/ExpectedReturns/ExcessExpectedReturns.jl")

# Gerber Covariance
include("./Moments/Covariance/GerberCovariance/GerberCovariance_AbstractTypes.jl")

## Base Gerber Covariance
include("./Moments/Covariance/GerberCovariance/BaseGerberCovariance/BaseGerberCovariance_AbstractTypes.jl")
include("./Moments/Covariance/GerberCovariance/BaseGerberCovariance/Gerber0Covariance.jl")
include("./Moments/Covariance/GerberCovariance/BaseGerberCovariance/Gerber0NormalisedCovariance.jl")
include("./Moments/Covariance/GerberCovariance/BaseGerberCovariance/Gerber1Covariance.jl")
include("./Moments/Covariance/GerberCovariance/BaseGerberCovariance/Gerber1NormalisedCovariance.jl")
include("./Moments/Covariance/GerberCovariance/BaseGerberCovariance/Gerber2Covariance.jl")
include("./Moments/Covariance/GerberCovariance/BaseGerberCovariance/Gerber2NormalisedCovariance.jl")

## Smyth Broby Covariance
include("./Moments/Covariance/GerberCovariance/SmythBrobyCovariance/SmythBrobyCovariance_AbstractTypes.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyCovariance/SmythBrobyDelta.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyCovariance/SmythBroby0Covariance.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyCovariance/SmythBroby0NormalisedCovariance.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyCovariance/SmythBroby1Covariance.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyCovariance/SmythBroby1NormalisedCovariance.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyCovariance/SmythBroby2Covariance.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyCovariance/SmythBroby2NormalisedCovariance.jl")

## Smyth Broby Gerber Covariance
include("./Moments/Covariance/GerberCovariance/SmythBrobyGerberCovariance/SmythBrobyGerberCovariance_AbstractTypes.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyGerberCovariance/SmythBrobyGerber0Covariance.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyGerberCovariance/SmythBrobyGerber0NormalisedCovariance.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyGerberCovariance/SmythBrobyGerber1Covariance.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyGerberCovariance/SmythBrobyGerber1NormalisedCovariance.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyGerberCovariance/SmythBrobyGerber2Covariance.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyGerberCovariance/SmythBrobyGerber2NormalisedCovariance.jl")

# Distances
include("./Moments/Distance/Distance_AbstractTypes.jl")

## Simple Distance
include("./Moments/Distance/SimpleDistance/SimpleDistance.jl")
include("./Moments/Distance/SimpleDistance/SimpleAbsoluteDistance.jl")
include("./Moments/Distance/SimpleDistance/SimpleDistanceDistance.jl")
include("./Moments/Distance/SimpleDistance/SimpleAbsoluteDistanceDistance.jl")

## General Distance
include("./Moments/Distance/GeneralDistance/GeneralDistance.jl")
include("./Moments/Distance/GeneralDistance/GeneralAbsoluteDistance.jl")
include("./Moments/Distance/GeneralDistance/GeneralDistanceDistance.jl")
include("./Moments/Distance/GeneralDistance/GeneralAbsoluteDistanceDistance.jl")

## Log Distance
include("./Moments/Distance/LogDistance/LogDistance.jl")
include("./Moments/Distance/LogDistance/LogDistanceDistance.jl")
include("./Moments/Distance/GeneralLogDistance/GeneralLogDistance.jl")
include("./Moments/Distance/GeneralLogDistance/GeneralLogDistanceDistance.jl")

## Variation of Information Distance
include("./Moments/Distance/VariationInfoDistance/VariationInfoDistance.jl")
include("./Moments/Distance/VariationInfoDistance/VariationInfoDistanceDistance.jl")
include("./Moments/Distance/GeneralVariationInfoDistance/GeneralVariationInfoDistance.jl")
include("./Moments/Distance/GeneralVariationInfoDistance/GeneralVariationInfoDistanceDistance.jl")

## Correlation Distance
include("./Moments/Distance/CorrelationDistance/CorrelationDistance.jl")
include("./Moments/Distance/CorrelationDistance/CorrelationDistanceDistance.jl")
include("./Moments/Distance/GeneralCorrelationDistance/GeneralCorrelationDistance.jl")
include("./Moments/Distance/GeneralCorrelationDistance/GeneralCorrelationDistanceDistance.jl")

## Canonical Distance
include("./Moments/Distance/CanonicalDistance/CanonicalDistance.jl")
include("./Moments/Distance/CanonicalDistance/CanonicalDistanceDistance.jl")
include("./Moments/Distance/GeneralCanonicalDistance/GeneralCanonicalDistance.jl")
include("./Moments/Distance/GeneralCanonicalDistance/GeneralCanonicalDistanceDistance.jl")

# DBHTs
include("./DBHT/Clustering/DBHT.jl")
include("./DBHT/LoGo/LoGo_AbstractTypes.jl")
include("./DBHT/LoGo/NoLoGo.jl")
include("./DBHT/LoGo/LoGo.jl")

# Matrix Processing
include("./MatrixProcessing/MatrixProcessing_AbstractTypes.jl")
include("./MatrixProcessing/DefaultMatrixProcessing.jl")

# PortfolioOptimisers Covariance Estimator
include("./Moments/Covariance/PortfolioOptimisersCovariance.jl")

# Coskewness
include("./Moments/Coskewness/Coskewness_AbstractTypes.jl")
include("./Moments/Coskewness/FullCoskewness.jl")
include("./Moments/Coskewness/SemiCoskewness.jl")

# Cokurtosis
include("./Moments/Cokurtosis/Cokurtosis_AbstractTypes.jl")
include("./Moments/Cokurtosis/FullCokurtosis.jl")
include("./Moments/Cokurtosis/SemiCokurtosis.jl")

# Regression
include("./Regression/Regression_AbstractTypes.jl")
include("./Regression/StepwiseRegression/StepwiseRegression.jl")
include("./Regression/StepwiseRegression/ForwardRegression.jl")
include("./Regression/StepwiseRegression/BackwardRegression.jl")
include("./Regression/DimensionReductionRegression/DimensionReduction_AbstractTypes.jl")
include("./Regression/DimensionReductionRegression/PCARegression/PCARegression_AbstractTypes.jl")
include("./Regression/DimensionReductionRegression/PCARegression/PCATarget.jl")
include("./Regression/DimensionReductionRegression/PCARegression/PPCATarget.jl")
include("./Regression/DimensionReductionRegression/PCARegression/PCARegression.jl")

# Prior
include("./Prior/Prior_AbstractTypes.jl")
include("./Prior/EmpiricalPrior.jl")
include("./Prior/BlackLittermanPrior.jl")
include("./Prior/FactorPrior.jl")
include("./Prior/BayesianBlackLittermanPrior.jl")
include("./Prior/FactorBlackLittermanPrior.jl")
include("./Prior/AugmentedBlackLittermanPrior.jl")
include("./Prior/HighOrderPrior.jl")

# Uncertainty sets
include("./UncertaintySets/UncertaintySets_AbstractTypes.jl")
include("./UncertaintySets/EllipseUncertaintySetClass.jl")
include("./UncertaintySets/BoxUncertaintySetClass.jl")
include("./UncertaintySets/DeltaUncertaintySet.jl")
include("./UncertaintySets/BootstrapUncertaintySets/BootstrapUncertaintySets_AbstractTypes.jl")
include("./UncertaintySets/BootstrapUncertaintySets/ARCHUncertaintySets/ARCHUncertaintySets.jl")
include("./UncertaintySets/BootstrapUncertaintySets/ARCHUncertaintySets/ARCHEllipseUncertaintySets.jl")
include("./UncertaintySets/BootstrapUncertaintySets/ARCHUncertaintySets/ARCHBoxUncertaintySets.jl")
include("./UncertaintySets/NormalUncertaintySets/NormalUncertaintySets.jl")
include("./UncertaintySets/NormalUncertaintySets/NormalBoxUncertaintySets.jl")
include("./UncertaintySets/NormalUncertaintySets/NormalEllipseUncertaintySets.jl")

# Utils
include("./Utils/BaseFunctionOverloads.jl")
include("./Utils/EquationParsing.jl")
include("./Utils/PricesToReturns.jl")
include("./Utils/Assertions.jl")

# Preselection
include("./PreSelection/PreSelection.jl")

# Risk measure
include("./Measures/Factories/ArrayClusterFactories.jl")
include("./Measures/Factories/RiskMeasureSolverFactories.jl")
include("./Measures/Factories/RiskMeasureFactories.jl")
include("./Measures/Factories/RiskMeasureClusterFactories.jl")
include("./Measures/TrackingTurnover/Turnover.jl")
include("./Measures/Returns/Fees.jl")
include("./Measures/TrackingTurnover/Tracking.jl")
include("./Measures/Risk/Risk_AbstractTypes.jl")
include("./Measures/Risk/MeanReturn.jl")
include("./Measures/Risk/MeanAbsoluteDeviation.jl")
include("./Measures/Risk/StandardDeviation.jl")
include("./Measures/Risk/Variance.jl")
include("./Measures/Risk/SemiVariance.jl")
include("./Measures/Risk/UncertaintyVariance.jl")
include("./Measures/Risk/SemiStandardDeviation.jl")
include("./Measures/Risk/FirstLowerPartialMoment.jl")
include("./Measures/Risk/WorstRealisation.jl")
include("./Measures/Risk/ConditionalValueatRisk.jl")
include("./Measures/Risk/DistributionallyRobustConditionalValueatRisk.jl")
include("./Measures/Risk/EntropicRiskMeasure.jl")
include("./Measures/Risk/EntropicValueatRisk.jl")
include("./Measures/Risk/EntropicValueatRiskRange.jl")
include("./Measures/Risk/RelativisticRiskMeasure.jl")
include("./Measures/Risk/RelativisticValueatRisk.jl")
include("./Measures/Risk/RelativisticValueatRiskRange.jl")
include("./Measures/Risk/MaximumDrawdown.jl")
include("./Measures/Risk/AverageDrawdown.jl")
include("./Measures/Risk/ConditionalDrawdownatRisk.jl")
include("./Measures/Risk/UlcerIndex.jl")
include("./Measures/Risk/EntropicDrawdownatRisk.jl")
include("./Measures/Risk/RelativisticDrawdownatRisk.jl")
include("./Measures/Risk/SquareRootKurtosis.jl")
include("./Measures/Risk/SquareRootSemiKurtosis.jl")
include("./Measures/Risk/Range.jl")
include("./Measures/Risk/ConditionalValueatRiskRange.jl")
include("./Measures/Risk/GiniMeanDifference.jl")
include("./Measures/Risk/TailGini.jl")
include("./Measures/Risk/TailGiniRange.jl")
include("./Measures/Risk/OrderedWeightsArray.jl")
include("./Measures/Risk/BrownianDistanceVariance.jl")
include("./Measures/Risk/NegativeQuadraticSkewness.jl")
include("./Measures/Risk/NegativeQuadraticSemiSkewness.jl")
include("./Measures/Risk/NegativeSkewness.jl")
include("./Measures/Risk/NegativeSemiSkewness.jl")
include("./Measures/Risk/TrackingRiskMeasure.jl")
include("./Measures/Risk/TurnoverRiskMeasure.jl")
include("./Measures/Risk/ValueatRisk.jl")
include("./Measures/Risk/ValueatRiskRange.jl")
include("./Measures/Risk/SquaredRiskMeasures.jl")

end
