# Introduction

This section explains `PortfolioOptimisers.jl` API in detail. The pages are organised in exactly the same way as the `src` folder itself. This means there should be a 1 to 1 correspondence between documentation and source files[^1].

## Design philosophy

There are three overarching design choices in `PortfolioOptimisers.jl`:

### 1. Well-defined type hierarchies

  - Easily and quickly add new features by sticking to defined interfaces.

### 2. Strongly typed immutable structs

  - All types are concrete and known at instantiation.
  - Constants can be propagated if necessary.
  - There is always a single immutable source of truth for every process.
  - If needed, modifying values must be done via interface functions, which simplifies finding and fixing bugs. If the interface for modification is not provided the code will throw a missing method exception.
  - Future developments may make use of [`Accessors.jl`](https://github.com/JuliaObjects/Accessors.jl) for certain things.

### 3. Compositional design

  - `PortfolioOptimisers.jl` is a toolkit whose components can interact in complex, deeply nested ways.
  - Separation of concerns lets us subdivide logical components into isolated, self-contained units. Leading to easier and fearless development and testing.
  - Extensive and judicious data validation checks are performed at the earliest possible moment---mostly at variable instantiation---to ensure correctness.
  - Turtles all the way down. Structures can be used, reused, and nested in many ways. This allows for efficient data reuse and arbitrary complexity.

## Design goals

This philosophy has three primary goals:

### 1. Maintainability and expandability

  - The only way to break existing functionality should be by modifying APIs.
  - Adding functionality should be a case of subtyping existing abstract types and implementing the correct interfaces.
  - Avoid leaking side effects to other components unless completely necessary. An example of this is entropy pooling requiring the use of a vector of observation weights which must be taken into account in different, largely unrelated places.

### 2. Correctness and robustness

  - Each subunit should perform its own data validation as early as possible unless it absolutely needs downstream data.

### 3. Performance

  - Types and constants are always fully known at inference time.
  - Immutability ensures smaller structs live in the stack.

## Contents

<!-- ```@contents
Pages = ["01_Base.md", "02_Tools.md", "03_Preprocessing.md", "04_PosdefMatrix.md", "05_Denoise.md", "06_Detone.md", "07_MatrixProcessing.md", "08_Moments/01_Base_Moments.md", "08_Moments/02_SimpleExpectedReturns.md", "08_Moments/03_Covariance.md", "08_Moments/04_SimpleVariance.md", "08_Moments/05_GerberCovariances.md", "08_Moments/06_SmythBrobyCovariance.md", "08_Moments/07_DistanceCovariance.md", "08_Moments/08_LowerTailDependenceCovariance.md", "08_Moments/09_RankCovariance.md", "08_Moments/10_Histogram.md", "08_Moments/11_MutualInfoCovariance.md", "08_Moments/12_PortfolioOptimisersCovariance.md", "08_Moments/13_ShrunkExpectedReturns.md", "08_Moments/14_EquilibriumExpectedReturns.md", "08_Moments/15_ExcessExpectedReturns.md", "08_Moments/16_Coskewness.md", "08_Moments/17_Cokurtosis.md", "08_Moments/18_Base_Regression.md", "08_Moments/19_StepwiseRegression.md", "08_Moments/20_DimensionReductionRegression.md", "08_Moments/21_ImpliedVolatility.md", "09_Distance/01_Base_Distance.md", "09_Distance/02_Distance.md", "09_Distance/03_DistanceDistance.md", "10_JuMPModelOptimisation.md", "11_OWA.md", "12_Phylogeny/01_Base_Phylogeny.md", "12_Phylogeny/02_Clustering.md", "12_Phylogeny/03_Hierarchical.md", "12_Phylogeny/04_DBHT.md", "12_Phylogeny/05_Phylogeny.md", "13_ConstraintGeneration/01_Base_ConstraintGeneration.md", "13_ConstraintGeneration/02_LinearConstraintGeneration.md", "13_ConstraintGeneration/03_PhylogenyConstraintGeneration.md", "13_ConstraintGeneration/04_WeightBoundsConstraintGeneration.md", "13_ConstraintGeneration/05_ThresholdConstraintGeneration.md", "14_Prior/01_Base_Prior.md", "14_Prior/02_EmpiricalPrior.md", "14_Prior/03_FactorPrior.md", "14_Prior/04_HighOrderPrior.md", "14_Prior/05_BlackLittermanViewsGeneration.md", "14_Prior/06_BlackLittermanPrior.md", "14_Prior/07_BayesianBlackLittermanPrior.md", "14_Prior/08_FactorBlackLittermanPrior.md", "14_Prior/09_AugmentedBlackLittermanPrior.md", "14_Prior/10_EntropyPoolingPrior.md", "14_Prior/11_OpinionPoolingPrior.md", "15_UncertaintySets/01_Base_UncertaintySets.md", "15_UncertaintySets/02_DeltaUncertaintySets.md", "15_UncertaintySets/03_NormalUncertaintySets.md", "15_UncertaintySets/04_BootstrapUncertaintySets.md", "16_Turnover.md", "17_Fees.md", "18_Tracking.md", "19_RiskMeasures/01_Base_RiskMeasures.md", "19_RiskMeasures/02_Variance.md", "19_RiskMeasures/03_MomentRiskMeasures.md", "19_RiskMeasures/04_Kurtosis.md", "19_RiskMeasures/05_NegativeSkewness.md", "19_RiskMeasures/06_XatRisk.md", "19_RiskMeasures/07_ConditionalXatRisk.md", "19_RiskMeasures/08_EntropicXatRisk.md", "19_RiskMeasures/09_RelativisticXatRisk.md", "19_RiskMeasures/10_OWARiskMeasures.md", "19_RiskMeasures/11_AverageDrawdown.md", "19_RiskMeasures/12_UlcerIndex.md", "19_RiskMeasures/13_MaximumDrawdown.md", "19_RiskMeasures/14_BrownianDistanceVariance.md", "19_RiskMeasures/15_WorstRealisation.md", "19_RiskMeasures/16_Range.md", "19_RiskMeasures/17_TurnoverRiskMeasure.md", "19_RiskMeasures/18_TrackingRiskMeasure.md", "19_RiskMeasures/19_RatioRiskMeasure.md", "19_RiskMeasures/20_EqualRiskMeasure.md", "19_RiskMeasures/21_MedianAbsoluteDeviationRisk.md", "19_RiskMeasures/22_NoOptimisationRiskMeasures.md", "19_RiskMeasures/23_AdjustRiskContributions.md", "19_RiskMeasures/24_ExpectedRisk.md", "19_RiskMeasures/25_RiskMeasureTools.md", "20_Optimisation/01_Base_Optimisation.md", "20_Optimisation/02_NaiveOptimisation.md", "20_Optimisation/03_Base_ClusteringOptimisation.md", "20_Optimisation/04_HierarchicalOptimiser.md", "20_Optimisation/05_HierarchicalRiskParity.md", "20_Optimisation/06_SchurComplementHierarchicalRiskParity.md", "20_Optimisation/07_HierarchicalEqualRiskContribution.md", "20_Optimisation/08_Base_JuMPOptimisation.md", "20_Optimisation/09_JuMPConstraints/01_Returns_and_ObjectiveFunctions.md", "20_Optimisation/09_JuMPConstraints/02_BudgetConstraints.md", "20_Optimisation/09_JuMPConstraints/03_WeightConstraints.md", "20_Optimisation/09_JuMPConstraints/04_SDPConstraints.md", "20_Optimisation/09_JuMPConstraints/05_MIPConstraints.md", "20_Optimisation/09_JuMPConstraints/06_TurnoverConstraints.md", "20_Optimisation/09_JuMPConstraints/07_FeesConstraints.md", "20_Optimisation/09_JuMPConstraints/08_TrackingErrorConstraints.md", "20_Optimisation/09_JuMPConstraints/09_EffectiveNumberAssetsConstraints.md", "20_Optimisation/09_JuMPConstraints/10_RegularisationConstraints.md", "20_Optimisation/10_JuMPOptimiser.md", "20_Optimisation/11_MeanRisk.md", "20_Optimisation/12_FactorRiskContribution.md", "20_Optimisation/13_NearOptimalCentering.md", "20_Optimisation/14_RiskBudgeting.md", "20_Optimisation/15_RelaxedRiskBudgeting.md", "20_Optimisation/16_NestedClustered.md", "20_Optimisation/17_Stacking.md", "20_Optimisation/18_RiskMeasureConstraints/01_BaseRiskConstraints.md", "20_Optimisation/18_RiskMeasureConstraints/02_VarianceConstraints.md", "20_Optimisation/18_RiskMeasureConstraints/03_MomentRiskMeasureConstraints.md", "20_Optimisation/18_RiskMeasureConstraints/04_KurtosisConstraints.md", "20_Optimisation/18_RiskMeasureConstraints/05_NegativeSkewnessConstraints.md", "20_Optimisation/18_RiskMeasureConstraints/06_XatRiskConstraints.md", "20_Optimisation/18_RiskMeasureConstraints/07_ConditionalXatRiskConstraints.md", "20_Optimisation/18_RiskMeasureConstraints/08_EntropicXatRiskConstraints.md", "20_Optimisation/18_RiskMeasureConstraints/09_RelativisticXatRiskConstraints.md", "20_Optimisation/18_RiskMeasureConstraints/10_OWARiskMeasuresConstraints.md", "20_Optimisation/18_RiskMeasureConstraints/11_AverageDrawdownConstraints.md", "20_Optimisation/18_RiskMeasureConstraints/12_UlcerIndexConstraints.md", "20_Optimisation/18_RiskMeasureConstraints/13_MaximumDrawdownConstraints.md", "20_Optimisation/18_RiskMeasureConstraints/14_BrownianDistanceVarianceConstraints.md", "20_Optimisation/18_RiskMeasureConstraints/15_WorstRealisationConstraints.md", "20_Optimisation/18_RiskMeasureConstraints/16_RangeConstraints.md", "20_Optimisation/18_RiskMeasureConstraints/17_TurnoverRiskMeasureConstraints.md", "20_Optimisation/18_RiskMeasureConstraints/18_TrackingRiskMeasureConstraints.md", "20_Optimisation/19_Base_FiniteAllocation.md", "20_Optimisation/20_DiscreteFiniteAllocation.md", "20_Optimisation/21_GreedyFiniteAllocation.md", "21_Expected_Returns.md", "22_Plotting.md", "23_Interfaces.md", "24_Precompilation.md"]
``` -->

[^1]: Except for a few cases, most of which are convenience function overloads. This means some links do not go to the exact method definition. Other than hard-coding links to specific lines of code, which is fragile, I haven't found an easy solution.
