# Type hierarchy

The trees below are generated automatically from the live type hierarchy
every time the documentation is built (see `[docs/generate_type_hierarchy.jl](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/docs/generate_type_hierarchy.jl)`),
so they always reflect the current state of the package. Each type links to
its docstring.

## [AbstractResult](@id type-hierarchy-AbstractResult)

```@raw html
<div class="type-tree">
```

[AbstractResult](@ref)

├──&nbsp;[AbstractConstraintResult](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractParsingResult](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[ParsingResult](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[RhoParsingResult](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractPhylogenyConstraintResult](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[IntegerPhylogeny](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[SemiDefinitePhylogeny](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[LinearConstraint](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[PartialLinearConstraint](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[RiskBudget](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[Threshold](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[WeightBounds](@ref)

├──&nbsp;[AbstractJuMPResult](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[JuMPResult](@ref)

├──&nbsp;[AbstractPhylogenyResult](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractClusteringResult](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[Clusters](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[PhylogenyResult](@ref)

├──&nbsp;[AbstractPredictionResult](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[MultiPeriodPredictionResult](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[PopulationPredictionResult](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[PredictionResult](@ref)

├──&nbsp;[AbstractPriorResult](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[HighOrderPrior](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[LowOrderPrior](@ref)

├──&nbsp;[AbstractRegressionResult](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[Regression](@ref)

├──&nbsp;[AbstractReturnsResult](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[PredictionReturnsResult](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[ReturnsResult](@ref)

├──&nbsp;[AbstractSearchCrossValidationResult](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[SearchCrossValidationResult](@ref)

├──&nbsp;[AbstractTracking](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[RiskTrackingError](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[TrackingError](@ref)

├──&nbsp;[AbstractUncertaintySetResult](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractEllipsoidalUncertaintySetResultClass](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[MuEllipsoidalUncertaintySet](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[SigmaEllipsoidalUncertaintySet](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[BoxUncertaintySet](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[EllipsoidalUncertaintySet](@ref)

├──&nbsp;[BaseJuMPOptimisationResult](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[JuMPOptimisationResult](@ref)

├──&nbsp;[BlackLittermanViews](@ref)

├──&nbsp;[ClusterNode](@ref)

├──&nbsp;[CrossValidationResult](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[NonOptimisationCrossValidationResult](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[NonOptimisationNonSequentialCrossValidationResult](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[NonOptimisationSequentialCrossValidationResult](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[MultipleRandomisedResult](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[OptimisationCrossValidationResult](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[NonSequentialCrossValidationResult](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[CombinatorialCrossValidationResult](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[KFoldResult](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[SequentialCrossValidationResult](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[WalkForwardResult](@ref)

├──&nbsp;[Fees](@ref)

├──&nbsp;[NearOptimalSetup](@ref)

├──&nbsp;[OptimisationModelResult](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[JuMPOptimisationSolution](@ref)

├──&nbsp;[OptimisationResult](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[FiniteAllocationOptimisationResult](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[DiscreteAllocationResult](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[GreedyAllocationResult](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[NonFiniteAllocationOptimisationResult](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[NonJuMPOptimisationResult](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[HierarchicalResult](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[NaiveOptimisationResult](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[NestedClusteredResult](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[SchurComplementHierarchicalRiskParityResult](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[StackingResult](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[SubsetResamplingResult](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[RiskJuMPOptimisationResult](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[FactorRiskContributionResult](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[MeanRiskResult](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[NearOptimalCenteringResult](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[RiskBudgetingResult](@ref)

├──&nbsp;[OptimisationReturnCode](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[OptimisationFailure](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[OptimisationSuccess](@ref)

├──&nbsp;[ProcessedAssetRiskBudgetingAttributes](@ref)

├──&nbsp;[ProcessedFactorRiskBudgetingAttributes](@ref)

├──&nbsp;[ProcessedJuMPOptimiserAttributes](@ref)

├──&nbsp;[RegimeAdjustedVarianceCache](@ref)

├──&nbsp;[Turnover](@ref)

└──&nbsp;[VecScalar](@ref)

```@raw html
</div>
```

## [AbstractEstimator](@id type-hierarchy-AbstractEstimator)

```@raw html
<div class="type-tree">
```

[AbstractEstimator](@ref)

├──&nbsp;[AbstractBaseRiskMeasure](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[NonOptimisationRiskMeasure](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[ExpectedReturn](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[ExpectedReturnRiskRatio](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[MeanReturn](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[MeanReturnRiskRatio](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[NonOptimisationRiskRatioRiskMeasure](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[Skewness](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[ThirdCentralMoment](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[OptimisationRiskMeasure](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[HierarchicalRiskMeasure](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[EqualRiskMeasure](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[HighOrderMoment](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[MedianAbsoluteDeviation](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[RelativeAverageDrawdown](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[RelativeConditionalDrawdownatRisk](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[RelativeDrawdownatRisk](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[RelativeEntropicDrawdownatRisk](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[RelativeMaximumDrawdown](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[RelativePowerNormDrawdownatRisk](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[RelativeRelativisticDrawdownatRisk](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[RelativeUlcerIndex](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[RiskRatioRiskMeasure](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[RiskMeasure](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[AverageDrawdown](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[BrownianDistanceVariance](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[ConditionalDrawdownatRisk](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[ConditionalValueatRisk](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[ConditionalValueatRiskRange](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[DistributionallyRobustConditionalDrawdownatRisk](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[DistributionallyRobustConditionalValueatRisk](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[DistributionallyRobustConditionalValueatRiskRange](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[DrawdownatRisk](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[EntropicDrawdownatRisk](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[EntropicValueatRisk](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[EntropicValueatRiskRange](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[GenericValueatRiskRange](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[Kurtosis](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[LowOrderMoment](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[MaximumDrawdown](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[NegativeSkewness](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[OrderedWeightsArray](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[OrderedWeightsArrayRange](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[PowerNormDrawdownatRisk](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[PowerNormValueatRisk](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[PowerNormValueatRiskRange](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[Range](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[RelativisticDrawdownatRisk](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[RelativisticValueatRisk](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[RelativisticValueatRiskRange](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[RiskTrackingRiskMeasure](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[StandardDeviation](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[TrackingRiskMeasure](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[TurnoverRiskMeasure](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[UlcerIndex](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[UncertaintySetVariance](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[ValueatRisk](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[ValueatRiskRange](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[Variance](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[VarianceSkewKurtosis](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[WorstRealisation](@ref)

├──&nbsp;[AbstractCentralityEstimator](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[CentralityEstimator](@ref)

├──&nbsp;[AbstractConstraintEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractCentralityConstraint](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[CentralityConstraint](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractPhylogenyConstraintEstimator](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[IntegerPhylogenyEstimator](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[SemiDefinitePhylogenyEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[AssetSetsMatrixEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[JuMPConstraintEstimator](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[BudgetConstraintEstimator](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[BudgetCostEstimator](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[BudgetCosts](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[BudgetMarketImpact](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[BudgetEstimator](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[BudgetRange](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[CustomJuMPConstraint](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[CustomJuMPObjective](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[LinearConstraintEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[RiskBudgetEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[ThresholdEstimator](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[WeightBoundsEstimator](@ref)

├──&nbsp;[AbstractCrossValidationScorer](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[PopulationScorer](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[PredictionScorer](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[NearestQuantilePrediction](@ref)

├──&nbsp;[AbstractDenoiseEstimator](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[Denoise](@ref)

├──&nbsp;[AbstractDetoneEstimator](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[Detone](@ref)

├──&nbsp;[AbstractDistanceEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[Distance](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[DistanceDistance](@ref)

├──&nbsp;[AbstractEntropyPoolingOptimiser](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[CVaREntropyPooling](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[JuMPEntropyPooling](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[OptimEntropyPooling](@ref)

├──&nbsp;[AbstractExpectedReturnsEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractShrunkExpectedReturnsEstimator](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[EquilibriumExpectedReturns](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[ExcessExpectedReturns](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[ShrunkExpectedReturns](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[CustomValueExpectedReturns](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[MedianExpectedReturns](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[SimpleExpectedReturns](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[StandardDeviationExpectedReturns](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[VarianceExpectedReturns](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[WindowedExpectedReturns](@ref)

├──&nbsp;[AbstractMatrixProcessingEstimator](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[MatrixProcessing](@ref)

├──&nbsp;[AbstractOptimalNumberClustersEstimator](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[OptimalNumberClusters](@ref)

├──&nbsp;[AbstractOptimisationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[BaseOptimisationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[BaseClusteringOptimisationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[HierarchicalOptimiser](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[BaseJuMPOptimisationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[JuMPOptimiser](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[OptimisationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[FiniteAllocationOptimisationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[DiscreteAllocation](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[GreedyAllocation](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[NonFiniteAllocationOptimisationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[BaseStackingOptimisationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[Stacking](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[BaseSubsetResamplingOptimisationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[SubsetResampling](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[ClusteringOptimisationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[HierarchicalEqualRiskContribution](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[HierarchicalRiskParity](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[NestedClustered](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[SchurComplementHierarchicalRiskParity](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[JuMPOptimisationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[RelaxedRiskBudgeting](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[RiskJuMPOptimisationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[FactorRiskContribution](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[MeanRisk](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[NearOptimalCentering](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[RiskBudgeting](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[NaiveOptimisationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[EqualWeighted](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[InverseVolatility](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[RandomWeighted](@ref)

├──&nbsp;[AbstractOrderedWeightsArrayEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[NormalisedConstantRelativeRiskAversion](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[OWAJuMP](@ref)

├──&nbsp;[AbstractPhylogenyEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractClustersEstimator](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[ClustersEstimator](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[NetworkClustersEstimator](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[AbstractNetworkEstimator](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[NetworkEstimator](@ref)

├──&nbsp;[AbstractPosdefEstimator](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[Posdef](@ref)

├──&nbsp;[AbstractPriorEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractHighOrderPriorEstimator](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractHighOrderPriorEstimator_F](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[HighOrderFactorPriorEstimator](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[HighOrderPriorEstimator](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[AbstractLowOrderPriorEstimator](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractLowOrderPriorEstimator_A](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[EmpiricalPrior](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractLowOrderPriorEstimator_AF](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[BlackLittermanPrior](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[EntropyPoolingPrior](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[OpinionPoolingPrior](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[AbstractLowOrderPriorEstimator_F](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[AugmentedBlackLittermanPrior](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[BayesianBlackLittermanPrior](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[FactorBlackLittermanPrior](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[FactorPrior](@ref)

├──&nbsp;[AbstractRegressionEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[DimensionReductionRegression](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[StepwiseRegression](@ref)

├──&nbsp;[AbstractRegularisationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[LpRegularisation](@ref)

├──&nbsp;[AbstractRiskMeasureSettings](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[HierarchicalRiskMeasureSettings](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[JuMPRiskMeasureSettings](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[MaxRiskMeasureSettings](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[RiskMeasureSettings](@ref)

├──&nbsp;[AbstractSearchCrossValidationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[GridSearchCrossValidation](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[RandomisedSearchCrossValidation](@ref)

├──&nbsp;[AbstractUncertaintySetEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[BootstrapUncertaintySetEstimator](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[ARCHUncertaintySet](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[DeltaUncertaintySet](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[NormalUncertaintySet](@ref)

├──&nbsp;[AssetSets](@ref)

├──&nbsp;[CokurtosisEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[Cokurtosis](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[WindowedCokurtosis](@ref)

├──&nbsp;[CoskewnessEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[Coskewness](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[WindowedCoskewness](@ref)

├──&nbsp;[CrossValidationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[NonOptimisationCrossValidationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[NonOptimisationNonSequentialCrossValidationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[NonOptimisationSequentialCrossValidationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[MultipleRandomised](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[OptimisationCrossValidationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[NonSequentialCrossValidationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[CombinatorialCrossValidation](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[KFold](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[SequentialCrossValidationEstimator](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[WalkForwardEstimator](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[DateWalkForward](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[IndexWalkForward](@ref)

├──&nbsp;[CrossValidationSearchScorer](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[HighestMeanScore](@ref)

├──&nbsp;[DateAdjusterEstimator](@ref)

├──&nbsp;[DynamicAbstractWeights](@ref)

├──&nbsp;[FeesEstimator](@ref)

├──&nbsp;[FrontierBoundEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[LinearBound](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[SquareRootBound](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[SquaredBound](@ref)

├──&nbsp;[GerberIQDecayEstimator](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[ExpGerberIQDecay](@ref)

├──&nbsp;[GerberIQEpsEstimator](@ref)

├──&nbsp;[GerberIQGammaEstimator](@ref)

├──&nbsp;[GerberIQScalerEstimator](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[AssetVolatilityGerberIQScaler](@ref)

├──&nbsp;[JuMPReturnsEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[ArithmeticReturn](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[LogarithmicReturn](@ref)

├──&nbsp;[NumberSubsetsEstimator](@ref)

├──&nbsp;[ObjectiveFunction](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[MaximumRatio](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[MaximumReturn](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[MaximumUtility](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[MinimumRisk](@ref)

├──&nbsp;[OptimisationCrossValidation](@ref)

├──&nbsp;[RegimeAdjustedMethod](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[FirstMomentRegimeAdjusted](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[LogRegimeAdjusted](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[RootMeanSquaredAdjusted](@ref)

├──&nbsp;[Scalariser](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[HierarchicalScalariser](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[MinScalariser](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[NonHierarchicalScalariser](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[LogSumExpScalariser](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[MaxScalariser](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[SumScalariser](@ref)

├──&nbsp;[Solver](@ref)

├──&nbsp;[SubsetSizeEstimator](@ref)

├──&nbsp;[TurnoverEstimator](@ref)

└──&nbsp;[WindowSizeEstimator](@ref)

```@raw html
</div>
```

## [AbstractAlgorithm](@id type-hierarchy-AbstractAlgorithm)

```@raw html
<div class="type-tree">
```

[AbstractAlgorithm](@ref)

├──&nbsp;[ARCHBootstrapSet](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[CircularBootstrap](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[MovingBootstrap](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[StationaryBootstrap](@ref)

├──&nbsp;[AbstractBins](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[AstroPyBins](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[FreedmanDiaconis](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[Knuth](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[Scott](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[HacineGharbiRavier](@ref)

├──&nbsp;[AbstractDenoiseAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[FixedDenoise](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[ShrunkDenoise](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[SpectralDenoise](@ref)

├──&nbsp;[AbstractDistanceAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[CanonicalDistance](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[CorrelationDistance](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[LogDistance](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[SimpleAbsoluteDistance](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[SimpleDistance](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[VariationInfoDistance](@ref)

├──&nbsp;[AbstractEntropyPoolingAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[H0_EntropyPooling](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[H1_EntropyPooling](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[H2_EntropyPooling](@ref)

├──&nbsp;[AbstractEntropyPoolingOptAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[ExpEntropyPooling](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[LogEntropyPooling](@ref)

├──&nbsp;[AbstractEstimatorValueAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[UniformValues](@ref)

├──&nbsp;[AbstractExpectedReturnsAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractShrunkExpectedReturnsAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[BayesStein](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[BodnarOkhrinParolya](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[JamesStein](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[AbstractShrunkExpectedReturnsTarget](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[GrandMean](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[MeanSquaredError](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[VolatilityWeighted](@ref)

├──&nbsp;[AbstractMatrixProcessingAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[InverseMatrixSparsificationAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[LoGo](@ref)

├──&nbsp;[AbstractMomentAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[Full](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[GerberCovarianceAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[Gerber0](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[Gerber1](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[Gerber2](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[GerberIQCovarianceAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[BasicGerberIQ](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[FullGerberIQ](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[PartialGerberIQ](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[Semi](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[SmythBrobyCovarianceAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[SmythBroby0](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[SmythBroby1](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[SmythBroby2](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[SmythBrobyCount0](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[SmythBrobyCount1](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[SmythBrobyCount2](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[SmythBrobyGerber0](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[SmythBrobyGerber1](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[SmythBrobyGerber2](@ref)

├──&nbsp;[AbstractOptimalNumberClustersAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[SecondOrderDifference](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[SilhouetteScore](@ref)

├──&nbsp;[AbstractOrderedWeightsArrayAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[MaximumEntropy](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[SquaredOrderedWeightsArrayAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[MinimumSquaredDistance](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[MinimumSumSquares](@ref)

├──&nbsp;[AbstractPhylogenyAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractCentralityAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[BetweennessCentrality](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[ClosenessCentrality](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[DegreeCentrality](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[EigenvectorCentrality](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[KatzCentrality](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[Pagerank](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[RadialityCentrality](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[StressCentrality](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractClustersAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractHierarchicalClusteringAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[DBHT](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[HClustAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[AbstractNonHierarchicalClusteringAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[KMeansAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[AbstractTreeType](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[BoruvkaTree](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[KruskalTree](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[PrimTree](@ref)

├──&nbsp;[AbstractPreorderBy](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[PreorderTreeByID](@ref)

├──&nbsp;[AbstractRegressionAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractRegressionTarget](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[GeneralisedLinearModel](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[LinearModel](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractStepwiseRegressionAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[Backward](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[Forward](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractStepwiseRegressionCriterion](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractMinMaxValStepwiseRegressionCriterion](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[AbstractMaxValStepwiseRegressionCriteria](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[AdjustedRSquared](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[RSquared](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[AbstractMinValStepwiseRegressionCriterion](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[AIC](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[AICC](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[BIC](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[PValue](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[DimensionReductionTarget](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[PCA](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[PPCA](@ref)

├──&nbsp;[AbstractSearchCrossValidationAlgorithm](@ref)

├──&nbsp;[AbstractSimilarityMatrixAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[ExponentialSimilarity](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[GeneralExponentialSimilarity](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[MaximumDistanceSimilarity](@ref)

├──&nbsp;[AbstractTrackingAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[ReturnsTracking](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[WeightsTracking](@ref)

├──&nbsp;[AbstractUncertaintyKAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[ChiSqKUncertaintyAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[GeneralKUncertaintyAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[NormalKUncertaintyAlgorithm](@ref)

├──&nbsp;[AbstractUncertaintySetAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[BoxUncertaintySetAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[EllipsoidalUncertaintySetAlgorithm](@ref)

├──&nbsp;[BrownianDistanceVarianceFormulation](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[IneqBrownianDistanceVariance](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[NormOneConeBrownianDistanceVariance](@ref)

├──&nbsp;[CrossValidationAlgorithm](@ref)

├──&nbsp;[DBHTRootMethod](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[EqualRoot](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[UniqueRoot](@ref)

├──&nbsp;[EntropyFormulation](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[ExponentialConeEntropy](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[RelativeEntropy](@ref)

├──&nbsp;[Frontier](@ref)

├──&nbsp;[ImpliedVolatilityAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[ImpliedVolatilityPremium](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[ImpliedVolatilityRegression](@ref)

├──&nbsp;[JuMPWeightFinaliserFormulation](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[AbsoluteErrorWeightFinaliser](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[RelativeErrorWeightFinaliser](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[SquaredAbsoluteErrorWeightFinaliser](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[SquaredRelativeErrorWeightFinaliser](@ref)

├──&nbsp;[MedianCenteringFunction](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[MeanCentering](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[MedianCentering](@ref)

├──&nbsp;[MomentMeasureAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[HighOrderMomentMeasureAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[StandardisedHighOrderMoment](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[UnstandardisedHighOrderMomentMeasureAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[FourthMoment](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[ThirdLowerMoment](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[LowOrderMomentMeasureAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[SecondMoment](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[UnstandardisedLowOrderMomentMeasureAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[EvenMoment](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[FirstLowerMoment](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[MeanAbsoluteDeviation](@ref)

├──&nbsp;[OpinionPoolingAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[LinearOpinionPooling](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[LogarithmicOpinionPooling](@ref)

├──&nbsp;[OptimisationAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[NearOptimalCenteringAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[ConstrainedNearOptimalCentering](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[UnconstrainedNearOptimalCentering](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[RelaxedRiskBudgetingAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[BasicRelaxedRiskBudgeting](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[RegularisedPenalisedRelaxedRiskBudgeting](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[RegularisedRelaxedRiskBudgeting](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[RiskBudgetingAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[AssetRiskBudgeting](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[FactorRiskBudgeting](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[RiskBudgetingFormulation](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[LogRiskBudgeting](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[MixedIntegerRiskBudgeting](@ref)

├──&nbsp;[OrderedWeightsArrayFormulation](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[ApproxOrderedWeightsArray](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[ExactOrderedWeightsArray](@ref)

├──&nbsp;[RegimeAdjustedTarget](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[DiagonalTarget](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[MahalanobisTarget](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[PortfolioTarget](@ref)

├──&nbsp;[SchurComplementAlgorithm](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[MonotonicSchurComplement](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[NonMonotonicSchurComplement](@ref)

├──&nbsp;[SchurComplementParams](@ref)

├──&nbsp;[SecondMomentFormulation](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[RSOCRiskExpr](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[SOCRiskExpr](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[VarianceFormulation](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[QuadRiskExpr](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[SquaredSOCRiskExpr](@ref)

├──&nbsp;[TrackingFormulation](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[NormTracking](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[L1Tracking](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[L2Tracking](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[LInfTracking](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;[LpTracking](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[SquaredL2Tracking](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[VariableTracking](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[DependentVariableTracking](@ref)

│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[IndependentVariableTracking](@ref)

├──&nbsp;[ValueatRiskFormulation](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[DistributionValueatRisk](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[MIPValueatRisk](@ref)

├──&nbsp;[VectorToScalarMeasure](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[MaxValue](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[MeanValue](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[MedianValue](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[MinValue](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[ModeValue](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[ProdValue](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[StandardisedValue](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[StdValue](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[SumValue](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[VarValue](@ref)

└──&nbsp;[WeightFinaliser](@ref)

&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;[IterativeWeightFinaliser](@ref)

&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;[JuMPWeightFinaliser](@ref)

```@raw html
</div>
```

## [AbstractCovarianceEstimator](@id type-hierarchy-AbstractCovarianceEstimator)

```@raw html
<div class="type-tree">
```

[AbstractCovarianceEstimator](@ref)

├──&nbsp;[AbstractVarianceEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[SimpleVariance](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[WindowedVariance](@ref)

├──&nbsp;[BaseGerberCovariance](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[BaseGerberIQCovariance](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[GerberIQCovariance](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[BaseSmythBrobyCovariance](@ref)

│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;[SmythBrobyCovariance](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[GerberCovariance](@ref)

├──&nbsp;[CorrelationCovariance](@ref)

├──&nbsp;[Covariance](@ref)

├──&nbsp;[DistanceCovariance](@ref)

├──&nbsp;[GeneralCovariance](@ref)

├──&nbsp;[ImpliedVolatility](@ref)

├──&nbsp;[LowerTailDependenceCovariance](@ref)

├──&nbsp;[MutualInfoCovariance](@ref)

├──&nbsp;[PortfolioOptimisersCovariance](@ref)

├──&nbsp;[RankCovarianceEstimator](@ref)

│&nbsp;&nbsp;&nbsp;├──&nbsp;[KendallCovariance](@ref)

│&nbsp;&nbsp;&nbsp;└──&nbsp;[SpearmanCovariance](@ref)

├──&nbsp;[RegimeAdjustedExpWeightedCovariance](@ref)

├──&nbsp;[RegimeAdjustedExpWeightedVariance](@ref)

└──&nbsp;[WindowedCovariance](@ref)

```@raw html
</div>
```
