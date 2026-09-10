# Type hierarchy

The trees below are generated automatically from the live type hierarchy
every time the documentation is built (see [docs/generate_type_hierarchy.jl](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/docs/generate_type_hierarchy.jl)),
so they always reflect the current state of the package. Each type links to
its docstring.

## [AbstractResult](@id type-hierarchy-AbstractResult)

```@raw html
<div class="type-tree">
```

[`AbstractResult`](@ref)\
├── [`AbstractConstraintResult`](@ref)\
│   ├── [`AbstractParsingResult`](@ref)\
│   │   ├── [`ParsingResult`](@ref)\
│   │   └── [`RhoParsingResult`](@ref)\
│   ├── [`AbstractPhylogenyConstraintResult`](@ref)\
│   │   ├── [`IntegerPhylogeny`](@ref)\
│   │   └── [`SemiDefinitePhylogeny`](@ref)\
│   ├── [`LinearConstraint`](@ref)\
│   ├── [`PartialLinearConstraint`](@ref)\
│   ├── [`RiskBudget`](@ref)\
│   ├── [`TargetedConstraint`](@ref)\
│   ├── [`Threshold`](@ref)\
│   └── [`WeightBounds`](@ref)\
├── [`AbstractEntropyPoolingTailView`](@ref)\
│   ├── [`AbstractSequentialTailViewConstraint`](@ref)\
│   │   ├── [`SequentialConditionalValueatRiskViewConstraint`](@ref)\
│   │   ├── [`SequentialEntropicValueatRiskViewConstraint`](@ref)\
│   │   └── [`SequentialRelativisticValueatRiskViewConstraint`](@ref)\
│   ├── [`ConicEntropicValueatRiskViewConstraint`](@ref)\
│   ├── [`ConicRelativisticValueatRiskViewConstraint`](@ref)\
│   ├── [`GridEntropicValueatRiskViewConstraint`](@ref)\
│   ├── [`GridRelativisticValueatRiskViewConstraint`](@ref)\
│   ├── [`IntegerConditionalValueatRiskViewConstraint`](@ref)\
│   └── [`LinearConditionalValueatRiskViewConstraint`](@ref)\
├── [`AbstractFactorFamilyBasis`](@ref)\
│   └── [`FactorFamilyBasis`](@ref)\
├── [`AbstractJuMPResult`](@ref)\
│   └── [`JuMPResult`](@ref)\
├── [`AbstractPanelField`](@ref)\
│   ├── [`CategoricalPanelField`](@ref)\
│   ├── [`NumericPanelField`](@ref)\
│   └── [`TensorPanelField`](@ref)\
├── [`AbstractPartialFitState`](@ref)\
│   ├── [`CokurtosisPartialFitState`](@ref)\
│   ├── [`CoskewnessPartialFitState`](@ref)\
│   ├── [`CovarianceState`](@ref)\
│   ├── [`ExpWeightedCovarianceState`](@ref)\
│   ├── [`ExpWeightedExpectedReturnsState`](@ref)\
│   ├── [`ExpWeightedVarianceState`](@ref)\
│   ├── [`RegimeAdjustedCovarianceState`](@ref)\
│   ├── [`RegimeAdjustedVarianceState`](@ref)\
│   ├── [`SampleBufferState`](@ref)\
│   ├── [`SimpleExpectedReturnsState`](@ref)\
│   └── [`SimpleVarianceState`](@ref)\
├── [`AbstractPhylogenyResult`](@ref)\
│   ├── [`AbstractClusteringResult`](@ref)\
│   │   └── [`Clusters`](@ref)\
│   └── [`PhylogenyResult`](@ref)\
├── [`AbstractPipelineResult`](@ref)\
│   └── [`PipelineResult`](@ref)\
├── [`AbstractPredictionResult`](@ref)\
│   ├── [`MultiPeriodPredictionResult`](@ref)\
│   ├── [`PopulationPredictionResult`](@ref)\
│   └── [`PredictionResult`](@ref)\
├── [`AbstractPreprocessingResult`](@ref)\
│   ├── [`AbstractPricesPreprocessingResult`](@ref)\
│   │   ├── [`MissingDataFilterResult`](@ref)\
│   │   └── [`PriceGapFillResult`](@ref)\
│   └── [`AbstractReturnsPreprocessingResult`](@ref)\
│       └── [`AssetSelectorResult`](@ref)\
├── [`AbstractPricesResult`](@ref)\
│   └── [`PricesResult`](@ref)\
├── [`AbstractPriorResult`](@ref)\
│   ├── [`HighOrderPrior`](@ref)\
│   └── [`LowOrderPrior`](@ref)\
├── [`AbstractRegressionResult`](@ref)\
│   ├── [`AbstractCrossSectionalRegressionResult`](@ref)\
│   │   └── [`CrossSectionalRegression`](@ref)\
│   └── [`AbstractLoadingsRegressionResult`](@ref)\
│       ├── [`CrossSectionalFactorModel`](@ref)\
│       └── [`Regression`](@ref)\
├── [`AbstractReturnForecastResult`](@ref)\
│   ├── [`CustomValueReturnForecastResult`](@ref)\
│   ├── [`ExpWeightedReturnForecastResult`](@ref)\
│   ├── [`FixedWeightedReturnForecastResult`](@ref)\
│   └── [`TargetReturnForecastResult`](@ref)\
├── [`AbstractReturnsResult`](@ref)\
│   ├── [`PredictionReturnsResult`](@ref)\
│   └── [`ReturnsResult`](@ref)\
├── [`AbstractSearchCrossValidationResult`](@ref)\
│   └── [`SearchCrossValidationResult`](@ref)\
├── [`AbstractTracking`](@ref)\
│   ├── [`RiskTrackingError`](@ref)\
│   └── [`TrackingError`](@ref)\
├── [`AbstractUncertaintySetResult`](@ref)\
│   ├── [`AbstractUncertaintySetClass`](@ref)\
│   │   ├── [`MuUncertaintySetClass`](@ref)\
│   │   └── [`SigmaUncertaintySetClass`](@ref)\
│   ├── [`BoxUncertaintySet`](@ref)\
│   ├── [`CompactCovarianceUncertaintySet`](@ref)\
│   ├── [`EllipsoidalUncertaintySet`](@ref)\
│   ├── [`L1UncertaintySet`](@ref)\
│   ├── [`NormBallUncertaintySet`](@ref)\
│   └── [`SignedL1UncertaintySet`](@ref)\
├── [`AssetPanel`](@ref)\
├── [`BaseHierarchicalOptimisationResult`](@ref)\
│   └── [`HierarchicalResult`](@ref)\
├── [`BaseJuMPOptimisationResult`](@ref)\
│   └── [`JuMPOptimisationResult`](@ref)\
├── [`BlackLittermanViews`](@ref)\
├── [`CalibrationContext`](@ref)\
├── [`ClusterNode`](@ref)\
├── [`CrossValidationResult`](@ref)\
│   ├── [`NonOptimisationCrossValidationResult`](@ref)\
│   │   ├── [`NonOptimisationNonSequentialCrossValidationResult`](@ref)\
│   │   └── [`NonOptimisationSequentialCrossValidationResult`](@ref)\
│   │       └── [`MultipleRandomisedResult`](@ref)\
│   └── [`OptimisationCrossValidationResult`](@ref)\
│       ├── [`NonSequentialCrossValidationResult`](@ref)\
│       │   ├── [`CombinatorialCrossValidationResult`](@ref)\
│       │   └── [`KFoldResult`](@ref)\
│       └── [`SequentialCrossValidationResult`](@ref)\
│           └── [`WalkForwardResult`](@ref)\
├── [`FactorAttributionResult`](@ref)\
├── [`FactorSummaryResult`](@ref)\
├── [`Fees`](@ref)\
├── [`ForecastEvaluationResult`](@ref)\
├── [`ForecastSummaryResult`](@ref)\
├── [`HeldWeightsResult`](@ref)\
├── [`NearOptimalSetup`](@ref)\
├── [`OptimisationModelResult`](@ref)\
│   └── [`JuMPOptimisationSolution`](@ref)\
├── [`OptimisationResult`](@ref)\
│   ├── [`FiniteAllocationOptimisationResult`](@ref)\
│   │   ├── [`DiscreteAllocationResult`](@ref)\
│   │   └── [`GreedyAllocationResult`](@ref)\
│   └── [`NonFiniteAllocationOptimisationResult`](@ref)\
│       ├── [`NonJuMPOptimisationResult`](@ref)\
│       │   ├── [`HierarchicalOptimisationResult`](@ref)\
│       │   │   ├── [`HierarchicalEqualRiskContributionResult`](@ref)\
│       │   │   ├── [`HierarchicalRiskParityResult`](@ref)\
│       │   │   └── [`SchurComplementHierarchicalRiskParityResult`](@ref)\
│       │   ├── [`NaiveOptimisationResult`](@ref)\
│       │   ├── [`NestedClusteredResult`](@ref)\
│       │   ├── [`StackingResult`](@ref)\
│       │   └── [`SubsetResamplingResult`](@ref)\
│       ├── [`NonRiskJuMPOptimisationResult`](@ref)\
│       │   └── [`RelaxedRiskBudgetingResult`](@ref)\
│       └── [`RiskJuMPOptimisationResult`](@ref)\
│           ├── [`FactorRiskContributionResult`](@ref)\
│           ├── [`MeanRiskResult`](@ref)\
│           ├── [`NearOptimalCenteringResult`](@ref)\
│           └── [`RiskBudgetingResult`](@ref)\
├── [`OptimisationReturnCode`](@ref)\
│   ├── [`OptimisationFailure`](@ref)\
│   └── [`OptimisationSuccess`](@ref)\
├── [`PerformanceSummaryResult`](@ref)\
├── [`PipelineContext`](@ref)\
├── [`PipelineUncertaintySets`](@ref)\
├── [`ProcessedAttributes`](@ref)\
│   ├── [`ProcessedJuMPOptimiserAttributes`](@ref)\
│   └── [`ProcessedRiskBudgetingAttributes`](@ref)\
│       ├── [`ProcessedAssetRiskBudgetingAttributes`](@ref)\
│       └── [`ProcessedFactorRiskBudgetingAttributes`](@ref)\
├── [`TimeDependentContext`](@ref)\
├── [`TrainTestSplitResult`](@ref)\
├── [`Turnover`](@ref)\
└── [`VecScalar`](@ref)

```@raw html
</div>
```

## [AbstractEstimator](@id type-hierarchy-AbstractEstimator)

```@raw html
<div class="type-tree">
```

[`AbstractEstimator`](@ref)\
├── [`AbstractAssetPanelEstimator`](@ref)\
│   ├── [`PhylogenyPanel`](@ref)\
│   └── [`RegressionPanel`](@ref)\
├── [`AbstractBaseRiskMeasure`](@ref)\
│   ├── [`NonOptimisationRiskMeasure`](@ref)\
│   │   ├── [`ExpectedReturn`](@ref)\
│   │   ├── [`ExpectedReturnRiskRatio`](@ref)\
│   │   ├── [`MeanReturn`](@ref)\
│   │   ├── [`MeanReturnRiskRatio`](@ref)\
│   │   ├── [`NonOptimisationRiskRatio`](@ref)\
│   │   ├── [`Skewness`](@ref)\
│   │   └── [`ThirdCentralMoment`](@ref)\
│   └── [`OptimisationRiskMeasure`](@ref)\
│       ├── [`HierarchicalRiskMeasure`](@ref)\
│       │   ├── [`EqualRisk`](@ref)\
│       │   ├── [`HighOrderMoment`](@ref)\
│       │   ├── [`MedianAbsoluteDeviation`](@ref)\
│       │   ├── [`RelativeAverageDrawdown`](@ref)\
│       │   ├── [`RelativeConditionalDrawdownatRisk`](@ref)\
│       │   ├── [`RelativeDrawdownatRisk`](@ref)\
│       │   ├── [`RelativeEntropicDrawdownatRisk`](@ref)\
│       │   ├── [`RelativeMaximumDrawdown`](@ref)\
│       │   ├── [`RelativePowerNormDrawdownatRisk`](@ref)\
│       │   ├── [`RelativeRelativisticDrawdownatRisk`](@ref)\
│       │   ├── [`RelativeUlcerIndex`](@ref)\
│       │   └── [`RiskRatio`](@ref)\
│       └── [`RiskMeasure`](@ref)\
│           ├── [`AverageDrawdown`](@ref)\
│           ├── [`BrownianDistanceVariance`](@ref)\
│           ├── [`ConditionalDrawdownatRisk`](@ref)\
│           ├── [`ConditionalValueatRisk`](@ref)\
│           ├── [`ConditionalValueatRiskRange`](@ref)\
│           ├── [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)\
│           ├── [`DistributionallyRobustConditionalValueatRisk`](@ref)\
│           ├── [`DistributionallyRobustConditionalValueatRiskRange`](@ref)\
│           ├── [`DrawdownatRisk`](@ref)\
│           ├── [`EntropicDrawdownatRisk`](@ref)\
│           ├── [`EntropicValueatRisk`](@ref)\
│           ├── [`EntropicValueatRiskRange`](@ref)\
│           ├── [`GenericValueatRiskRange`](@ref)\
│           ├── [`Kurtosis`](@ref)\
│           ├── [`LowOrderMoment`](@ref)\
│           ├── [`MaximumDrawdown`](@ref)\
│           ├── [`NegativeSkewness`](@ref)\
│           ├── [`NoRisk`](@ref)\
│           ├── [`OrderedWeightsArray`](@ref)\
│           ├── [`OrderedWeightsArrayRange`](@ref)\
│           ├── [`PowerNormDrawdownatRisk`](@ref)\
│           ├── [`PowerNormValueatRisk`](@ref)\
│           ├── [`PowerNormValueatRiskRange`](@ref)\
│           ├── [`Range`](@ref)\
│           ├── [`RelativisticDrawdownatRisk`](@ref)\
│           ├── [`RelativisticValueatRisk`](@ref)\
│           ├── [`RelativisticValueatRiskRange`](@ref)\
│           ├── [`RiskTrackingRiskMeasure`](@ref)\
│           ├── [`StandardDeviation`](@ref)\
│           ├── [`TrackingRiskMeasure`](@ref)\
│           ├── [`TurnoverRiskMeasure`](@ref)\
│           ├── [`UlcerIndex`](@ref)\
│           ├── [`UncertaintySetVariance`](@ref)\
│           ├── [`ValueatRisk`](@ref)\
│           ├── [`ValueatRiskRange`](@ref)\
│           ├── [`Variance`](@ref)\
│           ├── [`VarianceSkewKurtosis`](@ref)\
│           └── [`WorstRealisation`](@ref)\
├── [`AbstractCalibrationSeries`](@ref)\
│   ├── [`AbstractDrawdownSeries`](@ref)\
│   │   ├── [`AbsoluteDrawdownSeries`](@ref)\
│   │   └── [`RelativeDrawdownSeries`](@ref)\
│   └── [`ReturnsSeries`](@ref)\
├── [`AbstractCentralityEstimator`](@ref)\
│   └── [`CentralityEstimator`](@ref)\
├── [`AbstractConstraintEstimator`](@ref)\
│   ├── [`AbstractCentralityConstraint`](@ref)\
│   │   └── [`CentralityConstraint`](@ref)\
│   ├── [`AbstractPhylogenyConstraintEstimator`](@ref)\
│   │   ├── [`IntegerPhylogenyEstimator`](@ref)\
│   │   └── [`SemiDefinitePhylogenyEstimator`](@ref)\
│   ├── [`AssetSetsMatrixEstimator`](@ref)\
│   ├── [`ExposureConstraintEstimator`](@ref)\
│   ├── [`JuMPConstraintEstimator`](@ref)\
│   │   ├── [`BudgetConstraintEstimator`](@ref)\
│   │   │   ├── [`BudgetCostEstimator`](@ref)\
│   │   │   │   ├── [`BudgetCosts`](@ref)\
│   │   │   │   └── [`BudgetMarketImpact`](@ref)\
│   │   │   └── [`BudgetEstimator`](@ref)\
│   │   │       └── [`BudgetRange`](@ref)\
│   │   └── [`CustomJuMPConstraint`](@ref)\
│   ├── [`LinearConstraintEstimator`](@ref)\
│   ├── [`RiskBudgetEstimator`](@ref)\
│   ├── [`ThresholdEstimator`](@ref)\
│   └── [`WeightBoundsEstimator`](@ref)\
├── [`AbstractCrossSectionalTransform`](@ref)\
│   ├── [`CrossSectionalGaussianRank`](@ref)\
│   ├── [`CrossSectionalPercentileRank`](@ref)\
│   ├── [`CrossSectionalStandardiser`](@ref)\
│   ├── [`CrossSectionalTanhShrinker`](@ref)\
│   └── [`CrossSectionalWinsoriser`](@ref)\
├── [`AbstractCrossValidationScorer`](@ref)\
│   ├── [`PopulationScorer`](@ref)\
│   └── [`PredictionScorer`](@ref)\
│       └── [`NearestQuantilePrediction`](@ref)\
├── [`AbstractDenoiseEstimator`](@ref)\
│   └── [`Denoise`](@ref)\
├── [`AbstractDescriptorEstimator`](@ref)\
│   ├── [`ChangeInIntensity`](@ref)\
│   ├── [`ChangeToScale`](@ref)\
│   ├── [`DaysToCover`](@ref)\
│   ├── [`EWBeta`](@ref)\
│   ├── [`EWDownsideBeta`](@ref)\
│   ├── [`EWMacroSensitivity`](@ref)\
│   ├── [`EWMean`](@ref)\
│   ├── [`EWResidualVolatility`](@ref)\
│   ├── [`EWVolatility`](@ref)\
│   ├── [`EWVolumeRatio`](@ref)\
│   ├── [`GrowthRate`](@ref)\
│   ├── [`PanelFieldLog`](@ref)\
│   ├── [`PanelFieldRatio`](@ref)\
│   ├── [`Passthrough`](@ref)\
│   ├── [`RollingLogReturn`](@ref)\
│   └── [`RollingMax`](@ref)\
├── [`AbstractDetoneEstimator`](@ref)\
│   └── [`Detone`](@ref)\
├── [`AbstractDistanceEstimator`](@ref)\
│   ├── [`Distance`](@ref)\
│   ├── [`DistanceDistance`](@ref)\
│   └── [`FeatureDistance`](@ref)\
├── [`AbstractEntropyPoolingOptimiser`](@ref)\
│   ├── [`ConditionalValueatRiskEntropyPooling`](@ref)\
│   ├── [`JuMPEntropyPooling`](@ref)\
│   └── [`OptimEntropyPooling`](@ref)\
├── [`AbstractEntropyPoolingViewEstimator`](@ref)\
│   ├── [`AbstractEntropyPoolingTailViewEstimator`](@ref)\
│   │   ├── [`ConditionalValueatRiskView`](@ref)\
│   │   ├── [`EntropicValueatRiskView`](@ref)\
│   │   └── [`RelativisticValueatRiskView`](@ref)\
│   └── [`ValueatRiskView`](@ref)\
├── [`AbstractExpectedReturnsEstimator`](@ref)\
│   ├── [`AbstractShrunkExpectedReturnsEstimator`](@ref)\
│   │   ├── [`EquilibriumExpectedReturns`](@ref)\
│   │   ├── [`ExcessExpectedReturns`](@ref)\
│   │   └── [`ShrunkExpectedReturns`](@ref)\
│   ├── [`CustomValueExpectedReturns`](@ref)\
│   ├── [`ExpWeightedExpectedReturns`](@ref)\
│   ├── [`MedianExpectedReturns`](@ref)\
│   ├── [`SimpleExpectedReturns`](@ref)\
│   ├── [`StandardDeviationExpectedReturns`](@ref)\
│   ├── [`VarianceExpectedReturns`](@ref)\
│   └── [`WindowedExpectedReturns`](@ref)\
├── [`AbstractExposureEstimator`](@ref)\
│   ├── [`CompositeExposure`](@ref)\
│   ├── [`ConstantExposure`](@ref)\
│   ├── [`DerivedExposure`](@ref)\
│   └── [`OneHotExposure`](@ref)\
├── [`AbstractMatrixProcessingEstimator`](@ref)\
│   └── [`MatrixProcessing`](@ref)\
├── [`AbstractOptimalNumberClustersEstimator`](@ref)\
│   └── [`OptimalNumberClusters`](@ref)\
├── [`AbstractOptimisationEstimator`](@ref)\
│   ├── [`BaseOptimisationEstimator`](@ref)\
│   │   ├── [`BaseClusteringOptimisationEstimator`](@ref)\
│   │   │   └── [`HierarchicalOptimiser`](@ref)\
│   │   └── [`BaseJuMPOptimisationEstimator`](@ref)\
│   │       └── [`JuMPOptimiser`](@ref)\
│   └── [`OptimisationEstimator`](@ref)\
│       ├── [`FiniteAllocationOptimisationEstimator`](@ref)\
│       │   ├── [`DiscreteAllocation`](@ref)\
│       │   └── [`GreedyAllocation`](@ref)\
│       └── [`NonFiniteAllocationOptimisationEstimator`](@ref)\
│           ├── [`BaseStackingOptimisationEstimator`](@ref)\
│           │   └── [`Stacking`](@ref)\
│           ├── [`BaseSubsetResamplingOptimisationEstimator`](@ref)\
│           │   └── [`SubsetResampling`](@ref)\
│           ├── [`ClusteringOptimisationEstimator`](@ref)\
│           │   ├── [`HierarchicalEqualRiskContribution`](@ref)\
│           │   ├── [`HierarchicalRiskParity`](@ref)\
│           │   ├── [`NestedClustered`](@ref)\
│           │   └── [`SchurComplementHierarchicalRiskParity`](@ref)\
│           ├── [`JuMPOptimisationEstimator`](@ref)\
│           │   ├── [`RelaxedRiskBudgeting`](@ref)\
│           │   └── [`RiskJuMPOptimisationEstimator`](@ref)\
│           │       ├── [`FactorRiskContribution`](@ref)\
│           │       ├── [`MeanRisk`](@ref)\
│           │       ├── [`NearOptimalCentering`](@ref)\
│           │       └── [`RiskBudgeting`](@ref)\
│           └── [`NaiveOptimisationEstimator`](@ref)\
│               ├── [`EqualWeighted`](@ref)\
│               ├── [`InverseVolatility`](@ref)\
│               └── [`RandomWeighted`](@ref)\
├── [`AbstractOrderedWeightsArrayEstimator`](@ref)\
│   ├── [`NormalisedConstantRelativeRiskAversion`](@ref)\
│   └── [`OWAJuMP`](@ref)\
├── [`AbstractOrderedWeightsArrayFunction`](@ref)\
│   ├── [`LinearMoment`](@ref)\
│   ├── [`OrderedWeightsArrayConditionalValueatRisk`](@ref)\
│   ├── [`OrderedWeightsArrayConditionalValueatRiskRange`](@ref)\
│   ├── [`OrderedWeightsArrayTailGini`](@ref)\
│   └── [`OrderedWeightsArrayTailGiniRange`](@ref)\
├── [`AbstractPanelFieldInput`](@ref)\
│   ├── [`CategoricalPanelInput`](@ref)\
│   ├── [`NumericPanelInput`](@ref)\
│   └── [`TensorPanelInput`](@ref)\
├── [`AbstractPhylogenyEstimator`](@ref)\
│   ├── [`AbstractClustersEstimator`](@ref)\
│   │   ├── [`ClustersEstimator`](@ref)\
│   │   └── [`NetworkClustersEstimator`](@ref)\
│   └── [`AbstractNetworkEstimator`](@ref)\
│       └── [`NetworkEstimator`](@ref)\
├── [`AbstractPipelineEstimator`](@ref)\
│   └── [`Pipeline`](@ref)\
├── [`AbstractPosdefEstimator`](@ref)\
│   └── [`Posdef`](@ref)\
├── [`AbstractPreprocessingEstimator`](@ref)\
│   ├── [`AbstractPricesPreprocessingEstimator`](@ref)\
│   │   ├── [`MissingDataFilter`](@ref)\
│   │   └── [`PriceGapFill`](@ref)\
│   ├── [`AbstractReturnsPreprocessingEstimator`](@ref)\
│   │   └── [`AbstractAssetSelector`](@ref)\
│   │       ├── [`CompleteAssetSelector`](@ref)\
│   │       ├── [`RedundancySelector`](@ref)\
│   │       └── [`ScoreSelector`](@ref)\
│   ├── [`PricesToReturns`](@ref)\
│   └── [`TrainTestSplit`](@ref)\
├── [`AbstractPriorEstimator`](@ref)\
│   ├── [`AbstractHighOrderPriorEstimator`](@ref)\
│   │   ├── [`AbstractHighOrderPriorEstimator_F`](@ref)\
│   │   │   └── [`HighOrderFactorPriorEstimator`](@ref)\
│   │   └── [`HighOrderPriorEstimator`](@ref)\
│   └── [`AbstractLowOrderPriorEstimator`](@ref)\
│       ├── [`AbstractLowOrderPriorEstimator_A`](@ref)\
│       │   ├── [`CrossSectionalFactorPrior`](@ref)\
│       │   └── [`EmpiricalPrior`](@ref)\
│       ├── [`AbstractLowOrderPriorEstimator_AF`](@ref)\
│       │   ├── [`BlackLittermanPrior`](@ref)\
│       │   ├── [`EntropyPoolingPrior`](@ref)\
│       │   ├── [`MeucciEntropyPoolingPrior`](@ref)\
│       │   └── [`OpinionPoolingPrior`](@ref)\
│       └── [`AbstractLowOrderPriorEstimator_F`](@ref)\
│           ├── [`AugmentedBlackLittermanPrior`](@ref)\
│           ├── [`BayesianBlackLittermanPrior`](@ref)\
│           ├── [`FactorBlackLittermanPrior`](@ref)\
│           └── [`FactorPrior`](@ref)\
├── [`AbstractRegressionEstimator`](@ref)\
│   ├── [`AbstractCrossSectionalRegressionEstimator`](@ref)\
│   │   ├── [`CrossSectionalLinearRegression`](@ref)\
│   │   └── [`CrossSectionalTargetRegression`](@ref)\
│   └── [`AbstractTimeSeriesRegressionEstimator`](@ref)\
│       ├── [`DimensionReductionRegression`](@ref)\
│       └── [`StepwiseRegression`](@ref)\
├── [`AbstractRegularisationEstimator`](@ref)\
│   ├── [`L2Regularisation`](@ref)\
│   └── [`LpRegularisation`](@ref)\
├── [`AbstractReturnForecastEstimator`](@ref)\
│   ├── [`CustomValueReturnForecast`](@ref)\
│   ├── [`ExpWeightedReturnForecast`](@ref)\
│   ├── [`FixedWeightedReturnForecast`](@ref)\
│   └── [`TargetReturnForecast`](@ref)\
├── [`AbstractRiskMeasureSettings`](@ref)\
│   ├── [`HierarchicalRiskMeasureSettings`](@ref)\
│   └── [`JuMPRiskMeasureSettings`](@ref)\
│       ├── [`MaxRiskMeasureSettings`](@ref)\
│       └── [`RiskMeasureSettings`](@ref)\
├── [`AbstractSearchCrossValidationEstimator`](@ref)\
│   ├── [`GridSearchCrossValidation`](@ref)\
│   └── [`RandomisedSearchCrossValidation`](@ref)\
├── [`AbstractUncertaintySetEstimator`](@ref)\
│   ├── [`AbstractPriorUncertaintySetEstimator`](@ref)\
│   │   └── [`OrthogonalUncertaintySet`](@ref)\
│   ├── [`BootstrapUncertaintySetEstimator`](@ref)\
│   │   └── [`ARCHUncertaintySet`](@ref)\
│   ├── [`CharacteristicUncertaintySet`](@ref)\
│   ├── [`DeltaUncertaintySet`](@ref)\
│   └── [`NormalUncertaintySet`](@ref)\
├── [`CokurtosisEstimator`](@ref)\
│   ├── [`Cokurtosis`](@ref)\
│   └── [`WindowedCokurtosis`](@ref)\
├── [`CoskewnessEstimator`](@ref)\
│   ├── [`Coskewness`](@ref)\
│   └── [`WindowedCoskewness`](@ref)\
├── [`CoveragePolicy`](@ref)\
├── [`CrossValidationEstimator`](@ref)\
│   ├── [`NonOptimisationCrossValidationEstimator`](@ref)\
│   │   ├── [`NonOptimisationNonSequentialCrossValidationEstimator`](@ref)\
│   │   └── [`NonOptimisationSequentialCrossValidationEstimator`](@ref)\
│   │       └── [`MultipleRandomised`](@ref)\
│   └── [`OptimisationCrossValidationEstimator`](@ref)\
│       ├── [`NonSequentialCrossValidationEstimator`](@ref)\
│       │   ├── [`CombinatorialCrossValidation`](@ref)\
│       │   └── [`KFold`](@ref)\
│       └── [`SequentialCrossValidationEstimator`](@ref)\
│           └── [`WalkForwardEstimator`](@ref)\
│               ├── [`DateWalkForward`](@ref)\
│               └── [`IndexWalkForward`](@ref)\
├── [`CrossValidationSearchScorer`](@ref)\
│   └── [`HighestMeanScore`](@ref)\
├── [`CustomJuMPObjective`](@ref)\
├── [`DateAdjusterEstimator`](@ref)\
├── [`DescriptorScores`](@ref)\
├── [`DynamicAbstractWeights`](@ref)\
├── [`FeesEstimator`](@ref)\
├── [`FiniteAllocationInput`](@ref)\
├── [`FrontierBoundEstimator`](@ref)\
│   ├── [`LinearBound`](@ref)\
│   ├── [`SquareRootBound`](@ref)\
│   └── [`SquaredBound`](@ref)\
├── [`GerberIQDecayEstimator`](@ref)\
│   └── [`ExpGerberIQDecay`](@ref)\
├── [`GerberIQEpsEstimator`](@ref)\
├── [`GerberIQGammaEstimator`](@ref)\
├── [`GerberIQScalerEstimator`](@ref)\
│   └── [`AssetVolatilityGerberIQScaler`](@ref)\
├── [`JuMPReturnsEstimator`](@ref)\
│   ├── [`ArithmeticReturn`](@ref)\
│   ├── [`LogarithmicReturn`](@ref)\
│   └── [`NoReturn`](@ref)\
├── [`JuMPReturnsSettings`](@ref)\
├── [`NormError`](@ref)\
│   ├── [`L1Norm`](@ref)\
│   ├── [`L2Norm`](@ref)\
│   ├── [`LInfNorm`](@ref)\
│   ├── [`LpNorm`](@ref)\
│   └── [`SquaredL2Norm`](@ref)\
├── [`NumberSubsetsEstimator`](@ref)\
├── [`ObjectiveFunction`](@ref)\
│   ├── [`MaximumElementReturn`](@ref)\
│   ├── [`MaximumRatio`](@ref)\
│   ├── [`MaximumReturn`](@ref)\
│   ├── [`MaximumUtility`](@ref)\
│   └── [`MinimumRisk`](@ref)\
├── [`Online`](@ref)\
├── [`OptimisationCrossValidation`](@ref)\
├── [`PipelineStep`](@ref)\
├── [`PriceIngestion`](@ref)\
├── [`RegimeAdjustedMethod`](@ref)\
│   ├── [`FirstMomentRegimeAdjusted`](@ref)\
│   ├── [`LogRegimeAdjusted`](@ref)\
│   └── [`RootMeanSquaredAdjusted`](@ref)\
├── [`Scalariser`](@ref)\
│   ├── [`HierarchicalScalariser`](@ref)\
│   │   └── [`MinScalariser`](@ref)\
│   └── [`NonHierarchicalScalariser`](@ref)\
│       ├── [`LogSumExpScalariser`](@ref)\
│       ├── [`MaxScalariser`](@ref)\
│       └── [`SumScalariser`](@ref)\
├── [`Solver`](@ref)\
├── [`SubsetSizeEstimator`](@ref)\
├── [`TimeDependent`](@ref)\
├── [`TimeDependentCallable`](@ref)\
│   ├── [`TimeDependentConstraintCallable`](@ref)\
│   └── [`TimeDependentOptimiserCallable`](@ref)\
├── [`TurnoverEstimator`](@ref)\
├── [`UniverseSets`](@ref)\
└── [`WindowSizeEstimator`](@ref)

```@raw html
</div>
```

## [AbstractAlgorithm](@id type-hierarchy-AbstractAlgorithm)

```@raw html
<div class="type-tree">
```

[`AbstractAlgorithm`](@ref)\
├── [`ARCHBootstrapSet`](@ref)\
│   ├── [`CircularBootstrap`](@ref)\
│   ├── [`MovingBootstrap`](@ref)\
│   └── [`StationaryBootstrap`](@ref)\
├── [`AbstractBins`](@ref)\
│   ├── [`BinWidthBins`](@ref)\
│   │   ├── [`FreedmanDiaconis`](@ref)\
│   │   ├── [`Knuth`](@ref)\
│   │   └── [`Scott`](@ref)\
│   └── [`HacineGharbiRavier`](@ref)\
├── [`AbstractCalibrationAlgorithm`](@ref)\
│   ├── [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)\
│   │   ├── [`ConcentrationRadius`](@ref)\
│   │   ├── [`DimensionalRateRadius`](@ref)\
│   │   ├── [`DualNormRadius`](@ref)\
│   │   └── [`RateRadius`](@ref)\
│   ├── [`AbstractAmbiguityTailWeightCalibrationAlgorithm`](@ref)\
│   │   └── [`TailTermParity`](@ref)\
│   ├── [`AbstractDeformationCalibrationAlgorithm`](@ref)\
│   │   ├── [`EntropyBudget`](@ref)\
│   │   ├── [`HillTailDecay`](@ref)\
│   │   └── [`RadialTailDecay`](@ref)\
│   ├── [`AbstractNormCeilingCalibrationAlgorithm`](@ref)\
│   │   └── [`EffectiveAssetFloor`](@ref)\
│   └── [`AbstractSignificanceCalibrationAlgorithm`](@ref)\
│       ├── [`RateSignificance`](@ref)\
│       └── [`ScenarioCount`](@ref)\
├── [`AbstractCentralityPolarity`](@ref)\
│   ├── [`DistancePolarity`](@ref)\
│   └── [`SimilarityPolarity`](@ref)\
├── [`AbstractCollapseAlgorithm`](@ref)\
│   ├── [`MeanCollapse`](@ref)\
│   └── [`MedianCollapse`](@ref)\
├── [`AbstractCompactRadiusAlgorithm`](@ref)\
│   ├── [`ResidualInflation`](@ref)\
│   └── [`VarianceFraction`](@ref)\
├── [`AbstractConstraintSpace`](@ref)\
│   └── [`FactorSpace`](@ref)\
├── [`AbstractCoverageAlgorithm`](@ref)\
│   ├── [`DecayCoverage`](@ref)\
│   ├── [`ExpireCoverage`](@ref)\
│   └── [`ResetCoverage`](@ref)\
├── [`AbstractCrossSectionalWeightsAlgorithm`](@ref)\
│   ├── [`BlendedInverseVarianceWeights`](@ref)\
│   └── [`MarketCapWeights`](@ref)\
├── [`AbstractCustomValue`](@ref)\
│   └── [`CustomExpectedReturnsValueAlgorithm`](@ref)\
├── [`AbstractDenoiseAlgorithm`](@ref)\
│   ├── [`FixedDenoise`](@ref)\
│   ├── [`ShrunkDenoise`](@ref)\
│   └── [`SpectralDenoise`](@ref)\
├── [`AbstractDistanceAlgorithm`](@ref)\
│   ├── [`CanonicalDistance`](@ref)\
│   ├── [`CorrelationDistance`](@ref)\
│   ├── [`LogDistance`](@ref)\
│   ├── [`SimpleAbsoluteDistance`](@ref)\
│   ├── [`SimpleDistance`](@ref)\
│   └── [`VariationInfoDistance`](@ref)\
├── [`AbstractEntropyPoolingAlgorithm`](@ref)\
│   ├── [`H0_EntropyPooling`](@ref)\
│   ├── [`H1_EntropyPooling`](@ref)\
│   └── [`H2_EntropyPooling`](@ref)\
├── [`AbstractEntropyPoolingOptAlgorithm`](@ref)\
│   ├── [`ExpEntropyPooling`](@ref)\
│   └── [`LogEntropyPooling`](@ref)\
├── [`AbstractEntropyPoolingViewFormulation`](@ref)\
│   ├── [`AbstractConditionalValueatRiskViewFormulation`](@ref)\
│   │   ├── [`IntegerConditionalValueatRiskView`](@ref)\
│   │   ├── [`LinearConditionalValueatRiskView`](@ref)\
│   │   └── [`SequentialConditionalValueatRiskView`](@ref)\
│   ├── [`AbstractEntropicValueatRiskViewFormulation`](@ref)\
│   │   ├── [`ConicEntropicValueatRiskView`](@ref)\
│   │   ├── [`GridEntropicValueatRiskView`](@ref)\
│   │   └── [`SequentialEntropicValueatRiskView`](@ref)\
│   └── [`AbstractRelativisticValueatRiskViewFormulation`](@ref)\
│       ├── [`ConicRelativisticValueatRiskView`](@ref)\
│       ├── [`GridRelativisticValueatRiskView`](@ref)\
│       └── [`SequentialRelativisticValueatRiskView`](@ref)\
├── [`AbstractEstimatorValueAlgorithm`](@ref)\
│   └── [`VectorAbstractEstimatorValueAlgorithm`](@ref)\
│       └── [`UniformValues`](@ref)\
├── [`AbstractExpectedReturnsAlgorithm`](@ref)\
│   ├── [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)\
│   │   ├── [`BayesStein`](@ref)\
│   │   ├── [`BodnarOkhrinParolya`](@ref)\
│   │   └── [`JamesStein`](@ref)\
│   └── [`AbstractShrunkExpectedReturnsTarget`](@ref)\
│       ├── [`GrandMean`](@ref)\
│       ├── [`MeanSquaredError`](@ref)\
│       └── [`VolatilityWeighted`](@ref)\
├── [`AbstractFeatureCollapseAlgorithm`](@ref)\
│   ├── [`AggregateDistances`](@ref)\
│   ├── [`AggregateFeatures`](@ref)\
│   ├── [`LastObservation`](@ref)\
│   └── [`StackObservations`](@ref)\
├── [`AbstractFeeAmortisation`](@ref)\
│   ├── [`AmortisedFees`](@ref)\
│   └── [`FirstObservationFees`](@ref)\
├── [`AbstractForecastTarget`](@ref)\
│   ├── [`AssetReturnTarget`](@ref)\
│   ├── [`IdiosyncraticTarget`](@ref)\
│   └── [`PanelFieldTarget`](@ref)\
├── [`AbstractForecastUnit`](@ref)\
│   ├── [`IdiosyncraticReturnUnit`](@ref)\
│   └── [`IdiosyncraticSharpeUnit`](@ref)\
├── [`AbstractGapReturnAlgorithm`](@ref)\
│   └── [`CatchUpGapReturn`](@ref)\
├── [`AbstractMatrixProcessingAlgorithm`](@ref)\
│   └── [`InverseMatrixSparsificationAlgorithm`](@ref)\
│       └── [`LoGo`](@ref)\
├── [`AbstractMomentAlgorithm`](@ref)\
│   ├── [`FullMoment`](@ref)\
│   ├── [`GerberCovarianceAlgorithm`](@ref)\
│   │   ├── [`Gerber0`](@ref)\
│   │   ├── [`Gerber1`](@ref)\
│   │   └── [`Gerber2`](@ref)\
│   ├── [`GerberIQCovarianceAlgorithm`](@ref)\
│   │   ├── [`BasicGerberIQ`](@ref)\
│   │   ├── [`FullGerberIQ`](@ref)\
│   │   └── [`PartialGerberIQ`](@ref)\
│   ├── [`SemiMoment`](@ref)\
│   └── [`SmythBrobyCovarianceAlgorithm`](@ref)\
│       ├── [`SmythBroby0`](@ref)\
│       ├── [`SmythBroby1`](@ref)\
│       ├── [`SmythBroby2`](@ref)\
│       ├── [`SmythBrobyCount0`](@ref)\
│       ├── [`SmythBrobyCount1`](@ref)\
│       ├── [`SmythBrobyCount2`](@ref)\
│       ├── [`SmythBrobyGerber0`](@ref)\
│       ├── [`SmythBrobyGerber1`](@ref)\
│       └── [`SmythBrobyGerber2`](@ref)\
├── [`AbstractOptimalNumberClustersAlgorithm`](@ref)\
│   ├── [`SecondOrderDifference`](@ref)\
│   └── [`SilhouetteScore`](@ref)\
├── [`AbstractOrderedWeightsArrayAlgorithm`](@ref)\
│   ├── [`MaximumEntropy`](@ref)\
│   └── [`SquaredOrderedWeightsArrayAlgorithm`](@ref)\
│       ├── [`MinimumSquaredDistance`](@ref)\
│       └── [`MinimumSumSquares`](@ref)\
├── [`AbstractOrthogonalScaling`](@ref)\
│   ├── [`IdentityScaling`](@ref)\
│   └── [`IdiosyncraticVarianceScaling`](@ref)\
├── [`AbstractOrthogonalityMetric`](@ref)\
│   ├── [`BenchmarkWeightMetric`](@ref)\
│   ├── [`IdentityMetric`](@ref)\
│   ├── [`InverseIdiosyncraticVarianceMetric`](@ref)\
│   └── [`RegressionWeightMetric`](@ref)\
├── [`AbstractPanelFillAlgorithm`](@ref)\
│   ├── [`BackwardPanelFill`](@ref)\
│   ├── [`ConstantPanelFill`](@ref)\
│   ├── [`ForwardPanelFill`](@ref)\
│   └── [`NoPanelFill`](@ref)\
├── [`AbstractPhylogenyAlgorithm`](@ref)\
│   ├── [`AbstractCentralityAlgorithm`](@ref)\
│   │   ├── [`BetweennessCentrality`](@ref)\
│   │   ├── [`ClosenessCentrality`](@ref)\
│   │   ├── [`DegreeCentrality`](@ref)\
│   │   ├── [`EigenvectorCentrality`](@ref)\
│   │   ├── [`KatzCentrality`](@ref)\
│   │   ├── [`Pagerank`](@ref)\
│   │   ├── [`RadialityCentrality`](@ref)\
│   │   └── [`StressCentrality`](@ref)\
│   ├── [`AbstractClustersAlgorithm`](@ref)\
│   │   ├── [`AbstractHierarchicalClusteringAlgorithm`](@ref)\
│   │   │   ├── [`DBHT`](@ref)\
│   │   │   └── [`HClustAlgorithm`](@ref)\
│   │   └── [`AbstractNonHierarchicalClusteringAlgorithm`](@ref)\
│   │       └── [`KMeansAlgorithm`](@ref)\
│   └── [`AbstractTreeType`](@ref)\
│       ├── [`BoruvkaTree`](@ref)\
│       ├── [`KruskalTree`](@ref)\
│       └── [`PrimTree`](@ref)\
├── [`AbstractPhylogenyFeatureAlgorithm`](@ref)\
│   └── [`Proximity`](@ref)\
├── [`AbstractPreorderBy`](@ref)\
│   └── [`PreorderTreeByID`](@ref)\
├── [`AbstractPreviousWeightsSource`](@ref)\
│   └── [`DriftedWeights`](@ref)\
├── [`AbstractRedundancyAlgorithm`](@ref)\
│   ├── [`ClusterGroups`](@ref)\
│   ├── [`CorrelationComponents`](@ref)\
│   └── [`PairwiseCorrelation`](@ref)\
├── [`AbstractRegressionAlgorithm`](@ref)\
│   ├── [`AbstractCrossSectionalSolveAlgorithm`](@ref)\
│   │   ├── [`MinimumNormSolve`](@ref)\
│   │   ├── [`PseudoInverseFallback`](@ref)\
│   │   ├── [`RankDeficiencyRefusal`](@ref)\
│   │   └── [`UncheckedSolve`](@ref)\
│   ├── [`AbstractRegressionTarget`](@ref)\
│   │   ├── [`GeneralisedLinearModel`](@ref)\
│   │   └── [`LinearModel`](@ref)\
│   ├── [`AbstractStepwiseRegressionAlgorithm`](@ref)\
│   │   ├── [`BackwardElimination`](@ref)\
│   │   └── [`ForwardSelection`](@ref)\
│   ├── [`AbstractStepwiseRegressionCriterion`](@ref)\
│   │   └── [`PValue`](@ref)\
│   └── [`DimensionReductionTarget`](@ref)\
│       ├── [`PCA`](@ref)\
│       └── [`PPCA`](@ref)\
├── [`AbstractRiskSeriesAlgorithm`](@ref)\
│   ├── [`DrawdownRiskSeries`](@ref)\
│   └── [`NetReturnsRiskSeries`](@ref)\
├── [`AbstractSearchCrossValidationAlgorithm`](@ref)\
├── [`AbstractSelectionRule`](@ref)\
│   ├── [`QuantileRule`](@ref)\
│   ├── [`RankRule`](@ref)\
│   └── [`ThresholdRule`](@ref)\
├── [`AbstractSeparationAlgorithm`](@ref)\
│   ├── [`HopCount`](@ref)\
│   └── [`PathLength`](@ref)\
├── [`AbstractSeparationDecayAlgorithm`](@ref)\
│   ├── [`ExponentialDecay`](@ref)\
│   ├── [`LinearDecay`](@ref)\
│   ├── [`NoDecay`](@ref)\
│   └── [`ReciprocalDecay`](@ref)\
├── [`AbstractSimilarityMatrixAlgorithm`](@ref)\
│   ├── [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)\
│   │   ├── [`ComplementSimilarity`](@ref)\
│   │   ├── [`ExponentialSimilarity`](@ref)\
│   │   ├── [`GeneralExponentialSimilarity`](@ref)\
│   │   └── [`MaximumDistanceSimilarity`](@ref)\
│   └── [`AngularSimilarity`](@ref)\
├── [`AbstractTrackingAlgorithm`](@ref)\
│   ├── [`ReturnsTracking`](@ref)\
│   └── [`WeightsTracking`](@ref)\
├── [`AbstractUncertaintyEpsAlgorithm`](@ref)\
│   └── [`ActiveAssetsUncertaintyAlgorithm`](@ref)\
├── [`AbstractUncertaintyKAlgorithm`](@ref)\
│   ├── [`ChiSqKUncertaintyAlgorithm`](@ref)\
│   ├── [`GeneralKUncertaintyAlgorithm`](@ref)\
│   └── [`NormalKUncertaintyAlgorithm`](@ref)\
├── [`AbstractUncertaintySetAlgorithm`](@ref)\
│   ├── [`BoxUncertaintySetAlgorithm`](@ref)\
│   ├── [`EllipsoidalUncertaintySetAlgorithm`](@ref)\
│   ├── [`L1UncertaintySetAlgorithm`](@ref)\
│   ├── [`NormBallUncertaintySetAlgorithm`](@ref)\
│   └── [`SignedL1UncertaintySetAlgorithm`](@ref)\
├── [`AbstractWeightDrift`](@ref)\
│   └── [`SelfFinancingDrift`](@ref)\
├── [`BrownianDistanceVarianceFormulation`](@ref)\
│   ├── [`IneqBrownianDistanceVariance`](@ref)\
│   └── [`NormOneConeBrownianDistanceVariance`](@ref)\
├── [`CrossValidationAlgorithm`](@ref)\
├── [`DBHTRootMethod`](@ref)\
│   ├── [`EqualRoot`](@ref)\
│   └── [`UniqueRoot`](@ref)\
├── [`EntropyFormulation`](@ref)\
│   ├── [`ExponentialConeEntropy`](@ref)\
│   └── [`RelativeEntropy`](@ref)\
├── [`Frontier`](@ref)\
├── [`HopCountAlgorithm`](@ref)\
│   └── [`HopCountQuantile`](@ref)\
├── [`ImpliedVolatilityAlgorithm`](@ref)\
│   ├── [`ImpliedVolatilityPremium`](@ref)\
│   └── [`ImpliedVolatilityRegression`](@ref)\
├── [`JuMPWeightFinaliserFormulation`](@ref)\
│   ├── [`AbsoluteErrorWeightFinaliser`](@ref)\
│   ├── [`RelativeErrorWeightFinaliser`](@ref)\
│   ├── [`SquaredAbsoluteErrorWeightFinaliser`](@ref)\
│   └── [`SquaredRelativeErrorWeightFinaliser`](@ref)\
├── [`MedianCenteringFunction`](@ref)\
│   ├── [`MeanCentering`](@ref)\
│   └── [`MedianCentering`](@ref)\
├── [`MomentMeasureAlgorithm`](@ref)\
│   ├── [`HighOrderMomentMeasureAlgorithm`](@ref)\
│   │   ├── [`StandardisedHighOrderMoment`](@ref)\
│   │   └── [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref)\
│   │       ├── [`FourthMoment`](@ref)\
│   │       └── [`ThirdLowerMoment`](@ref)\
│   └── [`LowOrderMomentMeasureAlgorithm`](@ref)\
│       ├── [`SecondMoment`](@ref)\
│       └── [`UnstandardisedLowOrderMomentMeasureAlgorithm`](@ref)\
│           ├── [`EvenMoment`](@ref)\
│           ├── [`FirstLowerMoment`](@ref)\
│           └── [`MeanAbsoluteDeviation`](@ref)\
├── [`NoDefault`](@ref)\
├── [`OpinionPoolingAlgorithm`](@ref)\
│   ├── [`LinearOpinionPooling`](@ref)\
│   └── [`LogarithmicOpinionPooling`](@ref)\
├── [`OptimisationAlgorithm`](@ref)\
│   ├── [`NearOptimalCenteringAlgorithm`](@ref)\
│   │   ├── [`ConstrainedNearOptimalCentering`](@ref)\
│   │   └── [`UnconstrainedNearOptimalCentering`](@ref)\
│   ├── [`RelaxedRiskBudgetingAlgorithm`](@ref)\
│   │   ├── [`BasicRelaxedRiskBudgeting`](@ref)\
│   │   ├── [`RegularisedPenalisedRelaxedRiskBudgeting`](@ref)\
│   │   └── [`RegularisedRelaxedRiskBudgeting`](@ref)\
│   ├── [`RiskBudgetingAlgorithm`](@ref)\
│   │   ├── [`AssetRiskBudgeting`](@ref)\
│   │   └── [`FactorRiskBudgeting`](@ref)\
│   └── [`RiskBudgetingFormulation`](@ref)\
│       ├── [`LogRiskBudgeting`](@ref)\
│       └── [`MixedIntegerRiskBudgeting`](@ref)\
├── [`OrderedWeightsArrayFormulation`](@ref)\
│   ├── [`ApproxOrderedWeightsArray`](@ref)\
│   └── [`ExactOrderedWeightsArray`](@ref)\
├── [`PathLengthAlgorithm`](@ref)\
│   └── [`PathLengthQuantile`](@ref)\
├── [`PreviousWeightsFunction`](@ref)\
├── [`RegimeAdjustedTarget`](@ref)\
│   ├── [`DiagonalTarget`](@ref)\
│   ├── [`MahalanobisTarget`](@ref)\
│   └── [`PortfolioTarget`](@ref)\
├── [`RelativisticValueatRiskViewBracket`](@ref)\
├── [`SchurComplementAlgorithm`](@ref)\
│   ├── [`MonotonicSchurComplement`](@ref)\
│   └── [`NonMonotonicSchurComplement`](@ref)\
├── [`SchurComplementParams`](@ref)\
├── [`SecondMomentFormulation`](@ref)\
│   ├── [`RSOCRiskExpr`](@ref)\
│   ├── [`SOCRiskExpr`](@ref)\
│   └── [`VarianceFormulation`](@ref)\
│       ├── [`QuadRiskExpr`](@ref)\
│       └── [`SquaredSOCRiskExpr`](@ref)\
├── [`TopologyOnly`](@ref)\
├── [`ValueatRiskFormulation`](@ref)\
│   ├── [`DistributionValueatRisk`](@ref)\
│   └── [`MIPValueatRisk`](@ref)\
├── [`VariableTracking`](@ref)\
│   ├── [`DependentVariableTracking`](@ref)\
│   └── [`IndependentVariableTracking`](@ref)\
├── [`VectorToScalarMeasure`](@ref)\
│   ├── [`MaxValue`](@ref)\
│   ├── [`MeanValue`](@ref)\
│   ├── [`MedianValue`](@ref)\
│   ├── [`MinValue`](@ref)\
│   ├── [`ModeValue`](@ref)\
│   ├── [`ProdValue`](@ref)\
│   ├── [`StandardisedValue`](@ref)\
│   ├── [`StdValue`](@ref)\
│   ├── [`SumValue`](@ref)\
│   └── [`VarValue`](@ref)\
└── [`WeightFinaliser`](@ref)\
    ├── [`IterativeWeightFinaliser`](@ref)\
    └── [`JuMPWeightFinaliser`](@ref)

```@raw html
</div>
```

## [AbstractCovarianceEstimator](@id type-hierarchy-AbstractCovarianceEstimator)

```@raw html
<div class="type-tree">
```

[`AbstractCovarianceEstimator`](@ref)\
├── [`AbstractVarianceEstimator`](@ref)\
│   ├── [`ExpWeightedVariance`](@ref)\
│   ├── [`RegimeAdjustedExpWeightedVariance`](@ref)\
│   ├── [`SimpleVariance`](@ref)\
│   └── [`WindowedVariance`](@ref)\
├── [`BaseGerberCovariance`](@ref)\
│   ├── [`BaseGerberIQCovariance`](@ref)\
│   │   └── [`GerberIQCovariance`](@ref)\
│   ├── [`BaseSmythBrobyCovariance`](@ref)\
│   │   └── [`SmythBrobyCovariance`](@ref)\
│   └── [`GerberCovariance`](@ref)\
├── [`CorrelationCovariance`](@ref)\
├── [`Covariance`](@ref)\
├── [`DistanceCovariance`](@ref)\
├── [`ExpWeightedCovariance`](@ref)\
├── [`GeneralCovariance`](@ref)\
├── [`ImpliedVolatility`](@ref)\
├── [`LowerTailDependenceCovariance`](@ref)\
├── [`MutualInfoCovariance`](@ref)\
├── [`PortfolioOptimisersCovariance`](@ref)\
├── [`RankCovarianceEstimator`](@ref)\
│   ├── [`KendallCovariance`](@ref)\
│   └── [`SpearmanCovariance`](@ref)\
├── [`RegimeAdjustedExpWeightedCovariance`](@ref)\
└── [`WindowedCovariance`](@ref)

```@raw html
</div>
```
