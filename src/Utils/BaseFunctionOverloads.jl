for (op, name) ∈
    zip((Solver, LinearConstraintSide, BlackLittermanViewsEstimator, LinearConstraint,
         MeanReturn, Skewness, ThirdCentralMoment, FourthLowerMoment, SemiKurtosis,
         SemiSkewness, ThirdLowerMoment, FourthCentralMoment, Kurtosis, ValueatRiskRange,
         ValueatRisk, DrawdownatRisk, RelativeAverageDrawdown,
         RelativeConditionalDrawdownatRisk, RelativeDrawdownatRisk,
         RelativeEntropicDrawdownatRisk, RelativeMaximumDrawdown,
         RelativeRelativisticDrawdownatRisk, RelativeUlcerIndex, FirstLowerMoment,
         NegativeQuadraticSemiSkewness, NegativeSemiSkewness, SemiStandardDeviation,
         SecondLowerMoment, SquareRootSemiKurtosis, BrownianDistanceVariance,
         ConditionalValueatRiskRange, EntropicValueatRiskRange, MeanAbsoluteDeviation,
         NegativeQuadraticSkewness, NegativeSkewness, Range, RelativisticValueatRiskRange,
         SquareRootKurtosis, StandardDeviation, UncertaintySetVariance, Variance,
         ConditionalValueatRisk, DistributionallyRobustConditionalValueatRisk,
         EntropicValueatRisk, RelativisticValueatRisk, WorstRealisation, AverageDrawdown,
         ConditionalDrawdownatRisk, EntropicDrawdownatRisk, MaximumDrawdown,
         RelativisticDrawdownatRisk, UlcerIndex, TrackingRiskMeasure, TurnoverRiskMeasure,
         OrderedWeightsArray, ConstantEntropyPoolingConstraintEstimator,
         C0_LinearEntropyPoolingConstraintEstimator,
         C1_LinearEntropyPoolingConstraintEstimator, SkewnessEntropyPoolingViewAlgorithm,
         KurtosisEntropyPoolingAlgorithm, C2_LinearEntropyPoolingConstraintEstimator,
         C4_LinearEntropyPoolingConstraintEstimator, ContinuousEntropyPoolingViewEstimator,
         LowOrderPriorResult, HighOrderPriorResult, EqualRiskMeasure, CentralityConstraint),
        ("Solver", "LinearConstraintSide", "BlackLittermanViewsEstimator",
         "LinearConstraint", "MeanReturn", "Skewness", "ThirdCentralMoment",
         "FourthLowerMoment", "SemiKurtosis", "SemiSkewness", "ThirdLowerMoment",
         "FourthCentralMoment", "Kurtosis", "ValueatRiskRange", "ValueatRisk",
         "DrawdownatRisk", "RelativeAverageDrawdown", "RelativeConditionalDrawdownatRisk",
         "RelativeDrawdownatRisk", "RelativeEntropicDrawdownatRisk",
         "RelativeMaximumDrawdown", "RelativeRelativisticDrawdownatRisk",
         "RelativeUlcerIndex", "FirstLowerMoment", "NegativeQuadraticSemiSkewness",
         "NegativeSemiSkewness", "SemiStandardDeviation", "SecondLowerMoment",
         "SquareRootSemiKurtosis", "BrownianDistanceVariance",
         "ConditionalValueatRiskRange", "EntropicValueatRiskRange", "MeanAbsoluteDeviation",
         "NegativeQuadraticSkewness", "NegativeSkewness", "Range",
         "RelativisticValueatRiskRange", "SquareRootKurtosis", "StandardDeviation",
         "UncertaintySetVariance", "Variance", "ConditionalValueatRisk",
         "DistributionallyRobustConditionalValueatRisk", "EntropicValueatRisk",
         "RelativisticValueatRisk", "WorstRealisation", "AverageDrawdown",
         "ConditionalDrawdownatRisk", "EntropicDrawdownatRisk", "MaximumDrawdown",
         "RelativisticDrawdownatRisk", "UlcerIndex", "TrackingRiskMeasure",
         "TurnoverRiskMeasure", "OrderedWeightsArray",
         "ConstantEntropyPoolingConstraintEstimator",
         "C0_LinearEntropyPoolingConstraintEstimator",
         "C1_LinearEntropyPoolingConstraintEstimator",
         "SkewnessEntropyPoolingViewAlgorithm", "KurtosisEntropyPoolingAlgorithm",
         "C2_LinearEntropyPoolingConstraintEstimator",
         "C4_LinearEntropyPoolingConstraintEstimator",
         "ContinuousEntropyPoolingViewEstimator", "LowOrderPriorResult",
         "HighOrderPriorResult", "EqualRiskMeasure", "CentralityConstraint"))
    eval(quote
             Base.iterate(S::$op, state = 1) = state > 1 ? nothing : (S, state + 1)
             function Base.String(s::$op)
                 return $name
             end
             function Base.Symbol(::$op)
                 return Symbol($name)
             end
             function Base.length(::$op)
                 return 1
             end
             function Base.getindex(S::$op, ::Any)
                 return S
             end
             function Base.view(S::$op, ::Any)
                 return S
             end
         end)
end
