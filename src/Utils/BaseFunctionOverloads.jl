for (op, name) ∈ zip((Solver, LinearConstraint, LinearConstraintAtom, MeanReturn, Skewness,
                      ThirdCentralMoment, FourthLowerPartialMoment, SemiKurtosis, SemiSkewness,
                      ThirdLowerPartialMoment, FourthCentralMoment, Kurtosis, ValueatRiskRange,
                      ValueatRisk, DrawdownatRisk, RelativeAverageDrawdown,
                      RelativeConditionalDrawdownatRisk, RelativeDrawdownatRisk,
                      RelativeEntropicDrawdownatRisk, RelativeMaximumDrawdown,
                      RelativeRelativisticDrawdownatRisk, RelativeUlcerIndex, FirstLowerPartialModel,
                      NegativeQuadraticSemiSkewness, NegativeSemiSkewness, SemiStandardDeviation,
                      SemiVariance, SquareRootSemiKurtosis, BrownianDistanceVariance,
                      ConditionalValueatRiskRange, EntropicValueatRiskRange, GiniMeanDifference,
                      MeanAbsoluteDeviation, NegativeQuadraticSkewness, NegativeSkewness, Range,
                      RelativisticValueatRiskRange, SquareRootKurtosis, StandardDeviation, TailGiniRange,
                      UncertaintySetVariance, Variance, ConditionalValueatRisk,
                      DistributionallyRobustConditionalValueatRisk, EntropicValueatRisk,
                      RelativisticValueatRisk, TailGini, WorstRealisation, AverageDrawdown,
                      ConditionalDrawdownatRisk, EntropicDrawdownatRisk, MaximumDrawdown,
                      RelativisticDrawdownatRisk, UlcerIndex, TrackingRiskMeasure, TurnoverRiskMeasure,
                      OrderedWeightsArray),
                     ("Solver", "LinearConstraint", "LinearConstraintAtom", "MeanReturn", "Skewness",
                      "ThirdCentralMoment", "FourthLowerPartialMoment", "SemiKurtosis", "SemiSkewness",
                      "ThirdLowerPartialMoment", "FourthCentralMoment", "Kurtosis", "ValueatRiskRange",
                      "ValueatRisk", "DrawdownatRisk", "RelativeAverageDrawdown",
                      "RelativeConditionalDrawdownatRisk", "RelativeDrawdownatRisk",
                      "RelativeEntropicDrawdownatRisk", "RelativeMaximumDrawdown",
                      "RelativeRelativisticDrawdownatRisk", "RelativeUlcerIndex",
                      "FirstLowerPartialModel", "NegativeQuadraticSemiSkewness", "NegativeSemiSkewness",
                      "SemiStandardDeviation", "SemiVariance", "SquareRootSemiKurtosis",
                      "BrownianDistanceVariance", "ConditionalValueatRiskRange",
                      "EntropicValueatRiskRange", "GiniMeanDifference", "MeanAbsoluteDeviation",
                      "NegativeQuadraticSkewness", "NegativeSkewness", "Range",
                      "RelativisticValueatRiskRange", "SquareRootKurtosis", "StandardDeviation",
                      "TailGiniRange", "UncertaintySetVariance", "Variance", "ConditionalValueatRisk",
                      "DistributionallyRobustConditionalValueatRisk", "EntropicValueatRisk",
                      "RelativisticValueatRisk", "TailGini", "WorstRealisation", "AverageDrawdown",
                      "ConditionalDrawdownatRisk", "EntropicDrawdownatRisk", "MaximumDrawdown",
                      "RelativisticDrawdownatRisk", "UlcerIndex", "TrackingRiskMeasure",
                      "TurnoverRiskMeasure", "OrderedWeightsArray"))
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
