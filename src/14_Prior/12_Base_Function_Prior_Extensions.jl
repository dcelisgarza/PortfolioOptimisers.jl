for p ∈ (EmpiricalPriorResult, FactorPriorResult, HighOrderPriorResult,
         BlackLittermanPriorResult, BayesianBlackLittermanPriorResult,
         FactorBlackLittermanPriorResult, AugmentedBlackLittermanPriorResult,
         EntropyPoolingPriorResult)
    eval(quote
             Base.iterate(S::$(p), state = 1) = state > 1 ? nothing : (S, state + 1)
             #  function Base.String(s::$(p))
             #      return $name
             #  end
             #  function Base.Symbol(::$(p))
             #      return Symbol($name)
             #  end
             function Base.length(::$(p))
                 return 1
             end
             #  function Base.getindex(S::$(p), ::Any)
             #      return S
             #  end
             #  function Base.view(S::$(p), ::Any)
             #      return S
             #  end
         end)
end