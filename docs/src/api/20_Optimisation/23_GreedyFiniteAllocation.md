# Greedy allocation

```@docs
GreedyAllocationResult
factory(res::GreedyAllocationResult, fb::Option{<:FOptE_FOpt})
GreedyAllocation
roundmult
finite_sub_allocation!
optimise(ga::GreedyAllocation{<:Any, <:Any, <:Any, Nothing}, w::VecNum, p::VecNum,
                  cash::Number = 1e6, T::Option{<:Number} = nothing,
                  fees::Option{<:Fees} = nothing; kwargs...)
```
