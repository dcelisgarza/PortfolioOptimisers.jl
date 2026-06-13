# Discrete allocation

```@docs
DiscreteAllocationResult
factory(res::DiscreteAllocationResult, fb::Option{<:FOptE_FOpt})
DiscreteAllocation
finite_sub_allocation
optimise(da::DiscreteAllocation{<:Any, <:Any, <:Any, Nothing}, w::VecNum,
                  p::VecNum, cash::Number = 1e6, T::Option{<:Number} = nothing,
                  fees::Option{<:Fees} = nothing; str_names::Bool = false,
                  save::Bool = true, kwargs...)
set_discrete_error!
```
