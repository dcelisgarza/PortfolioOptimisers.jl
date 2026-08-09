# Custom value expected returns

```@docs
CustomExpectedReturnsValueAlgorithm
CER_Func_Num_VecNum
CustomValueExpectedReturns
mean(me::CustomValueExpectedReturns{<:Number}, X::MatNum; dims::Int = 1,
                         kwargs...)
mean(me::CustomValueExpectedReturns{<:VecNum}, X::MatNum; dims::Int = 1,
                         kwargs...)
mean(me::CustomValueExpectedReturns{<:Function}, X::MatNum; dims::Int = 1,
                         kwargs...)
```
