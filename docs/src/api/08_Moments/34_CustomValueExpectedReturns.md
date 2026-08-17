# Custom value expected returns

```@docs
CustomExpectedReturnsValueAlgorithm
CER_Func_Num_VecNum
CustomValueExpectedReturns
assert_custom_expected_returns_val
mean(me::CustomValueExpectedReturns{<:Number}, X::MatNum; dims::Int = 1,
                         kwargs...)
mean(me::CustomValueExpectedReturns{<:VecNum}, X::MatNum; dims::Int = 1,
                         kwargs...)
mean(me::CustomValueExpectedReturns{<:Union{<:Function, <:CustomExpectedReturnsValueAlgorithm}},
                         X::MatNum; dims::Int = 1, kwargs...)
```
