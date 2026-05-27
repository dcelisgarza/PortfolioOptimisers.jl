# Custom value expected returns

```@docs
CustomValueExpectedReturns
mean(me::CustomValueExpectedReturns{<:Number}, X::MatNum; dims::Int = 1,
                         kwargs...)
mean(me::CustomValueExpectedReturns{<:VecNum}, X::MatNum; dims::Int = 1,
                         kwargs...)
mean(me::CustomValueExpectedReturns{<:Function}, X::MatNum; dims::Int = 1,
                         kwargs...)
```
