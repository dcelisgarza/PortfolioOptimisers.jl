# Entropic X at Risk

```@docs
EntropicValueatRisk
factory(r::EntropicValueatRisk, pr::AbstractPriorResult, slv::Option{<:Slv_VecSlv} = nothing, args...; kwargs...)
factory(r::EntropicValueatRisk, slv::Slv_VecSlv, pr::Option{<:AbstractPriorResult} = nothing, args...; kwargs...)
EntropicValueatRiskRange
factory(r::EntropicValueatRiskRange, pr::AbstractPriorResult, slv::Option{<:Slv_VecSlv}, args...; kwargs...)
EntropicDrawdownatRisk
factory(r::EntropicDrawdownatRisk, pr::AbstractPriorResult, slv::Option{<:Slv_VecSlv} = nothing, args...; kwargs...)
factory(r::EntropicDrawdownatRisk, slv::Slv_VecSlv, pr::Option{<:AbstractPriorResult} = nothing, args...; kwargs...)
RelativeEntropicDrawdownatRisk
factory(r::RelativeEntropicDrawdownatRisk, pr::AbstractPriorResult, slv::Option{<:Slv_VecSlv} = nothing, args...; kwargs...)
factory(r::RelativeEntropicDrawdownatRisk, slv::Slv_VecSlv, pr::Option{<:AbstractPriorResult} = nothing, args...; kwargs...)
ERM
```
