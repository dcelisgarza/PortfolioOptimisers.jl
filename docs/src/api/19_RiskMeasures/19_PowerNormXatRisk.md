# Power Norm X at Risk

```@docs
PowerNormValueatRisk
factory(r::PowerNormValueatRisk, pr::AbstractPriorResult, slv::Option{<:Slv_VecSlv} = nothing, args...; kwargs...)
factory(r::PowerNormValueatRisk, slv::Slv_VecSlv, pr::Option{<:AbstractPriorResult} = nothing)
PowerNormValueatRiskRange
factory(r::PowerNormValueatRiskRange, pr::AbstractPriorResult, slv::Option{<:Slv_VecSlv}, args...; kwargs...)
PowerNormDrawdownatRisk
factory(r::PowerNormDrawdownatRisk, pr::AbstractPriorResult, slv::Option{<:Slv_VecSlv} = nothing, args...; kwargs...)
factory(r::PowerNormDrawdownatRisk, slv::Slv_VecSlv, pr::Option{<:AbstractPriorResult} = nothing)
RelativePowerNormDrawdownatRisk
factory(r::RelativePowerNormDrawdownatRisk, pr::AbstractPriorResult, slv::Option{<:Slv_VecSlv} = nothing, args...; kwargs...)
factory(r::RelativePowerNormDrawdownatRisk, slv::Slv_VecSlv, pr::Option{<:AbstractPriorResult} = nothing)
PRM
```
