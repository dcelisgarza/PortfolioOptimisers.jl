```@meta
Description = "Non-Optimisation Risk Measures, public API of PortfolioOptimisers.jl: MeanReturn, MeanReturnRiskRatio, ThirdCentralMoment, port_opt_view, factory."
```

# Non-Optimisation Risk Measures

```@docs
MeanReturn
MeanReturnRiskRatio
ThirdCentralMoment
port_opt_view(r::MeanReturn, ::Any, args...)
factory(r::MeanReturnRiskRatio, args...; kwargs...)
factory(r::MeanReturnRiskRatio, w::VecNum)
port_opt_view(r::ThirdCentralMoment, i, args...)
```
