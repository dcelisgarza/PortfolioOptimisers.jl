```@meta
Description = "Near optimal centering (b), private API of PortfolioOptimisers.jl: assemble_near_optimal_centering_model!, solve_near_optimal_centering!, …"
```

# Near optimal centering (b): private API

```@docs
assemble_near_optimal_centering_model!(::UnconstrainedNearOptimalCentering, model::JuMP.Model, noc::NearOptimalCentering, setup::NearOptimalSetup, rd::ReturnsResult)
solve_near_optimal_centering!(::UnconstrainedNearOptimalCentering, model::JuMP.Model, noc::NearOptimalCentering, setup::NearOptimalSetup)
get_overall_retcode
compute_risk_ubs(model::JuMP.Model, noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:ConstrainedNearOptimalCentering}, pr::AbstractPriorResult, fees::Option{<:Fees}, w_min::VecNum, w_max::VecNum, args...)
```
