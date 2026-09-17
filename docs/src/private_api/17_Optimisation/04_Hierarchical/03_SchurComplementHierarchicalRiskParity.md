```@meta
Description = "Schur Complement Hierarchical Risk Parity, private API of PortfolioOptimisers.jl: SchurComplementAlgorithm, Sd_Var, VecScP, ScP_VecScP, …"
```

# Schur Complement Hierarchical Risk Parity: private API

```@docs
SchurComplementAlgorithm
Sd_Var
VecScP
ScP_VecScP
naive_portfolio_risk(::Variance, sigma::MatNum)
symmetric_step_up_matrix(n1::Integer, n2::Integer)
schur_augmentation(A::MatNum, B::MatNum, C::MatNum, gamma::Number)
assert_schur_weights(w::Option{<:VecNum}, gamma::Number)
schur_complement_binary_search(objective::Function, lgamma::Number, hgamma::Number, lrisk::Number, tol::Number, iter::Option{<:Integer}, strict::Bool)
schur_complement_weights(pr::AbstractPriorResult, items::VecVecInt, wb::WeightBounds, params::SchurComplementParams{<:Any, <:Any, <:Any, <:NonMonotonicSchurComplement, <:Any}, gamma::Option{<:Number} = nothing)
schur_complement_weights(pr::AbstractPriorResult, items::VecVecInt, wb::WeightBounds, params::SchurComplementParams{<:Any, <:Any, <:Any, <:MonotonicSchurComplement, <:Any})
schur_complement_hrp_td_defaults
needs_previous_weights(opt::SchurComplementHierarchicalRiskParity)
```
