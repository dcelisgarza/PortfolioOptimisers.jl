# Schur Complement Hierarchical Risk Parity

```@docs
SchurComplementAlgorithm
NonMonotonicSchurComplement
MonotonicSchurComplement
SchurComplementParams
port_opt_view(sp::SchurComplementParams, i, X::MatNum, args...)
Sd_Var
naive_portfolio_risk(::Variance, sigma::MatNum)
symmetric_step_up_matrix(n1::Integer, n2::Integer)
schur_augmentation(A::MatNum, B::MatNum, C::MatNum, gamma::Number)
schur_complement_binary_search(objective::Function, lgamma::Number, hgamma::Number, lrisk::Number, tol::Number, iter::Option{<:Integer}, strict::Bool)
schur_complement_weights(pr::AbstractPriorResult, items::VecVecInt, wb::WeightBounds, params::SchurComplementParams{<:Any, <:Any, <:Any, <:NonMonotonicSchurComplement, <:Any}, gamma::Option{<:Number} = nothing)
schur_complement_weights(pr::AbstractPriorResult, items::VecVecInt, wb::WeightBounds, params::SchurComplementParams{<:Any, <:Any, <:Any, <:MonotonicSchurComplement, <:Any})
VecScP
ScP_VecScP
SchurComplementHierarchicalRiskParityResult
factory(res::SchurComplementHierarchicalRiskParityResult, fb::Option{<:OptE_Opt})
SchurComplementHierarchicalRiskParity
needs_previous_weights(opt::SchurComplementHierarchicalRiskParity)
port_opt_view(sh::SchurComplementHierarchicalRiskParity, i, X::MatNum, args...)
optimise(sh::SchurComplementHierarchicalRiskParity{<:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
```
