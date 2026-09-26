```@meta
Description = "Schur Complement Hierarchical Risk Parity, private API of PortfolioOptimisers.jl: Sd_Var, VecScP, ScP_VecScP, naive_portfolio_risk, …"
```

# Schur Complement Hierarchical Risk Parity: private API

```@docs
Sd_Var
VecScP
ScP_VecScP
naive_portfolio_risk(::Variance, sigma::MatNum)
symmetric_step_up_matrix(n1::Integer, n2::Integer)
schur_augmentation(A::MatNum, B::MatNum, C::MatNum, gamma::Number)
assert_schur_weights(w::Option{<:VecNum}, gamma::Number)
schur_complement_binary_search(objective::Function, lgamma::Number, hgamma::Number, lrisk::Number, lw::Option{<:VecNum}, tol::Number, iter::Option{<:Integer}, strict::Bool)
schur_complement_hrp_td_defaults
```
