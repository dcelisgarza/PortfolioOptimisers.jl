```@meta
Description = "Norm-Ball Uncertainty Sets, private API of PortfolioOptimisers.jl: assert_norm_ball_axis, assert_norm_ball_val, dual_norm_order, norm_ball_factor, …"
```

# Norm-Ball Uncertainty Sets: private API

```@docs
assert_norm_ball_axis
assert_norm_ball_val
dual_norm_order
norm_ball_factor
norm_ball_deviation_factor
k_norm_ball
norm_ball_set
norm_ball_deviation_set
expand_investable_ucs(set::NormBallUncertaintySet{<:Any, <:MatNum, <:Any, <:MuUncertaintySetClass}, imsk::BitVector, pr::AbstractPriorResult)
expand_investable_ucs(set::NormBallUncertaintySet{<:Any, <:MatNum, <:Any, <:SigmaUncertaintySetClass}, imsk::BitVector, pr::AbstractPriorResult)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
