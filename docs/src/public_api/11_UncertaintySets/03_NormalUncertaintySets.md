```@meta
Description = "Normal Uncertainty Sets, public API of PortfolioOptimisers.jl: AbstractDiagonalConstruction, DiagonalOfShape, ShapeOfDiagonal, diagonal_sigma_shape, …"
```

# Normal Uncertainty Sets

```@docs
AbstractDiagonalConstruction
DiagonalOfShape
ShapeOfDiagonal
diagonal_sigma_shape
NormalUncertaintySet
ucs(ue::NormalUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
ucs(ue::NormalUncertaintySet{Nothing, <:BoxUncertaintySetAlgorithm, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::NormalUncertaintySet{Nothing, <:BoxUncertaintySetAlgorithm, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{Nothing, <:BoxUncertaintySetAlgorithm, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
ucs(ue::NormalUncertaintySet{Nothing, <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
ucs(ue::NormalUncertaintySet{Nothing, <:NormBallUncertaintySetAlgorithm{<:Any, <:Any, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::NormalUncertaintySet{Nothing, <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::NormalUncertaintySet{Nothing, <:NormBallUncertaintySetAlgorithm{<:Any, <:Any, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{Nothing, <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{Nothing, <:NormBallUncertaintySetAlgorithm{<:Any, <:Any, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
