"""
$(DocStringExtensions.TYPEDEF)

Container type for equilibrium expected returns estimators.

`EquilibriumExpectedReturns` encapsulates the covariance estimator, equilibrium weights, and risk aversion parameter for computing equilibrium expected returns (e.g., as in Black-Litterman).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EquilibriumExpectedReturns(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        w::Option{<:VecNum} = nothing,
        l::Number = 1
    ) -> EquilibriumExpectedReturns

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ce`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `ce`: Recursively viewed via [`port_opt_view`](@ref).
  - `w`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> EquilibriumExpectedReturns()
EquilibriumExpectedReturns
  ce ┼ PortfolioOptimisersCovariance
     │   ce ┼ Covariance
     │      │    me ┼ SimpleExpectedReturns
     │      │       │   w ┴ nothing
     │      │    ce ┼ GeneralCovariance
     │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     │      │       │    w ┴ nothing
     │      │   alg ┴ FullMoment()
     │   mp ┼ MatrixProcessing
     │      │     pdm ┼ Posdef
     │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │      │      dn ┼ nothing
     │      │      dt ┼ nothing
     │      │     alg ┼ nothing
     │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
   w ┼ nothing
   l ┴ Int64: 1
```

# Related

  - [`AbstractShrunkExpectedReturnsEstimator`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct EquilibriumExpectedReturns <:
                               AbstractShrunkExpectedReturnsEstimator
    """
    $(field_dict[:ce])
    """
    @fprop @vprop ce
    """
    $(field_dict[:eqw])
    """
    @vprop w
    """
    $(field_dict[:l])
    """
    l
    function EquilibriumExpectedReturns(ce::StatsBase.CovarianceEstimator,
                                        w::Option{<:VecNum}, l::Number)
        assert_nonempty_finite_val(w, :w)
        return new{typeof(ce), typeof(w), typeof(l)}(ce, w, l)
    end
end
function EquilibriumExpectedReturns(;
                                    ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                    w::Option{<:VecNum} = nothing,
                                    l::Number = 1)::EquilibriumExpectedReturns
    return EquilibriumExpectedReturns(ce, w, l)
end
"""
    equilibrium_mu(l::Number, sigma::MatNum, w::Option{<:VecNum})

Compute equilibrium expected returns from a risk aversion parameter, a covariance block, and equilibrium weights.

`equilibrium_mu` is the **single owner** of the ``\\lambda \\mathbf{\\Sigma} \\boldsymbol{w}`` expression and of its equal-weight fallback. [`EquilibriumExpectedReturns`](@ref), [`FactorBlackLittermanPrior`](@ref) and [`AugmentedBlackLittermanPrior`](@ref) all reach it, so the fallback and the length check are stated once.

The result is an **excess** return. Reverse optimisation implies a risk premium, so no risk-free rate is in it and none is taken off it. This is why the Black-Litterman members apply [`remove_rf`](@ref) only on the branch where they do *not* call this function.

`sigma` is a covariance **block**, not necessarily a square covariance matrix. Its columns are the assets the weights are written over, so `size(sigma, 2)` is the length `w` must have. A square covariance gives the plain equilibrium returns. A rectangular block gives the equilibrium returns of the rows it spans, which is how the factor Black-Litterman members build a prior mean over factors from asset weights.

# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}}_{\\text{eq}} &= \\lambda \\, \\mathbf{\\Sigma} \\, \\boldsymbol{w}\\,.
\\end{align}
```

Where:

  - ``\\lambda``: Risk aversion parameter.
  - ``\\mathbf{\\Sigma}``: ``M \\times N`` covariance block.
  - ``\\boldsymbol{w}``: ``N \\times 1`` equilibrium portfolio weights.

# Arguments

  - `l`: Risk aversion parameter.
  - `sigma`: Covariance block whose columns are the assets.
  - `w`: Equilibrium weights, or `nothing` for equal weights.

# Validation

  - If `w` is a vector, `length(w) == size(sigma, 2)`.

# Returns

  - `mu::VecNum`: Equilibrium expected returns vector of length `size(sigma, 1)`.

# Related

  - [`EquilibriumExpectedReturns`](@ref)
  - [`FactorBlackLittermanPrior`](@ref)
  - [`AugmentedBlackLittermanPrior`](@ref)
"""
function equilibrium_mu(l::Number, sigma::MatNum, w::Nothing)
    N = size(sigma, 2)
    return l * sigma * fill(inv(N), N)
end
function equilibrium_mu(l::Number, sigma::MatNum, w::VecNum)
    @argcheck(length(w) == size(sigma, 2),
              DimensionMismatch("length(w) ($(length(w))) must match the number of assets, size(sigma, 2) ($(size(sigma, 2)))"))
    return l * sigma * w
end
"""
    Statistics.mean(me::EquilibriumExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)

Compute equilibrium expected returns from a covariance estimator, weights, and risk aversion.

This method computes equilibrium expected returns as `λ * Σ * w`, where `λ` is the risk aversion parameter, `Σ` is the covariance matrix, and `w` are the equilibrium weights. If `w` is not provided in the estimator, equal weights are used. The expression and the fallback belong to [`equilibrium_mu`](@ref).

# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}}_{\\text{eq}} &= \\lambda \\, \\hat{\\mathbf{\\Sigma}} \\, \\boldsymbol{w}\\,.
\\end{align}
```

Where:

  - ``\\lambda``: Risk aversion parameter (`me.l`).
  - ``\\hat{\\mathbf{\\Sigma}}``: `N × N` covariance matrix estimated from the data.
  - ``\\boldsymbol{w}``: `N × 1` equilibrium portfolio weights (equal weights if not provided).

# Arguments

  - `me`: Equilibrium expected returns estimator.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the covariance estimator.

# Returns

  - `mu::ArrNum`: Equilibrium expected returns vector.

# Related

  - [`EquilibriumExpectedReturns`](@ref)
  - [`equilibrium_mu`](@ref)
"""
function Statistics.mean(me::EquilibriumExpectedReturns, X::MatNum; dims::Int = 1,
                         kwargs...)
    sigma = Statistics.cov(me.ce, X; dims = dims, kwargs...)
    return equilibrium_mu(me.l, sigma, me.w)
end

export EquilibriumExpectedReturns
