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
    Statistics.mean(me::EquilibriumExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)

Compute equilibrium expected returns from a covariance estimator, weights, and risk aversion.

This method computes equilibrium expected returns as `λ * Σ * w`, where `λ` is the risk aversion parameter, `Σ` is the covariance matrix, and `w` are the equilibrium weights. If `w` is not provided in the estimator, equal weights are used.

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
"""
function Statistics.mean(me::EquilibriumExpectedReturns, X::MatNum; dims::Int = 1,
                         kwargs...)
    sigma = Statistics.cov(me.ce, X; dims = dims, kwargs...)
    w = !isnothing(me.w) ? me.w : fill(inv(size(sigma, 1)), size(sigma, 1))
    return me.l * sigma * w
end

export EquilibriumExpectedReturns
