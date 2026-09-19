"""
$(DocStringExtensions.TYPEDEF)

An expected-returns estimator whose mean is the `mu` of a Prior: the library-wide adapter that lets any `me` slot hold a prior estimator.

A Black–Litterman or shrunk mean may therefore drive a [`ForecastReversion`](@ref) step, an [`ExpectedReturn`](@ref) risk measure, or an [`EmpiricalPrior`](@ref). The adapter fits the prior on the returns it is handed and reads `mu` alone; the covariance and the scenarios the prior also computes are dropped. The expected-returns seam carries no factor returns anywhere in the library, so a prior that requires them is refused at construction; a *take what is given* prior is admitted and fitted without them.

The adapter folds exactly when its prior does under the library's rule ([`supports_partial_fit`](@ref)): [`partial_fit!`](@ref) forwards to the prior and `mean(me)` reads `prior(pe).mu`. An [`EmpiricalPrior`](@ref) carries its rows as memory rather than folding its moments alone, so under that rule it refits from the rows a host holds — which for the online portfolio selection head is the head's own buffer, held once.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PriorExpectedReturns(; pe::AbstractPriorEstimator = EmpiricalPrior()) -> PriorExpectedReturns

Keywords correspond to the struct's fields.

## Validation

  - `needs_factor_returns(pe) !== true`. An `ArgumentError` is thrown otherwise.

## View parameters

When [`port_opt_view`](@ref) is called on this type, `pe` is viewed through its own [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> PriorExpectedReturns()
PriorExpectedReturns
  pe ┼ EmpiricalPrior
     │           ce ┼ PortfolioOptimisersCovariance
     │              │   ce ┼ Covariance
     │              │      │    me ┼ SimpleExpectedReturns
     │              │      │       │   w ┴ nothing
     │              │      │    ce ┼ GeneralCovariance
     │              │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     │              │      │       │    w ┴ nothing
     │              │      │   alg ┼ FullMoment()
     │              │      │     w ┴ nothing
     │              │   mp ┼ MatrixProcessing
     │              │      │     pdm ┼ Posdef
     │              │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │              │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │              │      │      dn ┼ nothing
     │              │      │      dt ┼ nothing
     │              │      │     alg ┼ nothing
     │              │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
     │           me ┼ SimpleExpectedReturns
     │              │   w ┴ nothing
     │      horizon ┼ nothing
     │   fill_limit ┴ nothing
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`AbstractPriorEstimator`](@ref)
  - [`needs_factor_returns`](@ref)
  - [`PriceLevelExpectedReturns`](@ref)
  - [`ForecastReversion`](@ref)
"""
struct PriorExpectedReturns{T1 <: AbstractPriorEstimator} <:
       AbstractExpectedReturnsEstimator
    """
    The prior estimator whose `mu` is the expected return.
    """
    pe::T1
    function PriorExpectedReturns(pe::AbstractPriorEstimator)
        @argcheck(needs_factor_returns(pe) !== true,
                  ArgumentError("`$(typeof(pe).name.name)` requires factor returns, and an expected-returns estimator is fitted on returns alone: no `me` slot in the library carries a factor matrix. Hand the adapter a prior that fits on returns, or one whose factor argument is optional."))
        return new{typeof(pe)}(pe)
    end
end
function PriorExpectedReturns(;
                              pe::AbstractPriorEstimator = EmpiricalPrior())::PriorExpectedReturns
    return PriorExpectedReturns(pe)
end
function port_opt_view(me::PriorExpectedReturns, i, args...)
    return PriorExpectedReturns(; pe = port_opt_view(me.pe, i, args...))
end
"""
    Statistics.mean(me::PriorExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
    Statistics.mean(me::PriorExpectedReturns, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    Statistics.mean(me::PriorExpectedReturns; kwargs...)

Fits the prior on `X` (and the Asset Panel where one is given) with no factor returns, and answers its `mu` shaped `(1, N)` if `dims == 1` or `(N, 1)` if `dims == 2`; the no-data form reads the `mu` of a folded prior.

# Arguments

  - `me`: The adapter.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `kwargs...`: Forwarded to [`prior`](@ref).

# Validation

  - $(val_dict[:dims])

# Returns

  - `mu::Matrix{<:Number}`: The prior's expected return.

# Related

  - [`PriorExpectedReturns`](@ref)
  - [`prior`](@ref)
"""
function Statistics.mean(me::PriorExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
    assert_dims(dims)
    mu = prior(me.pe, X, nothing, nothing; dims = dims, kwargs...).mu
    return dims == 1 ? reshape(mu, 1, :) : reshape(mu, :, 1)
end
function Statistics.mean(me::PriorExpectedReturns, X::MatNum, pnl::Option{<:AssetPanel};
                         dims::Int = 1, kwargs...)
    assert_dims(dims)
    mu = prior(me.pe, X, nothing, pnl; dims = dims, kwargs...).mu
    return dims == 1 ? reshape(mu, 1, :) : reshape(mu, :, 1)
end
function Statistics.mean(me::PriorExpectedReturns; kwargs...)
    return reshape(prior(me.pe; kwargs...).mu, 1, :)
end
"""
    partial_fit!(me::PriorExpectedReturns, X::VecNum_MatNum; kwargs...)

Folds the rows into the adapter's prior through the prior's own [`partial_fit!`](@ref), and returns the adapter carrying the folded prior.

# Related

  - [`PriorExpectedReturns`](@ref)
  - [`supports_partial_fit`](@ref)
"""
function partial_fit!(me::PriorExpectedReturns, X::VecNum_MatNum; kwargs...)
    return PriorExpectedReturns(; pe = partial_fit!(me.pe, X; kwargs...))
end
# The adapter folds exactly when its prior does (see [`supports_partial_fit`](@ref)).
function supports_partial_fit(me::PriorExpectedReturns)
    return supports_partial_fit(me.pe)
end
export PriorExpectedReturns
