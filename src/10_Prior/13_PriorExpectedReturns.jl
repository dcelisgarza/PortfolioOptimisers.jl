"""
$(DocStringExtensions.TYPEDEF)

Reads the mean of a Prior as an expected-returns vector, so that any `me` slot can hold a prior estimator.

The adapter lets the posterior mean of a [`BlackLittermanPrior`](@ref) drive a [`ForecastReversion`](@ref) step, an [`ExpectedReturn`](@ref) risk measure or the `me` field of an [`EmpiricalPrior`](@ref). The adapter fits the prior on the returns it receives and keeps `mu` alone. It discards the covariance and the scenarios of the fit. No `me` slot in the library passes factor returns, so the constructor refuses a prior that requires them. A prior whose factor argument is optional is admitted, and the adapter fits it with no factor returns.

[`supports_partial_fit`](@ref) gives for the adapter the answer it gives for the prior. [`partial_fit!`](@ref) forwards the rows to the prior, and `mean(me)` reads the `mu` of the folded prior. An [`EmpiricalPrior`](@ref) keeps its rows, so `supports_partial_fit` is `false` for it, and a host refits it from the rows the host holds. For the online portfolio selection head, those rows are the head's own buffer. [`rows_needed`](@ref) is `nothing` for an adapter that does not fold, so the head keeps every row, and each step costs one prior fit over all the rows so far. To bound the rows, wrap the adapter in a [`WindowedExpectedReturns`](@ref).

The adapter fits the whole prior, so a covariance that the prior cannot form makes the mean throw, although `mu` alone is defined. Under the default [`EmpiricalPrior`](@ref), a column of constant returns, such as a suspended asset in the head's buffer, makes the positive-definite repair of the covariance throw an `ArgumentError`. To fit such rows, give the prior a covariance without that repair, `PortfolioOptimisersCovariance(; mp = MatrixProcessing(; pdm = nothing))`.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\mu} &= \\mathbb{E}_{\\mathcal{P}(\\mathbf{X})}\\left[\\boldsymbol{x}\\right]\\,.
\\end{align}
```

Where:

  - $(math_dict[:mu_er])
  - ``\\mathcal{P}(\\mathbf{X})``: The Prior fitted on ``\\mathbf{X}``, a distribution of the asset returns.
  - ``\\mathbf{X}``: Returns matrix ``T \\times N``.
  - ``\\boldsymbol{x}``: Asset returns vector ``N \\times 1``, distributed as ``\\mathcal{P}(\\mathbf{X})``.
  - $(math_dict[:x_t_obs])
  - $(math_dict[:T])
  - $(math_dict[:N])

Under the default [`EmpiricalPrior`](@ref), ``\\boldsymbol{\\mu} = \\frac{1}{T} \\sum_{t=1}^{T} \\boldsymbol{x}_t`` is the sample mean.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PriorExpectedReturns(; pe::AbstractPriorEstimator = EmpiricalPrior()) -> PriorExpectedReturns

Keywords correspond to the struct's fields.

## Validation

  - `needs_factor_returns(pe) !== true`. An `ArgumentError` is thrown otherwise.

## View parameters

`PriorExpectedReturns` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - `pe` recurses through [`port_opt_view`](@ref), with the same arguments.

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

Fits the prior on `X` with no factor returns and returns the prior's `mu` as a matrix.

The panel form passes the Asset Panel to the prior. The no-data form fits nothing, and reads the `mu` of a prior that [`partial_fit!`](@ref) folded.

# Algorithm

 1. Check `dims`.
 2. Fit the prior on `X` with [`prior`](@ref), giving its Result. The factor returns are `nothing`, and the panel form passes `pnl`.
 3. Read `mu`, the mean vector of the Result.
 4. Reshape `mu` to `(1, N)` when `dims == 1`, and to `(N, 1)` when `dims == 2`.

The no-data form reads the Result of `prior(me.pe)`, the read-out of the folded prior, in place of steps 1 and 2, and always reshapes `mu` to `(1, N)`.

# Arguments

  - `me`: The adapter.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `kwargs...`: Forwarded to [`prior`](@ref).

# Validation

  - $(val_dict[:dims])
  - The no-data form: the prior holds a partial-fit state. The prior's read-out throws an `ArgumentError` otherwise.
  - Everything the prior's fit refuses.

# Returns

  - `mu::Matrix{<:Number}`: The prior's expected return, shaped `(1, N)` when `dims == 1` and `(N, 1)` when `dims == 2`.

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

Folds the rows into the adapter's prior through the prior's own [`partial_fit!`](@ref), and returns an adapter that holds the folded prior.

A direct call folds every prior whose own fold accepts the rows. An [`EmpiricalPrior`](@ref) is one of them, although [`supports_partial_fit`](@ref) is `false` for it. For the default [`EmpiricalPrior`](@ref), `mean(me)` then equals the batch mean over every row folded so far.

# Arguments

  - `me`: The adapter.
  - `X`: The rows to fold, one row as a vector or a block of rows as a matrix.
  - `kwargs...`: Forwarded to the prior's [`partial_fit!`](@ref).

# Validation

  - Everything the prior's own fold refuses.

# Returns

  - `me::PriorExpectedReturns`: The adapter over the prior that the fold returns.

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
