"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all prior estimators.

A prior estimator computes a prior result from asset returns, from factor returns, or from both. Every concrete prior estimator subtypes one of the families below.

# Interfaces

To implement a new prior estimator, subtype the family that names the returns it reads, and hold every parameter in a field of the struct. The families are [`AbstractLowOrderPriorEstimator_A`](@ref), [`AbstractLowOrderPriorEstimator_F`](@ref) and [`AbstractLowOrderPriorEstimator_AF`](@ref) for a low order estimator. They are [`AbstractHighOrderPriorEstimator_F`](@ref) and [`AbstractHighOrderPriorEstimator`](@ref) for a high order estimator. Then implement this method:

  - `prior(pe::AbstractPriorEstimator, X::MatNum, F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, kwargs...) -> AbstractPriorResult`: Estimate the prior from the returns matrices and the Asset Panel.

The family sets the signature. A member of the `_A` family declares `F` as `args...` and does not read it. A member of a `_F` family declares `F::MatNum` and requires it. A member of the `_AF` family, or of the high order root, declares `F::Option{<:MatNum} = nothing` and reads `F` when the caller gives it.

`pnl` is the Asset Panel of the [`ReturnsResult`](@ref), and the `ReturnsResult` method gives it to every estimator. An estimator takes it and ignores it, unless the estimator fits on a panel, as [`CrossSectionalFactorPrior`](@ref) does. An estimator that wraps another over the assets passes `pnl` to it unchanged, so that the wrapped estimator can also fit on the panel. An estimator that wraps a prior over the factors does not pass it, because no panel describes a factor axis. An estimator that declares `args...` takes `pnl` there, and needs no other declaration.

The method returns the prior result of its own order. A low order estimator returns a [`LowOrderPrior`](@ref), and a high order estimator returns a [`HighOrderPrior`](@ref). An estimator that wraps another rebuilds the wrapped result with [`forward_prior`](@ref), not with a hand-written constructor call. Every field that the wrapper does not name then keeps its value.

This file defines the [`ReturnsResult`](@ref) method of [`prior`](@ref), so a new estimator does not implement it.

## Arguments

  - $(arg_dict[:pe])
  - $(arg_dict[:X])
  - `F`: Factor returns matrix, or `nothing`.
  - $(arg_dict[:pnl_prior])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the nested estimators.

## Returns

  - `pr::AbstractPriorResult`: Result object containing the estimated prior.

## `carry_folds` (optional)

  - `carry_folds(pe::MyPriorEstimator) -> Bool`: Returns `true` when the carry fold of a [`CrossSectionalFactorPrior`](@ref) folds the prior as its factor prior, one row at a time.

A prior that returns `true` also implements two methods. `partial_fit!(pe, f::MatNum) -> pe` folds factor returns into a state that the prior holds, and `prior(pe; strict::Bool = false) -> LowOrderPrior` reads that state. The fallback returns `false`, which is correct for every prior. The cost of `false` is on the carry fold. [`FoldOrRefit`](@ref) fits the factor prior again over every carried factor return at each step, and [`FoldOnly`](@ref) refuses the prior.

## `returns_source` (optional)

  - `returns_source(pe::MyPriorEstimator) -> Option{<:AbstractReturnsSource}`: Returns the source of the returns that the prior fits on. This is usually a field `x_src::Option{<:AbstractReturnsSource}`, which the keyword constructor checks with `PortfolioOptimisers.assert_returns_source(x_src, PortfolioOptimisers.DATA_RETURNS_SOURCES, :MyPriorEstimator)`.

The [`ReturnsResult`](@ref) method of [`prior`](@ref) reads the source through this method, and the frame walk of an optimiser also reads it. The fallback returns `nothing`, so a prior with no method reads the raw returns `rd.X`. A prior that holds another reader in a field refuses a source on that reader, with `PortfolioOptimisers.assert_inner_reader` in its keyword constructor. The outer prior picks the returns.

## `keeps_observations` (optional)

  - `keeps_observations(pe::MyPriorEstimator) -> Bool`: Returns `true` when the result carries one scenario for each observation that the prior receives, for every configuration and all data.

A prior that reweights observations, such as [`EntropyPoolingPrior`](@ref), reads this method on the prior that it wraps. It reads the count of the scenarios of the wrapped prior before it fits that prior under the prior probabilities of a caller. When the wrapped prior returns `true`, the count needs no fit. Otherwise the count runs only the steps that drop rows. The wrapped prior fits once more only when no method of `PortfolioOptimisers.scenario_count` gives the count. The fallback returns `false`, which is correct for every prior and costs at most that one fit. A prior that drops rows, such as one that holds a Scenario Cap, keeps the fallback. A prior that wraps another returns the answer of the prior that it wraps.

# Examples

This example makes a minimal prior estimator:

```jldoctest
julia> struct MyPriorEstimator <: PortfolioOptimisers.AbstractLowOrderPriorEstimator_A end

julia> function PortfolioOptimisers.prior(pe::MyPriorEstimator, X::PortfolioOptimisers.MatNum,
                                          args...; dims::Int = 1, kwargs...)
           mu = vec(sum(X; dims = 1)) / size(X, 1)
           sigma = Matrix(LinearAlgebra.I * 1.0, size(X, 2), size(X, 2))
           return LowOrderPrior(; X = X, mu = mu, sigma = sigma)
       end

julia> prior(MyPriorEstimator(), [0.01 0.02; 0.03 0.04])
LowOrderPrior
        X ┼ 2×2 Matrix{Float64}
      o_X ┼ nothing
       mu ┼ Vector{Float64}: [0.02, 0.03]
    sigma ┼ 2×2 Matrix{Float64}
     chol ┼ nothing
        w ┼ nothing
      ens ┼ nothing
      eoc ┼ Tuple{Nothing, Nothing}: (nothing, nothing)
      kld ┼ nothing
       ow ┼ nothing
       rr ┼ nothing
      fpr ┼ nothing
    x_src ┼ nothing
  ss_gaps ┼ nothing
       pc ┴ nothing
```

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractHighOrderPriorEstimator`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`prior`](@ref)
  - [`forward_prior`](@ref)
  - [`carry_folds`](@ref)
"""
abstract type AbstractPriorEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for low order prior estimators.

A low order prior estimator computes the mean and the covariance from asset returns, from factor returns, or from both. A member of this family returns a [`LowOrderPrior`](@ref), not a tuple of moments, so every consumer reads one result type. A member does not subtype this type directly. It subtypes the one of [`AbstractLowOrderPriorEstimator_A`](@ref), [`AbstractLowOrderPriorEstimator_F`](@ref) and [`AbstractLowOrderPriorEstimator_AF`](@ref) that names the returns it reads.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`LowOrderPrior`](@ref)
"""
abstract type AbstractLowOrderPriorEstimator <: AbstractPriorEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Low order prior estimator using only asset returns.

A member of this family computes the mean and the covariance from asset returns alone.

This is the first of the three source shapes. A member admits asset returns only. Its `prior` method declares the factor argument as `args...` and does not read it, so the method ignores factor returns and does not refuse them.

# Interfaces

  - `prior(pe::AbstractLowOrderPriorEstimator_A, X::MatNum, args...; dims::Int = 1, kwargs...) -> LowOrderPrior`: Estimate the prior from asset returns alone. `args...` takes the factor returns, and the method ignores them.

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`prior`](@ref)
"""
abstract type AbstractLowOrderPriorEstimator_A <: AbstractLowOrderPriorEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Low order prior estimator using factor returns.

A member of this family computes the mean and the covariance from asset returns and factor returns, and it requires both.

This is the second of the three source shapes. A member admits asset returns and requires factor returns. Its `prior` method declares the factor argument as `F::MatNum` with no default, so a call with no factor returns raises a `MethodError`. When a [`ReturnsResult`](@ref) with `F === nothing` reaches such an estimator, [`prior`](@ref) raises a named error before that call.

# Interfaces

  - `prior(pe::AbstractLowOrderPriorEstimator_F, X::MatNum, F::MatNum, args...; dims::Int = 1, kwargs...) -> LowOrderPrior`: Estimate the prior from asset and factor returns. A call with no `F` raises a `MethodError`.

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`AbstractHiLoOrderPriorEstimator_F`](@ref)
  - [`prior`](@ref)
"""
abstract type AbstractLowOrderPriorEstimator_F <: AbstractLowOrderPriorEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Low order prior estimator using both asset and factor returns.

A member of this family computes the mean and the covariance from asset returns, and it reads factor returns when the caller gives them.

This is the third of the three source shapes. A member admits asset returns, and admits factor returns as an option. Its `prior` method declares the factor argument as `F::Option{<:MatNum} = nothing`, and reads `F` when the caller gives it. The shape does not tell whether the result carries a regression. Use [`assert_prior_regression`](@ref) to find that.

The shape also does not tell whether the fit reads factor returns. [`needs_factor_returns`](@ref) gives one of three answers. A member that requires them answers `true`, and a member that does not read them answers `false`. This shape answers `nothing` by default, which means that the type does not decide and the fold takes the data it receives. The batch verb does the same. Every member of this shape in the library holds another prior and gives `F` to it, so each member defines the recursion and answers with the value of the innermost prior.

A new subtype that holds a prior must define the same recursion. A new subtype that reads `F` itself can keep the default.

# Interfaces

  - `prior(pe::AbstractLowOrderPriorEstimator_AF, X::MatNum, F::Option{<:MatNum} = nothing, args...; dims::Int = 1, kwargs...) -> LowOrderPrior`: Estimate the prior from asset returns, and read `F` when the caller gives it.

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`assert_prior_regression`](@ref)
  - [`needs_factor_returns`](@ref)
  - [`prior`](@ref)
"""
abstract type AbstractLowOrderPriorEstimator_AF <: AbstractLowOrderPriorEstimator end
"""
    const AbstractLowOrderPriorEstimator_A_AF = Union{AbstractLowOrderPriorEstimator_A,
                                                      AbstractLowOrderPriorEstimator_AF}

Union type for asset-only and asset-and-factor low order prior estimators.

A field typed `AbstractLowOrderPriorEstimator_A_AF` admits the asset-only shape and the optional-factor shape, and excludes the shape that requires factor returns. A wrapper uses this bound when it fits its nested estimator on one returns matrix that it gives itself. [`FactorPrior`](@ref) and [`FactorBlackLittermanPrior`](@ref) fit `pe` on the factor returns alone. [`AugmentedBlackLittermanPrior`](@ref) fits `a_pe` on the assets and `f_pe` on the factors. The wrapper has no second matrix to give, so the bound refuses a nested estimator that requires one. The refusal happens at construction, not at the call.

# Related

  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_F_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_F_AF`](@ref)
  - [`FactorPrior`](@ref)
  - [`FactorBlackLittermanPrior`](@ref)
  - [`AugmentedBlackLittermanPrior`](@ref)
"""
const AbstractLowOrderPriorEstimator_A_AF = Union{AbstractLowOrderPriorEstimator_A,
                                                  AbstractLowOrderPriorEstimator_AF}
"""
    const AbstractLowOrderPriorEstimator_F_AF = Union{AbstractLowOrderPriorEstimator_F,
                                                      AbstractLowOrderPriorEstimator_AF}

Union type for factor-only and asset-and-factor low order prior estimators.

A field typed `AbstractLowOrderPriorEstimator_F_AF` admits the factor-requiring shape and the optional-factor shape, and excludes the asset-only shape. A wrapper uses this bound when it passes both returns matrices to its nested estimator, and needs a result that can carry a factor block. [`HighOrderFactorPriorEstimator`](@ref) is such a wrapper. The bound limits the returns that the nested estimator reads. It does not limit the result that it returns. A member of the optional-factor shape can return a result with `rr === nothing`, so a consumer that reads the loadings checks them first with [`assert_prior_regression`](@ref).

# Related

  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_F_AF`](@ref)
  - [`HighOrderFactorPriorEstimator`](@ref)
  - [`assert_prior_regression`](@ref)
"""
const AbstractLowOrderPriorEstimator_F_AF = Union{AbstractLowOrderPriorEstimator_F,
                                                  AbstractLowOrderPriorEstimator_AF}
"""
    const AbstractLowOrderPriorEstimator_A_F_AF = Union{AbstractLowOrderPriorEstimator_A,
                                                        AbstractLowOrderPriorEstimator_F,
                                                        AbstractLowOrderPriorEstimator_AF}

Union type for asset-only, factor-only, and asset-and-factor low order prior estimators.

A field typed `AbstractLowOrderPriorEstimator_A_F_AF` admits all three source shapes, and excludes no subtype of the low order root. A wrapper uses this bound when it passes the returns matrices of the caller to its nested estimator unchanged. The nested estimator then receives the arguments of the caller, and its shape decides what it reads. [`EntropyPoolingPrior`](@ref) bounds `pe` this way. The union names the three shapes in place of [`AbstractLowOrderPriorEstimator`](@ref) for two reasons. Each field that admits the shapes names them, and a fourth shape joins this bound only when someone edits the union.

# Related

  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_F_AF`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
const AbstractLowOrderPriorEstimator_A_F_AF = Union{AbstractLowOrderPriorEstimator_A,
                                                    AbstractLowOrderPriorEstimator_F,
                                                    AbstractLowOrderPriorEstimator_AF}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for high order prior estimators.

A high order prior estimator computes the coskewness and the cokurtosis from asset returns, from factor returns, or from both. A member of this family returns a [`HighOrderPrior`](@ref). That result wraps the [`LowOrderPrior`](@ref) of the nested low order estimator of the member. A high order estimator adds an order and does not replace one, and the wrapper answers every low order name.

A member that requires factor returns subtypes [`AbstractHighOrderPriorEstimator_F`](@ref). A member that reads factor returns only when the caller gives them subtypes this type directly, as [`HighOrderPriorEstimator`](@ref) does.

# Interfaces

  - `prior(pe::AbstractHighOrderPriorEstimator, X::MatNum, F::Option{<:MatNum} = nothing, args...; dims::Int = 1, kwargs...) -> HighOrderPrior`: Estimate the prior from asset returns, and read `F` when the caller gives it.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractHighOrderPriorEstimator_F`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`prior`](@ref)
"""
abstract type AbstractHighOrderPriorEstimator <: AbstractPriorEstimator end
"""
$(DocStringExtensions.TYPEDEF)

High order prior estimator using factor returns.

A member of this family computes the coskewness and the cokurtosis from asset returns and factor returns, and it requires both.

A member admits asset returns and requires factor returns, as a member of [`AbstractLowOrderPriorEstimator_F`](@ref) does at the low order. Its `prior` method declares the factor argument as `F::MatNum` with no default. The two families are the members of [`AbstractHiLoOrderPriorEstimator_F`](@ref). That union lets [`prior`](@ref) find a factor prior of either order.

# Interfaces

  - `prior(pe::AbstractHighOrderPriorEstimator_F, X::MatNum, F::MatNum, args...; dims::Int = 1, kwargs...) -> HighOrderPrior`: Estimate the prior from asset and factor returns. A call with no `F` raises a `MethodError`.

# Related

  - [`AbstractHighOrderPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractHiLoOrderPriorEstimator_F`](@ref)
  - [`prior`](@ref)
"""
abstract type AbstractHighOrderPriorEstimator_F <: AbstractHighOrderPriorEstimator end
"""
    const AbstractHiLoOrderPriorEstimator_F = Union{AbstractLowOrderPriorEstimator_F,
                                                    AbstractHighOrderPriorEstimator_F}

Groups the two families that require factor returns, one for each order.

A member of `AbstractHiLoOrderPriorEstimator_F` cannot fit without factor returns, at either order, and it answers `true` to [`needs_factor_returns`](@ref). The checks at entry that look for a missing factor matrix call that predicate, not an `isa` test on this union. An estimator with an optional factor argument can hold a member of this union. The predicate reads the whole tree, and an `isa` test reads only the outer estimator.

# Related

  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractHighOrderPriorEstimator_F`](@ref)
  - [`needs_factor_returns`](@ref)
  - [`prior`](@ref)
  - [`ReturnsResult`](@ref)
"""
const AbstractHiLoOrderPriorEstimator_F = Union{AbstractLowOrderPriorEstimator_F,
                                                AbstractHighOrderPriorEstimator_F}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all prior result types.

A prior estimator returns a prior result. The result holds the moments, the asset returns and the factor returns that the estimator computed.

The library has two prior result types. [`LowOrderPrior`](@ref) holds the returns, the mean and the covariance. [`HighOrderPrior`](@ref) holds the co-moments, and wraps a [`LowOrderPrior`](@ref).

# Interfaces

To implement a new prior result type, subtype `AbstractPriorResult`, hold every value in a field of the struct, and implement these methods:

  - `reconstruct_prior(pr::AbstractPriorResult, patch::NamedTuple) -> AbstractPriorResult`: Rebuild `pr` through its own constructor with `patch` applied. [`forward_prior`](@ref) calls it. Each result type writes its own method, because the method names the constructor and does not find it by reflection.
  - `port_opt_view(pr::AbstractPriorResult, i, args...) -> AbstractPriorResult`: Restrict `pr` to the assets at index `i`, for hierarchical and subset optimisation.

`prior_field_values` reads the field list from the type, so a result type that gains a field needs no other method. An `@pprop` field can name only a field of [`LowOrderPrior`](@ref) or of [`HighOrderPrior`](@ref), because `prior_result_property_pool` writes those two types by name. `check_propagatable_contracts` refuses a name that only a new result type has.

## Arguments

  - $(arg_dict[:pr])
  - `patch`: Named tuple of field overrides.
  - `i`: Asset indices the view keeps.
  - `args...`: Additional arguments the view reads.

## Returns

  - `pr::AbstractPriorResult`: A prior result of the same type as the input.

# Examples

This example makes a minimal prior result type:

```jldoctest
julia> struct MyPriorResult <: PortfolioOptimisers.AbstractPriorResult
           X::Matrix{Float64}
           mu::Vector{Float64}
       end

julia> function PortfolioOptimisers.reconstruct_prior(pr::MyPriorResult, patch::NamedTuple)
           vals = merge(PortfolioOptimisers.prior_field_values(pr), patch)
           return MyPriorResult(vals.X, vals.mu)
       end

julia> function PortfolioOptimisers.port_opt_view(pr::MyPriorResult, i, args...)
           return MyPriorResult(pr.X[:, i], pr.mu[i])
       end

julia> pr = MyPriorResult([0.01 0.02; 0.03 0.04], [0.02, 0.03]);

julia> PortfolioOptimisers.forward_prior(pr; mu = [0.05, 0.06]).mu
2-element Vector{Float64}:
 0.05
 0.06

julia> PortfolioOptimisers.port_opt_view(pr, [1]).mu
1-element Vector{Float64}:
 0.02
```

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`AbstractResult`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`forward_prior`](@ref)
  - [`reconstruct_prior`](@ref)
  - [`prior`](@ref)
"""
abstract type AbstractPriorResult <: AbstractResult end
"""
    const PrE_Pr = Union{AbstractPriorEstimator, AbstractPriorResult}

Groups a prior estimator with a fitted prior result.

`PrE_Pr` is the bound of the `pe` field of every optimiser. A caller can thus give an optimiser a fitted prior result in place of a prior estimator. [`prior`](@ref) has a method for each of the two. The estimator method fits, and the result method returns its argument unchanged. The optimiser calls [`prior`](@ref) once, and has no branch for the kind of value that it holds.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`prior`](@ref)
"""
const PrE_Pr = Union{AbstractPriorEstimator, AbstractPriorResult}
"""
    const Pr_RR = Union{AbstractPriorResult, ReturnsResult}

Groups the two result types that hold an asset returns matrix `X`.

The clustering, phylogeny and centrality forwarders of `src/10_Prior/20_PriorReturnsSource.jl` dispatch on `Pr_RR`. Each forwarder reads `X` from the result that it receives, and calls the method on the asset returns. With one forwarder, an estimator can fit from a prior result or from the returns data. Each forwarder passes `dims = 1` after `kwargs`, because each result type holds its observations along the rows. A `dims` in `kwargs` has no effect. When the caller gives both results, [`returns_matrix_picker`](@ref) picks one of them. The forwarder passes both to the estimator tree as `pr` and `rd`, so a [`FeatureDistance`](@ref) finds its Asset Panel from them.

# Related

  - [`AbstractPriorResult`](@ref)
  - [`ReturnsResult`](@ref)
  - [`returns_matrix_picker`](@ref)
"""
const Pr_RR = Union{AbstractPriorResult, ReturnsResult}
"""
    prior(pe::AbstractPriorEstimator, rd::ReturnsResult; strict::Bool = false,
          dims::Int = 1, cache_reg::Bool = false)

Compute a prior result from the returns data with a prior estimator.

`prior` reads the asset returns and the factor returns from `rd`, and gives them to `pe` with the other data that `rd` holds. Every caller uses this method, and this file defines it once. Each estimator implements the returns-matrix method that this method calls. [`AbstractPriorEstimator`](@ref) states that contract.

# Algorithm

 1. Check that `rd` carries asset returns, so that the estimator does not receive `nothing` for `X`.
 2. When [`needs_factor_returns`](@ref) answers `true`, check that `rd` carries factor returns. The predicate reads the whole tree, so it finds a factor estimator inside an outer estimator whose factor argument is optional. The check runs here, so the caller gets a named error about `rd.F`. Without it, the caller gets a `MethodError` from the inner estimator one call later.
 3. Read the source that `pe` states with [`returns_source`](@ref). Call it `x_src`.
 4. Drop the observations of a Shared Series gap with [`shared_series_view`](@ref). These are a gap of a vector benchmark under `x_src = ExcessSource()`, and a gap of the factor returns when [`needs_factor_returns`](@ref) answers `true`. Every field of `rd` loses the same observations.
 5. Read the asset returns `X` with [`benchmark_relative_returns`](@ref) under `x_src`. `X` is `rd.X`, or `rd.X - rd.B` under `ExcessSource()`.
 6. Call the returns-matrix method of the estimator with `X`, `rd.F` and `rd.pnl`. Pass `rd.iv`, `rd.ivpa`, `rd.ne`, `rd.E`, `strict` and `cache_reg` as keyword arguments. A wrapper prior passes its keyword arguments to the estimator that it holds. The Exogenous Series gets to a [`CrossSectionalFactorPrior`](@ref) whose observed factors read it, and every other estimator ignores it. The call passes `dims = 1`, because a `ReturnsResult` holds its observations along the rows. The keyword `dims` has no effect.
 7. Record the observations that step 4 dropped with [`shared_series_record`](@ref). A reader that holds `rd` then pairs the rows of `X` with the correct observations.
 8. Record `x_src` in the prior result with [`stamp_returns_source`](@ref).

The Asset Panel is the third positional argument for the same reason that `rd.F` is the second. A wrapper prior holds no `ReturnsResult`. It can hold an estimator that fits on a panel only if the panel gets to the returns-matrix method of that estimator. Every returns-matrix method takes the argument. Every wrapper prior passes it unchanged to the estimator that it holds over the assets. An estimator that reads no panel ignores it.

# Arguments

  - $(arg_dict[:pe])
  - `rd`: Returns data that holds the asset returns, and the factor returns when the estimator reads them.
  - $(arg_dict[:strict_prior_rd])
  - $(arg_dict[:dims_rd])
  - $(arg_dict[:cache_reg])

# Validation

  - `!isnothing(rd.X)`.
  - The rules of [`benchmark_relative_returns`](@ref).
  - `!isnothing(rd.F)`, when `needs_factor_returns(pe) === true`.
  - The rules of [`shared_series_rows`](@ref).

# Returns

  - `pr::AbstractPriorResult`: The prior result that the estimator computed.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`needs_factor_returns`](@ref)
  - [`ReturnsResult`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
"""
function prior(pe::AbstractPriorEstimator, rd::ReturnsResult; strict::Bool = false,
               dims::Int = 1, cache_reg::Bool = false)
    @argcheck(!isnothing(rd.X), IsNothingError)
    assert_factor_returns(pe, rd.F)
    x_src = returns_source(pe)
    rf = needs_factor_returns(pe) === true
    T = size(rd.X, 1)
    rows = shared_series_rows(rd, x_src, rf)
    rd = shared_series_view(rd, x_src, rf)
    pr = prior(pe, benchmark_relative_returns(rd, x_src), rd.F, rd.pnl; iv = rd.iv,
               ivpa = rd.ivpa, ne = rd.ne, E = rd.E, strict = strict, dims = 1,
               cache_reg = cache_reg)
    return stamp_returns_source(shared_series_record(pr, rows, T), pe)
end
"""
    reads_factor_returns(pe::Union{AbstractPriorEstimator, Online}) -> Bool
    reads_factor_returns(pe::Union{Nothing, AbstractPriorResult, AbstractEstimator}) -> Bool

Answer whether a fit through `pe` reads the factor returns. When it does, a gap in the factor returns drops the observation, see [`shared_series_view`](@ref).

A tree reads them when [`needs_factor_returns`](@ref) answers `true`. A tree whose factor arguments are all optional answers `nothing`, and it reads no factor return of its own. When `pe` is `nothing`, or a prior result that a caller computed before, the fit reads no factor return. Any other estimator in the place of a prior is a schedule, and the nearest fold loop resolves it to a prior. That prior drops its own gaps at its entry, so the answer is `false`.

# Arguments

  - `pe`: The prior estimator, a prior result, or `nothing`.

# Returns

  - `flag::Bool`: `true` when the fit reads the factor returns.

# Related

  - [`needs_factor_returns`](@ref)
  - [`shared_series_view`](@ref)
"""
function reads_factor_returns(pe::Union{AbstractPriorEstimator, Online})
    return needs_factor_returns(pe) === true
end
function reads_factor_returns(::Union{Nothing, AbstractPriorResult, AbstractEstimator})
    return false
end
"""
    prior_regression_remedy

The cause and the remedy of every error that says that a prior carries no factor block.

The string is written once, so the two kinds of consumer give the same cause and remedy.
The consumers need the loadings for different purposes. Some project factor moments
through the loadings, and some plot them. Each consumer gives its own first sentence
in the `lead` of [`assert_prior_regression`](@ref). No consumer writes the cause again. A
prior has `rr === nothing` for one cause, and one remedy corrects it. Both follow from the
rule that every wrapper estimator passes on `rr` and the factor block `fpr`.

## Which errors carry it

Two errors carry it. They are the two ways in which a caller can ask for loadings that no
estimator computed.

  - The `IsNothingError` that [`assert_prior_regression`](@ref) raises. An estimator raises
    it when its `pe` field returns a prior with `rr === nothing`. A plotting function in
    the factor space raises it when it receives such a prior. Each caller gives its own
    `lead`, and adds this string unchanged after it.
  - The `IsNothingError` that [`constraint_space_basis`](@ref) raises when a factor exposure
    constraint has no basis for its loadings. The space states no basis, and the prior
    carries no loadings. That message starts with its own sentences about the space, and
    then adds this string, because the remedy for the prior is the same.

# Related

  - [`assert_prior_regression`](@ref)
  - [`constraint_space_basis`](@ref)
"""
const prior_regression_remedy = "No regression was ever computed: wrapping estimators forward `rr` and the factor block `fpr` from the prior they wrap, so nesting order does not matter, but nothing in the chain produces loadings (e.g. `EntropyPoolingPrior(; pe = EmpiricalPrior())`). Put an estimator that produces them at the bottom, such as `FactorPrior`."
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Check that a prior result carries a factor block, so that a consumer can read its loadings.

An estimator whose `pe` field is typed [`AbstractLowOrderPriorEstimator_F_AF`](@ref) admits the [`AbstractLowOrderPriorEstimator_AF`](@ref) part of that union. A member of that part reads factor returns only when the caller gives them. The type limits the returns that the estimator reads, and does not tell whether its result carries a regression. An estimator that projects factor moments through the loadings needs the regression, and must check for it.

A prior has `pr.rr === nothing` for one cause: no estimator in the chain computed a regression, as in `EntropyPoolingPrior(; pe = EmpiricalPrior())`. No estimator removes a regression, because every wrapper estimator passes on `rr` and the factor block `fpr`. The order of the estimators in the chain has no effect. A check of `rr` covers the whole factor block, because [`LowOrderPrior`](@ref) requires `rr` and `fpr` together or not at all. For this reason, a plotting function that reads `fpr.mu` or `fpr.sigma` calls this check on `rr` before it reads them.

The plotting functions in the factor space also read the factor block, and they get the same message. Only the first sentence is different. `lead` holds it, and [`prior_regression_remedy`](@ref) holds the rest.

# Arguments

  - `pr`: Prior result that the consumer received.
  - `sym`: Name of the field or argument through which the result came, for the error message.
  - `lead`: First sentence of the message. It names what needed the loadings and what it found. The default is for a wrapper estimator. A consumer that is not an estimator gives its own sentence, because the default says that the type of an estimator field does not make sure of a regression.

# Validation

  - `!isnothing(pr.rr)`, which raises an `IsNothingError` carrying `lead` followed by [`prior_regression_remedy`](@ref).

# Returns

  - `nothing`.

# Related

  - [`AbstractLowOrderPriorEstimator_F_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`Regression`](@ref)
  - [`prior_regression_remedy`](@ref)
"""
function assert_prior_regression(pr::AbstractPriorResult, sym::Sym_Str = :pe;
                                 lead::AbstractString = "this estimator projects factor moments through the regression loadings, so the prior it wraps must carry one, but `$sym` produced a result with `rr === nothing`. `$sym` accepts estimators that use factor returns only optionally, so the type does not guarantee a regression.")::Nothing
    @argcheck(!isnothing(pr.rr), IsNothingError("$lead $prior_regression_remedy"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the fields of a prior result as a named tuple, keyed in declaration order.

The function reads with `getfield`, so it reads only the values that `pr` stores. It does not read a name that a [`@forward_properties`](@ref) block adds. [`forward_prior`](@ref) needs this difference. `HighOrderPrior` forwards all of its `pr`, so `mu` and `sigma` are properties of it but not fields, and a patch can change only a field. The function reads the field list from the type, so a new field of a prior result type needs no change here.

# Algorithm

 1. Read the field names of `typeof(pr)` into `fnames`, in declaration order.
 2. Read each of those fields off `pr` with `getfield`, and return them as a `NamedTuple` keyed by `fnames`.

# Arguments

  - $(arg_dict[:pr])

# Returns

  - `vals::NamedTuple`: The fields of `pr`, keyed by name in declaration order.

# Related

  - [`forward_prior`](@ref)
  - [`reconstruct_prior`](@ref)
  - [`AbstractPriorResult`](@ref)
"""
function prior_field_values(pr::AbstractPriorResult)
    fnames = fieldnames(typeof(pr))
    return NamedTuple{fnames}(getfield.((pr,), fnames))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Forward a wrapped prior result, and name only the fields that the wrapper estimator changes or drops.

This function applies the composition rule of the prior results. A wrapper forwards a field when the value stays correct. It drops a field only when the forwarded value would state something false. The docstring of the estimator names every field that it drops.

The default is to forward, and the caller writes nothing for it. A wrapper cannot return fewer fields than the result that it wraps by mistake. The call site names every change. A new value is `field = value`, and a drop is `field = nothing`. A search of the source finds every drop. A hand-written call to the constructor names all fifteen fields of a [`LowOrderPrior`](@ref), and does not show which of them it changes.

The rebuild goes through the keyword constructor of `pr`, see [`reconstruct_prior`](@ref). Every `@argcheck` runs, and a forward that leaves the result inconsistent raises the same error as a hand-written constructor call. A caller can name only a field of `pr`. A forwarded or computed property is a view of a nested value, so a value for it can mean only a value for the field that holds it.

## The three enforced bindings

Three fields are bound to the value of another field. To forward one of them past a change to its field states something false. The function enforces each binding. When the caller names the field on the left, the caller must also name the fields on the right, with a rebuilt value or with `nothing`.

  - `sigma` binds `chol`. When `chol` is present, every consumer reads it in place of `sigma`. A stale `chol` makes the optimisation ignore the posterior covariance with no error.
  - `w` binds `ens`, `eoc`, `kld` and `ow`. These are diagnostics of `w` and the counts of the moments that `w` weights, so they are false for a different `w`.
  - `rr` binds `o_X`. `o_X` says that `X` is a reconstruction, and `rr` records the projection that made it, so the constructor refuses one without the other. A drop of the factor block also drops the original returns.

A binding has no effect when the bound field is `nothing`, or when `pr` has no such field. No stale value is then present.

The constructor checks the rest. `rr` and `fpr` must be given together or not at all. The constructor checks `w`, `chol`, `o_X`, `ss_gaps` and `pc` against the shape of `X` and `mu` again.

## What does not fit

[`FactorPrior`](@ref) and [`FactorBlackLittermanPrior`](@ref) lift a factor-axis prior into an asset-axis result. [`AugmentedBlackLittermanPrior`](@ref) merges two priors. None of the three forwards one wrapped result along its own axis. Each calls the constructor of its prior result directly, and does not use this function. Each still uses `forward_prior` on the factor block that it builds, because that is a usual forward of the factor prior.

# Algorithm

 1. Collect the keyword overrides into the named tuple `patch`. When `patch` is empty, return `pr` itself, because a forward that changes nothing rebuilds nothing.
 2. Compare the names of `patch` with the fields of `typeof(pr)`. Call the names that are not fields `extra`. When `extra` is not empty, raise an `ArgumentError` that names the fields of `pr`.
 3. Enforce the binding of `chol` to `sigma`. When `patch` names `sigma` and not `chol`, and [`bound_field_is_stale`](@ref) says that `pr` holds a `chol`, raise a [`ConflictingArgumentError`](@ref).
 4. Enforce the binding of `o_X` to `rr` with the same three tests. A failure raises the second [`ConflictingArgumentError`](@ref).
 5. Enforce the binding of `ens`, `eoc`, `kld` and `ow` to `w`. When `patch` names `w`, collect into `stale` each of the four that `patch` does not name and that `pr` holds. Raise when `stale` is not empty.
 6. Rebuild `pr` through [`reconstruct_prior`](@ref). It merges `patch` over [`prior_field_values`](@ref) and calls the keyword constructor, so every `@argcheck` of `typeof(pr)` runs on the result.

# Arguments

  - `pr`: Prior result that the wrapped estimator returned.
  - `overrides...`: Field overrides. Each is a value to replace, or `nothing` to drop.

# Validation

  - When `overrides` names `sigma`, it also names `chol`, unless `pr.chol` is `nothing`.
  - When `overrides` names `w`, it also names each of `ens`, `eoc`, `kld` and `ow` that `pr` states.
  - When `overrides` names `rr`, it also names `o_X`, unless `pr.o_X` is `nothing`.
  - Every name in `overrides` is a field of `typeof(pr)`.
  - Every `@argcheck` of the constructor of `typeof(pr)`.

# Returns

  - `pr::AbstractPriorResult`: The wrapped result with `overrides` applied, or `pr` itself when `overrides` is empty.

# Examples

```jldoctest
julia> pr = LowOrderPrior(; X = [0.01 0.02; 0.03 0.04], mu = [0.02, 0.03],
                          sigma = [0.0004 0.0002; 0.0002 0.0003], chol = [0.02 0.01; 0.0 0.01415]);

julia> PortfolioOptimisers.forward_prior(pr) === pr
true

julia> pr2 = PortfolioOptimisers.forward_prior(pr; mu = [0.05, 0.06], chol = nothing);

julia> (pr2.mu, pr2.chol, pr2.sigma === pr.sigma)
([0.05, 0.06], nothing, true)

julia> PortfolioOptimisers.forward_prior(pr; sigma = [0.0009 0.0001; 0.0001 0.0004])
ERROR: ConflictingArgumentError: forwarding `chol` past a change to `sigma` would state something false: `chol` takes precedence over `sigma` at every consumer, so a stale factor makes the optimisation silently ignore the updated covariance. Pass `chol = nothing` to drop it, or a factor rebuilt from the new `sigma`.
[...]
```

# Related

  - [`reconstruct_prior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`bound_field_is_stale`](@ref)
  - [`assert_prior_regression`](@ref)
  - [`ConflictingArgumentError`](@ref)
"""
function forward_prior(pr::AbstractPriorResult; overrides...)::AbstractPriorResult
    patch = NamedTuple(overrides)
    if isempty(patch)
        return pr
    end
    fnames = fieldnames(typeof(pr))
    extra = filter(sym -> sym ∉ fnames, propertynames(patch))
    @argcheck(isempty(extra),
              ArgumentError("$(extra) cannot be forwarded onto a $(nameof(typeof(pr))), whose fields are $(fnames). A forwarded or computed property is a view of a nested value; name the field holding that value instead."))
    if haskey(patch, :sigma) && !haskey(patch, :chol) && bound_field_is_stale(pr, :chol)
        throw(ConflictingArgumentError("forwarding `chol` past a change to `sigma` would state something false: `chol` takes precedence over `sigma` at every consumer, so a stale factor makes the optimisation silently ignore the updated covariance. Pass `chol = nothing` to drop it, or a factor rebuilt from the new `sigma`."))
    end
    if haskey(patch, :rr) && !haskey(patch, :o_X) && bound_field_is_stale(pr, :o_X)
        throw(ConflictingArgumentError("forwarding `o_X` past a change to `rr` would state something false: `o_X` says `X` is a reconstruction, and `rr` is what records the projection that produced it. Dropping the factor block drops the original with it. Pass `o_X` explicitly — `nothing` to drop it, or the matrix it should now name"))
    end
    if haskey(patch, :w)
        stale = filter(sym -> !haskey(patch, sym) && bound_field_is_stale(pr, sym),
                       (:ens, :eoc, :kld, :ow))
        @argcheck(isempty(stale),
                  ConflictingArgumentError("forwarding $(stale) past a change to `w` would state something false: they are diagnostics of the weights they were computed with, and diagnostics follow their weights. Pass each as `nothing` to drop it, or a value recomputed alongside the new `w`."))
    end
    return reconstruct_prior(pr, patch)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when field `sym` of `pr` holds a value that becomes stale if its bound field changes without it.

A field that `pr` does not have, or that holds `nothing`, cannot become stale. The function reads with `getfield`, so a forwarded property with the same name cannot answer for a field that `pr` does not have.

# Algorithm

 1. Check whether `typeof(pr)` declares a field named `sym`. When it does not, the binding has no effect on `pr`, so answer `false` and read nothing.
 2. Read that field with `getfield`, and answer `true` when its value is not `nothing`.

# Arguments

  - $(arg_dict[:pr])
  - `sym`: Name of the bound field to test.

# Returns

  - `stale::Bool`: `true` when `pr` holds a value under `sym` that a change to the field it is bound to would make stale.

# Related

  - [`forward_prior`](@ref)
  - [`AbstractPriorResult`](@ref)
"""
function bound_field_is_stale(pr::AbstractPriorResult, sym::Symbol)::Bool
    return hasfield(typeof(pr), sym) && !field_states_nothing(getfield(pr, sym))
end
"""
    field_states_nothing(v) -> Bool

Return `true` when a field value of a prior result states nothing.

A field states nothing when it holds `nothing`, or a tuple whose every entry is `nothing`. The tuple is the form of `eoc`, which holds one count for each moment, and `(nothing, nothing)` is its default.

# Arguments

  - `v`: The value of the field.

# Returns

  - `flag::Bool`: `true` when `v` states nothing.

# Related

  - [`bound_field_is_stale`](@ref)
  - [`forward_prior`](@ref)
"""
function field_states_nothing(v)
    return isnothing(v)
end
function field_states_nothing(v::Tuple)
    return all(isnothing, v)
end
"""
    prior(pr::AbstractPriorResult, args...; kwargs...)

Return a prior result unchanged.

This method is the result half of [`PrE_Pr`](@ref). A field with that bound calls `prior` once. Because of this method, a field that holds a fitted result needs no branch of its own. The method takes and ignores every other argument, so the call site is the same for an estimator and for a result.

# Arguments

  - $(arg_dict[:pr])
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `pr::AbstractPriorResult`: The input prior result, unchanged.

# Related

  - [`AbstractPriorResult`](@ref)
  - [`PrE_Pr`](@ref)
  - [`prior`](@ref)
"""
function prior(pr::AbstractPriorResult, args...; kwargs...)::AbstractPriorResult
    return pr
end
"""
    port_opt_view(pr::Option{<:AbstractPriorEstimator}, ::Any, args...; kwargs...)
    port_opt_view(pr::AbstractVector{<:Union{AbstractPriorResult, AbstractPriorEstimator}},
                  ::Any, args...; kwargs...)

Pass a prior estimator, or a vector of priors, through a view unchanged.

These two methods of [`port_opt_view`](@ref) cut nothing. An estimator holds a configuration and no data on an asset axis, so the view has nothing to cut. The subproblem fits the estimator again on its own universe. A vector holds one entry for each subproblem, such as each cluster or each subset. The caller picks the entry before it takes the view. A cut of the vector by an asset index cuts the wrong axis.

The prior result types that hold data on the asset axis have their own methods. See [`port_opt_view`](@ref) on [`LowOrderPrior`](@ref) and on [`HighOrderPrior`](@ref).

# Arguments

  - $(arg_dict[:per])
  - The second positional argument is the asset index. It has no name, because neither method reads it.
  - `args...`: Additional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `pr`: The input, unchanged.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(pr::Option{<:AbstractPriorEstimator}, ::Any, args...;
                       kwargs...)::Option{<:AbstractPriorEstimator}
    return pr
end
function port_opt_view(pr::AbstractVector{<:Union{AbstractPriorResult,
                                                  AbstractPriorEstimator}}, ::Any, args...;
                       kwargs...)
    return pr
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the returns, mean and covariance a low order prior estimator produced.

Every low order prior estimator returns a `LowOrderPrior`. It holds the asset returns, the mean vector and the covariance matrix. It can also hold a Cholesky factor, the observation weights and three diagnostics of them, a factor block, and the original returns. The last fields record the source of the returns, the marks of a Shared Series cut, and the cache of an opinion pool. The constructor checks every field.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LowOrderPrior(;
        X::MatNum,
        o_X::Option{<:MatNum} = nothing,
        mu::VecNum,
        sigma::MatNum,
        chol::Option{<:MatNum} = nothing,
        w::Option{<:ObsWeights} = nothing,
        ens::Option{<:Number} = nothing,
        eoc::Tuple{Option{<:Real}, Option{<:Real}} = (nothing, nothing),
        kld::Option{<:Num_VecNum} = nothing,
        ow::Option{<:VecNum} = nothing,
        rr::Option{<:AbstractLoadingsRegressionResult} = nothing,
        fpr::Option{<:LowOrderPrior} = nothing,
        x_src::Option{<:AbstractReturnsSource} = nothing,
        ss_gaps::Option{<:AbstractVector{Bool}} = nothing,
        pc::Option{<:NamedTuple} = nothing
    ) -> LowOrderPrior

Keywords correspond to the struct's fields.

## The factor block

A prior that a factor model fits carries two distributions. The fields of the `LowOrderPrior` hold the distribution over the assets. The field `fpr` holds the distribution over the factors, as a nested `LowOrderPrior`. A nested result, and not a set of flat fields with an `f_` prefix, gives the factor distribution every field of a `LowOrderPrior`. These are `w`, `ens`, `eoc`, `kld` and `ow` as well as `mu` and `sigma`. A field added later also applies to it with no second edit. Its `X` is the factor returns matrix, over the same observations as the asset `X`.

`fpr` and `rr` are the factor block, and the constructor requires them together or not at all. `rr` projects the block onto the assets, `mu ≈ rr.M * fpr.mu + rr.b`. A factor distribution with no loadings has no relation to this asset axis.

The bound of `rr` is [`AbstractLoadingsRegressionResult`](@ref). A member of that root carries the loadings matrix `M`, so the field admits a [`Regression`](@ref) and a [`CrossSectionalFactorModel`](@ref). The bound selects by the loadings, and not by the method of the fit. Every check of the constructor on the block reads `rr.M` alone, and `fpr` is on the axis that the columns of `M` name. Every consumer of the field reads `M`, or reads `L`, which returns `M` when `L` is not set.

The field does not tell one property of the block, so a consumer that needs it must ask. A member fitted in a re-based Factor Family answers `true` to [`has_family_rebasis`](@ref). Its `fpr.sigma` is then singular, because the raw factor axis is a linear image of the re-based axis. A projection through `M` stays correct, and [`HighOrderFactorPriorEstimator`](@ref) does such a projection. An inverse or a factorisation of `fpr.sigma` has no answer. The inverse raises no error, and returns a result whose scale looks usual. For this reason, [`BayesianBlackLittermanPrior`](@ref) refuses such a prior result.

The flat names read the nested block, so code written for the old shape still works. `pr.f_mu`, `pr.f_sigma` and `pr.f_w` return `fpr.mu`, `fpr.sigma` and `fpr.w`, or `nothing` when there is no factor block. `pr.f_ens`, `pr.f_kld` and `pr.f_ow` work in the same way. They are properties, not fields, so [`forward_prior`](@ref) and [`prior_field_values`](@ref) see only `fpr`.

### Which read to use

Read the factor block as `pr.fpr.mu`. The flat names with the `f_` prefix stay so that code written for the old shape still works. They also help where a read must return a value or `nothing` with no branch.

The flat names do not cover the block, and the set does not grow. There are six flat names for fifteen fields. `fpr.X`, the factor returns matrix, has no flat name, and neither have `fpr.chol`, `fpr.eoc` and `fpr.rr`. The set has six names here, and seven on [`HighOrderPrior`](@ref). A field added to `LowOrderPrior` later is available as `pr.fpr.<name>`, and gets no flat name, so no field needs an edit in two places.

The two reads are different when there is no factor block. `pr.f_mu` returns `nothing`, but `pr.fpr.mu` raises an error, because `fpr` is `nothing`. First check the block with [`assert_prior_regression`](@ref), and then read through `fpr`. The check reads `rr` only, which is enough, because `rr` and `fpr` are present together or not at all.

## Composition: what a wrapping estimator forwards

Most prior estimators wrap another, and build their prior result from the result that they receive. [`forward_prior`](@ref) enforces one rule for the fields that the wrapper keeps. A wrapper forwards a field when the value stays correct. It drops a field only when the forwarded value would state something false. The docstring of the estimator names every field that it drops.

The returned result must be consistent. A wrapper must not remove a value that the caller computed to get that consistency. The default is to forward, and the docstring of each estimator lists the fields that it drops and the reasons. Three bindings tie a field to another, so a wrapper never forwards the bound fields alone:

  - `sigma` binds `chol`. When `chol` is present, every consumer reads it in place of `sigma`, and a stale `chol` replaces the changed covariance with no error.
  - `w` binds `ens`, `eoc`, `kld` and `ow`. These are diagnostics and counts of the weights.
  - `rr` binds `o_X`. `rr` records the projection that made `X` from `o_X`.

`forward_prior` refuses a forward that breaks a binding.

## The original returns matrix

Four estimators overwrite `X`, and `o_X` exists for them. [`FactorPrior`](@ref), [`FactorBlackLittermanPrior`](@ref) and [`AugmentedBlackLittermanPrior`](@ref) write a reconstruction from the factors. [`CrossSectionalFactorPrior`](@ref) writes scenarios from the factor returns, the loadings and the standardised idiosyncratic returns. On their results, `X` is the asset distribution that the prior states, and not the returns of the caller. `o_X` holds the returns of the caller. It is `nothing` on every other result, where `X` is the returns of the caller.

The two matrices are different. The reconstruction of the first three estimators spans only the factors. It has rank `size(F, 2)`, and it has no residual. A consumer that fits a moment on the sample again must read the original. Otherwise it gets a singular matrix when there are more assets than factors.

Read the original returns as `original_X`, not as `o_X`. The property is always a matrix. It is the field when the field holds one, and `X` when it does not. A consumer needs no fallback. The field is storage, and it answers a different question. `isnothing(pr.o_X)` tells whether this prior result changed `X`. The field holds this state, and the property does not. [`forward_prior`](@ref) rebuilds through the keyword constructor with every field named. A `nothing` has no effect there, but a matrix in every result would become stale after a change to `X`.

`o_X` requires `rr`. Every estimator that overwrites `X` now builds it from a factor prior and loadings, and carries the loadings in `rr`. A prior result that has an `o_X` and no `rr` is a defect. This is a rule of the present library, and not of the domain. A new estimator that changes `X` with no regression must change this rule on purpose.

## The source of the returns

`x_src` records the frame of `X`. [`prior`](@ref) writes in it the source that the prior estimator states, through [`stamp_returns_source`](@ref). It does this when it fits on a [`ReturnsResult`](@ref), and when it reads out the state of an online fold. A returns-matrix method of `prior` cannot know the frame of the matrix that it receives, so it leaves `nothing`. A consumer that receives the Result in place of the estimator reads the frame from this field. The outer problem of a meta-optimiser adds the benchmark back to `X` when the field is [`ExcessSource`](@ref), see [`outer_returns_source`](@ref). A [`HighOrderPrior`](@ref) holds no such field, and answers `x_src` from the `LowOrderPrior` that it wraps.

## The observations of the rows

`ss_gaps` records the observations of the returns data that the fit dropped for a Shared Series gap. A fit also drops rows at the start. A Scenario Cap keeps the last rows, and a [`CrossSectionalFactorPrior`](@ref) drops the rows of its warm-up and of its exposure lag. So the rows of `X` are the last `size(X, 1)` observations that `ss_gaps` keeps. When `ss_gaps` is `nothing`, they are the last `size(X, 1)` observations of the returns data.

A reader that holds the returns data that the prior read pairs its rows with the rows of `X` through the marks, see [`shared_series_kept_rows`](@ref). A reader that holds the returns data after the same cut has fewer observations than the marks, and pairs the last rows. That was the rule before the record existed. [`prior`](@ref) writes the marks after it fits on the returns data, and after it reads out an online fold. A returns-matrix method of `prior` drops no observation, so it leaves `nothing`. A [`HighOrderPrior`](@ref) answers `ss_gaps` from the `LowOrderPrior` that it wraps.

## Validation

  - `X`, `mu`, and `sigma` must be non-empty.
  - `size(sigma, 1) == size(sigma, 2)`.
  - `size(X, 2) == length(mu) == size(sigma, 1)`.
  - Every entry of `sigma` between two investable assets is finite. An investable asset has a finite `mu` and a finite variance. Raises an `IsNonFiniteError`.
  - If `w` is not `nothing`, `!isempty(w)` and `length(w) == size(X, 1)`.
  - Each entry of `eoc` that is not `nothing` is finite and positive, through [`assert_moment_counts`](@ref).
  - If `kld` is an `AbstractVector`, `!isempty(kld)`.
  - If `ow` is not `nothing`, `!isempty(ow)`.
  - `rr` and `fpr` must be provided together or not at all.
  - If the factor block is present, `size(rr.M, 2) == length(fpr.mu) == size(fpr.sigma, 1)`, `size(rr.M, 1) == length(mu)`, and `size(fpr.X, 1) == size(X, 1)`. The two blocks describe the same observations. The constructor of the factor block checks the inside of the block, such as its own `w` against its own `X`.
  - If `o_X` is not `nothing`, `o_X !== X`, `size(o_X) == size(X)`, and `rr` is not `nothing`. `o_X !== X` tests identity, not equality. So `o_X = copy(X)` passes, and `o_X = X` raises an error. The two calls look the same, but only the first holds a matrix that a later change to `X` does not change. The check refuses a prior result whose original is the same object as its `X`. It does not refuse a matrix with equal values.
  - If `chol` is not `nothing`, `!isempty(chol)` and `length(mu) == size(chol, 2)`.
  - The rules of [`assert_shared_series_gaps`](@ref) on `ss_gaps`.
  - If `pc` is not `nothing`, `length(pc.w) == size(X, 1)`, and `size(pc.X, 1) == size(X, 1)` when `pc.X` is a matrix, through [`assert_pool_cache`](@ref).

## View parameters

`LowOrderPrior` defines its own [`port_opt_view`](@ref) method, and does not make one from field tags.

  - It reads no argument after `i`. It takes and ignores more positional arguments.
  - `rr` recurses through [`port_opt_view`](@ref) with `i`, which cuts the loadings on their asset axis.
  - `X`, `o_X`, `mu`, `sigma` and `chol` are cut to `i` on the asset axis. `o_X` takes the same cut as `X`, so the original returns of a subproblem are the returns of the caller for the assets of that subproblem.
  - `w`, `ens`, `eoc`, `kld` and `ow` pass through unchanged. They are on the observation axis, and `i` indexes assets.
  - `fpr` passes through unchanged, because it is a distribution over factors and not over assets. The view thus keeps `rr` and `fpr` together, as the rule of the factor block requires.
  - `ss_gaps` passes through unchanged. It is on the observation axis.
  - `pc` is dropped. Its rows hold every asset, and a fit of the subset of assets has no cache of its own.

# Examples

```jldoctest
julia> LowOrderPrior(; X = [0.01 0.02; 0.03 0.04], mu = [0.02, 0.03],
                     sigma = [0.0001 0.0002; 0.0002 0.0003])
LowOrderPrior
        X ┼ 2×2 Matrix{Float64}
      o_X ┼ nothing
       mu ┼ Vector{Float64}: [0.02, 0.03]
    sigma ┼ 2×2 Matrix{Float64}
     chol ┼ nothing
        w ┼ nothing
      ens ┼ nothing
      eoc ┼ Tuple{Nothing, Nothing}: (nothing, nothing)
      kld ┼ nothing
       ow ┼ nothing
       rr ┼ nothing
      fpr ┼ nothing
    x_src ┼ nothing
  ss_gaps ┼ nothing
       pc ┴ nothing
```

# Related

  - [`AbstractPriorResult`](@ref)
  - [`prior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`forward_prior`](@ref)
  - [`reconstruct_prior`](@ref)
  - [`port_opt_view`](@ref)
  - [`FeatureDistance`](@ref)
"""
@concrete struct LowOrderPrior <: AbstractPriorResult
    """
    $(field_dict[:X])
    """
    X
    """
    $(field_dict[:o_X])
    """
    o_X
    """
    $(field_dict[:mu])
    """
    mu
    """
    $(field_dict[:sigma])
    """
    sigma
    """
    $(field_dict[:chol])
    """
    chol
    """
    $(field_dict[:w_prior])
    """
    w
    """
    $(field_dict[:ens_prior])
    """
    ens
    """
    $(field_dict[:eoc_prior])
    """
    eoc
    """
    $(field_dict[:kld])
    """
    kld
    """
    $(field_dict[:op_w])
    """
    ow
    """
    $(field_dict[:reg_rr])
    """
    rr
    """
    $(field_dict[:fpr])
    """
    fpr
    """
    $(field_dict[:x_src_pr])
    """
    x_src
    """
    $(field_dict[:ss_gaps])
    """
    ss_gaps
    """
    $(field_dict[:pc])
    """
    pc
    function LowOrderPrior(X::MatNum, o_X::Option{<:MatNum}, mu::VecNum, sigma::MatNum,
                           chol::Option{<:MatNum}, w::Option{<:ObsWeights},
                           ens::Option{<:Number},
                           eoc::Tuple{Option{<:Real}, Option{<:Real}},
                           kld::Option{<:Num_VecNum}, ow::Option{<:VecNum},
                           rr::Option{<:AbstractLoadingsRegressionResult},
                           fpr::Option{<:LowOrderPrior},
                           x_src::Option{<:AbstractReturnsSource},
                           ss_gaps::Option{<:AbstractVector{Bool}},
                           pc::Option{<:NamedTuple})
        @argcheck(!isempty(X), IsEmptyError("X cannot be empty"))
        @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
        @argcheck(!isempty(sigma), IsEmptyError("sigma cannot be empty"))
        assert_matrix_issquare(sigma, :sigma)
        @argcheck(size(X, 2) == length(mu) == size(sigma, 1),
                  DimensionMismatch("size(X, 2) ($(size(X, 2))), length(mu) ($(length(mu))), and size(sigma, 1) ($(size(sigma, 1))) must all match"))
        imsk = isfinite.(mu) .& isfinite.(LinearAlgebra.diag(sigma))
        @argcheck(all(isfinite, view(sigma, imsk, imsk)),
                  IsNonFiniteError("sigma of investable assets must only contain finite values"))
        assert_nonempty_nonneg_finite_val(w, :w)
        if isa(w, StatsBase.AbstractWeights)
            @argcheck(length(w) == size(X, 1),
                      DimensionMismatch("length(w) ($(length(w))) must match size(X, 1) ($(size(X, 1)))"))
        end
        if isa(kld, VecNum)
            @argcheck(!isempty(kld), IsEmptyError("kld cannot be empty"))
        end
        if !isnothing(ow)
            @argcheck(!isempty(ow), IsEmptyError("ow cannot be empty"))
        end
        # `rr` and `fpr` are the factor block. The block is validated as a whole here and
        # only against this asset axis: everything internal to it — its own `mu`/`sigma`
        # shapes, and its own `w` against its own `X` — is its own constructor's job.
        @argcheck(isnothing(rr) == isnothing(fpr),
                  ArgumentError("rr and fpr are the factor block and must be provided together or not at all, isnothing(rr) = $(isnothing(rr)), isnothing(fpr) = $(isnothing(fpr))"))
        if !isnothing(rr)
            @argcheck(size(rr.M, 2) == length(fpr.mu) == size(fpr.sigma, 1),
                      DimensionMismatch("size(rr.M, 2) = $(size(rr.M, 2)), length(fpr.mu) = $(length(fpr.mu)), and size(fpr.sigma, 1) = $(size(fpr.sigma, 1)) must all match"))
            @argcheck(size(rr.M, 1) == length(mu),
                      DimensionMismatch("size(rr.M, 1) = $(size(rr.M, 1)) must match length(mu) = $(length(mu))"))
            @argcheck(size(fpr.X, 1) == size(X, 1),
                      DimensionMismatch("size(fpr.X, 1) ($(size(fpr.X, 1))) must match size(X, 1) ($(size(X, 1))): the asset and factor blocks describe the same observations"))
        end
        # `o_X` records that `X` is not the matrix the caller handed in. Three checks, all
        # O(1): no matrix is ever compared by value.
        #
        # The `rr` requirement is a *present-tense* constraint, not a law of the domain.
        # Every estimator that overwrites `X` today does so by projecting a factor prior
        # through regression loadings, so the loadings are always in hand and a prior result
        # claiming a reconstruction it cannot explain is a bug. A future estimator that
        # transforms `X` without a regression — a bootstrap or a simulation prior — is the
        # case that relaxes this, and it must relax it deliberately. See ADR 0046.
        if !isnothing(o_X)
            @argcheck(o_X !== X,
                      ArgumentError("o_X is X itself, so this prior result has no original distinct from the one it asserts. Pass o_X = nothing, which is what every consumer reads as \"X is the original\""))
            @argcheck(size(o_X) == size(X),
                      DimensionMismatch("size(o_X) ($(size(o_X))) must match size(X) ($(size(X))): the original and the matrix this prior result asserts describe the same observations and the same assets"))
            @argcheck(!isnothing(rr),
                      IsNothingError("o_X says X is not the caller's matrix, but rr === nothing, so this prior result does not record what produced X. Every estimator that overwrites X projects a factor prior through regression loadings and carries them in rr"))
        end
        if !isnothing(chol)
            @argcheck(!isempty(chol), IsEmptyError("chol cannot be empty"))
            @argcheck(length(mu) == size(chol, 2),
                      DimensionMismatch("length(mu) ($(length(mu))) must match size(chol, 2) ($(size(chol, 2)))"))
        end
        assert_moment_counts(eoc)
        assert_shared_series_gaps(ss_gaps, size(X, 1))
        assert_pool_cache(pc, size(X, 1))
        fs = (X, o_X, mu, sigma, chol, w, ens, eoc, kld, ow, rr, fpr, x_src, ss_gaps, pc)
        return new{map(typeof, fs)...}(fs...)
    end
end
"""
    assert_pool_cache(pc::Nothing, T::Integer)
    assert_pool_cache(pc::NamedTuple, T::Integer)

Check the cache of an opinion pool against the scenarios of the [`LowOrderPrior`](@ref) that carries it.

The pooled weights of the cache weigh the scenarios of the result, so the weights and the rows of the cache match the rows of the result.

# Arguments

  - `pc`: The cache of the pool, `(; X, w, pe2)` with the cache of `pe2` in `pe2`, or `nothing`.
  - `T`: The number of scenarios of the result.

# Validation

  - `length(pc.w) == T`. Raises a `DimensionMismatch` otherwise.
  - `size(pc.X, 1) == T` when `pc.X` is a matrix. Raises a `DimensionMismatch` otherwise.

# Returns

  - `nothing`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`opinion_pooling_cache`](@ref)
"""
function assert_pool_cache(::Nothing, ::Integer)::Nothing
    return nothing
end
function assert_pool_cache(pc::NamedTuple, T::Integer)::Nothing
    @argcheck(length(pc.w) == T,
              DimensionMismatch("length(pc.w) ($(length(pc.w))) must match size(X, 1) ($T): the pooled weights weigh the scenarios of the result"))
    @argcheck(isnothing(pc.X) || size(pc.X, 1) == T,
              DimensionMismatch("pc.X ($(size(pc.X, 1)) rows) must have the rows of X ($T)"))
    return nothing
end
function LowOrderPrior(; X::MatNum, o_X::Option{<:MatNum} = nothing, mu::VecNum,
                       sigma::MatNum, chol::Option{<:MatNum} = nothing,
                       w::Option{<:ObsWeights} = nothing, ens::Option{<:Number} = nothing,
                       eoc::Tuple{Option{<:Real}, Option{<:Real}} = (nothing, nothing),
                       kld::Option{<:Num_VecNum} = nothing, ow::Option{<:VecNum} = nothing,
                       rr::Option{<:AbstractLoadingsRegressionResult} = nothing,
                       fpr::Option{<:LowOrderPrior} = nothing,
                       x_src::Option{<:AbstractReturnsSource} = nothing,
                       ss_gaps::Option{<:AbstractVector{Bool}} = nothing,
                       pc::Option{<:NamedTuple} = nothing)::LowOrderPrior
    return LowOrderPrior(X, o_X, mu, sigma, chol, w, ens, eoc, kld, ow, rr, fpr, x_src,
                         ss_gaps, pc)
end
"""
    assert_moment_counts(eoc::Tuple) -> Nothing

Check the effective counts of the moments that a prior result states in `eoc`.

# Arguments

  - `eoc`: The tuple of counts, one entry for each moment. An entry is a count or `nothing`.

# Validation

  - Each entry that is not `nothing` is finite and positive, through [`assert_nonempty_gt0_finite_val`](@ref).

# Returns

  - `nothing`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`moment_observation_count`](@ref)
"""
function assert_moment_counts(eoc::Tuple)::Nothing
    foreach(c -> assert_nonempty_gt0_finite_val(c, :eoc), eoc)
    return nothing
end
"""
    moment_observation_count(pr::AbstractPriorResult, k::Integer) -> Real

Return the effective number of observations of moment `k` of a prior result.

The index follows `pr.eoc`: `1` is `mu`, `2` is `sigma`, `3` is the coskewness and `4` is the cokurtosis. An entry of `nothing` means `size(pr.X, 1)`, so a result that states no count gives the number of its rows. A consumer that needs the sample size of one moment reads this function, and not `size(pr.X, 1)`.

# Arguments

  - $(arg_dict[:pr])
  - `k`: The index of the moment.

# Returns

  - `T_e::Real`: The effective number of observations of the moment.

# Related

  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`effective_observation_count`](@ref)
"""
function moment_observation_count(pr::AbstractPriorResult, k::Integer)
    return something(pr.eoc[k], size(pr.X, 1))
end
"""
    moment_count_entry(T_e::Real, T::Integer) -> Option{<:Real}

Return the entry of `eoc` for a moment that is worth `T_e` observations, in a prior result that carries `T` rows.

An entry of `nothing` means the number of rows of the result, so a count equal to `T` is written as `nothing`. An unweighted fit with no Scenario Cap then states no count, as it did before the field existed.

# Arguments

  - `T_e`: The effective number of observations of the moment.
  - `T`: The number of rows that the prior result carries in `X`.

# Returns

  - `nothing` when `T_e == T`, and `T_e` otherwise.

# Related

  - [`moment_observation_count`](@ref)
  - [`EmpiricalPrior`](@ref)
"""
function moment_count_entry(T_e::Real, T::Integer)
    return T_e == T ? nothing : T_e
end
# The flat `f_`-prefixed names are virtual reads of the nested factor block, so code written
# against the pre-nesting shape is unaffected, and `f_ens`/`f_kld`/`f_ow` come for free.
# `compute` with a lambda rather than `alias(f_mu, fpr.mu)`: a dotted locator guards each
# intermediate and throws a [`PropertyPathError`](@ref) when a node is `nothing`, where these
# must return `nothing` — that is what the old flat fields did when there was no factor block.
#
# `original_X` is the read for `o_X`, and it is always a matrix. The field is the storage
# and answers `nothing` where `X` is already the original. The property answers the
# question every consumer asks, which is what returns the caller supplied.
#
# Two names, because one always-populated field cannot express both. `forward_prior`
# rebuilds through the keyword constructor with every field named, so a stated matrix would
# be carried past a change to `X` and would then name the wrong one. A `nothing` is inert.
@forward_properties LowOrderPrior begin
    compute(f_mu, obj -> isnothing(obj.fpr) ? nothing : obj.fpr.mu)
    compute(f_sigma, obj -> isnothing(obj.fpr) ? nothing : obj.fpr.sigma)
    compute(f_w, obj -> isnothing(obj.fpr) ? nothing : obj.fpr.w)
    compute(f_ens, obj -> isnothing(obj.fpr) ? nothing : obj.fpr.ens)
    compute(f_kld, obj -> isnothing(obj.fpr) ? nothing : obj.fpr.kld)
    compute(f_ow, obj -> isnothing(obj.fpr) ? nothing : obj.fpr.ow)
    compute(original_X, obj -> isnothing(obj.o_X) ? obj.X : obj.o_X)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of a [`LowOrderPrior`](@ref) restricted to assets at index `i`.

The view keeps every observation (`Colon`), because a fold cuts the observations before the prior fits.

The view forwards the factor block with no cut, because `i` indexes assets and `fpr` is a distribution over factors. Only `rr` is cut, on its asset axis.

# Algorithm

 1. Cut the Cholesky factor to `i` on its column axis. Call it `chol`. A prior result that holds none keeps `nothing`.
 2. Cut the original returns matrix to `i` on its asset axis. Call it `o_X`. A prior result that holds none keeps `nothing`. `o_X` takes the same cut as `X` in the next step, because the two have the same observations and the same assets.
 3. Rebuild the prior result through its keyword constructor. Cut `X` and `mu` to `i`, and `sigma` to `i` on both axes. Take `chol` and `o_X` from the two steps above. Recurse `rr` through [`port_opt_view`](@ref) with `i`. Forward `w`, `ens`, `eoc`, `kld`, `ow`, `fpr`, `x_src` and `ss_gaps` unchanged, and drop `pc`. Every `@argcheck` of the constructor thus runs on the view.

# Arguments

  - $(arg_dict[:pr])
  - `i`: Asset indices the view keeps.
  - `args...`: Additional arguments (ignored).

# Returns

  - `pr::LowOrderPrior`: The prior result restricted to the assets at `i`. It holds views, not copies.

# Related

  - [`LowOrderPrior`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(pr::LowOrderPrior, i, args...)::LowOrderPrior
    chol = isnothing(pr.chol) ? nothing : view(pr.chol, :, i)
    # `o_X` is assets-major over the same observations as `X`, so it takes the same cut. A
    # subproblem's original returns are the caller's returns for that subproblem's assets.
    o_X = isnothing(pr.o_X) ? nothing : view(pr.o_X, :, i)
    return LowOrderPrior(; X = view(pr.X, :, i), o_X = o_X, mu = view(pr.mu, i),
                         sigma = view(pr.sigma, i, i), chol = chol, w = pr.w, ens = pr.ens,
                         eoc = pr.eoc, kld = pr.kld, ow = pr.ow,
                         rr = port_opt_view(pr.rr, i), fpr = pr.fpr, x_src = pr.x_src,
                         ss_gaps = pr.ss_gaps)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Derive the Investable Mask of a prior result, which is `true` at every asset whose moments are finite.

A Prior Estimator fits on the coverage universe and returns a result on the full asset universe. An asset that it could not estimate carries `NaN` in `mu` and on the diagonal of `sigma`. This function derives the mask, and no prior result stores it in a field. A caller that needs the mask calls this function. A caller that reduces a result keeps the mask that it received. A reduced result cannot give the mask again, because its `mu` is finite at every asset.

When every asset is investable, the function returns `nothing`, and not a mask of `true` values. With `nothing`, a universe with no asset to exclude takes the same path as before the mask existed. That path has no reduction, no expansion and no allocation.

The function does not read the entries of `sigma` outside the diagonal. A non-investable asset can carry `NaN` in all of its row and its column. The diagonal alone decides, so the mask costs one pass over two vectors.

# Algorithm

 1. Take the elementwise conjunction of `isfinite.(pr.mu)` and `isfinite.(diag(pr.sigma))`.
 2. Raise an `IsEmptyError` when the conjunction holds no `true`. An optimisation over no asset has no answer. A problem with no asset that goes further fails at a point far from its cause.
 3. Return `nothing` when the conjunction holds no `false`.
 4. Otherwise, return the conjunction.

# Arguments

  - $(arg_dict[:pr])

# Validation

  - At least one asset must be investable.

# Returns

  - `imsk::Option{BitVector}`: `true` at every investable asset, or `nothing` when every asset is investable.

# Examples

```jldoctest
julia> pr = prior(EmpiricalPrior(),
                  ReturnsResult(; nx = [\"a\", \"b\"], X = [0.1 -0.2; -0.1 0.2; 0.05 0.1]));

julia> isnothing(PortfolioOptimisers.investable_mask(pr))
true
```

# Related

  - [`LowOrderPrior`](@ref)
  - [`port_opt_view`](@ref)
  - [`IsEmptyError`](@ref)
"""
function investable_mask(pr::AbstractPriorResult)::Option{BitVector}
    imsk = isfinite.(pr.mu) .& isfinite.(LinearAlgebra.diag(pr.sigma))
    @argcheck(any(imsk),
              IsEmptyError("no asset of the prior result is investable: every asset carries a NaN in `mu` or on the diagonal of `sigma`. Check that the prior estimator received enough observations, and that the universe holds at least one active asset."))
    return all(imsk) ? nothing : imsk
end
"""
    investable_views(pr::AbstractPriorResult, sets::Nothing) -> Tuple
    investable_views(pr::AbstractPriorResult, sets::UniverseSets) -> Tuple

Derive the Investable Mask of a fitted prior, and give a view builder the universe on which it can write rows.

A view is a dense linear form over the asset axis. A departed asset carries `NaN` in `mu` and on the diagonal of `sigma`. `A[i] == 0` does not remove the `NaN` from a row, because `0 * NaN` is `NaN`. A view that names only active assets gets a `NaN` row, the same as a view that names the departed asset. The solver receives a row of `NaN`, and the fit fails with an error that names a wrong cause. The fix is to build the row on the investable columns only. Every optimiser takes the same reduction at its entry, [`port_opt_view`](@ref) of the prior result at `findall(imsk)`. That reduction gives the same bits as a prior reduced by hand.

The two prior families that take views both reduce here. Each does its own work after the reduction. A row of entropy pooling is over the observations, so its solved probabilities have no asset axis, and nothing expands back. The moments come from the wrapped prior after its new fit, which already holds the `NaN` values of the full universe. A Black-Litterman posterior is a pair of moments over the reduced assets. The estimator must write it back into a `NaN` frame of the full width with [`expand_moment`](@ref) before it returns. For this difference, this function returns the index to the caller.

This function also adds the Non-Investable Axis to the sets that it gives the builders, with [`non_investable_sets`](@ref) after [`port_opt_view`](@ref). The order matters, because the view drops the axis, so that a subproblem does not get the departures of its parent. With the axis, a builder can tell a departed name from a wrong name. It drops the row of a departed name with no message. For a wrong name, the `strict_diagnostic` rule applies.

The method selects on `sets` by dispatch, and on the mask by a condition, for these reasons:

  - `sets` is a field of a `@concrete` estimator. Whether it is `nothing` is a fact of its type, fixed for each instance. Two methods let the compiler select at no cost, and keep the returned sets a concrete [`UniverseSets`](@ref). One method over `Option{<:UniverseSets}` returns a `Union` of values. The view builders declare `sets::UniverseSets`, so JET reports a missing method for the `Nothing` part at every builder call site, though no call reaches it.
  - [`investable_mask`](@ref) returns a `Union{Nothing, BitVector}` that the data decides. Julia splits a `Union` of two types and compiles the two cases in this branch. A dispatch on it gives nothing more, and adds a unit to the file.

When `sets` is `nothing`, the method returns at once, whatever the mask. There is nothing to build and nothing to reduce, because the constructor of an estimator that takes views refuses `nothing` sets when it holds a view. When every asset is investable, the method returns its arguments unchanged. A fit with no gap then reads two vectors once and allocates nothing.

# Algorithm

 1. When `sets` is `nothing`, return `nothing`, `sets` and no departed names. The field type of the estimator selects this method.
 2. Otherwise, derive the Investable Mask from the fitted prior with [`investable_mask`](@ref). When it is `nothing`, return `nothing`, `sets` and no departed names.
 3. Read the asset universe from `sets.dict[sets.xkey]`, and check it against the mask.
 4. Read the departed names with [`non_investable_names`](@ref).
 5. Take a [`port_opt_view`](@ref) of `sets` at `findall(imsk)`, and add the Non-Investable Axis to it with [`non_investable_sets`](@ref). Return the mask, the new sets and the departed names.

# Arguments

  - $(arg_dict[:pr])
  - `sets`: The [`UniverseSets`](@ref) of the estimator, or `nothing`.

# Validation

  - `length(sets.dict[sets.xkey]) == length(imsk)`. Otherwise, raises a `DimensionMismatch` that names both counts, in place of the `BoundsError` that the complement raises.

# Returns

  - `(imsk, sets, ni)`: The Investable Mask or `nothing`, the reduced sets that carry the Non-Investable Axis, and the departed names. [`investable_prior`](@ref) takes its view at the mask, and [`expand_moment`](@ref) writes back through it. A caller that reduces and expands needs nothing more.

# Related

  - [`investable_prior`](@ref)
  - [`investable_mask`](@ref)
  - [`non_investable_sets`](@ref)
  - [`non_investable_names`](@ref)
  - [`announce_non_investable`](@ref)
  - [`expand_moment`](@ref)
  - [`port_opt_view`](@ref)
"""
function investable_views(::AbstractPriorResult, sets::Nothing)
    return nothing, sets, String[]
end
function investable_views(pr::AbstractPriorResult, sets::UniverseSets)
    imsk = investable_mask(pr)
    # A condition here, where the split above is dispatch, and the asymmetry is the point.
    # `sets` is a field of a `@concrete` estimator, so its `nothing` is a TYPE fact and the
    # method pair is static — and it has to be, or the builders' `sets::UniverseSets` has no
    # method for what this returns and every call site reds JET. `imsk` is a VALUE fact, a
    # `Union` Julia union-splits back into this very branch. See the docstring.
    if isnothing(imsk)
        return nothing, sets, String[]
    end
    nx = sets.dict[sets.xkey]
    @argcheck(length(nx) == length(imsk),
              DimensionMismatch("the asset universe `$(sets.xkey)` and the fitted prior disagree on how many assets there are. Got\nlength(sets.dict[$(sets.xkey)]) => $(length(nx))\nassets in the prior => $(length(imsk))"))
    ni = non_investable_names(nx, imsk)
    return imsk, non_investable_sets(port_opt_view(sets, findall(imsk)), ni), ni
end
"""
    investable_prior(imsk::Nothing, pr::AbstractPriorResult) -> AbstractPriorResult
    investable_prior(imsk::BitVector, pr::AbstractPriorResult) -> AbstractPriorResult

View a fitted prior at the Investable Mask that [`investable_views`](@ref) derived.

This function is separate from [`investable_views`](@ref) because a staged entropy pooling fit fits its wrapped prior again between stages, once for each solve. Each new fit needs a new view before the builders of the next stage read it. The mask does not change, because no new weights of the observations can estimate a column that had no estimate. So the caller derives the mask once and applies this function many times.

A mask of `nothing` means that every asset is investable, and the function returns the prior unchanged. A fit with no gap thus allocates nothing. Here two methods select on the mask, and [`investable_views`](@ref) uses a condition. The two agree. The mask comes from a local variable whose `Union` Julia already split at the call site, so each branch calls this function with a concrete argument. One method over `Option{BitVector}` would bring the `Union` back, so the function must not become one.

# Arguments

  - `imsk`: The Investable Mask, or `nothing` when every asset is investable.
  - $(arg_dict[:pr])

# Returns

  - `pr::AbstractPriorResult`: The prior over the investable assets, or the prior unchanged.

# Related

  - [`investable_views`](@ref)
  - [`port_opt_view`](@ref)
  - [`investable_mask`](@ref)
"""
function investable_prior(::Nothing, pr::AbstractPriorResult)
    return pr
end
function investable_prior(imsk::BitVector, pr::AbstractPriorResult)
    return port_opt_view(pr, findall(imsk))
end
"""
    investable_universe_names(sets::Nothing, imsk) -> VecStr
    investable_universe_names(sets::UniverseSets, imsk::Nothing) -> VecStr
    investable_universe_names(sets::UniverseSets, imsk::BitVector) -> VecStr

Name the departed assets for an estimator that reduces its asset axis, and whose views are on a different axis.

Every estimator whose views resolve against `xkey` calls [`investable_views`](@ref). That function reduces the sets, adds the Non-Investable Axis to them, and finds the names. [`BayesianBlackLittermanPrior`](@ref) and [`FactorBlackLittermanPrior`](@ref) write their views on the factor axis. They reduce their assets, and do not change a view axis. They have no axis to add. A call of `investable_views` would make them require an asset universe that they do not use for any other purpose. They still report departed assets, and this function gives them only the names that they need.

When the caller states no sets, the function returns an empty list and raises no error. [`announce_non_investable`](@ref) prints nothing for an empty list. This is the correct result for a valid configuration. Both estimators admit `sets` of `nothing`, because a precomputed [`BlackLittermanViews`](@ref) resolves no name and needs no universe. A stated universe that does not match this fit also returns an empty list. Neither estimator reads an asset name for any other purpose, so no other check reads the length of the universe.

# Arguments

  - `sets`: The estimator's [`UniverseSets`](@ref), or `nothing`.
  - `imsk`: The Investable Mask, or `nothing` when every asset is investable.

# Returns

  - `ni::VecStr`: The names the mask left out, or an empty vector.

# Related

  - [`investable_views`](@ref)
  - [`non_investable_names`](@ref)
  - [`announce_non_investable`](@ref)
  - [`BayesianBlackLittermanPrior`](@ref)
  - [`FactorBlackLittermanPrior`](@ref)
"""
function investable_universe_names(::Nothing, ::Any)::VecStr
    return String[]
end
function investable_universe_names(::UniverseSets, ::Nothing)::VecStr
    return String[]
end
function investable_universe_names(sets::UniverseSets, imsk::BitVector)::VecStr
    # The asset universe is the one axis [`UniverseSets`](@ref) makes mandatory, so it is
    # always there to read. Its *length* is not checked anywhere on these two members —
    # neither reads an asset name for any other purpose — and a universe that does not
    # describe this fit answers silence rather than a refusal, because this reads names for
    # a message and refusing a message is not its job.
    nx = sets.dict[sets.xkey]
    return length(nx) == length(imsk) ? non_investable_names(nx, imsk) : String[]
end
"""
    held_non_investable(imsk::BitVector, w::VecNum)
    held_non_investable(imsk::BitVector, w::VecVecNum)
    held_non_investable(imsk::BitVector, W::MatNum)

Find the assets a portfolio holds and the Investable Mask excludes.

A non-investable asset carries `NaN` in `mu` and on the diagonal of `sigma`, so it has no moment. The reduction of a portfolio that holds no such asset loses nothing. A portfolio that holds such an asset needs a report to the caller, and this function finds those assets. A weight path holds an asset when one of its rows holds it. A population holds an asset when one of its members holds it.

# Arguments

  - `imsk`: The Investable Mask, `true` at every asset whose prior moments were finite.
  - `w`: Portfolio weights, a population of them, or a weight path (observations × assets).

# Returns

  - `held::VecInt`: Indices of the held non-investable assets, empty when there are none.

# Related

  - [`investable_mask`](@ref)
  - [`investable_weights_view`](@ref)
  - [`attribution_investable_diagnostic`](@ref)
"""
function held_non_investable(imsk::BitVector, w::VecNum)
    return findall(i -> !imsk[i] && !iszero(w[i]), eachindex(imsk))
end
function held_non_investable(imsk::BitVector, w::VecVecNum)
    return findall(i -> !imsk[i] && any(wi -> !iszero(wi[i]), w), eachindex(imsk))
end
function held_non_investable(imsk::BitVector, W::MatNum)
    return findall(i -> !imsk[i] && any(!iszero, view(W, :, i)), eachindex(imsk))
end
"""
    investable_weights_view(imsk::Nothing, w)
    investable_weights_view(imsk::BitVector, w::Nothing)
    investable_weights_view(imsk::BitVector, w::VecNum)
    investable_weights_view(imsk::BitVector, w::VecVecNum)
    investable_weights_view(imsk::BitVector, w::MatNum)

Take the view of the weights at the Investable Mask.

A weight vector is one cross-section, so the mask selects its entries. A weight path has one row of weights for each observation. The mask selects its columns, and every row keeps its own observation. The function reduces a population one member at a time. The asset return history of a fold has the shape of a path, observations × assets. The consumers of a [`HeldWeightsResult`](@ref) that take a fold use the same view of it, so this one column selection serves both.

When either argument is `nothing`, the function returns the argument `w` that it received. An `imsk` of `nothing` means that every asset is investable, so the function selects nothing and returns the weights unchanged. A `w` of `nothing` means that the caller gave no weights. [`equilibrium_mu`](@ref) then uses equal weights over the axis that it receives. There is nothing to reduce, and a reduction would need a length that does not exist. Dispatch selects both cases, so they cost nothing.

The `w` of a prior estimator is a per-asset configuration over the full universe of the caller. It meets a reduced axis for the same reason as a per-asset bound. The function was first written for the weights that an optimisation returns. A prior also reduces its own weights with it, because the two reductions are one operation.

# Arguments

  - `imsk`: The Investable Mask, `true` at every asset whose prior moments were finite, or `nothing` when every asset is investable.
  - `w`: Portfolio weights, a population of them, a weight path (observations × assets), or `nothing`.

# Returns

  - The view of `w` at the investable assets, or `w` itself when either argument is `nothing`.

# Related

  - [`held_non_investable`](@ref)
  - [`investable_mask`](@ref)
  - [`investable_views`](@ref)
  - [`equilibrium_mu`](@ref)
"""
function investable_weights_view(::Nothing, w)
    return w
end
function investable_weights_view(::BitVector, ::Nothing)
    return nothing
end
function investable_weights_view(imsk::BitVector, w::VecNum)
    return view(w, imsk)
end
function investable_weights_view(imsk::BitVector, w::VecVecNum)
    return [view(wi, imsk) for wi in w]
end
function investable_weights_view(imsk::BitVector, w::MatNum)
    return view(w, :, imsk)
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the coskewness and cokurtosis a high order prior estimator produced, over the low order prior it wraps.

Every high order prior estimator returns a `HighOrderPrior`. It wraps a low order prior result, and holds the cokurtosis matrix, the duplication, elimination and summation matrices, the coskewness matrix, and the negative spectral coskewness. It also holds the matrix processing estimator of the coskewness, and the factor co-moments. The constructor checks every field.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HighOrderPrior(;
        pr::AbstractPriorResult,
        kt::Option{<:MatNum} = nothing,
        D2::Option{<:MatNum} = nothing,
        L2::Option{<:MatNum} = nothing,
        S2::Option{<:MatNum} = nothing,
        sk::Option{<:MatNum} = nothing,
        V::Option{<:MatNum} = nothing,
        skmp::Option{<:AbstractMatrixProcessingEstimator} = nothing,
        eoc::Tuple{Option{<:Real}, Option{<:Real}} = (nothing, nothing),
        fpr::Option{<:HighOrderPrior} = nothing
    ) -> HighOrderPrior

Keywords correspond to the struct's fields.

## The counts of the moments

The field `eoc` holds the effective number of observations of the coskewness and of the cokurtosis. The property `pr.eoc` reads four counts. The first two are the `eoc` of `pr`, the counts of `mu` and `sigma`. So `pr.eoc[k]` is the count of moment `k` on both result types, and [`moment_observation_count`](@ref) reads it. A read of the field itself, with `getfield`, gives the pair. [`reconstruct_prior`](@ref) reads the fields so, and keeps the pair.

## The factor block

A high order prior that a factor model fits carries factor co-moments as well as asset co-moments. The field `fpr` holds them as a nested `HighOrderPrior`, and not as the flat fields `f_kt`, `f_sk` and `f_V`. The factor block thus has every field of a `HighOrderPrior`. These are `D2`, `L2`, `S2` and `skmp` as well as `kt`, `sk` and `V`. A field added later also applies to it with no second edit. The flat names still read the nested block. `pr.f_kt`, `pr.f_sk` and `pr.f_V` return `fpr.kt`, `fpr.sk` and `fpr.V`, or `nothing` when there is no factor block. `pr.f_D2`, `pr.f_L2`, `pr.f_S2` and `pr.f_skmp` work in the same way.

`fpr.pr` is the factor block at the low order, the [`LowOrderPrior`](@ref) over the factors. `pr.pr.fpr`, the factor block of the low order result, is the same distribution. The constructor requires that the two are the same object, see the validation below.

`fpr` is a field of the `HighOrderPrior`, so a read of `fpr` finds the field before the `forward(pr)` block. It names the high order factor block. Before the nested block existed, it named the low order factor block. Reads through it give the same values. The nested `HighOrderPrior` forwards to its own `pr`, which the constructor requires to be `pr.fpr`. So `hop.fpr.mu` is the factor mean in both cases, and `hop.fpr` is the factor prior at this order.

### Which read to use

Read the factor block as `pr.fpr.kt`, for the reasons that [`LowOrderPrior`](@ref) gives. The seven flat names here are `f_kt`, `f_sk`, `f_V`, `f_D2`, `f_L2`, `f_S2` and `f_skmp`, and the set does not grow. A field added to `HighOrderPrior` later is available as `pr.fpr.<name>`, and gets no flat name.

The two reads are different when there is no factor block. `pr.f_kt` returns `nothing`, and `pr.fpr.kt` raises an error. Check the block first, and then read through `fpr`.

## Validation

Let `N = length(pr.mu)`.

  - If one of `kt`, `L2` or `S2` is given, all three must be given and not empty, with `size(kt) == (N^2, N^2)` and `size(L2) == size(S2) == (div(N * (N + 1), 2), N^2)`.
  - If `sk` or `V` is given, both must be given and not empty, with `size(sk) == (N, N^2)` and `size(V) == (N, N)`.
  - If the three matrices of the first rule and `sk` are given, `D2` must be given and not empty, with `size(D2) == size(transpose(L2))`. `D2` has no other rule. It is the one moment field that the constructor admits alone, and a `HighOrderPrior` that holds only `D2` is valid.
  - If `fpr` is given, `pr.fpr` must be given, and `fpr.pr === pr.fpr`. The factor co-moments are computed against the factor block of the low order result, not against a copy of it. The opposite rule does not apply. A low order factor block with no factor co-moments is usual, so `fpr === nothing` always passes. The constructor of the factor block checks the inside of the block, such as its own shapes against its own `N`.

## View parameters

`HighOrderPrior` defines its own [`port_opt_view`](@ref) method, and does not make one from field tags.

  - It reads no argument after `i`. It takes and ignores more positional arguments.
  - `pr` recurses through [`port_opt_view`](@ref) with `i`, which cuts every low order field.
  - A fourth-moment index derived from `i` indexes `kt`, and `i` does not. `kt` is ``N^2 \\times N^2`` over ordered pairs of assets, so the asset index does not address it.
  - `sk` is cut by `i` on its asset axis, and by the same fourth-moment index on its pair axis.
  - `V` is computed again from the cut `sk`, and is not cut. It is a spectral quantity of the coskewness matrix, so the submatrix of `V` is not the `V` of the submatrix.
  - `D2`, `L2` and `S2` are built again for the number of assets of the subproblem, and are not cut. They depend only on that number, and hold no data of a specific asset.
  - `skmp` passes through unchanged. It is the matrix processing estimator that the new computation of `V` reads.
  - `eoc` passes through unchanged. It is on the observation axis, and `i` indexes assets.
  - `fpr` passes through unchanged, because it holds co-moments over factors and not over assets. The view forwards the same object, so `fpr.pr === pr.fpr` stays true. The low order view forwards its own factor block in the same way.

# Examples

```jldoctest
julia> HighOrderPrior(;
                      pr = LowOrderPrior(; X = [0.01 0.02; 0.03 0.04], mu = [0.02, 0.03],
                                         sigma = [0.0001 0.0002; 0.0002 0.0003]), kt = rand(4, 4),
                      D2 = PortfolioOptimisers.duplication_matrix(2),
                      L2 = PortfolioOptimisers.elimination_matrix(2),
                      S2 = PortfolioOptimisers.summation_matrix(2), sk = rand(2, 4),
                      V = rand(2, 2))
HighOrderPrior
    pr ┼ LowOrderPrior
       │         X ┼ 2×2 Matrix{Float64}
       │       o_X ┼ nothing
       │        mu ┼ Vector{Float64}: [0.02, 0.03]
       │     sigma ┼ 2×2 Matrix{Float64}
       │      chol ┼ nothing
       │         w ┼ nothing
       │       ens ┼ nothing
       │       eoc ┼ Tuple{Nothing, Nothing}: (nothing, nothing)
       │       kld ┼ nothing
       │        ow ┼ nothing
       │        rr ┼ nothing
       │       fpr ┼ nothing
       │     x_src ┼ nothing
       │   ss_gaps ┼ nothing
       │        pc ┴ nothing
    kt ┼ 4×4 Matrix{Float64}
    D2 ┼ 4×3 SparseArrays.SparseMatrixCSC{Int64, Int64}
    L2 ┼ 3×4 SparseArrays.SparseMatrixCSC{Int64, Int64}
    S2 ┼ 3×4 SparseArrays.SparseMatrixCSC{Int64, Int64}
    sk ┼ 2×4 Matrix{Float64}
     V ┼ 2×2 Matrix{Float64}
  skmp ┼ nothing
   eoc ┼ NTuple{4, Nothing}: (nothing, nothing, nothing, nothing)
   fpr ┴ nothing
```

# Related

  - [`AbstractPriorResult`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`HighOrderPriorEstimator`](@ref)
  - [`prior`](@ref)
  - [`forward_prior`](@ref)
  - [`reconstruct_prior`](@ref)
  - [`port_opt_view`](@ref)
"""
@concrete struct HighOrderPrior <: AbstractPriorResult
    """
    $(field_dict[:pr])
    """
    pr
    """
    $(field_dict[:kt])
    """
    kt
    """
    $(field_dict[:D2])
    """
    D2
    """
    $(field_dict[:L2])
    """
    L2
    """
    $(field_dict[:S2])
    """
    S2
    """
    $(field_dict[:sk])
    """
    sk
    """
    $(field_dict[:V])
    """
    V
    """
    $(field_dict[:skmp])
    """
    skmp
    """
    $(field_dict[:eoc_hop])
    """
    eoc
    """
    $(field_dict[:fpr])
    """
    fpr
    function HighOrderPrior(pr::AbstractPriorResult, kt::Option{<:MatNum},
                            D2::Option{<:MatNum}, L2::Option{<:MatNum},
                            S2::Option{<:MatNum}, sk::Option{<:MatNum}, V::Option{<:MatNum},
                            skmp::Option{<:AbstractMatrixProcessingEstimator},
                            eoc::Tuple{Option{<:Real}, Option{<:Real}},
                            fpr::Option{<:HighOrderPrior})
        N = length(pr.mu)
        sk_flag = isa(sk, MatNum)
        kt_flag = isa(kt, MatNum)
        L2_flag = isa(L2, MatNum)
        S2_flag = isa(S2, MatNum)
        if kt_flag || L2_flag || S2_flag
            @argcheck(kt_flag,
                      ArgumentError("kt must be provided when L2 or S2 is provided, isa(kt, MatNum) = $(kt_flag), isa(L2, MatNum) = $(L2_flag), isa(S2, MatNum) = $(S2_flag)"))
            @argcheck(L2_flag,
                      ArgumentError("L2 must be provided when kt or S2 is provided, isa(kt, MatNum) = $(kt_flag), isa(L2, MatNum) = $(L2_flag), isa(S2, MatNum) = $(S2_flag)"))
            @argcheck(S2_flag,
                      ArgumentError("S2 must be provided when kt or L2 is provided, isa(kt, MatNum) = $(kt_flag), isa(L2, MatNum) = $(L2_flag), isa(S2, MatNum) = $(S2_flag)"))
            @argcheck(!isempty(kt),
                      IsEmptyError("$(err_name_dict[:kt]) (`kt`) cannot be empty"))
            @argcheck(!isempty(L2),
                      IsEmptyError("$(err_name_dict[:L2]) (`L2`) cannot be empty"))
            @argcheck(!isempty(S2),
                      IsEmptyError("$(err_name_dict[:S2]) (`S2`) cannot be empty"))
            @argcheck(size(kt) == (N^2, N^2),
                      DimensionMismatch("size(kt) ($(size(kt))) must be ($(N^2), $(N^2))"))
            @argcheck(size(L2) == size(S2) == (div(N * (N + 1), 2), N^2),
                      DimensionMismatch("size(L2) ($(size(L2))) and size(S2) ($(size(S2))) must be ($(div(N * (N + 1), 2)), $(N^2))"))
            if sk_flag
                @argcheck(isa(D2, MatNum),
                          ArgumentError("D2 must be provided when sk is provided, isa(D2, MatNum) = $(isa(D2, MatNum)), isa(sk, MatNum) = $(sk_flag)"))
                @argcheck(!isempty(D2),
                          IsEmptyError("$(err_name_dict[:D2]) (`D2`) cannot be empty"))
                @argcheck(size(D2) == size(transpose(L2)),
                          DimensionMismatch("size(D2) = $(size(D2)) must match size(L2') = $(size(transpose(L2)))"))
            end
        end
        V_flag = isa(V, MatNum)
        if sk_flag || V_flag
            @argcheck(sk_flag,
                      ArgumentError("sk must be provided when V is provided, isa(sk, MatNum) = $(sk_flag), isa(V, MatNum) = $(V_flag)"))
            @argcheck(V_flag,
                      ArgumentError("V must be provided when sk is provided, isa(sk, MatNum) = $(sk_flag), isa(V, MatNum) = $(V_flag)"))
            @argcheck(!isempty(sk),
                      IsEmptyError("$(err_name_dict[:sk]) (`sk`) cannot be empty"))
            @argcheck(!isempty(V),
                      IsEmptyError("$(err_name_dict[:V]) (`V`) cannot be empty"))
            @argcheck(size(V) == (N, N),
                      DimensionMismatch("size(V) = $(size(V)) must be ($N, $N)"))
            @argcheck(size(sk) == (N, N^2),
                      DimensionMismatch("size(sk) = $(size(sk)) must be ($N, $(N^2))"))
        end
        # The factor low-order prior is reachable two ways once the factor co-moments nest:
        # `hop.fpr.pr` and `hop.pr.fpr`. They are the same distribution, so they must be the
        # same object — otherwise `hop.fpr.mu` and `hop.f_mu` could disagree, and nothing
        # downstream would say which is the prior the co-moments were computed against.
        # Everything internal to the block is validated by its own constructor, against its
        # own `N = length(fpr.pr.mu)`, which is why no factor shape is restated here.
        if !isnothing(fpr)
            inner = pr.fpr
            @argcheck(!isnothing(inner),
                      IsNothingError("factor co-moments (`fpr`) describe the same factors as the low order prior's own factor block, but the wrapped prior has none: `pr.fpr === nothing`. A `HighOrderPrior` only carries factor co-moments over a prior that already carries a factor distribution — fit it through a factor-based estimator such as `FactorPrior`."))
            @argcheck(fpr.pr === inner,
                      ConflictingArgumentError("the factor low order prior is reachable two ways, as `fpr.pr` and as `pr.fpr`, and they must be the same object, but they differ. `fpr.pr` is the distribution the factor co-moments were computed against, so a mismatch would make `hop.fpr.mu` and `hop.f_mu` disagree with no way to tell which is right. Build the nested block from the wrapped prior's own factor block: `HighOrderPrior(; pr = pr.fpr, kt = ...)`."))
        end
        assert_moment_counts(eoc)
        fs = (pr, kt, D2, L2, S2, sk, V, skmp, eoc, fpr)
        return new{map(typeof, fs)...}(fs...)
    end
end
function HighOrderPrior(; pr::AbstractPriorResult, kt::Option{<:MatNum} = nothing,
                        D2::Option{<:MatNum} = nothing, L2::Option{<:MatNum} = nothing,
                        S2::Option{<:MatNum} = nothing, sk::Option{<:MatNum} = nothing,
                        V::Option{<:MatNum} = nothing,
                        skmp::Option{<:AbstractMatrixProcessingEstimator} = nothing,
                        eoc::Tuple{Option{<:Real}, Option{<:Real}} = (nothing, nothing),
                        fpr::Option{<:HighOrderPrior} = nothing)::HighOrderPrior
    return HighOrderPrior(pr, kt, D2, L2, S2, sk, V, skmp, eoc, fpr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of a [`HighOrderPrior`](@ref) restricted to the assets at index `i`, with every moment matrix cut to match.

The view forwards the factor block with no cut, as on [`LowOrderPrior`](@ref), because `i` indexes assets and `fpr` holds co-moments over factors. The view forwards the same object, so `fpr.pr === pr.fpr` stays true. The low order view forwards its own factor block in the same way.

# Algorithm

 1. Make `idx` with [`fourth_moment_index_generator`](@ref) against the full number of assets of `pr`. `idx` is the fourth-moment index that addresses the co-moment matrices of the assets at `i`.
 2. Cut the coskewness matrix to `i` on its asset axis and to `idx` on its pair axis, with [`nothing_scalar_array_view_odd_order`](@ref). Call it `sk`. A prior result that holds none keeps `nothing`.
 3. Compute `V` again from the `sk` of step 2 and the cut returns matrix, with [`negative_spectral_coskewness`](@ref), the `skmp` of `pr` and the count of the coskewness from [`moment_observation_count`](@ref). `V` is a spectral quantity of the coskewness matrix, so the step builds it again and does not cut it. When step 2 gave `nothing`, `V` is `nothing`.
 4. Build `D2`, `L2` and `S2` again for the number of assets of the subproblem with [`dup_elim_sum_view`](@ref), and do not cut them. When `pr` holds `D2`, take all three. When it holds `S2` and no `D2`, take `L2` and `S2`, and keep `D2` as `nothing`. When it holds neither, take none.
 5. Rebuild the prior result through its keyword constructor. Recurse `pr` through [`port_opt_view`](@ref) with `i`, index `kt` by `idx`, take the values of steps 2 to 4, and forward `skmp`, the field `eoc` and `fpr` unchanged. Every `@argcheck` of the constructor thus runs on the view.

# Arguments

  - $(arg_dict[:pr])
  - `i`: Asset indices the view keeps.
  - `args...`: Additional arguments (ignored).

# Returns

  - `pr::HighOrderPrior`: The prior result restricted to the assets at `i`.

# Related

  - [`HighOrderPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`port_opt_view`](@ref)
  - [`fourth_moment_index_generator`](@ref)
  - [`dup_elim_sum_view`](@ref)
  - [`negative_spectral_coskewness`](@ref)
"""
function port_opt_view(pr::HighOrderPrior, i, args...)
    idx = fourth_moment_index_generator(length(pr.mu), i)
    kt = pr.kt
    sk = pr.sk
    skmp = pr.skmp
    sk = nothing_scalar_array_view_odd_order(sk, i, idx)
    if !isnothing(sk)
        V = negative_spectral_coskewness(sk, view(pr.X, :, i), skmp,
                                         moment_observation_count(pr, 3))
    else
        V = nothing
    end
    if !isnothing(pr.D2)
        D2, L2, S2 = dup_elim_sum_view(kt, length(i))
    elseif !isnothing(pr.S2)
        D2 = nothing
        L2, S2 = dup_elim_sum_view(kt, length(i))[2:3]
    else
        D2, L2, S2 = (nothing, nothing, nothing)
    end
    return HighOrderPrior(; pr = port_opt_view(pr.pr, i),
                          kt = nothing_scalar_array_view(kt, idx), D2 = D2, L2 = L2,
                          S2 = S2, sk = sk, V = V, skmp = skmp, eoc = getfield(pr, :eoc),
                          fpr = pr.fpr)
end
# The flat `f_`-prefixed names are virtual reads of the nested factor block, mirroring
# [`LowOrderPrior`](@ref): code written against the pre-nesting shape is unaffected, and
# `f_D2`/`f_L2`/`f_S2`/`f_skmp` come for free. `compute` with a lambda rather than a dotted
# locator, because a dotted locator throws a [`PropertyPathError`](@ref) on a `nothing` node
# where these must return `nothing` — that is what the old flat fields did with no factor
# block. They are declared before `forward(pr)` only for reading order: the embedded
# `LowOrderPrior` has no `f_kt`/`f_sk`/`f_V` of its own to shadow.
#
# `fpr` is a field of the `HighOrderPrior` itself, so it resolves before `forward(pr)` and
# names the *high* order factor block rather than the low order one. Reads through it are
# unaffected by the shift: the nested `HighOrderPrior` forwards to its own `pr`, which the
# constructor pins to `pr.fpr`, so `hop.fpr.mu` is the factor mean either way.
#
# ForwardSelection of the remaining unknown property names to the embedded `pr` prior gives
# transparent access to the low-order moment fields (see [`@forward_properties`](@ref)).
#
# `eoc` is a `swap`: the field holds the counts of the two higher moments, and the property
# reads the four counts of the result, with the two of the low order result first. So
# `pr.eoc[k]` indexes moment `k` on either result type. A rebuild reads the field with
# `getfield`, so it keeps the pair.
@forward_properties HighOrderPrior begin
    swap(eoc, obj -> (obj.pr.eoc..., getfield(obj, :eoc)...))
    compute(f_kt, obj -> isnothing(obj.fpr) ? nothing : obj.fpr.kt)
    compute(f_sk, obj -> isnothing(obj.fpr) ? nothing : obj.fpr.sk)
    compute(f_V, obj -> isnothing(obj.fpr) ? nothing : obj.fpr.V)
    compute(f_D2, obj -> isnothing(obj.fpr) ? nothing : obj.fpr.D2)
    compute(f_L2, obj -> isnothing(obj.fpr) ? nothing : obj.fpr.L2)
    compute(f_S2, obj -> isnothing(obj.fpr) ? nothing : obj.fpr.S2)
    compute(f_skmp, obj -> isnothing(obj.fpr) ? nothing : obj.fpr.skmp)
    forward(pr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuild a prior result through its keyword constructor, with the fields that `patch` names replaced.

Each prior result type has one method, because the method names the constructor of the type and does not find it by reflection. A generic method needs `Base.typename(T).wrapper` or a dependency on `ConstructionBase`, and neither gives more. [`prior_field_values`](@ref) already reads the field list from the type, so a result type that gains a field needs no change here. Only a new prior result type needs a method. Until it has one, a call raises a `MethodError` that names this function, and no generic code rebuilds a type that it does not know.

The rebuild runs all the checks of the type, which is the purpose of the call to the constructor. A patch that leaves the result inconsistent raises the same error as a hand-written constructor call. Keyword arguments have no order, so `patch` can name fields in any order.

This file defines these methods after both prior result types, because they dispatch on the concrete types.

# Algorithm

Both methods do the same three steps. Only the constructor of step 3 is different.

 1. Read the fields of `pr` into a named tuple with [`prior_field_values`](@ref), keyed in declaration order.
 2. Merge `patch` over that tuple. A field that `patch` names takes the value of the patch, and every other field keeps the value of `pr`.
 3. Splat the merged tuple into the keyword constructor of the type, `LowOrderPrior` in the first method and `HighOrderPrior` in the second. Return the prior result that it builds. Every `@argcheck` of that constructor runs on the merged values.

# Arguments

  - `pr`: Prior result to rebuild.
  - `patch`: Named tuple of field overrides. Every name must be a field of `typeof(pr)`. [`forward_prior`](@ref) checks the names before the call, so its error names the rule, and not an unknown keyword.

# Returns

  - `pr::AbstractPriorResult`: Reconstructed result of the same type.

# Related

  - [`forward_prior`](@ref)
  - [`prior_field_values`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
"""
function reconstruct_prior(pr::LowOrderPrior, patch::NamedTuple)::LowOrderPrior
    return LowOrderPrior(; merge(prior_field_values(pr), patch)...)
end
function reconstruct_prior(pr::HighOrderPrior, patch::NamedTuple)::HighOrderPrior
    return HighOrderPrior(; merge(prior_field_values(pr), patch)...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the field names of the two prior result types, with no duplicates.

[`propagatable_contract_violations`](@ref) checks an `@pprop` field name against this pool. The generated
`factory(x, pr::AbstractPriorResult, args...)` reads `getproperty(pr, :field)`, and the declaration does
not know which prior result type arrives.

The function names the two prior result types, as [`reconstruct_prior`](@ref) does, and reads their
fields from the types. So a result type that gains a field needs no change here. `HighOrderPrior`
forwards all of the `pr` that it wraps, so the low order names are properties of it but not fields.
For this reason, the field names of one result type are not the pool.

The pool holds field names only. A computed property, such as `f_mu`, `f_kt` or `original_X`, is not
in it, so an `@pprop` field cannot name one.

This file defines the function after both prior result types, because it names the concrete types.

# Algorithm

 1. Concatenate the field names of [`LowOrderPrior`](@ref) and of [`HighOrderPrior`](@ref) into one vector of `Symbol`.
 2. Remove the duplicates in place, and return the vector. `fpr` is a field of both types, so the concatenation has a duplicate.

# Returns

  - `pool::Vector{Symbol}`: The field names of [`LowOrderPrior`](@ref) and of [`HighOrderPrior`](@ref), with no duplicates.

# Related

  - [`propagatable_contract_violations`](@ref)
  - [`check_propagatable_contracts`](@ref)
  - [`@pprop`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
"""
function prior_result_property_pool()
    return unique!(Symbol[fieldnames(LowOrderPrior)..., fieldnames(HighOrderPrior)...])
end

"""
    shared_series_record(pr::AbstractPriorResult, rows::Nothing, T::Integer)
    shared_series_record(pr::LowOrderPrior, rows::VecInt, T::Integer)
    shared_series_record(pr::HighOrderPrior, rows::VecInt, T::Integer)
    shared_series_record(pr::AbstractPriorResult, state::SampleBufferState)

Record on a prior the observations of the returns data that its fit kept after a Shared Series cut.

A fit drops each observation with a Shared Series gap before it starts, see [`shared_series_view`](@ref) and [`shared_series_buffer`](@ref). After the cut, no part of the prior states which observation of the data each of its rows is. The function writes two records:

  - `ss_gaps`, the marks of the dropped observations, one for each of the `T` observations of the data. A reader that holds the data pairs the rows of `X` with its own rows through the marks, see [`shared_series_kept_rows`](@ref).
  - The row key of the factor model block. The batch verb counts the position of each row among the observations that it kept, and kept row `k` is observation `rows[k]` of the data. The function replaces each position with its observation, so that a realised [`factor_attribution`](@ref) of a cross-validation over the same data matches the correct rows. A key of positions matches each row after a gap to the next row, and raises no error. The timestamps come from the kept rows, so they need no change.

[`prior`](@ref) calls it after it fits on the returns data, and after it reads out a sample buffer. An optimiser calls it on the prior that it fits after its own cut, see [`shared_series_solve`](@ref).

# Algorithm

Julia selects the method by the types of the arguments. With `rows`, the method does these steps:

 1. Mark every one of the `T` observations as dropped. Call the marks `ss_gaps`.
 2. Remove the mark from the observations that the fit kept. A prior that an optimiser fits on its cut data can already hold marks of the `length(rows)` kept observations. Then remove the mark only from the `rows` at the positions that those marks keep, so that the two cuts combine.
 3. Map the row key of the factor model block through `rows` with [`map_row_positions`](@ref). Forward the key and `ss_gaps` onto `pr` with [`forward_prior`](@ref). A [`HighOrderPrior`](@ref) forwards the record of the `LowOrderPrior` that it wraps.

# Arguments

  - `pr`: The prior that the fit returned.
  - `rows`: The observations that the fit kept, from [`shared_series_rows`](@ref), or `nothing` when it kept every observation.
  - `T`: The number of observations of the data that the fit received.
  - `state`: The sample buffer that a new fit read. Its kept rows are those of [`buffer_kept_rows`](@ref), and its observations are its `n` rows.

# Validation

  - The rules of [`assert_shared_series_gaps`](@ref) on the marks.

# Returns

  - `pr::AbstractPriorResult`: `pr` itself when the fit kept every observation, and otherwise `pr` with the marks and with the positions of its block mapped through `rows`.

# Related

  - [`shared_series_kept_rows`](@ref)
  - [`map_row_positions`](@ref)
  - [`buffer_row_key`](@ref)
  - [`attribution_row_key`](@ref)
"""
function shared_series_record(pr::AbstractPriorResult, ::Nothing, ::Integer)
    return pr
end
function shared_series_record(pr::LowOrderPrior, rows::VecInt, T::Integer)
    ss_gaps = trues(T)
    kept = shared_series_kept_rows(pr, length(rows))
    ss_gaps[isnothing(kept) ? rows : view(rows, kept)] .= false
    # The positions are not part of the projection that `o_X` records, so `o_X` stays.
    return forward_prior(pr; rr = map_row_positions(pr.rr, rows), o_X = pr.o_X,
                         ss_gaps = ss_gaps)
end
function shared_series_record(pr::HighOrderPrior, rows::VecInt, T::Integer)
    return forward_prior(pr; pr = shared_series_record(pr.pr, rows, T))
end
function shared_series_record(pr::AbstractPriorResult, state::SampleBufferState)
    return shared_series_record(pr, buffer_kept_rows(state), state.n)
end
"""
    assert_shared_series_gaps(ss_gaps::Nothing, n::Integer)
    assert_shared_series_gaps(ss_gaps::AbstractVector{Bool}, n::Integer)

Check the Shared Series marks `ss_gaps` of a [`LowOrderPrior`](@ref) against the number of rows `n` of its `X`.

# Validation

  - At least one mark is `true`. Otherwise, raises an `ArgumentError`. A prior that dropped no observation records `nothing`. So the returns data after the cut always holds fewer observations than the marks, and [`shared_series_kept_rows`](@ref) tells the two apart by their length.
  - The marks keep at least `n` observations, because the rows of `X` are the last observations that the marks keep. Otherwise, raises a `DimensionMismatch`.

# Returns

  - `nothing`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`shared_series_record`](@ref)
"""
function assert_shared_series_gaps(::Nothing, ::Integer)
    return nothing
end
function assert_shared_series_gaps(ss_gaps::AbstractVector{Bool}, n::Integer)
    @argcheck(any(ss_gaps),
              ArgumentError("ss_gaps marks no observation. A prior that dropped no observation for a Shared Series gap records `nothing`, so that a reader can tell the returns data that the prior read from the returns data after the same cut"))
    k = count(!, ss_gaps)
    @argcheck(k >= n,
              DimensionMismatch("ss_gaps keeps $k observation(s), and X holds $n row(s). The rows of X are the last observations that ss_gaps keeps, so the marks must keep at least size(X, 1) observations"))
    return nothing
end
"""
    shared_series_kept_rows(pr, T::Integer) -> Option{Vector{Int}}

Return the observations that a prior kept after a Shared Series cut, when the returns data of a reader is the data that the prior read.

A reader holds `T` observations of the returns data. When `T` equals the number of marks in `pr.ss_gaps`, the reader holds the data that the prior read. The rows of `pr.X` are then the last of the observations that the marks keep. Otherwise, the reader holds the data after the same cut, or data with no record. The rows of `pr.X` are then its last observations, which was the rule of every reader before the record existed. The marks are `nothing` when the prior dropped no observation, so the cut data never has as many observations as the marks.

# Arguments

  - `pr`: The prior result, the returns data, a prior estimator, or `nothing`. Only a prior result holds marks: a [`LowOrderPrior`](@ref) in its field `ss_gaps`, and a [`HighOrderPrior`](@ref) in the low order prior that it wraps.
  - `T`: The number of observations of the returns data that the reader holds.

# Returns

  - `rows::Option{Vector{Int}}`: The positions of the kept observations in the data of the reader, or `nothing` when the reader pairs the last rows.

# Related

  - [`shared_series_record`](@ref)
  - [`implied_volatility_picker`](@ref)
  - [`optimiser_shared_series_rows`](@ref)
"""
function shared_series_kept_rows(pr::LowOrderPrior, T::Integer)
    return shared_series_kept_rows(getfield(pr, :ss_gaps), T)
end
function shared_series_kept_rows(pr::HighOrderPrior, T::Integer)
    return shared_series_kept_rows(getfield(pr, :pr), T)
end
function shared_series_kept_rows(ss_gaps::AbstractVector{Bool}, T::Integer)
    return length(ss_gaps) == T ? findall(!, ss_gaps) : nothing
end
function shared_series_kept_rows(::Any, ::Integer)
    return nothing
end
"""
    prior_data_rows(pr, T::Integer, n::Integer) -> AbstractVector{Int}

Return the observations of the returns data of a reader that pair the last `n` rows of the `X` of a prior.

A fit drops rows at the start, for a Scenario Cap and for the warm-up of a [`CrossSectionalFactorPrior`](@ref). It also drops the observations of a Shared Series gap. When the reader holds the data that the prior read, the rows of `X` are the last of the observations that the marks `pr.ss_gaps` keep, see [`shared_series_kept_rows`](@ref). Otherwise, they are the last `n` observations of the data of the reader.

# Arguments

  - `pr`: The prior result, or the returns data that a reader reads in its place.
  - `T`: The number of observations of the returns data that the reader holds.
  - `n`: The number of rows to pair, at most the number of rows of `pr.X`.

# Returns

  - `rows::AbstractVector{Int}`: The positions of the paired observations in the data of the reader, in order.

# Related

  - [`shared_series_kept_rows`](@ref)
  - [`implied_volatility_picker`](@ref)
  - [`outer_prior_rows`](@ref)
"""
function prior_data_rows(pr, T::Integer, n::Integer)
    rows = shared_series_kept_rows(pr, T)
    return if isnothing(rows)
        ((T - n + 1):T)
    else
        view(rows, (length(rows) - n + 1):length(rows))
    end
end
"""
    map_row_positions(rr::Option{<:AbstractLoadingsRegressionResult}, rows::VecInt)
    map_row_positions(rr::Regression, rows::VecInt)
    map_row_positions(rr::CrossSectionalFactorModel, rows::VecInt)

Return a factor model block whose positions are mapped through `rows`, and every other field unchanged.

Position `k` becomes `rows[k]`, with [`kept_row_positions`](@ref). The function returns unchanged a block that records no row key, and returns `nothing` when there is no block.

# Arguments

  - `rr`: The factor model block, or `nothing`.
  - `rows`: The observation of each position.

# Returns

  - `rr`: The block with its positions mapped, or `rr` itself.

# Related

  - [`shared_series_record`](@ref)
  - [`forget_row_positions`](@ref)
  - [`set_row_key`](@ref)
"""
function map_row_positions(rr::Option{<:AbstractLoadingsRegressionResult}, ::VecInt)
    return rr
end
function map_row_positions(rr::Regression, rows::VecInt)
    return set_row_key(rr, kept_row_positions(rows, rr.idx), rr.ts)
end
function map_row_positions(rr::CrossSectionalFactorModel, rows::VecInt)
    # Every field is read with `getfield`, for the reason `forget_row_positions` states.
    names = fieldnames(typeof(rr))
    fields = NamedTuple{names}(map(k -> getfield(rr, k), names))
    return CrossSectionalFactorModel(; fields...,
                                     idx = kept_row_positions(rows, getfield(rr, :idx)))
end
"""
    kept_row_positions(rows::Nothing, pos::Option{<:VecInt})
    kept_row_positions(rows::VecInt, pos::Nothing)
    kept_row_positions(rows::VecInt, pos::VecInt)

Return the observation of each position among the kept rows.

Kept row `k` is observation `rows[k]`. When `rows` is `nothing`, the fit kept every observation, so the positions are the observations. When `pos` is `nothing`, there is nothing to map.

# Arguments

  - `rows`: The kept observations, or `nothing` when the fit kept every observation.
  - `pos`: The positions among the kept rows, or `nothing`.

# Returns

  - `idx::Option{<:VecInt}`: `rows[pos]`, `pos` itself when `rows` is `nothing`, or `nothing` when `pos` is `nothing`.

# Related

  - [`map_row_positions`](@ref)
  - [`buffer_kept_rows`](@ref)
"""
function kept_row_positions(::Nothing, pos::Option{<:VecInt})
    return pos
end
function kept_row_positions(::VecInt, ::Nothing)
    return nothing
end
function kept_row_positions(rows::VecInt, pos::VecInt)
    return rows[pos]
end
export prior, LowOrderPrior, HighOrderPrior
public forward_prior, AbstractPriorEstimator, AbstractPriorResult, reconstruct_prior
# The `# Interfaces`-marked types of #1146 (ADR 0154): the five families
# `AbstractPriorEstimator`'s own section tells an author to subtype now carry the contract
# themselves. Their verb, `prior`, is already exported. `AbstractHighOrderPriorEstimator`
# joined under #531: `HighOrderPriorEstimator` subtypes it directly, with an optional `F`.
public AbstractLowOrderPriorEstimator_A, AbstractLowOrderPriorEstimator_F,
       AbstractLowOrderPriorEstimator_AF, AbstractHighOrderPriorEstimator_F,
       AbstractHighOrderPriorEstimator
