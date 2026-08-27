"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all prior estimators.

`AbstractPriorEstimator` is the base type for all estimators that compute prior information from asset and/or factor returns. All concrete prior estimators should subtype this type to ensure a consistent interface for prior computation and integration with portfolio optimisation workflows.

# Interfaces

In order to implement a new prior estimator which will work seamlessly with the library, subtype the family that names the returns it reads — [`AbstractLowOrderPriorEstimator_A`](@ref), [`AbstractLowOrderPriorEstimator_F`](@ref), [`AbstractLowOrderPriorEstimator_AF`](@ref) or [`AbstractHighOrderPriorEstimator_F`](@ref) — with all necessary parameters as part of the struct, and implement the following method:

  - `prior(pe::AbstractPriorEstimator, X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...) -> AbstractPriorResult`: Estimate the prior from the returns matrices.

The family fixes the signature. A member of the `_A` family declares `F` as `args...` and never reads it, a member of the `_F` family declares it `F::MatNum` and requires it, and a member of the `_AF` family declares it `F::Option{<:MatNum} = nothing` and reads it when it is there.

The method returns the carrier of its own order: a low order estimator returns a [`LowOrderPrior`](@ref), and a high order estimator returns a [`HighOrderPrior`](@ref). An estimator that wraps another rebuilds the wrapped result with [`forward_prior`](@ref) rather than by a hand-written constructor call, so that every field it does not name survives the hop.

The [`ReturnsResult`](@ref) method of [`prior`](@ref) is supplied by this file and needs no implementation.

## Arguments

  - $(arg_dict[:pe])
  - $(arg_dict[:X])
  - `F`: Factor returns matrix, or `nothing`.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the nested estimators.

## Returns

  - `pr::AbstractPriorResult`: Result object containing the estimated prior.

# Examples

We can create a dummy prior estimator as follows:

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
    kld ┼ nothing
     ow ┼ nothing
     rr ┼ nothing
    fpr ┼ nothing
      Z ┴ nothing
```

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractHighOrderPriorEstimator`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`prior`](@ref)
  - [`forward_prior`](@ref)
"""
abstract type AbstractPriorEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for low order prior estimators.

`AbstractLowOrderPriorEstimator` is the base type for estimators that compute low order moments (mean and covariance) from asset and/or factor returns. All concrete low order prior estimators should subtype this type for consistent moment estimation and integration. A member of this family returns a [`LowOrderPrior`](@ref), never a bare tuple of moments, so every consumer reads one carrier. It does not subtype this type directly: it subtypes the one of [`AbstractLowOrderPriorEstimator_A`](@ref), [`AbstractLowOrderPriorEstimator_F`](@ref) and [`AbstractLowOrderPriorEstimator_AF`](@ref) that names the returns it reads.

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

`AbstractLowOrderPriorEstimator_A` is the base type for estimators that compute low order moments (mean and covariance) using only asset returns data. All concrete asset-only prior estimators should subtype this type.

This is the first of the three source shapes. A member **admits asset returns only**: its `prior` method declares the factor argument as `args...` and never reads it, so factor returns handed to it are ignored rather than refused.

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

`AbstractLowOrderPriorEstimator_F` is the base type for estimators that compute low order moments (mean and covariance) requiring the use of both asset and factor returns data. All concrete factor-adjusted prior estimators should subtype this type.

This is the second of the three source shapes. A member **admits asset returns and requires factor returns**: its `prior` method declares the factor argument as `F::MatNum` with no default, so a call that omits factor returns is a `MethodError`. [`prior`](@ref) raises earlier and more clearly when a [`ReturnsResult`](@ref) with `F === nothing` reaches such an estimator.

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

`AbstractLowOrderPriorEstimator_AF` is the base type for estimators that compute low order moments (mean and covariance) using both asset and optionally factor returns data. All concrete prior estimators which may optionally use factor returns should subtype this type.

This is the third of the three source shapes. A member **admits asset returns and admits factor returns optionally**: its `prior` method declares the factor argument as `F::Option{<:MatNum} = nothing` and reads it when it is supplied. The shape therefore says nothing about whether the result carries a regression: use [`assert_prior_regression`](@ref) to establish that.

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`assert_prior_regression`](@ref)
  - [`prior`](@ref)
"""
abstract type AbstractLowOrderPriorEstimator_AF <: AbstractLowOrderPriorEstimator end
"""
    const AbstractLowOrderPriorEstimator_A_AF = Union{<:AbstractLowOrderPriorEstimator_A,
                                                      <:AbstractLowOrderPriorEstimator_AF}

Union type for asset-only and asset-and-factor low order prior estimators.

A field typed `AbstractLowOrderPriorEstimator_A_AF` **admits the asset-only and the optional-factor shapes, and excludes the shape that requires factor returns.** That is the bound for a wrapper which fits its nested estimator on **one** returns matrix it supplies itself: [`FactorPrior`](@ref) and [`FactorBlackLittermanPrior`](@ref) fit `pe` on the factor returns alone, and [`AugmentedBlackLittermanPrior`](@ref) fits `a_pe` on the assets and `f_pe` on the factors. A nested estimator that demanded a second matrix would have nothing to be handed, so the bound refuses it at construction rather than at the call.

# Related

  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_F_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_F_AF`](@ref)
  - [`FactorPrior`](@ref)
  - [`FactorBlackLittermanPrior`](@ref)
  - [`AugmentedBlackLittermanPrior`](@ref)
"""
const AbstractLowOrderPriorEstimator_A_AF = Union{<:AbstractLowOrderPriorEstimator_A,
                                                  <:AbstractLowOrderPriorEstimator_AF}
"""
    const AbstractLowOrderPriorEstimator_F_AF = Union{<:AbstractLowOrderPriorEstimator_F,
                                                      <:AbstractLowOrderPriorEstimator_AF}

Union type for factor-only and asset-and-factor low order prior estimators.

A field typed `AbstractLowOrderPriorEstimator_F_AF` **admits the factor-requiring and the optional-factor shapes, and excludes the asset-only shape.** That is the bound for a wrapper which forwards both returns matrices down and needs the result to be able to carry a factor block, as [`HighOrderFactorPriorEstimator`](@ref) does. The bound constrains what the nested estimator **consumes**, not what its result **produces**: the optional-factor half may still return a result with `rr === nothing`, so a consumer that reads the loadings guards with [`assert_prior_regression`](@ref).

# Related

  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_F_AF`](@ref)
  - [`HighOrderFactorPriorEstimator`](@ref)
  - [`assert_prior_regression`](@ref)
"""
const AbstractLowOrderPriorEstimator_F_AF = Union{<:AbstractLowOrderPriorEstimator_F,
                                                  <:AbstractLowOrderPriorEstimator_AF}
"""
    const AbstractLowOrderPriorEstimator_A_F_AF = Union{<:AbstractLowOrderPriorEstimator_A,
                                                        <:AbstractLowOrderPriorEstimator_F,
                                                        <:AbstractLowOrderPriorEstimator_AF}

Union type for asset-only, factor-only, and asset-and-factor low order prior estimators.

A field typed `AbstractLowOrderPriorEstimator_A_F_AF` **admits all three source shapes, and excludes nothing below the low order root.** That is the bound for a wrapper which passes the returns matrices it was handed straight through, so the nested estimator meets exactly the arguments the caller supplied and the shape is its own affair: [`EntropyPoolingPrior`](@ref) bounds `pe` this way. The union is written out rather than spelled [`AbstractLowOrderPriorEstimator`](@ref) so that the three shapes are named at every field that admits them, and so that a fourth shape added later reaches this bound only by a deliberate edit.

# Related

  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_F_AF`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
const AbstractLowOrderPriorEstimator_A_F_AF = Union{<:AbstractLowOrderPriorEstimator_A,
                                                    <:AbstractLowOrderPriorEstimator_F,
                                                    <:AbstractLowOrderPriorEstimator_AF}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for high order prior estimators.

`AbstractHighOrderPriorEstimator` is the base type for estimators that compute high order moments (such as coskewness and cokurtosis) from asset and/or factor returns. All concrete high order prior estimators should subtype this type to ensure a consistent interface for higher moment estimation and integration with portfolio optimisation workflows.

A member of this family returns a [`HighOrderPrior`](@ref), which wraps the [`LowOrderPrior`](@ref) its own nested low order estimator produced. So a high order estimator adds an order rather than replacing one, and every low order name stays readable through the wrapper.

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

`AbstractHighOrderPriorEstimator_F` is the base type for estimators that compute high order moments (such as coskewness and cokurtosis) requiring both asset and factor returns data. All concrete factor-based high order prior estimators should subtype this type.

A member **admits asset returns and requires factor returns**, on the same terms as [`AbstractLowOrderPriorEstimator_F`](@ref) one order down: its `prior` method declares the factor argument as `F::MatNum` with no default. The two are the members of [`AbstractHiLoOrderPriorEstimator_F`](@ref), which is how [`prior`](@ref) recognises a factor prior without naming an order.

# Related

  - [`AbstractHighOrderPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractHiLoOrderPriorEstimator_F`](@ref)
  - [`prior`](@ref)
"""
abstract type AbstractHighOrderPriorEstimator_F <: AbstractHighOrderPriorEstimator end
"""
    const AbstractHiLoOrderPriorEstimator_F = Union{<:AbstractLowOrderPriorEstimator_F,
                                                    <:AbstractHighOrderPriorEstimator_F}

Groups the two families that **require** factor returns, one per order.

`AbstractHiLoOrderPriorEstimator_F` is the one test for *this estimator cannot run without factor returns*, taken across both orders at once. [`prior`](@ref) dispatches its [`ReturnsResult`](@ref) method on every prior estimator, so it needs that test as a value rather than as a signature: it raises a named error when `rd.F` is `nothing` and the estimator is a member, in place of the `MethodError` the estimator's own signature would raise one call later.

# Related

  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractHighOrderPriorEstimator_F`](@ref)
  - [`prior`](@ref)
  - [`ReturnsResult`](@ref)
"""
const AbstractHiLoOrderPriorEstimator_F = Union{<:AbstractLowOrderPriorEstimator_F,
                                                <:AbstractHighOrderPriorEstimator_F}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all prior result types.

`AbstractPriorResult` is the base type for all result objects produced by prior estimators, containing computed prior information such as moments, asset returns, and factor returns. All concrete prior result types should subtype this to ensure a consistent interface for integration with portfolio optimisation workflows.

The library ships two carriers: [`LowOrderPrior`](@ref) holds the returns, the mean and the covariance, and [`HighOrderPrior`](@ref) holds the co-moments over a [`LowOrderPrior`](@ref) it wraps.

# Interfaces

In order to implement a new prior result carrier which will work seamlessly with the library, subtype `AbstractPriorResult` with all necessary fields as part of the struct, and implement the following methods:

  - `reconstruct_prior(pr::AbstractPriorResult, patch::NamedTuple) -> AbstractPriorResult`: Rebuild the carrier through its own constructor with `patch` applied. This is what makes [`forward_prior`](@ref) work on the carrier, and it is written per carrier because the constructor is named rather than recovered by reflection.
  - `port_opt_view(pr::AbstractPriorResult, i, args...) -> AbstractPriorResult`: Restrict the carrier to the assets at index `i`, for hierarchical and subset optimisation.

The field list is derived by [`prior_field_values`](@ref), so a carrier that gains a field needs no further method. Add the carrier's name to [`prior_result_property_pool`](@ref) so that an `@pprop` field naming one of its properties is recognised.

## Arguments

  - $(arg_dict[:pr])
  - `patch`: Named tuple of field overrides.
  - `i`: Asset indices the view keeps.
  - `args...`: Additional arguments the view reads.

## Returns

  - `pr::AbstractPriorResult`: A carrier of the same type as the input.

# Examples

We can create a dummy prior result carrier as follows:

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
    const PrE_Pr = Union{<:AbstractPriorEstimator, <:AbstractPriorResult}

Groups a prior estimator with an already-fitted prior result.

`PrE_Pr` is the bound of every optimiser's `pe` slot, and it is what lets a caller hand an optimiser a prior it has already fitted instead of the recipe for fitting one. The two are interchangeable there because [`prior`](@ref) has a method on each: the estimator method fits, and the result method returns its argument unchanged. So the optimiser calls [`prior`](@ref) once and never branches on which kind it holds.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`prior`](@ref)
"""
const PrE_Pr = Union{<:AbstractPriorEstimator, <:AbstractPriorResult}
"""
    const Pr_RR = Union{<:AbstractPriorResult, <:ReturnsResult}

Groups the two carriers that hold an asset returns matrix `X` and a feature matrix `Z`.

`Pr_RR` is the bridge the clustering, phylogeny and centrality forwarders below dispatch on. Each of them reads `X` and `Z` off its carrier and delegates to the asset-returns method, so an estimator that needs returns can be driven from a fitted prior or from the raw data with one method apiece rather than two. Where both carriers are present, [`returns_matrix_picker`](@ref) and [`feature_matrix_picker`](@ref) pick between them.

# Related

  - [`AbstractPriorResult`](@ref)
  - [`ReturnsResult`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`feature_matrix_picker`](@ref)
"""
const Pr_RR = Union{<:AbstractPriorResult, <:ReturnsResult}
"""
    prior(pe::AbstractPriorEstimator, rd::ReturnsResult; kwargs...)

Compute prior information from asset and/or factor returns using a prior estimator.

`prior` applies the specified prior estimator to a `ReturnsResult` object, extracting asset and factor returns and passing them, along with any additional information, to the estimator. Returns a prior result containing computed moments and other prior information for use in portfolio optimisation workflows.

This method is the entry point every caller uses, and it is written once here. What each estimator implements is the returns-matrix method that this one delegates to; [`AbstractPriorEstimator`](@ref) states that contract.

# Algorithm

 1. Check that `rd` carries asset returns, so that the estimator is not handed a `nothing` for `X`.
 2. When `pe` requires factor returns — when it is a member of [`AbstractHiLoOrderPriorEstimator_F`](@ref) — check that `rd` carries them. The check is made here so that the caller reads a named error against `rd.F` rather than a `MethodError` against the estimator's own signature one call later.
 3. Call the estimator's returns-matrix method with `rd.X` and `rd.F`, forwarding `rd.iv` and `rd.ivpa` as keyword arguments alongside `kwargs`, and return the prior result it produces.

# Arguments

  - $(arg_dict[:pe])
  - `rd`: Asset and/or factor returns result.
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Validation

  - `!isnothing(rd.X)`.
  - `!isnothing(rd.F)`, when `pe` is a member of [`AbstractHiLoOrderPriorEstimator_F`](@ref).

# Returns

  - `pr::AbstractPriorResult`: Result object containing computed prior information.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`AbstractHiLoOrderPriorEstimator_F`](@ref)
  - [`ReturnsResult`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
"""
function prior(pe::AbstractPriorEstimator, rd::ReturnsResult; kwargs...)
    @argcheck(!isnothing(rd.X), IsNothingError)
    if isa(pe, AbstractHiLoOrderPriorEstimator_F)
        @argcheck(!isnothing(rd.F),
                  IsNothingError("this is a factor prior; it needs factor returns. ReturnsResult.F is nothing — populate F (e.g. via prices_to_returns on factor prices)."))
    end
    return prior(pe, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)
end
"""
    prior_regression_remedy

The cause-and-remedy half of every "this prior carries no factor block" message, written
once so the two kinds of consumer cannot drift apart on it.

Consumers differ in *what* they wanted the loadings for — projecting factor moments through
them, or drawing them — so each supplies its own opening sentence via
[`assert_prior_regression`](@ref)'s `lead`. What none of them may restate is the diagnosis:
there is exactly one way to arrive with `rr === nothing`, and exactly one remedy, and both
are consequences of ADR 0046 rather than of the consumer.

## Which errors carry it

Two, and they are the two ways a caller can ask for loadings that were never computed:

  - The `IsNothingError` that [`assert_prior_regression`](@ref) raises. This is the estimator
    and plotting path: an estimator whose `pe` slot produced a prior with `rr === nothing`,
    or a factor-space plotting entry point handed the same prior. Each supplies its own
    `lead` and appends this string unchanged.
  - The `IsNothingError` that [`constraint_space_basis`](@ref) raises when a factor exposure
    constraint has no basis for its loadings — the space states none and the prior carries
    none. That message opens with its own sentences about the space, then appends this
    string, because the way out of the prior half of the diagnosis is the same one.

# Related

  - [`assert_prior_regression`](@ref)
  - [`constraint_space_basis`](@ref)
"""
const prior_regression_remedy = "No regression was ever computed: wrapping estimators forward `rr` and the factor block `fpr` (ADR 0046), so nesting order does not matter, but nothing in the chain produces loadings (e.g. `EntropyPoolingPrior(; pe = EmpiricalPrior())`). Put an estimator that produces them at the bottom, such as `FactorPrior`."
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that a prior result carries a factor block, so its loadings can be read.

Estimators whose `pe` field is typed [`AbstractLowOrderPriorEstimator_F_AF`](@ref) accept the [`AbstractLowOrderPriorEstimator_AF`](@ref) half of that union, whose members use factor returns only *optionally*. The type therefore constrains which returns an estimator **consumes**, not whether the result it **produces** carries a regression. An estimator that projects factor moments through the loadings needs the latter, and must check for it.

There is one way to arrive with `pr.rr === nothing`: nothing in the chain ever computed a regression (`EntropyPoolingPrior(; pe = EmpiricalPrior())`). Discarding one is no longer possible — every wrapping estimator forwards `rr` and the factor block `fpr` under ADR 0046, so nesting order does not matter. Checking `rr` covers the whole factor block, because [`LowOrderPrior`](@ref) already requires `rr` and `fpr` to be provided together or not at all — which is why the plotting entry points that want `fpr.mu` or `fpr.sigma` check `rr` here rather than testing the virtual read they are about to take.

Estimators are not the only consumer: the factor-space plotting entry points need the same block, and get the same diagnosis. Only the opening sentence differs, so `lead` carries it and [`prior_regression_remedy`](@ref) carries the rest.

# Arguments

  - `pr`: Prior result handed to the consumer.
  - `sym`: Name of the field or argument the result arrived through, used in the error message.
  - `lead`: Opening sentence naming what needed the loadings and what it found instead. Defaults to the wrapping-estimator case; a consumer that is not an estimator must supply its own, because the default's claim about the *type* not guaranteeing a regression is an estimator-field claim.

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

Return a prior result's own **fields** as a named tuple, keyed in declaration order.

Reads through `getfield`, so it sees only what the carrier stores — never a name a [`@forward_properties`](@ref) block exposes on top. That is the distinction [`forward_prior`](@ref) needs: `HighOrderPrior` forwards the whole of its `pr`, so `mu` and `sigma` are *properties* of it without being fields, and only a field can be patched. The field list is derived rather than written out, so adding a field to a carrier does not need an edit here.

# Algorithm

 1. Read the field names of `typeof(pr)` into `fnames`, in declaration order.
 2. Read each of those fields off `pr` with `getfield`, and return them as a `NamedTuple` keyed by `fnames`.

# Arguments

  - $(arg_dict[:pr])

# Returns

  - `vals::NamedTuple`: The carrier's own fields, keyed by name in declaration order.

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

Forward a wrapped prior result, spelling out only what the wrapping estimator changes or drops.

This is the mechanical half of the composition rule recorded in ADR 0046:

> **Forward when forwarding is correct; drop only where forwarding would state something false; document every drop in the estimator's docstring.**

Forwarding is the default and costs nothing to write, so a wrapper cannot accidentally return a narrower result than the one it wraps. Every deviation is spelled at the call site — a new value as `field = value`, a drop as `field = nothing` — which makes the set of drops greppable and reviewable instead of implicit in a hand-written constructor call listing all thirteen fields.

Reconstruction goes through the carrier's ordinary keyword constructor (see [`reconstruct_prior`](@ref)), so **every `@argcheck` runs**: a forward that leaves the carrier internally inconsistent throws exactly as a hand-written constructor call would. Only the carrier's own **fields** may be named — a forwarded or computed property is a view of a nested value, so setting it could only ever mean setting the field that value came from.

## The three enforced bindings

Three fields are *bound* to another field's value rather than being independent, so forwarding them past a change to the field they describe is what the rule calls stating something false. Because the binding is mechanical, the helper enforces it rather than leaving it to reviewer memory — naming the field on the left obliges the caller to name the fields on the right, either with a rebuilt value or with `nothing`:

  - **`sigma` binds `chol`.** `chol` *takes precedence over* `sigma` at every consumer, so a stale `chol` makes the optimisation silently ignore the posterior covariance.
  - **`w` binds `ens`, `kld` and `ow`.** Those are diagnostics *of* `w`; weights carrying another weighting's provenance cannot be interrogated.
  - **`rr` binds `o_X`.** `o_X` says `X` is a reconstruction, and `rr` is what records the projection that produced it, so the carrier refuses one without the other. Dropping the factor block therefore drops the original with it.

A binding is inert when the bound field is already `nothing` (there is nothing stale to carry) or absent from the carrier.

Everything else the constructor already covers: `rr` and `fpr` must be supplied together or not at all, and `w`, `chol` and `Z` are re-checked against the shape of `X` and `mu`.

## What does not fit

The estimators that *lift* a factor-axis prior into an asset-axis result ([`FactorPrior`](@ref), [`FactorBlackLittermanPrior`](@ref)) and the one that *merges two priors* ([`AugmentedBlackLittermanPrior`](@ref)) are not forwarding a single wrapped result along its own axis, so they construct their carrier directly and should not be forced through this helper. `forward_prior` still applies to the *factor block* they build, which is an ordinary forward of the factor prior.

# Algorithm

 1. Collect the keyword overrides into the named tuple `patch`. When `patch` is empty, return `pr` itself: a forward that changes nothing rebuilds nothing.
 2. Compare the names of `patch` against the fields of `typeof(pr)`, giving `extra`, the names that are not fields. A non-empty `extra` raises an `ArgumentError` naming the carrier's fields.
 3. Enforce the binding of `chol` to `sigma`. When `patch` names `sigma`, does not name `chol`, and [`bound_field_is_stale`](@ref) says `pr` holds a `chol`, raise a [`ConflictingArgumentError`](@ref).
 4. Enforce the binding of `o_X` to `rr`, on the same three tests, giving the second [`ConflictingArgumentError`](@ref).
 5. Enforce the binding of `ens`, `kld` and `ow` to `w`. When `patch` names `w`, collect into `stale` each of the three that `patch` does not name and that `pr` holds, and raise when `stale` is non-empty.
 6. Rebuild the carrier through [`reconstruct_prior`](@ref), which merges `patch` over [`prior_field_values`](@ref) and calls the ordinary keyword constructor, so every `@argcheck` of the carrier runs on the result.

# Arguments

  - `pr`: Prior result produced by the wrapped estimator.
  - `overrides...`: Field overrides; a value to replace, or `nothing` to drop.

# Validation

  - Naming `sigma` requires naming `chol`, unless `pr.chol` is already `nothing`.
  - Naming `w` requires naming each of `ens`, `kld` and `ow` that is not already `nothing`.
  - Naming `rr` requires naming `o_X`, unless `pr.o_X` is already `nothing`.
  - Every name in `overrides` is a field of `typeof(pr)`.
  - Every `@argcheck` of the constructor of `typeof(pr)`.

# Returns

  - `pr::AbstractPriorResult`: The wrapped result with `overrides` applied, or `pr` itself when there are none.

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
                       (:ens, :kld, :ow))
        @argcheck(isempty(stale),
                  ConflictingArgumentError("forwarding $(stale) past a change to `w` would state something false: they are diagnostics of the weights they were computed with, and diagnostics follow their weights. Pass each as `nothing` to drop it, or a value recomputed alongside the new `w`."))
    end
    return reconstruct_prior(pr, patch)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when field `sym` of `pr` holds a value that would go stale if the field it is bound to changed without it.

A field that the carrier does not have, or holds as `nothing`, has nothing to go stale. Reads through `getfield` so a forwarded property of the same name cannot answer for a field the carrier does not own.

# Algorithm

 1. Check whether `typeof(pr)` declares a field named `sym`. When it does not, the binding is inert on this carrier, so answer `false` without reading anything.
 2. Read that field with `getfield`, and answer `true` when the value it holds is not `nothing`.

# Arguments

  - $(arg_dict[:pr])
  - `sym`: Name of the bound field to test.

# Returns

  - `stale::Bool`: `true` when the carrier holds a value under `sym` that a change to the field it is bound to would make stale.

# Related

  - [`forward_prior`](@ref)
  - [`AbstractPriorResult`](@ref)
"""
function bound_field_is_stale(pr::AbstractPriorResult, sym::Symbol)::Bool
    return hasfield(typeof(pr), sym) && !isnothing(getfield(pr, sym))
end
"""
    prior(pr::AbstractPriorResult, args...; kwargs...)

Propagate or pass through prior result objects.

`prior` returns the input prior result object unchanged. This method is used to propagate already constructed prior results or enable uniform interface handling in workflows that accept either estimators or results.

It is the second half of [`PrE_Pr`](@ref): a slot bounded by that union calls `prior` once, and this method is why a slot holding an already-fitted result needs no branch of its own. Every further argument is accepted and ignored, so the call site does not change either.

# Arguments

  - $(arg_dict[:pr])
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `pr::AbstractPriorResult`: The input prior result object, unchanged.

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
    port_opt_view(pr::AbstractVector{<:Union{<:AbstractPriorResult, <:AbstractPriorEstimator}},
                  ::Any, args...; kwargs...)

Pass a prior estimator, or a vector of priors, through a view unchanged.

Both methods are the not-sliceable branch of [`port_opt_view`](@ref). An estimator carries a recipe rather than data on an asset axis, so there is nothing in it to cut down: the subproblem refits it on its own universe instead. A vector arrives already resolved per subproblem — one entry per cluster or per subset — so the entry has been chosen by the time the view is taken, and slicing the vector by an asset index would cut the wrong axis.

The carriers that *do* hold data on the asset axis take their own methods: see [`port_opt_view`](@ref) on [`LowOrderPrior`](@ref) and on [`HighOrderPrior`](@ref).

# Arguments

  - $(arg_dict[:per])
  - The second positional argument is the asset index. It is unnamed, because neither method reads it.
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
function port_opt_view(pr::AbstractVector{<:Union{<:AbstractPriorResult,
                                                  <:AbstractPriorEstimator}}, ::Any,
                       args...; kwargs...)
    return pr
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Pick the returns matrix the clustering, phylogeny and centrality estimators read.

Two carriers can supply asset returns: the prior result and the raw returns result. `x_src` names which one wins — `:prior` takes `pr.X`, `:data` takes `rd.X`. When no returns result is available there is nothing to select between, so `pr.X` is used and `x_src` is inert.

# Algorithm

 1. Check that `x_src` names one of the two carriers, with [`assert_source_selector`](@ref).
 2. Return `pr.X` when there is no returns result, or when `x_src` is `:prior`. Return `rd.X` otherwise.

# Arguments

  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Consulted only when `x_src` is `:data`.
  - $(arg_dict[:x_src])

# Validation

  - `x_src in (:prior, :data)`.

# Returns

  - `X::MatNum`: Asset returns matrix from the selected carrier.

# Related

  - [`assert_source_selector`](@ref)
  - [`clusterise`](@ref)
  - [`phylogeny_matrix`](@ref)
  - [`centrality_vector`](@ref)
"""
function returns_matrix_picker(pr::Pr_RR, rd::Option{<:ReturnsResult}, x_src::Symbol)
    assert_source_selector(x_src, :x_src)
    return isnothing(rd) || x_src == :prior ? pr.X : rd.X
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Pick the feature matrix a [`FeatureDistance`](@ref) inside the clustering, phylogeny or centrality estimator reads, and diagnose its absence.

The counterpart of [`returns_matrix_picker`](@ref), with the opposite default: `z_src = :data` prefers the user's own [`ReturnsResult`](@ref) over a derived one, because an explicit feature matrix outranks a produced one. (`x_src = :prior` prefers the prior, because a posterior returns matrix *is* the improvement being asked for. The differing defaults are the argument for naming the source rather than flagging it.)

A missing feature matrix is **not** an error here: `Z` is only required when a [`FeatureDistance`](@ref) is actually in the estimator tree, which this layer cannot see. Resolution therefore returns `nothing` and defers the throw to [`assert_feature_matrix_supplied`](@ref), passing a second return value that names *why* nothing was found — `:neither` when no carrier holds one, and the selector itself when it picked the empty carrier while the other held one.

# Algorithm

 1. Check that `z_src` names one of the two carriers, with [`assert_source_selector`](@ref).
 2. Read `Zp`, the prior carrier's feature matrix, and `Zd`, the returns result's. `Zd` is `nothing` when there is no returns result.
 3. Select `Z`: `Zp` when there is no returns result, or when `z_src` is `:prior`. `Zd` otherwise.
 4. Make the diagnostic `z_diag`: `:neither` when both `Zp` and `Zd` are `nothing`, and `z_src` itself otherwise. The two cases are distinct, because the second says a matrix exists on the carrier that was not selected.
 5. Return `Z` and `z_diag`.

# Arguments

  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Consulted only when `z_src` is `:data`.
  - $(arg_dict[:z_src])

# Validation

  - `z_src in (:prior, :data)`.

# Returns

  - `Z::Option{<:MatNum_Arr3Num}`: Feature matrix from the selected carrier, or `nothing`.
  - `z_diag::Symbol`: The diagnostic to forward as `z_src`; the selector itself, or `:neither`.

# Related

  - [`assert_source_selector`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`assert_feature_matrix_supplied`](@ref)
  - [`FeatureDistance`](@ref)
"""
function feature_matrix_picker(pr::Pr_RR, rd::Option{<:ReturnsResult}, z_src::Symbol)
    assert_source_selector(z_src, :z_src)
    Zp = pr.Z
    Zd = isnothing(rd) ? nothing : rd.Z
    Z = isnothing(rd) || z_src == :prior ? Zp : Zd
    return Z, isnothing(Zp) && isnothing(Zd) ? :neither : z_src
end
"""
    clusterise(cle::AbstractClustersEstimator, pr::AbstractPriorResult; kwargs...)

Clusterise asset or factor returns from a prior result using a clustering estimator.

`clusterise` applies the specified clustering estimator to the asset returns matrix contained in the prior result object, producing a clustering result for use in phylogeny analysis, constraint generation, or portfolio construction.

# Algorithm

 1. Pick the asset returns matrix `X` from the carrier that `x_src` names, with [`returns_matrix_picker`](@ref).
 2. Pick the feature matrix `Z` from the carrier that `z_src` names, with [`feature_matrix_picker`](@ref). It also gives `z_diag`, the diagnostic that names why nothing was found.
 3. Call the asset-returns method of [`clusterise`](@ref) with `X`, passing `Z` and `z_diag` on as `Z` and `z_src`, and return the clustering result it produces.

# Arguments

  - `cle`: Clustering estimator.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Consulted only when `x_src` or `z_src` is `:data`.
  - $(arg_dict[:x_src])
  - $(arg_dict[:z_src])
  - `kwargs...`: Additional keyword arguments passed to the clustering estimator.

# Returns

  - `clr::AbstractClusteringResult`: Result object containing clustering information.

# Related

  - [`ClustersEstimator`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`feature_matrix_picker`](@ref)
  - [`clusterise`](@ref)
"""
function clusterise(cle::AbstractClustersEstimator, pr::Pr_RR;
                    rd::Option{<:ReturnsResult} = nothing, x_src::Symbol = :prior,
                    z_src::Symbol = :data, kwargs...)
    X = returns_matrix_picker(pr, rd, x_src)
    Z, z_diag = feature_matrix_picker(pr, rd, z_src)
    return clusterise(cle, X; Z = Z, z_src = z_diag, kwargs...)
end
"""
    phylogeny_matrix(pl::NwE_ClE_Cl, pr::AbstractPriorResult;
                     kwargs...)

Compute the phylogeny matrix from asset returns in a prior result using a network or clustering estimator.

`phylogeny_matrix` applies the specified network or clustering estimator to the asset returns matrix contained in the prior result object, producing a phylogeny matrix for use in constraint generation, centrality analysis, or portfolio construction.

# Algorithm

 1. Pick the asset returns matrix `X` from the carrier that `x_src` names, with [`returns_matrix_picker`](@ref).
 2. Pick the feature matrix `Z` from the carrier that `z_src` names, with [`feature_matrix_picker`](@ref). It also gives `z_diag`, the diagnostic that names why nothing was found.
 3. Call the asset-returns method of [`phylogeny_matrix`](@ref) with `X`, passing `Z` and `z_diag` on as `Z` and `z_src`, and return the phylogeny result it produces.

# Arguments

  - `pl`: Network estimator, clusters estimator, or clustering result.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Consulted only when `x_src` or `z_src` is `:data`.
  - $(arg_dict[:x_src])
  - $(arg_dict[:z_src])
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Returns

  - `plr::PhylogenyResult`: Result object containing the phylogeny matrix.

# Related

  - [`NetworkEstimator`](@ref)
  - [`ClustersEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`feature_matrix_picker`](@ref)
  - [`phylogeny_matrix`](@ref)
"""
function phylogeny_matrix(pl::NwE_ClE_Cl, pr::Pr_RR; rd::Option{<:ReturnsResult} = nothing,
                          x_src::Symbol = :prior, z_src::Symbol = :data, kwargs...)
    X = returns_matrix_picker(pr, rd, x_src)
    Z, z_diag = feature_matrix_picker(pr, rd, z_src)
    return phylogeny_matrix(pl, X; Z = Z, z_src = z_diag, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute phylogeny constraints from asset returns in a prior result using a phylogeny constraint estimator.

`phylogeny_constraints` delegates to the asset-returns variant by extracting `X` from `pr` (or `rd` if provided and `x_src` is `:data`).

# Algorithm

 1. Pick the asset returns matrix `X` from the carrier that `x_src` names, with [`returns_matrix_picker`](@ref).
 2. Pick the feature matrix `Z` from the carrier that `z_src` names, with [`feature_matrix_picker`](@ref). It also gives `z_diag`, the diagnostic that names why nothing was found.
 3. Call the asset-returns method of [`phylogeny_constraints`](@ref) with `X`, passing `Z` and `z_diag` on as `Z` and `z_src`, and return the constraint result it produces.

# Arguments

  - `plc`: Phylogeny constraint estimator.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Consulted only when `x_src` or `z_src` is `:data`.
  - $(arg_dict[:x_src])
  - $(arg_dict[:z_src])
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Returns

  - Phylogeny constraint result.

# Related

  - [`AbstractPhylogenyConstraintEstimator`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`feature_matrix_picker`](@ref)
  - [`phylogeny_constraints`](@ref)
"""
function phylogeny_constraints(plc::AbstractPhylogenyConstraintEstimator, pr::Pr_RR;
                               rd::Option{<:ReturnsResult} = nothing,
                               x_src::Symbol = :prior, z_src::Symbol = :data, kwargs...)
    X = returns_matrix_picker(pr, rd, x_src)
    Z, z_diag = feature_matrix_picker(pr, rd, z_src)
    return phylogeny_constraints(plc, X; Z = Z, z_src = z_diag, kwargs...)
end
"""
    centrality_vector(cte::CentralityEstimator, pr::AbstractPriorResult; kwargs...)

Compute the centrality vector for a centrality estimator and prior result.

`centrality_vector` applies the centrality algorithm in the estimator to the network constructed from the asset returns in the prior result, returning centrality scores for each asset.

# Algorithm

 1. Pick the asset returns matrix `X` from the carrier that `x_src` names, with [`returns_matrix_picker`](@ref).
 2. Pick the feature matrix `Z` from the carrier that `z_src` names, with [`feature_matrix_picker`](@ref). It also gives `z_diag`, the diagnostic that names why nothing was found.
 3. Call the asset-returns method of [`centrality_vector`](@ref) with `X`, passing `Z` and `z_diag` on as `Z` and `z_src`, and return the centrality result it produces.

# Arguments

  - $(arg_dict[:cte])
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Consulted only when `x_src` or `z_src` is `:data`.
  - $(arg_dict[:x_src])
  - $(arg_dict[:z_src])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `plr::PhylogenyResult`: Result object containing the centrality vector.

# Related

  - [`CentralityEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`feature_matrix_picker`](@ref)
  - [`centrality_vector`](@ref)
"""
function centrality_vector(cte::CentralityEstimator, pr::Pr_RR;
                           rd::Option{<:ReturnsResult} = nothing, x_src::Symbol = :prior,
                           z_src::Symbol = :data, kwargs...)
    X = returns_matrix_picker(pr, rd, x_src)
    Z, z_diag = feature_matrix_picker(pr, rd, z_src)
    return centrality_vector(cte, X; Z = Z, z_src = z_diag, kwargs...)
end
"""
    centrality_vector(pl::NwE_ClE_Cl, ct::AbstractCentralityAlgorithm,
                      pr::AbstractPriorResult; kwargs...)

Compute the centrality vector for a network or clustering estimator and centrality algorithm.

`centrality_vector` constructs the phylogeny matrix from the asset returns in the prior result, builds a graph, and computes node centrality scores using the specified centrality algorithm.

# Algorithm

 1. Pick the asset returns matrix `X` from the carrier that `x_src` names, with [`returns_matrix_picker`](@ref).
 2. Pick the feature matrix `Z` from the carrier that `z_src` names, with [`feature_matrix_picker`](@ref). It also gives `z_diag`, the diagnostic that names why nothing was found.
 3. Call the asset-returns method of [`centrality_vector`](@ref) with `X`, passing `Z` and `z_diag` on as `Z` and `z_src`, and return the centrality result it produces.

# Arguments

  - `pl`: Network estimator, clusters estimator, or clustering result.
  - $(arg_dict[:cta])
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Consulted only when `x_src` or `z_src` is `:data`.
  - $(arg_dict[:x_src])
  - $(arg_dict[:z_src])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `plr::PhylogenyResult`: Result object containing the centrality vector.

# Related

  - [`NetworkEstimator`](@ref)
  - [`CentralityEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`feature_matrix_picker`](@ref)
  - [`centrality_vector`](@ref)
"""
function centrality_vector(pl::NwE_ClE_Cl, ct::AbstractCentralityAlgorithm, pr::Pr_RR;
                           rd::Option{<:ReturnsResult} = nothing, x_src::Symbol = :prior,
                           z_src::Symbol = :data, kwargs...)
    X = returns_matrix_picker(pr, rd, x_src)
    Z, z_diag = feature_matrix_picker(pr, rd, z_src)
    return centrality_vector(pl, ct, X; Z = Z, z_src = z_diag, kwargs...)
end
"""
    average_centrality(pl::NwE_Pl_ClE_Cl,
                       ct::AbstractCentralityAlgorithm, w::VecNum,
                       pr::AbstractPriorResult; kwargs...)

Compute the weighted average centrality for a network or phylogeny result.

`average_centrality` computes the centrality vector using the specified network or phylogeny estimator and centrality algorithm, then returns the weighted average using the provided portfolio weights.

# Algorithm

 1. Compute the centrality result with the [`Pr_RR`](@ref) method of [`centrality_vector`](@ref), forwarding `rd`, `x_src` and `z_src` unchanged. The source selection is therefore made once, there, and this method never reads a carrier itself.
 2. Return the dot product of that result's `X`, the centrality vector, with the weights `w`.

# Arguments

  - `pl`: Network estimator or phylogeny result.
  - $(arg_dict[:cta])
  - `w`: Portfolio weights vector.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Consulted only when `x_src` or `z_src` is `:data`.
  - $(arg_dict[:x_src])
  - $(arg_dict[:z_src])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `ac::Number`: Weighted average centrality.

# Related

  - [`NetworkEstimator`](@ref)
  - [`CentralityEstimator`](@ref)
  - [`Pr_RR`](@ref)
  - [`centrality_vector`](@ref)
  - [`average_centrality`](@ref)
"""
function average_centrality(pl::NwE_Pl_ClE_Cl, ct::AbstractCentralityAlgorithm, w::VecNum,
                            pr::Pr_RR; rd::Option{<:ReturnsResult} = nothing,
                            x_src::Symbol = :prior, z_src::Symbol = :data, kwargs...)
    return LinearAlgebra.dot(centrality_vector(pl, ct, pr; rd = rd, x_src = x_src,
                                               z_src = z_src, kwargs...).X, w)
end
"""
    average_centrality(cte::CentralityEstimator, w::VecNum, pr::AbstractPriorResult;
                       kwargs...)

Compute the weighted average centrality for a centrality estimator.

`average_centrality` applies the centrality algorithm in the estimator to the network constructed from the asset returns in the prior result, then returns the weighted average using the provided portfolio weights.

# Algorithm

 1. Pick the asset returns matrix `X` from the carrier that `x_src` names, with [`returns_matrix_picker`](@ref).
 2. Pick the feature matrix `Z` from the carrier that `z_src` names, with [`feature_matrix_picker`](@ref). It also gives `z_diag`, the diagnostic that names why nothing was found.
 3. Call the asset-returns method of [`average_centrality`](@ref) with `X`, passing `Z` and `z_diag` on as `Z` and `z_src`, and return the weighted average it produces.

The estimator method picks the carriers itself, where the network-and-algorithm method above delegates that to [`centrality_vector`](@ref). The two reach the same selection: `cte` carries `pl` and `ct` in its own fields, so the asset-returns method it calls is the one the other method's step 1 would have reached.

# Arguments

  - $(arg_dict[:cte])
  - `w`: Portfolio weights vector.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Consulted only when `x_src` or `z_src` is `:data`.
  - $(arg_dict[:x_src])
  - $(arg_dict[:z_src])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `ac::Number`: Weighted average centrality.

# Related

  - [`CentralityEstimator`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`feature_matrix_picker`](@ref)
  - [`centrality_vector`](@ref)
  - [`average_centrality`](@ref)
"""
function average_centrality(cte::CentralityEstimator, w::VecNum, pr::Pr_RR;
                            rd::Option{<:ReturnsResult} = nothing, x_src::Symbol = :prior,
                            z_src::Symbol = :data, kwargs...)
    X = returns_matrix_picker(pr, rd, x_src)
    Z, z_diag = feature_matrix_picker(pr, rd, z_src)
    return average_centrality(cte, w, X; Z = Z, z_src = z_diag, kwargs...)
end
"""
    asset_phylogeny(pl::NwE_ClE_Cl,
                    w::VecNum, pr::AbstractPriorResult; dims::Int = 1, kwargs...)

Compute the asset phylogeny score for a portfolio allocation using a phylogeny estimator or clustering result and a prior result.

This function computes the phylogeny matrix from the asset returns in the prior result using the specified phylogeny estimator or clustering result, then evaluates the asset phylogeny score for the given portfolio weights. The asset phylogeny score quantifies the degree of phylogenetic (network or cluster-based) structure present in the portfolio allocation.

# Algorithm

 1. Pick the asset returns matrix `X` from the carrier that `x_src` names, with [`returns_matrix_picker`](@ref).
 2. Pick the feature matrix `Z` from the carrier that `z_src` names, with [`feature_matrix_picker`](@ref). It also gives `z_diag`, the diagnostic that names why nothing was found.
 3. Call the asset-returns method of [`asset_phylogeny`](@ref) with `X`, passing `Z` and `z_diag` on as `Z` and `z_src`, and return the score it produces.

# Arguments

  - `pl`: Phylogeny estimator or clustering result used to compute the phylogeny matrix.
  - `w`: Portfolio weights vector.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Consulted only when `x_src` or `z_src` is `:data`.
  - $(arg_dict[:x_src])
  - $(arg_dict[:z_src])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the phylogeny matrix computation.

# Returns

  - `score::Number`: Asset phylogeny score.

# Related

  - [`phylogeny_matrix`](@ref)
  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractClusteringResult`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`feature_matrix_picker`](@ref)
  - [`asset_phylogeny`](@ref): The asset-returns methods this one delegates to. They build the phylogeny matrix, add up the gross weight of the related pairs, and divide by the gross weight of every pair. That is where the score's closed form and its numbered steps are stated.
"""
function asset_phylogeny(pl::NwE_ClE_Cl, w::VecNum, pr::Pr_RR;
                         rd::Option{<:ReturnsResult} = nothing, x_src::Symbol = :prior,
                         z_src::Symbol = :data, kwargs...)
    X = returns_matrix_picker(pr, rd, x_src)
    Z, z_diag = feature_matrix_picker(pr, rd, z_src)
    return asset_phylogeny(pl, w, X; Z = Z, z_src = z_diag, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute centrality constraints from asset returns in a prior result using a centrality constraint estimator.

`centrality_constraints` delegates to the asset-returns variant by extracting `X` from `pr` (or `rd` if provided and `x_src` is `:data`).

# Algorithm

 1. Pick the asset returns matrix `X` from the carrier that `x_src` names, with [`returns_matrix_picker`](@ref).
 2. Pick the feature matrix `Z` from the carrier that `z_src` names, with [`feature_matrix_picker`](@ref). It also gives `z_diag`, the diagnostic that names why nothing was found.
 3. Call the asset-returns method of [`centrality_constraints`](@ref) with `X`, passing `Z` and `z_diag` on as `Z` and `z_src`, and return the constraint result it produces.

# Arguments

  - `ccs`: Centrality constraint estimator or vector thereof.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Consulted only when `x_src` or `z_src` is `:data`.
  - $(arg_dict[:x_src])
  - $(arg_dict[:z_src])
  - `kwargs...`: Additional keyword arguments passed to the estimator. `strict` is read by the asset-returns variant, which reports a dropped zero centrality vector through it.

# Returns

  - Centrality constraint result.

# Related

  - [`AbstractPriorResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`feature_matrix_picker`](@ref)
  - [`centrality_constraints`](@ref)
"""
function centrality_constraints(ccs::CC_VecCC, pr::Pr_RR;
                                rd::Option{<:ReturnsResult} = nothing,
                                x_src::Symbol = :prior, z_src::Symbol = :data, kwargs...)
    X = returns_matrix_picker(pr, rd, x_src)
    Z, z_diag = feature_matrix_picker(pr, rd, z_src)
    return centrality_constraints(ccs, X; Z = Z, z_src = z_diag, kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the returns, mean and covariance a low order prior estimator produced.

`LowOrderPrior` stores the output of low order prior estimation routines, including asset returns, mean vector, covariance matrix, Cholesky factor, weights, entropy, Kullback-Leibler divergence, outlier weights, regression results, and optional factor moments. It is used throughout the package to represent validated prior information for portfolio optimisation and analytics.

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
        kld::Option{<:Num_VecNum} = nothing,
        ow::Option{<:VecNum} = nothing,
        rr::Option{<:Regression} = nothing,
        fpr::Option{<:LowOrderPrior} = nothing,
        Z::Option{<:MatNum_Arr3Num} = nothing
    ) -> LowOrderPrior

Keywords correspond to the struct's fields.

## The factor block

A prior fit through a factor model carries two distributions: one over the assets, in the carrier's own fields, and one over the factors. The factor one is a **nested `LowOrderPrior`** in `fpr` rather than a set of `f_`-prefixed flat fields, so it gains every field the carrier has — `w`, `ens`, `kld` and `ow` as well as `mu` and `sigma` — and gains any field added in future without a second edit. Its `X` is the factor returns matrix, over the same observations as the asset `X`; `fpr.Z` is therefore factors × features, which is why an asset-axis `Z` never comes from it.

`fpr` travels with `rr`: the two are the factor block, and the constructor requires them together or not at all. `rr` is what projects the block onto the assets (`mu ≈ rr.M * fpr.mu + rr.b`), so a factor distribution with no loadings could not be read against this asset axis.

The flat names are **virtual reads** of the nested block, so code written against the old shape is unaffected: `pr.f_mu`, `pr.f_sigma` and `pr.f_w` return `fpr.mu`, `fpr.sigma` and `fpr.w`, or `nothing` when there is no factor block, and `pr.f_ens`, `pr.f_kld` and `pr.f_ow` come with them. They are properties, not fields — [`forward_prior`](@ref) and [`prior_field_values`](@ref) see only `fpr`.

### Which read is idiomatic

**`pr.fpr.mu` is the public read**; the flat `f_`-prefixed names are a **compatibility surface**, kept so that code written against the pre-nesting shape keeps working, and useful where a value-or-`nothing` read without branching is wanted.

The reason is not taste. The flat surface is **partial and frozen**: there are six flat names over twelve fields, so `fpr.X` — the factor returns matrix — and `fpr.Z`, `fpr.chol` and `fpr.rr` have no flat spelling at all and never will. A surface that cannot express the whole block cannot be the way to read it. The set is fixed at the six here and the seven on [`HighOrderPrior`](@ref); a field added to a carrier in future is reachable as `pr.fpr.<name>` and gains no `f_` counterpart, so nothing has to be added in two places to stay complete.

The two reads also differ where the block is absent, which is the one case worth checking before choosing: `pr.f_mu` returns `nothing`, while `pr.fpr.mu` throws, because `fpr` is `nothing`. Guard with [`assert_prior_regression`](@ref) — `rr` and `fpr` are supplied together or not at all, so checking `rr` establishes the whole block — and then read through `fpr`.

## Composition: what a wrapping estimator forwards

Most prior estimators wrap another and return a carrier built from the one they were handed. Which fields survive that hop is governed by a single rule, recorded in ADR 0046 and enforced by [`forward_prior`](@ref):

> **Forward when forwarding is correct; drop only where forwarding would state something false; document every drop in the estimator's docstring.**

Consistency of the returned result is the criterion, and destroying a value the caller explicitly computed is not an acceptable way to buy it — so forwarding is the default, and each estimator's docstring lists the fields it drops and why. Two fields are *bound* to another and therefore never forwarded alone: `chol` is bound to `sigma` (it takes precedence over `sigma` at every consumer, so a stale factor is silently used in place of the updated covariance), and `ens`, `kld` and `ow` are bound to `w` (they are diagnostics *of* those weights). `forward_prior` refuses a forward that would break either binding.

## The feature matrix

`Z` is **derived only**: it is populated by a producer that declares a matrix to be features — [`FeaturePrior`](@ref) — and never by pass-through of a user's `ReturnsResult.Z`. That is what keeps the two carriers from disagreeing: they cannot both hold the same matrix, so `z_src` selects a provenance rather than one of two copies.

It carries **no feature names**. A producer runs inside `prior(pe, X, F; …)` with raw matrices, so names are structurally unavailable there — and it carries no squareness flag either: the prior carrier has no vocabulary for "the features *are* the assets", because every producer that builds a square feature matrix refits on the subproblem's own universe rather than having a full-universe matrix sliced. Exogenous square structure travels on the *data* carrier, where squareness is derived from `nz` against `nx` and therefore cannot be stated wrongly.

Every prior estimator that wraps another **forwards it**, so nesting order does not matter: `BlackLittermanPrior(; pe = FeaturePrior(…))` and `FeaturePrior(; pe = BlackLittermanPrior(…))` both arrive with `Z` set. This is unconditionally safe because no prior estimator changes the asset set or the observation count. The exceptions are the estimators whose wrapped prior is fit on **factors** rather than assets — [`FactorPrior`](@ref), [`FactorBlackLittermanPrior`](@ref), and the factor half of [`AugmentedBlackLittermanPrior`](@ref) — which drop it, because a factor-space feature matrix does not describe the asset axis.

`FactorPrior`, `FactorBlackLittermanPrior` and `AugmentedBlackLittermanPrior` *reconstruct* `X` as `F * transpose(M) .+ transpose(b)`, so a `Z` forwarded through them is dimension-correct but was derived from the pre-reconstruction returns.

## The original returns matrix

Those same three estimators are the reason `o_X` exists. They overwrite `X`, so on their carriers `X` is a **posterior** matrix — the asset distribution this prior asserts — and not the returns the caller supplied. `o_X` holds the returns the caller supplied. It is `nothing` everywhere else, where `X` already is them.

The two matrices are not interchangeable. The reconstruction spans only the factors: it has rank `size(F, 2)`, and the residual is absent. A consumer that refits a moment on the sample must therefore read the original, or it gets a singular matrix whenever there are more assets than factors.

**Read it as `original_X`, never as `o_X`.** The property is always a matrix — the field where there is one, `X` where there is not — so a consumer needs no fallback and cannot forget one. The field is storage, and it answers a different question: `isnothing(pr.o_X)` is how to ask whether this carrier reconstructed `X`. The field carries the state rather than the property carrying it, because [`forward_prior`](@ref) rebuilds through the keyword constructor with every field named, and a `nothing` is inert there where an always-populated matrix would go stale past a change to `X`.

`o_X` requires `rr`. Every estimator that overwrites `X` today does so by projecting a factor prior through regression loadings, so a carrier claiming a reconstruction it cannot explain is a bug. This is a present-tense constraint rather than a law of the domain: see the amendment to ADR 0046.

## Validation

  - `X`, `mu`, and `sigma` must be non-empty.
  - `size(sigma, 1) == size(sigma, 2)`.
  - `size(X, 2) == length(mu) == size(sigma, 1)`.
  - If `w` is not `nothing`, `!isempty(w)` and `length(w) == size(X, 1)`.
  - If `kld` is an `AbstractVector`, `!isempty(kld)`.
  - If `ow` is not `nothing`, `!isempty(ow)`.
  - `rr` and `fpr` must be provided together or not at all.
  - If the factor block is present, `size(rr.M, 2) == length(fpr.mu) == size(fpr.sigma, 1)`, `size(rr.M, 1) == length(mu)`, and `size(fpr.X, 1) == size(X, 1)` — the two blocks describe the same observations. Everything internal to the factor block, including its own `w` against its own `X`, is validated by its own constructor.
  - If `o_X` is not `nothing`, `o_X !== X`, `size(o_X) == size(X)`, and `rr` is not `nothing`.
  - If `chol` is not `nothing`, `!isempty(chol)` and `length(mu) == size(chol, 2)`.
  - If `Z` is not `nothing`, it is non-empty, all-finite, and assets-major against `X`: `size(Z, 1) == size(X, 2)` when static, `size(Z, 1) == size(X, 1)` and `size(Z, 2) == size(X, 2)` when time-varying (see [`check_feature_matrix`](@ref)).

## View parameters

`LowOrderPrior` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - It reads no argument beyond `i`. Further positional arguments are accepted and ignored.
  - `rr` recurses through [`port_opt_view`](@ref) with `i`, which cuts the loadings down on their asset axis.
  - `X`, `o_X`, `mu`, `sigma` and `chol` are sliced to `i` on the asset axis. `o_X` takes the same cut as `X`, so a subproblem's original returns stay the caller's returns for that subproblem's assets.
  - `Z` is sliced on its asset axis alone, through [`feature_matrix_view`](@ref). Its feature axis is never cut, and its observations are taken whole.
  - `w`, `ens`, `kld` and `ow` pass through unchanged. They live on the observation axis, and `i` indexes assets.
  - `fpr` passes through unchanged, because it is a distribution over factors rather than over assets. It is why the view keeps `rr` and `fpr` together, and so keeps the carrier's own factor-block rule satisfied.

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
    kld ┼ nothing
     ow ┼ nothing
     rr ┼ nothing
    fpr ┼ nothing
      Z ┴ nothing
```

# Related

  - [`AbstractPriorResult`](@ref)
  - [`prior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`forward_prior`](@ref)
  - [`reconstruct_prior`](@ref)
  - [`port_opt_view`](@ref)
  - [`feature_matrix_view`](@ref)
  - [`FeaturePrior`](@ref)
  - [`FeatureDistance`](@ref)
  - [`check_feature_matrix`](@ref)
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
    $(field_dict[:ens])
    """
    ens
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
    $(field_dict[:Z_prior])
    """
    Z
    function LowOrderPrior(X::MatNum, o_X::Option{<:MatNum}, mu::VecNum, sigma::MatNum,
                           chol::Option{<:MatNum}, w::Option{<:ObsWeights},
                           ens::Option{<:Number}, kld::Option{<:Num_VecNum},
                           ow::Option{<:VecNum}, rr::Option{<:Regression},
                           fpr::Option{<:LowOrderPrior}, Z::Option{<:MatNum_Arr3Num})
        @argcheck(!isempty(X), IsEmptyError("X cannot be empty"))
        @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
        @argcheck(!isempty(sigma), IsEmptyError("sigma cannot be empty"))
        assert_matrix_issquare(sigma, :sigma)
        @argcheck(size(X, 2) == length(mu) == size(sigma, 1),
                  DimensionMismatch("size(X, 2) ($(size(X, 2))), length(mu) ($(length(mu))), and size(sigma, 1) ($(size(sigma, 1))) must all match"))
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
        rr_is_nothing = isnothing(rr)
        fpr_is_nothing = isnothing(fpr)
        @argcheck(rr_is_nothing == fpr_is_nothing,
                  ArgumentError("rr and fpr are the factor block and must be provided together or not at all, isnothing(rr) = $(rr_is_nothing), isnothing(fpr) = $(fpr_is_nothing)"))
        if !rr_is_nothing
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
        # through regression loadings, so the loadings are always in hand and a carrier
        # claiming a reconstruction it cannot explain is a bug. A future estimator that
        # transforms `X` without a regression — a bootstrap or a simulation prior — is the
        # case that relaxes this, and it must relax it deliberately. See ADR 0046.
        if !isnothing(o_X)
            @argcheck(o_X !== X,
                      ArgumentError("o_X is X itself, so this carrier has no original distinct from the one it asserts. Pass o_X = nothing, which is what every consumer reads as \"X is the original\""))
            @argcheck(size(o_X) == size(X),
                      DimensionMismatch("size(o_X) ($(size(o_X))) must match size(X) ($(size(X))): the original and the matrix this carrier asserts describe the same observations and the same assets"))
            @argcheck(!rr_is_nothing,
                      IsNothingError("o_X says X is not the caller's matrix, but rr === nothing, so this carrier does not record what produced X. Every estimator that overwrites X projects a factor prior through regression loadings and carries them in rr"))
        end
        if !isnothing(chol)
            @argcheck(!isempty(chol), IsEmptyError("chol cannot be empty"))
            @argcheck(length(mu) == size(chol, 2),
                      DimensionMismatch("length(mu) ($(length(mu))) must match size(chol, 2) ($(size(chol, 2)))"))
        end
        check_feature_matrix(Z, size(X, 2), size(X, 1), "size(X, 2)")
        return new{typeof(X), typeof(o_X), typeof(mu), typeof(sigma), typeof(chol),
                   typeof(w), typeof(ens), typeof(kld), typeof(ow), typeof(rr), typeof(fpr),
                   typeof(Z)}(X, o_X, mu, sigma, chol, w, ens, kld, ow, rr, fpr, Z)
    end
end
function LowOrderPrior(; X::MatNum, o_X::Option{<:MatNum} = nothing, mu::VecNum,
                       sigma::MatNum, chol::Option{<:MatNum} = nothing,
                       w::Option{<:ObsWeights} = nothing, ens::Option{<:Number} = nothing,
                       kld::Option{<:Num_VecNum} = nothing, ow::Option{<:VecNum} = nothing,
                       rr::Option{<:Regression} = nothing,
                       fpr::Option{<:LowOrderPrior} = nothing,
                       Z::Option{<:MatNum_Arr3Num} = nothing)::LowOrderPrior
    return LowOrderPrior(X, o_X, mu, sigma, chol, w, ens, kld, ow, rr, fpr, Z)
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

The feature matrix is subselected on its asset axis only. Its feature axis is never sliced: a prior-side `Z` is *derived*, and every producer that builds a square one refits on the subproblem's own universe, so there is no full-universe square matrix here to cut down. Observations are taken whole (`Colon`): folds slice observations *before* the prior is fit, so a derived `Z` is already fold-local by the time it reaches here.

The factor block is forwarded **unsliced**: `i` indexes assets, and `fpr` is a distribution over factors. Only `rr` is cut down, on its asset axis.

# Algorithm

 1. Cut the Cholesky factor to `i` on its column axis, giving `chol`. A carrier that holds none keeps `nothing`.
 2. Cut the original returns matrix to `i` on its asset axis, giving `o_X`. A carrier that holds none keeps `nothing`. It takes the same cut `X` takes in the next step, because the two are assets-major over the same observations.
 3. Rebuild the carrier through its ordinary keyword constructor, naming every field: `X` and `mu` cut to `i`, `sigma` cut to `i` on both axes, `chol` and `o_X` from the two steps above, `rr` recursed through [`port_opt_view`](@ref) with `i`, `Z` cut with [`feature_matrix_view`](@ref) on its asset axis alone, and `w`, `ens`, `kld`, `ow` and `fpr` forwarded unchanged. Every `@argcheck` of the constructor therefore runs on the view.

# Arguments

  - $(arg_dict[:pr])
  - `i`: Asset indices the view keeps.
  - `args...`: Additional arguments (ignored).

# Returns

  - `pr::LowOrderPrior`: The carrier restricted to the assets at `i`, holding views rather than copies.

# Related

  - [`LowOrderPrior`](@ref)
  - [`port_opt_view`](@ref)
  - [`feature_matrix_view`](@ref)
"""
function port_opt_view(pr::LowOrderPrior, i, args...)::LowOrderPrior
    chol = isnothing(pr.chol) ? nothing : view(pr.chol, :, i)
    # `o_X` is assets-major over the same observations as `X`, so it takes the same cut. A
    # subproblem's original returns are the caller's returns for that subproblem's assets.
    o_X = isnothing(pr.o_X) ? nothing : view(pr.o_X, :, i)
    return LowOrderPrior(; X = view(pr.X, :, i), o_X = o_X, mu = view(pr.mu, i),
                         sigma = view(pr.sigma, i, i), chol = chol, w = pr.w, ens = pr.ens,
                         kld = pr.kld, ow = pr.ow, rr = port_opt_view(pr.rr, i),
                         fpr = pr.fpr, Z = feature_matrix_view(pr.Z, false, :, i))
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the coskewness and cokurtosis a high order prior estimator produced, over the low order prior it wraps.

`HighOrderPrior` stores the output of high order prior estimation routines, including low order prior results, cokurtosis tensor, elimination and summation matrices, coskewness tensor, quadratic skewness matrix, and matrix processing estimator. It is used throughout the package to represent validated prior information for portfolio optimisation and analytics involving higher moments.

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
        fpr::Option{<:HighOrderPrior} = nothing
    ) -> HighOrderPrior

Keywords correspond to the struct's fields.

## The factor block

A high order prior fit through a factor model carries factor co-moments alongside the asset ones. They are a **nested `HighOrderPrior`** in `fpr` rather than the `f_`-prefixed flat fields `f_kt`, `f_sk` and `f_V`, so the factor block gains every field the carrier has — `D2`, `L2`, `S2` and `skmp` as well as `kt`, `sk` and `V` — and gains any field added in future without a second edit. The flat names remain readable as **virtual reads** of it: `pr.f_kt`, `pr.f_sk` and `pr.f_V` return `fpr.kt`, `fpr.sk` and `fpr.V`, or `nothing` when there is no factor block, and `pr.f_D2`, `pr.f_L2`, `pr.f_S2` and `pr.f_skmp` come with them.

`fpr.pr` is the factor block one order down: the [`LowOrderPrior`](@ref) over the factors. The same distribution is also reachable as `pr.pr.fpr`, the low order carrier's own factor block, and the constructor **enforces that the two are the same object** — see the validation below.

`fpr` is this carrier's own field, so it resolves ahead of the `forward(pr)` block and names the **high** order factor block, where before nesting it resolved through to the low order one. Reads through it are unaffected by that shift: the nested carrier forwards to its own `pr`, which the invariant pins to `pr.fpr`, so `hop.fpr.mu` is the factor mean either way and `hop.fpr` is simply "the factor prior at this order".

### Which read is idiomatic

**`pr.fpr.kt` is the public read**, on the same terms as on [`LowOrderPrior`](@ref) — see the fuller reasoning there. The seven flat names here are a **frozen compatibility surface**: `f_kt`, `f_sk`, `f_V`, `f_D2`, `f_L2`, `f_S2` and `f_skmp`, and no more will be added. A field added to this carrier in future is reachable as `pr.fpr.<name>` and gains no `f_` counterpart.

As there, the two reads differ where the block is absent — `pr.f_kt` returns `nothing`, `pr.fpr.kt` throws — so guard first and then read through `fpr`.

## Validation

Defining `N = length(pr.mu)`.

  - If any of `kt`, `L2`, or `S2` are provided, all must be provided, non-empty, and `size(kt) == (N^2, N^2)`, `size(L2) == size(S2) == (div(N * (N + 1), 2), N^2)`.
  - If `sk` or `V` are provided, both must be provided, non-empty, and `size(sk) == (N, N^2)`, `size(V) == (N, N)`.
  - If that first triple is provided and `sk` is too, `D2` must be provided, non-empty, and `size(D2) == size(transpose(L2))`. `D2` carries no other rule: it is the one moment field the constructor accepts on its own, and a carrier holding it alone is legal.
  - If `fpr` is provided, `pr.fpr` must be provided and `fpr.pr === pr.fpr` — the factor distribution the factor co-moments were computed against is the low order carrier's own factor block, not a second copy of it. The converse does not hold: a low order factor block with no factor co-moments is ordinary, so `fpr === nothing` is always allowed. Everything internal to the factor block, including its own shapes against its own `N`, is validated by its own constructor.

## View parameters

`HighOrderPrior` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - It reads no argument beyond `i`. Further positional arguments are accepted and ignored.
  - `pr` recurses through [`port_opt_view`](@ref) with `i`, which is where every low order field is cut.
  - `kt` is indexed by a fourth-moment index derived from `i`, not by `i` itself. It is ``N^2 \\times N^2`` over ordered pairs of assets, so the asset index does not address it.
  - `sk` is cut by `i` on its asset axis and by that same fourth-moment index on its pair axis.
  - `V` is **recomputed** from the cut `sk` rather than cut. It is a spectral quantity of the coskewness matrix, so the submatrix of `V` is not the `V` of the submatrix.
  - `D2`, `L2` and `S2` are **rebuilt at the subproblem's asset count** rather than cut. They are combinatorial matrices of that count alone, and carry no asset-specific content to preserve.
  - `skmp` passes through unchanged. It is the matrix-processing estimator, which the recomputation of `V` reads.
  - `fpr` passes through unchanged, because it holds co-moments over factors rather than over assets. Forwarding it by identity is also what keeps `fpr.pr === pr.fpr` true of the view, since the low order view forwards its own factor block the same way.

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
       │       X ┼ 2×2 Matrix{Float64}
       │     o_X ┼ nothing
       │      mu ┼ Vector{Float64}: [0.02, 0.03]
       │   sigma ┼ 2×2 Matrix{Float64}
       │    chol ┼ nothing
       │       w ┼ nothing
       │     ens ┼ nothing
       │     kld ┼ nothing
       │      ow ┼ nothing
       │      rr ┼ nothing
       │     fpr ┼ nothing
       │       Z ┴ nothing
    kt ┼ 4×4 Matrix{Float64}
    D2 ┼ 4×3 SparseArrays.SparseMatrixCSC{Int64, Int64}
    L2 ┼ 3×4 SparseArrays.SparseMatrixCSC{Int64, Int64}
    S2 ┼ 3×4 SparseArrays.SparseMatrixCSC{Int64, Int64}
    sk ┼ 2×4 Matrix{Float64}
     V ┼ 2×2 Matrix{Float64}
  skmp ┼ nothing
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
    $(field_dict[:fpr])
    """
    fpr
    function HighOrderPrior(pr::AbstractPriorResult, kt::Option{<:MatNum},
                            D2::Option{<:MatNum}, L2::Option{<:MatNum},
                            S2::Option{<:MatNum}, sk::Option{<:MatNum}, V::Option{<:MatNum},
                            skmp::Option{<:AbstractMatrixProcessingEstimator},
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
        return new{typeof(pr), typeof(kt), typeof(D2), typeof(L2), typeof(S2), typeof(sk),
                   typeof(V), typeof(skmp), typeof(fpr)}(pr, kt, D2, L2, S2, sk, V, skmp,
                                                         fpr)
    end
end
function HighOrderPrior(; pr::AbstractPriorResult, kt::Option{<:MatNum} = nothing,
                        D2::Option{<:MatNum} = nothing, L2::Option{<:MatNum} = nothing,
                        S2::Option{<:MatNum} = nothing, sk::Option{<:MatNum} = nothing,
                        V::Option{<:MatNum} = nothing,
                        skmp::Option{<:AbstractMatrixProcessingEstimator} = nothing,
                        fpr::Option{<:HighOrderPrior} = nothing)::HighOrderPrior
    return HighOrderPrior(pr, kt, D2, L2, S2, sk, V, skmp, fpr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of a [`HighOrderPrior`](@ref) restricted to assets at index `i`, slicing all relevant moment tensors accordingly.

The factor block is forwarded **unsliced**, as it is on [`LowOrderPrior`](@ref): `i` indexes assets, and `fpr` holds co-moments over factors. Forwarding it by identity is also what keeps `fpr.pr === pr.fpr` true of the view, since the low order view forwards its own factor block the same way.

# Algorithm

 1. Make `idx`, the fourth-moment index that addresses the co-moment tensors of the assets at `i`, with [`fourth_moment_index_generator`](@ref) against the carrier's full asset count.
 2. Cut the coskewness matrix to `i` on its asset axis and to `idx` on its pair axis, with [`nothing_scalar_array_view_odd_order`](@ref), giving `sk`. A carrier that holds none keeps `nothing`.
 3. Recompute `V` from the `sk` of step 2 and the cut returns matrix, with [`negative_spectral_coskewness`](@ref) and the carrier's `skmp`. `V` is a spectral quantity of the coskewness matrix, so it is rebuilt rather than cut. When step 2 gave `nothing`, `V` is `nothing`.
 4. Rebuild `D2`, `L2` and `S2` at the subproblem's asset count with [`dup_elim_sum_view`](@ref), rather than cutting them. Take all three when the carrier holds `D2`, take `L2` and `S2` alone and leave `D2` as `nothing` when it holds `S2` but no `D2`, and take none when it holds neither.
 5. Rebuild the carrier through its ordinary keyword constructor: `pr` recursed through [`port_opt_view`](@ref) with `i`, `kt` indexed by `idx`, the values of steps 2 to 4, and `skmp` and `fpr` forwarded unchanged. Every `@argcheck` of the constructor therefore runs on the view.

# Arguments

  - $(arg_dict[:pr])
  - `i`: Asset indices the view keeps.
  - `args...`: Additional arguments (ignored).

# Returns

  - `pr::HighOrderPrior`: The carrier restricted to the assets at `i`.

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
        V = negative_spectral_coskewness(sk, view(pr.X, :, i), skmp)
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
                          S2 = S2, sk = sk, V = V, skmp = skmp, fpr = pr.fpr)
end
# The flat `f_`-prefixed names are virtual reads of the nested factor block, mirroring
# [`LowOrderPrior`](@ref): code written against the pre-nesting shape is unaffected, and
# `f_D2`/`f_L2`/`f_S2`/`f_skmp` come for free. `compute` with a lambda rather than a dotted
# locator, because a dotted locator throws a [`PropertyPathError`](@ref) on a `nothing` node
# where these must return `nothing` — that is what the old flat fields did with no factor
# block. They are declared before `forward(pr)` only for reading order: the embedded
# `LowOrderPrior` has no `f_kt`/`f_sk`/`f_V` of its own to shadow.
#
# `fpr` is the carrier's own field, so it resolves before `forward(pr)` and names the *high*
# order factor block rather than the low order one. Reads through it are unaffected by the
# shift: the nested carrier forwards to its own `pr`, which the constructor pins to
# `pr.fpr`, so `hop.fpr.mu` is the factor mean either way.
#
# ForwardSelection of the remaining unknown property names to the embedded `pr` prior gives
# transparent access to the low-order moment fields (see [`@forward_properties`](@ref)).
@forward_properties HighOrderPrior begin
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

Rebuild a prior result through its ordinary keyword constructor, patching the fields named in `patch`.

One method per carrier, because the carrier's constructor is *named* here rather than recovered by reflection. Recovering it generically would mean either `Base.typename(T).wrapper` or a dependency on `ConstructionBase`, and neither buys anything: the field *list* is already derived, via [`prior_field_values`](@ref), so a carrier that gains a field needs no edit here. Only a new carrier type needs a method — and until it has one it gets a `MethodError` naming this function, rather than being reconstructed by machinery that has never seen it.

Reconstruction runs the carrier's full validation, which is the point of routing through the constructor at all: a patch that leaves the carrier internally inconsistent throws exactly as a hand-written constructor call would. Keyword arguments are order-independent, so `patch` may name fields in any order.

These methods are defined here, after both carriers, because they dispatch on the concrete types.

# Algorithm

Both methods run the same three steps, and differ only in the constructor step 3 names.

 1. Read the carrier's own fields into a named tuple with [`prior_field_values`](@ref), keyed in declaration order.
 2. Merge `patch` over that tuple. A field `patch` names takes the patch's value, and every field it does not name keeps the carrier's.
 3. Splat the merged tuple into the carrier's keyword constructor — `LowOrderPrior` in the first method, `HighOrderPrior` in the second — and return the carrier it builds. Every `@argcheck` of that constructor runs on the merged values.

# Arguments

  - `pr`: Prior result to rebuild.
  - `patch`: Named tuple of field overrides. Every name must be a field of `typeof(pr)` — [`forward_prior`](@ref) checks that before calling, so a bad name is reported against the rule rather than as an unsupported keyword.

# Returns

  - `pr::AbstractPriorResult`: Reconstructed result of the same carrier type.

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

Return every property name a prior result can answer, unioned over the carriers.

This is the candidate pool [`propagatable_contract_violations`](@ref) checks an `@pprop` field
name against: the generated `factory(x, pr::AbstractPriorResult, args...)` reads
`getproperty(pr, :field)`, and the carrier that arrives is not known at the declaration.

The names of the two carriers are written out, as in [`reconstruct_prior`](@ref); their fields
are derived, so a carrier that gains a field needs no edit here. `HighOrderPrior` forwards the
whole of the `pr` it wraps, so the low-order names are properties of it too without being
fields — that forwarding is the reason a plain `fieldnames` of one carrier is not the pool.

These methods are defined here, after both carriers, because they name the concrete types.

# Algorithm

 1. Concatenate the field names of [`LowOrderPrior`](@ref) and of [`HighOrderPrior`](@ref) into one vector of `Symbol`.
 2. Remove the duplicates in place, and return the vector. `fpr` is a field of both carriers, so the concatenation is not already unique.

# Returns

  - `pool::Vector{Symbol}`: Every property name a prior result can answer, without duplicates.

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

export prior, LowOrderPrior, HighOrderPrior
public forward_prior
