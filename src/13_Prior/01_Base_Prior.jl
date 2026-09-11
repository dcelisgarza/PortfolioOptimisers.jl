"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all prior estimators.

`AbstractPriorEstimator` is the base type for all estimators that compute prior information from asset and/or factor returns. All concrete prior estimators should subtype this type to ensure a consistent interface for prior computation and integration with portfolio optimisation workflows.

# Interfaces

In order to implement a new prior estimator which will work seamlessly with the library, subtype the family that names the returns it reads — [`AbstractLowOrderPriorEstimator_A`](@ref), [`AbstractLowOrderPriorEstimator_F`](@ref), [`AbstractLowOrderPriorEstimator_AF`](@ref) or [`AbstractHighOrderPriorEstimator_F`](@ref) — with all necessary parameters as part of the struct, and implement the following method:

  - `prior(pe::AbstractPriorEstimator, X::MatNum, F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, kwargs...) -> AbstractPriorResult`: Estimate the prior from the returns matrices and the Asset Panel.

The family fixes the signature. A member of the `_A` family declares `F` as `args...` and never reads it, a member of the `_F` family declares it `F::MatNum` and requires it, and a member of the `_AF` family declares it `F::Option{<:MatNum} = nothing` and reads it when it is there.

`pnl` is the Asset Panel the carrier held, and the [`ReturnsResult`](@ref) method forwards it to every estimator. Take it and ignore it unless the estimator is fitted on a panel, as [`CrossSectionalFactorPrior`](@ref) is. An estimator that wraps another over the assets forwards it unchanged, so that the wrapped estimator composes; one that wraps a prior over the factors does not, because no panel describes a factor axis. An estimator that declares `args...` takes it there and needs no further declaration.

The method returns the carrier of its own order: a low order estimator returns a [`LowOrderPrior`](@ref), and a high order estimator returns a [`HighOrderPrior`](@ref). An estimator that wraps another rebuilds the wrapped result with [`forward_prior`](@ref) rather than by a hand-written constructor call, so that every field it does not name survives the hop.

The [`ReturnsResult`](@ref) method of [`prior`](@ref) is supplied by this file and needs no implementation.

## Arguments

  - $(arg_dict[:pe])
  - $(arg_dict[:X])
  - `F`: Factor returns matrix, or `nothing`.
  - $(arg_dict[:pnl_prior])
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
    fpr ┴ nothing
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

Nor does the shape say whether the fit *reads* factor returns, and [`needs_factor_returns`](@ref) answers that three ways. A member that requires them answers `true`, a member that never reads them answers `false`, and this shape answers `nothing` by default: *the type does not say; take what the fold is given*, which is what its batch verb does. Every member of this shape the library ships embeds another prior and hands `F` to it, so each defines the recursion and answers the leaf's value. A caller's own subtype that embeds a prior should define the same recursion; one that reads `F` itself may leave the default.

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

`AbstractHiLoOrderPriorEstimator_F` is the type-level half of the test for *this estimator cannot run without factor returns*, taken across both orders at once: a member answers `true` to [`needs_factor_returns`](@ref). The doors that check for a missing factor matrix ask the predicate rather than this union, because a factor leaf may sit under a host whose own factor argument is optional, and the predicate walks the tree where an `isa` test reads the host alone.

# Related

  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractHighOrderPriorEstimator_F`](@ref)
  - [`needs_factor_returns`](@ref)
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

`Pr_RR` is the bridge the clustering, phylogeny and centrality forwarders below dispatch on. Each of them reads `X` off its carrier and delegates to the asset-returns method, so an estimator that needs returns can be driven from a fitted prior or from the raw data with one method apiece rather than two. Where both carriers are present, [`returns_matrix_picker`](@ref) picks between them; both travel on to the estimator tree as `pr` and `rd`, so a [`FeatureDistance`](@ref) resolves its Asset Panel from them.

# Related

  - [`AbstractPriorResult`](@ref)
  - [`ReturnsResult`](@ref)
  - [`returns_matrix_picker`](@ref)
"""
const Pr_RR = Union{<:AbstractPriorResult, <:ReturnsResult}
"""
    prior(pe::AbstractPriorEstimator, rd::ReturnsResult; kwargs...)

Compute prior information from asset and/or factor returns using a prior estimator.

`prior` applies the specified prior estimator to a `ReturnsResult` object, extracting asset and factor returns and passing them, along with any additional information, to the estimator. Returns a prior result containing computed moments and other prior information for use in portfolio optimisation workflows.

This method is the entry point every caller uses, and it is written once here. What each estimator implements is the returns-matrix method that this one delegates to; [`AbstractPriorEstimator`](@ref) states that contract.

# Algorithm

 1. Check that `rd` carries asset returns, so that the estimator is not handed a `nothing` for `X`.
 2. When `pe` requires factor returns — when [`needs_factor_returns`](@ref) answers `true`, which walks the tree to a factor leaf under an optional-argument host — check that `rd` carries them. The check is made here so that the caller reads a named error against `rd.F` rather than a `MethodError` against the leaf's own signature one call later.
 3. Call the estimator's returns-matrix method with `rd.X`, `rd.F` and `rd.pnl`, forwarding `rd.iv` and `rd.ivpa` as keyword arguments alongside `kwargs`, and return the prior result it produces.

The Asset Panel travels as the third positional argument for the same reason `rd.F` travels as the second: a wrapping prior holds no carrier, so it can compose an estimator that is fitted on a panel only if the panel reaches its own returns-matrix method. Every returns-matrix method takes the argument, every wrapping prior forwards it unchanged to the estimator it nests over the assets, and an estimator that reads no panel ignores it.

# Arguments

  - $(arg_dict[:pe])
  - `rd`: Asset and/or factor returns result.
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Validation

  - `!isnothing(rd.X)`.
  - `!isnothing(rd.F)`, when `needs_factor_returns(pe) === true`.

# Returns

  - `pr::AbstractPriorResult`: Result object containing computed prior information.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`needs_factor_returns`](@ref)
  - [`ReturnsResult`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
"""
function prior(pe::AbstractPriorEstimator, rd::ReturnsResult; kwargs...)
    @argcheck(!isnothing(rd.X), IsNothingError)
    assert_factor_returns(pe, rd.F)
    return prior(pe, rd.X, rd.F, rd.pnl; iv = rd.iv, ivpa = rd.ivpa, kwargs...)
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
    clusterise(cle::AbstractClustersEstimator, pr::AbstractPriorResult; kwargs...)

Clusterise asset or factor returns from a prior result using a clustering estimator.

`clusterise` applies the specified clustering estimator to the asset returns matrix contained in the prior result object, producing a clustering result for use in phylogeny analysis, constraint generation, or portfolio construction.

# Algorithm

 1. Pick the asset returns matrix `X` from the carrier that `x_src` names, with [`returns_matrix_picker`](@ref).
 2. Call the asset-returns method of [`clusterise`](@ref) with `X`, passing both carriers on as `pr` and `rd`, and return the clustering result it produces.

# Arguments

  - `cle`: Clustering estimator.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Read for `X` only when `x_src` is `:data`, and passed on to the estimator tree.
  - $(arg_dict[:x_src])
  - `kwargs...`: Additional keyword arguments passed to the clustering estimator.

# Returns

  - `clr::AbstractClusteringResult`: Result object containing clustering information.

# Related

  - [`ClustersEstimator`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`clusterise`](@ref)
"""
function clusterise(cle::AbstractClustersEstimator, pr::Pr_RR;
                    rd::Option{<:ReturnsResult} = nothing, x_src::Symbol = :prior,
                    kwargs...)
    X = returns_matrix_picker(pr, rd, x_src)
    return clusterise(cle, X; pr = pr, rd = rd, kwargs...)
end
"""
    phylogeny_matrix(pl::NwE_ClE_Cl, pr::AbstractPriorResult;
                     kwargs...)

Compute the phylogeny matrix from asset returns in a prior result using a network or clustering estimator.

`phylogeny_matrix` applies the specified network or clustering estimator to the asset returns matrix contained in the prior result object, producing a phylogeny matrix for use in constraint generation, centrality analysis, or portfolio construction.

# Algorithm

 1. Pick the asset returns matrix `X` from the carrier that `x_src` names, with [`returns_matrix_picker`](@ref).
 2. Call the asset-returns method of [`phylogeny_matrix`](@ref) with `X`, passing both carriers on as `pr` and `rd`, and return the phylogeny result it produces.

# Arguments

  - `pl`: Network estimator, clusters estimator, or clustering result.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Read for `X` only when `x_src` is `:data`, and passed on to the estimator tree.
  - $(arg_dict[:x_src])
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Returns

  - `plr::PhylogenyResult`: Result object containing the phylogeny matrix.

# Related

  - [`NetworkEstimator`](@ref)
  - [`ClustersEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`phylogeny_matrix`](@ref)
"""
function phylogeny_matrix(pl::NwE_ClE_Cl, pr::Pr_RR; rd::Option{<:ReturnsResult} = nothing,
                          x_src::Symbol = :prior, kwargs...)
    X = returns_matrix_picker(pr, rd, x_src)
    return phylogeny_matrix(pl, X; pr = pr, rd = rd, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute phylogeny constraints from asset returns in a prior result using a phylogeny constraint estimator.

`phylogeny_constraints` delegates to the asset-returns variant by extracting `X` from `pr` (or `rd` if provided and `x_src` is `:data`).

# Algorithm

 1. Pick the asset returns matrix `X` from the carrier that `x_src` names, with [`returns_matrix_picker`](@ref).
 2. Call the asset-returns method of [`phylogeny_constraints`](@ref) with `X`, passing both carriers on as `pr` and `rd`, and return the constraint result it produces.

# Arguments

  - `plc`: Phylogeny constraint estimator.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Read for `X` only when `x_src` is `:data`, and passed on to the estimator tree.
  - $(arg_dict[:x_src])
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Returns

  - Phylogeny constraint result.

# Related

  - [`AbstractPhylogenyConstraintEstimator`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`phylogeny_constraints`](@ref)
"""
function phylogeny_constraints(plc::AbstractPhylogenyConstraintEstimator, pr::Pr_RR;
                               rd::Option{<:ReturnsResult} = nothing,
                               x_src::Symbol = :prior, kwargs...)
    X = returns_matrix_picker(pr, rd, x_src)
    return phylogeny_constraints(plc, X; pr = pr, rd = rd, kwargs...)
end
"""
    centrality_vector(cte::CentralityEstimator, pr::AbstractPriorResult; kwargs...)

Compute the centrality vector for a centrality estimator and prior result.

`centrality_vector` applies the centrality algorithm in the estimator to the network constructed from the asset returns in the prior result, returning centrality scores for each asset.

# Algorithm

 1. Pick the asset returns matrix `X` from the carrier that `x_src` names, with [`returns_matrix_picker`](@ref).
 2. Call the asset-returns method of [`centrality_vector`](@ref) with `X`, passing both carriers on as `pr` and `rd`, and return the centrality result it produces.

# Arguments

  - $(arg_dict[:cte])
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Read for `X` only when `x_src` is `:data`, and passed on to the estimator tree.
  - $(arg_dict[:x_src])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `plr::PhylogenyResult`: Result object containing the centrality vector.

# Related

  - [`CentralityEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`centrality_vector`](@ref)
"""
function centrality_vector(cte::CentralityEstimator, pr::Pr_RR;
                           rd::Option{<:ReturnsResult} = nothing, x_src::Symbol = :prior,
                           kwargs...)
    X = returns_matrix_picker(pr, rd, x_src)
    return centrality_vector(cte, X; pr = pr, rd = rd, kwargs...)
end
"""
    centrality_vector(pl::NwE_ClE_Cl, ct::AbstractCentralityAlgorithm,
                      pr::AbstractPriorResult; kwargs...)

Compute the centrality vector for a network or clustering estimator and centrality algorithm.

`centrality_vector` constructs the phylogeny matrix from the asset returns in the prior result, builds a graph, and computes node centrality scores using the specified centrality algorithm.

# Algorithm

 1. Pick the asset returns matrix `X` from the carrier that `x_src` names, with [`returns_matrix_picker`](@ref).
 2. Call the asset-returns method of [`centrality_vector`](@ref) with `X`, passing both carriers on as `pr` and `rd`, and return the centrality result it produces.

# Arguments

  - `pl`: Network estimator, clusters estimator, or clustering result.
  - $(arg_dict[:cta])
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Read for `X` only when `x_src` is `:data`, and passed on to the estimator tree.
  - $(arg_dict[:x_src])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `plr::PhylogenyResult`: Result object containing the centrality vector.

# Related

  - [`NetworkEstimator`](@ref)
  - [`CentralityEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`centrality_vector`](@ref)
"""
function centrality_vector(pl::NwE_ClE_Cl, ct::AbstractCentralityAlgorithm, pr::Pr_RR;
                           rd::Option{<:ReturnsResult} = nothing, x_src::Symbol = :prior,
                           kwargs...)
    X = returns_matrix_picker(pr, rd, x_src)
    return centrality_vector(pl, ct, X; pr = pr, rd = rd, kwargs...)
end
"""
    average_centrality(pl::NwE_Pl_ClE_Cl,
                       ct::AbstractCentralityAlgorithm, w::VecNum,
                       pr::AbstractPriorResult; kwargs...)

Compute the weighted average centrality for a network or phylogeny result.

`average_centrality` computes the centrality vector using the specified network or phylogeny estimator and centrality algorithm, then returns the weighted average using the provided portfolio weights.

# Algorithm

 1. Compute the centrality result with the [`Pr_RR`](@ref) method of [`centrality_vector`](@ref), forwarding `rd` and `x_src` unchanged. The source selection is therefore made once, there, and this method never reads a carrier itself.
 2. Return the dot product of that result's `X`, the centrality vector, with the weights `w`.

# Arguments

  - `pl`: Network estimator or phylogeny result.
  - $(arg_dict[:cta])
  - `w`: Portfolio weights vector.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Read for `X` only when `x_src` is `:data`, and passed on to the estimator tree.
  - $(arg_dict[:x_src])
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
                            x_src::Symbol = :prior, kwargs...)
    return LinearAlgebra.dot(centrality_vector(pl, ct, pr; rd = rd, x_src = x_src,
                                               kwargs...).X, w)
end
"""
    average_centrality(cte::CentralityEstimator, w::VecNum, pr::AbstractPriorResult;
                       kwargs...)

Compute the weighted average centrality for a centrality estimator.

`average_centrality` applies the centrality algorithm in the estimator to the network constructed from the asset returns in the prior result, then returns the weighted average using the provided portfolio weights.

# Algorithm

 1. Pick the asset returns matrix `X` from the carrier that `x_src` names, with [`returns_matrix_picker`](@ref).
 2. Call the asset-returns method of [`average_centrality`](@ref) with `X`, passing both carriers on as `pr` and `rd`, and return the weighted average it produces.

The estimator method picks the carriers itself, where the network-and-algorithm method above delegates that to [`centrality_vector`](@ref). The two reach the same selection: `cte` carries `pl` and `ct` in its own fields, so the asset-returns method it calls is the one the other method's step 1 would have reached.

# Arguments

  - $(arg_dict[:cte])
  - `w`: Portfolio weights vector.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Read for `X` only when `x_src` is `:data`, and passed on to the estimator tree.
  - $(arg_dict[:x_src])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `ac::Number`: Weighted average centrality.

# Related

  - [`CentralityEstimator`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`centrality_vector`](@ref)
  - [`average_centrality`](@ref)
"""
function average_centrality(cte::CentralityEstimator, w::VecNum, pr::Pr_RR;
                            rd::Option{<:ReturnsResult} = nothing, x_src::Symbol = :prior,
                            kwargs...)
    X = returns_matrix_picker(pr, rd, x_src)
    return average_centrality(cte, w, X; pr = pr, rd = rd, kwargs...)
end
"""
    asset_phylogeny(pl::NwE_ClE_Cl,
                    w::VecNum, pr::AbstractPriorResult; dims::Int = 1, kwargs...)

Compute the asset phylogeny score for a portfolio allocation using a phylogeny estimator or clustering result and a prior result.

This function computes the phylogeny matrix from the asset returns in the prior result using the specified phylogeny estimator or clustering result, then evaluates the asset phylogeny score for the given portfolio weights. The asset phylogeny score quantifies the degree of phylogenetic (network or cluster-based) structure present in the portfolio allocation.

# Algorithm

 1. Pick the asset returns matrix `X` from the carrier that `x_src` names, with [`returns_matrix_picker`](@ref).
 2. Call the asset-returns method of [`asset_phylogeny`](@ref) with `X`, passing both carriers on as `pr` and `rd`, and return the score it produces.

# Arguments

  - `pl`: Phylogeny estimator or clustering result used to compute the phylogeny matrix.
  - `w`: Portfolio weights vector.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Read for `X` only when `x_src` is `:data`, and passed on to the estimator tree.
  - $(arg_dict[:x_src])
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
  - [`asset_phylogeny`](@ref): The asset-returns methods this one delegates to. They build the phylogeny matrix, add up the gross weight of the related pairs, and divide by the gross weight of every pair. That is where the score's closed form and its numbered steps are stated.
"""
function asset_phylogeny(pl::NwE_ClE_Cl, w::VecNum, pr::Pr_RR;
                         rd::Option{<:ReturnsResult} = nothing, x_src::Symbol = :prior,
                         kwargs...)
    X = returns_matrix_picker(pr, rd, x_src)
    return asset_phylogeny(pl, w, X; pr = pr, rd = rd, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute centrality constraints from asset returns in a prior result using a centrality constraint estimator.

`centrality_constraints` delegates to the asset-returns variant by extracting `X` from `pr` (or `rd` if provided and `x_src` is `:data`).

# Algorithm

 1. Pick the asset returns matrix `X` from the carrier that `x_src` names, with [`returns_matrix_picker`](@ref).
 2. Call the asset-returns method of [`centrality_constraints`](@ref) with `X`, passing both carriers on as `pr` and `rd`, and return the constraint result it produces.

# Arguments

  - `ccs`: Centrality constraint estimator or vector thereof.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Read for `X` only when `x_src` is `:data`, and passed on to the estimator tree.
  - $(arg_dict[:x_src])
  - `kwargs...`: Additional keyword arguments passed to the estimator. `strict` is read by the asset-returns variant, which reports a dropped zero centrality vector through it.

# Returns

  - Centrality constraint result.

# Related

  - [`AbstractPriorResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`centrality_constraints`](@ref)
"""
function centrality_constraints(ccs::CC_VecCC, pr::Pr_RR;
                                rd::Option{<:ReturnsResult} = nothing,
                                x_src::Symbol = :prior, kwargs...)
    X = returns_matrix_picker(pr, rd, x_src)
    return centrality_constraints(ccs, X; pr = pr, rd = rd, kwargs...)
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
        rr::Option{<:AbstractLoadingsRegressionResult} = nothing,
        fpr::Option{<:LowOrderPrior} = nothing
    ) -> LowOrderPrior

Keywords correspond to the struct's fields.

## The factor block

A prior fit through a factor model carries two distributions: one over the assets, in the carrier's own fields, and one over the factors. The factor one is a **nested `LowOrderPrior`** in `fpr` rather than a set of `f_`-prefixed flat fields, so it gains every field the carrier has — `w`, `ens`, `kld` and `ow` as well as `mu` and `sigma` — and gains any field added in future without a second edit. Its `X` is the factor returns matrix, over the same observations as the asset `X`.

`fpr` travels with `rr`: the two are the factor block, and the constructor requires them together or not at all. `rr` is what projects the block onto the assets (`mu ≈ rr.M * fpr.mu + rr.b`), so a factor distribution with no loadings could not be read against this asset axis.

`rr` is bound to [`AbstractLoadingsRegressionResult`](@ref), the root that states a member carries the loadings matrix `M`, so a [`Regression`](@ref) and a [`CrossSectionalFactorModel`](@ref) both sit in the slot. The bound is the loadings criterion and not a fitting geometry: every invariant the constructor checks here reads `rr.M` alone, `fpr` sits on the axis `M`'s columns name, and every consumer of the slot reads `M`, or reads `L` and gets `M` back when `L` is unset.

One property of the block does not follow from the slot, and a consumer that needs it must ask. A member fitted in a re-based Factor Family states so through [`has_family_rebasis`](@ref), and its `fpr.sigma` is then singular by construction, because the raw factor axis is a linear image of the re-based one. Projecting through `M` is unaffected — that is what [`HighOrderFactorPriorEstimator`](@ref) does — but inverting or factorising `fpr.sigma` has no answer. The inversion does not say so: it raises nothing and returns a result whose scale looks ordinary, so [`BayesianBlackLittermanPrior`](@ref) refuses such a carrier rather than reporting one.

The flat names are **virtual reads** of the nested block, so code written against the old shape is unaffected: `pr.f_mu`, `pr.f_sigma` and `pr.f_w` return `fpr.mu`, `fpr.sigma` and `fpr.w`, or `nothing` when there is no factor block, and `pr.f_ens`, `pr.f_kld` and `pr.f_ow` come with them. They are properties, not fields — [`forward_prior`](@ref) and [`prior_field_values`](@ref) see only `fpr`.

### Which read is idiomatic

**`pr.fpr.mu` is the public read**; the flat `f_`-prefixed names are a **compatibility surface**, kept so that code written against the pre-nesting shape keeps working, and useful where a value-or-`nothing` read without branching is wanted.

The reason is not taste. The flat surface is **partial and frozen**: there are six flat names over eleven fields, so `fpr.X` — the factor returns matrix — and `fpr.chol` and `fpr.rr` have no flat spelling at all and never will. A surface that cannot express the whole block cannot be the way to read it. The set is fixed at the six here and the seven on [`HighOrderPrior`](@ref); a field added to a carrier in future is reachable as `pr.fpr.<name>` and gains no `f_` counterpart, so nothing has to be added in two places to stay complete.

The two reads also differ where the block is absent, which is the one case worth checking before choosing: `pr.f_mu` returns `nothing`, while `pr.fpr.mu` throws, because `fpr` is `nothing`. Guard with [`assert_prior_regression`](@ref) — `rr` and `fpr` are supplied together or not at all, so checking `rr` establishes the whole block — and then read through `fpr`.

## Composition: what a wrapping estimator forwards

Most prior estimators wrap another and return a carrier built from the one they were handed. Which fields survive that hop is governed by a single rule, recorded in ADR 0046 and enforced by [`forward_prior`](@ref):

> **Forward when forwarding is correct; drop only where forwarding would state something false; document every drop in the estimator's docstring.**

Consistency of the returned result is the criterion, and destroying a value the caller explicitly computed is not an acceptable way to buy it — so forwarding is the default, and each estimator's docstring lists the fields it drops and why. Two fields are *bound* to another and therefore never forwarded alone: `chol` is bound to `sigma` (it takes precedence over `sigma` at every consumer, so a stale factor is silently used in place of the updated covariance), and `ens`, `kld` and `ow` are bound to `w` (they are diagnostics *of* those weights). `forward_prior` refuses a forward that would break either binding.

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
  - If `o_X` is not `nothing`, `o_X !== X`, `size(o_X) == size(X)`, and `rr` is not `nothing`. `o_X !== X` is an **identity** test and not an equality test, so `o_X = copy(X)` is admitted where `o_X = X` raises. The two calls read identically at a call site, and only the first carries a matrix a later change to `X` cannot follow. What the guard rejects is the carrier that has no original distinct from the one it asserts, not a matrix whose values happen to agree.
  - If `chol` is not `nothing`, `!isempty(chol)` and `length(mu) == size(chol, 2)`.

## View parameters

`LowOrderPrior` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - It reads no argument beyond `i`. Further positional arguments are accepted and ignored.
  - `rr` recurses through [`port_opt_view`](@ref) with `i`, which cuts the loadings down on their asset axis.
  - `X`, `o_X`, `mu`, `sigma` and `chol` are sliced to `i` on the asset axis. `o_X` takes the same cut as `X`, so a subproblem's original returns stay the caller's returns for that subproblem's assets.
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
    fpr ┴ nothing
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
    function LowOrderPrior(X::MatNum, o_X::Option{<:MatNum}, mu::VecNum, sigma::MatNum,
                           chol::Option{<:MatNum}, w::Option{<:ObsWeights},
                           ens::Option{<:Number}, kld::Option{<:Num_VecNum},
                           ow::Option{<:VecNum},
                           rr::Option{<:AbstractLoadingsRegressionResult},
                           fpr::Option{<:LowOrderPrior})
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
        return new{typeof(X), typeof(o_X), typeof(mu), typeof(sigma), typeof(chol),
                   typeof(w), typeof(ens), typeof(kld), typeof(ow), typeof(rr),
                   typeof(fpr)}(X, o_X, mu, sigma, chol, w, ens, kld, ow, rr, fpr)
    end
end
function LowOrderPrior(; X::MatNum, o_X::Option{<:MatNum} = nothing, mu::VecNum,
                       sigma::MatNum, chol::Option{<:MatNum} = nothing,
                       w::Option{<:ObsWeights} = nothing, ens::Option{<:Number} = nothing,
                       kld::Option{<:Num_VecNum} = nothing, ow::Option{<:VecNum} = nothing,
                       rr::Option{<:AbstractLoadingsRegressionResult} = nothing,
                       fpr::Option{<:LowOrderPrior} = nothing)::LowOrderPrior
    return LowOrderPrior(X, o_X, mu, sigma, chol, w, ens, kld, ow, rr, fpr)
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
 3. Rebuild the carrier through its ordinary keyword constructor, naming every field: `X` and `mu` cut to `i`, `sigma` cut to `i` on both axes, `chol` and `o_X` from the two steps above, `rr` recursed through [`port_opt_view`](@ref) with `i`, and `w`, `ens`, `kld`, `ow` and `fpr` forwarded unchanged. Every `@argcheck` of the constructor therefore runs on the view.

# Arguments

  - $(arg_dict[:pr])
  - `i`: Asset indices the view keeps.
  - `args...`: Additional arguments (ignored).

# Returns

  - `pr::LowOrderPrior`: The carrier restricted to the assets at `i`, holding views rather than copies.

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
                         kld = pr.kld, ow = pr.ow, rr = port_opt_view(pr.rr, i),
                         fpr = pr.fpr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Derive the Investable Mask of a prior result: `true` at every asset whose moments are finite.

A Prior Estimator fits on the coverage universe and returns a result on the **full** asset universe, where an asset it could not estimate carries `NaN` in `mu` and on the diagonal of `sigma`. The mask is *derived here and stored nowhere*: no prior result carries a mask field, so a caller that needs one calls this, and a caller that reduces a result keeps the mask it was given. A reduced result can no longer yield it, because its `mu` is finite everywhere.

The all-investable case returns `nothing` rather than a mask of every `true`. That sentinel is what keeps a universe with nothing to exclude on the path it took before the mask existed: no reduction, no expansion, no allocation.

An off-diagonal `NaN` in `sigma` is not read. A non-investable asset may carry `NaN` across its whole row and column, and the diagonal alone decides, so the mask costs one pass over two vectors.

# Algorithm

 1. Take the elementwise conjunction of `isfinite.(pr.mu)` and `isfinite.(diag(pr.sigma))`.
 2. Throw an `IsEmptyError` when the conjunction holds no `true`. An optimisation over no asset has no answer to give, and a zero-asset problem passed downstream fails further from its cause.
 3. Return `nothing` when the conjunction holds no `false`.
 4. Return the conjunction otherwise.

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

Derive the Investable Mask of a fitted prior, and give a view builder the universe it may write rows over.

A view is a **dense linear form over the asset axis**, and a departed asset carries `NaN` in `mu` and on the diagonal of `sigma`. `A[i] == 0` does not protect a row from it, because `0 * NaN` is `NaN`, so a view naming only *live* assets is poisoned exactly as thoroughly as one naming the asset that left: the row reaches the solver all `NaN`, and the fit fails naming something that is not the cause. Building the row on the investable columns is the whole fix, and it is the same reduction every optimiser takes at its entry — [`port_opt_view`](@ref) of the carrier at `findall(imsk)`, which ADR 0115 states and [#919](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/919) measured bit-exact against the hand-reduced oracle.

**Both view-taking prior families reduce here, and what they owe afterwards is theirs, not this door's.** An entropy pooling row runs over *observations*, so its solved probabilities carry no asset axis and nothing is expanded back — the moments come from the refit wrapped prior, which already holds the full-universe `NaN` frame. A Black–Litterman posterior is a *moment pair over the reduced assets*, so it has to be written back into a `NaN` frame of the full width with [`expand_moment`](@ref) before it leaves the estimator. That is the whole of the difference, and it is why this verb hands back the index rather than swallowing it.

The door also **mints the Non-Investable Axis** on the sets it hands the builders, with [`non_investable_sets`](@ref) after [`port_opt_view`](@ref) — after, because the view drops the axis so that a sub-problem cannot inherit its parent's departures. That is what lets a builder tell a departed name from a typo: ADR 0125 drops the row whole and in silence for the first, and keeps today's `strict_diagnostic` for the second.

**`sets` splits by dispatch and the mask by a condition**, and the asymmetry is the whole of the reason. `sets` is a field of a `@concrete` estimator, so whether it is `nothing` is a **type** fact, fixed per instantiation: the pair is static dispatch, it costs nothing, and it is what keeps the returned sets concretely a [`UniverseSets`](@ref). A single method over `Option{<:UniverseSets}` would answer a value-level `Union`, and the view builders declare `sets::UniverseSets` — so JET finds no method for the `Nothing` half at every builder call site, none of them reachable. [`investable_mask`](@ref), by contrast, answers a `Union{Nothing, BitVector}` that depends on the **data**: Julia union-splits a two-member `Union` and compiles a method pair back into this very branch, so dispatching on it would buy nothing and cost a unit in a swept file. Dispatch where the fact is a type; branch where it is a value.

`sets` of `nothing` returns early whatever the mask says: a view-taking estimator's constructor refuses `nothing` sets the moment any view is stated, so there is nothing to build and nothing to reduce for. The all-investable path returns its arguments untouched, so a gap-free fit pays one pass over two vectors and allocates nothing.

# Algorithm

 1. Return `nothing`, `sets` and no departed names when `sets` is `nothing`, which is the method the estimator's own field type selects.
 2. Otherwise derive the Investable Mask from the fitted prior with [`investable_mask`](@ref), and return `nothing`, `sets` and no departed names when it is `nothing`.
 3. Read the asset universe off `sets.dict[sets.xkey]`, and check it against the mask.
 4. Read the departed names with [`non_investable_names`](@ref).
 5. Take a [`port_opt_view`](@ref) of `sets` at `findall(imsk)`, mint the Non-Investable Axis on it with [`non_investable_sets`](@ref), and return the mask, the minted sets and the departed names.

# Arguments

  - $(arg_dict[:pr])
  - `sets`: The estimator's [`UniverseSets`](@ref), or `nothing`.

# Validation

  - `length(sets.dict[sets.xkey]) == length(imsk)`. A `DimensionMismatch` naming both counts is thrown otherwise, in place of the `BoundsError` the complement would raise.

# Returns

  - `(imsk, sets, ni)`: The Investable Mask or `nothing`, the reduced sets carrying the Non-Investable Axis, and the departed names. The mask is what [`investable_prior`](@ref) views at and what [`expand_moment`](@ref) writes back through, so a caller that reduces and expands needs nothing else.

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

View a fitted prior at the Investable Mask [`investable_views`](@ref) derived.

It is separate from [`investable_views`](@ref) because a staged entropy pooling fit **refits** its wrapped prior between stages, once per solve, and every refit has to be viewed again before the next stage's builders read it. The mask itself does not move — a column that could not be estimated stays unestimable under any reweighting of the observations — so it is derived once and this is applied many times.

`nothing` is the all-investable path and hands the prior straight back, so a gap-free fit allocates nothing. The split is a method pair here and a condition inside [`investable_views`](@ref), and the two are not in conflict: the mask arrives from a local whose `Union` Julia has already split at the call site, so each branch reaches this with a concrete argument. What it must not become is one method over `Option{BitVector}`, which would put the union back.

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

Name the departed assets for an estimator that reduces its **asset** axis while its views live on another one.

[`investable_views`](@ref) is the door for an estimator whose views resolve against `xkey`: it reduces the sets, mints the Non-Investable Axis on them, and the names fall out on the way. [`BayesianBlackLittermanPrior`](@ref) and [`FactorBlackLittermanPrior`](@ref) write their views on the **factor** axis, so they reduce their asset side and touch no view axis at all — there is nothing for them to mint, and going through that door would make them demand an asset universe they have no other use for. They still have a departure to report, and this is the least they need to report it.

Sets that are not stated at all answer an empty list rather than throwing, and an empty list is what [`announce_non_investable`](@ref) says nothing about. That is the honest outcome, and it is a real configuration: both members admit `sets` of `nothing` entirely, because a precomputed [`BlackLittermanViews`](@ref) resolves no name and needs no universe. A stated universe that does not describe this fit answers the same silence, because neither member reads an asset name for any other purpose and so nothing else has checked its length.

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
$(DocStringExtensions.TYPEDSIGNATURES)

Find the entries a mask-aware prior fills: the non-finite returns of an **investable** asset.

A non-investable asset keeps its whole `NaN` column, because the Investable Mask is derived from that gap and filling it would erase the mask. Only a column whose moments came back finite is filled.

# Arguments

  - `X`: Asset returns, `observations × assets`.
  - `imsk`: The Investable Mask, `true` at every asset whose prior moments were finite.

# Returns

  - `filled::Vector{Tuple{Int, Int}}`: The `(observation, asset)` pairs to fill, asset-major.

# Related

  - [`scenario_fill`](@ref)
  - [`scenario_fill_msg`](@ref)
  - [`investable_mask`](@ref)
"""
function scenario_fill_pairs(X::MatNum, imsk::BitVector)
    return [(t, i) for i in axes(X, 2) if imsk[i] for t in axes(X, 1) if !isfinite(X[t, i])]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Write the message a scenario fill raises.

The message names the assets, the count of filled pairs and the first observation, so a caller can find the listing that made them. It then names the **worst investable column**, the share of its own observations the fill invented, and the limit that share was measured against — that share is the one the fill tests, because a matrix-wide denominator scales with the universe and hides the column the notice exists to catch. The matrix-wide share is reported beside it as context, and it trips nothing.

It states the consequence at each of the four consumers that a census of the readers of `pr.X` found, because a caller told only about the tail will not look for the other three, and it states the tail's cost as the identity rather than the adjective: the invented zeros do not enter the tail, they inflate the denominator, so a measure at level `alpha` over an admitted column of coverage `c` reads that column's observed `alpha / c` level.

The limit it names is the estimator's own, after [`resolve_fill_limit`](@ref). A `fill_limit` of `nothing` demands every observation of every investable asset, so the message says so rather than printing a share no caller chose.

# Arguments

  - `filled`: The `(observation, asset)` pairs [`scenario_fill_pairs`](@ref) found.
  - `wi`: The index of the worst investable column, the one with the most filled entries.
  - `worst`: The share of its own observations that column's fill invented.
  - `frac`: The share of the entries of the returns matrix every filled pair is, as context.
  - `fill_limit`: The share of a column the fit allowed in silence, or `nothing` when it allowed none.

# Returns

  - `msg::String`: The message.

# Related

  - [`scenario_fill`](@ref)
  - [`scenario_fill_pairs`](@ref)
  - [`resolve_fill_limit`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`strict_diagnostic`](@ref)
"""
function scenario_fill_msg(filled::AbstractVector{<:Tuple{Integer, Integer}}, wi::Integer,
                           worst::Real, frac::Real, fill_limit::Option{<:Real})
    assets = unique(last.(filled))
    against = if isnothing(fill_limit)
        "against `fill_limit = nothing`, which asks that every investable asset cover every observation"
    else
        "against a limit of $(fill_limit)"
    end
    return "a mask-aware prior estimated an asset from the observations it had, and the returns matrix still carries the gap. Assets $(assets) carry a non-finite return at $(length(filled)) (observation, asset) pair(s) of an investable column, the first at observation $(first(filled)[1]). The worst column is asset $(wi), $(worst) of whose own observations are invented, measured $(against); the whole matrix is $(frac) invented, which is context and trips nothing. Those entries are filled with zero, and four consumers read them. A scenario-based measure (CVaR, EVaR, CDaR and their kin) understates its risk over the filled rows, and the cost is exact: the zeros do not enter the tail, they inflate the denominator, so a measure at level `alpha` over an admitted column of coverage `c` reads that column's observed `alpha / c` level, and at `c = 0.3` a 5% CVaR is a 16.7% CVaR. A hierarchical optimiser may branch the asset alone and then overweight it, because `clusterise` recomputes its correlation from the returns matrix rather than from `sigma`, an invented column reads as idiosyncratic, and an inverse-variance allocation over that branch buys more of it. An entropy pooling view on the asset is calibrated on the filled column, because the view resolvers read the matrix column by column. And a meta-optimiser carries the fill into the outer problem, whose returns matrix is built from the inner Sub-Portfolios' net returns over this one. `mu` and `sigma` are untouched, because the estimator computed them from the rows it saw. Accept a share of an investable column in silence with the estimator's own field, `EmpiricalPrior(; fill_limit = ...)`, state the share as a `CoveragePolicy`'s `min_coverage` so that admission and the notice are the one number, pass `strict = true` to refuse any fill, or fit over a window every asset covers."
end
"""
    resolve_fill_limit(fill_limit::Nothing, floor::Nothing)
    resolve_fill_limit(fill_limit::Nothing, floor::Real)
    resolve_fill_limit(fill_limit::Real, floor::Nothing)
    resolve_fill_limit(fill_limit::Real, floor::Real)

Resolve the share of an investable column a [`scenario_fill`](@ref) may invent in silence, at the fit.

`fill_limit` and a [`CoveragePolicy`](@ref)'s `min_coverage` are two spellings of one number. [`admits`](@ref) reads an asset's coverage share as its own observation count over the number of observations folded, and the fill counts that column's non-finite entries over the same denominator, so `filled_share == 1 - coverage_share` identically and the admission test **is** the fill test. A limit that did not know this would fire on nearly every fold of an available-case walk-forward, naming the caller for doing precisely what they configured.

So `nothing` **derives**. Where any arm of the fitting estimator states a floor, `nothing` means `1 - floor`, and it never fires: every admitted column satisfies it by construction. Where no arm states one — the exponentially weighted family is mask-aware without a policy, gating on `min_obs`, a *count* that says nothing about the share of a long window — `nothing` keeps its original meaning and names every fill, because a caller who set no floor has weighed no trade.

An explicit `fill_limit` overrides the derivation and must be **tighter** than admission. A value above `1 - floor` is refused, because it is dead by construction: nothing that reaches the fill could trip it, and a knob that cannot fire is worse than no knob. What an explicit value buys is the one configuration the derivation cannot express — admit broadly and be told anyway, `min_coverage = 0.3` with `fill_limit = 0.5` (ADR 0118).

The four methods are dispatch rather than a branch, so a call site holding two `Option`s finds a method for every arm of its union split.

# Arguments

  - `fill_limit`: The estimator's own `fill_limit` field, or `nothing` to derive one.
  - `floor`: The binding coverage floor of the estimator's arms, or `nothing` when no arm states one.

# Validation

  - `fill_limit <= 1 - floor` where both are stated, else a `DomainError` naming both is thrown.

# Returns

  - `fill_limit::Option{<:Real}`: The share of an investable column the fill may invent in silence, or `nothing` when it may invent none.

# Related

  - [`scenario_fill`](@ref)
  - [`coverage_floor`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`EmpiricalPrior`](@ref)
"""
function resolve_fill_limit(::Nothing, ::Nothing)
    return nothing
end
function resolve_fill_limit(::Nothing, floor::Real)
    return one(floor) - floor
end
function resolve_fill_limit(fill_limit::Real, ::Nothing)
    return fill_limit
end
function resolve_fill_limit(fill_limit::Real, floor::Real)
    @argcheck(fill_limit <= one(floor) - floor,
              DomainError(fill_limit,
                          "fill_limit must be tighter than the coverage floor it is measured against, so it must be at most 1 - min_coverage = $(one(floor) - floor). A looser limit is dead by construction: an asset the fit admits covers at least $(floor) of the observations folded, so its column is at most $(one(floor) - floor) invented, and nothing that reaches the fill could trip $(fill_limit). Pass `nothing`, the default, to derive $(one(floor) - floor) and be told only when a coverage algorithm admits a column thinner than the floor it was given."))
    return fill_limit
end
"""
    scenario_window(::Nothing, X::MatNum)
    scenario_window(max_scenarios::Integer, X::MatNum)

Cut the returns matrix a Prior Result carries down to the last `max_scenarios` observations.

The Scenario Cap of ADR 0136, applied. A prior that carries `X` carries it for the scenario risk measures, and a caller who wants a long window of moments and a short window of scenarios says so with one field rather than with two fits. The cap therefore touches `X` alone: `mu` and `sigma` are already computed when this runs, over every observation the fit read, and nothing here can or does move them.

It is deliberately the **same** verb in batch and online. A cap is a property of the result, not of the fold, so `prior(pe, X)` and the read-out of a folded `pe` cut the same rows off the same tail.

A window at or above the number of observations is the matrix itself, and no copy is taken: the cut is a `view`, so a cap that does nothing costs nothing.

# Arguments

  - `max_scenarios`: The number of observations to carry, or `nothing` to carry every one.
  - `X`: Asset returns, `observations × assets`.

# Returns

  - `X::MatNum`: The last `max_scenarios` rows of `X`, or `X` itself.

# Related

  - [`scenario_fill`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function scenario_window(::Nothing, X::MatNum)
    return X
end
function scenario_window(max_scenarios::Integer, X::MatNum)
    t = size(X, 1)
    return max_scenarios >= t ? X : view(X, (t - max_scenarios + 1):t, :)
end
"""
    scenario_ens(::Nothing, X::MatNum)
    scenario_ens(max_scenarios::Integer, X::MatNum)

State the number of observations the moments of a capped Prior Result were fitted over, or `nothing` when the cap cuts nothing.

The count a Scenario Cap owes its readers (ADR 0138). [`scenario_window`](@ref) cuts the rows the result carries and leaves `mu` and `sigma` fitted over every observation, so a consumer that prices a sample size off `size(pr.X, 1)` — an uncertainty set's `T`, a calibration rule's count — would read `w` rows for moments fitted over `t`, and mis-price every count by `t / w`. The result therefore states `t` in `ens` exactly when the cap cuts, and every count reader takes `ens` before the shape. When the cap does not cut, or there is no cap, the rows carried *are* the observations fitted over, and `ens` stays `nothing`, so a fit without a cap is bit-identical to what it was.

It is the same verb in batch and at the folded read-out, as [`scenario_window`](@ref) is, and the two agree by construction: both read the same `t` off the same matrix.

# Arguments

  - `max_scenarios`: The number of observations carried, or `nothing`.
  - `X`: Asset returns the moments were fitted over, `observations × assets`, **before** the cut.

# Returns

  - `ens::Option{<:Integer}`: `size(X, 1)` when `max_scenarios` cuts it, `nothing` otherwise.

# Related

  - [`scenario_window`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`effective_sample_size`](@ref)
"""
function scenario_ens(::Nothing, ::MatNum)
    return nothing
end
function scenario_ens(max_scenarios::Integer, X::MatNum)
    t = size(X, 1)
    return max_scenarios >= t ? nothing : t
end
"""
    scenario_fill_report(filled, ::Nothing)
    scenario_fill_report(filled, named::AbstractSet{<:Integer})

Narrow the filled pairs a [`scenario_fill`](@ref) reports to the assets it has not named before.

A batch fit calls [`scenario_fill`](@ref) once, so it reports everything it finds and keeps no memory: that is the `nothing` method. A walk-forward reading out at every step calls it once per step over a growing window, and every step would otherwise name the same assets again, so a caller who read the first notice learns to ignore the channel and misses the asset that lists at step 500.

So a carry state holds the set of assets it has already named, and this verb narrows the report to the pairs of assets outside it. The set is the state's memory, and [`scenario_fill_remember!`](@ref) is what writes to it — only for a notice that actually fired, so an asset whose share was under the limit at one step is still named at the step where it goes over.

# Arguments

  - `filled`: The `(observation, asset)` pairs [`scenario_fill_pairs`](@ref) found.
  - `named`: The assets already named, or `nothing` when nothing remembers.

# Returns

  - `report::AbstractVector{<:Tuple{Integer, Integer}}`: The pairs to name, which is `filled` itself when nothing remembers.

# Related

  - [`scenario_fill`](@ref)
  - [`scenario_fill_remember!`](@ref)
  - [`PriorCarryState`](@ref)
"""
function scenario_fill_report(filled::AbstractVector{<:Tuple{Integer, Integer}}, ::Nothing)
    return filled
end
function scenario_fill_report(filled::AbstractVector{<:Tuple{Integer, Integer}},
                              named::AbstractSet{<:Integer})
    return filter(p -> last(p) ∉ named, filled)
end
"""
    scenario_fill_remember!(::Nothing, report)
    scenario_fill_remember!(named::AbstractSet{<:Integer}, report)

Record the assets a [`scenario_fill`](@ref) has just named, so a later read-out does not name them again.

The write side of [`scenario_fill_report`](@ref). It runs after the notice, not before it, so an asset is remembered exactly when a caller was told about it: under `strict` the notice throws and nothing is remembered, and under a limit the notice did not trip nothing is remembered either.

The set is a field of a [`PriorCarryState`](@ref) and is written **in place**, because a read-out returns a Prior Result rather than the estimator, so there is no other channel by which the memory could survive the call.

# Arguments

  - `named`: The assets already named, or `nothing` when nothing remembers.
  - `report`: The `(observation, asset)` pairs just named.

# Returns

  - `nothing`.

# Related

  - [`scenario_fill`](@ref)
  - [`scenario_fill_report`](@ref)
  - [`PriorCarryState`](@ref)
"""
function scenario_fill_remember!(::Nothing, ::AbstractVector{<:Tuple{Integer, Integer}})
    return nothing
end
function scenario_fill_remember!(named::AbstractSet{<:Integer},
                                 report::AbstractVector{<:Tuple{Integer, Integer}})
    for p in report
        push!(named, last(p))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the returns matrix with the missing rows of every investable asset filled with zero, and say so above the fitting estimator's resolved `fill_limit`.

A mask-aware moment estimator answers a young asset from the observations it has, so the asset is investable and its returns column still carries a `NaN` at every row before it listed. Every consumer of a Prior Result reads that column — the JuMP model, the meta-optimisers and the value-level door among them — so the fill is paid **once**, here, on the estimator's own pass, rather than at each of them (ADR 0118).

A non-investable asset keeps its `NaN` column, so [`investable_mask`](@ref) is unchanged. `mu`, `sigma` and every other block are untouched, because the estimator computed them from the rows it saw.

**The share the fill tests is per asset**: the worst investable column's own count of filled entries over the number of observations. A matrix-wide denominator scales with the universe, so the column the notice exists to catch disappears inside it — one asset of a hundred whose column is seven-tenths invented is seven thousandths of the matrix, under any limit a caller would set. The matrix-wide share is reported in the message as context and trips nothing.

How much of the trade passes in silence is the fitting estimator's own answer, carried in its `fill_limit` field and resolved at the fit by [`resolve_fill_limit`](@ref) against the coverage floor of its arms — the share is a property of one fit, not of the session, so two priors in one program may answer differently. Under `strict = false` the trade is silent while the worst column's share stays at or below the resolved limit, and is named through [`strict_diagnostic`](@ref) above it. Under `strict = true` **any** fill refuses, whatever the limit holds.

Two mask-aware families reach this verb. The exponentially weighted family carries no [`CoveragePolicy`](@ref), so a `fill_limit` of `nothing` names every fill there. The plain family is mask-aware exactly where a policy is set, and a policy derives a limit that never fires, so an available-case fit names nothing while it does what it was configured to do. A plain estimator with no policy never reaches this verb at all: under the whole-window rule of ADR 0117 an asset it could not cover leaves the Coverage Universe and is not investable.

# Algorithm

 1. Return `X` itself when every entry of `X` is finite, which is every fit over a complete window.
 2. Derive the Investable Mask from `mu` and the diagonal of `sigma`, as [`investable_mask`](@ref) does, and find the pairs to fill with [`scenario_fill_pairs`](@ref). Return `X` itself when there are none, which is a gap that belongs to a non-investable asset alone.
 3. Narrow the pairs to report with [`scenario_fill_report`](@ref), which drops the assets a carry state has already named and is the identity when nothing remembers. Under `strict` the narrowing is skipped, because `strict` refuses any fill and must not depend on how often the estimator has been read out.
 4. Count the reported pairs per asset, take the worst column's share of its own observations, and report through [`strict_diagnostic`](@ref) when `strict` holds, when `fill_limit` is `nothing`, or when that share exceeds `fill_limit`. Record the assets just named with [`scenario_fill_remember!`](@ref).
 5. Return a copy of `X` with zero written at each of the filled pairs — every one of them, not just the reported ones — and the `NaN` of every non-investable asset left where it is.

# The notice fires once per asset, not once per step

`named` is what separates the batch call from the online one. A batch fit passes `nothing`, reports every fill it finds, and remembers nothing. A read-out of a folded prior passes the set its [`PriorCarryState`](@ref) carries, so a walk-forward names an asset at the step it lists and stays quiet afterwards instead of emitting the same notice at every one of two thousand steps. The **fill itself is unchanged**: every filled pair is written at every call, whatever the set holds. Under `strict = true` the first fill still throws.

# Arguments

  - `X`: Asset returns, `observations × assets`.
  - `mu`: The expected returns the estimator answered, `NaN` at a non-investable asset.
  - `sigma`: The covariance the estimator answered, `NaN` on the diagonal at a non-investable asset.
  - `strict`: If `true`, any fill raises an `ArgumentError`; if `false`, a fill above the share warns.
  - `fill_limit`: The share of an investable column the fitting estimator accepts in silence, after [`resolve_fill_limit`](@ref), or `nothing` when it accepts none.
  - `named`: The assets already named, written in place when a notice fires, or `nothing` when nothing remembers.

# Validation

  - The worst investable column's filled share is at or below `fill_limit`, else a warning naming the assets is emitted, or an `ArgumentError` naming them is raised under `strict`, which any fill raises. A `fill_limit` of `nothing` is at or below no share, so any fill is named.

# Returns

  - `X::MatNum`: The returns matrix whose investable columns are finite.

# Related

  - [`scenario_fill_pairs`](@ref)
  - [`scenario_fill_msg`](@ref)
  - [`scenario_fill_report`](@ref)
  - [`scenario_fill_remember!`](@ref)
  - [`scenario_window`](@ref)
  - [`resolve_fill_limit`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`investable_mask`](@ref)
  - [`strict_diagnostic`](@ref)
  - [`filter_held_gaps`](@ref)
"""
function scenario_fill(X::MatNum, mu::VecNum, sigma::MatNum, strict::Bool,
                       fill_limit::Option{<:Real},
                       named::Option{<:AbstractSet{<:Integer}} = nothing)
    if all(isfinite, X)
        return X
    end
    imsk = isfinite.(mu) .& isfinite.(LinearAlgebra.diag(sigma))
    filled = scenario_fill_pairs(X, imsk)
    if isempty(filled)
        return X
    end
    # `strict` refuses **any** fill, so it never reads the memory: an asset a previous
    # read-out named is still a fill, and silencing it here would make `strict` depend on
    # how many times the estimator had been read out.
    report = strict ? filled : scenario_fill_report(filled, named)
    if !isempty(report)
        counts = zeros(Int, size(X, 2))
        for (_, i) in report
            counts[i] += 1
        end
        wi = argmax(counts)
        worst = counts[wi] / size(X, 1)
        frac = length(filled) / length(X)
        if strict || isnothing(fill_limit) || worst > fill_limit
            strict_diagnostic(scenario_fill_msg(report, wi, worst, frac, fill_limit),
                              strict)
            scenario_fill_remember!(named, report)
        end
    end
    Xf = copy(X)
    for (t, i) in filled
        Xf[t, i] = zero(eltype(Xf))
    end
    return Xf
end
"""
    held_non_investable(imsk::BitVector, w::VecNum)
    held_non_investable(imsk::BitVector, w::VecVecNum)
    held_non_investable(imsk::BitVector, W::MatNum)

Find the assets a portfolio holds and the Investable Mask excludes.

A non-investable asset carries `NaN` in `mu` and on the diagonal of `sigma`, so no moment of it exists. A portfolio that holds none of it is reduced exactly. A portfolio that holds some of it is what a caller must be told about, and this is the one scan that finds those assets. A weight path holds an asset when any of its rows does, and a population holds one when any of its members does.

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

A weight vector is one cross-section, so the mask selects its entries. A weight path is one row of weights per observation, so the mask selects its columns and every row keeps its own observation. A population is reduced member by member.

Both `nothing` sentinels answer the argument they were handed. `imsk` of `nothing` is the all-investable universe, so there is nothing to select and the weights come back untouched. `w` of `nothing` is a caller who stated no weights at all — [`equilibrium_mu`](@ref) falls back to equal weights over whatever axis it is handed — so there is nothing to reduce, and reducing it would have to invent a length. Both are dispatch, so a caller holding neither pays nothing.

The `w` a **prior estimator** carries is per-asset configuration written against the caller's full universe, and it meets a reduced axis for the same reason a per-asset bound does. That is why this verb, first written for the weights an optimisation returns, is also the one a prior reduces its own weights with: it is one operation, and it is stated in one place.

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
       │     fpr ┴ nothing
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
