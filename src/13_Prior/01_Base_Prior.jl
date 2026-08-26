"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all prior estimators.

`AbstractPriorEstimator` is the base type for all estimators that compute prior information from asset and/or factor returns. All concrete prior estimators should subtype this type to ensure a consistent interface for prior computation and integration with portfolio optimisation workflows.

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractHighOrderPriorEstimator`](@ref)
  - [`prior`](@ref)
"""
abstract type AbstractPriorEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for low order prior estimators.

`AbstractLowOrderPriorEstimator` is the base type for estimators that compute low order moments (mean and covariance) from asset and/or factor returns. All concrete low order prior estimators should subtype this type for consistent moment estimation and integration.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
"""
abstract type AbstractLowOrderPriorEstimator <: AbstractPriorEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Low order prior estimator using only asset returns.

`AbstractLowOrderPriorEstimator_A` is the base type for estimators that compute low order moments (mean and covariance) using only asset returns data. All concrete asset-only prior estimators should subtype this type.

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
"""
abstract type AbstractLowOrderPriorEstimator_A <: AbstractLowOrderPriorEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Low order prior estimator using factor returns.

`AbstractLowOrderPriorEstimator_F` is the base type for estimators that compute low order moments (mean and covariance) requiring the use of both asset and factor returns data. All concrete factor-adjusted prior estimators should subtype this type.

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
"""
abstract type AbstractLowOrderPriorEstimator_F <: AbstractLowOrderPriorEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Low order prior estimator using both asset and factor returns.

`AbstractLowOrderPriorEstimator_AF` is the base type for estimators that compute low order moments (mean and covariance) using both asset and optionally factor returns data. All concrete prior estimators which may optionally use factor returns should subtype this type.

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
"""
abstract type AbstractLowOrderPriorEstimator_AF <: AbstractLowOrderPriorEstimator end
"""
    const AbstractLowOrderPriorEstimator_A_AF = Union{<:AbstractLowOrderPriorEstimator_A,
                                                      <:AbstractLowOrderPriorEstimator_AF}

Union type for asset-only and asset-and-factor low order prior estimators.

`AbstractLowOrderPriorEstimator_A_AF` is a union type that allows dispatch on both asset-only and asset-and-factor prior estimators. This is useful for generic algorithms that operate on estimators using asset returns, with or without factor returns.

# Related

  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_F_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_F_AF`](@ref)
"""
const AbstractLowOrderPriorEstimator_A_AF = Union{<:AbstractLowOrderPriorEstimator_A,
                                                  <:AbstractLowOrderPriorEstimator_AF}
"""
    const AbstractLowOrderPriorEstimator_F_AF = Union{<:AbstractLowOrderPriorEstimator_F,
                                                      <:AbstractLowOrderPriorEstimator_AF}

Union type for factor-only and asset-and-factor low order prior estimators.

`AbstractLowOrderPriorEstimator_F_AF` is a union type that allows dispatch on both factor-only and asset-and-factor prior estimators. This is useful for generic algorithms that operate on estimators using factor returns, with or without asset returns.

# Related

  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_F_AF`](@ref)
"""
const AbstractLowOrderPriorEstimator_F_AF = Union{<:AbstractLowOrderPriorEstimator_F,
                                                  <:AbstractLowOrderPriorEstimator_AF}
"""
    const AbstractLowOrderPriorEstimator_A_F_AF = Union{<:AbstractLowOrderPriorEstimator_A,
                                                        <:AbstractLowOrderPriorEstimator_F,
                                                        <:AbstractLowOrderPriorEstimator_AF}

Union type for asset-only, factor-only, and asset-and-factor low order prior estimators.

`AbstractLowOrderPriorEstimator_A_F_AF` is a union type that allows dispatch on asset-only, factor-only, and asset-and-factor prior estimators. This is useful for generic algorithms that operate on estimators using any combination of asset and factor returns.

# Related

  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_F_AF`](@ref)
"""
const AbstractLowOrderPriorEstimator_A_F_AF = Union{<:AbstractLowOrderPriorEstimator_A,
                                                    <:AbstractLowOrderPriorEstimator_F,
                                                    <:AbstractLowOrderPriorEstimator_AF}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for high order prior estimators.

`AbstractHighOrderPriorEstimator` is the base type for estimators that compute high order moments (such as coskewness and cokurtosis) from asset and/or factor returns. All concrete high order prior estimators should subtype this type to ensure a consistent interface for higher moment estimation and integration with portfolio optimisation workflows.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`prior`](@ref)
"""
abstract type AbstractHighOrderPriorEstimator <: AbstractPriorEstimator end
"""
$(DocStringExtensions.TYPEDEF)

High order prior estimator using factor returns.

`AbstractHighOrderPriorEstimator_F` is the base type for estimators that compute high order moments (such as coskewness and cokurtosis) requiring both asset and factor returns data. All concrete factor-based high order prior estimators should subtype this type.

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

Alias for a union of low-order and high-order factor prior estimator types.

# Related

  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractHighOrderPriorEstimator_F`](@ref)
"""
const AbstractHiLoOrderPriorEstimator_F = Union{<:AbstractLowOrderPriorEstimator_F,
                                                <:AbstractHighOrderPriorEstimator_F}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all prior result types.

`AbstractPriorResult` is the base type for all result objects produced by prior estimators, containing computed prior information such as moments, asset returns, and factor returns. All concrete prior result types should subtype this to ensure a consistent interface for integration with portfolio optimisation workflows.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`prior`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractPriorResult <: AbstractResult end
"""
    const PrE_Pr = Union{<:AbstractPriorEstimator, <:AbstractPriorResult}

Alias for a union of prior estimator and prior result types.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`AbstractPriorResult`](@ref)
"""
const PrE_Pr = Union{<:AbstractPriorEstimator, <:AbstractPriorResult}
"""
    const Pr_RR = Union{<:AbstractPriorResult, <:ReturnsResult}

Alias for a union of prior result and returns result types.

# Related

  - [`AbstractPriorResult`](@ref)
  - [`ReturnsResult`](@ref)
"""
const Pr_RR = Union{<:AbstractPriorResult, <:ReturnsResult}
"""
    prior(pr::AbstractPriorEstimator, rd::ReturnsResult; kwargs...)

Compute prior information from asset and/or factor returns using a prior estimator.

`prior` applies the specified prior estimator to a `ReturnsResult` object, extracting asset and factor returns and passing them, along with any additional information, to the estimator. Returns a prior result containing computed moments and other prior information for use in portfolio optimisation workflows.

# Arguments

  - `pr`: Prior estimator.
  - `rd`: Asset and/or factor returns result.
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Returns

  - `pr::AbstractPriorResult`: Result object containing computed prior information.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`ReturnsResult`](@ref)
  - [`AbstractPriorResult`](@ref)
"""
function prior(pr::AbstractPriorEstimator, rd::ReturnsResult; kwargs...)
    @argcheck(!isnothing(rd.X), IsNothingError)
    if isa(pr, AbstractHiLoOrderPriorEstimator_F)
        @argcheck(!isnothing(rd.F),
                  IsNothingError("this is a factor prior; it needs factor returns. ReturnsResult.F is nothing — populate F (e.g. via prices_to_returns on factor prices)."))
    end
    return prior(pr, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)
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

# Related

  - [`assert_prior_regression`](@ref)
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

  - `!isnothing(pr.rr)`.

# Related

  - [`AbstractLowOrderPriorEstimator_F_AF`](@ref)
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

# Related

  - [`forward_prior`](@ref)
  - [`reconstruct_prior`](@ref)
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

# Related

  - [`forward_prior`](@ref)
"""
function bound_field_is_stale(pr::AbstractPriorResult, sym::Symbol)::Bool
    return hasfield(typeof(pr), sym) && !isnothing(getfield(pr, sym))
end
"""
    prior(pr::AbstractPriorResult, args...; kwargs...)

Propagate or pass through prior result objects.

`prior` returns the input prior result object unchanged. This method is used to propagate already constructed prior results or enable uniform interface handling in workflows that accept either estimators or results.

# Arguments

  - `pr`: Prior result object.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `pr::AbstractPriorResult`: The input prior result object, unchanged.

# Related

  - [`AbstractPriorResult`](@ref)
  - [`prior`](@ref)
"""
function prior(pr::AbstractPriorResult, args...; kwargs...)::AbstractPriorResult
    return pr
end
"""
    port_opt_view(pr, args...; kwargs...)

Get a view or subset of a prior estimator or result for slicing.

Returns the prior unchanged for estimators (they are not sliceable), or returns a sliced prior result for a given cluster or asset index. Used in hierarchical optimisation to provide cluster-specific priors.

# Arguments

  - $(arg_dict[:per])
  - `args...`: Additional arguments (index, etc.).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - Sliced prior result or unchanged estimator.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`LowOrderPrior`](@ref)
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

# Arguments

  - `pr`: Prior result or returns result object.
  - `rd`: Optional returns result.
  - `x_src`: Source selector, `:prior` or `:data`.

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

# Arguments

  - `pr`: Prior result or returns result object.
  - `rd`: Optional returns result.
  - `z_src`: Source selector, `:prior` or `:data`.

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

# Arguments

  - `cle`: Clustering estimator.
  - `pr`: Prior result object.
  - `kwargs...`: Additional keyword arguments passed to the clustering estimator.

# Returns

  - `clr::AbstractClusteringResult`: Result object containing clustering information.

# Related

  - [`ClustersEstimator`](@ref)
  - [`AbstractPriorResult`](@ref)
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

# Arguments

  - `pl`: Network estimator, res estimator, or clustering result.
  - `pr`: Prior result object.
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Returns

  - `plr::PhylogenyResult`: Result object containing the phylogeny matrix.

# Related

  - [`NetworkEstimator`](@ref)
  - [`ClustersEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
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

# Arguments

  - `plc`: Phylogeny constraint estimator.
  - `pr`: Prior result or returns result object.
  - `rd`: Optional returns result (used when `x_src = :data` or `z_src = :data`).
  - `x_src`: If `:prior`, use asset returns from `pr`; if `:data`, use `rd`. Default is `:prior`.
  - `z_src`: Which carrier supplies the feature matrix a [`FeatureDistance`](@ref) reads: `:data` takes `rd.Z`, `:prior` takes `pr.Z`. Default is `:data`. Ignored when no [`FeatureDistance`](@ref) is in the estimator.
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Returns

  - Phylogeny constraint result.

# Related

  - [`AbstractPhylogenyConstraintEstimator`](@ref)
  - [`AbstractPriorResult`](@ref)
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

# Arguments

  - $(arg_dict[:cte])
  - `pr`: Prior result object.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `plr::PhylogenyResult`: Result object containing the centrality vector.

# Related

  - [`CentralityEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
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

# Arguments

  - `pl`: Network estimator, res estimator, or clustering result.
  - $(arg_dict[:cta])
  - `pr`: Prior result object.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `plr::PhylogenyResult`: Result object containing the centrality vector.

# Related

  - [`NetworkEstimator`](@ref)
  - [`CentralityEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
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

# Arguments

  - `pl`: Network estimator or phylogeny result.
  - $(arg_dict[:cta])
  - `w`: Portfolio weights vector.
  - `pr`: Prior result object.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `ac::Number`: Weighted average centrality.

# Related

  - [`NetworkEstimator`](@ref)
  - [`CentralityEstimator`](@ref)
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

# Arguments

  - $(arg_dict[:cte])
  - `w`: Portfolio weights vector.
  - `pr`: Prior result object.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `ac::Number`: Weighted average centrality.

# Related

  - [`CentralityEstimator`](@ref)
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

# Arguments

  - `pl`: Phylogeny estimator or clustering result used to compute the phylogeny matrix.
  - `w`: Portfolio weights vector.
  - `pr`: Prior result object containing asset returns.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the phylogeny matrix computation.

# Returns

  - `score::Number`: Asset phylogeny score.

# Details

  - Computes the phylogeny matrix from the asset returns in `pr` using `pl`.
  - Evaluates the weighted sum of the phylogeny matrix using the weights `w`.
  - Normalises the score by the sum of absolute weights.
  - Returns a real-valued score quantifying the phylogenetic structure of the allocation.

# Related

  - [`phylogeny_matrix`](@ref)
  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractClusteringResult`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`asset_phylogeny`](@ref)
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

# Arguments

  - `ccs`: Centrality constraint estimator or vector thereof.
  - `pr`: Prior result or returns result object.
  - `rd`: Optional returns result (used when `x_src = :data` or `z_src = :data`).
  - `x_src`: If `:prior`, use asset returns from `pr`; if `:data`, use `rd`. Default is `:prior`.
  - `z_src`: Which carrier supplies the feature matrix a [`FeatureDistance`](@ref) reads: `:data` takes `rd.Z`, `:prior` takes `pr.Z`. Default is `:data`. Ignored when no [`FeatureDistance`](@ref) is in the estimator.
  - `kwargs...`: Additional keyword arguments passed to the estimator. `strict` is read by the asset-returns variant, which reports a dropped zero centrality vector through it.

# Returns

  - Centrality constraint result.

# Related

  - [`AbstractPriorResult`](@ref)
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

# Related

  - [`HighOrderPrior`](@ref)
  - [`port_opt_view`](@ref)
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
