"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all portfolio optimisation estimators.

All optimisers and optimisation components should subtype `AbstractOptimisationEstimator` to participate in the optimisation dispatch system.

# Interfaces

`AbstractOptimisationEstimator` declares no method of its own. It carries the default [`port_opt_view`](@ref), which returns the estimator unchanged, and it splits into two halves. Subtype [`BaseOptimisationEstimator`](@ref) for a configuration an optimiser holds, and [`OptimisationEstimator`](@ref) for an estimator [`optimise`](@ref) runs.

# Related

  - [`BaseOptimisationEstimator`](@ref)
  - [`OptimisationEstimator`](@ref)
  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
"""
abstract type AbstractOptimisationEstimator <: AbstractEstimator end
"""
    const VecOptE = AbstractVector{<:AbstractOptimisationEstimator}

Alias for a vector of portfolio optimisation estimators.

Represents a collection of [`AbstractOptimisationEstimator`](@ref) objects, used for dispatch in routines that process multiple optimisers simultaneously.

# Related

  - [`AbstractOptimisationEstimator`](@ref)
"""
const VecOptE = AbstractVector{<:AbstractOptimisationEstimator}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for base portfolio optimisation estimators.

`BaseOptimisationEstimator` is the parent for all internal optimiser components that configure the optimisation problem but are not directly invokable as top-level optimisers.

# Interfaces

A subtype gains the time-dependent host methods from this supertype: [`is_time_dependent`](@ref), [`update_time_dependent_estimator`](@ref), [`reset_time_dependent_estimator`](@ref) and [`assert_time_dependent_fold_count`](@ref) all scan its fields generically, through [`time_dependent_fields`](@ref). One method is worth implementing:

## `time_dependent_field_defaults`

  - `time_dependent_field_defaults(opt::MyConfiguration) -> NamedTuple`: The static default of each field that may hold a [`TimeDependent`](@ref), for those whose default is not `nothing`. A *required* field is listed with [`NoDefault`](@ref), which declares that a schedule there must carry its own `default`.

### Arguments

  - `opt`: The concrete subtype instance.

### Returns

  - `defaults::NamedTuple`: The fold-less value of each listed field. The fallback method returns an empty tuple, which gives every scheduled field the fold-less value `nothing`.

# Related

  - [`AbstractOptimisationEstimator`](@ref)
  - [`OptimisationEstimator`](@ref)
"""
abstract type BaseOptimisationEstimator <: AbstractOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for portfolio optimisation estimators that produce portfolio weights.

Subtype `OptimisationEstimator` to implement concrete portfolio optimisers. All optimisers that can be invoked with `optimise` should subtype this.

# Interfaces

In order to implement a new optimiser that works seamlessly with the library, subtype `OptimisationEstimator`, give it an `fb` field, and implement the following method:

## `_optimise`

  - `_optimise(opt::MyOptimiser, rd::ReturnsResult, args...; kwargs...) -> OptimisationResult`: Solves the problem `opt` states over the data in `rd`, and returns the optimiser's own result type.

### Arguments

  - `opt`: The concrete subtype instance.
  - `rd`: Returns data.
  - `args...`, `kwargs...`: Forwarded from [`optimise`](@ref).

### Returns

  - `res::OptimisationResult`: The result, whose `retcode` decides whether [`optimise`](@ref) walks on to the fallback.

## The `fb` field

[`optimise`](@ref) reads `opt.fb` to walk the fallback chain, so every subtype carries one. It holds the next optimiser to try, or `nothing` to end the chain.

# Related

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`AbstractOptimisationEstimator`](@ref)
"""
abstract type OptimisationEstimator <: AbstractOptimisationEstimator end
function reset_time_dependent_estimator(opt::OptimisationEstimator)
    return opt
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for portfolio optimisation estimators that produce continuous (non-integer) portfolio weights.

# Interfaces

`NonFiniteAllocationOptimisationEstimator` adds no method to [`OptimisationEstimator`](@ref). It marks the optimisers whose weights are continuous, which is what admits them to the cross-validation and meta-optimisation entry points (see [`OptE_Opt`](@ref)).

# Related

  - [`OptimisationEstimator`](@ref)
  - [`NaiveOptimisationEstimator`](@ref)
  - [`ClusteringOptimisationEstimator`](@ref)
"""
abstract type NonFiniteAllocationOptimisationEstimator <: OptimisationEstimator end
"""
    pipe_route(x, ::Val{target}, v)

Absorb a [Routing Target](@ref PIPELINE_ROUTING_TARGETS)'s value into an optimiser, returning the rebuilt optimiser.

This is the optimiser-owned half of the [`Pipeline`](@ref) seam: [`inject_context`](@ref) fans a [`PipelineContext`](@ref) slot out into routing targets and delivers each one here, knowing nothing about where it lands.

Targets are named after the field they land in — `:pe`, `:cle`, `:wb`, `:lcse`, `:ple` — because those names are this package's shared vocabulary (see `field_dict`) rather than any one optimiser's private layout. The default method therefore *is* the routing rule: a target lands in the like-named field of any optimiser that has one. Nothing is declared per type, so nothing can drift.

Two targets are exceptions, because they carry validation policy and name no plain field: `:mu_ucs` (requires an [`ArithmeticReturn`](@ref), lands in `ret.ucs`) and `:sigma_ucs` (lands in the [`UncertaintySetVariance`](@ref) measures of `r`, see [`@pipe_route_sigma_ucs`](@ref)).

Optimisers holding their configuration in a field rather than carrying the target fields themselves declare [`@pipe_delegates`](@ref).

The lookup is `hasfield` rather than `hasproperty`, because routing rebuilds the object through the field: a name reachable only as a forwarded property could be read but not set.

A target with no home falls through to [`unroutable_target`](@ref), which ignores the [optional](@ref PIPELINE_OPTIONAL_TARGETS) ones and throws for the rest.

Internal machinery — not part of the user-facing API.

# Arguments

  - `x`: The optimiser or optimiser configuration.
  - `::Val{target}`: One of [`PIPELINE_ROUTING_TARGETS`](@ref).
  - `v`: The computed result to absorb.

# Returns

  - `x′`: The rebuilt optimiser.

# Related

  - [`pipe_accepts`](@ref)
  - [`@pipe_delegates`](@ref)
  - [`inject_context`](@ref)
"""
function pipe_route(x, ::Val{target}, v) where {target}
    return if hasfield(typeof(x), target)
        Accessors.set(x, Accessors.PropertyLens{target}(), v)
    else
        unroutable_target(x, Val(target), v)
    end
end
"""
    pipe_accepts(x, ::Val{target}) -> Bool

Whether a [Routing Target](@ref PIPELINE_ROUTING_TARGETS) has a home in `x`.

For the field-named targets this is exactly `hasfield` — the same fact [`pipe_route`](@ref) dispatches on, so acceptance and routing cannot disagree. The `:mu_ucs`/`:sigma_ucs` exceptions and the [`@pipe_delegates`](@ref) forwarders override it.

Internal machinery — not part of the user-facing API.

# Related

  - [`pipe_route`](@ref)
  - [`unroutable_target`](@ref)
"""
function pipe_accepts(x, ::Val{target})::Bool where {target}
    return hasfield(typeof(x), target)
end
"""
    unroutable_target(x, ::Val{target}, v)

Handle a [Routing Target](@ref PIPELINE_ROUTING_TARGETS) that `x` has no home for.

Declared here so [`pipe_route`](@ref) can call it; defined beside [`PIPELINE_OPTIONAL_TARGETS`](@ref), which is where the ignore-versus-throw policy belongs.

Internal machinery — not part of the user-facing API.

# Related

  - [`pipe_route`](@ref)
  - [`PIPELINE_OPTIONAL_TARGETS`](@ref)
"""
function unroutable_target end
"""
    pipe_config_field(x) -> Union{Nothing, Symbol}

The field holding `x`'s optimiser configuration, or `nothing` if it has none.

Declared by [`@pipe_delegates`](@ref) rather than probed, so an estimator whose `opt` field holds an inner *estimator* — [`SubsetResampling`](@ref) — is never mistaken for one holding a [`JuMPOptimiser`](@ref) configuration.

Internal machinery — not part of the user-facing API.

# Related

  - [`@pipe_delegates`](@ref)
  - [`pipe_route`](@ref)
"""
pipe_config_field(::Any) = nothing
"""
    @pipe_delegates T field

Declare that optimiser type `T` forwards every [Routing Target](@ref PIPELINE_ROUTING_TARGETS) to the configuration held in `field`.

Emits [`pipe_config_field`](@ref) plus forwarding [`pipe_route`](@ref) and [`pipe_accepts`](@ref) methods. Targets the configuration has no home for reach its own [`unroutable_target`](@ref), so the resulting error names the configuration — matching the pre-inversion messages.

A type that absorbs a target *itself* rather than through its configuration declares that target on the concrete type (see [`@pipe_route_sigma_ucs`](@ref)), which out-specialises this forwarder.

# Examples

```julia
@pipe_delegates MeanRisk opt
```

# Related

  - [`pipe_route`](@ref)
  - [`pipe_config_field`](@ref)
"""
macro pipe_delegates(T, field)
    f = QuoteNode(isa(field, QuoteNode) ? field.value : field)
    #! The block is escaped whole: hygiene would otherwise rename the three generics being
    #! extended into gensyms, silently defining the methods on throwaway functions.
    return esc(quote
                   pipe_config_field(::$T) = $f
                   function pipe_route(x::$T, t::Val, v)
                       return Accessors.set(x, Accessors.PropertyLens{$f}(),
                                            pipe_route(getfield(x, $f), t, v))
                   end
                   function pipe_accepts(x::$T, t::Val)::Bool
                       return pipe_accepts(getfield(x, $f), t)
                   end
               end)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Route a covariance uncertainty set into the [`UncertaintySetVariance`](@ref) risk measure(s) held in an optimiser's `r` field.

Each [`UncertaintySetVariance`](@ref) found — directly or inside a vector — has its `ucs` replaced with `sig`. An `r` field carrying no such measure is an error rather than a silent no-op: a computed uncertainty set that reaches no risk measure would be dropped.

Internal machinery — not part of the user-facing API.

# Arguments

  - `x`: The optimiser, which must carry an `r` field.
  - `sig`: The covariance uncertainty set result.

# Returns

  - `x′`: The rebuilt optimiser.

# Related

  - [`@pipe_route_sigma_ucs`](@ref)
  - [`pipe_route`](@ref)
"""
function route_sigma_ucs(x, sig::AbstractUncertaintySetResult)
    replace_usv(y) =
        if isa(y, UncertaintySetVariance)
            Accessors.set(y, Accessors.PropertyLens{:ucs}(), sig)
        else
            y
        end
    r = x.r
    newr = isa(r, AbstractVector) ? identity.([replace_usv(y) for y in r]) : replace_usv(r)
    found = if isa(newr, AbstractVector)
        any(y -> isa(y, UncertaintySetVariance), newr)
    else
        isa(newr, UncertaintySetVariance)
    end
    @argcheck(found,
              ArgumentError("cannot route a covariance uncertainty set into a $(Base.typename(typeof(x)).wrapper): no UncertaintySetVariance risk measure in its r field"))
    return Accessors.set(x, Accessors.PropertyLens{:r}(), newr)
end
"""
    @pipe_route_sigma_ucs T

Declare that optimiser type `T` absorbs the `:sigma_ucs` [Routing Target](@ref PIPELINE_ROUTING_TARGETS) into its own `r` field via [`route_sigma_ucs`](@ref).

Declared per concrete type rather than on a supertype: the covariance uncertainty set lands in the *estimator's* risk measures while every other target is forwarded to its configuration, so this method must out-specialise the [`@pipe_delegates`](@ref) forwarder on the same type. It is opt-in because carrying a configuration does not imply carrying risk measures — [`RelaxedRiskBudgeting`](@ref) has no `r` field.

# Related

  - [`route_sigma_ucs`](@ref)
  - [`@pipe_delegates`](@ref)
"""
macro pipe_route_sigma_ucs(T)
    #! Escaped whole, for the same reason as `@pipe_delegates`.
    return esc(quote
                   function pipe_route(x::$T, ::Val{:sigma_ucs}, v)
                       return route_sigma_ucs(x, v)
                   end
                   pipe_accepts(::$T, ::Val{:sigma_ucs})::Bool = true
               end)
end
"""
    @pipe_route_rkb T

Declare that optimiser type `T` absorbs the `:rkb` [Routing Target](@ref PIPELINE_ROUTING_TARGETS) into the `rkb` field of its risk-budgeting algorithm.

`:rkb` is the one target named after a field an optimiser does not carry directly. A risk budget belongs to the *algorithm* — `AssetRiskBudgeting` budgets assets, `FactorRiskBudgeting` budgets factors — so it lands one level down, at `rba.rkb`, and the derived `hasfield` rule cannot reach it.

Acceptance is not a constant: it asks the algorithm the optimiser is actually carrying. A [`TimeDependent`](@ref) schedule in `rba` has no `rkb` to write, so such an optimiser declines the target and a pipeline computing a budget for it is refused at construction rather than failing in the fold loop.

Declared per concrete type, for the same reason as [`@pipe_route_sigma_ucs`](@ref): it must out-specialise the [`@pipe_delegates`](@ref) forwarder on the same type.

# Related

  - [`@pipe_delegates`](@ref)
  - [`pipe_route`](@ref)
"""
macro pipe_route_rkb(T)
    #! Escaped whole, for the same reason as `@pipe_delegates`.
    return esc(quote
                   function pipe_route(x::$T, ::Val{:rkb}, v)
                       @argcheck(hasfield(typeof(x.rba), :rkb),
                                 ArgumentError("cannot route a risk budget into a $(Base.typename(typeof(x)).wrapper): its rba field holds a $(Base.typename(typeof(x.rba)).wrapper), which has no rkb field to receive it"))
                       rba = Accessors.set(x.rba, Accessors.PropertyLens{:rkb}(), v)
                       return Accessors.set(x, Accessors.PropertyLens{:rba}(), rba)
                   end
                   function pipe_accepts(x::$T, ::Val{:rkb})::Bool
                       return hasfield(typeof(x.rba), :rkb)
                   end
               end)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for optimisation algorithms used by portfolio optimisers.

# Interfaces

A subtype is a tag that an optimiser dispatches on, so it declares no method of its own. To add a behaviour, subtype `OptimisationAlgorithm` and add the methods of the consuming optimiser that are specialised on it.

# Related

  - [`AbstractAlgorithm`](@ref)
"""
abstract type OptimisationAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for portfolio optimisation result types.

All concrete optimisation result types should subtype `OptimisationResult`.

# Interfaces

A subtype declares no method, but [`optimise`](@ref) and [`factory`](@ref) read three properties of it. A subtype exposes them either as its own fields or by forwarding from an embedded core, as the JuMP and hierarchical leaves do:

  - `w`: The portfolio weights.
  - `retcode`: An [`OptimisationReturnCode`](@ref). [`optimise`](@ref) walks the fallback chain until it reads an [`OptimisationSuccess`](@ref).
  - `fb`: The record of the fallbacks that ran. It must be the **last field** of the struct, because [`factory`](@ref) rebuilds the result by replacing its trailing field.

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`OptimisationReturnCode`](@ref)
"""
abstract type OptimisationResult <: AbstractResult end
function reset_time_dependent_estimator(opt::OptimisationResult)
    return opt
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for continuous (non-integer allocation) optimisation results.

# Interfaces

The family adds no method to [`OptimisationResult`](@ref), but it is the bound of the generic [`factory`](@ref)`(res, fb)` that rebuilds a result with a new fallback record, which is why the trailing `fb` field is required here rather than one level up.

# Related

  - [`OptimisationResult`](@ref)
  - [`NaiveOptimisationResult`](@ref)
  - [`HierarchicalOptimisationResult`](@ref)
  - [`MeanRiskResult`](@ref)
"""
abstract type NonFiniteAllocationOptimisationResult <: OptimisationResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for non-JuMP continuous optimisation results.

Groups the results that do not carry a JuMP model (naive, clustering, and meta-optimiser results). Mirrors the JuMP/non-JuMP split on the result side. The JuMP side is itself two halves, [`RiskJuMPOptimisationResult`](@ref) for the results that carry a risk measure and [`NonRiskJuMPOptimisationResult`](@ref) for those that carry none.

The hierarchical members are grouped one level further down, under [`HierarchicalOptimisationResult`](@ref).

# Interfaces

`NonJuMPOptimisationResult` adds no method to [`NonFiniteAllocationOptimisationResult`](@ref). It is a classification, and it is what lets a method state "carries no JuMP model" in a signature.

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`RiskJuMPOptimisationResult`](@ref)
  - [`NonRiskJuMPOptimisationResult`](@ref)
  - [`NaiveOptimisationResult`](@ref)
  - [`HierarchicalOptimisationResult`](@ref)
"""
abstract type NonJuMPOptimisationResult <: NonFiniteAllocationOptimisationResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the **core** field block shared by hierarchical optimisation results.

Sits off the optimisation-result tree on purpose, exactly as [`BaseJuMPOptimisationResult`](@ref) does on the JuMP side. A core is not a thing `optimise` returns, so it must not satisfy methods bounded on the result family — [`factory`](@ref)`(res::NonFiniteAllocationOptimisationResult, fb)` included.

Its one subtype is [`HierarchicalResult`](@ref), embedded as `hr` by each leaf.

# Interfaces

A subtype is a field block, not a result. It declares no method, and it must **not** be given one that is bounded on the optimisation-result family, because the core is never what [`optimise`](@ref) returns.

# Related

  - [`HierarchicalResult`](@ref)
  - [`HierarchicalOptimisationResult`](@ref)
  - [`BaseJuMPOptimisationResult`](@ref)
"""
abstract type BaseHierarchicalOptimisationResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the results of the estimators that embed a hierarchical optimiser.

The optimiser they embed is a [`HierarchicalOptimiser`](@ref), held in their estimator's `opt` field.

The membership rule is exact: [`HierarchicalRiskParity`](@ref), [`HierarchicalEqualRiskContribution`](@ref) and [`SchurComplementHierarchicalRiskParity`](@ref) each hold an `opt::HierarchicalOptimiser`. [`NestedClustered`](@ref) does not, and its result is **not** in this family.

The family is deliberately **not** called `ClusteringOptimisationResult`: [`ClusteringOptimisationEstimator`](@ref) has four subtypes and the fourth is [`NestedClustered`](@ref), so that name would claim a set this type does not hold.

Two of the three members embed [`HierarchicalResult`](@ref) as `hr`; [`SchurComplementHierarchicalRiskParityResult`](@ref) keeps a flat field block, which is why the property forwarding lives on the leaves rather than here.

# Interfaces

The family adds no method to [`NonJuMPOptimisationResult`](@ref). Because the field block is not shared, a leaf that embeds [`HierarchicalResult`](@ref) declares its own property forwarding, so that the `w` and `retcode` properties [`OptimisationResult`](@ref) requires resolve through `hr`.

# Related

  - [`BaseHierarchicalOptimisationResult`](@ref)
  - [`HierarchicalRiskParityResult`](@ref)
  - [`HierarchicalEqualRiskContributionResult`](@ref)
  - [`SchurComplementHierarchicalRiskParityResult`](@ref)
"""
abstract type HierarchicalOptimisationResult <: NonJuMPOptimisationResult end
"""
    const VecOpt = AbstractVector{<:NonFiniteAllocationOptimisationResult}

Alias for a vector of non-finite allocation optimisation results.

Represents a collection of [`NonFiniteAllocationOptimisationResult`](@ref) objects.

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`OptE_Opt`](@ref)
"""
const VecOpt = AbstractVector{<:NonFiniteAllocationOptimisationResult}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for optimisation return codes.

Concrete subtypes indicate whether an optimisation succeeded or failed.

# Interfaces

A subtype declares no method. It carries one field, `res`, which holds the diagnostic text of a failure or `nothing`. [`optimise`](@ref) tests the code by type alone: only an [`OptimisationSuccess`](@ref) ends the fallback chain, so any other subtype is read as a failure.

# Related

  - [`OptimisationSuccess`](@ref)
  - [`OptimisationFailure`](@ref)
"""
abstract type OptimisationReturnCode <: AbstractResult end
"""
    const VecOptRetCode = AbstractVector{<:OptimisationReturnCode}

Alias for a vector of optimisation return codes.

# Related

  - [`OptimisationReturnCode`](@ref)
"""
const VecOptRetCode = AbstractVector{<:OptimisationReturnCode}
"""
    const OptRetCode_VecOptRetCode = Union{<:OptimisationReturnCode, <:VecOptRetCode}

Alias for either a single optimisation return code or a vector of return codes.

# Related

  - [`OptimisationReturnCode`](@ref)
  - [`VecOptRetCode`](@ref)
"""
const OptRetCode_VecOptRetCode = Union{<:OptimisationReturnCode, <:VecOptRetCode}

"""
    set_retcode(res::NonFiniteAllocationOptimisationResult, retcode::OptRetCode_VecOptRetCode)

Rebuild an optimisation result with a different return code, and every other member unchanged.

A cross-validation fold that drifts a population's weights drops the members whose wealth is not positive, and it drops them by failing their entry of the result's return code. A result is an immutable record, so the drop rebuilds it. The rebuild is a per-type method that writes the constructor name once, rather than a reflection pass over the field list.

Only a result that can carry a population of weight vectors needs a method here, because only such a result can hold one return code per member. A result that reaches the fallback raises, and the message names the type that is missing its method.

# Arguments

  - `res`: Optimisation result to rebuild.
  - `retcode`: Return code, or one per member of the population.

# Validation

  - The type of `res` declares a method of its own, else an `ArgumentError` is raised.

# Returns

  - `NonFiniteAllocationOptimisationResult`: The result, with the new return code.

# Related

  - [`mark_ruined_members`](@ref)
  - [`OptRetCode_VecOptRetCode`](@ref)
  - [`OptimisationFailure`](@ref)
"""
function set_retcode(res::NonFiniteAllocationOptimisationResult, ::OptRetCode_VecOptRetCode)
    return throw(ArgumentError("`set_retcode` has no method for `$(Base.typename(typeof(res)).wrapper)`, so a ruined population member of it cannot be dropped. A result that carries one return code per member needs a method of `set_retcode` that rebuilds it."))
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for intermediate optimisation model results.

Sits off the optimisation-result tree, like [`BaseHierarchicalOptimisationResult`](@ref) does: an intermediate record is not a thing [`optimise`](@ref) returns. Its one subtype is [`JuMPOptimisationSolution`](@ref), the record of what a solver returned.

# Interfaces

A subtype is a record of one solver attempt, and it declares no method. It is held by a result rather than returned by an optimiser.

# Related

  - [`OptimisationResult`](@ref)
"""
abstract type OptimisationModelResult <: AbstractResult end
"""
    const OptE_Opt = Union{<:NonFiniteAllocationOptimisationEstimator,
                           <:NonFiniteAllocationOptimisationResult}

Alias for a non-finite allocation optimisation estimator or result.

Matches either a [`NonFiniteAllocationOptimisationEstimator`](@ref) (specifying an optimiser configuration) or a [`NonFiniteAllocationOptimisationResult`](@ref) (a pre-computed result). Used for dispatch in cross-validation and optimisation workflows that accept either form.

# Related

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`VecOptE_Opt`](@ref)
"""
const OptE_Opt = Union{<:NonFiniteAllocationOptimisationEstimator,
                       <:NonFiniteAllocationOptimisationResult}
"""
    factory(res::NonFiniteAllocationOptimisationResult, fb::Option{<:OptE_Opt})

Rebuild a continuous optimisation result with an updated fallback optimiser `fb`.

Every optimisation result carries `fb` as its last field, so the generic rebuild copies all fields unchanged except the trailing `fb`. Concrete result types may override this method when rebuilding requires more than swapping `fb`.

# Related

  - [`OptE_Opt`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
function factory(res::NonFiniteAllocationOptimisationResult, fb::Option{<:OptE_Opt})
    flds = ntuple(i -> getfield(res, i), Val(fieldcount(typeof(res))))
    return (typeof(res).name.wrapper)(Base.front(flds)..., fb)
end
"""
    assert_special_nco_requirements(opt)

Assert that the optimiser meets special requirements for Nested Clustered Optimisation (NCO).

The default implementation does nothing. Overridden for estimators (e.g. [`Stacking`](@ref)) that have requirements which must be validated before NCO can proceed.

# Arguments

  - `opt`: Optimisation estimator, result, or vector thereof.

# Returns

  - `nothing`.

# Related

  - [`NestedClustered`](@ref)
  - [`Stacking`](@ref)
"""
function assert_special_nco_requirements(::OptE_Opt)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `opt` unchanged.

Default pass-through factory for optimisation estimators and results. Overridden for estimators that carry parameters requiring update at each optimisation step.

# Related

  - [`OptE_Opt`](@ref)
  - [`factory`](@ref)
"""
function factory(opt::OptE_Opt, ::Any)
    return opt
end
"""
    needs_previous_weights(opt)

Return `true` if the optimiser requires the previous period's weights.

The default returns `false`. Overridden for optimisers that contain turnover constraints, tracking error constraints, or other time-dependent components that require the previous optimisation's weights.

# Arguments

  - `opt`: Optimisation estimator, result, risk measure, fee structure, or vector thereof.

# Returns

  - `Bool`: `true` if previous weights are needed.

# Related

  - [`is_time_dependent`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function needs_previous_weights(::OptE_Opt)
    return false
end
#! Start: Overload these for all estimators which can use time-dependent constraints.
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the callable structs used as time-dependent values.

A subtype is a data-carrying alternative to a bare function inside a [`TimeDependent`](@ref): it must implement a functor `(x::MySubtype)(ctx::TimeDependentContext)` returning the fold's field value. Because it is a struct, it participates in a trait a bare function cannot: define `needs_previous_weights(::MySubtype) = true` to declare a previous-weights requirement directly (the default is `false`), instead of wrapping in [`PreviousWeightsFunction`](@ref).

Being a struct also makes it the natural home for **recording what a callable schedule chose**: a bare `ctx -> …` selects nothing by index, so its per-fold decision is not otherwise recoverable (see the provenance note on [`TimeDependent`](@ref)). A functor can carry a mutable field — e.g. a vector it writes at `ctx.i` — and log the fold's resolved value as a side effect of computing it.

The family classifies by what the functor returns, and a subtype declares that kind in its type. Subtype [`TimeDependentConstraintCallable`](@ref) when the per-fold value is a constraint value, and [`TimeDependentOptimiserCallable`](@ref) when it is an optimiser. Only the second is statically admissible in an optimiser-valued field (see [`TD_OptE_Opt`](@ref)), so the classification is what that admissibility is read off. Do not subtype this root directly.

# Interfaces

Subtype one of the two children, not this root, and implement the following:

## The functor

  - `(x::MySubtype)(ctx::TimeDependentContext)`: Returns the field value for the fold that `ctx` describes.

### Arguments

  - `x`: The concrete subtype instance.
  - `ctx`: The fold's context, which carries the fold index, the fold loop's data and, when the loop runs sequentially, the previous fold's weights.

### Returns

  - The complete field value for that fold. Its kind is the one the subtype's supertype declares.

## `needs_previous_weights`

  - `needs_previous_weights(::MySubtype) -> Bool`: Declares whether the functor reads `ctx.w_prev`. The default is `false`. Define it as `true` to force sequential fold execution, which is what [`PreviousWeightsFunction`](@ref) does for a bare function.

# Related

  - [`TimeDependentConstraintCallable`](@ref)
  - [`TimeDependentOptimiserCallable`](@ref)
  - [`TimeDependent`](@ref)
  - [`TimeDependentContext`](@ref)
  - [`PreviousWeightsFunction`](@ref)
  - [`needs_previous_weights`](@ref)
"""
abstract type TimeDependentCallable <: AbstractEstimator end
function needs_previous_weights(::TimeDependentCallable)::Bool
    return false
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for callable structs whose per-fold value is a *constraint value*.

A subtype implements a functor `(x::MySubtype)(ctx::TimeDependentContext)` returning the fold's value for a constraint-position field — a budget, a set of weight bounds, a fee structure, a turnover limit, anything a [`TimeDependent`](@ref) may carry other than an optimiser. The value is checked when the fold loop swaps it into the field, by the host's own keyword constructor.

This is the kind to subtype for a functor whose output is *not* an optimiser. A functor returning an optimiser declares [`TimeDependentOptimiserCallable`](@ref) instead, which is what makes an optimiser-position schedule statically admissible (see [`TD_OptE_Opt`](@ref)).

# Interfaces

The methods are those of [`TimeDependentCallable`](@ref): the functor `(x::MySubtype)(ctx::TimeDependentContext)`, and the optional `needs_previous_weights`. This child adds no method. It states what the functor returns — a constraint value — and the host's own keyword constructor checks that value when the fold loop swaps it in.

# Related

  - [`TimeDependentCallable`](@ref)
  - [`TimeDependentOptimiserCallable`](@ref)
  - [`TimeDependent`](@ref)
  - [`TimeDependentContext`](@ref)
  - [`needs_previous_weights`](@ref)
"""
abstract type TimeDependentConstraintCallable <: TimeDependentCallable end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for callable structs whose per-fold value is an *optimiser*.

A subtype implements a functor `(x::MySubtype)(ctx::TimeDependentContext)` returning the fold's optimiser (an [`OptE_Opt`](@ref)), so a [`TimeDependent`](@ref) holding it is admissible wherever an optimiser-valued field accepts a schedule (see [`TD_OptE_Opt`](@ref)). Declaring the functor's output kind in the type is what makes the schedule *statically* admissible: a bare `ctx -> optimiser` is admitted as a `Base.Callable` and checked only when the fold loop swaps its value in.

# Interfaces

The methods are those of [`TimeDependentCallable`](@ref): the functor `(x::MySubtype)(ctx::TimeDependentContext)`, and the optional `needs_previous_weights`. This child adds no method. It states that the functor returns an [`OptE_Opt`](@ref), and [`assert_time_dependent_optimiser`](@ref) checks that promise when the fold loop swaps the value in.

# Related

  - [`TimeDependentCallable`](@ref)
  - [`TimeDependentConstraintCallable`](@ref)
  - [`TimeDependent`](@ref)
  - [`TD_OptE_Opt`](@ref)
  - [`TimeDependentContext`](@ref)
"""
abstract type TimeDependentOptimiserCallable <: TimeDependentCallable end
"""
$(DocStringExtensions.TYPEDEF)

Declares that a callable time-dependent entry requires the previous optimisation's weights.

A bare callable inside a [`TimeDependent`](@ref) cannot be inspected for previous-weight requirements, so it contributes `false` to [`needs_previous_weights`](@ref) and its context's `w_prev` is only populated when something else makes the fold loop sequential. Wrapping the callable in `PreviousWeightsFunction` declares the requirement as data: it contributes `true` to [`needs_previous_weights`](@ref), forcing sequential fold execution and a populated `w_prev` in the [`TimeDependentContext`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PreviousWeightsFunction(; f) -> PreviousWeightsFunction

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> PreviousWeightsFunction(; f = identity)
PreviousWeightsFunction
  f ┴ typeof(identity): identity
```

# Related

  - [`TimeDependent`](@ref)
  - [`TimeDependentContext`](@ref)
  - [`needs_previous_weights`](@ref)
"""
struct PreviousWeightsFunction{T} <: AbstractAlgorithm
    """
    Callable evaluated per fold as `f(ctx::TimeDependentContext)`, returning the fold's field value.
    """
    f::T
    function PreviousWeightsFunction(f)
        return new{typeof(f)}(f)
    end
end
function PreviousWeightsFunction(; f)::PreviousWeightsFunction
    return PreviousWeightsFunction(f)
end
function needs_previous_weights(::PreviousWeightsFunction)::Bool
    return true
end
"""
$(DocStringExtensions.TYPEDEF)

States that no fold-less value exists. It stands in the two places such a value may be missing.

  - As a [`TimeDependent`](@ref)'s `default`: the schedule states no fold-less value of its own, so a fold-less solve falls back to the field's static default (see [`time_dependent_field_defaults`](@ref)).
  - As an entry of a host's [`time_dependent_field_defaults`](@ref): the field is *required* and has no static default (the optimiser-valued fields), so a schedule there must carry its own `default`. A fold-less solve of a host whose required field holds a defaultless schedule throws a [`TimeDependentDefaultError`](@ref).

# Constructors

    NoDefault() -> NoDefault

# Examples

```jldoctest
julia> NoDefault()
NoDefault()
```

# Related

  - [`TimeDependent`](@ref)
  - [`TimeDependentDefaultError`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`reset_time_dependent_estimator`](@ref)
"""
struct NoDefault <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Exception thrown when a fold-less solve reaches a [`TimeDependent`](@ref) schedule that has no value to fall back to: the field has no static default and the schedule supplies no `default`.

A schedule is defined *only* over the folds of a cross-validation scheme. Fields with a static default reset to it silently; a required field (the optimiser-valued ones) has nothing to reset to, so the schedule must state the value a fold-less solve should use, via `TimeDependent(val; default = x)`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TimeDependentDefaultError(msg)

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`TimeDependent`](@ref)
  - [`NoDefault`](@ref)
  - [`reset_time_dependent_estimator`](@ref)
"""
@concrete struct TimeDependentDefaultError <: PortfolioOptimisersError
    """
    $(field_dict[:msg])
    """
    msg
end
"""
$(DocStringExtensions.TYPEDEF)

Varies one optimiser input across the folds of a cross-validation scheme.

A `TimeDependent` is stored *directly in the optimiser field it varies* — e.g. `JuMPOptimiser(; lt = TimeDependent([...]))` — so the field's position names the target and a field holds either a static value or a per-fold schedule, never both. It is recognised at top-level optimiser fields only, never nested inside another input (e.g. inside a [`Fees`](@ref) or a risk measure).

`val` is either a vector of per-fold values — entry `i` is the complete field value for fold `i` of the consuming scheme's `split` enumeration — or a callable evaluated per fold: a bare function `f(ctx::TimeDependentContext)` (optionally wrapped in [`PreviousWeightsFunction`](@ref)) or a [`TimeDependentCallable`](@ref) functor struct.

For a field that itself accepts a *vector of constraints* statically, a per-fold entry is that whole vector, so a schedule of per-fold constraint vectors is a vector of vectors — `TimeDependent([[c₁ᵃ, c₁ᵇ], [c₂ᵃ, c₂ᵇ], …])`, entry `i` being fold `i`'s complete constraint vector. There is no separate "vector of `TimeDependent`" facility and none is needed: `TimeDependent` is recognised only at a top-level field, so to vary individual constraints within a vector, build the fold's vector in a callable — `TimeDependent(ctx -> [dynamic(ctx), static])` — which keeps the shared static parts in one place.

The machinery imposes no ordering of its own: fold `i` is whatever `split(cv, rd)` enumerates `i`-th, which is chronological for walk-forward and (unshuffled) KFold schemes. For schemes whose enumeration is not a timeline (combinatorial splits, randomised paths) it is the user's responsibility to key entries off the fold's indices — a callable sees its own fold's windows via `ctx.train_idx[ctx.i]`/`ctx.test_idx[ctx.i]` and may derive any ordering from them.

A time-dependent constraint participates only where folds exist and is inert everywhere else — a fold-less `optimise` replaces it with the field's fold-less value (see [`reset_time_dependent_estimator`](@ref)). Vector entries must have length equal to the number of folds of the consuming cross-validation scheme, validated at `split` time. Entries may be `nothing`, giving the field `nothing` for that fold.

The fold-less value is the field's static default, unless `default` overrides it. A field with *no* static default — the required, optimiser-valued fields — has nothing to reset to, so a schedule there **must** supply `default`; a fold-less solve of one that does not throws a [`TimeDependentDefaultError`](@ref).

A vector whose entries are all optimisers or precomputed results ([`OptE_Opt`](@ref)) is stored as a `Vector{OptE_Opt}`, so a *mixed* schedule — fold `i` optimising or predicting depending on what entry `i` is — is admissible in an optimiser-valued field on its element type alone (see [`TD_OptE_Opt`](@ref)) rather than falling out to a `Vector{Any}` the field cannot accept.

Schedules do not nest: neither `val`, nor a vector entry of `val`, nor `default` may be a `TimeDependent`. Entry `i` is fold `i`'s *complete* field value, and the fold-less value is by definition outside every fold loop, so nesting has no meaning. An estimator swapped in by a schedule may itself carry schedules — those resolve against the same fold context after the swap — but they live in *its* fields, not inside this wrapper.

**Recovering which entry a fold ran** needs no stored provenance, because a vector schedule is keyed by the fold index and nothing else. Entry `i` runs at fold `i` of the consuming scheme's `split` enumeration ([`time_dependent_value`](@ref) indexes `val[ctx.i]`), so `val[i]` *is* fold `i`'s value — the same index you keyed the schedule by. Under the time-ordered schemes (walk-forward, unshuffled [`KFold`](@ref), [`Pipeline`](@ref)) fold `i` is also the `i`-th entry of the returned [`MultiPeriodPredictionResult`](@ref); under schemes that regroup for reporting ([`MultipleRandomised`](@ref) sorts by test index, combinatorial recombines each split's test groups into paths) the prediction order no longer tracks the fold order, so re-run `split(cv, rd)` and read the fold→path map off its `path_ids` — it is keyed by the very enumeration index the schedule was, so entry `k` still governs enumeration fold `k`. A **callable** schedule computes its value rather than selecting an entry, so there is no index to recover: what it returned is knowable only by re-running it on the fold's [`TimeDependentContext`](@ref), or by having it record its own choice. Recording is a logging concern the caller owns, and the [`TimeDependentCallable`](@ref) struct interface is its natural home — a functor can stash the regime it picked per fold in a field of its own.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TimeDependent(val, bind::Symbol = :outermost; default = NoDefault())
    TimeDependent(; val::Union{<:AbstractVector, <:Base.Callable,
                               <:PreviousWeightsFunction, <:TimeDependentCallable,
                               <:TimeDependent}, bind::Symbol = :outermost,
                  default = NoDefault())

## Validation

  - If `val` is a vector: `!isempty(val)`, and no entry is a `TimeDependent`.
  - `val` is not a `TimeDependent`.
  - `default` is not a `TimeDependent`.
  - `bind in (:outermost, :nearest)`.

# Examples

```jldoctest
julia> TimeDependent([Fees(; l = 0.001), Fees(; l = 0.002)])
TimeDependent
      val ┼ 2-element Vector{Fees}
          │ Fees ⋯
          │ Fees ⋯
     bind ┼ Symbol: :outermost
  default ┴ NoDefault()
```

# Related

  - [`TimeDependentContext`](@ref)
  - [`PreviousWeightsFunction`](@ref)
  - [`NoDefault`](@ref)
  - [`TimeDependentDefaultError`](@ref)
  - [`TD_OptE_Opt`](@ref)
  - [`is_time_dependent`](@ref)
  - [`update_time_dependent_estimator`](@ref)
  - [`reset_time_dependent_estimator`](@ref)
"""
struct TimeDependent{T1, T2} <: AbstractEstimator
    """
    Vector of per-fold values (in the consuming scheme's `split` enumeration order), or a callable of the fold's [`TimeDependentContext`](@ref): a bare function (optionally wrapped in [`PreviousWeightsFunction`](@ref)) or a [`TimeDependentCallable`](@ref) functor struct.
    """
    val::T1
    """
    Which fold loop consumes the schedule: `:outermost` (default) binds it to the outermost fold loop processing the estimator tree; `:nearest` binds it to the nearest enclosing fold loop — inside a meta-optimiser's inner estimators that is the meta's own cross-validation leg, which then consumes the schedule even when the meta is backtested under an outer fold loop.
    """
    bind::Symbol
    """
    Value the field takes outside every fold loop, overriding the host's static default (see [`time_dependent_field_defaults`](@ref)). [`NoDefault`](@ref) (the default) defers to the host's static default; a field that has none requires this to be set.
    """
    default::T2
    function TimeDependent(val::Union{<:AbstractVector, <:Base.Callable,
                                      <:PreviousWeightsFunction, <:TimeDependentCallable},
                           bind::Symbol = :outermost; default = NoDefault())
        if isa(val, AbstractVector)
            @argcheck(!isempty(val), IsEmptyError("val cannot be empty"))
            @argcheck(!any(x -> isa(x, TimeDependent), val),
                      ArgumentError("no entry of val may be a TimeDependent: entry i is fold i's complete field value, so schedules do not nest. To vary parts of a vector-valued field, assemble the fold's vector in a callable: TimeDependent(ctx -> [dynamic(ctx), static])."))
            if !(eltype(val) <: OptE_Opt) && all(x -> isa(x, OptE_Opt), val)
                val = convert(Vector{OptE_Opt}, val)
            end
        end
        @argcheck(!isa(default, TimeDependent),
                  ArgumentError("default cannot be a TimeDependent: it is the field's value outside every fold loop, where a schedule is undefined."))
        @argcheck(bind in (:outermost, :nearest),
                  ArgumentError("bind must be :outermost or :nearest, got :$bind"))
        return new{typeof(val), typeof(default)}(val, bind, default)
    end
end
function TimeDependent(::TimeDependent, args...; kwargs...)
    return throw(ArgumentError("val cannot be a TimeDependent: schedules do not nest. An estimator swapped in by a schedule may carry schedules of its own — they resolve against the same fold context after the swap — but they belong in its fields, not inside this wrapper."))
end
function TimeDependent(;
                       val::Union{<:AbstractVector, <:Base.Callable,
                                  <:PreviousWeightsFunction, <:TimeDependentCallable,
                                  <:TimeDependent}, bind::Symbol = :outermost,
                       default = NoDefault())::TimeDependent
    return TimeDependent(val, bind; default = default)
end
"""
    const TD_Option{X} = Union{Nothing, <:TimeDependent, X}

Alias for an optimiser field that accepts `nothing`, a static value of type `X`, or a per-fold [`TimeDependent`](@ref) schedule.

The set of fields whose constructor signatures use this alias is the single source of truth for which optimiser inputs may vary over folds.

# Related

  - [`TimeDependent`](@ref)
  - [`Option`](@ref)
"""
const TD_Option{X} = Union{Nothing, <:TimeDependent, X}
"""
    const TD{X} = Union{<:TimeDependent, X}

Alias for a *required* optimiser field that accepts a static value of type `X` or a per-fold [`TimeDependent`](@ref) schedule, but not `nothing`.

The problem-definition fields that always carry a value — the prior estimator, the returns model, the scalariser, the clustering estimator, the weight finaliser — are time-dependent through this alias rather than [`TD_Option`](@ref), so `nothing` stays inadmissible where it was never a legal static value. Such a field still has a *static default*, so a schedule in one resets to that default on a fold-less solve, unlike the optimiser-valued fields (see [`TD_OptE_Opt`](@ref)).

# Related

  - [`TimeDependent`](@ref)
  - [`TD_Option`](@ref)
  - [`time_dependent_field_defaults`](@ref)
"""
const TD{X} = Union{<:TimeDependent, X}
"""
    const TD_OptE_Opt = Union{TimeDependent{<:AbstractVector{<:OptE_Opt}},
                              TimeDependent{<:TimeDependentOptimiserCallable},
                              TimeDependent{<:PreviousWeightsFunction},
                              TimeDependent{<:Base.Callable}}

The [`TimeDependent`](@ref) forms admissible in an *optimiser-valued* field — where the scheduled thing is the optimiser itself, not one of its inputs.

Two of the four are statically checked: a vector schedule whose entries are all [`OptE_Opt`](@ref) (an optimiser or a precomputed result — a mixed schedule is allowed, fold `i` optimising or predicting depending on what entry `i` is), and a [`TimeDependentOptimiserCallable`](@ref), which declares its output kind in its type. The other two — a bare `ctx -> optimiser` and a [`PreviousWeightsFunction`](@ref) wrapping one — cannot be checked before they run, so their output is checked when the fold loop swaps it into the field, by the host's own keyword constructor.

Because an optimiser-valued field is *required*, a schedule in one has no static default to reset to on a fold-less solve and must supply `default` (see [`NoDefault`](@ref), [`TimeDependentDefaultError`](@ref)).

# Related

  - [`TimeDependent`](@ref)
  - [`TimeDependentOptimiserCallable`](@ref)
  - [`TDO_Option`](@ref)
  - [`TD_Option`](@ref)
  - [`OptE_Opt`](@ref)
"""
const TD_OptE_Opt = Union{TimeDependent{<:AbstractVector{<:OptE_Opt}},
                          TimeDependent{<:TimeDependentOptimiserCallable},
                          TimeDependent{<:PreviousWeightsFunction},
                          TimeDependent{<:Base.Callable}}
"""
    const TDO_OptE_Opt = Union{<:TD_OptE_Opt,
                               <:TimeDependent{<:AbstractVector{<:Option{<:OptE_Opt}}}}

The [`TimeDependent`](@ref) forms admissible in an *optional* optimiser-valued field (a fallback): every [`TD_OptE_Opt`](@ref) form, plus a vector schedule whose entries may be `nothing`.

`nothing` was always a legal static value of an optional field, and the [`TimeDependent`](@ref) contract says a vector entry may be `nothing`, giving the field `nothing` for that fold — so an optional optimiser field admits `TimeDependent([mr, nothing])`, a fallback switched off on some folds. A *required* optimiser position (the optimiser itself) never admits `nothing`, statically or per fold, so it stays on the strict [`TD_OptE_Opt`](@ref) bound.

# Related

  - [`TD_OptE_Opt`](@ref)
  - [`TDO_Option`](@ref)
  - [`Option`](@ref)
"""
const TDO_OptE_Opt = Union{<:TD_OptE_Opt,
                           <:TimeDependent{<:AbstractVector{<:Option{<:OptE_Opt}}}}
"""
    const TDO_Option{X} = Union{Nothing, <:TDO_OptE_Opt, X}

Alias for an *optional* optimiser-valued field (e.g. a fallback) that accepts `nothing`, a static value of type `X`, or a per-fold schedule of optimisers whose entries may be `nothing` (see [`TDO_OptE_Opt`](@ref)).

A required optimiser-valued field spells its union out — `Union{<:X, <:TD_OptE_Opt}` — since `nothing` is not one of its values.

# Related

  - [`TDO_OptE_Opt`](@ref)
  - [`TD_OptE_Opt`](@ref)
  - [`TD_Option`](@ref)
  - [`Option`](@ref)
"""
const TDO_Option{X} = Union{Nothing, <:TDO_OptE_Opt, X}
"""
    const OptE_TD = Union{<:NonFiniteAllocationOptimisationEstimator, <:TD_OptE_Opt}

Alias for an optimisation estimator, or a [`TimeDependent`](@ref) schedule standing in its place.

This is the entry-point type of the cross-validation fold loops that *fit*: a schedule handed straight to [`cross_val_predict`](@ref) is the optimiser, and fold `i` runs entry `i`. Precomputed results are excluded because a bare result takes the predict-only path, which has no fold loop to resolve a schedule against — but a schedule *whose entries* are results is admissible here, and each such entry takes the predict-only path per fold (see [`OptE_Opt_TD`](@ref)).

# Related

  - [`TD_OptE_Opt`](@ref)
  - [`OptE_Opt_TD`](@ref)
  - [`cross_val_predict`](@ref)
"""
const OptE_TD = Union{<:NonFiniteAllocationOptimisationEstimator, <:TD_OptE_Opt}
"""
    const OptE_Opt_TD = Union{<:OptE_Opt, <:TD_OptE_Opt}

Alias for an optimisation estimator or a precomputed result, or a [`TimeDependent`](@ref) schedule standing in their place.

The entry-point type of the fold loops that accept a precomputed result as well as an estimator. A schedule's entries are [`OptE_Opt`](@ref), so a *mixed* schedule is admissible: fold `i` optimises when entry `i` is an estimator and predicts when it is a result, which the single-fold [`fit_and_predict`](@ref) methods already distinguish by dispatch.

# Related

  - [`OptE_TD`](@ref)
  - [`TD_OptE_Opt`](@ref)
  - [`OptE_Opt`](@ref)
"""
const OptE_Opt_TD = Union{<:OptE_Opt, <:TD_OptE_Opt}
"""
    const VecOptE_Opt_TD = AbstractVector{<:OptE_Opt_TD}

Alias for a vector of optimisation estimators or results in which individual *elements* may be [`TimeDependent`](@ref) schedules.

This is the element-level admission of schedules, needed where a vector-valued field's elements are themselves optimiser positions consumed by a fold loop one at a time — `Stacking.opti`, whose inner cross-validation is entered per candidate. It is a superset of [`VecOptE_Opt`](@ref), so every method taking it continues to accept plain vectors.

# Related

  - [`OptE_Opt_TD`](@ref)
  - [`VecOptE_Opt`](@ref)
"""
const VecOptE_Opt_TD = AbstractVector{<:OptE_Opt_TD}
"""
    const TD_VecOptE_Opt = Union{TimeDependent{<:AbstractVector{<:VecOptE_Opt_TD}},
                                 TimeDependent{<:TimeDependentOptimiserCallable},
                                 TimeDependent{<:PreviousWeightsFunction},
                                 TimeDependent{<:Base.Callable}}

The [`TimeDependent`](@ref) forms admissible in a *vector-of-optimisers* field (`Stacking.opti`): a vector schedule whose entries are per-fold optimiser vectors, or a callable returning the fold's vector.

Entry `i` is fold `i`'s complete vector of candidates, so a field-level schedule varies the whole candidate set per fold; an entry's own elements may in turn be schedules (a [`VecOptE_Opt_TD`](@ref)), which the consuming host's inner fold loop resolves as usual. Only `bind = :outermost` is admissible at the field level — see the host's constructor for why.

# Related

  - [`TD_OptE_Opt`](@ref)
  - [`VecOptE_Opt_TD`](@ref)
"""
const TD_VecOptE_Opt = Union{TimeDependent{<:AbstractVector{<:VecOptE_Opt_TD}},
                             TimeDependent{<:TimeDependentOptimiserCallable},
                             TimeDependent{<:PreviousWeightsFunction},
                             TimeDependent{<:Base.Callable}}
"""
$(DocStringExtensions.TYPEDEF)

Describes one fold to the time-dependent constraints that resolve against it.

Carries the fold's position in the consuming scheme's `split` enumeration and the data needed for a callable entry to compute its value. `i` indexes `train_idx`/`test_idx`, so `ctx.train_idx[ctx.i]`/`ctx.test_idx[ctx.i]` are always the fold's own windows; no ordering beyond the scheme's enumeration is implied. `rd` is the fold loop's (possibly asset-viewed) input data, so callables see the current universe and timestamps: the returns-level data at the optimiser fold loops, or the raw, pre-preprocessing price- or returns-level input at the [`Pipeline`](@ref) fold loop — a pipeline-level callable sees the fold's data *before* any pipeline step has transformed it. `w_prev` is populated only when the fold loop runs sequentially and a previous fold exists; `path_id` only under multi-path schemes.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TimeDependentContext(;
        i::Integer, n::Integer, rd::Prices_RR, train_idx, test_idx,
        w_prev::Option{<:VecNum} = nothing, path_id::Option{<:Integer} = nothing
    ) -> TimeDependentContext

Keywords correspond to the struct's fields.

## Validation

  - `1 <= i <= n`.

# Related

  - [`TimeDependent`](@ref)
  - [`update_time_dependent_estimator`](@ref)
"""
struct TimeDependentContext{T1, T2, T3, T4, T5, T6, T7} <: AbstractResult
    """
    Index of the fold within the scheme's `split` enumeration (1-based); indexes `train_idx`/`test_idx`.
    """
    i::T1
    """
    Number of folds within the path.
    """
    n::T2
    """
    The fold loop's (possibly asset-viewed) returns data.
    """
    rd::T3
    """
    Per-path training index vectors.
    """
    train_idx::T4
    """
    Per-path test index vectors.
    """
    test_idx::T5
    """
    Previous fold's portfolio weights, when threaded; `nothing` otherwise.
    """
    w_prev::T6
    """
    Path identifier under multi-path schemes; `nothing` otherwise.
    """
    path_id::T7
    function TimeDependentContext(i::Integer, n::Integer, rd::Prices_RR, train_idx,
                                  test_idx, w_prev::Option{<:VecNum},
                                  path_id::Option{<:Integer})
        @argcheck(1 <= i <= n, DomainError(i, "fold index i must be in 1:$n"))
        return new{typeof(i), typeof(n), typeof(rd), typeof(train_idx), typeof(test_idx),
                   typeof(w_prev), typeof(path_id)}(i, n, rd, train_idx, test_idx, w_prev,
                                                    path_id)
    end
end
function TimeDependentContext(; i::Integer, n::Integer, rd::Prices_RR, train_idx, test_idx,
                              w_prev::Option{<:VecNum} = nothing,
                              path_id::Option{<:Integer} = nothing)::TimeDependentContext
    return TimeDependentContext(i, n, rd, train_idx, test_idx, w_prev, path_id)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve a time-dependent constraint to its value for the fold described by `ctx`.

Vector values index entry `ctx.i`; callables are invoked with `ctx`.

# Related

  - [`TimeDependent`](@ref)
  - [`TimeDependentContext`](@ref)
"""
function time_dependent_value(td::TimeDependent, ctx::TimeDependentContext)
    v = td.val
    if isa(v, AbstractVector)
        return v[ctx.i]
    elseif isa(v, PreviousWeightsFunction)
        return v.f(ctx)
    end
    return v(ctx)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if a time-dependent constraint requires the previous optimisation's weights.

`true` for a [`PreviousWeightsFunction`](@ref) value; for vector values, delegates to [`needs_previous_weights`](@ref) on entries that support it (turnover, fees, tracking), descending into per-fold vector entries. Bare callables contribute `false` — their output cannot be inspected.

# Related

  - [`TimeDependent`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function needs_previous_weights(td::TimeDependent)::Bool
    v = td.val
    if isa(v, PreviousWeightsFunction)
        return true
    elseif isa(v, TimeDependentCallable)
        return needs_previous_weights(v)
    elseif isa(v, AbstractVector)
        return any(time_dependent_entry_needs_previous_weights, v)
    end
    return false
end
"""
    time_dependent_entry_needs_previous_weights(x)

Return `true` if a per-fold entry value of a [`TimeDependent`](@ref) requires the previous optimisation's weights.

Delegates to [`needs_previous_weights`](@ref) for the value types that support the trait (turnover, fees, tracking); every other value contributes `false`.

# Related

  - [`TimeDependent`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function time_dependent_entry_needs_previous_weights(::Any)::Bool
    return false
end
function time_dependent_entry_needs_previous_weights(x::Union{<:TnE_Tn, <:FeesE_Fees,
                                                              <:Tr_VecTr,
                                                              <:AbstractBaseRiskMeasure})::Bool
    return needs_previous_weights(x)
end
function time_dependent_entry_needs_previous_weights(x::AbstractVector)::Bool
    return any(time_dependent_entry_needs_previous_weights, x)
end
function time_dependent_entry_needs_previous_weights(x::OptE_Opt)::Bool
    return needs_previous_weights(x)
end
function time_dependent_entry_needs_previous_weights(x::TimeDependent)::Bool
    return needs_previous_weights(x)
end
function port_opt_view(td::TimeDependent, i, args...)
    v = td.val
    if isa(v, AbstractVector)
        v = [port_opt_view(x, i, args...) for x in v]
    end
    return TimeDependent(v, td.bind; default = port_opt_view(td.default, i, args...))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Slice a [`TimeDependent`](@ref) schedule of scalar-or-array field values (warm starts, initial weights) to asset indices `i`.

Vector schedules slice each per-fold entry, and the `default` when one is set; callable schedules pass through — they see the sliced universe via their fold context's `rd`.

# Related

  - [`TimeDependent`](@ref)
  - [`port_opt_view`](@ref)
"""
function nothing_scalar_array_view(td::TimeDependent, i)
    v = td.val
    if isa(v, AbstractVector)
        v = [nothing_scalar_array_view(x, i) for x in v]
    end
    d = td.default
    if !isa(d, NoDefault)
        d = nothing_scalar_array_view(d, i)
    end
    return TimeDependent(v, td.bind; default = d)
end
"""
    inner_fold_fields(opt)

Field names of `opt` that the host hands across a fold loop it opens *itself*, so the loop that merely reaches the host is never the nearest one for them.

The default is the empty tuple: an ordinary host opens no inner fold loop, so the loop reaching it is both outermost and nearest for every field. A meta-optimiser whose inner cross-validation consumes a field directly declares that field here (e.g. `NestedClustered`'s `opti`, entered per cluster as `cross_val_predict(opti, …; cols = cl)`), and every generic pass — [`time_dependent_fields`](@ref), and through it update, reset and the fold-count assertion — then leaves a `bind = :nearest` schedule in that field for the host's own inner loop. Without the reset leg of this rule, the fold-less reset at the top of `_optimise` would replace a `:nearest` optimiser schedule with its `default` before the inner cross-validation ever saw it.

# Related

  - [`entitled`](@ref)
  - [`time_dependent_fields`](@ref)
  - [`reset_time_dependent_fields`](@ref)
"""
function inner_fold_fields(::Any)::Tuple
    return ()
end
"""
    time_dependent_candidate_fields(opt)

Field names of `opt` whose *type* admits a [`TimeDependent`](@ref) value — the candidate set [`time_dependent_fields`](@ref) narrows by value.

Whether a field can hold a schedule is decidable from `fieldtype` alone: a host built through the widened constructor signatures (see [`TD_Option`](@ref)) records a schedule in the field's type parameter, so a field that holds no schedule cannot have a type intersecting [`TimeDependent`](@ref). The tuple is therefore computed once per host type by a generated function, and a fold-invariant scan over a wide static host such as `JuMPOptimiser`, whose fields number in the dozens, folds to an empty tuple at compile time rather than walking every field dynamically on every `split` and `_optimise`.

This stays derived from the field types — no hand-maintained list — so the constructor signatures remain the single source of truth for which fields may vary over folds (ADR 0030).

# Related

  - [`TimeDependent`](@ref)
  - [`TD_Option`](@ref)
  - [`time_dependent_fields`](@ref)
"""
@generated function time_dependent_candidate_fields(::T) where {T}
    fns = Tuple(f
                for f in fieldnames(T)
                if typeintersect(fieldtype(T, f), TimeDependent) !== Union{})
    return :($fns)
end
"""
    entitled(opt, f::Symbol, all_binds::Bool)

Return `true` when the recursion position described by `all_binds` may consume a `bind = :nearest` schedule in field `f` of `opt`.

Entitlement is per-field, not per-host: a loop scanning with `all_binds = true` takes `:nearest` schedules everywhere *except* in the fields the host hands across its own inner fold loop (see [`inner_fold_fields`](@ref)) — for those, the host's inner loop is the nearest one, whatever loop is doing the scanning. A field is taken by a pass iff `entitled(opt, f, all_binds) || bind === :outermost`.

# Related

  - [`inner_fold_fields`](@ref)
  - [`time_dependent_fields`](@ref)
"""
function entitled(opt, f::Symbol, all_binds::Bool)::Bool
    return all_binds && !(f in inner_fold_fields(opt))
end
"""
    time_dependent_fields(opt, all_binds::Bool = true)

Return the tuple of field names of `opt` whose values are [`TimeDependent`](@ref).

The scan is generic over the host's fields, so the widened constructor signatures (see [`TD_Option`](@ref)) remain the single source of truth for which fields may vary over folds — there is no hand-maintained list. Only the fields whose type admits a schedule are visited (see [`time_dependent_candidate_fields`](@ref)); the rest are ruled out at compile time, so a static host returns an empty tuple without touching its fields.

# The `all_binds` argument

`all_binds` encodes something the schedule's own `bind` field cannot: it is a property of the *recursion position*, not of the schedule. A [`TimeDependent`](@ref)'s `bind` (`:outermost` / `:nearest`) says *which* fold loop the schedule wants; `all_binds` says whether the loop currently recursing is *entitled* to consume nearest-bound schedules at this depth. The second fact is not on the schedule.

Why position matters: under `outer CV loop → meta → (meta's inner CV loop) → inner estimator with a :nearest field`, the same `:nearest` field is visited by two loops. The outer loop recurses through the meta (mandatory — that recursion is how an inner estimator's `:outermost` field is resolved against the outer folds) and must *skip* the `:nearest` field, because it is not the nearest enclosing loop. The meta's inner CV loop drives the same estimator directly and must *consume* it, because it is. Same field, same `bind`, opposite actions — the difference is whether a nearer fold-loop boundary was crossed to reach it, which is exactly what `all_binds` carries.

So `all_binds` is `true` at every ordinary (outermost/standalone) fold loop — which is both outermost and nearest, and therefore takes everything remaining, including `:nearest`. It is forced to `false` only where a meta-optimiser recurses into the estimators its own inner CV owns, leaving their `:nearest` schedules for that inner loop. With `all_binds = false`, only fields with `bind === :outermost` are returned.

Entitlement is refined **per field** by [`inner_fold_fields`](@ref): even at `all_binds = true`, a `:nearest` schedule in a field the host hands across its *own* inner fold loop is left alone — the host's inner loop, not the scanning one, is nearest for that field (see [`entitled`](@ref)).

# Related

  - [`TimeDependent`](@ref)
  - [`TD_Option`](@ref)
  - [`is_time_dependent`](@ref)
  - [`inner_fold_fields`](@ref)
  - [`entitled`](@ref)
"""
function time_dependent_fields(opt, all_binds::Bool = true)
    fns = time_dependent_candidate_fields(opt)
    return filter(f -> begin
                      x = getfield(opt, f)
                      if isa(x, TimeDependent)
                          (entitled(opt, f, all_binds) || x.bind === :outermost)
                      else
                          false
                      end
                  end, fns)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Test-substitute every vector entry of the [`TimeDependent`](@ref)-valued fields in `args` through the keyword constructor of `T`.

`args` holds the host constructor's arguments; `defaults` the static defaults of the fields that may be time-dependent (see [`time_dependent_field_defaults`](@ref)). Each per-fold entry, and an explicit `default`, is substituted into its field — with every other time-dependent field standing at a value of its own (see [`time_dependent_stand_in`](@ref)) — and the constructor re-run, surfacing type and cross-field errors at construction time instead of mid-backtest. Substituted calls contain no `TimeDependent` values, so the recursion terminates.

Validation is skipped when a time-dependent field has no stand-in at all — a callable schedule in a required field, whose value only exists once a fold context does.

# Related

  - [`TimeDependent`](@ref)
  - [`time_dependent_fields`](@ref)
  - [`time_dependent_stand_in`](@ref)
"""
function assert_time_dependent_substitution(::Type{T}, args::NamedTuple,
                                            defaults::NamedTuple)::Nothing where {T}
    tdfs = filter(f -> isa(args[f], TimeDependent), keys(args))
    if isempty(tdfs)
        return nothing
    end
    stand_ins = map(f -> time_dependent_stand_in(args[f], defaults, f), tdfs)
    if any(isnothing, stand_ins)
        return nothing
    end
    base = merge(args, NamedTuple{tdfs}(map(something, stand_ins)))
    for f in tdfs
        td = args[f]
        v = td.val
        if isa(v, AbstractVector)
            for x in v
                T(; merge(base, NamedTuple{(f,)}((x,)))...)
            end
        end
        d = td.default
        if !isa(d, NoDefault)
            T(; merge(base, NamedTuple{(f,)}((d,)))...)
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a valid static value for a [`TimeDependent`](@ref)-valued field, wrapped in `Some`, or `nothing` if none exists.

Used by [`assert_time_dependent_substitution`](@ref) to stand the *other* time-dependent fields at a valid value while it test-substitutes one of them: the schedule's `default`, else the field's static default, else the schedule's first entry. A callable schedule in a field with no default of either kind has no stand-in — its value exists only inside a fold — so it returns `nothing` and validation is skipped.

Unlike [`time_dependent_reset_value`](@ref) this never throws: a schedule without a fold-less value is legitimate at construction time and only fails if it reaches a fold-less solve.

# Related

  - [`assert_time_dependent_substitution`](@ref)
  - [`time_dependent_reset_value`](@ref)
  - [`NoDefault`](@ref)
"""
function time_dependent_stand_in(td::TimeDependent, defaults::NamedTuple, field::Symbol)
    d = td.default
    if !isa(d, NoDefault)
        return Some(d)
    end
    d = get(defaults, field, nothing)
    if !isa(d, NoDefault)
        return Some(d)
    end
    v = td.val
    return isa(v, AbstractVector) ? Some(v[1]) : nothing
end
"""
    assert_time_dependent_fold_count(opt, n::Integer, all_binds::Bool = true)

Assert that every vector-valued time-dependent constraint in `opt` has exactly `n` entries.

Called by the cross-validation fold loops immediately after `split`, before any fold runs. The default is a no-op; hosts scan their [`time_dependent_fields`](@ref) and wrapper optimisers recurse. When `all_binds` is `false`, `bind === :nearest` schedules are skipped — they are validated by the nearest enclosing fold loop against its own fold count instead (see [`TimeDependent`](@ref)).

# Related

  - [`TimeDependent`](@ref)
  - [`time_dependent_fields`](@ref)
  - [`is_time_dependent`](@ref)
"""
function assert_time_dependent_fold_count(::OptE_Opt, ::Integer, ::Bool = true)::Nothing
    return nothing
end
function assert_time_dependent_fold_count(::Nothing, ::Integer, ::Bool = true)::Nothing
    return nothing
end
function assert_time_dependent_fold_count(td::TimeDependent, n::Integer,
                                          field::Symbol)::Nothing
    v = td.val
    if isa(v, AbstractVector)
        @argcheck(length(v) == n,
                  DimensionMismatch("time-dependent entries for $field ($(length(v))) must equal the number of folds ($n)"))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert the fold count of every [`TimeDependent`](@ref)-valued field of a host optimiser.

# Related

  - [`assert_time_dependent_fold_count`](@ref)
  - [`time_dependent_fields`](@ref)
"""
function assert_time_dependent_fields_fold_count(opt, n::Integer,
                                                 all_binds::Bool = true)::Nothing
    for f in time_dependent_fields(opt, all_binds)
        assert_time_dependent_fold_count(getfield(opt, f), n, f)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if the base optimiser configuration carries time-dependent constraints.

# Related

  - [`BaseOptimisationEstimator`](@ref)
  - [`TimeDependent`](@ref)
  - [`is_time_dependent`](@ref)
"""
function is_time_dependent(opt::BaseOptimisationEstimator)
    return !isempty(time_dependent_fields(opt))
end
function assert_time_dependent_fold_count(opt::BaseOptimisationEstimator, n::Integer,
                                          all_binds::Bool = true)::Nothing
    assert_time_dependent_fields_fold_count(opt, n, all_binds)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the time-dependent constraints of a base optimiser configuration for the fold described by `ctx`.

Rebuilds the configuration through its validated keyword constructor with each [`TimeDependent`](@ref)-valued field replaced by its resolved per-fold value, so the result is an ordinary static configuration. When `all_binds` is `false`, `bind === :nearest` fields are left in place for the nearest enclosing fold loop to consume.

# Related

  - [`BaseOptimisationEstimator`](@ref)
  - [`update_time_dependent_estimator`](@ref)
  - [`update_time_dependent_fields`](@ref)
"""
function update_time_dependent_estimator(opt::BaseOptimisationEstimator,
                                         ctx::TimeDependentContext, all_binds::Bool = true)
    return update_time_dependent_fields(opt, ctx, all_binds)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Replace the time-dependent constraints of a base optimiser configuration with their static defaults (see [`time_dependent_field_defaults`](@ref)).

# Related

  - [`BaseOptimisationEstimator`](@ref)
  - [`reset_time_dependent_estimator`](@ref)
  - [`reset_time_dependent_fields`](@ref)
"""
function reset_time_dependent_estimator(opt::BaseOptimisationEstimator)
    return reset_time_dependent_fields(opt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuild an estimator through its keyword constructor with the fields in `repl` replaced.

All remaining fields are carried through unchanged. Because the rebuild goes through the validated keyword constructor, every construction invariant re-runs.

# Related

  - [`update_time_dependent_fields`](@ref)
  - [`reset_time_dependent_fields`](@ref)
"""
function rebuild_estimator(x, repl::NamedTuple)
    fns = fieldnames(typeof(x))
    nt = NamedTuple{fns}(map(f -> getfield(x, f), fns))
    return (typeof(x).name.wrapper)(; merge(nt, repl)...)
end
"""
    is_time_dependent(opt)

Return `true` if the optimiser carries time-dependent constraints.

The default returns `false`. Hosts return `true` when any of their fields holds a [`TimeDependent`](@ref) (see [`time_dependent_fields`](@ref)); wrapper optimisers recurse into their inner optimiser and fallback.

# Arguments

  - `opt`: Optimisation estimator, result, or vector thereof.

# Returns

  - `Bool`: `true` if the estimator is time-dependent.

# Related

  - [`TimeDependent`](@ref)
  - [`update_time_dependent_estimator`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function is_time_dependent(::OptE_Opt)
    return false
end
function is_time_dependent(::Nothing)
    return false
end
"""
    update_time_dependent_estimator(opt, ctx::TimeDependentContext, all_binds::Bool = true)

Resolve the time-dependent constraints of `opt` for the fold described by `ctx`.

The default returns the estimator unchanged. Hosts rebuild themselves through their validated keyword constructor with each [`TimeDependent`](@ref)-valued field replaced by its resolved per-fold value, so the result is an ordinary static estimator; wrapper optimisers recurse.

# Arguments

  - `opt`: Optimisation estimator or result.
  - `ctx::TimeDependentContext`: The fold's context.
  - `all_binds::Bool`: When `false`, `bind === :nearest` schedules are skipped, leaving them for the nearest enclosing fold loop to consume. Meta-optimisers pass `false` when recursing into the estimators their internal fold loop processes; fold loops call with the default `true`.

# Returns

  - Updated estimator.

# Related

  - [`TimeDependent`](@ref)
  - [`TimeDependentContext`](@ref)
  - [`is_time_dependent`](@ref)
"""
function update_time_dependent_estimator(opt::OptE_Opt, ::TimeDependentContext,
                                         ::Bool = true)
    return opt
end
function update_time_dependent_estimator(::Nothing, ::TimeDependentContext, ::Bool = true)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuild a host optimiser with each [`TimeDependent`](@ref)-valued field replaced by its per-fold value for `ctx`.

Shared implementation behind the hosts' [`update_time_dependent_estimator`](@ref) methods. Returns `opt` unchanged when no field is time-dependent.

# Related

  - [`update_time_dependent_estimator`](@ref)
  - [`time_dependent_fields`](@ref)
  - [`time_dependent_value`](@ref)
"""
function update_time_dependent_fields(opt, ctx::TimeDependentContext,
                                      all_binds::Bool = true)
    tdfs = time_dependent_fields(opt, all_binds)
    if isempty(tdfs)
        return opt
    end
    repl = NamedTuple{tdfs}(map(f -> time_dependent_value(getfield(opt, f), ctx), tdfs))
    return rebuild_estimator(opt, repl)
end
"""
    time_dependent_field_defaults(opt)

Return a `NamedTuple` of the static defaults of the optimiser fields that may hold a [`TimeDependent`](@ref), for those whose default is not `nothing`.

Used by [`reset_time_dependent_estimator`](@ref) to replace per-fold schedules with their static defaults on fold-less solves; fields absent from the tuple default to `nothing`. A *required* field — one with no static default at all, i.e. the optimiser-valued fields — is listed with [`NoDefault`](@ref), which is not a value it can take but a declaration that a schedule there must carry its own `default`. The fallback method returns an empty tuple.

# Related

  - [`reset_time_dependent_estimator`](@ref)
  - [`time_dependent_reset_value`](@ref)
  - [`NoDefault`](@ref)
  - [`TD_Option`](@ref)
"""
function time_dependent_field_defaults(::Any)::NamedTuple
    return (;)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the value a [`TimeDependent`](@ref)-valued field takes outside every fold loop.

The schedule's own `default` wins; absent one ([`NoDefault`](@ref)), the host's static default for `field` is used ([`time_dependent_field_defaults`](@ref), `nothing` for fields it omits). Throws a [`TimeDependentDefaultError`](@ref) when neither exists — a schedule in a required field that never said what a fold-less solve should do.

# Related

  - [`reset_time_dependent_fields`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`TimeDependentDefaultError`](@ref)
"""
function time_dependent_reset_value(td::TimeDependent, defaults::NamedTuple, field::Symbol,
                                    opt)
    d = td.default
    if !isa(d, NoDefault)
        return d
    end
    d = get(defaults, field, nothing)
    if isa(d, NoDefault)
        throw(TimeDependentDefaultError("field `$field` of $(nameof(typeof(opt))) holds a TimeDependent schedule, but the field has no static default and the schedule supplies none, so there is no value to use outside a fold loop. A schedule is defined only over the folds of a cross-validation scheme; this solve has none. Give the schedule a fold-less value: TimeDependent(val; default = x)."))
    end
    return d
end
"""
    reset_time_dependent_estimator(opt)

Replace every [`TimeDependent`](@ref)-valued field of `opt` with its static default, recursing through wrapper optimisers.

A time-dependent constraint is defined only over the folds of a cross-validation scheme, so a fold-less solve runs with the affected fields at their static defaults (see [`time_dependent_field_defaults`](@ref)). Called at the top of the `_optimise` methods; per-fold estimators produced by [`update_time_dependent_estimator`](@ref) contain no `TimeDependent` values, so they pass through unchanged. The default returns the estimator unchanged; hosts rebuild themselves, wrapper optimisers recurse.

# Related

  - [`TimeDependent`](@ref)
  - [`update_time_dependent_estimator`](@ref)
  - [`is_time_dependent`](@ref)
"""
function reset_time_dependent_estimator(opt::OptE_Opt)
    return opt
end
function reset_time_dependent_estimator(::Nothing)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuild a host optimiser with each [`TimeDependent`](@ref)-valued field replaced by its fold-less value (see [`time_dependent_reset_value`](@ref)).

Shared implementation behind the hosts' [`reset_time_dependent_estimator`](@ref) methods. Returns `opt` unchanged when no field is time-dependent.

# Related

  - [`reset_time_dependent_estimator`](@ref)
  - [`time_dependent_reset_value`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`time_dependent_fields`](@ref)
"""
function reset_time_dependent_fields(opt)
    tdfs = time_dependent_fields(opt)
    if isempty(tdfs)
        return opt
    end
    defaults = time_dependent_field_defaults(opt)
    repl = NamedTuple{tdfs}(map(f -> time_dependent_reset_value(getfield(opt, f), defaults,
                                                                f, opt), tdfs))
    return rebuild_estimator(opt, repl)
end
#! End: Overload these for all estimators which can use time-dependent constraints.
#! Begin: TimeDependent as an optimiser in its own right.
"""
$(DocStringExtensions.TYPEDSIGNATURES)

A [`TimeDependent`](@ref) schedule is time-dependent by construction.

# Related

  - [`is_time_dependent`](@ref)
"""
function is_time_dependent(::TimeDependent)
    return true
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that a [`TimeDependent`](@ref) schedule in an optimiser position resolved to something that can be optimised or predicted.

The vector and [`TimeDependentOptimiserCallable`](@ref) forms of a schedule declare their output kind in their type and are checked statically (see [`TD_OptE_Opt`](@ref)). The two callable forms — a bare `ctx -> optimiser` and a [`PreviousWeightsFunction`](@ref) wrapping one — cannot be, so their output is checked here, when the fold loop swaps it in.

# Related

  - [`TD_OptE_Opt`](@ref)
  - [`update_time_dependent_estimator`](@ref)
"""
function assert_time_dependent_optimiser(::OptE_Opt)::Nothing
    return nothing
end
function assert_time_dependent_optimiser(opt)::Nothing
    return throw(ArgumentError("a TimeDependent schedule in an optimiser position resolved to a $(typeof(opt)), which is neither an optimisation estimator nor a precomputed optimisation result. A callable schedule standing in for an optimiser must return an OptE_Opt for every fold."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve a [`TimeDependent`](@ref) schedule standing in for an optimiser to the optimiser of fold `ctx.i`.

Entry `i` may be an estimator or a precomputed result, so a *mixed* schedule optimises on some folds and predicts on others. After the swap the resolved estimator is recursed into with the **same** context, so its own `:outermost` schedules bind to this fold loop rather than going unresolved.

Returns the schedule unchanged when `all_binds` is `false` and it is not `:outermost`-bound — a `:nearest` schedule in an optimiser position is consumed by a fold loop the host itself opens, not by the loop that reached the host.

# Related

  - [`TD_OptE_Opt`](@ref)
  - [`update_time_dependent_estimator`](@ref)
  - [`assert_time_dependent_optimiser`](@ref)
"""
function update_time_dependent_estimator(td::TD_OptE_Opt, ctx::TimeDependentContext,
                                         all_binds::Bool = true)
    if !all_binds && td.bind !== :outermost
        return td
    end
    opt = time_dependent_value(td, ctx)
    assert_time_dependent_optimiser(opt)
    return update_time_dependent_estimator(opt, ctx, all_binds)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that a [`TimeDependent`](@ref) schedule standing in for an optimiser has one entry per fold, and that the schedules *within* each entry are sized to the same fold loop.

Entry `i` runs at fold `i` of this loop, so its own `:outermost` schedules bind here too (see [`update_time_dependent_estimator`](@ref)) and are validated against this loop's fold count. The `default` is not — it runs only outside a fold loop, where its schedules reset instead.

Skipped when `all_binds` is `false` and the schedule is not `:outermost`-bound — the fold loop the host opens validates it against its own fold count instead.

# Related

  - [`assert_time_dependent_fold_count`](@ref)
  - [`TD_OptE_Opt`](@ref)
"""
function assert_time_dependent_fold_count(td::TDO_OptE_Opt, n::Integer,
                                          all_binds::Bool = true)::Nothing
    if !all_binds && td.bind !== :outermost
        return nothing
    end
    v = td.val
    if isa(v, AbstractVector)
        @argcheck(length(v) == n,
                  DimensionMismatch("a TimeDependent schedule of optimisers has $(length(v)) entries, which must equal the number of folds ($n)"))
        for opt in v
            assert_time_dependent_fold_count(opt, n, all_binds)
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the optimiser a [`TimeDependent`](@ref) schedule takes outside every fold loop.

An optimiser position is *required* — there is no static default to fall back to — so the schedule must supply its own `default`, and one that does not throws a [`TimeDependentDefaultError`](@ref). The fold-less optimiser is itself reset, so its own schedules resolve to their defaults too.

# Related

  - [`reset_time_dependent_estimator`](@ref)
  - [`NoDefault`](@ref)
  - [`TimeDependentDefaultError`](@ref)
"""
function reset_time_dependent_estimator(td::TD_OptE_Opt)
    d = td.default
    if isa(d, NoDefault)
        throw(TimeDependentDefaultError("a TimeDependent schedule stands in for the optimiser itself but supplies no `default`, so there is no optimiser to run outside a fold loop. A schedule is defined only over the folds of a cross-validation scheme; this solve has none. Give the schedule a fold-less optimiser: TimeDependent(val; default = opt)."))
    end
    return reset_time_dependent_estimator(d)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the statically inspectable entries of a [`TimeDependent`](@ref) schedule: the per-fold values of a vector schedule, plus its `default` when it has one. A callable schedule contributes nothing — its per-fold values cannot be inspected before it runs.

# Related

  - [`TimeDependent`](@ref)
"""
function time_dependent_entries(td::TimeDependent)
    v = td.val
    entries = isa(v, AbstractVector) ? collect(Any, v) : Any[]
    if !isa(td.default, NoDefault)
        push!(entries, td.default)
    end
    return entries
end
function assert_internal_optimiser(td::Union{<:TD_OptE_Opt, <:TD_VecOptE_Opt})::Nothing
    for opt in time_dependent_entries(td)
        assert_internal_optimiser(opt)
    end
    return nothing
end
function assert_external_optimiser(td::Union{<:TD_OptE_Opt, <:TD_VecOptE_Opt})::Nothing
    for opt in time_dependent_entries(td)
        assert_external_optimiser(opt)
    end
    return nothing
end
function assert_special_nco_requirements(td::Union{<:TD_OptE_Opt, <:TD_VecOptE_Opt})::Nothing
    for opt in time_dependent_entries(td)
        assert_special_nco_requirements(opt)
    end
    return nothing
end
"""
    assert_no_nearest_bind_optimiser_schedule(x, field::Symbol, host::Symbol)

Reject a `bind = :nearest` [`TimeDependent`](@ref) schedule in an optimiser-valued position no inner fold loop consumes.

`bind` picks *which fold loop supplies the schedule's index*. `:nearest` therefore says something different from `:outermost` only where the host opens a fold loop of its **own** and hands the field across it — `NestedClustered.opti` (its inner cross-validation is entered per cluster) and `Stacking.opti[k]` (entered per candidate), the positions declared by [`inner_fold_fields`](@ref). Everywhere else the loop that reaches the host *is* the nearest one, and the two binds would name the same loop.

The positions this guards have no such inner loop:

  - **A fallback (`fb`), on every host.** The fallback walk is a retry chain *within a single fold's solve* — it has no fold indices of its own — so `:nearest` there is either redundant with `:outermost` or, behind a meta's inner cross-validation, silently wrong: it would resolve against the inner loop's fold numbers (tuning folds) instead of the backtest's periods, changing meaning with nesting depth. A per-fold fallback is fully expressible with `:outermost`, including `nothing` entries to switch it off on some folds (see [`TDO_OptE_Opt`](@ref)).
  - **The outer optimisers (`opto`).** They consume the *combined* inner output, once per solve.
  - **`SubsetResampling.opt`.** Its internal loop is over randomly drawn asset subsets, not time folds.

So a `:nearest` schedule in any of them has no nearest fold loop to bind to, and is rejected at construction rather than resolving against a loop the caller did not mean. No-op for anything that is not a [`TimeDependent`](@ref).

# Related

  - [`assert_nearest_optimiser_schedule`](@ref)
  - [`inner_fold_fields`](@ref)
  - [`TDO_OptE_Opt`](@ref)
"""
function assert_no_nearest_bind_optimiser_schedule(x, field::Symbol, host::Symbol)::Nothing
    if isa(x, TimeDependent)
        @argcheck(x.bind !== :nearest,
                  ArgumentError("field `$field` of $host holds a `bind = :nearest` TimeDependent schedule, but no inner fold loop of $host consumes `$field`, so there is no nearest fold loop for it to bind to. Use `bind = :outermost`: the fold loop that reaches the $host resolves the schedule."))
    end
    return nothing
end
"""
    assert_nearest_optimiser_schedule(x, field::Symbol, cv, host::Symbol)

Validate a `bind = :nearest` [`TimeDependent`](@ref) schedule in an optimiser-valued position that a host's inner cross-validation *does* consume.

Two construction-time requirements, both consequences of the position's double consumer: the inner cross-validation leg resolves the schedule per fold, while the full-sample leg (the meta's `wi` fit, or the per-cluster optimise) always resolves it fold-lessly to its `default`.

  - An explicit `default` is required — without one, every solve would throw a [`TimeDependentDefaultError`](@ref) when the full-sample leg reaches the schedule, so the error is moved to construction. This deliberately departs from the rule that a defaultless schedule is legal at construction, for this position only.
  - `cv !== nothing` is required — without an inner cross-validation there is no inner fold loop, so the schedule could only ever be its `default`: silently inert.

No-op for anything that is not a `bind = :nearest` [`TimeDependent`](@ref).

# Related

  - [`assert_no_nearest_bind_optimiser_schedule`](@ref)
  - [`inner_fold_fields`](@ref)
  - [`TimeDependentDefaultError`](@ref)
"""
function assert_nearest_optimiser_schedule(x, field::Symbol, cv, host::Symbol)::Nothing
    if isa(x, TimeDependent) && x.bind === :nearest
        @argcheck(!isa(x.default, NoDefault),
                  TimeDependentDefaultError("a `bind = :nearest` schedule in `$field` of $host must supply a `default`: besides the inner cross-validation fold loop, `$field` also has a fold-less full-sample consumer that always resolves the schedule to its `default`, so a defaultless one would throw on every solve. Give it a fold-less optimiser: TimeDependent(val, :nearest; default = opt)."))
        @argcheck(!isnothing(cv),
                  ArgumentError("a `bind = :nearest` schedule in `$field` of $host requires `cv`: without an inner cross-validation there is no inner fold loop, so the schedule could only ever resolve to its `default` — silently inert. Provide `cv`, or use `bind = :outermost` so the fold loop that reaches the $host consumes it."))
    end
    return nothing
end
#! End: TimeDependent as an optimiser in its own right.
"""
    const VecOptE_Opt = AbstractVector{<:OptE_Opt}

Alias for a vector of optimisation estimators or results.

Represents a collection of [`OptE_Opt`](@ref) objects for batch processing.

# Related

  - [`OptE_Opt`](@ref)
"""
const VecOptE_Opt = AbstractVector{<:OptE_Opt}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply [`factory`](@ref) through a [`TimeDependent`](@ref) schedule: to each vector entry and to the `default`, rebuilding the schedule.

A schedule can survive a fold loop's resolution pass (a `bind = :nearest` element left for a meta's inner cross-validation), so the factory pass that follows resolution must see through it. Callable forms pass through unchanged — their per-fold values do not exist yet, and a callable receives the fold's context (including `w_prev`) when it runs.

# Related

  - [`factory`](@ref)
  - [`TimeDependent`](@ref)
"""
function factory(td::TimeDependent, args...)
    v = td.val
    if isa(v, AbstractVector)
        v = [factory(x, args...) for x in v]
    end
    d = td.default
    if !isa(d, NoDefault)
        d = factory(d, args...)
    end
    return TimeDependent(v, td.bind; default = d)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert special NCO requirements for each element of a vector of optimisation estimators or results.

# Related

  - [`assert_special_nco_requirements`](@ref)
  - [`NestedClustered`](@ref)
"""
function assert_special_nco_requirements(opt::VecOptE_Opt_TD)::Nothing
    for opti in opt
        assert_special_nco_requirements(opti)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if any element of the vector of optimisation estimators or results requires previous portfolio weights.

# Related

  - [`needs_previous_weights`](@ref)
  - [`VecOptE_Opt`](@ref)
"""
function needs_previous_weights(opt::VecOptE_Opt_TD)
    return any(needs_previous_weights.(opt))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if any element of the vector of optimisation estimators or results is time-dependent.

# Related

  - [`is_time_dependent`](@ref)
  - [`VecOptE_Opt`](@ref)
"""
function is_time_dependent(opt::VecOptE_Opt_TD)
    return any(is_time_dependent.(opt))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply [`update_time_dependent_estimator`](@ref) element-wise to a vector of optimisation estimators or results.

# Related

  - [`update_time_dependent_estimator`](@ref)
  - [`VecOptE_Opt`](@ref)
"""
function update_time_dependent_estimator(opt::VecOptE_Opt_TD, ctx::TimeDependentContext,
                                         all_binds::Bool = true)
    return [update_time_dependent_estimator(opti, ctx, all_binds) for opti in opt]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply [`assert_time_dependent_fold_count`](@ref) element-wise to a vector of optimisation estimators or results.

# Related

  - [`assert_time_dependent_fold_count`](@ref)
  - [`VecOptE_Opt`](@ref)
"""
function assert_time_dependent_fold_count(opt::VecOptE_Opt_TD, n::Integer,
                                          all_binds::Bool = true)::Nothing
    for opti in opt
        assert_time_dependent_fold_count(opti, n, all_binds)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for JuMP-based weight finaliser formulations.

Defines the interface for norm types used when adjusting portfolio weights to satisfy bounds via a JuMP model.

# Interfaces

In order to implement a new formulation that works seamlessly with the library, subtype `JuMPWeightFinaliserFormulation` and implement the following method:

## `set_clustering_weight_finaliser_alg!`

  - `set_clustering_weight_finaliser_alg!(model::JuMP.Model, alg::MyFormulation, wi::VecNum) -> Nothing`: Adds the deviation objective to a model that already carries the decision vector `w`, the budget equality and the weight bounds.

### Arguments

  - `model`: The JuMP model, built by [`opt_weight_bounds`](@ref).
  - `alg`: The concrete subtype instance.
  - `wi`: The weights the optimisation produced, which the model repairs.

### Returns

  - `nothing`. The method works by adding variables, constraints and the objective to `model`.

# Related

  - [`RelativeErrorWeightFinaliser`](@ref)
  - [`SquaredRelativeErrorWeightFinaliser`](@ref)
  - [`AbsoluteErrorWeightFinaliser`](@ref)
  - [`SquaredAbsoluteErrorWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
abstract type JuMPWeightFinaliserFormulation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Minimises the L1 norm of relative weight deviations when enforcing weight bounds.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\min} &\\quad \\left\\lVert \\boldsymbol{w} \\oslash \\boldsymbol{w}_{0} - \\boldsymbol{1} \\right\\rVert_{1}\\,, \\\\
\\textrm{s.t.} &\\quad \\boldsymbol{1}^\\intercal \\boldsymbol{w} = \\boldsymbol{1}^\\intercal \\boldsymbol{w}_{0}\\,, \\\\
&\\quad \\boldsymbol{l} \\leq \\boldsymbol{w} \\leq \\boldsymbol{u}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:w_0_finaliser])
  - $(math_dict[:lb_ub_finaliser])
  - ``\\oslash``: Elementwise division. A zero entry of ``\\boldsymbol{w}_{0}`` is replaced by `eps` before the division, so the ratio stays finite.

# Constructors

    RelativeErrorWeightFinaliser() -> RelativeErrorWeightFinaliser

# Examples

```jldoctest
julia> RelativeErrorWeightFinaliser()
RelativeErrorWeightFinaliser()
```

# Related

  - [`JuMPWeightFinaliserFormulation`](@ref)
  - [`SquaredRelativeErrorWeightFinaliser`](@ref)
  - [`AbsoluteErrorWeightFinaliser`](@ref)
  - [`SquaredAbsoluteErrorWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
struct RelativeErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Minimises the L2 norm of relative weight deviations when enforcing weight bounds.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\min} &\\quad \\left\\lVert \\boldsymbol{w} \\oslash \\boldsymbol{w}_{0} - \\boldsymbol{1} \\right\\rVert_{2}\\,, \\\\
\\textrm{s.t.} &\\quad \\boldsymbol{1}^\\intercal \\boldsymbol{w} = \\boldsymbol{1}^\\intercal \\boldsymbol{w}_{0}\\,, \\\\
&\\quad \\boldsymbol{l} \\leq \\boldsymbol{w} \\leq \\boldsymbol{u}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:w_0_finaliser])
  - $(math_dict[:lb_ub_finaliser])
  - ``\\oslash``: Elementwise division. A zero entry of ``\\boldsymbol{w}_{0}`` is replaced by `eps` before the division, so the ratio stays finite.

The second-order cone bounds the norm itself, so the objective value is the L2 norm and not its square. The name records the squared-error criterion, whose minimiser is the same because the square is monotonic on a non-negative norm. [`RelativeErrorWeightFinaliser`](@ref) differs in the norm, not in the power.

# Constructors

    SquaredRelativeErrorWeightFinaliser() -> SquaredRelativeErrorWeightFinaliser

# Examples

```jldoctest
julia> SquaredRelativeErrorWeightFinaliser()
SquaredRelativeErrorWeightFinaliser()
```

# Related

  - [`JuMPWeightFinaliserFormulation`](@ref)
  - [`RelativeErrorWeightFinaliser`](@ref)
  - [`AbsoluteErrorWeightFinaliser`](@ref)
  - [`SquaredAbsoluteErrorWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
struct SquaredRelativeErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Minimises the L1 norm of absolute weight deviations when enforcing weight bounds.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\min} &\\quad \\left\\lVert \\boldsymbol{w} - \\boldsymbol{w}_{0} \\right\\rVert_{1}\\,, \\\\
\\textrm{s.t.} &\\quad \\boldsymbol{1}^\\intercal \\boldsymbol{w} = \\boldsymbol{1}^\\intercal \\boldsymbol{w}_{0}\\,, \\\\
&\\quad \\boldsymbol{l} \\leq \\boldsymbol{w} \\leq \\boldsymbol{u}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:w_0_finaliser])
  - $(math_dict[:lb_ub_finaliser])

# Constructors

    AbsoluteErrorWeightFinaliser() -> AbsoluteErrorWeightFinaliser

# Examples

```jldoctest
julia> AbsoluteErrorWeightFinaliser()
AbsoluteErrorWeightFinaliser()
```

# Related

  - [`JuMPWeightFinaliserFormulation`](@ref)
  - [`RelativeErrorWeightFinaliser`](@ref)
  - [`SquaredRelativeErrorWeightFinaliser`](@ref)
  - [`SquaredAbsoluteErrorWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
struct AbsoluteErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Minimises the L2 norm of absolute weight deviations when enforcing weight bounds.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\min} &\\quad \\left\\lVert \\boldsymbol{w} - \\boldsymbol{w}_{0} \\right\\rVert_{2}\\,, \\\\
\\textrm{s.t.} &\\quad \\boldsymbol{1}^\\intercal \\boldsymbol{w} = \\boldsymbol{1}^\\intercal \\boldsymbol{w}_{0}\\,, \\\\
&\\quad \\boldsymbol{l} \\leq \\boldsymbol{w} \\leq \\boldsymbol{u}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:w_0_finaliser])
  - $(math_dict[:lb_ub_finaliser])

The second-order cone bounds the norm itself, so the objective value is the L2 norm and not its square. The name records the squared-error criterion, whose minimiser is the same because the square is monotonic on a non-negative norm. [`AbsoluteErrorWeightFinaliser`](@ref) differs in the norm, not in the power.

# Constructors

    SquaredAbsoluteErrorWeightFinaliser() -> SquaredAbsoluteErrorWeightFinaliser

# Examples

```jldoctest
julia> SquaredAbsoluteErrorWeightFinaliser()
SquaredAbsoluteErrorWeightFinaliser()
```

# Related

  - [`JuMPWeightFinaliserFormulation`](@ref)
  - [`RelativeErrorWeightFinaliser`](@ref)
  - [`SquaredRelativeErrorWeightFinaliser`](@ref)
  - [`AbsoluteErrorWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
struct SquaredAbsoluteErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for weight finaliser strategies.

A `WeightFinaliser` enforces weight bounds after the optimisation has produced unconstrained weights.

# Interfaces

In order to implement a new strategy that works seamlessly with the library, subtype `WeightFinaliser` and implement the following method:

## `opt_weight_bounds`

  - `opt_weight_bounds(wf::MyFinaliser, wb::WeightBounds, w::VecNum) -> VecNum`: Moves `w` into the bounds `wb`, keeping the budget it already carries.

### Arguments

  - `wf`: The concrete subtype instance.
  - `wb`: The weight bounds. Either bound may be `nothing`.
  - `w`: The weights the optimisation produced.

### Returns

  - `w::VecNum`: The repaired weight vector. [`finalise_weight_bounds`](@ref) then pairs it with a return code.

# Related

  - [`IterativeWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
abstract type WeightFinaliser <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Iteratively projects weights into the feasible region defined by weight bounds.

Each pass clips the weights to the bounds, then redistributes the clipped mass over the entries that lie strictly inside the bounds, in proportion to their own weights. The pass ends by rescaling the vector to the budget it started with, so the sum is preserved. Passes run until the bounds hold or until `iter` passes are done. An absent bound is read as `typemin` or `typemax` of the weight element type.

The bounds are not guaranteed on exit. A bound set that no rescaled vector can satisfy exhausts the passes and returns the last vector: four assets summing to `1` under `lb = 0.3` return `[0.25, 0.25, 0.25, 0.25]`. [`finalise_weight_bounds`](@ref) tests finiteness alone, so such a vector is reported as an [`OptimisationSuccess`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    IterativeWeightFinaliser(;
        iter::Integer = 100
    ) -> IterativeWeightFinaliser

Keywords correspond to the struct's fields.

## Validation

  - `iter > 0`.

# Examples

```jldoctest
julia> IterativeWeightFinaliser()
IterativeWeightFinaliser
  iter ┴ Int64: 100
```

# Related

  - [`WeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
@concrete struct IterativeWeightFinaliser <: WeightFinaliser
    """
    $(field_dict[:iter])
    """
    iter
    function IterativeWeightFinaliser(iter::Integer)
        @argcheck(iter > 0, DomainError(iter, "iter must be > 0"))
        return new{typeof(iter)}(iter)
    end
end
function IterativeWeightFinaliser(; iter::Integer = 100)::IterativeWeightFinaliser
    return IterativeWeightFinaliser(iter)
end
"""
$(DocStringExtensions.TYPEDEF)

Uses a JuMP optimisation model to enforce weight bounds.

The programme keeps the budget of the input weights and holds every weight between the bounds, and `alg` states which deviation it minimises over that set. An absent bound adds no constraint. A failed solve raises a warning and falls back to a default [`IterativeWeightFinaliser`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    JuMPWeightFinaliser(;
        slv::Slv_VecSlv,
        sc::Number = 1.0,
        so::Number = 1.0,
        alg::JuMPWeightFinaliserFormulation = RelativeErrorWeightFinaliser()
    ) -> JuMPWeightFinaliser

Keywords correspond to the struct's fields.

## Validation

  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - `sc > 0`, `so > 0`.

# Examples

```jldoctest
julia> JuMPWeightFinaliser(; slv = Solver(; solver = nothing))
JuMPWeightFinaliser
  slv ┼ Solver
      │          name ┼ String: ""
      │        solver ┼ nothing
      │      settings ┼ nothing
      │     check_sol ┼ @NamedTuple{}: NamedTuple()
      │   add_bridges ┴ Bool: true
   sc ┼ Float64: 1.0
   so ┼ Float64: 1.0
  alg ┴ RelativeErrorWeightFinaliser()
```

# Related

  - [`WeightFinaliser`](@ref)
  - [`IterativeWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliserFormulation`](@ref)
"""
@concrete struct JuMPWeightFinaliser <: WeightFinaliser
    """
    $(field_dict[:slv])
    """
    slv
    """
    $(field_dict[:sc])
    """
    sc
    """
    $(field_dict[:so])
    """
    so
    """
    $(field_dict[:wfalg])
    """
    alg
    function JuMPWeightFinaliser(slv::Slv_VecSlv, sc::Number, so::Number,
                                 alg::JuMPWeightFinaliserFormulation)
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        @argcheck(sc > zero(sc), DomainError(sc, "sc must be positive"))
        @argcheck(so > zero(so), DomainError(so, "so must be positive"))
        return new{typeof(slv), typeof(sc), typeof(so), typeof(alg)}(slv, sc, so, alg)
    end
end
function JuMPWeightFinaliser(; slv::Slv_VecSlv, sc::Number = 1.0, so::Number = 1.0,
                             alg::JuMPWeightFinaliserFormulation = RelativeErrorWeightFinaliser())::JuMPWeightFinaliser
    return JuMPWeightFinaliser(slv, sc, so, alg)
end
"""
    set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                         alg::JuMPWeightFinaliserFormulation,
                                         wi::VecNum)

Add the deviation objective of `alg` to the weight finalisation model.

[`opt_weight_bounds`](@ref) has already added the decision vector `w`, the budget equality and the weight bounds. This method adds the epigraph variable `t`, the cone that bounds the deviation of `w` from `wi`, and the objective `Min so * t`. The cone is a `NormOneCone` for the two L1 formulations and a `SecondOrderCone` for the two L2 formulations.

# Arguments

  - `model`: JuMP model, which must already carry `w` and the two scale expressions.
  - `alg`: The deviation formulation, one of the four [`JuMPWeightFinaliserFormulation`](@ref) subtypes.
  - `wi`: The weights the optimisation produced, which the model repairs.

# Returns

  - `nothing`.

# Details

  - The two relative formulations divide by `wi`, so they first replace each zero entry of `wi` **in place** with `eps(eltype(wi))`. The caller's vector carries that substitution afterwards.

# Related

  - [`JuMPWeightFinaliserFormulation`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
  - [`opt_weight_bounds`](@ref)
"""
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::RelativeErrorWeightFinaliser, wi::VecNum)
    wi[iszero.(wi)] .= eps(eltype(wi))
    w = get_w(model)
    sc = get_constraint_scale(model)
    so = get_objective_scale(model)
    JuMP.@variable(model, t)
    JuMP.@constraint(model,
                     [sc * t;
                      sc * (w ⊘ wi .- one(eltype(wi)))] in
                     JuMP.MOI.NormOneCone(length(w) + 1))
    JuMP.@objective(model, Min, so * t)
    return nothing
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::SquaredRelativeErrorWeightFinaliser,
                                              wi::VecNum)
    wi[iszero.(wi)] .= eps(eltype(wi))
    w = get_w(model)
    sc = get_constraint_scale(model)
    so = get_objective_scale(model)
    JuMP.@variable(model, t)
    JuMP.@constraint(model,
                     [sc * t; sc * (w ⊘ wi .- one(eltype(wi)))] in JuMP.SecondOrderCone())
    JuMP.@objective(model, Min, so * t)
    return nothing
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::AbsoluteErrorWeightFinaliser, wi::VecNum)
    w = get_w(model)
    sc = get_constraint_scale(model)
    so = get_objective_scale(model)
    JuMP.@variable(model, t)
    JuMP.@constraint(model, [sc * t; sc * (w - wi)] in JuMP.MOI.NormOneCone(length(w) + 1))
    JuMP.@objective(model, Min, so * t)
    return nothing
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::SquaredAbsoluteErrorWeightFinaliser,
                                              wi::VecNum)
    w = get_w(model)
    sc = get_constraint_scale(model)
    so = get_objective_scale(model)
    JuMP.@variable(model, t)
    JuMP.@constraint(model, [sc * t; sc * (w - wi)] in JuMP.SecondOrderCone())
    JuMP.@objective(model, Min, so * t)
    return nothing
end
"""
    opt_weight_bounds(wf::JuMPWeightFinaliser, wb::WeightBounds, wi::VecNum) -> VecNum
    opt_weight_bounds(wf::IterativeWeightFinaliser, wb::WeightBounds, w::VecNum) -> VecNum

Move a weight vector into the bounds `wb`, keeping the budget it already carries.

The bounds themselves are not changed. Weights that already satisfy the bounds are returned unchanged, without a solve.

The [`JuMPWeightFinaliser`](@ref) method builds the programme of its `alg` (see [`set_clustering_weight_finaliser_alg!`](@ref)) and solves it. A failed solve warns and falls back to a default [`IterativeWeightFinaliser`](@ref). The [`IterativeWeightFinaliser`](@ref) method clips and redistributes instead, and may exhaust its passes with the bounds still violated.

# Arguments

  - `wf`: Weight finaliser algorithm.
  - `wb`: Weight bounds.
  - `wi`, `w`: The weights the optimisation produced.

# Returns

  - `w::VecNum`: The repaired weight vector.

# Related

  - [`WeightBounds`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
  - [`IterativeWeightFinaliser`](@ref)
  - [`finalise_weight_bounds`](@ref)
"""
function opt_weight_bounds(wf::JuMPWeightFinaliser, wb::WeightBounds, wi::VecNum)
    lb = wb.lb
    ub = wb.ub
    if !(!isnothing(lb) && any(map((x, y) -> x > y, lb, wi)) ||
         !isnothing(ub) && any(map((x, y) -> x < y, ub, wi)))
        return wi
    end
    model = JuMP.Model()
    JuMP.@expression(model, sc, wf.sc)
    JuMP.@expression(model, so, wf.so)
    JuMP.@variable(model, w[1:length(wi)])
    JuMP.@constraint(model, sc * (sum(w) - sum(wi)) == 0)
    if !isnothing(lb)
        JuMP.@constraint(model, sc * (w ⊖ lb) >= 0)
    end
    if !isnothing(ub)
        JuMP.@constraint(model, sc * (w ⊖ ub) <= 0)
    end
    set_clustering_weight_finaliser_alg!(model, wf.alg, wi)
    return if optimise_JuMP_model!(model, wf.slv).success
        JuMP.value.(get_w(model))
    else
        @warn("Version: $(wf.alg)\nReverting to Heuristic type.")
        opt_weight_bounds(IterativeWeightFinaliser(), wb, wi)
    end
end
function opt_weight_bounds(wf::IterativeWeightFinaliser, wb::WeightBounds, w::VecNum)
    lb = wb.lb
    ub = wb.ub
    if isnothing(lb)
        lb = typemin(eltype(w))
    end
    if isnothing(ub)
        ub = typemax(eltype(w))
    end
    if !(any(map((x, y) -> x > y, lb, w)) || any(map((x, y) -> x < y, ub, w)))
        return w
    end
    iter = wf.iter
    s1 = sum(w)
    for _ in 1:iter
        if !(any(map((x, y) -> x > y, lb, w)) || any(map((x, y) -> x < y, ub, w)))
            break
        end
        old_w = copy(w)
        w = max.(min.(w, ub), lb)
        idx = w .< ub .&& w .> lb
        w_add = sum(max.(old_w ⊖ ub, zero(eltype(w))))
        w_sub = sum(min.(old_w ⊖ lb, zero(eltype(w))))
        delta = w_add + w_sub
        if !iszero(delta)
            w[idx] += delta * w[idx] / sum(w[idx])
        end
        w *= s1 / sum(w)
    end
    return w
end
"""
    finalise_weight_bounds(wf::WeightFinaliser, wb::WeightBounds, w::VecNum)

Apply weight finalisation to enforce bounds and determine the optimisation return code.

Runs [`opt_weight_bounds`](@ref) with the given finaliser and bounds, then returns a success or failure return code based on whether all weights are finite.

Finiteness is the whole test. A vector that still violates the bounds — which an [`IterativeWeightFinaliser`](@ref) returns when it exhausts its passes — is reported as an [`OptimisationSuccess`](@ref).

# Arguments

  - `wf::WeightFinaliser`: Weight finaliser algorithm.
  - `wb::WeightBounds`: Weight bounds configuration.
  - `w::VecNum`: Portfolio weights to finalise.

# Returns

  - `(retcode, w)`: Tuple of return code and adjusted weights.

# Related

  - [`WeightFinaliser`](@ref)
  - [`WeightBounds`](@ref)
  - [`OptimisationSuccess`](@ref)
  - [`OptimisationFailure`](@ref)
"""
function finalise_weight_bounds(wf::WeightFinaliser, wb::WeightBounds, w::VecNum)
    w = opt_weight_bounds(wf, wb, w)
    retcode = if !any(!isfinite, w)
        OptimisationSuccess()
    else
        OptimisationFailure(; res = "Failure to set bounds\n$wf\n$wb.")
    end
    return retcode, w
end
"""
$(DocStringExtensions.TYPEDEF)

Indicates that a portfolio optimisation completed successfully.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OptimisationSuccess(; res = nothing) -> OptimisationSuccess

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> OptimisationSuccess()
OptimisationSuccess
  res ┴ nothing
```

# Related

  - [`OptimisationReturnCode`](@ref)
  - [`OptimisationFailure`](@ref)
"""
@concrete struct OptimisationSuccess <: OptimisationReturnCode
    """
    $(field_dict[:res_retcode])
    """
    res
end
function OptimisationSuccess(; res = nothing)
    return OptimisationSuccess(res)
end
"""
$(DocStringExtensions.TYPEDEF)

Indicates that a portfolio optimisation failed.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OptimisationFailure(; res = nothing) -> OptimisationFailure

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> OptimisationFailure()
OptimisationFailure
  res ┴ nothing
```

# Related

  - [`OptimisationReturnCode`](@ref)
  - [`OptimisationSuccess`](@ref)
"""
@concrete struct OptimisationFailure <: OptimisationReturnCode
    """
    $(field_dict[:res_retcode])
    """
    res
end
function OptimisationFailure(; res = nothing)
    return OptimisationFailure(res)
end
"""
    port_opt_view(opt, i, args...)

Return a view or subset of an optimisation estimator for a given cluster index `i`.

Default fallback returns the estimator unchanged. Overridden for composite estimators (e.g. [`JuMPOptimiser`](@ref), [`HierarchicalRiskParity`](@ref)) to slice all sub-estimators for the `i`-th cluster.

# Arguments

  - `opt`: Optimisation estimator or result.
  - `i`: Cluster or asset index.
  - `args...`: Additional arguments (e.g. asset returns matrix).

# Returns

  - Sliced or unchanged optimisation estimator.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`NestedClustered`](@ref)
"""
function port_opt_view(opt::AbstractOptimisationEstimator, ::Any, args...)
    return opt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

A precomputed optimisation result cannot be restricted to an asset subset.

Its weights were solved over the full universe and a sub-portfolio of them has no defined meaning, so an asset-subset view of a result throws. In particular, a [`TimeDependent`](@ref) schedule holding result entries is incompatible with asset-subsampling cross-validation ([`MultipleRandomised`](@ref)), whose fold loops view the optimiser to each fold's asset subset before the swap. The trivial all-assets view (`Colon`) passes the result through unchanged.

# Related

  - [`port_opt_view`](@ref)
  - [`TimeDependent`](@ref)
  - [`MultipleRandomised`](@ref)
"""
function port_opt_view(res::NonFiniteAllocationOptimisationResult, ::Colon, args...)
    return res
end
function port_opt_view(::NonFiniteAllocationOptimisationResult, ::Any, args...)
    return throw(ArgumentError("a precomputed optimisation result cannot be viewed to an asset subset: its weights were solved over the full universe and a sub-portfolio of them has no defined meaning. A TimeDependent schedule holding precomputed results is therefore incompatible with asset-subsampling cross-validation (e.g. MultipleRandomised); use estimator entries there instead."))
end
"""
    optimise(opt::OptimisationEstimator, args...; kwargs...) -> OptimisationResult
    optimise(opt::OptimisationResult, args...; kwargs...) -> OptimisationResult

Run portfolio optimisation using the given estimator `opt` and return an [`OptimisationResult`](@ref).

If `opt` returns an [`OptimisationFailure`](@ref), the fallback estimator is tried automatically until either a successful result is obtained or all fallbacks are exhausted.

Passing an [`OptimisationResult`](@ref) directly returns it unchanged (pass-through method).

# Arguments

  - `opt`: Optimisation estimator (e.g. a [`JuMPOptimisationEstimator`](@ref) subtype).
  - $(arg_dict[:ignargs])
  - $(arg_dict[:ignkwargs])

# Returns

  - [`OptimisationResult`](@ref): The optimisation result.

# Related

  - [`OptimisationEstimator`](@ref)
  - [`OptimisationResult`](@ref)
  - [`OptimisationSuccess`](@ref)
  - [`OptimisationFailure`](@ref)
"""
function optimise(opt::OptimisationResult, args...; kwargs...)
    return opt
end
"""
    _optimise(opt, rd, args...; dims, str_names, save, kwargs...)

Internal dispatch function for portfolio optimisation.

Called by [`optimise`](@ref) to perform the actual optimisation. Each optimisation estimator type implements its own overload. Returns the estimator-specific result type.

# Arguments

  - `opt`: Optimisation estimator (e.g. [`MeanRisk`](@ref), [`RiskBudgeting`](@ref), etc.).
  - `rd::ReturnsResult`: Returns data.
  - `dims::Int`: Observation dimension.
  - `str_names::Bool`: Whether to use string names in the JuMP model.
  - `save::Bool`: Whether to save the JuMP model in the result.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - Estimator-specific optimisation result.

# Related

  - [`optimise`](@ref)
  - [`MeanRisk`](@ref)
  - [`RiskBudgeting`](@ref)
  - [`NearOptimalCentering`](@ref)
"""
function _optimise end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

High level optimisation function that wraps around estimator-specific optimisation functions. This takes care of fallback methods if the primary optimisation fails. It returns the first successful optimisation result but stores all fallback results in the `fb` field of the result.

This is a fold-less entry point, so time-dependent schedules are inert here: the estimator is reset to its fold-less values (see [`reset_time_dependent_estimator`](@ref)) before the solve — in particular a scheduled fallback resets to its `default`, or to `nothing` (no fallback) when it has none, *before* the fallback chain is walked. Inside a fold loop this reset is a no-op, because the loop resolves every schedule before optimising.

# Arguments

  - `opt::OptimisationEstimator`: The optimisation estimator to use.
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
"""
function optimise(opt::OptimisationEstimator, args...; kwargs...)
    fb = Tuple{OptimisationEstimator, OptimisationResult}[]
    current_opt = reset_time_dependent_estimator(opt)
    res = nothing
    while true
        res = _optimise(current_opt, args...; kwargs...)
        if isa(res.retcode, OptimisationSuccess) || isnothing(current_opt.fb)
            break
        else
            push!(fb, (current_opt, res))
            current_opt = current_opt.fb
            @warn("Using fallback method. Please ignore previous optimisation failure warnings.")
        end
    end
    return isempty(fb) ? res : factory(res, fb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Optimise with a [`TimeDependent`](@ref) schedule standing in for the optimiser, outside any fold loop.

There are no folds to index, so the schedule resolves to its `default` and that optimiser runs (see [`reset_time_dependent_estimator`](@ref)); a schedule with no `default` throws a [`TimeDependentDefaultError`](@ref). Inside a fold loop this method is never reached — the loop resolves entry `i` first.

# Related

  - [`TD_OptE_Opt`](@ref)
  - [`reset_time_dependent_estimator`](@ref)
  - [`cross_val_predict`](@ref)
"""
function optimise(td::TD_OptE_Opt, args...; kwargs...)
    return optimise(reset_time_dependent_estimator(td), args...; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that `res` is a valid internal optimisation result.

Default no-op. Overridden for result types that must satisfy internal constraints before use.

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
function assert_internal_optimiser(::NonFiniteAllocationOptimisationResult)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that `res` is a valid external optimisation result.

Default no-op. Overridden for result types that must satisfy external interface constraints.

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
function assert_external_optimiser(::NonFiniteAllocationOptimisationResult)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Obtains the fees to use for net return calculations from an optimisation result.

An explicitly provided `fees` wins. Otherwise the fees are read from the `fees` property of `res`, and a result exposing no such property gives `nothing`.

# Arguments

  - `res`: Optimisation result, potentially containing a `fees` property.
  - `fees`: Optional fees to use, which take precedence over `res.fees` if provided.

# Returns

  - `Option{<:Fees}`: The fees to use for net return calculations, or `nothing` if not found.

# Related

  - [`calc_net_returns`](@ref)
  - [`OptimisationResult`](@ref)
  - [`Fees`](@ref)
"""
function extract_fees(res::OptimisationResult, fees::Option{<:Fees} = nothing)
    if isnothing(fees) && hasproperty(res, :fees)
        fees = res.fees
    end
    return fees
end
"""
    calc_net_returns(res::OptimisationResult, X::MatNum, fees = nothing, wd = nothing, obs = nothing)
    calc_net_returns(res::OptimisationResult, pr::Pr_RR, fees = nothing, wd = nothing, obs = nothing)

Compute net returns for a [`OptimisationResult`](@ref).

`fees` takes precedence over `res.fees` if both are provided. Delegates to [`calc_net_returns(w, X, fees, wd, obs)`](@ref).

When `pr::Pr_RR` is passed, extracts `X` from `pr.X` and delegates.

`wd` is the Weight Drift the window is read under. `nothing` reads the window at the constant weights `res.w`, which is the library's original behaviour. A [`SelfFinancingDrift`](@ref) reads it as the wealth ratio of the drifted holdings, and `obs` then names the observations of the message a non-positive wealth raises.

# Related

  - [`calc_net_returns`](@ref)
  - [`OptimisationResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`AbstractWeightDrift`](@ref)
  - [`SelfFinancingDrift`](@ref)
"""
function calc_net_returns(res::OptimisationResult, X::MatNum,
                          fees::Option{<:Fees} = nothing,
                          wd::Option{<:AbstractWeightDrift} = nothing, obs = nothing)
    fees = extract_fees(res, fees)
    return calc_net_returns(res.w, X, fees, wd, obs)
end
function calc_net_returns(res::OptimisationResult, pr::Pr_RR,
                          fees::Option{<:Fees} = nothing,
                          wd::Option{<:AbstractWeightDrift} = nothing, obs = nothing)
    return calc_net_returns(res, pr.X, fees, wd, obs)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Normalises inner weights into the convex weights that collapse real assets onto a meta-optimiser's synthetic assets.

Quantities carried alongside the returns matrix are either *extensive* (returns, benchmark returns) and collapse as a plain weighted sum `w'x`, or *intensive* (rates such as `rd.iv` and `rd.ivpa`) and collapse as a weighted *average*. A plain weighted sum scales an intensive quantity by the gross exposure `sⱼ = Σᵢ|wᵢⱼ|`, so a shorting or leveraged portfolio (`sⱼ ≠ 1`) inflates a rate that should not depend on gross exposure at all.

Normalising the weights once here makes every subsequent product a convex combination, so callers collapsing an intensive quantity need only pass their weights through this function.

# Arguments

  - `w`: Inner weights. A vector collapses onto a single synthetic asset; a matrix (assets × synthetic assets) collapses each column independently.

# Returns

  - `w`: `abs.(w)`, with each column scaled to sum to one. A column summing to zero — a degenerate synthetic asset — is left as-is rather than divided, preserving the zero row it already produced.

# Related

  - [`prepare_outer_rd`](@ref)
  - [`reconstruct_rd`](@ref)
"""
function synthetic_asset_weights(w::VecNum)
    w = abs.(w)
    s = sum(w)
    return iszero(s) ? w : w / s
end
function synthetic_asset_weights(w::MatNum)
    w = abs.(w)
    s = sum(w; dims = 1)
    return w ./ map(x -> iszero(x) ? one(x) : x, s)
end
"""
    collapse_feature_matrix(Z::Nothing, sq::Bool, wi::MatNum)
    collapse_feature_matrix(Z::MatNum, sq::Bool, wi::MatNum)
    collapse_feature_matrix(Z::Arr3Num, sq::Bool, wi::MatNum)
    collapse_feature_matrix(Z::Nothing, w::VecNum)
    collapse_feature_matrix(Z::MatNum, w::VecNum)
    collapse_feature_matrix(Z::Arr3Num, w::VecNum)

Aggregate a feature matrix onto the synthetic assets a meta-optimiser builds for its outer problem.

A meta-optimiser's outer problem allocates across *synthetic* assets — [`NestedClustered`](@ref)'s clusters, [`Stacking`](@ref)'s inner portfolios — each of which is a weighted combination of the real ones. Every quantity the outer [`ReturnsResult`](@ref) carries has to be re-expressed on that universe, and a feature matrix is no exception: without this collapse the outer optimiser has no feature matrix at all, so a [`FeatureDistance`](@ref) there throws rather than clustering the synthetic universe.

Features are treated as **intensive**, exactly as `iv` and `ivpa` are: the collapse is a convex combination, obtained by pushing the inner weights through [`synthetic_asset_weights`](@ref) first. An un-normalised weighted sum would scale each synthetic asset's feature vector by its gross exposure `sⱼ = Σᵢ|wᵢⱼ|`, inflating it under leverage or shorting. Under the default [`AngularDist`](@ref) the normalisation is a mathematical no-op for a rectangular feature matrix — scaling one row of the result leaves every cosine unchanged — but it is *not* one in the square case, where the two-sided product rescales feature columns as well, and it is what keeps the collapse bounded for any `sⱼ > 0`. An extensive feature (a market capitalisation, a headcount) wanting a weighted *sum* is not supported: the divisor depends on the inner solve, so a caller cannot pre-scale their way to one.

## The two weight shapes

  - A weight **matrix** `wi` (assets × synthetic assets) collapses the whole universe at once. When `sq` is `true` the feature axis *is* the asset axis ([`features_are_assets`](@ref)), so it is contracted too and the result is again square on the synthetic universe.
  - A weight **vector** `w` collapses onto a *single* synthetic asset, which is all [`reconstruct_rd`](@ref) has in scope within a cross-validation fold. It takes no `sq` argument, and that absence is the statement: the second contraction of the square case needs every synthetic asset's weights simultaneously, and contracting the feature axis with the one vector available would collapse it to a single number per synthetic asset — a feature space in which every asset is trivially identical. A square feature matrix therefore keeps the real assets as its feature axis through the fold path, reading as "this synthetic asset's weighted-average neighbourhood".

## Degenerate synthetic assets

A synthetic asset whose weights are entirely zero has `sⱼ = 0`; [`synthetic_asset_weights`](@ref) leaves the column alone rather than dividing, so the collapse gives that asset a **zero feature vector** instead of throwing. It then lands on the zero-feature-vector convention the distance kernel already implements, matching the zero returns column, `iv` and `ivpa` the same degenerate weights already produce.

# Arguments

  - `Z`: Feature matrix, static (assets × features) or time-varying (observations × assets × features).
  - `sq`: Whether the feature axis is the asset axis, from [`features_are_assets`](@ref).
  - `wi`: Inner weights, assets × synthetic assets.
  - `w`: Inner weights for a single synthetic asset, assets × 1.

# Returns

  - `nothing` when `Z` is `nothing`.
  - Matrix arity: `synthetic assets × features`, or `synthetic assets × synthetic assets` when `sq`; `observations × …` with the same trailing axes for a time-varying `Z`.
  - Vector arity: a `features`-length vector for a static `Z`, an `observations × features` matrix for a time-varying one.

# Related

  - [`synthetic_asset_weights`](@ref)
  - [`features_are_assets`](@ref)
  - [`prepare_outer_rd`](@ref)
  - [`reconstruct_rd`](@ref)
  - [`FeatureDistance`](@ref)
"""
function collapse_feature_matrix(::Nothing, ::Bool, ::MatNum)
    return nothing
end
function collapse_feature_matrix(Z::MatNum, sq::Bool, wi::MatNum)
    wi = synthetic_asset_weights(wi)
    Zc = transpose(wi) * Z
    return sq ? Zc * wi : Zc
end
function collapse_feature_matrix(Z::Arr3Num, sq::Bool, wi::MatNum)
    wi = synthetic_asset_weights(wi)
    k = size(wi, 2)
    nf = sq ? k : size(Z, 3)
    Zc = Array{promote_type(eltype(Z), eltype(wi))}(undef, size(Z, 1), k, nf)
    @inbounds for t in axes(Z, 1)
        Zt = transpose(wi) * view(Z, t, :, :)
        Zc[t, :, :] = sq ? Zt * wi : Zt
    end
    return Zc
end
function collapse_feature_matrix(::Nothing, ::VecNum)
    return nothing
end
function collapse_feature_matrix(Z::MatNum, w::VecNum)
    return transpose(Z) * synthetic_asset_weights(w)
end
function collapse_feature_matrix(Z::Arr3Num, w::VecNum)
    w = synthetic_asset_weights(w)
    Zc = Matrix{promote_type(eltype(Z), eltype(w))}(undef, size(Z, 1), size(Z, 3))
    @inbounds for t in axes(Z, 1)
        Zc[t, :] = transpose(view(Z, t, :, :)) * w
    end
    return Zc
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Extracts the prior result for risk calculation from an optimisation result.

An explicitly provided `pr` wins. Otherwise the one test is `hasproperty(res, :pr)`, which property forwarding answers for a nested result: a JuMP leaf reaches its prior at `res.jr.pa.pr`, and `res.pr` resolves to it. A result that exposes no `pr` property throws.

# Arguments

  - `res`: Optimisation result, which carries a prior result as its `pr` property or reaches one by property forwarding.
  - `pr`: Optional prior result to use for risk calculation, which takes precedence over the one found in `res`.

# Returns

  - `pr::Pr_RR`: The prior result to use for risk calculation. Throws an `ArgumentError` when none is found.

# Related

  - [`expected_risk`](@ref)
  - [`extract_fees`](@ref)
  - [`OptimisationResult`](@ref)
"""
function extract_pr(res::OptimisationResult, pr::Option{<:Pr_RR} = nothing)
    return if !isnothing(pr)
        pr
    elseif hasproperty(res, :pr)
        res.pr
    else
        throw(ArgumentError("`$(nameof(typeof(res)))` exposes no `.pr` property, directly or through property forwarding; provide `pr` explicitly"))
    end
end
"""
    expected_risk(r::BaseRM_VecBaseRM, res::OptimisationResult, X::MatNum, fees = nothing; kwargs...)
    expected_risk(r::BaseRM_VecBaseRM, res::OptimisationResult, pr = nothing, fees = nothing; kwargs...)

Compute the expected risk for an [`OptimisationResult`](@ref).

Extracts `w` from `res` and delegates to the weight-based [`expected_risk`](@ref). `fees` takes precedence over `res.fees` if both are provided.

When `pr::Pr_RR` is `nothing`, [`extract_pr`](@ref) takes the prior result from the `pr` property of `res` before delegating.

`r` is one measure or a vector of them; a vector is scalarised by `sca`, defaulting to [`SumScalariser`](@ref). The measure is **not** read from `res`, so a result that carries its own `r` and `sca` reports the figure it optimised only when the caller passes them back, as `expected_risk(res.r, res; sca = res.sca)`.

The prior-taking method forwards the carrier whole, so a caller's **own** measure resolves against it: an unstated slot takes the prior's field and a **Deferred Quantity** is fitted, exactly as in `expected_risk(r, w, pr)`. Pass a matrix instead to opt out and supply every slot yourself.

# Related

  - [`expected_risk`](@ref)
  - [`OptimisationResult`](@ref)
  - [`BaseRM_VecBaseRM`](@ref)
  - [`resolve_risk_inputs`](@ref)
"""
function expected_risk(r::BaseRM_VecBaseRM, res::OptimisationResult, X::MatNum,
                       fees::Option{<:Fees} = nothing; kwargs...)
    fees = extract_fees(res, fees)
    return expected_risk(r, res.w, X, fees; kwargs...)
end
function expected_risk(r::BaseRM_VecBaseRM, res::OptimisationResult,
                       pr::Option{<:Pr_RR} = nothing, fees::Option{<:Fees} = nothing;
                       kwargs...)
    # The carrier is forwarded whole, never unwrapped to `pr.X`. `expected_risk`'s own
    # `Pr_RR` route resolves the measure through `resolve_risk_inputs`, which has an arm for
    # each carrier: a prior result resolves the measure, a `ReturnsResult` only unwraps `X`.
    # Unwrapping here dropped the prior fallback, so `expected_risk(Variance(), res)` — the
    # call this docstring asks callers to make — hit the kernel with an unstated `sigma`.
    fees = extract_fees(res, fees)
    return expected_risk(r, res.w, extract_pr(res, pr), fees; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `false`.

`nothing` never requires previous portfolio weights.

# Related

  - [`needs_previous_weights`](@ref)
"""
function needs_previous_weights(::Option{<:Union{<:AbstractEstimator, <:AbstractAlgorithm,
                                                 <: AbstractResult}})
    return false
end

export optimise, OptimisationSuccess, OptimisationFailure, IterativeWeightFinaliser,
       RelativeErrorWeightFinaliser, SquaredRelativeErrorWeightFinaliser,
       AbsoluteErrorWeightFinaliser, SquaredAbsoluteErrorWeightFinaliser,
       JuMPWeightFinaliser, TimeDependent, TimeDependentContext, PreviousWeightsFunction,
       NoDefault, TimeDependentDefaultError
