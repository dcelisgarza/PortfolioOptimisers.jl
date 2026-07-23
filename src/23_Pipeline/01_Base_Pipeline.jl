"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all pipeline estimator types.

A pipeline reifies an end-to-end workflow — price preprocessing, prices-to-returns conversion, returns preprocessing, prior estimation, phylogeny, uncertainty sets, constraint generation, and optimisation — as an ordered list of steps executed left-to-right over a [`PipelineContext`](@ref). Pipelines widen the cross-validation and hyperparameter-tuning boundary to the entire workflow, data preparation included.

All concrete pipeline estimators should subtype `AbstractPipelineEstimator`.

See `docs/adr/0028-pipeline-workflow-estimator.md` for the design rationale.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractPipelineResult`](@ref)
  - [`PipelineContext`](@ref)
  - [`PipelineStep`](@ref)
"""
abstract type AbstractPipelineEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all pipeline result types.

Pipeline results carry the fitted per-step results of executing an [`AbstractPipelineEstimator`](@ref), the final [`PipelineContext`](@ref), and the terminal optimisation result when one exists.

All concrete pipeline results should subtype `AbstractPipelineResult`.

# Related

  - [`AbstractResult`](@ref)
  - [`AbstractPipelineEstimator`](@ref)
"""
abstract type AbstractPipelineResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

The mu/sigma pair held by the `uncertainty` slot of a [`PipelineContext`](@ref).

A computed uncertainty-set result cannot always reveal which parameter it bounds (a `BoxUncertaintySet` may bound either the mean or the covariance), so the slot stores the two targets explicitly. Uncertainty-set steps declare their target through a [`PipelineStep`](@ref) wrapper (`target = :mu`, `target = :sigma`, or `target = :both`); a narrowed step fills its half of the pair and leaves the other untouched, so `:mu` and `:sigma` steps compose, while `:both` derives the two halves from a single [`ucs`](@ref) call.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PipelineUncertaintySets(;
        mu::Option{<:AbstractUncertaintySetResult} = nothing,
        sigma::Option{<:AbstractUncertaintySetResult} = nothing,
    ) -> PipelineUncertaintySets

Keywords correspond to the struct's fields.

# Related

  - [`PipelineContext`](@ref)
  - [`PipelineStep`](@ref)
"""
@concrete struct PipelineUncertaintySets <: AbstractResult
    """
    Uncertainty set bounding the expected returns vector ([`AbstractUncertaintySetResult`](@ref)).
    """
    mu
    """
    Uncertainty set bounding the covariance matrix ([`AbstractUncertaintySetResult`](@ref)).
    """
    sigma
    function PipelineUncertaintySets(mu::Option{<:AbstractUncertaintySetResult},
                                     sigma::Option{<:AbstractUncertaintySetResult})
        return new{typeof(mu), typeof(sigma)}(mu, sigma)
    end
end
function PipelineUncertaintySets(; mu::Option{<:AbstractUncertaintySetResult} = nothing,
                                 sigma::Option{<:AbstractUncertaintySetResult} = nothing)::PipelineUncertaintySets
    return PipelineUncertaintySets(mu, sigma)
end
"""
    const PIPELINE_SLOTS = (:prices, :returns, :prior, :phylogeny, :uncertainty, :constraints, :opt)

The named slots of a [`PipelineContext`](@ref). Each pipeline step reads the slots it needs and writes the slot its estimator family produces.
"""
const PIPELINE_SLOTS = (:prices, :returns, :prior, :phylogeny, :uncertainty, :constraints,
                        :opt)
"""
    const PIPELINE_DATA_SLOTS = (:prices, :returns)

The [`PIPELINE_SLOTS`](@ref) whose write *changes the asset universe* — equivalently, the two slots a [`Pipeline`](@ref) input can fill directly. Writing one of these makes every slot *derived* from it stale, which is exactly what [`PIPELINE_INVALIDATES`](@ref) is derived from. Every other slot is computed from the data and reorders nothing, so writing it invalidates nothing.

# Related

  - [`PIPELINE_SLOTS`](@ref)
  - [`PIPELINE_INVALIDATES`](@ref)
  - [`PipelineContext`](@ref)
"""
const PIPELINE_DATA_SLOTS = (:prices, :returns)
"""
    PIPELINE_INVALIDATES

The [`PipelineContext`](@ref) slots each written slot invalidates, derived from [`PIPELINE_SLOTS`](@ref) order and [`PIPELINE_DATA_SLOTS`](@ref):

```julia
(prices = (:returns, :prior, :phylogeny, :uncertainty, :constraints),
 returns = (:prior, :phylogeny, :uncertainty, :constraints))
```

Writing a [data slot](@ref PIPELINE_DATA_SLOTS) makes every slot *derived* from that data stale: a prior, phylogeny, uncertainty set, or constraint result computed on one asset universe does not match a later, different one. [`Pipeline`](@ref) rejects such an ordering at construction rather than letting a stale, asset-misdimensioned result reach [`inject_context`](@ref).

The derivation: only a data slot invalidates, and it invalidates every slot after it in [`PIPELINE_SLOTS`](@ref) *except* the terminal `:opt`. `:opt` is the workflow's output — nothing derives from it, so a stale `:opt` is never read by a later step; it is excluded from the invalidatable set by construction. A slot filled by the pipeline *input* rather than by a step is not "written", so the usual `MissingDataFilter → Imputer → PricesToReturns → …` ordering is unaffected, and a non-data write (`prior`, `phylogeny`, `uncertainty`, `constraints`, `opt`) invalidates nothing.

# Related

  - [`PIPELINE_SLOTS`](@ref)
  - [`PIPELINE_DATA_SLOTS`](@ref)
  - [`Pipeline`](@ref)
  - [`PipelineContext`](@ref)
"""
const PIPELINE_INVALIDATES = let slots = PIPELINE_SLOTS, terminal = last(PIPELINE_SLOTS)
    idx(x) = findfirst(==(x), slots)
    NamedTuple{PIPELINE_DATA_SLOTS}(map(PIPELINE_DATA_SLOTS) do s
                                        return Tuple(x
                                                     for x in slots
                                                     if x != terminal && idx(x) > idx(s))
                                    end)
end
"""
    const PIPELINE_ROUTING_TARGETS = (:pe, :cle, :wb, :lcse, :ple, :mu_ucs, :sigma_ucs)

The destinations [`inject_context`](@ref) can deliver a computed slot to.

Routing targets are finer than [`PIPELINE_SLOTS`](@ref) and address a different audience. A slot is *pipeline-author* vocabulary: it names a stage of the workflow, and a step's estimator family decides which slot it writes. A target is *optimiser-author* vocabulary: it names a destination, and an optimiser's fields decide which targets it accepts. Users writing pipelines never name a target.

The fan-out from slots to targets is the [`Pipeline`](@ref)'s job, and it is where the slot-level heterogeneity is resolved:

  - `prior` → `:pe`.
  - `phylogeny` → `:cle`, when the result is a clustering structure.
  - `uncertainty` → `:mu_ucs` and `:sigma_ucs`, one per populated half of the [`PipelineUncertaintySets`](@ref) pair.
  - `constraints` → `:wb`, `:lcse` or `:ple`, per element, by result type; multiple `:lcse`/`:ple` elements are packed into a vector.

Five of the seven are named after the field they land in, using this package's shared field vocabulary (see `field_dict`), so [`pipe_route`](@ref) needs no per-optimiser declaration — the target lands in the like-named field of whichever optimiser has one. `:mu_ucs` and `:sigma_ucs` are the exceptions: they carry validation policy and name no plain field.

Note that `:cle` and `:ple` are one letter apart and come from the *same* `phylogeny` slot; they are not interchangeable. `:cle` is a clustering structure the optimiser uses to build a hierarchy, `:ple` is a phylogeny *constraint* result. That is why only one of them is optional below.

Internal machinery — not part of the user-facing API.

# Related

  - [`PIPELINE_SLOTS`](@ref)
  - [`PIPELINE_OPTIONAL_TARGETS`](@ref)
  - [`pipe_route`](@ref)
  - [`inject_context`](@ref)
"""
const PIPELINE_ROUTING_TARGETS = (:pe, :cle, :wb, :lcse, :ple, :mu_ucs, :sigma_ucs)
"""
    const PIPELINE_OPTIONAL_TARGETS = (:pe, :cle)

The [routing targets](@ref PIPELINE_ROUTING_TARGETS) an optimiser may have no home for without it being an error.

The asymmetry is the whole of the injection policy, and it turns on whether dropping the value changes the *answer*:

  - `:pe` and `:cle` do not. An optimiser with no `pe` field either needs no prior ([`EqualWeighted`](@ref)) or computes an equivalent one internally, which is what ADR 0028 means by every stage being optional. A [`JuMPOptimiser`](@ref) has no `cle` field because phylogeny reaches it as constraint results — generated from returns, not from this slot — so the structure is genuinely surplus to it.
  - Everything else does. A weight bound, linear constraint, phylogeny constraint or uncertainty set that reaches no optimiser field would silently change the solved portfolio, so it is an error — the same reason [`PipelineStep`](@ref)'s `target` rejects an uncertainty half it cannot place.

Note the cost that *is* paid: no step reads the `prior` or `phylogeny` slots — [`inject_context`](@ref) is their only consumer — so a step writing a slot the terminal optimiser cannot receive is wasted computation, silently. That is a performance trap rather than a correctness one, which is why it is tolerated rather than rejected, but it is the reason to keep this list short.

Internal machinery — not part of the user-facing API.

# Related

  - [`PIPELINE_ROUTING_TARGETS`](@ref)
  - [`unroutable_target`](@ref)
"""
const PIPELINE_OPTIONAL_TARGETS = (:pe, :cle)
function unroutable_target(x, ::Val{target}, _) where {target}
    if target in PIPELINE_OPTIONAL_TARGETS
        return x
    end
    return throw(ArgumentError("cannot route the :$target pipeline target into a $(Base.typename(typeof(x)).wrapper): it has no :$target field to receive it"))
end
"""
$(DocStringExtensions.TYPEDEF)

The accumulating blackboard threaded through a pipeline's steps.

A `PipelineContext` holds one typed slot per stage of the workflow. Steps run in user-given order; each reads the slots it needs and writes the slot its estimator family produces. Heterogeneous slots (`uncertainty`, `constraints`) hold one result or a vector of results whose elements are routed to their optimiser targets by result type.

Internal machinery — not part of the user-facing API.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PipelineContext(;
        prices::Option{<:AbstractPricesResult} = nothing,
        returns::Option{<:AbstractReturnsResult} = nothing,
        prior::Option{<:AbstractPriorResult} = nothing,
        phylogeny::Option{<:AbstractPhylogenyResult} = nothing,
        uncertainty::Option{<:PipelineUncertaintySets} = nothing,
        constraints::Option{<:Union{<:AbstractConstraintResult, <:AbstractVector{<:AbstractConstraintResult}}} = nothing,
        opt::Option{<:OptimisationResult} = nothing,
    ) -> PipelineContext

Keywords correspond to the struct's fields.

# Related

  - [`PIPELINE_SLOTS`](@ref)
  - [`AbstractPipelineEstimator`](@ref)
  - [`PipelineStep`](@ref)
"""
@concrete struct PipelineContext <: AbstractResult
    """
    Price-level data ([`AbstractPricesResult`](@ref)).
    """
    prices
    """
    Returns-level data ([`AbstractReturnsResult`](@ref)).
    """
    returns
    """
    Prior statistics ([`AbstractPriorResult`](@ref)).
    """
    prior
    """
    Phylogeny structure ([`AbstractPhylogenyResult`](@ref)).
    """
    phylogeny
    """
    Uncertainty set results as an explicit mu/sigma pair ([`PipelineUncertaintySets`](@ref)).
    """
    uncertainty
    """
    Constraint result(s) ([`AbstractConstraintResult`](@ref) or a vector thereof).
    """
    constraints
    """
    Terminal optimisation result ([`OptimisationResult`](@ref)).
    """
    opt
    function PipelineContext(prices::Option{<:AbstractPricesResult},
                             returns::Option{<:AbstractReturnsResult},
                             prior::Option{<:AbstractPriorResult},
                             phylogeny::Option{<:AbstractPhylogenyResult},
                             uncertainty::Option{<:PipelineUncertaintySets},
                             constraints::Option{<:Union{<:AbstractConstraintResult,
                                                         <:AbstractVector{<:AbstractConstraintResult}}},
                             opt::Option{<:OptimisationResult})
        return new{typeof(prices), typeof(returns), typeof(prior), typeof(phylogeny),
                   typeof(uncertainty), typeof(constraints), typeof(opt)}(prices, returns,
                                                                          prior, phylogeny,
                                                                          uncertainty,
                                                                          constraints, opt)
    end
end
function PipelineContext(; prices::Option{<:AbstractPricesResult} = nothing,
                         returns::Option{<:AbstractReturnsResult} = nothing,
                         prior::Option{<:AbstractPriorResult} = nothing,
                         phylogeny::Option{<:AbstractPhylogenyResult} = nothing,
                         uncertainty::Option{<:PipelineUncertaintySets} = nothing,
                         constraints::Option{<:Union{<:AbstractConstraintResult,
                                                     <:AbstractVector{<:AbstractConstraintResult}}} = nothing,
                         opt::Option{<:OptimisationResult} = nothing)::PipelineContext
    return PipelineContext(prices, returns, prior, phylogeny, uncertainty, constraints, opt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the [`PipelineContext`](@ref) slot a pipeline step writes.

The estimator's family determines the slot via dispatch on the existing abstract-type taxonomy. Estimators whose family cannot be inferred must be wrapped in a [`PipelineStep`](@ref); the fallback method throws an `ArgumentError` saying so.

# Arguments

  - `est`: The step estimator.

# Returns

  - `slot::Symbol`: One of [`PIPELINE_SLOTS`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.pipe_writes(EmpiricalPrior())
:prior
```

# Related

  - [`pipe_reads`](@ref)
  - [`PIPELINE_SLOTS`](@ref)
  - [`PipelineStep`](@ref)
"""
function pipe_writes(est)
    return throw(ArgumentError("cannot infer the pipeline slot written by a $(typeof(est)); wrap it in a PipelineStep to declare its reads/writes explicitly"))
end
pipe_writes(::AbstractPricesPreprocessingEstimator) = :prices
pipe_writes(::AbstractReturnsPreprocessingEstimator) = :returns
pipe_writes(::AbstractPriorEstimator) = :prior
pipe_writes(::AbstractPhylogenyEstimator) = :phylogeny
pipe_writes(::AbstractUncertaintySetEstimator) = :uncertainty
pipe_writes(::AbstractConstraintEstimator) = :constraints
pipe_writes(::OptimisationEstimator) = :opt
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the [`PipelineContext`](@ref) slots a pipeline step requires to be populated before it runs.

These are the *required* reads used for construction-time dependency validation, not every slot the step may consume. Slots an estimator can compute internally when absent (for example a phylogeny-constraint estimator's own phylogeny) are not listed.

# Arguments

  - `est`: The step estimator.

# Returns

  - `slots::Tuple{Vararg{Symbol}}`: Subset of [`PIPELINE_SLOTS`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.pipe_reads(EmpiricalPrior())
(:returns,)
```

# Related

  - [`pipe_writes`](@ref)
  - [`PIPELINE_SLOTS`](@ref)
  - [`PipelineStep`](@ref)
"""
function pipe_reads(est)
    return throw(ArgumentError("cannot infer the pipeline slots read by a $(typeof(est)); wrap it in a PipelineStep to declare its reads/writes explicitly"))
end
pipe_reads(::AbstractPricesPreprocessingEstimator) = (:prices,)
pipe_reads(::AbstractReturnsPreprocessingEstimator) = (:returns,)
pipe_reads(::AbstractPriorEstimator) = (:returns,)
pipe_reads(::AbstractPhylogenyEstimator) = (:returns,)
pipe_reads(::AbstractUncertaintySetEstimator) = (:returns,)
pipe_reads(::AbstractConstraintEstimator) = (:returns,)
pipe_reads(::OptimisationEstimator) = (:returns,)
"""
$(DocStringExtensions.TYPEDEF)

Explicit pipeline step wrapper — used when a step's slots or its routing intent must be stated rather than inferred.

Most estimators are used as pipeline steps directly: their family determines which [`PipelineContext`](@ref) slots they read and write via [`pipe_reads`](@ref)/[`pipe_writes`](@ref). `PipelineStep` covers the two cases that dispatch alone cannot settle:

  - **Slots dispatch cannot infer**: a custom callable, or an estimator routed to a nonstandard slot. `reads` and `writes` supply what the family would otherwise declare. This includes a bare-callable [`TimeDependent`](@ref) schedule of optimisers (`TimeDependent(ctx -> optimiser)`), whose output kind is not in its type: it enters via `PipelineStep(; est = td, writes = :opt)` and its output is type-checked when the fold loop swaps it in (see [`TD_OptE_Opt_Inferable`](@ref)).
  - **Routing intent dispatch must not guess**: an uncertainty-set estimator writes the `uncertainty` slot either way, so the slot is never in doubt; what the wrapper declares through `target` is *which parameters you want bounded* — `:mu`, `:sigma`, or `:both`. Since [`ucs`](@ref) derives both halves from a single fit, this is a statement of intent, not a disambiguation, and every populated half must reach the optimiser or [`inject_context`](@ref) rejects it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PipelineStep(;
        est,
        writes::Symbol,
        reads::Tuple{Vararg{Symbol}} = (),
        target::Option{Symbol} = nothing,
    ) -> PipelineStep

Keywords correspond to the struct's fields.

## Validation

  - `writes in PIPELINE_SLOTS`.
  - `all(r -> r in PIPELINE_SLOTS, reads)`.
  - A [`TimeDependent`](@ref) `est` must be an optimiser-position schedule ([`TD_OptE_Opt`](@ref)) and declare `writes = :opt`: schedules of non-optimiser families are not steppable — a per-fold prior/constraint/… is spelled as a `TimeDependent` *field* of the optimisation step instead.

# Examples

```jldoctest
julia> ps = PipelineStep(; est = NormalUncertaintySet(), reads = (:returns,),
                         writes = :uncertainty, target = :mu);

julia> PortfolioOptimisers.pipe_writes(ps)
:uncertainty

julia> PortfolioOptimisers.pipe_reads(ps)
(:returns,)
```

# Related

  - [`pipe_reads`](@ref)
  - [`pipe_writes`](@ref)
  - [`PIPELINE_SLOTS`](@ref)
  - [`AbstractPipelineEstimator`](@ref)
"""
@concrete struct PipelineStep <: AbstractEstimator
    """
    The wrapped step: an estimator or a callable.
    """
    est
    """
    Slot the step requires to be populated before it runs (subset of [`PIPELINE_SLOTS`](@ref)).
    """
    reads
    """
    Slot the step writes (one of [`PIPELINE_SLOTS`](@ref)).
    """
    writes
    """
    Optional routing annotation for heterogeneous slots (for uncertainty sets: `:mu`, `:sigma`, or `:both`).
    """
    target
    function PipelineStep(est::Union{<:AbstractEstimator, <:Function},
                          reads::Tuple{Vararg{Symbol}}, writes::Symbol,
                          target::Option{Symbol})
        @argcheck(writes in PIPELINE_SLOTS,
                  ArgumentError("writes must be one of $(PIPELINE_SLOTS), got :$writes"))
        @argcheck(all(r -> r in PIPELINE_SLOTS, reads),
                  ArgumentError("all reads must be members of $(PIPELINE_SLOTS), got $reads"))
        if isa(est, TimeDependent)
            @argcheck(isa(est, TD_OptE_Opt),
                      ArgumentError("a TimeDependent schedule is only steppable when it stands in for the optimiser (an optimiser-position schedule, see TD_OptE_Opt); schedules of non-optimiser families are not pipeline steps. To vary a prior, constraint, or other input per fold, put a TimeDependent in the corresponding field of the optimisation step instead."))
            @argcheck(writes === :opt,
                      ArgumentError("a TimeDependent schedule step stands in for the optimiser, so it must declare writes = :opt, got :$writes"))
        end
        return new{typeof(est), typeof(reads), typeof(writes), typeof(target)}(est, reads,
                                                                               writes,
                                                                               target)
    end
end
function PipelineStep(; est::Union{<:AbstractEstimator, <:Function}, writes::Symbol,
                      reads::Tuple{Vararg{Symbol}} = (),
                      target::Option{Symbol} = nothing)::PipelineStep
    return PipelineStep(est, reads, writes, target)
end
pipe_writes(ps::PipelineStep) = ps.writes
pipe_reads(ps::PipelineStep) = ps.reads
"""
    const TD_OptE_Opt_Inferable = Union{TimeDependent{<:AbstractVector{<:OptE_Opt}},
                                        TimeDependent{<:TimeDependentOptimiserCallable}}

The [`TimeDependent`](@ref) optimiser-position schedule forms whose pipeline slot is inferable from their type: a vector schedule whose entries are all optimisers or precomputed results, and a declared [`TimeDependentOptimiserCallable`](@ref) functor. Both are optimisation steps, so they write `:opt` and read `:returns` like any [`OptimisationEstimator`](@ref) step.

The other two forms of [`TD_OptE_Opt`](@ref) — a bare `ctx -> optimiser` and a [`PreviousWeightsFunction`](@ref) wrapping one — declare nothing in their type, so they keep the cannot-infer throw of [`pipe_writes`](@ref) and enter via `PipelineStep(; est = td, writes = :opt)`, their output type-checked when the fold loop swaps it in (see [`assert_time_dependent_optimiser`](@ref)).

# Related

  - [`TD_OptE_Opt`](@ref)
  - [`pipe_writes`](@ref)
  - [`PipelineStep`](@ref)
"""
const TD_OptE_Opt_Inferable = Union{TimeDependent{<:AbstractVector{<:OptE_Opt}},
                                    TimeDependent{<:TimeDependentOptimiserCallable}}
pipe_writes(::TD_OptE_Opt_Inferable) = :opt
pipe_reads(::TD_OptE_Opt_Inferable) = (:returns,)

export PipelineStep
