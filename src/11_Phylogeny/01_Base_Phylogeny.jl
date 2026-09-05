"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all phylogeny estimator types.

All concrete and/or abstract types implementing phylogeny-based estimation algorithms should be subtypes of `AbstractPhylogenyEstimator`.

# Related

  - [`AbstractPhylogenyAlgorithm`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
"""
abstract type AbstractPhylogenyEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all phylogeny algorithm types.

All concrete and/or abstract types implementing specific phylogeny algorithms should be subtypes of `AbstractPhylogenyAlgorithm`.

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
"""
abstract type AbstractPhylogenyAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all phylogeny result types.

All concrete and/or abstract types representing the result of a phylogeny estimation should be subtypes of `AbstractPhylogenyResult`.

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyAlgorithm`](@ref)
"""
abstract type AbstractPhylogenyResult <: AbstractResult end
"""
    const PlE_Pl = Union{<:AbstractPhylogenyEstimator, <:AbstractPhylogenyResult}

Alias for a phylogeny estimator or result.

Matches either an [`AbstractPhylogenyEstimator`](@ref) or an [`AbstractPhylogenyResult`](@ref). Used internally for dispatch when either a phylogeny estimation configuration or pre-computed result is accepted.

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
"""
const PlE_Pl = Union{<:AbstractPhylogenyEstimator, <:AbstractPhylogenyResult}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the phylogeny estimator or result `pl` unchanged.

Identity pass-through used when a phylogeny estimator or pre-computed result is provided in a context that calls [`factory`](@ref). A phylogeny estimator carries no prior-dependent field to rebuild, so every field is passed through and none is replaced.

# Algorithm

 1. Return `pl` itself. No field of it is rebuilt, and `args` and `kwargs` are discarded.

# Arguments

  - `pl`: Phylogeny estimator or phylogeny result.
  - `args...`: Optional arguments (ignored).
  - `kwargs...`: Optional keyword arguments (ignored).

# Returns

  - `pl::PlE_Pl`: The original phylogeny estimator or result.

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
  - [`factory`](@ref)
"""
function factory(pl::PlE_Pl, args...; kwargs...)::PlE_Pl
    return pl
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the phylogeny algorithm `alg` unchanged.

Identity pass-through used when a phylogeny algorithm is provided in a context that calls [`factory`](@ref). A phylogeny algorithm carries no prior-dependent field to rebuild, so every field is passed through and none is replaced.

# Algorithm

 1. Return `alg` itself. No field of it is rebuilt, and `args` and `kwargs` are discarded.

# Arguments

  - `alg`: Phylogeny algorithm.
  - `args...`: Optional arguments (ignored).
  - `kwargs...`: Optional keyword arguments (ignored).

# Returns

  - `alg::AbstractPhylogenyAlgorithm`: The original phylogeny algorithm.

# Related

  - [`AbstractPhylogenyAlgorithm`](@ref)
  - [`factory`](@ref)
"""
function factory(alg::AbstractPhylogenyAlgorithm, args...;
                 kwargs...)::AbstractPhylogenyAlgorithm
    return alg
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all separation algorithms.

A separation algorithm is the rule saying **how far apart** two assets sit in a network, and **how far is too far**. It answers two questions with one object, through three kernels: [`separation_graph`](@ref) builds the structure the member measures over, [`separation_matrix`](@ref) reads the dense `assets × assets` separations off that structure, and [`separation_budget`](@ref) resolves the budget beyond which a pair counts as unrelated. The family is open: a new member is a struct and one method of each.

The two questions travel together because they share a unit. A hop count is budgeted in hops and a weighted path length in the distance estimator's units, so a budget stated apart from the rule that measures it would be a number nobody could interpret — which is why the budget lives on the member rather than on [`NetworkEstimator`](@ref).

# Building the structure is a separate kernel from measuring it

The measuring kernel takes **the structure**, not the estimator that produces one: `separation_matrix(sep, g)` is the interface, and `separation_matrix(sep, nte, X)` is a wrapper that calls [`separation_graph`](@ref) first. The split is [`calc_weighted_adjacency_graph`](@ref)'s two-entry-point shape, for the same reason — the structure is expensive and a caller often holds one already. Under [`VariationInfoDistance`](@ref) building it is `98%` of [`clusterise`](@ref)'s runtime, so a consumer that resolves a budget rule *and* measures the separations must build once and pass the graph, not call two estimator-taking kernels.

It is also the seam a test or an extension enters through. Every structure a shipped estimator can build is connected — a spanning tree or a PMFG — so a disconnected graph, and with it the unreachable sentinel below, is reachable only by handing one in.

# Two more kernels, and why neither is a third question

[`resolve_separation`](@ref) turns a member whose budget is a **rule** into one whose budget is a value, and is called by every consumer before the other two kernels. It is not a third question about the network: a member whose budget is already a number is returned unchanged by the fallback on this type, so an extension inherits the kernel and never writes one.

[`is_related`](@ref) applies the budget to one entry of the separation matrix, over [`is_reachable`](@ref)'s sentinel test. Both are single generic methods on this type rather than per-member ones, because the rule — *not the sentinel, and no further than the budget* — is the same rule whatever the unit; a member whose underlying routine reports an exotic sentinel overrides [`is_reachable`](@ref) alone. Every consumer applying a budget calls them instead of open-coding the comparison, which is what keeps the ordering obligation of [`separation_matrix`](@ref)'s "the unreachable sentinel" inside an interface.

The two shipped members widen their budget field to admit a rule — [`HopCountValue`](@ref) and [`PathLengthValue`](@ref) — so a caller who cannot state the budget in advance states what would produce it instead. The resolution happens where the data is in hand, which is the only place a rule can be answered, and it is why [`separation_budget`](@ref) refuses an unresolved member rather than returning a function.

# Separation is not decay

[`AbstractSeparationDecayAlgorithm`](@ref) turns a separation into a *score*; this family produces the separation and says where it runs out. The seam is that `sep` decides **which pairs are related** — every consumer of a network needs that — while `decay` decides **how strongly, as a number**, which only the feature producer wants. That is why `sep` sits on [`NetworkEstimator`](@ref) and `decay` sits on [`Proximity`](@ref).

# The family is unqualified on purpose

The name says nothing about graphs. A taxonomy depth is a separation too, so the room is left for a member that measures one, rather than being closed off by an `AbstractGraphSeparationAlgorithm`.

# Related

  - [`HopCount`](@ref)
  - [`PathLength`](@ref)
  - [`separation_graph`](@ref)
  - [`separation_matrix`](@ref)
  - [`separation_budget`](@ref)
  - [`resolve_separation`](@ref)
  - [`is_reachable`](@ref)
  - [`is_related`](@ref)
  - [`HopCountAlgorithm`](@ref)
  - [`PathLengthAlgorithm`](@ref)
  - [`AbstractSeparationDecayAlgorithm`](@ref)
  - [`NetworkEstimator`](@ref)
  - [`Proximity`](@ref)
"""
abstract type AbstractSeparationAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all rules computing a hop count from the network and the data.

A [`HopCount`](@ref) budget is usually a number the caller states. It does not have to be. A subtype of `HopCountAlgorithm` is a **callable struct** standing in the `n` field, and [`resolve_separation`](@ref) calls it with the network estimator and the data matrix in hand. That is what lets a budget follow a universe whose size the caller cannot know in advance — a cross-validation fold, or a subproblem of a meta optimiser such as [`NestedClustered`](@ref).

# The extension contract

A subtype defines one method, the functor:

    (rule::MySubtype)(nte::AbstractNetworkEstimator, X::MatNum, g::Graphs.AbstractGraph;
                      dims::Int = 1, kwargs...) -> Integer

`g` is the structure [`separation_graph`](@ref) built for the separation the rule stands in, so a rule reads what it needs off a graph it did **not** pay to build — through [`separation_matrix`](@ref), which takes `g` directly. It still pays for the all-pairs traversal; see [`resolve_separation`](@ref).

`nte` owns the separation and `X` is the data `g` was built from. Both are inert for the shipped rule, and are the channel through which an extension reaches what the graph does not carry — the distance estimator, the observation count, a covariance.

**The return value must be an `Integer`, and this is checked rather than bounded.** A functor's return type is not part of its signature, so the family cannot state the requirement in the type system. [`resolve_separation`](@ref) checks it instead, and the check is not a formality: three readers use `0:n` as a **matrix-power count**, where `0:1.5` silently drops a power rather than failing.

A bare `Function` is admitted in the same field and carries the same obligation, unchecked at construction. Subtype this instead when the rule has parameters — the struct holds them, prints them, and is comparable.

# Related

  - [`HopCount`](@ref)
  - [`HopCountQuantile`](@ref)
  - [`HopCountRule`](@ref)
  - [`HopCountValue`](@ref)
  - [`PathLengthAlgorithm`](@ref)
  - [`resolve_separation`](@ref)
"""
abstract type HopCountAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all rules computing a path-length budget from the network and the data.

The [`PathLength`](@ref) counterpart of [`HopCountAlgorithm`](@ref): a **callable struct** standing in the `dmax` field, called by [`resolve_separation`](@ref) with the network estimator, the data matrix, and the structure already built from them in hand.

The two families are separate because their return obligations differ, and the split is what lets one of them be checked. A hop count must be an `Integer`; a path-length budget is stated in the distance estimator's units, so it is any `Number` — or `nothing`, which resolves to the observed diameter exactly as a stated `nothing` does.

# The extension contract

A subtype defines one method, the functor:

    (rule::MySubtype)(nte::AbstractNetworkEstimator, X::MatNum, g::Graphs.AbstractGraph;
                      dims::Int = 1, kwargs...) -> Number

`g` is [`separation_graph`](@ref)'s structure, weighted by distance on both branches under a [`PathLength`](@ref). A bare `Function` is admitted in the same field and carries the same obligation, unchecked at construction.

**A rule must return a `Number`, and `nothing` is not one.** `nothing` in the `dmax` field means the observed diameter, which is a statement the caller makes *instead of* stating a rule. A rule that meant to ask for the diameter is asking for something the field already spells, and a rule that returned `nothing` by accident would silently get the maximal ball. So [`PathLengthValue`](@ref) covers the rules and the numbers, and the field is an `Option` of it.

# Related

  - [`PathLength`](@ref)
  - [`PathLengthQuantile`](@ref)
  - [`PathLengthRule`](@ref)
  - [`PathLengthValue`](@ref)
  - [`HopCountAlgorithm`](@ref)
  - [`resolve_separation`](@ref)
"""
abstract type PathLengthAlgorithm <: AbstractAlgorithm end
"""
    const HopCountRule = Union{<:HopCountAlgorithm, <:Function}

Alias for the **dynamic** forms of a hop count.

Matches the two things [`resolve_separation`](@ref) *calls* rather than reads: a [`HopCountAlgorithm`](@ref) and a bare `Function`. Used for dispatch, so that `HopCount{<:HopCountRule}` names an unresolved separation and `HopCount{<:Integer}` a resolved one.

# Related

  - [`HopCountAlgorithm`](@ref)
  - [`HopCountValue`](@ref)
  - [`HopCount`](@ref)
  - [`resolve_separation`](@ref)
"""
const HopCountRule = Union{<:HopCountAlgorithm, <:Function}
"""
    const HopCountValue = Union{<:Integer, <:HopCountAlgorithm, <:Function}

Alias for everything [`HopCount`](@ref)'s `n` field accepts.

Widens the field from the stated `Integer` to the rules of [`HopCountRule`](@ref) as well. The `Integer` case is the resolved one and every reader takes it directly; a rule is resolved by [`resolve_separation`](@ref) before any reader sees it.

# Related

  - [`HopCount`](@ref)
  - [`HopCountRule`](@ref)
  - [`HopCountAlgorithm`](@ref)
  - [`PathLengthValue`](@ref)
"""
const HopCountValue = Union{<:Integer, <:HopCountAlgorithm, <:Function}
"""
    const PathLengthRule = Union{<:PathLengthAlgorithm, <:Function}

Alias for the **dynamic** forms of a path-length budget.

The [`PathLength`](@ref) counterpart of [`HopCountRule`](@ref).

# Related

  - [`PathLengthAlgorithm`](@ref)
  - [`PathLengthValue`](@ref)
  - [`PathLength`](@ref)
  - [`resolve_separation`](@ref)
"""
const PathLengthRule = Union{<:PathLengthAlgorithm, <:Function}
"""
    const PathLengthValue = Union{<:Number, <:PathLengthAlgorithm, <:Function}

Alias for everything [`PathLength`](@ref)'s `dmax` field accepts, apart from `nothing`.

The [`PathLength`](@ref) counterpart of [`HopCountValue`](@ref), and the field is `Option{PathLengthValue}` rather than this alias alone. The asymmetry is deliberate: `nothing` in that field means the observed diameter, which is one of the *stated* budgets and not something a rule may answer with. Keeping it outside the alias is what makes [`resolve_separation`](@ref)'s check a plain `isa(dmax, Number)`.

# Related

  - [`PathLength`](@ref)
  - [`PathLengthRule`](@ref)
  - [`PathLengthAlgorithm`](@ref)
  - [`HopCountValue`](@ref)
  - [`Option`](@ref)
"""
const PathLengthValue = Union{<:Number, <:PathLengthAlgorithm, <:Function}
"""
$(DocStringExtensions.TYPEDEF)

Separation measured as the number of graph edges between two assets.

The separation between two assets is the length of the shortest path between them counted in **edges**, ignoring the weights those edges carry, and the budget is `n` of them. It is the separation the network family has always used: [`phylogeny_matrix`](@ref)'s `sum(A^i for i in 0:n)` and both [`clusterise`](@ref) methods' power sums are hop budgets, and this member is where that `n` now lives.

The budget is a **field** rather than an argument because it is stated in hops, a unit only this member uses. [`PathLength`](@ref) measures the same structure in the distance estimator's units and carries its own budget in those, so no caller has to know which unit is in play.

# The power sum is the source's range connection matrix

[`phylogeny_matrix`](@ref)'s ``\\mathbf{P} = \\mathbb{1}_{x \\geq 1}\\left(\\sum_{i=0}^{n} \\mathbf{A}^{i}\\right) - \\mathbf{I}`` is the **range connection matrix** ``\\mathbf{B}_{1,n}`` of the source, spelled with one indicator instead of `n` of them. The source builds a per-length connection matrix ``\\mathbf{B}_{k} = \\mathbb{1}_{x \\geq 1}(\\mathbf{A}^{k} + \\mathbf{I}) - \\mathbf{I}`` and then indicates their sum; the library adds the powers first, and the ``\\mathbf{A}^{0} = \\mathbf{I}`` term the sum picks up is the term the trailing ``- \\mathbf{I}`` removes again. The two agree entry for entry: on the source's own six-node example the matrices are **identical for every** `n` from `1` to `5`.

# The budget may be a rule instead of a number

`n` also takes a [`HopCountAlgorithm`](@ref) or a bare `Function`, which [`resolve_separation`](@ref) calls as `n(nte, X, g; dims = dims, kwargs...)` at the point of use, `g` being the structure the consumer already built. A caller who cannot state the budget in advance — because the universe is a cross-validation fold or a subproblem of a meta optimiser — states the *rule* that produces it instead of a number that was right for one universe. [`HopCountQuantile`](@ref) is the shipped rule.

`n` is still an `Integer` **once resolved**, and never a `Real`. Three readers use `0:(nte.sep.n)` as a **matrix-power count**, where `0:1.5` silently drops a power instead of failing, so [`resolve_separation`](@ref) checks the rule's return value rather than trusting it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HopCount(;
        n::HopCountValue = 1
    ) -> HopCount

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:ntn])

# Examples

```jldoctest
julia> HopCount()
HopCount
  n ┴ Int64: 1

julia> HopCount(; n = HopCountQuantile())
HopCount
  n ┼ HopCountQuantile
    │   q ┴ Float64: 0.25
```

# Related

  - [`AbstractSeparationAlgorithm`](@ref)
  - [`PathLength`](@ref)
  - [`HopCountValue`](@ref)
  - [`HopCountAlgorithm`](@ref)
  - [`HopCountQuantile`](@ref)
  - [`resolve_separation`](@ref)
  - [`separation_matrix`](@ref)
  - [`separation_budget`](@ref)
  - [`NetworkEstimator`](@ref)
  - [`Proximity`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 13.1.2, Equations 13.1-13.2.
"""
@concrete struct HopCount <: AbstractSeparationAlgorithm
    """
    $(field_dict[:ntn])
    """
    n
    function HopCount(n::HopCountValue)
        # A rule is a promise about a value that does not exist yet, so there is nothing to
        # check here. `resolve_separation` checks what it returns -- and it returns it
        # through this constructor, so the resource cap below covers a computed `n` too.
        if isa(n, Integer)
            @argcheck(n >= one(n), DomainError(n, "n must be >= 1"))
            assert_resource_cap(n, RESOURCE_LIMITS[].max_hop_count, :n, :max_hop_count)
        end
        return new{typeof(n)}(n)
    end
end
function HopCount(; n::HopCountValue = 1)::HopCount
    return HopCount(n)
end
"""
$(DocStringExtensions.TYPEDEF)

Separation measured as the length of the shortest weighted path between two assets.

The separation between two assets is the sum of the **distances** along the shortest path joining them in the network, and the budget is `dmax` of the same units. It is the graded counterpart of [`HopCount`](@ref): both measure how far apart two assets sit in the same structure, but one counts the edges and the other adds up how long they are.

It is a **library generalisation and rests on no published source**. [`HopCount`](@ref) has one — it is the range connection matrix of a walk length — but that literature counts edges throughout, and states no budget in the units a distance estimator emits.

# The path runs over distances on both branches

The path is taken over the distance matrix **restricted to the structure's edge set** — [`calc_distance_weighted_graph`](@ref) — whichever branch built the structure. On the tree branch that is the graph's own weights; on the PMFG branch the structure is selected by similarity and then re-weighted by the distance that the similarity is a strictly decreasing function of.

Pathing over the PMFG's similarities instead is not a second convention, it is backwards: a shortest path over similarities *minimises total similarity*, so it prefers the route through the weakest links. It fails quietly — measured over the four similarity algorithms, the backwards answer correlates `0.95` to `0.97` with the right one, which is far too close to catch by looking.

# It is not comparable with a hop count, only interchangeable with one

`PathLength` and [`HopCount`](@ref) satisfy the same contract, so any consumer reading a separation *through that contract* takes either — [`Proximity`](@ref) and [`phylogeny_matrix`](@ref) both do. Their outputs are **not comparable as values**: the budgets are in different units, the supports differ, and under [`LinearDecay`](@ref) the scales differ.

Both [`clusterise`](@ref) methods are the exception, and refuse `PathLength` at dispatch. They do not read the separation through the contract at all: they index a matrix power by `sep.n`, and a radius has no analogue of one.

On a real universe the two agree far more than that suggests — measured over twenty assets, `rho = 0.99` on a minimum spanning tree and `0.95` to `0.98` on a PMFG, with `0.16%` of pairs of pairs strictly inverted on the tree and **none at all** on the PMFG — because both structures are selected by distance to begin with. That agreement is **empirical, not guaranteed**: it is a fact about the graphs an [`AbstractDistanceEstimator`](@ref) tends to produce, not a property of either separation.

# The budget

`dmax = nothing` is the default and means **the whole connected component**, implemented as the observed diameter: the largest finite entry of the separation matrix. It is the default because nobody has an intuition for a summed path in the units an [`AbstractDistanceEstimator`](@ref) emits — `dmax = 0.37` is not a number a caller can reason about, whereas "look at the whole component and let the decay do the falling off" is. Choosing a number is how a caller buys fold-stability.

[`separation_budget`](@ref) clamps a chosen `dmax` to the observed diameter. The clamp cuts nothing — no pair sits beyond the diameter — so it is a **scale-top correction** and bites only [`LinearDecay`](@ref), the one decay that reads the budget: without it, a `dmax` far above the diameter would flatten `Z` towards a constant while forbidding no pair at all.

The default reads very differently through [`phylogeny_matrix`](@ref), which **selects** on the budget instead of shaping a fall-off inside it. "The whole connected component" there means every reachable pair is related, so `NetworkEstimator(; sep = PathLength())` yields a matrix of ones off the diagonal — the opposite end of the dial from [`HopCount`](@ref)'s default `n = 1`. State a numeric `dmax` to select anything narrower.

# The budget may be a rule instead of a number

`dmax` also takes a [`PathLengthAlgorithm`](@ref) or a bare `Function`, which [`resolve_separation`](@ref) calls as `dmax(nte, X, g; dims = dims, kwargs...)` at the point of use, `g` being the structure the consumer already built. This is the answer to the paragraph above for a caller who *cannot* state a number: [`PathLengthQuantile`](@ref) asks for the budget that relates a stated **fraction of the reachable pairs**, which is a quantity a caller does have an intuition for, and which means the same thing on every fold of a cross-validation and in every subproblem of a meta optimiser.

A fixed `dmax` and a rule buy different things, and the difference is the whole point. A fixed `dmax` holds the **radius** still and lets the related-pair count move with the graph. A quantile rule holds the **count** still and lets the radius move. Neither is fold-stable in both senses at once, because the graph is refitted either way.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PathLength(;
        dmax::Option{PathLengthValue} = nothing
    ) -> PathLength

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:sepdmax])

# Examples

```jldoctest
julia> PathLength()
PathLength
  dmax ┴ nothing

julia> PathLength(; dmax = 0.5)
PathLength
  dmax ┴ Float64: 0.5

julia> PathLength(; dmax = PathLengthQuantile(; q = 0.3))
PathLength
  dmax ┼ PathLengthQuantile
       │   q ┴ Float64: 0.3
```

# Related

  - [`AbstractSeparationAlgorithm`](@ref)
  - [`HopCount`](@ref)
  - [`PathLengthValue`](@ref)
  - [`PathLengthAlgorithm`](@ref)
  - [`PathLengthQuantile`](@ref)
  - [`resolve_separation`](@ref)
  - [`separation_matrix`](@ref)
  - [`separation_budget`](@ref)
  - [`calc_distance_weighted_graph`](@ref)
  - [`NetworkEstimator`](@ref)
  - [`Proximity`](@ref)
"""
@concrete struct PathLength <: AbstractSeparationAlgorithm
    """
    $(field_dict[:sepdmax])
    """
    dmax
    function PathLength(dmax::Option{PathLengthValue})
        # A rule is a promise about a value that does not exist yet, so there is nothing to
        # check here. `resolve_separation` checks what it returns.
        if isa(dmax, Number)
            @argcheck(dmax > zero(dmax), DomainError(dmax, "dmax must be > 0"))
        end
        return new{typeof(dmax)}(dmax)
    end
end
function PathLength(; dmax::Option{PathLengthValue} = nothing)::PathLength
    return PathLength(dmax)
end
"""
    is_reachable(sep::AbstractSeparationAlgorithm, d::Number)

Is `d` a separation at all, or the sentinel an unreachable pair carries?

[`separation_matrix`](@ref) passes the underlying routine's sentinel through **unrepaired**, so this is the test that tells a measured separation from a missing one. It is one generic method on [`AbstractSeparationAlgorithm`](@ref) rather than one per member, because the two shipped sentinels are both covered by the same expression; a member whose routine reports something else overrides this method, and inherits [`is_related`](@ref) unchanged.

# The test is not `isfinite` alone, and not `typemax` alone

Both clauses carry a sentinel of their own.

  - `isfinite` is `true` for every `Integer`, so on its own it admits [`HopCount`](@ref)'s `typemax(Int)` — which [`ReciprocalDecay`](@ref) then overflows.
  - `typemax` of a `Float64` *is* `Inf`, so the comparison covers [`PathLength`](@ref)'s sentinel as well; `isfinite` stays to reject a `NaN`, which no shipped path produces and which would compare `false` against every budget anyway.

# Algorithm

 1. Test `isfinite(d)`, rejecting a `NaN` and rejecting [`PathLength`](@ref)'s `Inf`.
 2. Compare `d` against `typemax(typeof(d))`, rejecting [`HopCount`](@ref)'s `typemax(Int)`, which step 1 admits.
 3. Return `reachable`, the conjunction of the two tests. `&&` short-circuits, so step 2 runs only on a finite `d`.

# Arguments

  - `sep`: Separation algorithm. Inert for the shipped members, and the dispatch channel for an extension whose routine reports a different sentinel.
  - `d`: One entry of a separation matrix from [`separation_matrix`](@ref).

# Returns

  - `reachable::Bool`: `true` when `d` is a measured separation.

# Related

  - [`AbstractSeparationAlgorithm`](@ref)
  - [`is_related`](@ref)
  - [`separation_matrix`](@ref)
  - [`separation_quantile`](@ref)
  - [`separation_budget`](@ref)
"""
function is_reachable(::AbstractSeparationAlgorithm, d::Number)::Bool
    return isfinite(d) && d != typemax(typeof(d))
end
"""
    is_related(sep::AbstractSeparationAlgorithm, d::Number, dmax::Number)

Does a separation of `d` count as related under a budget of `dmax`?

The one place the budget is applied to an entry of a separation matrix: reachable, and no further than `dmax`. Every consumer that selects on a budget calls this instead of writing the comparison out, so the rule has one spelling — [`phylogeny_matrix`](@ref) and [`phylogeny_features`](@ref) had two, and one of them was a budget test with no sentinel test behind it.

# The reachability test comes first

`d <= dmax` is not sufficient on its own. It happens to reject both shipped sentinels, because [`separation_budget`](@ref) clamps a [`PathLength`](@ref) budget to the observed **finite** diameter and a [`HopCount`](@ref) budget is capped far below `typemax(Int)` — but that is a property of the two shipped budgets, not of the comparison. [`is_reachable`](@ref) makes the rejection the predicate's own, so a budget that reached its unit's ceiling would still exclude an unreachable pair.

# It does not remove the caller's obligation to short-circuit

A consumer that scores the separation must still keep the *evaluation* of the score inside a short-circuiting branch — `is_related(...) ? separation_decay(...) : zero(...)`, never `ifelse` — because an `ifelse` evaluates both arms and [`ReciprocalDecay`](@ref) overflows `1 + d` at `typemax(Int)`, which a fractional `power` turns into a `DomainError`. The predicate owns the rule; the call site owns the laziness.

# Algorithm

 1. Call [`is_reachable`](@ref) on `sep` and `d`, giving the reachability test. This runs first, so a sentinel is rejected whatever `dmax` is.
 2. Compare `d` against `dmax`, giving the budget test.
 3. Return `related`, the conjunction of the two tests. `&&` short-circuits, so step 2 runs only on a reachable `d`.

# Arguments

  - `sep`: Separation algorithm, forwarded to [`is_reachable`](@ref).
  - `d`: One entry of a separation matrix from [`separation_matrix`](@ref).
  - `dmax`: Separation budget in scope, from [`separation_budget`](@ref). In the units `sep` measures in, which is why the two arrive together.

# Returns

  - `related::Bool`: `true` when the pair is reachable and inside the budget.

# Related

  - [`AbstractSeparationAlgorithm`](@ref)
  - [`is_reachable`](@ref)
  - [`separation_budget`](@ref)
  - [`separation_decay`](@ref)
  - [`phylogeny_matrix`](@ref)
  - [`phylogeny_features`](@ref)
"""
function is_related(sep::AbstractSeparationAlgorithm, d::Number, dmax::Number)::Bool
    return is_reachable(sep, d) && d <= dmax
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all separation decay algorithms.

A separation decay turns a **separation** `d >= 0` — how far apart two assets sit in whatever structure the caller is reading — into a score, and is applied by [`separation_decay`](@ref). The family is open: a caller wanting a different fall-off defines a member and a [`separation_decay`](@ref) method for it, exactly as [`AbstractSimilarityMatrixAlgorithm`](@ref) is extended through [`distance_to_similarity`](@ref).

`d` is a **real** separation rather than an integer hop count, so one family serves an unweighted graph — where hop counts enter as integer-valued reals — and any structure whose separation is continuous.

# The contract

  - Defined for every `d >= 0`.
  - `f(0) > 0` and maximal. Self-inclusion is load-bearing rather than cosmetic: a decay that does not put an asset at the top of its own scale silently produces a *structural equivalence* matrix instead of a proximity one — see [`PhylogenyFeatures`](@ref)'s "Why the diagonal includes self".
  - Monotone non-increasing in `d`.
  - Never assumed to reach zero. **Truncation is a separate knob**: the consumer applies its own budget — [`separation_budget`](@ref) of the [`AbstractSeparationAlgorithm`](@ref) in scope — and the decay only shapes the fall-off inside it. An exponential never reaches zero, so budget and fall-off cannot be the same dial.
  - `f(d) >= 0` for `0 <= d <= dmax`. `0` is the unreachable sentinel, so a negative score *inside* the budget would place a **reachable** pair strictly below an **unreachable** one — an ordering inversion within the producer's own scale. It is not a claim that a signed score is wrong in general: the feature matrix is signed-tolerant by decision, and [`assert_metric_domain`](@ref) checks non-negativity per metric at the consumer rather than blanket. This clause is producer-local, and it is non-negativity rather than strict positivity because a decay that bottoms out at zero says *no relatedness*, which is the same claim an unreachable pair makes.

The clause is **scoped to the budget** because the sign outside it is unobservable — the consumer's `h[u] <= n` test short-circuits before the decay is ever evaluated there — and because the family's own default violates the wider statement: [`LinearDecay`](@ref) crosses zero at `d = dmax + 1` and is negative above it. Binding the clause on all `d >= 0` would need the `max(0, ⋅)` floor the budget knob exists to avoid, a second truncation biting before `n` does.

A zero in the resulting feature matrix therefore means **functionally unreachable**: either the graph is disconnected there, or the decay has fallen to nothing — the same claim about the pair, and nothing downstream can act on the difference. No shipped member emits zero anywhere inside the budget, so for what ships a zero is disconnection and nothing else.

# The budget is an argument, not a field

[`separation_decay`](@ref) takes the budget in scope as its third argument, `dmax`, and members may ignore it — only [`LinearDecay`](@ref) reads it, to set `f(0)`. Keeping it off the member is what makes the two knobs impossible to desync: the [`AbstractSeparationAlgorithm`](@ref) stays the single source of truth for the budget, rather than mirroring it on an algorithm that cannot see it at construction. [`ExponentialDecay`](@ref) provides the self-versus-neighbour contrast a free top-of-scale would have bought, without the hazard of a second truncation hiding inside the decay.

# Enforcement

The contract is enforced rather than merely documented, by a probing [`assert_separation_decay`](@ref) fallback on this type. The shipped members satisfy it by construction and override that fallback to a no-op, so the check is **opt-out**: an extension that says nothing about itself gets probed.

# Related

  - [`LinearDecay`](@ref)
  - [`ExponentialDecay`](@ref)
  - [`ReciprocalDecay`](@ref)
  - [`NoDecay`](@ref)
  - [`separation_decay`](@ref)
  - [`assert_separation_decay`](@ref)
  - [`AbstractSeparationAlgorithm`](@ref)
  - [`Proximity`](@ref)
  - [`PhylogenyFeatures`](@ref)
"""
abstract type AbstractSeparationDecayAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Separation decay falling off linearly to the edge of the budget.

# Mathematical definition

```math
\\begin{align}
f(d) &= d_{\\mathrm{max}} + 1 - d\\,,
\\end{align}
```

Where:

  - $(math_dict[:d_sep])
  - ``d_{\\mathrm{max}}``: Separation budget in scope.

The default, and the only member that reads the budget. It is the fall-off the graded neighbourhood hardcoded before the family existed, so it reproduces those values exactly: a direct neighbour scores ``d_{\\mathrm{max}}``, the asset itself ``d_{\\mathrm{max}} + 1``.

Because truncation lives with the budget rather than in the decay, no `max(0, ⋅)` floor is needed — on the kept range `0 <= d <= dmax` the expression is strictly positive, bottoming out at `1`.

# Examples

```jldoctest
julia> separation_decay.(Ref(LinearDecay()), 0:3, 3)
4-element Vector{Int64}:
 4
 3
 2
 1
```

# Related

  - [`AbstractSeparationDecayAlgorithm`](@ref)
  - [`ExponentialDecay`](@ref)
  - [`ReciprocalDecay`](@ref)
  - [`separation_decay`](@ref)
  - [`Proximity`](@ref)
"""
struct LinearDecay <: AbstractSeparationDecayAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Separation decay falling off exponentially.

# Mathematical definition

```math
\\begin{align}
f(d) &= e^{-\\lambda d}\\,,
\\end{align}
```

Where:

  - $(math_dict[:d_sep])
  - ``\\lambda``: Rate of the fall-off, `rate`.

Pins `f(0) = 1` and lets `rate` set the self-versus-neighbour contrast independently of the budget, which is what a caller wanting relatedness to drop *sharply* needs — the budget only says how far to look.

Parameterised by rate rather than by per-step retention. ``\\rho^d`` and ``e^{-\\lambda d}`` are the same function (``\\lambda = -\\log\\rho``), but "retention per step" is a statement about integers, and the family's argument is a real separation. The rate form also matches [`ExponentialSimilarity`](@ref) and [`ExpGerberIQDecay`](@ref), and needs only a one-sided bound to stay monotone.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExponentialDecay(;
        rate::Number = 1.0
    ) -> ExponentialDecay

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:sdrate])

# Examples

```jldoctest
julia> ExponentialDecay()
ExponentialDecay
  rate ┴ Float64: 1.0

julia> separation_decay.(Ref(ExponentialDecay()), 0:3, 3)
4-element Vector{Float64}:
 1.0
 0.36787944117144233
 0.1353352832366127
 0.049787068367863944
```

# Related

  - [`AbstractSeparationDecayAlgorithm`](@ref)
  - [`LinearDecay`](@ref)
  - [`ReciprocalDecay`](@ref)
  - [`separation_decay`](@ref)
  - [`ExponentialSimilarity`](@ref)
"""
@concrete struct ExponentialDecay <: AbstractSeparationDecayAlgorithm
    """
    $(field_dict[:sdrate])
    """
    rate
    function ExponentialDecay(rate::Number)
        @argcheck(zero(rate) < rate,
                  DomainError(rate, "the rate of an ExponentialDecay must be > 0"))
        return new{typeof(rate)}(rate)
    end
end
function ExponentialDecay(; rate::Number = 1.0)::ExponentialDecay
    return ExponentialDecay(rate)
end
"""
$(DocStringExtensions.TYPEDEF)

Separation decay falling off as a power of the separation.

# Mathematical definition

```math
\\begin{align}
f(d) &= \\left(1 + d\\right)^{-p}\\,,
\\end{align}
```

Where:

  - $(math_dict[:d_sep])
  - ``p``: Exponent of the fall-off, `power`.

The middle ground between [`LinearDecay`](@ref) and [`ExponentialDecay`](@ref): heavier-tailed than the exponential, so distant assets keep a small but non-negligible score.

The `1 +` is what makes classical inverse-distance weighting finite at ``d = 0``; it also pins `f(0) = 1`, matching [`ExponentialDecay`](@ref)'s scale for free. The alternative spelling ``(1 + d^p)^{-1}`` is *not* used: it pins ``f(1) = 1/2`` for every `p`, and its response to `p` at fixed `d` **flips sign about that pivot** — raising the exponent scores a pair nearer than ``d = 1`` *higher* and a pair further away lower. At ``p = 1`` and ``p = 3`` it scores ``d = 0.5`` as ``0.6667`` and ``0.8889``, and ``d = 2`` as ``0.3333`` and ``0.1111``. So `p` is no fall-off dial there, because it cannot sharpen the decay inside the pivot at all. Under the spelling above, raising `p` lowers the score at every ``d > 0``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ReciprocalDecay(;
        power::Number = 1.0
    ) -> ReciprocalDecay

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:sdpower])

# Examples

```jldoctest
julia> ReciprocalDecay()
ReciprocalDecay
  power ┴ Float64: 1.0

julia> separation_decay.(Ref(ReciprocalDecay()), 0:3, 3)
4-element Vector{Float64}:
 1.0
 0.5
 0.3333333333333333
 0.25
```

# Related

  - [`AbstractSeparationDecayAlgorithm`](@ref)
  - [`LinearDecay`](@ref)
  - [`ExponentialDecay`](@ref)
  - [`separation_decay`](@ref)
"""
@concrete struct ReciprocalDecay <: AbstractSeparationDecayAlgorithm
    """
    $(field_dict[:sdpower])
    """
    power
    function ReciprocalDecay(power::Number)
        @argcheck(zero(power) < power,
                  DomainError(power, "the power of a ReciprocalDecay must be > 0"))
        return new{typeof(power)}(power)
    end
end
function ReciprocalDecay(; power::Number = 1.0)::ReciprocalDecay
    return ReciprocalDecay(power)
end
"""
$(DocStringExtensions.TYPEDEF)

Separation decay that does not fall off at all.

# Mathematical definition

```math
\\begin{align}
f(d) &= 1\\,,
\\end{align}
```

Where:

  - $(math_dict[:d_sep])

# No decay is not no truncation

The name is about the *fall-off* and nothing else. The budget still cuts: a pair outside it scores `0`, because truncation was never the decay's job — see [`AbstractSeparationDecayAlgorithm`](@ref)'s "The budget is an argument, not a field". What comes out is therefore an **indicator** of the neighbourhood the budget selects, not a matrix of ones.

That is exactly what makes it useful. Under [`HopCount`](@ref) it turns [`Proximity`](@ref) into the `n`-hop neighbourhood indicator, which is what the retired `BinaryNeighbourhood` produced; under [`PathLength`](@ref) it is an ε-ball, and neither needs a type of its own once the fall-off is a knob.

It is the flat end of the family, so it is the only member that is not strictly decreasing. The contract asks for monotone **non**-increasing, which a constant satisfies.

# Examples

```jldoctest
julia> separation_decay.(Ref(NoDecay()), 0:3, 3)
4-element Vector{Int64}:
 1
 1
 1
 1
```

# Related

  - [`AbstractSeparationDecayAlgorithm`](@ref)
  - [`LinearDecay`](@ref)
  - [`separation_decay`](@ref)
  - [`Proximity`](@ref)
  - [`HopCount`](@ref)
"""
struct NoDecay <: AbstractSeparationDecayAlgorithm end
"""
    separation_decay(dk::LinearDecay, d::Number, dmax::Number)
    separation_decay(dk::ExponentialDecay, d::Number, dmax::Number)
    separation_decay(dk::ReciprocalDecay, d::Number, dmax::Number)
    separation_decay(dk::NoDecay, d::Number, dmax::Number)

Score a separation under a decay algorithm.

The whole extension contract of [`AbstractSeparationDecayAlgorithm`](@ref): a new member is a struct and one method of this function.

# The method the call selects

This function is a **selector** and runs no step of its own. It dispatches on `dk`, and the selected method evaluates one closed form. Each form is stated under `# Mathematical definition` on the member that owns it, and none is restated here.

| `dk`                       | The fall-off it selects            | Reads `dmax` |
|:-------------------------- |:---------------------------------- |:------------ |
| [`LinearDecay`](@ref)      | Linear to the edge of the budget   | yes          |
| [`ExponentialDecay`](@ref) | Exponential in `rate`              | no           |
| [`ReciprocalDecay`](@ref)  | A power of `1 + d`, set by `power` | no           |
| [`NoDecay`](@ref)          | Flat                               | no           |

[`LinearDecay`](@ref) is the only member that reads the budget, which is why the other three methods leave their third argument unnamed.

# Arguments

  - `dk`: Separation decay algorithm.
  - `d`: Separation between two assets, `d >= 0`. Real rather than integral, so a weighted path length is as admissible as a hop count.
  - `dmax`: Separation budget in scope. **Inert for members that do not need it** — only [`LinearDecay`](@ref) reads it. Inert arguments have precedent here: [`phylogeny_features`](@ref) ignores its `alg` entirely for a partition source.

# Validation

The contract is not checked here — this runs inside an `assets × assets` loop. Callers probe once up front with [`assert_separation_decay`](@ref) instead.

# Returns

  - `f::Number`: Score for the separation. Non-negative for `0 <= d <= dmax`; **above** the budget the sign is unconstrained and [`LinearDecay`](@ref) does go negative, which is harmless because the consumer's budget test short-circuits before the call.

# Examples

```jldoctest
julia> separation_decay(LinearDecay(), 2, 3)
2

julia> separation_decay(ExponentialDecay(; rate = 2.0), 2, 3)
0.01831563888873418

julia> separation_decay(ReciprocalDecay(; power = 2.0), 2, 3)
0.1111111111111111

julia> separation_decay(NoDecay(), 2, 3)
1
```

# Related

  - [`AbstractSeparationDecayAlgorithm`](@ref)
  - [`LinearDecay`](@ref)
  - [`ExponentialDecay`](@ref)
  - [`ReciprocalDecay`](@ref)
  - [`NoDecay`](@ref)
  - [`assert_separation_decay`](@ref)
  - [`Proximity`](@ref)
"""
function separation_decay end
function separation_decay(::LinearDecay, d::Number, dmax::Number)::Number
    return dmax + one(dmax) - d
end
function separation_decay(dk::ExponentialDecay, d::Number, ::Number)::Number
    return exp(-dk.rate * d)
end
function separation_decay(dk::ReciprocalDecay, d::Number, ::Number)::Number
    return inv((one(d) + d)^dk.power)
end
function separation_decay(::NoDecay, d::Number, ::Number)::Number
    return one(d)
end
"""
    assert_separation_decay(dk::AbstractSeparationDecayAlgorithm, ds, dmax::Number)
    assert_separation_decay(dk::Union{<:LinearDecay, <:ExponentialDecay, <:ReciprocalDecay,
                                      <:NoDecay}, ds, dmax::Number)

Check that a separation decay honours its contract over the separations it will be asked about.

The fallback **probes**: it evaluates [`separation_decay`](@ref) over `ds` and checks the result is finite, that `f(0)` is strictly positive and maximal, that the values are monotone non-increasing, and that none of them is negative. The four shipped members satisfy the contract by construction and override this to a no-op, so the probe costs nothing for what ships and is fail-safe for extensions.

Probing is cheap where it is used because `ds` is small and the loop it guards is not: [`Proximity`](@ref) passes `0:dmax`, which under [`HopCount`](@ref) is *exhaustive* — every separation the `assets × assets` loop can ever ask about, in `dmax + 1` evaluations. Under a separation whose budget is not an integer the same range is a unit-spaced *sample*, which is all a continuum admits and all the clauses below need.

Non-negativity gets **one extra evaluation at `d = dmax`**, whether or not `dmax` appears in `ds`, mirroring the out-of-loop evaluation of `f(0)`. That endpoint is what closes the clause over a *continuum*: monotonicity is already promised, so `f(dmax) >= 0` implies `f(d) >= 0` for every `d` in `[0, dmax]`, and a `ds` that can only ever be a sample — as it must be once separations are weighted path lengths — costs this clause nothing. Monotonicity itself gains nothing from the endpoint and remains genuinely sampled.

# Algorithm

These are the steps of the probing fallback. The method on the four shipped members runs none of them.

 1. Score the separation `zero(dmax)` with [`separation_decay`](@ref), giving `f0`, the top of the scale.
 2. Check that `f0` is finite and strictly positive.
 3. Sort `ds` when it is not sorted already, so that step 5 reads the separations in increasing order.
 4. Set `fp`, the score of the previous separation, to `f0`.
 5. For each `d` of `ds`, score it with [`separation_decay`](@ref), giving `f`. Check that `f` is finite and does not exceed `f0`. Check that `f` does not exceed `fp`. Check that `f` is non-negative. Set `fp` to `f`.
 6. Score `dmax` itself, giving `fmax`, and check that `fmax` is non-negative. This is the endpoint the paragraph above states, and step 5 need not have reached it.
 7. Return `nothing`.

# Arguments

  - `dk`: Separation decay algorithm.
  - `ds`: Separations to probe. Need not be sorted. Precondition: `ds ⊆ [0, dmax]` — `ds` is what the guarded loop will ask about, and the loop never asks outside the budget.
  - `dmax`: Separation budget in scope, forwarded to [`separation_decay`](@ref) and probed as an endpoint in its own right.

# Validation

  - Every probed value is finite.
  - `f(0) > 0`.
  - `f(0) >= f(d)` for every probed `d`.
  - The probed values are monotone non-increasing in `d`.
  - `f(d) >= 0` for every probed `d`, and at `d = dmax` whether or not it was probed.

# Returns

  - `nothing`.

# Related

  - [`AbstractSeparationDecayAlgorithm`](@ref)
  - [`separation_decay`](@ref)
  - [`Proximity`](@ref)
"""
function assert_separation_decay(dk::AbstractSeparationDecayAlgorithm, ds,
                                 dmax::Number)::Nothing
    f0 = separation_decay(dk, zero(dmax), dmax)
    @argcheck(isfinite(f0) && f0 > zero(f0),
              DomainError(f0,
                          "a separation decay must score d = 0 finite and strictly positive, got $(f0) from $(typeof(dk))"))
    ds = issorted(ds) ? ds : sort!(collect(ds))
    fp = f0
    for d in ds
        f = separation_decay(dk, d, dmax)
        @argcheck(isfinite(f) && f <= f0,
                  DomainError(f,
                              "a separation decay must be finite and maximal at d = 0, got $(f) at d = $(d) against $(f0) at d = 0 from $(typeof(dk))"))
        @argcheck(f <= fp,
                  DomainError(f,
                              "a separation decay must be monotone non-increasing, got $(f) at d = $(d) from $(typeof(dk))"))
        @argcheck(f >= zero(f),
                  DomainError(f,
                              "a separation decay must be non-negative inside the budget, got $(f) at d = $(d) against a budget of dmax = $(dmax) from $(typeof(dk))"))
        fp = f
    end
    # `ds` need not contain `dmax`, and a real separation makes an exhaustive `ds` impossible.
    # Monotonicity is already promised, so this one endpoint closes non-negativity over the
    # whole of `0 <= d <= dmax` -- the only clause whose domain closes from a single point.
    fmax = separation_decay(dk, dmax, dmax)
    @argcheck(fmax >= zero(fmax),
              DomainError(fmax,
                          "a separation decay must be non-negative inside the budget, got $(fmax) at d = $(dmax) against a budget of dmax = $(dmax) from $(typeof(dk))"))
    return nothing
end
# The shipped members satisfy the contract by construction, so the probe is turned off for
# them and left on for anything an extension defines -- opt-out, not opt-in.
function assert_separation_decay(::Union{<:LinearDecay, <:ExponentialDecay,
                                         <:ReciprocalDecay, <:NoDecay}, ::Any,
                                 ::Number)::Nothing
    return nothing
end

export HopCount, PathLength, LinearDecay, ExponentialDecay, ReciprocalDecay, NoDecay,
       separation_decay
