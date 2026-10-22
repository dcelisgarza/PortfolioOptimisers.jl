"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all phylogeny estimator types.

A subtype holds the configuration that finds the relations between assets from the data. A network estimator and a clusters estimator are the two families.

# Related

  - [`AbstractPhylogenyAlgorithm`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
"""
abstract type AbstractPhylogenyEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all phylogeny algorithm types.

A subtype selects one method inside a phylogeny estimator. A clustering algorithm, a tree type and a centrality algorithm are the three families.

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
"""
abstract type AbstractPhylogenyAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all phylogeny result types.

A subtype holds the output of a phylogeny estimator, such as a clustering or a phylogeny matrix.

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyAlgorithm`](@ref)
"""
abstract type AbstractPhylogenyResult <: AbstractResult end
"""
    const PlE_Pl = Union{<:AbstractPhylogenyEstimator, <:AbstractPhylogenyResult}

Alias for a phylogeny estimator or result.

Matches an [`AbstractPhylogenyEstimator`](@ref) or an [`AbstractPhylogenyResult`](@ref). The library dispatches on it where a function accepts a phylogeny estimator or a precomputed result.

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
"""
const PlE_Pl = Union{<:AbstractPhylogenyEstimator, <:AbstractPhylogenyResult}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the phylogeny estimator or result `pl` unchanged.

A caller that rebuilds an estimator with [`factory`](@ref) reaches this method for a phylogeny estimator or a precomputed result. A phylogeny estimator has no field that depends on a prior, so the method keeps every field and replaces none.

# Algorithm

 1. Return `pl` itself. The method rebuilds no field of it, and it discards `args` and `kwargs`.

# Arguments

  - `pl`: Phylogeny estimator or phylogeny result.
  - `args...`: Optional arguments. The method does not read them.
  - `kwargs...`: Optional keyword arguments. The method does not read them.

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

A caller that rebuilds an estimator with [`factory`](@ref) reaches this method for a phylogeny algorithm that has no nearer method. A phylogeny algorithm has no field that depends on a prior, so the method keeps every field and replaces none.

# Algorithm

 1. Return `alg` itself. The method rebuilds no field of it, and it discards `args` and `kwargs`.

# Arguments

  - `alg`: Phylogeny algorithm.
  - `args...`: Optional arguments. The method does not read them.
  - `kwargs...`: Optional keyword arguments. The method does not read them.

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

A separation algorithm states how far apart two assets are in a network, and how far apart two related assets can be. One object answers the two questions through three functions:

  - [`separation_graph`](@ref) builds the structure that the member measures.
  - [`separation_matrix`](@ref) reads the dense `assets × assets` separations from that structure.
  - [`separation_budget`](@ref) resolves the budget. A pair further apart than the budget is unrelated.

The family is open. A new member is a struct and one method of each function.

The two questions share a unit. A hop count has its budget in hops, and a weighted path length has its budget in the units of the distance estimator. A budget apart from the rule that measures it is a number with no unit. So the budget is a field of the member, not of [`NetworkEstimator`](@ref).

# The structure and the measurement are two functions

The function that measures takes the structure, not the estimator that builds it. `separation_matrix(sep, g)` is the interface, and `separation_matrix(sep, nte, X)` is a wrapper that calls [`separation_graph`](@ref) first. [`calc_weighted_adjacency_graph`](@ref) has the same two entry points for the same reason. The structure is expensive, and a caller often has one already.

Under [`VariationInfoDistance`](@ref), the structure takes `98%` of the runtime of [`clusterise`](@ref). A consumer that resolves a budget rule and also measures the separations must build the structure once and pass the graph to both functions.

A test or an extension also calls `separation_matrix(sep, g)`. Every structure that a shipped estimator builds is connected, because it is a spanning tree or a PMFG. Only a graph that a caller gives can be disconnected and carry the unreachable sentinel.

# Two more functions

[`resolve_separation`](@ref) changes a member whose budget is a rule into a member whose budget is a value. Every consumer calls it before the other two functions. It asks nothing new about the network. The fallback on this type returns a member with a numeric budget unchanged, so an extension inherits it and writes no method.

[`is_related`](@ref) applies the budget to one entry of the separation matrix, after the sentinel test of [`is_reachable`](@ref). Each is one generic method on this type, because the rule is the same in every unit. A related pair is not the sentinel, and it is not further apart than the budget.

A member whose routine reports a different sentinel overrides [`is_reachable`](@ref) alone. Every consumer that applies a budget calls these two functions and writes no comparison of its own. So the rule of "The unreachable sentinel" of [`separation_matrix`](@ref) stays inside one interface.

The two shipped members accept a rule in the budget field, through [`HopCountValue`](@ref) and [`PathLengthValue`](@ref). A caller who cannot state the budget in advance gives the rule that calculates it. The library resolves the rule where the data is available, because a rule needs the data. For this reason [`separation_budget`](@ref) refuses an unresolved member and does not return a function.

# Separation is not decay

[`AbstractSeparationDecayAlgorithm`](@ref) changes a separation into a score. This family calculates the separation and its budget. `sep` decides which pairs are related, and every consumer of a network needs that. `decay` decides how strongly a pair is related, and only the feature producer needs that number. So `sep` is a field of [`NetworkEstimator`](@ref) and `decay` is a field of [`Proximity`](@ref).

# The name does not say graph

The name of the family says nothing about graphs. A taxonomy depth is also a separation. The name leaves room for a member that measures one, and `AbstractGraphSeparationAlgorithm` does not.

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

Abstract supertype for all rules that compute a hop count from the network and the data.

The caller usually states a [`HopCount`](@ref) budget as a number, but the budget can also be a rule. A subtype of `HopCountAlgorithm` is a callable struct in the `n` field. [`resolve_separation`](@ref) calls it with the network estimator and the data matrix.

A rule lets the budget follow a universe whose size the caller does not know in advance. Examples are a cross-validation fold and a subproblem of a meta optimiser such as [`NestedClustered`](@ref).

# The extension contract

A subtype defines one method, the functor:

    (rule::MySubtype)(nte::AbstractNetworkEstimator, X::MatNum, g::Graphs.AbstractGraph;
                      dims::Int = 1, kwargs...) -> Integer

`g` is the structure that [`separation_graph`](@ref) built for the separation that holds the rule. The rule reads what it needs from `g` through [`separation_matrix`](@ref), which takes `g` directly, so the rule builds no structure. It still runs the traversal of all pairs, as [`resolve_separation`](@ref) states.

`nte` holds the separation, and `X` is the data that `g` comes from. The shipped rule reads neither. An extension reads them to get what the graph does not hold, for example the distance estimator, the number of observations or a covariance.

The return value must be an `Integer`. The return type of a functor is not part of its signature, so the type system cannot state this requirement. [`resolve_separation`](@ref) checks the value instead. Three readers use `0:n` as a count of matrix powers, and `0:1.5` drops a power with no error.

The same field also accepts a bare `Function`, which has the same requirement. The constructor does not check it. Subtype this type instead when the rule has parameters, because the struct holds them, prints them and compares equal by them.

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

Abstract supertype for all rules that compute a path-length budget from the network and the data.

This is the [`PathLength`](@ref) counterpart of [`HopCountAlgorithm`](@ref). A subtype is a callable struct in the `dmax` field. [`resolve_separation`](@ref) calls it with the network estimator, the data matrix and the structure that it built from them.

The two families have different return types, so they are two types. The split lets [`resolve_separation`](@ref) check each return type. A hop count must be an `Integer`. A path-length budget is in the units of the distance estimator, so it can be any `Number`.

# The extension contract

A subtype defines one method, the functor:

    (rule::MySubtype)(nte::AbstractNetworkEstimator, X::MatNum, g::Graphs.AbstractGraph;
                      dims::Int = 1, kwargs...) -> Number

`g` is the structure of [`separation_graph`](@ref). Under a [`PathLength`](@ref) its edges carry distances on both branches. The same field also accepts a bare `Function`, which has the same requirement. The constructor does not check it.

A rule must return a `Number`, and `nothing` is not a `Number`. `nothing` in the `dmax` field means the observed diameter. The caller states it in place of a rule. A rule cannot ask for the diameter with `nothing`, because the field states that value already. If a rule returns `nothing` in error, [`resolve_separation`](@ref) throws an `ArgumentError` and does not give the largest ball. So [`PathLengthValue`](@ref) holds the rules and the numbers, and the field is an `Option` of it.

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

Alias for the forms of a hop count that are rules.

Matches the two forms that [`resolve_separation`](@ref) calls and does not read: a [`HopCountAlgorithm`](@ref) and a bare `Function`. The library dispatches on it. `HopCount{<:HopCountRule}` is an unresolved separation, and `HopCount{<:Integer}` is a resolved one.

# Related

  - [`HopCountAlgorithm`](@ref)
  - [`HopCountValue`](@ref)
  - [`HopCount`](@ref)
  - [`resolve_separation`](@ref)
"""
const HopCountRule = Union{<:HopCountAlgorithm, <:Function}
"""
    const HopCountValue = Union{<:Integer, <:HopCountAlgorithm, <:Function}

Alias for every value that the `n` field of [`HopCount`](@ref) accepts.

The field accepts a stated `Integer` and the rules of [`HopCountRule`](@ref). An `Integer` is the resolved form, and every reader reads it directly. [`resolve_separation`](@ref) resolves a rule before a reader gets it.

# Related

  - [`HopCount`](@ref)
  - [`HopCountRule`](@ref)
  - [`HopCountAlgorithm`](@ref)
  - [`PathLengthValue`](@ref)
"""
const HopCountValue = Union{<:Integer, <:HopCountAlgorithm, <:Function}
"""
    const PathLengthRule = Union{<:PathLengthAlgorithm, <:Function}

Alias for the forms of a path-length budget that are rules.

This is the [`PathLength`](@ref) counterpart of [`HopCountRule`](@ref).

# Related

  - [`PathLengthAlgorithm`](@ref)
  - [`PathLengthValue`](@ref)
  - [`PathLength`](@ref)
  - [`resolve_separation`](@ref)
"""
const PathLengthRule = Union{<:PathLengthAlgorithm, <:Function}
"""
    const PathLengthValue = Union{<:Number, <:PathLengthAlgorithm, <:Function}

Alias for every value that the `dmax` field of [`PathLength`](@ref) accepts, except `nothing`.

This is the [`PathLength`](@ref) counterpart of [`HopCountValue`](@ref), but the field is `Option{PathLengthValue}`. `nothing` in that field means the observed diameter. It is a stated budget, and a rule cannot return it. The alias leaves `nothing` out, so the check of [`resolve_separation`](@ref) is `isa(dmax, Number)` alone.

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

The separation between two assets is the number of edges on the shortest path between them. The weights of the edges have no effect. The budget is `n` edges. The power sum `sum(A^i for i in 0:n)` of [`phylogeny_matrix`](@ref) and the power sums of the two [`clusterise`](@ref) methods read this `n`.

The budget is a field and not an argument, because its unit is the hop, and only this member uses that unit. [`PathLength`](@ref) measures the same structure in the units of the distance estimator and holds its budget in those units. So a caller does not have to know the unit.

# The power sum is the range connection matrix of the source

The matrix ``\\mathbf{P} = \\mathbb{1}_{x \\geq 1}\\left(\\sum_{i=0}^{n} \\mathbf{A}^{i}\\right) - \\mathbf{I}`` of [`phylogeny_matrix`](@ref) is the range connection matrix ``\\mathbf{B}_{1,n}`` of the source, with one indicator in place of `n` indicators. The source builds a connection matrix for each walk length, ``\\mathbf{B}_{k} = \\mathbb{1}_{x \\geq 1}(\\mathbf{A}^{k} + \\mathbf{I}) - \\mathbf{I}``, and then applies the indicator to their sum.

The library adds the powers first. The sum then has the term ``\\mathbf{A}^{0} = \\mathbf{I}``, and the last ``- \\mathbf{I}`` removes it. The two forms are equal for every graph and every `n`. Off the diagonal, each one is `1` when a walk of length `1` to `n` joins the pair. On the diagonal, each one is `0`.

The original, [graphpo1](@cite), prints the indicator as ``\\mathbb{1}_{x > 1}``. With that indicator, a pair that one walk joins is not in ``\\mathbf{B}_{k}``, and Figure 9 of the same source shows that it is. Equations 13.1 and 13.2 of [cajas2025](@cite) print ``\\mathbb{1}_{x \\geq 1}``, which gives that figure. The library builds this form.

# The budget can be a rule

`n` also accepts a [`HopCountAlgorithm`](@ref) or a bare `Function`. [`resolve_separation`](@ref) calls it as `n(nte, X, g; dims = dims, kwargs...)`, where `g` is the structure that the consumer built. A caller who cannot state the budget in advance gives the rule that calculates it. This applies when the universe is a cross-validation fold or a subproblem of a meta optimiser. [`HopCountQuantile`](@ref) is the shipped rule.

After resolution, `n` is an `Integer` and never a `Real`. Three readers use `0:n` as a count of matrix powers, and `0:1.5` drops a power with no error. So [`resolve_separation`](@ref) checks the return value of the rule.

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

  - $(ref_dict[:graphpo1]) Section 2.2, Equations 1-2.
  - $(ref_dict[:cajas2025]) Section 13.1.2, Equations 13.1-13.2, for the indicator ``\\mathbb{1}_{x \\geq 1}``.
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

The separation between two assets is the sum of the distances along the shortest path that joins them in the network. The budget `dmax` has the same units. It is the graded counterpart of [`HopCount`](@ref). Both measure how far apart two assets are in the same structure. `HopCount` counts the edges, and `PathLength` adds their lengths.

No published source states this separation. The library adds it. [`HopCount`](@ref) has a source, the range connection matrix of a walk length. That literature counts edges only, and states no budget in the units of a distance estimator.

# The path uses distances on both branches

The path uses the distance matrix on the edges of the structure alone, through [`calc_distance_weighted_graph`](@ref). This is true for each branch that builds the structure. On the tree branch, the weights of the graph are these distances. On the PMFG branch, the similarity selects the structure. The distance then weights its edges, and the similarity is a strictly decreasing function of that distance.

A path over the similarities of the PMFG is wrong, not a second convention. A shortest path over similarities has the smallest total similarity, so it goes through the weakest links. The error is hard to see. Over the four similarity algorithms, the wrong answer has a Pearson correlation of `0.95` to `0.97` with the correct one.

# It can replace a hop count, but its values are not comparable

`PathLength` and [`HopCount`](@ref) satisfy the same contract. [`Proximity`](@ref) and [`phylogeny_matrix`](@ref) read a separation through that contract, so each one accepts the two members. Their outputs are not comparable as values. The budgets have different units and the supports are different. Under [`LinearDecay`](@ref), the scales are also different.

The two [`clusterise`](@ref) methods refuse `PathLength` at dispatch. They do not read the separation through the contract. They use `sep.n` as the count of matrix powers, and a radius has no equivalent of that count.

On real data the two separations agree more than this suggests, because the distance selects both structures. Over twenty assets, the Pearson correlation of the two separation matrices is `0.99` on a minimum spanning tree and `0.97` to `0.98` on a PMFG. On the tree `0.16%` of the pairs of pairs are in the opposite order, and on the PMFG none are. This agreement is a measurement, not a guarantee. It comes from the graphs that an [`AbstractDistanceEstimator`](@ref) usually gives, not from a property of the two separations.

# The budget

The default `dmax = nothing` means the whole connected component. The library uses the observed diameter, which is the largest finite entry of the separation matrix. A caller cannot easily choose a summed path length in the units of an [`AbstractDistanceEstimator`](@ref), such as `dmax = 0.37`. The default reads the whole component and lets the decay make the fall-off. A caller who needs the same radius in each fold states a number.

[`separation_budget`](@ref) clamps a stated `dmax` to the observed diameter. No pair is further apart than the diameter, so the clamp removes no pair. It changes only the top of the scale of [`LinearDecay`](@ref), which is the one decay that reads the budget. With no clamp, a `dmax` far above the diameter makes the scores of [`LinearDecay`](@ref) almost constant and excludes no pair.

[`phylogeny_matrix`](@ref) uses the budget to select pairs, not to shape a fall-off. There the default relates every reachable pair. So the phylogeny matrix of `NetworkEstimator(; sep = PathLength())` has ones at each entry off the diagonal. This is the opposite of the default `n = 1` of [`HopCount`](@ref), which gives the smallest neighbourhood. State a numeric `dmax` to select fewer pairs.

# The budget can be a rule

`dmax` also accepts a [`PathLengthAlgorithm`](@ref) or a bare `Function`. [`resolve_separation`](@ref) calls it as `dmax(nte, X, g; dims = dims, kwargs...)`, where `g` is the structure that the consumer built. A rule helps a caller who cannot state a number. [`PathLengthQuantile`](@ref) gives the budget that relates a stated fraction of the reachable pairs. A caller can choose that fraction, and it means the same in each fold of a cross-validation and in each subproblem of a meta optimiser.

A fixed `dmax` and a rule keep different quantities constant. A fixed `dmax` keeps the radius constant, and the number of related pairs changes with the graph. A quantile rule keeps the number constant, and the radius changes. Neither keeps both constant across folds, because each fold fits a new graph.

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

[`separation_matrix`](@ref) does not repair the sentinel of the routine that it calls. This test tells a measured separation from a missing one. It is one generic method on [`AbstractSeparationAlgorithm`](@ref), because one expression covers the two shipped sentinels. A member whose routine reports a different sentinel overrides this method, and it inherits [`is_related`](@ref) unchanged.

# The test is not `isfinite` alone, and not `typemax` alone

Each clause rejects a sentinel that the other clause accepts.

  - `isfinite` is `true` for every `Integer`. Alone, it accepts the `typemax(Int)` of [`HopCount`](@ref), and [`ReciprocalDecay`](@ref) overflows at that value.
  - `typemax` of a `Float64` is `Inf`, so the comparison also rejects the sentinel of [`PathLength`](@ref). `isfinite` rejects a `NaN`. No shipped path gives a `NaN`, and a `NaN` compares `false` with every budget.

# Algorithm

 1. Test `isfinite(d)`. This rejects a `NaN` and the `Inf` of [`PathLength`](@ref).
 2. Compare `d` with `typemax(typeof(d))`. This rejects the `typemax(Int)` of [`HopCount`](@ref), which step 1 accepts.
 3. Return `reachable`, the conjunction of the two tests. `&&` short-circuits, so step 2 runs only on a finite `d`.

# Arguments

  - `sep`: Separation algorithm. The shipped members do not read it. An extension whose routine reports a different sentinel dispatches on it.
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

This is the one function that applies the budget to an entry of a separation matrix. A related pair is reachable, and it is not further apart than `dmax`. Each consumer that compares a separation with a budget calls this function and writes no comparison of its own. The [`PathLength`](@ref) branch of [`phylogeny_matrix`](@ref) and the [`Proximity`](@ref) method of [`phylogeny_features`](@ref) call it.

# The reachability test comes first

`d <= dmax` alone is not sufficient. It rejects the two shipped sentinels only because of the two shipped budgets. [`separation_budget`](@ref) clamps a [`PathLength`](@ref) budget to the observed finite diameter, and a [`HopCount`](@ref) budget has a cap far below `typemax(Int)`. [`is_reachable`](@ref) makes the rejection part of the predicate. A budget at the largest value of its unit then still excludes an unreachable pair.

# The caller must still short-circuit

A consumer that scores the separation must put the score inside a branch that short-circuits. Write `is_related(...) ? separation_decay(...) : zero(...)`, and do not use `ifelse`. An `ifelse` evaluates the two branches. [`ReciprocalDecay`](@ref) overflows `1 + d` at `typemax(Int)`, and a fractional `power` then gives a `DomainError`. The predicate holds the rule, and the call site must not evaluate the score of an unrelated pair.

# Algorithm

 1. Call [`is_reachable`](@ref) on `sep` and `d`, which gives the reachability test. This test runs first, so the function rejects a sentinel at each value of `dmax`.
 2. Compare `d` with `dmax`, which gives the budget test.
 3. Return `related`, the conjunction of the two tests. `&&` short-circuits, so step 2 runs only on a reachable `d`.

# Arguments

  - `sep`: Separation algorithm, which the function passes to [`is_reachable`](@ref).
  - `d`: One entry of a separation matrix from [`separation_matrix`](@ref).
  - `dmax`: Separation budget in scope, from [`separation_budget`](@ref). It has the units that `sep` measures in, so the two arguments come together.

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

A separation decay changes a separation `d >= 0` into a score. The separation is how far apart two assets are in the structure that the caller reads. [`separation_decay`](@ref) applies the decay. The family is open. For a different fall-off, define a member and a [`separation_decay`](@ref) method for it. [`AbstractSimilarityMatrixAlgorithm`](@ref) and [`distance_to_similarity`](@ref) have the same pattern.

`d` is a real separation and not an integer hop count. So one family serves an unweighted graph, where a hop count is a real with an integer value, and each structure whose separation is continuous.

# The contract

  - `f` has a value for every `d >= 0`.
  - `f(0) > 0`, and `f(0)` is the largest value. A decay must put an asset at the top of its own scale. If not, the result is a structural equivalence matrix, not a proximity matrix. See "Why the diagonal includes self" of [`PhylogenyPanel`](@ref).
  - `f` is monotone non-increasing in `d`.
  - The contract does not ask `f` to reach zero. The budget truncates, and the decay does not. The consumer applies the budget, which is [`separation_budget`](@ref) of the [`AbstractSeparationAlgorithm`](@ref) in scope. The decay shapes the fall-off inside the budget. An exponential never reaches zero, so one parameter cannot set the budget and the fall-off.
  - `f(d) >= 0` for `0 <= d <= dmax`. The score of an unreachable pair is `0`. A negative score inside the budget puts a reachable pair below an unreachable pair on the scale of the producer.

The last clause does not say that a signed score is always wrong. The feature matrix accepts signed values by decision, and [`assert_metric_domain`](@ref) checks non-negativity for each metric at the consumer. The clause applies to the producer alone. It asks for non-negativity and not strict positivity, because a decay that falls to zero says that the pair is unrelated. An unreachable pair says the same.

The clause applies inside the budget alone, for two reasons. First, the sign outside the budget has no effect, because the consumer tests [`is_related`](@ref) first and does not evaluate the decay there. Second, the default of the family breaks the wider clause. [`LinearDecay`](@ref) is zero at `d = dmax + 1` and negative above it. A clause on all `d >= 0` needs a `max(0, ⋅)` floor, which is a second truncation inside the decay.

So a zero in the feature matrix means that the pair is unrelated. The pair is unreachable, it is outside the budget, or the decay fell to zero. A consumer of the feature matrix cannot tell these cases apart. No shipped member gives zero inside the budget. For the shipped members, a zero means unreachable or outside the budget.

# The budget is an argument, not a field

[`separation_decay`](@ref) takes the budget in scope as its third argument, `dmax`. A member can ignore it, and only [`LinearDecay`](@ref) reads it, to set `f(0)`. The member does not hold the budget, so the budget and the decay cannot disagree. The [`AbstractSeparationAlgorithm`](@ref) is the one place that holds the budget. A decay algorithm cannot see the budget when the caller constructs it. [`ExponentialDecay`](@ref) sets the contrast between an asset and its neighbour with its own rate, and adds no second truncation inside the decay.

# Enforcement

The fallback [`assert_separation_decay`](@ref) on this type probes a member and enforces the contract. The shipped members satisfy the contract by construction, and their method of that function does nothing. So an extension that defines no method of [`assert_separation_decay`](@ref) gets the probe.

# Related

  - [`LinearDecay`](@ref)
  - [`ExponentialDecay`](@ref)
  - [`ReciprocalDecay`](@ref)
  - [`NoDecay`](@ref)
  - [`separation_decay`](@ref)
  - [`assert_separation_decay`](@ref)
  - [`AbstractSeparationAlgorithm`](@ref)
  - [`Proximity`](@ref)
  - [`PhylogenyPanel`](@ref)
"""
abstract type AbstractSeparationDecayAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Separation decay that falls linearly to the edge of the budget.

# Mathematical definition

```math
\\begin{align}
f(d) &= d_{\\mathrm{max}} + 1 - d\\,,
\\end{align}
```

Where:

  - $(math_dict[:d_sep])
  - $(math_dict[:dmax_sep])

This is the default member, and the only member that reads the budget. A direct neighbour scores ``d_{\\mathrm{max}}``, and the asset itself scores ``d_{\\mathrm{max}} + 1``.

The budget truncates, and the decay does not, so the decay needs no `max(0, ⋅)` floor. On `0 <= d <= dmax` the score is strictly positive, and its smallest value is `1`.

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

Separation decay that falls exponentially.

# Mathematical definition

```math
\\begin{align}
f(d) &= e^{-\\lambda d}\\,,
\\end{align}
```

Where:

  - $(math_dict[:d_sep])
  - ``\\lambda``: Rate of the fall-off, `rate`.

The score of the asset itself is `f(0) = 1`. `rate` sets the contrast between an asset and its neighbour, and the budget has no effect on it. A large `rate` makes the score fall fast. The budget states only how far apart a related pair can be.

The parameter is a rate, not a retention per step. ``\\rho^d`` and ``e^{-\\lambda d}`` are the same function when ``\\lambda = -\\log\\rho``. A retention per step applies to integer steps, but the separation is a real number. [`ExponentialSimilarity`](@ref) and [`ExpGerberIQDecay`](@ref) also use a rate. The rate needs one bound, `rate > 0`, for a monotone decay.

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

Separation decay that falls as a power of the separation.

# Mathematical definition

```math
\\begin{align}
f(d) &= \\left(1 + d\\right)^{-p}\\,,
\\end{align}
```

Where:

  - $(math_dict[:d_sep])
  - ``p``: Exponent of the fall-off, `power`.

Its tail is heavier than the tail of [`ExponentialDecay`](@ref), so a distant asset keeps a small score.

The `1 +` keeps the classical inverse-distance weight finite at ``d = 0``. It also sets `f(0) = 1`, which is the top of the scale of [`ExponentialDecay`](@ref).

The library does not use the form ``(1 + d^p)^{-1}``. That form gives ``f(1) = 1/2`` for every `p`. A larger `p` raises the score of a pair nearer than ``d = 1``, and lowers the score of a pair further away. From `p = 1` to `p = 3`, the score at ``d = 0.5`` goes from `0.6667` to `0.8889`, and the score at ``d = 2`` goes from `0.3333` to `0.1111`. So in that form, `p` cannot make the decay steeper below ``d = 1``. In the form above, a larger `p` lowers the score at every ``d > 0``.

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

# The budget still truncates

The name is about the fall-off alone. A pair outside the budget scores `0`, because the budget truncates and the decay does not. See "The contract" of [`AbstractSeparationDecayAlgorithm`](@ref). So the result is the indicator of the neighbourhood that the budget selects, not a matrix of ones.

Under [`HopCount`](@ref), [`Proximity`](@ref) with this decay gives the indicator of the `n`-hop neighbourhood. Under [`PathLength`](@ref), it gives the indicator of the ball of radius `dmax`. Neither indicator needs a type of its own.

It is the only member that is not strictly decreasing. The contract asks for a monotone non-increasing function, and a constant is one.

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

This function is the extension contract of [`AbstractSeparationDecayAlgorithm`](@ref). A new member is a struct and one method of this function.

# The method the call selects

This function selects a method and runs no step of its own. It dispatches on `dk`, and the selected method evaluates one closed form. The `# Mathematical definition` of each member states its form, and this docstring does not repeat it.

| `dk`                       | The fall-off it selects            | Reads `dmax` |
|:-------------------------- |:---------------------------------- |:------------ |
| [`LinearDecay`](@ref)      | Linear to the edge of the budget   | yes          |
| [`ExponentialDecay`](@ref) | Exponential in `rate`              | no           |
| [`ReciprocalDecay`](@ref)  | A power of `1 + d`, set by `power` | no           |
| [`NoDecay`](@ref)          | Flat                               | no           |

[`LinearDecay`](@ref) is the only member that reads the budget. So the other three methods leave their third argument unnamed.

# Arguments

  - `dk`: Separation decay algorithm.
  - `d`: Separation between two assets, `d >= 0`. It is a real number, so it can be a weighted path length or a hop count.
  - `dmax`: Separation budget in scope. Only [`LinearDecay`](@ref) reads it, and the other members ignore it.

# Validation

This function does not check the contract, because it runs inside an `assets × assets` loop. A caller probes the decay once before the loop with [`assert_separation_decay`](@ref).

# Returns

  - `f::Number`: Score for the separation. It is non-negative for `0 <= d <= dmax`. Above the budget the contract does not set the sign, and [`LinearDecay`](@ref) is negative there. The consumer tests the budget first, so it does not call this function there.

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

Check that a separation decay satisfies its contract over the separations that the caller will score.

The fallback probes the decay. It evaluates [`separation_decay`](@ref) over `ds` and checks four properties:

  - Each value is finite.
  - `f(0)` is strictly positive, and it is the largest value.
  - The values are monotone non-increasing.
  - No value is negative.

The four shipped members satisfy the contract by construction, and their method does nothing. So the probe costs nothing for a shipped member, and it checks each extension.

The probe is cheap where the library uses it, because `ds` is small and the loop after the probe is large. [`Proximity`](@ref) passes `0:dmax`. Under [`HopCount`](@ref), this range holds every separation that the `assets × assets` loop can score, in `dmax + 1` evaluations. Under a budget that is not an integer, the same range is a sample with a step of one. A continuum permits only a sample, and the checks below need no more.

The non-negativity check has one more evaluation at `d = dmax`, also when `ds` does not contain `dmax`. It is the counterpart of the evaluation of `f(0)` before the loop. If the decay is monotone, `f(dmax) >= 0` gives `f(d) >= 0` for every `d` in `[0, dmax]`. So the check holds over the continuum, also when `ds` is a sample, as it is for weighted path lengths. The endpoint does not help the monotonicity check, which stays a sample.

# Algorithm

These are the steps of the probe in the fallback. The method on the four shipped members runs none of them.

 1. Score the separation `zero(dmax)` with [`separation_decay`](@ref), which gives `f0`, the top of the scale.
 2. Check that `f0` is finite and strictly positive.
 3. If `ds` is not sorted, sort it, so that step 5 reads the separations in increasing order.
 4. Set `fp`, the score of the previous separation, to `f0`.
 5. For each `d` of `ds`, score it with [`separation_decay`](@ref), which gives `f`. Check that `f` is finite and not more than `f0`. Check that `f` is not more than `fp`. Check that `f` is non-negative. Set `fp` to `f`.
 6. Score `dmax`, which gives `fmax`, and check that `fmax` is non-negative. This is the endpoint that the paragraph above states. Step 5 does not always reach it.
 7. Return `nothing`.

# Arguments

  - `dk`: Separation decay algorithm.
  - `ds`: Separations to probe, in any order. Each separation must be in `[0, dmax]`, because the loop after the probe scores no separation outside the budget.
  - `dmax`: Separation budget in scope. The function passes it to [`separation_decay`](@ref), and also probes it as an endpoint.

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
