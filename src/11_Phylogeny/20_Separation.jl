"""
    separation_graph(sep::HopCount, G::Graphs.AbstractGraph)
    separation_graph(sep::HopCount, nte::AbstractNetworkEstimator, X::MatNum;
                     dims::Int = 1, kwargs...)
    separation_graph(sep::PathLength, nte::AbstractNetworkEstimator, X::MatNum;
                     dims::Int = 1, kwargs...)

Build the structure a separation measures over.

One third of the extension contract of [`AbstractSeparationAlgorithm`](@ref); [`separation_matrix`](@ref) and [`separation_budget`](@ref) are the other two. It exists as a kernel of its own so that **the structure is built once per consumer call**: [`separation_matrix`](@ref) and [`resolve_separation`](@ref) both take the graph, and a consumer that needs the separations *and* a budget rule answered would otherwise build the same structure twice, through two estimator-taking kernels that each derive it privately.

# What each member measures over

  - [`HopCount`](@ref): [`calc_adjacency`](@ref)'s structure as a `Graphs.SimpleGraph`. Binary, because a hop count ignores the weights and its consumers do not — [`_phylogeny_matrix`](@ref) reads `Graphs.adjacency_matrix` off this graph for a power sum, where a weight would make `A^i` sum *products of distances* instead of counting walks.
  - [`PathLength`](@ref): [`calc_distance_weighted_graph`](@ref)'s structure. The same edge set, weighted by **distance** on either branch, because a shortest path over the PMFG's similarities seeks the route through the weakest links.

# Two entry points, and why only the hop count gets the graph-taking one

The estimator-taking methods derive the structure from `X`. The graph-taking method is for a caller that already holds it: both [`clusterise`](@ref) methods build the structure from the selecting quantity they already paid for, through [`calc_weighted_adjacency_graph`](@ref)'s own two-argument entry point, and would otherwise re-derive the distance — `98%` of `clusterise`'s runtime under [`VariationInfoDistance`](@ref) — to answer a budget rule.

[`PathLength`](@ref) has **no** graph-taking method, and cannot: a graph carries no polarity tag, so `G` is a distance-weighted structure on the tree branch and a similarity-weighted one on the PMFG branch, and nothing in the argument distinguishes them. Handing the PMFG's similarities to a shortest path returns an answer instead of raising — see [`calc_distance_weighted_graph`](@ref). The hop count is exempt because it discards the weights.

# Algorithm

 1. Build the structure `X` implies, through the branch that `sep` selects.

      + [`HopCount`](@ref): [`calc_adjacency`](@ref)'s binary matrix.
      + [`PathLength`](@ref): [`calc_distance_weighted_graph`](@ref)'s distance-weighted structure, which is already the answer.

 2. Rebuild a [`HopCount`](@ref)'s structure as a `Graphs.SimpleGraph`, which discards the weights. The graph-taking method is handed `G` and starts here.

# Arguments

  - `sep`: Separation algorithm. Dispatched on, and its budget is not read — a member whose budget is still a **rule** measures over the same structure as one whose budget is a number, which is what lets [`resolve_separation`](@ref) be handed this graph.
  - `G`: Network structure a caller already holds, in either polarity.
  - $(arg_dict[:nte])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `g::Graphs.AbstractGraph`: The structure `sep` measures over.

# Related

  - [`AbstractSeparationAlgorithm`](@ref)
  - [`HopCount`](@ref)
  - [`PathLength`](@ref)
  - [`separation_matrix`](@ref)
  - [`resolve_separation`](@ref)
  - [`calc_adjacency`](@ref)
  - [`calc_distance_weighted_graph`](@ref)
  - [`calc_weighted_adjacency_graph`](@ref)
"""
function separation_graph end
function separation_graph(::HopCount, G::Graphs.AbstractGraph)
    # A hop count ignores the weights, but `_phylogeny_matrix`'s power sum does not, so the
    # structure is handed on binarised rather than as it arrived.
    return Graphs.SimpleGraph(G)
end
function separation_graph(::HopCount, nte::AbstractNetworkEstimator, X::MatNum;
                          dims::Int = 1, kwargs...)
    return Graphs.SimpleGraph(calc_adjacency(nte, X; dims = dims, kwargs...))
end
function separation_graph(::PathLength, nte::AbstractNetworkEstimator, X::MatNum;
                          dims::Int = 1, kwargs...)
    return calc_distance_weighted_graph(nte, X; dims = dims, kwargs...)
end
"""
    separation_matrix(sep::HopCount, g::Graphs.AbstractGraph)
    separation_matrix(sep::PathLength, g::Graphs.AbstractGraph)
    separation_matrix(sep::AbstractSeparationAlgorithm, nte::AbstractNetworkEstimator,
                      X::MatNum; dims::Int = 1, kwargs...)

Compute the dense `assets × assets` matrix of separations under a separation algorithm.

One third of the extension contract of [`AbstractSeparationAlgorithm`](@ref); [`separation_graph`](@ref) and [`separation_budget`](@ref) are the other two.

# The graph-taking form is the interface

The separations depend on the structure alone, so `separation_matrix(sep, g)` is where each member's method lives and the estimator-taking form is a wrapper that calls [`separation_graph`](@ref) first. A consumer holding a graph — because it built one for [`resolve_separation`](@ref), or because a test chose one — enters at the graph, and pays for one structure rather than two.

The wrapper is generic rather than per-member: it is [`separation_graph`](@ref) that knows which structure the member reads, so there is nothing left for a member to say here.

# The unreachable sentinel

An unreachable pair carries whatever sentinel the underlying routine uses, **not** a repaired value: `Graphs.gdistances` reports `typemax(Int)` for [`HopCount`](@ref), and `Graphs.floyd_warshall_shortest_paths` reports `typemax(T)` for [`PathLength`](@ref), which on the `Float64` weights it is handed is `Inf`. A consumer therefore reads an entry through [`is_related`](@ref) rather than comparing it against the budget itself, and keeps the *evaluation* of anything it scores the entry with inside a short-circuiting branch — an `ifelse` evaluates both arms, and [`ReciprocalDecay`](@ref) overflows `1 + d` at `typemax(Int)`, which a fractional `power` turns into a `DomainError` rather than a discarded number.

# The two shipped members read the same structure differently

[`HopCount`](@ref) counts the edges of the binarised structure; [`PathLength`](@ref) sums the distances along them. Which structure each reads is [`separation_graph`](@ref)'s answer, not this function's. All-pairs shortest paths come from one `floyd_warshall_shortest_paths` call rather than a Dijkstra per vertex — measured about **7 times faster** on this shape, and within about `1.3` times of the breadth-first loop the hop count uses.

# Algorithm

 1. Build the structure with [`separation_graph`](@ref). The graph-taking methods are handed `g` and start at step 2.

 2. Measure every pair over that structure, through the branch that `sep` selects.

      + [`HopCount`](@ref): one `Graphs.gdistances` breadth-first traversal per vertex, whose answer becomes that vertex's column of a dense `Matrix{Int}`.
      + [`PathLength`](@ref): one `Graphs.floyd_warshall_shortest_paths` call over the whole graph, whose `dists` field is the matrix.

# Arguments

  - `sep`: Separation algorithm. Its budget is not read; a member whose budget is still a rule measures the same separations.
  - `g`: Structure to measure over, from [`separation_graph`](@ref).
  - `nte`: Network estimator. On the wrapper only, where the structure is derived from `X` on every call.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments forwarded to the underlying phylogeny routines.

# Returns

  - `d::Matrix`: Square matrix of separations. `d[i, k]` is the separation between assets `i` and `k`, `d[i, i]` is zero, and an unreachable pair carries the sentinel above.

# Related

  - [`AbstractSeparationAlgorithm`](@ref)
  - [`HopCount`](@ref)
  - [`PathLength`](@ref)
  - [`separation_graph`](@ref)
  - [`separation_budget`](@ref)
  - [`is_related`](@ref)
  - [`Proximity`](@ref)
"""
function separation_matrix end
function separation_matrix(::HopCount, g::Graphs.AbstractGraph)
    d = Matrix{Int}(undef, Graphs.nv(g), Graphs.nv(g))
    for v in Graphs.vertices(g)
        @inbounds d[:, v] = Graphs.gdistances(g, v)
    end
    return d
end
function separation_matrix(::PathLength, g::Graphs.AbstractGraph)
    return Graphs.floyd_warshall_shortest_paths(g).dists
end
# One wrapper for the whole family: which structure the member measures over is
# `separation_graph`'s answer, so there is nothing per-member left to say here.
function separation_matrix(sep::AbstractSeparationAlgorithm, nte::AbstractNetworkEstimator,
                           X::MatNum; dims::Int = 1, kwargs...)
    return separation_matrix(sep, separation_graph(sep, nte, X; dims = dims, kwargs...))
end
"""
    separation_budget(sep::HopCount, nte::AbstractNetworkEstimator, d::MatNum)
    separation_budget(sep::PathLength, nte::AbstractNetworkEstimator, d::MatNum)

Resolve the separation budget in scope: the separation beyond which a pair counts as unrelated.

One third of the extension contract of [`AbstractSeparationAlgorithm`](@ref); [`separation_graph`](@ref) and [`separation_matrix`](@ref) are the other two. Split from `separation_matrix` because a consumer needs the budget on its own — to probe a decay before entering the `assets × assets` loop, or to threshold a matrix it already holds.

# The separations are passed in, not recomputed

`d` is the matrix [`separation_matrix`](@ref) produced, so a member whose budget is *observed* rather than configured — the diameter of what the graph actually turned out to be — can read it without a second traversal. That is why the argument is the **matrix** and not a diameter: finding the largest finite entry is itself the `assets²` reduction, so passing a diameter would charge [`HopCount`](@ref) for one it ignores. Handing over `d` pushes the reduction into [`PathLength`](@ref), the member that wants it.

`nte` is inert for what ships: it is the channel through which an extension budget can see the estimator that owns it. Inert arguments have precedent here — [`separation_decay`](@ref)'s `dmax` is read by only one of five members.

# The observed diameter is a ceiling, not only a default

[`PathLength`](@ref) clamps a chosen `dmax` to the observed diameter as well as substituting the diameter for `nothing`. The clamp **truncates nothing** — no pair sits beyond the diameter — so it is a scale-top correction and is visible only through [`LinearDecay`](@ref), the one decay reading the budget. Without it, `dmax = 100` on a graph of diameter `3.5` would flatten the scores towards a constant while forbidding no pair at all.

# Algorithm

 1. Refuse a separation whose budget is still a rule. A [`HopCount`](@ref) carrying a [`HopCountAlgorithm`](@ref), and a [`PathLength`](@ref) carrying a [`PathLengthAlgorithm`](@ref), each throw here.

 2. Answer the budget, through the branch that `sep` selects.

      + [`HopCount`](@ref): `sep.n`, the number of hops the caller stated. `d` is not read.
      + [`PathLength`](@ref): walk `d`, keep the largest entry [`is_reachable`](@ref) admits, and call it `delta`, the observed diameter. Answer `delta` when `sep.dmax` is `nothing`, and the smaller of `sep.dmax` and `delta` otherwise.

# Arguments

  - `sep`: Separation algorithm.
  - `nte`: Network estimator that owns `sep`. **Inert** for the shipped members.
  - `d`: Separation matrix from [`separation_matrix`](@ref). **Inert** for [`HopCount`](@ref), whose budget is configured rather than observed; read by [`PathLength`](@ref), whose budget is capped by what the graph turned out to be.

# Validation

  - Throws an `ArgumentError` if `sep` carries a budget rule rather than a value. Call [`resolve_separation`](@ref) first.

# Returns

  - `dmax::Number`: Separation budget. Stated in the units `sep` measures in — hops for [`HopCount`](@ref), the distance estimator's units for [`PathLength`](@ref) — so it is only ever compared against entries of `d`.

# Related

  - [`AbstractSeparationAlgorithm`](@ref)
  - [`HopCount`](@ref)
  - [`PathLength`](@ref)
  - [`separation_matrix`](@ref)
  - [`separation_decay`](@ref)
  - [`Proximity`](@ref)
"""
function separation_budget end
function separation_budget(sep::HopCount, ::AbstractNetworkEstimator, ::MatNum)::Number
    return sep.n
end
function separation_budget(sep::PathLength, ::AbstractNetworkEstimator, d::MatNum)::Number
    # The diameter of what the graph turned out to be. The sentinel is excluded rather than
    # repaired: an unreachable pair is not a long one, and taking it as the diameter would
    # make the budget `Inf`, which `LinearDecay` scores `Inf` at every separation. The
    # exclusion is `is_reachable`'s and not an inline `isfinite`, which is true of every
    # `Integer` and would take a `typemax(Int)` sentinel for the diameter.
    delta = zero(eltype(d))
    for dij in d
        if is_reachable(sep, dij) && dij > delta
            delta = dij
        end
    end
    return isnothing(sep.dmax) ? delta : min(sep.dmax, delta)
end
# An unresolved budget is a rule, and a rule cannot be compared against an entry of `d`.
# Returning it would put a function where every caller expects a number, so it is refused
# here rather than three frames later inside `d .<= dmax`.
function separation_budget(sep::HopCount{<:HopCountRule}, ::AbstractNetworkEstimator,
                           ::MatNum)
    return throw(ArgumentError("separation_budget needs a resolved separation, and this HopCount carries the rule $(typeof(sep.n)) in `n`.\nCall resolve_separation(sep, nte, X) first; every shipped consumer of a network already does."))
end
function separation_budget(sep::PathLength{<:PathLengthRule}, ::AbstractNetworkEstimator,
                           ::MatNum)
    return throw(ArgumentError("separation_budget needs a resolved separation, and this PathLength carries the rule $(typeof(sep.dmax)) in `dmax`.\nCall resolve_separation(sep, nte, X) first; every shipped consumer of a network already does."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Quantile of the reachable off-diagonal entries of a separation matrix.

The population is the pairs a budget can be *about*: the diagonal is zero by construction and an unreachable pair carries [`separation_matrix`](@ref)'s sentinel, so neither is a separation. It is also the population [`phylogeny_matrix`](@ref) selects from, which is what makes `q` read as the fraction of pairs the resulting budget relates.

# The sentinel test is the family's

The population excludes an unreachable pair through [`is_reachable`](@ref) rather than through a test written out here. `isfinite` alone would not do it: it is `true` for every `Integer`, so it admits [`HopCount`](@ref)'s `typemax(Int)`.

# Algorithm

 1. Collect the off-diagonal entries of `d` that [`is_reachable`](@ref) admits, giving `v`, the population of the quantile.
 2. Throw an `ArgumentError` when `v` is empty.
 3. Take the `q`-quantile of `v` with `Statistics.quantile`.

# Arguments

  - `sep`: Separation algorithm the matrix was measured under, forwarded to [`is_reachable`](@ref).
  - `d`: Separation matrix from [`separation_matrix`](@ref).
  - `q`: Quantile in `[0, 1]`.

# Validation

  - Throws an `ArgumentError` if no off-diagonal entry of `d` is reachable, because a budget cannot be placed at a quantile of an empty population.

# Returns

  - `dmax::Number`: The `q`-quantile of the reachable off-diagonal separations.

# Related

  - [`HopCountQuantile`](@ref)
  - [`PathLengthQuantile`](@ref)
  - [`separation_matrix`](@ref)
  - [`is_reachable`](@ref)
"""
function separation_quantile(sep::AbstractSeparationAlgorithm, d::MatNum, q::Number)::Number
    v = [d[i, j] for j in axes(d, 2)
         for i in axes(d, 1) if i != j && is_reachable(sep, d[i, j])]
    @argcheck(!isempty(v),
              ArgumentError("a separation quantile needs at least one reachable pair of distinct assets, and this $(size(d, 1))-asset structure has none."))
    return Statistics.quantile(v, q)
end
"""
$(DocStringExtensions.TYPEDEF)

Places the hop budget at a quantile of the observed hop separations.

The shipped [`HopCountAlgorithm`](@ref). `HopCount(; n = HopCountQuantile(; q = 0.25))` asks for the hop budget that relates about a quarter of the reachable pairs, instead of naming a number of hops that was right for one universe.

# What it holds still

A stated `n` holds the **number of hops** still and lets the related-pair count move with the graph. This rule holds the **count** still — about `q` of the reachable pairs — and lets the number of hops move. On a cross-validation fold or a subproblem of a meta optimiser the second is usually what the caller meant, because the constraint strength a downstream consumer feels is the cardinality, not the hop number.

# The rounding is where the two stop matching

A hop count is an `Integer` and the quantile is not, so the budget is rounded to the nearest hop. The related-pair count therefore lands *near* `q` rather than on it, and on a small graph the shells are coarse enough that it can miss by a lot — a hop budget can only ever select one of a handful of cardinalities. [`PathLengthQuantile`](@ref) has no such step and hits `q` closely; that is the sharpest practical difference between the two separations.

# It pays for a traversal, but not for a structure

Resolving this rule runs [`separation_matrix`](@ref) once, which the hop-ball branch of [`_phylogeny_matrix`](@ref) does not otherwise do — it walks matrix powers instead. A dynamic budget costs one all-pairs traversal that a stated one does not.

It does **not** cost a second structure. The rule is handed the graph its consumer already built, through [`separation_graph`](@ref), so the distance derivation — `98%` of [`clusterise`](@ref)'s runtime under [`VariationInfoDistance`](@ref) — is paid once per consumer call whether the budget is a rule or a number.

# Algorithm

The steps below are the call operator's, which [`resolve_separation`](@ref) invokes.

 1. Measure the hop separations over `g` with [`separation_matrix`](@ref), giving `d`. A bare [`HopCount`](@ref) is the probe, because that kernel dispatches on the separation's type and reads no field of it.
 2. Take the `q`-quantile of the reachable off-diagonal entries of `d` with [`separation_quantile`](@ref).
 3. Round that quantile to the nearest `Int`, giving the hop budget. The smallest off-diagonal hop separation is `1`, so the quantile never falls below one and the round never has to be clamped up to [`HopCount`](@ref)'s floor.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HopCountQuantile(;
        q::Number = 0.25
    ) -> HopCountQuantile

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:sepq])

# Examples

```jldoctest
julia> HopCountQuantile()
HopCountQuantile
  q ┴ Float64: 0.25
```

# Related

  - [`HopCountAlgorithm`](@ref)
  - [`HopCount`](@ref)
  - [`PathLengthQuantile`](@ref)
  - [`resolve_separation`](@ref)
  - [`separation_quantile`](@ref)
"""
@concrete struct HopCountQuantile <: HopCountAlgorithm
    """
    $(field_dict[:sepq])
    """
    q
    function HopCountQuantile(q::Number)
        @argcheck(zero(q) <= q <= one(q), DomainError(q, "q must be in [0, 1]"))
        return new{typeof(q)}(q)
    end
end
function HopCountQuantile(; q::Number = 0.25)::HopCountQuantile
    return HopCountQuantile(q)
end
function (alg::HopCountQuantile)(::AbstractNetworkEstimator, ::MatNum,
                                 g::Graphs.AbstractGraph; kwargs...)::Integer
    # `separation_matrix` dispatches on the separation's *type* and reads no field of it, so
    # a bare `HopCount()` here is a probe saying "measure in hops" rather than a budget. `g`
    # is already the structure a hop count measures over, so nothing is rebuilt.
    d = separation_matrix(HopCount(), g)
    # The smallest off-diagonal hop separation is `1`, so the quantile never falls below one
    # and the round never has to be clamped up to `HopCount`'s floor.
    return round(Int, separation_quantile(HopCount(), d, alg.q))
end
"""
$(DocStringExtensions.TYPEDEF)

Places the path-length budget at a quantile of the observed path separations.

The shipped [`PathLengthAlgorithm`](@ref), and the direct answer to [`PathLength`](@ref)'s own complaint that nobody has an intuition for a summed path in the units an [`AbstractDistanceEstimator`](@ref) emits. `dmax = 0.37` is not a number a caller can reason about; "the budget that relates a quarter of the reachable pairs" is.

# What it holds still

A stated `dmax` holds the **radius** still and lets the related-pair count move with the graph. This rule holds the **count** still and lets the radius move. Both are refitted per fold, so neither is stable in both senses at once — the choice is which of the two a downstream consumer is sensitive to, and for [`SemiDefinitePhylogeny`](@ref) and [`IntegerPhylogeny`](@ref) the constraint strength *is* the cardinality.

# It reaches the cardinalities a hop count cannot

This is where the radius ball's one real gain becomes reachable by name. A hop budget steps through a handful of shell cardinalities and cannot stop between them; `q` is continuous, so `PathLengthQuantile(; q = 0.3)` asks for a cardinality directly and lands on it closely.

# Algorithm

The steps below are the call operator's, which [`resolve_separation`](@ref) invokes.

 1. Measure the path separations over `g` with [`separation_matrix`](@ref), giving `d`. A bare [`PathLength`](@ref) is the probe, for the same reason as on the hop rule.
 2. Take the `q`-quantile of the reachable off-diagonal entries of `d` with [`separation_quantile`](@ref). That quantile is the radius budget, and it takes no rounding.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PathLengthQuantile(;
        q::Number = 0.25
    ) -> PathLengthQuantile

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:sepq])

# Examples

```jldoctest
julia> PathLengthQuantile()
PathLengthQuantile
  q ┴ Float64: 0.25
```

# Related

  - [`PathLengthAlgorithm`](@ref)
  - [`PathLength`](@ref)
  - [`HopCountQuantile`](@ref)
  - [`resolve_separation`](@ref)
  - [`separation_quantile`](@ref)
"""
@concrete struct PathLengthQuantile <: PathLengthAlgorithm
    """
    $(field_dict[:sepq])
    """
    q
    function PathLengthQuantile(q::Number)
        @argcheck(zero(q) <= q <= one(q), DomainError(q, "q must be in [0, 1]"))
        return new{typeof(q)}(q)
    end
end
function PathLengthQuantile(; q::Number = 0.25)::PathLengthQuantile
    return PathLengthQuantile(q)
end
function (alg::PathLengthQuantile)(::AbstractNetworkEstimator, ::MatNum,
                                   g::Graphs.AbstractGraph; kwargs...)::Number
    # A bare `PathLength()` is the probe here, for the same reason as on the hop rule.
    d = separation_matrix(PathLength(), g)
    return separation_quantile(PathLength(), d, alg.q)
end
"""
    resolve_separation(sep::AbstractSeparationAlgorithm, nte::AbstractNetworkEstimator,
                       X::MatNum, g::Graphs.AbstractGraph; dims::Int = 1, kwargs...)
    resolve_separation(sep::HopCount{<:HopCountRule}, nte::AbstractNetworkEstimator,
                       X::MatNum, g::Graphs.AbstractGraph; dims::Int = 1, kwargs...)
    resolve_separation(sep::PathLength{<:PathLengthRule}, nte::AbstractNetworkEstimator,
                       X::MatNum, g::Graphs.AbstractGraph; dims::Int = 1, kwargs...)
    resolve_separation(sep::AbstractSeparationAlgorithm, nte::AbstractNetworkEstimator,
                       X::MatNum; dims::Int = 1, kwargs...)
    resolve_separation(sep::Union{<:HopCount{<:HopCountRule},
                                  <:PathLength{<:PathLengthRule}},
                       nte::AbstractNetworkEstimator, X::MatNum; dims::Int = 1, kwargs...)

Replace a separation whose budget is a **rule** by one whose budget is a value.

The fourth kernel of [`AbstractSeparationAlgorithm`](@ref), and the only one an extension does not have to write: the fallback on the abstract type returns `sep` unchanged, so a member whose budget is already a number passes through at no cost and gains nothing to maintain.

# It is called by the consumer, not by the other kernels

Every shipped consumer of a network resolves `nte.sep` **first** and passes the resolved separation to [`separation_matrix`](@ref) and [`separation_budget`](@ref) — [`phylogeny_matrix`](@ref), both [`clusterise`](@ref) methods, and [`phylogeny_features`](@ref) for [`Proximity`](@ref).

The alternative was to resolve inside [`separation_budget`](@ref), and it does not work: that kernel takes the separation **matrix** rather than the data, deliberately, so that [`HopCount`](@ref) never pays for a diameter reduction it ignores. A rule needs the structure, which is the one thing the budget kernel does not have. So `separation_budget` refuses an unresolved separation instead, and this kernel is where the structure is still in hand.

# The rule is handed the structure, not asked to build one

`g` is [`separation_graph`](@ref)'s structure, and the graph-taking methods are the interface: a consumer builds once, resolves the rule against that graph, and measures the separations over the same graph. The rule reads what it needs through `separation_matrix(sep, g)`.

The estimator-taking methods are wrappers, and the resolved case has one of its own so that **a stated budget builds nothing at all**. Dispatching the wrapper on the rule-carrying parameterisation is what keeps that true — a single generic wrapper would derive a structure before discovering that the fallback ignores it.

# The return check is a run-time one, and it has to be

A functor's return type is not part of its signature, so a [`HopCountAlgorithm`](@ref) cannot promise an `Integer` in the type system. This kernel checks the value and throws otherwise. The check earns its place: three readers use `0:n` as a **matrix-power count**, where `0:1.5` drops a power silently.

Resolution goes back through the ordinary constructor, so the rule's answer meets exactly the validation a stated budget meets — `n >= 1`, `n <= RESOURCE_LIMITS[].max_hop_count`, and `dmax > 0`. That is why the resource cap needs no second check here: a rule that returns an absurd hop count is rejected by the same `assert_resource_cap` a stated one meets.

# Algorithm

 1. Answer `sep` unchanged when its budget is already a value. The wrapper carries a method of its own for this case, so a stated budget builds no structure at all.
 2. Build the structure the rule measures over with [`separation_graph`](@ref), on the wrapper. The graph-taking methods are handed `g` and start at step 3.
 3. Call the rule in the budget field with `nte`, `X` and `g`, forwarding `dims` and `kwargs`, giving the budget the rule answers.
 4. Check the type of that answer. A [`HopCount`](@ref) rule must answer with an `Integer`, and a [`PathLength`](@ref) rule with a `Number`. Throw an `ArgumentError` otherwise.
 5. Rebuild the member through its ordinary constructor, carrying the answer as its budget.

# Arguments

  - `sep`: Separation algorithm, resolved or not.
  - `nte`: Network estimator that owns `sep`, handed to the rule as the channel to anything the graph does not carry.
  - `X`: Data matrix (observations × assets).
  - `g`: Structure the rule measures over, from [`separation_graph`](@ref). Derived from `X` by the wrappers.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments, forwarded to the rule.

# Validation

  - A [`HopCountAlgorithm`](@ref) or `Function` in `HopCount`'s `n` must return an `Integer`.
  - A [`PathLengthAlgorithm`](@ref) or `Function` in `PathLength`'s `dmax` must return a `Number`. `nothing` is a stated budget, not a computed one.

# Returns

  - `sep::AbstractSeparationAlgorithm`: The same member with a resolved budget. `sep` itself when the budget was already a value.

# Related

  - [`AbstractSeparationAlgorithm`](@ref)
  - [`HopCount`](@ref)
  - [`PathLength`](@ref)
  - [`HopCountAlgorithm`](@ref)
  - [`PathLengthAlgorithm`](@ref)
  - [`HopCountQuantile`](@ref)
  - [`PathLengthQuantile`](@ref)
  - [`separation_graph`](@ref)
  - [`separation_matrix`](@ref)
  - [`separation_budget`](@ref)
"""
function resolve_separation end
function resolve_separation(sep::AbstractSeparationAlgorithm, ::AbstractNetworkEstimator,
                            ::MatNum, ::Graphs.AbstractGraph;
                            kwargs...)::AbstractSeparationAlgorithm
    return sep
end
function resolve_separation(sep::HopCount{<:HopCountRule}, nte::AbstractNetworkEstimator,
                            X::MatNum, g::Graphs.AbstractGraph; dims::Int = 1,
                            kwargs...)::HopCount
    n = sep.n(nte, X, g; dims = dims, kwargs...)
    @argcheck(isa(n, Integer),
              ArgumentError("a hop count rule must return an Integer, because three readers use `0:n` as a matrix-power count.\nGot $(n)::$(typeof(n)) from $(typeof(sep.n))."))
    return HopCount(n)
end
function resolve_separation(sep::PathLength{<:PathLengthRule},
                            nte::AbstractNetworkEstimator, X::MatNum,
                            g::Graphs.AbstractGraph; dims::Int = 1, kwargs...)::PathLength
    dmax = sep.dmax(nte, X, g; dims = dims, kwargs...)
    # `nothing` is a stated budget rather than a computed one, so it is not an admissible
    # answer here -- see `PathLengthValue`.
    @argcheck(isa(dmax, Number),
              ArgumentError("a path length rule must return a Number.\nGot $(dmax)::$(typeof(dmax)) from $(typeof(sep.dmax))."))
    return PathLength(dmax)
end
# The resolved case takes the wrapper too, and builds nothing: a stated budget must not pay
# for a structure to be told that it is already a value.
function resolve_separation(sep::AbstractSeparationAlgorithm, ::AbstractNetworkEstimator,
                            ::MatNum; kwargs...)::AbstractSeparationAlgorithm
    return sep
end
function resolve_separation(sep::Union{<:HopCount{<:HopCountRule},
                                       <:PathLength{<:PathLengthRule}},
                            nte::AbstractNetworkEstimator, X::MatNum; dims::Int = 1,
                            kwargs...)::AbstractSeparationAlgorithm
    return resolve_separation(sep, nte, X,
                              separation_graph(sep, nte, X; dims = dims, kwargs...);
                              dims = dims, kwargs...)
end

export separation_matrix, separation_budget, resolve_separation, HopCountQuantile,
       PathLengthQuantile
