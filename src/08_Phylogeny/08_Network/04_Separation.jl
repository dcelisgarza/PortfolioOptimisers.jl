"""
    separation_graph(sep::HopCount, G::Graphs.AbstractGraph)
    separation_graph(sep::HopCount, nte::AbstractNetworkEstimator, X::MatNum;
                     dims::Int = 1, kwargs...)
    separation_graph(sep::PathLength, nte::AbstractNetworkEstimator, X::MatNum;
                     dims::Int = 1, kwargs...)

Build the structure a separation measures over.

This function is one of the three kernels that a member of [`AbstractSeparationAlgorithm`](@ref) implements. [`separation_matrix`](@ref) and [`separation_budget`](@ref) are the other two. It is a kernel of its own, so that a consumer builds the structure one time for each call. [`separation_matrix`](@ref) and [`resolve_separation`](@ref) both take the graph. Without this kernel, a consumer that needs the separations and a resolved budget rule builds the same structure two times.

# What each member measures over

  - [`HopCount`](@ref): the structure of [`calc_adjacency`](@ref), as a `Graphs.SimpleGraph`. The graph is binary, because a hop count does not read the weights and its consumers do. [`_phylogeny_matrix`](@ref) reads `Graphs.adjacency_matrix` of this graph for a sum of powers. With weights, `A^i` sums products of distances and does not count walks.
  - [`PathLength`](@ref): the structure of [`calc_distance_weighted_graph`](@ref). It has the same edge set, with distances as weights on either branch. A shortest path over the similarities of a PMFG takes the route through the weakest links.

# Two entry points, and why only the hop count takes a graph

The methods that take `nte` build the structure from `X`. The method that takes `G` is for a caller that already holds the structure. Both [`clusterise`](@ref) methods build it from the quantity that selects the edges, through the entry point of [`calc_weighted_adjacency_graph`](@ref) that takes `alg`. Without this method, they compute the distance again to resolve a budget rule. Under [`VariationInfoDistance`](@ref), that computation is `98%` of the run time of `clusterise`.

[`PathLength`](@ref) has no method that takes a graph, and it cannot have one. A graph carries no polarity tag. `G` has distances as weights on the tree branch and similarities on the PMFG branch, and the argument does not show which. A shortest path over the similarities of a PMFG returns an answer and raises no error, see [`calc_distance_weighted_graph`](@ref). The hop count is not affected, because it removes the weights.

# Algorithm

 1. Build the structure from `X`, through the branch that `sep` selects.

      + [`HopCount`](@ref): the binary matrix of [`calc_adjacency`](@ref).
      + [`PathLength`](@ref): the structure of [`calc_distance_weighted_graph`](@ref), with distances as weights. It is the answer.

 2. Convert the structure of a [`HopCount`](@ref) to a `Graphs.SimpleGraph`, which removes the weights. The method that takes `G` receives it and starts here.

# Arguments

  - `sep`: Separation algorithm. The function dispatches on its type and does not read its budget. A member whose budget is a rule measures over the same structure as a member whose budget is a number. [`resolve_separation`](@ref) can therefore take this graph.
  - `G`: Network structure that the caller holds, in either polarity.
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

This function is one of the three kernels that a member of [`AbstractSeparationAlgorithm`](@ref) implements. [`separation_graph`](@ref) and [`separation_budget`](@ref) are the other two.

# The method that takes a graph is the interface

The separations depend on the structure alone. Each member implements `separation_matrix(sep, g)`, and the method that takes `nte` is a wrapper that first calls [`separation_graph`](@ref). A consumer that holds a graph calls the method that takes `g`, and builds one structure and not two. It can hold a graph because it built one for [`resolve_separation`](@ref), or because a test selected one.

One wrapper serves every member. [`separation_graph`](@ref) selects the structure that the member reads, so a member has nothing to add here.

# The unreachable sentinel

An unreachable pair keeps the sentinel of the routine that measures it, and the function does not repair it. `Graphs.gdistances` gives `typemax(Int)` for [`HopCount`](@ref). `Graphs.floyd_warshall_shortest_paths` gives `typemax(T)` for [`PathLength`](@ref), which is `Inf` for its `Float64` weights. A consumer reads an entry through [`is_related`](@ref), and does not compare it with the budget itself.

A consumer must also put the score of an entry inside a branch that short-circuits. An `ifelse` evaluates both arms. [`ReciprocalDecay`](@ref) overflows `1 + d` at `typemax(Int)`, and a fractional `power` then raises a `DomainError`.

# The two shipped members read the same structure differently

[`HopCount`](@ref) counts the edges of the binary structure. [`PathLength`](@ref) sums the distances along them. [`separation_graph`](@ref) selects the structure that each member reads. One call to `floyd_warshall_shortest_paths` gives the shortest paths of all pairs, and the function does not run Dijkstra from each vertex. On this shape, the measured call is about 7 times faster than Dijkstra, and about `1.3` times slower than the breadth-first loop of the hop count.

# Algorithm

 1. Build the structure with [`separation_graph`](@ref). The methods that take `g` receive it and start at step 2.

 2. Measure every pair over that structure, through the branch that `sep` selects.

      + [`HopCount`](@ref): one breadth-first traversal with `Graphs.gdistances` from each vertex. Its answer is the column of that vertex in a dense `Matrix{Int}`.
      + [`PathLength`](@ref): one call to `Graphs.floyd_warshall_shortest_paths` over the whole graph. Its `dists` field is the matrix.

# Arguments

  - `sep`: Separation algorithm. The function does not read its budget, so a member whose budget is a rule measures the same separations.
  - `g`: Structure to measure over, from [`separation_graph`](@ref).
  - `nte`: Network estimator. Only the wrapper takes it, and the wrapper builds the structure from `X` on every call.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Keyword arguments that the wrapper passes to [`separation_graph`](@ref).

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

Return the separation budget, the separation beyond which a pair counts as unrelated.

This function is one of the three kernels that a member of [`AbstractSeparationAlgorithm`](@ref) implements. [`separation_graph`](@ref) and [`separation_matrix`](@ref) are the other two. It is separate from `separation_matrix`, because a consumer can need the budget alone. The consumer can test a decay before its loop over the `assets × assets` pairs, or apply a threshold to a matrix that it already holds.

# The function takes the separations, and does not compute them again

`d` is the matrix that [`separation_matrix`](@ref) made. A member can observe its budget from the graph, and not take it from the configuration. The diameter of the graph is such a budget, and the member reads it from `d` with no second traversal. This is why the argument is the matrix, and not a diameter. The largest finite entry is a reduction over `assets²` entries, and a diameter argument makes [`HopCount`](@ref) pay for a reduction that it does not read. With `d`, only [`PathLength`](@ref), the member that needs the reduction, does it.

The shipped members do not read `nte`. It lets the budget of an extension read the estimator that owns it. [`separation_decay`](@ref) has a similar argument, because only one of its four members reads `dmax`.

# The observed diameter is a ceiling, not only a default

[`PathLength`](@ref) puts the observed diameter in place of `dmax = nothing`. It also clamps a stated `dmax` to the observed diameter. The clamp removes no pair, because no pair is beyond the diameter. It corrects the top of the scale, and only [`LinearDecay`](@ref), the one decay that reads the budget, shows it. Without the clamp, `dmax = 100` on a graph of diameter `3.5` moves every score near to one constant, and refuses no pair.

# Algorithm

 1. Refuse a separation whose budget is not resolved. A [`HopCount`](@ref) that holds a [`FractionOf`](@ref) or a [`HopCountAlgorithm`](@ref), and a [`PathLength`](@ref) that holds a [`FractionOf`](@ref) or a [`PathLengthAlgorithm`](@ref), each throw here.

 2. Return the budget, through the branch that `sep` selects.

      + [`HopCount`](@ref): `sep.n.val`, the number of hops of its [`CountOf`](@ref). The method does not read `d`.
      + [`PathLength`](@ref): find `delta`, the largest entry of `d` that [`is_reachable`](@ref) admits. It is the observed diameter. Return `delta` when `sep.dmax` is `nothing`, and the smaller of `sep.dmax` and `delta` otherwise.

# Arguments

  - `sep`: Separation algorithm.
  - `nte`: Network estimator that owns `sep`. The shipped members do not read it.
  - `d`: Separation matrix from [`separation_matrix`](@ref). [`HopCount`](@ref) does not read it, because its budget comes from the configuration. [`PathLength`](@ref) reads it, because the observed diameter caps its budget.

# Validation

  - Throws an `ArgumentError` if `sep` holds a fraction or a budget rule and not a value. Call [`resolve_separation`](@ref) first.

# Returns

  - `dmax::Number`: Separation budget, in the units of `sep`. The unit is the hop for [`HopCount`](@ref), and the unit of the distance estimator for [`PathLength`](@ref). A consumer compares it only with entries of `d`.

# Related

  - [`AbstractSeparationAlgorithm`](@ref)
  - [`HopCount`](@ref)
  - [`PathLength`](@ref)
  - [`separation_matrix`](@ref)
  - [`separation_decay`](@ref)
  - [`Proximity`](@ref)
"""
function separation_budget end
function separation_budget(sep::HopCount{<:CountOf}, ::AbstractNetworkEstimator,
                           ::MatNum)::Number
    return sep.n.val
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
# An unresolved budget is a fraction or a rule, and neither can be compared against an entry
# of `d`. Returning it would put a wrapper or a function where every caller expects a number,
# so it is refused here rather than three frames later inside `d .<= dmax`.
function separation_budget(sep::HopCount{<:Union{HopCountRule, FractionOf}},
                           ::AbstractNetworkEstimator, ::MatNum)
    return throw(ArgumentError("separation_budget needs a resolved separation, and this HopCount carries the budget $(typeof(sep.n)) in `n`.\nCall resolve_separation(sep, nte, X) first; every shipped consumer of a network already does."))
end
function separation_budget(sep::PathLength{<:Union{PathLengthRule, FractionOf}},
                           ::AbstractNetworkEstimator, ::MatNum)
    return throw(ArgumentError("separation_budget needs a resolved separation, and this PathLength carries the budget $(typeof(sep.dmax)) in `dmax`.\nCall resolve_separation(sep, nte, X) first; every shipped consumer of a network already does."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the quantile of the reachable off-diagonal entries of a separation matrix.

The population is the set of pairs that a budget can relate. The diagonal is zero, and an unreachable pair holds the sentinel of [`separation_matrix`](@ref), so neither is a separation. [`phylogeny_matrix`](@ref) selects from the same population. `q` is then the fraction of the pairs that the budget relates.

# The test for the sentinel belongs to the family

The function removes an unreachable pair from the population with [`is_reachable`](@ref), and not with a test of its own. `isfinite` does not work. It is `true` for every `Integer`, so it admits the sentinel `typemax(Int)` of [`HopCount`](@ref).

# Algorithm

 1. Collect into `v` the off-diagonal entries of `d` that [`is_reachable`](@ref) admits. `v` is the population of the quantile.
 2. Throw an `ArgumentError` when `v` is empty.
 3. Return the `q`-quantile of `v` from `Statistics.quantile`.

# Arguments

  - `sep`: Separation algorithm that measured the matrix. The function passes it to [`is_reachable`](@ref).
  - `d`: Separation matrix from [`separation_matrix`](@ref).
  - `q`: Quantile in `[0, 1]`. [`resolve_separation`](@ref) passes the `val` of a [`FractionOf`](@ref) budget.

# Validation

  - Throws an `ArgumentError` if no off-diagonal entry of `d` is reachable, because an empty population has no quantile.

# Returns

  - `dmax::Number`: The `q`-quantile of the reachable off-diagonal separations.

# Related

  - [`resolve_separation`](@ref)
  - [`FractionOf`](@ref)
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
    resolve_separation(sep::AbstractSeparationAlgorithm, nte::AbstractNetworkEstimator,
                       X::MatNum, g::Graphs.AbstractGraph; dims::Int = 1, kwargs...)
    resolve_separation(sep::HopCount{<:HopCountRule}, nte::AbstractNetworkEstimator,
                       X::MatNum, g::Graphs.AbstractGraph; dims::Int = 1, kwargs...)
    resolve_separation(sep::PathLength{<:PathLengthRule}, nte::AbstractNetworkEstimator,
                       X::MatNum, g::Graphs.AbstractGraph; dims::Int = 1, kwargs...)
    resolve_separation(sep::HopCount{<:FractionOf}, nte::AbstractNetworkEstimator,
                       X::MatNum, g::Graphs.AbstractGraph; kwargs...)
    resolve_separation(sep::PathLength{<:FractionOf}, nte::AbstractNetworkEstimator,
                       X::MatNum, g::Graphs.AbstractGraph; kwargs...)
    resolve_separation(sep::AbstractSeparationAlgorithm, nte::AbstractNetworkEstimator,
                       X::MatNum; dims::Int = 1, kwargs...)
    resolve_separation(sep::Union{HopCount{<:Union{HopCountRule, FractionOf}},
                                  PathLength{<:Union{PathLengthRule, FractionOf}}},
                       nte::AbstractNetworkEstimator, X::MatNum; dims::Int = 1, kwargs...)

Replace a separation whose budget is a fraction or a rule by one whose budget is a value.

This function is the fourth kernel of [`AbstractSeparationAlgorithm`](@ref), and the only one that an extension does not have to implement. The method on the abstract type returns `sep` unchanged. A member whose budget is a number costs nothing here, and needs no method of its own.

# A fraction is a quantile of the separations

A [`FractionOf`](@ref) budget names the share of the reachable pairs that the budget relates. The kernel measures the separations over `g` with [`separation_matrix`](@ref), and takes their quantile at `val` over the reachable off-diagonal pairs with [`separation_quantile`](@ref). [`HopCount`](@ref) rounds the quantile to the nearest `Int`, because a hop count is an `Integer`. The smallest off-diagonal hop separation is `1`, so the rounded value is never below one. [`PathLength`](@ref) takes the quantile as its radius, with no rounding.

The fraction costs one traversal of all pairs, and a stated budget does not. The hop branch of [`_phylogeny_matrix`](@ref) does not run the traversal otherwise, because that branch sums matrix powers. The fraction does not cost a second structure, because it reads the graph that its consumer built. Under [`VariationInfoDistance`](@ref), the distance is `98%` of the run time of [`clusterise`](@ref).

# The consumer calls it, and the other kernels do not

Every shipped consumer of a network first resolves `nte.sep`. It then passes the resolved separation to [`separation_matrix`](@ref) and [`separation_budget`](@ref). The consumers are [`phylogeny_matrix`](@ref), both [`clusterise`](@ref) methods, and [`phylogeny_features`](@ref) for [`Proximity`](@ref).

[`separation_budget`](@ref) cannot resolve a fraction or a rule. It takes the separation matrix and not the data, so that [`HopCount`](@ref) does not pay for a reduction to a diameter that it does not read. A fraction and a rule need the structure, and the budget kernel does not have it. `separation_budget` refuses a separation that is not resolved, and this kernel resolves it while the structure is available.

# The rule receives the structure, and does not build one

`g` is the structure of [`separation_graph`](@ref), and the methods that take `g` are the interface. A consumer builds the structure one time. It resolves the rule against that graph, and measures the separations over the same graph. The rule reads what it needs through `separation_matrix(sep, g)`.

The methods that take only `nte` and `X` are wrappers. The resolved case has a wrapper of its own, so that a stated budget builds no structure. The wrapper for a rule dispatches on the type parameter that holds the rule. One generic wrapper builds a structure before it finds that the method on the abstract type does not read it.

# The check of the return value runs at run time, and it must

The return type of a functor is not part of its signature, so the type system cannot make a [`HopCountAlgorithm`](@ref) return an `Integer`. This kernel checks the value, and throws if it is not an `Integer`. The check is necessary. Three readers use `0:n.val` as a count of matrix powers, and `0:1.5` removes a power with no error.

The kernel builds the resolved member through its usual constructor, which wraps a hop count in a [`CountOf`](@ref). The answer of a fraction or a rule meets the same checks as a stated budget: `n.val >= 1`, `n.val <= RESOURCE_LIMITS[].max_hop_count` and `dmax > 0`. For this reason, the resource cap needs no second check here. The same `assert_resource_cap` refuses a hop count that is too large, from a rule or from the caller.

# Algorithm

 1. Return `sep` unchanged when its budget is a value. The wrapper has a method of its own for this case, so a stated budget builds no structure.
 2. In the wrapper, build the structure that the fraction or the rule measures over with [`separation_graph`](@ref). The methods that take `g` receive it and start at step 3.
 3. For a [`FractionOf`](@ref), measure `d = separation_matrix(sep, g)` and take `separation_quantile(sep, d, val)`. Round it to the nearest `Int` for a [`HopCount`](@ref). Go to step 6.
 4. For a rule, call it with `nte`, `X` and `g`, and pass on `dims` and `kwargs`. The rule returns the budget.
 5. Check the type of that answer. A [`HopCount`](@ref) rule must return an `Integer`, and a [`PathLength`](@ref) rule must return a `Number`. Throw an `ArgumentError` otherwise.
 6. Build the member again through its usual constructor, with the answer as its budget.

# Arguments

  - `sep`: Separation algorithm, resolved or not.
  - `nte`: Network estimator that owns `sep`. A rule receives it, and can read from it what the graph does not hold. A fraction does not read it.
  - $(arg_dict[:X])
  - `g`: Structure that the fraction or the rule measures over, from [`separation_graph`](@ref). The wrappers build it from `X`.
  - $(arg_dict[:dims])
  - `kwargs...`: Keyword arguments that the function passes to the rule.

# Validation

  - A [`HopCountAlgorithm`](@ref) or `Function` in `HopCount`'s `n` must return an `Integer`.
  - A [`PathLengthAlgorithm`](@ref) or `Function` in `PathLength`'s `dmax` must return a `Number`. `nothing` is a stated budget, not a computed one.
  - A fraction needs at least one reachable pair of distinct assets. [`separation_quantile`](@ref) throws an `ArgumentError` otherwise.

# Returns

  - `sep::AbstractSeparationAlgorithm`: The same member with a resolved budget. It is `sep` when the budget is a value.

# Related

  - [`AbstractSeparationAlgorithm`](@ref)
  - [`HopCount`](@ref)
  - [`PathLength`](@ref)
  - [`HopCountAlgorithm`](@ref)
  - [`PathLengthAlgorithm`](@ref)
  - [`FractionOf`](@ref)
  - [`separation_quantile`](@ref)
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
# `separation_matrix` dispatches on the separation's *type* and reads no field of it, so `sep`
# itself, with its budget unresolved, measures the separations it is to be resolved over. `g`
# is already the structure the member measures over, so nothing is rebuilt.
function resolve_separation(sep::HopCount{<:FractionOf}, ::AbstractNetworkEstimator,
                            ::MatNum, g::Graphs.AbstractGraph; kwargs...)::HopCount
    d = separation_matrix(sep, g)
    return HopCount(round(Int, separation_quantile(sep, d, sep.n.val)))
end
function resolve_separation(sep::PathLength{<:FractionOf}, ::AbstractNetworkEstimator,
                            ::MatNum, g::Graphs.AbstractGraph; kwargs...)::PathLength
    d = separation_matrix(sep, g)
    return PathLength(separation_quantile(sep, d, sep.dmax.val))
end
# The resolved case takes the wrapper too, and builds nothing: a stated budget must not pay
# for a structure to be told that it is already a value.
function resolve_separation(sep::AbstractSeparationAlgorithm, ::AbstractNetworkEstimator,
                            ::MatNum; kwargs...)::AbstractSeparationAlgorithm
    return sep
end
function resolve_separation(sep::Union{HopCount{<:Union{HopCountRule, FractionOf}},
                                       PathLength{<:Union{PathLengthRule, FractionOf}}},
                            nte::AbstractNetworkEstimator, X::MatNum; dims::Int = 1,
                            kwargs...)::AbstractSeparationAlgorithm
    return resolve_separation(sep, nte, X,
                              separation_graph(sep, nte, X; dims = dims, kwargs...);
                              dims = dims, kwargs...)
end

export separation_matrix, separation_budget, resolve_separation
