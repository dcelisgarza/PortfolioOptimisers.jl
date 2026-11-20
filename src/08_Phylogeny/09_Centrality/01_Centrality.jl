"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the algorithms that score how central each asset is in a network.

Every member wraps one routine of [`Graphs.jl`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/).

Every concrete or abstract type that selects a centrality algorithm must subtype `AbstractCentralityAlgorithm`.

# Each member declares the weights it needs

A member states through [`centrality_polarity`](@ref) which quantity its edge weights must be, and [`centrality_graph`](@ref) gives it that quantity. The declaration is about correctness, because a shortest path over similarities gives the reverse order. It is not about capability. A member that declares nothing, and a source that carries no weights, both run on the plain graph with no error. The method on the abstract type declares `nothing`, so a new member is unweighted until it declares a polarity.

The five members that declare a polarity have an `ov` field. [`TopologyOnly`](@ref) in that field withdraws the declaration for that instance. [`centrality_polarity`](@ref) returns the effective polarity, and not the declared one.

The rule is the one of [pozzi2013](@cite). It weights the degree and the eigenvector centrality by a similarity, ``1 + \\rho_{i,\\,j}``, and the betweenness, the eccentricity and the closeness by the distance ``\\sqrt{2(1 - \\rho_{i,\\,j})}``.

# Interfaces

To add a centrality algorithm, subtype `AbstractCentralityAlgorithm` with the configuration of its routine as fields. Then implement `calc_centrality`. Implement `centrality_polarity` too if the algorithm reads edge weights.

## `calc_centrality`

  - `calc_centrality(ct::MyCentrality, g::Graphs.AbstractGraph) -> VecNum`: Score every vertex of `g`. `centrality_graph` builds `g` with the weights of the polarity that the algorithm declares. The method only calls the routine, and checks nothing.

### Arguments

  - `ct`: The instance of the subtype.
  - `g`: The graph to score. The declared polarity decides whether it is weighted or plain.

### Returns

  - `scores::VecNum`: One score for each vertex of `g`, in the order of the vertices.

## `centrality_polarity`

  - `centrality_polarity(ct::MyCentrality) -> Option{<:AbstractCentralityPolarity}`: Declare which quantity the edge weights must be. The method on the abstract type returns `nothing`, so a new algorithm runs on the plain graph until it declares a polarity. Declare [`DistancePolarity`](@ref) for a routine over shortest paths, and [`SimilarityPolarity`](@ref) for a routine that reads the adjacency matrix.

### Arguments

  - `ct`: The instance of the subtype.

### Returns

  - `polarity::Option{<:AbstractCentralityPolarity}`: The effective polarity, or `nothing` for an algorithm that reads the topology alone.

# Related

  - [`centrality_polarity`](@ref)
  - [`centrality_graph`](@ref)
  - [`TopologyOnly`](@ref)
  - [`BetweennessCentrality`](@ref)
  - [`ClosenessCentrality`](@ref)
  - [`DegreeCentrality`](@ref)
  - [`EigenvectorCentrality`](@ref)
  - [`KatzCentrality`](@ref)
  - [`Pagerank`](@ref)
  - [`RadialityCentrality`](@ref)
  - [`StressCentrality`](@ref)

# References

  - $(ref_dict[:pozzi2013]) Section "Methods", "Centrality and Peripherality measures".
"""
abstract type AbstractCentralityAlgorithm <: AbstractPhylogenyAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Withdraws an algorithm's polarity declaration, so it reads the network's topology alone.

An algorithm that declares a polarity receives weights wherever the source carries them. `TopologyOnly` in its `ov` field withdraws that request. [`centrality_polarity`](@ref) then returns `nothing`, and [`centrality_graph`](@ref) builds the plain `Graphs.SimpleGraph` of [`phylogeny_matrix`](@ref). [`DegreeCentrality`](@ref), [`Pagerank`](@ref) and [`KatzCentrality`](@ref) already run on that graph. `TopologyOnly` selects a route that exists, and it adds no estimator.

# The override runs one way only

It removes weights and never adds them. No value forces a polarity on an algorithm, and the type of the field is not [`AbstractCentralityPolarity`](@ref) on purpose. A forced polarity runs with no error, because [`calc_distance_weighted_graph`](@ref) carries distances on both branches. The algorithm then reads a distance where it needs a similarity, and reverses its order with no error. No check at run time can find a wrong polarity.

# Every source honours the request

Every source can give the answer over the topology alone. The override never warns, and no source ignores it. A partition source, a precomputed [`PhylogenyResult`](@ref), and the tree branch under [`SimilarityPolarity`](@ref) already build the plain graph. On these sources, the answer is already the answer that the override asks for.

Only the five algorithms that declare a polarity have an `ov` field. [`DegreeCentrality`](@ref), [`Pagerank`](@ref) and [`KatzCentrality`](@ref) already return the answer over the topology alone. They have nothing to override, so `DegreeCentrality(; ov = TopologyOnly())` raises a `MethodError`.

# It is a choice, not a simplification, and it moves no default

A centrality over the topology alone can be more stable across folds, for the same reason that a fixed `dmax` is stable across folds under [`PathLength`](@ref). That is not a reason to make it the default.

**The shipped default is already unweighted.** The default `ct` of [`CentralityEstimator`](@ref) is [`DegreeCentrality`](@ref), which declares no polarity. A caller who names no algorithm already gets this answer. A default of `ov = TopologyOnly()` changes the answer only for a caller who selected one of the five. For those five, the weights of the source give the correct answer, and [`AbstractCentralityPolarity`](@ref) states why.

**The override makes `sep` active again.** The route of the plain graph reads the separation closure that [`phylogeny_matrix`](@ref) builds, and the weighted routes do not read it. The override replaces the edge weights by a second parameter, and does not remove a parameter. Over twenty assets, all five algorithms with the override give different answers at `HopCount(; n = 1)` and at `n = 3`. Four of them do not read `sep` without the override. Under a default [`PathLength`](@ref), that parameter is the observed diameter. The diameter depends on the data, and the argument for stability across folds tries to avoid such a quantity.

# Examples

```jldoctest
julia> ClosenessCentrality(; ov = TopologyOnly())
ClosenessCentrality
    args ┼ Tuple{}: ()
  kwargs ┼ @NamedTuple{}: NamedTuple()
      ov ┴ TopologyOnly()

julia> isnothing(centrality_polarity(ClosenessCentrality(; ov = TopologyOnly())))
true

julia> centrality_polarity(ClosenessCentrality())
DistancePolarity()
```

# Related

  - [`centrality_polarity`](@ref)
  - [`centrality_graph`](@ref)
  - [`AbstractCentralityPolarity`](@ref)
  - [`BetweennessCentrality`](@ref)
  - [`ClosenessCentrality`](@ref)
  - [`EigenvectorCentrality`](@ref)
  - [`RadialityCentrality`](@ref)
  - [`StressCentrality`](@ref)
"""
struct TopologyOnly <: AbstractAlgorithm end
"""
    assert_no_weight_channel_args(::Type{T}, args::Tuple, S::Type, shape::AbstractString,
                                  channel::AbstractString) where {T}

Refuse an entry of `args` that reaches a second weight channel.

Both guards of a splatted `args` make this refusal. An entry point of `Graphs.jl` takes its weights in a positional slot. An entry of `args` with that type is then a second weight channel, and the declared channel already gives the weights. [`assert_centrality_args`](@ref) and [`assert_tree_args`](@ref) differ only in the type that reaches a channel, and in the declared channel that they name. Both call this function with their own `S`, `shape` and `channel`.

The message gives the index and the type of the entry. The library splats `args`, so the caller sees no argument names.

# Algorithm

 1. Find `idx`, the index of the first entry of `args` that is an `S`.
 2. Throw a [`ConflictingArgumentError`](@ref) when `idx` is not `nothing`. The message holds `T`, `shape`, `channel`, `idx` and the type of the entry.

# Arguments

  - `T`: Algorithm type, which the error message names.
  - `args`: Positional arguments for the function of `Graphs.jl`.
  - `S`: The type of an entry that reaches a weight slot, such as `AbstractMatrix`.
  - `shape`: Name of that type, for the error message.
  - `channel`: Sentence that names the declared channel that the caller must use.

# Validation

  - Throws a [`ConflictingArgumentError`](@ref) if any entry of `args` is an `S`.

# Returns

  - `nothing`.

# Related

  - [`assert_centrality_args`](@ref)
  - [`assert_tree_args`](@ref)
"""
function assert_no_weight_channel_args(::Type{T}, args::Tuple, S::Type,
                                       shape::AbstractString,
                                       channel::AbstractString) where {T}
    idx = findfirst(a -> isa(a, S), args)
    @argcheck(isnothing(idx),
              ConflictingArgumentError("`args` of a $(T) must not contain a $(shape): $(channel). Got\nargs[$(idx)] => $(isnothing(idx) ? nothing : typeof(args[idx]))"))
    return nothing
end
"""
    assert_centrality_args(::Type{T}, args::Tuple) where {T}

Refuse a matrix inside a centrality algorithm's `args`.

[`calc_centrality`](@ref) splats `args` into the centrality function of `Graphs.jl`. A matrix in `args` is then a `distmx`, a second way to weight the graph that no polarity declares. [`centrality_polarity`](@ref) is the declared way. It selects the weights that the mathematics of the algorithm needs, from the structure that the estimator built. The second channel is not safe:

  - `distmx` is the third positional argument of `Graphs.betweenness_centrality`. A matrix in `args` binds to `vs`, and the call overflows the stack inside `Graphs.degree`. A caller can catch the `StackOverflowError`, and the process continues. Only the call fails.
  - `distmx` is the second positional argument of `Graphs.closeness_centrality`. A matrix in `args` ran, and replaced the polarity with no error. A matrix of the wrong size raised a `BoundsError`, and not a `DimensionMismatch`.
  - `Graphs.stress_centrality` has no `distmx`.

The guard accepts every other entry. A list of vertices or a sample count is a real positional argument of those functions, and it does not set weights.

`kwargs` needs no second guard. A keyword binds by name, so a matrix in `kwargs` cannot reach a positional slot. None of the four functions declares a keyword whose value is a matrix. Each of `normalize`, `endpoints`, `rng` and `seed` refuses a matrix. A matrix in `kwargs` always raises a `MethodError` or a `TypeError`.

# Algorithm

 1. Refuse an `AbstractMatrix` in `args` with [`assert_no_weight_channel_args`](@ref), which names [`centrality_polarity`](@ref) as the declared weight channel.

# Arguments

  - `T`: Centrality algorithm type, which the error message names.
  - `args`: Positional arguments for the centrality function of `Graphs.jl`.

# Validation

  - Throws a [`ConflictingArgumentError`](@ref) if any entry of `args` is an `AbstractMatrix`.

# Returns

  - `nothing`.

# Related

  - [`centrality_polarity`](@ref)
  - [`assert_tree_args`](@ref)
  - [`BetweennessCentrality`](@ref)
  - [`ClosenessCentrality`](@ref)
  - [`StressCentrality`](@ref)
"""
function assert_centrality_args(::Type{T}, args::Tuple) where {T}
    assert_no_weight_channel_args(T, args, AbstractMatrix, "matrix",
                                  "a weight matrix reaches the centrality algorithm through `centrality_polarity`, not through `args`")
    return nothing
end
"""
    assert_tree_args(::Type{T}, args::Tuple, kwargs::NamedTuple) where {T}

Refuse a second weight channel in the `args` and the `kwargs` of a spanning tree algorithm.

[`calc_mst`](@ref) splats both fields into the spanning tree function of `Graphs.jl`. Every weight channel that they can reach re-weights or re-orients the tree that [`calc_weighted_adjacency_graph`](@ref) builds. The graph that it gives to [`calc_mst`](@ref) carries the distances from the `de` and the `ce` of the estimator, and the default `distmx` of `Graphs.jl` is those weights. A caller who fills these fields replaces weights that are already correct, and no error shows it:

  - `kruskal_mst`, `boruvka_mst` and `prim_mst` all take `distmx` as their second positional argument. A matrix in `args` replaces the distances of the estimator. Such a matrix often has the correct size, and the call then returns a valid tree of the wrong graph.
  - `kruskal_mst` also takes a `weight_vector` in that slot, which replaces the weights in the same way.
  - `minimize` in `kwargs` inverts the direction of the search. A minimised distance defines the tree branch, as [`calc_weighted_adjacency_graph`](@ref) and [`SimilarityPolarity`](@ref) state. `minimize = false` gives a maximum spanning tree, and every consumer reads it as a minimum spanning tree.

The guard accepts every entry that is not a matrix or a vector, and every other keyword. Those reach no weight channel. The three functions declare no other weight channel, so a wrong entry raises an error at the call.

# Algorithm

 1. Refuse an `AbstractMatrix` or an `AbstractVector` in `args` with [`assert_no_weight_channel_args`](@ref). The message names the graph that [`calc_weighted_adjacency_graph`](@ref) built as the declared weight channel.
 2. Throw a [`ConflictingArgumentError`](@ref) when `kwargs` holds the key `minimize`. The message holds `T` and the value of that key.

# Arguments

  - `T`: Spanning tree algorithm type, which the error message names.
  - `args`: Positional arguments for the spanning tree function of `Graphs.jl`.
  - `kwargs`: Keyword arguments for the same function.

# Validation

  - Throws a [`ConflictingArgumentError`](@ref) if any entry of `args` is an `AbstractMatrix` or an `AbstractVector`.
  - Throws a [`ConflictingArgumentError`](@ref) if `kwargs` contains `minimize`.

# Returns

  - `nothing`.

# Related

  - [`assert_centrality_args`](@ref)
  - [`calc_mst`](@ref)
  - [`calc_weighted_adjacency_graph`](@ref)
  - [`KruskalTree`](@ref)
  - [`BoruvkaTree`](@ref)
  - [`PrimTree`](@ref)
"""
function assert_tree_args(::Type{T}, args::Tuple, kwargs::NamedTuple) where {T}
    assert_no_weight_channel_args(T, args, Union{AbstractMatrix, AbstractVector},
                                  "matrix or vector",
                                  "the weights of a spanning tree reach it through the graph the `NetworkEstimator` built, not through `args`")
    @argcheck(!haskey(kwargs, :minimize),
              ConflictingArgumentError("`kwargs` of a $(T) must not contain `minimize`: the tree branch is defined by minimising a distance, so `minimize = false` silently yields a maximum spanning tree. Got\nkwargs.minimize => $(get(kwargs, :minimize, nothing))"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the centrality vector of a precomputed phylogeny matrix with the algorithm `ct`.

The graph is always unweighted, whatever polarity `ct` declares. `Graphs.SimpleGraph` reads every nonzero entry of `plr.X` as an edge, and removes its value. A precomputed [`PhylogenyResult`](@ref) is one of the sources with no weights that the warning of [`centrality_vector`](@ref) lists. To get the weighted answer, pass the estimator and not its result.

# Algorithm

 1. Convert `plr.X`, the precomputed phylogeny matrix, to a plain `Graphs.SimpleGraph` `G`.
 2. Score the vertices of `G` with [`calc_centrality`](@ref).
 3. Return the scores in a [`PhylogenyResult`](@ref).

# Arguments

  - `plr`: Precomputed phylogeny matrix.
  - $(arg_dict[:cta])
  - `args...`: Positional arguments. The method does not read them.
  - `kwargs...`: Keyword arguments. The method does not read them.

# Returns

  - `plr::PhylogenyResult{<:VecNum}`: One centrality score for each asset.

# Related

  - [`PhylogenyResult`](@ref)
  - [`AbstractCentralityAlgorithm`](@ref)
  - [`calc_centrality`](@ref)
  - [`centrality_graph`](@ref)
"""
function centrality_vector(plr::PhylogenyResult{<:MatNum}, ct::AbstractCentralityAlgorithm,
                           args...; kwargs...)
    G = Graphs.SimpleGraph(plr.X)
    return PhylogenyResult(; X = calc_centrality(ct, G))
end
"""
$(DocStringExtensions.TYPEDEF)

Scores each asset by the share of the network's shortest paths that run through it.

`BetweennessCentrality` selects the [betweenness centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.betweenness_centrality) of `Graphs.jl`. It measures how much an asset lies on the shortest paths between other assets.

It declares [`DistancePolarity`](@ref), unless `ov` overrides it. It is defined over shortest paths, so its weights must be distances. On a tree, the weighted answer is equal to the unweighted one. A tree has exactly one path between two vertices, so no weights can change the set of shortest paths. This is a property of the tree, and it does not hold on the similarity branch. Set `ov` to [`TopologyOnly`](@ref) to withdraw the declaration and read the topology alone.

# Mathematical definition

```math
\\begin{align}
    \\mathrm{BC}_i &= \\dfrac{1}{(n - 1)(n - 2)} \\sum_{s \\neq i \\neq t} \\dfrac{\\sigma_{s,\\,t}(i)}{\\sigma_{s,\\,t}}\\,,
\\end{align}
```

Where:

  - ``\\mathrm{BC}_i``: Betweenness centrality of asset ``i``.
  - $(math_dict[:sigma_st_paths])
  - $(math_dict[:sigma_st_i_paths])
  - $(math_dict[:n_network])

The sum runs over the ordered pairs of distinct assets, so an undirected network counts each pair two times. The leading factor is therefore the reciprocal of ``(n - 1)(n - 2)``, and not of half of it. `Graphs.jl` applies that factor by default. `kwargs = (; normalize = false)` replaces it by ``1/2``, which gives the count over the unordered pairs. `kwargs = (; endpoints = true)` also counts the two ends of every path, which raises every score.

A pair that several shortest paths join gives one unit of score, divided among those paths, because the summand is a fraction of ``\\sigma_{s,\\,t}``. [`StressCentrality`](@ref) does not divide, and counts the paths.

[freeman1979](@cite) states the measure of [freeman1977](@cite) again over the unordered pairs. It divides the measure by its largest value, ``(n^2 - 3n + 2) / 2``, which the centre of a star reaches. That is the factor above, over the ordered pairs. No copy of the work of 1977 is available.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BetweennessCentrality(;
        args::Tuple = (),
        kwargs::NamedTuple = (;),
        ov::Option{TopologyOnly} = nothing
    ) -> BetweennessCentrality

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:ctargs_nomat])

# Examples

```jldoctest
julia> BetweennessCentrality()
BetweennessCentrality
    args ┼ Tuple{}: ()
  kwargs ┼ @NamedTuple{}: NamedTuple()
      ov ┴ nothing
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`centrality_polarity`](@ref)
  - [`DistancePolarity`](@ref)
  - [`TopologyOnly`](@ref)
  - [`Graphs.betweenness_centrality`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.betweenness_centrality)

# References

  - $(ref_dict[:freeman1977])
  - $(ref_dict[:freeman1979]) Pages 223-224. $(ref_dict[:no_original_source])
  - $(ref_dict[:brandes2001]) Sections 2 and 4, for the sum over the ordered pairs and the algorithm that `Graphs.jl` runs.
"""
@concrete struct BetweennessCentrality <: AbstractCentralityAlgorithm
    """
    $(field_dict[:ctargs])
    """
    args
    """
    $(field_dict[:ctkwargs])
    """
    kwargs
    """
    $(field_dict[:ctov])
    """
    ov
    function BetweennessCentrality(args::Tuple, kwargs::NamedTuple,
                                   ov::Option{TopologyOnly})
        assert_centrality_args(BetweennessCentrality, args)
        return new{typeof(args), typeof(kwargs), typeof(ov)}(args, kwargs, ov)
    end
end
function BetweennessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;),
                               ov::Option{TopologyOnly} = nothing)::BetweennessCentrality
    return BetweennessCentrality(args, kwargs, ov)
end
"""
$(DocStringExtensions.TYPEDEF)

Scores each asset by the reciprocal of its mean shortest-path distance to the others.

`ClosenessCentrality` selects the [closeness centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.closeness_centrality) of `Graphs.jl`. It measures how near an asset is to all other assets.

It declares [`DistancePolarity`](@ref), unless `ov` overrides it. It sums the lengths of shortest paths, so its weights must be distances. It reads them on both branches, so its answer on a [`NetworkEstimator`](@ref) source is different from the unweighted one. Over twenty assets, the measured largest absolute change is `0.713` on a triangulated maximally filtered graph and `0.538` on a tree. Set `ov` to [`TopologyOnly`](@ref) to withdraw the declaration and read the topology alone.

# Mathematical definition

```math
\\begin{align}
    \\mathrm{CC}_i &= \\dfrac{r_i}{\\displaystyle\\sum_{j \\in \\mathcal{R}_i} \\ell_{i,\\,j}} \\cdot \\dfrac{r_i}{n - 1}\\,,
\\end{align}
```

Where:

  - ``\\mathrm{CC}_i``: Closeness centrality of asset ``i``.
  - ``\\mathcal{R}_i``: Set of assets other than asset ``i`` that asset ``i`` reaches.
  - ``r_i``: Cardinality of ``\\mathcal{R}_i``.
  - $(math_dict[:ell_ij_path])
  - $(math_dict[:n_network])

The first factor is the reciprocal of the mean length from asset ``i`` to the assets that it reaches. The second factor is the share of the assets that it reaches. `Graphs.jl` applies the second factor by default, and `kwargs = (; normalize = false)` removes it. On a connected network, ``r_i = n - 1`` and the second factor is one. The two settings agree there, and are different only on a network with more than one component.

An asset that reaches no asset scores zero, and not an infinity. The sum in the denominator is over ``\\mathcal{R}_i`` alone, so an unreachable asset never enters it.

[freeman1979](@cite) states the first factor for a connected network, as the reciprocal of the mean distance. It gives the factor to two earlier works, and no copy of those works is available. The second factor extends the measure to a network with more than one component. It is the rule of `Graphs.jl`, and no available work states it first.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ClosenessCentrality(;
        args::Tuple = (),
        kwargs::NamedTuple = (;),
        ov::Option{TopologyOnly} = nothing
    ) -> ClosenessCentrality

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:ctargs_nomat])

# Examples

```jldoctest
julia> ClosenessCentrality()
ClosenessCentrality
    args ┼ Tuple{}: ()
  kwargs ┼ @NamedTuple{}: NamedTuple()
      ov ┴ nothing
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`centrality_polarity`](@ref)
  - [`DistancePolarity`](@ref)
  - [`TopologyOnly`](@ref)
  - [`Graphs.closeness_centrality`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.closeness_centrality)

# References

  - $(ref_dict[:freeman1979]) Pages 225-226. $(ref_dict[:no_original_source])
"""
@concrete struct ClosenessCentrality <: AbstractCentralityAlgorithm
    """
    $(field_dict[:ctargs])
    """
    args
    """
    $(field_dict[:ctkwargs])
    """
    kwargs
    """
    $(field_dict[:ctov])
    """
    ov
    function ClosenessCentrality(args::Tuple, kwargs::NamedTuple, ov::Option{TopologyOnly})
        assert_centrality_args(ClosenessCentrality, args)
        return new{typeof(args), typeof(kwargs), typeof(ov)}(args, kwargs, ov)
    end
end
function ClosenessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;),
                             ov::Option{TopologyOnly} = nothing)::ClosenessCentrality
    return ClosenessCentrality(args, kwargs, ov)
end
"""
$(DocStringExtensions.TYPEDEF)

Counts the network edges that touch each asset, divided by the number of other assets.

`DegreeCentrality` selects the [degree centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.degree_centrality-Tuple%7BAbstractGraph%7D) of `Graphs.jl`. It is the simplest score of the family, and the default `ct` of [`CentralityEstimator`](@ref).

It declares no polarity, and runs on the plain graph. `Graphs._degree_centrality` counts edges and does not read their weights. The `sep` of the estimator has an effect on it. The unweighted route reads the separation closure that [`phylogeny_matrix`](@ref) builds, so `HopCount(; n = 2)` changes this answer.

It has no `ov` field, and [`TopologyOnly`](@ref) does not apply to it. It already reads the topology alone, so it has no declaration to withdraw. `DegreeCentrality(; ov = TopologyOnly())` raises a `MethodError`.

# Mathematical definition

The degree vector of an adjacency matrix ``\\mathbf{A}`` over ``n`` assets is

```math
\\begin{align}
    \\mathbf{D}_n &= \\mathbf{A}\\,\\mathbf{1}_n\\,,
\\end{align}
```

Where:

  - ``\\mathbf{D}_n``: Degree vector of the network, whose ``i``-th entry counts the edges that touch asset ``i``.
  - $(math_dict[:A_network])
  - ``\\mathbf{1}_n``: Column vector of ones of length ``n``.
  - $(math_dict[:n_network])

`Graphs._degree_centrality` normalises that vector by default, and this type returns ``\\mathbf{D}_n / (n - 1)``, and not ``\\mathbf{D}_n``. `kwargs = (; normalize = false)` gives ``\\mathbf{D}_n``.

The factor is the only difference, and it does not change the ranking. [`average_centrality`](@ref) is linear in the score vector, so the average changes by the same factor.

# The three `kind` values agree on these structures

`kind = 0` counts the total degree, `kind = 1` the in-degree, and `kind = 2` the out-degree. Every graph that this library builds is undirected, and on an undirected graph the three are one number. On the same tree, `kind = 0`, `1` and `2` give the same vector. The field exists because `Graphs.jl` takes it, and it selects nothing here.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DegreeCentrality(;
        kind::Integer = 0,
        kwargs::NamedTuple = (;)
    ) -> DegreeCentrality

Keywords correspond to the struct's fields.

## Validation

  - `0 <= kind <= 2`.

# Examples

```jldoctest
julia> DegreeCentrality(; kind = 1)
DegreeCentrality
    kind ┼ Int64: 1
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`centrality_polarity`](@ref)
  - [`TopologyOnly`](@ref)
  - [`Graphs._degree_centrality`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.degree_centrality-Tuple%7BAbstractGraph%7D)

# References

  - $(ref_dict[:graphpo1]) Section 2.3.1, Equation 3.
  - $(ref_dict[:freeman1979]) Page 221, for the division by ``n - 1``.
"""
@concrete struct DegreeCentrality <: AbstractCentralityAlgorithm
    """
    Degree to count. `0` counts the total degree, `1` the in-degree, and `2` the out-degree.
    """
    kind
    """
    $(field_dict[:ctkwargs])
    """
    kwargs
    function DegreeCentrality(kind::Integer, kwargs::NamedTuple)
        @argcheck(kind in 0:2, DomainError(kind, "kind must be in 0:2"))
        return new{typeof(kind), typeof(kwargs)}(kind, kwargs)
    end
end
function DegreeCentrality(; kind::Integer = 0, kwargs::NamedTuple = (;))::DegreeCentrality
    return DegreeCentrality(kind, kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

Scores each asset by the leading eigenvector of the network's adjacency matrix.

`EigenvectorCentrality` selects the eigenvector centrality of `Graphs.jl`. The score of an asset grows with the scores of its neighbours.

# Mathematical definition

```math
\\begin{align}
    \\mathbf{EC}_n &= \\dfrac{1}{\\lambda_{\\mathrm{max}}}\\,\\mathbf{A}\\,\\mathbf{q}_{\\mathrm{max}}\\,,
\\end{align}
```

Where:

  - ``\\mathbf{EC}_n``: Eigenvector centrality vector of the network.
  - $(math_dict[:A_network])
  - $(math_dict[:lambda_max_network])
  - ``\\mathbf{q}_{\\mathrm{max}}``: Eigenvector of ``\\lambda_{\\mathrm{max}}``.

The right-hand side is ``\\mathbf{q}_{\\mathrm{max}}``, so the score is the leading eigenvector, with the normalisation of the eigensolver. `Graphs.jl` returns it with unit 2-norm, and takes the absolute value of every entry. By the Perron-Frobenius theorem, the entries of the leading eigenvector of a non-negative matrix have one sign, so the absolute value changes no order. The returned vector agrees with the formula above to the precision of the eigensolver, which changes by a few ulps between calls on one graph.

The formula above is Equation 4 of [graphpo1](@cite). [bonacich1987](@cite) states the measure as ``\\lambda \\mathbf{e} = \\mathbf{R}\\,\\mathbf{e}``, and prefers the largest eigenvalue. It gives the measure to a work of 1972 by the same author, and no copy of that work is available.

It declares [`SimilarityPolarity`](@ref), unless `ov` overrides it, and it is the only member that declares it. It is the leading eigenvector of the adjacency matrix, so a stronger link must give a larger entry. It reads weights on the similarity branch alone. A minimised distance selects a tree, and a tree carries no similarity. This algorithm runs unweighted on a tree, and does not receive the wrong quantity. Set `ov` to [`TopologyOnly`](@ref) to withdraw the declaration and read the topology alone.

The weights change this answer less than they change the answers of the shortest path algorithms, but they change it. The weighted and the unweighted vectors are different, and their correlation is high. A withdrawal of the weights also changes the ``\\lambda_{\\mathrm{max}}`` of the structure.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EigenvectorCentrality(;
        ov::Option{TopologyOnly} = nothing
    ) -> EigenvectorCentrality

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> EigenvectorCentrality()
EigenvectorCentrality
  ov ┴ nothing
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`centrality_polarity`](@ref)
  - [`SimilarityPolarity`](@ref)
  - [`TopologyOnly`](@ref)
  - [`Graphs.eigenvector_centrality`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.eigenvector_centrality-Tuple%7BAbstractGraph%7D)

# References

  - $(ref_dict[:graphpo1]) Section 2.3.2, Equation 4.
  - $(ref_dict[:bonacich1987]) Page 1172, Equations 1-2. $(ref_dict[:no_original_source])
"""
@concrete struct EigenvectorCentrality <: AbstractCentralityAlgorithm
    """
    $(field_dict[:ctov])
    """
    ov
    function EigenvectorCentrality(ov::Option{TopologyOnly})
        return new{typeof(ov)}(ov)
    end
end
function EigenvectorCentrality(; ov::Option{TopologyOnly} = nothing)::EigenvectorCentrality
    return EigenvectorCentrality(ov)
end
"""
$(DocStringExtensions.TYPEDEF)

Scores each asset by every walk that reaches it, discounted geometrically by the walk's length.

`KatzCentrality` selects the [Katz centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.katz_centrality) of `Graphs.jl`. It counts the walks that reach an asset, and the attenuation factor `alpha` discounts each walk by its length.

It declares no polarity, and runs on the plain graph. `Graphs.katz_centrality` reads the adjacency matrix with the element type `Bool`, through `adjacency_matrix(g, Bool)`. A weight other than `0` or `1` cannot become a `Bool`, so on a weighted graph the call throws an `InexactError`. The unweighted route is therefore necessary for this algorithm.

It has no `ov` field, and [`TopologyOnly`](@ref) does not apply to it. It already reads the topology alone, so it has no declaration to withdraw. `KatzCentrality(; ov = TopologyOnly())` raises a `MethodError`.

# Mathematical definition

```math
\\begin{align}
    \\boldsymbol{v} &= \\sum_{k \\geq 0} \\alpha^{k}\\,\\mathbf{A}^{k}\\,\\mathbf{1}_n = \\left(\\mathbf{I}_n - \\alpha\\,\\mathbf{A}\\right)^{-1}\\mathbf{1}_n\\,, \\\\
    \\mathbf{KC}_n &= \\dfrac{\\boldsymbol{v}}{\\lVert \\boldsymbol{v} \\rVert_2}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{v}``: Walk sum of the network, whose ``i``-th entry adds up every walk that reaches asset ``i``, discounted by ``\\alpha`` for each edge on it.
  - ``\\mathbf{KC}_n``: Katz centrality vector, the walk sum at unit 2-norm.
  - ``\\alpha``: Attenuation factor, `alpha`.
  - $(math_dict[:A_network])
  - $(math_dict[:lambda_max_network])
  - ``\\mathbf{I}_n``: Identity matrix of order ``n``.
  - ``\\mathbf{1}_n``: Column vector of ones of length ``n``.
  - $(math_dict[:n_network])

The series converges to the resolvent only for ``\\alpha < 1 / \\lambda_{\\mathrm{max}}``, and the two right-hand sides are equal only there. Outside that range, the resolvent is defined at almost every ``\\alpha``, but its vector is not a sum of walks.

[katz1953](@cite) sums the walks of length one and more, ``\\left(\\mathbf{I}_n - \\alpha\\,\\mathbf{A}\\right)^{-1} - \\mathbf{I}_n``. On ``\\mathbf{1}_n``, that sum gives ``\\boldsymbol{v} - \\mathbf{1}_n``. [katz1953](@cite) then divides by a constant that depends on ``n`` and ``\\alpha``. `Graphs.jl` keeps the walk of length zero, and scales to unit 2-norm. The ranking is the same, and the scores are not proportional. [katz1953](@cite) states the same bound on ``\\alpha``.

# `alpha` must be below the reciprocal of the largest eigenvalue

Above the bound, the linear solve still returns a vector, but the vector is not a centrality. Some of its scores become negative, and a negative centrality has no meaning.

**The constructor cannot check this.** ``\\lambda_{\\mathrm{max}}`` is a property of the graph, and [`centrality_graph`](@ref) builds the graph later. The constructor checks only `alpha > 0`, and the caller must obey the bound. A dense network has a larger ``\\lambda_{\\mathrm{max}}``, and so a smaller bound. A value that holds on a tree can fail on a triangulated maximally filtered graph over the same assets. The default `alpha = 0.3` is not safe for every shape of graph.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    KatzCentrality(;
        alpha::Number = 0.3
    ) -> KatzCentrality

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:katz_alpha])

# Examples

```jldoctest
julia> KatzCentrality(; alpha = 0.1)
KatzCentrality
  alpha ┴ Float64: 0.1
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`centrality_polarity`](@ref)
  - [`TopologyOnly`](@ref)
  - [`Graphs.katz_centrality`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.katz_centrality)

# References

  - $(ref_dict[:katz1953]) Pages 41-42.
"""
@concrete struct KatzCentrality <: AbstractCentralityAlgorithm
    """
    Attenuation factor. The series of walks converges only when `alpha` is below the reciprocal of the largest eigenvalue of the adjacency matrix, and the constructor does not check it.
    """
    alpha
    function KatzCentrality(alpha::Number)
        @argcheck(zero(alpha) < alpha, DomainError(alpha, "`alpha` must be positive"))
        return new{typeof(alpha)}(alpha)
    end
end
function KatzCentrality(; alpha::Number = 0.3)::KatzCentrality
    return KatzCentrality(alpha)
end
"""
$(DocStringExtensions.TYPEDEF)

Scores each asset by the stationary distribution of a damped random walk over the network.

`Pagerank` selects the [PageRank](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.pagerank-Union%7BTuple%7BAbstractGraph%7BU%7D%7D,%20Tuple%7BU%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any,%20Integer%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any,%20Integer,%20Any%7D%7D%20where%20U%3C:Integer) of `Graphs.jl`. The score of an asset grows with the scores of the assets that link to it. The damping factor `alpha`, the number of iterations `n` and the convergence tolerance `epsilon` control the iteration.

It declares no polarity, and runs on the plain graph. `Graphs.pagerank` reads only `outdegree` and `inneighbors`, and never reads an edge weight. The weighted graph and the plain graph give the same vector. As for [`DegreeCentrality`](@ref), the `sep` of the estimator has an effect on it, because it reads the separation closure and not the structure.

It has no `ov` field, and [`TopologyOnly`](@ref) does not apply to it. It already reads the topology alone, so it has no declaration to withdraw. `Pagerank(; ov = TopologyOnly())` raises a `MethodError`.

# Mathematical definition

```math
\\begin{align}
    \\mathrm{PR}_i &= \\dfrac{1 - \\alpha}{n} + \\dfrac{\\alpha}{n}\\sum_{j \\in \\mathcal{D}} \\mathrm{PR}_j + \\alpha \\sum_{j \\in \\mathcal{I}_i} \\dfrac{\\mathrm{PR}_j}{k_j}\\,,
\\end{align}
```

Where:

  - ``\\mathrm{PR}_i``: PageRank of asset ``i``, the share of time that the damped walk is at asset ``i``.
  - ``\\alpha``: Damping factor, `alpha`. It is the probability that the walk follows an edge, and does not teleport.
  - ``\\mathcal{I}_i``: Set of assets with an edge into asset ``i``.
  - ``k_j``: Number of edges that leave asset ``j``.
  - ``\\mathcal{D}``: Set of dangling assets, which no edge leaves.
  - $(math_dict[:n_network])

The three terms are the three moves of the walk. It teleports to an asset drawn uniformly, it teleports out of a dangling asset that it cannot leave, or it follows one of the edges into asset ``i``. Every score is non-negative, and the vector sums to one. The members of the family that count edges or paths do not have this property.

[brin1998](@cite) writes ``\\mathrm{PR}_i = (1 - \\alpha) + \\alpha \\sum_{j \\in \\mathcal{I}_i} \\mathrm{PR}_j / k_j``, with no division by ``n`` and no term for the dangling assets. The division by ``n`` makes the vector sum to one, as [brin1998](@cite) states. The uniform spread of the mass of the dangling assets is the rule of `Graphs.jl`, and no available work states it first.

Every network that this library builds is undirected. On it, ``\\mathcal{I}_i`` is the neighbourhood of asset ``i``, and ``k_j`` is the degree of asset ``j``. On a connected undirected network, the solution goes to a scaled copy of the degree vector of [`DegreeCentrality`](@ref) as ``\\alpha`` goes to one. A smaller ``\\alpha`` mixes that limit with the uniform distribution.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Pagerank(;
        n::Integer = 100,
        alpha::Number = 0.85,
        epsilon::Number = 1e-6
    ) -> Pagerank

Keywords correspond to the struct's fields.

## Validation

  - `n > 0`.
  - `0 < alpha < 1`.
  - `epsilon > 0`.

# Examples

```jldoctest
julia> Pagerank(; alpha = 0.9, n = 200, epsilon = 1e-8)
Pagerank
        n ┼ Int64: 200
    alpha ┼ Float64: 0.9
  epsilon ┴ Float64: 1.0e-8
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`centrality_polarity`](@ref)
  - [`TopologyOnly`](@ref)
  - [`Graphs.pagerank`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.pagerank-Union%7BTuple%7BAbstractGraph%7BU%7D%7D,%20Tuple%7BU%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any,%20Integer%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any,%20Integer,%20Any%7D%7D%20where%20U%3C:Integer)

# References

  - $(ref_dict[:brin1998]) Section 2.1.1. $(ref_dict[:no_original_source])
"""
@concrete struct Pagerank <: AbstractCentralityAlgorithm
    """
    Largest number of iterations. `Graphs.pagerank` raises an error if the scores do not converge in this number of sweeps.
    """
    n
    """
    Damping factor. It is the probability that the walk follows an edge, and does not teleport.
    """
    alpha
    """
    Convergence threshold. A sweep converges when the L1 change of the score vector falls below this value multiplied by the number of assets.
    """
    epsilon
    function Pagerank(n::Integer, alpha::Number, epsilon::Number)
        assert_gt0(n, :n)
        assert_unit_interval(alpha, :alpha)
        assert_gt0(epsilon, :epsilon)
        return new{typeof(n), typeof(alpha), typeof(epsilon)}(n, alpha, epsilon)
    end
end
function Pagerank(; n::Integer = 100, alpha::Number = 0.85,
                  epsilon::Number = 1e-6)::Pagerank
    return Pagerank(n, alpha, epsilon)
end
"""
$(DocStringExtensions.TYPEDEF)

Scores each asset by its mean shortest-path distance, measured against the network's diameter.

`RadialityCentrality` selects the radiality centrality of `Graphs.jl`. It measures how near an asset is to all other assets, relative to the diameter of the network.

It declares [`DistancePolarity`](@ref), unless `ov` overrides it. It reads the lengths of shortest paths against the diameter of the graph, so its weights must be distances. It reads them on both branches, and the weights change its answer. Over twenty assets, the measured largest absolute change is `0.248` on a triangulated maximally filtered graph and `0.234` on a tree. Set `ov` to [`TopologyOnly`](@ref) to withdraw the declaration and read the topology alone.

# Mathematical definition

```math
\\begin{align}
    \\bar{\\ell}_i &= \\dfrac{1}{n - 1}\\sum_{j} \\ell_{i,\\,j}\\,, \\\\
    \\mathrm{RC}_i &= \\dfrac{D + 1 - \\bar{\\ell}_i}{D}\\,.
\\end{align}
```

Where:

  - ``\\bar{\\ell}_i``: Mean length from asset ``i`` to every other asset.
  - ``\\mathrm{RC}_i``: Radiality centrality of asset ``i``.
  - ``D``: Diameter of the network, the largest ``\\ell_{i,\\,j}`` over every pair.
  - $(math_dict[:ell_ij_path])
  - $(math_dict[:n_network])

The diameter makes this score different from [`ClosenessCentrality`](@ref). Closeness takes the reciprocal of the mean length. Radiality subtracts the mean length from the diameter plus one, so the score states how far inside the reach of the network an asset is.

The smallest score is ``1/D``, for an asset whose mean length is equal to the diameter. On an unweighted network every length is at least ``1``, so the largest score is ``1``, for an asset one edge away from every other asset. On a weighted network a length can be less than ``1``, and a score can be larger than ``1``. The bounds change with ``D``, so the scores of two networks are comparable only when the two have the same diameter.

[brandes2001](@cite) states the formula, and gives it to [valente1998](@cite). No copy of that work is available. [brandes2001](@cite) sums ``D + 1 - \\ell_{i,\\,j}`` over every asset ``j``, with the asset ``i``, and divides by ``(n - 1) D``. In that form every score is larger by ``(D + 1) / ((n - 1) D)``, so the ranking is the same. The form above, which `Graphs.jl` computes, does not include ``j = i``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RadialityCentrality(;
        ov::Option{TopologyOnly} = nothing
    ) -> RadialityCentrality

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> RadialityCentrality()
RadialityCentrality
  ov ┴ nothing
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`centrality_polarity`](@ref)
  - [`DistancePolarity`](@ref)
  - [`TopologyOnly`](@ref)
  - [`Graphs.radiality_centrality`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.radiality_centrality-Tuple%7BAbstractGraph%7D)

# References

  - $(ref_dict[:valente1998])
  - $(ref_dict[:brandes2001]) Section 4. $(ref_dict[:no_original_source])
"""
@concrete struct RadialityCentrality <: AbstractCentralityAlgorithm
    """
    $(field_dict[:ctov])
    """
    ov
    function RadialityCentrality(ov::Option{TopologyOnly})
        return new{typeof(ov)}(ov)
    end
end
function RadialityCentrality(; ov::Option{TopologyOnly} = nothing)::RadialityCentrality
    return RadialityCentrality(ov)
end
"""
$(DocStringExtensions.TYPEDEF)

Counts the shortest paths of the network that pass through each asset.

`StressCentrality` selects the stress centrality of `Graphs.jl`. It counts the shortest paths that go through each asset.

It declares [`DistancePolarity`](@ref), unless `ov` overrides it. It counts shortest paths, so its weights must be distances. As for [`BetweennessCentrality`](@ref), the weights do not change it on a tree, where the structure alone sets the shortest paths. They change it on the similarity branch. Set `ov` to [`TopologyOnly`](@ref) to withdraw the declaration and read the topology alone.

# Mathematical definition

```math
\\begin{align}
    \\mathrm{SC}_i &= \\sum_{s \\neq i \\neq t} \\sigma_{s,\\,t}(i)\\,,
\\end{align}
```

Where:

  - ``\\mathrm{SC}_i``: Stress centrality of asset ``i``.
  - $(math_dict[:sigma_st_i_paths])

The sum runs over the ordered pairs of distinct assets, so an undirected network counts each pair two times. The score has no normalisation, so it is a count and not a rate. It grows with the size of the network. The scores of two networks are comparable only when the two have the same number of assets.

[brandes2001](@cite) states this sum, and gives it to [shimbel1953](@cite). No copy of that work is available.

It is the sum of [`BetweennessCentrality`](@ref) without the division by ``\\sigma_{s,\\,t}``. A pair that many shortest paths join adds many units here, and one unit to the betweenness. This score measures how many shortest paths go through an asset. The betweenness measures the share of the shortest paths of each pair that go through it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    StressCentrality(;
        args::Tuple = (),
        kwargs::NamedTuple = (;),
        ov::Option{TopologyOnly} = nothing
    ) -> StressCentrality

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:ctargs_nomat])

# Examples

```jldoctest
julia> StressCentrality()
StressCentrality
    args ┼ Tuple{}: ()
  kwargs ┼ @NamedTuple{}: NamedTuple()
      ov ┴ nothing
```

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`centrality_polarity`](@ref)
  - [`DistancePolarity`](@ref)
  - [`TopologyOnly`](@ref)
  - [`Graphs.stress_centrality`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.stress_centrality)

# References

  - $(ref_dict[:shimbel1953])
  - $(ref_dict[:brandes2001]) Section 2. $(ref_dict[:no_original_source])
"""
@concrete struct StressCentrality <: AbstractCentralityAlgorithm
    """
    $(field_dict[:ctargs])
    """
    args
    """
    $(field_dict[:ctkwargs])
    """
    kwargs
    """
    $(field_dict[:ctov])
    """
    ov
    function StressCentrality(args::Tuple, kwargs::NamedTuple, ov::Option{TopologyOnly})
        assert_centrality_args(StressCentrality, args)
        return new{typeof(args), typeof(kwargs), typeof(ov)}(args, kwargs, ov)
    end
end
function StressCentrality(; args::Tuple = (), kwargs::NamedTuple = (;),
                          ov::Option{TopologyOnly} = nothing)::StressCentrality
    return StressCentrality(args, kwargs, ov)
end
"""
    calc_centrality(ct::AbstractCentralityAlgorithm, g::Graphs.AbstractGraph)

Compute the centrality score of every vertex of a graph with the algorithm `ct`.

The type of `ct` selects one routine of [`Graphs.jl`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/).

`g` can be weighted or unweighted, and this function does not check which. The default `distmx` of every routine of `Graphs.jl` that takes one is `weights(g)`. [`centrality_graph`](@ref) makes the choice one time, and this function only calls the routine. [`centrality_graph`](@ref) does not give a weighted graph to an algorithm that declares no polarity, because `Graphs.katz_centrality` throws an `InexactError` on a weighted graph.

# Algorithm

 1. Select the routine of `Graphs.jl` that the type of `ct` names, from the list under `# Arguments`.

 2. Splat the configuration of `ct` into that call.

      + [`BetweennessCentrality`](@ref), [`ClosenessCentrality`](@ref) and [`StressCentrality`](@ref): `args` and `kwargs`.
      + [`DegreeCentrality`](@ref): `kind` and `kwargs`.
      + [`KatzCentrality`](@ref): `alpha`.
      + [`Pagerank`](@ref): `alpha`, `n` and `epsilon`.
      + [`EigenvectorCentrality`](@ref) and [`RadialityCentrality`](@ref): nothing.

 3. Return the scores of that routine, one for each vertex of `g`.

# Arguments

  - `ct`: Centrality algorithm.

      + `ct::BetweennessCentrality`: `Graphs.betweenness_centrality`.
      + `ct::ClosenessCentrality`: `Graphs.closeness_centrality`.
      + `ct::DegreeCentrality`: `Graphs._degree_centrality`.
      + `ct::EigenvectorCentrality`: `Graphs.eigenvector_centrality`.
      + `ct::KatzCentrality`: `Graphs.katz_centrality`.
      + `ct::Pagerank`: `Graphs.pagerank`.
      + `ct::RadialityCentrality`: `Graphs.radiality_centrality`.
      + `ct::StressCentrality`: `Graphs.stress_centrality`.

  - `g`: Graph to score.

# Returns

  - `scores::VecNum`: One centrality score for each vertex of the graph.

# Related

  - [`AbstractCentralityAlgorithm`](@ref)
  - [`BetweennessCentrality`](@ref)
  - [`ClosenessCentrality`](@ref)
  - [`DegreeCentrality`](@ref)
  - [`EigenvectorCentrality`](@ref)
  - [`KatzCentrality`](@ref)
  - [`Pagerank`](@ref)
  - [`RadialityCentrality`](@ref)
  - [`StressCentrality`](@ref)
"""
function calc_centrality(ct::BetweennessCentrality, g::Graphs.AbstractGraph)
    return Graphs.betweenness_centrality(g, ct.args...; ct.kwargs...)
end
function calc_centrality(ct::ClosenessCentrality, g::Graphs.AbstractGraph)
    return Graphs.closeness_centrality(g, ct.args...; ct.kwargs...)
end
function calc_centrality(ct::DegreeCentrality, g::Graphs.AbstractGraph)
    return Graphs._degree_centrality(g, ct.kind; ct.kwargs...)
end
function calc_centrality(::EigenvectorCentrality, g::Graphs.AbstractGraph)
    return Graphs.eigenvector_centrality(g::Graphs.AbstractGraph)
end
function calc_centrality(ct::KatzCentrality, g::Graphs.AbstractGraph)
    return Graphs.katz_centrality(g, ct.alpha)
end
function calc_centrality(ct::Pagerank, g::Graphs.AbstractGraph)
    return Graphs.pagerank(g, ct.alpha, ct.n, ct.epsilon)
end
function calc_centrality(::RadialityCentrality, g::Graphs.AbstractGraph)
    return Graphs.radiality_centrality(g::Graphs.AbstractGraph)
end
function calc_centrality(ct::StressCentrality, g::Graphs.AbstractGraph)
    return Graphs.stress_centrality(g, ct.args...; ct.kwargs...)
end

export AbstractCentralityAlgorithm, TopologyOnly, BetweennessCentrality,
       ClosenessCentrality, DegreeCentrality, EigenvectorCentrality, KatzCentrality,
       Pagerank, RadialityCentrality, StressCentrality
# The verb the `# Interfaces` section of `AbstractCentralityAlgorithm` names (ADR 0154).
# Public, not exported: it is only ever extended, and an extension must qualify it as
# `PortfolioOptimisers.calc_centrality` anyway.
public calc_centrality
