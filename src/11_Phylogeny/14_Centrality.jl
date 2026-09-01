"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the algorithms that score how central each asset is in a network.

Every member wraps one routine of [`Graphs.jl`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/).

All concrete and/or abstract types implementing specific centrality algorithms (e.g., betweenness, closeness, degree, eigenvector, Katz, pagerank, radiality, stress) should be subtypes of `AbstractCentralityAlgorithm`.

# Each member declares the weights it needs

A member says which quantity its edge weights must be, through [`centrality_polarity`](@ref), and [`centrality_graph`](@ref) supplies it. The declaration is about **correctness** — a shortest path over similarities is backwards — and never about capability: a member that declares nothing, and a source that carries no weights, both run on the plain graph rather than raising. The fallback declares `nothing`, so a new member is unweighted until it opts in.

The five members that do declare one carry an `ov` field, and [`TopologyOnly`](@ref) in it withdraws the declaration for that instance. [`centrality_polarity`](@ref) therefore answers the **effective** polarity, not the declared one.

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

  - $(ref_dict[:cajas2025]) Section 13.1.3.
  - $(ref_dict[:estrada2011]) Chapter 7.
"""
abstract type AbstractCentralityAlgorithm <: AbstractPhylogenyAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Withdraws an algorithm's polarity declaration, so it reads the network's topology alone.

An algorithm that declares a polarity is handed weights wherever the source carries them. `TopologyOnly` in its `ov` field withdraws that request: [`centrality_polarity`](@ref) then answers `nothing`, and [`centrality_graph`](@ref) routes to the plain `Graphs.SimpleGraph` of [`phylogeny_matrix`](@ref). The computation is the one that already runs for [`DegreeCentrality`](@ref), [`Pagerank`](@ref) and [`KatzCentrality`](@ref), so this is a redirect and never a new estimator.

# The override runs one way only

It removes weights and never supplies them. There is no value that forces a polarity onto an algorithm, and the field is deliberately **not** typed over [`AbstractCentralityPolarity`](@ref). A forced polarity would succeed rather than raise — [`calc_distance_weighted_graph`](@ref) carries distances on both branches — and the algorithm would read a distance where it needs a similarity, reversing its own ordering in silence. Polarity correctness is not a runtime property, so nothing could catch it.

# Every request is honoured, on every source

The answer over the topology alone is available from every source, so the override never warns and never goes inert. On a partition source, on a precomputed [`PhylogenyResult`](@ref), and on the tree branch under [`SimilarityPolarity`](@ref), the plain graph is what those routes already build, so the request is satisfied before it is made.

Only the five algorithms that declare a polarity carry an `ov` field. [`DegreeCentrality`](@ref), [`Pagerank`](@ref) and [`KatzCentrality`](@ref) already return the topology-only answer, so there is nothing for them to override and `DegreeCentrality(; ov = TopologyOnly())` is a `MethodError`.

# It is a choice, not a simplification, and it moves no default

A topology-only centrality is often argued to be the more **fold-stable** of the two, by the same reasoning that makes a fixed `dmax` fold-stable under [`PathLength`](@ref). That is not a reason to default to it.

**The shipped default is already unweighted.** [`CentralityEstimator`](@ref)'s `ct` defaults to [`DegreeCentrality`](@ref), which declares no polarity, so a caller who names no algorithm gets this answer already. Defaulting `ov` to `TopologyOnly` would change the answer only for a caller who named one of the five deliberately — and for those five, reading the weights the source carries is the correct answer, which is what [`AbstractCentralityPolarity`](@ref) exists to say.

**The override re-arms `sep`.** The plain-graph route reads the separation closure [`phylogeny_matrix`](@ref) builds, and the weighted routes bypass it. So the override trades the edge weights for a second knob rather than removing one: measured over twenty assets, all five algorithms answer differently at `HopCount(; n = 1)` and at `n = 3` once they carry it, including the four that are inert to `sep` without it. Under a bare [`PathLength`](@ref) that knob is the **observed diameter**, which is the data-dependent quantity the fold-stability argument set out to avoid.

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

Refuse an entry of `args` that would reach a second weighting channel.

The one refusal both splat guards make. A `Graphs.jl` entry point takes its weights in a positional slot, so an entry of `args` with that shape is a second channel answering a question the declared one already answered. [`assert_centrality_args`](@ref) and [`assert_tree_args`](@ref) differ only in which shape reaches a channel and in which declared channel they name, so both call this with their own `S`, `shape` and `channel`.

The index of the offending entry is reported with its type, because `args` is splatted and the caller sees no argument names.

# Algorithm

 1. Find the index of the first entry of `args` that is an `S`, giving `idx`.
 2. Throw a [`ConflictingArgumentError`](@ref) when `idx` is an index rather than `nothing`. The message carries `T`, `shape`, `channel`, `idx` and the type of the offending entry.

# Arguments

  - `T`: Algorithm type, named in the error message.
  - `args`: Positional arguments destined for the `Graphs.jl` function.
  - `S`: The shape that reaches a weight slot, e.g. `AbstractMatrix`.
  - `shape`: Name of that shape, used in the error message.
  - `channel`: Sentence naming the declared channel the caller must use instead.

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

`args` is splatted straight into the `Graphs.jl` centrality function, so a matrix in it is a `distmx` — a second, undeclared way to weight the graph. [`centrality_polarity`](@ref) is the declared one, and it picks the weights the algorithm's own mathematics needs, from the structure that was actually built. Two channels answering the same question is one too many, and this one was never safe:

  - `Graphs.betweenness_centrality`'s `distmx` is its **third** positional argument, so a matrix in `args` binds to `vs` instead and the call **overflows the stack** inside `Graphs.degree`. The `StackOverflowError` is catchable and the process survives it, so what is lost is the call and not the session.
  - `Graphs.closeness_centrality`'s is its second, so that one worked — silently overriding the polarity, and reporting a wrong-sized matrix as a `BoundsError` rather than a `DimensionMismatch`.
  - `Graphs.stress_centrality` has no `distmx` at all.

Non-matrix entries are untouched: a vertex list or a sample count is a genuine positional argument of those functions and says nothing about weights.

`kwargs` needs no companion guard. A keyword binds **by name**, so a matrix there cannot reach a positional slot: none of the four functions declares a matrix-valued keyword, and every one of `normalize`, `endpoints`, `rng` and `seed` refuses a matrix on its own. The whole family fails closed with a `MethodError` or a `TypeError`.

# Algorithm

 1. Refuse an `AbstractMatrix` in `args` with [`assert_no_weight_channel_args`](@ref), naming [`centrality_polarity`](@ref) as the declared weight channel.

# Arguments

  - `T`: Centrality algorithm type, named in the error message.
  - `args`: Positional arguments destined for the `Graphs.jl` centrality function.

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

Refuse a second weighting channel inside a spanning-tree algorithm's `args` and `kwargs`.

Both fields are splatted straight into the `Graphs.jl` spanning-tree function, and **every** channel they can reach re-weights or re-orients a tree that [`calc_weighted_adjacency_graph`](@ref) has already weighted. The graph it hands to [`calc_mst`](@ref) carries the distances the estimator's `de` and `ce` produced, and `Graphs.jl` defaults `distmx` to exactly those weights. A caller who fills these fields therefore answers a question that was already answered, and the wrong answer is **silent**:

  - `kruskal_mst`, `boruvka_mst` and `prim_mst` all take `distmx` as their second positional argument, so a matrix in `args` replaces the estimator's distances outright. It is correctly sized often enough to succeed, and the tree it builds is a legitimate-looking tree of the wrong graph.
  - `kruskal_mst` also takes a `weight_vector` there, which is the same override in the other shape.
  - `minimize` in `kwargs` inverts the sense of the search. The tree branch is *defined* by minimising a distance — [`calc_weighted_adjacency_graph`](@ref) and [`SimilarityPolarity`](@ref) both say so — and `minimize = false` yields a **maximum** spanning tree while everything downstream still reads it as a minimum one.

Non-matrix, non-vector entries are untouched, and so is every other keyword. Those reach no weighting channel, and the three functions declare none, so they fail closed at the call.

# Algorithm

 1. Refuse an `AbstractMatrix` or an `AbstractVector` in `args` with [`assert_no_weight_channel_args`](@ref), naming the graph [`calc_weighted_adjacency_graph`](@ref) built as the declared weight channel.
 2. Throw a [`ConflictingArgumentError`](@ref) when `kwargs` carries the key `minimize`. The message carries `T` and the value of that key.

# Arguments

  - `T`: Spanning-tree algorithm type, named in the error message.
  - `args`: Positional arguments destined for the `Graphs.jl` spanning-tree function.
  - `kwargs`: Keyword arguments destined for the same function.

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

Compute the centrality vector from a matrix [`PhylogenyResult`](@ref) using the specified centrality algorithm.

Builds a graph from the phylogeny matrix and applies `ct` to compute node centrality scores.

The graph is **always unweighted**, whatever polarity `ct` declares. A precomputed [`PhylogenyResult`](@ref) is a matrix of `0`s and `1`s, so it is one of the weightless sources listed on [`centrality_vector`](@ref)'s warning, and the weights it does not carry cannot be recovered from it. Pass the estimator instead of its result to get the weighted answer.

# Algorithm

 1. Read `plr.X`, the precomputed phylogeny matrix, into a plain `Graphs.SimpleGraph`, giving the structure `G`.
 2. Score the vertices of `G` with [`calc_centrality`](@ref), giving the centrality vector.
 3. Wrap that vector in a [`PhylogenyResult`](@ref).

# Arguments

  - `plr`: Phylogeny matrix result object.
  - $(arg_dict[:cta])
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `plr::PhylogenyResult{<:VecNum}`: Centrality scores for each asset.

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

`BetweennessCentrality` computes the [betweenness centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.betweenness_centrality) of nodes in a graph, measuring the extent to which a node lies on shortest paths between other nodes.

Declares [`DistancePolarity`](@ref), unless `ov` overrides it: it is defined over shortest paths, so its weights must be distances. On a tree the weighted answer equals the unweighted one — a tree has exactly one path between any two vertices, so no weighting can change the shortest-path set — which is a theorem about the graph rather than a limitation, and it does not hold on the similarity branch. Set `ov` to [`TopologyOnly`](@ref) to withdraw the declaration and read the topology alone.

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

The sum runs over the ordered pairs of distinct assets, so an undirected network counts each pair twice and the leading factor is the reciprocal of ``(n - 1)(n - 2)`` rather than of half of it. `Graphs.jl` applies that factor by default, and `kwargs = (; normalize = false)` replaces it by ``1/2``, which is the count over the unordered pairs. `kwargs = (; endpoints = true)` counts the two ends of every path as well, which raises every score.

A pair joined by several shortest paths shares one unit of score between them, because the summand is a fraction of ``\\sigma_{s,\\,t}``. [`StressCentrality`](@ref) omits that division and counts the paths themselves.

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
  - $(ref_dict[:brandes2001])
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

`ClosenessCentrality` computes the [closeness centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.closeness_centrality) of nodes in a graph, measuring how close a node is to all other nodes.

Declares [`DistancePolarity`](@ref), unless `ov` overrides it: it sums shortest-path lengths, so its weights must be distances. It reads them on **both** branches, so its answer on a [`NetworkEstimator`](@ref) source differs from the unweighted one — measured over twenty assets, a maximum absolute change of `0.713` on a triangulated maximally filtered graph and `0.538` on a tree. Set `ov` to [`TopologyOnly`](@ref) to withdraw the declaration and read the topology alone.

# Mathematical definition

```math
\\begin{align}
    \\mathrm{CC}_i &= \\dfrac{r_i}{\\displaystyle\\sum_{j \\in \\mathcal{R}_i} \\ell_{i,\\,j}} \\cdot \\dfrac{r_i}{n - 1}\\,,
\\end{align}
```

Where:

  - ``\\mathrm{CC}_i``: Closeness centrality of asset ``i``.
  - ``\\mathcal{R}_i``: Set of assets that asset ``i`` reaches, excluding itself.
  - ``r_i``: Cardinality of ``\\mathcal{R}_i``.
  - $(math_dict[:ell_ij_path])
  - $(math_dict[:n_network])

The first factor is the reciprocal of the mean length from asset ``i`` to the assets it reaches. The second is the share of the universe it reaches, which `Graphs.jl` applies by default and `kwargs = (; normalize = false)` drops. On a connected network ``r_i = n - 1`` and the second factor is one, so the two settings agree there and part only where the network falls into components.

An asset that reaches nothing scores zero rather than an infinity, because the sum in the denominator is over ``\\mathcal{R}_i`` alone and an unreachable asset never enters it.

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

  - $(ref_dict[:freeman1979])
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

`DegreeCentrality` computes the [degree centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.degree_centrality-Tuple%7BAbstractGraph%7D) of nodes in a graph. It is the simplest score of the family, and the shipped default of [`CentralityEstimator`](@ref)'s `ct`.

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

`Graphs.jl` **normalises** that vector by default, so what this type returns is ``\\mathbf{D}_n / (n - 1)`` and not ``\\mathbf{D}_n``. Measured over the minimum spanning tree of the last 253 observations of the 20-asset sample in `test/assets/SP500.csv.gz`, the first six entries of ``\\mathbf{D}_n`` are `[3, 2, 1, 1, 2, 3]` and the returned scores are `[0.1579, 0.1053, 0.0526, 0.0526, 0.1053, 0.1579]`, a maximum absolute difference of `4.7368421052631575`. `kwargs = (; normalize = false)` recovers ``\\mathbf{D}_n`` exactly.

The factor is the whole difference, and it re-ranks nothing. [`average_centrality`](@ref) is linear in the score vector, so a constant scale moves the average by that same constant.

# The three `kind` values coincide on these structures

`kind` selects the total, the in- or the out-degree. Every graph this library builds is **undirected**, where the three are one number: measured over the same tree, `kind = 0`, `1` and `2` agree exactly. The field is kept because `Graphs.jl` takes it, not because it selects anything here.

Declares no polarity and runs on the plain graph: `Graphs.degree_centrality` counts edges and ignores what they weigh. It is therefore one of the algorithms for which the estimator's `sep` stays **live** — the unweighted route reads the separation closure [`phylogeny_matrix`](@ref) builds, so `HopCount(; n = 2)` does change this answer.

It carries no `ov` field, and [`TopologyOnly`](@ref) is not applicable to it: the topology alone is what it already reads, so there is no declaration to withdraw. `DegreeCentrality(; ov = TopologyOnly())` is a `MethodError`.

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

  - $(ref_dict[:cajas2025]) Section 13.1.3.1, Equation 13.3.
  - $(ref_dict[:freeman1979])
"""
@concrete struct DegreeCentrality <: AbstractCentralityAlgorithm
    """
    Degree type (0: total, 1: in-degree, 2: out-degree).
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

`EigenvectorCentrality` computes the eigenvector centrality of nodes in a graph, measuring the influence of a node based on the centrality of its neighbours.

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

The right-hand side is ``\\mathbf{q}_{\\mathrm{max}}`` itself, so the score is the leading eigenvector under whatever normalisation the eigensolver applies. `Graphs.jl` returns it with unit 2-norm, and takes the absolute value of every entry — the leading eigenvector of a non-negative matrix shares one sign, by the Perron-Frobenius theorem, so that changes no ordering. Measured over the triangulated maximally filtered graph of the last 253 observations of the 20-asset sample in `test/assets/SP500.csv.gz`, weighted by the similarities that selected its edges, the returned vector matches the formula above to within `1.0e-15`, has 2-norm `1.0` and runs from `0.07199` to `0.40756`. The residual is quoted as a bound rather than as a decimal because the eigensolver moves by about `6.0e-16` between calls on one graph.

Declares [`SimilarityPolarity`](@ref) — the only member that declares it — unless `ov` overrides it: it is the leading eigenvector of the adjacency matrix itself, so a stronger link must contribute a larger entry. It therefore reads weights on the similarity branch alone. A tree is selected by minimising a distance and carries no similarity, so this algorithm runs unweighted there rather than being handed the wrong quantity. Set `ov` to [`TopologyOnly`](@ref) to withdraw the declaration and read the topology alone.

The weights change the answer by less than the shortest-path algorithms do, and they do change it: over the same triangulated maximally filtered graph the weighted and unweighted vectors differ by a maximum absolute `0.02561` and correlate `0.99361`, on entries whose median is `0.1969` on either vector. Withdrawing the weights also moves the structure's ``\\lambda_{\\mathrm{max}}``, from `4.844612909369407` to `6.174911353215694`.

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

  - $(ref_dict[:cajas2025]) Section 13.1.3.2, Equation 13.4.
  - $(ref_dict[:bonacich1987])
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

`KatzCentrality` computes the [Katz centrality](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.katz_centrality) of nodes in a graph, measuring the influence of a node based on the number and length of walks between nodes, controlled by the attenuation factor `alpha`.

Declares no polarity and runs on the plain graph: `Graphs.katz_centrality` binarises its input through `adjacency_matrix(g, Bool)`, and throws an `InexactError` when the graph is weighted. The unweighted route is real code here rather than an absent check.

It carries no `ov` field, and [`TopologyOnly`](@ref) is not applicable to it: the topology alone is what it already reads, so there is no declaration to withdraw. `KatzCentrality(; ov = TopologyOnly())` is a `MethodError`.

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

The series converges to the resolvent only for ``\\alpha < 1 / \\lambda_{\\mathrm{max}}``, and the two right-hand sides are equal only there. Outside that range the resolvent is still defined at almost every ``\\alpha``, and the vector it gives is the walk sum of nothing.

# `alpha` must be below the reciprocal of the largest eigenvalue

Above the bound the linear solve still returns a vector, and the vector is not a centrality: measured over the minimum spanning tree of the last 253 observations of the 20-asset sample in `test/assets/SP500.csv.gz`, ``\\lambda_{\\mathrm{max}} = 2.57344493899609`` and the bound is `0.388584183343788`. At `alpha = 0.3` every score is positive, between `0.10752` and `0.46421`. At `alpha = 0.5` the scores run `-0.55627` to `0.19624`, eleven of the twenty are negative, and a negative centrality has no reading.

**The constructor cannot check this.** ``\\lambda_{\\mathrm{max}}`` is a property of the graph, and the graph is built later by [`centrality_graph`](@ref), so the validation is `alpha > 0` and the bound is the caller's to respect. A dense network raises ``\\lambda_{\\mathrm{max}}`` and lowers the bound, so a value that held on a tree can fail on a triangulated maximally filtered graph over the same assets: over those same 20 assets the filtered graph has ``\\lambda_{\\mathrm{max}} = 6.174911353215694``, a bound of `0.16194564468998124`, and the default `alpha = 0.3` is outside it.

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

  - $(ref_dict[:katz1953])
"""
@concrete struct KatzCentrality <: AbstractCentralityAlgorithm
    """
    Attenuation factor for Katz centrality.
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

`Pagerank` computes the [PageRank](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/#Graphs.pagerank-Union%7BTuple%7BAbstractGraph%7BU%7D%7D,%20Tuple%7BU%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any,%20Integer%7D,%20Tuple%7BAbstractGraph%7BU%7D,%20Any,%20Integer,%20Any%7D%7D%20where%20U%3C:Integer) of nodes in a graph, measuring the importance of nodes based on the structure of incoming links. The algorithm is controlled by the damping factor `alpha`, number of iterations `n`, and convergence tolerance `epsilon`.

Declares no polarity and runs on the plain graph: `Graphs.pagerank` walks `outdegree` and `inneighbors` alone and never reads an edge weight. Measured over a 20-asset triangulated maximally filtered graph, the weighted and the plain graph give the identical vector, to `0.0`. Like [`DegreeCentrality`](@ref) it therefore keeps the estimator's `sep` live, reading the separation closure rather than the structure.

It carries no `ov` field, and [`TopologyOnly`](@ref) is not applicable to it: the topology alone is what it already reads, so there is no declaration to withdraw. `Pagerank(; ov = TopologyOnly())` is a `MethodError`.

# Mathematical definition

```math
\\begin{align}
    \\mathrm{PR}_i &= \\dfrac{1 - \\alpha}{n} + \\dfrac{\\alpha}{n}\\sum_{j \\in \\mathcal{D}} \\mathrm{PR}_j + \\alpha \\sum_{j \\in \\mathcal{I}_i} \\dfrac{\\mathrm{PR}_j}{k_j}\\,,
\\end{align}
```

Where:

  - ``\\mathrm{PR}_i``: PageRank of asset ``i``, the share of its time the damped walk spends there.
  - ``\\alpha``: Damping factor, `alpha`. It is the probability that the walk follows an edge rather than teleporting.
  - ``\\mathcal{I}_i``: Set of assets carrying an edge into asset ``i``.
  - ``k_j``: Number of edges leaving asset ``j``.
  - ``\\mathcal{D}``: Set of dangling assets, those that no edge leaves.
  - $(math_dict[:n_network])

The three terms are the walk's three moves: it teleports to a uniformly drawn asset, it teleports out of a dangling asset it cannot leave, or it follows one of the edges into asset ``i``. Every score is therefore non-negative and the vector sums to one, which is what separates this member from the counting scores of the family.

Every network this library builds is **undirected**, where ``\\mathcal{I}_i`` is the neighbourhood of asset ``i`` and ``k_j`` is its degree. On a connected undirected network the solution approaches the degree vector [`DegreeCentrality`](@ref) counts, up to a scale, as ``\\alpha`` approaches one, and a smaller ``\\alpha`` blends that limit with the uniform distribution.

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

  - $(ref_dict[:brin1998])
"""
@concrete struct Pagerank <: AbstractCentralityAlgorithm
    """
    Number of iterations. `Graphs.pagerank` raises when the scores have not converged after this many sweeps.
    """
    n
    """
    Damping factor. It is the probability that the walk follows an edge rather than teleporting.
    """
    alpha
    """
    Convergence threshold. A sweep converges when the L1 change of the score vector falls below this value multiplied by the number of assets.
    """
    epsilon
    function Pagerank(n::Integer, alpha::Number, epsilon::Number)
        @argcheck(0 < n, DomainError)
        assert_unit_interval(alpha, :alpha)
        @argcheck(zero(epsilon) < epsilon, DomainError)
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

`RadialityCentrality` computes the radiality centrality of nodes in a graph, measuring how close a node is to all other nodes, adjusted for the maximum possible distance.

Declares [`DistancePolarity`](@ref), unless `ov` overrides it: it reads shortest-path lengths against the graph's diameter, so its weights must be distances. It reads them on both branches, and its answer moves when they arrive — measured over twenty assets, a maximum absolute change of `0.248` on a triangulated maximally filtered graph and `0.234` on a tree. Set `ov` to [`TopologyOnly`](@ref) to withdraw the declaration and read the topology alone.

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

The diameter is what separates this score from [`ClosenessCentrality`](@ref). Closeness reciprocates the mean length and reads a scale of its own; radiality subtracts it from the longest length the network holds, so the score says how far inside the network's own reach an asset sits.

The score is at most ``1``, which an asset one edge away from every other reaches, and at least ``1/D``, which an asset whose mean length equals the diameter reaches. Both bounds move with ``D``, so two networks give comparable scores only when the two have the same diameter.

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

`StressCentrality` computes the stress centrality of nodes in a graph, measuring the number of shortest paths passing through each node.

Declares [`DistancePolarity`](@ref), unless `ov` overrides it: it counts shortest paths, so its weights must be distances. Like [`BetweennessCentrality`](@ref) it is unchanged by them on a tree, where the shortest-path set is fixed by the structure alone, and does move on the similarity branch. Set `ov` to [`TopologyOnly`](@ref) to withdraw the declaration and read the topology alone.

# Mathematical definition

```math
\\begin{align}
    \\mathrm{SC}_i &= \\sum_{s \\neq i \\neq t} \\sigma_{s,\\,t}(i)\\,,
\\end{align}
```

Where:

  - ``\\mathrm{SC}_i``: Stress centrality of asset ``i``.
  - $(math_dict[:sigma_st_i_paths])

The sum runs over the ordered pairs of distinct assets, so an undirected network counts each pair twice. There is no normalisation, so the score is a count rather than a rate: it grows with the size of the network, and two networks give comparable scores only when the two hold the same number of assets.

It is [`BetweennessCentrality`](@ref)'s sum without the division by ``\\sigma_{s,\\,t}``. A pair joined by many shortest paths therefore contributes many units here and one unit there, so this score reads how much traffic an asset carries and betweenness reads how much of it the asset alone can carry.

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

Compute node centrality scores for a graph using the specified centrality algorithm.

This function dispatches to the appropriate centrality computation from [`Graphs.jl`](https://juliagraphs.org/Graphs.jl/stable/algorithms/centrality/) based on the type of `ct`. Supported algorithms include betweenness, closeness, degree, eigenvector, Katz, pagerank, radiality, and stress centrality.

`g` may be weighted or unweighted, and nothing here inspects which. `Graphs.jl` weights implicitly — the `distmx` of every routine that takes one defaults to `weights(g)` — so the choice is made once, by [`centrality_graph`](@ref), and this function only forwards. Handing a weighted graph to an algorithm that declares no polarity is what [`centrality_graph`](@ref) exists to prevent: `Graphs.katz_centrality` throws an `InexactError` on one.

# Algorithm

 1. Select the `Graphs.jl` routine that the type of `ct` names, from the list under `# Arguments`.
 2. Splat the configuration `ct` carries into that call. [`BetweennessCentrality`](@ref), [`ClosenessCentrality`](@ref) and [`StressCentrality`](@ref) pass `args` and `kwargs`; [`DegreeCentrality`](@ref) passes `kind` and `kwargs`; [`KatzCentrality`](@ref) passes `alpha`; [`Pagerank`](@ref) passes `alpha`, `n` and `epsilon`; [`EigenvectorCentrality`](@ref) and [`RadialityCentrality`](@ref) pass nothing.
 3. Return the scores that routine produced, one entry per vertex of `g`.

# Arguments

  - `ct`: Centrality algorithm to use.

      + `ct::BetweennessCentrality`: Computes betweenness centrality.
      + `ct::ClosenessCentrality`: Computes closeness centrality.
      + `ct::DegreeCentrality`: Computes degree centrality.
      + `ct::EigenvectorCentrality`: Computes eigenvector centrality.
      + `ct::KatzCentrality`: Computes Katz centrality.
      + `ct::Pagerank`: Computes PageRank.
      + `ct::RadialityCentrality`: Computes radiality centrality.
      + `ct::StressCentrality`: Computes stress centrality.

  - `g`: Graph to compute centrality on.

# Returns

  - `ct::VecNum`: Centrality scores for each node in the graph.

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
