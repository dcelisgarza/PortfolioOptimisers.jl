"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the algorithms that build a minimum spanning tree.

Every concrete or abstract type that selects a minimum spanning tree algorithm must subtype `AbstractTreeType`. A subtype goes in the `alg` field of a [`NetworkEstimator`](@ref), and [`calc_mst`](@ref) runs the algorithm that it selects.

# Related

  - [`KruskalTree`](@ref)
  - [`BoruvkaTree`](@ref)
  - [`PrimTree`](@ref)

# References

  - $(ref_dict[:mantegna1999]) Page 194, for the minimum spanning tree of the distances of Equation 2.
"""
abstract type AbstractTreeType <: AbstractPhylogenyAlgorithm end
"""
    const Tree_SimMat = Union{<:AbstractNonNegativeSimilarityMatrixAlgorithm,
                              <:AbstractTreeType}

Union of the tree algorithms and the non-negative similarity algorithms.

It is the bound of the `alg` field of a [`NetworkEstimator`](@ref), and of the `alg` argument of [`calc_weighted_adjacency`](@ref). A tree algorithm selects a minimum spanning tree, and a similarity algorithm selects a PMFG.

The similarity members come from [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref), and not from the wider [`AbstractSimilarityMatrixAlgorithm`](@ref). The similarity branch builds a PMFG, and the consumers of a PMFG cannot take a negative weight. [`MaximumDistanceSimilarity`](@ref), [`ExponentialSimilarity`](@ref), [`GeneralExponentialSimilarity`](@ref) and [`ComplementSimilarity`](@ref) are members, and [`AngularSimilarity`](@ref) is not.

# Related

  - [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)
  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`AbstractTreeType`](@ref)
  - [`NetworkEstimator`](@ref)
"""
const Tree_SimMat = Union{<:AbstractNonNegativeSimilarityMatrixAlgorithm,
                          <:AbstractTreeType}
"""
$(DocStringExtensions.TYPEDEF)

Grows the minimum spanning tree with the lightest edge that joins two components, one edge at a time.

`KruskalTree` selects [Kruskal's algorithm](https://juliagraphs.org/Graphs.jl/stable/algorithms/spanningtrees/#Graphs.kruskal_mst) for [`calc_mst`](@ref), which calls `Graphs.kruskal_mst`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    KruskalTree(;
        args::Tuple = (),
        kwargs::NamedTuple = (;)
    ) -> KruskalTree

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:treeargs_nochan])

# Examples

```jldoctest
julia> KruskalTree()
KruskalTree
    args ┼ Tuple{}: ()
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractTreeType`](@ref)
  - [`Graphs.kruskal_mst`](https://juliagraphs.org/Graphs.jl/stable/algorithms/spanningtrees/#Graphs.kruskal_mst)

# References

  - $(ref_dict[:kruskal1956]) Page 49, Construction A.
"""
@concrete struct KruskalTree <: AbstractTreeType
    """
    $(field_dict[:treeargs])
    """
    args
    """
    $(field_dict[:treekwargs])
    """
    kwargs
    function KruskalTree(args::Tuple, kwargs::NamedTuple)
        assert_tree_args(KruskalTree, args, kwargs)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function KruskalTree(; args::Tuple = (), kwargs::NamedTuple = (;))::KruskalTree
    return KruskalTree(args, kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

Grows the minimum spanning tree with one edge from every component to its lightest neighbour, in each round.

`BoruvkaTree` selects [Boruvka's algorithm](https://juliagraphs.org/Graphs.jl/stable/algorithms/spanningtrees/#Graphs.boruvka_mst) for [`calc_mst`](@ref), which calls `Graphs.boruvka_mst`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BoruvkaTree(;
        args::Tuple = (),
        kwargs::NamedTuple = (;)
    ) -> BoruvkaTree

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:treeargs_nochan])

# Examples

```jldoctest
julia> BoruvkaTree()
BoruvkaTree
    args ┼ Tuple{}: ()
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractTreeType`](@ref)
  - [`Graphs.boruvka_mst`](https://juliagraphs.org/Graphs.jl/stable/algorithms/spanningtrees/#Graphs.boruvka_mst)

# References

  - $(ref_dict[:boruvka1926])
  - $(ref_dict[:nesetril2001]) Section 2, for an English translation, and Section 4, for the step that joins each vertex to its nearest neighbour.
"""
@concrete struct BoruvkaTree <: AbstractTreeType
    """
    $(field_dict[:treeargs])
    """
    args
    """
    $(field_dict[:treekwargs])
    """
    kwargs
    function BoruvkaTree(args::Tuple, kwargs::NamedTuple)
        assert_tree_args(BoruvkaTree, args, kwargs)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function BoruvkaTree(; args::Tuple = (), kwargs::NamedTuple = (;))::BoruvkaTree
    return BoruvkaTree(args, kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

Grows the minimum spanning tree outward from one first vertex.

`PrimTree` selects [Prim's algorithm](https://juliagraphs.org/Graphs.jl/stable/algorithms/spanningtrees/#Graphs.prim_mst) for [`calc_mst`](@ref), which calls `Graphs.prim_mst`.

Construction B of [kruskal1956](@cite), one year before [prim1957](@cite), grows the tree from a fixed set of vertices. With one vertex in that set, it is the same algorithm.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PrimTree(;
        args::Tuple = (),
        kwargs::NamedTuple = (;)
    ) -> PrimTree

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:treeargs_nochan])

# Examples

```jldoctest
julia> PrimTree()
PrimTree
    args ┼ Tuple{}: ()
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractTreeType`](@ref)
  - [`Graphs.prim_mst`](https://juliagraphs.org/Graphs.jl/stable/algorithms/spanningtrees/#Graphs.prim_mst)

# References

  - $(ref_dict[:prim1957]) Pages 1391-1392, Principles 1 and 2, and the variant that grows a single fragment.
  - $(ref_dict[:kruskal1956]) Page 49, Construction B.
"""
@concrete struct PrimTree <: AbstractTreeType
    """
    $(field_dict[:treeargs])
    """
    args
    """
    $(field_dict[:treekwargs])
    """
    kwargs
    function PrimTree(args::Tuple, kwargs::NamedTuple)
        assert_tree_args(PrimTree, args, kwargs)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function PrimTree(; args::Tuple = (), kwargs::NamedTuple = (;))::PrimTree
    return PrimTree(args, kwargs)
end
"""
    calc_mst(alg::AbstractTreeType, g::Graphs.AbstractGraph)

Compute the minimum spanning tree of a graph with the algorithm that `alg` selects.

The type of `alg` selects one of three routines of `Graphs.jl`.

# Algorithm

 1. Select the spanning tree routine of `Graphs.jl` that the type of `alg` names.
 2. Splat `alg.args` and `alg.kwargs` into that call. The constructor of `alg` called [`assert_tree_args`](@ref), which refuses every entry that can re-weight or re-orient the search. The routine then minimises the tree over the weights that `g` carries.
 3. Return the vector of edges. `Graphs.boruvka_mst` returns a named tuple whose first field holds the vector, and the other two routines return the vector.

# Arguments

  - `alg`: Minimum spanning tree algorithm.

      + `alg::KruskalTree`: Kruskal's algorithm, through `Graphs.kruskal_mst`.
      + `alg::BoruvkaTree`: Boruvka's algorithm, through `Graphs.boruvka_mst`.
      + `alg::PrimTree`: Prim's algorithm, through `Graphs.prim_mst`.

  - `g::Graphs.AbstractGraph`: Weighted graph to span.

# Returns

  - `tree::Vector`: Edges of the minimum spanning tree.

# Related

  - [`KruskalTree`](@ref)
  - [`BoruvkaTree`](@ref)
  - [`PrimTree`](@ref)
"""
function calc_mst(ct::KruskalTree, g::Graphs.AbstractGraph)
    return Graphs.kruskal_mst(g, ct.args...; ct.kwargs...)
end
function calc_mst(ct::BoruvkaTree, g::Graphs.AbstractGraph)
    return Graphs.boruvka_mst(g, ct.args...; ct.kwargs...)[1]
end
function calc_mst(ct::PrimTree, g::Graphs.AbstractGraph)
    return Graphs.prim_mst(g, ct.args...; ct.kwargs...)
end

export KruskalTree, BoruvkaTree, PrimTree
