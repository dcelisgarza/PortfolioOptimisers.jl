"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all minimum spanning tree (MST) algorithm types.

All concrete and/or abstract types implementing specific MST algorithms (e.g., Kruskal, Boruvka, Prim) should be subtypes of `AbstractTreeType`.

# Related

  - [`KruskalTree`](@ref)
  - [`BoruvkaTree`](@ref)
  - [`PrimTree`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 13.1.4.1.
  - $(ref_dict[:mantegna1999])
"""
abstract type AbstractTreeType <: AbstractPhylogenyAlgorithm end
"""
    const Tree_SimMat = Union{<:AbstractNonNegativeSimilarityMatrixAlgorithm,
                              <:AbstractTreeType}

Alias for a tree or similarity matrix algorithm.

Matches either an [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref) or an [`AbstractTreeType`](@ref). Used for dispatch in phylogeny estimation where either a spanning tree or a similarity matrix approach may be used.

The similarity half is the **narrow** family, not [`AbstractSimilarityMatrixAlgorithm`](@ref): the similarity branch builds a PMFG, whose consumers cannot take a negative weight. So [`MaximumDistanceSimilarity`](@ref), [`ExponentialSimilarity`](@ref), [`GeneralExponentialSimilarity`](@ref) and [`ComplementSimilarity`](@ref) match, and [`AngularSimilarity`](@ref) does not.

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

Grows the minimum spanning tree by taking the lightest edge that joins two components.

`KruskalTree` specifies the use of [Kruskal's algorithm](https://juliagraphs.org/Graphs.jl/stable/algorithms/spanningtrees/#Graphs.kruskal_mst) for constructing a minimum spanning tree from a graph.

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

  - $(ref_dict[:kruskal1956])
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

Grows the minimum spanning tree by joining every component to its own lightest neighbour at once.

`BoruvkaTree` specifies the use of [Boruvka's algorithm](https://juliagraphs.org/Graphs.jl/stable/algorithms/spanningtrees/#Graphs.boruvka_mst) for constructing a minimum spanning tree from a graph.

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

Grows the minimum spanning tree outward from a single starting vertex.

`PrimTree` specifies the use of [Prim's algorithm](https://juliagraphs.org/Graphs.jl/stable/algorithms/spanningtrees/#Graphs.prim_mst) for constructing a minimum spanning tree from a graph.

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

  - $(ref_dict[:prim1957])
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
function PrimTree(; args::Tuple = (), kwargs::NamedTuple = (;))
    return PrimTree(args, kwargs)
end
"""
    calc_mst(alg::AbstractTreeType, g::Graphs.AbstractGraph)

Compute the minimum spanning tree (MST) of a graph using the specified algorithm.

This function dispatches to the appropriate MST computation from `Graphs.jl` based on the type of `alg`. Supported algorithms include Kruskal, Boruvka, and Prim.

# Algorithm

 1. Select the `Graphs.jl` spanning-tree routine that the type of `alg` names.
 2. Splat `alg.args` and `alg.kwargs` into that call. [`assert_tree_args`](@ref) refused every entry that could re-weight or re-orient the search, so the tree is minimised over the weights `g` already carries.
 3. Read the edge vector out of the answer. `Graphs.boruvka_mst` answers with a named tuple whose first field holds it, and the other two routines answer with the vector itself.

# Arguments

  - `alg`: MST algorithm to use.

      + `alg::KruskalTree`: Computes the MST using Kruskal's algorithm.
      + `alg::BoruvkaTree`: Computes the MST using Boruvka's algorithm.
      + `alg::PrimTree`: Computes the MST using Prim's algorithm.

  - `g::Graphs.AbstractGraph`: Graph to compute the MST on.

# Returns

  - `tree::Vector`: Vector of edges representing the MST.

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
