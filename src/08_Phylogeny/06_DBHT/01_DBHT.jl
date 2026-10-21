"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Direct Bubble Hierarchical Tree (DBHT) root selection methods.

The root is chosen inside [`CliqHierarchyTree2s`](@ref), which builds the clique hierarchy of the planar graph. A hierarchy needs one node with no parent, and the planar clique tree can present several candidates, so the choice of which of them becomes the root is a member of this family.

The candidates are the maximal 3-cliques of the graph, and there can be more than one. The hierarchy of [NHPG](@cite) is then a forest with one tree for each of them. [DBHTs](@cite) states no rule for the choice. The two members come from the MATLAB code that accompanies it.

# Related

  - [`UniqueRoot`](@ref)
  - [`EqualRoot`](@ref)
  - [`DBHT`](@ref)
  - [`CliqHierarchyTree2s`](@ref)

# References

  - $(ref_dict[:NHPG]) Theorem 2.8 and Corollary 3.6.
"""
abstract type DBHTRootMethod <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Joins every root candidate of the planar hierarchy under one root.

When the clique hierarchy has one root candidate, that 3-clique is the root. When it has several, [`CliqueRoot`](@ref) adds one synthetic clique and makes it the parent of every candidate. That clique is the imaginary 3-clique of [NHPG](@cite): its interior is the whole graph, so the hierarchy has a single maximal element.

# Related

  - [`DBHTRootMethod`](@ref)
  - [`EqualRoot`](@ref)
  - [`DBHT`](@ref)
  - [`CliqueRoot`](@ref)

# References

  - $(ref_dict[:NHPG]) Definition 4.1 and Corollary 4.2.
"""
struct UniqueRoot <: DBHTRootMethod end
"""
$(DocStringExtensions.TYPEDEF)

Builds one root from the adjacency tree of every root candidate.

This keeps several equally plausible roots of the DBHT hierarchy rather than choosing between them. [`CliqueRoot`](@ref) adds no clique. It joins two root candidates when they share two vertices.

[NHPG](@cite) leaves the hierarchy a forest of one tree for each candidate, and states no rule that joins them. The rule comes from the MATLAB code that accompanies [DBHTs](@cite), and no published work states it.

# Related

  - [`DBHTRootMethod`](@ref)
  - [`UniqueRoot`](@ref)
  - [`DBHT`](@ref)
  - [`CliqueRoot`](@ref)

# References

  - $(ref_dict[:NHPG]) Corollary 3.6. $(ref_dict[:no_original_source])
"""
struct EqualRoot <: DBHTRootMethod end
"""
$(DocStringExtensions.TYPEDEF)

Clusters assets by the bubble hierarchy of a triangulated maximally filtered graph.

`DBHT` is a composable clustering algorithm type for constructing hierarchical clusterings using the Direct Bubble Hierarchical Tree (DBHT) method, as described in [DBHTs](@cite).

[DBHTs](@cite) builds the planar maximally filtered graph of [tumminello2005](@cite), which inserts edges one at a time. This type builds the triangulated maximally filtered graph of [PMFG](@cite) with [`PMFG_T2s`](@ref) instead, as the MATLAB code that accompanies [DBHTs](@cite) does. Both are maximal planar graphs, so the bubble hierarchy of [DBHTs](@cite) applies to either.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DBHT(;
        sim::AbstractNonNegativeSimilarityMatrixAlgorithm = MaximumDistanceSimilarity(),
        root::DBHTRootMethod = UniqueRoot()
    ) -> DBHT

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> DBHT()
DBHT
   sim ┼ MaximumDistanceSimilarity()
  root ┴ UniqueRoot()
```

# Related

  - [`AbstractHierarchicalClusteringAlgorithm`](@ref)
  - [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)
  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`DBHTRootMethod`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)
  - [`UniqueRoot`](@ref)
  - [`EqualRoot`](@ref)

# References

  - $(ref_dict[:DBHTs]) Section "Methods", Equations 1-5.
  - $(ref_dict[:tumminello2005]) For the planar maximally filtered graph of the original DBHT method.
  - $(ref_dict[:PMFG]) Section III.A, Algorithm 1, for the graph that this type builds.
"""
@concrete struct DBHT <: AbstractHierarchicalClusteringAlgorithm
    """
    $(field_dict[:sim])
    """
    sim <: AbstractNonNegativeSimilarityMatrixAlgorithm
    """
    $(field_dict[:root])
    """
    root
    function DBHT(sim::AbstractNonNegativeSimilarityMatrixAlgorithm, root::DBHTRootMethod)
        return new{typeof(sim), typeof(root)}(sim, root)
    end
end
function DBHT(;
              sim::AbstractNonNegativeSimilarityMatrixAlgorithm = MaximumDistanceSimilarity(),
              root::DBHTRootMethod = UniqueRoot())::DBHT
    return DBHT(sim, root)
end

export UniqueRoot, EqualRoot, DBHT, Clusters
