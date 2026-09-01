"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Direct Bubble Hierarchy Tree (DBHT) root selection methods.

The root is chosen inside [`CliqHierarchyTree2s`](@ref), which builds the clique hierarchy of the planar graph. A hierarchy needs one node with no parent, and the planar clique tree can present several candidates, so the choice of which of them becomes the root is a member of this family.

# Related

  - [`UniqueRoot`](@ref)
  - [`EqualRoot`](@ref)
  - [`DBHT`](@ref)
  - [`CliqHierarchyTree2s`](@ref)

# References

  - $(ref_dict[:NHPG])
  - $(ref_dict[:DBHTs])
"""
abstract type DBHTRootMethod <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Takes one clique of the planar hierarchy as its single root.

# Related

  - [`DBHTRootMethod`](@ref)
  - [`EqualRoot`](@ref)
  - [`DBHT`](@ref)
  - [`CliqueRoot`](@ref)

# References

  - $(ref_dict[:NHPG])
"""
struct UniqueRoot <: DBHTRootMethod end
"""
$(DocStringExtensions.TYPEDEF)

Builds one root from the adjacency tree of every root candidate.

This keeps several equally plausible roots of the DBHT hierarchy rather than choosing between them.

# Related

  - [`DBHTRootMethod`](@ref)
  - [`UniqueRoot`](@ref)
  - [`DBHT`](@ref)
  - [`CliqueRoot`](@ref)

# References

  - $(ref_dict[:NHPG])
"""
struct EqualRoot <: DBHTRootMethod end
"""
$(DocStringExtensions.TYPEDEF)

Clusters assets by the bubble hierarchy of a triangulated maximally filtered graph.

`DBHT` is a composable clustering algorithm type for constructing hierarchical clusterings using the Direct Bubble Hierarchical Tree (DBHT) method, as described in [DBHTs](@cite).

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

  - $(ref_dict[:DBHTs])
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
