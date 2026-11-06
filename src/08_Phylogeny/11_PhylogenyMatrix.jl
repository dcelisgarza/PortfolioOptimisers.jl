"""
    _phylogeny_matrix(sep::HopCount, nte::AbstractNetworkEstimator,
                      g::Graphs.AbstractGraph)
    _phylogeny_matrix(sep::PathLength, nte::AbstractNetworkEstimator,
                      g::Graphs.AbstractGraph)

Select the related pairs of a structure, through the branch of the separation `sep`.

[`phylogeny_matrix`](@ref) calls this function for its body. The separation decides the neighbourhood that [`phylogeny_matrix`](@ref) selects, and the estimator does not. This function therefore dispatches on the separation, and the public method does not. A dispatch on the estimator ties the choice to `NetworkEstimator`, and puts every other [`AbstractNetworkEstimator`](@ref) on one branch. The other kernels of the family, [`separation_matrix`](@ref) and [`separation_budget`](@ref), also take the separation first, for the same reason.

# The structure is already built

`g` comes from [`separation_graph`](@ref). The public method builds it one time, and also gives it to [`resolve_separation`](@ref). Neither branch computes a distance of its own. The function passes `nte` to [`separation_budget`](@ref), and reads it for no other purpose.

# The two balls

  - [`HopCount`](@ref): the hop ball, `sum(A^i for i in 0:n)` clamped to `0` or `1`, over `Graphs.adjacency_matrix(g)`. The matrix is binary, because [`separation_graph`](@ref) gives a hop count a binary structure, and a power of a weighted matrix sums products of distances. The function reads `sep.n` directly as a count of matrix powers, and not through [`separation_budget`](@ref).
  - [`PathLength`](@ref): the radius ball, the matrix of [`separation_matrix`](@ref) with a threshold at [`separation_budget`](@ref). It needs no second traversal.

# Algorithm

**Under a [`HopCount`](@ref):**

 1. Read the binary matrix `A` from `g` with `Graphs.adjacency_matrix`.
 2. Add up `P` as the sum of `A^i` over `i in 0:sep.n`. Each entry counts the walks of length at most `sep.n` between its pair.
 3. Clamp `P` to `0` or `1`, which changes the count of walks into the selection. Subtract the identity to set the diagonal to zero.

**Under a [`PathLength`](@ref):**

 1. Measure the separations over `g` into `d` with [`separation_matrix`](@ref).
 2. Find the budget `dmax` with [`separation_budget`](@ref).
 3. Select every pair that [`is_related`](@ref) admits at `dmax`. Subtract the identity to set the diagonal to zero. [`is_related`](@ref) tests the unreachable sentinel as well as the budget.

# Arguments

  - `sep`: Resolved separation algorithm. The public method takes it from `nte.sep`.
  - $(arg_dict[:nte])
  - `g`: Structure to read, from [`separation_graph`](@ref).

# Returns

  - `P::Matrix{Int}`: Phylogeny matrix. An entry is `1` for a related pair and `0` otherwise, and the diagonal is `0`.

# Related

  - [`phylogeny_matrix`](@ref)
  - [`HopCount`](@ref)
  - [`PathLength`](@ref)
  - [`separation_graph`](@ref)
  - [`separation_matrix`](@ref)
  - [`separation_budget`](@ref)
  - [`is_related`](@ref)
"""
function _phylogeny_matrix end
function _phylogeny_matrix(sep::HopCount, ::AbstractNetworkEstimator,
                           g::Graphs.AbstractGraph)
    A = Graphs.adjacency_matrix(g)
    P = zeros(Int, size(A))
    # A matrix-power count, hence `sep.n` directly rather than `separation_budget`: this is
    # the hop branch, and a separation measuring anything else needs its own method.
    for i in 0:(sep.n)
        P .+= A^i
    end
    P .= clamp!(P, 0, 1) - LinearAlgebra.I
    return P
end
function _phylogeny_matrix(sep::PathLength, nte::AbstractNetworkEstimator,
                           g::Graphs.AbstractGraph)
    d = separation_matrix(sep, g)
    dmax = separation_budget(sep, nte, d)
    # `is_related` carries both halves of the rule -- the budget and the unreachable sentinel
    # `separation_matrix` passes through unrepaired. The diagonal is zero and therefore always
    # inside the budget, which `- I` then clears, matching the hop branch exactly.
    return Int.(is_related.(Ref(sep), d, dmax)) - LinearAlgebra.I
end
"""
    phylogeny_matrix(nte::AbstractNetworkEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the phylogeny matrix of the network that `nte` builds.

The method builds the network from `X`. It returns the binary matrix of the pairs that `nte.sep` counts as related, with a zero diagonal. The separation selects the neighbourhood, through [`_phylogeny_matrix`](@ref). [`HopCount`](@ref) gives the hop ball, the clamped sum of powers `sum(A^i for i in 0:n)`. [`PathLength`](@ref) gives the radius ball, the matrix of [`separation_matrix`](@ref) with a threshold at [`separation_budget`](@ref).

# The hop ball is the range connection matrix

The hop branch computes the range connection matrix of the walks of length at most `n`, which is Equations 13.1 and 13.2 of [cajas2025](@cite). With the binary adjacency matrix ``\\mathbf{A}`` and the identity ``\\mathbf{I}_n``,

```math
\\begin{align}
    \\mathbf{B}_{k} &= \\mathbf{1}_{x \\geq 1}\\left(\\mathbf{A}^{k} + \\mathbf{I}_n\\right) - \\mathbf{I}_n\\,, \\\\
    \\mathbf{B}_{1,\\,l} &= \\mathbf{1}_{x \\geq 1}\\left(\\sum_{k=1}^{l} \\mathbf{B}_{k}\\right)\\,,
\\end{align}
```

Where:

  - ``\\mathbf{1}_{x \\geq 1}(\\cdot)``: Element-wise indicator of the entries that are at least one.
  - ``\\mathbf{B}_{k}``: Pairs joined by at least one walk of length exactly ``k``.
  - ``\\mathbf{B}_{1,\\,l}``: Pairs joined by at least one walk of length at most ``l``.

The code adds up `sum(A^i for i in 0:n)`, clamps it to `0` or `1`, and subtracts the identity. This is the same selection, written one time for all the shells.

# The result is `Int` under either separation

The two separations select different pairs, and give values of the same kind. The matrix of the [`PhylogenyResult`](@ref) is `Int` here, as it is everywhere else, because no consumer needs a graded value. The values of the matrix do not change [`SemiDefinitePhylogeny`](@ref), because `A ⊙ W == 0` is the same constraint at any magnitude. [`IntegerPhylogeny`](@ref) counts an integer cardinality. [`centrality_vector`](@ref) removes the values before a centrality algorithm runs. [`Proximity`](@ref) gives the graded value of a separation.

# What the radius ball gives, measured

**It changes the ranking little, and on the PMFG not at all.** Compare a hop shell with the first pairs of the order of path lengths, at the same cardinality. On a PMFG of 20 assets, the two sets are the same at every shell. `0` pairs are different out of `54`, `121`, `165` and `186`. On the minimum spanning tree, they are the same at the shells of `19` and `48`. They are different by `1`, `1`, `3` and `2` pairs at the shells of `84`, `115`, `144` and `170`.

The distance selects both structures, so a path length refines a hop count and does not compete with it. The radius ball is not a different kind of neighbourhood.

It gives cardinalities between the shells. Over the same PMFG, the hop count relates `54`, then `121`, then `165` of the `190` pairs. A caller who wants about `100` pairs cannot ask for it. A sweep of `dmax` over the same graph reaches `36`, `55`, `100`, `122`, `151` and `179`. That is the only gain. It is real for [`SemiDefinitePhylogeny`](@ref) and [`IntegerPhylogeny`](@ref), because their constraint strength is that cardinality.

# The default budget of `PathLength` relates every reachable pair

`PathLength()` sets `dmax = nothing`, and [`separation_budget`](@ref) resolves it to the observed diameter. No reachable pair is outside the budget, and every entry off the diagonal is one. The measured count is `190` of `190` pairs on both branches. This is the correct meaning of a budget that the caller does not state, but it is the opposite end of the scale from the default `n = 1` of [`HopCount`](@ref). A caller who replaces one separation by the other, and changes nothing else, gets the largest ball in place of the smallest. State a numeric `dmax` to select fewer pairs.

# Algorithm

 1. Build the structure `g` that `nte.sep` measures over, with [`separation_graph`](@ref). The method builds one structure for each call, and the two steps below share it.
 2. Resolve `nte.sep` against `g` with [`resolve_separation`](@ref). A budget that is a value passes through unchanged and builds nothing.
 3. Select the related pairs with [`_phylogeny_matrix`](@ref), through the branch that the resolved separation names.
 4. Return the selection in a [`PhylogenyResult`](@ref).

# Arguments

  - $(arg_dict[:nte])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `plr::PhylogenyResult{<:Matrix{Int}}`: Phylogeny matrix of the assets. An entry is `1` for a related pair and `0` otherwise, and the diagonal is `0`.

# Related

  - [`NetworkEstimator`](@ref)
  - [`_phylogeny_matrix`](@ref)
  - [`HopCount`](@ref)
  - [`PathLength`](@ref)
  - [`calc_adjacency`](@ref)
  - [`separation_matrix`](@ref)
  - [`separation_budget`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 13.1.2, Equations 13.1-13.2.
"""
function phylogeny_matrix(nte::AbstractNetworkEstimator, X::MatNum; dims::Int = 1,
                          kwargs...)
    # One structure per call, built here and handed to both readers below. A rule in the
    # separation's budget field is answered against that structure rather than deriving a
    # second one; a member whose budget is already a value passes through untouched.
    g = separation_graph(nte.sep, nte, X; dims = dims, kwargs...)
    sep = resolve_separation(nte.sep, nte, X, g; dims = dims, kwargs...)
    return PhylogenyResult(; X = _phylogeny_matrix(sep, nte, g))
end
"""
    phylogeny_matrix(cle::ClE_Cl,
                     X::MatNum; branchorder::Symbol = :optimal, dims::Int = 1,
                     kwargs...)

Compute the phylogeny matrix of a clustering estimator or result.

The method clusters the data into the number of clusters that the estimator selects. It returns the binary phylogeny matrix whose entry is `1` for two assets in the same cluster, with a zero diagonal.

# Algorithm

 1. Partition the assets into the [`Clusters`](@ref) result `res` with [`clusterise`](@ref).
 2. Read the cluster of every asset with [`assignments`](@ref).
 3. Build the `assets × res.k` membership matrix `P`, whose entry is one when the asset belongs to the cluster of its column.
 4. Multiply `P` by its transpose. The product is one for a pair in the same cluster, and zero otherwise. Subtract the identity to set the diagonal to zero.
 5. Return the selection in a [`PhylogenyResult`](@ref).

# Arguments

  - `cle`: Clustering estimator or result.
  - $(arg_dict[:X])
  - `branchorder`: Order method for the dendrogram branches, `:optimal` by default. [`branch_ordered_hclust`](@ref) and [`DBHTs`](@ref) state what each value does.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `plr::PhylogenyResult{<:Matrix{Int}}`: Phylogeny matrix of the clusters. An entry is `1` for two assets in the same cluster and `0` otherwise, and the diagonal is `0`.

# Related

  - [`ClustersEstimator`](@ref)
  - [`AbstractClusteringResult`](@ref)
  - [`clusterise`](@ref)
"""
function phylogeny_matrix(cle::ClE_Cl, X::MatNum; branchorder::Symbol = :optimal,
                          dims::Int = 1, kwargs...)
    res = clusterise(cle, X; branchorder = branchorder, dims = dims, kwargs...)
    clusters = assignments(res)
    # One row per asset. `size(X, 2)` counts the observations when `dims = 2`.
    P = zeros(Int, length(clusters), res.k)
    for i in axes(P, 2)
        idx = clusters .== i
        P[idx, i] .= one(eltype(P))
    end
    return PhylogenyResult(; X = P * transpose(P) - LinearAlgebra.I)
end

export phylogeny_matrix
