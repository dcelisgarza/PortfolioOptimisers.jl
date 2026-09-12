"""
    _phylogeny_matrix(sep::HopCount, nte::AbstractNetworkEstimator,
                      g::Graphs.AbstractGraph)
    _phylogeny_matrix(sep::PathLength, nte::AbstractNetworkEstimator,
                      g::Graphs.AbstractGraph)

Internal dispatch helper carrying [`phylogeny_matrix`](@ref)'s per-separation body.

The neighbourhood [`phylogeny_matrix`](@ref) selects is a question about the separation, not about the estimator, so the split lives here rather than on the public method's argument. Dispatching on the estimator instead would pin the choice to `NetworkEstimator` and leave every other [`AbstractNetworkEstimator`](@ref) on one branch — and this family's other kernels, [`separation_matrix`](@ref) and [`separation_budget`](@ref), already take the separation first for the same reason.

# The structure arrives built

`g` is [`separation_graph`](@ref)'s, built once by the public method and shared with [`resolve_separation`](@ref), so neither branch derives a distance of its own. `nte` stays for [`separation_budget`](@ref)'s estimator channel and is otherwise inert here.

# The two balls

  - [`HopCount`](@ref): the **hop ball**, `sum(A^i for i in 0:n)` clamped to `0` or `1`, over `Graphs.adjacency_matrix(g)` — binary, because [`separation_graph`](@ref) hands a hop count a binarised structure and a power of a weighted matrix would sum products of distances. `sep.n` is read directly as a **matrix-power count** rather than through [`separation_budget`](@ref), which is what makes it a power count and not a budget.
  - [`PathLength`](@ref): the **radius ball**, [`separation_matrix`](@ref) thresholded at [`separation_budget`](@ref). No second traversal.

# Algorithm

**Under a [`HopCount`](@ref):**

 1. Read `Graphs.adjacency_matrix` off `g`, giving the binary matrix `A`.
 2. Accumulate `P` as the sum of `A^i` over `i in 0:sep.n`. Each entry counts the walks of length at most `sep.n` between its pair.
 3. Clamp `P` to `0` or `1`, which turns the walk count into the selection, and subtract the identity to clear the diagonal.

**Under a [`PathLength`](@ref):**

 1. Measure the separations over `g` with [`separation_matrix`](@ref), giving `d`.
 2. Resolve the budget with [`separation_budget`](@ref), giving `dmax`.
 3. Select every pair [`is_related`](@ref) admits at `dmax`, and subtract the identity to clear the diagonal. [`is_related`](@ref) carries the unreachable sentinel as well as the budget.

# Arguments

  - `sep`: Separation algorithm, taken from `nte.sep` by the public method and resolved.
  - $(arg_dict[:nte])
  - `g`: Structure to read, from [`separation_graph`](@ref).

# Returns

  - `P::Matrix{Int}`: Phylogeny matrix. `1` for a related pair, `0` otherwise, `0` on the diagonal.

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

Compute the phylogeny matrix for a network estimator.

Builds the network from `X` and returns the binary matrix of the pairs `nte.sep` counts as related, with self-loops removed. Which neighbourhood that is comes from the separation, through [`_phylogeny_matrix`](@ref): [`HopCount`](@ref) gives the **hop ball**, the clamped power sum `sum(A^i for i in 0:n)` the network family has always used; [`PathLength`](@ref) gives the **radius ball**, [`separation_matrix`](@ref) thresholded at [`separation_budget`](@ref).

# The hop ball is the range connection matrix

The hop branch computes the range connection matrix of walks of length at most `n`, which is [`NetworkEstimator`](@ref)'s Equations 13.1 and 13.2. Writing ``\\mathbf{A}`` for the binary adjacency matrix and ``\\mathbf{I}_n`` for the identity,

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

The code accumulates `sum(A^i for i in 0:n)`, clamps to `0` or `1`, and subtracts the identity, which is the same selection written once rather than shell by shell. Measured over the minimum spanning tree of the last 253 observations of the 20-asset sample in `test/assets/SP500.csv.gz`, the two agree entry for entry at `n = 1, 2, 3, 4` — a maximum absolute difference of `0`, over `19`, `48`, `84` and `115` related pairs.

# The result is `Int` under either separation

Selection changes; the values do not. [`PhylogenyResult`](@ref)'s matrix is `Int` here as everywhere else, because no consumer of one wants a number: [`SemiDefinitePhylogeny`](@ref) is weight-inert (`A ⊙ W == 0` is the same constraint at any magnitude), [`IntegerPhylogeny`](@ref) counts an integer cardinality, and [`centrality_vector`](@ref) binarises before any centrality algorithm runs. The graded reading of a separation lives on [`Proximity`](@ref) instead.

# What the radius ball buys, measured

**It barely re-ranks, and on the PMFG not at all.** Compare a hop shell against the equal-cardinality prefix of the path-length ordering: on a 20-asset PMFG the two sets are **identical** at every shell — `0` pairs differ out of `54`, `121`, `165` and `186`. On the minimum spanning tree they are identical at the shells of `19` and `48`, and differ by `1`, `1`, `3` and `2` pairs at the shells of `84`, `115`, `144` and `170`. Both structures are selected by distance in the first place, so a path length **refines** a hop count rather than rivalling it. A reader who takes the radius ball for a conceptually different neighbourhood will be wrong.

What it buys is **intermediate cardinalities between the shells**. Over the same PMFG the hop knob relates `54`, then `121`, then `165` of the `190` pairs; a caller wanting about `100` cannot ask for it. Sweeping `dmax` across the same graph reaches `36`, `55`, `100`, `122`, `151` and `179`. That is the whole gain, and it is real for [`SemiDefinitePhylogeny`](@ref) and [`IntegerPhylogeny`](@ref), whose constraint strength is that cardinality.

# `PathLength`'s default budget relates everything reachable

`PathLength()` leaves `dmax = nothing`, which [`separation_budget`](@ref) resolves to the **observed diameter** — so no reachable pair sits outside it and the matrix is all ones off the diagonal. Measured: `190` of `190` pairs on both branches. This is the honest reading of an unstated budget rather than a fall-back, but it is the *opposite* end of the dial from [`HopCount`](@ref)'s default `n = 1`: a caller who swaps one separation for the other and changes nothing else gets the maximal ball where they had the minimal one. State a numeric `dmax` to select anything narrower.

# Algorithm

 1. Build the structure `nte.sep` measures over with [`separation_graph`](@ref), giving `g`. One structure is built per call and both readers below share it.
 2. Resolve `nte.sep` against `g` with [`resolve_separation`](@ref). A budget that is already a value passes through and builds nothing.
 3. Select the related pairs with [`_phylogeny_matrix`](@ref), through the branch the resolved separation names.
 4. Wrap the selection in a [`PhylogenyResult`](@ref).

# Arguments

  - `nte`: NetworkEstimator estimator.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `plr::PhylogenyResult{<:Matrix{Int}}`: Phylogeny matrix representing asset relationships. `1` for a related pair, `0` otherwise, `0` on the diagonal.

# Related

  - [`NetworkEstimator`](@ref)
  - [`_phylogeny_matrix`](@ref)
  - [`HopCount`](@ref)
  - [`PathLength`](@ref)
  - [`calc_adjacency`](@ref)
  - [`separation_matrix`](@ref)
  - [`separation_budget`](@ref)
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

Compute the phylogeny matrix for a clustering estimator or result.

This function clusterises the data, cuts the tree into the optimal number of clusters, and constructs a binary phylogeny matrix indicating shared cluster membership, with self-loops removed.

# Algorithm

 1. Partition the assets with [`clusterise`](@ref), giving the [`Clusters`](@ref) result `res`.
 2. Read the cluster of every asset with [`assignments`](@ref).
 3. Build the `assets × res.k` membership matrix `P`, whose entry is one when the asset belongs to the cluster of its column.
 4. Multiply `P` by its own transpose, which is one exactly for a pair that shares a cluster, and subtract the identity to clear the diagonal.
 5. Wrap the selection in a [`PhylogenyResult`](@ref).

# Arguments

  - `cle`: Clustering estimator or result.
  - `X`: Data matrix (observations × assets).
  - `branchorder`: Branch ordering strategy for hierarchical clustering.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `P::Matrix{Int}`: Phylogeny matrix representing cluster relationships.

# Related

  - [`ClustersEstimator`](@ref)
  - [`AbstractClusteringResult`](@ref)
  - [`clusterise`](@ref)
"""
function phylogeny_matrix(cle::ClE_Cl, X::MatNum; branchorder::Symbol = :optimal,
                          dims::Int = 1, kwargs...)
    res = clusterise(cle, X; branchorder = branchorder, dims = dims, kwargs...)
    clusters = assignments(res)
    P = zeros(Int, size(X, 2), res.k)
    for i in axes(P, 2)
        idx = clusters .== i
        P[idx, i] .= one(eltype(P))
    end
    return PhylogenyResult(; X = P * transpose(P) - LinearAlgebra.I)
end

export phylogeny_matrix
