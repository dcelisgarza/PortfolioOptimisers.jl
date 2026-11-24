---
status: accepted
---

# The network readers read an Exogenous Graph at the assets

## Context

ADR 0242 put named Exogenous Graphs on the returns data. The network readers, which are the
separations, the phylogeny matrix and its constraints, the centralities and `PhylogenyPanel`, took
a network that the library fits to the returns. An Exogenous Graph is directed, it can have nodes
that are not assets, and its weights state a strength and not a length. Issue #2289 decided this
record in a grilling session, and issue #2295 built it.

## Decision

1. **`ExogenousNetwork(; ng, dir, scope, nrm, len, sep, strict) <: AbstractNetworkEstimator`** is a
   member of the network family. It reads `rd.G` at the point of use, and a call with no returns
   data refuses. `NetworkEstimator` does not change, because its `ce`, `de`, `alg` and `x_src` have
   no effect on an Exogenous Graph. `ng` takes `nothing`, one name or a `GraphUnion`.
2. **`dir`** takes the four members of `AbstractEdgeDirection`, with `AnyEdges()` as the default,
   the one that every reader takes. `AnyEdges` is the path that can turn at each node. `BothEdges`
   is the smaller of the two one-way separations. `OutEdges` and `InEdges` are the one-way
   separations, and row `i` holds the separations from asset `i`.
3. **A reader refuses a direction that has no meaning for it**, in its constructor, and a direct
   call refuses at the call. A phylogeny matrix is symmetric, so it refuses `OutEdges` and
   `InEdges`. A centrality is found in one graph, so it refuses `BothEdges`. `Proximity` takes all.
4. **`scope = AllNodes()`** lets a path go through any node, and **`AssetNodes()`** keeps the
   subgraph of the assets of the fit. The separations are measured from the assets, with one
   traversal for each, so a `FractionOf` budget and the observed diameter read the pairs of assets.
   A budget rule receives the whole graph.
5. **`len = ReciprocalLength(; power = 1)`**, `1 / w^power`, **`NegativeLogLength()`**, `-log w`,
   which refuses `w > 1`, or **`nothing`** for weights that are lengths. A zero length takes the
   repair of `graph_weight_matrix`. Two opposite edges give the shorter of their two lengths under
   `AnyEdges`, and the sum of their two weights as a strength.
6. **`nrm = nothing`** or **`WeightShare()`**, before the length, along the single direction of a
   part, and along the out-edges under `AnyEdges`. A `GraphUnion` with a normalisation of its own
   acts along the same direction, so a union of one graph under `WeightShare` equals the graph with
   `nrm = WeightShare()`. A self-loop is removed before the normalisation.
7. **A centrality** is found in the graph of `scope`, and the asset entries are the result. With no
   polarity it reads the closure of `sep` over that graph. Under `len = nothing` an eigenvector
   centrality reads the plain graph, as it does on a tree.
8. **An asset that is not a node** is an isolated vertex that stays investable, with one warning.
   Under `strict` it is an error.
9. **It reads no returns.** A gap in the returns does not refuse the fit, the Investable Mask does
   not change, it has no `x_src`, and a reader that holds it adds no frame to its head.
10. **Refused**: `NetworkClustersEstimator`, whose kernel reads the similarities and the distances
    of the returns, the outer problem of `NestedClustered` and `Stacking`, and `frc_ple` of
    `FactorRiskContribution`, whose factors are not nodes. `holds_node_reader` finds a reader of the
    nodes of a graph at any depth, so the last two also refuse a `WalkProfilePanel`, and they no
    longer refuse it at the solve alone.
11. **`PhylogenyPanel` passes `pr` and `rd`** to the phylogeny routines of its `pl`.

## Consequences

- The kernels read an `AssetVertexGraph`, which holds the graph of each part of the direction and
  the vertex of each asset. A `NetworkEstimator` keeps its plain graph and its old methods.
- Under `BothEdges` with a normalisation, the two parts have shares of their own, so the two one-way
  separations of a pair differ between the parts. The separation is the smallest over both parts and
  both orders of the pair, so it stays symmetric. With no normalisation it is the smaller of the
  `OutEdges` and the `InEdges` separations.
- The sum of lengths along a path depends on the end where it starts. A symmetric separation takes
  the smaller of the two sums, so a budget at a separation cannot relate a pair in one order alone.
- The closure over the whole graph is one traversal for each vertex. At one hop the closure is the
  graph, and no traversal runs.
- `plot_network` and `plot_centrality` pass the returns data to the network.
