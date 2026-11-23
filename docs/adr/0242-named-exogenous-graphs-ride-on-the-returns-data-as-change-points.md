---
status: accepted
---

# Named Exogenous Graphs ride on the returns data as change points

## Context

The oracle of map #2162 reads a graph distance and a dependency exposure from a user graph: a
relation of assets and other entities, such as the suppliers of each asset, that the returns do
not show. The library had no type for it. A tensor Panel Field holds an axis of assets first, so it
cannot hold an edge between two nodes that are not assets, such as a supplier of a supplier. The
oracle `Graph` also refuses a matrix that is not square, and holds a relation of assets and
suppliers as a square graph over the union of the names. A graph changes over time, and a fold of
a cross-validation must not read the graph of its test window.

The grilling of #2165 (2026-11-18) settled fourteen rows, and #2167 added two points. #2171 builds
them. This ADR records the decision and the details that the build settled.

## Decision

**`ExogenousGraph`** holds one directed, weighted graph: `nodes::Vector{String}` and the
adjacency `At`, with the sources in the **columns**. `At[j, i]` is the weight of the edge
`i → j`. Every consumer walks out-edges, so the out-edges of a node are one contiguous CSC column,
the weight share of a node is a column sum, and the exposure of `w` is `At * w`. `adjacency(g)` is
`transpose(g.At)` with no copy. `Graphs.SimpleDiGraph(g)` and
`SimpleWeightedGraphs.SimpleWeightedDiGraph(g)` convert at the point of use, and the weighted
conversion needs no transpose, because that type stores `weights[dst, src]`. The type is not a
wrapper of `SimpleWeightedDiGraph`, which is mutable, has no names and is sparse only, and not a
subtype of `Graphs.AbstractGraph`, which would lose `AbstractResult`.

The storage is as given, dense or sparse. Each constructor takes its input with the **sources in
the rows**, as an edge list and the oracle write it, and transposes once. One keyword constructor
takes three forms: the square form `nodes` and `A`, the block form `sources`, `targets` and `A`,
and the edge list `edges` with optional `nodes`. Julia keeps one keyword method for one positional
signature, so the form is a dispatch of `exogenous_graph` on the keywords that are given. A block
that is not square embeds into a sparse matrix, its nodes are the sources and then each new
target, and `sources == targets` is the square form. A sparse adjacency holds no stored zero. An
integer or `Bool` adjacency floats once at construction (#1972), and the file is in the `ENTRY`
table of `test/test_55_numeric_coercion_census.jl`. `==`, `isequal` and `hash` compare `nodes`
and `At` by value, so a dense and a sparse adjacency with the same entries are equal.

**`ExogenousGraphSeries`** is a list of change points: strictly increasing keys of one type,
1-based row positions or timestamps, and an `ExogenousGraph` or `nothing` at each key. A graph is
in force from its key to the next key, and a `nothing` key ends the graph before it. A static graph
is a plain `ExogenousGraph`. Both subtype `AbstractResult`, and the private alias
`ExG_ExGS = Union{ExogenousGraph, ExogenousGraphSeries}` names the pair. The family is closed: a
graph is in force at each row, or it changes at its keys. An abstract supertype would need an
`# Interfaces` section (test_61, test_66), and an honest one would make eight internal methods
public for a family that no user extends. The field `G` takes any vector, because the literal
`[g, s]` of a static graph and a series has the element type `AbstractResult`, and
`check_exogenous_graphs` refuses an entry outside the alias by dispatch.

**Location.** `ReturnsResult` and `PricesResult` take `ng` (unique names) and `G` (one graph or
series for each name) after `ne` and `E`. `port_opt_view`, `prices_to_returns`,
`vcat_observations` and each rebuild that keeps the rows of its source carry them (ADR 0204). The
asset view passes them through: a consumer finds the assets with `indexin(rd.nx, g.nodes)`. A
`PricesResult` keys a series on the rows of `X`, and `prices_to_returns` views the graphs on the
rows of the price clock that the returns keep, so an integer key moves to its return row.

**As-of rule.** `graph_in_force(rd, name)` reads the graph in force at the last row, by
`searchsortedlast` on the keys. No graph in force refuses, with the graph, the last row and the
first key in the message. `name = nothing` reads the only graph and refuses two or more.

**Observation view.** For sorted parent rows, `graph_view` keeps the anchor (the last key at or
before the first row, which can hold `nothing`) and the keys inside the window, and drops the rest.
An integer key moves to `searchsortedfirst(rows, key)`, the anchor to 1, and of two keys on one row
the last stays. A date key stays. When no key is in force at the first row and none is inside the
window, the view is one `nothing` key at its first row, because a series has one key or more. A
view refuses rows out of clock order. A randomised check of the test file reads the graph in force
at each row of 500 views against its parent row.

**`vcat`.** `vcat_observations` refuses another `ng`. Two static graphs must be equal. A static
graph and a series, or a series of positions and a series of dates, are refused. Each series is
first viewed on its own rows. The integer keys of `b` move by the rows of `a`. The anchor of `b` is
dropped when it holds the graph in force at the last row of `a`. A date anchor that holds another
graph at or before the last row of `a` is refused. When no key of `b` is in force at its first row
and `a` ends with a graph, a `nothing` key at the first row of `b` keeps its rows without one. So
the join of the views of every split is the series of the whole data.

**Round trip.** `graph_dataframe` writes one row for each edge (`graph`, `key`, `source`,
`target`, `weight`), and `graph_manifest` writes what the edges cannot show, as `panel_manifest`
does: a `graph` row (`static` or `series`), a `snapshot` row for each key (`dense`, `sparse` or
`none`) and a `node` row for each node in order. `exogenous_graphs(df, mf)` gives back the same
`ng` and `G`. The edge table alone gives nodes in the order of first use, sparse storage, a series
for each graph with keys, and a static graph for a graph whose keys are all `missing`. The reader
reads each cell as text when a table reader typed it so.

**Validation of the data.** The data types refuse what row 12 of #2165 lists: empty names, an
empty or repeated name, a wrong shape, a weight that is not finite, a repeated edge of an edge
list, a series with no key, keys that are not strictly increasing or of two kinds, an integer key
below 1 or after the last row, a date key with no timestamps, a date type that does not compare
with them, a repeated name in `ng`, and `length(ng) != length(G)`. They accept negative weights,
self-loops, cycles, nodes that are not assets and assets that are not nodes.

**Validation of a consumer.** A consumer states what it cannot read with `graph_requirements`,
a tuple of `AbstractGraphRequirement`s, and names its graphs with `graph_names`. Its default
`nothing` reads the only graph. `NonNegativeWeights` is the first requirement. A requirement
implements `graph_requirement_breach` and `graph_requirement_rule`. `optimise` and the fold loop
of every cross-validation call `assert_graph_entry` one time, before the first view or fold. It
walks every field of the estimator tree, vectors, tuples and Pairs included, as
`online_entry_state` does, and checks every snapshot of each named graph. So a bad snapshot that
fold 80 reads fails before fold 1, and the error names the consumer, its path in the tree, the
graph, the key and the edge. A direct call of a consumer checks the one snapshot that it reads with
`consumer_graph`. The walk is a plain recursion and not a new channel of `PROP_TAG_CHANNELS`,
because every channel rebuilds the struct, and this check only reads it. By #2167, the consumer of
the graph distance is the producer `WalkProfilePanel` (#2172), and `graph_requirements` of that
producer is `(NonNegativeWeights(),)`. The exposure of #2173 declares its own.

**Load order.** The types load before `PricesResult`, so the family is the directory
`src/03_InputData/05_ExogenousGraph/` (ADR 0150). `05_PanelViews.jl` held only methods and moved
to `23_PanelViews.jl`, beside the other panel files, so no other file was renumbered. The methods
that take a `ReturnsResult` or a `PricesResult` sit at the end of the files of those types.

## Consequences

- The positional constructors of `ReturnsResult` and `PricesResult` take two more arguments, and
  their show prints two more fields. `docs/src/migration.md` names the break.
- The entry check walks the estimator tree on each call of `optimise`, also in a nested optimiser
  of a meta-optimiser, where the view of the fold holds fewer snapshots. The walk reads fields and
  allocates a path string only for a consumer. No flag marks a check that already ran.
- `prior(pe, rd)` does not check, because the inner method of a prior reads `X`, `F` and the panel
  and no consumer of a prior tree can read a graph.
- The online buffer of `ReturnsBufferState` keeps no graph. A consumer that reads the data that
  the buffer gives back refuses, because that data holds no graph. The online route of a series,
  the Fold Context and a series whose node set changes stay in *Not yet specified* of map #2162.
- The rule that joins two named graphs in one consumer belongs to #2203 and #2208.
- The `PredictionReturnsResult` of a fold holds the graphs of its test rows, the view of
  `graph_view`, so `dependency_exposure` (#2173) reads the graph in force at each row of a fold.
  The series of a path holds none, because the folds of a combinatorial path are not in time
  order, and each fold holds its own.
- The outer problem of a meta-optimiser that a cross-validation fits (`rebuild_returns_result`)
  and the factor returns of a fold (`fold_factor_returns`) build their `ReturnsResult` from a
  `PredictionReturnsResult`, and they do not read its graphs, so they carry none. The outer problem of a
  meta-optimiser with no cross-validation keeps the graphs of its rows. A consumer at the outer
  level of a cross-validated NCO is a question of #2172.
