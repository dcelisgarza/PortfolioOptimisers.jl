```@meta
Description = "Graph requirements, public API of PortfolioOptimisers.jl: AbstractGraphRequirement, NonNegativeWeights, graph_requirements, graph_names, …"
```

# Graph requirements

## Types

A graph of the returns data accepts a negative weight, a self-loop and a cycle. A consumer that cannot read
one of them states a requirement, and the entry of a user verb checks every snapshot of the graph
that the consumer names, before the first fold.

```@docs
PortfolioOptimisers.AbstractGraphRequirement
PortfolioOptimisers.NonNegativeWeights
```

## Functions

```@docs
PortfolioOptimisers.graph_requirements
PortfolioOptimisers.graph_names
PortfolioOptimisers.graph_requirement_breach
PortfolioOptimisers.graph_requirement_rule
PortfolioOptimisers.consumer_graph
PortfolioOptimisers.assert_graph_entry
```
