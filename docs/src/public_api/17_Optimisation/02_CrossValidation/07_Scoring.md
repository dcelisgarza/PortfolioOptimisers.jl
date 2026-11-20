```@meta
Description = "Scoring, public API of PortfolioOptimisers.jl: NearestQuantilePrediction, AbstractSearchReduction, EachValue, PerCandidate."
```

# Scoring

```@docs
NearestQuantilePrediction
```

The slot `r_sca` of a search takes a rule that a scalariser cannot spell. Each rule keeps one score matrix for each risk measure.

```@docs
AbstractSearchReduction
```

A search keeps each value apart under [`EachValue`](@ref), and gives one ranking for each value per candidate.

```@docs
EachValue
```

A search combines its risk measures once per candidate under [`PerCandidate`](@ref), after the search scorer reduces the folds or the paths of each measure.

```@docs
PerCandidate
```
