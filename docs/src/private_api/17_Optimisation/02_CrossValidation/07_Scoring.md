```@meta
Description = "Scoring, private API of PortfolioOptimisers.jl: AbstractCrossValidationScorer, PredictionScorer, PredictionCrossValScorer, member_solved, …"
```

# Scoring: private API

```@docs
AbstractCrossValidationScorer
PredictionScorer
PredictionCrossValScorer
```

A scorer ranks only the members of a population whose every fold solved.

```@docs
PortfolioOptimisers.member_solved
PortfolioOptimisers.successful_members
```

A [`PopulationPredictionResult`](@ref) gives each member without an `id` its position in the population as its `id`. So the `id` of the path that a scorer selects is its position in the population.

```@docs
PortfolioOptimisers.lacks_id
PortfolioOptimisers.with_position_id
PortfolioOptimisers.population_ids
```
