```@meta
Description = "Scoring, private API of PortfolioOptimisers.jl: AbstractCrossValidationScorer, PredictionScorer, PopulationScorer, PredictionCrossValScorer, …"
```

# Scoring: private API

```@docs
AbstractCrossValidationScorer
PredictionScorer
PopulationScorer
PredictionCrossValScorer
PopulationCrossValScorer
```

A [`PopulationPredictionResult`](@ref) gives each member without an `id` its position in the population as its `id`. So the `id` of the path that a scorer selects is its position in the population.

```@docs
PortfolioOptimisers.successful_members
PortfolioOptimisers.lacks_id
PortfolioOptimisers.with_position_id
PortfolioOptimisers.population_ids
```
