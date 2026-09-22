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

A [`PopulationPredictionResult`](@ref) numbers its members on construction, so the path a scorer selects names its place in the population.

```@docs
PortfolioOptimisers.successful_members
PortfolioOptimisers.lacks_id
PortfolioOptimisers.with_position_id
PortfolioOptimisers.population_ids
```
