```@meta
Description = "Scoring, private API of PortfolioOptimisers.jl: AbstractCrossValidationScorer, PredictionScorer, PredictionCrossValScorer, oriented_score, oriented_value, …"
```

# Scoring: private API

```@docs
AbstractCrossValidationScorer
PredictionScorer
PredictionCrossValScorer
```

A scorer reads the oriented score of a path. The score orients each risk measure before it reduces a vector, so a higher score is always better.

```@docs
PortfolioOptimisers.oriented_score
PortfolioOptimisers.oriented_value
PortfolioOptimisers.reduce_member_scores
PortfolioOptimisers.oriented_member_score
PortfolioOptimisers.prediction_returns
```

A reader of a population takes its reduction as an argument. So it refuses a `sca` in the keywords that it forwards to the score.

```@docs
PortfolioOptimisers.assert_path_reduction
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
