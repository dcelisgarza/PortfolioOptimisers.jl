```@meta
Description = "A NearOptimalCentering sweep gives one return code for each point, and a frontier sweep with a fallback gives the sweep when one point or more solved."
```

# [A frontier sweep and its fallback](@id migration-next-frontier-fallback)

## [Return codes that change shape](@id migration-next-frontier-fallback-retcodes)

- **`NearOptimalCentering` gives one return code for each point of a frontier sweep**, as every other optimiser that sweeps does. `res.retcode` of a sweep was one code, and it failed when one point failed. Write `all(rc -> rc isa OptimisationSuccess, res.retcode)` in place of `res.retcode isa OptimisationSuccess`. A solve of one portfolio keeps its one code. A point whose anchor is infeasible now fails alone. It made every point of the sweep fail in the previous release.

## [Calls that now raise](@id migration-next-frontier-fallback-raises)

An optimiser with a frontier sweep and a fallback `fb` gave the result of the fallback, also when every point of the sweep solved. It now gives the sweep when one point or more solved, and the fallback runs only when no point solved. These calls get the sweep where they got the one portfolio of the fallback.

| Call | The previous release | This release |
| --- | --- | --- |
| `NestedClustered` or `Stacking` with an inner optimiser that holds a frontier sweep and a fallback | Ran on the portfolio of the fallback | `ArgumentError`, as with no fallback. An inner optimiser gives one portfolio. |
| A walk-forward of an optimiser that holds a frontier sweep and the fallback `PreviousWeights()` | Ran the fallback on every fold | `ArgumentError` at the fold after a solved sweep, because a sweep gives no one portfolio as the previous weights. Optimise one portfolio per fold, or use a fallback that reads no previous weights. |
| `sort_by_measure` of a population whose paths hold a frontier sweep | An empty vector | `ArgumentError`. A path that holds a sweep has one score for each point and no rank. |
| `quantile_by_measure` and `NearestQuantilePrediction` of such a population | `ArgumentError` that named an empty data vector | `ArgumentError` that names the sweep |
