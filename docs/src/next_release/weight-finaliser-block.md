```@meta
Description = "The block of a weight finaliser: JuMPWeightFinaliser holds a ConstraintBlock, and a weight finaliser of your own takes the context of the head."
```

# [The block of a weight finaliser](@id migration-next-weight-finaliser-block)

`JuMPWeightFinaliser` now holds a `ConstraintBlock` in the new field `cbe`, so a clustering or naive optimiser takes each slot of a block, such as `card`, `conc`, `lcse` or `tn`. A finaliser that states no slot gives the weights of before. Code that builds it by keyword does not change.

- **`JuMPWeightFinaliser`** now takes five positional arguments, `slv`, `cbe`, `sc`, `so` and `alg`, where it took four before. Write `JuMPWeightFinaliser(slv, ConstraintBlock(; wb = nothing), sc, so, alg)`, or use the keyword constructor. The type parameters gain one entry after the one of `slv`, and the show of the finaliser prints the rows of `cbe`.
- **A `ConstraintBlock` with a slot keyed by name and no `sets`** raised an `IsNothingError` at its construction before. Now the constructor of its owner raises it: `JuMPOptimiser`, `ProgrammeAllocationSet`, or the optimiser that holds the finaliser.
- **A weight finaliser of your own** now takes the context of the head. Write `PortfolioOptimisers.opt_weight_bounds(wf::MyFinaliser, wb, w, ctx...)` where you wrote `opt_weight_bounds(wf::MyFinaliser, wb, w)` before. The context is `pr`, `rd`, `sets`, `fees` and `strict`.
- **A deviation of your own for `JuMPWeightFinaliser`**, a subtype of `JuMPWeightFinaliserFormulation`, now gives its deviation to `PortfolioOptimisers.set_divergence_objective!`, so the penalties of the block join the objective. A method that sets the objective itself still runs, and it ignores a penalty.
