```@meta
Description = "Time-dependent estimators, public API of PortfolioOptimisers.jl: TimeDependent, TimeDependentContext, PreviousWeightsFunction, NoDefault, …"
```

# Time-dependent estimators

```@docs
TimeDependent
TimeDependentContext
PreviousWeightsFunction
NoDefault
TimeDependentDefaultError
factory(td::TimeDependent, args...)
TimeDependentCallable
TimeDependentConstraintCallable
TimeDependentOptimiserCallable
needs_previous_weights(opt::VecOptE_Opt)
needs_previous_weights(td::TimeDependent)
needs_previous_weights(opt::VecOptE_Opt_TD)
time_dependent_field_defaults
```
