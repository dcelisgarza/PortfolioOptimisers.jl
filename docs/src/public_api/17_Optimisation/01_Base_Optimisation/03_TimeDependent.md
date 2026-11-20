```@meta
Description = "Time-dependent, public API of PortfolioOptimisers.jl: TimeDependent, TimeDependentContext, PreviousWeightsFunction, NoDefault, TimeDependentDefaultError, …"
```

# Time-dependent

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
needs_previous_weights(::Any)
time_dependent_field_defaults
```
