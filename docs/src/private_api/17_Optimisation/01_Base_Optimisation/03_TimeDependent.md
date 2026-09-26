```@meta
Description = "Time-dependent estimators, private API of PortfolioOptimisers.jl: TD_Option, TD, TD_OptE_Opt, TDO_Option, OptE_TD, OptE_Opt_TD, VecOptE_Opt_TD, …"
```

# Time-dependent estimators: private API

```@docs
TD_Option
TD
TD_OptE_Opt
TDO_Option
OptE_TD
OptE_Opt_TD
VecOptE_Opt_TD
TD_VecOptE_Opt
TDO_OptE_Opt
VecOptE_Opt
assert_special_nco_requirements(opt::VecOptE_Opt)
assert_nearest_optimiser_schedule
inner_fold_fields
time_dependent_value
time_dependent_candidate_fields
time_dependent_fields
time_dependent_entries
time_dependent_entry_needs_previous_weights
assert_time_dependent_substitution
substitute_time_dependent_entries
time_dependent_stand_in
time_dependent_reset_value
assert_time_dependent_optimiser
assert_time_dependent_fold_count(::OptE_Opt, ::Integer, ::Bool = true)
assert_time_dependent_fold_count(td::TDO_OptE_Opt, n::Integer, all_binds::Bool = true)
assert_time_dependent_fold_count(opt::VecOptE_Opt_TD, n::Integer, all_binds::Bool = true)
assert_time_dependent_fields_fold_count
rebuild_estimator
is_time_dependent(::OptE_Opt)
is_time_dependent(opt::VecOptE_Opt)
is_time_dependent(opt::BaseOptimisationEstimator)
is_time_dependent(::TimeDependent)
is_time_dependent(opt::VecOptE_Opt_TD)
update_time_dependent_estimator
update_time_dependent_fields
reset_time_dependent_estimator(opt::OptE_Opt)
reset_time_dependent_estimator(opt::BaseOptimisationEstimator)
reset_time_dependent_estimator(td::TD_OptE_Opt)
reset_time_dependent_fields
assert_no_nearest_bind_optimiser_schedule(x, field::Symbol, host::Symbol)
entitled
```
