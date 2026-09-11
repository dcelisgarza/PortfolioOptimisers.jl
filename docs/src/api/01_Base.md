# Base

[`src/01_Base/`](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/src/01_Base) implements the most basal symbols used in `PortfolioOptimisers.jl`. One file per concept: the docstring dictionaries, the type roots, the pretty-show macro, the `ScopedConfig` holders, the load-time preferences, the message builders, the error hierarchy, the type aliases, the observation weights, the `assert_*` family, `VecScalar`, the `NormError` family, the Kaniadakis logarithm, the partial-fit state seam and the sample buffer the online step folds into.

```@docs
PortfolioOptimisers
```

## Base abstract types

`PortfolioOptimisers.jl` is designed in a deliberately structured and hierarchical way. Enabling us to create self-contained, independent, composable processes. These abstract types form the basis of this hierarchy.

```@docs
AbstractEstimator
AbstractAlgorithm
AbstractResult
CrossValidationEstimator
```

## Configuration

Package-level configuration values (pretty-printing collapse, fuzzy-suggestion distance, equation-parser resource caps, the scenario-fill share) are held in thread-safe [`ScopedConfig`](@ref) holders: a `set_*!` setter swaps the global default atomically, a `with_*` helper overrides it for the dynamic extent of a call (task-scoped, automatically restored), and per-project defaults can be seeded at load time via Preferences.jl.

```@docs
ScopedConfig
Base.getindex(cfg::ScopedConfig)
set_default!
with_config
apply_preferences!
apply_show_preferences!
PortfolioOptimisers.__init__
PREFERENCE_KEYS
PREFERENCE_DISTANCES
RESOURCE_LIMITS
ResourceLimits
assert_resource_cap
assert_ep_grid_size
set_resource_limits!
with_resource_limits
```

## Pretty printing

`PortfolioOptimisers.jl`'s types tend to contain quite a lot of information, these functions enable pretty printing so they are easier to interpret. A field that holds `nothing` is hidden by default and shown in this documentation; [`set_show_nothing_fields!`](@ref) is the switch, and [`show_fields`](@ref) is the hook a type overloads to hide a field of its own choice.

```@docs
@define_pretty_show
show_fields
pretty_show_fields
has_pretty_show_method
set_compact_show!
with_compact_show
COMPACT_SHOW
compact_show_budget
ShowNothingFields
SHOW_NOTHING_FIELDS
set_show_nothing_fields!
with_show_nothing_fields
pretty_show_vector_summary
pretty_show_vector_element
pretty_show_vector_body
```

## Utilities

Custom types are the bread and butter of `PorfolioOptimisers.jl`, the following types and utilities are non-specific and used throughout the library.

```@docs
DynamicAbstractWeights
AbstractCustomValue
VecScalar
AbstractEstimatorValueAlgorithm
VectorAbstractEstimatorValueAlgorithm
get_observation_weights
NormError
L2Norm
SquaredL2Norm
L1Norm
LpNorm
LInfNorm
norm_error
norm_factor
kappa_log
resolve_rng
```

## Logging

Functionality for logging messages.

```@docs
StringDistanceConfig
STRING_DISTANCE
set_string_distance!
with_string_distance
did_you_mean
suggest_declared_key
unknown_variable_msg
misaligned_axis_msg
strict_diagnostic
missing_group_assets_msg
empty_row_msg
empty_projected_row_msg
zero_centrality_msg
gross_budget_bounds_msg
failed_solve_msg
relaxed_preferences_msg
first_error_line
EquationLimits
EQUATION_LIMITS
set_equation_limits!
with_equation_limits
```

## Error types

Many of the types defined in `PortfolioOptimisers.jl` make use of extensive data validation to ensure values meet various criteria. This simplifies the implementation of methods, and improves performance and by delegating as many checks as possible to variable instantiation. In cases where validation cannot be performed at variable instantiation, they are performed as soon as possible within functions.

`PortfolioOptimisers.jl` aims to catch potential data validation issues as soon as possible and in an informative manner, in order to do so it makes use of a few custom error types.

```@docs
PortfolioOptimisersError
Base.showerror(io::IO, err::PortfolioOptimisersError)
IsNothingError
IsEmptyError
IsNonFiniteError
PropertyPathError
ConflictingArgumentError
ObservationWeightsError
NonPositiveWealthError
```

## Assertions

In order to increase correctness, robustness, and safety, we make extensive use of [defensive programming](https://en.wikipedia.org/wiki/Defensive_programming). The following functions perform some of these validations and are usually called at variable instantiation.

```@docs
assert_nonempty
assert_finite
assert_nonneg
assert_gt0
assert_nonempty_nonneg_finite_val
assert_nonempty_gt0_finite_val
assert_nonempty_finite_val
assert_matrix_issquare
assert_unit_interval
assert_closed_unit_interval
assert_all_finite
assert_source_selector
assert_returns_result_dims
```

## Base type aliases

`PortfolioOptimisers.jl` heavily relies on `Julia`'s dispatch and type system to ensure data validity. Many custom types and functions/methods can accept different data types. These can be represented as type unions, many of which are used throughout the library. The following type aliases centralise these union definitions, as well as improving correctness and maintainability.

```@docs
Option{T}
VecNum
VecInt
MatNum
ArrNum
Arr3Num
VecNum_MatNum
MatNum_Arr3Num
Num_VecNum
Func_Num_VecNum
CVal_Func_Num_VecNum
Num_ArrNum
PairStrNum
DictStrNum
MultiEstValType
EstValType
PairGSCV
DictGSCV
GSCVKey
RSCVVal
MultiGSCVValType
VecMultiGSCVValType
MultiGSCVValType_VecMultiGSCVValType
Str_Expr
VecStr_Expr
EqnType
VecVecNum
VecVecInt
VecInt_VecVecInt
VecVecVecInt
VecMatNum
VecStr
VecPair
VecJuMPScalar
MatNum_VecMatNum
Int_VecInt
VecNum_VecVecNum
VecDate
Dict_Vec
Sym_Str
Str_Vec
ObsWeights
Num_VecNum_VecScalar
Num_ArrNum_VecScalar_DynWeights
Func_VecNum
```

## Glossaries

In order to standardise the documentation we use a arg_dict of terms.

```@docs
unique_key_dict
arg_dict
val_dict
ret_dict
field_dict
math_dict
err_name_dict
ref_dict
```

## Partial fit

An incremental fit folds one observation into an estimate without reading the sample again. [`partial_fit!`](@ref) is the verb each family writes, [`partial_fit`](@ref) is the value form that folds a copy of the state, its running quantities live in a [`AbstractPartialFitState`](@ref), and [`merge_states`](@ref) combines the states of two disjoint blocks of observations into the state of the concatenated block.

```@docs
partial_fit!
partial_fit
partial_fit(est::Union{<:PortfolioOptimisers.AbstractEstimator, <:StatsBase.CovarianceEstimator}, args...; kwargs...)
PortfolioOptimisers.AbstractPartialFitState
PortfolioOptimisers.merge_states
PortfolioOptimisers.assert_mergeable_states
PortfolioOptimisers.chan_merge
PortfolioOptimisers.assert_partial_fit_state
PortfolioOptimisers.partial_fit_cache
PortfolioOptimisers.observation_count
PortfolioOptimisers.obs_weights_view(::PortfolioOptimisers.AbstractPartialFitState, ::Any)
```

## The online step

An estimator with no exact incremental fold keeps the observations it has seen in a [`PortfolioOptimisers.SampleBufferState`](@ref), and [`Online`](@ref) is the configuration that seeds one. The wrapper is transient: [`PortfolioOptimisers.update_online_estimator`](@ref) resolves it at warm-up, so no wrapper survives into the run.

```@docs
PortfolioOptimisers.SampleBufferState
PortfolioOptimisers.assert_sample_buffer_state
PortfolioOptimisers.sample_buffer
PortfolioOptimisers.assert_sample_buffer(est::Union{<:PortfolioOptimisers.AbstractEstimator, <:StatsBase.CovarianceEstimator})
PortfolioOptimisers.assert_sample_buffer(::PortfolioOptimisers.Online)
PortfolioOptimisers.sample_buffer_seed
PortfolioOptimisers.fold_buffer
PortfolioOptimisers.partial_fit!(state::PortfolioOptimisers.SampleBufferState, X::PortfolioOptimisers.MatNum; dims::Int = 1)
PortfolioOptimisers.partial_fit!(state::PortfolioOptimisers.SampleBufferState, x::PortfolioOptimisers.VecNum)
PortfolioOptimisers.reserve_sample_buffer
PortfolioOptimisers.partial_fit!(est::Union{<:PortfolioOptimisers.AbstractEstimator, <:StatsBase.CovarianceEstimator}, X::PortfolioOptimisers.VecNum_MatNum; dims::Int = 1)
PortfolioOptimisers.merge_states(a::PortfolioOptimisers.SampleBufferState, b::PortfolioOptimisers.SampleBufferState)
Base.copy(x::PortfolioOptimisers.SampleBufferState)
PortfolioOptimisers.port_opt_view(x::PortfolioOptimisers.SampleBufferState, i, args...)
PortfolioOptimisers.supports_partial_fit
Online
PortfolioOptimisers.Online_Option
PortfolioOptimisers.Onl
PortfolioOptimisers.online_candidate_fields
PortfolioOptimisers.online_fields
PortfolioOptimisers.online_state_seed(::Union{<:PortfolioOptimisers.AbstractEstimator, <:StatsBase.CovarianceEstimator}, ::PortfolioOptimisers.Option{<:Integer})
PortfolioOptimisers.update_online_estimator
```

### The paired buffer

A family whose batch verb reads a returns matrix **and** a factor matrix has two sequences to keep, and one buffer cannot keep them. [`PortfolioOptimisers.FactorSampleBufferState`](@ref) is the pair, and every operation of the seam on it is the operation of its two halves.

```@docs
PortfolioOptimisers.FactorSampleBufferState
PortfolioOptimisers.assert_factor_sample_buffer_state
PortfolioOptimisers.assert_factor_sample_buffer
PortfolioOptimisers.partial_fit!(state::PortfolioOptimisers.FactorSampleBufferState, x::PortfolioOptimisers.VecNum, f::PortfolioOptimisers.VecNum)
PortfolioOptimisers.merge_states(a::PortfolioOptimisers.FactorSampleBufferState, b::PortfolioOptimisers.FactorSampleBufferState)
Base.copy(x::PortfolioOptimisers.FactorSampleBufferState)
PortfolioOptimisers.port_opt_view(x::PortfolioOptimisers.FactorSampleBufferState, i, args...)
PortfolioOptimisers.partial_fit!(est::Union{<:PortfolioOptimisers.AbstractEstimator, <:StatsBase.CovarianceEstimator}, X::PortfolioOptimisers.VecNum_MatNum, F::PortfolioOptimisers.VecNum_MatNum; dims::Int = 1, active_mask = nothing, estimation_mask = nothing)
```

## The coverage policy

A moment estimator that carries a [`CoveragePolicy`](@ref) fits each cell of its answer on the observations that cell has, instead of reducing its window to the Coverage Universe. The rule for a delisted asset is an [`PortfolioOptimisers.AbstractCoverageAlgorithm`](@ref), whose two verbs are [`PortfolioOptimisers.fold_inactive!`](@ref) at fold time and [`PortfolioOptimisers.admits`](@ref) at read-out, and the per-cell denominators live in a [`PortfolioOptimisers.CoverageCounts`](@ref) the partial-fit state carries.

```@docs
CoveragePolicy
PortfolioOptimisers.AbstractCoverageAlgorithm
DecayCoverage
ResetCoverage
ExpireCoverage
PortfolioOptimisers.fold_inactive!
PortfolioOptimisers.fold_inactive!(::Union{<:DecayCoverage, <:ExpireCoverage}, state::PortfolioOptimisers.AbstractPartialFitState, ::AbstractVector{<:Bool})
PortfolioOptimisers.admits
PortfolioOptimisers.admits(::Union{<:DecayCoverage, <:ResetCoverage}, share::Real, active::Bool, ::Integer, min_coverage::Real)
PortfolioOptimisers.admits(alg::ExpireCoverage, share::Real, active::Bool, stale::Integer, min_coverage::Real)
PortfolioOptimisers.CoverageCounts
PortfolioOptimisers.coverage_counts_seed
Base.copy(x::PortfolioOptimisers.CoverageCounts)
PortfolioOptimisers.coverage_counts_view
PortfolioOptimisers.coverage_valid
PortfolioOptimisers.coverage_valid_block
PortfolioOptimisers.coverage_step!
PortfolioOptimisers.coverage_merge_stale
PortfolioOptimisers.coverage_reset!
PortfolioOptimisers.coverage_admission
PortfolioOptimisers.coverage_divide
PortfolioOptimisers.coverage_frame
PortfolioOptimisers.coverage_refuse!
```

## Iteration and indexing

Estimators, algorithms, and results behave as length-1 iterables and containers to simplify dispatch and slicing in hierarchical workflows.

```@docs
Base.iterate(obj::Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}, state)
Base.getindex(obj::Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}, i::Int)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
