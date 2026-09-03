# Base

[`src/01_Base/`](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/src/01_Base) implements the most basal symbols used in `PortfolioOptimisers.jl`. One file per concept: the docstring dictionaries, the type roots, the pretty-show macro, the `ScopedConfig` holders, the load-time preferences, the message builders, the error hierarchy, the type aliases, the observation weights, the `assert_*` family, `VecScalar`, the `NormError` family, the Kaniadakis logarithm and the partial-fit state seam.

```@docs
PortfolioOptimisers
```

## Base abstract types

`PortfolioOptimisers.jl` is designed in a deliberately structured and hierarchical way. Enabling us to create self-contained, independent, composable processes. These abstract types form the basis of this hierarchy.

```@docs
AbstractEstimator
AbstractAlgorithm
AbstractResult
```

## Configuration

Package-level configuration values (pretty-printing collapse, fuzzy-suggestion distance, equation-parser resource caps) are held in thread-safe [`ScopedConfig`](@ref) holders: a `set_*!` setter swaps the global default atomically, a `with_*` helper overrides it for the dynamic extent of a call (task-scoped, automatically restored), and per-project defaults can be seeded at load time via Preferences.jl.

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

An incremental fit folds one observation into an estimate without reading the sample again. [`partial_fit!`](@ref) is the verb that folds it, its running quantities live in a [`AbstractPartialFitState`](@ref), and [`merge_states`](@ref) combines the states of two disjoint blocks of observations into the state of the concatenated block.

```@docs
partial_fit!
PortfolioOptimisers.AbstractPartialFitState
PortfolioOptimisers.merge_states
PortfolioOptimisers.assert_mergeable_states
PortfolioOptimisers.chan_merge
PortfolioOptimisers.assert_partial_fit_state
PortfolioOptimisers.partial_fit_cache
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
