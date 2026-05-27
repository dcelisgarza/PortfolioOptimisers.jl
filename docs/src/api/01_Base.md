# Base

[`01_Base.jl`](https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/main/src/01_Base.jl) implements the most basal symbols used in `PortfolioOptimisers.jl`.

```@docs
PortfolioOptimisers
```

## Base abstract types

`PortfolioOptimisers.jl` is designed in a deliberately structured and hierarchical way. Enabling us to create self-contained, independent, composable processes. These abstract types form the basis of this hierarchy.

```@docs
AbstractEstimator
AbstractAlgorithm
AbstractResult
DynamicAbstractWeights
```

## Pretty printing

`PortfolioOptimisers.jl`'s types tend to contain quite a lot of information, these functions enable pretty printing so they are easier to interpret.

```@docs
@define_pretty_show
has_pretty_show_method
```

## Utilities

Custom types are the bread and butter of `PorfolioOptimisers.jl`, the following types and utilities are non-specific and used throughout the library.

```@docs
VecScalar
AbstractEstimatorValueAlgorithm
SingletonVector
get_observation_weights
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
```

## Assertions

In order to increase correctness, robustness, and safety, we make extensive use of [defensive programming](https://en.wikipedia.org/wiki/Defensive_programming). The following functions perform some of these validations and are usually called at variable instantiation.

```@docs
assert_nonempty_nonneg_finite_val
assert_nonempty_gt0_finite_val
assert_nonempty_finite_val
assert_matrix_issquare
```

## Base type aliases

`PortfolioOptimisers.jl` heavily relies on `Julia`'s dispatch and type system to ensure data validity. Many custom types and functions/methods can accept different data types. These can be represented as type unions, many of which are used throughout the library. The following type aliases centralise these union definitions, as well as improving correctness and maintainability.

```@docs
Option{T}
VecNum
VecInt
MatNum
ArrNum
VecNum_MatNum
Num_VecNum
Func_Num_VecNum
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
```

## Glossaries

In order to standardise the documentation we use a arg_dict of terms.

```@docs
arg_dict
val_dict
ret_dict
field_dict
math_dict
```

## Iteration and indexing

Estimators, algorithms, and results behave as length-1 iterables and containers to simplify dispatch and slicing in hierarchical workflows.

```@docs
Base.iterate(obj::Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}, state)
Base.getindex(obj::Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}, i::Int)
Base.getindex(A::SingletonVector, i::Int)
```
