# Base

[`01_Base.jl`](https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/main/src/01_Base.jl) implements the most basal symbols used in `PortfolioOptimisers.jl`.

## Base abstract types

`PortfolioOptimisers.jl` is designed in a deliberately structured and hierarchical way. This enables us to create self-contained, independent, composable processes. These abstract types form the basis of this hierarchy.

```@docs
AbstractEstimator
AbstractAlgorithm
AbstractResult
```

## Pretty printing

`PortfolioOptimisers.jl`'s types tend to contain quite a lot of information, these functions enable pretty printing so they are easier to interpret.

```@docs
@define_pretty_show
has_pretty_show_method
```

## Error types

Many of the types defined in `PortfolioOptimisers.jl` make use of extensive data validation to ensure values meet various criteria. This simplifies the implementation of methods, and improves performance and by delegating as many checks as possible to variable instantiation. In cases where validation cannot be performed at variable instantiation, they are performed as soon as possible within functions.

`PortfolioOptimisers.jl` aims to catch potential data validation issues as soon as possible and in an informative manner, in order to do so it makes use of a few custom error types.

```@docs
PortfolioOptimisersError
IsNothingError
IsEmptyError
IsNonFiniteError
```

## Utility types

Custom types are the bread and butter of `PorfolioOptimisers.jl`, the following types non-specific and used throughout the library.

```@docs
VecScalar
AbstractEstimatorValueAlgorithm
```

## Base type aliases

`PortfolioOptimisers.jl` heavily relies on `Julia`'s dispatch and type system to ensure data validity. Many custom types and functions/methods can accept different data types. These can be represented as type unions, many of which are used throughout the library. The following type aliases centralise these union definitions, as well as improving correctness and maintainability.

```@docs
Option{T}
VecNum
VecInt
MatNum
ArrNum
Num_VecNum
Num_ArrNum
PairStrNum
DictStrNum
MultiEstValType
EstValType
Str_Expr
VecStr_Expr
EqnType
VecVecNum
VecVecInt
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
Num_VecNum_VecScalar
Num_ArrNum_VecScalar
```

## Documentation glossary

In order to standardise the documentation we use a glossary of terms.

```@docs
glossary
validation
```
