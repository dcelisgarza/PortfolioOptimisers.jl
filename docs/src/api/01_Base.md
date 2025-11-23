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

```@docs
PortfolioOptimisersError
IsNothingError
IsEmptyError
IsNonFiniteError
```

## Private

```@docs
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
Option{T}
MatNum_VecMatNum
Int_VecInt
VecNum_VecVecNum
VecDate
Dict_Vec
Sym_Str
Str_Vec
```
