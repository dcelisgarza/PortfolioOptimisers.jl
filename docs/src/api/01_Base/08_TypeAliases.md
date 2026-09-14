# Type aliases

## Utilities

```@docs
AbstractCustomValue
AbstractEstimatorValueAlgorithm
VectorAbstractEstimatorValueAlgorithm
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
Func_VecNum
```
