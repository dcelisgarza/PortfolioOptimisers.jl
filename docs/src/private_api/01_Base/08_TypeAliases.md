```@meta
Description = "Type aliases, private API of PortfolioOptimisers.jl: AbstractCustomValue, Option, VecNum, VecInt, MatNum, ArrNum, Arr3Num, VecNum_MatNum, MatNum_Arr3Num, …"
```

# Type aliases: private API

Many functions and fields of `PortfolioOptimisers.jl` accept more than one type, for example a vector or a matrix of numbers. The aliases below name those unions. A signature then reads `VecNum_MatNum`, and every method that accepts the same inputs uses the same union.

```@docs
AbstractCustomValue
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
