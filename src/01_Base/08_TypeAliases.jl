"""
    const VecNum = AbstractVector{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}

Alias for an abstract vector of numeric types or JuMP scalar types.

# Related

  - [`VecInt`](@ref)
  - [`MatNum`](@ref)
  - [`JuMP.AbstractJuMPScalar`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.JuMP.AbstractJuMPScalar)
"""
const VecNum = AbstractVector{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}
"""
    const VecInt = AbstractVector{<:Integer}

Alias for an abstract vector of integer types.

# Related

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
  - [`ArrNum`](@ref)
"""
const VecInt = AbstractVector{<:Integer}
"""
    const MatNum = AbstractMatrix{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}

Alias for an abstract matrix of numeric types or JuMP scalar types.

# Related

  - [`VecNum`](@ref)
  - [`ArrNum`](@ref)
  - [`VecMatNum`](@ref)
"""
const MatNum = AbstractMatrix{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}
"""
    const ArrNum = AbstractArray{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}

Alias for an abstract array of numeric types or JuMP scalar types.

# Related

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
"""
const ArrNum = AbstractArray{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}
"""
    const Arr3Num = AbstractArray{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}, 3}

Alias for an abstract 3-dimensional array of numeric types or JuMP scalar types.

It is [`ArrNum`](@ref) with the rank fixed at three, for data that is a stack of matrices and not one matrix. A window of time-varying features is such data, and its first axis holds the observations. A method that dispatches on this alias keeps the 2-D and 3-D entry points apart with no `ndims` branch at run time.

# Related

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
  - [`ArrNum`](@ref)
"""
const Arr3Num = AbstractArray{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}, 3}
"""
    const VecNum_MatNum = Union{<:VecNum, <:MatNum}

Alias for a union of an abstract vector of numeric types and an abstract matrix of numeric types.

# Related

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
"""
const VecNum_MatNum = Union{<:VecNum, <:MatNum}
"""
    const MatNum_Arr3Num = Union{<:MatNum, <:Arr3Num}

Alias for a union of an abstract matrix and an abstract 3-dimensional array of numeric types.

A Feature Matrix has one of two shapes. It is a static `assets × features` matrix, or a window of time-varying features, `observations × assets × features`, whose first axis holds the observations. [`PricesResult`](@ref) and [`ReturnsResult`](@ref) hold both shapes. [`FeatureDistance`](@ref) reads both, and it tells them apart by dispatch and not by an `ndims` branch.

# Related

  - [`MatNum`](@ref)
  - [`Arr3Num`](@ref)
"""
const MatNum_Arr3Num = Union{<:MatNum, <:Arr3Num}
"""
    const Num_VecNum = Union{<:Number, <:VecNum}

Alias for a union of a numeric type and an abstract vector of numeric types.

# Related

  - [`VecNum`](@ref)
  - [`ArrNum`](@ref)
"""
const Num_VecNum = Union{<:Number, <:VecNum}
"""
    const Func_VecNum = Union{<:Function, <:VecNum}

Alias for a union of a function and a vector of numeric types.

# Related

  - [`VecNum`](@ref)
  - [`Func_Num_VecNum`](@ref)
"""
const Func_VecNum = Union{<:Function, <:VecNum}
"""
    const Func_Num_VecNum = Union{<:Number, <:Func_VecNum}

Alias for a union of a function type, a numeric type and an abstract vector of numeric types.

# Related

  - [`Func_VecNum`](@ref)
"""
const Func_Num_VecNum = Union{<:Number, <:Func_VecNum}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for custom value algorithms. A user writes a custom value algorithm to give an estimator a value that the library does not compute.

The methods that a subtype must implement depend on the estimator that reads it.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`CVal_Func_Num_VecNum`](@ref)
  - [`CustomValueExpectedReturns`](@ref)
"""
abstract type AbstractCustomValue <: AbstractAlgorithm end
"""
    const CVal_Func_Num_VecNum = Union{<:AbstractCustomValue, <:Func_Num_VecNum}

Alias for the two ways in which a caller gives a custom value. The caller gives an [`AbstractCustomValue`](@ref) algorithm, or one of the plain forms of [`Func_Num_VecNum`](@ref).

The `val` field of [`CustomValueExpectedReturns`](@ref) accepts both ways. One bound on that field keeps the two ways in agreement, so the alias exists.

# Related

  - [`AbstractCustomValue`](@ref)
  - [`Func_Num_VecNum`](@ref)
  - [`CustomValueExpectedReturns`](@ref)
"""
const CVal_Func_Num_VecNum = Union{<:AbstractCustomValue, <:Func_Num_VecNum}
"""
    const Num_ArrNum = Union{<:Number, <:ArrNum}

Alias for a union of a numeric type and an abstract array of numeric types.

# Related

  - [`ArrNum`](@ref)
  - [`VecNum`](@ref)
"""
const Num_ArrNum = Union{<:Number, <:ArrNum}
"""
    const PairStrNum = Pair{<:AbstractString, <:Number}

Alias for a pair of an abstract string and a numeric type.

# Related

  - [`DictStrNum`](@ref)
  - [`MultiEstValType`](@ref)
"""
const PairStrNum = Pair{<:AbstractString, <:Number}
"""
    const GSCVKey = Union{<:AbstractString, Expr, Symbol, <:ComposedFunction,
                          <:Accessors.PropertyLens, <:Accessors.IndexLens, <:Integer}

Alias for the type of a key of a grid search cross-validation. A key is an abstract string, an expression, a symbol, a composed function, an accessor lens, or an integer. An integer is the position of a step when the search tunes a `Pipeline`.

# Related

  - [`PairGSCV`](@ref)
  - [`DictGSCV`](@ref)
  - [`MultiGSCVValType`](@ref)
"""
const GSCVKey = Union{<:AbstractString, Expr, Symbol, <:ComposedFunction,
                      <:Accessors.PropertyLens, <:Accessors.IndexLens, <:Integer}
"""
    const RSCVVal = Union{<:AbstractVector, <:Distributions.Distribution}

Alias for the type of a value of a randomised search cross-validation. A value is an abstract vector or a distribution.

# Related

  - [`PairGSCV`](@ref)
  - [`DictGSCV`](@ref)
  - [`MultiGSCVValType`](@ref)
"""
const RSCVVal = Union{<:AbstractVector, <:Distributions.Distribution}
"""
    const PairGSCV = Pair{<:GSCVKey, <:AbstractVector}

Alias for a pair of a [`GSCVKey`](@ref) and an abstract vector.

# Related

  - [`DictGSCV`](@ref)
  - [`MultiGSCVValType`](@ref)
"""
const PairGSCV = Pair{<:GSCVKey, <:AbstractVector}
"""
    const DictStrNum = AbstractDict{<:AbstractString, <:Number}

Alias for an abstract dictionary with string keys and numeric values.

# Related

  - [`PairStrNum`](@ref)
  - [`MultiEstValType`](@ref)
"""
const DictStrNum = AbstractDict{<:AbstractString, <:Number}
"""
    const DictGSCV = AbstractDict{<:GSCVKey, <:AbstractVector}

Alias for an abstract dictionary with [`GSCVKey`](@ref) keys and abstract vector values.

# Related

  - [`PairGSCV`](@ref)
  - [`MultiGSCVValType`](@ref)
"""
const DictGSCV = AbstractDict{<:GSCVKey, <:AbstractVector}
"""
    const MultiEstValType = Union{<:DictStrNum, <:AbstractVector{<:PairStrNum}}

Alias for a union of a dictionary with string keys and numeric values, and a vector of string-number pairs.

# Related

  - [`DictStrNum`](@ref)
  - [`PairStrNum`](@ref)
  - [`EstValType`](@ref)
"""
const MultiEstValType = Union{<:DictStrNum, <:AbstractVector{<:PairStrNum}}
"""
    const MultiGSCVValType = Union{<:DictGSCV, <:AbstractVector{<:PairGSCV}}

Alias for a union of a [`DictGSCV`](@ref) and a vector of [`PairGSCV`](@ref) pairs.

# Related

  - [`DictGSCV`](@ref)
  - [`PairGSCV`](@ref)
  - [`VecMultiGSCVValType`](@ref)
  - [`MultiGSCVValType_VecMultiGSCVValType`](@ref)
"""
const MultiGSCVValType = Union{<:DictGSCV, <:AbstractVector{<:PairGSCV}}
"""
    const VecMultiGSCVValType = AbstractVector{<:MultiGSCVValType}

Alias for an abstract vector of `MultiGSCVValType` elements.

# Related

  - [`DictGSCV`](@ref)
  - [`PairGSCV`](@ref)
  - [`MultiGSCVValType`](@ref)
  - [`MultiGSCVValType_VecMultiGSCVValType`](@ref)
"""
const VecMultiGSCVValType = AbstractVector{<:MultiGSCVValType}
"""
    const MultiGSCVValType_VecMultiGSCVValType = Union{<:MultiGSCVValType,
                                                       <:VecMultiGSCVValType}

Alias for a union of a `MultiGSCVValType` and a `VecMultiGSCVValType`.

# Related

  - [`DictGSCV`](@ref)
  - [`PairGSCV`](@ref)
  - [`MultiGSCVValType`](@ref)
  - [`VecMultiGSCVValType`](@ref)
"""
const MultiGSCVValType_VecMultiGSCVValType = Union{<:MultiGSCVValType,
                                                   <:VecMultiGSCVValType}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all estimator value algorithm types.

A subtype of `AbstractEstimatorValueAlgorithm` computes the value of a constraint or a bound over the universe, from the asset sets. An estimator slot that accepts a value can hold such an algorithm in the place of the value.

The family has branches by the shape of the value that the algorithm returns, because a slot accepts one shape and refuses the others. [`VectorAbstractEstimatorValueAlgorithm`](@ref) is the branch that returns a `Num_VecNum`, and [`VectorEstValType`](@ref) is the bound of a slot that accepts that branch. [`EstValType`](@ref) takes the branch as its type parameter. A slot that parses equations, such as the `val` of [`LinearConstraintEstimator`](@ref), takes an [`EqnType`](@ref), and no branch of this family is an `EqnType`.

# Interfaces

To implement a new estimator value algorithm, subtype the branch of `AbstractEstimatorValueAlgorithm` for the shape of the value that it returns, with a struct that holds its parameters. Then implement the method below.

  - `estimator_to_val(alg::AbstractEstimatorValueAlgorithm, sets::UniverseSets, val::Option{<:Number} = nothing, key::Option{<:AbstractString} = nothing; datatype::DataType = Float64, strict::Bool = false) -> Num_VecNum`: Computes a number, or a vector of numbers, over the universe. A usual method does these two steps:
      + `val = ifelse(isnothing(val), <default value use with datatype element type>, val)`: Computes the default value when `val` is `nothing`.
      + `nx = sets.dict[ifelse(isnothing(key), sets.xkey, key)]`: Gets the universe whose assets the values map to.

## Arguments

  - `alg`: A concrete subtype of a branch of `AbstractEstimatorValueAlgorithm`.
  - $(arg_dict[:sets])
  - $(arg_dict[:val])
  - $(arg_dict[:ekey])
  - $(arg_dict[:datatype])
  - $(arg_dict[:strict])

## Returns

  - `val::Num_VecNum`: The number, or the vector of numbers.

# Examples

A minimal estimator value algorithm:

```jldoctest
julia> struct MyIncreasingValue <: PortfolioOptimisers.VectorAbstractEstimatorValueAlgorithm end

julia> function PortfolioOptimisers.estimator_to_val(alg::MyIncreasingValue, sets::UniverseSets,
                                                     val::PortfolioOptimisers.Option{<:Number} = nothing,
                                                     key::PortfolioOptimisers.Option{<:AbstractString} = nothing;
                                                     datatype::DataType = Float64,
                                                     strict::Bool = false)
           val = ifelse(isnothing(val), zero(datatype), val)
           nx = sets.dict[ifelse(isnothing(key), sets.xkey, key)]
           arr = ((1 - val):(length(nx) - val))
           return arr
       end

julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"sha\", \"bis\", \"man\"]))
UniverseSets
    xkey ┼ String: "nx"
   uxkey ┼ String: "ux"
   tfkey ┼ String: "nf"
  utfkey ┼ String: "uf"
   cfkey ┼ String: "ncf"
  ucfkey ┼ String: "ucf"
   nikey ┼ String: "ni"
    dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["sha", "bis", "man"])

julia> estimator_to_val(MyIncreasingValue(), sets)
1.0:1.0:3.0
```

# Related

  - [`VectorAbstractEstimatorValueAlgorithm`](@ref)
  - [`VectorEstValType`](@ref)
  - [`EstValType`](@ref)
  - [`estimator_to_val`](@ref)
"""
abstract type AbstractEstimatorValueAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the estimator value algorithms that return a numeric or a vector of numeric value.

A slot that holds a number, or a vector of numbers, over the universe accepts this branch of [`AbstractEstimatorValueAlgorithm`](@ref) and no other. The slot writes that bound as [`VectorEstValType`](@ref), which also refuses a matrix. The branch and the shape that a slot accepts are thus part of the type of the slot, and not a check at the time when the library resolves the value.

# Interfaces

Subtype `VectorAbstractEstimatorValueAlgorithm`, and implement the `estimator_to_val` method that [`AbstractEstimatorValueAlgorithm`](@ref) documents. The method must return a `Num_VecNum`.

# Related

  - [`AbstractEstimatorValueAlgorithm`](@ref)
  - [`VectorEstValType`](@ref)
  - [`estimator_to_val`](@ref)
  - [`UniformValues`](@ref)
"""
abstract type VectorAbstractEstimatorValueAlgorithm <: AbstractEstimatorValueAlgorithm end
"""
    const EstValType{T <: AbstractEstimatorValueAlgorithm} = Union{<:Num_VecNum, <:MatNum,
                                                                   <:PairStrNum,
                                                                   <:MultiEstValType, T}

Alias for a union of a number, a vector of numbers, a matrix of numbers, a string-number pair, a multi-estimator value, and the branch `T` of [`AbstractEstimatorValueAlgorithm`](@ref) that computes such a value.

`T` names the branch that the slot accepts, so the type of a slot states which algorithms the slot can resolve. The alias with no parameter accepts every branch, and it is the widest bound of the family.

`T` does not change the shape of the plain value: every `EstValType` accepts a matrix. A slot that holds a number, or a vector of numbers, over the universe resolves no matrix, so it writes [`VectorEstValType`](@ref) instead.

# Related

  - [`Num_VecNum`](@ref)
  - [`MatNum`](@ref)
  - [`PairStrNum`](@ref)
  - [`MultiEstValType`](@ref)
  - [`VectorEstValType`](@ref)
  - [`AbstractEstimatorValueAlgorithm`](@ref)
  - [`VectorAbstractEstimatorValueAlgorithm`](@ref)
"""
const EstValType{T <: AbstractEstimatorValueAlgorithm} = Union{<:Num_VecNum, <:MatNum,
                                                               <:PairStrNum,
                                                               <:MultiEstValType, T}
"""
    const VectorEstValType = Union{<:Num_VecNum, <:PairStrNum, <:MultiEstValType,
                                   <:VectorAbstractEstimatorValueAlgorithm}

Alias for a union of a number, a vector of numbers, a string-number pair, a multi-estimator value, and a [`VectorAbstractEstimatorValueAlgorithm`](@ref).

A slot that holds a number, or a vector of numbers, over the universe takes this alias as its bound. Each shape that the alias accepts resolves to a `Num_VecNum` through [`estimator_to_val`](@ref). The alias refuses a matrix, which [`EstValType`](@ref) accepts, because the result of such a slot holds a `Num_VecNum` only. The constructor of the estimator thus refuses a matrix, and not the resolver.

# Related

  - [`Num_VecNum`](@ref)
  - [`PairStrNum`](@ref)
  - [`MultiEstValType`](@ref)
  - [`EstValType`](@ref)
  - [`VectorAbstractEstimatorValueAlgorithm`](@ref)
"""
const VectorEstValType = Union{<:Num_VecNum, <:PairStrNum, <:MultiEstValType,
                               <:VectorAbstractEstimatorValueAlgorithm}
"""
    const Str_Expr = Union{<:AbstractString, Expr}

Alias for a union of an abstract string and a Julia expression.

# Related

  - [`VecStr_Expr`](@ref)
  - [`EqnType`](@ref)
"""
const Str_Expr = Union{<:AbstractString, Expr}
"""
    const VecStr_Expr = AbstractVector{<:Str_Expr}

Alias for an abstract vector of strings or Julia expressions.

# Related

  - [`Str_Expr`](@ref)
  - [`EqnType`](@ref)
"""
const VecStr_Expr = AbstractVector{<:Str_Expr}
"""
    const EqnType = Union{<:AbstractString, Expr, <:VecStr_Expr}

Alias for a union of a string, a Julia expression, and a vector of strings and expressions.

Every consumer of this alias parses the value with [`parse_equation`](@ref), so the alias holds equation text and nothing more. No branch of [`AbstractEstimatorValueAlgorithm`](@ref) is an `EqnType`. An algorithm computes a value over the universe. A value has no comparison operator and no side, so the library cannot build a constraint row from it. A slot that accepts an algorithm is an [`EstValType`](@ref).

# Related

  - [`Str_Expr`](@ref)
  - [`VecStr_Expr`](@ref)
  - [`EstValType`](@ref)
  - [`parse_equation`](@ref)
"""
const EqnType = Union{<:AbstractString, Expr, <:VecStr_Expr}
"""
    const VecVecNum = AbstractVector{<:VecNum}

Alias for an abstract vector of numeric vectors.

# Related

  - [`VecNum`](@ref)
  - [`VecMatNum`](@ref)
"""
const VecVecNum = AbstractVector{<:VecNum}
"""
    const VecVecInt = AbstractVector{<:VecInt}

Alias for an abstract vector of integer vectors.

# Related

  - [`VecInt`](@ref)
"""
const VecVecInt = AbstractVector{<:VecInt}
"""
    const VecInt_VecVecInt = Union{<:VecInt, <:VecVecInt}

Alias for a union of an abstract vector of integers and an abstract vector of integer vectors.

# Related

  - [`VecInt`](@ref)
  - [`VecVecInt`](@ref)
"""
const VecInt_VecVecInt = Union{<:VecInt, <:VecVecInt}
"""
    const VecVecVecInt = AbstractVector{<:VecVecInt}

Alias for an abstract vector of abstract vectors of integer vectors.

# Related

  - [`VecVecInt`](@ref)
"""
const VecVecVecInt = AbstractVector{<:VecVecInt}
"""
    const VecMatNum = AbstractVector{<:MatNum}

Alias for an abstract vector of numeric matrices.

# Related

  - [`MatNum`](@ref)
  - [`VecNum`](@ref)
"""
const VecMatNum = AbstractVector{<:MatNum}
"""
    const VecStr = AbstractVector{<:AbstractString}

Alias for an abstract vector of strings.

# Related

  - [`Str_Expr`](@ref)
  - [`VecStr_Expr`](@ref)
"""
const VecStr = AbstractVector{<:AbstractString}
"""
    const VecPair = AbstractVector{<:Pair}

Alias for an abstract vector of pairs.

# Related

  - [`PairStrNum`](@ref)
"""
const VecPair = AbstractVector{<:Pair}
"""
    const VecJuMPScalar = Union{<:AbstractVector{<:JuMP.AbstractJuMPScalar}}

Alias for an abstract vector of JuMP scalar types.

# Related

  - [`VecNum`](@ref)
"""
const VecJuMPScalar = Union{<:AbstractVector{<:JuMP.AbstractJuMPScalar}}
"""
    const Option{T} = Union{Nothing, T}

Alias for an optional value of type `T`, which can be `nothing`.

# Related

  - [`EstValType`](@ref)
"""
const Option{T} = Union{Nothing, T}
"""
    const MatNum_VecMatNum = Union{<:MatNum, <:VecMatNum}

Alias for a union of a numeric matrix and a vector of numeric matrices.

# Related

  - [`MatNum`](@ref)
  - [`VecMatNum`](@ref)
"""
const MatNum_VecMatNum = Union{<:MatNum, <:VecMatNum}
"""
    const Int_VecInt = Union{<:Integer, <:VecInt}

Alias for a union of an integer and a vector of integers.

# Related

  - [`VecInt`](@ref)
"""
const Int_VecInt = Union{<:Integer, <:VecInt}
"""
    const VecNum_VecVecNum = Union{<:VecNum, <:VecVecNum}

Alias for a union of a numeric vector and a vector of numeric vectors.

# Related

  - [`VecNum`](@ref)
  - [`VecVecNum`](@ref)
"""
const VecNum_VecVecNum = Union{<:VecNum, <:VecVecNum}
"""
    const VecDate = AbstractVector{<:Dates.AbstractTime}

Alias for an abstract vector of date or time types.

# Related

  - [`VecNum`](@ref)
  - [`VecStr`](@ref)
"""
const VecDate = AbstractVector{<:Dates.AbstractTime}
"""
    const Dict_Vec = Union{<:AbstractDict, <:AbstractVector}

Alias for a union of an abstract dictionary and an abstract vector.

# Related

  - [`DictStrNum`](@ref)
  - [`VecNum`](@ref)
"""
const Dict_Vec = Union{<:AbstractDict, <:AbstractVector}
"""
    const Sym_Str = Union{Symbol, <:AbstractString}

Alias for a union of a symbol and an abstract string.

# Related

  - [`VecStr`](@ref)
"""
const Sym_Str = Union{Symbol, <:AbstractString}
"""
    const Str_Vec = Union{<:AbstractString, <:AbstractVector}

Alias for a union of an abstract string and an abstract vector.

# Related

  - [`VecStr`](@ref)
  - [`Str_Expr`](@ref)
"""
const Str_Vec = Union{<:AbstractString, <:AbstractVector}
public AbstractEstimatorValueAlgorithm, VectorAbstractEstimatorValueAlgorithm
