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

Rank-restricted counterpart of [`ArrNum`](@ref), for the data that is a stack of matrices rather than a single one — a window of time-varying features, whose observation axis leads. Dispatching on this alias keeps the 2-D and 3-D entry points apart without a runtime `ndims` branch.

# Related

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
  - [`ArrNum`](@ref)
"""
const Arr3Num = AbstractArray{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}, 3}
"""
    const VecNum_MatNum = Union{<:VecNum, <:MatNum}

Alias for a union of a numeric type or an abstract matrix of numeric types.

# Related

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
"""
const VecNum_MatNum = Union{<:VecNum, <:MatNum}
"""
    const MatNum_Arr3Num = Union{<:MatNum, <:Arr3Num}

Alias for a union of an abstract matrix and an abstract 3-dimensional array of numeric types.

The two admissible shapes of a feature matrix: a static `assets × features` matrix, and a window of time-varying features whose observation axis leads, `observations × assets × features`. Both shapes are carried by [`PricesResult`](@ref) and [`ReturnsResult`](@ref) and consumed by [`FeatureDistance`](@ref), which distinguishes them by dispatch rather than by an `ndims` branch.

# Related

  - [`MatNum`](@ref)
  - [`Arr3Num`](@ref)
"""
const MatNum_Arr3Num = Union{<:MatNum, <:Arr3Num}
"""
    const Num_VecNum = Union{<:Number, <:VecNum}

Alias for a union of a numeric type or an abstract vector of numeric types.

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

Alias for a union of a function type or a numeric type or an abstract vector of numeric types.

# Related

  - [`Func_VecNum`](@ref)
"""
const Func_Num_VecNum = Union{<:Number, <:Func_VecNum}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for custom value algorithms. These are user defined algorithms that return a custom value for an estimator.

The interfaces users must implement depend on the estimator type.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`CVal_Func_Num_VecNum`](@ref)
  - [`CustomValueExpectedReturns`](@ref)
"""
abstract type AbstractCustomValue <: AbstractAlgorithm end
"""
    const CVal_Func_Num_VecNum = Union{<:AbstractCustomValue, <:Func_Num_VecNum}

Alias for the two ways a caller supplies a custom value: an [`AbstractCustomValue`](@ref) algorithm, or the plain forms that [`Func_Num_VecNum`](@ref) already groups.

The group exists because the `val` field of [`CustomValueExpectedReturns`](@ref) accepts both, and one bound on that field is what keeps the two routes from drifting apart.

# Related

  - [`AbstractCustomValue`](@ref)
  - [`Func_Num_VecNum`](@ref)
  - [`CustomValueExpectedReturns`](@ref)
"""
const CVal_Func_Num_VecNum = Union{<:AbstractCustomValue, <:Func_Num_VecNum}
"""
    const Num_ArrNum = Union{<:Number, <:ArrNum}

Alias for a union of a numeric type or an abstract array of numeric types.

# Related

  - [`ArrNum`](@ref)
  - [`VecNum`](@ref)
"""
const Num_ArrNum = Union{<:Number, <:ArrNum}
"""
    const PairStrNum = Pair{<:AbstractString, <:Number}

Alias for a pair consisting of an abstract string and a numeric type.

# Related

  - [`DictStrNum`](@ref)
  - [`MultiEstValType`](@ref)
"""
const PairStrNum = Pair{<:AbstractString, <:Number}
"""
    const GSCVKey = Union{<:AbstractString, Expr, Symbol, <:ComposedFunction,
                          <:Accessors.PropertyLens, <:Accessors.IndexLens, <:Integer}

Alias for a key type used in grid search cross-validation, which can be an abstract string, an expression, a symbol, a composed function, an accessor lens, or an integer (a step position when tuning a `Pipeline`).

# Related

  - [`PairGSCV`](@ref)
  - [`DictGSCV`](@ref)
  - [`MultiGSCVValType`](@ref)
"""
const GSCVKey = Union{<:AbstractString, Expr, Symbol, <:ComposedFunction,
                      <:Accessors.PropertyLens, <:Accessors.IndexLens, <:Integer}
"""
    const RSCVVal = Union{<:AbstractVector, <:Distributions.Distribution}

Alias for a value type used in randomised search cross-validation, which can be an abstract vector or a distribution.

# Related

  - [`PairGSCV`](@ref)
  - [`DictGSCV`](@ref)
  - [`MultiGSCVValType`](@ref)
"""
const RSCVVal = Union{<:AbstractVector, <:Distributions.Distribution}
"""
    const PairGSCV = Pair{<:GSCVKey, <:AbstractVector}

Alias for a pair consisting of an abstract string and an abstract vector.

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

Alias for an abstract dictionary with string keys and abstract vector values.

# Related

  - [`PairGSCV`](@ref)
  - [`MultiGSCVValType`](@ref)
"""
const DictGSCV = AbstractDict{<:GSCVKey, <:AbstractVector}
"""
    const MultiEstValType = Union{<:DictStrNum, <:AbstractVector{<:PairStrNum}}

Alias for a union of a dictionary with string keys and numeric values, or a vector of string-number pairs.

# Related

  - [`DictStrNum`](@ref)
  - [`PairStrNum`](@ref)
  - [`EstValType`](@ref)
"""
const MultiEstValType = Union{<:DictStrNum, <:AbstractVector{<:PairStrNum}}
"""
    const MultiGSCVValType = Union{<:DictGSCV, <:AbstractVector{<:PairGSCV}}

Alias for a union of an abstract dictionary with string keys and abstract vector values, or a vector of string-vector pairs.

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

Alias for a union of `MultiGSCVValType` and `VecMultiGSCVValType` elements.

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

Subtypes of `AbstractEstimatorValueAlgorithm` implement algorithms for computing constraint result values. These are used to extend or modify the behavior of estimators in a composable and modular fashion.

The family is split by the shape of the value the algorithm returns, because a slot admits one shape and refuses the others. [`VectorAbstractEstimatorValueAlgorithm`](@ref) is the branch that returns a `Num_VecNum`, and [`EstValType`](@ref) takes the branch a slot admits as its type parameter. A slot that parses equations, such as [`LinearConstraintEstimator`](@ref)'s `val`, takes an [`EqnType`](@ref), and no branch of this family is an `EqnType`.

# Interfaces

In order to implement a new estimator value algorithm which will work seamlessly with the library, subtype the branch of `AbstractEstimatorValueAlgorithm` whose value shape the algorithm returns with all necessary parameters struct, and implement the following method:

  - `estimator_to_val(alg::AbstractEstimatorValueAlgorithm, sets::UniverseSets, val::Option{<:Number} = nothing, key::Option{<:AbstractString} = nothing; datatype::DataType = Float64, strict::Bool = false) -> Num_VecNum`: Converts an estimator value dictionary to a numeric or vector of numeric value. Usually this should compute some version of:
      + `val = ifelse(isnothing(val), <default value use with datatype element type>, val)`: Computes the default value to use if `val` is `nothing`.
      + `nx = sets.dict[ifelse(isnothing(key), sets.xkey, key)]`: Gets the universe to use for mapping values to assets.

## Arguments

  - `alg`: Concrete subtype of a branch of `AbstractEstimatorValueAlgorithm`.
  - $(arg_dict[:sets])
  - $(arg_dict[:val])
  - $(arg_dict[:ekey])
  - $(arg_dict[:datatype])
  - $(arg_dict[:strict])

## Returns

  - `val::Num_VecNum`: The numeric or vector of numeric value.

# Examples

We can create a dummy estimator value algorithm as follows:

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
    zkey ┼ String: "nz"
    dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["sha", "bis", "man"])

julia> estimator_to_val(MyIncreasingValue(), sets)
1.0:1.0:3.0
```

# Related

  - [`VectorAbstractEstimatorValueAlgorithm`](@ref)
  - [`EstValType`](@ref)
  - [`estimator_to_val`](@ref)
"""
abstract type AbstractEstimatorValueAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the estimator value algorithms that return a numeric or a vector of numeric value.

A slot that holds a number, or a vector of numbers, over the universe admits this branch of [`AbstractEstimatorValueAlgorithm`](@ref) and no other. The slot writes that bound as `EstValType{<:VectorAbstractEstimatorValueAlgorithm}`, so the branch a slot admits is part of the slot's type, not a check made when the value is resolved.

# Interfaces

Subtype `VectorAbstractEstimatorValueAlgorithm`, and implement the `estimator_to_val` method that [`AbstractEstimatorValueAlgorithm`](@ref) documents. The method must return a `Num_VecNum`.

# Related

  - [`AbstractEstimatorValueAlgorithm`](@ref)
  - [`EstValType`](@ref)
  - [`estimator_to_val`](@ref)
  - [`UniformValues`](@ref)
"""
abstract type VectorAbstractEstimatorValueAlgorithm <: AbstractEstimatorValueAlgorithm end
"""
    const EstValType{T <: AbstractEstimatorValueAlgorithm} = Union{<:Num_VecNum, <:MatNum,
                                                                   <:PairStrNum,
                                                                   <:MultiEstValType, T}

Alias for a union of numeric, vector of numeric, matrix of numeric, string-number pair, or multi-estimator value types, and the branch `T` of [`AbstractEstimatorValueAlgorithm`](@ref) that computes such a value.

`T` names the branch the slot admits, so a slot states in its own type which algorithms it can resolve. A slot that holds a number, or a vector of numbers, writes `EstValType{<:VectorAbstractEstimatorValueAlgorithm}`. The alias written without a parameter admits every branch, which is the widest bound the family gives.

# Related

  - [`Num_VecNum`](@ref)
  - [`PairStrNum`](@ref)
  - [`MultiEstValType`](@ref)
  - [`AbstractEstimatorValueAlgorithm`](@ref)
  - [`VectorAbstractEstimatorValueAlgorithm`](@ref)
"""
const EstValType{T <: AbstractEstimatorValueAlgorithm} = Union{<:Num_VecNum, <:MatNum,
                                                               <:PairStrNum,
                                                               <:MultiEstValType, T}
"""
    const Str_Expr = Union{<:AbstractString, Expr}

Alias for a union of abstract string or Julia expression.

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

Alias for a union of string, Julia expression, or vector of strings/expressions.

Every consumer of this alias parses the value with [`parse_equation`](@ref), so the alias holds equation text and nothing else. No branch of [`AbstractEstimatorValueAlgorithm`](@ref) is an `EqnType`: an algorithm computes a value over the universe, and a value carries neither a comparison operator nor a side, so no constraint row can be assembled from it. A slot that admits an algorithm is an [`EstValType`](@ref).

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

Alias for a union of an abstract vector of integers or an abstract vector of integer vectors.

# Related

  - [`VecInt`](@ref)
  - [`VecVecInt`](@ref)
"""
const VecInt_VecVecInt = Union{<:VecInt, <:VecVecInt}
"""
    const VecVecVecInt = AbstractVector{<:VecVecInt}

Alias for an abstract vector of abstract vector of integer vectors.

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

Alias for an optional value of type `T`, which may be `nothing`.

# Related

  - [`EstValType`](@ref)
"""
const Option{T} = Union{Nothing, T}
"""
    const MatNum_VecMatNum = Union{<:MatNum, <:VecMatNum}

Alias for a union of a numeric matrix or a vector of numeric matrices.

# Related

  - [`MatNum`](@ref)
  - [`VecMatNum`](@ref)
"""
const MatNum_VecMatNum = Union{<:MatNum, <:VecMatNum}
"""
    const Int_VecInt = Union{<:Integer, <:VecInt}

Alias for a union of an integer or a vector of integers.

# Related

  - [`VecInt`](@ref)
"""
const Int_VecInt = Union{<:Integer, <:VecInt}
"""
    const VecNum_VecVecNum = Union{<:VecNum, <:VecVecNum}

Alias for a union of a numeric vector or a vector of numeric vectors.

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

Alias for a union of an abstract dictionary or an abstract vector.

# Related

  - [`DictStrNum`](@ref)
  - [`VecNum`](@ref)
"""
const Dict_Vec = Union{<:AbstractDict, <:AbstractVector}
"""
    const Sym_Str = Union{Symbol, <:AbstractString}

Alias for a union of a symbol or an abstract string.

# Related

  - [`VecStr`](@ref)
"""
const Sym_Str = Union{Symbol, <:AbstractString}
"""
    const Str_Vec = Union{<:AbstractString, <:AbstractVector}

Alias for a union of an abstract string or an abstract vector.

# Related

  - [`VecStr`](@ref)
  - [`Str_Expr`](@ref)
"""
const Str_Vec = Union{<:AbstractString, <:AbstractVector}
