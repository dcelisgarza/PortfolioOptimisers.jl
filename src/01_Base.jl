"""
    abstract type AbstractEstimator end

Abstract supertype for all estimator types in PortfolioOptimisers.jl.

All custom estimators (e.g., for moments, risk, or priors) should subtype `AbstractEstimator`.
This enables a consistent interface for estimation routines throughout the package.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractEstimator end
"""
    abstract type AbstractAlgorithm end

Abstract supertype for all algorithm types in PortfolioOptimisers.jl.

All algorithms (e.g., solvers, metaheuristics) should subtype `AbstractAlgorithm`.
This allows for flexible extension and dispatch of routines.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractAlgorithm end
"""
    abstract type AbstractResult end

Abstract supertype for all result types returned by optimizers in PortfolioOptimisers.jl.

All result objects (e.g., optimization outputs, solution summaries) should subtype `AbstractResult`.
This ensures a unified interface for accessing results across different estimators and algorithms.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
"""
abstract type AbstractResult end
"""
    abstract type AbstractCovarianceEstimator <: StatsBase.CovarianceEstimator end

Abstract supertype for all covariance estimator types in PortfolioOptimisers.jl.

All concrete types that implement covariance estimation (e.g., sample covariance, shrinkage estimators) should subtype `AbstractCovarianceEstimator`. This enables a consistent interface for covariance estimation routines throughout the package.

# Related

  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/)
  - [`AbstractMomentAlgorithm`](@ref)
"""
abstract type AbstractCovarianceEstimator <: StatsBase.CovarianceEstimator end
"""
    abstract type AbstractVarianceEstimator <: AbstractCovarianceEstimator end

Abstract supertype for all variance estimator types in PortfolioOptimisers.jl.

All concrete types that implement variance estimation (e.g., sample variance, robust variance estimators) should subtype `AbstractVarianceEstimator`. This enables a consistent interface for variance estimation routines and allows for flexible extension and dispatch within the package.

# Related

  - [`AbstractCovarianceEstimator`](@ref)
"""
abstract type AbstractVarianceEstimator <: AbstractCovarianceEstimator end
"""
    @define_pretty_show(T)

Macro to define a custom pretty-printing `Base.show` method for types in PortfolioOptimisers.jl.

This macro generates a `show` method that displays the type name and all fields in a readable, aligned format. For fields that are themselves custom types or collections, the macro recursively applies pretty-printing for nested structures. Handles compact and multiline IO contexts gracefully.

# Arguments

  - `T`: The type for which to define the pretty-printing method.

# Returns

  - Defines a `Base.show(io::IO, obj::T)` method for the given type.

# Details

  - Prints the type name and all fields with aligned labels.
  - Recursively pretty-prints nested custom types and collections.
  - Handles compact and multiline IO contexts.
  - Displays matrix/vector fields with their size and type.
  - Skips fields that are not present or are `nothing`.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`Base.show`](https://docs.julialang.org/en/v1/base/io/#Base.show)
"""
macro define_pretty_show(T)
    quote
        function Base.show(io::IO, obj::$(esc(T)))
            fields = propertynames(obj)
            if isempty(fields)
                return print(io, string(typeof(obj), "()"), '\n')
            end
            if get(io, :compact, false) || get(io, :multiline, false)
                return print(io, string(typeof(obj)), '\n')
            end
            name = Base.typename(typeof(obj)).wrapper
            print(io, name, '\n')
            padding = maximum(map(length, map(string, fields))) + 2
            for (i, field) in enumerate(fields)
                if hasproperty(obj, field)
                    val = getproperty(obj, field)
                else
                    continue
                end
                flag = has_pretty_show_method(val)
                sym1 = ifelse(i == length(fields) &&
                              (!flag || (flag && isempty(propertynames(val)))), '┴', '┼')
                print(io, lpad(string(field), padding), " ")
                if isnothing(val)
                    print(io, "$(sym1) nothing", '\n')
                elseif flag || (isa(val, AbstractVector) &&
                                length(val) <= 6 &&
                                all(has_pretty_show_method, val))
                    ioalg = IOBuffer()
                    show(ioalg, val)
                    algstr = String(take!(ioalg))
                    alglines = split(algstr, '\n')
                    print(io, "$(sym1) ", alglines[1], '\n')
                    for l in alglines[2:end]
                        if isempty(l) || l == '\n'
                            continue
                        end
                        sym2 = '│'
                        print(io, lpad("$sym2 ", padding + 3), l, '\n')
                    end
                elseif isa(val, AbstractMatrix)
                    print(io, "$(sym1) $(size(val,1))×$(size(val,2)) $(typeof(val))", '\n')
                elseif isa(val, AbstractVector) && length(val) > 6
                    print(io, "$(sym1) $(length(val))-element $(typeof(val))", '\n')
                elseif isa(val, DataType)
                    tval = typeof(val)
                    valstr = Base.typename(tval).wrapper
                    print(io, "$(sym1) $(tval): ", valstr, '\n')
                else
                    print(io, "$(sym1) $(typeof(val)): ", repr(val), '\n')
                end
            end
            return nothing
        end
    end
end
has_pretty_show_method(::Any) = false
has_pretty_show_method(::JuMP.Model) = true
has_pretty_show_method(::Clustering.Hclust) = true
function has_pretty_show_method(::Union{<:AbstractEstimator, <:AbstractAlgorithm,
                                        <:AbstractResult, <:AbstractCovarianceEstimator})
    return true
end
@define_pretty_show(Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult,
                          <:AbstractCovarianceEstimator})
"""
    abstract type PortfolioOptimisersError <: Exception end

Abstract supertype for all custom exception types in PortfolioOptimisers.jl.

All error types specific to PortfolioOptimisers.jl should subtype `PortfolioOptimisersError`. This enables consistent error handling and dispatch throughout the package.

# Related Types

  - [`IsNothingError`](@ref)
  - [`IsEmptyError`](@ref)
  - [`IsNonFiniteError`](@ref)
"""
abstract type PortfolioOptimisersError <: Exception end
"""
    struct IsNothingError{T1} <: PortfolioOptimisersError
        msg::T1
    end

Exception type thrown when an argument or value is unexpectedly `nothing`.

# Fields

  - `msg`: Error message describing the condition that triggered the exception.

# Constructors

    IsNothingError(msg)

Arguments correspond to the fields above.

# Examples

```jldoctest
julia> throw(IsNothingError("Input data must not be nothing"))
ERROR: IsNothingError: Input data must not be nothing
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`IsEmptyError`](@ref)
  - [`IsNonFiniteError`](@ref)
"""
struct IsNothingError{T1} <: PortfolioOptimisersError
    msg::T1
end
"""
    struct IsEmptyError{T1} <: PortfolioOptimisersError
        msg::T1
    end

Exception type thrown when an argument or value is unexpectedly empty.

# Fields

  - `msg`: Error message describing the condition that triggered the exception.

# Constructors

    IsEmptyError(msg)

Arguments correspond to the fields above.

# Examples

```jldoctest
julia> throw(IsEmptyError("Input array must not be empty"))
ERROR: IsEmptyError: Input array must not be empty
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`IsNothingError`](@ref)
  - [`IsNonFiniteError`](@ref)
"""
struct IsEmptyError{T1} <: PortfolioOptimisersError
    msg::T1
end
"""
    struct IsNonFiniteError{T1} <: PortfolioOptimisersError
        msg::T1
    end

Exception type thrown when an argument or value is unexpectedly non-finite (e.g., contains `NaN` or `Inf`).

# Fields

  - `msg`: Error message describing the condition that triggered the exception.

# Constructors

    IsNonFiniteError(msg)

Arguments correspond to the fields above.

# Examples

```jldoctest
julia> throw(IsNonFiniteError("Input array contains non-finite values"))
ERROR: IsNonFiniteError: Input array contains non-finite values
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`IsNothingError`](@ref)
  - [`IsEmptyError`](@ref)
"""
struct IsNonFiniteError{T1} <: PortfolioOptimisersError
    msg::T1
end
function Base.showerror(io::IO, err::PortfolioOptimisersError)
    name = string(typeof(err))
    name = name[1:(findfirst(x -> (x == '{' || x == '('), name) - 1)]
    return print(io, "$name: $(err.msg)")
end
function Base.iterate(obj::Union{<:AbstractEstimator, <:AbstractAlgorithm,
                                 <:AbstractResult}, state = 1)
    return state > 1 ? nothing : (obj, state + 1)
end
Base.length(::Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}) = 1
function Base.getindex(obj::Union{<:AbstractEstimator, <:AbstractAlgorithm,
                                  <:AbstractResult}, i::Int)
    return i == 1 ? obj : throw(BoundsError())
end
"""
    const VecNum = AbstractVector{<:Union{<:Number, <:AbstractJuMPScalar}}

Alias for an abstract vector of numeric types or JuMP scalar types.

# Related Types

  - [`VecInt`](@ref)
  - [`MatNum`](@ref)
  - [`AbstractJuMPScalar`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.AbstractJuMPScalar)
"""
const VecNum = AbstractVector{<:Union{<:Number, <:AbstractJuMPScalar}}
"""
    const VecInt = AbstractVector{<:Integer}

Alias for an abstract vector of integer types.

# Related Types

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
  - [`ArrNum`](@ref)
"""
const VecInt = AbstractVector{<:Integer}
"""
    const MatNum = AbstractMatrix{<:Union{<:Number, <:AbstractJuMPScalar}}

Alias for an abstract matrix of numeric types or JuMP scalar types.

# Related Types

  - [`VecNum`](@ref)
  - [`ArrNum`](@ref)
  - [`VecMatNum`](@ref)
"""
const MatNum = AbstractMatrix{<:Union{<:Number, <:AbstractJuMPScalar}}
"""
    const ArrNum = AbstractArray{<:Union{<:Number, <:AbstractJuMPScalar}}

Alias for an abstract array of numeric types or JuMP scalar types.

# Related Types

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
"""
const ArrNum = AbstractArray{<:Union{<:Number, <:AbstractJuMPScalar}}
"""
    const Num_VecNum = Union{<:Number, <:VecNum}

Alias for a union of a numeric type or an abstract vector of numeric types.

# Related Types

  - [`VecNum`](@ref)
  - [`ArrNum`](@ref)
"""
const Num_VecNum = Union{<:Number, <:VecNum}
"""
    const Num_ArrNum = Union{<:Number, <:ArrNum}

Alias for a union of a numeric type or an abstract array of numeric types.

# Related Types

  - [`ArrNum`](@ref)
  - [`VecNum`](@ref)
"""
const Num_ArrNum = Union{<:Number, <:ArrNum}
"""
    const PairStrNum = Pair{<:AbstractString, <:Number}

Alias for a pair consisting of an abstract string and a numeric type.

# Related Types

  - [`DictStrNum`](@ref)
  - [`MultiEstValType`](@ref)
"""
const PairStrNum = Pair{<:AbstractString, <:Number}
"""
    const DictStrNum = AbstractDict{<:AbstractString, <:Number}

Alias for an abstract dictionary with string keys and numeric values.

# Related Types

  - [`PairStrNum`](@ref)
  - [`MultiEstValType`](@ref)
"""
const DictStrNum = AbstractDict{<:AbstractString, <:Number}
"""
    const MultiEstValType = Union{<:DictStrNum, <:AbstractVector{<:PairStrNum}}

Alias for a union of a dictionary with string keys and numeric values, or a vector of string-number pairs.

# Related Types

  - [`DictStrNum`](@ref)
  - [`PairStrNum`](@ref)
  - [`EstValType`](@ref)
"""
const MultiEstValType = Union{<:DictStrNum, <:AbstractVector{<:PairStrNum}}
"""
    const EstValType = Union{<:Num_VecNum, <:PairStrNum, <:MultiEstValType}

Alias for a union of numeric, vector of numeric, string-number pair, or multi-estimator value types.

# Related Types

  - [`Num_VecNum`](@ref)
  - [`PairStrNum`](@ref)
  - [`MultiEstValType`](@ref)
"""
const EstValType = Union{<:Num_VecNum, <:PairStrNum, <:MultiEstValType}
"""
    const Str_Expr = Union{<:AbstractString, Expr}

Alias for a union of abstract string or Julia expression.

# Related Types

  - [`VecStr_Expr`](@ref)
  - [`EqnType`](@ref)
"""
const Str_Expr = Union{<:AbstractString, Expr}
"""
    const VecStr_Expr = AbstractVector{<:Str_Expr}

Alias for an abstract vector of strings or Julia expressions.

# Related Types

  - [`Str_Expr`](@ref)
  - [`EqnType`](@ref)
"""
const VecStr_Expr = AbstractVector{<:Str_Expr}
"""
    const EqnType = Union{<:AbstractString, Expr, <:VecStr_Expr}

Alias for a union of string, Julia expression, or vector of strings/expressions.

# Related Types

  - [`Str_Expr`](@ref)
  - [`VecStr_Expr`](@ref)
"""
const EqnType = Union{<:AbstractString, Expr, <:VecStr_Expr}
"""
    const VecVecNum = AbstractVector{<:VecNum}

Alias for an abstract vector of numeric vectors.

# Related Types

  - [`VecNum`](@ref)
  - [`VecMatNum`](@ref)
"""
const VecVecNum = AbstractVector{<:VecNum}
"""
    const VecVecInt = AbstractVector{<:VecInt}

Alias for an abstract vector of integer vectors.

# Related Types

  - [`VecInt`](@ref)
"""
const VecVecInt = AbstractVector{<:VecInt}
"""
    const VecMatNum = AbstractVector{<:MatNum}

Alias for an abstract vector of numeric matrices.

# Related Types

  - [`MatNum`](@ref)
  - [`VecNum`](@ref)
"""
const VecMatNum = AbstractVector{<:MatNum}
"""
    const VecStr = AbstractVector{<:AbstractString}

Alias for an abstract vector of strings.

# Related Types

  - [`Str_Expr`](@ref)
  - [`VecStr_Expr`](@ref)
"""
const VecStr = AbstractVector{<:AbstractString}
"""
    const VecPair = AbstractVector{<:Pair}

Alias for an abstract vector of pairs.

# Related Types

  - [`PairStrNum`](@ref)
"""
const VecPair = AbstractVector{<:Pair}
"""
    const VecJuMPScalar = AbstractVector{<:AbstractJuMPScalar}

Alias for an abstract vector of JuMP scalar types.

# Related Types

  - [`VecNum`](@ref)
"""
const VecJuMPScalar = AbstractVector{<:AbstractJuMPScalar}
"""
    const Option{T} = Union{Nothing, T}

Alias for an optional value of type `T`, which may be `nothing`.

# Related Types

  - [`EstValType`](@ref)
"""
const Option{T} = Union{Nothing, T}
"""
    const MatNum_VecMatNum = Union{<:MatNum, <:VecMatNum}

Alias for a union of a numeric matrix or a vector of numeric matrices.

# Related Types

  - [`MatNum`](@ref)
  - [`VecMatNum`](@ref)
"""
const MatNum_VecMatNum = Union{<:MatNum, <:VecMatNum}
"""
    const Int_VecInt = Union{<:Integer, <:VecInt}

Alias for a union of an integer or a vector of integers.

# Related Types

  - [`VecInt`](@ref)
"""
const Int_VecInt = Union{<:Integer, <:VecInt}
"""
    const VecNum_VecVecNum = Union{<:VecNum, <:VecVecNum}

Alias for a union of a numeric vector or a vector of numeric vectors.

# Related Types

  - [`VecNum`](@ref)
  - [`VecVecNum`](@ref)
"""
const VecNum_VecVecNum = Union{<:VecNum, <:VecVecNum}
"""
    const VecDate = AbstractVector{<:Dates.AbstractTime}

Alias for an abstract vector of date or time types.

# Related Types

  - [`VecNum`](@ref)
  - [`VecStr`](@ref)
"""
const VecDate = AbstractVector{<:Dates.AbstractTime}
"""
    const Dict_Vec = Union{<:AbstractDict, <:AbstractVector}

Alias for a union of an abstract dictionary or an abstract vector.

# Related Types

  - [`DictStrNum`](@ref)
  - [`VecNum`](@ref)
"""
const Dict_Vec = Union{<:AbstractDict, <:AbstractVector}
"""
    const Sym_Str = Union{Symbol, <:AbstractString}

Alias for a union of a symbol or an abstract string.

# Related Types

  - [`VecStr`](@ref)
"""
const Sym_Str = Union{Symbol, <:AbstractString}
"""
    const Str_Vec = Union{<:AbstractString, <:AbstractVector}

Alias for a union of an abstract string or an abstract vector.

# Related Types

  - [`VecStr`](@ref)
  - [`Str_Expr`](@ref)
"""
const Str_Vec = Union{<:AbstractString, <:AbstractVector}

export IsEmptyError, IsNothingError, IsNonFiniteError
