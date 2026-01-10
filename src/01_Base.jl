"""
    abstract type AbstractEstimator end

Abstract supertype for all estimator types in `PortfolioOptimisers.jl`.

All custom estimators (e.g., for moments, risk, or priors) should subtype `AbstractEstimator`.

This enables a consistent interface for estimation routines throughout the package.

Estimators consume data to estimate parameters or models. Some estimators may utilise different algorithms. These can range from simple implementation details that don't change the result much but may have different characteristics, to entirely different methodologies or algorithms yielding different results. Results are often encapsulated in result types, this simplifies dispatch and usage.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractEstimator end
"""
    abstract type AbstractAlgorithm end

Abstract supertype for all algorithm types in `PortfolioOptimisers.jl`.

All algorithms (e.g., solvers, metaheuristics) should subtype `AbstractAlgorithm`.

This allows for flexible extension and dispatch of routines.

Algorithms are often used by estimators to perform specific tasks. These can be in the form of simple implementation details or entirely different procedures.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractAlgorithm end
"""
    abstract type AbstractResult end

Abstract supertype for all result types returned by optimizers in `PortfolioOptimisers.jl`.

All result objects (e.g., optimization outputs, solution summaries) should subtype `AbstractResult`.

This ensures a unified interface for accessing results across different estimators and algorithms.

Result types encapsulate the outcomes of estimators. This makes dispatch and usage more straightforward, especially when the results encapsulate a variety of information.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
"""
abstract type AbstractResult end
"""
    @define_pretty_show(T)

Macro to define a custom pretty-printing `Base.show` method for types in `PortfolioOptimisers.jl`.

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
        function Base.show(io::IO, obj::$T)
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
"""
    has_pretty_show_method(::Any)

Default method indicating whether a type has a custom pretty-printing `show` method.

Overloading this method to return `true` indicates that type already has a custom pretty-printing method.

# Arguments

  - `::Any`: Any type.

# Returns

  - `flag::Bool`: `false` by default, indicating no custom pretty-printing method.

# Related

  - [`@define_pretty_show`](@ref)
"""
has_pretty_show_method(::Any) = false
has_pretty_show_method(::JuMP.Model) = true
has_pretty_show_method(::Clustering.Hclust) = true
has_pretty_show_method(::Clustering.KmeansResult) = true
function has_pretty_show_method(::Union{<:AbstractEstimator, <:AbstractAlgorithm,
                                        <:AbstractResult})
    return true
end
@define_pretty_show(Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult})
"""
    abstract type PortfolioOptimisersError <: Exception end

Abstract supertype for all custom exception types in `PortfolioOptimisers.jl`.

All error types specific to `PortfolioOptimisers.jl` should subtype `PortfolioOptimisersError`. This enables consistent error handling and dispatch throughout the package.

# Related

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
    const VecNum = Union{<:AbstractVector{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}}

Alias for an abstract vector of numeric types or JuMP scalar types.

# Related

  - [`VecInt`](@ref)
  - [`MatNum`](@ref)
  - [`JuMP.AbstractJuMPScalar`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.JuMP.AbstractJuMPScalar)
"""
const VecNum = Union{<:AbstractVector{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}}
"""
    const VecInt = Union{<:AbstractVector{<:Integer}}

Alias for an abstract vector of integer types.

# Related

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
  - [`ArrNum`](@ref)
"""
const VecInt = Union{<:AbstractVector{<:Integer}}
"""
    const MatNum = Union{<:AbstractMatrix{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}}

Alias for an abstract matrix of numeric types or JuMP scalar types.

# Related

  - [`VecNum`](@ref)
  - [`ArrNum`](@ref)
  - [`VecMatNum`](@ref)
"""
const MatNum = Union{<:AbstractMatrix{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}}
"""
    const ArrNum = Union{<:AbstractArray{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}}

Alias for an abstract array of numeric types or JuMP scalar types.

# Related

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
"""
const ArrNum = Union{<:AbstractArray{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}}
"""
    const Num_VecNum = Union{<:Number, <:VecNum}

Alias for a union of a numeric type or an abstract vector of numeric types.

# Related

  - [`VecNum`](@ref)
  - [`ArrNum`](@ref)
"""
const Num_VecNum = Union{<:Number, <:VecNum}
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
    const DictStrNum = AbstractDict{<:AbstractString, <:Number}

Alias for an abstract dictionary with string keys and numeric values.

# Related

  - [`PairStrNum`](@ref)
  - [`MultiEstValType`](@ref)
"""
const DictStrNum = AbstractDict{<:AbstractString, <:Number}
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
    abstract type AbstractEstimatorValueAlgorithm <: AbstractAlgorithm end

Abstract supertype for all estimator value algorithm types in `PortfolioOptimisers.jl`.

Subtypes of `AbstractEstimatorValueAlgorithm` implement algorithms for computing constraint result values. These are used to extend or modify the behavior of estimators in a composable and modular fashion.

# Related

  - [`EstValType`](@ref)
"""
abstract type AbstractEstimatorValueAlgorithm <: AbstractAlgorithm end
"""
    const EstValType = Union{<:Num_VecNum, <:PairStrNum, <:MultiEstValType, <:AbstractEstimatorValueAlgorithm}

Alias for a union of numeric, vector of numeric, string-number pair, or multi-estimator value types.

# Related

  - [`Num_VecNum`](@ref)
  - [`PairStrNum`](@ref)
  - [`MultiEstValType`](@ref)
  - [`AbstractEstimatorValueAlgorithm`](@ref)
"""
const EstValType = Union{<:Num_VecNum, <:PairStrNum, <:MultiEstValType,
                         <:AbstractEstimatorValueAlgorithm}
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

# Related

  - [`Str_Expr`](@ref)
  - [`VecStr_Expr`](@ref)
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
    const VecMatNum = AbstractVector{<:MatNum}

Alias for an abstract vector of numeric matrices.

# Related

  - [`MatNum`](@ref)
  - [`VecNum`](@ref)
"""
const VecMatNum = AbstractVector{<:MatNum}
"""
    const VecStr = Union{<:AbstractVector{<:AbstractString}}

Alias for an abstract vector of strings.

# Related

  - [`Str_Expr`](@ref)
  - [`VecStr_Expr`](@ref)
"""
const VecStr = Union{<:AbstractVector{<:AbstractString}}
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
"""
    struct VecScalar{T1, T2} <: AbstractResult
        v::T1
        s::T2
    end

Represents a composite result containing a vector and a scalar in `PortfolioOptimisers.jl`.

Encapsulates a vector and a scalar value, commonly used for storing results that combine both types of data (e.g., weighted statistics, risk measures).

# Fields

  - `v`: Vector value.
  - `s`: Scalar value.

# Constructors

    VecScalar(; v::VecNum, s::Number)

Keyword arguments correspond to the fields above.

## Validation

  - `v`: `!isempty(v)` and `all(isfinite, v)`.
  - `s`: `isfinite(s)`.

# Examples

```jldoctest
julia> VecScalar([1.0, 2.0, 3.0], 4.2)
VecScalar
  v ┼ Vector{Float64}: [1.0, 2.0, 3.0]
  s ┴ Float64: 4.2
```

# Related

  - [`AbstractResult`](@ref)
  - [`VecNum`](@ref)
"""
struct VecScalar{T1, T2} <: AbstractResult
    v::T1
    s::T2
    function VecScalar(v::VecNum, s::Number)
        assert_nonempty_finite_val(v, :v)
        assert_nonempty_finite_val(s, :s)
        return new{typeof(v), typeof(s)}(v, s)
    end
end
function VecScalar(; v::VecNum, s::Number)
    return VecScalar(v, s)
end
"""
    const Num_VecNum_VecScalar = Union{<:Num_VecNum, <:VecScalar}

Alias for a union of a numeric type, a vector of numeric types, or a `VecScalar` result.

# Related

  - [`Num_VecNum`](@ref)
  - [`VecScalar`](@ref)
"""
const Num_VecNum_VecScalar = Union{<:Num_VecNum, <:VecScalar}
"""
    const Num_ArrNum_VecScalar = Union{<:Num_ArrNum, <:VecScalar}

Alias for a union of a numeric type, an array of numeric types, or a `VecScalar` result.

# Related

  - [`Num_ArrNum`](@ref)
  - [`VecScalar`](@ref)
"""
const Num_ArrNum_VecScalar = Union{<:Num_ArrNum, <:VecScalar}

export IsEmptyError, IsNothingError, IsNonFiniteError, VecScalar
