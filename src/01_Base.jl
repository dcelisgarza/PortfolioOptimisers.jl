"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all estimator types in `PortfolioOptimisers.jl`.

All custom estimators should subtype `AbstractEstimator`.

Estimators consume data to estimate parameters or models. Some estimators may utilise different algorithms. These can range from simple implementation details that don't change the result much but may have different numerical characteristics, to entirely different methodologies or algorithms yielding different results.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all algorithm types in `PortfolioOptimisers.jl`.

All algorithms should subtype `AbstractAlgorithm`.

Algorithms are often used by estimators to perform specific tasks. These can be in the form of simple implementation details to entirely different procedures for estimating a quantity.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all result types in `PortfolioOptimisers.jl`.

All result objects should subtype `AbstractResult`.

Result types encapsulate the outcomes of estimators. This makes dispatch and usage more straightforward, especially when the results encapsulate a wide range of information.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
"""
abstract type AbstractResult end
"""
    define_pretty_show(T, flag::Bool = true)

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
macro define_pretty_show(T, flag::Bool = true)
    esc(quote
            if $flag
                has_pretty_show_method(::$T) = true
            end
            function Base.show(io::IO, obj::$T)
                fields = propertynames(obj)
                tobj = typeof(obj)
                if isempty(fields)
                    return print(io, string(tobj, "()"), '\n')
                end
                if get(io, :compact, false) || get(io, :multiline, false)
                    return print(io, string(tobj), '\n')
                end
                name = Base.typename(tobj).wrapper
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
                                  (!flag || (flag && isempty(propertynames(val)))), '┴',
                                  '┼')
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
                        print(io, "$(sym1) $(size(val,1))×$(size(val,2)) $(typeof(val))",
                              '\n')
                    elseif isa(val, AbstractVector) && length(val) > 6 ||
                           isa(val, AbstractVector{<:AbstractArray})
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
        end)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

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
@define_pretty_show(Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult})
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all custom exception types in `PortfolioOptimisers.jl`.

All error types specific to `PortfolioOptimisers.jl` should be subtypes of `PortfolioOptimisersError`.

# Related

  - [`IsNothingError`](@ref)
  - [`IsEmptyError`](@ref)
  - [`IsNonFiniteError`](@ref)
"""
abstract type PortfolioOptimisersError <: Exception end
"""
$(DocStringExtensions.TYPEDEF)

Exception type thrown when an argument or value is unexpectedly `nothing`.

# Fields

$(DocStringExtensions.FIELDS)

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
@concrete struct IsNothingError <: PortfolioOptimisersError
    "error message describing the condition that triggered the exception."
    msg
end
"""
$(DocStringExtensions.TYPEDEF)

Exception type thrown when an argument or value is unexpectedly empty.

# Fields

$(DocStringExtensions.FIELDS)

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
@concrete struct IsEmptyError <: PortfolioOptimisersError
    "error message describing the condition that triggered the exception."
    msg
end
"""
$(DocStringExtensions.TYPEDEF)

Exception type thrown when an argument or value is unexpectedly non-finite (e.g., contains `NaN` or `Inf`).

# Fields

$(DocStringExtensions.FIELDS)

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
@concrete struct IsNonFiniteError <: PortfolioOptimisersError
    "error message describing the condition that triggered the exception."
    msg
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
    return i == 1 ? obj : throw(BoundsError(obj, i))
end
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of numeric types or JuMP scalar types.

# Related

  - [`VecInt`](@ref)
  - [`MatNum`](@ref)
  - [`JuMP.AbstractJuMPScalar`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.JuMP.AbstractJuMPScalar)
"""
const VecNum = AbstractVector{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of integer types.

# Related

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
  - [`ArrNum`](@ref)
"""
const VecInt = AbstractVector{<:Integer}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract matrix of numeric types or JuMP scalar types.

# Related

  - [`VecNum`](@ref)
  - [`ArrNum`](@ref)
  - [`VecMatNum`](@ref)
"""
const MatNum = AbstractMatrix{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract array of numeric types or JuMP scalar types.

# Related

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
"""
const ArrNum = AbstractArray{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}
"""
    const VecNum_MatNum = Union{<:VecNum, <:MatNum}

Alias for a union of a numeric type or an abstract matrix of numeric types.

# Related

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
"""
const VecNum_MatNum = Union{<:VecNum, <:MatNum}
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
$(DocStringExtensions.TYPEDEF)

Alias for a pair consisting of an abstract string and a numeric type.

# Related

  - [`DictStrNum`](@ref)
  - [`MultiEstValType`](@ref)
"""
const PairStrNum = Pair{<:AbstractString, <:Number}
"""
$(DocStringExtensions.TYPEDEF)

Alias for a pair consisting of an abstract string and an abstract vector.

# Related

  - [`DictGSCV`](@ref)
  - [`MultiGSCVValType`](@ref)
"""
const PairGSCV = Pair{<:AbstractString, <:AbstractVector}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract dictionary with string keys and numeric values.

# Related

  - [`PairStrNum`](@ref)
  - [`MultiEstValType`](@ref)
"""
const DictStrNum = AbstractDict{<:AbstractString, <:Number}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract dictionary with string keys and abstract vector values.

# Related

  - [`PairGSCV`](@ref)
  - [`MultiGSCVValType`](@ref)
"""
const DictGSCV = AbstractDict{<:AbstractString, <:AbstractVector}
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
$(DocStringExtensions.TYPEDEF)

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

Abstract supertype for all estimator value algorithm types in `PortfolioOptimisers.jl`.

Subtypes of `AbstractEstimatorValueAlgorithm` implement algorithms for computing constraint result values. These are used to extend or modify the behavior of estimators in a composable and modular fashion.

# Related

  - [`EstValType`](@ref)
"""
abstract type AbstractEstimatorValueAlgorithm <: AbstractAlgorithm end
"""
    const EstValType = Union{<:Num_VecNum, <:PairStrNum, <:MultiEstValType,
                             <:AbstractEstimatorValueAlgorithm}

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
$(DocStringExtensions.TYPEDEF)

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
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of numeric vectors.

# Related

  - [`VecNum`](@ref)
  - [`VecMatNum`](@ref)
"""
const VecVecNum = AbstractVector{<:VecNum}
"""
$(DocStringExtensions.TYPEDEF)

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
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of abstract vector of integer vectors.

# Related

  - [`VecVecInt`](@ref)
"""
const VecVecVecInt = AbstractVector{<:VecVecInt}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of numeric matrices.

# Related

  - [`MatNum`](@ref)
  - [`VecNum`](@ref)
"""
const VecMatNum = AbstractVector{<:MatNum}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of strings.

# Related

  - [`Str_Expr`](@ref)
  - [`VecStr_Expr`](@ref)
"""
const VecStr = AbstractVector{<:AbstractString}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of pairs.

# Related

  - [`PairStrNum`](@ref)
"""
const VecPair = AbstractVector{<:Pair}
"""
$(DocStringExtensions.TYPEDEF)

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
$(DocStringExtensions.TYPEDEF)

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
$(DocStringExtensions.TYPEDEF)

Represents a composite result containing a vector and a scalar in `PortfolioOptimisers.jl`.

Encapsulates a vector and a scalar value, commonly used for storing results that combine both types of data (e.g., weighted statistics, risk measures).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    VecScalar(;
        v::VecNum,
        s::Number
    ) -> VecScalar

Keywords correspond to the struct's fields.

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
@concrete struct VecScalar <: AbstractResult
    "Vector component."
    v
    "Scalar component."
    s
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

"""
$(DocStringExtensions.TYPEDEF)

Singleton vector type that represents a vector with a single element with value equal to 1. Used for reducing matrix vector products to dropping the matrix's second dimension.

# Constructors

    SingletonVector()
"""
struct SingletonVector{T} <: AbstractVector{T} end
function SingletonVector()
    return SingletonVector{Int}()
end
Base.length(::SingletonVector) = 1
function Base.getindex(A::SingletonVector, i::Int)
    return isone(i) ? 1 : throw(BoundsError(A, i))
end
Base.:*(M::Matrix, ::SingletonVector) = dropdims(M; dims = 2)
Base.size(::SingletonVector) = (1,)
"""
    arg_dict = Dict(
                 # Weight vectors.
                 :pw => "`w`: Portfolio weights vector.",
                 :ow => "`w`: Observation weights vector.",
                 :oow => "`w`: Optional observation weights vector.",
                 # Matrix processing.
                 :pdm => "`pdm`: Positive definite matrix estimator.",
                 :dn => "`dn`: Matrix denoising estimator.",
                 :dt => "`dt`: Matrix detoning estimator.",
                 :mp => "`mp`: Matrix processing estimator.",
                 # Moments.
                 :me => "`me`: Expected returns estimator.",
                 :ce => "`ce`: Covariance estimator.",
                 :ve => "`ve`: Variance estimator.",
                 :ske => "`ske`: Coskewness estimator.",
                 :kte => "`kte`: Cokurtosis estimator.",
                 :de => "`de`: Distance matrix estimator.",
                 # Priors.
                 :pe => "`pe`: Prior estimator.",
                 :pr => "`pr`: Prior result.",
                 :per => "`pe`: Prior estimator or result.",
                 # Phylogeny.
                 :cle => "`cle`: Clusters estimator.",
                 :clr => "`clr`: Clusters result.",
                 :cler => "`cle`: Clusters estimator or result.",
                 :ple => "`pl`: Phylogeny estimator.",
                 :plr => "`pl`: Phylogeny result.",
                 :pler => "`pl`: Phylogeny estimator or result.",
                 :nte => "`pl`: Network estimator.",
                 :ntr => "`pl`: Network result.",
                 :nter => "`pl`: Network estimator or result.",
                 :cte => "`cte`: Centrality estimator.",
                 :cta => "`ct`: Centrality algorithm.",
                 :ctr => "`ct`: Centrality result.",
                 :cter => "`ct`: Centrality estimator or result.",
                 # Turnover.
                 :tne => "`tn`: Turnover estimator.",
                 :tnr => "`tn`: Turnover result.",
                 :tner => "`tn`: Turnover estimator or result.",
                 :tnes => "`tn`: Turnover estimator(s).",
                 :tnrs => "`tn`: Turnover result(s).",
                 :tners => "`tn`: Turnover estimator(s) or result(s).",
                 # Tracking.
                 :tre => "`tr`: Tracking error estimator.",
                 :trr => "`tr`: Tracking error result.",
                 :trer => "`tr`: Tracking error estimator or result.",
                 :tres => "`tr`: Tracking error estimator(s).",
                 :trrs => "`tr`: Tracking error result(s).",
                 :trers => "`tr`: Tracking error estimator(s) or result(s).",
                 # Weight bounds.
                 :wbe => "`wb`: Weight bounds estimator.",
                 :wbr => "`wb`: Weight bounds result.",
                 :wber => "`wb`: Weight bounds estimator or result.",
                 # Fees.
                 :feese => "`fees`: Fees estimator.",
                 :feesr => "`fees`: Fees result.",
                 :feeser => "`fees`: Fees estimator or result.")

This dictionary contains the arg_dict terms and their corresponding descriptions used in the documentation of `PortfolioOptimisers.jl`.
"""
const arg_dict = Dict(
                      # Weight vectors.
                      :pw => "`w`: Portfolio weights vector `assets × 1`.",
                      :ow => "`w`: Observation weights vector `observations × 1`.",
                      :oow => "`w`: Optional observation weights vector `observations × 1`. If `nothing`, the computation is unweighted.",
                      # Matrix processing.
                      :pdm => "`pdm`: Positive definite matrix estimator.",
                      :opdm => "`pdm`: Optional positive definite matrix estimator.",
                      :dn => "`dn`: Matrix denoising estimator.",
                      :odn => "`dn`: Optional matrix denoising estimator.",
                      :dna => "`dna`: Matrix denoising algorithm.",
                      :dt => "`dt`: Matrix detoning estimator.",
                      :odt => "`dt`: Optional matrix detoning estimator.",
                      :mp => "`mp`: Matrix processing estimator.",
                      :omp => "`mp`: Optional matrix processing estimator.",
                      :mpa => "`mpa`: Matrix processing algorithm.",
                      # Moments.
                      :me => "`me`: Expected returns estimator.",
                      :ome => "`me`: Optional expected returns estimator. It is not needed when used on a vector, if `nothing` and used on a matrix defaults to [`SimpleExpectedReturns`](@ref).",
                      :ce => "`ce`: Covariance estimator.",#
                      :ve => "`ve`: Variance estimator.",#
                      :ske => "`ske`: Coskewness estimator.",
                      :kte => "`kte`: Cokurtosis estimator.",
                      :de => "`de`: Distance matrix estimator.",
                      :oidx => "`oidx`: Optional indices of the observations to use for estimation `Y × 1` where `Y <= observations`. If `nothing`, all observations are used.",
                      :malg => "`alg`: Moment algorithm.",
                      :corrected => "`corrected`: Whether to apply Bessel's correction.",
                      ## Gerber
                      :gerbt => "`t`: Threshold value.",
                      :gerbalg => "`alg`: Gerber covariance algorithm.",
                      :gerbce => "`ce`: Gerber covariance estimator.",
                      :stdarr => "`sd`: Standard deviation vector of `X`, shaped to be consistent with `X`.",
                      # Priors.
                      :pe => "`pe`: Prior estimator.",#
                      :pr => "`pr`: Prior result.",#
                      :per => "`pr`: Prior estimator or result.",
                      # Phylogeny.
                      :cle => "`cle`: Clusters estimator.",#
                      :clr => "`clr`: Clusters result.",#
                      :cler => "`clr`: Clusters estimator or result.",#
                      :ple => "`ple`: Phylogeny estimator.",#
                      :plr => "`plr`: Phylogeny result.",
                      :pler => "`pl`: Phylogeny estimator or result.",
                      :nte => "`nte`: Network estimator.",#
                      :ntr => "`pl`: Network result.",
                      :nter => "`pl`: Network estimator or result.",
                      :cte => "`cte`: Centrality estimator.",#
                      :cta => "`ct`: Centrality algorithm.",
                      :ctr => "`ct`: Centrality result.",
                      :cter => "`ct`: Centrality estimator or result.",
                      # Turnover.
                      :tne => "`tn`: Turnover estimator.",#
                      :tnr => "`tn`: Turnover result.",
                      :tner => "`tn`: Turnover estimator or result.",
                      :tnes => "`tn`: Turnover estimator(s).",
                      :tnrs => "`tn`: Turnover result(s).",
                      :tners => "`tn`: Turnover estimator(s) or result(s).",
                      # Tracking.
                      :tre => "`tr`: Tracking error estimator.",
                      :trr => "`tr`: Tracking error result.",
                      :trer => "`tr`: Tracking error estimator or result.",
                      :tres => "`tr`: Tracking error estimator(s).",
                      :trrs => "`tr`: Tracking error result(s).",
                      :trers => "`tr`: Tracking error estimator(s) or result(s).",
                      # Weight bounds.
                      :wbe => "`wb`: Weight bounds estimator.",
                      :wbr => "`wb`: Weight bounds result.",
                      :wber => "`wb`: Weight bounds estimator or result.",
                      # Fees.
                      :feese => "`fees`: Fees estimator.",#
                      :feesr => "`fees`: Fees result.",
                      :feeser => "`fees`: Fees estimator or result.",
                      # Stats.
                      :sigma => "`sigma`: Covariance matrix `features × features`.",#
                      :mu => "`mu`: Expected returns vector `features × 1`.",#
                      :rho => "`rho`: Correlation matrix `features × features`.",
                      :sigrho => "`sigma`: Covariance-like or correlation-like matrix `features × features`.",
                      :sigrhoX => "`X`: Covariance-like or correlation-like matrix `features × features`.",
                      :kt => "`kt`: Cokurtosis matrix `features^2 × features^2`.",#
                      :sk => "`sk`: Coskewness matrix `features × features^2`.",#
                      :V => "`V`: Sum of the negative spectral slices of the cokurtosis matrix `features × features`.",
                      :X => "`X`: Data matrix `observations × features` if the `dims` keyword does not exist or `dims = 1`, `features × observations` when `dims = 2`.",#
                      :F => "`F`: Data matrix `observations × factors` if the `dims` keyword does not exist or `dims = 1`, `factors × observations` when `dims = 2`.",#
                      :Xv => "`X`: Data vector `observations × 1`.",#
                      :dims => "`dims`: Dimension along which to perform the computation.",#
                      :omean => "`mean`: Optional mean value to use for centering.",
                      :stdvec => "`std_vec`: Vector of standard deviations for each asset, used to scale the threshold.")
const field_dict = Dict(key => strip(val[(findfirst(":", val)[1] + 1):end])
                        for (key, val) in arg_dict)
"""
    val_dict = Dict(:oow => "If `w` is not `nothing`, `!isempty(w)`.")

Validation rules for certain arg_dict terms used in the documentation of `PortfolioOptimisers.jl`.
"""
val_dict = Dict(:oow => "If `w` is not `nothing`, `!isempty(w)`.",
                :oidx => "If `idx` is not `nothing`, `!isempty(idx)` and all indices are positive integers.",
                :gerbt => "`0 < t < 1`.",#
                :dims => "`dims in (1, 2)`.")

"""
Dictionary containing return value descriptions for common parameters used in `PortfolioOptimisers.jl`.
"""
ret_dict = Dict(:mu => "`mu::ArrNum`: Expected returns vector `features x 1` if the `dims` keyword does not exist or `dims = 2`, `1 x features` if `dims = 1`.",#
                :sigma => "`sigma::MatNum`: Covariance matrix `features x features`.",#
                :rho => "`rho::MatNum`: Correlation matrix `features x features`.",#
                :sigrho => "`sigrho::MatNum`: Covariance/correlation matrix `features x features`.",#
                :sk => "`sk::MatNum`: Coskewness matrix `features x features`.",#
                :kt => "`kt::MatNum`: Cokurtosis matrix `features x features`.",#
                :me => "`me`: New expected returns estimator of the same type as the argument, with the appropriate weights applied.",#
                :ce => "`ce`: New covariance estimator of the same type as the argument, with the new weights applied.",#
                :ve => "`ve`: New variance estimator of the same type as the argument, with the new weights applied.",
                :stdvar => "`res::ArrNum`: Variance or standard deviation vector of `X`, reshaped to be consistent with the dimension along which the value is computed.",#
                :stdvarnum => "`res::Number`: Variance or standard deviation `X`",#
                :stdarr => "`sd::ArrNum`: Standard deviation vector of `X`, reshaped to be consistent with the dimension along which the value is computed.",
                :vararr => "`vr::ArrNum`: Variance vector of `X`, reshaped to be consistent with the dimension along which the value is computed.",
                :stdnum => "`vr::Number`: Standard deviation of `X`",
                :varnum => "`vr::Number`: Variance of `X`",
                :algw => "`alg`: New algorithm instance of the same type as the argument, with the new weights applied.",
                :alg => "`alg`: The original algorithm instance.")
math_dict = Dict(:Xv => "``\\boldsymbol{X}``: Data vector `observations × 1`.",#
                 :tgt => "``\\boldsymbol{t}``: Target value, usually the unweighted (or weighted) expected value ``E[\\boldsymbol{X}]``.")

export IsEmptyError, IsNothingError, IsNonFiniteError, VecScalar
