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
#=
function Base.show(io::IO,
                   ear::Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult,
                              <:AbstractCovarianceEstimator})
    fields = propertynames(ear)
    if isempty(fields)
        return print(io, string(typeof(ear), "()"),'\n')
    end
    name = Base.typename(typeof(ear)).wrapper
    print(io, name,'\n')
    padding = maximum(map(length, map(string, fields))) + 2
    for field in fields
        if hasproperty(ear, field)
            val = getproperty(ear, field)
        else
            continue
        end
        print(io, lpad(string(field), padding), " ")
        if isnothing(val)
            print(io, "| nothing",'\n')
        elseif isa(val, NumMat)
            print(io, "| $(size(val,1))×$(size(val,2)) $(typeof(val))",'\n')
        elseif isa(val, NumVec) && length(val) > 6
            print(io, "| $(length(val))-element $(typeof(val))",'\n')
        elseif isa(val,
                   Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult,
                         <:AbstractCovarianceEstimator, <:JuMP.Model, <:Clustering.Hclust})
            ioalg = IOBuffer()
            print(ioalg, val,'\n')
            algstr = String(take!(ioalg))
            alglines = split(algstr, '\n')
            print(io, "| ", alglines[1],'\n')
            for l in alglines[2:end]
                if isempty(l) || l == '\n'
                    continue
                end
                print(io, lpad("| ", padding + 3), l,'\n')
            end
        elseif isa(val, DataType)
            tval = typeof(val)
            val = Base.typename(tval).wrapper
            print(io, "| $(tval): ", val,'\n')
        else
            print(io, "| $(typeof(val)): ", repr(val),'\n')
        end
    end
    return nothing
end
=#
#=
function Base.show(io::IO,
                   ear::Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult,
                              <:AbstractCovarianceEstimator})
    fields = propertynames(ear)
    if isempty(fields)
        return print(io, string(typeof(ear), "()"), '\n')
    end
    if get(io, :compact, false)
        return print(io, string(typeof(ear), "{...}(...)"), '\n')
    end
    custom_type = Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult,
                        <:AbstractCovarianceEstimator, <:JuMP.Model, <:Clustering.Hclust}
    name = Base.typename(typeof(ear)).wrapper
    print(io, name, '\n')
    padding = maximum(map(length, map(string, fields))) + 2
    for (i, field) in enumerate(fields)
        if hasproperty(ear, field)
            val = getproperty(ear, field)
        else
            continue
        end
        flag = isa(val, custom_type)
        sym1 = ifelse(i == length(fields) && (!flag || flag && isempty(propertynames(val))),
                      '┴', '┼')#┴ ┼ └ ├
        # sym1 = ifelse(i == length(fields), '┴', '┼')
        print(io, lpad(string(field), padding), " ")
        if isnothing(val)
            print(io, "$(sym1) nothing", '\n')
        elseif flag || isa(val, AbstractVector{<:custom_type}) && length(val) <= 6
            ioalg = IOBuffer()
            show(ioalg, val)
            algstr = String(take!(ioalg))
            alglines = split(algstr, '\n')
            print(io, "$(sym1) ", alglines[1], '\n')
            for l in alglines[2:end]
                if isempty(l) || l == '\n'
                    continue
                end
                # sym2 = ifelse(i == length(fields), ' ', '│')
                sym2 = '│'
                print(io, lpad("$sym2 ", padding + 3), l, '\n')
            end
        elseif isa(val, NumMat)
            print(io, "$(sym1) $(size(val,1))×$(size(val,2)) $(typeof(val))", '\n')
        elseif isa(val, NumVec) && length(val) > 6
            print(io, "$(sym1) $(length(val))-element $(typeof(val))", '\n')
        elseif isa(val, DataType)
            tval = typeof(val)
            val = Base.typename(tval).wrapper
            print(io, "$(sym1) $(tval): ", val, '\n')
        else
            print(io, "$(sym1) $(typeof(val)): ", repr(val), '\n')
        end
    end
    return nothing
end
=#
has_pretty_show_method(::Any) = false
has_pretty_show_method(::JuMP.Model) = true
has_pretty_show_method(::Clustering.Hclust) = true
function has_pretty_show_method(::Union{<:AbstractEstimator, <:AbstractAlgorithm,
                                        <:AbstractResult, <:AbstractCovarianceEstimator})
    return true
end
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
@define_pretty_show(Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult,
                          <:AbstractCovarianceEstimator})
"""
    abstract type PortfolioOptimisersError <: Exception end

Abstract supertype for all custom exception types in PortfolioOptimisers.jl.

All error types specific to PortfolioOptimisers.jl should subtype `PortfolioOptimisersError`. This enables consistent error handling and dispatch for package-specific exceptions.

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

Error type for `nothing` values in PortfolioOptimisers.jl.

`IsNothingError` is thrown when an argument or value required by an estimator, algorithm, or result is `nothing`. This enables consistent error handling for nothing data or configuration throughout the package.

# Fields

  - `msg`: Error message describing the nothing value or context.

# Constructors

    IsNothingError(msg)

Argument names correspond to the fields above.

# Examples

```jldoctest
julia> throw(IsNothingError("Input data is nothing"))
ERROR: IsNothingError: Input data is nothing
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related Types

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

Error type for empty values in PortfolioOptimisers.jl.

`IsEmptyError` is thrown when an argument or value required by an estimator, algorithm, or result is empty (e.g., an empty array, dictionary, or missing data structure). This enables consistent error handling for cases where required data is present but contains no elements.

# Fields

  - `msg`: Error message describing the empty value or context.

# Constructors

    IsEmptyError(msg)

Argument names correspond to the fields above.

# Examples

```jldoctest
julia> throw(IsEmptyError("Input array is empty"))
ERROR: IsEmptyError: Input array is empty
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related Types

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

Error type for non-finite values in PortfolioOptimisers.jl.

`IsNonFiniteError` is thrown when an argument or value required by an estimator, algorithm, or result is not finite (e.g., contains `Inf`, `-Inf`, or `NaN`). This enables consistent error handling for invalid numerical data throughout the package.

# Fields

  - `msg`: Error message describing the non-finite value or context.

# Constructors

    IsNonFiniteError(msg)

Argument names correspond to the fields above.

# Examples

```jldoctest
julia> throw(IsNonFiniteError("Input contains NaN"))
ERROR: IsNonFiniteError: Input contains NaN
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
    const NumVec = AbstractVector{<:Union{<:Number, <:AbstractJuMPScalar}}

Type alias for vectors of scalars in PortfolioOptimisers.jl.

`NumVec` is used throughout the package to represent vectors containing either numeric types or abstract JuMP scalars. This alias enables flexible and consistent handling of vector data in estimators, algorithms, and results.

# Details

  - Used for portfolio weights, returns, and other numeric vector data.
  - Supports both standard numeric types and JuMP scalar types for optimization.
  - Ensures type consistency across estimation and optimization routines.

# Related

  - [`NumMat`](@ref)
  - [`NumArr`](@ref)
  - [`IntVec`](@ref)
  - [`AbstractJuMPScalar`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.AbstractJuMPScalar)
"""
const NumVec = AbstractVector{<:Union{<:Number, <:AbstractJuMPScalar}}
"""
    const IntVec = AbstractVector{<:Integer}

Type alias for vectors of integers in PortfolioOptimisers.jl.

`IntVec` is used throughout the package to represent vectors containing integer values. This alias enables consistent handling of integer vector data in estimators, algorithms, and results.

# Details

  - Used for asset indices, counts, and other integer vector data.
  - Ensures type consistency across estimation and optimization routines.
  - Supports all subtypes of `Integer`.

# Related

  - [`NumVec`](@ref)
  - [`NumMat`](@ref)
  - [`NumArr`](@ref)
  - [`EstValType`](@ref)
"""
const IntVec = AbstractVector{<:Integer}
"""
    const NumMat = AbstractMatrix{<:Union{<:Number, <:AbstractJuMPScalar}}

Type alias for matrices of scalars in PortfolioOptimisers.jl.

`NumMat` is used throughout the package to represent matrices containing either numeric types or abstract JuMP scalars. This alias enables flexible and consistent handling of matrix data in estimators, algorithms, and results.

# Details

  - Used for covariance, correlation, and other matrix-valued data.
  - Supports both standard numeric types and JuMP scalar types for optimization.
  - Ensures type consistency across estimation and optimization routines.

# Related

  - [`NumVec`](@ref)
  - [`NumArr`](@ref)
  - [`IntVec`](@ref)
  - [`AbstractJuMPScalar`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.AbstractJuMPScalar)
"""
const NumMat = AbstractMatrix{<:Union{<:Number, <:AbstractJuMPScalar}}
"""
    const NumArr = AbstractArray{<:Union{<:Number, <:AbstractJuMPScalar}}

Type alias for arrays of scalars in PortfolioOptimisers.jl.

`NumArr` is used throughout the package to represent arrays containing either numeric types or abstract JuMP scalars. This alias enables flexible and consistent handling of array data in estimators, algorithms, and results.

# Details

  - Used in cases where functions may accept vectors or matrices.
  - Supports both standard numeric types and JuMP scalar types for optimization.
  - Ensures type consistency across estimation and optimization routines.

# Related

  - [`NumVec`](@ref)
  - [`NumMat`](@ref)
  - [`IntVec`](@ref)
  - [`AbstractJuMPScalar`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.AbstractJuMPScalar)
"""
const NumArr = AbstractArray{<:Union{<:Number, <:AbstractJuMPScalar}}
"""
    const EstValType = Union{<:Pair{<:AbstractString, <:Number},
                             <:AbstractVector{<:Pair{<:AbstractString, <:Number}},
                             <:AbstractDict{<:AbstractString, <:Number}}

Type alias for value types used in the `val` field of estimators in PortfolioOptimisers.jl.

`EstValType` is used to represent estimator values as a string-number pair, a vector of such pairs, or a dictionary mapping strings to numbers. This enables flexible and consistent handling of estimator outputs and configuration values.

# Details

  - Used whenever a value needs to be mapped to a name by a concrete subtype of [`AbstractEstimator`](@ref) to produce a concrete type of [`AbstractResult`](@ref).
  - Supports both single and multiple named values.
  - Ensures type consistency for estimator outputs and configuration.

# Related

  - [`NumVec`](@ref)
  - [`NumMat`](@ref)
  - [`NumArr`](@ref)
  - [`IntVec`](@ref)
  - [`AbstractEstimator`](@ref)
  - [`AbstractResult`](@ref)
"""
const EstValType = Union{<:Pair{<:AbstractString, <:Number},
                         <:AbstractVector{<:Pair{<:AbstractString, <:Number}},
                         <:AbstractDict{<:AbstractString, <:Number}}

export IsEmptyError, IsNothingError, IsNonFiniteError, NumVec, IntVec, NumMat, NumArr,
       EstValType
