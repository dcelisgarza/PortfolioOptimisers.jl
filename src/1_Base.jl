"""
    AbstractEstimator

Abstract supertype for all estimator types in PortfolioOptimisers.jl.

All custom estimators (e.g., for moments, risk, or priors) should subtype `AbstractEstimator`.
This enables a consistent interface for estimation routines throughout the package.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractEstimator end

"""
    AbstractAlgorithm

Abstract supertype for all algorithm types in PortfolioOptimisers.jl.

All algorithms (e.g., solvers, metaheuristics) should subtype `AbstractAlgorithm`.
This allows for flexible extension and dispatch of routines.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractAlgorithm end

"""
    AbstractResult

Abstract supertype for all result types returned by optimizers in PortfolioOptimisers.jl.

All result objects (e.g., optimization outputs, solution summaries) should subtype `AbstractResult`.
This ensures a unified interface for accessing results across different estimators and algorithms.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
"""
abstract type AbstractResult end

"""
    AbstractCovarianceEstimator <: StatsBase.CovarianceEstimator

Abstract supertype for all covariance estimator types in PortfolioOptimisers.jl.

All concrete types that implement covariance estimation (e.g., sample covariance, shrinkage estimators) should subtype `AbstractCovarianceEstimator`. This enables a consistent interface for covariance estimation routines throughout the package.

# Related

  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/)
  - [`AbstractMomentAlgorithm`](@ref)
"""
abstract type AbstractCovarianceEstimator <: StatsBase.CovarianceEstimator end

"""
    AbstractVarianceEstimator <: AbstractCovarianceEstimator

Abstract supertype for all variance estimator types in PortfolioOptimisers.jl.

All concrete types that implement variance estimation (e.g., sample variance, robust variance estimators) should subtype `AbstractVarianceEstimator`. This enables a consistent interface for variance estimation routines and allows for flexible extension and dispatch within the package.

# Related

  - [`AbstractCovarianceEstimator`](@ref)
"""
abstract type AbstractVarianceEstimator <: AbstractCovarianceEstimator end
function Base.show(
    io::IO,
    ear::Union{
        <:AbstractEstimator,
        <:AbstractAlgorithm,
        <:AbstractResult,
        <:AbstractCovarianceEstimator,
    },
)
    name = string(typeof(ear))
    fields = propertynames(ear)
    if isempty(fields)
        return println(io, string(typeof(ear), "()"))
    end
    name = name[1:(findfirst(x->(x=='{'||x=='('), name)-1)]
    println(io, name)
    padding = maximum(map(length, map(string, fields))) + 2
    for field in fields
        val = try
            getfield(ear, field)
        catch
            continue
        end
        print(io, lpad(string(field), padding), " ")
        if isnothing(val)
            println(io, "| nothing")
        elseif isa(val, AbstractMatrix)
            println(io, "| $(size(val,1))×$(size(val,2)) $(typeof(val))")
        elseif isa(val, AbstractVector) && length(val) > 6
            println(io, "| $(length(val))-element $(typeof(val))")
        elseif isa(
            val,
            Union{
                <:AbstractEstimator,
                <:AbstractAlgorithm,
                <:AbstractResult,
                <:AbstractCovarianceEstimator,
                <:JuMP.Model,
                <:Clustering.Hclust,
            },
        )
            ioalg = IOBuffer()
            show(ioalg, val)
            algstr = String(take!(ioalg))
            alglines = split(algstr, '\n')
            println(io, "| ", alglines[1])
            for l in alglines[2:end]
                if isempty(l) || l == '\n'
                    continue
                end
                println(io, lpad("| ", padding + 3), l)
            end
        elseif isa(val, DataType)
            tval = typeof(val)
            val = repr(val)
            if !isnothing(match(r"[\(\{]", val))
                val = val[1:(findfirst(x->(x=='{'||x=='('), val)-1)]
            end
            println(io, "| $(tval): ", val)
        else
            println(io, "| $(typeof(val)): ", repr(val))
        end
    end
    return nothing
end
function mul_cond_msg(conds::AbstractString...)
    N = isa(conds, Tuple) ? length(conds) : 1
    msg = "the following conditions must hold:\n"
    for (i, val) in enumerate(conds)
        mi = i == N ? "$val." : "$val.\n"
        msg *= mi
    end
    return msg
end
abstract type PortfolioOptimisersError <: Exception end
struct IsNothingError{T1} <: PortfolioOptimisersError
    msg::T1
end
struct IsEmptyError{T1} <: PortfolioOptimisersError
    msg::T1
end
struct IsNothingEmptyError{T1} <: PortfolioOptimisersError
    msg::T1
end
struct IsNonFiniteError{T1} <: PortfolioOptimisersError
    msg::T1
end
function Base.showerror(io::IO, err::PortfolioOptimisersError)
    name = string(typeof(err))
    name = name[1:(findfirst(x->(x=='{'||x=='('), name)-1)]
    return print(io, "$name: $(err.msg)")
end
function non_finite_msg(a)
    return "$a must finite"
end
function non_zero_msg(a, va = nothing)
    return "$a$(!isnothing(va) ? " ($va)" : "") must be non-zero"
end
function non_neg_msg(a, va = nothing)
    return "$a$(!isnothing(va) ? " ($va)" : "") must be non-negative"
end
function non_pos_msg(a, va = nothing)
    return "$a$(!isnothing(va) ? " ($va)" : "") must be non-positive"
end
function some_msg(a, va = nothing)
    return "$a (isnothing($a) => $(isnothing(va))) must not be `nothing`"
end
function non_empty_msg(a, va = nothing)
    return "$a$(!isnothing(va) ? " (isempty($a) => $(isempty(va)))" : "") must be non-empty"
end
function nothing_non_empty_msg(a, va = nothing)
    return "$a (isnothing($a) => $(isnothing(va))) must not be `nothing`, and non-empty$(!isnothing(va) ? " (isempty($a) => $(isempty(va)))" : "")"
end
function range_msg(a, b, c, va = nothing, bi::Bool = false, ci::Bool = false)
    return "$a$(!isnothing(va) ? " ($va)" : "") must be in $(bi ? '[' : '(')$b, $c$(ci ? ']' : ')')"
end
function comp_msg(a, b, c = :eq, va = nothing, vb = nothing)
    msg = (;
        :eq => "must be equal to",
        :gt => "must be greater than",
        :lt => "must be smaller than",
        :geq => "must be greater than or equal to",
        :leq => "must be smaller than or equal to",
    )
    return "$a$(!isnothing(va) ? " ($va)" : "") $(msg[c]) $b$(!isnothing(vb) ? " ($vb)" : "")"
end

export IsEmptyError, IsNothingError, IsNothingEmptyError
