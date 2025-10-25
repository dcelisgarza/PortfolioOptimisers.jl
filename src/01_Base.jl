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
        elseif isa(val, AbstractMatrix)
            print(io, "| $(size(val,1))×$(size(val,2)) $(typeof(val))",'\n')
        elseif isa(val, AbstractVector) && length(val) > 6
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
function Base.show(io::IO,
                   ear::Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult,
                              <:AbstractCovarianceEstimator})
    fields = propertynames(ear)
    if isempty(fields)
        return print(io, string(typeof(ear), "()"), '\n')
    end
    name = Base.typename(typeof(ear)).wrapper
    print(io, name, '\n')
    padding = maximum(map(length, map(string, fields))) + 2
    for (i, field) in enumerate(fields)
        if hasproperty(ear, field)
            val = getproperty(ear, field)
        else
            continue
        end
        flag = isa(val,
                   Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult,
                         <:AbstractCovarianceEstimator, <:JuMP.Model, <:Clustering.Hclust})
        sym1 = ifelse(i == length(fields) && (!flag || flag && isempty(propertynames(val))),
                      '└', '├')#┴ ┼ └ ├
        # sym1 = ifelse(i == length(fields), '└', '├')
        print(io, lpad(string(field), padding), " ")
        if isnothing(val)
            print(io, "$(sym1) nothing", '\n')
        elseif isa(val, AbstractMatrix)
            print(io, "$(sym1) $(size(val,1))×$(size(val,2)) $(typeof(val))", '\n')
        elseif isa(val, AbstractVector) && length(val) > 6
            print(io, "$(sym1) $(length(val))-element $(typeof(val))", '\n')
        elseif flag
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
    name = name[1:(findfirst(x -> (x == '{' || x == '('), name) - 1)]
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
    msg = (; :eq => "must be equal to", :gt => "must be greater than",
           :lt => "must be smaller than", :geq => "must be greater than or equal to",
           :leq => "must be smaller than or equal to")
    return "$a$(!isnothing(va) ? " ($va)" : "") $(msg[c]) $b$(!isnothing(vb) ? " ($vb)" : "")"
end
function Base.iterate(obj::Union{<:AbstractEstimator, <:AbstractAlgorithm,
                                 <:AbstractResult}, state = 1)
    return state > 1 ? nothing : (obj, state + 1)
end
Base.length(::Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}) = 1

export IsEmptyError, IsNothingError, IsNothingEmptyError, IsNonFiniteError
