abstract type PhilogenyEstimator <: AbstractEstimator end
abstract type PhilogenyResult <: AbstractResult end
struct SemiDefinitePhilogenyEstimator{T1, T2} <: PhilogenyEstimator
    pe::T1
    p::T2
end
function SemiDefinitePhilogenyEstimator(;
                                        pe::Union{<:AbstractPhilogenyEstimator,
                                                  <:AbstractClusteringResult} = Network(),
                                        p::Real = 0.05)
    @argcheck(p >= zero(p))
    return SemiDefinitePhilogenyEstimator(pe, p)
end
struct SemiDefinitePhilogeny{T1, T2} <: PhilogenyResult
    A::T1
    p::T2
end
function SemiDefinitePhilogeny(; A::AbstractMatrix{<:Real}, p::Real = 0.05)
    @argcheck(!isempty(A))
    @argcheck(p >= zero(p))
    return SemiDefinitePhilogeny(A, p)
end
function philogeny_constraints(plc::SemiDefinitePhilogenyEstimator, X::AbstractMatrix;
                               dims::Int = 1, kwargs...)
    return SemiDefinitePhilogeny(; A = philogeny_matrix(plc.pe, X; dims = dims, kwargs...),
                                 p = plc.p)
end
struct IntegerPhilogenyEstimator{T1, T2, T3} <: PhilogenyEstimator
    pe::T1
    B::T2
    scale::T3
end
function _validate_length_integer_philogeny_constraint_B(alg::PredefinedNumberClusters,
                                                         B::AbstractVector)
    @argcheck(length(B) <= alg.k)
    return nothing
end
function _validate_length_integer_philogeny_constraint_B(args...)
    return nothing
end
function validate_length_integer_philogeny_constraint_B(pe::ClusteringEstimator,
                                                        B::AbstractVector)
    if !isnothing(pe.onc.max_k)
        @argcheck(length(B) <= pe.onc.max_k)
    end
    _validate_length_integer_philogeny_constraint_B(pe.onc.alg, B)
    return nothing
end
function validate_length_integer_philogeny_constraint_B(args...)
    return nothing
end
function IntegerPhilogenyEstimator(;
                                   pe::Union{<:AbstractPhilogenyEstimator,
                                             <:AbstractClusteringResult} = Network(),
                                   B::Union{<:Integer, <:AbstractVector{<:Integer}} = 1,
                                   scale::Real = 100_000.0)
    if isa(B, AbstractVector)
        @argcheck(!isempty(B))
        @argcheck(all(x -> x >= zero(x), B))
        validate_length_integer_philogeny_constraint_B(pe, B)
    else
        @argcheck(B >= zero(B))
    end
    return IntegerPhilogenyEstimator(pe, B, scale)
end
struct IntegerPhilogeny{T1, T2, T3} <: PhilogenyResult
    A::T1
    B::T2
    scale::T3
end
function IntegerPhilogeny(; A::AbstractMatrix{<:Real},
                          B::Union{<:Integer, <:AbstractVector{<:Integer}} = 1,
                          scale::Real = 100_000.0)
    @argcheck(!isempty(A))
    A = unique(A + I; dims = 1)
    if isa(B, AbstractVector)
        @argcheck(!isempty(B))
        @argcheck(size(A, 1) == length(B))
        @argcheck(all(x -> x >= zero(x), B))
    else
        @argcheck(B >= zero(B))
    end
    return IntegerPhilogeny(A, B, scale)
end
function philogeny_constraints(plc::IntegerPhilogenyEstimator, X::AbstractMatrix;
                               dims::Int = 1, kwargs...)
    return IntegerPhilogeny(; A = philogeny_matrix(plc.pe, X; dims = dims, kwargs...),
                            B = plc.B, scale = plc.scale)
end
function philogeny_constraints(plc::Union{<:SemiDefinitePhilogeny, <:IntegerPhilogeny},
                               args...; kwargs...)
    return plc
end
function philogeny_constraints(::Nothing, args...; kwargs...)
    return nothing
end
abstract type VectorToRealMeasure <: AbstractAlgorithm end
struct MinValue <: VectorToRealMeasure end
function vec_to_real_measure(::MinValue, val::AbstractVector)
    return minimum(val)
end
struct MeanValue <: VectorToRealMeasure end
function vec_to_real_measure(::MeanValue, val::AbstractVector)
    return mean(val)
end
struct MedianValue <: VectorToRealMeasure end
function vec_to_real_measure(::MedianValue, val::AbstractVector)
    return median(val)
end
struct MaxValue <: VectorToRealMeasure end
function vec_to_real_measure(::MaxValue, val::AbstractVector)
    return maximum(val)
end
function vec_to_real_measure(val::Real, ::AbstractVector)
    return val
end
struct CentralityEstimator{T1, T2, T3} <: PhilogenyEstimator
    A::T1
    B::T2
    comp::T3
end
function CentralityEstimator(; A::Centrality = Centrality(),
                             B::Union{<:Real, <:VectorToRealMeasure} = MinValue(),
                             comp::ComparisonOperators = LEQ())
    return CentralityEstimator(A, B, comp)
end
function Base.iterate(S::CentralityEstimator, state = 1)
    return state > 1 ? nothing : (S, state + 1)
end
function centrality_constraints(ccs::Union{<:CentralityEstimator,
                                           <:AbstractVector{<:CentralityEstimator}},
                                X::AbstractMatrix; dims::Int = 1, kwargs...)
    if isa(ccs, AbstractVector)
        @argcheck(!isempty(ccs))
    end
    A_ineq = Vector{eltype(X)}(undef, 0)
    B_ineq = Vector{eltype(X)}(undef, 0)
    A_eq = Vector{eltype(X)}(undef, 0)
    B_eq = Vector{eltype(X)}(undef, 0)
    for cc in ccs
        A = centrality_vector(cc.A, X; dims = dims, kwargs...)
        lhs_flag = isempty(A) || all(iszero, A)
        if lhs_flag
            continue
        end
        d, flag_ineq = comparison_sign_ineq_flag(cc.comp)
        A .*= d
        B = d * vec_to_real_measure(cc.B, A)
        if flag_ineq
            append!(A_ineq, A)
            append!(B_ineq, B)
        else
            append!(A_eq, A)
            append!(B_eq, B)
        end
    end
    ineq_flag = !isempty(A_ineq)
    eq_flag = !isempty(A_eq)
    if ineq_flag
        A_ineq = transpose(reshape(A_ineq, size(X, 2), :))
    end
    if eq_flag
        A_eq = transpose(reshape(A_eq, size(X, 2), :))
    end
    return if !ineq_flag && !eq_flag
        nothing
    else
        LinearConstraint(; ineq = if ineq_flag
                             PartialLinearConstraint(; A = A_ineq, B = B_ineq)
                         else
                             nothing
                         end, eq = if eq_flag
                             PartialLinearConstraint(; A = A_eq, B = B_eq)
                         else
                             nothing
                         end)
    end
end
function centrality_constraints(ccs::LinearConstraint, args...; kwargs...)
    return ccs
end
function centrality_constraints(::Nothing, args...; kwargs...)
    return nothing
end

export SemiDefinitePhilogenyEstimator, SemiDefinitePhilogeny, IntegerPhilogenyEstimator,
       IntegerPhilogeny, MinValue, MeanValue, MedianValue, MaxValue, CentralityEstimator,
       philogeny_constraints, centrality_constraints
