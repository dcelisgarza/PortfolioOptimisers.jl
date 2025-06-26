abstract type PhilogenyConstraintEstimator <: AbstractEstimator end
abstract type PhilogenyConstraintResult <: AbstractResult end
struct SemiDefinitePhilogenyConstraintEstimator{T1 <: AbstractPhilogenyEstimator,
                                                T2 <: Real} <: PhilogenyConstraintEstimator
    pe::T1
    p::T2
end
function SemiDefinitePhilogenyConstraintEstimator(;
                                                  pe::AbstractPhilogenyEstimator = NetworkEstimator(),
                                                  p::Real = 0.05)
    @smart_assert(p >= zero(p))
    return SemiDefinitePhilogenyConstraintEstimator{typeof(pe), typeof(p)}(pe, p)
end
struct SemiDefinitePhilogenyResult{T1 <: AbstractMatrix{<:Real}, T2 <: Real} <:
       PhilogenyConstraintResult
    A::T1
    p::T2
end
function SemiDefinitePhilogenyResult(; A::AbstractMatrix{<:Real}, p::Real = 0.05)
    @smart_assert(!isempty(A))
    @smart_assert(p >= zero(p))
    return SemiDefinitePhilogenyResult{typeof(A), typeof(p)}(A, p)
end
function philogeny_constraints(plc::SemiDefinitePhilogenyConstraintEstimator,
                               X::AbstractMatrix; dims::Int = 1, kwargs...)
    return SemiDefinitePhilogenyResult(;
                                       A = philogeny_matrix(plc.pe, X; dims = dims,
                                                            kwargs...), p = plc.p)
end
struct IntegerPhilogenyConstraintEstimator{T1 <: AbstractPhilogenyEstimator,
                                           T2 <:
                                           Union{<:Integer, <:AbstractVector{<:Integer}},
                                           T3 <: Real} <: PhilogenyConstraintEstimator
    pe::T1
    B::T2
    scale::T3
end
function _validate_length_integer_philogeny_constraint_B(alg::PredefinedNumberClusters,
                                                         B::AbstractVector)
    @smart_assert(length(B) <= alg.k)
    return nothing
end
function _validate_length_integer_philogeny_constraint_B(args...)
    return nothing
end
function validate_length_integer_philogeny_constraint_B(pe::ClusteringEstimator,
                                                        B::AbstractVector)
    if !isnothing(pe.onc.max_k)
        @smart_assert(length(B) <= pe.onc.max_k)
    end
    _validate_length_integer_philogeny_constraint_B(pe.onc.alg, B)
    return nothing
end
function validate_length_integer_philogeny_constraint_B(args...)
    return nothing
end
function IntegerPhilogenyConstraintEstimator(;
                                             pe::AbstractPhilogenyEstimator = NetworkEstimator(),
                                             B::Union{<:Integer,
                                                      <:AbstractVector{<:Integer}} = 1,
                                             scale::Real = 100_000.0)
    if isa(B, AbstractVector)
        @smart_assert(!isempty(B))
        @smart_assert(all(B .>= zero(B)))
        validate_length_integer_philogeny_constraint_B(pe, B)
    else
        @smart_assert(B >= zero(B))
    end
    return IntegerPhilogenyConstraintEstimator{typeof(pe), typeof(B), typeof(scale)}(pe, B,
                                                                                     scale)
end
struct IntegerPhilogenyResult{T1 <: AbstractMatrix{<:Real},
                              T2 <: Union{<:Integer, <:AbstractVector{<:Integer}},
                              T3 <: Real} <: PhilogenyConstraintResult
    A::T1
    B::T2
    scale::T3
end
function IntegerPhilogenyResult(; A::AbstractMatrix{<:Real},
                                B::Union{<:Integer, <:AbstractVector{<:Integer}} = 1,
                                scale::Real = 100_000.0)
    @smart_assert(!isempty(A))
    A = unique(A + I; dims = 1)
    if isa(B, AbstractVector)
        @smart_assert(!isempty(B))
        @smart_assert(size(A, 1) == length(B))
        @smart_assert(all(B .> zero(B)))
    else
        @smart_assert(B > zero(B))
    end
    return IntegerPhilogenyResult{typeof(A), typeof(B), typeof(scale)}(A, B, scale)
end
function philogeny_constraints(plc::IntegerPhilogenyConstraintEstimator, X::AbstractMatrix;
                               dims::Int = 1, kwargs...)
    return IntegerPhilogenyResult(; A = philogeny_matrix(plc.pe, X; dims = dims, kwargs...),
                                  B = plc.B, scale = plc.scale)
end
function philogeny_constraints(plc::Union{<:SemiDefinitePhilogenyResult,
                                          <:IntegerPhilogenyResult}, args...; kwargs...)
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
struct CentralityConstraintEstimator{T1 <: CentralityEstimator,
                                     T2 <: Union{<:Real, <:VectorToRealMeasure},
                                     T3 <: ComparisonOperators} <:
       PhilogenyConstraintEstimator
    A::T1
    B::T2
    comp::T3
end
function CentralityConstraintEstimator(; A::CentralityEstimator = CentralityEstimator(),
                                       B::Union{<:Real, <:VectorToRealMeasure} = MinValue(),
                                       comp::ComparisonOperators = LEQ())
    return CentralityConstraintEstimator{typeof(A), typeof(B), typeof(comp)}(A, B, comp)
end
function Base.iterate(S::CentralityConstraintEstimator, state = 1)
    return state > 1 ? nothing : (S, state + 1)
end
function centrality_constraints(ccs::Union{<:CentralityConstraintEstimator,
                                           <:AbstractVector{<:CentralityConstraintEstimator}},
                                X::AbstractMatrix; dims::Int = 1, kwargs...)
    if isa(ccs, AbstractVector)
        @smart_assert(!isempty(ccs))
    end
    A_ineq = Vector{eltype(X)}(undef, 0)
    B_ineq = Vector{eltype(X)}(undef, 0)
    A_eq = Vector{eltype(X)}(undef, 0)
    B_eq = Vector{eltype(X)}(undef, 0)
    for cc ∈ ccs
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
        LinearConstraintResult(;
                               ineq = if ineq_flag
                                   PartialLinearConstraintResult(; A = A_ineq, B = B_ineq)
                               else
                                   nothing
                               end,
                               eq = if eq_flag
                                   PartialLinearConstraintResult(; A = A_eq, B = B_eq)
                               else
                                   nothing
                               end)
    end
end
function centrality_constraints(ccs::LinearConstraintResult, args...; kwargs...)
    return ccs
end
function centrality_constraints(::Nothing, args...; kwargs...)
    return nothing
end

export SemiDefinitePhilogenyConstraintEstimator, SemiDefinitePhilogenyResult,
       IntegerPhilogenyConstraintEstimator, IntegerPhilogenyResult, MinValue, MeanValue,
       MedianValue, MaxValue, CentralityConstraintEstimator, philogeny_constraints,
       centrality_constraints
