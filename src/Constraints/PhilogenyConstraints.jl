abstract type PhilogenyConstraintModel end
abstract type PhilogenyConstraint end
struct SemiDefinitePhilogenyConstraint{T1 <:
                                       Union{<:NetworkEstimator, <:ClusteringEstimator},
                                       T2 <: Real} <: PhilogenyConstraint
    pe::T1
    p::T2
end
function SemiDefinitePhilogenyConstraint(;
                                         pe::Union{<:NetworkEstimator,
                                                   <:ClusteringEstimator} = NetworkEstimator(),
                                         p::Real = 0.05)
    @smart_assert(!isempty(pe))
    @smart_assert(p >= zero(p))
    return SemiDefinitePhilogenyConstraint{typeof(pe), typeof(p)}(pe, p)
end
struct SemiDefinitePhilogenyModel{T1 <: AbstractMatrix{<:Real}, T2 <: Real} <:
       PhilogenyConstraintModel
    A::T1
    p::T2
end
function SemiDefinitePhilogenyModel(; A::AbstractMatrix{<:Real}, p::Real = 0.05)
    @smart_assert(!isempty(A))
    @smart_assert(p >= zero(p))
    return SemiDefinitePhilogenyModel{typeof(A), typeof(p)}(A, p)
end
struct IntegerPhilogenyConstraint{T1 <: Union{<:NetworkEstimator, <:ClusteringEstimator},
                                  T2 <: Union{<:Integer, <:AbstractVector{<:Integer}},
                                  T3 <: Real} <: PhilogenyConstraintModel
    pe::T1
    B::T2
    scale::T3
end
function IntegerPhilogenyConstraint(;
                                    pe::Union{<:NetworkEstimator, <:ClusteringEstimator} = NetworkEstimator(),
                                    B::Union{<:Integer, <:AbstractVector{<:Integer}} = 1,
                                    scale::Real = 100_000.0)
    @smart_assert(!isempty(pe))
    if isa(B, AbstractVector)
        @smart_assert(!isempty(B) && all(B .>= zero(B)))
        if isa(pe, ClusteringEstimator) && !isnothing(pe.nch.max_k)
            @smart_assert(length(B) <= pe.nch.max_k)
        end
    else
        @smart_assert(B >= zero(B))
    end
    return IntegerPhilogenyConstraint{typeof(pe), typeof(B), typeof(scale)}(pe, B, scale)
end
struct IntegerPhilogenyModel{T1 <: AbstractMatrix{<:Real},
                             T2 <: Union{<:Integer, <:AbstractVector{<:Integer}},
                             T3 <: Real} <: PhilogenyConstraintModel
    A::T1
    B::T2
    scale::T3
end
function IntegerPhilogenyModel(; A::AbstractMatrix{<:Real},
                               B::Union{<:Integer, <:AbstractVector{<:Integer}} = 1,
                               scale::Real = 100_000.0)
    @smart_assert(!isempty(A))
    A = unique(A + I; dims = 1)
    if isa(B, AbstractVector)
        @smart_assert(!isempty(B) && size(A, 1) == length(B) && all(B .> zero(B)))
    else
        @smart_assert(B > zero(B))
    end
    return IntegerPhilogenyModel{typeof(A), typeof(B), typeof(scale)}(A, B, scale)
end
abstract type VectorToRealMeasure end
struct MinimumValue <: VectorToRealMeasure end
function vec_to_real_measure(::MinimumValue, val::AbstractVector)
    return min(val)
end
struct MeanValue <: VectorToRealMeasure end
function vec_to_real_measure(::MeanValue, val::AbstractVector)
    return mean(val)
end
struct MedianValue <: VectorToRealMeasure end
function vec_to_real_measure(::MedianValue, val::AbstractVector)
    return median(val)
end
function vec_to_real_measure(val::Real, ::AbstractVector)
    return val
end
struct CentralityConstraint{T1 <: CentralityEstimator,
                            T2 <: Union{<:Real, <:VectorToRealMeasure},
                            T3 <: ComparisonOperators} <: PhilogenyConstraintModel
    A::T1
    B::T2
    comp::T3
end
function CentralityConstraint(; A::CentralityEstimator = CentralityConstraint(),
                              B::Union{<:Real, <:VectorToRealMeasure} = MinimumValue(),
                              comp::ComparisonOperators = EQ())
    @smart_assert(!isempty(A))
    return CentralityConstraint{typeof(A), typeof(B), typeof(comp)}(A, B, comp)
end
function centrality_constraints(ccs::Union{<:CentralityConstraint,
                                           <:AbstractVector{<:CentralityConstraint}},
                                X::AbstractMatrix; dims::Int = 1, kwargs...)
    if isa(ccs, AbstractVector)
        @smart_assert(!isempty(ccs))
    end
    A_ineq = Vector{eltype(X)}(undef, 0)
    B_ineq = Vector{eltype(X)}(undef, 0)
    A_eq = Vector{eltype(X)}(undef, 0)
    B_eq = Vector{eltype(X)}(undef, 0)
    for cc ∈ ccs
        A = centrality_vector(cc.A, X; dims = dims)
        B = vec_to_real_measure(cc.B, A)
        lhs_flag = isempty(A) || all(iszero.(A))
        if lhs_flag
            continue
        end
        d, flag_ineq = comparison_sign_ineq_flag(cc.comp)
        A = d * A
        B = d * cc.B
        if flag_ineq
            append!(A_ineq, A)
            append!(B_ineq, B)
        else
            append!(A_eq, A)
            append!(B_eq, B)
        end
    end
    if !isempty(A_ineq)
        A_ineq = transpose(reshape(A_ineq, size(X, 2), :))
    else
        A_ineq = nothing
        B_ineq = nothing
    end
    if !isempty(A_eq)
        A_eq = transpose(reshape(A_eq, size(X, 2), :))
    else
        A_eq = nothing
        B_eq = nothing
    end
    return LinearConstraintModel(;
                                 ineq = PartialLinearConstraintModel(; A = A_ineq,
                                                                     B = B_ineq),
                                 eq = PartialLinearConstraintModel(; A = A_eq, B = B_eq))
end
function centrality_constraints(ccs::LinearConstraintModel, args...; kwargs...)
    return ccs
end
function centrality_constraints(::Nothing, args...; kwargs...)
    return nothing
end
function philogeny_constraints(plc::SemiDefinitePhilogenyConstraint, X::AbstractMatrix;
                               dims::Int = 1, kwargs...)
    return SemiDefinitePhilogenyModel(;
                                      A = philogeny_matrix(plc.pe, X; dims = dims,
                                                           kwargs...), p = plc.p)
end
function philogeny_constraints(plc::IntegerPhilogenyConstraint, X::AbstractMatrix;
                               dims::Int = 1, kwargs...)
    return IntegerPhilogenyModel(; A = philogeny_matrix(plc.pe, X; dims = dims, kwargs...),
                                 B = plc.B, scale = plc.scale)
end
function philogeny_constraints(plc::Union{<:SemiDefinitePhilogenyModel,
                                          <:IntegerPhilogenyModel}, args...; kwargs...)
    return plc
end
function philogeny_constraints(::Nothing, args...; kwargs...)
    return nothing
end
