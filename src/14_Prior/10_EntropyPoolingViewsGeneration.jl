abstract type EntropyPoolingConstraintEstimator <: AbstractEstimator end
abstract type ContinuousEntropyPoolingConstraintEstimator <:
              EntropyPoolingConstraintEstimator end
abstract type AbstractConstantEntropyPoolingConstraintEstimator <:
              ContinuousEntropyPoolingConstraintEstimator end
abstract type AbstractNonConstantEntropyPoolingConstraintEstimator <:
              ContinuousEntropyPoolingConstraintEstimator end
abstract type LinearEntropyPoolingConstraintEstimator <:
              AbstractNonConstantEntropyPoolingConstraintEstimator end
abstract type BilinearEntropyPoolingConstraintEstimator <:
              AbstractNonConstantEntropyPoolingConstraintEstimator end
abstract type AbstractEntropyPoolingConstraintAlgorithm <: AbstractAlgorithm end
struct ConstantEntropyPoolingConstraintEstimator{T1 <: Real} <:
       AbstractConstantEntropyPoolingConstraintEstimator
    coef::T1
end
function ConstantEntropyPoolingConstraintEstimator(; coef::Real = 0.0)
    return ConstantEntropyPoolingConstraintEstimator{typeof(coef)}(coef)
end
abstract type AbstractDiscontinuousEntropyPoolingConstraintEstimator <:
              EntropyPoolingConstraintEstimator end
abstract type C0_EntropyPoolingAlgorithm <: AbstractEntropyPoolingConstraintAlgorithm end
struct MeanEntropyPoolingViewAlgorithm <: C0_EntropyPoolingAlgorithm end
struct ValueatRiskEntropyPoolingAlgorithm{T1 <: Real} <: C0_EntropyPoolingAlgorithm
    alpha::T1
end
function ValueatRiskEntropyPoolingAlgorithm(; alpha::Real = 0.05)
    return ValueatRiskEntropyPoolingAlgorithm{typeof(alpha)}(alpha)
end
struct C0_LinearEntropyPoolingConstraintEstimator{T1, T2,
                                                  T3 <:
                                                  Union{<:Real, <:AbstractVector{<:Real}},
                                                  T4 <: C0_EntropyPoolingAlgorithm} <:
       LinearEntropyPoolingConstraintEstimator
    group::T1
    name::T2
    coef::T3
    kind::T4
end
function C0_LinearEntropyPoolingConstraintEstimator(; group, name,
                                                    coef::Union{<:Real,
                                                                <:AbstractVector{<:Real}} = 1.0,
                                                    kind::C0_EntropyPoolingAlgorithm = MeanEntropyPoolingViewAlgorithm())
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    coef_flag = isa(coef, AbstractVector)
    if group_flag || name_flag || coef_flag
        @smart_assert(group_flag && name_flag && coef_flag)
        @smart_assert(!isempty(group) && !isempty(name) && !isempty(coef))
        @smart_assert(length(group) == length(name) == length(coef))
    end
    return C0_LinearEntropyPoolingConstraintEstimator{typeof(group), typeof(name),
                                                      typeof(coef), typeof(kind)}(group,
                                                                                  name,
                                                                                  coef,
                                                                                  kind)
end
struct C1_LinearEntropyPoolingConstraintEstimator{T1, T2,
                                                  T3 <:
                                                  Union{<:Real, <:AbstractVector{<:Real}},
                                                  T4 <: Real} <:
       LinearEntropyPoolingConstraintEstimator
    group::T1
    name::T2
    coef::T3
    exponent::T4
end
function C1_LinearEntropyPoolingConstraintEstimator(; group, name,
                                                    coef::Union{<:Real,
                                                                <:AbstractVector{<:Real}} = 1.0,
                                                    exponent::Real = 1)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    coef_flag = isa(coef, AbstractVector)
    if group_flag || name_flag || coef_flag
        @smart_assert(group_flag && name_flag && coef_flag)
        @smart_assert(!isempty(group) && !isempty(name) && !isempty(coef))
        @smart_assert(length(group) == length(name) == length(coef))
    end
    return C1_LinearEntropyPoolingConstraintEstimator{typeof(group), typeof(name),
                                                      typeof(coef), typeof(exponent)}(group,
                                                                                      name,
                                                                                      coef,
                                                                                      exponent)
end
abstract type C2_EntropyPoolingAlgorithm <: AbstractEntropyPoolingConstraintAlgorithm end
struct SkewnessEntropyPoolingViewAlgorithm <: C2_EntropyPoolingAlgorithm end
struct KurtosisEntropyPoolingAlgorithm <: C2_EntropyPoolingAlgorithm end
struct C2_LinearEntropyPoolingConstraintEstimator{T1, T2,
                                                  T3 <:
                                                  Union{<:Real, <:AbstractVector{<:Real}},
                                                  T4 <: C2_EntropyPoolingAlgorithm} <:
       LinearEntropyPoolingConstraintEstimator
    group::T1
    name::T2
    coef::T3
    kind::T4
end
function C2_LinearEntropyPoolingConstraintEstimator(; group, name,
                                                    coef::Union{<:Real,
                                                                <:AbstractVector{<:Real}} = 1.0,
                                                    kind::C2_EntropyPoolingAlgorithm)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    coef_flag = isa(coef, AbstractVector)
    if group_flag || name_flag || coef_flag
        @smart_assert(group_flag && name_flag && coef_flag)
        @smart_assert(!isempty(group) && !isempty(name) && !isempty(coef))
        @smart_assert(length(group) == length(name) == length(coef))
    end
    return C2_LinearEntropyPoolingConstraintEstimator{typeof(group), typeof(name),
                                                      typeof(coef), typeof(kind)}(group,
                                                                                  name,
                                                                                  coef,
                                                                                  kind)
end
struct C4_LinearEntropyPoolingConstraintEstimator{T1, T2, T3, T4,
                                                  T5 <:
                                                  Union{<:Real, <:AbstractVector{<:Real}}} <:
       BilinearEntropyPoolingConstraintEstimator
    group1::T1
    group2::T2
    name1::T3
    name2::T4
    coef::T5
end
function C4_LinearEntropyPoolingConstraintEstimator(; group1, name1, group2, name2,
                                                    coef::Union{<:Real,
                                                                <:AbstractVector{<:Real}} = 1.0,)
    group1_flag = isa(group1, AbstractVector)
    name1_flag = isa(name1, AbstractVector)
    coef_flag = isa(coef, AbstractVector)
    group2_flag = isa(group2, AbstractVector)
    name2_flag = isa(name2, AbstractVector)
    if group1_flag || name1_flag || coef_flag || group2_flag || name2_flag
        @smart_assert(group1_flag && name1_flag && coef_flag && group2_flag && name2_flag)
        @smart_assert(!isempty(group1) &&
                      !isempty(name1) &&
                      !isempty(coef) &&
                      !isempty(group2) &&
                      !isempty(name2))
        @smart_assert(length(group1) ==
                      length(name1) ==
                      length(coef) ==
                      length(group2) ==
                      length(name2))
    end
    return C4_LinearEntropyPoolingConstraintEstimator{typeof(group1), typeof(group2),
                                                      typeof(name1), typeof(name2),
                                                      typeof(coef)}(group1, group2, name1,
                                                                    name2, coef)
end
struct ContinuousEntropyPoolingViewEstimator{T1 <:
                                             AbstractNonConstantEntropyPoolingConstraintEstimator,
                                             T2 <:
                                             Union{<:ContinuousEntropyPoolingConstraintEstimator,
                                                   <:AbstractVector{<:ContinuousEntropyPoolingConstraintEstimator}},
                                             T3 <: ComparisonOperators}
    A::T1
    B::T2
    comp::T3
end
function ContinuousEntropyPoolingViewEstimator(;
                                               A::AbstractNonConstantEntropyPoolingConstraintEstimator,
                                               B::Union{<:ContinuousEntropyPoolingConstraintEstimator,
                                                        <:AbstractVector{<:ContinuousEntropyPoolingConstraintEstimator}},
                                               comp::ComparisonOperators = LEQ())
    return ContinuousEntropyPoolingViewEstimator{typeof(A), typeof(B), typeof(comp)}(A, B,
                                                                                     comp)
end
function Base.setindex!(obj::ContinuousEntropyPoolingViewEstimator, args...)
    return obj
end
function Base.sort(obj::ContinuousEntropyPoolingViewEstimator)
    return obj
end
function Base.sort!(obj::ContinuousEntropyPoolingViewEstimator)
    return obj
end
function to_be_frozen(epv::ContinuousEntropyPoolingViewEstimator)
    return isequal(epv.A, epv.B)
end
function constant_entropy_pooling_constraint!(pr::AbstractPriorResult, cache::AbstractSet,
                                              idx::AbstractVector,
                                              epvs::Union{<:ContinuousEntropyPoolingViewEstimator,
                                                          <:AbstractVector{<:ContinuousEntropyPoolingViewEstimator}},
                                              sets::DataFrame; strict::Bool = false,
                                              w::AbstractWeights = pweights(range(;
                                                                                  start = 1,
                                                                                  stop = 1,
                                                                                  length = size(pr.X,
                                                                                                1))))
    for (i, epv) ∈ zip(idx, view(epvs, idx))
        if i ∈ cache
            continue
        end
        # Value at risk views are frozen in set_var_cvar_A_B from the start.
        if !isa(epv,
                C0_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                           <:AbstractVector,
                                                           <:AbstractVector,
                                                           <:ValueatRiskEntropyPoolingAlgorithm})
            # Sets A to be constant with its updated value.
            epvs[i] = ContinuousEntropyPoolingViewEstimator(; A = freeze_A_view(epv.A),
                                                            B = freeze_B_view(pr, epv.A,
                                                                              sets, strict;
                                                                              w = w),
                                                            comp = epv.comp)
        end
        push!(cache, i)
    end
    return nothing
end
function freeze_A_view(A::C0_LinearEntropyPoolingConstraintEstimator{<:Any, <:Any, <:Any,
                                                                     <:Any})
    return C0_LinearEntropyPoolingConstraintEstimator(; group = A.group, name = A.name,
                                                      coef = sign(A.coef), kind = A.kind)
end
function freeze_A_view(A::C1_LinearEntropyPoolingConstraintEstimator{<:Any, <:Any, <:Any})
    return C1_LinearEntropyPoolingConstraintEstimator(; group = A.group, name = A.name,
                                                      coef = sign(A.coef))
end
function freeze_A_view(A::C2_LinearEntropyPoolingConstraintEstimator{<:Any, <:Any, <:Any,
                                                                     <:Any})
    return C2_LinearEntropyPoolingConstraintEstimator(; group = A.group, name = A.name,
                                                      coef = sign(A.coef), kind = A.kind)
end
function freeze_A_view(A::C4_LinearEntropyPoolingConstraintEstimator{<:Any, <:Any, <:Any,
                                                                     <:Any, <:Any})
    return C4_LinearEntropyPoolingConstraintEstimator(; group1 = A.group1,
                                                      group2 = A.group2, name1 = A.name1,
                                                      name2 = A.name2, coef = sign(A.coef))
end
function freeze_A_view(A::C0_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                     <:AbstractVector,
                                                                     <:AbstractVector,
                                                                     <:Any})
    return C0_LinearEntropyPoolingConstraintEstimator(; group = A.group, name = A.name,
                                                      coef = sign.(A.coef), kind = A.kind)
end
function freeze_A_view(A::C1_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                     <:AbstractVector,
                                                                     <:AbstractVector})
    return C1_LinearEntropyPoolingConstraintEstimator(; group = A.group, name = A.name,
                                                      coef = sign.(A.coef))
end
function freeze_A_view(A::C2_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                     <:AbstractVector,
                                                                     <:AbstractVector,
                                                                     <:Any})
    return C2_LinearEntropyPoolingConstraintEstimator(; group = A.group, name = A.name,
                                                      coef = sign.(A.coef), kind = A.kind)
end
function freeze_A_view(A::C4_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                     <:AbstractVector,
                                                                     <:AbstractVector,
                                                                     <:AbstractVector,
                                                                     <:AbstractVector})
    return C4_LinearEntropyPoolingConstraintEstimator(; group1 = A.group1,
                                                      group2 = A.group2, name1 = A.name1,
                                                      name2 = A.name2, coef = sign.(A.coef))
end
function _freeze_view(epc::C0_LinearEntropyPoolingConstraintEstimator{<:Any, <:Any, <:Any,
                                                                      <:MeanEntropyPoolingViewAlgorithm},
                      pr::AbstractPriorResult, idx::AbstractVector, coef::Real; kwargs...)
    mu = view(pr.mu, idx)
    return sign(coef) * sum(mu)
end
function _freeze_view(epc::C1_LinearEntropyPoolingConstraintEstimator,
                      pr::AbstractPriorResult, idx::AbstractVector, coef::Real; kwargs...)
    sigma = pr.sigma
    dsigma = view(diag(sigma), idx)
    if !isone(epc.exponent)
        if isapprox(epc.exponent, 1 / 2)
            dsigma = sqrt.(dsigma)
        elseif isapprox(epc.exponent, 1 / 3)
            dsigma = cbrt.(dsigma)
        else
            dsigma = dsigma .^ epc.exponent
        end
    end
    return sign(coef) * sum(dsigma)
end
function _freeze_view(epc::C2_LinearEntropyPoolingConstraintEstimator{<:Any, <:Any, <:Any,
                                                                      <:SkewnessEntropyPoolingViewAlgorithm},
                      pr::AbstractPriorResult, idx::AbstractVector, coef::Real;
                      w::AbstractWeights = pweights(range(; start = 1, stop = 1,
                                                          length = size(pr.X, 1))),
                      kwargs...)
    X = view(pr.X, :, idx)
    return sign(coef) * sum([skewness(X[:, i], w) for i ∈ axes(X, 2)])
end
function _freeze_view(epc::C2_LinearEntropyPoolingConstraintEstimator{<:Any, <:Any, <:Any,
                                                                      <:KurtosisEntropyPoolingAlgorithm},
                      pr::AbstractPriorResult, idx::AbstractVector, coef::Real;
                      w::AbstractWeights = pweights(range(; start = 1, stop = 1,
                                                          length = size(pr.X, 1))),
                      kwargs...)
    X = view(pr.X, :, idx)
    return sign(coef) * sum([kurtosis(X[:, i], w) + 3 for i ∈ axes(X, 2)])
end
function _freeze_view(epc::C4_LinearEntropyPoolingConstraintEstimator,
                      pr::AbstractPriorResult, idx1::AbstractVector, idx2::AbstractVector,
                      coef::Real; kwargs...)
    sigma = pr.sigma
    dsigma = diag(sigma)
    dsigma1 = sqrt.(view(dsigma, idx1))
    dsigma2 = sqrt.(view(dsigma, idx2))
    return sign(coef) * sum(dsigma1 ⊙ dsigma2)
end
function freeze_B_view(::AbstractPriorResult,
                       epv::ConstantEntropyPoolingConstraintEstimator, ::DataFrame, ::Bool,
                       args...; kwargs...)
    return epv
end
function freeze_B_view(pr::AbstractPriorResult,
                       epv::Union{<:C0_LinearEntropyPoolingConstraintEstimator{<:Any, <:Any,
                                                                               <:Real,
                                                                               <:Any},
                                  <:C1_LinearEntropyPoolingConstraintEstimator{<:Any, <:Any,
                                                                               <:Real},
                                  <:C2_LinearEntropyPoolingConstraintEstimator{<:Any, <:Any,
                                                                               <:Real,
                                                                               <:Any}},
                       sets::DataFrame, strict::Bool = false;
                       w::AbstractWeights = pweights(range(; start = 1, stop = 1,
                                                           length = size(pr.X, 1))),
                       kwargs...)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    (; group, name, coef) = epv
    return ConstantEntropyPoolingConstraintEstimator(;
                                                     coef = if !(isnothing(group) ||
                                                                 string(group) ∉
                                                                 group_names)
                                                         idx = sets[!, group] .== name
                                                         if all(iszero, idx)
                                                             if strict
                                                                 throw(ArgumentError("$(string(name)) is not in $(group).\n$(epv)"))
                                                             else
                                                                 @warn("$(string(name)) is not in $(group).\n$(epv)")
                                                             end
                                                             zero(eltype(pr.X))
                                                         else
                                                             _freeze_view(epv, pr, idx,
                                                                          coef; w = w)
                                                         end
                                                     elseif strict
                                                         throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(epv)"))
                                                     else
                                                         @warn("$(string(group)) is not in $(group_names).\n$(epv)")
                                                         zero(eltype(pr.X))
                                                     end)
end
function freeze_B_view(pr::AbstractPriorResult,
                       epv::Union{<:C0_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                               <:AbstractVector,
                                                                               <:AbstractVector,
                                                                               <:Any},
                                  <:C1_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                               <:AbstractVector,
                                                                               <:AbstractVector},
                                  <:C2_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                               <:AbstractVector,
                                                                               <:AbstractVector,
                                                                               <:Any}},
                       sets::DataFrame, strict::Bool = false;
                       w::AbstractWeights = pweights(range(; start = 1, stop = 1,
                                                           length = size(pr.X, 1))),
                       kwargs...)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    B = Vector{eltype(pr.X)}(undef, 0)
    for (group, name, coef) ∈ zip(epv.group, epv.name, epv.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            if all(iszero, idx)
                if strict
                    throw(ArgumentError("$(string(name)) is not in $(group).\n$(epv)"))
                else
                    @warn("$(string(name)) is not in $(group).\n$(epv)")
                end
                continue
            end
            append!(B, _freeze_view(epv, pr, idx, coef; w = w))
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(epv)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(epv).")
        end
    end
    return ConstantEntropyPoolingConstraintEstimator(; coef = if !isempty(B)
                                                         sum(B)
                                                     else
                                                         zero(eltype(pr.X))
                                                     end)
end
function freeze_B_view(pr::AbstractPriorResult,
                       epv::C4_LinearEntropyPoolingConstraintEstimator{<:Any, <:Any, <:Any,
                                                                       <:Any, <:Real},
                       sets::DataFrame, strict::Bool = false;
                       w::AbstractWeights = pweights(range(; start = 1, stop = 1,
                                                           length = size(pr.X, 1))),
                       kwargs...)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    (; group1, group2, name1, name2, coef) = epv
    return ConstantEntropyPoolingConstraintEstimator(;
                                                     coef = if !(isnothing(group1) ||
                                                                 string(group1) ∉
                                                                 group_names ||
                                                                 isnothing(group2) ||
                                                                 string(group2) ∉
                                                                 group_names)
                                                         idx1 = sets[!, group1] .== name1
                                                         idx2 = sets[!, group2] .== name2
                                                         if all(iszero, idx1) ||
                                                            all(iszero, idx2)
                                                             if strict
                                                                 throw(ArgumentError("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)"))
                                                             else
                                                                 @warn("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)")
                                                             end
                                                             zero(eltype(pr.X))
                                                         else
                                                             _freeze_view(epv, pr, idx1,
                                                                          idx2, coef;
                                                                          w = w)
                                                         end
                                                     elseif strict
                                                         throw(ArgumentError("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epv)."))
                                                     else
                                                         @warn("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epv).")
                                                     end)
end
function freeze_B_view(pr::AbstractPriorResult,
                       epv::C4_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                       <:AbstractVector,
                                                                       <:AbstractVector,
                                                                       <:AbstractVector,
                                                                       <:AbstractVector},
                       sets::DataFrame, strict::Bool = false;
                       w::AbstractWeights = pweights(range(; start = 1, stop = 1,
                                                           length = size(pr.X, 1))),
                       kwargs...)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    B = Vector{eltype(pr.X)}(undef, 0)
    for (group1, group2, name1, name2, coef) ∈
        zip(epv.group1, epv.group2, epv.name1, epv.name2, epv.coef)
        if !(isnothing(group1) ||
             string(group1) ∉ group_names ||
             isnothing(group2) ||
             string(group2) ∉ group_names)
            idx1 = sets[!, group1] .== name1
            idx2 = sets[!, group2] .== name2
            if all(iszero, idx1) || all(iszero, idx2)
                if strict
                    throw(ArgumentError("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)"))
                else
                    @warn("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)")
                end
                continue
            end
            append!(B, _freeze_view(epv, pr, idx1, idx2, coef; w = w))
        elseif strict
            throw(ArgumentError("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epv)."))
        else
            @warn("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epv).")
        end
    end
    return ConstantEntropyPoolingConstraintEstimator(; coef = if !isempty(B)
                                                         sum(B)
                                                     else
                                                         zero(eltype(pr.X))
                                                     end)
end
function get_view_level(::ContinuousEntropyPoolingViewEstimator{<:C0_LinearEntropyPoolingConstraintEstimator,
                                                                <:Any, <:Any})
    return 0
end
function get_view_level(::ContinuousEntropyPoolingViewEstimator{<:C1_LinearEntropyPoolingConstraintEstimator,
                                                                <:Any, <:Any})
    return 1
end
function get_view_level(::ContinuousEntropyPoolingViewEstimator{<:C2_LinearEntropyPoolingConstraintEstimator,
                                                                <:Any, <:Any})
    return 2
end
function get_view_level(::ContinuousEntropyPoolingViewEstimator{<:C4_LinearEntropyPoolingConstraintEstimator,
                                                                <:Any, <:Any})
    return 4
end
function Base.isless(a::ContinuousEntropyPoolingViewEstimator,
                     b::ContinuousEntropyPoolingViewEstimator)
    return Base.isless(get_view_level(a), get_view_level(b))
end
function get_B_entropy_pooling_view_data(::AbstractPriorResult,
                                         epv::ConstantEntropyPoolingConstraintEstimator,
                                         ::DataFrame, ::Bool, args...; kwargs...)
    return epv.coef
end
#########
struct ConditionalValueatRiskPoolingConstraintEstimator{T1, T2,
                                                        T3 <: Union{<:Real,
                                                                    <:AbstractVector{<:Real}},
                                                        T4 <: Real} <:
       AbstractDiscontinuousEntropyPoolingConstraintEstimator
    group::T1
    name::T2
    coef::T3
    alpha::T4
end
function ConditionalValueatRiskPoolingConstraintEstimator(; group, name,
                                                          coef::Union{<:Real,
                                                                      <:AbstractVector{<:Real}} = 1.0,
                                                          alpha::Real = 0.05)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    coef_flag = isa(coef, AbstractVector)
    if group_flag || name_flag || coef_flag
        @smart_assert(group_flag && name_flag && coef_flag)
        @smart_assert(!isempty(group) && !isempty(name) && !isempty(coef))
        @smart_assert(length(group) == length(name) == length(coef))
    end
    return ConditionalValueatRiskPoolingConstraintEstimator{typeof(group), typeof(name),
                                                            typeof(coef), typeof(alpha)}(group,
                                                                                         name,
                                                                                         coef,
                                                                                         alpha)
end
struct DiscontinuousEntropyPoolingViewEstimator{T1 <:
                                                AbstractDiscontinuousEntropyPoolingConstraintEstimator,
                                                T2 <:
                                                Union{<:AbstractDiscontinuousEntropyPoolingConstraintEstimator,
                                                      <:ConstantEntropyPoolingConstraintEstimator,
                                                      <:AbstractVector{<:AbstractDiscontinuousEntropyPoolingConstraintEstimator},
                                                      <:AbstractVector{<:ConstantEntropyPoolingConstraintEstimator}},
                                                T3 <: ComparisonOperators}
    A::T1
    B::T2
    comp::T3
end
function DiscontinuousEntropyPoolingViewEstimator(;
                                                  A::AbstractDiscontinuousEntropyPoolingConstraintEstimator,
                                                  B::Union{<:AbstractDiscontinuousEntropyPoolingConstraintEstimator,
                                                           <:ConstantEntropyPoolingConstraintEstimator,
                                                           <:AbstractVector{<:AbstractDiscontinuousEntropyPoolingConstraintEstimator},
                                                           <:AbstractVector{<:ConstantEntropyPoolingConstraintEstimator}})
    if isa(B, AbstractVector)
        @smart_assert(!isempty(B))
        idx = isa.(B, AbstractDiscontinuousEntropyPoolingConstraintEstimator)
        @smart_assert(all(getproperty.(B[idx], :alpha) .== A.alpha))
    else
        @smart_assert(A.alpha == B.alpha)
    end
    return DiscontinuousEntropyPoolingViewEstimator{typeof(A), typeof(B), EQ}(A, B, EQ())
end
function Base.length(::Union{<:ContinuousEntropyPoolingViewEstimator,
                             <:DiscontinuousEntropyPoolingViewEstimator})
    return 1
end
function Base.iterate(S::Union{<:ContinuousEntropyPoolingViewEstimator,
                               <:DiscontinuousEntropyPoolingViewEstimator}, state = 1)
    return state > 1 ? nothing : (S, state + 1)
end
function cvar(x::AbstractVector, alpha::Real, w::AbstractWeights)
    idx = sortperm(x)
    w = w[idx] / sum(w)
    cw = cumsum(w)
    i = findlast(x -> x <= alpha, cw)
    x = view(x, idx)
    return -if isone(i)
        x[1]
    else
        dot(view(x, 1, i), view(w, 1, i)) + x[i] * (one(eltype(cw)) - cw[i - 1] / alpha)
    end
end
function cvar(x::AbstractMatrix, alpha::Real, w::AbstractWeights)
    idx = sortperm(x; dims = 1)
    sw = sum(w)
    w = [w[idx[i]] / sw for i ∈ axes(idx, 2)]
    cw = cumsum(w; dims = 1)
    i = [findlast(x -> x <= alpha, cwi) for cwi ∈ axes(idx, 2)]
    x = view(x, idx)
    ialpha = inv(alpha)
    function f(_x, _i, _w, _cw)
        return -if isone(_i)
            _x[1]
        else
            dot(view(_x, 1, _i), view(_w, 1, _i)) +
            _x[_i] * (one(eltype(_cw)) - _cw[_i - 1] * ialpha)
        end
    end
    return [f(view(x, :, i), i, view(w, :, i), view(cw, :, i)) for i ∈ axes(idx, 2)]
end
function _get_B_entropy_pooling_view_data(epv::ConditionalValueatRiskPoolingConstraintEstimator,
                                          pr::AbstractPriorResult, idx::AbstractVector,
                                          coef::Real, args...;
                                          w::AbstractWeights = pweights(range(; start = 1,
                                                                              stop = 1,
                                                                              length = size(pr.X,
                                                                                            1))),
                                          kwargs...)
    alpha = epv.alpha
    X = pr.X
    cv = cvar(X, alpha, w)
    return coef * sum(cv)
end
function _get_B_entropy_pooling_view_data(epv::C0_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                          <:Any,
                                                                                          <:Any,
                                                                                          <:ValueatRiskEntropyPoolingAlgorithm},
                                          pr::AbstractPriorResult, idx::AbstractVector,
                                          coef::Real, args...; kwargs...)
    alpha = epv.kind.alpha
    X = view(pr.X, :, idx)
    j = ceil(Int, alpha * size(X, 1))
    var = sum([-partialsort(view(X, :, i), j) for i ∈ idx])
    @smart_assert(all(var .>= zero(eltype(var))))
    return coef * var
end
function set_var_cvar_A_B(::Any, A, B)
    return A, B
end
function set_var_cvar_A_B(epv::C0_LinearEntropyPoolingConstraintEstimator{<:Any, <:Any,
                                                                          <:Any,
                                                                          <:ValueatRiskEntropyPoolingAlgorithm},
                          A::AbstractVector, B::Real)
    idx = A .<= -abs(B)
    @smart_assert(count(idx) > 0,
                  "Value at risk view too extreme, please increase `alpha`, lower the value of the view, or use a prior that accounts that includes more extreme loss values.")
    ea = eltype(A)
    A = zeros(ea, length(idx))
    A[idx] .= one(ea)
    return A, epv.kind.alpha
end
function set_var_cvar_A_B(::ConditionalValueatRiskPoolingConstraintEstimator,
                          A::AbstractVector, B::Real)
    @smart_assert(B >= -minimum(A),
                  "Conditional value at risk view too extreme, please increase `alpha`, lower the value of the view, or use a prior that accounts that includes more extreme loss values.")
    return A, B
end
#########
function _get_B_entropy_pooling_view_data(::C0_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                       <:Any,
                                                                                       <:Any,
                                                                                       <:MeanEntropyPoolingViewAlgorithm},
                                          pr::AbstractPriorResult, idx::AbstractVector,
                                          coef::Real, args...; kwargs...)
    mu = view(pr.mu, idx)
    return coef * sum(mu)
end
function _get_B_entropy_pooling_view_data(epc::C1_LinearEntropyPoolingConstraintEstimator,
                                          pr::AbstractPriorResult, idx::AbstractVector,
                                          coef::Real, args...; kwargs...)
    sigma = pr.sigma
    dsigma = view(diag(sigma), idx)
    if !isone(epc.exponent)
        if isapprox(epc.exponent, 1 / 2)
            dsigma = sqrt.(dsigma)
        elseif isapprox(epc.exponent, 1 / 3)
            dsigma = cbrt.(dsigma)
        else
            dsigma = dsigma .^ epc.exponent
        end
    end
    return coef * sum(dsigma)
end
function _get_B_entropy_pooling_view_data(::C2_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                       <:Any,
                                                                                       <:Any,
                                                                                       <:SkewnessEntropyPoolingViewAlgorithm},
                                          pr::AbstractPriorResult, idx::AbstractVector,
                                          coef::Real;
                                          w::AbstractWeights = pweights(range(; start = 1,
                                                                              stop = 1,
                                                                              length = size(pr.X,
                                                                                            1))),
                                          kwargs...)
    X = view(pr.X, :, idx)
    return coef * sum([skewness(X[:, i], w)] for i ∈ axes(X, 2))
end
function _get_B_entropy_pooling_view_data(::C2_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                       <:Any,
                                                                                       <:Any,
                                                                                       <:KurtosisEntropyPoolingAlgorithm},
                                          pr::AbstractPriorResult, idx::AbstractVector,
                                          coef::Real;
                                          w::AbstractWeights = pweights(range(; start = 1,
                                                                              stop = 1,
                                                                              length = size(pr.X,
                                                                                            1))),
                                          kwargs...)
    X = view(pr.X, :, idx)
    return coef * sum([kurtosis(X[:, i], w) + 3] for i ∈ axes(X, 2))
end
function _get_B_entropy_pooling_view_data(::C4_LinearEntropyPoolingConstraintEstimator,
                                          pr::AbstractPriorResult, idx1::AbstractVector,
                                          idx2::AbstractVector, coef::Real, args...;
                                          kwargs...)
    sigma = pr.sigma
    dsigma = diag(sigma)
    dsigma1 = sqrt.(view(dsigma, idx1))
    dsigma2 = sqrt.(view(dsigma, idx2))
    return coef * sum(dsigma1 ⊙ dsigma2)
end
function get_B_entropy_pooling_view_data(pr::AbstractPriorResult,
                                         epvbs::AbstractVector{<:ContinuousEntropyPoolingConstraintEstimator},
                                         sets::DataFrame, strict::Bool = false, args...;
                                         w::AbstractWeights = pweights(range(; start = 1,
                                                                             stop = 1,
                                                                             length = size(pr.X,
                                                                                           1))))
    B = zero(eltype(pr.X))
    for epv ∈ epvbs
        B += get_B_entropy_pooling_view_data(pr, epv, sets, strict, args...; w = w)
    end
    return B
end
function get_B_entropy_pooling_view_data(pr::AbstractPriorResult,
                                         epv::C4_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                         <:Any,
                                                                                         <:Any,
                                                                                         <:Any,
                                                                                         <:Real},
                                         sets::DataFrame, strict::Bool = false, args...;
                                         w::AbstractWeights = pweights(range(; start = 1,
                                                                             stop = 1,
                                                                             length = size(pr.X,
                                                                                           1))))
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    (; group1, group2, name1, name2, coef) = epv
    B = if !(isnothing(group1) ||
             string(group1) ∉ group_names ||
             isnothing(group2) ||
             string(group2) ∉ group_names)
        idx1 = sets[!, group1] .== name1
        idx2 = sets[!, group1] .== name2
        if all(iszero, idx1) || all(iszero, idx2)
            if strict
                throw(ArgumentError("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)"))
            else
                @warn("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)")
            end
            zero(eltype(pr.X))
        else
            _get_B_entropy_pooling_view_data(epv, pr, idx1, idx2, coef, args...; w = w)
        end
    elseif strict
        throw(ArgumentError("$(string(group1)) or $(string(group2)) are not in $(group_names).\n$(epv)"))
    else
        @warn("$(string(group1)) or $(string(group2)) are not in $(group_names).\n$(epv)")
        zero(eltype(pr.X))
    end
    return B
end
function get_B_entropy_pooling_view_data(pr::AbstractPriorResult,
                                         epv::C4_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                                         <:AbstractVector,
                                                                                         <:AbstractVector,
                                                                                         <:AbstractVector,
                                                                                         <:AbstractVector},
                                         sets::DataFrame, strict::Bool = false, args...;
                                         w::AbstractWeights = pweights(range(; start = 1,
                                                                             stop = 1,
                                                                             length = size(pr.X,
                                                                                           1))))
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    B = zero(eltype(pr.X))
    for (group1, group2, name1, name2, coef) ∈
        zip(epv.group1, epv.group2, epv.name1, epv.name2, epv.coef)
        if !(isnothing(group1) ||
             string(group1) ∉ group_names ||
             isnothing(group2) ||
             string(group2) ∉ group_names)
            idx1 = sets[!, group1] .== name1
            idx2 = sets[!, group1] .== name2
            if all(iszero, idx1) || all(iszero, idx2)
                if strict
                    throw(ArgumentError("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)"))
                else
                    @warn("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)")
                end
                continue
            end
            B += _get_B_entropy_pooling_view_data(epv, pr, idx1, idx2, coef, args...; w = w)
        elseif strict
            throw(ArgumentError("$(string(group1)) or $(string(group2)) are not in $(group_names).\n$(epv)."))
        else
            @warn("$(string(group1)) or $(string(group2)) are not in $(group_names).\n$(epv).")
        end
    end
    return B
end
function get_B_entropy_pooling_view_data(pr::AbstractPriorResult,
                                         epv::Union{<:C0_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                                 <:Any,
                                                                                                 <:Real,
                                                                                                 <:Any},
                                                    <:C1_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                                 <:Any,
                                                                                                 <:Real},
                                                    <:C2_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                                 <:Any,
                                                                                                 <:Real,
                                                                                                 <:Any},
                                                    <:ConditionalValueatRiskPoolingConstraintEstimator{<:Any,
                                                                                                       <:Any,
                                                                                                       <:Real,
                                                                                                       <:Any}},
                                         sets::DataFrame, strict::Bool = false, args...;
                                         w::AbstractWeights = pweights(range(; start = 1,
                                                                             stop = 1,
                                                                             length = size(pr.X,
                                                                                           1))))
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    (; group, name, coef) = epv
    B = if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        if all(iszero, idx)
            if strict
                throw(ArgumentError("$(string(name)) is not in $(group).\n$(epv)"))
            else
                @warn("$(string(name)) is not in $(group).\n$(epv)")
            end
            zero(eltype(pr.X))
        else
            _get_B_entropy_pooling_view_data(epv, pr, idx, coef, args...; w = w)
        end
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(epv)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(epv)")
        zero(eltype(pr.X))
    end
    return B
end
function get_view_index(epv::Union{<:C0_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                                <:AbstractVector,
                                                                                <:AbstractVector},
                                   <:C1_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                                <:AbstractVector,
                                                                                <:AbstractVector},
                                   <:C2_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                                <:AbstractVector,
                                                                                <:AbstractVector,
                                                                                <:Any},
                                   <:ConditionalValueatRiskPoolingConstraintEstimator{<:AbstractVector,
                                                                                      <:AbstractVector,
                                                                                      <:AbstractVector,
                                                                                      <:Any}},
                        sets::DataFrame, strict::Bool = false, args...)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    return for (group, name) ∈ zip(epv.group, epv.name)
        if !(isnothing(group) || string(group) ∉ group_names)
            sets[!, group] .== name
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(epv)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(epv).")
        end
    end
end
function get_view_index(epv::Union{<:C0_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                <:Any,
                                                                                <:Real},
                                   <:C1_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                <:Any,
                                                                                <:Real},
                                   <:C2_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                <:Any,
                                                                                <:Real,
                                                                                <:Any}},
                        sets::DataFrame, strict::Bool = false, args...)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    (; group, name) = epv
    return if !(isnothing(group) || string(group) ∉ group_names)
        sets[!, group] .== name
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(epv)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(epv)")
    end
end
function get_B_entropy_pooling_view_data(pr::AbstractPriorResult,
                                         epv::Union{<:C0_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                                                 <:AbstractVector,
                                                                                                 <:AbstractVector},
                                                    <:C1_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                                                 <:AbstractVector,
                                                                                                 <:AbstractVector},
                                                    <:C2_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                                                 <:AbstractVector,
                                                                                                 <:AbstractVector,
                                                                                                 <:Any},
                                                    <:ConditionalValueatRiskPoolingConstraintEstimator{<:AbstractVector,
                                                                                                       <:AbstractVector,
                                                                                                       <:AbstractVector,
                                                                                                       <:Any}},
                                         sets::DataFrame, strict::Bool = false, args...;
                                         w::AbstractWeights = pweights(range(; start = 1,
                                                                             stop = 1,
                                                                             length = size(pr.X,
                                                                                           1))))
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    B = zero(eltype(pr.X))
    for (group, name, coef) ∈ zip(epv.group, epv.name, epv.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            if all(iszero, idx)
                if strict
                    throw(ArgumentError("$(string(name)) is not in $(group).\n$(epv)"))
                else
                    @warn("$(string(name)) is not in $(group).\n$(epv)")
                end
                continue
            end
            B += _get_B_entropy_pooling_view_data(epv, pr, idx, coef, args...; w = w)
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(epv)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(epv).")
        end
    end
    return B
end
function _get_A_entropy_pooling_view_data(::Union{<:C0_LinearEntropyPoolingConstraintEstimator,
                                                  <:ConditionalValueatRiskPoolingConstraintEstimator},
                                          pr::AbstractPriorResult, idx::AbstractVector,
                                          coef::Real)
    X = view(pr.X, :, idx)
    return coef * X
end
function _get_A_entropy_pooling_view_data(::C1_LinearEntropyPoolingConstraintEstimator,
                                          pr::AbstractPriorResult, idx::AbstractVector,
                                          coef::Real)
    X = view(pr.X, :, idx)
    mu = view(pr.mu, idx)
    return coef * (X .- transpose(mu)) .^ 2
end
function _get_A_entropy_pooling_view_data(::C2_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                       <:Any,
                                                                                       <:Any,
                                                                                       <:SkewnessEntropyPoolingViewAlgorithm},
                                          pr::AbstractPriorResult, idx::AbstractVector,
                                          coef::Real)
    X = view(pr.X, :, idx)
    mu = view(pr.mu, idx)
    dsigma = view(diag(pr.sigma), idx)
    return coef * ((X .- transpose(mu)) .^ 3) ⊘ (dsigma ⊙ sqrt.(dsigma))
end
function _get_A_entropy_pooling_view_data(::C2_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                       <:Any,
                                                                                       <:Any,
                                                                                       <:KurtosisEntropyPoolingAlgorithm},
                                          pr::AbstractPriorResult, idx::AbstractVector,
                                          coef::Real)
    X = view(pr.X, :, idx)
    mu = view(pr.mu, idx)
    dsigma = view(diag(pr.sigma), idx)
    return coef * ((X .- transpose(mu)) .^ 4) ⊘ (dsigma .^ 2)
end
function _get_A_entropy_pooling_view_data(::C4_LinearEntropyPoolingConstraintEstimator,
                                          pr::AbstractPriorResult, idx1::AbstractVector,
                                          idx2::AbstractVector, coef::Real)
    X1 = view(pr.X, :, idx1)
    X2 = view(pr.X, :, idx2)
    mu1 = view(pr.mu, idx1)
    mu2 = view(pr.mu, idx2)
    return coef * vec((X1 .- transpose(mu1)) ⊙ (X2 .- transpose(mu2)))
end
function get_A_entropy_pooling_view_data(pr::AbstractPriorResult,
                                         epv::C4_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                         <:Any,
                                                                                         <:Any,
                                                                                         <:Any,
                                                                                         <:Real},
                                         sets::DataFrame, strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    (; group1, group2, name1, name2, coef) = epv
    A = if !(isnothing(group1) ||
             string(group1) ∉ group_names ||
             isnothing(group2) ||
             string(group2) ∉ group_names)
        idx1 = sets[!, group1] .== name1
        idx2 = sets[!, group2] .== name2
        if all(iszero, idx1) || all(iszero, idx2)
            if strict
                throw(ArgumentError("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)"))
            else
                @warn("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)")
            end
            zeros(eltype(pr.X), size(pr.X, 1))
        else
            _get_A_entropy_pooling_view_data(epv, pr, idx1, idx2, coef)
        end
    elseif strict
        throw(ArgumentError("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epv)."))
    else
        @warn("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epv).")
    end
    return A
end
function get_A_entropy_pooling_view_data(pr::AbstractPriorResult,
                                         epv::C4_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                                         <:AbstractVector,
                                                                                         <:AbstractVector,
                                                                                         <:AbstractVector,
                                                                                         <:AbstractVector},
                                         sets::DataFrame, strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    A = Vector{eltype(pr.X)}(undef, 0)
    for (group1, group2, name1, name2, coef) ∈
        zip(epv.group1, epv.group2, epv.name1, epv.name2, epv.coef)
        if !(isnothing(group1) ||
             string(group1) ∉ group_names ||
             isnothing(group2) ||
             string(group2) ∉ group_names)
            idx1 = sets[!, group1] .== name1
            idx2 = sets[!, group2] .== name2
            if all(iszero, idx1) || all(iszero, idx2)
                if strict
                    throw(ArgumentError("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)"))
                else
                    @warn("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)")
                end
                continue
            else
                append!(A, _get_A_entropy_pooling_view_data(epv, pr, idx1, idx2, coef))
            end
        elseif strict
            throw(ArgumentError("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epv)."))
        else
            @warn("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epv).")
        end
    end
    if !isempty(A)
        A = vec(sum(reshape(A, size(pr.X, 1), :); dims = 2))
    end
    return A
end
function get_A_entropy_pooling_view_data(pr::AbstractPriorResult,
                                         epv::Union{<:C0_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                                 <:Any,
                                                                                                 <:Real},
                                                    <:C1_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                                 <:Any,
                                                                                                 <:Real},
                                                    <:C2_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                                 <:Any,
                                                                                                 <:Real,
                                                                                                 <:Any},
                                                    <:ConditionalValueatRiskPoolingConstraintEstimator},
                                         sets::DataFrame, strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    (; group, name, coef) = epv
    A = if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        if all(iszero, idx)
            if strict
                throw(ArgumentError("$(string(name)) is not in $(group).\n$(epv)"))
            else
                @warn("$(string(name)) is not in $(group).\n$(epv)")
                zeros(eltype(pr.X), size(pr.X, 1))
            end
        else
            _get_A_entropy_pooling_view_data(epv, pr, idx, coef)
        end
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(epv)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(epv)")
        Vector{eltype(pr.X)}(undef, 0)
    end
    return A
end
function get_A_entropy_pooling_view_data(pr::AbstractPriorResult,
                                         epv::Union{<:C0_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                                                 <:AbstractVector,
                                                                                                 <:AbstractVector},
                                                    <:C1_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                                                 <:AbstractVector,
                                                                                                 <:AbstractVector},
                                                    <:C2_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                                                 <:AbstractVector,
                                                                                                 <:AbstractVector,
                                                                                                 <:Any}},
                                         sets::DataFrame, strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    A = Vector{eltype(pr.X)}(undef, 0)
    for (group, name, coef) ∈ zip(epv.group, epv.name, epv.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            if all(iszero, idx)
                if strict
                    throw(ArgumentError("$(string(name)) is not in $(group).\n$(epv)"))
                else
                    @warn("$(string(name)) is not in $(group).\n$(epv)")
                    zeros(eltype(pr.X), size(pr.X, 1))
                end
                append!(A, _get_A_entropy_pooling_view_data(epv, pr, idx, coef))
            end
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(epv)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(epv).")
        end
    end
    if !isempty(A)
        A = vec(sum(reshape(A, size(pr.X, 1), :); dims = 2))
    end
    return A
end
function entropy_pooling_views(pr::AbstractPriorResult,
                               epvs::Union{<:ContinuousEntropyPoolingViewEstimator,
                                           <:DiscontinuousEntropyPoolingViewEstimator,
                                           <:AbstractVector{<:ContinuousEntropyPoolingViewEstimator},
                                           <:AbstractVector{<:DiscontinuousEntropyPoolingViewEstimator}},
                               sets::DataFrame; strict::Bool = false,
                               w::AbstractWeights = pweights(range(; start = 1, stop = 1,
                                                                   length = size(pr.X, 1))))
    if isa(epvs, AbstractVector)
        @smart_assert(!isempty(epvs))
    end
    @smart_assert(!isempty(sets))
    A_ineq = Vector{eltype(pr.X)}(undef, 0)
    B_ineq = Vector{eltype(pr.X)}(undef, 0)
    A_eq = Vector{eltype(pr.X)}(undef, 0)
    B_eq = Vector{eltype(pr.X)}(undef, 0)
    for epv ∈ epvs
        A = get_A_entropy_pooling_view_data(pr, epv.A, sets, strict)
        if isempty(A)
            continue
        end
        B = get_B_entropy_pooling_view_data(pr, epv.B, sets, strict; w = w)
        A, B = set_var_cvar_A_B(epv, A, B)
        d, flag_ineq = comparison_sign_ineq_flag(epv.comp)
        A = d * A
        B = d * B
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
        A_ineq = transpose(reshape(A_ineq, size(pr.X, 1), :))
    end
    if eq_flag
        A_eq = transpose(reshape(A_eq, size(pr.X, 1), :))
    end
    return if !ineq_flag && !eq_flag
        nothing
    else
        return LinearConstraintResult(;
                                      ineq = if ineq_flag
                                          PartialLinearConstraintResult(; A = A_ineq,
                                                                        B = B_ineq)
                                      else
                                          nothing
                                      end,
                                      eq = if eq_flag
                                          PartialLinearConstraintResult(; A = A_eq,
                                                                        B = B_eq)
                                      else
                                          nothing
                                      end)
    end
end
function entropy_pooling_views(pr::AbstractPriorResult, epvs::LinearConstraintResult,
                               args...; kwargs...)
    return lcm
end
function entropy_pooling_views(pr::AbstractPriorResult, ::Nothing, args...; kwargs...)
    return nothing
end

export entropy_pooling_views, ConstantEntropyPoolingConstraintEstimator,
       C0_LinearEntropyPoolingConstraintEstimator,
       C1_LinearEntropyPoolingConstraintEstimator, SkewnessEntropyPoolingViewAlgorithm,
       KurtosisEntropyPoolingAlgorithm, C2_LinearEntropyPoolingConstraintEstimator,
       C4_LinearEntropyPoolingConstraintEstimator, ContinuousEntropyPoolingViewEstimator,
       to_be_frozen
