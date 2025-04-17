abstract type EntropyPoolingConstraintEstimator <: AbstractEstimator end
abstract type AbstractConstantEntropyPoolingConstraintEstimator <:
              EntropyPoolingConstraintEstimator end
abstract type AbstractNonConstantEntropyPoolingConstraintEstimator <:
              EntropyPoolingConstraintEstimator end
abstract type LinearEntropyPoolingConstraintEstimator <:
              AbstractNonConstantEntropyPoolingConstraintEstimator end
abstract type BilinearEntropyPoolingConstraintEstimator <:
              AbstractNonConstantEntropyPoolingConstraintEstimator end
struct ConstantEntropyPoolingConstraintEstimator{T1 <: Real} <:
       AbstractConstantEntropyPoolingConstraintEstimator
    coef::T1
end
function ConstantEntropyPoolingConstraintEstimator(; coef::Real = 0.0)
    return ConstantEntropyPoolingConstraintEstimator{typeof(coef)}(coef)
end
struct C0_LinearEntropyPoolingConstraintEstimator{T1, T2,
                                                  T3 <:
                                                  Union{<:Real, <:AbstractVector{<:Real}}} <:
       LinearEntropyPoolingConstraintEstimator
    group::T1
    name::T2
    coef::T3
end
function C0_LinearEntropyPoolingConstraintEstimator(; group, name,
                                                    coef::Union{<:Real,
                                                                <:AbstractVector{<:Real}} = 1.0)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    coef_flag = isa(coef, AbstractVector)
    if group_flag || name_flag || coef_flag
        @smart_assert(group_flag && name_flag && coef_flag)
        @smart_assert(!isempty(group) && !isempty(name) && !isempty(coef))
        @smart_assert(length(group) == length(name) == length(coef))
    end
    return C0_LinearEntropyPoolingConstraintEstimator{typeof(group), typeof(name),
                                                      typeof(coef)}(group, name, coef)
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
abstract type C2_EntropyPoolingAlgorithm <: AbstractAlgorithm end
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
struct EntropyPoolingViewEstimator{T1 <:
                                   AbstractNonConstantEntropyPoolingConstraintEstimator,
                                   T2 <: Union{<:EntropyPoolingConstraintEstimator,
                                               <:AbstractVector{<:EntropyPoolingConstraintEstimator}},
                                   T3 <: ComparisonOperators}
    A::T1
    B::T2
    comp::T3
end
function EntropyPoolingViewEstimator(;
                                     A::AbstractNonConstantEntropyPoolingConstraintEstimator,
                                     B::Union{<:EntropyPoolingConstraintEstimator,
                                              <:AbstractVector{<:EntropyPoolingConstraintEstimator}},
                                     comp::ComparisonOperators = LEQ())
    return EntropyPoolingViewEstimator{typeof(A), typeof(B), typeof(comp)}(A, B, comp)
end
function Base.setindex!(obj::EntropyPoolingViewEstimator, args...)
    return obj
end
function Base.sort(obj::EntropyPoolingViewEstimator)
    return obj
end
function Base.sort!(obj::EntropyPoolingViewEstimator)
    return obj
end
function to_be_frozen(epv::EntropyPoolingViewEstimator)
    return isequal(epv.A, epv.B)
end
function constant_entropy_pooling_constraint!(pm::AbstractPriorResult, cache::AbstractSet,
                                              idx::AbstractVector,
                                              epvs::Union{<:EntropyPoolingViewEstimator,
                                                          <:AbstractVector{<:EntropyPoolingViewEstimator}},
                                              sets::DataFrame; strict::Bool = false,
                                              w::AbstractWeights = pweights(range(;
                                                                                  start = 1,
                                                                                  stop = 1,
                                                                                  length = size(pm.X,
                                                                                                1))))
    for (i, epv) ∈ zip(idx, view(epvs, idx))
        if i ∈ cache
            continue
        end
        # Sets A to be constant with its updated value.
        epvs[i] = EntropyPoolingViewEstimator(;
                                              B = freeze_B_view(pm, epv.A, sets, strict;
                                                                w = w),
                                              A = freeze_A_view(epv.A), comp = epv.comp)
        push!(cache, i)
    end
    return nothing
end
function freeze_A_view(A::C0_LinearEntropyPoolingConstraintEstimator{<:Any, <:Any, <:Any})
    return C0_LinearEntropyPoolingConstraintEstimator(; group = A.group, name = A.name,
                                                      coef = sign(A.coef))
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
                                                                     <:AbstractVector})
    return C0_LinearEntropyPoolingConstraintEstimator(; group = A.group, name = A.name,
                                                      coef = sign.(A.coef))
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
function _freeze_view(epc::C0_LinearEntropyPoolingConstraintEstimator,
                      pm::AbstractPriorResult, idx::AbstractVector, coef::Real; kwargs...)
    mu = view(pm.mu, idx)
    return sign(coef) * sum(mu)
end
function _freeze_view(epc::C1_LinearEntropyPoolingConstraintEstimator,
                      pm::AbstractPriorResult, idx::AbstractVector, coef::Real; kwargs...)
    sigma = pm.sigma
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
                      pm::AbstractPriorResult, idx::AbstractVector, coef::Real;
                      w::AbstractWeights = pweights(range(; start = 1, stop = 1,
                                                          length = size(pm.X, 1))),
                      kwargs...)
    X = view(pm.X, :, idx)
    return sign(coef) * sum([skewness(X[:, i], w) for i ∈ axes(X, 2)])
end
function _freeze_view(epc::C2_LinearEntropyPoolingConstraintEstimator{<:Any, <:Any, <:Any,
                                                                      <:KurtosisEntropyPoolingAlgorithm},
                      pm::AbstractPriorResult, idx::AbstractVector, coef::Real;
                      w::AbstractWeights = pweights(range(; start = 1, stop = 1,
                                                          length = size(pm.X, 1))),
                      kwargs...)
    X = view(pm.X, :, idx)
    return sign(coef) * sum([kurtosis(X[:, i], w) + 3 for i ∈ axes(X, 2)])
end
function _freeze_view(epc::C4_LinearEntropyPoolingConstraintEstimator,
                      pm::AbstractPriorResult, idx1::AbstractVector, idx2::AbstractVector,
                      coef::Real; kwargs...)
    sigma = pm.sigma
    dsigma = diag(sigma)
    dsigma1 = sqrt.(view(dsigma, idx1))
    dsigma2 = sqrt.(view(dsigma, idx2))
    return sign(coef) * sum(dsigma1 .* dsigma2)
end
function freeze_B_view(::AbstractPriorResult,
                       epv::ConstantEntropyPoolingConstraintEstimator, ::DataFrame, ::Bool,
                       args...; kwargs...)
    return epv
end
function freeze_B_view(pm::AbstractPriorResult,
                       epv::Union{<:C0_LinearEntropyPoolingConstraintEstimator{<:Any, <:Any,
                                                                               <:Real},
                                  <:C1_LinearEntropyPoolingConstraintEstimator{<:Any, <:Any,
                                                                               <:Real},
                                  <:C2_LinearEntropyPoolingConstraintEstimator{<:Any, <:Any,
                                                                               <:Real,
                                                                               <:Any}},
                       sets::DataFrame, strict::Bool = false;
                       w::AbstractWeights = pweights(range(; start = 1, stop = 1,
                                                           length = size(pm.X, 1))),
                       kwargs...)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    (; group, name, coef) = epv
    return ConstantEntropyPoolingConstraintEstimator(;
                                                     coef = if !(isnothing(group) ||
                                                                 string(group) ∉
                                                                 group_names)
                                                         idx = sets[!, group] .== name
                                                         if all(iszero.(idx))
                                                             if strict
                                                                 throw(ArgumentError("$(string(name)) is not in $(group).\n$(epv)"))
                                                             else
                                                                 @warn("$(string(name)) is not in $(group).\n$(epv)")
                                                             end
                                                             zero(eltype(pm.X))
                                                         else
                                                             _freeze_view(epv, pm, idx,
                                                                          coef; w = w)
                                                         end
                                                     elseif strict
                                                         throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(epv)"))
                                                     else
                                                         @warn("$(string(group)) is not in $(group_names).\n$(epv)")
                                                         zero(eltype(pm.X))
                                                     end)
end
function freeze_B_view(pm::AbstractPriorResult,
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
                       sets::DataFrame, strict::Bool = false;
                       w::AbstractWeights = pweights(range(; start = 1, stop = 1,
                                                           length = size(pm.X, 1))),
                       kwargs...)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    B = Vector{eltype(pm.X)}(undef, 0)
    for (group, name, coef) ∈ zip(epv.group, epv.name, epv.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            if all(iszero.(idx))
                if strict
                    throw(ArgumentError("$(string(name)) is not in $(group).\n$(epv)"))
                else
                    @warn("$(string(name)) is not in $(group).\n$(epv)")
                end
                continue
            end
            append!(B, _freeze_view(epv, pm, idx, coef; w = w))
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(epv)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(epv).")
        end
    end
    return ConstantEntropyPoolingConstraintEstimator(; coef = if !isempty(B)
                                                         sum(B)
                                                     else
                                                         zero(eltype(pm.X))
                                                     end)
end
function freeze_B_view(pm::AbstractPriorResult,
                       epv::C4_LinearEntropyPoolingConstraintEstimator{<:Any, <:Any, <:Any,
                                                                       <:Any, <:Real},
                       sets::DataFrame, strict::Bool = false;
                       w::AbstractWeights = pweights(range(; start = 1, stop = 1,
                                                           length = size(pm.X, 1))),
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
                                                         if all(iszero.(idx1)) ||
                                                            all(iszero.(idx2))
                                                             if strict
                                                                 throw(ArgumentError("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)"))
                                                             else
                                                                 @warn("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)")
                                                             end
                                                             zero(eltype(pm.X))
                                                         else
                                                             _freeze_view(epv, pm, idx1,
                                                                          idx2, coef;
                                                                          w = w)
                                                         end
                                                     elseif strict
                                                         throw(ArgumentError("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epv)."))
                                                     else
                                                         @warn("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epv).")
                                                     end)
end
function freeze_B_view(pm::AbstractPriorResult,
                       epv::C4_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                       <:AbstractVector,
                                                                       <:AbstractVector,
                                                                       <:AbstractVector,
                                                                       <:AbstractVector},
                       sets::DataFrame, strict::Bool = false;
                       w::AbstractWeights = pweights(range(; start = 1, stop = 1,
                                                           length = size(pm.X, 1))),
                       kwargs...)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    B = Vector{eltype(pm.X)}(undef, 0)
    for (group1, group2, name1, name2, coef) ∈
        zip(epv.group1, epv.group2, epv.name1, epv.name2, epv.coef)
        if !(isnothing(group1) ||
             string(group1) ∉ group_names ||
             isnothing(group2) ||
             string(group2) ∉ group_names)
            idx1 = sets[!, group1] .== name1
            idx2 = sets[!, group2] .== name2
            if all(iszero.(idx1)) || all(iszero.(idx2))
                if strict
                    throw(ArgumentError("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)"))
                else
                    @warn("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)")
                end
                continue
            end
            append!(B, _freeze_view(epv, pm, idx1, idx2, coef; w = w))
        elseif strict
            throw(ArgumentError("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epv)."))
        else
            @warn("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epv).")
        end
    end
    return ConstantEntropyPoolingConstraintEstimator(; coef = if !isempty(B)
                                                         sum(B)
                                                     else
                                                         zero(eltype(pm.X))
                                                     end)
end
function get_view_level(::EntropyPoolingViewEstimator{<:C0_LinearEntropyPoolingConstraintEstimator,
                                                      <:Any, <:Any})
    return 0
end
function get_view_level(::EntropyPoolingViewEstimator{<:C1_LinearEntropyPoolingConstraintEstimator,
                                                      <:Any, <:Any})
    return 1
end
function get_view_level(::EntropyPoolingViewEstimator{<:C2_LinearEntropyPoolingConstraintEstimator,
                                                      <:Any, <:Any})
    return 2
end
function get_view_level(::EntropyPoolingViewEstimator{<:C4_LinearEntropyPoolingConstraintEstimator,
                                                      <:Any, <:Any})
    return 4
end
function Base.isless(a::EntropyPoolingViewEstimator, b::EntropyPoolingViewEstimator)
    return Base.isless(get_view_level(a), get_view_level(b))
end
function get_B_entropy_pooling_view_data(::AbstractPriorResult,
                                         epv::ConstantEntropyPoolingConstraintEstimator,
                                         ::DataFrame, ::Bool, args...; kwargs...)
    return epv.coef
end
function _get_B_entropy_pooling_view_data(::C0_LinearEntropyPoolingConstraintEstimator,
                                          pm::AbstractPriorResult, idx::AbstractVector,
                                          coef::Real, args...; kwargs...)
    mu = view(pm.mu, idx)
    return coef * sum(mu)
end
function _get_B_entropy_pooling_view_data(epc::C1_LinearEntropyPoolingConstraintEstimator,
                                          pm::AbstractPriorResult, idx::AbstractVector,
                                          coef::Real, args...; kwargs...)
    sigma = pm.sigma
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
                                          pm::AbstractPriorResult, idx::AbstractVector,
                                          coef::Real;
                                          w::AbstractWeights = pweights(range(; start = 1,
                                                                              stop = 1,
                                                                              length = size(pm.X,
                                                                                            1))),
                                          kwargs...)
    X = view(pm.X, :, idx)
    return coef * sum([skewness(X[:, i], w)] for i ∈ axes(X, 2))
end
function _get_B_entropy_pooling_view_data(::C2_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                       <:Any,
                                                                                       <:Any,
                                                                                       <:KurtosisEntropyPoolingAlgorithm},
                                          pm::AbstractPriorResult, idx::AbstractVector,
                                          coef::Real;
                                          w::AbstractWeights = pweights(range(; start = 1,
                                                                              stop = 1,
                                                                              length = size(pm.X,
                                                                                            1))),
                                          kwargs...)
    X = view(pm.X, :, idx)
    return coef * sum([kurtosis(X[:, i], w) + 3] for i ∈ axes(X, 2))
end
function _get_B_entropy_pooling_view_data(::C4_LinearEntropyPoolingConstraintEstimator,
                                          pm::AbstractPriorResult, idx1::AbstractVector,
                                          idx2::AbstractVector, coef::Real, args...;
                                          kwargs...)
    sigma = pm.sigma
    dsigma = diag(sigma)
    dsigma1 = sqrt.(view(dsigma, idx1))
    dsigma2 = sqrt.(view(dsigma, idx2))
    return coef * sum(dsigma1 .* dsigma2)
end
function get_B_entropy_pooling_view_data(pm::AbstractPriorResult,
                                         epvbs::AbstractVector{<:EntropyPoolingConstraintEstimator},
                                         sets::DataFrame, strict::Bool = false, args...;
                                         w::AbstractWeights = pweights(range(; start = 1,
                                                                             stop = 1,
                                                                             length = size(pm.X,
                                                                                           1))))
    B = zero(eltype(pm.X))
    for epv ∈ epvbs
        B += get_B_entropy_pooling_view_data(pm, epv, sets, strict, args...; w = w)
    end
    return B
end
function get_B_entropy_pooling_view_data(pm::AbstractPriorResult,
                                         epv::C4_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                         <:Any,
                                                                                         <:Any,
                                                                                         <:Any,
                                                                                         <:Real},
                                         sets::DataFrame, strict::Bool = false, args...;
                                         w::AbstractWeights = pweights(range(; start = 1,
                                                                             stop = 1,
                                                                             length = size(pm.X,
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
        if all(iszero.(idx1)) || all(iszero.(idx2))
            if strict
                throw(ArgumentError("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)"))
            else
                @warn("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)")
            end
            zero(eltype(pm.X))
        else
            _get_B_entropy_pooling_view_data(epv, pm, idx1, idx2, coef, args...; w = w)
        end
    elseif strict
        throw(ArgumentError("$(string(group1)) or $(string(group2)) are not in $(group_names).\n$(epv)"))
    else
        @warn("$(string(group1)) or $(string(group2)) are not in $(group_names).\n$(epv)")
        zero(eltype(pm.X))
    end
    return B
end
function get_B_entropy_pooling_view_data(pm::AbstractPriorResult,
                                         epv::C4_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                                         <:AbstractVector,
                                                                                         <:AbstractVector,
                                                                                         <:AbstractVector,
                                                                                         <:AbstractVector},
                                         sets::DataFrame, strict::Bool = false, args...;
                                         w::AbstractWeights = pweights(range(; start = 1,
                                                                             stop = 1,
                                                                             length = size(pm.X,
                                                                                           1))))
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    B = zero(eltype(pm.X))
    for (group1, group2, name1, name2, coef) ∈
        zip(epv.group1, epv.group2, epv.name1, epv.name2, epv.coef)
        if !(isnothing(group1) ||
             string(group1) ∉ group_names ||
             isnothing(group2) ||
             string(group2) ∉ group_names)
            idx1 = sets[!, group1] .== name1
            idx2 = sets[!, group1] .== name2
            if all(iszero.(idx1)) || all(iszero.(idx2))
                if strict
                    throw(ArgumentError("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)"))
                else
                    @warn("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)")
                end
                continue
            end
            B += _get_B_entropy_pooling_view_data(epv, pm, idx1, idx2, coef, args...; w = w)
        elseif strict
            throw(ArgumentError("$(string(group1)) or $(string(group2)) are not in $(group_names).\n$(epv)."))
        else
            @warn("$(string(group1)) or $(string(group2)) are not in $(group_names).\n$(epv).")
        end
    end
    return B
end
function get_B_entropy_pooling_view_data(pm::AbstractPriorResult,
                                         epv::Union{<:C0_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                                 <:Any,
                                                                                                 <:Real},
                                                    <:C1_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                                 <:Any,
                                                                                                 <:Real},
                                                    <:C2_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                                 <:Any,
                                                                                                 <:Real,
                                                                                                 <:Any}},
                                         sets::DataFrame, strict::Bool = false, args...;
                                         w::AbstractWeights = pweights(range(; start = 1,
                                                                             stop = 1,
                                                                             length = size(pm.X,
                                                                                           1))))
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    (; group, name, coef) = epv
    B = if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        if all(iszero.(idx))
            if strict
                throw(ArgumentError("$(string(name)) is not in $(group).\n$(epv)"))
            else
                @warn("$(string(name)) is not in $(group).\n$(epv)")
            end
            zero(eltype(pm.X))
        else
            _get_B_entropy_pooling_view_data(epv, pm, idx, coef, args...; w = w)
        end
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(epv)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(epv)")
        zero(eltype(pm.X))
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
function get_B_entropy_pooling_view_data(pm::AbstractPriorResult,
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
                                         sets::DataFrame, strict::Bool = false, args...;
                                         w::AbstractWeights = pweights(range(; start = 1,
                                                                             stop = 1,
                                                                             length = size(pm.X,
                                                                                           1))))
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    B = zero(eltype(pm.X))
    for (group, name, coef) ∈ zip(epv.group, epv.name, epv.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            if all(iszero.(idx))
                if strict
                    throw(ArgumentError("$(string(name)) is not in $(group).\n$(epv)"))
                else
                    @warn("$(string(name)) is not in $(group).\n$(epv)")
                end
                continue
            end
            B += _get_B_entropy_pooling_view_data(epv, pm, idx, coef, args...; w = w)
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(epv)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(epv).")
        end
    end
    return B
end
function _get_A_entropy_pooling_view_data(::C0_LinearEntropyPoolingConstraintEstimator,
                                          pm::AbstractPriorResult, idx::AbstractVector,
                                          coef::Real)
    X = view(pm.X, :, idx)
    return coef * X
end
function _get_A_entropy_pooling_view_data(::C1_LinearEntropyPoolingConstraintEstimator,
                                          pm::AbstractPriorResult, idx::AbstractVector,
                                          coef::Real)
    X = view(pm.X, :, idx)
    mu = view(pm.mu, idx)
    return coef * (X .- transpose(mu)) .^ 2
end
function _get_A_entropy_pooling_view_data(::C2_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                       <:Any,
                                                                                       <:Any,
                                                                                       <:SkewnessEntropyPoolingViewAlgorithm},
                                          pm::AbstractPriorResult, idx::AbstractVector,
                                          coef::Real)
    X = view(pm.X, :, idx)
    mu = view(pm.mu, idx)
    dsigma = view(diag(pm.sigma), idx)
    return coef * ((X .- transpose(mu)) .^ 3) ./ (dsigma .* sqrt.(dsigma))
end
function _get_A_entropy_pooling_view_data(::C2_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                       <:Any,
                                                                                       <:Any,
                                                                                       <:KurtosisEntropyPoolingAlgorithm},
                                          pm::AbstractPriorResult, idx::AbstractVector,
                                          coef::Real)
    X = view(pm.X, :, idx)
    mu = view(pm.mu, idx)
    dsigma = view(diag(pm.sigma), idx)
    return coef * ((X .- transpose(mu)) .^ 4) ./ (dsigma .^ 2)
end
function _get_A_entropy_pooling_view_data(::C4_LinearEntropyPoolingConstraintEstimator,
                                          pm::AbstractPriorResult, idx1::AbstractVector,
                                          idx2::AbstractVector, coef::Real)
    X1 = view(pm.X, :, idx1)
    X2 = view(pm.X, :, idx2)
    mu1 = view(pm.mu, idx1)
    mu2 = view(pm.mu, idx2)
    return coef * vec((X1 .- transpose(mu1)) .* (X2 .- transpose(mu2)))
end
function get_A_entropy_pooling_view_data(pm::AbstractPriorResult,
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
        if all(iszero.(idx1)) || all(iszero.(idx2))
            if strict
                throw(ArgumentError("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)"))
            else
                @warn("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)")
            end
            zeros(eltype(pm.X), size(pm.X, 1))
        else
            _get_A_entropy_pooling_view_data(epv, pm, idx1, idx2, coef)
        end
    elseif strict
        throw(ArgumentError("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epv)."))
    else
        @warn("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epv).")
    end
    return A
end
function get_A_entropy_pooling_view_data(pm::AbstractPriorResult,
                                         epv::C4_LinearEntropyPoolingConstraintEstimator{<:AbstractVector,
                                                                                         <:AbstractVector,
                                                                                         <:AbstractVector,
                                                                                         <:AbstractVector,
                                                                                         <:AbstractVector},
                                         sets::DataFrame, strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    A = Vector{eltype(pm.X)}(undef, 0)
    for (group1, group2, name1, name2, coef) ∈
        zip(epv.group1, epv.group2, epv.name1, epv.name2, epv.coef)
        if !(isnothing(group1) ||
             string(group1) ∉ group_names ||
             isnothing(group2) ||
             string(group2) ∉ group_names)
            idx1 = sets[!, group1] .== name1
            idx2 = sets[!, group2] .== name2
            if all(iszero.(idx1)) || all(iszero.(idx2))
                if strict
                    throw(ArgumentError("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)"))
                else
                    @warn("$(string(name1)) or $(string(name1)) are not in $(group1) or $(group2).\n$(epv)")
                end
                continue
            else
                append!(A, _get_A_entropy_pooling_view_data(epv, pm, idx1, idx2, coef))
            end
        elseif strict
            throw(ArgumentError("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epv)."))
        else
            @warn("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epv).")
        end
    end
    if !isempty(A)
        A = vec(sum(reshape(A, size(pm.X, 1), :); dims = 2))
    end
    return A
end
function get_A_entropy_pooling_view_data(pm::AbstractPriorResult,
                                         epv::Union{<:C0_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                                 <:Any,
                                                                                                 <:Real},
                                                    <:C1_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                                 <:Any,
                                                                                                 <:Real},
                                                    <:C2_LinearEntropyPoolingConstraintEstimator{<:Any,
                                                                                                 <:Any,
                                                                                                 <:Real,
                                                                                                 <:Any}},
                                         sets::DataFrame, strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    (; group, name, coef) = epv
    A = if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        if all(iszero.(idx))
            if strict
                throw(ArgumentError("$(string(name)) is not in $(group).\n$(epv)"))
            else
                @warn("$(string(name)) is not in $(group).\n$(epv)")
                zeros(eltype(pm.X), size(pm.X, 1))
            end
        else
            _get_A_entropy_pooling_view_data(epv, pm, idx, coef)
        end
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(epv)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(epv)")
        Vector{eltype(pm.X)}(undef, 0)
    end
    return A
end
function get_A_entropy_pooling_view_data(pm::AbstractPriorResult,
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
    A = Vector{eltype(pm.X)}(undef, 0)
    for (group, name, coef) ∈ zip(epv.group, epv.name, epv.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            if all(iszero.(idx))
                if strict
                    throw(ArgumentError("$(string(name)) is not in $(group).\n$(epv)"))
                else
                    @warn("$(string(name)) is not in $(group).\n$(epv)")
                    zeros(eltype(pm.X), size(pm.X, 1))
                end
                append!(A, _get_A_entropy_pooling_view_data(epv, pm, idx, coef))
            end
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(epv)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(epv).")
        end
    end
    if !isempty(A)
        A = vec(sum(reshape(A, size(pm.X, 1), :); dims = 2))
    end
    return A
end
function entropy_pooling_views(pm::AbstractPriorResult,
                               epvs::Union{<:EntropyPoolingViewEstimator,
                                           <:AbstractVector{<:EntropyPoolingViewEstimator}},
                               sets::DataFrame; strict::Bool = false,
                               w::AbstractWeights = pweights(range(; start = 1, stop = 1,
                                                                   length = size(pm.X, 1))))
    if isa(epvs, AbstractVector)
        @smart_assert(!isempty(epvs))
    end
    @smart_assert(!isempty(sets))
    A_ineq = Vector{eltype(pm.X)}(undef, 0)
    B_ineq = Vector{eltype(pm.X)}(undef, 0)
    A_eq = Vector{eltype(pm.X)}(undef, 0)
    B_eq = Vector{eltype(pm.X)}(undef, 0)
    for epv ∈ epvs
        A = get_A_entropy_pooling_view_data(pm, epv.A, sets, strict)
        B = get_B_entropy_pooling_view_data(pm, epv.B, sets, strict; w = w)
        if isempty(A)
            continue
        end
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
        A_ineq = transpose(reshape(A_ineq, size(pm.X, 1), :))
    end
    if eq_flag
        A_eq = transpose(reshape(A_eq, size(pm.X, 1), :))
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
function entropy_pooling_views(pm::AbstractPriorResult, epvs::LinearConstraintResult,
                               args...; kwargs...)
    return lcm
end

export entropy_pooling_views, ConstantEntropyPoolingConstraintEstimator,
       C0_LinearEntropyPoolingConstraintEstimator,
       C1_LinearEntropyPoolingConstraintEstimator, SkewnessEntropyPoolingViewAlgorithm,
       KurtosisEntropyPoolingAlgorithm, C2_LinearEntropyPoolingConstraintEstimator,
       C4_LinearEntropyPoolingConstraintEstimator, EntropyPoolingViewEstimator, to_be_frozen
