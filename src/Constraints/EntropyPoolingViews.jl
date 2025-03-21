abstract type EntropyPoolingConstraint end
abstract type AbstractConstantEntropyPoolingConstraint <: EntropyPoolingConstraint end
abstract type AbstractNonConstantEntropyPoolingConstraint <: EntropyPoolingConstraint end
abstract type LinearEntropyPoolingConstraint <: AbstractNonConstantEntropyPoolingConstraint end
abstract type BilinearEntropyPoolingConstraint <:
              AbstractNonConstantEntropyPoolingConstraint end
struct ConstantEntropyPoolingConstraint{T1 <: Real} <:
       AbstractConstantEntropyPoolingConstraint
    coef::T1
end
function ConstantEntropyPoolingConstraint(; coef::Real = 0.0)
    return ConstantEntropyPoolingConstraint{typeof(coef)}(coef)
end
struct C0_LinearEntropyPoolingConstraint{T1, T2,
                                         T3 <: Union{<:Real, <:AbstractVector{<:Real}}} <:
       LinearEntropyPoolingConstraint
    group::T1
    name::T2
    coef::T3
end
function C0_LinearEntropyPoolingConstraint(; group = nothing, name = nothing,
                                           coef::Union{<:Real, <:AbstractVector{<:Real}} = 1.0)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    coef_flag = isa(coef, AbstractVector)
    if any((group_flag, name_flag, coef_flag))
        @smart_assert(all((group_flag, name_flag, coef_flag)))
        @smart_assert(!isempty(group) && !isempty(name) && !isempty(coef))
        @smart_assert(length(group) == length(name) == length(coef))
        for (g, n) ∈ zip(group, name)
            if isnothing(g) || isnothing(n)
                @smart_assert(isnothing(g) && isnothing(n))
            end
        end
    else
        if isnothing(group) || isnothing(name)
            @smart_assert(isnothing(group) && isnothing(name))
        end
    end
    return C0_LinearEntropyPoolingConstraint{typeof(group), typeof(name), typeof(coef)}(group,
                                                                                        name,
                                                                                        coef)
end
struct C1_LinearEntropyPoolingConstraint{T1, T2,
                                         T3 <: Union{<:Real, <:AbstractVector{<:Real}},
                                         T4 <: Real} <: LinearEntropyPoolingConstraint
    group::T1
    name::T2
    coef::T3
    exponent::T4
end
function C1_LinearEntropyPoolingConstraint(; group = nothing, name = nothing,
                                           coef::Union{<:Real, <:AbstractVector{<:Real}} = 1.0,
                                           exponent::Real = 1)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    coef_flag = isa(coef, AbstractVector)
    if any((group_flag, name_flag, coef_flag))
        @smart_assert(all((group_flag, name_flag, coef_flag)))
        @smart_assert(!isempty(group) && !isempty(name) && !isempty(coef))
        @smart_assert(length(group) == length(name) == length(coef))
        for (g, n) ∈ zip(group, name)
            if isnothing(g) || isnothing(n)
                @smart_assert(isnothing(g) && isnothing(n))
            end
        end
    else
        if isnothing(group) || isnothing(name)
            @smart_assert(isnothing(group) && isnothing(name))
        end
    end
    return C1_LinearEntropyPoolingConstraint{typeof(group), typeof(name), typeof(coef),
                                             typeof(exponent)}(group, name, coef, exponent)
end
abstract type C2_EntropyPoolingKind end
struct SkewnessEntropyPoolingView <: C2_EntropyPoolingKind end
struct KurtosisEntropyPoolingView <: C2_EntropyPoolingKind end
struct C2_LinearEntropyPoolingConstraint{T1, T2,
                                         T3 <: Union{<:Real, <:AbstractVector{<:Real}},
                                         T4 <: C2_EntropyPoolingKind} <:
       LinearEntropyPoolingConstraint
    group::T1
    name::T2
    coef::T3
    kind::T4
end
function C2_LinearEntropyPoolingConstraint(; group = nothing, name = nothing,
                                           coef::Union{<:Real, <:AbstractVector{<:Real}} = 1.0,
                                           kind::C2_EntropyPoolingKind = SkewnessEntropyPoolingView())
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    coef_flag = isa(coef, AbstractVector)
    if any((group_flag, name_flag, coef_flag))
        @smart_assert(all((group_flag, name_flag, coef_flag)))
        @smart_assert(!isempty(group) && !isempty(name) && !isempty(coef))
        @smart_assert(length(group) == length(name) == length(coef))
        for (g, n) ∈ zip(group, name)
            if isnothing(g) || isnothing(n)
                @smart_assert(isnothing(g) && isnothing(n))
            end
        end
    else
        if isnothing(group) || isnothing(name)
            @smart_assert(isnothing(group) && isnothing(name))
        end
    end
    return C2_LinearEntropyPoolingConstraint{typeof(group), typeof(name), typeof(coef),
                                             typeof(kind)}(group, name, coef, kind)
end
struct C4_LinearEntropyPoolingConstraint{T1, T2, T3, T4,
                                         T5 <: Union{<:Real, <:AbstractVector{<:Real}}} <:
       BilinearEntropyPoolingConstraint
    group1::T1
    group2::T2
    name1::T3
    name2::T4
    coef::T5
end
function C4_LinearEntropyPoolingConstraint(; group1 = nothing, name1 = nothing,
                                           group2 = nothing, name2 = nothing,
                                           coef::Union{<:Real, <:AbstractVector{<:Real}} = 1.0,)
    group1_flag = isa(group1, AbstractVector)
    name1_flag = isa(name1, AbstractVector)
    coef_flag = isa(coef, AbstractVector)
    group2_flag = isa(group2, AbstractVector)
    name2_flag = isa(name2, AbstractVector)
    if any((group1_flag, name1_flag, coef_flag, group2_flag, name2_flag))
        @smart_assert(all((group1_flag, name1_flag, coef_flag, group2_flag, name2_flag)))
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
        for (g1, n1, g2, n2) ∈ zip(group1, name1, group2, name2)
            if isnothing(g1) || isnothing(n1) || isnothing(g2) || isnothing(n2)
                @smart_assert(isnothing(g1) &&
                              isnothing(n1) &&
                              isnothing(g2) &&
                              isnothing(n2))
            end
        end
    else
        if isnothing(group1) || isnothing(name1) || isnothing(group2) || isnothing(name2)
            @smart_assert(isnothing(group1) &&
                          isnothing(name1) &&
                          isnothing(group2) &&
                          isnothing(name2))
        end
    end
    return C4_LinearEntropyPoolingConstraint{typeof(group1), typeof(group2), typeof(name1),
                                             typeof(name2), typeof(coef)}(group1, group2,
                                                                          name1, name2,
                                                                          coef)
end
struct EntropyPoolingView{T1 <: AbstractNonConstantEntropyPoolingConstraint,
                          T2 <: Union{<:EntropyPoolingConstraint,
                                      <:AbstractVector{<:EntropyPoolingConstraint}},
                          T3 <: ComparisonOperators}
    A::T1
    B::T2
    comp::T3
end
function EntropyPoolingView(;
                            A::AbstractNonConstantEntropyPoolingConstraint = C0_LinearEntropyPoolingConstraint(),
                            B::Union{<:EntropyPoolingConstraint,
                                     <:AbstractVector{<:EntropyPoolingConstraint}} = C0_LinearEntropyPoolingConstraint(),
                            comp::ComparisonOperators = LEQ())
    return EntropyPoolingView{typeof(A), typeof(B), typeof(comp)}(A, B, comp)
end
function Base.setindex!(obj::EntropyPoolingView, args...)
    return obj
end
function Base.sort(obj::EntropyPoolingView)
    return obj
end
function isfixed(epv::EntropyPoolingView)
    return isequal(epv.A, epv.B)
end
function fixed_entropy_pooling_view_factory(epv::EntropyPoolingView{<:C0_LinearEntropyPoolingConstraint,
                                                                    <:Any, <:Any})
    A = C0_LinearEntropyPoolingConstraint(; group = epv.A.group, name = epv.A.name,
                                          coef = if isa(epv.A.group, AbstractVector)
                                              range(; start = 1, stop = 1,
                                                    length = length(epv.A.group))
                                          else
                                              1
                                          end)
    return EntropyPoolingView(; A = A, B = A, comp = EQ())
end
function fixed_entropy_pooling_view_factory(epv::EntropyPoolingView{<:C1_LinearEntropyPoolingConstraint,
                                                                    <:Any, <:Any})
    A = C1_LinearEntropyPoolingConstraint(; group = epv.A.group, name = epv.A.name,
                                          coef = if isa(epv.A.group, AbstractVector)
                                              range(; start = 1, stop = 1,
                                                    length = length(epv.A.group))
                                          else
                                              1
                                          end, exponent = epv.A.exponent)
    return EntropyPoolingView(; A = A, B = A, comp = EQ())
end
function fixed_entropy_pooling_view_factory(epv::EntropyPoolingView{<:C2_LinearEntropyPoolingConstraint,
                                                                    <:Any, <:Any})
    # return epv
    A = C2_LinearEntropyPoolingConstraint(; group = epv.A.group, name = epv.A.name,
                                          coef = if isa(epv.A.group, AbstractVector)
                                              range(; start = 1, stop = 1,
                                                    length = length(epv.A.group))
                                          else
                                              1
                                          end, kind = epv.A.kind)
    return EntropyPoolingView(; A = A, B = A, comp = EQ())
end
function fixed_entropy_pooling_view_factory(epv::EntropyPoolingView{<:C4_LinearEntropyPoolingConstraint,
                                                                    <:Any, <:Any})
    A = C4_LinearEntropyPoolingConstraint(; group1 = epv.A.group1, group2 = epv.A.group2,
                                          name1 = epv.A.name1, name2 = epv.A.name2,
                                          coef = if isa(epv.A.group1, AbstractVector)
                                              range(; start = 1, stop = 1,
                                                    length = length(epv.A.group1))
                                          else
                                              1
                                          end)
    return EntropyPoolingView(; A = A, B = A, comp = EQ())
end
function get_view_level(::EntropyPoolingView{<:C0_LinearEntropyPoolingConstraint, <:Any,
                                             <:Any})
    return 0
end
function get_view_level(::EntropyPoolingView{<:C1_LinearEntropyPoolingConstraint, <:Any,
                                             <:Any})
    return 1
end
function get_view_level(::EntropyPoolingView{<:C2_LinearEntropyPoolingConstraint, <:Any,
                                             <:Any})
    return 2
end
function get_view_level(::EntropyPoolingView{<:C4_LinearEntropyPoolingConstraint, <:Any,
                                             <:Any})
    return 4
end
function Base.isless(a::EntropyPoolingView, b::EntropyPoolingView)
    return Base.isless(get_view_level(a), get_view_level(b))
end
function get_B_entropy_pooling_view_data(::AbstractPriorModel,
                                         epvb::ConstantEntropyPoolingConstraint,
                                         ::DataFrame, ::Bool, args...; kwargs...)
    return epvb.coef
end
function _get_B_entropy_pooling_view_data(::C0_LinearEntropyPoolingConstraint,
                                          pm::AbstractPriorModel, idx::AbstractVector,
                                          coef::Real, args...; kwargs...)
    mu = view(pm.mu, idx)
    return coef * sum(mu)
end
function _get_B_entropy_pooling_view_data(epc::C1_LinearEntropyPoolingConstraint,
                                          pm::AbstractPriorModel, idx::AbstractVector,
                                          coef::Real, args...;
                                          old_sigma::Union{Nothing, AbstractMatrix} = nothing,
                                          kwargs...)
    sigma = isnothing(old_sigma) ? pm.sigma : old_sigma
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
function _get_B_entropy_pooling_view_data(::C2_LinearEntropyPoolingConstraint{<:Any, <:Any,
                                                                              <:Any,
                                                                              <:SkewnessEntropyPoolingView},
                                          pm::AbstractPriorModel, idx::AbstractVector,
                                          coef::Real;
                                          old_w::AbstractWeights = pweights(range(;
                                                                                  start = 1,
                                                                                  stop = 1,
                                                                                  length = size(pm.X,
                                                                                                1))),
                                          kwargs...)
    X = view(pm.X, :, idx)
    return coef * sum([skewness(X[:, i], old_w)] for i ∈ axes(X, 2))
end
function _get_B_entropy_pooling_view_data(::C2_LinearEntropyPoolingConstraint{<:Any, <:Any,
                                                                              <:Any,
                                                                              <:KurtosisEntropyPoolingView},
                                          pm::AbstractPriorModel, idx::AbstractVector,
                                          coef::Real;
                                          old_w::AbstractWeights = pweights(range(;
                                                                                  start = 1,
                                                                                  stop = 1,
                                                                                  length = size(pm.X,
                                                                                                1))),
                                          kwargs...)
    X = view(pm.X, :, idx)
    return coef * sum([kurtosis(X[:, i], old_w) + 3] for i ∈ axes(X, 2))
end
function _get_B_entropy_pooling_view_data(::C4_LinearEntropyPoolingConstraint,
                                          pm::AbstractPriorModel, idx1::AbstractVector,
                                          idx2::AbstractVector, coef::Real, args...;
                                          old_sigma::Union{Nothing, AbstractMatrix} = nothing,
                                          kwargs...)
    sigma = isnothing(old_sigma) ? pm.sigma : old_sigma
    dsigma = diag(sigma)
    dsigma1 = sqrt.(view(dsigma, idx1))
    dsigma2 = sqrt.(view(dsigma, idx2))
    return coef * sum(dsigma1 .* dsigma2)
end
function get_B_entropy_pooling_view_data(pm::AbstractPriorModel,
                                         epvbs::AbstractVector{<:EntropyPoolingConstraint},
                                         sets::DataFrame, strict::Bool = false, args...;
                                         old_mu::Union{Nothing, AbstractVector} = nothing,
                                         old_sigma::Union{Nothing, AbstractMatrix} = nothing,
                                         old_w::AbstractWeights = pweights(range(;
                                                                                 start = 1,
                                                                                 stop = 1,
                                                                                 length = size(pm.X,
                                                                                               1))))
    B = zero(eltype(pm.X))
    for epvb ∈ epvbs
        B += get_B_entropy_pooling_view_data(pm, epvb, sets, strict, args...;
                                             old_mu = old_mu, old_sigma = old_sigma,
                                             old_w = old_w)
    end
    return B
end
function get_B_entropy_pooling_view_data(pm::AbstractPriorModel,
                                         epvb::C4_LinearEntropyPoolingConstraint{<:Any,
                                                                                 <:Any,
                                                                                 <:Any,
                                                                                 <:Any,
                                                                                 <:Real},
                                         sets::DataFrame, strict::Bool = false, args...;
                                         old_mu::Union{Nothing, AbstractVector} = nothing,
                                         old_sigma::Union{Nothing, AbstractMatrix} = nothing,
                                         old_w::AbstractWeights = pweights(range(;
                                                                                 start = 1,
                                                                                 stop = 1,
                                                                                 length = size(pm.X,
                                                                                               1))))
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    (; group1, group2, name1, name2, coef) = epvb
    B = if !(isnothing(group1) ||
             string(group1) ∉ group_names ||
             isnothing(group2) ||
             string(group2) ∉ group_names)
        idx1 = sets[!, group1] .== name1
        idx2 = sets[!, group1] .== name2
        _get_B_entropy_pooling_view_data(epvb, pm, idx1, idx2, coef, args...;
                                         old_mu = old_mu, old_sigma = old_sigma,
                                         old_w = old_w)
    elseif strict
        throw(ArgumentError("$(string(group1)) or $(string(group2)) are not in $(group_names).\n$(epvb)"))
    else
        @warn("$(string(group1)) or $(string(group2)) are not in $(group_names).\n$(epvb)")
        zero(eltype(pm.X))
    end
    return B
end
function get_B_entropy_pooling_view_data(pm::AbstractPriorModel,
                                         epvb::C4_LinearEntropyPoolingConstraint{<:AbstractVector,
                                                                                 <:AbstractVector,
                                                                                 <:AbstractVector,
                                                                                 <:AbstractVector,
                                                                                 <:AbstractVector},
                                         sets::DataFrame, strict::Bool = false, args...;
                                         old_mu::Union{Nothing, AbstractVector} = nothing,
                                         old_sigma::Union{Nothing, AbstractMatrix} = nothing,
                                         old_w::AbstractWeights = pweights(range(;
                                                                                 start = 1,
                                                                                 stop = 1,
                                                                                 length = size(pm.X,
                                                                                               1))))
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    B = zero(eltype(pm.X))
    for (group1, group2, name1, name2, coef) ∈
        zip(epvb.group1, epvb.group2, epvb.name1, epvb.name2, epvb.coef)
        if !(isnothing(group1) ||
             string(group1) ∉ group_names ||
             isnothing(group2) ||
             string(group2) ∉ group_names)
            idx1 = sets[!, group1] .== name1
            idx2 = sets[!, group1] .== name2
            B += _get_B_entropy_pooling_view_data(epvb, pm, idx1, idx2, coef, args...;
                                                  old_mu = old_mu, old_sigma = old_sigma,
                                                  old_w = old_w)
        elseif strict
            throw(ArgumentError("$(string(group1)) or $(string(group2)) are not in $(group_names).\n$(epvb)."))
        else
            @warn("$(string(group1)) or $(string(group2)) are not in $(group_names).\n$(epvb).")
        end
    end
    return B
end
function get_B_entropy_pooling_view_data(pm::AbstractPriorModel,
                                         epvb::Union{<:C0_LinearEntropyPoolingConstraint{<:Any,
                                                                                         <:Any,
                                                                                         <:Real},
                                                     <:C1_LinearEntropyPoolingConstraint{<:Any,
                                                                                         <:Any,
                                                                                         <:Real},
                                                     <:C2_LinearEntropyPoolingConstraint{<:Any,
                                                                                         <:Any,
                                                                                         <:Real}},
                                         sets::DataFrame, strict::Bool = false, args...;
                                         old_mu::Union{Nothing, AbstractVector} = nothing,
                                         old_sigma::Union{Nothing, AbstractMatrix} = nothing,
                                         old_w::AbstractWeights = pweights(range(;
                                                                                 start = 1,
                                                                                 stop = 1,
                                                                                 length = size(pm.X,
                                                                                               1))))
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    (; group, name, coef) = epvb
    B = if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        _get_B_entropy_pooling_view_data(epvb, pm, idx, coef, args...; old_mu = old_mu,
                                         old_sigma = old_sigma, old_w = old_w)
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(epvb)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(epvb)")
        zero(eltype(pm.X))
    end
    return B
end
function get_B_entropy_pooling_view_data(pm::AbstractPriorModel,
                                         epvb::Union{<:C0_LinearEntropyPoolingConstraint{<:AbstractVector,
                                                                                         <:AbstractVector,
                                                                                         <:AbstractVector},
                                                     <:C1_LinearEntropyPoolingConstraint{<:AbstractVector,
                                                                                         <:AbstractVector,
                                                                                         <:AbstractVector},
                                                     <:C2_LinearEntropyPoolingConstraint{<:AbstractVector,
                                                                                         <:AbstractVector,
                                                                                         <:AbstractVector}},
                                         sets::DataFrame, strict::Bool = false, args...;
                                         old_mu::Union{Nothing, AbstractVector} = nothing,
                                         old_sigma::Union{Nothing, AbstractMatrix} = nothing,
                                         old_w::AbstractWeights = pweights(range(;
                                                                                 start = 1,
                                                                                 stop = 1,
                                                                                 length = size(pm.X,
                                                                                               1))))
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    B = zero(eltype(pm.X))
    for (group, name, coef) ∈ zip(epvb.group, epvb.name, epvb.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            B += _get_B_entropy_pooling_view_data(epvb, pm, idx, coef, args...;
                                                  old_mu = old_mu, old_sigma = old_sigma,
                                                  old_w = old_w)
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(epvb)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(epvb).")
        end
    end
    return B
end
function _get_A_entropy_pooling_view_data(::C0_LinearEntropyPoolingConstraint,
                                          pm::AbstractPriorModel, idx::AbstractVector,
                                          coef::Real)
    X = view(pm.X, :, idx)
    return coef * X
end
function _get_A_entropy_pooling_view_data(::C1_LinearEntropyPoolingConstraint,
                                          pm::AbstractPriorModel, idx::AbstractVector,
                                          coef::Real)
    X = view(pm.X, :, idx)
    mu = view(pm.mu, idx)
    return coef * (X .- transpose(mu)) .^ 2
end
function _get_A_entropy_pooling_view_data(::C2_LinearEntropyPoolingConstraint{<:Any, <:Any,
                                                                              <:Any,
                                                                              <:SkewnessEntropyPoolingView},
                                          pm::AbstractPriorModel, idx::AbstractVector,
                                          coef::Real)
    X = view(pm.X, :, idx)
    mu = view(pm.mu, idx)
    dsigma = view(diag(pm.sigma), idx)
    return coef * ((X .- transpose(mu)) .^ 3) ./ (dsigma .* sqrt.(dsigma))
end
function _get_A_entropy_pooling_view_data(::C2_LinearEntropyPoolingConstraint{<:Any, <:Any,
                                                                              <:Any,
                                                                              <:KurtosisEntropyPoolingView},
                                          pm::AbstractPriorModel, idx::AbstractVector,
                                          coef::Real)
    X = view(pm.X, :, idx)
    mu = view(pm.mu, idx)
    dsigma = view(diag(pm.sigma), idx)
    return coef * ((X .- transpose(mu)) .^ 4) ./ (dsigma .^ 2)
end
function _get_A_entropy_pooling_view_data(::C4_LinearEntropyPoolingConstraint,
                                          pm::AbstractPriorModel, idx1::AbstractVector,
                                          idx2::AbstractVector, coef::Real)
    X1 = view(pm.X, :, idx1)
    X2 = view(pm.X, :, idx2)
    mu1 = view(pm.mu, idx1)
    mu2 = view(pm.mu, idx2)
    return coef * vec((X1 .- transpose(mu1)) .* (X2 .- transpose(mu2)))
end
function get_A_entropy_pooling_view_data(pm::AbstractPriorModel,
                                         epva::C4_LinearEntropyPoolingConstraint{<:Any,
                                                                                 <:Any,
                                                                                 <:Any,
                                                                                 <:Any,
                                                                                 <:Real},
                                         sets::DataFrame, strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    (; group1, group2, name1, name2, coef) = epva
    A = if !(isnothing(group1) ||
             string(group1) ∉ group_names ||
             isnothing(group2) ||
             string(group2) ∉ group_names)
        idx1 = sets[!, group1] .== name1
        idx2 = sets[!, group2] .== name2
        _get_A_entropy_pooling_view_data(epva, pm, idx1, idx2, coef)
    elseif strict
        throw(ArgumentError("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epva)."))
    else
        @warn("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epva).")
    end
    return A
end
function get_A_entropy_pooling_view_data(pm::AbstractPriorModel,
                                         epva::C4_LinearEntropyPoolingConstraint{<:AbstractVector,
                                                                                 <:AbstractVector,
                                                                                 <:AbstractVector,
                                                                                 <:AbstractVector,
                                                                                 <:AbstractVector},
                                         sets::DataFrame, strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    A = Vector{eltype(pm.X)}(undef, 0)
    for (group1, group2, name1, name2, coef) ∈
        zip(epva.group1, epva.group2, epva.name1, epva.name2, epva.coef)
        if !(isnothing(group1) ||
             string(group1) ∉ group_names ||
             isnothing(group2) ||
             string(group2) ∉ group_names)
            idx1 = sets[!, group1] .== name1
            idx2 = sets[!, group2] .== name2
            append!(A, _get_A_entropy_pooling_view_data(epva, pm, idx1, idx2, coef))
        elseif strict
            throw(ArgumentError("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epva)."))
        else
            @warn("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epva).")
        end
    end
    if !isempty(A)
        A = vec(sum(reshape(A, size(pm.X, 1), :); dims = 2))
    end
    return A
end
function get_A_entropy_pooling_view_data(pm::AbstractPriorModel,
                                         epva::Union{<:C0_LinearEntropyPoolingConstraint{<:Any,
                                                                                         <:Any,
                                                                                         <:Real},
                                                     <:C1_LinearEntropyPoolingConstraint{<:Any,
                                                                                         <:Any,
                                                                                         <:Real},
                                                     <:C2_LinearEntropyPoolingConstraint{<:Any,
                                                                                         <:Any,
                                                                                         <:Real}},
                                         sets::DataFrame, strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    (; group, name, coef) = epva
    A = if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        _get_A_entropy_pooling_view_data(epva, pm, idx, coef)
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(epva)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(epva)")
        Vector{eltype(pm.X)}(undef, 0)
    end
    return A
end
function get_A_entropy_pooling_view_data(pm::AbstractPriorModel,
                                         epva::Union{<:C0_LinearEntropyPoolingConstraint{<:AbstractVector,
                                                                                         <:AbstractVector,
                                                                                         <:AbstractVector},
                                                     <:C1_LinearEntropyPoolingConstraint{<:AbstractVector,
                                                                                         <:AbstractVector,
                                                                                         <:AbstractVector},
                                                     <:C2_LinearEntropyPoolingConstraint{<:AbstractVector,
                                                                                         <:AbstractVector,
                                                                                         <:AbstractVector}},
                                         sets::DataFrame, strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    A = Vector{eltype(pm.X)}(undef, 0)
    for (group, name, coef) ∈ zip(epva.group, epva.name, epva.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            append!(A, _get_A_entropy_pooling_view_data(epva, pm, idx, coef))
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(epva)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(epva).")
        end
    end
    if !isempty(A)
        A = vec(sum(reshape(A, size(pm.X, 1), :); dims = 2))
    end
    return A
end
function entropy_pooling_views(pm::AbstractPriorModel,
                               epvs::Union{<:EntropyPoolingView,
                                           <:AbstractVector{<:EntropyPoolingView}},
                               sets::DataFrame; strict::Bool = false,
                               old_mu::Union{Nothing, AbstractVector} = nothing,
                               old_sigma::Union{Nothing, AbstractMatrix} = nothing,
                               old_w::AbstractWeights = pweights(range(; start = 1,
                                                                       stop = 1,
                                                                       length = size(pm.X,
                                                                                     1))))
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
        B = get_B_entropy_pooling_view_data(pm, epv.B, sets, strict; old_mu = old_mu,
                                            old_sigma = old_sigma, old_w = old_w)

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

    if !isempty(A_ineq)
        A_ineq = transpose(reshape(A_ineq, size(pm.X, 1), :))
    else
        A_ineq = nothing
        B_ineq = nothing
    end
    if !isempty(A_eq)
        A_eq = transpose(reshape(A_eq, size(pm.X, 1), :))
    else
        A_eq = nothing
        B_eq = nothing
    end

    return LinearConstraintModel(;
                                 ineq = PartialLinearConstraintModel(; A = A_ineq,
                                                                     B = B_ineq),
                                 eq = PartialLinearConstraintModel(; A = A_eq, B = B_eq))
end

export entropy_pooling_views, ConstantEntropyPoolingConstraint,
       C0_LinearEntropyPoolingConstraint, C1_LinearEntropyPoolingConstraint,
       SkewnessEntropyPoolingView, KurtosisEntropyPoolingView,
       C2_LinearEntropyPoolingConstraint, C4_LinearEntropyPoolingConstraint,
       EntropyPoolingView, isfixed
