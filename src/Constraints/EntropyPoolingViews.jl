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
                                         T3 <: Union{<:Real, <:AbstractVector{<:Real}}} <:
       LinearEntropyPoolingConstraint
    group::T1
    name::T2
    coef::T3
end
function C1_LinearEntropyPoolingConstraint(; group = nothing, name = nothing,
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
    return C1_LinearEntropyPoolingConstraint{typeof(group), typeof(name), typeof(coef)}(group,
                                                                                        name,
                                                                                        coef)
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
                                         T5 <: Union{<:Real, <:AbstractVector{<:Real}}}
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
function A_B_entropy_pooling_stats(::AbsoluteMeanEntropyPoolingView, pm::AbstractPriorModel,
                                   idx::AbstractVector, coef::Real, cnst::Real)
    return pm.X[:, idx] * coef, [cnst]
end
function A_B_entropy_pooling_stats(::RelativeMeanEntropyPoolingView, pm::AbstractPriorModel,
                                   idx::AbstractVector, coef::Real, cnst::Real)
    return pm.X[:, idx] * coef, cnst * pm.mu[idx]
end
function A_B_entropy_pooling_stats(::AbsoluteVolatilityEntropyPoolingView,
                                   pm::AbstractPriorModel, idx::AbstractVector, coef::Real,
                                   cnst::Real)
    return coef * (pm.X[:, idx] .- transpose(pm.mu[idx])) .^ 2, [cnst]
end
function A_B_entropy_pooling_stats(::RelativeVolatilityEntropyPoolingView,
                                   pm::AbstractPriorModel, idx::AbstractVector, coef::Real,
                                   cnst::Real)
    return coef * (pm.X[:, idx] .- transpose(pm.mu[idx])) .^ 2, cnst * diag(pm.sigma)[idx]
end
function A_B_entropy_pooling_stats(::SkewnessEntropyPoolingView, pm::AbstractPriorModel,
                                   idx::AbstractVector, coef::Real, cnst::Real)
    return coef *
           ((pm.X[:, idx] .- transpose(pm.mu[idx])) ./ sqrt.(diag(pm.sigma)[idx])) .^ 3,
           [cnst]
end
function A_B_entropy_pooling_stats(::KurtosisEntropyPoolingView, pm::AbstractPriorModel,
                                   idx::AbstractVector, coef::Real, cnst::Real)
    return coef *
           ((pm.X[:, idx] .- transpose(pm.mu[idx])) .^ 4 ./ (diag(pm.sigma)[idx] .^ 2)),
           [cnst]
end
function A_B_entropy_pooling_stats(::AbsoluteCorrelationEntropyPoolingView,
                                   pm::AbstractPriorModel, idx1::AbstractVector,
                                   idx2::AbstractVector, coef::Real, cnst::Real)
    return coef * vec((pm.X[:, idx1] .- transpose(pm.mu[idx1])) .*
                      (pm.X[:, idx2] .- transpose(pm.mu[idx2]))), [cnst]
end
function A_B_entropy_pooling_stats(::RelativeCorrelationEntropyPoolingView,
                                   pm::AbstractPriorModel, idx1::AbstractVector,
                                   idx2::AbstractVector, coef::Real, cnst::Real)
    dsigma = diag(pm.sigma)
    B = cnst * vec(sqrt.(dsigma[idx1]) .* sqrt.(dsigma[idx2]))
    return coef * vec((pm.X[:, idx1] .- transpose(pm.mu[idx1])) .*
                      (pm.X[:, idx2] .- transpose(pm.mu[idx2]))), B
end
function _get_B_entropy_pooling_view_data(::AbstractPriorModel,
                                          lcb::ConstantEntropyPoolingConstraint,
                                          ::DataFrame, ::Bool)
    return lcb.coef
end
function _get_B_entropy_pooling_view_data(::C0_LinearEntropyPoolingConstraint,
                                          pm::AbstractPriorModel, idx::AbstractVector,
                                          coef::Real)
    return coef * sum(pm.mu[idx])
end
function _get_B_entropy_pooling_view_data(::C1_LinearEntropyPoolingConstraint,
                                          pm::AbstractPriorModel, idx::AbstractVector,
                                          coef::Real)
    return coef * sum(diag(pm.sigma)[idx])
end
function _get_B_entropy_pooling_view_data(::C4_LinearEntropyPoolingConstraint,
                                          pm::AbstractPriorModel, idx1::AbstractVector,
                                          idx2::AbstractVector, coef::Real)
    dsigma = diag(pm.sigma)
    return coef * sum(sqrt.(dsigma[idx1]) .* sqrt.(dsigma[idx2]))
end
function get_B_entropy_pooling_view_data(pm::AbstractPriorModel,
                                         lcb::C4_LinearEntropyPoolingConstraint{<:Any,
                                                                                <:Any,
                                                                                <:Real},
                                         sets::DataFrame, strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    (; group1, group2, name1, name2, coef) = lcb
    B = if !(isnothing(group1) ||
             string(group1) ∉ group_names ||
             isnothing(group2) ||
             string(group2) ∉ group_names)
        idx1 = sets[!, group1] .== name1
        idx2 = sets[!, group1] .== name2
        _get_B_entropy_pooling_view_data(lcb, pm, idx1, idx2, coef)
    elseif strict
        throw(ArgumentError("$(string(group1)) or $(string(group2)) are not in $(group_names).\n$(lcb)"))
    else
        @warn("$(string(group1)) or $(string(group2)) are not in $(group_names).\n$(lcb)")
        zero(eltype(pm.X))
    end
    return B
end
function get_B_entropy_pooling_view_data(pm::AbstractPriorModel,
                                         lcb::C4_LinearEntropyPoolingConstraint{<:AbstractVector,
                                                                                <:AbstractVector,
                                                                                <:AbstractVector},
                                         sets::DataFrame, strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    B = zero(eltype(pm.X))
    for (group1, group2, name1, name2, coef) ∈
        zip(lcb.group1, lcb.group2, lcb.name1, lcb.name2, lcb.coef)
        if !(isnothing(group1) ||
             string(group1) ∉ group_names ||
             isnothing(group2) ||
             string(group2) ∉ group_names)
            idx1 = sets[!, group1] .== name1
            idx2 = sets[!, group1] .== name2
            B += _get_B_entropy_pooling_view_data(lcb, pm, idx1, idx2, coef)
        elseif strict
            throw(ArgumentError("$(string(group1)) or $(string(group2)) are not in $(group_names).\n$(lcb)."))
        else
            @warn("$(string(group1)) or $(string(group2)) are not in $(group_names).\n$(lcb).")
        end
    end
    return B
end
function get_B_entropy_pooling_view_data(pm::AbstractPriorModel,
                                         lcb::Union{<:C0_LinearEntropyPoolingConstraint{<:Any,
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
    (; group, name, coef) = lcb
    B = if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        _get_B_entropy_pooling_view_data(lcb, pm, idx, coef)
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(lcb)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(lcb)")
        zero(eltype(pm.X))
    end
    return B
end
function get_B_entropy_pooling_view_data(pm::AbstractPriorModel,
                                         lcb::Union{<:C0_LinearEntropyPoolingConstraint{<:AbstractVector,
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
    B = zero(eltype(pm.X))
    for (group, name, coef) ∈ zip(lcb.group, lcb.name, lcb.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            B += _get_B_entropy_pooling_view_data(lcb, pm, idx, coef)
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(lcb)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(lcb).")
        end
    end
    return B
end
function _get_A_entropy_pooling_view_data(::C0_LinearEntropyPoolingConstraint,
                                          pm::AbstractPriorModel, idx::AbstractVector,
                                          coef::Real)
    return coef * pm.X[:, idx]
end
function _get_A_entropy_pooling_view_data(::C1_LinearEntropyPoolingConstraint,
                                          pm::AbstractPriorModel, idx::AbstractVector,
                                          coef::Real)
    return coef * (pm.X[:, idx] .- transpose(pm.mu[idx])) .^ 2
end
function _get_A_entropy_pooling_view_data(::C2_LinearEntropyPoolingConstraint{<:Any, <:Any,
                                                                              <:Any,
                                                                              <:SkewnessEntropyPoolingView},
                                          pm::AbstractPriorModel, idx::AbstractVector,
                                          coef::Real)
    dsigma = diag(pm.sigma)[idx]
    return coef * ((pm.X[:, idx] .- transpose(pm.mu[idx])) .^ 3) ./
           (dsigma .* sqrt.(dsigma))
end
function _get_A_entropy_pooling_view_data(::C2_LinearEntropyPoolingConstraint{<:Any, <:Any,
                                                                              <:Any,
                                                                              <:KurtosisEntropyPoolingView},
                                          pm::AbstractPriorModel, idx::AbstractVector,
                                          coef::Real)
    dsigma = diag(pm.sigma)[idx]
    return coef * ((pm.X[:, idx] .- transpose(pm.mu[idx])) .^ 4) ./ (dsigma .^ 2)
end
function _get_A_entropy_pooling_view_data(::C4_LinearEntropyPoolingConstraint,
                                          pm::AbstractPriorModel, idx1::AbstractVector,
                                          idx2::AbstractVector, coef::Real)
    return coef * vec((pm.X[:, idx1] .- transpose(pm.mu[idx1])) .*
                      (pm.X[:, idx2] .- transpose(pm.mu[idx2])))
end
function get_A_entropy_pooling_view_data(pm::AbstractPriorModel,
                                         lca::C4_LinearEntropyPoolingConstraint{<:Any,
                                                                                <:Any,
                                                                                <:Real},
                                         sets::DataFrame, strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    (; group1, group2, name1, name2, coef) = lca
    A = if !(isnothing(group1) ||
             string(group1) ∉ group_names ||
             isnothing(group2) ||
             string(group2) ∉ group_names)
        idx1 = sets[!, group1] .== name1
        idx2 = sets[!, group2] .== name2
        _get_A_entropy_pooling_view_data(lca, pm, idx1, idx2, coef)
    elseif strict
        throw(ArgumentError("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(lca)."))
    else
        @warn("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(lca).")
    end
    return A
end
function get_A_entropy_pooling_view_data(pm::AbstractPriorModel,
                                         lca::C4_LinearEntropyPoolingConstraint{<:AbstractVector,
                                                                                <:AbstractVector,
                                                                                <:AbstractVector},
                                         sets::DataFrame, strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    A = Vector{eltype(pm.X)}(undef, 0)
    for (group1, group2, name1, name2, coef1, coef2) ∈
        zip(lca.group1, lca.group2, lca.name1, lca.name2, lca.coef)
        if !(isnothing(group1) ||
             string(group1) ∉ group_names ||
             isnothing(group2) ||
             string(group2) ∉ group_names)
            idx1 = sets[!, group1] .== name1
            idx2 = sets[!, group2] .== name2
            append!(A, _get_A_entropy_pooling_view_data(lca, pm, idx1, idx2, coef))
        elseif strict
            throw(ArgumentError("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(lca)."))
        else
            @warn("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(lca).")
        end
    end
    if !isempty(A)
        A = vec(sum(reshape(A, size(pm.X, 1), :); dims = 2))
    end
    return A
end
function get_A_entropy_pooling_view_data(pm::AbstractPriorModel,
                                         lca::Union{<:C0_LinearEntropyPoolingConstraint{<:Any,
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
    (; group, name, coef) = lca
    A = if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        _get_A_entropy_pooling_view_data(lca, pm, idx, coef)
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(lca)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(lca)")
        Vector{eltype(pm.X)}(undef, 0)
    end
    return A
end
function get_A_entropy_pooling_view_data(pm::AbstractPriorModel,
                                         lca::Union{<:C0_LinearEntropyPoolingConstraint{<:AbstractVector,
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
    for (group, name, coef) ∈ zip(lca.group, lca.name, lca.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            append!(A, _get_A_entropy_pooling_view_data(lca, pm, idx, coef))
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(lca)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(lca).")
        end
    end
    if !isempty(A)
        A = vec(sum(reshape(A, size(pm.X, 1), :); dims = 2))
    end
    return A
end
function get_entropy_pooling_view_data(pm::AbstractPriorModel,
                                       lca::QuadraticEntropyConstraintAtom{<:Any, <:Any,
                                                                           <:Any, <:Any,
                                                                           <:Real, <:Real},
                                       kind::CorrelationEntropyPoolingViewKind,
                                       sets::DataFrame, strict::Bool = false)
    group_names = names(sets)
    (; group1, name1, group2, name2, coef, cnst) = lca
    A, B = if !(isnothing(group1) ||
                string(group1) ∉ group_names ||
                isnothing(group2) ||
                string(group2) ∉ group_names)
        idx1 = sets[!, group1] .== name1
        idx2 = sets[!, group2] .== name2
        A_B_entropy_pooling_stats(kind, pm, idx1, idx2, coef, cnst)
    elseif strict
        throw(ArgumentError("$(string(group1)) or $(string(group2)) is not in $(group_names).\n$(lca)"))
    else
        @warn("$(string(group1)) or $(string(group2)) is not in $(group_names).\n$(lca)")
        Vector{eltype(pm.X)}(undef, 0), Vector{eltype(pm.X)}(undef, 0)
    end
    return A, B
end
function get_entropy_pooling_view_data(pm::AbstractPriorModel,
                                       lca::QuadraticEntropyConstraintAtom{<:AbstractVector,
                                                                           <:AbstractVector,
                                                                           <:AbstractVector,
                                                                           <:AbstractVector,
                                                                           <:AbstractVector},
                                       kind::CorrelationEntropyPoolingViewKind,
                                       sets::DataFrame, strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    A = Vector{eltype(pm.X)}(undef, 0)
    B = Vector{eltype(pm.X)}(undef, 0)
    for (group1, name1, group2, name2, coef, cnst) ∈
        zip(lca.group1, lca.name1, lca.group2, lca.name2, lca.coef, lca.cnst)
        if !(isnothing(group1) ||
             string(group1) ∉ group_names ||
             isnothing(group2) ||
             string(group2) ∉ group_names)
            idx1 = sets[!, group1] .== name1
            idx2 = sets[!, group1] .== name2
            _A, _B = A_B_entropy_pooling_stats(kind, pm, idx1, idx2, coef, cnst)
            append!(A, vec(_A))
            append!(B, vec(_B))
        elseif strict
            throw(ArgumentError("$(string(group1)) or $(string(group2)) is not in $(group_names).\n$(lca)."))
        else
            @warn("$(string(group1)) or $(string(group2)) is not in $(group_names).\n$(lca).")
        end
    end
    if !isempty(A)
        A = vec(sum(reshape(A, size(pm.X, 1), :); dims = 2))
    end
    return A, B
end
function entropy_pooling_views(pm::AbstractPriorModel,
                               lcs::Union{<:EntropyPoolingView,
                                          <:AbstractVector{<:EntropyPoolingView}},
                               sets::DataFrame; strict::Bool = false)
    if isa(lcs, AbstractVector)
        @smart_assert(!isempty(lcs))
    end
    @smart_assert(!isempty(sets))
    A_ineq = Vector{eltype(pm.X)}(undef, 0)
    B_ineq = Vector{eltype(pm.X)}(undef, 0)
    A_eq = Vector{eltype(pm.X)}(undef, 0)
    B_eq = Vector{eltype(pm.X)}(undef, 0)
    for lc ∈ lcs
        A, B = get_entropy_pooling_view_data(pm, lc, sets, strict)

        lhs_flag = isempty(A)

        if lhs_flag
            continue
        end

        d, flag_ineq = comparison_sign_ineq_flag(lc.comp)
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

export LinearEntropyConstraintAtom, QuadraticEntropyConstraintAtom, C0_EntropyPoolingView,
       C1_EntropyPoolingView, C2_EntropyPoolingView, C4_EntropyPoolingView,
       entropy_pooling_views, AbsoluteMeanEntropyPoolingView,
       RelativeMeanEntropyPoolingView, AbsoluteVolatilityEntropyPoolingView,
       RelativeVolatilityEntropyPoolingView, SkewnessEntropyPoolingView,
       KurtosisEntropyPoolingView, AbsoluteCorrelationEntropyPoolingView,
       RelativeCorrelationEntropyPoolingView
