abstract type EntropyPoolingViewKind end
struct AbsoluteEntropyPoolingView <: EntropyPoolingViewKind end
struct RelativeEntropyPoolingView <: EntropyPoolingViewKind end
struct EntropyConstraintAtom{T1 <: PartialLinearConstraintAtom,
                             T2 <: Union{<:Real, <:AbstractVector{<:Real}}}
    plca::T1
    cnst::T2
end
function EntropyConstraintAtom(; group = nothing, name = nothing,
                               coef::Union{<:Real, <:AbstractVector{<:Real}} = 1.0,
                               cnst::Union{<:Real, <:AbstractVector{<:Real}} = 0.0)
    plca = PartialLinearConstraintAtom(; group = group, name = name, coef = coef)
    if isa(group, AbstractVector)
        @smart_assert(isa(cnst, AbstractVector) &&
                      !isempty(cnst) &&
                      length(group) == length(cnst))
    else
        @smart_assert(isa(cnst, Real))
    end
    return EntropyConstraintAtom{typeof(plca), typeof(cnst)}(plca, cnst)
end
function Base.getproperty(obj::EntropyConstraintAtom, sym::Symbol)
    return if sym == :group
        obj.plca.group
    elseif sym == :name
        obj.plca.name
    elseif sym == :coef
        obj.plca.coef
    else
        return getfield(obj, sym)
    end
end
struct QuadraticEntropyConstraintAtom{T1, T2, T3, T4, T5,
                                      T6 <: Union{<:Real, <:AbstractVector{<:Real}}}
    group1::T1
    group2::T2
    name1::T3
    name2::T4
    coef::T5
    cnst::T6
end
function QuadraticEntropyConstraintAtom(; group1 = nothing, name1 = nothing,
                                        group2 = nothing, name2 = nothing,
                                        coef::Union{<:Real, <:AbstractVector{<:Real}} = 1.0,
                                        cnst::Union{<:Real, <:AbstractVector{<:Real}} = 0.0)
    group1_flag = isa(group1, AbstractVector)
    group2_flag = isa(group2, AbstractVector)
    name1_flag = isa(name1, AbstractVector)
    name2_flag = isa(name2, AbstractVector)
    coef_flag = isa(coef, AbstractVector)
    cnst_flag = isa(cnst, AbstractVector)
    if any((group1_flag, name1_flag, group2_flag, name2_flag, coef_flag, cnst_flag))
        @smart_assert(all((group1_flag, name1_flag, group2_flag, name2_flag, coef_flag,
                           cnst_flag)))
        @smart_assert(!isempty(group1) &&
                      !isempty(name1) &&
                      !isempty(group2) &&
                      !isempty(name2) &&
                      !isempty(coef) &&
                      !isempty(cnst))
        @smart_assert(length(group1) ==
                      length(name1) ==
                      length(group2) ==
                      length(name2) ==
                      length(coef) ==
                      length(cnst))
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
    return QuadraticEntropyConstraintAtom{typeof(group1), typeof(group2), typeof(name1),
                                          typeof(name2), typeof(coef), typeof(cnst)}(group1,
                                                                                     group2,
                                                                                     name1,
                                                                                     name2,
                                                                                     coef,
                                                                                     cnst)
end
struct CorEntropyPoolingConstraint{T1 <: QuadraticEntropyConstraintAtom,
                                   T2 <: QuadraticEntropyConstraintAtom,
                                   T3 <: ComparisonOperators, T4 <: EntropyPoolingViewKind}
    lhs::T1
    rhs::T2
    comp::T3
    kind::T4
end
struct MeanEntropyPoolingConstraint{T1 <: EntropyConstraintAtom,
                                    T2 <: EntropyConstraintAtom, T3 <: ComparisonOperators,
                                    T4 <: EntropyPoolingViewKind}
    lhs::T1
    rhs::T2
    comp::T3
    kind::T4
end
function MeanEntropyPoolingConstraint(; lhs::EntropyConstraintAtom,
                                      rhs::EntropyConstraintAtom,
                                      comp::ComparisonOperators = LEQ(),
                                      kind::EntropyPoolingViewKind = AbsoluteEntropyPoolingView())
    return MeanEntropyPoolingConstraint{typeof(lhs), typeof(rhs), typeof(comp),
                                        typeof(kind)}(lhs, rhs, comp, kind)
end
function get_mean_entropy_pooling_data(pm::AbstractPriorModel,
                                       lca::EntropyConstraintAtom{<:PartialLinearConstraintAtom{<:Any,
                                                                                                <:Any,
                                                                                                <:Real},
                                                                  <:Real},
                                       kind::EntropyPoolingViewKind, sets::DataFrame;
                                       strict::Bool = false)
    group_names = names(sets)
    (; group, name, coef, cnst) = lca
    if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        A = pm.X[:, idx] * coef
        B = if isa(kind, AbsoluteEntropyPoolingView)
            cnst
        else
            cnst * pm.mu[idx]
        end
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(lca)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(lca)")
    end
    return A, B
end
function get_mean_entropy_pooling_data(pm::AbstractPriorModel,
                                       lca::EntropyConstraintAtom{<:PartialLinearConstraintAtom{<:AbstractVector,
                                                                                                <:AbstractVector,
                                                                                                <:AbstractVector},
                                                                  <:AbstractVector},
                                       kind::EntropyPoolingViewKind, sets::DataFrame;
                                       strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    T = size(pm.X, 1)
    A = Vector{eltype(pm.X)}(undef, 0)
    B = Vector{eltype(pm.X)}(undef, 0)
    for (group, name, coef, cnst) ∈ zip(lca.group, lca.name, lca.coef, lca.cnst)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            append!(A, vec(pm.X[:, idx] * coef))
            append!(B, if isa(kind, AbsoluteEntropyPoolingView)
                        cnst
                    else
                        cnst * vec(pm.mu[idx])
                    end)
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(lca)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(lca).")
        end
    end
    return if isempty(A)
        A, B
    else
        vec(sum(reshape(A, T, :); dims = 2)), sum(B)
    end
end
function get_vol_entropy_pooling_data(pm::AbstractPriorModel,
                                      lca::EntropyConstraintAtom{<:PartialLinearConstraintAtom{<:Any,
                                                                                               <:Any,
                                                                                               <:Real},
                                                                 <:Real},
                                      kind::EntropyPoolingViewKind, sets::DataFrame;
                                      strict::Bool = false)
    group_names = names(sets)
    (; group, name, coef, cnst) = lca
    if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        A = coef * (pm.X[:, idx] .- transpose(pm.mu[idx])) .^ 2
        B = if isa(kind, AbsoluteEntropyPoolingView)
            cnst
        else
            cnst * diag(pm.sigma)[idx]
        end
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(lca)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(lca)")
    end
    return A, B
end
function get_vol_entropy_pooling_data(pm::AbstractPriorModel,
                                      lca::EntropyConstraintAtom{<:PartialLinearConstraintAtom{<:AbstractVector,
                                                                                               <:AbstractVector,
                                                                                               <:AbstractVector},
                                                                 <:AbstractVector},
                                      kind::EntropyPoolingViewKind, sets::DataFrame;
                                      strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    T = size(pm.X, 1)
    A = Vector{eltype(pm.X)}(undef, 0)
    B = Vector{eltype(pm.X)}(undef, 0)
    for (group, name, coef, cnst) ∈ zip(lca.group, lca.name, lca.coef, lca.cnst)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            append!(A, coef * vec((pm.X[:, idx] .- transpose(pm.mu[idx])) .^ 2))
            append!(B, if isa(kind, AbsoluteEntropyPoolingView)
                        cnst
                    else
                        cnst * vec(diag(pm.sigma)[idx])
                    end)
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(lca)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(lca).")
        end
    end
    return if isempty(A)
        A, B
    else
        vec(sum(reshape(A, T, :); dims = 2)), sum(B)
    end
end
function get_skew_kurt_entropy_pooling_data(pm::AbstractPriorModel,
                                            lca::EntropyConstraintAtom{<:PartialLinearConstraintAtom{<:Any,
                                                                                                     <:Any,
                                                                                                     <:Real},
                                                                       <:Real},
                                            sets::DataFrame, skew::Bool = true;
                                            strict::Bool = false)
    d = skew ? 3 : 4
    group_names = names(sets)
    (; group, name, coef, cnst) = lca
    if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        A = coef *
            ((pm.X[:, idx] .- transpose(pm.mu[idx])) ./ sqrt.(diag(pm.sigma)[idx])) .^ d
        B = cnst
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(lca)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(lca)")
    end
    return A, B
end
function get_skew_kurt_entropy_pooling_data(pm::AbstractPriorModel,
                                            lca::EntropyConstraintAtom{<:PartialLinearConstraintAtom{<:AbstractVector,
                                                                                                     <:AbstractVector,
                                                                                                     <:AbstractVector},
                                                                       <:AbstractVector},
                                            sets::DataFrame, skew::Bool = true;
                                            strict::Bool = false)
    d = skew ? 3 : 4
    group_names = names(sets)
    T = size(pm.X, 1)
    A = Vector{eltype(pm.X)}(undef, 0)
    B = Vector{eltype(pm.X)}(undef, 0)
    for (group, name, coef, cnst) ∈ zip(lca.group, lca.name, lca.coef, lca.cnst)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            append!(A,
                    coef *
                    ((pm.X[:, idx] .- transpose(pm.mu[idx])) ./ sqrt.(diag(pm.sigma)[idx]))) .^
            d
            append!(B, cnst)
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(lca)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(lca).")
        end
    end
    return if isempty(A)
        A, B
    else
        vec(sum(reshape(A, T, :); dims = 2)), sum(B)
    end
end
function get_cor_entropy_pooling_data(pm::AbstractPriorModel,
                                      lca::QuadraticEntropyConstraintAtom{<:Any, <:Any,
                                                                          <:Any, <:Any,
                                                                          <:Real, <:Real},
                                      kind::EntropyPoolingViewKind, sets::DataFrame;
                                      strict::Bool = false)
    group_names = names(sets)
    (; group1, name1, group2, name2, coef, cnst) = lca
    if !(isnothing(group1) ||
         string(group1) ∉ group_names ||
         isnothing(group2) ||
         string(group2) ∉ group_names)
        idx1 = sets[!, group1] .== name1
        idx2 = sets[!, group1] .== name2
        A = coef * (pm.X[:, idx1] .- transpose(pm.mu[idx1])) .*
            (pm.X[:, idx2] .- transpose(pm.mu[idx2]))
        B = if isa(kind, AbsoluteEntropyPoolingView)
            cnst
        else
            dsigma = sqrt.(diag(pm.sigma))
            cnst * vec(dsigma[idx1] .* dsigma[idx2])
        end
    elseif strict
        throw(ArgumentError("$(string(group1)) or $(string(group2)) is not in $(group_names).\n$(lca)"))
    else
        @warn("$(string(group1)) or $(string(group2)) is not in $(group_names).\n$(lca)")
    end
    return A, B
end
function get_cor_entropy_pooling_data(pm::AbstractPriorModel,
                                      lca::QuadraticEntropyConstraintAtom{<:AbstractVector,
                                                                          <:AbstractVector,
                                                                          <:AbstractVector,
                                                                          <:AbstractVector,
                                                                          <:AbstractVector},
                                      kind::EntropyPoolingViewKind, sets::DataFrame;
                                      strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    T = size(pm.X, 1)
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
            append!(A,
                    vec(coef * (pm.X[:, idx1] .- transpose(pm.mu[idx1])) .*
                        (pm.X[:, idx2] .- transpose(pm.mu[idx2]))))
            append!(B, if isa(kind, AbsoluteEntropyPoolingView)
                        cnst
                    else
                        dsigma = sqrt.(diag(pm.sigma))
                        cnst * vec(dsigma[idx1] .* dsigma[idx2])
                    end)
        elseif strict
            throw(ArgumentError("$(string(group1)) or $(string(group2)) is not in $(group_names).\n$(lca)."))
        else
            @warn("$(string(group1)) or $(string(group2)) is not in $(group_names).\n$(lca).")
        end
    end
    return if isempty(A)
        A, B
    else
        vec(sum(reshape(A, T, :); dims = 2)), sum(B)
    end
end

export EntropyConstraintAtom, MeanEntropyPoolingConstraint, AbsoluteEntropyPoolingView,
       QuadraticEntropyConstraintAtom, RelativeEntropyPoolingView,
       get_mean_entropy_pooling_data, get_vol_entropy_pooling_data,
       get_skew_kurt_entropy_pooling_data, get_cor_entropy_pooling_data

abstract type EntropyPoolingView end
struct C0_EntropyPoolingView{T1 <: EntropyConstraintAtom, T2 <: EntropyConstraintAtom,
                             T3 <: ComparisonOperators, T4 <: EntropyPoolingViewKind} <:
       EntropyPoolingView
    lhs::T1
    rhs::T2
    comp::T3
    kind::T4
end
function C0_EntropyPoolingView(; lhs::EntropyConstraintAtom, rhs::EntropyConstraintAtom,
                               comp::ComparisonOperators = LEQ(),
                               kind::EntropyPoolingViewKind = AbsoluteEntropyPoolingView())
    return C0_EntropyPoolingView{typeof(lhs), typeof(rhs), typeof(comp), typeof(kind)}(lhs,
                                                                                       rhs,
                                                                                       comp,
                                                                                       kind)
end
struct C1_EntropyPoolingView{T1 <: EntropyConstraintAtom, T2 <: EntropyConstraintAtom,
                             T3 <: ComparisonOperators, T4 <: EntropyPoolingViewKind} <:
       EntropyPoolingView
    lhs::T1
    rhs::T2
    comp::T3
    kind::T4
end
function C1_EntropyPoolingView(; lhs::EntropyConstraintAtom, rhs::EntropyConstraintAtom,
                               comp::ComparisonOperators = LEQ(),
                               kind::EntropyPoolingViewKind = AbsoluteEntropyPoolingView())
    return C1_EntropyPoolingView{typeof(lhs), typeof(rhs), typeof(comp), typeof(kind)}(lhs,
                                                                                       rhs,
                                                                                       comp,
                                                                                       kind)
end
struct C2_EntropyPoolingView{T1 <: EntropyConstraintAtom, T2 <: EntropyConstraintAtom,
                             T3 <: ComparisonOperators, T4 <: EntropyPoolingViewKind} <:
       EntropyPoolingView
    lhs::T1
    rhs::T2
    comp::T3
    kind::T4
end
function C2_EntropyPoolingView(; lhs::EntropyConstraintAtom, rhs::EntropyConstraintAtom,
                               comp::ComparisonOperators = LEQ(),
                               kind::EntropyPoolingViewKind = AbsoluteEntropyPoolingView())
    return C2_EntropyPoolingView{typeof(lhs), typeof(rhs), typeof(comp), typeof(kind)}(lhs,
                                                                                       rhs,
                                                                                       comp,
                                                                                       kind)
end
struct C4_EntropyPoolingView{T1 <: QuadraticEntropyConstraintAtom,
                             T2 <: QuadraticEntropyConstraintAtom,
                             T3 <: ComparisonOperators, T4 <: EntropyPoolingViewKind} <:
       EntropyPoolingView
    lhs::T1
    rhs::T2
    comp::T3
    kind::T4
end
function C4_EntropyPoolingView(; lhs::QuadraticEntropyConstraintAtom,
                               rhs::QuadraticEntropyConstraintAtom,
                               comp::ComparisonOperators = LEQ(),
                               kind::EntropyPoolingViewKind = AbsoluteEntropyPoolingView())
    return C4_EntropyPoolingView{typeof(lhs), typeof(rhs), typeof(comp), typeof(kind)}(lhs,
                                                                                       rhs,
                                                                                       comp,
                                                                                       kind)
end
function get_view_level(::C0_EntropyPoolingView)
    return 0
end
function get_view_level(::C1_EntropyPoolingView)
    return 1
end
function get_view_level(::C2_EntropyPoolingView)
    return 2
end
function get_view_level(::C4_EntropyPoolingView)
    return 4
end
function Base.isless(a::EntropyPoolingView, b::EntropyPoolingView)
    return Base.isless(get_view_level(a), get_view_level(b))
end

export C0_EntropyPoolingView, C1_EntropyPoolingView, C2_EntropyPoolingView,
       C4_EntropyPoolingView
