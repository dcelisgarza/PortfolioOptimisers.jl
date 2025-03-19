abstract type EntropyPoolingView end
abstract type EntropyConstraintAtom end
abstract type EntropyPoolingViewKind end
abstract type MeanEntropyPoolingViewKind <: EntropyPoolingViewKind end
struct AbsoluteMeanEntropyPoolingView <: MeanEntropyPoolingViewKind end
struct RelativeMeanEntropyPoolingView <: MeanEntropyPoolingViewKind end
abstract type VolatilityEntropyPoolingViewKind <: EntropyPoolingViewKind end
struct AbsoluteVolatilityEntropyPoolingView <: VolatilityEntropyPoolingViewKind end
struct RelativeVolatilityEntropyPoolingView <: VolatilityEntropyPoolingViewKind end
abstract type SkewKurtEntropyPoolingViewKind <: EntropyPoolingViewKind end
struct SkewnessEntropyPoolingView <: SkewKurtEntropyPoolingViewKind end
struct KurtosisEntropyPoolingView <: SkewKurtEntropyPoolingViewKind end
abstract type CorrelationEntropyPoolingViewKind <: EntropyPoolingViewKind end
struct AbsoluteCorrelationEntropyPoolingView <: CorrelationEntropyPoolingViewKind end
struct RelativeCorrelationEntropyPoolingView <: CorrelationEntropyPoolingViewKind end
struct LinearEntropyConstraintAtom{T1 <: PartialLinearConstraintAtom,
                                   T2 <: Union{<:Real, <:AbstractVector{<:Real}}} <:
       EntropyConstraintAtom
    plca::T1
    cnst::T2
end
function LinearEntropyConstraintAtom(; group = nothing, name = nothing,
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
    return LinearEntropyConstraintAtom{typeof(plca), typeof(cnst)}(plca, cnst)
end
function Base.getproperty(obj::LinearEntropyConstraintAtom, sym::Symbol)
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
                                      T6 <: Union{<:Real, <:AbstractVector{<:Real}}} <:
       EntropyConstraintAtom
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
struct C0_EntropyPoolingView{T1 <: LinearEntropyConstraintAtom,
                             T2 <: LinearEntropyConstraintAtom, T3 <: ComparisonOperators,
                             T4 <: MeanEntropyPoolingViewKind} <: EntropyPoolingView
    lhs::T1
    rhs::T2
    comp::T3
    kind::T4
end
function C0_EntropyPoolingView(;
                               lhs::LinearEntropyConstraintAtom = LinearEntropyConstraintAtom(),
                               rhs::LinearEntropyConstraintAtom = LinearEntropyConstraintAtom(),
                               comp::ComparisonOperators = LEQ(),
                               kind::MeanEntropyPoolingViewKind = AbsoluteMeanEntropyPoolingView())
    return C0_EntropyPoolingView{typeof(lhs), typeof(rhs), typeof(comp), typeof(kind)}(lhs,
                                                                                       rhs,
                                                                                       comp,
                                                                                       kind)
end
struct C1_EntropyPoolingView{T1 <: LinearEntropyConstraintAtom,
                             T2 <: LinearEntropyConstraintAtom, T3 <: ComparisonOperators,
                             T4 <: VolatilityEntropyPoolingViewKind} <: EntropyPoolingView
    lhs::T1
    rhs::T2
    comp::T3
    kind::T4
end
function C1_EntropyPoolingView(;
                               lhs::LinearEntropyConstraintAtom = LinearEntropyConstraintAtom(),
                               rhs::LinearEntropyConstraintAtom = LinearEntropyConstraintAtom(),
                               comp::ComparisonOperators = LEQ(),
                               kind::VolatilityEntropyPoolingViewKind = AbsoluteVolatilityEntropyPoolingView())
    return C1_EntropyPoolingView{typeof(lhs), typeof(rhs), typeof(comp), typeof(kind)}(lhs,
                                                                                       rhs,
                                                                                       comp,
                                                                                       kind)
end
struct C2_EntropyPoolingView{T1 <: LinearEntropyConstraintAtom,
                             T2 <: LinearEntropyConstraintAtom, T3 <: ComparisonOperators,
                             T4 <: EntropyPoolingViewKind} <: EntropyPoolingView
    lhs::T1
    rhs::T2
    comp::T3
    kind::T4
end
function C2_EntropyPoolingView(;
                               lhs::LinearEntropyConstraintAtom = LinearEntropyConstraintAtom(),
                               rhs::LinearEntropyConstraintAtom = LinearEntropyConstraintAtom(),
                               comp::ComparisonOperators = LEQ(),
                               kind::SkewKurtEntropyPoolingViewKind = SkewnessEntropyPoolingView())
    return C2_EntropyPoolingView{typeof(lhs), typeof(rhs), typeof(comp), typeof(kind)}(lhs,
                                                                                       rhs,
                                                                                       comp,
                                                                                       kind)
end
struct C4_EntropyPoolingView{T1 <: QuadraticEntropyConstraintAtom,
                             T2 <: QuadraticEntropyConstraintAtom,
                             T3 <: ComparisonOperators,
                             T4 <: CorrelationEntropyPoolingViewKind} <: EntropyPoolingView
    lhs::T1
    rhs::T2
    comp::T3
    kind::T4
end
function C4_EntropyPoolingView(;
                               lhs::QuadraticEntropyConstraintAtom = QuadraticEntropyConstraintAtom(),
                               rhs::QuadraticEntropyConstraintAtom = QuadraticEntropyConstraintAtom(),
                               comp::ComparisonOperators = LEQ(),
                               kind::CorrelationEntropyPoolingViewKind = AbsoluteCorrelationEntropyPoolingView())
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
    return coef * (pm.X[:, idx1] .- transpose(pm.mu[idx1])) .*
           (pm.X[:, idx2] .- transpose(pm.mu[idx2])), [cnst]
end
function A_B_entropy_pooling_stats(::RelativeCorrelationEntropyPoolingView,
                                   pm::AbstractPriorModel, idx1::AbstractVector,
                                   idx2::AbstractVector, coef::Real, cnst::Real)
    dsigma = diag(pm.sigma)
    B = cnst * vec(sqrt.(dsigma[idx1]) .* sqrt.(dsigma[idx2]))
    return coef * (pm.X[:, idx1] .- transpose(pm.mu[idx1])) .*
           (pm.X[:, idx2] .- transpose(pm.mu[idx2])), B
end
function get_entropy_pooling_view_data(pm::AbstractPriorModel,
                                       lca::LinearEntropyConstraintAtom{<:PartialLinearConstraintAtom{<:Any,
                                                                                                      <:Any,
                                                                                                      <:Real},
                                                                        <:Real},
                                       kind::EntropyPoolingViewKind, sets::DataFrame,
                                       strict::Bool = false)
    group_names = names(sets)
    (; group, name, coef, cnst) = lca
    A, B = if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        A_B_entropy_pooling_stats(kind, pm, idx, coef, cnst)
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(lca)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(lca)")
        Vector{eltype(pm.X)}(undef, 0), Vector{eltype(pm.X)}(undef, 0)
    end
    return A, B
end
function get_entropy_pooling_view_data(pm::AbstractPriorModel,
                                       lca::LinearEntropyConstraintAtom{<:PartialLinearConstraintAtom{<:AbstractVector,
                                                                                                      <:AbstractVector,
                                                                                                      <:AbstractVector},
                                                                        <:AbstractVector},
                                       kind::EntropyPoolingViewKind, sets::DataFrame,
                                       strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    A = Vector{eltype(pm.X)}(undef, 0)
    B = Vector{eltype(pm.X)}(undef, 0)
    for (group, name, coef, cnst) ∈ zip(lca.group, lca.name, lca.coef, lca.cnst)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            _A, _B = A_B_entropy_pooling_stats(kind, pm, idx, coef, cnst)
            append!(A, _A)
            append!(B, _B)
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(lca)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(lca).")
        end
    end
    if !isempty(A)
        A = vec(sum(reshape(A, size(pm.X, 1), :); dims = 2))
    end
    return A, B
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
        idx2 = sets[!, group1] .== name2
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
        lhs_A, lhs_B = get_entropy_pooling_view_data(pm, lc.lhs, lc.kind, sets, strict)
        rhs_A, rhs_B = get_entropy_pooling_view_data(pm, lc.rhs, lc.kind, sets, strict)

        lhs_flag = isempty(lhs_A) || all(iszero.(lhs_A))
        rhs_flag = isempty(rhs_A) || all(iszero.(rhs_A))

        if lhs_flag && rhs_flag
            continue
        end

        d, flag_ineq = comparison_sign_ineq_flag(lc.comp)
        rlhs_A, rlhs_B = if lhs_flag
            -rhs_A * d, -lhs_B * d
        elseif rhs_flag
            lhs_A * d, lhs_B * d
        else
            (lhs_A .- rhs_A) * d, (rhs_B .- lhs_B) * d
        end

        if flag_ineq
            append!(A_ineq, rlhs_A)
            append!(B_ineq, rlhs_B)
        else
            append!(A_eq, rlhs_A)
            append!(B_eq, rlhs_B)
        end
    end

    if !isempty(A_ineq)
        A_ineq = transpose(reshape(A_ineq, size(pm.X, 1), :))
        A_ineq = convert.(typeof(promote(A_ineq...)[1]), A_ineq)
        B_ineq = convert.(typeof(promote(B_ineq...)[1]), B_ineq)
    else
        A_ineq = nothing
        B_ineq = nothing
    end
    if !isempty(A_eq)
        A_eq = transpose(reshape(A_eq, size(pm.X, 1), :))
        A_eq = convert.(typeof(promote(A_eq...)[1]), A_eq)
        B_eq = convert.(typeof(promote(B_eq...)[1]), B_eq)
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
