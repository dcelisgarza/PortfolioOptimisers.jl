struct TurnoverEstimator{T1,T2,T3} <: AbstractEstimator
    w::T1
    val::T2
    default::T3
end
function TurnoverEstimator(;
    w::AbstractVector{<:Real},
    val::Union{<:AbstractDict,<:Pair{<:Any,<:Real},<:AbstractVector{<:Pair{<:Any,<:Real}}},
    default::Real = 0.0,
)
    @argcheck(
        !isempty(w) && !isempty(val) && default >= zero(default),
        AssertionError(
            "The following conditions must hold:\n`w` must be non-empty => $(!isempty(w))\n`val` must be non-empty => $(!isempty(val))\n`default` must be non-negative: $default",
        )
    )
    return TurnoverEstimator(w, val, default)
end
function turnover_constraints(::Nothing, args...; kwargs...)
    return nothing
end
function turnover_constraints(tn::TurnoverEstimator, sets::AssetSets; strict::Bool = false)
    return Turnover(;
        w = tn.w,
        val = estimator_to_val(tn.val, sets, tn.default; strict = strict),
    )
end
struct Turnover{T1,T2} <: AbstractResult
    w::T1
    val::T2
end
function Turnover(;
    w::AbstractVector{<:Real},
    val::Union{<:Real,<:AbstractVector{<:Real}} = 0.0,
)
    @argcheck(!isempty(w), IsEmptyError(non_empty_msg("`w`") * "."))
    if isa(val, AbstractVector)
        @argcheck(
            !isempty(val) &&
            length(val) == length(w) &&
            any(isfinite, val) &&
            all(x -> x >= zero(x), val),
            AssertionError(
                "The following conditions must hold:\n`val` must be non-empty => $(!isempty(val))\n`val` must have the same length as `w` => $(length(val) == length(w))\n`val` must be non-negative and finite => $(all(x -> isfinite(x) && x >= zero(x), val))",
            )
        )
    else
        @argcheck(
            isfinite(val) && val >= zero(eltype(val)),
            DomainError("`val` must be non-negative and finite:\nval => $val")
        )
    end
    return Turnover(w, val)
end
function turnover_constraints(tn::Turnover, args...; kwargs...)
    return tn
end
function turnover_constraints(
    tn::Union{
        <:AbstractVector{<:TurnoverEstimator},
        <:AbstractVector{<:Turnover},
        <:AbstractVector{<:Union{<:TurnoverEstimator,<:Turnover}},
    },
    sets::AssetSets;
    strict::Bool = false,
)
    return turnover_constraints.(tn, Ref(sets); strict = strict)
end
function turnover_view(::Nothing, ::Any)
    return nothing
end
function turnover_view(tn::TurnoverEstimator, i::AbstractVector)
    w = view(tn.w, i)
    return TurnoverEstimator(; w = w, val = tn.val, default = tn.default)
end
function turnover_view(tn::Turnover, i::AbstractVector)
    w = view(tn.w, i)
    val = nothing_scalar_array_view(tn.val, i)
    return Turnover(; w = w, val = val)
end
function turnover_view(tn::AbstractVector{<:Turnover}, i::AbstractVector)
    return turnover_view.(tn, Ref(i))
end
function factory(tn::Turnover, w::AbstractVector)
    return Turnover(; w = w, val = tn.val)
end
Base.iterate(S::Turnover, state = 1) = state > 1 ? nothing : (S, state + 1)

export TurnoverEstimator, Turnover, turnover_constraints
