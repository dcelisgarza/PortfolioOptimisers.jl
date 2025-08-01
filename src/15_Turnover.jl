struct TurnoverEstimator{T1, T2} <: AbstractEstimator
    w::T1
    val::T2
end
function TurnoverEstimator(; w::AbstractVector{<:Real},
                           val::Union{<:AbstractDict,
                                      <:AbstractVector{<:Pair{<:Any, <:Real}}})
    @smart_assert(!isempty(w))
    @smart_assert(!isempty(val))
    return TurnoverEstimator(w, val)
end
function turnover_constraints(::Nothing, args...; kwargs...)
    return nothing
end
function turnover_constraints(tn::TurnoverEstimator, sets::AssetSets; strict::Bool = false,
                              datatype::DataType = Float64)
    return Turnover(; w = tn.w,
                    val = estimator_to_val(tn.val, sets, zero(datatype); strict = strict))
end
struct Turnover{T1, T2} <: AbstractResult
    w::T1
    val::T2
end
function Turnover(; w::AbstractVector{<:Real},
                  val::Union{<:Real, <:AbstractVector{<:Real}} = 0.0)
    @smart_assert(!isempty(w))
    if isa(val, AbstractVector)
        @smart_assert(!isempty(val))
        @smart_assert(length(val) == length(w))
        @smart_assert(all(isfinite, val) && all(x -> x >= zero(x), val))
    else
        @smart_assert(isfinite(val) && val >= zero(eltype(val)))
    end
    return Turnover(w, val)
end
function turnover_constraints(tn::Turnover, args...; kwargs...)
    return tn
end
function turnover_constraints(tn::Union{<:AbstractVector{<:TurnoverEstimator},
                                        <:AbstractVector{<:Turnover},
                                        <:AbstractVector{<:Union{<:TurnoverEstimator,
                                                                 <:Turnover}}},
                              sets::AssetSets; strict::Bool = false,
                              datatype::DataType = Float64)
    return turnover_constraints.(tn, Ref(sets); strict = strict, datatype = datatype)
end
function turnover_view(::Nothing, ::Any)
    return nothing
end
function turnover_view(tn::TurnoverEstimator, i::AbstractVector)
    w = view(tn.w, i)
    return TurnoverEstimator(; w = w, val = tn.val)
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

export TurnoverEstimator, Turnover, turnover_constraints
