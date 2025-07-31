struct BuyInThresholdResult{T1 <: Union{<:Real, <:AbstractVector{<:Real}}} <: AbstractResult
    val::T1
end
function BuyInThresholdResult(; val::Union{<:Real, <:AbstractVector{<:Real}})
    if isa(val, Real)
        @smart_assert(isfinite(val) && val >= zero(val))
    elseif isa(val, AbstractVector)
        @smart_assert(all(x -> (isfinite(x) && x >= 0), val))
    end
    return BuyInThresholdResult{typeof(val)}(val)
end
function threshold_view(t::Union{Nothing, <:AbstractDict,
                                 <:AbstractVector{<:Pair{<:Any, <:Real}},
                                 <:AbstractVector{Nothing},
                                 <:AbstractVector{<:AbstractDict},
                                 <:AbstractVector{<:AbstractVector{<:Pair{<:Any, <:Real}}},
                                 <:AbstractVector{<:Union{Nothing, <:AbstractDict,
                                                          <:AbstractVector{<:Pair{<:Any,
                                                                                  <:Real}}}}},
                        ::Any)
    return t
end
function threshold_view(t::BuyInThresholdResult, i::AbstractVector)
    return BuyInThresholdResult(; val = nothing_scalar_array_view(t.val, i))
end
function threshold_view(t::AbstractVector{<:BuyInThresholdResult}, i::AbstractVector)
    return threshold_view.(t, Ref(i))
end
function threshold_constraints(t::Union{Nothing, <:BuyInThresholdResult}, args...;
                               kwargs...)
    return t
end
function threshold_constraints(t::Union{<:AbstractDict,
                                        <:AbstractVector{<:Pair{<:Any, <:Real}}},
                               sets::AssetSets; datatype::DataType = Float64,
                               strict::Bool = false)
    return BuyInThresholdResult(;
                                val = estimator_to_val(t, sets, zero(datatype);
                                                       strict = strict))
end
function threshold_constraints(t::Union{<:AbstractVector{Nothing},
                                        <:AbstractVector{<:AbstractDict},
                                        <:AbstractVector{<:AbstractVector{<:Pair{<:Any,
                                                                                 <:Real}}},
                                        <:AbstractVector{<:Union{Nothing, <:AbstractDict,
                                                                 <:AbstractVector{<:Pair{<:Any,
                                                                                         <:Real}}}}},
                               sets::AssetSets; kwargs...)
    return threshold_constraints.(t, Ref(sets); kwargs...)
end

export BuyInThresholdResult
