struct BuyInThresholdEstimator{T1 <: Union{<:AbstractDict,
                                           <:AbstractVector{<:Pair{<:Any, <:Real}}}} <:
       AbstractEstimator
    val::T1
end
function BuyInThresholdEstimator(;
                                 val::Union{<:AbstractDict,
                                            <:AbstractVector{<:Pair{<:Any, <:Real}}})
    @smart_assert(!isempty(val))
    return BuyInThresholdEstimator{typeof(val)}(val)
end
struct BuyInThreshold{T1 <: Union{<:Real, <:AbstractVector{<:Real}}} <: AbstractResult
    val::T1
end
function BuyInThreshold(; val::Union{<:Real, <:AbstractVector{<:Real}})
    if isa(val, Real)
        @smart_assert(isfinite(val) && val >= zero(val))
    elseif isa(val, AbstractVector)
        @smart_assert(all(x -> (isfinite(x) && x >= 0), val))
    end
    return BuyInThreshold{typeof(val)}(val)
end
function threshold_view(t::Union{Nothing, <:BuyInThresholdEstimator,
                                 <:AbstractVector{<:Union{Nothing,
                                                          <:BuyInThresholdEstimator}}},
                        ::Any)
    return t
end
function threshold_view(t::BuyInThreshold, i::AbstractVector)
    return BuyInThreshold(; val = nothing_scalar_array_view(t.val, i))
end
function threshold_view(t::AbstractVector{<:Union{Nothing, <:BuyInThreshold}},
                        i::AbstractVector)
    return threshold_view.(t, Ref(i))
end
function threshold_constraints(t::Union{Nothing, <:BuyInThreshold}, args...; kwargs...)
    return t
end
function threshold_constraints(t::BuyInThresholdEstimator, sets::AssetSets;
                               datatype::DataType = Float64, strict::Bool = false)
    return BuyInThreshold(;
                          val = estimator_to_val(t.val, sets, zero(datatype);
                                                 strict = strict))
end
function threshold_constraints(t::AbstractVector{<:Union{Nothing, <:BuyInThresholdEstimator,
                                                         <:BuyInThreshold}},
                               sets::AssetSets; kwargs...)
    return threshold_constraints.(t, Ref(sets); kwargs...)
end

#! Start: delete
function threshold_constraints(t::Union{<:AbstractDict,
                                        <:AbstractVector{<:Pair{<:Any, <:Real}}},
                               sets::AssetSets; datatype::DataType = Float64,
                               strict::Bool = false)
    return BuyInThreshold(;
                          val = estimator_to_val(t, sets, zero(datatype); strict = strict))
end
function threshold_constraints(t::Union{<:AbstractVector{<:AbstractDict},
                                        <:AbstractVector{<:AbstractVector{<:Pair{<:Any,
                                                                                 <:Real}}},
                                        <:AbstractVector{<:Union{Nothing, <:AbstractDict,
                                                                 <:AbstractVector{<:Pair{<:Any,
                                                                                         <:Real}}}}},
                               sets::AssetSets; kwargs...)
    return threshold_constraints.(t, Ref(sets); kwargs...)
end
function threshold_view(t::Union{<:AbstractDict, <:AbstractVector{<:Pair{<:Any, <:Real}},
                                 <:AbstractVector{Nothing},
                                 <:AbstractVector{<:AbstractDict},
                                 <:AbstractVector{<:AbstractVector{<:Pair{<:Any, <:Real}}},
                                 <:AbstractVector{<:Union{Nothing, <:AbstractDict,
                                                          <:AbstractVector{<:Pair{<:Any,
                                                                                  <:Real}}}}},
                        ::Any)
    return t
end
#! Stop: delete

export BuyInThreshold
