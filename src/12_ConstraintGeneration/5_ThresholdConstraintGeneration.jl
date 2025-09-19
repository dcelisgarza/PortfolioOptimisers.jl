struct BuyInThresholdEstimator{T1} <: AbstractConstraintEstimator
    val::T1
end
function BuyInThresholdEstimator(;
                                 val::Union{<:AbstractDict,
                                            <:Pair{<:AbstractString, <:Real},
                                            <:AbstractVector{<:Pair{<:AbstractString,
                                                                    <:Real}}})
    @argcheck(!isempty(val))
    return BuyInThresholdEstimator(val)
end
struct BuyInThreshold{T1} <: AbstractConstraintResult
    val::T1
end
function BuyInThreshold(; val::Union{<:Real, <:AbstractVector{<:Real}})
    if isa(val, Real)
        @argcheck(isfinite(val) && val >= zero(val))
    elseif isa(val, AbstractVector)
        @argcheck(all(x -> (isfinite(x) && x >= 0), val))
    end
    return BuyInThreshold(val)
end
function threshold_view(t::Union{Nothing, <:BuyInThresholdEstimator}, ::Any)
    return t
end
function threshold_view(t::BuyInThreshold, i::AbstractVector)
    return BuyInThreshold(; val = nothing_scalar_array_view(t.val, i))
end
function threshold_view(t::AbstractVector{<:Union{Nothing, <:BuyInThreshold,
                                                  <:BuyInThresholdEstimator}},
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
function threshold_constraints(bounds::UniformScaledBounds, sets::AssetSets; kwargs...)
    return BuyInThreshold(; val = inv(length(sets.dict[sets.key])))
end
function threshold_constraints(t::AbstractVector{<:Union{Nothing, <:BuyInThresholdEstimator,
                                                         <:BuyInThreshold}},
                               sets::AssetSets; kwargs...)
    return threshold_constraints.(t, Ref(sets); kwargs...)
end

export BuyInThreshold, BuyInThresholdEstimator
