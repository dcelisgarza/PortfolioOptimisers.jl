function threshold_view(t::Union{<:AbstractDict, <:AbstractVector{<:Pair{<:Any, <:Real}}},
                        ::Any)
    return t
end
function threshold_view(t::Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                        i::AbstractVector)
    return nothing_scalar_array_view(t, i)
end
function threshold_constraints(t::Union{Nothing, <:Real, <:AbstractVector{<:Real}}, args...;
                               kwargs...)
    return t
end
function threshold_constraints(t::Union{<:AbstractDict,
                                        <:AbstractVector{<:Pair{<:Any, <:Real}}},
                               sets::AssetSets; datatype::DataType = Float64,
                               strict::Bool = false)
    t = estimator_to_val(t, sets, zero(datatype); strict = strict)
    if isa(t, Real) || isa(t, AbstractVector{<:Real})
        assert_finite_nonnegative_real_or_vec(t)
    end
    return t
end
