struct NearOptimalCentering{T1 <: Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}},
                            T2 <: ObjectiveFunction, T3 <: JuMPOptimiser, T4 <: Real,
                            T5 <: Union{Nothing, <:AbstractVector},
                            T6 <: Union{Nothing, <:AbstractVector},
                            T7 <: Union{Nothing, <:AbstractVector},
                            T8 <: Union{Nothing, <:AbstractVector},
                            T9 <: Union{Nothing, <:AbstractVector},
                            T10 <: Union{Nothing, <:AbstractVector}} <:
       JuMPOptimisationEstimator
    r::T1
    obj::T2
    opt::T3
    bins::T4
    w_min::T5
    w_min_ini::T6
    w_opt::T7
    w_opt_ini::T8
    w_max::T9
    w_max_ini::T10
end
function NearOptimalCentering(;
                              r::Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}} = Variance(),
                              obj::ObjectiveFunction = MinimumRisk(),
                              opt::JuMPOptimiser = JuMPOptimiser(), bins::Real = 20,
                              w_min::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                              w_min_ini::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                              w_opt::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                              w_opt_ini::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                              w_max::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                              w_max_ini::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    if isa(r, AbstractVector)
        @smart_assert(!isempty(r))
    end
    @smart_assert(isfinite(bins) && bins > 0)
    return NearOptimalCentering{typeof(r), typeof(obj), typeof(opt), typeof(bins),
                                typeof(w_min), typeof(w_min_ini), typeof(w_opt),
                                typeof(w_opt_ini), typeof(w_max), typeof(w_max_ini)}(r, obj,
                                                                                     opt,
                                                                                     w_min,
                                                                                     w_min_ini,
                                                                                     w_opt,
                                                                                     w_opt_ini,
                                                                                     w_max,
                                                                                     w_max_ini)
end