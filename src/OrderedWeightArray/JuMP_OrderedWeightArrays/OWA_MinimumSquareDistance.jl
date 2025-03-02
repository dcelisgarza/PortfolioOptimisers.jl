struct OWA_MinimumSquareDistance{T1 <: Real, T2 <: Real, T3 <: Real,
                                 T4 <: Union{Solver, <:AbstractVector{Solver}}} <:
       JuMP_OrderedWeightArrays
    max_phi::T1
    scale_constr::T2
    scale_obj::T3
    solvers::T4
end
function OWA_MinimumSquareDistance(; max_phi::Real = 0.5, scale_constr::Real = 1.0,
                                   scale_obj::Real = 1.0,
                                   solvers::Union{Solver, <:AbstractVector{Solver}} = Solver())
    @smart_assert(zero(max_phi) < max_phi < one(max_phi))
    @smart_assert(scale_constr > zero(scale_constr))
    @smart_assert(scale_obj > zero(scale_obj))
    return OWA_MinimumSquareDistance{typeof(max_phi), typeof(scale_constr),
                                     typeof(scale_obj), typeof(solvers)}(max_phi,
                                                                         scale_constr,
                                                                         scale_obj, solvers)
end
function owa_l_moment_crm(method::OWA_MinimumSquareDistance,
                          weights::AbstractMatrix{<:Real})
    scale_constr = method.scale_constr
    scale_obj = method.scale_obj
    model = owa_model_setup(method, weights)
    theta = model[:theta]
    @variable(model, t)
    @constraint(model,
                [scale_constr * t; scale_constr * (theta[2:end] .- theta[1:(end - 1)])] ∈
                SecondOrderCone())
    @objective(model, Min, scale_obj * t)
    return owa_model_solve(model, method, weights)
end

export OWA_MinimumSquareDistance
