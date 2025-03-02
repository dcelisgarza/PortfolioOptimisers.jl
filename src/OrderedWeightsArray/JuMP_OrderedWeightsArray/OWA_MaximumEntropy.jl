struct OWA_MaximumEntropy{T1 <: Real, T2 <: Real, T3 <: Real,
                          T4 <: Union{Solver, <:AbstractVector{Solver}}} <:
       JuMP_OrderedWeightsArray
    max_phi::T1
    scale_constr::T2
    scale_obj::T3
    solvers::T4
end
function OWA_MaximumEntropy(; max_phi::Real = 0.5, scale_constr::Real = 1.0,
                            scale_obj::Real = 1.0,
                            solvers::Union{Solver, <:AbstractVector{Solver}} = Solver())
    @smart_assert(zero(max_phi) < max_phi < one(max_phi))
    @smart_assert(scale_constr > zero(scale_constr))
    @smart_assert(scale_obj > zero(scale_obj))
    return OWA_MaximumEntropy{typeof(max_phi), typeof(scale_constr), typeof(scale_obj),
                              typeof(solvers)}(max_phi, scale_constr, scale_obj, solvers)
end
function owa_l_moment_crm(method::OWA_MaximumEntropy, weights::AbstractMatrix{<:Real})
    T = size(weights, 1)
    scale_constr = method.scale_constr
    scale_obj = method.scale_obj
    ovec = range(; start = 1, stop = 1, length = T)
    model = owa_model_setup(method, weights)
    theta = model[:theta]
    @variables(model, begin
                   t
                   x[1:T]
               end)
    @constraints(model,
                 begin
                     scale_constr * sum(x) == scale_constr * 1
                     scale_constr * [t; ovec; x] ∈ MOI.RelativeEntropyCone(2 * T + 1)
                     [i = 1:T], scale_constr * [x[i]; theta[i]] ∈ MOI.NormOneCone(2)
                 end)
    @objective(model, Max, -scale_obj * t)
    return owa_model_solve(model, method, weights)
end

export OWA_MaximumEntropy
