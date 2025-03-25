struct OWA_MinimumSquareDistance{T1 <: Real, T2 <: Real, T3 <: Real,
                                 T4 <: Union{Solver, <:AbstractVector{Solver}}} <:
       JuMP_OrderedWeightsArray
    max_phi::T1
    sc::T2
    so::T3
    solvers::T4
end
function OWA_MinimumSquareDistance(; max_phi::Real = 0.5, sc::Real = 1.0, so::Real = 1.0,
                                   solvers::Union{Solver, <:AbstractVector{Solver}} = Solver())
    if isa(solvers, AbstractVector)
        @smart_assert(!isempty(solvers))
    end
    @smart_assert(zero(max_phi) < max_phi < one(max_phi))
    @smart_assert(sc > zero(sc))
    @smart_assert(so > zero(so))
    return OWA_MinimumSquareDistance{typeof(max_phi), typeof(sc), typeof(so),
                                     typeof(solvers)}(max_phi, sc, so, solvers)
end
function owa_l_moment_crm(method::OWA_MinimumSquareDistance,
                          weights::AbstractMatrix{<:Real})
    sc = method.sc
    so = method.so
    model = owa_model_setup(method, weights)
    theta = model[:theta]
    @variable(model, t)
    @constraint(model, sc * [t; (theta[2:end] .- theta[1:(end - 1)])] ∈ SecondOrderCone())
    @objective(model, Min, so * t)
    return owa_model_solve(model, method, weights)
end

export OWA_MinimumSquareDistance
