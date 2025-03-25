struct OWA_MaximumEntropy{T1 <: Real, T2 <: Real, T3 <: Real,
                          T4 <: Union{Solver, <:AbstractVector{Solver}}} <:
       JuMP_OrderedWeightsArray
    max_phi::T1
    sc::T2
    so::T3
    solvers::T4
end
function OWA_MaximumEntropy(; max_phi::Real = 0.5, sc::Real = 1.0, so::Real = 1.0,
                            solvers::Union{Solver, <:AbstractVector{Solver}} = Solver())
    if isa(solvers, AbstractVector)
        @smart_assert(!isempty(solvers))
    end
    @smart_assert(zero(max_phi) < max_phi < one(max_phi))
    @smart_assert(sc > zero(sc))
    @smart_assert(so > zero(so))
    return OWA_MaximumEntropy{typeof(max_phi), typeof(sc), typeof(so), typeof(solvers)}(max_phi,
                                                                                        sc,
                                                                                        so,
                                                                                        solvers)
end
function owa_l_moment_crm(method::OWA_MaximumEntropy, weights::AbstractMatrix{<:Real})
    T = size(weights, 1)
    sc = method.sc
    so = method.so
    ovec = range(; start = 1, stop = 1, length = T)
    model = owa_model_setup(method, weights)
    theta = model[:theta]
    @variables(model, begin
                   t
                   x[1:T]
               end)
    @constraints(model, begin
                     sc * sum(x) == sc * 1
                     sc * [t; ovec; x] ∈ MOI.RelativeEntropyCone(2 * T + 1)
                     [i = 1:T], sc * [x[i]; theta[i]] ∈ MOI.NormOneCone(2)
                 end)
    @objective(model, Max, -so * t)
    return owa_model_solve(model, method, weights)
end

export OWA_MaximumEntropy
