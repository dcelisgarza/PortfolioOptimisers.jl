function RRM(x::AbstractVector, slv::Union{<:Solver, <:AbstractVector{<:Solver}},
             alpha::Real = 0.05, kappa::Real = 0.3)
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    T = length(x)
    at = alpha * T
    invat = 1 / at
    ln_k = (invat^kappa - invat^(-kappa)) / (2 * kappa)
    invk2 = 1 / (2 * kappa)
    opk = 1 + kappa
    omk = 1 - kappa
    invk = 1 / kappa
    invopk = 1 / opk
    invomk = 1 / omk
    model = JuMP.Model()
    set_string_names_on_creation(model, false)
    @variables(model, begin
                   t
                   z >= 0
                   omega[1:T]
                   psi[1:T]
                   theta[1:T]
                   epsilon[1:T]
               end)
    @constraints(model,
                 begin
                     [i = 1:T],
                     [z * opk * invk2, psi[i] * opk * invk, epsilon[i]] ∈
                     MOI.PowerCone(invopk)
                     [i = 1:T],
                     [omega[i] * invomk, theta[i] * invk, -z * invk2] ∈ MOI.PowerCone(omk)
                     -x .- t .+ epsilon .+ omega .<= 0
                 end)
    @expression(model, risk, t + ln_k * z + sum(psi .+ theta))
    @objective(model, Min, risk)
    (; trials, success) = optimise_JuMP_model!(model, slv)
    return if success
        objective_value(model)
    else
        model = JuMP.Model()
        set_string_names_on_creation(model, false)
        @variables(model, begin
                       z[1:T]
                       nu[1:T]
                       tau[1:T]
                   end)
        @constraints(model, begin
                         sum(z) == 1
                         sum(nu .- tau) * invk2 <= ln_k
                         [i = 1:T], [nu[i], 1, z[i]] ∈ MOI.PowerCone(invopk)
                         [i = 1:T], [z[i], 1, tau[i]] ∈ MOI.PowerCone(omk)
                     end)
        @expression(model, risk, -dot(z, x))
        @objective(model, Max, risk)
        (; trials, success) = optimise_JuMP_model!(model, slv)
        if success
            objective_value(model)
        else
            @warn("Model could not be optimised satisfactorily.\nSolvers: $trials.")
            NaN
        end
    end
end
