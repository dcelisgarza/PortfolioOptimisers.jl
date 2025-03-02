abstract type JuMP_OrderedWeightsArray <: AbstractOrderedWeightsArray end
function owa_model_setup(method::JuMP_OrderedWeightsArray, weights::AbstractMatrix{<:Real})
    T, N = size(weights)
    model = JuMP.Model()
    max_phi = method.max_phi
    scale_constr = method.scale_constr
    @variables(model, begin
                   theta[1:T]
                   phi[1:N]
               end)
    @constraints(model,
                 begin
                     scale_constr * 0 .<= scale_constr * phi .<= scale_constr * max_phi
                     scale_constr * sum(phi) == scale_constr * 1
                     scale_constr * theta .== scale_constr * weights * phi
                     scale_constr * phi[2:end] .<= scale_constr * phi[1:(end - 1)]
                     scale_constr * theta[2:end] .>= scale_constr * theta[1:(end - 1)]
                 end)
    return model
end
function owa_model_solve(model::JuMP.Model, method::JuMP_OrderedWeightsArray,
                         weights::AbstractMatrix)
    solvers = method.solvers
    success = optimise_JuMP_model(model, solvers)[1]
    return if success
        phi = model[:phi]
        phis = value.(phi)
        phis ./= sum(phis)
        w = weights * phis
    else
        @warn("model could not be optimised satisfactorily.\nType: $method\nReverting to ncrra_weights.")
        w = ncrra_weights(weights, 0.5)
    end
end
