abstract type JuMP_OrderedWeightsArray <: AbstractOrderedWeightsArray end
function owa_model_setup(method::JuMP_OrderedWeightsArray, weights::AbstractMatrix{<:Real})
    T, N = size(weights)
    model = JuMP.Model()
    max_phi = method.max_phi
    sc = method.sc
    @variables(model, begin
                   theta[1:T]
                   phi[1:N]
               end)
    @constraints(model, begin
                     sc * 0 .<= sc * phi .<= sc * max_phi
                     sc * sum(phi) == sc * 1
                     sc * theta .== sc * weights * phi
                     sc * phi[2:end] .<= sc * phi[1:(end - 1)]
                     sc * theta[2:end] .>= sc * theta[1:(end - 1)]
                 end)
    return model
end
function owa_model_solve(model::JuMP.Model, method::JuMP_OrderedWeightsArray,
                         weights::AbstractMatrix)
    slv = method.slv
    success = optimise_JuMP_model(model, slv)[1]
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
