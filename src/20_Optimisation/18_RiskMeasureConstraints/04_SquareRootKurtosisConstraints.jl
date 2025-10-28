function set_kurtosis_risk!(model::JuMP.Model,
                            r::SquareRootKurtosis{<:Any, <:Any, <:Any, <:Any, <:Integer,
                                                  <:Any, <:SOCRiskExpr},
                            opt::RiskJuMPOptimisationEstimator,
                            sqrt_kurtosis_risk::AbstractJuMPScalar, key::Symbol)
    set_risk_bounds_and_expression!(model, opt, sqrt_kurtosis_risk, r.settings, key)
    return sqrt_kurtosis_risk
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::SquareRootKurtosis{<:Any, <:Any, <:Any, <:Any, <:Integer,
                                                     <:Any, <:Any},
                               opt::RiskJuMPOptimisationEstimator, pr::HighOrderPrior,
                               args...; kwargs...)
    key = Symbol(:sqrt_kurtosis_risk_, i)
    sc = model[:sc]
    W = set_sdp_constraints!(model)
    kt = isnothing(r.kt) ? pr.kt : r.kt
    N = size(W, 1)
    f = clamp(r.N, 1, N)
    Nf = f * N
    sqrt_kurtosis_risk, x_kurt = model[key], model[Symbol(:x_kurt_, i)] = @variables(model,
                                                                                     begin
                                                                                         ()
                                                                                         [1:Nf]
                                                                                     end)
    A = block_vec_pq(kt, N, N)
    vals_A, vecs_A = eigen(A)
    vals_A = clamp.(real(vals_A), 0, Inf) .+ clamp.(imag(vals_A), 0, Inf)im
    Bi = Vector{Matrix{eltype(kt)}}(undef, Nf)
    N_eig = length(vals_A)
    for i in eachindex(Bi)
        j = i - 1
        B = reshape(real(complex(sqrt(vals_A[end - j])) * view(vecs_A, :, N_eig - j)), N, N)
        Bi[i] = B
    end
    model[Symbol(:capprox_kurt_soc_, i)], model[Symbol(:capprox_kurt_, i)] = @constraints(model,
                                                                                          begin
                                                                                              [sc *
                                                                                               sqrt_kurtosis_risk
                                                                                               sc *
                                                                                               x_kurt] in
                                                                                              SecondOrderCone()
                                                                                              [i = 1:Nf],
                                                                                              sc *
                                                                                              (x_kurt[i] -
                                                                                               tr(Bi[i] *
                                                                                                  W)) ==
                                                                                              0
                                                                                          end)
    return set_kurtosis_risk!(model, r, opt, sqrt_kurtosis_risk, key)
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::SquareRootKurtosis{<:Any, <:Any, <:Any, <:Any, Nothing,
                                                     <:Any, <:Any},
                               opt::RiskJuMPOptimisationEstimator, pr::HighOrderPrior,
                               args...; kwargs...)
    key = Symbol(:sqrt_kurtosis_risk_, i)
    sc = model[:sc]
    W = set_sdp_constraints!(model)
    kt = isnothing(r.kt) ? pr.kt : r.kt
    sqrt_kurtosis_risk = model[key] = @variable(model)
    L2 = pr.L2
    S2 = pr.S2
    sqrt_sigma_4 = cholesky(S2 * kt * transpose(S2)).U
    zkurt = model[Symbol(:zkurt_, i)] = @expression(model, L2 * vec(W))
    model[Symbol(:ckurt_soc_, i)] = @constraint(model,
                                                [sc * sqrt_kurtosis_risk;
                                                 sc * sqrt_sigma_4 * zkurt] in
                                                SecondOrderCone())
    return set_kurtosis_risk!(model, r, opt, sqrt_kurtosis_risk, key)
end
function set_risk_constraints!(::JuMP.Model, ::Any, ::SquareRootKurtosis,
                               ::Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting},
                               pr::LowOrderPrior, args...; kwargs...)
    throw(ArgumentError("SquareRootKurtosis requires a HighOrderPrior, not a $(typeof(pr))."))
end
