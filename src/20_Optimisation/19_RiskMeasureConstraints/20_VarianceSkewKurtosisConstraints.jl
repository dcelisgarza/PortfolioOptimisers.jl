function set_risk_constraints!(model::JuMP.Model, i::Any, r::VarianceSkewKurtosis,
                               opt::RiskJuMPOptimisationEstimator, pr::HighOrderPrior,
                               args...; kwargs...)
    key = Symbol(:vr_sk_kt_risk_, i)
    w = model[:w]
    sc = model[:sc]
    k = ifelse(haskey(model, :crkb), 1, model[:k])
    sigma = ifelse(isnothing(r.vr.sigma), pr.sigma, r.vr.sigma)
    sk = ifelse(isnothing(r.sk.sk), pr.sk, r.sk.sk)
    kt = ifelse(isnothing(r.kt.kt), pr.kt, r.kt.kt)
    D2 = pr.D2
    L2 = pr.L2
    S2 = pr.S2
    T, N = size(pr.X)
    M = div(N * (N + 1), 2)
    if !haskey(model, :W1_vr_sk_kt)
        JuMP.@variables(model, begin
                            W1_vr_sk_kt[1:N, 1:N], Symmetric
                            W2_vr_sk_kt[1:M, 1:N]
                            W3_vr_sk_kt[1:M, 1:M], Symmetric
                        end)
        JuMP.@expression(model, L2W1_vr_sk_kt, L2 * vec(W1_vr_sk_kt))
        JuMP.@expression(model, M_vr_sk_kt,
                         vcat(hcat(k, transpose(w), transpose(L2W1_vr_sk_kt)),
                              hcat(w, W1_vr_sk_kt, transpose(W2_vr_sk_kt)),
                              hcat(L2W1_vr_sk_kt, W2_vr_sk_kt, W3_vr_sk_kt)))
        JuMP.@constraint(model, M_vr_sk_kt_PSD, sc * M_vr_sk_kt in JuMP.PSDCone())
    end
    W1 = model[:W1_vr_sk_kt]
    W2 = model[:W2_vr_sk_kt]
    W3 = model[:W3_vr_sk_kt]
    vr_risk, sk_risk, kt_risk = model[Symbol(:vr_risk_, i)], model[Symbol(:sk_risk_, i)], model[Symbol(:kt_risk_, i)] = JuMP.@expressions(model,
                                                                                                                                          begin
                                                                                                                                              LinearAlgebra.tr(sigma *
                                                                                                                                                               W1)
                                                                                                                                              LinearAlgebra.tr(sk *
                                                                                                                                                               D2 *
                                                                                                                                                               W2)
                                                                                                                                              LinearAlgebra.tr(S2 *
                                                                                                                                                               kt *
                                                                                                                                                               transpose(S2) *
                                                                                                                                                               W3)
                                                                                                                                          end)
    #! check
    ub = variance_risk_bounds_val(sdp_flag, r.settings.ub)
    set_variance_risk_bounds_and_expression!(model, opt, var_bound_expr, ub, var_bound_key,
                                             variance_risk, r.settings)
    set_negative_skewness_risk!(model, r, opt, nskew_risk, key, V)
    set_kurtosis_risk!(model, r, opt, sqrt_kurtosis_risks, x_kurt, key, i)

    return nothing
end
