"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the joint Variance–Skewness–Kurtosis SDP risk constraints for a [`VarianceSkewKurtosis`](@ref) risk measure.

Constructs the semidefinite lifting variables `W1`, `W2`, `W3` and the PSD cone constraint that jointly encodes variance, skewness, and kurtosis. Each sub-risk expression is then bounded and registered separately using [`set_variance_risk_bounds_and_expression!`](@ref), with the skewness term using a lower bound (`flag = false`) because higher skewness is preferred. The composite expression `scale_vr * vr - scale_sk * sk + scale_kt * kt` is stored and passed to [`set_risk_bounds_and_expression!`](@ref).

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::VarianceSkewKurtosis`: Composite risk measure.
  - $(arg_dict[:opt_rjumpe])
  - `pr::HighOrderPrior`: High-order prior result providing `sigma`, `sk`, `kt`, `D2`, `L2`, `S2`.

# Returns

  - The composite `vr_sk_kt_risk` JuMP expression.

# Related

  - [`VarianceSkewKurtosis`](@ref)
  - [`set_variance_risk_bounds_and_expression!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
  - [`variance_risk_bounds_val`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::VarianceSkewKurtosis,
                               opt::RiskJuMPOptimisationEstimator, pr::HighOrderPrior,
                               args...; kwargs...)
    key = Symbol(:vr_sk_kt_risk_, i)
    vr_key = Symbol(:vr_risk_, i)
    sk_key = Symbol(:sk_risk_, i)
    kt_key = Symbol(:kt_risk_, i)
    w = get_w(model)
    sc = get_constraint_scale(model)
    k = ifelse(haskey(model, :crkb), 1, get_k(model))
    sigma = ifelse(isnothing(r.vr.sigma), pr.sigma, r.vr.sigma)
    sk = ifelse(isnothing(r.sk.sk), pr.sk, r.sk.sk)
    kt = ifelse(isnothing(r.kt.kt), pr.kt, r.kt.kt)
    D2 = pr.D2
    L2 = pr.L2
    S2 = pr.S2
    N = size(pr.X, 2)
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
    vr_bound = variance_risk_bounds_val(LinearBound(), r.vr.settings.ub)
    sk_bound = variance_risk_bounds_val(LinearBound(), r.sk.settings.lb)
    kt_bound = variance_risk_bounds_val(ifelse(isa(r.kt.alg1, QuadSecondMomentFormulations),
                                               LinearBound(), SquaredBound()),
                                        r.kt.settings.ub)
    set_variance_risk_bounds_and_expression!(model, opt, vr_risk, vr_bound, vr_key, vr_risk,
                                             r.vr.settings)
    # We want to maximise the skewness (distribution skewed towards more positive values), so we set a lower bound instead.
    set_variance_risk_bounds_and_expression!(model, opt, sk_risk, sk_bound, sk_key, sk_risk,
                                             r.sk.settings, false)
    set_variance_risk_bounds_and_expression!(model, opt, kt_risk, kt_bound, kt_key, kt_risk,
                                             r.kt.settings)
    vr_sk_kt_risk = model[key] = JuMP.@expression(model,
                                                  r.vr.settings.scale * vr_risk -
                                                  r.sk.settings.scale * sk_risk +
                                                  r.kt.settings.scale * kt_risk)
    set_risk_bounds_and_expression!(model, opt, vr_sk_kt_risk, r.settings, key)
    return vr_sk_kt_risk
end
