"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the joint Variance–Skewness–Kurtosis SDP risk constraints for a [`VarianceSkewKurtosis`](@ref) risk measure.

Constructs the semidefinite lifting variables `W1`, `W2`, `W3` and the PSD cone constraint that jointly encodes variance, skewness, and kurtosis. Each sub-risk expression is then bounded and registered separately using [`set_variance_risk_bounds_and_expression!`](@ref), with the skewness term using a lower bound (`flag = false`) because higher skewness is preferred. The composite expression `scale_vr * vr - scale_sk * sk + scale_kt * kt` is stored and passed to [`set_risk_bounds_and_expression!`](@ref).

Any prior result is accepted. Both tensors must resolve, on their child or on the prior, and [`assert_high_order_quantity`](@ref) refuses the measure when either resolves on neither. The three vectorisation matrices come from [`dup_elim_sum_selector`](@ref), so a container whose `sk` and `kt` children hold their own tensors is buildable against a [`LowOrderPrior`](@ref), which already carries the `sigma` the third child needs.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::VarianceSkewKurtosis`: Composite risk measure.
  - $(arg_dict[:opt_rjumpe])
  - `pr::AbstractPriorResult`: Prior result. It supplies `sigma`, `sk` and `kt` where the children state none.

# Returns

  - The composite `vr_sk_kt_risk` JuMP expression.

# Related

  - [`VarianceSkewKurtosis`](@ref)
  - [`set_variance_risk_bounds_and_expression!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
  - [`variance_risk_bounds_val`](@ref)
  - [`assert_high_order_quantity`](@ref)
  - [`dup_elim_sum_selector`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::VarianceSkewKurtosis,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    assert_high_order_quantity(r.sk.sk, pr, :VarianceSkewKurtosis, :sk,
                               :CoskewnessEstimator)
    assert_high_order_quantity(r.kt.kt, pr, :VarianceSkewKurtosis, :kt,
                               :CokurtosisEstimator)
    w = get_w(model, prefix)
    sc = get_constraint_scale(model)
    k = effective_k(model)
    sigma = nothing_scalar_array_selector(r.vr.sigma, pr.sigma)
    sk = nothing_scalar_array_selector(r.sk.sk, prior_high_order_quantity(pr, :sk))
    kt = nothing_scalar_array_selector(r.kt.kt, prior_high_order_quantity(pr, :kt))
    N = size(pr.X, 2)
    D2, L2, S2 = dup_elim_sum_selector(pr, N)
    M = div(N * (N + 1), 2)
    W1 = state_build!(model, prefix, :W1_vr_sk_kt) do
        W1_vr_sk_kt = JuMP.@variable(model, [1:N, 1:N], Symmetric)
        W2_vr_sk_kt = state_set!(model, prefix, :W2_vr_sk_kt,
                                 JuMP.@variable(model, [1:M, 1:N]))
        W3_vr_sk_kt = state_set!(model, prefix, :W3_vr_sk_kt,
                                 JuMP.@variable(model, [1:M, 1:M], Symmetric))
        L2W1_vr_sk_kt = state_set!(model, prefix, :L2W1_vr_sk_kt,
                                   JuMP.@expression(model, L2 * vec(W1_vr_sk_kt)))
        M_vr_sk_kt = state_set!(model, prefix, :M_vr_sk_kt,
                                JuMP.@expression(model,
                                                 vcat(hcat(k, transpose(w),
                                                           transpose(L2W1_vr_sk_kt)),
                                                      hcat(w, W1_vr_sk_kt,
                                                           transpose(W2_vr_sk_kt)),
                                                      hcat(L2W1_vr_sk_kt, W2_vr_sk_kt,
                                                           W3_vr_sk_kt))))
        state_set!(model, prefix, :M_vr_sk_kt_PSD,
                   JuMP.@constraint(model, sc * M_vr_sk_kt in JuMP.PSDCone()))
        return W1_vr_sk_kt
    end
    W2 = state_get(model, prefix, :W2_vr_sk_kt)
    W3 = state_get(model, prefix, :W3_vr_sk_kt)
    vr_risk, sk_risk, kt_risk = JuMP.@expressions(model,
                                                  begin
                                                      LinearAlgebra.tr(sigma * W1)
                                                      LinearAlgebra.tr(sk * D2 * W2)
                                                      LinearAlgebra.tr(S2 *
                                                                       kt *
                                                                       transpose(S2) *
                                                                       W3)
                                                  end)
    state_set!(model, prefix, :vr_risk_, i, vr_risk)
    state_set!(model, prefix, :sk_risk_, i, sk_risk)
    state_set!(model, prefix, :kt_risk_, i, kt_risk)
    vr_bound = variance_risk_bounds_val(LinearBound(), r.vr.settings.ub)
    sk_bound = variance_risk_bounds_val(LinearBound(), r.sk.settings.lb)
    kt_bound = variance_risk_bounds_val(ifelse(isa(r.kt.alg1, QuadSecondMomentFormulations),
                                               LinearBound(), SquaredBound()),
                                        r.kt.settings.ub)
    set_variance_risk_bounds_and_expression!(model, opt, vr_risk, vr_bound, :vr_risk_, i,
                                             vr_risk, r.vr.settings; prefix = prefix)
    # We want to maximise the skewness (distribution skewed towards more positive values), so we set a lower bound instead.
    set_variance_risk_bounds_and_expression!(model, opt, sk_risk, sk_bound, :sk_risk_, i,
                                             sk_risk, r.sk.settings, false; prefix = prefix)
    set_variance_risk_bounds_and_expression!(model, opt, kt_risk, kt_bound, :kt_risk_, i,
                                             kt_risk, r.kt.settings; prefix = prefix)
    vr_sk_kt_risk = state_set!(model, prefix, :vr_sk_kt_risk_, i,
                               JuMP.@expression(model,
                                                r.vr.settings.scale * vr_risk -
                                                r.sk.settings.scale * sk_risk +
                                                r.kt.settings.scale * kt_risk))
    set_risk_bounds_and_expression!(model, opt, vr_sk_kt_risk, r.settings, :vr_sk_kt_risk_,
                                    i; prefix = prefix)
    return vr_sk_kt_risk
end
