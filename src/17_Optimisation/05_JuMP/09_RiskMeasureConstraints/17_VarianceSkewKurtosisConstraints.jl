"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the semidefinite relaxation of a [`VarianceSkewKurtosis`](@ref) to the model, and return its risk expression.

The model is Problem 12 of [sdpmom](@cite), with the budget homogenised by ``k``. One positive semidefinite block, Equation 11 of that work, lifts the products of the weights into the matrices ``\\mathbf{W}_1``, ``\\mathbf{W}_2`` and ``\\mathbf{W}_3``, and each moment becomes a trace that is affine in a block. The block is built once for each namespace, so two measures of one model share it.

Any prior result is accepted. Both tensors must resolve, on their child or on the prior, and [`assert_high_order_quantity`](@ref) refuses the measure when either resolves on neither. The three vectorisation matrices come from [`dup_elim_sum_selector`](@ref), so a container whose `sk` and `kt` children hold their own tensors builds against a [`LowOrderPrior`](@ref), which carries the `sigma` that the third child needs.

# Algorithm

 1. Check that `r.sk.sk` or `pr` holds a coskewness matrix, and that `r.kt.kt` or `pr` holds a cokurtosis matrix, with [`assert_high_order_quantity`](@ref).
 2. Read `w`, the weights, `sc`, the constraint scale, and `k`, the homogenisation variable.
 3. Select `sigma`, `sk` and `kt`, the matrix of each child, else the matrix of the prior.
 4. Read `D2`, `L2` and `S2` for the `N` assets with [`dup_elim_sum_selector`](@ref).
 5. Build `W1`, `W2`, `W3`, `L2W1` and the block `M` once for the namespace, and register the cone row below.
 6. Register `vr_risk`, `sk_risk` and `kt_risk`, the three traces.
 7. Compute the bound of each term with [`variance_risk_bounds_val`](@ref). The bound of `kt` is the square of `kt.settings.ub` under [`SOCRiskExpr`](@ref), because that `ub` is in the units of ``\\sqrt{m_4}``.
 8. Bound each term with [`set_variance_risk_bounds_and_expression!`](@ref). The skewness bound is a floor, so its row has the opposite sign. The children hold `rke = false`, so no term enters `risk_vec` alone. Record `r.vr`, `r.sk` and `r.kt` as the owners of the frontier entries of their terms with [`set_risk_frontier_owner!`](@ref), so that a sweep on a term takes its span from that term. [`frontier_risk`](@ref) values `r.sk` as the third moment of the coskewness matrix, the units of `sk.settings.lb`.
 9. Register `vr_sk_kt_risk`, the weighted sum, bound it and add it to `risk_vec` with [`set_risk_bounds_and_expression!`](@ref), and return it.

# JuMP formulation

## Variables

  - `w`: read, ``\\boldsymbol{w}``.
  - `k`: read, ``k``.
  - `W1_vr_sk_kt`: created, the symmetric ``N \\times N`` block ``\\mathbf{W}_1``, which stands for ``\\boldsymbol{w} \\boldsymbol{w}^\\intercal / k``.
  - `W2_vr_sk_kt`: created, the ``M \\times N`` block ``\\mathbf{W}_2``, which stands for ``\\mathbf{L}_2 (\\boldsymbol{w} \\otimes \\boldsymbol{w}) \\boldsymbol{w}^\\intercal / k^2``.
  - `W3_vr_sk_kt`: created, the symmetric ``M \\times M`` block ``\\mathbf{W}_3``, which stands for ``\\mathbf{L}_2 (\\boldsymbol{w} \\otimes \\boldsymbol{w}) (\\boldsymbol{w} \\otimes \\boldsymbol{w})^\\intercal \\mathbf{L}_2^\\intercal / k^3``.

## Expressions

  - `L2W1_vr_sk_kt`: ``\\mathbf{L}_2 \\mathrm{vec}(\\mathbf{W}_1)``.
  - `M_vr_sk_kt`: the block matrix ``\\mathbf{M}`` below.
  - `vr_risk_`: ``\\mathrm{Tr}(\\mathbf{\\Sigma} \\mathbf{W}_1)``.
  - `sk_risk_`: ``\\mathrm{Tr}(\\mathbf{M}_3 \\mathbf{D}_2 \\mathbf{W}_2)``.
  - `kt_risk_`: ``\\mathrm{Tr}(\\mathbf{S}_2 \\mathbf{K} \\mathbf{S}_2^\\intercal \\mathbf{W}_3)``.
  - `vr_sk_kt_risk_`: ``s_{\\sigma^2} \\mathrm{Tr}(\\mathbf{\\Sigma} \\mathbf{W}_1) - s_{\\mathrm{sk}} \\mathrm{Tr}(\\mathbf{M}_3 \\mathbf{D}_2 \\mathbf{W}_2) + s_{\\kappa} \\mathrm{Tr}(\\mathbf{S}_2 \\mathbf{K} \\mathbf{S}_2^\\intercal \\mathbf{W}_3)``.

```math
\\begin{align}
\\mathbf{M} &= \\begin{bmatrix}
    k & \\boldsymbol{w}^\\intercal & \\left(\\mathbf{L}_2 \\mathrm{vec}(\\mathbf{W}_1)\\right)^\\intercal \\\\
    \\boldsymbol{w} & \\mathbf{W}_1 & \\mathbf{W}_2^\\intercal \\\\
    \\mathbf{L}_2 \\mathrm{vec}(\\mathbf{W}_1) & \\mathbf{W}_2 & \\mathbf{W}_3
\\end{bmatrix}\\,.
\\end{align}
```

## Constraints

  - `M_vr_sk_kt_PSD`: ``s_c \\mathbf{M} \\succeq 0``.
  - `vr_risk_<i>_ub`: ``s_c \\left(\\mathrm{Tr}(\\mathbf{\\Sigma} \\mathbf{W}_1) - u_{\\sigma^2} k\\right) \\leq 0``, when `vr.settings.ub` is a number.
  - `sk_risk_<i>_ub`: ``-s_c \\left(\\mathrm{Tr}(\\mathbf{M}_3 \\mathbf{D}_2 \\mathbf{W}_2) - l_{\\mathrm{sk}} k\\right) \\leq 0``, when `sk.settings.lb` is a number.
  - `kt_risk_<i>_ub`: ``s_c \\left(\\mathrm{Tr}(\\mathbf{S}_2 \\mathbf{K} \\mathbf{S}_2^\\intercal \\mathbf{W}_3) - u_{\\kappa} k\\right) \\leq 0``, when `kt.settings.ub` is a number.
  - `vr_sk_kt_risk_<i>_ub`: the bound of `settings.ub` on `vr_sk_kt_risk_`, from [`set_risk_bounds_and_expression!`](@ref).

[`set_risk_upper_bound!`](@ref) registers the four bound rows. A frontier in a bound replaces its row with the frontier variable of that function. Each name except those of the block carries the index `i` of the measure, and every name carries the namespace `prefix`.

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])
  - $(math_dict[:sc_scale])
  - ``M = N(N+1)/2``: Number of entries on and below the diagonal of an ``N \\times N`` matrix.
  - $(math_dict[:Sigma_rm])
  - $(math_dict[:M3_cosk])
  - $(math_dict[:K_cokurt])
  - ``\\mathbf{D}_2``: Duplication matrix, ``N^2 \\times N(N+1)/2``. For a symmetric ``\\mathbf{A}``, ``\\mathbf{D}_2 \\mathbf{L}_2 \\mathrm{vec}(\\mathbf{A}) = \\mathrm{vec}(\\mathbf{A})``.
  - $(math_dict[:L_2_elim])
  - $(math_dict[:S_2_sum])
  - ``s_{\\sigma^2}``, ``s_{\\mathrm{sk}}``, ``s_{\\kappa}``: The `settings.scale` of `vr`, `sk` and `kt`.
  - ``u_{\\sigma^2}``: `vr.settings.ub`.
  - ``l_{\\mathrm{sk}}``: `sk.settings.lb`.
  - ``u_{\\kappa}``: `kt.settings.ub`, squared under [`SOCRiskExpr`](@ref).

## Relaxation

$(val_dict[:relax])

  - The corner ``k`` of ``\\mathbf{M}`` gives ``\\mathbf{W}_1 \\succeq \\boldsymbol{w} \\boldsymbol{w}^\\intercal / k``, so `vr_risk_` lies at or above ``\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w} / k``.
  - The same block gives ``\\mathbf{W}_3 \\succeq \\mathbf{L}_2 \\mathrm{vec}(\\mathbf{W}_1) \\mathrm{vec}(\\mathbf{W}_1)^\\intercal \\mathbf{L}_2^\\intercal / k``. For a sample cokurtosis, `kt_risk_` then lies at or above ``m_4(\\boldsymbol{w}) / k^3``, by the argument of the exact formulation of [`Kurtosis`](@ref).
  - No row ties ``\\mathbf{W}_2`` to the weights, so `sk_risk_` bounds ``m_3(\\boldsymbol{w}) / k^2`` in neither direction. The objective subtracts it, so a minimisation pushes it up, and a large ``s_{\\mathrm{sk}}`` moves it far from ``m_3``.
  - So `vr_sk_kt_risk_` bounds the value of the functor in neither direction. All four terms are exact when ``\\mathbf{M}`` has rank one, and Section 3 of [sdpmom](@cite) states that the relaxation is tight over a range of skewness values only.
  - The ceilings on `vr_risk_` and `kt_risk_` hold the weights, because each term lies at or above its moment. The floor on `sk_risk_` holds the relaxation and not the weights, so the returned weights can have a third moment below `sk.settings.lb`.
  - A [`Frontier`](@ref) sweep binds the relaxation in the same way. The span of a sweep on `sk.settings.lb` is ``m_3`` at the two corner portfolios, and the span of a sweep on `settings.ub` is the value of the functor at the two corners. Each level is a correct value, but the floor on `sk_risk_` binds no weight, so a sweep on it can return one portfolio at every level. A sweep on `settings.ub` meets each level while the relaxation is tight. When a large ``s_{\\mathrm{sk}}`` opens the gap, the levels can return one portfolio, whose value of the functor lies far from its level.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::VarianceSkewKurtosis`: Composite risk measure.
  - $(arg_dict[:opt_rjumpe])
  - `pr::AbstractPriorResult`: Prior result. It supplies `sigma`, `sk` and `kt` where the children state none.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Validation

  - `r.sk.sk` or `pr` holds a coskewness matrix, and `r.kt.kt` or `pr` holds a cokurtosis matrix. [`assert_high_order_quantity`](@ref) raises an `ArgumentError` otherwise.

# Returns

  - `vr_sk_kt_risk`: The weighted sum of the three traces.

# Related

  - [`VarianceSkewKurtosis`](@ref)
  - [`set_variance_risk_bounds_and_expression!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
  - [`set_risk_upper_bound!`](@ref)
  - [`variance_risk_bounds_val`](@ref)
  - [`assert_high_order_quantity`](@ref)
  - [`dup_elim_sum_selector`](@ref)

# References

  - $(ref_dict[:sdpmom]) Section 3, Equation 11 and Problems 12 to 14.
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::VarianceSkewKurtosis,
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
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
                                                      LinearAlgebra.dot(transpose(sigma),
                                                                        W1)
                                                      LinearAlgebra.dot(transpose(sk * D2),
                                                                        W2)
                                                      LinearAlgebra.dot(transpose(S2 *
                                                                                  kt *
                                                                                  transpose(S2)),
                                                                        W3)
                                                  end)
    state_set!(model, prefix, :vr_risk_, i, vr_risk)
    state_set!(model, prefix, :sk_risk_, i, sk_risk)
    state_set!(model, prefix, :kt_risk_, i, kt_risk)
    vr_bound = variance_risk_bounds_val(LinearBound(), r.vr.settings.ub)
    sk_bound = variance_risk_bounds_val(LinearBound(), r.sk.settings.lb)
    kt_bound = variance_risk_bounds_val(ifelse(isa(r.kt.alg2, QuadSecondMomentFormulations),
                                               LinearBound(), SquaredBound()),
                                        r.kt.settings.ub)
    # A sweep on a term takes its span from the measure of that term, so each term records
    # its own measure as the owner of its entry before the whole measure claims the rest.
    first = risk_frontier_length(model)
    set_variance_risk_bounds_and_expression!(model, opt, vr_risk, vr_bound, :vr_risk_, i,
                                             vr_risk, r.vr.settings; prefix = prefix)
    set_risk_frontier_owner!(model, first, r.vr)
    # We want to maximise the skewness (distribution skewed towards more positive values), so we set a lower bound instead.
    first = risk_frontier_length(model)
    set_variance_risk_bounds_and_expression!(model, opt, sk_risk, sk_bound, :sk_risk_, i,
                                             sk_risk, r.sk.settings, false; prefix = prefix)
    set_risk_frontier_owner!(model, first, r.sk)
    first = risk_frontier_length(model)
    set_variance_risk_bounds_and_expression!(model, opt, kt_risk, kt_bound, :kt_risk_, i,
                                             kt_risk, r.kt.settings; prefix = prefix)
    set_risk_frontier_owner!(model, first, r.kt)
    vr_sk_kt_risk = state_set!(model, prefix, :vr_sk_kt_risk_, i,
                               JuMP.@expression(model,
                                                r.vr.settings.scale * vr_risk -
                                                r.sk.settings.scale * sk_risk +
                                                r.kt.settings.scale * kt_risk))
    set_risk_bounds_and_expression!(model, opt, vr_sk_kt_risk, r.settings, :vr_sk_kt_risk_,
                                    i; prefix = prefix)
    return vr_sk_kt_risk
end
