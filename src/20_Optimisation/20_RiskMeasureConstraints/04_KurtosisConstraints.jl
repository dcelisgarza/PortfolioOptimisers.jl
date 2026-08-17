"""
$(DocStringExtensions.TYPEDSIGNATURES)

Retrieve or compute and cache the Cholesky factor of the co-kurtosis matrix.

If `model` does not yet contain `Gkt`, computes the upper Cholesky factor of
`pr.S2 * pr.kt * pr.S2'` and stores it as the `:Gkt` Model State entry.

# Arguments

  - $(arg_dict[:model])
  - `pr::HighOrderPrior`: High-order prior containing `kt` and `S2`.

# Returns

  - `Gkt::Matrix`: Upper Cholesky factor of the co-kurtosis projected matrix.

# Related

  - [`get_kt_Akt_pm`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function get_chol_or_Gkt_pm(model::JuMP.Model, pr::HighOrderPrior)
    if !shared_has(model, :Gkt)
        #=
        #! figure out how to add chol
        G = if isnothing(pr.chol_kt)
            LinearAlgebra.cholesky(pr.S2 * pr.kt * transpose(pr.S2)).U
        else
            N1, N2 = size(pr.chol_kt)
            vcat(pr.S2 * view(pr.chol_kt, 1:N2, 1:N2) * transpose(pr.S2),
                 view(pr.chol_kt, (N2 + 1):N1, :) * transpose(pr.S2))
        end
        =#
        G = LinearAlgebra.cholesky(pr.S2 * pr.kt * transpose(pr.S2)).U
        JuMP.@expression(model, Gkt, G)
    end
    return shared_get(model, :Gkt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Retrieve or compute and cache the eigendecomposition of the co-kurtosis matrix.

Builds the block-vectorised kurtosis matrix `A`, clamps its eigenvalues to be non-negative,
and stores `vals_Akt` and `vecs_Akt` in `model`.

# Arguments

  - $(arg_dict[:model])
  - `pr::HighOrderPrior`: High-order prior containing `kt` and `mu`.

# Returns

  - A 2-tuple `(vals_Akt, vecs_Akt)` of eigenvalues and eigenvectors.

# Related

  - [`get_chol_or_Gkt_pm`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function get_kt_Akt_pm(model::JuMP.Model, pr::HighOrderPrior)
    if !shared_has(model, :vecs_Akt)
        N = length(pr.mu)
        A = block_vec_pq(pr.kt, N, N)
        vals_A, vecs_A = LinearAlgebra.eigen(A)
        vals_A = clamp.(real(vals_A), 0, Inf) .+ clamp.(imag(vals_A), 0, Inf)im
        JuMP.@expressions(model, begin
                              vecs_Akt, vecs_A
                              vals_Akt, vals_A
                          end)
    end
    return shared_get(model, :vals_Akt), shared_get(model, :vecs_Akt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Finalise the kurtosis risk expression and apply bounds according to the chosen formulation.

The `SOCRiskExpr` overload passes the SOC variable directly to
[`set_risk_bounds_and_expression!`](@ref). The `SquaredSOCRiskExpr` overload squares the
variable and bounds the original variable. The `QuadRiskExpr` overload uses a quadratic
dot product of `x_kurt`. The `RSOCRiskExpr` overload adds a rotated second-order cone
constraint.

# Arguments

  - $(arg_dict[:model])
  - `r::Kurtosis`: Kurtosis risk measure instance.
  - $(arg_dict[:opt_rjumpe])
  - `sqrt_kurtosis_risk`: SOC variable representing the square root of kurtosis risk.
  - `x_kurt`: Auxiliary vector expression used in Quad/RSOC formulations.
  - $(arg_dict[:ci])

# Returns

  - The kurtosis risk JuMP expression.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`variance_risk_bounds_val`](@ref)
"""
function set_kurtosis_risk!(model::JuMP.Model,
                            r::Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:SOCRiskExpr}, opt::RiskJuMPOptimisationEstimator,
                            sqrt_kurtosis_risk::JuMP.AbstractJuMPScalar, ::Any, i;
                            prefix::Symbol = Symbol(""))
    set_risk_bounds_and_expression!(model, opt, sqrt_kurtosis_risk, r.settings,
                                    :kurtosis_risk_, i; prefix = prefix)
    return sqrt_kurtosis_risk
end
function set_kurtosis_risk!(model::JuMP.Model,
                            r::Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:SquaredSOCRiskExpr},
                            opt::RiskJuMPOptimisationEstimator,
                            sqrt_kurtosis_risk::JuMP.AbstractJuMPScalar, ::Any, i;
                            prefix::Symbol = Symbol(""))
    qsqrt_kurtosis_risk = state_set!(model, prefix, :sq_kurtosis_risk_, i,
                                     JuMP.@expression(model, sqrt_kurtosis_risk^2))
    ub = variance_risk_bounds_val(SquareRootBound(), r.settings.ub)
    set_risk_upper_bound!(model, opt, sqrt_kurtosis_risk, ub,
                          state_key(prefix, :kurtosis_risk_, i))
    set_risk_expression!(model, qsqrt_kurtosis_risk, r.settings.scale, r.settings.rke)
    return qsqrt_kurtosis_risk
end
function set_kurtosis_risk!(model::JuMP.Model,
                            r::Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:QuadRiskExpr}, opt::RiskJuMPOptimisationEstimator,
                            sqrt_kurtosis_risk::JuMP.AbstractJuMPScalar, x_kurt, i;
                            prefix::Symbol = Symbol(""))
    qsqrt_kurtosis_risk = state_set!(model, prefix, :qd_kurtosis_risk_, i,
                                     JuMP.@expression(model,
                                                      LinearAlgebra.dot(x_kurt, x_kurt)))
    ub = variance_risk_bounds_val(SquareRootBound(), r.settings.ub)
    set_risk_upper_bound!(model, opt, sqrt_kurtosis_risk, ub,
                          state_key(prefix, :kurtosis_risk_, i))
    set_risk_expression!(model, qsqrt_kurtosis_risk, r.settings.scale, r.settings.rke)
    return qsqrt_kurtosis_risk
end
function set_kurtosis_risk!(model::JuMP.Model,
                            r::Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:RSOCRiskExpr}, opt::RiskJuMPOptimisationEstimator,
                            sqrt_kurtosis_risk::JuMP.AbstractJuMPScalar, x_kurt, i;
                            prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    tkurtosis = state_set!(model, prefix, :tkurtosis_risk, i, JuMP.@variable(model))
    qsqrt_kurtosis_risk = state_set!(model, prefix, :ckurtosis_rsoc_, i,
                                     JuMP.@constraint(model,
                                                      [sc * tkurtosis
                                                       0.5
                                                       sc * x_kurt] in
                                                      JuMP.RotatedSecondOrderCone()))
    ub = variance_risk_bounds_val(SquareRootBound(), r.settings.ub)
    set_risk_upper_bound!(model, opt, sqrt_kurtosis_risk, ub,
                          state_key(prefix, :kurtosis_risk_, i))
    set_risk_expression!(model, qsqrt_kurtosis_risk, r.settings.scale, r.settings.rke)
    return qsqrt_kurtosis_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add kurtosis risk constraints to `model`.

The `Integer N` overload uses an approximate spectral decomposition of the co-kurtosis tensor
to build `N` eigen-directions and encodes kurtosis via SOC and equality constraints. The
`Nothing N` overload uses the full Cholesky-based formulation with the duplication matrix.

Both accept any prior result. The cokurtosis matrix must resolve on one side or the other,
and [`assert_high_order_quantity`](@ref) refuses the measure when it resolves on neither.

# Mathematical definition

```math
\\begin{align}
\\sqrt{\\mathrm{Kurt}(\\boldsymbol{w})} &= \\lVert \\mathbf{G}_{kt}(\\boldsymbol{w} \\otimes \\boldsymbol{w}) \\rVert_2\\,, \\\\
\\mathbf{G}_{kt} &= \\mathrm{chol}(\\mathbf{S}_2 \\mathbf{K} \\mathbf{S}_2^\\intercal)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{Kurt}(\\boldsymbol{w})``: Portfolio kurtosis risk measure.
  - ``\\mathbf{G}_{kt}``: Cholesky factor of the projected co-kurtosis matrix.
  - ``\\mathbf{K}``: Co-kurtosis matrix.
  - ``\\mathbf{S}_2``: Duplication matrix.
  - $(math_dict[:w_port])
  - ``\\otimes``: Kronecker product.

where ``\\mathbf{K}`` is the co-kurtosis matrix and ``\\mathbf{S}_2`` is the duplication matrix.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::Kurtosis`: Kurtosis risk measure instance.
  - $(arg_dict[:opt_rjumpe])
  - `pr::AbstractPriorResult`: Prior result. It supplies `kt` when the measure states none.

# Returns

  - `nothing`.

# Related

  - [`get_chol_or_Gkt_pm`](@ref)
  - [`get_kt_Akt_pm`](@ref)
  - [`set_kurtosis_risk!`](@ref)
  - [`assert_high_order_quantity`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Integer, <:Any,
                                           <:Any}, opt::RiskJuMPOptimisationEstimator,
                               pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    assert_high_order_quantity(r.kt, pr, :Kurtosis, :kt, :CokurtosisEstimator)
    sc = get_constraint_scale(model)
    W = set_sdp_constraints!(model; prefix = prefix)
    N = size(W, 1)
    f = clamp(r.N, 1, N)
    Nf = f * N
    sqrt_kurtosis_risk, x_kurt = JuMP.@variables(model, begin
                                                     ()
                                                     [1:Nf]
                                                 end)
    state_set!(model, prefix, :kurtosis_risk_, i, sqrt_kurtosis_risk)
    state_set!(model, prefix, :x_kurt_, i, x_kurt)
    vals_A, vecs_A = if isnothing(r.kt)
        get_kt_Akt_pm(model, pr)
    else
        A = block_vec_pq(r.kt, N, N)
        vals_A, vecs_A = LinearAlgebra.eigen(A)
        vals_A = clamp.(real(vals_A), 0, Inf) .+ clamp.(imag(vals_A), 0, Inf)im
        vals_A, vecs_A
    end
    Bi = Vector{Matrix{eltype(vals_A)}}(undef, Nf)
    N_eig = length(vals_A)
    for i in eachindex(Bi)
        j = i - 1
        B = reshape(real(complex(sqrt(vals_A[end - j])) * view(vecs_A, :, N_eig - j)), N, N)
        Bi[i] = B
    end
    capprox_kurt_soc, capprox_kurt = JuMP.@constraints(model,
                                                       begin
                                                           [sc * sqrt_kurtosis_risk
                                                            sc * x_kurt] in
                                                           JuMP.SecondOrderCone()
                                                           [i = 1:Nf],
                                                           sc * (x_kurt[i] -
                                                                 LinearAlgebra.tr(Bi[i] * W)) ==
                                                           0
                                                       end)
    state_set!(model, prefix, :capprox_kurt_soc_, i, capprox_kurt_soc)
    state_set!(model, prefix, :capprox_kurt_, i, capprox_kurt)
    return set_kurtosis_risk!(model, r, opt, sqrt_kurtosis_risk, x_kurt, i; prefix = prefix)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `Kurtosis` with a continuous `Nothing` truncation parameter
to `model`.

Uses the full Cholesky-based SDP formulation to compute the portfolio kurtosis risk as a
second-order cone constraint over the vectorised weight matrix `W`. This overload applies
when the kurtosis truncation rank is `Nothing` (no truncation).

The elimination and summation matrices come from [`dup_elim_sum_selector`](@ref), so this
formulation is reachable under a [`LowOrderPrior`](@ref) whenever the measure holds its own
cokurtosis matrix: those two were the only other thing the kernel took from the prior, and
they are a pure function of the asset count.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::Kurtosis{<:Any, <:Any, <:Any, <:Any, Nothing, <:Any, <:Any}`: The kurtosis risk
    measure with no truncation.
  - $(arg_dict[:opt_rjumpe])
  - `pr::AbstractPriorResult`: Prior result. It supplies `kt` when the measure states none.

# Returns

  - `nothing`.

# Related

  - [`Kurtosis`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`set_kurtosis_risk!`](@ref)
  - [`assert_high_order_quantity`](@ref)
  - [`dup_elim_sum_selector`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::Kurtosis{<:Any, <:Any, <:Any, <:Any, Nothing, <:Any,
                                           <:Any}, opt::RiskJuMPOptimisationEstimator,
                               pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    assert_high_order_quantity(r.kt, pr, :Kurtosis, :kt, :CokurtosisEstimator)
    sc = get_constraint_scale(model)
    W = set_sdp_constraints!(model; prefix = prefix)
    L2, S2 = dup_elim_sum_selector(pr, size(W, 1))[2:3]
    G = if isnothing(r.kt)
        get_chol_or_Gkt_pm(model, pr)
    else
        LinearAlgebra.cholesky(S2 * r.kt * transpose(S2)).U
    end
    sqrt_kurtosis_risk = state_set!(model, prefix, :kurtosis_risk_, i,
                                    JuMP.@variable(model))
    L2W = state_build!(model, prefix, :L2W) do
        JuMP.@expression(model, L2 * vec(W))
    end
    x_kurt = state_set!(model, prefix, :x_kurt_, i, JuMP.@expression(model, G * L2W))
    state_set!(model, prefix, :ckurt_soc_, i,
               JuMP.@constraint(model,
                                [sc * sqrt_kurtosis_risk; sc * x_kurt] in
                                JuMP.SecondOrderCone()))
    return set_kurtosis_risk!(model, r, opt, sqrt_kurtosis_risk, x_kurt, i; prefix = prefix)
end
