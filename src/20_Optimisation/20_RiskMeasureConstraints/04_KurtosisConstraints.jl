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
  - $(arg_dict[:key_sym])
  - `i`: Constraint index for unique naming (used by the RSOC overload).

# Returns

  - The kurtosis risk JuMP expression.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`variance_risk_bounds_val`](@ref)
"""
function set_kurtosis_risk!(model::JuMP.Model,
                            r::Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:SOCRiskExpr}, opt::RiskJuMPOptimisationEstimator,
                            sqrt_kurtosis_risk::JuMP.AbstractJuMPScalar, ::Any, key::Symbol,
                            args...)
    set_risk_bounds_and_expression!(model, opt, sqrt_kurtosis_risk, r.settings, key)
    return sqrt_kurtosis_risk
end
function set_kurtosis_risk!(model::JuMP.Model,
                            r::Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:SquaredSOCRiskExpr},
                            opt::RiskJuMPOptimisationEstimator,
                            sqrt_kurtosis_risk::JuMP.AbstractJuMPScalar, ::Any, key::Symbol,
                            args...)
    qsqrt_kurtosis_risk = model[Symbol(:sq_, key)] = JuMP.@expression(model,
                                                                      sqrt_kurtosis_risk^2)
    ub = variance_risk_bounds_val(SquareRootBound(), r.settings.ub)
    set_risk_upper_bound!(model, opt, sqrt_kurtosis_risk, ub, key)
    set_risk_expression!(model, qsqrt_kurtosis_risk, r.settings.scale, r.settings.rke)
    return qsqrt_kurtosis_risk
end
function set_kurtosis_risk!(model::JuMP.Model,
                            r::Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:QuadRiskExpr}, opt::RiskJuMPOptimisationEstimator,
                            sqrt_kurtosis_risk::JuMP.AbstractJuMPScalar, x_kurt,
                            key::Symbol, args...)
    qsqrt_kurtosis_risk = model[Symbol(:qd_, key)] = JuMP.@expression(model,
                                                                      LinearAlgebra.dot(x_kurt,
                                                                                        x_kurt))
    ub = variance_risk_bounds_val(SquareRootBound(), r.settings.ub)
    set_risk_upper_bound!(model, opt, sqrt_kurtosis_risk, ub, key)
    set_risk_expression!(model, qsqrt_kurtosis_risk, r.settings.scale, r.settings.rke)
    return qsqrt_kurtosis_risk
end
function set_kurtosis_risk!(model::JuMP.Model,
                            r::Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:RSOCRiskExpr}, opt::RiskJuMPOptimisationEstimator,
                            sqrt_kurtosis_risk::JuMP.AbstractJuMPScalar, x_kurt,
                            key::Symbol, i::Any)
    sc = get_constraint_scale(model)
    tkurtosis = model[Symbol(:tkurtosis_risk, i)] = JuMP.@variable(model)
    qsqrt_kurtosis_risk = model[Symbol(:ckurtosis_rsoc_, i)] = JuMP.@constraint(model,
                                                                                [sc *
                                                                                 tkurtosis
                                                                                 0.5;
                                                                                 sc *
                                                                                 x_kurt] in
                                                                                JuMP.RotatedSecondOrderCone())
    ub = variance_risk_bounds_val(SquareRootBound(), r.settings.ub)
    set_risk_upper_bound!(model, opt, sqrt_kurtosis_risk, ub, key)
    set_risk_expression!(model, qsqrt_kurtosis_risk, r.settings.scale, r.settings.rke)
    return qsqrt_kurtosis_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add kurtosis risk constraints to `model`.

The `Integer N` overload uses an approximate spectral decomposition of the co-kurtosis tensor
to build `N` eigen-directions and encodes kurtosis via SOC and equality constraints. The
`Nothing N` overload uses the full Cholesky-based formulation with the duplication matrix.
The `LowOrderPrior` overload unconditionally throws an `ArgumentError` since kurtosis
estimation requires a high-order prior.

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
  - `pr::HighOrderPrior`: High-order prior containing `kt`, `S2`, and `L2`.

# Returns

  - `nothing`.

# Related

  - [`get_chol_or_Gkt_pm`](@ref)
  - [`get_kt_Akt_pm`](@ref)
  - [`set_kurtosis_risk!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Integer, <:Any,
                                           <:Any}, opt::RiskJuMPOptimisationEstimator,
                               pr::HighOrderPrior, args...; prefix::Symbol = Symbol(""),
                               kwargs...)
    key = Symbol(:kurtosis_risk_, i)
    sc = get_constraint_scale(model)
    W = set_sdp_constraints!(model; prefix = prefix)
    N = size(W, 1)
    f = clamp(r.N, 1, N)
    Nf = f * N
    sqrt_kurtosis_risk, x_kurt = model[key], model[Symbol(:x_kurt_, i)] = JuMP.@variables(model,
                                                                                          begin
                                                                                              ()
                                                                                              [1:Nf]
                                                                                          end)
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
    model[Symbol(:capprox_kurt_soc_, i)], model[Symbol(:capprox_kurt_, i)] = JuMP.@constraints(model,
                                                                                               begin
                                                                                                   [sc *
                                                                                                    sqrt_kurtosis_risk
                                                                                                    sc *
                                                                                                    x_kurt] in
                                                                                                   JuMP.SecondOrderCone()
                                                                                                   [i = 1:Nf],
                                                                                                   sc *
                                                                                                   (x_kurt[i] -
                                                                                                    LinearAlgebra.tr(Bi[i] *
                                                                                                                     W)) ==
                                                                                                   0
                                                                                               end)
    return set_kurtosis_risk!(model, r, opt, sqrt_kurtosis_risk, x_kurt, key, i)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `Kurtosis` with a continuous `Nothing` truncation parameter
to `model`.

Uses the full Cholesky-based SDP formulation to compute the portfolio kurtosis risk as a
second-order cone constraint over the vectorised weight matrix `W`. This overload applies
when the kurtosis truncation rank is `Nothing` (no truncation).

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::Kurtosis{<:Any, <:Any, <:Any, <:Any, Nothing, <:Any, <:Any}`: The kurtosis risk
    measure with no truncation.
  - $(arg_dict[:opt_rjumpe])
  - `pr::HighOrderPrior`: High-order prior containing the kurtosis matrix `kt`.

# Returns

  - `nothing`.

# Related

  - [`Kurtosis`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`set_kurtosis_risk!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::Kurtosis{<:Any, <:Any, <:Any, <:Any, Nothing, <:Any,
                                           <:Any}, opt::RiskJuMPOptimisationEstimator,
                               pr::HighOrderPrior, args...; prefix::Symbol = Symbol(""),
                               kwargs...)
    key = Symbol(:kurtosis_risk_, i)
    sc = get_constraint_scale(model)
    W = set_sdp_constraints!(model; prefix = prefix)
    G = if isnothing(r.kt)
        get_chol_or_Gkt_pm(model, pr)
    else
        LinearAlgebra.cholesky(pr.S2 * r.kt * transpose(pr.S2)).U
    end
    sqrt_kurtosis_risk = model[key] = JuMP.@variable(model)
    L2W = state_build!(model, prefix, :L2W) do
        L2 = pr.L2
        JuMP.@expression(model, L2 * vec(W))
    end
    x_kurt = model[Symbol(:x_kurt_, i)] = JuMP.@expression(model, G * L2W)
    model[Symbol(:ckurt_soc_, i)] = JuMP.@constraint(model,
                                                     [sc * sqrt_kurtosis_risk;
                                                      sc * x_kurt] in
                                                     JuMP.SecondOrderCone())
    return set_kurtosis_risk!(model, r, opt, sqrt_kurtosis_risk, x_kurt, key, i)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Throw an `ArgumentError` indicating that `Kurtosis` requires a `HighOrderPrior`.

This fall-through overload is triggered when a `LowOrderPrior` is passed and always
raises an error.

# Arguments

  - `r::Kurtosis`: The kurtosis risk measure (unused).
  - `pr::LowOrderPrior`: A low-order prior (not compatible with kurtosis).

# Returns

  - Does not return; always throws `ArgumentError`.

# Related

  - [`Kurtosis`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(::JuMP.Model, ::Any, ::Kurtosis,
                               ::RiskJuMPOptimisationEstimator, pr::LowOrderPrior, args...;
                               kwargs...)
    return throw(ArgumentError("Kurtosis requires a HighOrderPrior, not a $(typeof(pr))."))
end
