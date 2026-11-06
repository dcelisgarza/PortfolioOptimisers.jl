"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the factor of the prior's projected cokurtosis matrix, and register it in the model on the first call.

The factor is the transpose of the square root of ``\\mathbf{S}_2 \\mathbf{K} \\mathbf{S}_2^\\intercal`` under `mtx_sqrt`, which [`matrix_square_root`](@ref) takes. `Gkt` is a shared entry, so every exact [`Kurtosis`](@ref) of the model that reads the prior's matrix reads one factor, and a later call returns it without a second factorisation. The first measure sets the algorithm. On a positive definite matrix every algorithm gives the Cholesky factor.

# JuMP formulation

## Expressions

  - `Gkt`: ``\\mathbf{G}_{kt}``, registered on the first call.

Where:

  - $(math_dict[:G_kt_factor])
  - $(math_dict[:S_2_sum])
  - ``\\mathbf{K}``: The prior's cokurtosis matrix, `pr.kt`.

# Arguments

  - $(arg_dict[:model])
  - `pr::HighOrderPrior`: The prior result. The function reads its `kt` and its `S2`.
  - `mtx_sqrt`: Square-root algorithm of the projected matrix, or `nothing` for the plain Cholesky factor.

# Validation

  - The projected matrix has a square root under `mtx_sqrt`, as [`matrix_square_root`](@ref) states. Raises a `LinearAlgebra.PosDefException`.

# Returns

  - `Gkt::AbstractMatrix`: The factor ``\\mathbf{G}_{kt}``.

# Related

  - [`matrix_square_root`](@ref)
  - [`get_kt_Akt_pm`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function get_chol_or_Gkt_pm(model::JuMP.Model, pr::HighOrderPrior,
                            mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot())
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
        G = transpose(matrix_square_root(mtx_sqrt, pr.S2 * pr.kt * transpose(pr.S2)))
        JuMP.@expression(model, Gkt, G)
    end
    return shared_get(model, :Gkt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the eigenvalues and the eigenvectors of the block vectorisation of the cokurtosis matrix `kt`, with each eigenvalue clamped at zero.

Row ``p + (q - 1) N`` of the block vectorisation holds the vectorisation of block ``(p, q)`` of ``\\mathbf{K}``. An entry of a cokurtosis matrix is a fourth comoment, which does not change when its four indices are permuted. So the block vectorisation of a cokurtosis matrix equals the matrix, and it is symmetric. The function reads it as `Symmetric`, so the decomposition is real.

# Mathematical definition

```math
\\begin{align}
\\mathrm{vec}_{N,N}(\\mathbf{K}) &= \\sum_{i=1}^{N^2} \\lambda_i \\boldsymbol{q}_i \\boldsymbol{q}_i^\\intercal\\,, \\\\
\\mathbf{K} &= \\sum_{i=1}^{N^2} \\mathbf{B}_i \\otimes \\mathbf{B}_i\\,, \\\\
m_4(\\boldsymbol{w}) &= \\sum_{i=1}^{N^2} \\left(\\boldsymbol{w}^\\intercal \\mathbf{B}_i \\boldsymbol{w}\\right)^2\\,.
\\end{align}
```

Where:

  - ``\\mathrm{vec}_{N,N}(\\mathbf{K})``: Block vectorisation of ``\\mathbf{K}`` in blocks of ``N \\times N``, [`block_vec_pq`](@ref).
  - ``\\lambda_i``, ``\\boldsymbol{q}_i``: Eigenvalue ``i`` and its unit eigenvector.
  - $(math_dict[:B_i_kt])
  - ``\\mathbf{K}``: The cokurtosis matrix `kt`, ``N^2 \\times N^2``.
  - $(math_dict[:m4_port])
  - $(math_dict[:w_port])

The second and third rows hold when no eigenvalue is negative. A cokurtosis matrix is positive semidefinite, so the clamp moves a rounding error only. The rank of ``\\mathbf{K}`` is at most ``N(N+1)/2``, because each column is the vectorisation of a symmetric matrix. So at most ``N(N+1)/2`` eigenvalues are not zero.

# Algorithm

 1. Build `A`, the block vectorisation of `kt`, with [`block_vec_pq`](@ref).
 2. Decompose the `Symmetric` view of `A` with `LinearAlgebra.eigen`, giving `vals` in ascending order and `vecs`.
 3. Clamp each entry of `vals` at zero.

# Arguments

  - `kt::MatNum`: The cokurtosis matrix, ``N^2 \\times N^2``.
  - `N::Integer`: The number of assets.

# Validation

  - Both sizes of `kt` are integer multiples of `N`. [`block_vec_pq`](@ref) raises a `DomainError` otherwise.

# Returns

  - `(vals, vecs)`: The clamped eigenvalues in ascending order, and the unit eigenvectors in the columns of `vecs`.

# Related

  - [`block_vec_pq`](@ref)
  - [`get_kt_Akt_pm`](@ref)
  - [`set_risk_constraints!`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 7.2.5.2, Equations 7.108 to 7.110. $(ref_dict[:no_original_source])
  - $(ref_dict[:pkurtapprox])
"""
function cokurtosis_spectrum(kt::MatNum, N::Integer)
    vals, vecs = LinearAlgebra.eigen(LinearAlgebra.Symmetric(block_vec_pq(kt, N, N)))
    return max.(vals, zero(eltype(vals))), vecs
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the spectrum of the prior's cokurtosis matrix, and register it in the model on the first call.

[`cokurtosis_spectrum`](@ref) computes the spectrum. `vals_Akt` and `vecs_Akt` are shared entries, so every approximate [`Kurtosis`](@ref) of the model that reads the prior's matrix reads one decomposition, and a later call returns it without a second decomposition.

# JuMP formulation

## Expressions

  - `vals_Akt`: The clamped eigenvalues in ascending order, registered on the first call.
  - `vecs_Akt`: The unit eigenvectors, registered on the first call.

# Arguments

  - $(arg_dict[:model])
  - `pr::HighOrderPrior`: The prior result. The function reads its `kt`, and the number of assets from its `mu`.

# Returns

  - `(vals_Akt, vecs_Akt)`: The clamped eigenvalues, and the unit eigenvectors in the columns of `vecs_Akt`.

# Related

  - [`cokurtosis_spectrum`](@ref)
  - [`get_chol_or_Gkt_pm`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function get_kt_Akt_pm(model::JuMP.Model, pr::HighOrderPrior)
    if !shared_has(model, :vecs_Akt)
        vals_A, vecs_A = cokurtosis_spectrum(pr.kt, length(pr.mu))
        JuMP.@expressions(model, begin
                              vecs_Akt, vecs_A
                              vals_Akt, vals_A
                          end)
    end
    return shared_get(model, :vals_Akt), shared_get(model, :vecs_Akt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Bound the epigraph of the square root of the kurtosis, and add the risk expression of the formulation that `r.alg` selects.

Both formulations of [`set_risk_constraints!`](@ref) for [`Kurtosis`](@ref) end with this call. The bound acts on ``g`` under every formulation. Under [`SOCRiskExpr`](@ref) the risk expression is ``g``, in the units of the square root of the fourth moment, and `settings.ub` is in the same units. Under the other three the risk expression is in the units of the fourth moment, and the bound row reads the square root of `settings.ub`. [`set_risk_upper_bound!`](@ref) adds the bound row, and [`set_risk_expression!`](@ref) adds the risk expression to `risk_vec`.

# JuMP formulation

## Variables

  - `kurtosis_risk_`: read, ``g``.
  - `x_kurt_`: read under [`QuadRiskExpr`](@ref) and [`RSOCRiskExpr`](@ref), ``\\boldsymbol{x}_{kt}``.
  - `tkurtosis_risk_`: created under [`RSOCRiskExpr`](@ref), the scalar ``t``.

## Expressions

  - `sq_kurtosis_risk_`: ``g^2``, under [`SquaredSOCRiskExpr`](@ref).
  - `qd_kurtosis_risk_`: ``\\lVert \\boldsymbol{x}_{kt} \\rVert_2^2``, under [`QuadRiskExpr`](@ref).

## Constraints

  - `ckurtosis_rsoc_`: under [`RSOCRiskExpr`](@ref), ``\\left(s_c t,\\, s_c / 2,\\, s_c \\boldsymbol{x}_{kt}\\right)`` lies in the rotated second-order cone, which is ``t \\geq \\lVert \\boldsymbol{x}_{kt} \\rVert_2^2``.

Each name carries the index `i` of the measure, and the namespace `prefix`.

Where:

  - $(math_dict[:g_kt_epi])
  - $(math_dict[:x_kt])
  - ``t``: Epigraph variable of the squared norm of ``\\boldsymbol{x}_{kt}``.
  - $(math_dict[:sc_scale])

## Relaxation

$(val_dict[:relax])

  - Under [`SquaredSOCRiskExpr`](@ref) `sq_kurtosis_risk_` lies at or above ``\\lVert \\boldsymbol{x}_{kt} \\rVert_2^2``, because ``g`` lies at or above ``\\lVert \\boldsymbol{x}_{kt} \\rVert_2``. Under [`RSOCRiskExpr`](@ref) ``t`` lies at or above ``\\lVert \\boldsymbol{x}_{kt} \\rVert_2^2``. Both are tight when the objective minimises the risk.
  - Under [`QuadRiskExpr`](@ref) `qd_kurtosis_risk_` is exact in ``\\boldsymbol{x}_{kt}``, and ``g`` serves the bound alone.
  - A bound on ``g`` is exact in ``\\boldsymbol{x}_{kt}``.

The relation of ``\\lVert \\boldsymbol{x}_{kt} \\rVert_2^2`` to the fourth moment of the weights is a property of the formulation, and each method of [`set_risk_constraints!`](@ref) states it.

# Arguments

  - $(arg_dict[:model])
  - `r::Kurtosis`: The measure, read for `alg` and `settings`.
  - $(arg_dict[:opt_rjumpe])
  - `sqrt_kurtosis_risk::JuMP.AbstractJuMPScalar`: ``g``.
  - `x_kurt`: ``\\boldsymbol{x}_{kt}``. The methods for [`SOCRiskExpr`](@ref) and [`SquaredSOCRiskExpr`](@ref) do not read it.
  - $(arg_dict[:ci])

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Returns

  - `risk`: The risk expression. It is ``g`` under [`SOCRiskExpr`](@ref), `sq_kurtosis_risk_` under [`SquaredSOCRiskExpr`](@ref), `qd_kurtosis_risk_` under [`QuadRiskExpr`](@ref), and ``t`` under [`RSOCRiskExpr`](@ref).

# Related

  - [`set_risk_constraints!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
  - [`set_risk_upper_bound!`](@ref)
  - [`set_risk_expression!`](@ref)
  - [`variance_risk_bounds_val`](@ref)
"""
function set_kurtosis_risk!(model::JuMP.Model,
                            r::Kurtosis{<:Any, <:Any, <:Any, <:SOCRiskExpr},
                            opt::RiskConstraintOwner,
                            sqrt_kurtosis_risk::JuMP.AbstractJuMPScalar, ::Any, i;
                            prefix::Symbol = Symbol(""))
    set_risk_bounds_and_expression!(model, opt, sqrt_kurtosis_risk, r.settings,
                                    :kurtosis_risk_, i; prefix = prefix)
    return sqrt_kurtosis_risk
end
function set_kurtosis_risk!(model::JuMP.Model,
                            r::Kurtosis{<:Any, <:Any, <:Any, <:SquaredSOCRiskExpr},
                            opt::RiskConstraintOwner,
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
                            r::Kurtosis{<:Any, <:Any, <:Any, <:QuadRiskExpr},
                            opt::RiskConstraintOwner,
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
                            r::Kurtosis{<:Any, <:Any, <:Any, <:RSOCRiskExpr},
                            opt::RiskConstraintOwner,
                            sqrt_kurtosis_risk::JuMP.AbstractJuMPScalar, x_kurt, i;
                            prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    tkurtosis = state_set!(model, prefix, :tkurtosis_risk_, i, JuMP.@variable(model))
    state_set!(model, prefix, :ckurtosis_rsoc_, i,
               JuMP.@constraint(model,
                                [sc * tkurtosis
                                 sc * 0.5
                                 sc * x_kurt] in JuMP.RotatedSecondOrderCone()))
    ub = variance_risk_bounds_val(SquareRootBound(), r.settings.ub)
    set_risk_upper_bound!(model, opt, sqrt_kurtosis_risk, ub,
                          state_key(prefix, :kurtosis_risk_, i))
    set_risk_expression!(model, tkurtosis, r.settings.scale, r.settings.rke)
    return tkurtosis
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the approximate kurtosis of a [`Kurtosis`](@ref) to the model, and return its risk expression.

This method applies when `r.N` is an integer. [`cokurtosis_spectrum`](@ref) writes the fourth moment as a sum of ``N^2`` squared quadratic forms, and the method keeps the ``N_f`` forms of the largest eigenvalues. A larger `r.N` is more accurate and more expensive. The rank of ``\\mathbf{K}`` is at most ``N(N+1)/2``, so every ``N_f \\geq N(N+1)/2`` keeps every form whose eigenvalue is not zero.

The method reads the measure's `kt` when the measure states one, and the prior's otherwise. It accepts any prior result, and [`assert_high_order_quantity`](@ref) refuses the measure when neither holds a cokurtosis matrix. Every approximate measure of the model that reads the prior's matrix reads one decomposition, [`get_kt_Akt_pm`](@ref).

# Mathematical definition

```math
\\begin{align}
\\widetilde{m}_4(\\boldsymbol{w}) &= \\sum_{i=1}^{N_f} \\left(\\boldsymbol{w}^\\intercal \\mathbf{B}_i \\boldsymbol{w}\\right)^2 \\leq m_4(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - ``\\widetilde{m}_4(\\boldsymbol{w})``: Approximate fourth central moment of the portfolio returns.
  - ``N_f``: Number of forms that the method keeps, `r.N` clamped to the range from ``1`` to ``N``, times ``N``.
  - $(math_dict[:m4_port])
  - $(math_dict[:B_i_kt])
  - $(math_dict[:K_cokurt])
  - $(math_dict[:w_port])

The model replaces ``\\boldsymbol{w} \\boldsymbol{w}^\\intercal`` with the lift ``\\mathbf{W}``, so each quadratic form becomes ``\\mathrm{Tr}(\\mathbf{B}_i \\mathbf{W})``, which is affine in the variables. A second-order cone bounds the norm of the vector of the ``N_f`` forms.

# Algorithm

 1. Check that `r.kt` or `pr` holds a cokurtosis matrix, with [`assert_high_order_quantity`](@ref).
 2. Read `sc`, the constraint scale, and `W`, the lift of [`set_sdp_constraints!`](@ref).
 3. Compute `Nf`, which is `r.N` clamped to `1:N`, times `N`.
 4. Create `sqrt_kurtosis_risk` and `x_kurt`, and register them under their keys.
 5. Read `vals_A` and `vecs_A`, the spectrum of `r.kt` from [`cokurtosis_spectrum`](@ref), else the spectrum of the prior's matrix from [`get_kt_Akt_pm`](@ref).
 6. Build `Bi`, the matrices of the `Nf` largest eigenvalues.
 7. Register the two rows below under their keys.
 8. Bound `sqrt_kurtosis_risk` and add the risk expression with [`set_kurtosis_risk!`](@ref), and return the risk expression.

# JuMP formulation

## Variables

  - `W`: read, ``\\mathbf{W}``, the lift that [`set_sdp_constraints!`](@ref) builds once for the owner of the weights.
  - `kurtosis_risk_`: created, the scalar ``g``.
  - `x_kurt_`: created, the vector ``\\boldsymbol{x}_{kt}`` of ``N_f`` entries.

## Constraints

  - `capprox_kurt_soc_`: ``(s_c g,\\, s_c \\boldsymbol{x}_{kt}) \\in \\mathcal{K}_{2}``, which is ``g \\geq \\lVert \\boldsymbol{x}_{kt} \\rVert_2``.
  - `capprox_kurt_`: ``s_c \\left(x_{kt,j} - \\mathrm{Tr}(\\mathbf{B}_j \\mathbf{W})\\right) = 0``, ``\\forall\\, j = 1,\\ldots,N_f``.

Each name carries the index `i` of the measure, and the namespace `prefix`.

Where:

  - $(math_dict[:g_kt_epi])
  - $(math_dict[:x_kt])
  - $(math_dict[:W_lift])
  - $(math_dict[:k_budget])
  - $(math_dict[:K_q_norm])
  - $(math_dict[:sc_scale])

## Relaxation

$(val_dict[:relax])

  - At ``\\mathbf{W} = \\boldsymbol{w} \\boldsymbol{w}^\\intercal / k``, ``\\lVert \\boldsymbol{x}_{kt} \\rVert_2^2 = \\widetilde{m}_4(\\boldsymbol{w}) / k^2``, which lies at or below ``m_4(\\boldsymbol{w}) / k^2``. The two are equal when ``N_f \\geq N(N+1)/2``.
  - With ``N_f \\geq N(N+1)/2``, ``\\lVert \\boldsymbol{x}_{kt} \\rVert_2^2`` is the squared norm of the exact formulation in another basis, and the relaxation of the exact formulation applies. With a smaller ``N_f`` the expression bounds ``m_4`` in neither direction away from rank one.
  - `kurtosis_risk_` lies at or above ``\\lVert \\boldsymbol{x}_{kt} \\rVert_2``. It is tight when the objective minimises the risk.
  - A bound on `kurtosis_risk_` holds the weights to ``\\widetilde{m}_4``, not to ``m_4``. With ``N_f < N(N+1)/2`` the returned weights can have a fourth moment above the bound.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::Kurtosis`: The measure, read for `kt`, `N`, `alg` and `settings`.
  - $(arg_dict[:opt_rjumpe])
  - `pr::AbstractPriorResult`: The prior result. It supplies `kt` when the measure states none.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Validation

  - `r.kt` or `pr` holds a cokurtosis matrix. [`assert_high_order_quantity`](@ref) raises an `ArgumentError` otherwise.

# Returns

  - `risk`: The kurtosis risk expression of [`set_kurtosis_risk!`](@ref).

# Related

  - [`Kurtosis`](@ref)
  - [`cokurtosis_spectrum`](@ref)
  - [`get_kt_Akt_pm`](@ref)
  - [`set_kurtosis_risk!`](@ref)
  - [`set_sdp_constraints!`](@ref)
  - [`assert_high_order_quantity`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 7.2.5.2, Equations 7.110 and 7.111. $(ref_dict[:no_original_source])
  - $(ref_dict[:pkurtapprox])
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::Kurtosis{<:Any, <:Any, <:Integer, <:Any, <:Any},
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
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
        cokurtosis_spectrum(r.kt, N)
    end
    # The eigenvalues ascend, so the last `Nf` pairs are the largest.
    N_eig = length(vals_A)
    Bi = [reshape(sqrt(vals_A[N_eig - j]) * view(vecs_A, :, N_eig - j), N, N)
          for j in 0:(Nf - 1)]
    capprox_kurt_soc, capprox_kurt = JuMP.@constraints(model,
                                                       begin
                                                           [sc * sqrt_kurtosis_risk
                                                            sc * x_kurt] in
                                                           JuMP.SecondOrderCone()
                                                           [j = 1:Nf],
                                                           sc * (x_kurt[j] -
                                                                 LinearAlgebra.dot(transpose(Bi[j]),
                                                                                   W)) == 0
                                                       end)
    state_set!(model, prefix, :capprox_kurt_soc_, i, capprox_kurt_soc)
    state_set!(model, prefix, :capprox_kurt_, i, capprox_kurt)
    return set_kurtosis_risk!(model, r, opt, sqrt_kurtosis_risk, x_kurt, i; prefix = prefix)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the kurtosis of a [`Kurtosis`](@ref) to the model in the exact semidefinite formulation, and return its risk expression.

This method applies when `r.N` is `nothing`. It is problem 13 of [pkurt](@cite), with the budget homogenised by ``k``. Equation 6 of that work states the fourth moment as a quadratic form in the entries on and below the diagonal of ``\\boldsymbol{w} \\boldsymbol{w}^\\intercal``, and the model replaces ``\\boldsymbol{w} \\boldsymbol{w}^\\intercal`` with the lift ``\\mathbf{W}``.

The method reads the measure's `kt` when the measure states one, and the prior's otherwise. Every exact measure of the model that reads the prior's matrix reads one factor, [`get_chol_or_Gkt_pm`](@ref). The elimination and summation matrices come from [`dup_elim_sum_selector`](@ref). They depend on the number of assets alone, so the method builds under a [`LowOrderPrior`](@ref) when the measure holds its own cokurtosis matrix. [`assert_high_order_quantity`](@ref) refuses the measure when neither holds one.

# Mathematical definition

```math
\\begin{align}
m_4(\\boldsymbol{w}) &= \\left[\\mathbf{L}_2 \\mathrm{vec}(\\boldsymbol{w} \\boldsymbol{w}^\\intercal)\\right]^\\intercal \\mathbf{S}_2 \\mathbf{K} \\mathbf{S}_2^\\intercal \\left[\\mathbf{L}_2 \\mathrm{vec}(\\boldsymbol{w} \\boldsymbol{w}^\\intercal)\\right] = \\left\\lVert \\mathbf{G}_{kt} \\mathbf{L}_2 \\mathrm{vec}(\\boldsymbol{w} \\boldsymbol{w}^\\intercal) \\right\\rVert_2^2\\,.
\\end{align}
```

Where:

  - $(math_dict[:m4_port])
  - $(math_dict[:L_2_elim])
  - $(math_dict[:S_2_sum])
  - $(math_dict[:K_cokurt])
  - $(math_dict[:G_kt_factor])
  - $(math_dict[:w_port])

# Algorithm

 1. Check that `r.kt` or `pr` holds a cokurtosis matrix, with [`assert_high_order_quantity`](@ref).
 2. Read `sc`, the constraint scale, and `W`, the lift of [`set_sdp_constraints!`](@ref).
 3. Read `L2` and `S2` for the size of `W`, with [`dup_elim_sum_selector`](@ref).
 4. Read `G`, the factor of `r.kt` projected by `S2` under `r.mtx_sqrt`, else the factor of the prior's matrix from [`get_chol_or_Gkt_pm`](@ref).
 5. Create `sqrt_kurtosis_risk`, and register it under its key.
 6. Build `L2W` once for the namespace, and build `x_kurt`.
 7. Register the cone row below under its key.
 8. Bound `sqrt_kurtosis_risk` and add the risk expression with [`set_kurtosis_risk!`](@ref), and return the risk expression.

# JuMP formulation

## Variables

  - `W`: read, ``\\mathbf{W}``, the lift that [`set_sdp_constraints!`](@ref) builds once for the owner of the weights.
  - `kurtosis_risk_`: created, the scalar ``g``.

## Expressions

  - `L2W`: ``\\mathbf{L}_2 \\mathrm{vec}(\\mathbf{W})``, registered once for each namespace.
  - `x_kurt_`: ``\\boldsymbol{x}_{kt} = \\mathbf{G}_{kt} \\mathbf{L}_2 \\mathrm{vec}(\\mathbf{W})``.

## Constraints

  - `ckurt_soc_`: ``(s_c g,\\, s_c \\boldsymbol{x}_{kt}) \\in \\mathcal{K}_{2}``, which is ``g \\geq \\lVert \\boldsymbol{x}_{kt} \\rVert_2``.

Each name except `L2W` carries the index `i` of the measure, and every name carries the namespace `prefix`.

Where:

  - $(math_dict[:g_kt_epi])
  - $(math_dict[:x_kt])
  - $(math_dict[:W_lift])
  - $(math_dict[:k_budget])
  - $(math_dict[:K_q_norm])
  - $(math_dict[:sc_scale])

## Relaxation

$(val_dict[:relax])

  - ``\\lVert \\boldsymbol{x}_{kt} \\rVert_2^2 = \\mathrm{vec}(\\mathbf{W})^\\intercal \\mathbf{K} \\mathrm{vec}(\\mathbf{W})``. A sample cokurtosis is a mean of ``(\\boldsymbol{y}_t \\otimes \\boldsymbol{y}_t)(\\boldsymbol{y}_t \\otimes \\boldsymbol{y}_t)^\\intercal`` over centred observations ``\\boldsymbol{y}_t``, with non-negative weights. For such a matrix the expression is the mean of ``(\\boldsymbol{y}_t^\\intercal \\mathbf{W} \\boldsymbol{y}_t)^2``, and ``\\mathbf{W} \\succeq \\boldsymbol{w} \\boldsymbol{w}^\\intercal / k`` gives ``\\boldsymbol{y}_t^\\intercal \\mathbf{W} \\boldsymbol{y}_t \\geq (\\boldsymbol{y}_t^\\intercal \\boldsymbol{w})^2 / k``. So the expression lies at or above ``m_4(\\boldsymbol{w}) / k^2``.
  - `kurtosis_risk_` lies at or above ``\\lVert \\boldsymbol{x}_{kt} \\rVert_2``, so at or above ``\\sqrt{m_4(\\boldsymbol{w})} / k``. It is tight at ``\\mathbf{W} = \\boldsymbol{w} \\boldsymbol{w}^\\intercal / k``, which the objective reaches when it minimises the risk and no other row acts on ``\\mathbf{W}``.
  - A bound on `kurtosis_risk_` is exact, because the weights satisfy it when ``\\sqrt{m_4(\\boldsymbol{w})}`` is at most the bound.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::Kurtosis`: The measure, read for `kt`, `mtx_sqrt`, `alg` and `settings`.
  - $(arg_dict[:opt_rjumpe])
  - `pr::AbstractPriorResult`: The prior result. It supplies `kt` when the measure states none.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Validation

  - `r.kt` or `pr` holds a cokurtosis matrix. [`assert_high_order_quantity`](@ref) raises an `ArgumentError` otherwise.
  - The projected matrix has a square root under `r.mtx_sqrt`, as [`matrix_square_root`](@ref) states. Raises a `LinearAlgebra.PosDefException`.

# Returns

  - `risk`: The kurtosis risk expression of [`set_kurtosis_risk!`](@ref).

# Related

  - [`Kurtosis`](@ref)
  - [`get_chol_or_Gkt_pm`](@ref)
  - [`set_kurtosis_risk!`](@ref)
  - [`set_sdp_constraints!`](@ref)
  - [`dup_elim_sum_selector`](@ref)
  - [`assert_high_order_quantity`](@ref)

# References

  - $(ref_dict[:pkurt]) Section 2, Equation 6, and Section 3.1, Equation 13.
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::Kurtosis{<:Any, <:Any, Nothing, <:Any, <:Any},
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    assert_high_order_quantity(r.kt, pr, :Kurtosis, :kt, :CokurtosisEstimator)
    sc = get_constraint_scale(model)
    W = set_sdp_constraints!(model; prefix = prefix)
    L2, S2 = dup_elim_sum_selector(pr, size(W, 1))[2:3]
    G = if isnothing(r.kt)
        get_chol_or_Gkt_pm(model, pr, r.mtx_sqrt)
    else
        transpose(matrix_square_root(r.mtx_sqrt, S2 * r.kt * transpose(S2)))
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
