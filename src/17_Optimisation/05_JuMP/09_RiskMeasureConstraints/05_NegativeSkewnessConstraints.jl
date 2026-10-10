"""
$(DocStringExtensions.TYPEDSIGNATURES)

Retrieve or compute and cache the square-root matrix of the co-skewness matrix `V`.

If `model` does not yet contain `GV`, computes the transpose of the square root of `pr.V` under
`mtx_sqrt` with [`matrix_square_root`](@ref), and stores it as the `:GV` Model State entry.
Every measure of the model that reads the prior's matrix reads this one factor, so the first
measure sets the algorithm. On a positive definite matrix every algorithm gives the Cholesky
factor.

# Arguments

  - $(arg_dict[:model])
  - `pr::HighOrderPrior`: High-order prior containing `V`.
  - `mtx_sqrt`: Square-root algorithm, or `nothing` for the plain Cholesky factor.

# Returns

  - `GV::Matrix`: Square-root factor of the co-skewness matrix.

# Related

  - [`set_negative_skewness_risk!`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function get_chol_or_V_pm(model::JuMP.Model, pr::HighOrderPrior,
                          mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot())
    if !shared_has(model, :GV)
        G = transpose(matrix_square_root(mtx_sqrt, pr.V))
        JuMP.@expression(model, GV, G)
    end
    return shared_get(model, :GV)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Finalise the negative-skewness risk expression and apply bounds according to the formulation.

The `SOCRiskExpr` overload passes the SOC variable directly to
[`set_risk_bounds_and_expression!`](@ref). The `SquaredSOCRiskExpr` overload squares it. The
`QuadRiskExpr` overload encodes skewness as the quadratic form `w' * V * w`.

# Arguments

  - $(arg_dict[:model])
  - `r::NegativeSkewness`: Negative-skewness risk measure instance.
  - $(arg_dict[:opt_rjumpe])
  - `nskew_risk`: SOC variable for negative-skewness risk.
  - $(arg_dict[:ci])
  - `V::MatNum`: Co-skewness matrix (used only by the Quad overload).

# Returns

  - The negative-skewness risk JuMP expression.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`variance_risk_bounds_val`](@ref)
"""
function set_negative_skewness_risk!(model::JuMP.Model,
                                     r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                         <:SOCRiskExpr},
                                     opt::RiskConstraintOwner,
                                     nskew_risk::JuMP.AbstractJuMPScalar, i, args...;
                                     prefix::Symbol = Symbol(""))
    set_risk_bounds_and_expression!(model, opt, nskew_risk, r.settings, :nskew_risk_, i;
                                    prefix = prefix)
    return nskew_risk
end
function set_negative_skewness_risk!(model::JuMP.Model,
                                     r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                         <:SquaredSOCRiskExpr},
                                     opt::RiskConstraintOwner,
                                     nskew_risk::JuMP.AbstractJuMPScalar, i, args...;
                                     prefix::Symbol = Symbol(""))
    qnskew_risk = state_set!(model, prefix, :sq_nskew_risk_, i,
                             JuMP.@expression(model, nskew_risk^2))
    ub = variance_risk_bounds_val(SquareRootBound(), r.settings.ub)
    set_risk_upper_bound!(model, opt, nskew_risk, ub, state_key(prefix, :nskew_risk_, i))
    set_risk_expression!(model, qnskew_risk, r.settings.scale, r.settings.rke)
    return qnskew_risk
end
function set_negative_skewness_risk!(model::JuMP.Model,
                                     r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                         <:QuadRiskExpr},
                                     opt::RiskConstraintOwner,
                                     nskew_risk::JuMP.AbstractJuMPScalar, i, V::MatNum;
                                     prefix::Symbol = Symbol(""))
    w = get_w(model, prefix)
    qnskew_risk = state_set!(model, prefix, :qd_nskew_risk_, i,
                             JuMP.@expression(model, LinearAlgebra.dot(w, V, w)))
    ub = variance_risk_bounds_val(SquareRootBound(), r.settings.ub)
    set_risk_upper_bound!(model, opt, nskew_risk, ub, state_key(prefix, :nskew_risk_, i))
    set_risk_expression!(model, qnskew_risk, r.settings.scale, r.settings.rke)
    return qnskew_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add negative-skewness risk constraints to `model`.

Selects the co-skewness matrix (from `r.V` or `pr.V`), creates a scalar variable, adds the
SOC constraint `[sc * nskew_risk; sc * G * w] in SOC`, and dispatches to
[`set_negative_skewness_risk!`](@ref) for bounding.

Any prior result is accepted. `V` must resolve on one side or the other, and
[`assert_high_order_quantity`](@ref) refuses the measure when it resolves on neither. `V`
travels with `sk` out of one fit, so a measure that holds its own pair takes nothing else
from the prior.

# Mathematical definition

```math
\\begin{align}
\\mathrm{NSkew}(\\boldsymbol{w}) &= \\lVert \\mathbf{G}_V \\boldsymbol{w} \\rVert_2\\,, \\\\
\\mathbf{G}_V^\\intercal \\mathbf{G}_V &= \\mathbf{V}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{NSkew}(\\boldsymbol{w})``: Negative skewness risk measure.
  - ``\\mathbf{G}_V``: Transpose of the square root of the projected co-skewness matrix ``\\mathbf{V}`` that [`matrix_square_root`](@ref) takes under the `mtx_sqrt` of the measure.
  - $(math_dict[:w_port])

where ``\\mathbf{V}`` is the co-skewness matrix projected onto the weight space.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::NegativeSkewness`: Negative-skewness risk measure instance.
  - $(arg_dict[:opt_rjumpe])
  - `pr::AbstractPriorResult`: Prior result. It supplies `V` when the measure states none.

# Returns

  - `nothing`.

# Related

  - [`get_chol_or_V_pm`](@ref)
  - [`set_negative_skewness_risk!`](@ref)
  - [`assert_high_order_quantity`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::NegativeSkewness,
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    # `sk` and `V` are both-or-neither on either side, so the gate reads the pair through
    # `sk` and the kernel below can take `pr.V` whenever it takes `pr.sk`.
    assert_high_order_quantity(r.sk, pr, :NegativeSkewness, :sk, :CoskewnessEstimator)
    sc = get_constraint_scale(model)
    w = get_w(model, prefix)
    V, G = if isnothing(r.V)
        (pr.V, get_chol_or_V_pm(model, pr, r.mtx_sqrt))
    else
        (r.V, transpose(matrix_square_root(r.mtx_sqrt, r.V)))
    end
    nskew_risk = state_set!(model, prefix, :nskew_risk_, i, JuMP.@variable(model))
    state_set!(model, prefix, :cnskew_soc_, i,
               JuMP.@constraint(model,
                                [sc * nskew_risk; sc * G * w] in JuMP.SecondOrderCone()))
    return set_negative_skewness_risk!(model, r, opt, nskew_risk, i, V; prefix = prefix)
end
