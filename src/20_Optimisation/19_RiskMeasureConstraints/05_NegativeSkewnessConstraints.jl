"""
$(DocStringExtensions.TYPEDSIGNATURES)

Retrieve or compute and cache the square-root matrix of the co-skewness matrix `V`.

If `model` does not yet contain `GV`, attempts a Cholesky factorisation of `pr.V` and falls
back to `sqrt(pr.V)` for positive-semidefinite matrices. Stores the result as `model[:GV]`.

# Arguments

  - $(arg_dict[:model])
  - `pr::HighOrderPrior`: High-order prior containing `V`.

# Returns

  - `GV::Matrix`: Square-root factor of the co-skewness matrix.

# Related

  - [`set_negative_skewness_risk!`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function get_chol_or_V_pm(model::JuMP.Model, pr::HighOrderPrior)
    if !haskey(model, :GV)
        G = try
            LinearAlgebra.cholesky(pr.V).U
        catch err
            if isa(err, LinearAlgebra.PosDefException)
                sqrt(pr.V)
            else
                rethrow(err)
            end
        end
        JuMP.@expression(model, GV, G)
    end
    return model[:GV]
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
  - $(arg_dict[:key_sym])
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
                                     opt::RiskJuMPOptimisationEstimator,
                                     nskew_risk::JuMP.AbstractJuMPScalar, key::Symbol,
                                     args...)
    set_risk_bounds_and_expression!(model, opt, nskew_risk, r.settings, key)
    return nskew_risk
end
function set_negative_skewness_risk!(model::JuMP.Model,
                                     r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                         <:SquaredSOCRiskExpr},
                                     opt::RiskJuMPOptimisationEstimator,
                                     nskew_risk::JuMP.AbstractJuMPScalar, key::Symbol,
                                     args...)
    qnskew_risk = model[Symbol(:sq_, key)] = JuMP.@expression(model, nskew_risk^2)
    ub = variance_risk_bounds_val(false, r.settings.ub)
    set_risk_upper_bound!(model, opt, nskew_risk, ub, key)
    set_risk_expression!(model, qnskew_risk, r.settings.scale, r.settings.rke)
    return qnskew_risk
end
function set_negative_skewness_risk!(model::JuMP.Model,
                                     r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                         <:QuadRiskExpr},
                                     opt::RiskJuMPOptimisationEstimator,
                                     nskew_risk::JuMP.AbstractJuMPScalar, key::Symbol,
                                     V::MatNum)
    w = model[:w]
    qnskew_risk = model[Symbol(:qd_, key)] = JuMP.@expression(model,
                                                              LinearAlgebra.dot(w, V, w))
    ub = variance_risk_bounds_val(false, r.settings.ub)
    set_risk_upper_bound!(model, opt, nskew_risk, ub, key)
    set_risk_expression!(model, qnskew_risk, r.settings.scale, r.settings.rke)
    return qnskew_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add negative-skewness risk constraints to `model`.

The `HighOrderPrior` overload selects the co-skewness matrix (from `r.V` or `pr.V`),
creates a scalar variable, adds the SOC constraint `[sc * nskew_risk; sc * G * w] in SOC`,
and dispatches to [`set_negative_skewness_risk!`](@ref) for bounding. The `LowOrderPrior`
overload unconditionally throws an `ArgumentError`.

# Mathematical definition

```math
\\begin{align}
\\mathrm{NSkew}(\\boldsymbol{w}) &= \\|\\mathbf{G}_V \\boldsymbol{w}\\|_2\\,, \\\\
\\mathbf{G}_V &= \\mathrm{chol}(\\mathbf{V})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{NSkew}(\\boldsymbol{w})``: Negative skewness risk measure.
  - ``\\mathbf{G}_V``: Cholesky factor of the projected co-skewness matrix ``\\mathbf{V}``.
  - $(math_dict[:w_port])

where ``\\mathbf{V}`` is the co-skewness matrix projected onto the weight space.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::NegativeSkewness`: Negative-skewness risk measure instance.
  - $(arg_dict[:opt_rjumpe])
  - `pr::HighOrderPrior`: High-order prior containing `V`.

# Returns

  - `nothing`.

# Related

  - [`get_chol_or_V_pm`](@ref)
  - [`set_negative_skewness_risk!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::NegativeSkewness,
                               opt::RiskJuMPOptimisationEstimator, pr::HighOrderPrior,
                               args...; kwargs...)
    key = Symbol(:nskew_risk_, i)
    sc = model[:sc]
    w = model[:w]
    V, G = if isnothing(r.V)
        (pr.V, get_chol_or_V_pm(model, pr))
    else
        (r.V, LinearAlgebra.cholesky(r.V).U)
    end
    nskew_risk = model[key] = JuMP.@variable(model)
    model[Symbol(:cnskew_soc_, i)] = JuMP.@constraint(model,
                                                      [sc * nskew_risk; sc * G * w] in
                                                      JuMP.SecondOrderCone())
    return set_negative_skewness_risk!(model, r, opt, nskew_risk, key, V)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Throw an `ArgumentError` indicating that `NegativeSkewness` requires a `HighOrderPrior`.

This fall-through overload is triggered when a `LowOrderPrior` is passed and always
raises an error.

# Arguments

  - `r::NegativeSkewness`: The negative skewness risk measure (unused).
  - `pr::LowOrderPrior`: A low-order prior (not compatible with negative skewness).

# Returns

  - Does not return; always throws `ArgumentError`.

# Related

  - [`NegativeSkewness`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(::JuMP.Model, ::Any, ::NegativeSkewness,
                               ::RiskJuMPOptimisationEstimator, pr::LowOrderPrior, args...;
                               kwargs...)
    throw(ArgumentError("NegativeSkewness requires a HighOrderPrior, not a $(typeof(pr))."))
end
