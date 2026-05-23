"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add an effective number of assets constraint to the JuMP optimisation model.

The fall-through method does nothing. The concrete method introduces an auxiliary variable `nea` and enforces `‖w‖₂ ≤ nea` via a SecondOrderCone constraint, combined with `nea * √val ≤ k`. This is equivalent to requiring the effective number of assets to be at least `val`.

# Summary Statistics

```math
\\mathrm{nea} \\geq \\|\\boldsymbol{w}\\|_2, \\qquad \\mathrm{nea} \\cdot \\sqrt{\\mathrm{val}} \\leq k
\\quad \\Leftrightarrow \\quad \\mathrm{ENA}(\\boldsymbol{w}) = \\frac{1}{\\|\\boldsymbol{w}\\|_2^2} \\geq \\mathrm{val}
```

where ``\\mathrm{ENA}(\\boldsymbol{w})`` is the effective number of assets, ``k`` is the budget scaling variable, and `val` is the minimum required effective number of assets.

# Arguments

  - $(arg_dict[:model])
  - `val::Number`: Minimum required effective number of assets.

# Returns

  - `nothing`.

# Related

  - [`number_effective_assets`](@ref)
  - [`EqualRiskMeasure`](@ref)
"""
function set_number_effective_assets!(args...)
    return nothing
end
function set_number_effective_assets!(model::JuMP.Model, val::Number)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    JuMP.@variable(model, nea)
    JuMP.@constraints(model, begin
                          cnea_soc, [sc * nea; sc * w] in JuMP.SecondOrderCone()
                          cnea, sc * (nea * sqrt(val) - k) <= 0
                      end)
    return nothing
end
