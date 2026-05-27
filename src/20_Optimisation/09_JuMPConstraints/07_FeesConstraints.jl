"""
$(DocStringExtensions.TYPEDSIGNATURES)

Accumulate a JuMP expression into the `:fees` expression of the optimisation model.

Creates the `:fees` expression if it does not yet exist; otherwise adds `expr` to it in place.

# Arguments

  - $(arg_dict[:model])
  - `expr::JuMP.AbstractJuMPScalar`: The fee expression to accumulate.

# Returns

  - `nothing`.

# Related

  - [`set_non_fixed_fees!`](@ref)
  - [`set_long_non_fixed_fees!`](@ref)
  - [`set_short_non_fixed_fees!`](@ref)
  - [`set_turnover_fees!`](@ref)
  - [`Fees`](@ref)
"""
function add_to_fees!(model::JuMP.Model, expr::JuMP.AbstractJuMPScalar)
    if !haskey(model, :fees)
        JuMP.@expression(model, fees, expr)
    else
        fees = model[:fees]
        JuMP.add_to_expression!(fees, expr)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add a turnover-based transaction fee expression to the JuMP optimisation model.

The fall-through method does nothing. The concrete method computes `val' * |w - wt|` via NormOneCone constraints and accumulates the result into the model's `:fees` expression via [`add_to_fees!`](@ref).

# Mathematical definition

```math
\\begin{align}
t_{ftn,i} &\\geq |w_i - w_{t,i}\\, k|\\,, \\\\
f_{tn} &= \\boldsymbol{v}^\\intercal \\boldsymbol{t}_{ftn}\\,.
\\end{align}
```

Where:

  - ``w_i``: Portfolio weight for asset ``i``.
  - ``w_{t,i}``: Benchmark weight for asset ``i``.
  - $(math_dict[:k_budget])
  - ``\\boldsymbol{v}``: Per-asset fee rate vector.
  - ``\\boldsymbol{t}_{ftn}``: Auxiliary absolute-deviation variable vector.
  - ``f_{tn}``: Total turnover fee.

# Arguments

  - $(arg_dict[:model])
  - `tn::Turnover`: Turnover specification containing benchmark weights `w` and per-unit fee `val`.

# Returns

  - `nothing`.

# Related

  - [`add_to_fees!`](@ref)
  - [`set_non_fixed_fees!`](@ref)
  - [`set_turnover_constraints!`](@ref)
  - [`Turnover`](@ref)
  - [`Fees`](@ref)
"""
function set_turnover_fees!(args...)
    return nothing
end
function set_turnover_fees!(model::JuMP.Model, tn::Turnover)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    N = length(w)
    wt = tn.w
    val = tn.val
    JuMP.@variable(model, t_ftn[1:N])
    JuMP.@expressions(model, begin
                          x_ftn, w - wt * k
                          ftn, dot_scalar(val, t_ftn)
                      end)
    JuMP.@constraint(model, cftn[i = 1:N],
                     [sc * t_ftn[i]; sc * x_ftn[i]] in JuMP.MOI.NormOneCone(2))
    add_to_fees!(model, ftn)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add all non-fixed (proportional and turnover) fee expressions to the JuMP optimisation model.

The fall-through method does nothing. The concrete method delegates to [`set_long_non_fixed_fees!`](@ref), [`set_short_non_fixed_fees!`](@ref), and [`set_turnover_fees!`](@ref).

# Arguments

  - $(arg_dict[:model])
  - `fees::Fees`: Fee specification containing long, short, and turnover fee rates.

# Returns

  - `nothing`.

# Related

  - [`add_to_fees!`](@ref)
  - [`set_long_non_fixed_fees!`](@ref)
  - [`set_short_non_fixed_fees!`](@ref)
  - [`set_turnover_fees!`](@ref)
  - [`Fees`](@ref)
"""
function set_non_fixed_fees!(args...)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add proportional long-side fee expression to the JuMP optimisation model.

The fall-through method does nothing. The concrete method adds `fl' * lw` to the model's `:fees` expression via [`add_to_fees!`](@ref).

# Mathematical definition

```math
\\begin{align}
f_l &= \\boldsymbol{f}_l^\\intercal \\boldsymbol{lw}\\,.
\\end{align}
```

Where:

  - ``f_l``: Total long-side fee.
  - ``\\boldsymbol{f}_l``: Per-asset long-side fee rate vector.
  - ``\\boldsymbol{lw}``: Long-weight vector.

# Arguments

  - $(arg_dict[:model])
  - `fl`: Long-side fee rate(s). Accepts a scalar `Number` or a `VecNum`.

# Returns

  - `nothing`.

# Related

  - [`add_to_fees!`](@ref)
  - [`set_short_non_fixed_fees!`](@ref)
  - [`set_non_fixed_fees!`](@ref)
  - [`Fees`](@ref)
"""
function set_long_non_fixed_fees!(args...)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add proportional short-side fee expression to the JuMP optimisation model.

The fall-through method does nothing. The concrete method adds `fs' * sw` to the model's `:fees` expression via [`add_to_fees!`](@ref). Does nothing when no short-weight variable `:sw` exists in the model.

# Mathematical definition

```math
\\begin{align}
f_s &= \\boldsymbol{f}_s^\\intercal \\boldsymbol{sw}\\,.
\\end{align}
```

Where:

  - ``f_s``: Total short-side fee.
  - ``\\boldsymbol{f}_s``: Per-asset short-side fee rate vector.
  - ``\\boldsymbol{sw}``: Short-weight vector.

# Arguments

  - $(arg_dict[:model])
  - `fs`: Short-side fee rate(s). Accepts a scalar `Number` or a `VecNum`.

# Returns

  - `nothing`.

# Related

  - [`add_to_fees!`](@ref)
  - [`set_long_non_fixed_fees!`](@ref)
  - [`set_non_fixed_fees!`](@ref)
  - [`Fees`](@ref)
"""
function set_short_non_fixed_fees!(args...)
    return nothing
end
function set_long_non_fixed_fees!(model::JuMP.Model, fl::Num_VecNum)
    lw = model[:lw]
    JuMP.@expression(model, fl, dot_scalar(fl, lw))
    add_to_fees!(model, fl)
    return nothing
end
function set_short_non_fixed_fees!(model::JuMP.Model, fs::Num_VecNum)
    if !haskey(model, :sw)
        return nothing
    end
    sw = model[:sw]
    JuMP.@expression(model, fs, dot_scalar(fs, sw))
    add_to_fees!(model, fs)
    return nothing
end
function set_non_fixed_fees!(model::JuMP.Model, fees::Fees)
    set_long_non_fixed_fees!(model, fees.l)
    set_short_non_fixed_fees!(model, fees.s)
    set_turnover_fees!(model, fees.tn)
    return nothing
end
