"""
$(DocStringExtensions.TYPEDSIGNATURES)

Accumulate a JuMP expression into the `:fees` expression of the optimisation model.

Creates the `:fees` expression if it does not yet exist; otherwise adds `expr` to it in place.

# Arguments

  - $(arg_dict[:model])
  - `expr`: The fee expression to accumulate. A plain number is accepted beside a JuMP
    scalar, because a charge that does not depend on the decision variables — the forced
    liquidation of [`set_liquidation_fees!`](@ref) — is a constant under any objective
    whose `k` is not a variable.

# Returns

  - `nothing`.

# Related

  - [`set_non_fixed_fees!`](@ref)
  - [`set_long_non_fixed_fees!`](@ref)
  - [`set_short_non_fixed_fees!`](@ref)
  - [`set_turnover_fees!`](@ref)
  - [`Fees`](@ref)
"""
function add_to_fees!(model::JuMP.Model, expr::Union{<:Number, <:JuMP.AbstractJuMPScalar})
    if !shared_has(model, :fees)
        JuMP.@expression(model, fees, expr)
    else
        fees = shared_get(model, :fees)
        JuMP.add_to_expression!(fees, expr)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Accumulate a one-off fee expression into the model's `:one_time_fees` expression.

The twin of [`add_to_fees!`](@ref), for the terms that are charged one time for the whole holding
period rather than on every observation. Only the two fixed fees reach it, because `l`, `s` and
`tn` are rates per period. [`set_net_portfolio_returns!`](@ref) subtracts this expression from the
first observation alone, and [`add_fees_to_ret!`](@ref) divides it by the observation count,
because an expected return is a per period number.

# Arguments

  - $(arg_dict[:model])
  - `expr`: The fee expression to accumulate.

# Returns

  - `nothing`.

# Related

  - [`add_to_fees!`](@ref)
  - [`set_fixed_fees!`](@ref)
  - [`set_net_portfolio_returns!`](@ref)
  - [`add_fees_to_ret!`](@ref)
"""
function add_to_one_time_fees!(model::JuMP.Model,
                               expr::Union{<:Number, <:JuMP.AbstractJuMPScalar})
    if !shared_has(model, :one_time_fees)
        JuMP.@expression(model, one_time_fees, expr)
    else
        one_time_fees = shared_get(model, :one_time_fees)
        JuMP.add_to_expression!(one_time_fees, expr)
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
    w = get_w(model)
    k = get_k(model)
    sc = get_constraint_scale(model)
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
    set_liquidation_fees!(::JuMP.Model, ::Nothing)
    set_liquidation_fees!(model::JuMP.Model, lq::Turnover)

Add the proportional cost of a forced exit to the model's `:fees` expression.

`lq` prices the positions that leave the Investable Mask, and those assets are **not** in the model's `w`: the optimisation reduced them away at its entry. The charge is therefore a **constant**, not a function of the decision variables, and it needs no auxiliary variable and no norm constraint — which is what separates it from [`set_turnover_fees!`](@ref), whose `|w - wt * k|` does depend on `w`.

The constant is multiplied by the homogenising variable `k`, exactly as the turnover term is, so a ratio objective sees the charge in the same units as every other fee and the exit moves the argmin rather than riding outside the programme.

`lq` is a rate per period, so the charge joins `:fees` beside `l`, `s` and `tn` through [`add_to_fees!`](@ref), and no clock reaches it.

# Algorithm

 1. On a `nothing` `lq`, do nothing. No asset left the universe.
 2. Otherwise read `k`, the homogenising variable.
 3. Compute the constant through [`calc_liquidation_fees`](@ref), which reads the carrier alone: a forced exit trades to zero, so the charge is the rate times `abs.(lq.w)`.
 4. Register `constant * k` and add it to `:fees` through [`add_to_fees!`](@ref).

# Arguments

  - `model`: JuMP model.
  - `lq`: The proportional liquidation carrier, or `nothing`.

# Returns

  - `nothing`: The model is modified in place.

# Related

  - [`Fees`](@ref)
  - [`set_non_fixed_fees!`](@ref)
  - [`set_turnover_fees!`](@ref)
  - [`add_to_fees!`](@ref)
  - [`calc_liquidation_fees`](@ref)
  - [`set_fixed_liquidation_fees!`](@ref)
"""
function set_liquidation_fees!(::JuMP.Model, ::Nothing)
    return nothing
end
function set_liquidation_fees!(model::JuMP.Model, lq::Turnover)
    k = get_k(model)
    val = calc_liquidation_fees(lq)
    JuMP.@expression(model, flq_prop, val * k)
    add_to_fees!(model, flq_prop)
    return nothing
end
"""
    set_fixed_liquidation_fees!(::JuMP.Model, ::Nothing, ::NamedTuple)
    set_fixed_liquidation_fees!(model::JuMP.Model, flq::Turnover, kwargs::NamedTuple)

Add the fixed cost of a forced exit to the model's `:one_time_fees` expression.

The fixed twin of [`set_liquidation_fees!`](@ref), and a constant for the same reason: the liquidated assets are not in `w`. Unlike [`set_fixed_fees!`](@ref) it therefore needs **no binary indicator**, because whether each position is held is already known from `flq.w` rather than decided by the programme.

`flq` is a currency amount charged one time for the whole holding period, so it joins `:one_time_fees` beside `fl` and `fs` through [`add_to_one_time_fees!`](@ref), and [`charge_one_time_fees`](@ref) then lands it on the clock `fees.fa` names.

# Algorithm

 1. On a `nothing` `flq`, do nothing.
 2. Otherwise read `k`, the homogenising variable.
 3. Compute the constant through [`calc_fixed_liquidation_fees`](@ref), which charges both the liquidated long and the liquidated short side.
 4. Register `constant * k` and add it to `:one_time_fees` through [`add_to_one_time_fees!`](@ref).

# Arguments

  - `model`: JuMP model.
  - `flq`: The fixed liquidation carrier, or `nothing`.
  - `kwargs`: Forwarded to `isapprox` to decide how near zero counts as zero.

# Returns

  - `nothing`: The model is modified in place.

# Related

  - [`Fees`](@ref)
  - [`set_non_fixed_fees!`](@ref)
  - [`set_fixed_fees!`](@ref)
  - [`add_to_one_time_fees!`](@ref)
  - [`calc_fixed_liquidation_fees`](@ref)
  - [`set_liquidation_fees!`](@ref)
"""
function set_fixed_liquidation_fees!(::JuMP.Model, ::Nothing, ::NamedTuple)
    return nothing
end
function set_fixed_liquidation_fees!(model::JuMP.Model, flq::Turnover, kwargs::NamedTuple)
    k = get_k(model)
    val = calc_fixed_liquidation_fees(flq, kwargs)
    JuMP.@expression(model, flq_fixed, val * k)
    add_to_one_time_fees!(model, flq_fixed)
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
    lw = shared_get(model, :lw)
    JuMP.@expression(model, fl, dot_scalar(fl, lw))
    add_to_fees!(model, fl)
    return nothing
end
function set_short_non_fixed_fees!(model::JuMP.Model, fs::Num_VecNum)
    if !shared_has(model, :sw)
        return nothing
    end
    sw = shared_get(model, :sw)
    JuMP.@expression(model, fs, dot_scalar(fs, sw))
    add_to_fees!(model, fs)
    return nothing
end
function set_non_fixed_fees!(model::JuMP.Model, fees::Fees)
    set_long_non_fixed_fees!(model, fees.l)
    set_short_non_fixed_fees!(model, fees.s)
    set_turnover_fees!(model, fees.tn)
    set_liquidation_fees!(model, fees.lq)
    set_fixed_liquidation_fees!(model, fees.flq, fees.kwargs)
    # The clock decides whether the one-off terms are spread at all, and it belongs to the
    # fee, so the fee builder registers it. The holding period they are spread over is the
    # observation count of the fit, which the head has already registered as `:T`.
    # `set_net_portfolio_returns!` reads the clock, and `add_fees_to_ret!` reads the count.
    if !shared_has(model, :fee_fa)
        shared_set!(model, :fee_fa, fees.fa)
    end
    return nothing
end
"""
    charge_one_time_fees(model::JuMP.Model, net, one_time, T::Number,
                         fa::Option{<:AbstractFeeAmortisation})

Lay the model's one-off fee expression onto a net return series, on the clock the fee states.

The model's twin of [`charge_fees`](@ref), and it states the same rule. The two fixed fees are
charged one time for the whole holding period, so a `nothing` or [`FirstObservationFees`](@ref)
clock subtracts them from the first observation alone, and an [`AmortisedFees`](@ref) spreads them
evenly over `T`, the observation count of the fit.

# Algorithm

 1. On a `nothing` or [`FirstObservationFees`](@ref) `fa`, subtract `one_time` from the first entry
    of `net` and leave the rest.
 2. On an [`AmortisedFees`](@ref) `fa`, build the share `one_time / T` once and subtract it
    from every entry of `net`.

# Arguments

  - $(arg_dict[:model])
  - `net`: The net return expression, already charged the per period terms.
  - `one_time`: The model's `:one_time_fees` expression.
  - `T`: Observation count of the fit, from [`get_T`](@ref).
  - `fa`: The fee's clock, from `:fee_fa`.

# Returns

  - The net return expression, charged the one-off terms.

# Related

  - [`charge_fees`](@ref)
  - [`add_to_one_time_fees!`](@ref)
  - [`set_net_portfolio_returns!`](@ref)
  - [`AmortisedFees`](@ref)
  - [`FirstObservationFees`](@ref)
"""
function charge_one_time_fees(model::JuMP.Model, net, one_time, ::Number,
                              ::Union{Nothing, <:FirstObservationFees})
    # One entry is charged, so one entry is touched. Writing the charge as a selector vector
    # times `one_time` would build two arrays and put a zero multiple of the fee into every
    # other observation's expression, which the model then carries for nothing. `net` is
    # built by the one caller and handed straight here, so the term is added in place.
    JuMP.add_to_expression!(net[1], -1, one_time)
    return net
end
function charge_one_time_fees(model::JuMP.Model, net, one_time, T::Number, ::AmortisedFees)
    # Every entry is charged, but all of them are charged the same share, so the share is
    # built once rather than once per observation. Writing it as a broadcast would build a
    # second array the length of `net` and throw the first away. `net` is built by the one
    # caller and handed straight here, so the term is added in place.
    share = JuMP.@expression(model, one_time / T)
    for n in net
        JuMP.add_to_expression!(n, -1, share)
    end
    return net
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add fixed-fee expressions to the JuMP optimisation model.

A fixed fee is charged per *position held*, whatever its size, so unlike the proportional fees
above it cannot be written against the weights — it needs a binary saying whether the position
is there at all. That is the only reason this one takes an indicator bundle, and the only
reason a MIP builder has to run before it.

# Mathematical definition

```math
\\begin{align}
f_{fl} &= \\boldsymbol{f}_{fl}^\\intercal \\boldsymbol{b}^l\\,, &
f_{fs} &= \\boldsymbol{f}_{fs}^\\intercal \\boldsymbol{b}^s\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{b}^l``, ``\\boldsymbol{b}^s``: Long and short binaries ([`long_bin`](@ref), [`short_bin`](@ref)). These are the binaries themselves, never the gates: a fee is incurred by the decision to hold, which is what the bit records, and the gates relax to continuous variables when the budget is free.
  - ``\\boldsymbol{f}_{fl}``, ``\\boldsymbol{f}_{fs}``: Long and short fixed-fee rates.

Under a long-only builder the held bit is the long bit ([`HeldIndicators`](@ref)), and there is
no short side to charge.

# Arguments

  - $(arg_dict[:model])
  - `sp::AbstractMIPSpace`: Weight space the fees are charged in.
  - `ind::AbstractMIPIndicators`: Indicator bundle supplying the binaries.
  - `ffl::Option{<:Num_VecNum}`: Long-side fixed fee rate(s).
  - `ffs::Option{<:Num_VecNum}`: Short-side fixed fee rate(s).
  - `ffl_flag::Bool`: Whether to add the long fixed-fee expression.
  - `ffs_flag::Bool`: Whether to add the short fixed-fee expression.

# Returns

  - `nothing`.

# Related

  - [`add_to_fees!`](@ref)
  - [`set_non_fixed_fees!`](@ref)
  - [`AbstractMIPIndicators`](@ref)
  - [`Fees`](@ref)
"""
function set_fixed_fees!(model::JuMP.Model, sp::AbstractMIPSpace,
                         ind::AbstractMIPIndicators, ffl::Option{<:Num_VecNum},
                         ffs::Option{<:Num_VecNum}, ffl_flag::Bool, ffs_flag::Bool)
    if ffl_flag
        ffl = model[mip_key(sp, :ffl)] = JuMP.@expression(model,
                                                          dot_scalar(ffl, long_bin(ind)))
        add_to_one_time_fees!(model, ffl)
    end
    if ffs_flag
        ffs = model[mip_key(sp, :ffs)] = JuMP.@expression(model,
                                                          dot_scalar(ffs, short_bin(ind)))
        add_to_one_time_fees!(model, ffs)
    end
    return nothing
end
