"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add a term to the per period fee of the JuMP model.

The first call registers `:fees` as an affine expression equal to `expr`, so a number can take a later term. Each later call adds `expr` to it in place. `:fees` is a shared entry, so a nested build reads it under the bare key.

# JuMP formulation

## Expressions

  - `fees`: ``f_r = \\sum_j e_j``, the sum of the terms that the builders pass. [`set_net_portfolio_returns!`](@ref) subtracts it from every observation, and [`add_fees_to_ret!`](@ref) subtracts it from the expected return.

Where:

  - $(math_dict[:f_r_fee])
  - ``e_j``: The ``j``-th term passed in, `expr`.

# Arguments

  - $(arg_dict[:model])
  - `expr`: The term to add. A plain number is accepted beside a JuMP scalar, because the forced liquidation of [`set_liquidation_fees!`](@ref) does not depend on the decision variables, and it is a number when `k` is a constant.

# Returns

  - `nothing`.

# Related

  - [`add_to_one_time_fees!`](@ref): The twin for the terms that the holding period pays one time.
  - [`set_long_non_fixed_fees!`](@ref)
  - [`set_short_non_fixed_fees!`](@ref)
  - [`set_turnover_fees!`](@ref)
  - [`set_liquidation_fees!`](@ref)
  - [`Fees`](@ref)
"""
function add_to_fees!(model::JuMP.Model, expr::Union{<:Number, <:JuMP.AbstractJuMPScalar})
    if !shared_has(model, :fees)
        # A number takes no later term in place, so the first term starts an affine
        # expression, whatever its own type.
        T = JuMP.value_type(typeof(model))
        V = JuMP.variable_ref_type(model)
        JuMP.@expression(model, fees, zero(JuMP.GenericAffExpr{T, V}) + expr)
    else
        fees = shared_get(model, :fees)
        JuMP.add_to_expression!(fees, expr)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add a term to the one-off fee of the JuMP model.

The holding period pays these terms one time, and each observation does not. The three fixed fees reach it: `fl` and `fs` through [`set_fixed_fees!`](@ref), and `flq` through [`set_fixed_liquidation_fees!`](@ref). [`set_net_portfolio_returns!`](@ref) charges it on the clock of the fee, and [`add_fees_to_ret!`](@ref) divides it by the observation count, because an expected return is a per period number. The first call registers `:one_time_fees` as an affine expression equal to `expr`, and each later call adds `expr` to it in place. It is a shared entry, so a nested build reads it under the bare key.

# JuMP formulation

## Expressions

  - `one_time_fees`: ``f_o = \\sum_j e_j``, the sum of the terms that the builders pass.

Where:

  - $(math_dict[:f_o_fee])
  - ``e_j``: The ``j``-th term passed in, `expr`.

# Arguments

  - $(arg_dict[:model])
  - `expr`: The term to add. A plain number is accepted beside a JuMP scalar, because the fixed liquidation charge is a number when `k` is a constant.

# Returns

  - `nothing`.

# Related

  - [`add_to_fees!`](@ref): The twin for the terms that each observation pays.
  - [`set_fixed_fees!`](@ref)
  - [`set_fixed_liquidation_fees!`](@ref)
  - [`charge_one_time_fees`](@ref)
  - [`set_net_portfolio_returns!`](@ref)
  - [`add_fees_to_ret!`](@ref)
"""
function add_to_one_time_fees!(model::JuMP.Model,
                               expr::Union{<:Number, <:JuMP.AbstractJuMPScalar})
    if !shared_has(model, :one_time_fees)
        # A number takes no later term in place, so the first term starts an affine
        # expression, whatever its own type.
        T = JuMP.value_type(typeof(model))
        V = JuMP.variable_ref_type(model)
        JuMP.@expression(model, one_time_fees, zero(JuMP.GenericAffExpr{T, V}) + expr)
    else
        one_time_fees = shared_get(model, :one_time_fees)
        JuMP.add_to_expression!(one_time_fees, expr)
    end
    return nothing
end
"""
    set_turnover_fees!(model::JuMP.Model, tn::Turnover)
    set_turnover_fees!(args...)

Add the turnover fee to the per period fee of the JuMP model.

The fee charges the rate of each asset on the distance of its weight from the reference weight of `tn`. The model holds each distance in an epigraph variable, so the charge is linear. The reference weight is multiplied by `k`, so under [`MaximumRatio`](@ref) the fee acts on the homogenised weights ``k \\boldsymbol{w}`` and scales with them. The method on `args...` does nothing, and it runs when the fee states no turnover term.

# Mathematical definition

```math
\\begin{align}
F_{\\text{tn}} &= \\sum_{i = 1}^{N} v_i \\left\\lvert w_i - w_{\\text{tn},i} \\right\\rvert\\,.
\\end{align}
```

Where:

  - ``F_{\\text{tn}}``: Turnover fee.
  - ``v_i``: Fee rate of asset ``i``, the `val` of `tn`. A scalar applies to every asset.
  - ``w_{\\text{tn},i}``: Reference weight of asset ``i``, the `w` of `tn`.
  - $(math_dict[:w_i_asset])
  - $(math_dict[:N])

# JuMP formulation

## Variables

  - `w`, `k`: read from the model.
  - `t_ftn`: created, one epigraph entry for each asset.

## Expressions

  - `x_ftn`: ``\\boldsymbol{x} = \\boldsymbol{w} - \\boldsymbol{w}_{\\text{tn}} k``.
  - `ftn`: ``\\boldsymbol{v}^\\intercal \\boldsymbol{t}``. [`add_to_fees!`](@ref) adds it to ``f_r``.

## Constraints

  - `cftn[i]`: ``\\left(s_c t_i,\\, s_c x_i\\right)`` lies in the 1-norm cone of dimension two, which is ``t_i \\geq \\lvert x_i \\rvert``.

Where:

  - ``\\boldsymbol{t}``: The epigraph variable `t_ftn`.
  - ``\\boldsymbol{x}``: The deviation `x_ftn`.
  - ``\\boldsymbol{v}``: Fee rate vector, the `val` of `tn`.
  - ``\\boldsymbol{w}_{\\text{tn}}``: Reference weight vector, the `w` of `tn`.
  - $(math_dict[:f_r_fee])
  - $(math_dict[:sc_scale])
  - $(math_dict[:k_budget])
  - $(math_dict[:w_port])

## Relaxation

$(val_dict[:relax])

  - The rows give ``t_i \\geq \\lvert x_i \\rvert``, so `ftn` lies at or above ``\\boldsymbol{v}^\\intercal \\lvert \\boldsymbol{w} - \\boldsymbol{w}_{\\text{tn}} k \\rvert``.
  - The bound is tight when the model pulls `:fees` down. An objective that maximises a return net of the fee does, and so do a binding return floor and a risk measure that reads the net return series. When nothing reads the fee, `t_ftn` can lie above the distance, and the weights do not depend on it.

# Arguments

  - $(arg_dict[:model])
  - `tn::Turnover`: The reference weights `w` and the fee rate `val`.

# Returns

  - `nothing`.

# Related

  - [`add_to_fees!`](@ref)
  - [`set_non_fixed_fees!`](@ref)
  - [`set_turnover_constraints!`](@ref): Reads the same [`Turnover`](@ref) as a bound on each distance.
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

Add the proportional cost of a forced exit to the per period fee of the JuMP model.

`lq` prices the positions that leave the Investable Mask. Those assets are not in the model's `w`, because the optimisation removed them before it built the model. So the charge is a constant of the model, and it needs no epigraph variable and no row. The constant is multiplied by `k`, as the turnover term is, so a ratio objective sees it in the units of every other fee. `lq` is a rate per period, so the charge joins ``f_r`` beside `l`, `s` and `tn`.

# Mathematical definition

```math
\\begin{align}
F_{\\text{lq}} &= \\sum_{i} v_i \\left\\lvert w_{\\text{lq},i} \\right\\rvert\\,.
\\end{align}
```

Where:

  - ``F_{\\text{lq}}``: Proportional liquidation fee.
  - ``v_i``: Fee rate of the ``i``-th asset that left, the `val` of `lq`. A scalar applies to every such asset.
  - ``w_{\\text{lq},i}``: Previous weight of the ``i``-th asset that left, the `w` of `lq`. A forced exit sells each position to zero, so the trade is its absolute value.

# Algorithm

 1. On a `nothing` `lq`, do nothing. No asset left the universe.
 2. Otherwise read `k`.
 3. Compute ``F_{\\text{lq}}`` through [`calc_liquidation_fees`](@ref).
 4. Register ``F_{\\text{lq}} k`` and add it to ``f_r`` through [`add_to_fees!`](@ref).

# JuMP formulation

## Variables

  - `k`: read from the model.

## Expressions

  - `flq_prop`: ``F_{\\text{lq}} k``.

Where:

  - ``F_{\\text{lq}}``: Proportional liquidation fee.
  - $(math_dict[:f_r_fee])
  - $(math_dict[:k_budget])

# Arguments

  - $(arg_dict[:model])
  - `lq`: The proportional liquidation carrier, or `nothing`.

# Returns

  - `nothing`.

# Related

  - [`Fees`](@ref)
  - [`set_non_fixed_fees!`](@ref)
  - [`set_turnover_fees!`](@ref): Charges a distance that depends on `w`, so it needs an epigraph variable.
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

Add the fixed cost of a forced exit to the one-off fee of the JuMP model.

It is the fixed twin of [`set_liquidation_fees!`](@ref), and it is a constant of the model for the same reason. [`set_fixed_fees!`](@ref) needs a binary for each asset, because the model decides which positions it holds. This charge needs none, because the positions that the exit sold are known from `flq.w`. The holding period pays it one time, so it joins ``f_o`` beside `fl` and `fs`, and [`charge_one_time_fees`](@ref) puts it on the clock of the fee.

# Mathematical definition

```math
\\begin{align}
F_{\\text{flq}} &= \\sum_{i} v_i\\, 1\\left\\{w_{\\text{flq},i} \\neq 0\\right\\}\\,.
\\end{align}
```

Where:

  - ``F_{\\text{flq}}``: Fixed liquidation fee.
  - ``v_i``: Fixed fee of the ``i``-th asset that left, the `val` of `flq`. A scalar applies to every such asset.
  - ``w_{\\text{flq},i}``: Previous weight of the ``i``-th asset that left, the `w` of `flq`. A long and a short position both pay.
  - ``1\\left\\{\\cdot\\right\\}``: Indicator function. ``w \\neq 0`` is `!isapprox(w, 0; kwargs...)`.

# Algorithm

 1. On a `nothing` `flq`, do nothing.
 2. Otherwise read `k`.
 3. Compute ``F_{\\text{flq}}`` through [`calc_fixed_liquidation_fees`](@ref).
 4. Register ``F_{\\text{flq}} k`` and add it to ``f_o`` through [`add_to_one_time_fees!`](@ref).

# JuMP formulation

## Variables

  - `k`: read from the model.

## Expressions

  - `flq_fixed`: ``F_{\\text{flq}} k``.

Where:

  - ``F_{\\text{flq}}``: Fixed liquidation fee.
  - $(math_dict[:f_o_fee])
  - $(math_dict[:k_budget])

# Arguments

  - $(arg_dict[:model])
  - `flq`: The fixed liquidation carrier, or `nothing`.
  - `kwargs`: Forwarded to `isapprox`, to decide how near zero a weight counts as zero.

# Returns

  - `nothing`.

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
    set_non_fixed_fees!(model::JuMP.Model, fees::Fees)
    set_non_fixed_fees!(args...)

Add every fee of `fees` that needs no binary to the JuMP model.

These are the proportional fees `l` and `s`, the turnover fee `tn`, and the two liquidation charges `lq` and `flq`. The fixed fees `fl` and `fs` need a binary for each asset, so the MIP builder charges them through [`set_fixed_fees!`](@ref) before this builder runs. The method on `args...` does nothing, and it runs when the optimisation states no fee.

# Algorithm

 1. Charge `fees.l` through [`set_long_non_fixed_fees!`](@ref).
 2. Charge `fees.s` through [`set_short_non_fixed_fees!`](@ref).
 3. Charge `fees.tn` through [`set_turnover_fees!`](@ref).
 4. Charge `fees.lq` through [`set_liquidation_fees!`](@ref).
 5. Charge `fees.flq` through [`set_fixed_liquidation_fees!`](@ref), with `fees.kwargs`.
 6. When the model holds no `:fee_fa`, register `fees.fa` under it. [`set_net_portfolio_returns!`](@ref) reads it as the clock of the one-off fee.

# Arguments

  - $(arg_dict[:model])
  - `fees::Fees`: The fee of the optimisation.

# Returns

  - `nothing`.

# Related

  - [`add_to_fees!`](@ref)
  - [`set_long_non_fixed_fees!`](@ref)
  - [`set_short_non_fixed_fees!`](@ref)
  - [`set_turnover_fees!`](@ref)
  - [`set_liquidation_fees!`](@ref)
  - [`set_fixed_liquidation_fees!`](@ref)
  - [`set_fixed_fees!`](@ref)
  - [`Fees`](@ref)
"""
function set_non_fixed_fees!(args...)
    return nothing
end
"""
    set_long_non_fixed_fees!(model::JuMP.Model, fl::Num_VecNum)
    set_long_non_fixed_fees!(args...)

Add the proportional fee of the long positions to the per period fee of the JuMP model.

The fee charges the model's long part `lw`, and not the weights. `lw` holds the homogenised long part, so under [`MaximumRatio`](@ref) the fee scales with the weights. The method on `args...` does nothing, and it runs when the fee states no `l`.

# Mathematical definition

```math
\\begin{align}
F_l &= \\sum_{i = 1}^{N} f_{l,i} \\max\\left(w_i, 0\\right)\\,.
\\end{align}
```

Where:

  - ``F_l``: Proportional long fee.
  - ``f_{l,i}``: Fee rate of asset ``i``, `fl`. A scalar applies to every asset.
  - $(math_dict[:w_i_asset])
  - $(math_dict[:N])

# JuMP formulation

## Variables

  - `lw`: read from the model.

## Expressions

  - `fl`: ``\\boldsymbol{f}_l^\\intercal \\boldsymbol{w}^{l}``. [`add_to_fees!`](@ref) adds it to ``f_r``.

Where:

  - ``\\boldsymbol{f}_l``: Fee rate vector, `fl`.
  - ``\\boldsymbol{w}^{l}``: Long part of the weights, `lw`.
  - $(math_dict[:f_r_fee])

## Relaxation

$(val_dict[:relax])

  - Under a [`PartsBoundWeights`](@ref) head, `lw` lies at or above the long part of `w`, so `fl` lies at or above ``F_l``.
  - The bound is tight when the decomposition is pinned, which `xbgt = true` on [`JuMPOptimiser`](@ref) does, and under a [`WeightsFromParts`](@ref) head with a sign pattern. The budget pins `sum(lw)` and `sum(sw)` whether or not the weights reach them, so without the pin a long-short model can pay the fee on an exposure that it does not hold.

# Arguments

  - $(arg_dict[:model])
  - `fl::Num_VecNum`: Fee rate of each asset, or one rate for every asset.

# Returns

  - `nothing`.

# Related

  - [`add_to_fees!`](@ref)
  - [`set_short_non_fixed_fees!`](@ref)
  - [`set_non_fixed_fees!`](@ref)
  - [`PartsBoundWeights`](@ref)
  - [`Fees`](@ref)
"""
function set_long_non_fixed_fees!(args...)
    return nothing
end
"""
    set_short_non_fixed_fees!(model::JuMP.Model, fs::Num_VecNum)
    set_short_non_fixed_fees!(args...)

Add the proportional fee of the short positions to the per period fee of the JuMP model.

The fee charges the model's short part `sw`, and not the weights. A long-only model registers no `sw`, and then the method does nothing. The method on `args...` does nothing, and it runs when the fee states no `s`.

# Mathematical definition

```math
\\begin{align}
F_s &= \\sum_{i = 1}^{N} f_{s,i} \\max\\left(-w_i, 0\\right)\\,.
\\end{align}
```

Where:

  - ``F_s``: Proportional short fee.
  - ``f_{s,i}``: Fee rate of asset ``i``, `fs`. A scalar applies to every asset.
  - $(math_dict[:w_i_asset])
  - $(math_dict[:N])

# JuMP formulation

## Variables

  - `sw`: read from the model.

## Expressions

  - `fs`: ``\\boldsymbol{f}_s^\\intercal \\boldsymbol{w}^{s}``. [`add_to_fees!`](@ref) adds it to ``f_r``.

Where:

  - ``\\boldsymbol{f}_s``: Fee rate vector, `fs`.
  - ``\\boldsymbol{w}^{s}``: Short part of the weights, `sw`. It is non-negative.
  - $(math_dict[:f_r_fee])

## Relaxation

$(val_dict[:relax])

  - Under a [`PartsBoundWeights`](@ref) head, `sw` lies at or above the short part of `w`, so `fs` lies at or above ``F_s``.
  - The bound is tight under the conditions that [`set_long_non_fixed_fees!`](@ref) states. Without the pin, a model that holds no short position still pays the fee on the whole short budget.

# Arguments

  - $(arg_dict[:model])
  - `fs::Num_VecNum`: Fee rate of each asset, or one rate for every asset.

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

Charge the one-off fee of the JuMP model to a net return series, on the clock of the fee.

It is the model's twin of [`charge_fees`](@ref), and it states the same rule. A `nothing` or [`FirstObservationFees`](@ref) clock charges the whole fee to the first observation. An [`AmortisedFees`](@ref) clock charges an equal part of it to every observation of the fit.

# Mathematical definition

```math
\\begin{align}
\\hat{r}^{\\prime}_t &= \\hat{r}_t - c_t f_o\\,.
\\end{align}
```

Where:

  - ``\\hat{r}^{\\prime}_t``: Entry ``t`` of the returned series.
  - $(math_dict[:rhat_t_net])
  - $(math_dict[:c_t_one_off])
  - $(math_dict[:f_o_fee])
  - $(math_dict[:T])

# Algorithm

 1. On a `nothing` or [`FirstObservationFees`](@ref) `fa`, subtract `one_time` from the first entry of `net` in place, and leave the other entries.
 2. On an [`AmortisedFees`](@ref) `fa`, build `share = one_time / T` one time, and subtract it from every entry of `net` in place.

# JuMP formulation

## Expressions

  - `share`, not registered: ``f_o / T``. Only the [`AmortisedFees`](@ref) method builds it.

Where:

  - $(math_dict[:f_o_fee])
  - $(math_dict[:T])

# Arguments

  - $(arg_dict[:model])
  - `net`: The net return series, already charged ``f_r``. The function changes its entries.
  - `one_time`: The one-off fee `:one_time_fees`.
  - `T`: Observation count of the fit, from [`get_T`](@ref).
  - `fa`: The clock of the fee, from `:fee_fa`.

# Returns

  - `net`, charged the one-off fee.

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
    set_fixed_fees!(model::JuMP.Model, sp::AbstractMIPSpace, ind::HeldIndicators,
                    ffl::Option{<:Num_VecNum}, ffs::Option{<:Num_VecNum},
                    ffl_flag::Bool, ffs_flag::Bool)
    set_fixed_fees!(model::JuMP.Model, sp::AbstractMIPSpace, ind::LongShortIndicators,
                    ffl::Option{<:Num_VecNum}, ffs::Option{<:Num_VecNum},
                    ffl_flag::Bool, ffs_flag::Bool)

Add the fixed fees of the held positions to the one-off fee of the JuMP model.

A fixed fee charges each position that the portfolio holds, whatever its size. So it cannot be written against the weights, and it reads the held binaries of `ind`. That is the reason a MIP builder calls it.

The fee charges the gate of each binary, which is the binary times `k`, and not the binary. Under [`MaximumRatio`](@ref) the weights are the homogenised ``k \\boldsymbol{w}`` and `k` is a variable, so the fee scales with the weights, as every other fee does, and the weights do not depend on `ohf`. When `k` is a constant, or the long-short asset space runs on a unit budget, the gate is the binary itself. [`HeldIndicators`](@ref) has no short side, so it charges `ffl` alone.

# Mathematical definition

```math
\\begin{align}
F_{\\text{f}} &= \\boldsymbol{f}_{\\text{fl}}^\\intercal \\boldsymbol{b}^{l} + \\boldsymbol{f}_{\\text{fs}}^\\intercal \\boldsymbol{b}^{s}\\,.
\\end{align}
```

Where:

  - ``F_{\\text{f}}``: Fixed fee.
  - ``\\boldsymbol{f}_{\\text{fl}}``, ``\\boldsymbol{f}_{\\text{fs}}``: Long and short fixed fee of each position, `ffl` and `ffs`. A scalar applies to every position.
  - ``\\boldsymbol{b}^{l}``, ``\\boldsymbol{b}^{s}``: Long and short held binaries. Under [`HeldIndicators`](@ref) ``\\boldsymbol{b}^{l}`` is the held binary and ``\\boldsymbol{b}^{s} = \\boldsymbol{0}``.

# JuMP formulation

## Variables

  - The long gate ``\\boldsymbol{g}^{l}``: read from `ind`. It is `i_mip` of [`HeldIndicators`](@ref), and `il` of [`LongShortIndicators`](@ref).
  - The short gate ``\\boldsymbol{g}^{s}``: read from `ind`. It is `is` of [`LongShortIndicators`](@ref).

## Expressions

  - `ffl`, under the key that [`mip_key`](@ref) gives in `sp`: ``\\boldsymbol{f}_{\\text{fl}}^\\intercal \\boldsymbol{g}^{l}``, when `ffl_flag` is set. [`add_to_one_time_fees!`](@ref) adds it to ``f_o``.
  - `ffs`, under the key that [`mip_key`](@ref) gives in `sp`: ``\\boldsymbol{f}_{\\text{fs}}^\\intercal \\boldsymbol{g}^{s}``, when `ffs_flag` is set. [`add_to_one_time_fees!`](@ref) adds it to ``f_o``.

Where:

  - ``\\boldsymbol{g}^{l} = k \\boldsymbol{b}^{l}``, ``\\boldsymbol{g}^{s} = k \\boldsymbol{b}^{s}``: Long and short gates. The big-M rows of the declaration make each entry equal to ``k`` or to zero, when the big-M is not below ``k``.
  - ``\\boldsymbol{f}_{\\text{fl}}``, ``\\boldsymbol{f}_{\\text{fs}}``: Long and short fixed fee of each position.
  - $(math_dict[:f_o_fee])
  - $(math_dict[:k_budget])

# Arguments

  - $(arg_dict[:model])
  - `sp::AbstractMIPSpace`: Weight space of the positions.
  - `ind`: Indicator bundle that holds the gates.
  - `ffl::Option{<:Num_VecNum}`: Long fixed fee of each position.
  - `ffs::Option{<:Num_VecNum}`: Short fixed fee of each position. [`HeldIndicators`](@ref) does not read it.
  - `ffl_flag::Bool`: Whether to charge `ffl`.
  - `ffs_flag::Bool`: Whether to charge `ffs`. [`HeldIndicators`](@ref) does not read it.

# Returns

  - `nothing`.

# Related

  - [`add_to_one_time_fees!`](@ref)
  - [`set_non_fixed_fees!`](@ref)
  - [`mip_constraints`](@ref)
  - [`short_mip_threshold_constraints`](@ref)
  - [`AbstractMIPIndicators`](@ref)
  - [`Fees`](@ref)
"""
function set_fixed_fees!(model::JuMP.Model, sp::AbstractMIPSpace, ind::HeldIndicators,
                         ffl::Option{<:Num_VecNum}, ::Option{<:Num_VecNum}, ffl_flag::Bool,
                         ::Bool)
    return set_fixed_fees!(model, sp, (held(ind), nothing), ffl, nothing, ffl_flag, false)
end
function set_fixed_fees!(model::JuMP.Model, sp::AbstractMIPSpace, ind::LongShortIndicators,
                         ffl::Option{<:Num_VecNum}, ffs::Option{<:Num_VecNum},
                         ffl_flag::Bool, ffs_flag::Bool)
    return set_fixed_fees!(model, sp, (long_gate(ind), short_gate(ind)), ffl, ffs, ffl_flag,
                           ffs_flag)
end
function set_fixed_fees!(model::JuMP.Model, sp::AbstractMIPSpace, gates::Tuple,
                         ffl::Option{<:Num_VecNum}, ffs::Option{<:Num_VecNum},
                         ffl_flag::Bool, ffs_flag::Bool)
    lg, sg = gates
    # The gate is the binary times `k`, so the fee scales with the homogenised weights as
    # every other fee does. A charge on the bare binary would not scale, and under a ratio
    # objective the weights would then depend on `ohf`.
    if ffl_flag
        ffl = model[mip_key(sp, :ffl)] = JuMP.@expression(model, dot_scalar(ffl, lg))
        add_to_one_time_fees!(model, ffl)
    end
    if ffs_flag
        ffs = model[mip_key(sp, :ffs)] = JuMP.@expression(model, dot_scalar(ffs, sg))
        add_to_one_time_fees!(model, ffs)
    end
    return nothing
end
