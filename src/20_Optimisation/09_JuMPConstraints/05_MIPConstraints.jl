"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the big-M style scaling constant used by MIP constraints.

When `ss` is a number it is returned directly. When `nothing`, the value is derived from the largest finite bound magnitude in `wb` multiplied by 1000, defaulting to `1000.0` when all bounds are infinite or absent.

# Arguments

  - `ss::Number`: Scaling constant. Returned as-is.

# Returns

  - `ss::Number`: The scaling constant.

# Related

  - [`mip_constraints`](@ref)
  - [`smip_constraints`](@ref)
  - [`WeightBounds`](@ref)
"""
function get_mip_ss(ss::Number, args...)
    return ss
end
function get_mip_ss(::Nothing, wb::WeightBounds)
    lb = wb.lb
    ub = wb.ub
    lb_mag = if isnothing(lb)
        0.0
    else
        idx = isfinite.(lb)
        lbv = view(lb, idx)
        isempty(lbv) ? 0.0 : maximum(abs.(lbv))
    end
    ub_mag = if isnothing(ub)
        0.0
    else
        idx = isfinite.(ub)
        ubv = view(ub, idx)
        isempty(ubv) ? 0.0 : maximum(abs.(ubv))
    end
    return (iszero(lb_mag) && iszero(ub_mag)) ? 1000.0 : max(lb_mag, ub_mag) * 1000.0
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add MIP-compatible weight bound constraints using binary selection variables.

The fall-through method does nothing when `wb` is `nothing`. The concrete method adds `w ≥ is ⊙ lb` and `w ≤ il ⊙ ub` constraints when the respective finite bounds are present.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{i}_s \\odot \\boldsymbol{\\ell} \\leq \\boldsymbol{w} \\leq \\boldsymbol{i}_l \\odot \\boldsymbol{u}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - ``\\boldsymbol{i}_l``, ``\\boldsymbol{i}_s``: Long and short binary indicator vectors.
  - ``\\boldsymbol{\\ell}``, ``\\boldsymbol{u}``: Lower and upper bound vectors from `wb`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - $(arg_dict[:il_arg])
  - $(arg_dict[:is_arg])

# Returns

  - `nothing`.

# Related

  - [`mip_constraints`](@ref)
  - [`smip_wb`](@ref)
  - [`w_finite_flag`](@ref)
  - [`WeightBounds`](@ref)
"""
function mip_wb(::Any, ::Nothing, args...)
    return nothing
end
function mip_wb(model::JuMP.Model, wb::WeightBounds, il::VecNum, is::VecNum)
    sc = get_constraint_scale(model)
    w = get_w(model)
    lb = wb.lb
    if !isnothing(lb) && w_finite_flag(lb)
        JuMP.@constraint(model, w_mip_lb, sc * (w - is ⊙ lb) >= 0)
    end
    ub = wb.ub
    if !isnothing(ub) && w_finite_flag(ub)
        JuMP.@constraint(model, w_mip_ub, sc * (w - il ⊙ ub) <= 0)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add MIP binary selection variables and threshold constraints for long-short portfolios.

Creates `ilb`/`isb` binary indicator variables (or their continuous relaxations `ilf`/`isf` when `k` is a JuMP variable), enforces that each asset is either long or short but not both (`ilb + isb ≤ 1`), and applies long/short minimum-holding threshold and rebalancing constraints based on the flags provided.

# Mathematical definition

Mutual-exclusivity and big-M linearisation:

```math
\\begin{align}
ilb_i + isb_i \\leq 1, \\quad ilf_i &= ilb_i \\cdot k, \\quad isf_i = isb_i \\cdot k\\,.
\\end{align}
```

Where:

  - ``ilb_i``, ``isb_i``: Long and short binary indicator variables for asset ``i``.
  - ``ilf_i``, ``isf_i``: Continuous relaxations of ``ilb_i \\cdot k`` and ``isb_i \\cdot k``.
  - $(math_dict[:k_budget])

Minimum-holding thresholds:

```math
\\begin{align}
w_i &\\geq ilf_i\\, \\ell_i - M(1 - ilb_i)\\,, \\\\
w_i &\\leq -isf_i\\, u_i + M(1 - isb_i)\\,.
\\end{align}
```

Where:

  - ``w_i``: Portfolio weight for asset ``i``.
  - ``ilf_i``, ``isf_i``: Long and short continuous indicator expressions.
  - ``ilb_i``, ``isb_i``: Long and short binary indicator variables for asset ``i``.
  - ``\\ell_i``, ``u_i``: Long and short minimum-holding thresholds for asset ``i``.
  - ``M``: Big-M constant.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - $(arg_dict[:lt_arg])
  - $(arg_dict[:st_arg])
  - `ffl::Option{<:Num_VecNum}`: Long-side fixed fee rate(s).
  - `ffs::Option{<:Num_VecNum}`: Short-side fixed fee rate(s).
  - $(arg_dict[:ss_arg])
  - $(arg_dict[:lt_flag_arg])
  - $(arg_dict[:st_flag_arg])
  - `ffl_flag::Bool`: Whether to add long fixed fee expressions.
  - `ffs_flag::Bool`: Whether to add short fixed fee expressions.
  - $(arg_dict[:miprb_flag_arg])

# Returns

  - `i_mip`: Combined long+short indicator expression.

# Related

  - [`mip_constraints`](@ref)
  - [`get_mip_ss`](@ref)
  - [`mip_wb`](@ref)
  - [`Threshold`](@ref)
  - [`WeightBounds`](@ref)
"""
function short_mip_threshold_constraints(model::JuMP.Model, wb::WeightBounds,
                                         lt::Option{<:Threshold}, st::Option{<:Threshold},
                                         ffl::Option{<:Num_VecNum},
                                         ffs::Option{<:Num_VecNum}, ss::Option{<:Number},
                                         lt_flag::Bool, st_flag::Bool, ffl_flag::Bool,
                                         ffs_flag::Bool, miprb_flag::Bool)
    w = get_w(model)
    k = get_k(model)
    sc = get_constraint_scale(model)
    JuMP.@expression(model, ss, get_mip_ss(ss, wb))
    N = length(w)
    JuMP.@variables(model, begin
                        ilb[1:N], (binary = true)
                        isb[1:N], (binary = true)
                    end)
    JuMP.@expression(model, i_mip, ilb + isb)
    if isa(k, Number) || haskey(model, :crkb)
        JuMP.@expressions(model, begin
                              il, ilb
                              is, isb
                          end)
    else
        JuMP.@variables(model, begin
                            ilf[1:N] >= 0
                            isf[1:N] >= 0
                        end)
        JuMP.@constraints(model, begin
                              ilf_ub, sc * (ilf .- k) <= 0
                              isf_ub, sc * (isf .- k) <= 0
                              ilfd_ub, sc * (ilf - ss * ilb) <= 0
                              isfd_ub, sc * (isf - ss * isb) <= 0
                              ilfd_lb, sc * ((ilf + ss * (1 .- ilb)) .- k) >= 0
                              isfd_lb, sc * ((isf + ss * (1 .- isb)) .- k) >= 0
                          end)
        JuMP.@expressions(model, begin
                              il, ilf
                              is, isf
                          end)
    end
    JuMP.@constraint(model, i_mip_ub, sc * (i_mip .- 1) <= 0)
    mip_wb(model, wb, il, is)
    if lt_flag
        JuMP.@constraint(model, w_mip_lt, sc * (w - il ⊙ lt.val + ss * (1 .- ilb)) >= 0)
    end
    if st_flag
        JuMP.@constraint(model, w_mip_st, sc * (w + is ⊙ st.val - ss * (1 .- isb)) <= 0)
    end
    if ffl_flag
        JuMP.@expression(model, ffl, dot_scalar(ffl, ilb))
        add_to_fees!(model, ffl)
    end
    if ffs_flag
        JuMP.@expression(model, ffs, dot_scalar(ffs, isb))
        add_to_fees!(model, ffs)
    end
    if miprb_flag
        lw = model[:lw]
        sw = model[:sw]
        JuMP.@constraints(model, begin
                              lmiprb, sc * (lw - ss * il) <= 0
                              smiprb, sc * (sw - ss * is) <= 0
                          end)
    end
    return i_mip
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add long-only MIP binary indicator variable and associated constraints to the JuMP optimisation model.

Creates binary variable `ib[i]` per asset indicating whether the asset is held. When `k` is a JuMP variable, introduces continuous relaxation `ibf` with big-M linking constraints. Optionally applies minimum-holding threshold, fixed-fee, and rebalancing constraints.

# Mathematical definition

Big-M linearisation of ``ibf_i = ib_i \\cdot k``:

```math
\\begin{align}
ibf_i &\\leq k, \\quad ibf_i \\leq M\\, ib_i, \\quad ibf_i + M(1 - ib_i) \\geq k\\,.
\\end{align}
```

Where:

  - ``ib_i``: Binary inclusion indicator for asset ``i``.
  - ``ibf_i``: Continuous relaxation of ``ib_i \\cdot k``.
  - $(math_dict[:k_budget])
  - ``M``: Big-M constant.

Minimum-holding threshold and cardinality:

```math
\\begin{align}
w_i &\\geq ibf_i\\, \\ell_i\\,, \\\\
\\sum_i ib_i &\\leq \\mathrm{card}\\,.
\\end{align}
```

Where:

  - ``w_i``: Portfolio weight for asset ``i``.
  - ``ibf_i``: Continuous relaxation of ``ib_i \\cdot k``.
  - ``\\ell_i``: Minimum-holding threshold for asset ``i``.
  - ``\\mathrm{card}``: Maximum number of non-zero assets (cardinality bound).

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `ffl::Option{<:Num_VecNum}`: Long-side fixed fee rate(s).
  - $(arg_dict[:lt_arg])
  - $(arg_dict[:ss_arg])
  - $(arg_dict[:lt_flag_arg])
  - `ffl_flag::Bool`: Whether to add fixed fee expressions.
  - $(arg_dict[:miprb_flag_arg])

# Returns

  - `ib`: Binary indicator JuMP variable vector.

# Related

  - [`short_mip_threshold_constraints`](@ref)
  - [`set_mip_constraints!`](@ref)
  - [`mip_wb`](@ref)
  - [`get_mip_ss`](@ref)
  - [`set_iplg_constraints!`](@ref)
  - [`Threshold`](@ref)
  - [`WeightBounds`](@ref)
"""
function mip_constraints(model::JuMP.Model, wb::WeightBounds, ffl::Option{<:Num_VecNum},
                         lt::Option{<:Threshold}, ss::Option{<:Number}, lt_flag::Bool,
                         ffl_flag::Bool, miprb_flag::Bool)
    w = get_w(model)
    k = get_k(model)
    sc = get_constraint_scale(model)
    N = length(w)
    JuMP.@variable(model, ib[1:N], binary = true)
    if isa(k, Number)
        JuMP.@expression(model, i_mip, ib)
    else
        JuMP.@expression(model, ss, get_mip_ss(ss, wb))
        JuMP.@variable(model, ibf[1:N] >= 0)
        JuMP.@constraints(model, begin
                              ibf_ub, sc * (ibf .- k) <= 0
                              ibfd_ub, sc * (ibf - ss * ib) <= 0
                              ibfd_lb, sc * ((ibf + ss * (1 .- ib)) .- k) >= 0
                          end)
        JuMP.@expression(model, i_mip, ibf)
    end
    mip_wb(model, wb, i_mip, i_mip)
    if lt_flag
        JuMP.@constraint(model, w_mip_lt, sc * (w - i_mip ⊙ lt.val) >= 0)
    end
    if ffl_flag
        JuMP.@expression(model, ffl, dot_scalar(ffl, ib))
        add_to_fees!(model, ffl)
    end
    if miprb_flag
        lw = model[:lw]
        sw = model[:sw]
        JuMP.@constraints(model, begin
                              lmiprb, sc * (lw - ss * ib) <= 0
                              smiprb, sc * (sw - ss * (1 .- ib)) <= 0
                          end)
    end
    return ib
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add integer phylogeny cardinality constraints to the JuMP optimisation model.

Iterates over `plgs` and, for each [`IntegerPhylogeny`](@ref) entry, enforces `A * ib ≤ B` where `ib` is the binary indicator variable created by [`mip_constraints`](@ref).

# Arguments

  - $(arg_dict[:model])
  - `plgs`: Collection of phylogeny constraint objects.

# Returns

  - `nothing`.

# Related

  - [`mip_constraints`](@ref)
  - [`set_mip_constraints!`](@ref)
  - [`IntegerPhylogeny`](@ref)
"""
function set_iplg_constraints!(model::JuMP.Model, plgs::PlC_VecPlC)
    ib = model[:ib]
    sc = get_constraint_scale(model)
    for (i, pl) in enumerate(plgs)
        if !isa(pl, IntegerPhylogeny)
            continue
        end
        A = pl.A
        B = pl.B
        model[Symbol(:card_plg_, i)] = JuMP.@constraint(model, sc * (A * ib ⊖ B) <= 0)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add all long-only MIP constraints to the JuMP optimisation model.

Orchestrates cardinality, group cardinality, integer phylogeny, minimum-holding threshold, fixed-fee, and rebalancing MIP constraints for long-only portfolios, dispatching to [`mip_constraints`](@ref), [`short_mip_threshold_constraints`](@ref), and [`set_iplg_constraints!`](@ref) as appropriate.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `card::Option{<:Integer}`: Optional maximum cardinality (number of non-zero assets).
  - `gcard::Option{<:LinearConstraint}`: Optional group cardinality constraint.
  - `pl::Option{<:PlC_VecPlC}`: Optional phylogeny constraint(s).
  - $(arg_dict[:lt_arg])
  - $(arg_dict[:st_arg])
  - `fees::Option{<:Fees}`: Optional fee specification.
  - $(arg_dict[:ss_arg])
  - `miprb_flag::Bool = false`: Whether to add MIP rebalancing constraints.

# Returns

  - `nothing`.

# Related

  - [`mip_constraints`](@ref)
  - [`short_mip_threshold_constraints`](@ref)
  - [`set_iplg_constraints!`](@ref)
  - [`set_smip_constraints!`](@ref)
  - [`WeightBounds`](@ref)
  - [`Threshold`](@ref)
  - [`IntegerPhylogeny`](@ref)
"""
function set_mip_constraints!(model::JuMP.Model, wb::WeightBounds, card::Option{<:Integer},
                              gcard::Option{<:LinearConstraint}, pl::Option{<:PlC_VecPlC},
                              lt::Option{<:Threshold}, st::Option{<:Threshold},
                              fees::Option{<:Fees}, ss::Option{<:Number},
                              miprb_flag::Bool = false)
    card_flag = !isnothing(card)
    gcard_flag = !isnothing(gcard)
    iplg_flag = isa(pl, IntegerPhylogeny) ||
                isa(pl, AbstractVector) && any(x -> isa(x, IntegerPhylogeny), pl)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    ffl_flag, ffs_flag, ffl, ffs = if !isnothing(fees)
        non_zero_real_or_vec(fees.fl), non_zero_real_or_vec(fees.fs), fees.fl, fees.fs
    else
        false, false, nothing, nothing
    end
    if !(card_flag ||
         gcard_flag ||
         iplg_flag ||
         lt_flag ||
         st_flag ||
         ffl_flag ||
         ffs_flag ||
         miprb_flag)
        return nothing
    end
    ib = if (st_flag || ffl_flag || ffs_flag) && haskey(model, :sw)
        short_mip_threshold_constraints(model, wb, lt, st, ffl, ffs, ss, lt_flag, st_flag,
                                        ffl_flag, ffs_flag, miprb_flag)
    else
        mip_constraints(model, wb, ffl, lt, ss, lt_flag, ffl_flag, miprb_flag)
    end
    sc = get_constraint_scale(model)
    if card_flag
        JuMP.@constraint(model, card, sc * (sum(ib) - card) <= 0)
    end
    if gcard_flag
        if !isnothing(gcard.ineq)
            A = gcard.ineq.A
            B = gcard.ineq.B
            JuMP.@constraint(model, gcard_ineq, sc * (A * ib ⊖ B) <= 0)
        end
        if !isnothing(gcard.eq)
            A = gcard.eq.A
            B = gcard.eq.B
            JuMP.@constraint(model, gcard_eq, sc * (A * ib ⊖ B) == 0)
        end
    end
    if iplg_flag
        set_iplg_constraints!(model, pl)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add sub-group MIP weight bound constraints using a selection matrix.

The fall-through method does nothing when `wb` is `nothing`. The concrete method applies bounds `smtx * lb` and `smtx * ub` gated by binary selection variables `is` and `il` respectively.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `smtx::MatNum`: Selection matrix mapping assets to sub-groups.
  - `smtx_expr`: JuMP expression for the sub-group weight combination.
  - $(arg_dict[:il_arg])
  - $(arg_dict[:is_arg])
  - `key::Symbol = :set_w_mip_`: Base key for naming constraints.
  - `i::Integer = 1`: Index for generating unique constraint names.

# Returns

  - `nothing`.

# Related

  - [`mip_wb`](@ref)
  - [`smip_constraints`](@ref)
  - [`WeightBounds`](@ref)
"""
function smip_wb(::Any, ::Nothing, args...)
    return nothing
end
function smip_wb(model::JuMP.Model, wb::WeightBounds, smtx::MatNum,
                 smtx_expr::VecJuMPScalar, il::VecNum, is::VecNum,
                 key::Symbol = :set_w_mip_, i::Integer = 1)
    sc = get_constraint_scale(model)
    lb = wb.lb
    if !isnothing(lb) && w_finite_flag(lb)
        lb = smtx * lb
        model[Symbol(key, :lb_, i)] = JuMP.@constraint(model,
                                                       sc * (smtx_expr - lb ⊙ is) >= 0)
    end
    ub = wb.ub
    if !isnothing(ub) && w_finite_flag(ub)
        ub = smtx * ub
        model[Symbol(key, :ub_, i)] = JuMP.@constraint(model,
                                                       sc * (smtx_expr - ub ⊙ il) <= 0)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add sub-group MIP binary selection variables and threshold constraints for long-short portfolios using a selection matrix.

Creates per-group binary indicator variables and, when `k` is a JuMP variable, their continuous relaxations with big-M linking constraints. Applies long/short minimum-holding thresholds gated by the selection matrix `smtx`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - $(arg_dict[:smtx_arg])
  - $(arg_dict[:lt_arg])
  - $(arg_dict[:st_arg])
  - $(arg_dict[:ss_arg])
  - $(arg_dict[:lt_flag_arg])
  - $(arg_dict[:st_flag_arg])
  - `key1::Symbol = :si`: Base key for long indicator variables.
  - `key7::Symbol = :smtx_expr_`: Base key for sub-group weight expressions.
  - `key8::Symbol = :set_w_mip_`: Base key for weight bound constraints.
  - `i::Integer = 1`: Index for generating unique names.

# Returns

  - `i_mip`: Combined long+short indicator expression for the sub-group.

# Related

  - [`short_mip_threshold_constraints`](@ref)
  - [`smip_constraints`](@ref)
  - [`smip_wb`](@ref)
  - [`get_mip_ss`](@ref)
  - [`Threshold`](@ref)
  - [`WeightBounds`](@ref)
"""
function short_smip_threshold_constraints(model::JuMP.Model, wb::WeightBounds,
                                          smtx::Option{<:MatNum}, lt::Option{<:Threshold},
                                          st::Option{<:Threshold}, ss::Option{<:Number},
                                          lt_flag::Bool, st_flag::Bool, key1::Symbol = :si,
                                          key7::Symbol = :smtx_expr_,
                                          key8::Symbol = :set_w_mip_, i::Integer = 1)
    w = get_w(model)
    k = get_k(model)
    sc = get_constraint_scale(model)
    JuMP.@expression(model, ss, get_mip_ss(ss, wb))
    N = size(smtx, 1)
    ilb, isb = model[Symbol(key1, :lb_, i)], model[Symbol(key1, :ub_, i)] = JuMP.@variables(model,
                                                                                            begin
                                                                                                [1:N],
                                                                                                (binary = true)
                                                                                                [1:N],
                                                                                                (binary = true)
                                                                                            end)
    key2 = Symbol(key1, :_mip_)
    i_mip = model[Symbol(key2, i)] = JuMP.@expression(model, ilb + isb)
    il, is = if isa(k, Number)
        model[Symbol(key1, :l_, i)], model[Symbol(key1, :s_, i)] = JuMP.@expressions(model,
                                                                                     begin
                                                                                         ilb
                                                                                         isb
                                                                                     end)
    else
        key3 = Symbol(key1, :lf_)
        key4 = Symbol(key1, :sf_)
        ilf, isf = model[Symbol(key3, i)], model[Symbol(key4, i)] = JuMP.@variables(model,
                                                                                    begin
                                                                                        [1:N],
                                                                                        (lower_bound = 0)
                                                                                        [1:N],
                                                                                        (lower_bound = 0)
                                                                                    end)
        key5 = Symbol(key1, :lfd_)
        key6 = Symbol(key1, :sfd_)
        model[Symbol(key3, :ub_, i)], model[Symbol(key4, :ub_, i)], model[Symbol(key5, :ub_, i)], model[Symbol(key6, :ub_, i)], model[Symbol(key5, :lb_, i)], model[Symbol(key6, :lb_, i)] = JuMP.@constraints(model,
                                                                                                                                                                                                               begin
                                                                                                                                                                                                                   sc *
                                                                                                                                                                                                                   (ilf .-
                                                                                                                                                                                                                    k) <=
                                                                                                                                                                                                                   0
                                                                                                                                                                                                                   sc *
                                                                                                                                                                                                                   (isf .-
                                                                                                                                                                                                                    k) <=
                                                                                                                                                                                                                   0
                                                                                                                                                                                                                   sc *
                                                                                                                                                                                                                   (ilf -
                                                                                                                                                                                                                    ss *
                                                                                                                                                                                                                    ilb) <=
                                                                                                                                                                                                                   0
                                                                                                                                                                                                                   sc *
                                                                                                                                                                                                                   (isf -
                                                                                                                                                                                                                    ss *
                                                                                                                                                                                                                    isb) <=
                                                                                                                                                                                                                   0
                                                                                                                                                                                                                   sc *
                                                                                                                                                                                                                   ((ilf +
                                                                                                                                                                                                                     ss *
                                                                                                                                                                                                                     (1 .-
                                                                                                                                                                                                                      ilb)) .-
                                                                                                                                                                                                                    k) >=
                                                                                                                                                                                                                   0
                                                                                                                                                                                                                   sc *
                                                                                                                                                                                                                   ((isf +
                                                                                                                                                                                                                     ss *
                                                                                                                                                                                                                     (1 .-
                                                                                                                                                                                                                      isb)) .-
                                                                                                                                                                                                                    k) >=
                                                                                                                                                                                                                   0
                                                                                                                                                                                                               end)
        model[Symbol(key1, :l_, i)], model[Symbol(key1, :s_, i)] = JuMP.@expressions(model,
                                                                                     begin
                                                                                         ilf
                                                                                         isf
                                                                                     end)
    end
    model[Symbol(key2, :ub, i)] = JuMP.@constraint(model, sc * (i_mip .- 1) <= 0)
    smtx_expr = model[Symbol(key7, i)] = JuMP.@expression(model, smtx * w)
    smip_wb(model, wb, smtx, smtx_expr, il, is, key8, i)
    if lt_flag
        model[Symbol(key8, :lt, i)] = JuMP.@constraint(model,
                                                       sc * (smtx_expr - il ⊙ lt.val +
                                                             ss * (1 .- ilb)) >= 0)
    end
    if st_flag
        model[Symbol(key8, :st, i)] = JuMP.@constraint(model,
                                                       sc * (smtx_expr + is ⊙ st.val -
                                                             ss * (1 .- isb)) <= 0)
    end
    return i_mip
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add a sub-group MIP binary indicator variable and associated weight and threshold constraints for a single selection matrix group.

Creates binary variable `sib` (or continuous relaxation when `k` is a JuMP variable) representing asset inclusion within the sub-group, then applies sub-group weight expression and optional threshold bounds via [`smip_wb`](@ref).

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `smtx::Option{<:MatNum}`: Selection matrix for this sub-group.
  - $(arg_dict[:lt_arg])
  - $(arg_dict[:ss_arg])
  - $(arg_dict[:lt_flag_arg])
  - `key1::Symbol = :sib_`: Base key for binary indicator variable.
  - `key2::Symbol = :i_smip_`: Base key for the indicator expression.
  - `key3::Symbol = :isbf_`: Base key for continuous relaxation variable.
  - `key4::Symbol = :smtx_expr_`: Base key for sub-group weight expression.
  - `key5::Symbol = :set_w_mip_`: Base key for weight bound constraints.
  - `key6::Symbol = :w_smip_lt_`: Base key for threshold constraints.
  - `i::Integer = 1`: Index for generating unique names.

# Returns

  - `sib`: Binary indicator JuMP variable for this sub-group.

# Related

  - [`smip_wb`](@ref)
  - [`set_all_smip_constraints!`](@ref)
  - [`set_scardmip_constraints!`](@ref)
  - [`get_mip_ss`](@ref)
  - [`WeightBounds`](@ref)
"""
function smip_constraints(model::JuMP.Model, wb::WeightBounds, smtx::Option{<:MatNum},
                          lt::Option{<:Threshold}, ss::Option{<:Number}, lt_flag::Bool,
                          key1::Symbol = :sib_, key2::Symbol = :i_smip_,
                          key3::Symbol = :isbf_, key4::Symbol = :smtx_expr_,
                          key5::Symbol = :set_w_mip_, key6::Symbol = :w_smip_lt_,
                          i::Integer = 1)
    w = get_w(model)
    k = get_k(model)
    sc = get_constraint_scale(model)
    N = size(smtx, 1)
    sib = model[Symbol(key1, i)] = JuMP.@variable(model, [1:N], binary = true)
    i_smip = if isa(k, Number)
        model[Symbol(key2, i)] = JuMP.@expression(model, sib)
    else
        JuMP.@expression(model, ss, get_mip_ss(ss, wb))
        isbf = model[Symbol(key3, i)] = JuMP.@variable(model, [1:N], lower_bound = 0)
        model[Symbol(key3, :_ub_, i)], model[Symbol(key3, :d_ub_, i)], model[Symbol(key3, :d_lb_, i)] = JuMP.@constraints(model,
                                                                                                                          begin
                                                                                                                              sc *
                                                                                                                              (isbf .-
                                                                                                                               k) <=
                                                                                                                              0
                                                                                                                              sc *
                                                                                                                              (isbf -
                                                                                                                               ss *
                                                                                                                               sib) <=
                                                                                                                              0
                                                                                                                              sc *
                                                                                                                              ((isbf +
                                                                                                                                ss *
                                                                                                                                (1 .-
                                                                                                                                 sib)) .-
                                                                                                                               k) >=
                                                                                                                              0
                                                                                                                          end)
        model[Symbol(key2, i)] = JuMP.@expression(model, isbf)
    end
    smtx_expr = model[Symbol(key4, i)] = JuMP.@expression(model, smtx * w)
    smip_wb(model, wb, smtx, smtx_expr, i_smip, i_smip, key5, i)
    if lt_flag
        model[Symbol(key6, i)] = JuMP.@constraint(model,
                                                  sc * (smtx_expr - i_smip ⊙ lt.val) >= 0)
    end
    return sib
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add all sub-group MIP constraints for a single or multiple selection matrices.

The single-matrix method handles cardinality, group cardinality, long/short threshold, and weight-bound constraints for one sub-group. The vector method iterates over collections of cardinalities, group constraints, and selection matrices.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `card`: Cardinality bound(s) for the sub-group(s).
  - `gcard`: Group-cardinality constraint(s) for the sub-group(s).
  - `smtx`: Selection matrix (or vector thereof) for the sub-group(s).
  - `lt`: Long-side minimum-holding threshold(s).
  - `st`: Short-side minimum-holding threshold(s).
  - $(arg_dict[:ss_arg])
  - `i::Integer = 1`: Index for generating unique names (single-matrix method only).

# Returns

  - `nothing`.

# Related

  - [`smip_constraints`](@ref)
  - [`short_smip_threshold_constraints`](@ref)
  - [`set_scardmip_constraints!`](@ref)
  - [`set_sgcardmip_constraints!`](@ref)
  - [`set_smip_constraints!`](@ref)
  - [`WeightBounds`](@ref)
"""
function set_all_smip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                   card::Option{<:Integer},
                                   gcard::Option{<:LinearConstraint},
                                   smtx::Option{<:MatNum}, lt::Option{<:Threshold},
                                   st::Option{<:Threshold}, ss::Option{<:Number},
                                   i::Integer = 1)
    card_flag = !isnothing(card)
    gcard_flag = !isnothing(gcard)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    if !(card_flag || gcard_flag || lt_flag || st_flag)
        return nothing
    end
    sc = get_constraint_scale(model)
    sib = if st_flag && haskey(model, :sw)
        short_smip_threshold_constraints(model, wb, smtx, lt, st, ss, lt_flag, st_flag, :si,
                                         :smtx_expr_, :set_w_mip_, i)
    else
        smip_constraints(model, wb, smtx, lt, ss, lt_flag, :sib_, :i_smip_, :isbf_,
                         :smtx_expr_, :set_w_mip_, :w_smip_lt_, i)
    end
    if card_flag
        model[Symbol(:scard_, i)] = JuMP.@constraint(model, sc * (sum(sib) - card) <= 0)
    end
    if gcard_flag
        if !isnothing(gcard.ineq)
            A = gcard.ineq.A
            B = gcard.ineq.B
            model[Symbol(:sgcard_ineq_, i)] = JuMP.@constraint(model,
                                                               sc * (A * sib ⊖ B) <= 0)
        end
        if !isnothing(gcard.eq)
            A = gcard.eq.A
            B = gcard.eq.B
            model[Symbol(:sgcard_eq_, i)] = JuMP.@constraint(model, sc * (A * sib ⊖ B) == 0)
        end
    end
    return nothing
end
function set_all_smip_constraints!(model::JuMP.Model, wb::WeightBounds, card::VecInt,
                                   gcard::VecLc, smtx::VecMatNum, lt::Option{<:Bt_VecOptBt},
                                   st::Option{<:Bt_VecOptBt}, ss::Option{<:Number})
    for (i, (c, g, s)) in enumerate(zip(card, gcard, smtx))
        lti = isa(lt, Option{<:Threshold}) ? lt : lt[i]
        sti = isa(st, Option{<:Threshold}) ? st : st[i]
        set_all_smip_constraints!(model, wb, c, g, s, lti, sti, ss, i)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add sub-group cardinality MIP constraints using a selection matrix.

The single-matrix method enforces `sum(sib) ≤ card` for one sub-group. The vector method iterates over collections of cardinalities and selection matrices.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `card`: Cardinality bound(s) for the sub-group(s).
  - `smtx`: Selection matrix (or vector thereof).
  - `lt`: Long-side minimum-holding threshold(s).
  - `st`: Short-side minimum-holding threshold(s).
  - $(arg_dict[:ss_arg])
  - `i::Integer = 1`: Index for generating unique names (single-matrix method only).

# Returns

  - `nothing`.

# Related

  - [`smip_constraints`](@ref)
  - [`short_smip_threshold_constraints`](@ref)
  - [`set_sgcardmip_constraints!`](@ref)
  - [`set_smip_constraints!`](@ref)
  - [`WeightBounds`](@ref)
"""
function set_scardmip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                   card::Option{<:Integer}, smtx::Option{<:MatNum},
                                   lt::Option{<:Threshold}, st::Option{<:Threshold},
                                   ss::Option{<:Number}, i::Integer = 1)
    card_flag = !isnothing(card)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    if !(card_flag || lt_flag || st_flag)
        return nothing
    end
    sc = get_constraint_scale(model)
    sib = if st_flag && haskey(model, :sw)
        short_smip_threshold_constraints(model, wb, smtx, lt, st, ss, lt_flag, st_flag, :si,
                                         :smtx_expr_, :set_w_mip_, i)
    else
        smip_constraints(model, wb, smtx, lt, ss, lt_flag, :sib_, :i_smip_, :isbf_,
                         :smtx_expr_, :set_w_mip_, :w_smip_lt_, i)
    end
    model[Symbol(:scard_, i)] = JuMP.@constraint(model, sc * (sum(sib) - card) <= 0)
    return nothing
end
function set_scardmip_constraints!(model::JuMP.Model, wb::WeightBounds, card::VecInt,
                                   smtx::VecMatNum, lt::Option{<:Bt_VecOptBt},
                                   st::Option{<:Bt_VecOptBt}, ss::Option{<:Number})
    for (i, (c, s)) in enumerate(zip(card, smtx))
        lti = isa(lt, Option{<:Threshold}) ? lt : lt[i]
        sti = isa(st, Option{<:Threshold}) ? st : st[i]
        set_scardmip_constraints!(model, wb, c, s, lti, sti, ss, i)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add sub-group group-cardinality MIP constraints using a selection matrix.

The single-matrix method enforces linear group cardinality constraints `A * sib ≤ B` and `A * sib = B` on the sub-group binary indicator. The vector method iterates over multiple group constraints and selection matrices.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `gcard`: Group-cardinality constraint(s).
  - `smtx`: Selection matrix (or vector thereof).
  - `lt`: Long-side minimum-holding threshold(s).
  - `st`: Short-side minimum-holding threshold(s).
  - $(arg_dict[:ss_arg])
  - `i::Integer = 1`: Index for generating unique names (single-matrix method only).

# Returns

  - `nothing`.

# Related

  - [`smip_constraints`](@ref)
  - [`short_smip_threshold_constraints`](@ref)
  - [`set_scardmip_constraints!`](@ref)
  - [`set_smip_constraints!`](@ref)
  - [`WeightBounds`](@ref)
"""
function set_sgcardmip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                    gcard::Option{<:LinearConstraint},
                                    smtx::Option{<:MatNum}, lt::Option{<:Threshold},
                                    st::Option{<:Threshold}, ss::Option{<:Number},
                                    i::Integer = 1)
    gcard_flag = !isnothing(gcard)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    if !(gcard_flag || lt_flag || st_flag)
        return nothing
    end
    sc = get_constraint_scale(model)
    sib = if st_flag && haskey(model, :sw)
        short_smip_threshold_constraints(model, wb, smtx, lt, st, ss, lt_flag, st_flag,
                                         :sgi, :sgmtx_expr_, :setg_w_mip_, i)
    else
        smip_constraints(model, wb, smtx, lt, ss, lt_flag, :sgib_, :i_sgmip_, :isgbf_,
                         :sgmtx_expr_, :setg_w_mip_, :w_sgmip_lt_, i)
    end
    if !isnothing(gcard.ineq)
        A = gcard.ineq.A
        B = gcard.ineq.B
        model[Symbol(:sgcard_ineq_, i)] = JuMP.@constraint(model, sc * (A * sib ⊖ B) <= 0)
    end
    if !isnothing(gcard.eq)
        A = gcard.eq.A
        B = gcard.eq.B
        model[Symbol(:sgcard_eq_, i)] = JuMP.@constraint(model, sc * (A * sib ⊖ B) == 0)
    end
    return nothing
end
function set_sgcardmip_constraints!(model::JuMP.Model, wb::WeightBounds, gcard::VecLc,
                                    smtx::VecMatNum, lt::Option{<:Bt_VecOptBt},
                                    st::Option{<:Bt_VecOptBt}, ss::Option{<:Number})
    for (i, (gc, s)) in enumerate(zip(gcard, smtx))
        lti = isa(lt, Option{<:Threshold}) ? lt : lt[i]
        sti = isa(st, Option{<:Threshold}) ? st : st[i]
        set_sgcardmip_constraints!(model, wb, gc, s, lti, sti, ss, i)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add all sub-group MIP constraints (both cardinality and group-cardinality) to the JuMP optimisation model.

Dispatches between combined selection matrices (calling [`set_all_smip_constraints!`](@ref)) and separate cardinality/group-cardinality selection matrices (calling [`set_scardmip_constraints!`](@ref) and [`set_sgcardmip_constraints!`](@ref) independently).

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `card`: Cardinality bound(s).
  - `gcard`: Group-cardinality constraint(s).
  - `smtx`: Cardinality selection matrix (or vector thereof).
  - `sgmtx`: Group-cardinality selection matrix (or vector thereof).
  - `lt`: Long-side minimum-holding threshold(s) for cardinality sub-groups.
  - `st`: Short-side minimum-holding threshold(s) for cardinality sub-groups.
  - `glt`: Long-side minimum-holding threshold(s) for group-cardinality sub-groups.
  - `gst`: Short-side minimum-holding threshold(s) for group-cardinality sub-groups.
  - $(arg_dict[:ss_arg])

# Returns

  - `nothing`.

# Related

  - [`set_all_smip_constraints!`](@ref)
  - [`set_scardmip_constraints!`](@ref)
  - [`set_sgcardmip_constraints!`](@ref)
  - [`set_mip_constraints!`](@ref)
  - [`WeightBounds`](@ref)
"""
function set_smip_constraints!(model::JuMP.Model, wb::WeightBounds,
                               card::Option{<:Int_VecInt}, gcard::Option{<:Lc_VecLc},
                               smtx::Option{<:MatNum_VecMatNum},
                               sgmtx::Option{<:MatNum_VecMatNum}, lt::Option{<:Bt_VecOptBt},
                               st::Option{<:Bt_VecOptBt}, glt::Option{<:Bt_VecOptBt},
                               gst::Option{<:Bt_VecOptBt}, ss::Option{<:Number})
    if smtx === sgmtx
        set_all_smip_constraints!(model, wb, card, gcard, smtx, lt, st, ss)
    else
        set_scardmip_constraints!(model, wb, card, smtx, lt, st, ss)
        set_sgcardmip_constraints!(model, wb, gcard, sgmtx, glt, gst, ss)
    end
    return nothing
end
