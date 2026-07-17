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
  - [`set_mip_ss_expr!`](@ref)
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

Register the big-M scaling constant as the model expression `:ss` and return it.

Asset-space and sub-group MIP builders share the single `:ss` key; the first caller registers it via [`get_mip_ss`](@ref), subsequent callers reuse the registered expression.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ss_arg])
  - $(arg_dict[:wb_arg])

# Returns

  - `ss`: The registered scaling-constant expression.

# Related

  - [`get_mip_ss`](@ref)
  - [`short_mip_threshold_constraints`](@ref)
  - [`mip_constraints`](@ref)
"""
function set_mip_ss_expr!(model::JuMP.Model, ss::Option{<:Number}, wb::WeightBounds)
    if haskey(model, :ss)
        return model[:ss]
    end
    return model[:ss] = JuMP.@expression(model, get_mip_ss(ss, wb))
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the weight space a MIP constraint builder acts on.

A MIP space tells the shared builders which expression the binary indicators gate ([`mip_wx!`](@ref)), how weight bounds map into that space ([`mip_bounds`](@ref)), and how model keys are named ([`mip_key`](@ref)).

# Related

  - [`AssetMIPSpace`](@ref)
  - [`SubsetMIPSpace`](@ref)
"""
abstract type AbstractMIPSpace end
"""
$(DocStringExtensions.TYPEDEF)

Asset-space MIP constraints: indicators gate the portfolio weights `w` directly and model keys use the bare names (`:ib`, `:i_mip`, `:w_mip_lt`, ...).

# Related

  - [`AbstractMIPSpace`](@ref)
  - [`SubsetMIPSpace`](@ref)
"""
struct AssetMIPSpace <: AbstractMIPSpace end
"""
$(DocStringExtensions.TYPEDEF)

Sub-group MIP constraints: indicators gate the sub-group weights `smtx * w` and weight bounds map through the selection matrix. Model keys are namespaced as `Symbol(pfx, name, :_, i)` so multiple sub-groups (and the cardinality/group-cardinality variants) do not collide.

# Fields

  - `smtx`: Selection matrix mapping assets to sub-groups.
  - `pfx::Symbol`: Key prefix; `:s` for cardinality sub-groups, `:sg` for group-cardinality sub-groups.
  - `i::Integer`: Index of the selection matrix, used in key naming.

# Related

  - [`AbstractMIPSpace`](@ref)
  - [`AssetMIPSpace`](@ref)
  - [`mip_key`](@ref)
"""
struct SubsetMIPSpace{T1 <: MatNum, T2 <: Integer} <: AbstractMIPSpace
    smtx::T1
    pfx::Symbol
    i::T2
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Map a logical MIP builder name to the model key for the given space.

Asset space returns the name unchanged; sub-group space returns `Symbol(pfx, name, :_, i)`.

# Related

  - [`AbstractMIPSpace`](@ref)
"""
function mip_key(::AssetMIPSpace, name::Symbol)
    return name
end
function mip_key(sp::SubsetMIPSpace, name::Symbol)
    return Symbol(sp.pfx, name, :_, sp.i, :_)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the weight expression the MIP indicators gate.

Asset space returns the portfolio weights `w`; sub-group space registers and returns the sub-group weight expression `smtx * w` under the space's `:mtx_expr` key.

# Related

  - [`AbstractMIPSpace`](@ref)
  - [`mip_bounds`](@ref)
"""
function mip_wx!(model::JuMP.Model, ::AssetMIPSpace)
    return get_w(model)
end
function mip_wx!(model::JuMP.Model, sp::SubsetMIPSpace)
    return model[mip_key(sp, :mtx_expr)] = JuMP.@expression(model, sp.smtx * get_w(model))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Map a weight bound vector into the MIP space.

Asset space returns the bounds unchanged; sub-group space maps them through the selection matrix, `smtx * b`.

# Related

  - [`AbstractMIPSpace`](@ref)
  - [`mip_wx!`](@ref)
"""
function mip_bounds(::AssetMIPSpace, b::VecNum)
    return b
end
function mip_bounds(sp::SubsetMIPSpace, b::VecNum)
    return sp.smtx * b
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Whether the binary indicators can gate the weights directly, without the continuous big-M relaxation of `indicator * k`.

True when the budget `k` is a constant. The asset space additionally accepts a unit budget ([`is_unit_budget`](@ref)): the head has normalised the scale, so the indicators may gate the weights directly even though `k` is still a free variable.

# Related

  - [`short_mip_threshold_constraints`](@ref)
"""
function use_direct_mip_indicators(model::JuMP.Model, ::AssetMIPSpace, k)
    return isa(k, Number) || is_unit_budget(model)
end
function use_direct_mip_indicators(::JuMP.Model, ::SubsetMIPSpace, k)
    return isa(k, Number)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add MIP-compatible weight bound constraints using binary selection variables.

The fall-through method does nothing when `wb` is `nothing`. The concrete method adds `wx ≥ is ⊙ lb` and `wx ≤ il ⊙ ub` constraints when the respective finite bounds are present, where `wx` and the bounds live in the space selected by `sp` ([`mip_wx!`](@ref), [`mip_bounds`](@ref)).

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{i}_s \\odot \\boldsymbol{\\ell} \\leq \\boldsymbol{w} \\leq \\boldsymbol{i}_l \\odot \\boldsymbol{u}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - ``\\boldsymbol{i}_l``, ``\\boldsymbol{i}_s``: Long and short binary indicator vectors.
  - ``\\boldsymbol{\\ell}``, ``\\boldsymbol{u}``: Lower and upper bound vectors from `wb`, mapped into the space of `sp`.

# Arguments

  - $(arg_dict[:model])
  - `sp::AbstractMIPSpace`: Weight space the constraints act on.
  - $(arg_dict[:wb_arg])
  - `wx::VecNum`: Weight expression from [`mip_wx!`](@ref).
  - $(arg_dict[:il_arg])
  - $(arg_dict[:is_arg])

# Returns

  - `nothing`.

# Related

  - [`mip_constraints`](@ref)
  - [`short_mip_threshold_constraints`](@ref)
  - [`w_finite_flag`](@ref)
  - [`WeightBounds`](@ref)
"""
function mip_wb(::JuMP.Model, ::AbstractMIPSpace, ::Nothing, args...)
    return nothing
end
function mip_wb(model::JuMP.Model, sp::AbstractMIPSpace, wb::WeightBounds, wx::VecNum,
                il::VecNum, is::VecNum)
    sc = get_constraint_scale(model)
    lb = wb.lb
    if !isnothing(lb) && w_finite_flag(lb)
        model[mip_key(sp, :w_mip_lb)] = JuMP.@constraint(model,
                                                         sc *
                                                         (wx - mip_bounds(sp, lb) ⊙ is) >=
                                                         0)
    end
    ub = wb.ub
    if !isnothing(ub) && w_finite_flag(ub)
        model[mip_key(sp, :w_mip_ub)] = JuMP.@constraint(model,
                                                         sc *
                                                         (wx - mip_bounds(sp, ub) ⊙ il) <=
                                                         0)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add MIP binary selection variables and threshold constraints for long-short portfolios.

Creates long/short binary indicator variables (or their continuous relaxations when `k` is a JuMP variable), enforces that each position is either long or short but not both, and applies long/short minimum-holding threshold, fixed-fee, and rebalancing constraints based on the flags provided. The weight expression, bound mapping, and model key naming are selected by `sp`, so the same builder serves asset-space and sub-group constraints.

# Mathematical definition

Mutual-exclusivity and big-M linearisation:

```math
\\begin{align}
ilb_i + isb_i \\leq 1, \\quad ilf_i &= ilb_i \\cdot k, \\quad isf_i = isb_i \\cdot k\\,.
\\end{align}
```

Where:

  - ``ilb_i``, ``isb_i``: Long and short binary indicator variables for position ``i``.
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

  - ``w_i``: Weight expression for position ``i`` in the space of `sp`.
  - ``ilf_i``, ``isf_i``: Long and short continuous indicator expressions.
  - ``ilb_i``, ``isb_i``: Long and short binary indicator variables for position ``i``.
  - ``\\ell_i``, ``u_i``: Long and short minimum-holding thresholds for position ``i``.
  - ``M``: Big-M constant.

# Arguments

  - $(arg_dict[:model])
  - `sp::AbstractMIPSpace`: Weight space the constraints act on.
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
  - `xbgt_flag::Bool = false`: Whether to pin the long/short decomposition exactly. See the section below.

# Pinning the decomposition (`xbgt_flag`)

`lw` and `sw` are *upper* bounds on the parts of `w` (`lw >= w`, `lw >= 0`, `sw >= -w`, `sw >= 0`), never equal to them, so every budget built on them (`bgt`, `sbgt`, `gbgt`) **bounds** the realised exposure rather than pinning it: `sbgt = 0.3` means at most 30% short. The slack is usually invisible — the objective normally pushes against the budget, making the bound tight — but it is not the same problem.

The complementarity pair emitted for `miprb_flag` (`lw <= ss * il`, `sw <= ss * is`) forces the long-xor-short *sign pattern*, but does not close that slack. `xbgt_flag` emits the pair regardless of `miprb_flag` and adds the two constraints that do:

```math
\\begin{align}
lw_i - w_i - M(1 - ilb_i) &\\leq 0\\,, \\\\
sw_i + w_i - M(1 - isb_i) &\\leq 0\\,.
\\end{align}
```

So `ilb_i = 1` gives `lw_i <= w_i`, which with `lw_i >= w_i` gives `lw_i == max(w_i, 0)`; `isb_i = 1` gives `sw_i == max(-w_i, 0)`; and both zero gives `lw_i == sw_i == 0`, hence `w_i == 0`. The budgets then hold exactly.

This reuses `ilb`/`isb` rather than introducing a sign indicator of its own — they already mean long and short here, and `i_mip = ilb + isb` remains the *held* indicator that cardinality counts. The constraints are keyed on the binaries rather than on `il`/`is`, which relax to continuous variables when `k` is free.

# Returns

  - `i_mip`: Combined long+short indicator expression.

# Related

  - [`mip_constraints`](@ref)
  - [`set_mip_ss_expr!`](@ref)
  - [`mip_wb`](@ref)
  - [`set_gross_budget_constraints!`](@ref)
  - [`JuMPOptimiser`](@ref)
  - [`Threshold`](@ref)
  - [`WeightBounds`](@ref)
"""
function short_mip_threshold_constraints(model::JuMP.Model, sp::AbstractMIPSpace,
                                         wb::WeightBounds, lt::Option{<:Threshold},
                                         st::Option{<:Threshold}, ffl::Option{<:Num_VecNum},
                                         ffs::Option{<:Num_VecNum}, ss::Option{<:Number},
                                         lt_flag::Bool, st_flag::Bool, ffl_flag::Bool,
                                         ffs_flag::Bool, miprb_flag::Bool,
                                         xbgt_flag::Bool = false)
    wx = mip_wx!(model, sp)
    k = get_k(model)
    sc = get_constraint_scale(model)
    ss = set_mip_ss_expr!(model, ss, wb)
    N = length(wx)
    ilb = model[mip_key(sp, :ilb)] = JuMP.@variable(model, [1:N], binary = true)
    isb = model[mip_key(sp, :isb)] = JuMP.@variable(model, [1:N], binary = true)
    i_mip = model[mip_key(sp, :i_mip)] = JuMP.@expression(model, ilb + isb)
    if use_direct_mip_indicators(model, sp, k)
        il = model[mip_key(sp, :il)] = JuMP.@expression(model, ilb)
        is = model[mip_key(sp, :is)] = JuMP.@expression(model, isb)
    else
        ilf = model[mip_key(sp, :ilf)] = JuMP.@variable(model, [1:N], lower_bound = 0)
        isf = model[mip_key(sp, :isf)] = JuMP.@variable(model, [1:N], lower_bound = 0)
        model[mip_key(sp, :ilf_ub)] = JuMP.@constraint(model, sc * (ilf .- k) <= 0)
        model[mip_key(sp, :isf_ub)] = JuMP.@constraint(model, sc * (isf .- k) <= 0)
        model[mip_key(sp, :ilfd_ub)] = JuMP.@constraint(model, sc * (ilf - ss * ilb) <= 0)
        model[mip_key(sp, :isfd_ub)] = JuMP.@constraint(model, sc * (isf - ss * isb) <= 0)
        model[mip_key(sp, :ilfd_lb)] = JuMP.@constraint(model,
                                                        sc *
                                                        ((ilf + ss * (1 .- ilb)) .- k) >= 0)
        model[mip_key(sp, :isfd_lb)] = JuMP.@constraint(model,
                                                        sc *
                                                        ((isf + ss * (1 .- isb)) .- k) >= 0)
        il = model[mip_key(sp, :il)] = JuMP.@expression(model, ilf)
        is = model[mip_key(sp, :is)] = JuMP.@expression(model, isf)
    end
    model[mip_key(sp, :i_mip_ub)] = JuMP.@constraint(model, sc * (i_mip .- 1) <= 0)
    mip_wb(model, sp, wb, wx, il, is)
    if lt_flag
        model[mip_key(sp, :w_mip_lt)] = JuMP.@constraint(model,
                                                         sc * (wx - il ⊙ lt.val +
                                                               ss * (1 .- ilb)) >= 0)
    end
    if st_flag
        model[mip_key(sp, :w_mip_st)] = JuMP.@constraint(model,
                                                         sc * (wx + is ⊙ st.val -
                                                               ss * (1 .- isb)) <= 0)
    end
    if ffl_flag
        ffl = model[mip_key(sp, :ffl)] = JuMP.@expression(model, dot_scalar(ffl, ilb))
        add_to_fees!(model, ffl)
    end
    if ffs_flag
        ffs = model[mip_key(sp, :ffs)] = JuMP.@expression(model, dot_scalar(ffs, isb))
        add_to_fees!(model, ffs)
    end
    if miprb_flag || xbgt_flag
        lw = model[:lw]
        sw = model[:sw]
        model[mip_key(sp, :lmiprb)] = JuMP.@constraint(model, sc * (lw - ss * il) <= 0)
        model[mip_key(sp, :smiprb)] = JuMP.@constraint(model, sc * (sw - ss * is) <= 0)
        if xbgt_flag
            # The pair above only forces lw = 0 on the short side and sw = 0 on the long
            # side. These close the remaining slack, so lw == max(wx, 0) and
            # sw == max(-wx, 0) rather than merely bounding them. Keyed on the binaries,
            # not on il/is, which relax to continuous variables when k is free.
            model[mip_key(sp, :xbgt_lw_eq)] = JuMP.@constraint(model,
                                                               sc *
                                                               (lw - wx - ss * (1 .- ilb)) <=
                                                               0)
            model[mip_key(sp, :xbgt_sw_eq)] = JuMP.@constraint(model,
                                                               sc *
                                                               (sw + wx - ss * (1 .- isb)) <=
                                                               0)
        end
    end
    return i_mip
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add a long-only MIP binary indicator variable and associated constraints to the JuMP optimisation model.

Creates a binary variable per position indicating whether it is held. When `k` is a JuMP variable, introduces a continuous relaxation with big-M linking constraints. Optionally applies minimum-holding threshold, fixed-fee, and rebalancing constraints. The weight expression, bound mapping, and model key naming are selected by `sp`, so the same builder serves asset-space and sub-group constraints.

# Mathematical definition

Big-M linearisation of ``ibf_i = ib_i \\cdot k``:

```math
\\begin{align}
ibf_i &\\leq k, \\quad ibf_i \\leq M\\, ib_i, \\quad ibf_i + M(1 - ib_i) \\geq k\\,.
\\end{align}
```

Where:

  - ``ib_i``: Binary inclusion indicator for position ``i``.
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

  - ``w_i``: Weight expression for position ``i`` in the space of `sp`.
  - ``ibf_i``: Continuous relaxation of ``ib_i \\cdot k``.
  - ``\\ell_i``: Minimum-holding threshold for position ``i``.
  - ``\\mathrm{card}``: Maximum number of non-zero positions (cardinality bound).

# Arguments

  - $(arg_dict[:model])
  - `sp::AbstractMIPSpace`: Weight space the constraints act on.
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
  - [`set_mip_ss_expr!`](@ref)
  - [`set_iplg_constraints!`](@ref)
  - [`Threshold`](@ref)
  - [`WeightBounds`](@ref)
"""
function mip_constraints(model::JuMP.Model, sp::AbstractMIPSpace, wb::WeightBounds,
                         ffl::Option{<:Num_VecNum}, lt::Option{<:Threshold},
                         ss::Option{<:Number}, lt_flag::Bool, ffl_flag::Bool,
                         miprb_flag::Bool)
    wx = mip_wx!(model, sp)
    k = get_k(model)
    sc = get_constraint_scale(model)
    N = length(wx)
    ib = model[mip_key(sp, :ib)] = JuMP.@variable(model, [1:N], binary = true)
    i_mip = if isa(k, Number)
        model[mip_key(sp, :i_mip)] = JuMP.@expression(model, ib)
    else
        ss = set_mip_ss_expr!(model, ss, wb)
        ibf = model[mip_key(sp, :ibf)] = JuMP.@variable(model, [1:N], lower_bound = 0)
        model[mip_key(sp, :ibf_ub)] = JuMP.@constraint(model, sc * (ibf .- k) <= 0)
        model[mip_key(sp, :ibfd_ub)] = JuMP.@constraint(model, sc * (ibf - ss * ib) <= 0)
        model[mip_key(sp, :ibfd_lb)] = JuMP.@constraint(model,
                                                        sc *
                                                        ((ibf + ss * (1 .- ib)) .- k) >= 0)
        model[mip_key(sp, :i_mip)] = JuMP.@expression(model, ibf)
    end
    mip_wb(model, sp, wb, wx, i_mip, i_mip)
    if lt_flag
        model[mip_key(sp, :w_mip_lt)] = JuMP.@constraint(model,
                                                         sc * (wx - i_mip ⊙ lt.val) >= 0)
    end
    if ffl_flag
        ffl = model[mip_key(sp, :ffl)] = JuMP.@expression(model, dot_scalar(ffl, ib))
        add_to_fees!(model, ffl)
    end
    if miprb_flag
        lw = model[:lw]
        sw = model[:sw]
        model[mip_key(sp, :lmiprb)] = JuMP.@constraint(model, sc * (lw - ss * ib) <= 0)
        model[mip_key(sp, :smiprb)] = JuMP.@constraint(model,
                                                       sc * (sw - ss * (1 .- ib)) <= 0)
    end
    return ib
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Pin the long/short decomposition using a single binary *sign* indicator per asset.

This is the lean form of the `xbgt_flag` branch of [`short_mip_threshold_constraints`](@ref), for when nothing else in the model needs binaries. It produces `N` of them rather than `2N`.

The long-short builder splits each asset three ways — long, short, or inactive — because a threshold has to tell `w_i = 0` apart from `w_i >= lt_i`, and cardinality has to count the assets that are held. Pinning the decomposition needs neither: the sign split is *total*, since `w_i = 0` satisfies both `w_i >= 0` and `w_i <= 0` and is picked up by either branch. One bit per asset therefore suffices, and the inactive state is pure redundancy.

# Mathematical definition

```math
\\begin{align}
lw_i - M\\, xb_i &\\leq 0\\,, & sw_i - M(1 - xb_i) &\\leq 0\\,, \\\\
lw_i - w_i - M(1 - xb_i) &\\leq 0\\,, & sw_i + w_i - M\\, xb_i &\\leq 0\\,.
\\end{align}
```

Where:

  - ``xb_i``: Binary sign indicator; `1` selects `w_i >= 0`, `0` selects `w_i <= 0`.
  - $(math_dict[:w_port])
  - ``lw_i``, ``sw_i``: Long and short parts of ``w_i``.
  - ``M``: Big-M constant.

The first pair forces `lw_i == 0` on the short side and `sw_i == 0` on the long side; the second closes the slack, giving `lw_i == max(w_i, 0)` and `sw_i == max(-w_i, 0)`.

# Arguments

  - $(arg_dict[:model])
  - `sp::AbstractMIPSpace`: Weight space the constraints act on.
  - $(arg_dict[:wb_arg])
  - $(arg_dict[:ss_arg])

# Returns

  - `nothing`: There is no held indicator to return — this builder runs only when nothing would consume one.

# Related

  - [`short_mip_threshold_constraints`](@ref)
  - [`set_mip_constraints!`](@ref)
  - [`set_mip_ss_expr!`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function sign_mip_constraints(model::JuMP.Model, sp::AbstractMIPSpace, wb::WeightBounds,
                              ss::Option{<:Number})
    wx = mip_wx!(model, sp)
    lw = model[:lw]
    sw = model[:sw]
    sc = get_constraint_scale(model)
    ss = set_mip_ss_expr!(model, ss, wb)
    N = length(wx)
    xb = model[mip_key(sp, :xbgt_ib)] = JuMP.@variable(model, [1:N], binary = true)
    model[mip_key(sp, :xbgt_lw_ub)] = JuMP.@constraint(model, sc * (lw - ss * xb) <= 0)
    model[mip_key(sp, :xbgt_sw_ub)] = JuMP.@constraint(model,
                                                       sc * (sw - ss * (1 .- xb)) <= 0)
    model[mip_key(sp, :xbgt_lw_eq)] = JuMP.@constraint(model,
                                                       sc * (lw - wx - ss * (1 .- xb)) <= 0)
    model[mip_key(sp, :xbgt_sw_eq)] = JuMP.@constraint(model, sc * (sw + wx - ss * xb) <= 0)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add integer phylogeny cardinality constraints to the JuMP optimisation model.

Iterates over `plgs` and, for each [`IntegerPhylogeny`](@ref) entry, enforces `A * ib ≤ B` where `ib` is the *held* indicator of the builder that ran — `ib` from [`mip_constraints`](@ref) or `i_mip` from [`short_mip_threshold_constraints`](@ref). It is passed in rather than read from the model, because only the long-only builder registers a `:ib` key.

# Arguments

  - $(arg_dict[:model])
  - `plgs`: Collection of phylogeny constraint objects.
  - `ib::VecNum`: Held indicator returned by the MIP builder that ran.

# Returns

  - `nothing`.

# Related

  - [`mip_constraints`](@ref)
  - [`set_mip_constraints!`](@ref)
  - [`IntegerPhylogeny`](@ref)
"""
function set_iplg_constraints!(model::JuMP.Model, plgs::PlC_VecPlC, ib::VecNum)
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
  - `xbgt::Bool = false`: Whether to pin the long/short decomposition so the budgets hold exactly. Needs a per-asset sign bit, which the long-only [`mip_constraints`](@ref) does not have — its `ib` marks an asset as *held* and is what cardinality counts. It is therefore served by [`sign_mip_constraints`](@ref) when nothing else in the model needs binaries (`N` of them), and otherwise routed to [`short_mip_threshold_constraints`](@ref), whose `ilb`/`isb` it reuses for free. Ignored when the model has no short side.

# Returns

  - `nothing`.

# Related

  - [`mip_constraints`](@ref)
  - [`short_mip_threshold_constraints`](@ref)
  - [`sign_mip_constraints`](@ref)
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
                              miprb_flag::Bool = false, xbgt::Bool = false)
    card_flag = !isnothing(card)
    gcard_flag = !isnothing(gcard)
    iplg_flag = isa(pl, IntegerPhylogeny) ||
                isa(pl, AbstractVector) && any(x -> isa(x, IntegerPhylogeny), pl)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    # Pinning the decomposition is meaningless without a short side to pin: with no `sw`
    # the weights are their own long part and the budgets are already exact.
    xbgt_flag = xbgt && haskey(model, :sw)
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
         miprb_flag ||
         xbgt_flag)
        return nothing
    end
    sp = AssetMIPSpace()
    # `xbgt` needs a per-asset *sign* bit, which only the long-short builder has: the
    # long-only builder's `ib` marks an asset as held and carries no sign.
    ib = if xbgt_flag && !(card_flag ||
                           gcard_flag ||
                           iplg_flag ||
                           lt_flag ||
                           st_flag ||
                           ffl_flag ||
                           ffs_flag ||
                           miprb_flag)
        # Nothing here consumes a held indicator, so the long-short builder's third state
        # (inactive) is dead weight: one sign bit per asset does the job with half the
        # binaries. Returns nothing -- there is no held indicator, and by this branch's
        # own condition nothing below would read one.
        sign_mip_constraints(model, sp, wb, ss)
    elseif (st_flag || ffl_flag || ffs_flag || xbgt_flag) && haskey(model, :sw)
        short_mip_threshold_constraints(model, sp, wb, lt, st, ffl, ffs, ss, lt_flag,
                                        st_flag, ffl_flag, ffs_flag, miprb_flag, xbgt_flag)
    else
        mip_constraints(model, sp, wb, ffl, lt, ss, lt_flag, ffl_flag, miprb_flag)
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
        set_iplg_constraints!(model, pl, ib)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add all sub-group MIP constraints for a single or multiple selection matrices.

The single-matrix method handles cardinality, group cardinality, long/short threshold, and weight-bound constraints for one sub-group via the shared builders ([`short_mip_threshold_constraints`](@ref), [`mip_constraints`](@ref)) on a [`SubsetMIPSpace`](@ref). The vector method iterates over collections of cardinalities, group constraints, and selection matrices.

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

  - [`mip_constraints`](@ref)
  - [`short_mip_threshold_constraints`](@ref)
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
    sp = SubsetMIPSpace(smtx, :s, i)
    sib = if st_flag && haskey(model, :sw)
        short_mip_threshold_constraints(model, sp, wb, lt, st, nothing, nothing, ss,
                                        lt_flag, st_flag, false, false, false)
    else
        mip_constraints(model, sp, wb, nothing, lt, ss, lt_flag, false, false)
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

  - [`mip_constraints`](@ref)
  - [`short_mip_threshold_constraints`](@ref)
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
    sp = SubsetMIPSpace(smtx, :s, i)
    sib = if st_flag && haskey(model, :sw)
        short_mip_threshold_constraints(model, sp, wb, lt, st, nothing, nothing, ss,
                                        lt_flag, st_flag, false, false, false)
    else
        mip_constraints(model, sp, wb, nothing, lt, ss, lt_flag, false, false)
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

  - [`mip_constraints`](@ref)
  - [`short_mip_threshold_constraints`](@ref)
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
    sp = SubsetMIPSpace(smtx, :sg, i)
    sib = if st_flag && haskey(model, :sw)
        short_mip_threshold_constraints(model, sp, wb, lt, st, nothing, nothing, ss,
                                        lt_flag, st_flag, false, false, false)
    else
        mip_constraints(model, sp, wb, nothing, lt, ss, lt_flag, false, false)
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
