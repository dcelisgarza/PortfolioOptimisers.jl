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
$(DocStringExtensions.TYPEDEF)

Abstract supertype for a declared set of MIP binary indicators.

A MIP space ([`AbstractMIPSpace`](@ref)) says *which weights* the indicators gate; an
indicator bundle says *which bits exist and what they mean*. The two are independent axes:
the declaration step picks the bundle from the features the model needs, and every emitter
is written once against the bundle's accessors rather than reading binaries back out of the
model by key.

The bits divide into two kinds, and conflating them is a bug — a *held* bit says an asset is
in the portfolio (what cardinality counts), a *sign* bit says which side it is on. Bundles
differ in which of the two they carry:

| bundle                        | binaries    | states                  |
|:----------------------------- |:----------- |:----------------------- |
| [`HeldIndicators`](@ref)      | `ib`        | held / not              |
| [`LongShortIndicators`](@ref) | `ilb`,`isb` | long / short / inactive |
| [`SignIndicators`](@ref)      | `xb`        | long / short            |

# Related

  - [`HeldIndicators`](@ref)
  - [`LongShortIndicators`](@ref)
  - [`SignIndicators`](@ref)
  - [`held`](@ref)
"""
abstract type AbstractMIPIndicators end
"""
$(DocStringExtensions.TYPEDEF)

Long-only indicators: one *held* bit per asset, and no sign.

# Fields

  - `ib`: Binary held bit.
  - `i_mip`: Held gate — `ib` itself, or its continuous relaxation `ibf` when the budget is free.

# Related

  - [`AbstractMIPIndicators`](@ref)
"""
struct HeldIndicators{T1, T2} <: AbstractMIPIndicators
    ib::T1
    i_mip::T2
end
"""
$(DocStringExtensions.TYPEDEF)

Long-short indicators: a long bit and a short bit per asset, mutually exclusive, so an asset
is long, short, or inactive. Carries *both* a sign (which of `ilb`/`isb`) and a held
indicator (`i_mip = ilb + isb`), which is why this bundle can serve cardinality and exact
budgets at once.

# Fields

  - `ilb`, `isb`: Binary long and short bits.
  - `il`, `is`: Long and short gates — the bits themselves, or their continuous relaxations `ilf`/`isf` when the budget is free.
  - `i_mip`: Held gate, `ilb + isb`.

# Related

  - [`AbstractMIPIndicators`](@ref)
"""
struct LongShortIndicators{T1, T2, T3, T4, T5} <: AbstractMIPIndicators
    ilb::T1
    isb::T2
    il::T3
    is::T4
    i_mip::T5
end
"""
$(DocStringExtensions.TYPEDEF)

Sign-only indicators: one *sign* bit per asset and no held indicator, for when nothing in the
model needs to know whether an asset is held.

The sign split is *total* — `w_i = 0` satisfies both `w_i >= 0` and `w_i <= 0` and is picked
up by either branch — so one bit per asset suffices and the inactive state of
[`LongShortIndicators`](@ref) is pure redundancy. This is what lets a model that only pins the
decomposition use `N` binaries rather than `2N`.

# Fields

  - `xb`: Binary sign bit; `1` selects `w_i >= 0`, `0` selects `w_i <= 0`.

# Related

  - [`AbstractMIPIndicators`](@ref)
"""
struct SignIndicators{T1} <: AbstractMIPIndicators
    xb::T1
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the *held* indicator — the per-asset quantity that is 1 when the asset is in the
portfolio. This is what cardinality counts and what integer phylogeny gates on.

[`SignIndicators`](@ref) deliberately has no method: it carries no held bit, and the
declaration step only chooses it when nothing in the model consumes one. A caller that would
need a held indicator therefore fails at the seam with a `MethodError`, rather than reading a
sign bit and silently miscounting.

# Related

  - [`AbstractMIPIndicators`](@ref)
"""
function held(ind::HeldIndicators)
    return ind.i_mip
end
function held(ind::LongShortIndicators)
    return ind.i_mip
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the gates gating the lower and upper weight bounds in [`mip_wb`](@ref).

The long-only bundle gates both bounds with the held indicator (`w = 0` unless held); the
long-short bundle gates the lower bound with the short gate and the upper with the long gate.

# Related

  - [`mip_wb`](@ref)
"""
function lb_gate(ind::HeldIndicators)
    return ind.i_mip
end
function ub_gate(ind::HeldIndicators)
    return ind.i_mip
end
function lb_gate(ind::LongShortIndicators)
    return ind.is
end
function ub_gate(ind::LongShortIndicators)
    return ind.il
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the gates and binaries the exact-decomposition constraints key on.

`long_gate`/`short_gate` bound `lw`/`sw` and may be continuous relaxations; `long_bin`/`short_bin`
are always the binaries themselves, because the slack-closing constraints must key on a bit
that is exactly 0 or 1 even when the budget is free and the gates relax.

For [`HeldIndicators`](@ref) the held bit doubles as the sign: this bundle is only asked for a
decomposition under a long-only weight bound, where held and long coincide.

# Related

  - [`AbstractMIPIndicators`](@ref)
"""
function long_gate(ind::HeldIndicators)
    return ind.ib
end
function short_gate(ind::HeldIndicators)
    return 1 .- ind.ib
end
function long_bin(ind::HeldIndicators)
    return ind.ib
end
function short_bin(ind::HeldIndicators)
    return 1 .- ind.ib
end
function long_gate(ind::LongShortIndicators)
    return ind.il
end
function short_gate(ind::LongShortIndicators)
    return ind.is
end
function long_bin(ind::LongShortIndicators)
    return ind.ilb
end
function short_bin(ind::LongShortIndicators)
    return ind.isb
end
function long_gate(ind::SignIndicators)
    return ind.xb
end
function short_gate(ind::SignIndicators)
    return 1 .- ind.xb
end
function long_bin(ind::SignIndicators)
    return ind.xb
end
function short_bin(ind::SignIndicators)
    return 1 .- ind.xb
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
function mip_wb(model::JuMP.Model, sp::AbstractMIPSpace, wb::WeightBounds, wx::VecNum,
                ind::AbstractMIPIndicators)
    return mip_wb(model, sp, wb, wx, ub_gate(ind), lb_gate(ind))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Declare the long-only *held* indicators and the constraints that give them meaning.

Creates one binary per asset and, when the budget `k` is a free variable, the continuous
relaxation `ibf = ib * k` with its big-M linking constraints. Applies [`mip_wb`](@ref), which
is what forces `w = 0` for an unheld asset and so makes the bit *mean* held.

Emits no feature constraints — thresholds, fees, cardinality, phylogeny, and budgets are the
business of their own emitters, which take the returned bundle.

# Arguments

  - $(arg_dict[:model])
  - `sp::AbstractMIPSpace`: Weight space the indicators act on.
  - $(arg_dict[:wb_arg])
  - `wx::VecNum`: Weight expression from [`mip_wx!`](@ref).
  - $(arg_dict[:ss_arg])

# Returns

  - `ind::HeldIndicators`: The declared indicator bundle.

# Related

  - [`HeldIndicators`](@ref)
  - [`declare_long_short_indicators!`](@ref)
  - [`declare_sign_indicators!`](@ref)
"""
function declare_held_indicators!(model::JuMP.Model, sp::AbstractMIPSpace,
                                  wb::Option{<:WeightBounds}, wx::VecNum,
                                  ss::Option{<:Number})
    k = get_k(model)
    sc = get_constraint_scale(model)
    N = length(wx)
    ib = model[mip_key(sp, :ib)] = JuMP.@variable(model, [1:N], binary = true)
    i_mip = if isa(k, Number)
        model[mip_key(sp, :i_mip)] = JuMP.@expression(model, ib)
    else
        ssx = set_mip_ss_expr!(model, ss, wb)
        ibf = model[mip_key(sp, :ibf)] = JuMP.@variable(model, [1:N], lower_bound = 0)
        model[mip_key(sp, :ibf_ub)] = JuMP.@constraint(model, sc * (ibf .- k) <= 0)
        model[mip_key(sp, :ibfd_ub)] = JuMP.@constraint(model, sc * (ibf - ssx * ib) <= 0)
        model[mip_key(sp, :ibfd_lb)] = JuMP.@constraint(model,
                                                        sc *
                                                        ((ibf + ssx * (1 .- ib)) .- k) >= 0)
        model[mip_key(sp, :i_mip)] = JuMP.@expression(model, ibf)
    end
    ind = HeldIndicators(ib, i_mip)
    mip_wb(model, sp, wb, wx, ind)
    return ind
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Declare the long-short indicators and the constraints that give them meaning.

Creates a long and a short binary per asset, forces them mutually exclusive (`ilb + isb <= 1`,
so an asset is long, short, or inactive), builds the continuous relaxations `ilf`/`isf` with
their big-M linking constraints when the budget is free, and applies [`mip_wb`](@ref).

Emits no feature constraints; see [`declare_held_indicators!`](@ref).

# Arguments

  - $(arg_dict[:model])
  - `sp::AbstractMIPSpace`: Weight space the indicators act on.
  - $(arg_dict[:wb_arg])
  - `wx::VecNum`: Weight expression from [`mip_wx!`](@ref).
  - $(arg_dict[:ss_arg])

# Returns

  - `ind::LongShortIndicators`: The declared indicator bundle.

# Related

  - [`LongShortIndicators`](@ref)
  - [`declare_held_indicators!`](@ref)
"""
function declare_long_short_indicators!(model::JuMP.Model, sp::AbstractMIPSpace,
                                        wb::Option{<:WeightBounds}, wx::VecNum,
                                        ss::Option{<:Number})
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
    ind = LongShortIndicators(ilb, isb, il, is, i_mip)
    mip_wb(model, sp, wb, wx, ind)
    return ind
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Declare a single *sign* binary per asset.

Unlike the held and long-short declarations there is no [`mip_wb`](@ref) here and no held
indicator: a sign bit needs no weight-bound gating to mean what it means, and this bundle is
only chosen when nothing in the model consumes a held indicator.

# Arguments

  - $(arg_dict[:model])
  - `sp::AbstractMIPSpace`: Weight space the indicators act on.
  - $(arg_dict[:wb_arg])
  - `wx::VecNum`: Weight expression from [`mip_wx!`](@ref).
  - $(arg_dict[:ss_arg])

# Returns

  - `ind::SignIndicators`: The declared indicator bundle.

# Related

  - [`SignIndicators`](@ref)
  - [`declare_long_short_indicators!`](@ref)
"""
function declare_sign_indicators!(model::JuMP.Model, sp::AbstractMIPSpace,
                                  wb::Option{<:WeightBounds}, wx::VecNum,
                                  ss::Option{<:Number})
    set_mip_ss_expr!(model, ss, wb)
    N = length(wx)
    xb = model[mip_key(sp, :xbgt_ib)] = JuMP.@variable(model, [1:N], binary = true)
    return SignIndicators(xb)
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
  - `ffl_flag::Bool`: Whether to add long fixed fee expressions. Emitted by [`set_fixed_fees!`](@ref) against `ilb`.
  - `ffs_flag::Bool`: Whether to add short fixed fee expressions. Emitted by [`set_fixed_fees!`](@ref) against `isb`.
  - $(arg_dict[:xbgt_flag_arg]) Served by [`set_exact_budget_constraints!`](@ref), which reuses this builder's `ilb`/`isb` rather than introducing a sign indicator of its own — they already mean long and short here, and `i_mip = ilb + isb` remains the *held* indicator that cardinality counts.

# Returns

  - `i_mip`: Combined long+short indicator expression.

# Related

  - [`mip_constraints`](@ref)
  - [`set_mip_ss_expr!`](@ref)
  - [`mip_wb`](@ref)
  - [`set_fixed_fees!`](@ref)
  - [`set_exact_budget_constraints!`](@ref)
  - [`JuMPOptimiser`](@ref)
  - [`Threshold`](@ref)
  - [`WeightBounds`](@ref)
"""
function short_mip_threshold_constraints(model::JuMP.Model, sp::AbstractMIPSpace,
                                         wb::WeightBounds, lt::Option{<:Threshold},
                                         st::Option{<:Threshold}, ffl::Option{<:Num_VecNum},
                                         ffs::Option{<:Num_VecNum}, ss::Option{<:Number},
                                         lt_flag::Bool, st_flag::Bool, ffl_flag::Bool,
                                         ffs_flag::Bool, xbgt_flag::Bool = false)
    wx = mip_wx!(model, sp)
    ind = declare_long_short_indicators!(model, sp, wb, wx, ss)
    ss = set_mip_ss_expr!(model, ss, wb)
    set_threshold_constraints!(model, sp, ind, wx, lt, st, ss, lt_flag, st_flag)
    set_fixed_fees!(model, sp, ind, ffl, ffs, ffl_flag, ffs_flag)
    if xbgt_flag
        set_exact_budget_constraints!(model, sp, ind, wx, ss)
    end
    return ind.i_mip
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
  - `ffl_flag::Bool`: Whether to add fixed fee expressions. Emitted by [`set_fixed_fees!`](@ref) against `ib`.
  - $(arg_dict[:xbgt_flag_arg]) Under a long-only weight bound the held bit doubles as the sign, so this builder's `ib` can serve it.

# Returns

  - `ib`: Binary indicator JuMP variable vector.

# Related

  - [`short_mip_threshold_constraints`](@ref)
  - [`set_mip_constraints!`](@ref)
  - [`mip_wb`](@ref)
  - [`set_mip_ss_expr!`](@ref)
  - [`set_fixed_fees!`](@ref)
  - [`set_iplg_constraints!`](@ref)
  - [`Threshold`](@ref)
  - [`WeightBounds`](@ref)
"""
function mip_constraints(model::JuMP.Model, sp::AbstractMIPSpace, wb::WeightBounds,
                         ffl::Option{<:Num_VecNum}, lt::Option{<:Threshold},
                         ss::Option{<:Number}, lt_flag::Bool, ffl_flag::Bool,
                         xbgt_flag::Bool)
    wx = mip_wx!(model, sp)
    ind = declare_held_indicators!(model, sp, wb, wx, ss)
    set_threshold_constraints!(model, sp, ind, wx, lt, lt_flag)
    set_fixed_fees!(model, sp, ind, ffl, nothing, ffl_flag, false)
    if xbgt_flag
        # Registered here rather than read from a local: the held declaration only builds the
        # big-M expression on its free-budget branch, so under a constant budget `ss` is still
        # the raw argument. `set_mip_ss_expr!` is idempotent.
        ss = set_mip_ss_expr!(model, ss, wb)
        set_exact_budget_constraints!(model, sp, ind, wx, ss)
    end
    return ind.ib
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Pin the long/short decomposition using a single binary *sign* indicator per asset.

This is the lean alternative to running a whole long-short builder just to get a sign, for when nothing else in the model needs binaries. It produces `N` of them rather than `2N`.

The long-short builder splits each asset three ways — long, short, or inactive — because a threshold has to tell `w_i = 0` apart from `w_i >= lt_i`, and cardinality has to count the assets that are held. Pinning the decomposition needs neither: the sign split is *total*, since `w_i = 0` satisfies both `w_i >= 0` and `w_i <= 0` and is picked up by either branch. One bit per asset therefore suffices, and the inactive state is pure redundancy.

It declares the sign bit and hands it to [`set_exact_budget_constraints!`](@ref), which emits the constraints; see there for the mathematical definition.

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
    ind = declare_sign_indicators!(model, sp, wb, wx, ss)
    ss = set_mip_ss_expr!(model, ss, wb)
    set_exact_budget_constraints!(model, sp, ind, wx, ss)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run the MIP indicator builder the requested features call for and return its held indicator.

Selects among the three builders on flags shared by the asset space and every sub-group:

  - [`sign_mip_constraints`](@ref) when `sign_only` — an explicit `xbgt` with nothing else to
    consume a held indicator, so a single sign bit per asset suffices (returns `nothing`).
  - [`short_mip_threshold_constraints`](@ref) when a short side exists (`model[:sw]`) and
    something needs it — a short threshold, a fixed fee, or a pinned decomposition.
  - [`mip_constraints`](@ref) otherwise — the long-only held builder.

The asset caller ([`set_mip_constraints!`](@ref)) and the sub-group callers
([`set_all_smip_constraints!`](@ref), [`set_scardmip_constraints!`](@ref),
[`set_sgcardmip_constraints!`](@ref)) all route through here, so builder selection lives in one
place. The sub-groups leave every keyword at its default — no fees, no pinning, no sign-only
branch — differing from the asset space only through `sp`.

# Arguments

  - $(arg_dict[:model])
  - `sp::AbstractMIPSpace`: Weight space the indicators act on.
  - $(arg_dict[:wb_arg])
  - $(arg_dict[:lt_arg])
  - $(arg_dict[:st_arg])
  - $(arg_dict[:ss_arg])
  - $(arg_dict[:lt_flag_arg])
  - $(arg_dict[:st_flag_arg])
  - `ffl::Option{<:Num_VecNum} = nothing`: Long-side fixed fee rate(s).
  - `ffs::Option{<:Num_VecNum} = nothing`: Short-side fixed fee rate(s).
  - `ffl_flag::Bool = false`: Whether to emit long fixed fees.
  - `ffs_flag::Bool = false`: Whether to emit short fixed fees.
  - `xbgt_flag::Bool = false`: Whether an explicit `xbgt` asked to pin the decomposition (asset space only).
  - `pin_flag::Bool = false`: Whether the running builder should pin the long/short decomposition.
  - `sign_only::Bool = false`: Whether to use the lean sign-bit builder (asset space only).

# Returns

  - The held indicator of the builder that ran, or `nothing` for the sign-bit branch.

# Related

  - [`mip_constraints`](@ref)
  - [`short_mip_threshold_constraints`](@ref)
  - [`sign_mip_constraints`](@ref)
  - [`set_mip_constraints!`](@ref)
"""
function run_mip_builder!(model::JuMP.Model, sp::AbstractMIPSpace, wb::WeightBounds,
                          lt::Option{<:Threshold}, st::Option{<:Threshold},
                          ss::Option{<:Number}, lt_flag::Bool, st_flag::Bool;
                          ffl::Option{<:Num_VecNum} = nothing,
                          ffs::Option{<:Num_VecNum} = nothing, ffl_flag::Bool = false,
                          ffs_flag::Bool = false, xbgt_flag::Bool = false,
                          pin_flag::Bool = false, sign_only::Bool = false)
    return if sign_only
        sign_mip_constraints(model, sp, wb, ss)
    elseif (st_flag || ffl_flag || ffs_flag || xbgt_flag) && haskey(model, :sw)
        short_mip_threshold_constraints(model, sp, wb, lt, st, ffl, ffs, ss, lt_flag,
                                        st_flag, ffl_flag, ffs_flag, pin_flag)
    else
        mip_constraints(model, sp, wb, ffl, lt, ss, lt_flag, ffl_flag, pin_flag)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the asset-space MIP constraints to the JuMP optimisation model and return the held indicator.

Chooses the indicator builder the requested features need ([`mip_constraints`](@ref), [`short_mip_threshold_constraints`](@ref), or [`sign_mip_constraints`](@ref)), then lets each feature emit against the bundle it returns: minimum-holding thresholds and fixed fees inside the builder, then cardinality, group cardinality, and pinning the long/short decomposition here. Integer phylogeny is *not* emitted here — it is applied late in [`assemble_jump_model!`](@ref) beside its semidefinite sibling — but its presence still forces a builder to run, so the held indicator exists for that later call to gate on. That held indicator is the return value.

# Pinning the decomposition

Two callers want the decomposition pinned, and they are the same request:

  - `xbgt`, asked for explicitly by a head whose `lw`/`sw` merely *bound* the weights ([`PartsBoundWeights`](@ref)).
  - A [`WeightsFromParts`](@ref) head, which asks for it by construction — `w = lw - sw` is what its parts are *for* — and so passes no flag.

Both route to [`set_exact_budget_constraints!`](@ref), which reads the contract back off the model and emits what that contract needs.

Pinning needs a per-asset *sign* bit, which the long-only [`mip_constraints`](@ref) does not have — its `ib` marks an asset as *held* and is what cardinality counts. So an explicit `xbgt` is served by [`sign_mip_constraints`](@ref) when nothing else in the model needs binaries (`N` of them), and otherwise routed to [`short_mip_threshold_constraints`](@ref), whose `ilb`/`isb` it reuses for free. It is ignored when the model has no short side, there being no decomposition to pin.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `card::Option{<:Integer}`: Optional maximum cardinality (number of non-zero assets).
  - `gcard::Option{<:LinearConstraint}`: Optional group cardinality constraint.
  - `pl::Option{<:PlC_VecPlC}`: Optional phylogeny constraint(s). Only used to detect whether an [`IntegerPhylogeny`](@ref) is present, which forces a builder to run so a held indicator is returned for the late [`set_iplg_constraints!`](@ref) call.
  - $(arg_dict[:lt_arg])
  - $(arg_dict[:st_arg])
  - `fees::Option{<:Fees}`: Optional fee specification.
  - $(arg_dict[:ss_arg])
  - `xbgt::Bool = false`: Whether to pin the long/short decomposition so the budgets hold exactly. See the section above.

# Returns

  - `ib`: The held indicator of the builder that ran, or `nothing` when none did (the sign-bit branch, or no MIP features at all). [`set_iplg_constraints!`](@ref) gates on it.

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
                              xbgt::Bool = false)
    card_flag = !isnothing(card)
    gcard_flag = !isnothing(gcard)
    iplg_flag = isa(pl, IntegerPhylogeny) ||
                isa(pl, AbstractVector) && any(x -> isa(x, IntegerPhylogeny), pl)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    # Pinning the decomposition is meaningless without a short side to pin: with no `sw`
    # the weights are their own long part and the budgets are already exact.
    xbgt_flag = xbgt && haskey(model, :sw)
    # A `WeightsFromParts` head needs no flag of its own to ask for the decomposition to be
    # pinned: `w = lw - sw` is what its parts are *for*. It stays distinct from `xbgt_flag`
    # because it also picks the builder below -- its `lw`/`sw` are the primitive variables,
    # so it wants the long-only builder's gate on them, not a sign builder's.
    wfp_flag = isa(decomposition_contract(model), WeightsFromParts)
    pin_flag = xbgt_flag || wfp_flag
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
         pin_flag)
        return nothing
    end
    sp = AssetMIPSpace()
    # `xbgt` needs a per-asset *sign* bit, which only the long-short builder has: the
    # long-only builder's `ib` marks an asset as held and carries no sign. When nothing here
    # consumes a held indicator, the long-short builder's third state (inactive) is dead
    # weight, so the lean sign-bit builder does the job with half the binaries. That branch
    # returns `nothing` -- there is no held indicator, and by its own condition nothing below
    # would read one.
    sign_only = xbgt_flag && !(card_flag ||
                               gcard_flag ||
                               iplg_flag ||
                               lt_flag ||
                               st_flag ||
                               ffl_flag ||
                               ffs_flag ||
                               wfp_flag)
    ib = run_mip_builder!(model, sp, wb, lt, st, ss, lt_flag, st_flag; ffl, ffs, ffl_flag,
                          ffs_flag, xbgt_flag, pin_flag, sign_only)
    set_card_constraints!(model, sp, ib, card)
    set_gcard_constraints!(model, sp, ib, gcard)
    # Hand the held indicator back so integer phylogeny can gate on it from its own emitter,
    # called late in `assemble_jump_model!` beside the semidefinite one. `ib` is `nothing` when
    # no builder ran (the sign-bit branch, or the no-flags early return) -- and by this
    # function's own guards that is exactly when no integer phylogeny can be present.
    return ib
end
