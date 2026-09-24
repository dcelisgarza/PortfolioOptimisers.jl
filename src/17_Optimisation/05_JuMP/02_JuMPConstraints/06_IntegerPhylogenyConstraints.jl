"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the integer phylogeny rows, which cap the number of held assets in each row of an [`IntegerPhylogeny`](@ref).

Each [`IntegerPhylogeny`](@ref) entry of `plgs` adds its rows on the held binary of the asset bundle. [`mip_indicators`](@ref) returns the bundle that [`set_mip_constraints!`](@ref) registered, and [`held_bin`](@ref) returns its held binary. An entry of another kind adds no row here. The method for `plgs = nothing` adds nothing, and the main method adds nothing when no MIP builder ran.

[`set_mip_constraints!`](@ref) runs a builder that declares a held binary whenever `plgs` holds an [`IntegerPhylogeny`](@ref), so the bundle is present whenever a row is due. The lean sign-bit builder registers a [`SignIndicators`](@ref), which has no held binary, and it never runs beside an integer phylogeny. The head calls this function after the risk and the return rows, beside [`set_sdp_phylogeny_constraints!`](@ref). It reads no mark that a later builder writes, so the position is a grouping and not a dependency.

# JuMP formulation

## Variables

  - The held binary of the asset bundle, read through [`held_bin`](@ref). It is `ib` of the long-only bundle, and `ilb + isb` of the long-short bundle.

## Constraints

  - `card_plg_<i>`, for the `i`-th entry of `plgs` when that entry is an [`IntegerPhylogeny`](@ref): ``s_c \\left(\\mathbf{A} \\boldsymbol{b} - \\boldsymbol{B}\\right) \\leq 0``.

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:b_held])
  - $(math_dict[:A_iplg])
  - $(math_dict[:B_iplg])
  - $(math_dict[:i_plg])

# Arguments

  - $(arg_dict[:model])
  - `plgs::Option{<:PlC_VecPlC}`: The phylogeny constraint result, a vector of them, or `nothing`.

# Validation

  - Each [`IntegerPhylogeny`](@ref) entry has one column for each asset of the model, else `DimensionMismatch`.

# Returns

  - `nothing`.

# Related

  - [`mip_indicators`](@ref)
  - [`held_bin`](@ref)
  - [`set_mip_constraints!`](@ref): Registers the bundle that the rows read.
  - [`set_frc_iplg_constraints!`](@ref): Adds the same rows on the factor weights.
  - [`set_sdp_phylogeny_constraints!`](@ref)
  - [`IntegerPhylogeny`](@ref)
"""
function set_iplg_constraints!(model::JuMP.Model, plgs::PlC_VecPlC)
    ind = mip_indicators(model)
    if isnothing(ind)
        return nothing
    end
    sc = get_constraint_scale(model)
    ib = held_bin(ind)
    for (i, pl) in enumerate(plgs)
        if !isa(pl, IntegerPhylogeny)
            continue
        end
        A = pl.A
        B = pl.B
        @argcheck(size(A, 2) == length(ib),
                  DimensionMismatch("entry $i of the integer phylogeny has $(size(A, 2)) columns, but the model has $(length(ib)) assets"))
        state_set!(model, Symbol(""), :card_plg_, i,
                   JuMP.@constraint(model, sc * (A * ib ⊖ B) <= 0))
    end
    return nothing
end
function set_iplg_constraints!(::JuMP.Model, ::Nothing)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

MIP space of the factor weights `w1` of a [`FactorRiskContribution`](@ref) model.

[`mip_wx!`](@ref) returns `w1`, and [`mip_key`](@ref) writes the prefix `frc_` into each model key, so the held binary of this space is `frc_ib`. [`factor_weight_bounds`](@ref) states the gate bounds on the factor axis, so [`mip_bounds`](@ref) returns them unchanged. This space registers no bundle in Model State. [`set_frc_iplg_constraints!`](@ref) declares the bundle and reads it in the same call, so [`mip_indicators`](@ref) still returns the asset bundle.

# Related

  - [`AbstractMIPSpace`](@ref)
  - [`AssetMIPSpace`](@ref)
  - [`set_frc_iplg_constraints!`](@ref)
"""
struct FactorMIPSpace <: AbstractMIPSpace end
function mip_key(::FactorMIPSpace, name::Symbol)
    return Symbol(:frc_, name)
end
function mip_wx!(model::JuMP.Model, ::FactorMIPSpace)
    return shared_get(model, :w1)
end
function mip_bounds(::FactorMIPSpace, b::VecNum)
    return b
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the bounds of the factor weights that the asset weight bounds imply, widened to hold zero.

[`set_factor_risk_contribution_constraints!`](@ref) builds the factor basis `b1` as the pseudoinverse of ``\\mathbf{B}^\\intercal``. When the loadings have full column rank, ``\\mathbf{B}^\\intercal`` times `b1` is the identity, so ``\\boldsymbol{w}_1 = \\mathbf{B}^\\intercal \\boldsymbol{w}``. The off-factor block `b2 * w2` adds nothing to that product, because its columns lie in the null space of ``\\mathbf{B}^\\intercal``. Each factor weight is therefore a fixed linear combination of the asset weights, and the asset bounds bound it. The bounds gate a held bit, and a factor that is not held has a weight of zero, so the function widens each bound to hold zero.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\ell}_{f} &= \\min\\left(\\mathbf{0},\\, \\mathbf{B}_{+}^\\intercal \\boldsymbol{\\ell} + \\mathbf{B}_{-}^\\intercal \\boldsymbol{u}\\right)\\,, \\\\
\\boldsymbol{u}_{f} &= \\max\\left(\\mathbf{0},\\, \\mathbf{B}_{+}^\\intercal \\boldsymbol{u} + \\mathbf{B}_{-}^\\intercal \\boldsymbol{\\ell}\\right)\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{\\ell}``, ``\\boldsymbol{u}``: Lower and upper asset weight bounds from `wb`. A scalar bound applies to every asset.
  - $(math_dict[:B_loadings])
  - ``\\mathbf{B}_{+}``, ``\\mathbf{B}_{-}``: The positive and the negative part of ``\\mathbf{B}``, taken elementwise, so that ``\\mathbf{B} = \\mathbf{B}_{+} + \\mathbf{B}_{-}``.
  - $(math_dict[:lu_factor]) The minimum and the maximum are elementwise.

Before the widening, each bound is the exact extreme of ``\\mathbf{B}^\\intercal \\boldsymbol{w}`` over the box ``\\boldsymbol{\\ell} \\leq \\boldsymbol{w} \\leq \\boldsymbol{u}``. The budget and the other rows of the model do not tighten it.

# Arguments

  - `wb::Option{<:WeightBounds}`: The asset weight bounds of the optimisation.
  - `Bt::MatNum`: The transposed loading matrix ``\\mathbf{B}^\\intercal``, of size ``N_{f} \\times N``.

# Validation

  - `wb`, `wb.lb` and `wb.ub` are not `nothing`, and every lower and upper bound is finite. A factor weight has no bound of its own, so an asset without a finite bound leaves the gate of each factor that loads on it without a value.

# Returns

  - `WeightBounds`: The factor weight bounds, one entry for each factor.

# Related

  - [`set_frc_iplg_constraints!`](@ref)
  - [`FactorMIPSpace`](@ref)
  - [`WeightBounds`](@ref)
"""
function factor_weight_bounds(wb::Option{<:WeightBounds}, Bt::MatNum)
    @argcheck(!isnothing(wb) &&
              !isnothing(wb.lb) &&
              !isnothing(wb.ub) &&
              all(isfinite, wb.lb) &&
              all(isfinite, wb.ub),
              ArgumentError("an IntegerPhylogeny on the factor weights needs finite lower and upper asset weight bounds, because the bounds of the factor weights are derived from them"))
    N = size(Bt, 2)
    lb = isa(wb.lb, Number) ? fill(wb.lb, N) : wb.lb
    ub = isa(wb.ub, Number) ? fill(wb.ub, N) : wb.ub
    Bp = max.(Bt, zero(eltype(Bt)))
    Bn = min.(Bt, zero(eltype(Bt)))
    flb = min.(Bp * lb + Bn * ub, zero(eltype(Bt)))
    fub = max.(Bp * ub + Bn * lb, zero(eltype(Bt)))
    return WeightBounds(flb, fub)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the integer phylogeny rows on the factor weights of a [`FactorRiskContribution`](@ref) model.

The asset builder [`set_iplg_constraints!`](@ref) reads the held binary of the asset bundle. This builder declares a held binary of its own, one bit for each factor, through [`declare_held_indicators!`](@ref) in the [`FactorMIPSpace`](@ref). The gate rows of that call force the weight of a factor to zero when its bit is zero. Each [`IntegerPhylogeny`](@ref) entry of `plgs` then adds its rows on the factor bits. A factor weight is free in sign, so its gate needs a bound on each side, and [`factor_weight_bounds`](@ref) derives both from the asset weight bounds.

The builder adds nothing when `plgs` holds no [`IntegerPhylogeny`](@ref). The head calls it after [`assemble_jump_model!`](@ref), beside [`set_sdp_frc_phylogeny_constraints!`](@ref). When the budget ``k`` is a variable, the gate reads the big-M constant `ss`. An asset MIP builder registered it if one ran. If none ran, this builder registers it from the asset weight bounds, as the asset builder does.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\ell}_{f} \\odot \\boldsymbol{b} \\leq \\boldsymbol{w}_1 &\\leq \\boldsymbol{u}_{f} \\odot \\boldsymbol{b}\\,, \\\\
\\mathbf{A} \\boldsymbol{b} &\\leq \\boldsymbol{B}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_1_factor])
  - $(math_dict[:b_held])
  - $(math_dict[:lu_factor])
  - $(math_dict[:A_iplg])
  - $(math_dict[:B_iplg])

# JuMP formulation

## Variables

  - `w1`: The factor weights, read from the model.
  - `k`: The budget, read from the model.
  - `frc_ib`: The held binary of the factors, created here.
  - `frc_ibf`: The continuous product ``\\boldsymbol{f} = \\boldsymbol{b} k``, created here when ``k`` is a variable.

## Expressions

  - `frc_i_mip`: The gate ``\\boldsymbol{g}``. It is ``\\boldsymbol{b}`` when ``k`` is a number, and ``\\boldsymbol{f}`` when ``k`` is a variable.

## Constraints

  - `frc_ibf_ub`: ``s_c \\left(\\boldsymbol{f} - k\\right) \\leq 0``, when ``k`` is a variable.
  - `frc_ibfd_ub`: ``s_c \\left(\\boldsymbol{f} - s_s \\boldsymbol{b}\\right) \\leq 0``, when ``k`` is a variable.
  - `frc_ibfd_lb`: ``s_c \\left(\\boldsymbol{f} + s_s (\\mathbf{1} - \\boldsymbol{b}) - k\\right) \\geq 0``, when ``k`` is a variable.
  - `frc_w_mip_lb`: ``s_c \\left(\\boldsymbol{w}_1 - \\boldsymbol{\\ell}_{f} \\odot \\boldsymbol{g}\\right) \\geq 0``.
  - `frc_w_mip_ub`: ``s_c \\left(\\boldsymbol{w}_1 - \\boldsymbol{u}_{f} \\odot \\boldsymbol{g}\\right) \\leq 0``.
  - `frc_card_plg_<i>`, for the `i`-th entry of `plgs` when that entry is an [`IntegerPhylogeny`](@ref): ``s_c \\left(\\mathbf{A} \\boldsymbol{b} - \\boldsymbol{B}\\right) \\leq 0``.

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:k_budget])
  - ``s_s``: The big-M constant `ss`. The three rows of ``\\boldsymbol{f}`` give ``f_j = k`` when ``b_j = 1`` and ``f_j = 0`` when ``b_j = 0``, provided that ``k \\leq s_s``.
  - $(math_dict[:i_plg])

The rows `frc_card_plg_<i>` read the bits ``\\boldsymbol{b}`` and never the gate, so they count held factors when ``k`` is a variable too.

# Arguments

  - $(arg_dict[:model])
  - `plgs::Option{<:PlC_VecPlC}`: The phylogeny constraint result on the factors, a vector of them, or `nothing`.
  - `wb::Option{<:WeightBounds}`: The asset weight bounds of the optimisation.
  - `Bt::MatNum`: The transposed loading matrix ``\\mathbf{B}^\\intercal``.
  - $(arg_dict[:ss_arg])

# Validation

  - The asset weight bounds pass the check of [`factor_weight_bounds`](@ref).
  - Each [`IntegerPhylogeny`](@ref) entry has one column for each factor, `size(Bt, 1)` in all, else `DimensionMismatch`.

# Returns

  - `nothing`.

# Related

  - [`set_iplg_constraints!`](@ref): Adds the same rows on the asset weights.
  - [`set_sdp_frc_phylogeny_constraints!`](@ref)
  - [`factor_weight_bounds`](@ref)
  - [`declare_held_indicators!`](@ref)
  - [`FactorMIPSpace`](@ref)
  - [`IntegerPhylogeny`](@ref)
"""
function set_frc_iplg_constraints!(model::JuMP.Model, plgs::Option{<:PlC_VecPlC},
                                   wb::Option{<:WeightBounds}, Bt::MatNum,
                                   ss::Option{<:Number})
    if !(isa(plgs, IntegerPhylogeny) ||
         isa(plgs, AbstractVector) && any(x -> isa(x, IntegerPhylogeny), plgs))
        return nothing
    end
    fwb = factor_weight_bounds(wb, Bt)
    if !isa(get_k(model), Number)
        # `factor_weight_bounds` refused a `nothing` bound above.
        set_mip_ss_expr!(model, ss, wb::WeightBounds)
    end
    sp = FactorMIPSpace()
    ib = held_bin(declare_held_indicators!(model, sp, fwb, mip_wx!(model, sp), ss))
    sc = get_constraint_scale(model)
    for (i, pl) in enumerate(plgs)
        if !isa(pl, IntegerPhylogeny)
            continue
        end
        @argcheck(size(pl.A, 2) == length(ib),
                  DimensionMismatch("entry $i of the integer phylogeny on the factors has $(size(pl.A, 2)) columns, but the model has $(length(ib)) factors"))
        state_set!(model, Symbol(""), :frc_card_plg_, i,
                   JuMP.@constraint(model, sc * (pl.A * ib ⊖ pl.B) <= 0))
    end
    return nothing
end
