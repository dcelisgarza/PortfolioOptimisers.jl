"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add integer phylogeny cardinality constraints to the JuMP optimisation model.

Iterates over `plgs` and, for each [`IntegerPhylogeny`](@ref) entry, enforces `A * ib ≤ B` where `ib` is the *held binary* ([`held_bin`](@ref)) of the MIP builder that ran. The bundle is read back from Model State with [`mip_indicators`](@ref) rather than hand-threaded from [`set_mip_constraints!`](@ref): the builder registered it under one canonical entry, and each builder's own raw keys stay private to it.

This runs in the same neighbourhood of [`assemble_jump_model!`](@ref) as its semidefinite sibling [`set_sdp_phylogeny_constraints!`](@ref): both are phylogeny constraints applied to the fully-assembled weights. Unlike the semidefinite one it reads nothing that later builders set (no `:variance_flag`), so its position there is a grouping, not a dependency.

`held_bin` is read only inside the [`IntegerPhylogeny`](@ref) branch, so a bundle that carries no held indicator (a [`SignIndicators`](@ref) from the lean sign-bit builder) is never asked for one — by [`set_mip_constraints!`](@ref)'s own guards that builder rules out any integer phylogeny. The `::Nothing` method and the early `mip_indicators` guard make the call *total*: no phylogeny of any kind, or no MIP builder having run at all.

# Arguments

  - $(arg_dict[:model])
  - `plgs`: Collection of phylogeny constraint objects (or `nothing`).

# Returns

  - `nothing`.

# Related

  - [`mip_indicators`](@ref)
  - [`held_bin`](@ref)
  - [`set_mip_constraints!`](@ref)
  - [`set_sdp_phylogeny_constraints!`](@ref)
  - [`IntegerPhylogeny`](@ref)
"""
function set_iplg_constraints!(model::JuMP.Model, plgs::PlC_VecPlC)
    ind = mip_indicators(model)
    if isnothing(ind)
        return nothing
    end
    sc = get_constraint_scale(model)
    for (i, pl) in enumerate(plgs)
        if !isa(pl, IntegerPhylogeny)
            continue
        end
        A = pl.A
        B = pl.B
        ib = held_bin(ind)
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

Factor-space MIP constraints: the indicators gate the factor weights `w1` of a [`FactorRiskContribution`](@ref) model, and the model keys take the prefix `frc_` (`:frc_ib`, `:frc_w_mip_lb`, …).

The bounds that the indicators gate with are stated on the factor axis already, by [`factor_weight_bounds`](@ref), so [`mip_bounds`](@ref) returns them unchanged. No bundle of this space is registered in Model State: [`set_frc_iplg_constraints!`](@ref) declares it and reads it in the same call, so [`mip_indicators`](@ref) keeps the asset bundle.

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

The factor basis `b1` of [`set_factor_risk_contribution_constraints!`](@ref) is the pseudoinverse of ``\\mathbf{B}^\\intercal``, so ``\\boldsymbol{w}_1 = \\mathbf{B}^\\intercal \\boldsymbol{w}`` when the loadings have full column rank. The off-factor block `b2 * w2` adds nothing to that product, because its columns span the null space of ``\\mathbf{B}^\\intercal``. Each factor weight is therefore a fixed combination of the asset weights, and the asset bounds bound it. The bounds are widened to hold zero, because they gate a *held* bit: a factor that is not held has a weight of zero.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\ell}_{f} &= \\min\\left(\\mathbf{0},\\, \\mathbf{B}_{+}^\\intercal \\boldsymbol{\\ell} + \\mathbf{B}_{-}^\\intercal \\boldsymbol{u}\\right)\\,, \\\\
\\boldsymbol{u}_{f} &= \\max\\left(\\mathbf{0},\\, \\mathbf{B}_{+}^\\intercal \\boldsymbol{u} + \\mathbf{B}_{-}^\\intercal \\boldsymbol{\\ell}\\right)\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{\\ell}``, ``\\boldsymbol{u}``: Lower and upper asset weight bounds from `wb`.
  - ``\\mathbf{B}_{+}``, ``\\mathbf{B}_{-}``: The positive and the negative part of the loading matrix, taken elementwise, so that ``\\mathbf{B} = \\mathbf{B}_{+} + \\mathbf{B}_{-}``.
  - ``\\boldsymbol{\\ell}_{f}``, ``\\boldsymbol{u}_{f}``: Lower and upper factor weight bounds. The minimum and the maximum are elementwise.

# Arguments

  - `wb::Option{<:WeightBounds}`: The asset weight bounds of the optimisation.
  - `Bt::MatNum`: The transposed loading matrix ``\\mathbf{B}^\\intercal``, of size ``N_{f} \\times N``.

# Validation

  - `wb`, `wb.lb` and `wb.ub` are not `nothing`, and every lower and upper bound is finite. A factor weight has no bound of its own, so an asset without a finite bound leaves the big-M gate of each factor that loads on it without a value.

# Returns

  - `WeightBounds`: The factor weight bounds, one entry per factor.

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

Add integer phylogeny cardinality constraints on the factor weights of a [`FactorRiskContribution`](@ref) model.

Iterates over `plgs` and, for each [`IntegerPhylogeny`](@ref) entry, enforces `A * ib ≤ B`, where `ib` holds one *held* bit per factor. The asset builder [`set_iplg_constraints!`](@ref) gates on the asset bits, and this builder declares bits of its own on `w1`, through [`declare_held_indicators!`](@ref) in the [`FactorMIPSpace`](@ref). A factor weight is free in sign, so its gate needs bounds on both sides: [`factor_weight_bounds`](@ref) derives them from the asset weight bounds.

Does nothing when `plgs` contains no [`IntegerPhylogeny`](@ref). The head calls it after [`assemble_jump_model!`](@ref), beside [`set_sdp_frc_phylogeny_constraints!`](@ref), so an asset MIP builder has registered the big-M constant `:ss` of the budget when one ran. When none ran and the budget `k` is a variable, this builder registers `:ss` from the *asset* bounds, as the asset builder does.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\ell}_{f} \\odot \\boldsymbol{z} \\leq \\boldsymbol{w}_1 &\\leq \\boldsymbol{u}_{f} \\odot \\boldsymbol{z}\\,, \\\\
\\mathbf{A} \\boldsymbol{z} &\\leq \\boldsymbol{b}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}_1``: Factor weight vector.
  - ``\\boldsymbol{z}``: Binary held bits, one per factor. When the budget ``k`` is a variable, the gate reads the continuous product ``\\boldsymbol{z} k`` in place of ``\\boldsymbol{z}``, and the rows ``\\mathbf{A} \\boldsymbol{z} \\leq \\boldsymbol{b}`` read the bits.
  - ``\\boldsymbol{\\ell}_{f}``, ``\\boldsymbol{u}_{f}``: Factor weight bounds from [`factor_weight_bounds`](@ref).
  - ``\\mathbf{A}``, ``\\boldsymbol{b}``: The rows and the right-hand side of the [`IntegerPhylogeny`](@ref).

# JuMP formulation

  - `frc_ib`, `frc_i_mip`, `frc_w_mip_lb`, `frc_w_mip_ub`: The bits, the gate and the two gate rows of [`declare_held_indicators!`](@ref). A variable budget adds `frc_ibf`, `frc_ibf_ub`, `frc_ibfd_ub` and `frc_ibfd_lb`.
  - `frc_card_plg_<i>`: The rows of the `i`-th entry of `plgs`, indexed by its position.

# Arguments

  - $(arg_dict[:model])
  - `plgs`: Phylogeny constraint(s). Accepts `nothing`, a single phylogeny, or a vector.
  - `wb::Option{<:WeightBounds}`: The asset weight bounds of the optimisation.
  - `Bt::MatNum`: The transposed loading matrix ``\\mathbf{B}^\\intercal``.
  - $(arg_dict[:ss_arg])

# Returns

  - `nothing`.

# Related

  - [`set_iplg_constraints!`](@ref)
  - [`set_sdp_frc_phylogeny_constraints!`](@ref)
  - [`factor_weight_bounds`](@ref)
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
        set_mip_ss_expr!(model, ss, wb)
    end
    sp = FactorMIPSpace()
    ib = held_bin(declare_held_indicators!(model, sp, fwb, mip_wx!(model, sp), ss))
    sc = get_constraint_scale(model)
    for (i, pl) in enumerate(plgs)
        if !isa(pl, IntegerPhylogeny)
            continue
        end
        state_set!(model, Symbol(""), :frc_card_plg_, i,
                   JuMP.@constraint(model, sc * (pl.A * ib ⊖ pl.B) <= 0))
    end
    return nothing
end
