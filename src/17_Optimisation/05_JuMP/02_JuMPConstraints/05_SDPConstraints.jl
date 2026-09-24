"""
    weights_prefix(model::JuMP.Model, prefix::Symbol)

Returns the Model State namespace that owns the weights registered under `prefix`.

One lifted matrix ``\\mathbf{W}`` belongs to one weight vector, and so do the marks that describe the measures built on it, `variance_flag` and `rc_variance`. A programme Allocation Set in a leader's model registers the model's own `w` under `:aset_` and marks the prefix `w_shared`, so the bare namespace owns it. The set's variance ceiling, its semidefinite phylogeny and the leader's measures then read one ``\\mathbf{W}`` and one set of marks, as the measures of one head do. A prefix without the mark owns itself, as a tracking build's prefix does.

# Arguments

  - $(arg_dict[:model])
  - `prefix`: The Model State namespace the weights are registered under.

# Returns

  - `owner::Symbol`: `Symbol("")` when `prefix` carries the `w_shared` mark, and `prefix` otherwise.

# Related

  - [`set_sdp_constraints!`](@ref)
  - [`get_w`](@ref)
  - [`add_allocation_set_constraints!`](@ref)
"""
function weights_prefix(model::JuMP.Model, prefix::Symbol)
    return state_has(model, prefix, :w_shared) ? Symbol("") : prefix
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add a positive semidefinite (PSD) constraint to the JuMP optimisation model for the portfolio weights.

Creates a symmetric matrix variable `W` and enforces that the bordered matrix `[W w; wᵀ k]` lies in the PSD cone. Returns immediately if `W` already exists in `model`. The matrix is registered under the namespace that owns the weights under `prefix`, [`weights_prefix`](@ref), so weights registered twice have one `W`.

# Mathematical definition

```math
\\begin{align}
\\mathbf{M} &= \\begin{bmatrix} \\mathbf{W} & \\boldsymbol{w} \\\\ \\boldsymbol{w}^\\intercal & k \\end{bmatrix} \\succeq 0 \\\\
&\\quad\\Leftrightarrow\\quad \\mathbf{W} \\succeq \\frac{\\boldsymbol{w}\\boldsymbol{w}^\\intercal}{k}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{M}``: Bordered positive semidefinite matrix.
  - ``\\mathbf{W}``: Symmetric ``N \\times N`` matrix variable.
  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])

# Arguments

  - $(arg_dict[:model])

# Returns

  - `W`: Symmetric JuMP variable matrix of size `N × N`.

# Related

  - [`set_sdp_frc_constraints!`](@ref)
  - [`set_sdp_phylogeny_constraints!`](@ref)
  - [`SemiDefinitePhylogeny`](@ref)
"""
function set_sdp_constraints!(model::JuMP.Model; prefix::Symbol = Symbol(""))
    prefix = weights_prefix(model, prefix)
    return state_build!(model, prefix, :W) do
        w = get_w(model, prefix)
        k = effective_k(model)
        sc = get_constraint_scale(model)
        N = length(w)
        W = JuMP.@variable(model, [1:N, 1:N], Symmetric)
        M = state_set!(model, prefix, :M,
                       JuMP.@expression(model, hcat(vcat(W, transpose(w)), vcat(w, k))))
        state_set!(model, prefix, :M_PSD, JuMP.@constraint(model, sc * M in JuMP.PSDCone()))
        return W
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add a positive semidefinite (PSD) constraint for factor risk contribution to the JuMP optimisation model.

Creates a symmetric matrix variable `frc_W` and enforces that the bordered matrix `[frc_W w1; w1ᵀ k]` lies in the PSD cone. Returns immediately if `frc_W` already exists in `model`.

# Arguments

  - $(arg_dict[:model])

# Returns

  - `frc_W`: Symmetric JuMP variable matrix of size `Nf × Nf`.

# Related

  - [`set_sdp_constraints!`](@ref)
  - [`set_sdp_frc_phylogeny_constraints!`](@ref)
  - [`SemiDefinitePhylogeny`](@ref)
"""
function set_sdp_frc_constraints!(model::JuMP.Model)
    if shared_has(model, :frc_W)
        return shared_get(model, :frc_W)
    end
    w1 = shared_get(model, :w1)
    sc = get_constraint_scale(model)
    k = get_k(model)
    Nf = length(w1)
    JuMP.@variable(model, frc_W[1:Nf, 1:Nf], Symmetric)
    JuMP.@expression(model, frc_M, hcat(vcat(frc_W, transpose(w1)), vcat(w1, k)))
    JuMP.@constraint(model, frc_M_PSD, sc * frc_M in JuMP.PSDCone())
    return frc_W
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add semidefinite phylogeny constraints to the JuMP optimisation model.

Iterates over `plgs` and, for each [`SemiDefinitePhylogeny`](@ref) entry, enforces `A ⊙ W = 0` and adds `p * tr(W)` to the objective penalty when no [`Variance`](@ref) was built on the same weights. A variance marks `variance_flag`, and the rule is the same whether the variance is in the objective or is a ceiling. Does nothing when `plgs` contains no [`SemiDefinitePhylogeny`](@ref) instances.

# Arguments

  - $(arg_dict[:model])
  - `plgs`: Phylogeny constraint(s). Accepts `nothing`, a single phylogeny, or a vector.
  - `prefix::Symbol`: The Model State namespace the rows and the penalty are registered under, `Symbol("")` for a head's own. The lifted `W` and the `variance_flag` it reads belong to the weights, [`weights_prefix`](@ref), so a programme Allocation Set's phylogeny in a leader's model reuses the leader's `W` and prefixes only its rows.

# Returns

  - `nothing`.

# Related

  - [`set_sdp_constraints!`](@ref)
  - [`set_sdp_frc_phylogeny_constraints!`](@ref)
  - [`SemiDefinitePhylogeny`](@ref)
"""
function set_sdp_phylogeny_constraints!(model::JuMP.Model, plgs::Option{<:PlC_VecPlC};
                                        prefix::Symbol = Symbol(""))
    if !(isa(plgs, SemiDefinitePhylogeny) ||
         isa(plgs, AbstractVector) && any(x -> isa(x, SemiDefinitePhylogeny), plgs))
        return nothing
    end
    sc = get_constraint_scale(model)
    W = set_sdp_constraints!(model; prefix = prefix)
    owner = weights_prefix(model, prefix)
    for (i, pl) in enumerate(plgs)
        if !isa(pl, SemiDefinitePhylogeny)
            continue
        end
        A = pl.A
        state_set!(model, prefix, :sdp_plg_, i, JuMP.@constraint(model, sc * A ⊙ W == 0))
        if !state_has(model, owner, :variance_flag)
            p = pl.p
            plp = state_set!(model, prefix, :sdp_plg_p_, i,
                             JuMP.@expression(model, p * LinearAlgebra.tr(W)))
            add_to_objective_penalty!(model, plp)
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add semidefinite phylogeny constraints for factor risk contribution to the JuMP optimisation model.

Iterates over `plgs` and, for each [`SemiDefinitePhylogeny`](@ref) entry, enforces `A ⊙ frc_W = 0` and optionally adds `p * tr(frc_W)` to the objective penalty. Does nothing when `plgs` contains no [`SemiDefinitePhylogeny`](@ref) instances.

# Arguments

  - $(arg_dict[:model])
  - `plgs`: Phylogeny constraint(s). Accepts `nothing`, a single phylogeny, or a vector.

# Returns

  - `nothing`.

# Related

  - [`set_sdp_frc_constraints!`](@ref)
  - [`set_sdp_phylogeny_constraints!`](@ref)
  - [`SemiDefinitePhylogeny`](@ref)
"""
function set_sdp_frc_phylogeny_constraints!(model::JuMP.Model,
                                            plgs::Option{<:PlCE_PlC_VecPlCE_PlC})
    if !(isa(plgs, SemiDefinitePhylogeny) ||
         isa(plgs, AbstractVector) && any(x -> isa(x, SemiDefinitePhylogeny), plgs))
        return nothing
    end
    sc = get_constraint_scale(model)
    W = set_sdp_frc_constraints!(model)
    for (i, pl) in enumerate(plgs)
        if !isa(pl, SemiDefinitePhylogeny)
            continue
        end
        A = pl.A
        state_set!(model, Symbol(""), :frc_sdp_plg_, i,
                   JuMP.@constraint(model, sc * A ⊙ W == 0))
        if !shared_has(model, :variance_flag)
            p = pl.p
            plp = state_set!(model, Symbol(""), :frc_sdp_plg_p_, i,
                             JuMP.@expression(model, p * LinearAlgebra.tr(W)))
            add_to_objective_penalty!(model, plp)
        end
    end
    return nothing
end
