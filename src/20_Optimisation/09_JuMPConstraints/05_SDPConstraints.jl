"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add a positive semidefinite (PSD) constraint to the JuMP optimisation model for the portfolio weights.

Creates a symmetric matrix variable `W` and enforces that the bordered matrix `[W w; wᵀ k]` lies in the PSD cone. Returns immediately if `W` already exists in `model`.

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
    if haskey(model, Symbol(prefix, :W))
        return model[Symbol(prefix, :W)]
    end
    w = get_w(model, prefix)
    k = effective_k(model)
    sc = get_constraint_scale(model)
    N = length(w)
    W = preg!(model, prefix, :W, JuMP.@variable(model, [1:N, 1:N], Symmetric))
    M = preg!(model, prefix, :M,
              JuMP.@expression(model, hcat(vcat(W, transpose(w)), vcat(w, k))))
    preg!(model, prefix, :M_PSD, JuMP.@constraint(model, sc * M in JuMP.PSDCone()))
    return W
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
    if haskey(model, :frc_W)
        return model[:frc_W]
    end
    w1 = model[:w1]
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

Iterates over `plgs` and, for each [`SemiDefinitePhylogeny`](@ref) entry, enforces `A ⊙ W = 0` and optionally adds `p * tr(W)` to the objective penalty. Does nothing when `plgs` contains no [`SemiDefinitePhylogeny`](@ref) instances.

# Arguments

  - $(arg_dict[:model])
  - `plgs`: Phylogeny constraint(s). Accepts `nothing`, a single phylogeny, or a vector.

# Returns

  - `nothing`.

# Related

  - [`set_sdp_constraints!`](@ref)
  - [`set_sdp_frc_phylogeny_constraints!`](@ref)
  - [`SemiDefinitePhylogeny`](@ref)
"""
function set_sdp_phylogeny_constraints!(model::JuMP.Model, plgs::Option{<:PlC_VecPlC})
    if !(isa(plgs, SemiDefinitePhylogeny) ||
         isa(plgs, AbstractVector) && any(x -> isa(x, SemiDefinitePhylogeny), plgs))
        return nothing
    end
    sc = get_constraint_scale(model)
    W = set_sdp_constraints!(model)
    for (i, pl) in enumerate(plgs)
        if !isa(pl, SemiDefinitePhylogeny)
            continue
        end
        key = Symbol(:sdp_plg_, i)
        A = pl.A
        model[key] = JuMP.@constraint(model, sc * A ⊙ W == 0)
        if !haskey(model, :variance_flag)
            key = Symbol(key, :_p)
            p = pl.p
            plp = model[key] = JuMP.@expression(model, p * LinearAlgebra.tr(W))
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
                                            plgs::Option{<:PlCE_PhC_VecPlCE_PlC})
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
        key = Symbol(:frc_sdp_plg_, i)
        A = pl.A
        model[key] = JuMP.@constraint(model, sc * A ⊙ W == 0)
        if !haskey(model, :variance_flag)
            key = Symbol(key, :_p)
            p = pl.p
            plp = model[key] = JuMP.@expression(model, p * LinearAlgebra.tr(W))
            add_to_objective_penalty!(model, plp)
        end
    end
    return nothing
end
