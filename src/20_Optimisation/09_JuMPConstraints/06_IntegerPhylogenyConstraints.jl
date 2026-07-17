"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add integer phylogeny cardinality constraints to the JuMP optimisation model.

Iterates over `plgs` and, for each [`IntegerPhylogeny`](@ref) entry, enforces `A * ib ≤ B` where `ib` is the *held* indicator of the MIP builder that ran — `ib` from [`mip_constraints`](@ref) or `i_mip` from [`short_mip_threshold_constraints`](@ref), whichever [`set_mip_constraints!`](@ref) returned. It is passed in rather than read from the model, because the held indicator is registered under a different key by each builder.

This runs in the same neighbourhood of [`assemble_jump_model!`](@ref) as its semidefinite sibling [`set_sdp_phylogeny_constraints!`](@ref): both are phylogeny constraints applied to the fully-assembled weights. Unlike the semidefinite one it reads nothing that later builders set (no `:variance_flag`), so its position there is a grouping, not a dependency.

The two fall-through methods make the call *total* over what [`set_mip_constraints!`](@ref) hands back: `::Nothing` for the held indicator (no MIP builder ran, so there is nothing to gate and, by construction, no integer phylogeny to gate it), and `::Nothing` for `plgs` (no phylogeny of any kind).

# Arguments

  - $(arg_dict[:model])
  - `plgs`: Collection of phylogeny constraint objects (or `nothing`).
  - `ib::VecNum`: Held indicator returned by the MIP builder that ran (or `nothing`).

# Returns

  - `nothing`.

# Related

  - [`mip_constraints`](@ref)
  - [`short_mip_threshold_constraints`](@ref)
  - [`set_mip_constraints!`](@ref)
  - [`set_sdp_phylogeny_constraints!`](@ref)
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
function set_iplg_constraints!(::JuMP.Model, ::Option{<:PlC_VecPlC}, ::Nothing)
    return nothing
end
function set_iplg_constraints!(::JuMP.Model, ::Nothing, ::VecNum)
    return nothing
end
