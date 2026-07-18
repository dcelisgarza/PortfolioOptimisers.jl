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
        model[Symbol(:card_plg_, i)] = JuMP.@constraint(model, sc * (A * ib ⊖ B) <= 0)
    end
    return nothing
end
function set_iplg_constraints!(::JuMP.Model, ::Nothing)
    return nothing
end
