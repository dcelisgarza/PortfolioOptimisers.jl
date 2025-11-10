function set_sdp_constraints!(model::JuMP.Model)
    if haskey(model, :W)
        return model[:W]
    end
    w = model[:w]
    k = ifelse(haskey(model, :crkb), 1, model[:k])
    sc = model[:sc]
    N = length(w)
    @variable(model, W[1:N, 1:N], Symmetric)
    @expression(model, M, hcat(vcat(W, transpose(w)), vcat(w, k)))
    @constraint(model, M_PSD, sc * M in PSDCone())
    return W
end
function set_sdp_frc_constraints!(model::JuMP.Model)
    if haskey(model, :frc_W)
        return model[:frc_W]
    end
    w1 = model[:w1]
    sc = model[:sc]
    k = model[:k]
    Nf = length(w1)
    @variable(model, frc_W[1:Nf, 1:Nf], Symmetric)
    @expression(model, frc_M, hcat(vcat(frc_W, transpose(w1)), vcat(w1, k)))
    @constraint(model, frc_M_PSD, sc * frc_M in PSDCone())
    return frc_W
end
function set_sdp_phylogeny_constraints!(model::JuMP.Model, plgs::Option{<:PhC_VecPhC})
    if !(isa(plgs, SemiDefinitePhylogeny) ||
         isa(plgs, AbstractVector) && any(x -> isa(x, SemiDefinitePhylogeny), plgs))
        return nothing
    end
    sc = model[:sc]
    W = set_sdp_constraints!(model)
    for (i, plg) in enumerate(plgs)
        if !isa(plg, SemiDefinitePhylogeny)
            continue
        end
        key = Symbol(:sdp_plg_, i)
        A = plg.A
        model[key] = @constraint(model, sc * A ⊙ W == 0)
        if !haskey(model, :variance_flag)
            key = Symbol(key, :_p)
            p = plg.p
            plp = model[key] = @expression(model, p * tr(W))
            add_to_objective_penalty!(model, plp)
        end
    end
    return nothing
end
function set_sdp_frc_phylogeny_constraints!(model::JuMP.Model,
                                            plgs::Option{<:PhCE_PhC_VecPhCE_PhC})
    if !(isa(plgs, SemiDefinitePhylogeny) ||
         isa(plgs, AbstractVector) && any(x -> isa(x, SemiDefinitePhylogeny), plgs))
        return nothing
    end
    sc = model[:sc]
    W = set_sdp_frc_constraints!(model)
    for (i, plg) in enumerate(plgs)
        if !isa(plg, SemiDefinitePhylogeny)
            continue
        end
        key = Symbol(:frc_sdp_plg_, i)
        A = plg.A
        model[key] = @constraint(model, sc * A ⊙ W == 0)
        if !haskey(model, :variance_flag)
            key = Symbol(key, :_p)
            p = plg.p
            plp = model[key] = @expression(model, p * tr(W))
            add_to_objective_penalty!(model, plp)
        end
    end
    return nothing
end
