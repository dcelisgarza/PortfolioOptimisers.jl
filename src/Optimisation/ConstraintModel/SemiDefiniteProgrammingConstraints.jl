function set_sdp_constraints!(model::JuMP.Model)
    if haskey(model, :W)
        return nothing
    end
    w, k, sc = get_w_k_sc(model)
    N = length(w)
    @variable(model, W[1:N, 1:N], Symmetric)
    @expression(model, M, hcat(vcat(W, transpose(w)), vcat(w, k)))
    @constraint(model, M_PSD, sc * M ∈ PSDCone())
    return nothing
end
function set_sdp_frc_constraints!(model::JuMP.Model)
    if haskey(model, :W1)
        return nothing
    end
    sc = model[:sc]
    w1 = model[:w1]
    k = model[:k]
    Nf = length(w1)
    @variable(model, W1[1:Nf, 1:Nf], Symmetric)
    @expression(model, M1, hcat(vcat(W1, transpose(w1)), vcat(w1, k)))
    @constraint(model, M1_PSD, sc * M1 ∈ PSDCone())
    return nothing
end
function set_sdp_network_cluster_constraints!(model::JuMP.Model, cadj::AdjacencyConstraint,
                                              nadj::AdjacencyConstraint)
    c_flag = isa(cadj, SDP)
    n_flag = isa(nadj, SDP)
    if !(n_flag || c_flag)
        return nothing
    end
    sc = model[:sc]
    set_sdp_constraints!(model)
    W = model[:W]
    if c_flag
        @constraint(model, csdp, sc * cadj.A .* W == 0)
        if !haskey(model, :variance_flag)
            add_to_expression!(model[:obj_pen], cadj.p, tr(W))
        end
    end
    if n_flag
        @constraint(model, nsdp, sc * nadj.A .* W == 0)
        if !haskey(model, :variance_flag)
            add_to_expression!(model[:obj_pen], nadj.p, tr(W))
        end
    end
    return nothing
end
function set_sdp_frc_network_cluster_constraints!(model::JuMP.Model,
                                                  cadj::AdjacencyConstraint,
                                                  nadj::AdjacencyConstraint)
    c_flag = isa(cadj, SDP)
    n_flag = isa(nadj, SDP)
    if !(n_flag || c_flag)
        return nothing
    end
    sc = model[:sc]
    set_sdp_frc_constraints!(model)
    W1 = model[:W1]
    if c_flag
        @constraint(model, csdp, sc * cadj.A .* W1 == 0)
        if !haskey(model, :variance_flag)
            add_to_expression!(model[:obj_pen], cadj.p, tr(W1))
        end
    end
    if n_flag
        @constraint(model, nsdp, sc * nadj.A .* W1 == 0)
        if !haskey(model, :variance_flag)
            add_to_expression!(model[:obj_pen], nadj.p, tr(W1))
        end
    end
    return nothing
end
