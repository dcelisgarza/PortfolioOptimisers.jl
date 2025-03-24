struct HierarchicalEqualRiskParity{T1 <: HierarchicalOptimiser,
                                   T2 <: Union{<:OptimisationRiskMeasure,
                                               <:AbstractVector{<:OptimisationRiskMeasure}},
                                   T3 <: Union{<:OptimisationRiskMeasure,
                                               <:AbstractVector{<:OptimisationRiskMeasure}}}
    opt::T1
    ri::T2
    ro::T3
end
function HierarchicalEqualRiskParity(; opt::HierarchicalOptimiser = HierarchicalOptimiser(),
                                     ri::Union{<:OptimisationRiskMeasure,
                                               <:AbstractVector{<:OptimisationRiskMeasure}} = Variance(),
                                     ro::Union{<:OptimisationRiskMeasure,
                                               <:AbstractVector{<:OptimisationRiskMeasure}} = ri)
    if isa(ri, AbstractVector)
        @smart_assert(!isempty(ri))
    end
    if isa(ro, AbstractVector)
        @smart_assert(!isempty(ro))
    end
    return HierarchicalEqualRiskParity{typeof(opt), typeof(ri), typeof(ro)}(opt, ri, ro)
end
function optimise!(hc::HierarchicalEqualRiskParity, rd::ReturnsData = ReturnsData())
    pm = prior(hc.opt.pe, rd.X, rd.F)
    ri = risk_measure_factory(hc.ri; prior = pm, solvers = hc.opt.slv)
    riku = unitary_expected_risks(ri, pm.X, hc.opt.fees, hc.opt.sce)
    if hc.ri === hc.ro
        ro = ri
        roku = riku
    else
        ro = risk_measure_factory(hc.ro; prior = pm, solvers = hc.opt.slv)
        roku = unitary_expected_risks(ro, pm.X, hc.opt.fees, hc.opt.sce)
    end
    clm = clusterise(hc.opt.cle, pm.X)
    idx = cutree(clm.clustering; k = clm.k)
    cls = [findall(x -> x == i, idx) for i ∈ 1:(clm.k)]
    rkbi = Vector{eltype(pm.X)}(undef, size(pm.X, 2))
    rkbo = Vector{eltype(pm.X)}(undef, size(pm.X, 2))
    rkcl = Vector{eltype(pm.X)}(undef, clm.k)
    w = ones(eltype(pm.X), size(pm.X, 2))
    for (i, cl) ∈ pairs(cls)
        fill!(rkbi, zero(eltype(pm.X)))
        fill!(rkbo, zero(eltype(pm.X)))
        rkbo[cl] = inv.(view(roku, cl))
        rkbo /= sum(rkbo)
        rkcl[i] = expected_risk(ro, rkbo, pm.X, hc.opt.fees, hc.opt.sce)
        rkbi[cl] = inv.(view(riku, cl))
        rkbi /= sum(rkbi)
        w[cl] .*= view(rkbi, cl)
    end
    nd = to_tree(clm.clustering)[2]
    hs = [i.height for i ∈ nd]
    nd = nd[sortperm(hs; rev = true)]

    # Treat each cluster as its own portfolio and optimise each one individually.
    # Calculate the weight of each cluster relative to the other clusters.
    for i ∈ nd[1:(clm.k - 1)]
        if is_leaf(i)
            continue
        end
        # Do this recursively accounting for the dendrogram structure.
        ln = pre_order(i.left)
        rn = pre_order(i.right)
        lc = Int[]
        rc = Int[]
        for (i, cl) ∈ pairs(cls)
            if issubset(cl, ln)
                push!(lc, i)
            elseif issubset(cl, rn)
                push!(rc, i)
            end
        end
        lrisk = sum(view(rkcl, lc))
        rrisk = sum(view(rkcl, rc))
        risk = lrisk + rrisk
        # Allocate weight to clusters.
        alpha = one(lrisk) - lrisk / risk
        # Weight constraints.
        w[ln] *= alpha
        w[rn] *= one(alpha) - alpha
    end
    return finalise_hierarchical_weights(hc.opt.cwf,
                                         create_array_weight_limits(hc.opt.wl, length(w)),
                                         w / sum(w))
end

export HierarchicalEqualRiskParity
