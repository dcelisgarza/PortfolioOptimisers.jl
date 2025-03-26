struct HierarchicalEqualRiskContribution{T1 <: HierarchicalOptimiser,
                                         T2 <: Union{<:OptimisationRiskMeasure,
                                                     <:AbstractVector{<:OptimisationRiskMeasure}},
                                         T3 <: Union{<:OptimisationRiskMeasure,
                                                     <:AbstractVector{<:OptimisationRiskMeasure}}}
    opt::T1
    ri::T2
    ro::T3
end
function HierarchicalEqualRiskContribution(;
                                           opt::HierarchicalOptimiser = HierarchicalOptimiser(),
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
    return HierarchicalEqualRiskContribution{typeof(opt), typeof(ri), typeof(ro)}(opt, ri,
                                                                                  ro)
end
function herc_scalarised_risk_o!(::SumScalariser, wk::AbstractVector, riro::Bool,
                                 roku::AbstractVector, rkbo::AbstractVector,
                                 cl::AbstractVector,
                                 ros::AbstractVector{<:OptimisationRiskMeasure},
                                 X::AbstractMatrix, fees::Union{Nothing, <:Fees})
    crisk = zero(eltype(X))
    for ro ∈ ros
        if !riro
            unitary_expected_risks!(wk, roku, ro, X, fees)
        end
        rkbo[cl] .= inv.(view(roku, cl))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk += ro.settings.scale * expected_risk(ro, rkbo, X, fees)
    end
    return crisk
end
function herc_scalarised_risk_o!(::MaxScalariser, wk::AbstractVector, riro::Bool,
                                 roku::AbstractVector, rkbo::AbstractVector,
                                 cl::AbstractVector,
                                 ros::AbstractVector{<:OptimisationRiskMeasure},
                                 X::AbstractMatrix, fees::Union{Nothing, <:Fees})
    crisk = typemin(eltype(X))
    for ro ∈ ros
        if !riro
            unitary_expected_risks!(wk, roku, ro, X, fees)
        end
        rkbo[cl] .= inv.(view(roku, cl))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk_i = ro.settings.scale * expected_risk(ro, rkbo, X, fees)
        if crisk_i > crisk
            crisk = crisk_i
        end
    end
    return crisk
end
function herc_scalarised_risk_o!(sce::LogSumExpScalariser, wk::AbstractVector, riro::Bool,
                                 roku::AbstractVector, rkbo::AbstractVector,
                                 cl::AbstractVector,
                                 ros::AbstractVector{<:OptimisationRiskMeasure},
                                 X::AbstractMatrix, fees::Union{Nothing, <:Fees})
    crisk = zero(eltype(X))
    for ro ∈ ros
        if !riro
            unitary_expected_risks!(wk, roku, ro, X, fees)
        end
        rkbo[cl] .= inv.(view(roku, cl))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk += ro.settings.scale * sce.gamma * expected_risk(ro, rkbo, X, fees)
    end
    return log(exp(crisk)) / sce.gamma
end
function herc_scalarised_risk_i!(::SumScalariser, wk::AbstractVector, riku::AbstractVector,
                                 cl::AbstractVector,
                                 ris::AbstractVector{<:OptimisationRiskMeasure},
                                 X::AbstractMatrix, fees::Union{Nothing, <:Fees})
    risk = zeros(eltype(X), length(cl), 2)
    for ri ∈ ris
        unitary_expected_risks!(wk, riku, ri, X, fees)
        risk[:, 1] .= inv.(view(riku, cl))
        risk[:, 1] ./= sum(view(risk, :, 1))
        risk[:, 2] .+= ri.settings.scale * view(risk, :, 1)
    end
    return view(risk, :, 2)
end
function herc_scalarised_risk_i!(::MaxScalariser, wk::AbstractVector, riku::AbstractVector,
                                 cl::AbstractVector,
                                 ris::AbstractVector{<:OptimisationRiskMeasure},
                                 X::AbstractMatrix, fees::Union{Nothing, <:Fees})
    risk_t = typemin(eltype(X))
    risk = zeros(eltype(X), length(cl), 2)
    for ri ∈ ris
        unitary_expected_risks!(wk, riku, ri, X, fees)
        risk[:, 1] .= ri.settings.scale * view(risk, :, 1)
        risk_i = sum(view(risk, :, 1))
        if risk_i > risk_t
            risk_t = risk_i
            risk[:, 2] .= inv.(view(risk, :, 1))
            risk[:, 2] .= ri.settings.scale * view(risk, :, 2) / sum(view(risk, :, 2))
        end
    end
    return view(risk, :, 2)
end
function herc_scalarised_risk_i!(sce::LogSumExpScalariser, wk::AbstractVector,
                                 riku::AbstractVector, cl::AbstractVector,
                                 ris::AbstractVector{<:OptimisationRiskMeasure},
                                 X::AbstractMatrix, fees::Union{Nothing, <:Fees})
    risk = zeros(eltype(X), length(cl), 2)
    for ri ∈ ris
        unitary_expected_risks!(wk, riku, ri, X, fees)
        risk[:, 1] .= inv.(view(riku, cl))
        risk[:, 1] ./= sum(view(risk, :, 1))
        risk[:, 2] .+= ri.settings.scale * sce.gamma * view(risk, :, 1)
    end
    return log.(exp.(view(risk, :, 2))) / sce.gamma
end
function herc_risk(hc::HierarchicalEqualRiskContribution{<:Any, <:OptimisationRiskMeasure,
                                                         <:OptimisationRiskMeasure},
                   pm::AbstractPriorModel, cls::AbstractVector)
    ri = risk_measure_factory(hc.ri; prior = pm, solvers = hc.opt.slv)
    riku = unitary_expected_risks(ri, pm.X, hc.opt.fees)
    if hc.ri === hc.ro
        ro = ri
        roku = riku
    else
        ro = risk_measure_factory(hc.ro; prior = pm, solvers = hc.opt.slv)
        roku = unitary_expected_risks(ro, pm.X, hc.opt.fees)
    end
    rkbo = zeros(eltype(pm.X), size(pm.X, 2))
    rkcl = Vector{eltype(pm.X)}(undef, length(cls))
    w = Vector{eltype(pm.X)}(undef, size(pm.X, 2))
    for (i, cl) ∈ pairs(cls)
        w[cl] .= inv.(view(riku, cl))
        w[cl] ./= sum(view(w, cl))
        rkbo[cl] .= inv.(view(roku, cl))
        rkbo[cl] ./= sum(view(rkbo, cl))
        rkcl[i] = expected_risk(ro, rkbo, pm.X, hc.opt.fees)
        rkbo[cl] .= zero(eltype(pm.X))
    end
    return w, rkcl
end
function herc_risk(hc::HierarchicalEqualRiskContribution{<:Any,
                                                         <:AbstractVector{<:OptimisationRiskMeasure},
                                                         <:AbstractVector{<:OptimisationRiskMeasure}},
                   pm::AbstractPriorModel, cls::AbstractVector)
    ri = risk_measure_factory(hc.ri; prior = pm, solvers = hc.opt.slv)
    riro = false
    riku = Vector{eltype(pm.X)}(undef, size(pm.X, 2))
    if hc.ri === hc.ro
        ro = ri
        roku = riku
        riro = true
    else
        ro = risk_measure_factory(hc.ro; prior = pm, solvers = hc.opt.slv)
        roku = Vector{eltype(pm.X)}(undef, size(pm.X, 2))
    end
    rkbo = zeros(eltype(pm.X), size(pm.X, 2))
    rkcl = Vector{eltype(pm.X)}(undef, length(cls))
    w = Vector{eltype(pm.X)}(undef, size(pm.X, 2))
    wk = zeros(eltype(pm.X), size(pm.X, 2))
    for (i, cl) ∈ pairs(cls)
        w[cl] .= herc_scalarised_risk_i!(hc.opt.sce, wk, riku, cl, ri, pm.X, hc.opt.fees)
        rkcl[i] = herc_scalarised_risk_o!(hc.opt.sce, wk, riro, roku, rkbo, cl, ro, pm.X,
                                          hc.opt.fees)
        rkbo[cl] .= zero(eltype(pm.X))
    end
    return w, rkcl
end
function herc_risk(hc::HierarchicalEqualRiskContribution{<:Any, <:OptimisationRiskMeasure,
                                                         <:AbstractVector{<:OptimisationRiskMeasure}},
                   pm::AbstractPriorModel, cls::AbstractVector)
    ri = risk_measure_factory(hc.ri; prior = pm, solvers = hc.opt.slv)
    riku = unitary_expected_risks(ri, pm.X, hc.opt.fees)
    ro = risk_measure_factory(hc.ro; prior = pm, solvers = hc.opt.slv)
    roku = Vector{eltype(pm.X)}(undef, size(pm.X, 2))
    rkbo = zeros(eltype(pm.X), size(pm.X, 2))
    rkcl = Vector{eltype(pm.X)}(undef, length(cls))
    w = Vector{eltype(pm.X)}(undef, size(pm.X, 2))
    wk = zeros(eltype(pm.X), size(pm.X, 2))
    for (i, cl) ∈ pairs(cls)
        w[cl] .= inv.(view(riku, cl))
        w[cl] ./= sum(view(w, cl))
        rkcl[i] = herc_scalarised_risk_o!(hc.opt.sce, wk, false, roku, rkbo, cl, ro, pm.X,
                                          hc.opt.fees)
        rkbo[cl] .= zero(eltype(pm.X))
    end
    return w, rkcl
end
function herc_risk(hc::HierarchicalEqualRiskContribution{<:Any,
                                                         <:AbstractVector{<:OptimisationRiskMeasure},
                                                         <:OptimisationRiskMeasure},
                   pm::AbstractPriorModel, cls::AbstractVector)
    ri = risk_measure_factory(hc.ri; prior = pm, solvers = hc.opt.slv)
    riku = Vector{eltype(pm.X)}(undef, size(pm.X, 2))
    ro = risk_measure_factory(hc.ro; prior = pm, solvers = hc.opt.slv)
    roku = unitary_expected_risks(ro, pm.X, hc.opt.fees)
    rkbo = zeros(eltype(pm.X), size(pm.X, 2))
    rkcl = Vector{eltype(pm.X)}(undef, length(cls))
    w = Vector{eltype(pm.X)}(undef, size(pm.X, 2))
    wk = zeros(eltype(pm.X), size(pm.X, 2))
    for (i, cl) ∈ pairs(cls)
        w[cl] .= herc_scalarised_risk_i!(hc.opt.sce, wk, riku, cl, ri, pm.X, hc.opt.fees)
        rkbo[cl] .= inv.(view(roku, cl))
        rkbo[cl] ./= sum(view(rkbo, cl))
        rkcl[i] = expected_risk(ro, rkbo, pm.X, hc.opt.fees)
        rkbo[cl] .= zero(eltype(pm.X))
    end
    return w, rkcl
end
function optimise!(hc::HierarchicalEqualRiskContribution, rd::ReturnsData = ReturnsData();
                   strict::Bool = false)
    pm = prior(hc.opt.pe, rd.X, rd.F)
    clm = clusterise(hc.opt.cle, pm.X)
    idx = cutree(clm.clustering; k = clm.k)
    cls = [findall(x -> x == i, idx) for i ∈ 1:(clm.k)]
    w, rkcl = herc_risk(hc, pm, cls)
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
        # This implicitly multiplies the asset risks by the cluster risk. We eliminate the allocation of a vector of cluster weights, and a loop at the end.
        # wcl = ones(eltype(pm.X), clm.k)
        # for i ∈ nd[1:(clm.k - 1)]
        #     ...
        #     <this loop>
        #     ...
        #     wcl[lc] *= alpha
        #     wcl[rc] *= one(alpha) - alpha
        # end
        # for (i, cl) ∈ pairs(cls)
        #     w[cl] *= view(rkcl, i)
        # end
        # because `ln` and `rn` contain `cl`.
        w[ln] *= alpha
        w[rn] *= one(alpha) - alpha
    end
    return finalise_hierarchical_weights(hc.opt.cwf,
                                         weight_bounds_constraints(hc.opt.wb; N = length(w),
                                                                   strict = strict),
                                         w / sum(w))
end

export HierarchicalEqualRiskContribution
