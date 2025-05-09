struct HierarchicalEqualRiskContribution{T1 <: HierarchicalOptimiser,
                                         T2 <: Union{<:OptimisationRiskMeasure,
                                                     <:AbstractVector{<:OptimisationRiskMeasure}},
                                         T3 <: Union{<:OptimisationRiskMeasure,
                                                     <:AbstractVector{<:OptimisationRiskMeasure}}} <:
       ClusteringOptimisationEstimator
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
function opt_view(hc::HierarchicalEqualRiskContribution, i::AbstractVector)
    ri = hc.ri
    ro = hc.ro
    if ri === ro
        ri, ro = risk_measure_view(ri, i)
    else
        ri = risk_measure_view(ri, i)
        ro = risk_measure_view(ro, i)
    end
    opt = opt_view(hc.opt, i)
    return HierarchicalEqualRiskContribution(; ri = ri, ro = ro, opt = opt)
end
function herc_scalarised_risk_o!(::SumScalariser, wk::AbstractVector, roku::AbstractVector,
                                 rkbo::AbstractVector, cl::AbstractVector,
                                 ros::AbstractVector{<:OptimisationRiskMeasure},
                                 X::AbstractMatrix, fees::Union{Nothing, <:Fees})
    crisk = zero(eltype(X))
    for ro ∈ ros
        unitary_expected_risks!(wk, roku, ro, X, fees)
        rkbo[cl] .= inv.(view(roku, cl))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk += ro.settings.scale * expected_risk(ro, rkbo, X, fees)
    end
    return crisk
end
function herc_scalarised_risk_o!(::SumScalariser, wk::AbstractVector, roku::AbstractMatrix,
                                 rkbo::AbstractVector, cl::AbstractVector,
                                 ros::AbstractVector{<:OptimisationRiskMeasure},
                                 X::AbstractMatrix, fees::Union{Nothing, <:Fees})
    crisk = zero(eltype(X))
    for (i, ro) ∈ pairs(ros)
        rkbo[cl] .= inv.(view(roku, cl, i))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk += ro.settings.scale * expected_risk(ro, rkbo, X, fees)
    end
    return crisk
end
function herc_scalarised_risk_o!(::MaxScalariser, wk::AbstractVector, roku::AbstractVector,
                                 rkbo::AbstractVector, cl::AbstractVector,
                                 ros::AbstractVector{<:OptimisationRiskMeasure},
                                 X::AbstractMatrix, fees::Union{Nothing, <:Fees})
    crisk = typemin(eltype(X))
    for ro ∈ ros
        unitary_expected_risks!(wk, roku, ro, X, fees)
        rkbo[cl] .= inv.(view(roku, cl))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk_i = ro.settings.scale * expected_risk(ro, rkbo, X, fees)
        if crisk_i > crisk
            crisk = crisk_i
        end
    end
    return crisk
end
function herc_scalarised_risk_o!(::MaxScalariser, wk::AbstractVector, roku::AbstractMatrix,
                                 rkbo::AbstractVector, cl::AbstractVector,
                                 ros::AbstractVector{<:OptimisationRiskMeasure},
                                 X::AbstractMatrix, fees::Union{Nothing, <:Fees})
    crisk = typemin(eltype(X))
    for (i, ro) ∈ pairs(ros)
        rkbo[cl] .= inv.(view(roku, cl, i))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk_i = ro.settings.scale * expected_risk(ro, rkbo, X, fees)
        if crisk_i > crisk
            crisk = crisk_i
        end
    end
    return crisk
end
function herc_scalarised_risk_o!(sce::LogSumExpScalariser, wk::AbstractVector,
                                 roku::AbstractVector, rkbo::AbstractVector,
                                 cl::AbstractVector,
                                 ros::AbstractVector{<:OptimisationRiskMeasure},
                                 X::AbstractMatrix, fees::Union{Nothing, <:Fees})
    crisk = zero(eltype(X))
    for ro ∈ ros
        unitary_expected_risks!(wk, roku, ro, X, fees)
        rkbo[cl] .= inv.(view(roku, cl))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk += ro.settings.scale * sce.gamma * expected_risk(ro, rkbo, X, fees)
    end
    return log(exp(crisk)) / sce.gamma
end
function herc_scalarised_risk_o!(sce::LogSumExpScalariser, ::AbstractVector,
                                 roku::AbstractMatrix, rkbo::AbstractVector,
                                 cl::AbstractVector,
                                 ros::AbstractVector{<:OptimisationRiskMeasure},
                                 X::AbstractMatrix, fees::Union{Nothing, <:Fees})
    crisk = zero(eltype(X))
    for (i, ro) ∈ pairs(ros)
        rkbo[cl] .= inv.(view(roku, cl, i))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk += ro.settings.scale * sce.gamma * expected_risk(ro, rkbo, X, fees)
    end
    return log(exp(crisk)) / sce.gamma
end
function herc_scalarised_risk_i!(::SumScalariser, wk::AbstractVector, riku::AbstractVector,
                                 cl::AbstractVector,
                                 ris::AbstractVector{<:OptimisationRiskMeasure},
                                 X::AbstractMatrix, fees::Union{Nothing, <:Fees},
                                 cache::Bool = false)
    risk = zeros(eltype(X), length(cl), 2)
    for ri ∈ ris
        unitary_expected_risks!(wk, riku, ri, X, fees)
        risk[:, 1] .= inv.(view(riku, cl))
        risk[:, 1] ./= sum(view(risk, :, 1))
        risk[:, 2] .+= ri.settings.scale * view(risk, :, 1)
    end
    return view(risk, :, 2)
end
function herc_scalarised_risk_i!(::SumScalariser, wk::AbstractVector, riku::AbstractMatrix,
                                 cl::AbstractVector,
                                 ris::AbstractVector{<:OptimisationRiskMeasure},
                                 X::AbstractMatrix, fees::Union{Nothing, <:Fees},
                                 cache::Bool = false)
    risk = zeros(eltype(X), length(cl), 2)
    for (i, ri) ∈ pairs(ris)
        unitary_expected_risks!(wk, view(riku, :, i), ri, X, fees)
        risk[:, 1] .= inv.(view(view(riku, :, i), cl))
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
        risk[:, 1] .= ri.settings.scale * view(riku, cl)
        risk_i = sum(view(risk, :, 1))
        if risk_i > risk_t
            risk_t = risk_i
            risk[:, 2] .= inv.(view(risk, :, 1))
            risk[:, 2] .= view(risk, :, 2) / sum(view(risk, :, 2))
        end
    end
    return view(risk, :, 2)
end
function herc_scalarised_risk_i!(::MaxScalariser, wk::AbstractVector, riku::AbstractMatrix,
                                 cl::AbstractVector,
                                 ris::AbstractVector{<:OptimisationRiskMeasure},
                                 X::AbstractMatrix, fees::Union{Nothing, <:Fees})
    risk_t = typemin(eltype(X))
    risk = zeros(eltype(X), length(cl), 2)
    for (i, ri) ∈ pairs(ris)
        unitary_expected_risks!(wk, view(riku, :, i), ri, X, fees)
        risk[:, 1] .= ri.settings.scale * view(riku, cl, i)
        risk_i = sum(view(risk, :, 1))
        if risk_i > risk_t
            risk_t = risk_i
            risk[:, 2] .= inv.(view(risk, :, 1))
            risk[:, 2] .= view(risk, :, 2) / sum(view(risk, :, 2))
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
function herc_scalarised_risk_i!(sce::LogSumExpScalariser, wk::AbstractVector,
                                 riku::AbstractMatrix, cl::AbstractVector,
                                 ris::AbstractVector{<:OptimisationRiskMeasure},
                                 X::AbstractMatrix, fees::Union{Nothing, <:Fees})
    risk = zeros(eltype(X), length(cl), 2)
    for (i, ri) ∈ pairs(ris)
        unitary_expected_risks!(wk, view(riku, :, i), ri, X, fees)
        risk[:, 1] .= inv.(view(riku, cl, i))
        risk[:, 1] ./= sum(view(risk, :, 1))
        risk[:, 2] .+= ri.settings.scale * sce.gamma * view(risk, :, 1)
    end
    return log.(exp.(view(risk, :, 2))) / sce.gamma
end
function herc_risk(hc::HierarchicalEqualRiskContribution{<:Any, <:OptimisationRiskMeasure,
                                                         <:OptimisationRiskMeasure},
                   pr::AbstractPriorResult, cls::AbstractVector)
    ri = risk_measure_factory(hc.ri, pr, hc.opt.slv)
    riku = unitary_expected_risks(ri, pr.X, hc.opt.fees)
    if hc.ri === hc.ro
        ro = ri
        roku = riku
    else
        ro = risk_measure_factory(hc.ro, pr, hc.opt.slv)
        roku = unitary_expected_risks(ro, pr.X, hc.opt.fees)
    end
    rkbo = zeros(eltype(pr.X), size(pr.X, 2))
    rkcl = Vector{eltype(pr.X)}(undef, length(cls))
    w = Vector{eltype(pr.X)}(undef, size(pr.X, 2))
    for (i, cl) ∈ pairs(cls)
        w[cl] .= inv.(view(riku, cl))
        w[cl] ./= sum(view(w, cl))
        rkbo[cl] .= inv.(view(roku, cl))
        rkbo[cl] ./= sum(view(rkbo, cl))
        rkcl[i] = expected_risk(ro, rkbo, pr.X, hc.opt.fees)
        rkbo[cl] .= zero(eltype(pr.X))
    end
    return w, rkcl
end
function herc_risk(hc::HierarchicalEqualRiskContribution{<:Any,
                                                         <:AbstractVector{<:OptimisationRiskMeasure},
                                                         <:AbstractVector{<:OptimisationRiskMeasure}},
                   pr::AbstractPriorResult, cls::AbstractVector)
    ri = risk_measure_factory(hc.ri, Ref(pr), Ref(hc.opt.slv))
    if hc.ri === hc.ro
        ro = ri
        rku = zeros(eltype(pr.X), size(pr.X, 2), length(cls))
    else
        rku = Vector{eltype(pr.X)}(undef, size(pr.X, 2))
        ro = risk_measure_factory(hc.ro, Ref(pr), Ref(hc.opt.slv))
    end
    rkbo = zeros(eltype(pr.X), size(pr.X, 2))
    rkcl = Vector{eltype(pr.X)}(undef, length(cls))
    w = Vector{eltype(pr.X)}(undef, size(pr.X, 2))
    wk = zeros(eltype(pr.X), size(pr.X, 2))
    for (i, cl) ∈ pairs(cls)
        w[cl] .= herc_scalarised_risk_i!(hc.opt.sce, wk, rku, cl, ri, pr.X, hc.opt.fees)
        rkcl[i] = herc_scalarised_risk_o!(hc.opt.sce, wk, rku, rkbo, cl, ro, pr.X,
                                          hc.opt.fees)
        rkbo[cl] .= zero(eltype(pr.X))
    end
    return w, rkcl
end
function herc_risk(hc::HierarchicalEqualRiskContribution{<:Any, <:OptimisationRiskMeasure,
                                                         <:AbstractVector{<:OptimisationRiskMeasure}},
                   pr::AbstractPriorResult, cls::AbstractVector)
    ri = risk_measure_factory(hc.ri, pr, hc.opt.slv)
    riku = unitary_expected_risks(ri, pr.X, hc.opt.fees)
    ro = risk_measure_factory(hc.ro, Ref(pr), Ref(hc.opt.slv))
    roku = Vector{eltype(pr.X)}(undef, size(pr.X, 2))
    rkbo = zeros(eltype(pr.X), size(pr.X, 2))
    rkcl = Vector{eltype(pr.X)}(undef, length(cls))
    w = Vector{eltype(pr.X)}(undef, size(pr.X, 2))
    wk = zeros(eltype(pr.X), size(pr.X, 2))
    for (i, cl) ∈ pairs(cls)
        w[cl] .= inv.(view(riku, cl))
        w[cl] ./= sum(view(w, cl))
        rkcl[i] = herc_scalarised_risk_o!(hc.opt.sce, wk, roku, rkbo, cl, ro, pr.X,
                                          hc.opt.fees)
        rkbo[cl] .= zero(eltype(pr.X))
    end
    return w, rkcl
end
function herc_risk(hc::HierarchicalEqualRiskContribution{<:Any,
                                                         <:AbstractVector{<:OptimisationRiskMeasure},
                                                         <:OptimisationRiskMeasure},
                   pr::AbstractPriorResult, cls::AbstractVector)
    ri = risk_measure_factory(hc.ri, Ref(pr), Ref(hc.opt.slv))
    riku = Vector{eltype(pr.X)}(undef, size(pr.X, 2))
    ro = risk_measure_factory(hc.ro, pr, hc.opt.slv)
    roku = unitary_expected_risks(ro, pr.X, hc.opt.fees)
    rkbo = zeros(eltype(pr.X), size(pr.X, 2))
    rkcl = Vector{eltype(pr.X)}(undef, length(cls))
    w = Vector{eltype(pr.X)}(undef, size(pr.X, 2))
    wk = zeros(eltype(pr.X), size(pr.X, 2))
    for (i, cl) ∈ pairs(cls)
        w[cl] .= herc_scalarised_risk_i!(hc.opt.sce, wk, riku, cl, ri, pr.X, hc.opt.fees)
        rkbo[cl] .= inv.(view(roku, cl))
        rkbo[cl] ./= sum(view(rkbo, cl))
        rkcl[i] = expected_risk(ro, rkbo, pr.X, hc.opt.fees)
        rkbo[cl] .= zero(eltype(pr.X))
    end
    return w, rkcl
end
function optimise!(hc::HierarchicalEqualRiskContribution,
                   rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   branchorder::Symbol = :optimal, kwargs...)
    pr = prior(hc.opt.pe, rd.X, rd.F; dims = dims)
    clr = clusterise(hc.opt.cle, pr.X; dims = dims, branchorder = branchorder)
    idx = cutree(clr.clustering; k = clr.k)
    cls = [findall(x -> x == i, idx) for i ∈ 1:(clr.k)]
    w, rkcl = herc_risk(hc, pr, cls)
    nd = to_tree(clr.clustering)[2]
    hs = [i.height for i ∈ nd]
    nd = nd[sortperm(hs; rev = true)]
    # Treat each cluster as its own portfolio and optimise each one individually.
    # Calculate the weight of each cluster relative to the other clusters.
    lc = Vector{Int}(undef, 0)
    rc = Vector{Int}(undef, 0)
    @inbounds for i ∈ nd[1:(clr.k - 1)]
        if is_leaf(i)
            continue
        end
        # Do this recursively accounting for the dendrogram structure.
        ln = pre_order(i.left)
        rn = pre_order(i.right)
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
        # wcl = ones(eltype(pr.X), clr.k)
        # for i ∈ nd[1:(clr.k - 1)]
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
        empty!(lc)
        empty!(rc)
    end
    wb = weight_bounds_constraints(hc.opt.wb, hc.opt.sets; N = length(w),
                                   strict = hc.opt.strict)
    retcode, w = clustering_optimisation_result(hc.opt.cwf, wb, w / sum(w))
    return HierarchicalOptimisationResult(typeof(HierarchicalEqualRiskContribution), pr, wb,
                                          clr, retcode, w)
end

export HierarchicalEqualRiskContribution
