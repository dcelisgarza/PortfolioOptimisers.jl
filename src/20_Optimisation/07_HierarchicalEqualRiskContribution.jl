struct HierarchicalEqualRiskContribution{T1, T2, T3, T4, T5, T6, T7} <:
       ClusteringOptimisationEstimator
    opt::T1
    ri::T2
    ro::T3
    scai::T4
    scao::T5
    ex::T6
    fb::T7
    function HierarchicalEqualRiskContribution(opt::HierarchicalOptimiser,
                                               ri::OptRM_VecOptRM, ro::OptRM_VecOptRM,
                                               scai::Scalariser, scao::Scalariser,
                                               ex::FLoops.Transducers.Executor,
                                               fb::Option{<:OptimisationEstimator})
        if isa(ri, AbstractVector)
            @argcheck(!isempty(ri))
        end
        if isa(ro, AbstractVector)
            @argcheck(!isempty(ro))
        end
        return new{typeof(opt), typeof(ri), typeof(ro), typeof(scai), typeof(scao),
                   typeof(ex), typeof(fb)}(opt, ri, ro, scai, scao, ex, fb)
    end
end
function HierarchicalEqualRiskContribution(;
                                           opt::HierarchicalOptimiser = HierarchicalOptimiser(),
                                           ri::OptRM_VecOptRM = Variance(),
                                           ro::OptRM_VecOptRM = ri,
                                           scai::Scalariser = SumScalariser(),
                                           scao::Scalariser = scai,
                                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                                           fb::Option{<:OptimisationEstimator} = nothing)
    return HierarchicalEqualRiskContribution(opt, ri, ro, scai, scao, ex, fb)
end
function opt_view(hec::HierarchicalEqualRiskContribution, i, X::MatNum)
    X = isa(hec.opt.pe, AbstractPriorResult) ? hec.opt.pe.X : X
    ri = hec.ri
    ro = hec.ro
    if ri === ro
        ri = ro = risk_measure_view(ri, i, X)
    else
        ri = risk_measure_view(ri, i, X)
        ro = risk_measure_view(ro, i, X)
    end
    opt = opt_view(hec.opt, i)
    return HierarchicalEqualRiskContribution(; ri = ri, ro = ro, opt = opt, scai = hec.scai,
                                             scao = hec.scao, ex = hec.ex, fb = hec.fb)
end
function herc_scalarised_risk_o!(::SumScalariser, wk::VecNum, roku::VecNum, rkbo::VecNum,
                                 cl::VecInt, ros::VecOptRM, X::MatNum, fees::Option{<:Fees})
    crisk = zero(eltype(X))
    for ro in ros
        unitary_expected_risks!(wk, roku, ro, X, fees)
        rkbo[cl] .= inv.(view(roku, cl))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk += ro.settings.scale * expected_risk(ro, rkbo, X, fees)
    end
    return crisk
end
function herc_scalarised_risk_o!(::SumScalariser, wk::VecNum, roku::MatNum, rkbo::VecNum,
                                 cl::VecInt, ros::VecOptRM, X::MatNum, fees::Option{<:Fees})
    crisk = zero(eltype(X))
    for (i, ro) in pairs(ros)
        rkbo[cl] .= inv.(view(roku, cl, i))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk += ro.settings.scale * expected_risk(ro, rkbo, X, fees)
    end
    return crisk
end
function herc_scalarised_risk_o!(::MaxScalariser, wk::VecNum, roku::VecNum, rkbo::VecNum,
                                 cl::VecInt, ros::VecOptRM, X::MatNum, fees::Option{<:Fees})
    crisk = typemin(eltype(X))
    for ro in ros
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
function herc_scalarised_risk_o!(::MaxScalariser, wk::VecNum, roku::MatNum, rkbo::VecNum,
                                 cl::VecInt, ros::VecOptRM, X::MatNum, fees::Option{<:Fees})
    crisk = typemin(eltype(X))
    for (i, ro) in pairs(ros)
        rkbo[cl] .= inv.(view(roku, cl, i))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk_i = ro.settings.scale * expected_risk(ro, rkbo, X, fees)
        if crisk_i > crisk
            crisk = crisk_i
        end
    end
    return crisk
end
function herc_scalarised_risk_o!(sca::LogSumExpScalariser, wk::VecNum, roku::VecNum,
                                 rkbo::VecNum, cl::VecInt, ros::VecOptRM, X::MatNum,
                                 fees::Option{<:Fees})
    crisk = Vector{eltype(X)}(undef, length(ros))
    for (i, ro) in enumerate(ros)
        unitary_expected_risks!(wk, roku, ro, X, fees)
        rkbo[cl] .= inv.(view(roku, cl))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk[i] = ro.settings.scale * sca.gamma * expected_risk(ro, rkbo, X, fees)
    end
    return logsumexp(crisk) / sca.gamma
end
function herc_scalarised_risk_o!(sca::LogSumExpScalariser, ::VecNum, roku::MatNum,
                                 rkbo::VecNum, cl::VecInt, ros::VecOptRM, X::MatNum,
                                 fees::Option{<:Fees})
    crisk = Vector{eltype(X)}(undef, length(ros))
    for (i, ro) in enumerate(ros)
        rkbo[cl] .= inv.(view(roku, cl, i))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk[i] = ro.settings.scale * sca.gamma * expected_risk(ro, rkbo, X, fees)
    end
    return logsumexp(crisk) / sca.gamma
end
function herc_scalarised_risk_i!(::SumScalariser, wk::VecNum, riku::VecNum, cl::VecInt,
                                 ris::VecOptRM, X::MatNum, fees::Option{<:Fees})
    risk = zeros(eltype(X), length(cl), 2)
    for ri in ris
        unitary_expected_risks!(wk, riku, ri, X, fees)
        risk[:, 1] .= inv.(view(riku, cl))
        risk[:, 1] ./= sum(view(risk, :, 1))
        risk[:, 2] += ri.settings.scale * view(risk, :, 1)
    end
    return view(risk, :, 2)
end
function herc_scalarised_risk_i!(::SumScalariser, wk::VecNum, riku::MatNum, cl::VecInt,
                                 ris::VecOptRM, X::MatNum, fees::Option{<:Fees})
    risk = zeros(eltype(X), length(cl), 2)
    for (i, ri) in pairs(ris)
        unitary_expected_risks!(wk, view(riku, :, i), ri, X, fees)
        risk[:, 1] .= inv.(view(view(riku, :, i), cl))
        risk[:, 1] ./= sum(view(risk, :, 1))
        risk[:, 2] += ri.settings.scale * view(risk, :, 1)
    end
    return view(risk, :, 2)
end
function herc_scalarised_risk_i!(::MaxScalariser, wk::VecNum, riku::VecNum, cl::VecInt,
                                 ris::VecOptRM, X::MatNum, fees::Option{<:Fees})
    risk_t = typemin(eltype(X))
    risk = zeros(eltype(X), length(cl), 2)
    for ri in ris
        unitary_expected_risks!(wk, riku, ri, X, fees)
        risk[:, 1] = ri.settings.scale * view(riku, cl)
        risk_i = sum(view(risk, :, 1))
        if risk_i > risk_t
            risk_t = risk_i
            risk[:, 2] .= inv.(view(risk, :, 1))
            risk[:, 2] = view(risk, :, 2) / sum(view(risk, :, 2))
        end
    end
    return view(risk, :, 2)
end
function herc_scalarised_risk_i!(::MaxScalariser, wk::VecNum, riku::MatNum, cl::VecInt,
                                 ris::VecOptRM, X::MatNum, fees::Option{<:Fees})
    risk_t = typemin(eltype(X))
    risk = zeros(eltype(X), length(cl), 2)
    for (i, ri) in pairs(ris)
        unitary_expected_risks!(wk, view(riku, :, i), ri, X, fees)
        risk[:, 1] = ri.settings.scale * view(riku, cl, i)
        risk_i = sum(view(risk, :, 1))
        if risk_i > risk_t
            risk_t = risk_i
            risk[:, 2] .= inv.(view(risk, :, 1))
            risk[:, 2] = view(risk, :, 2) / sum(view(risk, :, 2))
        end
    end
    return view(risk, :, 2)
end
function herc_scalarised_risk_i!(sca::LogSumExpScalariser, wk::VecNum, riku::VecNum,
                                 cl::VecInt, ris::VecOptRM, X::MatNum, fees::Option{<:Fees})
    risk = zeros(eltype(X), length(cl), 2)
    for ri in ris
        unitary_expected_risks!(wk, riku, ri, X, fees)
        risk[:, 1] .= inv.(view(riku, cl))
        risk[:, 1] ./= sum(view(risk, :, 1))
        risk[:, 2] += ri.settings.scale * sca.gamma * view(risk, :, 1)
    end
    return log.(exp.(view(risk, :, 2))) / sca.gamma
end
function herc_scalarised_risk_i!(sca::LogSumExpScalariser, wk::VecNum, riku::MatNum,
                                 cl::VecInt, ris::VecOptRM, X::MatNum, fees::Option{<:Fees})
    risk = zeros(eltype(X), length(cl), 2)
    for (i, ri) in pairs(ris)
        unitary_expected_risks!(wk, view(riku, :, i), ri, X, fees)
        risk[:, 1] .= inv.(view(riku, cl, i))
        risk[:, 1] ./= sum(view(risk, :, 1))
        risk[:, 2] += ri.settings.scale * sca.gamma * view(risk, :, 1)
    end
    return log.(exp.(view(risk, :, 2))) / sca.gamma
end
function herc_risk(hec::HierarchicalEqualRiskContribution{<:Any, <:OptimisationRiskMeasure,
                                                          <:OptimisationRiskMeasure, <:Any,
                                                          <:Any, <:FLoops.SequentialEx},
                   pr::AbstractPriorResult, cls::VecVecInt)
    ri = factory(hec.ri, pr, hec.opt.slv)
    fees = fees_constraints(hec.opt.fees, hec.opt.sets; strict = hec.opt.strict,
                            datatype = eltype(pr.X))
    riku = unitary_expected_risks(ri, pr.X, fees)
    if hec.ri === hec.ro
        ro = ri
        roku = riku
    else
        ro = factory(hec.ro, pr, hec.opt.slv)
        roku = unitary_expected_risks(ro, pr.X, fees)
    end
    rkbo = zeros(eltype(pr.X), size(pr.X, 2))
    rkcl = Vector{eltype(pr.X)}(undef, length(cls))
    w = Vector{eltype(pr.X)}(undef, size(pr.X, 2))
    let
        roku_i, ro_i = roku, ro
        FLoops.@floop hec.ex for (i, cl) in pairs(cls)
            w[cl] .= inv.(view(riku, cl))
            w[cl] ./= sum(view(w, cl))
            rkbo[cl] .= inv.(view(roku_i, cl))
            rkbo[cl] ./= sum(view(rkbo, cl))
            rkcl[i] = expected_risk(ro_i, rkbo, pr.X, fees)
            rkbo[cl] .= zero(eltype(pr.X))
        end
    end
    return w, rkcl, fees
end
function herc_risk(hec::HierarchicalEqualRiskContribution{<:Any, <:OptimisationRiskMeasure,
                                                          <:OptimisationRiskMeasure, <:Any,
                                                          <:Any,
                                                          <:FLoops.Transducers.Executor},
                   pr::AbstractPriorResult, cls::VecVecInt)
    ri = factory(hec.ri, pr, hec.opt.slv)
    fees = fees_constraints(hec.opt.fees, hec.opt.sets; strict = hec.opt.strict,
                            datatype = eltype(pr.X))
    riku = unitary_expected_risks(ri, pr.X, fees)
    if hec.ri === hec.ro
        ro = ri
        roku = riku
    else
        ro = factory(hec.ro, pr, hec.opt.slv)
        roku = unitary_expected_risks(ro, pr.X, fees)
    end
    Nc = length(cls)
    rkbo = zeros(eltype(pr.X), size(pr.X, 2), Nc)
    rkcl = Vector{eltype(pr.X)}(undef, Nc)
    w = Vector{eltype(pr.X)}(undef, size(pr.X, 2))
    let
        roku_i, ro_i = roku, ro
        FLoops.@floop hec.ex for (i, cl) in pairs(cls)
            w[cl] .= inv.(view(riku, cl))
            w[cl] ./= sum(view(w, cl))
            rkbo[cl, i] .= inv.(view(roku_i, cl))
            rkbo[cl, i] ./= sum(view(rkbo, cl, i))
            rkcl[i] = expected_risk(ro_i, view(rkbo, :, i), pr.X, fees)
        end
    end
    return w, rkcl, fees
end
function herc_risk(hec::HierarchicalEqualRiskContribution{<:Any, <:VecOptRM, <:VecOptRM,
                                                          <:Any, <:Any,
                                                          <:FLoops.SequentialEx},
                   pr::AbstractPriorResult, cls::VecVecInt)
    ri = factory(hec.ri, pr, hec.opt.slv)
    if hec.ri === hec.ro
        ro = ri
        rku = zeros(eltype(pr.X), size(pr.X, 2), length(ri))
    else
        ro = factory(hec.ro, pr, hec.opt.slv)
        rku = Vector{eltype(pr.X)}(undef, size(pr.X, 2))
    end
    rkcl = Vector{eltype(pr.X)}(undef, length(cls))
    w = Vector{eltype(pr.X)}(undef, size(pr.X, 2))
    wk = zeros(eltype(pr.X), size(pr.X, 2))
    rkbo = zeros(eltype(pr.X), size(pr.X, 2))
    fees = fees_constraints(hec.opt.fees, hec.opt.sets; strict = hec.opt.strict,
                            datatype = eltype(pr.X))
    let
        rku_i, ro_i = rku, ro
        FLoops.@floop hec.ex for (i, cl) in pairs(cls)
            w[cl] = herc_scalarised_risk_i!(hec.scai, wk, rku_i, cl, ri, pr.X, fees)
            rkcl[i] = herc_scalarised_risk_o!(hec.scao, wk, rku_i, rkbo, cl, ro_i, pr.X,
                                              fees)
            rkbo[cl] .= zero(eltype(pr.X))
        end
    end
    return w, rkcl, fees
end
function herc_risk(hec::HierarchicalEqualRiskContribution{<:Any, <:VecOptRM, <:VecOptRM,
                                                          <:Any, <:Any,
                                                          <:FLoops.Transducers.Executor},
                   pr::AbstractPriorResult, cls::VecVecInt)
    ri = factory(hec.ri, pr, hec.opt.slv)
    if hec.ri === hec.ro
        ro = ri
    else
        ro = factory(hec.ro, pr, hec.opt.slv)
    end
    Nc = length(cls)
    rkcl = Vector{eltype(pr.X)}(undef, Nc)
    w = Vector{eltype(pr.X)}(undef, size(pr.X, 2))
    rku = Matrix{eltype(pr.X)}(undef, size(pr.X, 2), Nc)
    wk = zeros(eltype(pr.X), size(pr.X, 2), Nc)
    rkbo = zeros(eltype(pr.X), size(pr.X, 2), Nc)
    fees = fees_constraints(hec.opt.fees, hec.opt.sets; strict = hec.opt.strict,
                            datatype = eltype(pr.X))
    let
        ro_i = ro
        FLoops.@floop hec.ex for (i, cl) in pairs(cls)
            w[cl] = herc_scalarised_risk_i!(hec.scai, view(wk, :, i), view(rku, :, i), cl,
                                            ri, pr.X, fees)
            rkcl[i] = herc_scalarised_risk_o!(hec.scao, view(wk, :, i), view(rku, :, i),
                                              view(rkbo, :, i), cl, ro_i, pr.X, fees)
        end
    end
    return w, rkcl, fees
end
function herc_risk(hec::HierarchicalEqualRiskContribution{<:Any, <:OptimisationRiskMeasure,
                                                          <:VecOptRM, <:Any, <:Any,
                                                          <:FLoops.SequentialEx},
                   pr::AbstractPriorResult, cls::VecVecInt)
    ri = factory(hec.ri, pr, hec.opt.slv)
    fees = fees_constraints(hec.opt.fees, hec.opt.sets; strict = hec.opt.strict,
                            datatype = eltype(pr.X))
    riku = unitary_expected_risks(ri, pr.X, fees)
    ro = factory(hec.ro, pr, hec.opt.slv)
    rkcl = Vector{eltype(pr.X)}(undef, length(cls))
    w = Vector{eltype(pr.X)}(undef, size(pr.X, 2))
    roku = Vector{eltype(pr.X)}(undef, size(pr.X, 2))
    wk = zeros(eltype(pr.X), size(pr.X, 2))
    rkbo = zeros(eltype(pr.X), size(pr.X, 2))
    FLoops.@floop hec.ex for (i, cl) in pairs(cls)
        w[cl] .= inv.(view(riku, cl))
        w[cl] ./= sum(view(w, cl))
        rkcl[i] = herc_scalarised_risk_o!(hec.scao, wk, roku, rkbo, cl, ro, pr.X, fees)
        rkbo[cl] .= zero(eltype(pr.X))
    end
    return w, rkcl, fees
end
function herc_risk(hec::HierarchicalEqualRiskContribution{<:Any, <:OptimisationRiskMeasure,
                                                          <:VecOptRM, <:Any, <:Any,
                                                          <:FLoops.Transducers.Executor},
                   pr::AbstractPriorResult, cls::VecVecInt)
    ri = factory(hec.ri, pr, hec.opt.slv)
    fees = fees_constraints(hec.opt.fees, hec.opt.sets; strict = hec.opt.strict,
                            datatype = eltype(pr.X))
    riku = unitary_expected_risks(ri, pr.X, fees)
    ro = factory(hec.ro, pr, hec.opt.slv)
    Nc = length(cls)
    rkcl = Vector{eltype(pr.X)}(undef, Nc)
    w = Vector{eltype(pr.X)}(undef, size(pr.X, 2))
    roku = Matrix{eltype(pr.X)}(undef, size(pr.X, 2), Nc)
    wk = zeros(eltype(pr.X), size(pr.X, 2), Nc)
    rkbo = zeros(eltype(pr.X), size(pr.X, 2), Nc)
    FLoops.@floop hec.ex for (i, cl) in pairs(cls)
        w[cl] .= inv.(view(riku, cl))
        w[cl] ./= sum(view(w, cl))
        rkcl[i] = herc_scalarised_risk_o!(hec.scao, view(wk, :, i), view(roku, :, i),
                                          view(rkbo, :, i), cl, ro, pr.X, fees)
    end
    return w, rkcl, fees
end
function herc_risk(hec::HierarchicalEqualRiskContribution{<:Any, <:VecOptRM,
                                                          <:OptimisationRiskMeasure, <:Any,
                                                          <:Any, <:FLoops.SequentialEx},
                   pr::AbstractPriorResult, cls::VecVecInt)
    ri = factory(hec.ri, pr, hec.opt.slv)
    ro = factory(hec.ro, pr, hec.opt.slv)
    fees = fees_constraints(hec.opt.fees, hec.opt.sets; strict = hec.opt.strict,
                            datatype = eltype(pr.X))
    roku = unitary_expected_risks(ro, pr.X, fees)
    rkcl = Vector{eltype(pr.X)}(undef, length(cls))
    w = Vector{eltype(pr.X)}(undef, size(pr.X, 2))
    wk = zeros(eltype(pr.X), size(pr.X, 2))
    riku = Vector{eltype(pr.X)}(undef, size(pr.X, 2))
    rkbo = zeros(eltype(pr.X), size(pr.X, 2))
    FLoops.@floop hec.ex for (i, cl) in pairs(cls)
        w[cl] = herc_scalarised_risk_i!(hec.scai, wk, riku, cl, ri, pr.X, fees)
        rkbo[cl] .= inv.(view(roku, cl))
        rkbo[cl] ./= sum(view(rkbo, cl))
        rkcl[i] = expected_risk(ro, rkbo, pr.X, fees)
        rkbo[cl] .= zero(eltype(pr.X))
    end
    return w, rkcl, fees
end
function herc_risk(hec::HierarchicalEqualRiskContribution{<:Any, <:VecOptRM,
                                                          <:OptimisationRiskMeasure, <:Any,
                                                          <:Any,
                                                          <:FLoops.Transducers.Executor},
                   pr::AbstractPriorResult, cls::VecVecInt)
    ri = factory(hec.ri, pr, hec.opt.slv)
    ro = factory(hec.ro, pr, hec.opt.slv)
    fees = fees_constraints(hec.opt.fees, hec.opt.sets; strict = hec.opt.strict,
                            datatype = eltype(pr.X))
    roku = unitary_expected_risks(ro, pr.X, fees)
    Nc = length(cls)
    rkcl = Vector{eltype(pr.X)}(undef, Nc)
    w = Vector{eltype(pr.X)}(undef, size(pr.X, 2))
    wk = zeros(eltype(pr.X), size(pr.X, 2), Nc)
    riku = Matrix{eltype(pr.X)}(undef, size(pr.X, 2), Nc)
    rkbo = zeros(eltype(pr.X), size(pr.X, 2), Nc)
    FLoops.@floop hec.ex for (i, cl) in pairs(cls)
        w[cl] = herc_scalarised_risk_i!(hec.scai, view(wk, :, i), view(riku, :, i), cl, ri,
                                        pr.X, fees)
        rkbo[cl, i] .= inv.(view(roku, cl))
        rkbo[cl, i] ./= sum(view(rkbo, cl, i))
        rkcl[i] = expected_risk(ro, view(rkbo, :, i), pr.X, fees)
    end
    return w, rkcl, fees
end
function _optimise(hec::HierarchicalEqualRiskContribution,
                   rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   branchorder::Symbol = :optimal, kwargs...)
    pr = prior(hec.opt.pe, rd; dims = dims)
    clr = clusterise(hec.opt.cle, pr.X; iv = rd.iv, ivpa = rd.ivpa, dims = dims,
                     branchorder = branchorder)
    idx = Clustering.cutree(clr.clustering; k = clr.k)
    cls = [findall(x -> x == i, idx) for i in 1:(clr.k)]
    w, rkcl, fees = herc_risk(hec, pr, cls)
    nd = to_tree(clr.clustering)[2]
    hs = [i.height for i in nd]
    nd = nd[sortperm(hs; rev = true)]
    # Treat each cluster as its own portfolio and optimise each one individually.
    # Calculate the weight of each cluster relative to the other clusters.
    lc = Vector{Int}(undef, 0)
    rc = Vector{Int}(undef, 0)
    for i in nd[1:(clr.k - 1)]
        if is_leaf(i)
            continue
        end
        # Do this recursively accounting for the dendrogram structure.
        ln = pre_order(i.left)
        rn = pre_order(i.right)
        for (i, cl) in pairs(cls)
            if issubset(cl, ln)
                push!(lc, i)
            elseif issubset(cl, rn)
                push!(rc, i)
            end
        end
        # Cluster risk contributions.
        lrisk = sum(view(rkcl, lc))
        rrisk = sum(view(rkcl, rc))
        risk = lrisk + rrisk
        alpha = one(lrisk) - lrisk / risk
        # Asset risk contribution times cluster risk contribution.
        w[ln] *= alpha
        w[rn] *= one(alpha) - alpha
        empty!(lc)
        empty!(rc)
    end
    wb = weight_bounds_constraints(hec.opt.wb, hec.opt.sets; N = length(w),
                                   strict = hec.opt.strict, datatype = eltype(pr.X))
    retcode, w = clustering_optimisation_result(hec.opt.cwf, wb, w / sum(w))
    return HierarchicalOptimisation(typeof(hec), pr, fees, wb, clr, retcode, w, nothing)
end
function optimise(hec::HierarchicalEqualRiskContribution{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                         <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  branchorder::Symbol = :optimal, kwargs...)
    return _optimise(hec, rd; dims = dims, branchorder = branchorder, kwargs...)
end

export HierarchicalEqualRiskContribution
