struct HierarchicalRiskParity{T1 <: HierarchicalOptimiser,
                              T2 <: Union{<:OptimisationRiskMeasure,
                                          <:AbstractVector{<:OptimisationRiskMeasure}}} <:
       ClusteringOptimisationEstimator
    opt::T1
    r::T2
end
function HierarchicalRiskParity(; opt::HierarchicalOptimiser = HierarchicalOptimiser(),
                                r::Union{<:OptimisationRiskMeasure,
                                         <:AbstractVector{<:OptimisationRiskMeasure}} = Variance())
    if isa(r, AbstractVector)
        @smart_assert(!isempty(r))
    end
    return HierarchicalRiskParity{typeof(opt), typeof(r)}(opt, r)
end
function opt_view(hc::HierarchicalRiskParity, i::AbstractVector, X::AbstractMatrix)
    r = risk_measure_view(hc.r, i, X)
    opt = opt_view(hc.opt, i)
    return HierarchicalRiskParity(; r = r, opt = opt)
end
function split_factor_weight_constraints(alpha::Real, wb::WeightBoundsResult,
                                         w::AbstractVector, lc::AbstractVector,
                                         rc::AbstractVector)
    lb = wb.lb
    ub = wb.ub
    alpha = min(sum(view(ub, lc)) / w[lc[1]], max(sum(view(lb, lc)) / w[lc[1]], alpha))
    return one(alpha) - min(sum(view(ub, rc)) / w[rc[1]],
                            max(sum(view(lb, rc)) / w[rc[1]], one(alpha) - alpha))
end
function optimise!(hc::HierarchicalRiskParity{<:Any, <:OptimisationRiskMeasure},
                   rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
    pr = prior(hc.opt.pe, rd.X, rd.F; dims = dims)
    clr = clusterise(hc.opt.cle, pr.X; dims = dims)
    r = factory(hc.r, pr, hc.opt.slv)
    wu = Matrix{eltype(pr.X)}(undef, size(pr.X, 2), 2)
    rku = unitary_expected_risks(r, pr.X, hc.opt.fees)
    wb = weight_bounds_constraints(hc.opt.wb, hc.opt.sets; N = size(pr.X, 2),
                                   strict = hc.opt.strict)
    w = ones(eltype(pr.X), size(pr.X, 2))
    items = [clr.clustering.order]
    @inbounds while length(items) > 0
        items = [i[j:k] for i ∈ items
                 for (j, k) ∈ ((1, div(length(i), 2)), (1 + div(length(i), 2), length(i)))
                 if length(i) > 1]
        for i ∈ 1:2:length(items)
            fill!(wu, zero(eltype(pr.X)))
            lc = items[i]
            rc = items[i + 1]
            wu[lc, 1] .= inv.(view(rku, lc))
            wu[lc, 1] ./= sum(view(wu, lc, 1))
            wu[rc, 2] .= inv.(view(rku, rc))
            wu[rc, 2] ./= sum(view(wu, rc, 2))
            lrisk = expected_risk(r, view(wu, :, 1), pr.X, hc.opt.fees)
            rrisk = expected_risk(r, view(wu, :, 2), pr.X, hc.opt.fees)
            # Allocate weight to clusters.
            alpha = one(lrisk) - lrisk / (lrisk + rrisk)
            alpha = split_factor_weight_constraints(alpha, wb, w, lc, rc)
            # Weight constraints.
            w[lc] .*= alpha
            w[rc] .*= one(alpha) - alpha
        end
    end
    retcode, w = clustering_optimisation_result(hc.opt.cwf, wb, w / sum(w))
    return HierarchicalOptimisationResult(typeof(HierarchicalRiskParity), pr, wb, clr,
                                          retcode, w)
end
function hrp_scalarised_risk(::SumScalariser, wu::AbstractMatrix, wk::AbstractVector,
                             rku::AbstractVector, lc::AbstractVector, rc::AbstractVector,
                             rs::AbstractVector{<:OptimisationRiskMeasure},
                             X::AbstractMatrix, fees::Union{Nothing, <:Fees})
    lrisk = zero(eltype(X))
    rrisk = zero(eltype(X))
    for r ∈ rs
        fill!(wu, zero(eltype(X)))
        unitary_expected_risks!(wk, rku, r, X, fees)
        wu[lc, 1] .= inv.(view(rku, lc))
        wu[lc, 1] ./= sum(view(wu, lc, 1))
        wu[rc, 2] .= inv.(view(rku, rc))
        wu[rc, 2] ./= sum(view(wu, rc, 2))
        lrisk += expected_risk(r, view(wu, :, 1), X, fees) * r.settings.scale
        rrisk += expected_risk(r, view(wu, :, 2), X, fees) * r.settings.scale
    end
    return lrisk, rrisk
end
function hrp_scalarised_risk(::MaxScalariser, wu::AbstractMatrix, wk::AbstractVector,
                             rku::AbstractVector, lc::AbstractVector, rc::AbstractVector,
                             rs::AbstractVector{<:OptimisationRiskMeasure},
                             X::AbstractMatrix, fees::Union{Nothing, <:Fees})
    lrisk = zero(eltype(X))
    rrisk = zero(eltype(X))
    trisk = typemin(eltype(X))
    for r ∈ rs
        fill!(wu, zero(eltype(X)))
        unitary_expected_risks!(wk, rku, r, X, fees)
        wu[lc, 1] .= inv.(view(rku, lc))
        wu[lc, 1] ./= sum(view(wu, lc, 1))
        wu[rc, 2] .= inv.(view(rku, rc))
        wu[rc, 2] ./= sum(view(wu, rc, 2))
        lrisk_i = expected_risk(r, view(wu, :, 1), X, fees) * r.settings.scale
        rrisk_i = expected_risk(r, view(wu, :, 2), X, fees) * r.settings.scale
        trisk_i = lrisk_i + rrisk_i
        if trisk_i > trisk
            lrisk = lrisk_i
            rrisk = rrisk_i
            trisk = trisk_i
        end
    end
    return lrisk, rrisk
end
function hrp_scalarised_risk(sce::LogSumExpScalariser, wu::AbstractMatrix,
                             wk::AbstractVector, rku::AbstractVector, lc::AbstractVector,
                             rc::AbstractVector,
                             rs::AbstractVector{<:OptimisationRiskMeasure},
                             X::AbstractMatrix, fees::Union{Nothing, <:Fees})
    lrisk = zero(eltype(X))
    rrisk = zero(eltype(X))
    for r ∈ rs
        fill!(wu, zero(eltype(X)))
        unitary_expected_risks!(wk, rku, r, X, fees)
        wu[lc, 1] .= inv.(view(rku, lc))
        wu[lc, 1] ./= sum(view(wu, lc, 1))
        wu[rc, 2] .= inv.(view(rku, rc))
        wu[rc, 2] ./= sum(view(wu, rc, 2))
        scale = r.settings.scale * sce.gamma
        lrisk += expected_risk(r, view(wu, :, 1), X, fees) * scale
        rrisk += expected_risk(r, view(wu, :, 2), X, fees) * scale
    end
    return log(exp(lrisk)) / sce.gamma, log(exp(rrisk)) / sce.gamma
end
function optimise!(hc::HierarchicalRiskParity{<:Any,
                                              <:AbstractVector{<:OptimisationRiskMeasure}},
                   rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
    pr = prior(hc.opt.pe, rd.X, rd.F; dims = dims)
    clr = clusterise(hc.opt.cle, pr.X; dims = dims)
    r = factory(hc.r, Ref(pr), Ref(hc.opt.slv))
    wu = Matrix{eltype(pr.X)}(undef, size(pr.X, 2), 2)
    wk = zeros(eltype(pr.X), size(pr.X, 2))
    rku = Vector{eltype(pr.X)}(undef, size(pr.X, 2))
    wb = weight_bounds_constraints(hc.opt.wb, hc.opt.sets; N = size(pr.X, 2),
                                   strict = hc.opt.strict)
    w = ones(eltype(pr.X), size(pr.X, 2))
    items = [clr.clustering.order]
    @inbounds while length(items) > 0
        items = [i[j:k] for i ∈ items
                 for (j, k) ∈ ((1, div(length(i), 2)), (1 + div(length(i), 2), length(i)))
                 if length(i) > 1]
        for i ∈ 1:2:length(items)
            lc = items[i]
            rc = items[i + 1]
            lrisk, rrisk = hrp_scalarised_risk(hc.opt.sce, wu, wk, rku, lc, rc, r, pr.X,
                                               hc.opt.fees)
            # Allocate weight to clusters.
            alpha = one(lrisk) - lrisk / (lrisk + rrisk)
            alpha = split_factor_weight_constraints(alpha, wb, w, lc, rc)
            # Weight constraints.
            w[lc] .*= alpha
            w[rc] .*= one(alpha) - alpha
        end
    end
    retcode, w = clustering_optimisation_result(hc.opt.cwf, wb, w / sum(w))
    return HierarchicalOptimisationResult(typeof(HierarchicalRiskParity), pr, wb, clr,
                                          retcode, w)
end

export HierarchicalRiskParity
