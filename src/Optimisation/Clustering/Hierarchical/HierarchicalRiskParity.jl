struct HierarchicalRiskParity{T1 <: HierarchicalOptimiser,
                              T2 <: Union{<:OptimisationRiskMeasure,
                                          <:AbstractVector{<:OptimisationRiskMeasure}}}
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
function split_factor_weight_constraints(alpha::Real, wb::WeightBounds, w::AbstractVector,
                                         lc::AbstractVector, rc::AbstractVector)
    alpha = min(sum(view(wb.ub, lc)) / w[lc[1]],
                max(sum(view(wb.lb, lc)) / w[lc[1]], alpha))
    return one(alpha) - min(sum(view(wb.ub, rc)) / w[rc[1]],
                            max(sum(view(wb.lb, rc)) / w[rc[1]], one(alpha) - alpha))
end
function optimise!(hc::HierarchicalRiskParity{<:Any, <:OptimisationRiskMeasure},
                   rd::ReturnsData = ReturnsData(); strict::Bool = false)
    pm = prior(hc.opt.pe, rd.X, rd.F)
    clm = clusterise(hc.opt.cle, pm.X)
    r = risk_measure_factory(hc.r, pm, hc.opt.slv)
    wu = Matrix{eltype(pm.X)}(undef, size(pm.X, 2), 2)
    rku = unitary_expected_risks(r, pm.X, hc.opt.fees)
    wb = weight_bounds_constraints(hc.opt.wb; N = size(pm.X, 2), strict = strict)
    w = ones(eltype(pm.X), size(pm.X, 2))
    items = [clm.clustering.order]
    @inbounds while length(items) > 0
        items = [i[j:k] for i ∈ items
                 for (j, k) ∈ ((1, div(length(i), 2)), (1 + div(length(i), 2), length(i)))
                 if length(i) > 1]
        for i ∈ 1:2:length(items)
            fill!(wu, zero(eltype(pm.X)))
            lc = items[i]
            rc = items[i + 1]
            wu[lc, 1] .= inv.(view(rku, lc))
            wu[lc, 1] ./= sum(view(wu, lc, 1))
            wu[rc, 2] .= inv.(view(rku, rc))
            wu[rc, 2] ./= sum(view(wu, rc, 2))
            lrisk = expected_risk(r, view(wu, :, 1), pm.X, hc.opt.fees)
            rrisk = expected_risk(r, view(wu, :, 2), pm.X, hc.opt.fees)
            # Allocate weight to clusters.
            alpha = one(lrisk) - lrisk / (lrisk + rrisk)
            alpha = split_factor_weight_constraints(alpha, wb, w, lc, rc)
            # Weight constraints.
            w[lc] .*= alpha
            w[rc] .*= one(alpha) - alpha
        end
    end
    return finalise_hierarchical_weights(hc.opt.cwf, wb, w / sum(w))
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
        lrisk += expected_risk(r, view(wu, :, 1), X, fees)
        rrisk += expected_risk(r, view(wu, :, 2), X, fees)
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
        lrisk_i = expected_risk(r, view(wu, :, 1), X, fees)
        rrisk_i = expected_risk(r, view(wu, :, 2), X, fees)
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
                   rd::ReturnsData = ReturnsData(); strict::Bool = false)
    pm = prior(hc.opt.pe, rd.X, rd.F)
    clm = clusterise(hc.opt.cle, pm.X)
    r = risk_measure_factory(hc.r, pm, hc.opt.slv)
    wu = Matrix{eltype(pm.X)}(undef, size(pm.X, 2), 2)
    wk = zeros(eltype(pm.X), size(pm.X, 2))
    rku = Vector{eltype(pm.X)}(undef, size(pm.X, 2))
    wb = weight_bounds_constraints(hc.opt.wb; N = size(pm.X, 2), strict = strict)
    w = ones(eltype(pm.X), size(pm.X, 2))
    items = [clm.clustering.order]
    @inbounds while length(items) > 0
        items = [i[j:k] for i ∈ items
                 for (j, k) ∈ ((1, div(length(i), 2)), (1 + div(length(i), 2), length(i)))
                 if length(i) > 1]
        for i ∈ 1:2:length(items)
            lc = items[i]
            rc = items[i + 1]
            lrisk, rrisk = hrp_scalarised_risk(hc.opt.sce, wu, wk, rku, lc, rc, r, pm.X,
                                               hc.opt.fees)
            # Allocate weight to clusters.
            alpha = one(lrisk) - lrisk / (lrisk + rrisk)
            alpha = split_factor_weight_constraints(alpha, wb, w, lc, rc)
            # Weight constraints.
            w[lc] .*= alpha
            w[rc] .*= one(alpha) - alpha
        end
    end
    return finalise_hierarchical_weights(hc.opt.cwf, wb, w / sum(w))
end

export HierarchicalRiskParity
