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
function optimise!(hc::HierarchicalRiskParity, rd::ReturnsData = ReturnsData();
                   strict::Bool = false)
    pm = prior(hc.opt.pe, rd.X, rd.F)
    clm = clusterise(hc.opt.cle, pm.X)
    r = risk_measure_factory(hc.r; prior = pm, solvers = hc.opt.slv)
    rku = unitary_expected_risks(r, pm.X; fees = hc.opt.fees, scalariser = hc.opt.sce)
    w = ones(eltype(pm.X), size(pm.X, 2))
    wb = weight_bounds_constraints(hc.opt.wb; N = length(w), strict = strict)
    wu = Matrix{eltype(pm.X)}(undef, size(pm.X, 2), 2)
    items = [clm.clustering.order]
    @inbounds while length(items) > 0
        items = [i[j:k] for i ∈ items
                 for (j, k) ∈ ((1, div(length(i), 2)), (1 + div(length(i), 2), length(i)))
                 if length(i) > 1]
        for i ∈ 1:2:length(items)
            fill!(wu, zero(eltype(pm.X)))
            lc = items[i]
            rc = items[i + 1]
            rku_lc = inv.(view(rku, lc))
            rku_rc = inv.(view(rku, rc))
            wu[lc, 1] = rku_lc / sum(rku_lc)
            wu[rc, 2] = rku_rc / sum(rku_rc)
            lrisk = expected_risk(r, view(wu, :, 1), pm.X; fees = hc.opt.fees,
                                  scalariser = hc.opt.sce)
            rrisk = expected_risk(r, view(wu, :, 2), pm.X; fees = hc.opt.fees,
                                  scalariser = hc.opt.sce)
            # Allocate weight to clusters.
            alpha = one(lrisk) - lrisk / (lrisk + rrisk)
            alpha = split_factor_weight_constraints(alpha, wb, w, lc, rc)
            # Weight constraints.
            w[lc] *= alpha
            w[rc] *= one(alpha) - alpha
        end
    end
    return finalise_hierarchical_weights(hc.opt.cwf, wb, w / sum(w))
end

export HierarchicalRiskParity
