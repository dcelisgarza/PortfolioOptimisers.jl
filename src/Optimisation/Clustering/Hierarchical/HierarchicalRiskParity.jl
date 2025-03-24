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
function optimise!(rd::ReturnsData, hc::HierarchicalRiskParity)
    pm = prior(hc.opt.pe, rd.X, rd.F)
    r = risk_measure_factory(hc.r; prior = pm, solvers = hc.opt.slv)
    clm = clusterise(hc.opt.cle, pm.X)
    rku = unitary_expected_risks(r, pm.X, hc.opt.fees, hc.opt.sce)
    w = ones(eltype(pm.X), size(pm.X, 2))
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
            lrisk = expected_risk(r, view(wu, :, 1), pm.X, hc.opt.fees, hc.opt.sce)
            rrisk = expected_risk(r, view(wu, :, 2), pm.X, hc.opt.fees, hc.opt.sce)
            # Allocate weight to clusters.
            alpha = one(lrisk) - lrisk / (lrisk + rrisk)
            # Weight constraints.
            w[lc] *= alpha
            w[rc] *= one(alpha) - alpha
        end
    end
    return w / sum(w)
end
export HierarchicalRiskParity, optimise!