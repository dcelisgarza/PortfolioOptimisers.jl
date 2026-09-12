using PortfolioOptimisers, Test, Clustering, TimeSeries, Dates, StableRNGs, StatsBase,
      LinearAlgebra, Clarabel

const PO = PortfolioOptimisers

#=
This file owns the collapse of an `AssetPanel` onto the synthetic assets a meta-optimiser
builds for its outer problem. Before it, `NestedClustered` and `Stacking` dropped the feature
data and an outer `FeatureDistance` threw; now the outer problem is measured on features
aggregated from its members'. `#807` made the collapse act one Panel Field at a time and
return a panel, so a selector written for the inner problem resolves on the outer one.

The failure mode is the same quiet one the rest of the feature suite guards: a wrong
collapse still produces a finite, symmetric, plausible distance matrix over `k` synthetic
assets, and `HierarchicalRiskParity` consumes the dendrogram's *leaf ordering* rather than
its merges, so it can return byte-identical weights from a wrong matrix. Every end-to-end
test below therefore asserts on the matrix that reached the kernel, through the same
`RecordingDistance` instrument `test_13c_feature_views.jl` uses, and not on the weights.
=#
mutable struct RecordingDistance{T} <: PO.AbstractDistanceEstimator
    de::T
    seen::Vector{Any}
    lock::ReentrantLock
end
RecordingDistance(de) = RecordingDistance(de, Any[], ReentrantLock())
function record!(de::RecordingDistance, Z)
    lock(de.lock) do
        return push!(de.seen, Z)
    end
    return nothing
end
function PO.distance(de::RecordingDistance, ce, X; pr = nothing, rd = nothing, kwargs...)
    record!(de, feature_matrix(de.de, pr, rd, X))
    return PO.distance(de.de, ce, X; pr = pr, rd = rd, kwargs...)
end
function PO.cor_and_dist(de::RecordingDistance, ce, X; pr = nothing, rd = nothing,
                         kwargs...)
    record!(de, feature_matrix(de.de, pr, rd, X))
    return PO.cor_and_dist(de.de, ce, X; pr = pr, rd = rd, kwargs...)
end
PO.distance(de::RecordingDistance, Z; kwargs...) = PO.distance(de.de, Z; kwargs...)
function PO.cor_and_dist(de::RecordingDistance, Z; kwargs...)
    return PO.cor_and_dist(de.de, Z; kwargs...)
end

include(joinpath(@__DIR__, "asset_panel_fixture.jl"))
#=
The collapse is a fact about the panel now, so the arithmetic assertions below drive it
through a one-field panel: a tensor field whose labels are the asset names is the square
case, and one whose labels are anything else is the rectangular one.
=#
function sqpanel(labels, vals)
    return asset_panel([TensorPanelInput(; name = "prox", axis = "asset", labels = labels,
                                         vals = vals)])
end
function rectpanel(labels, vals)
    return asset_panel([TensorPanelInput(; name = "beta", axis = "factor", labels = labels,
                                         vals = vals)])
end
@testset "Collapsing the feature matrix onto synthetic assets" begin
    saw = PO.synthetic_asset_weights
    features_are_assets = PO.features_are_assets
    seq = PO.FLoops.SequentialEx()
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false))
    hopt(de) = HierarchicalOptimiser(; cle = ClustersEstimator(; de = de), slv = slv)
    plain_hrp() = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; slv = slv))

    T, N, K, k = 250, 8, 3, 3
    rng = StableRNG(20260729)
    F = randn(rng, T, K) ./ 50
    Bl = randn(rng, N, K)
    X = F * transpose(Bl) .+ randn(rng, T, N) ./ 500
    nx = ["A$i" for i in 1:N]
    nf = ["F$i" for i in 1:K]
    ts = collect(Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(T - 1)))

    # A rectangular feature matrix, strictly positive so the non-negative-domain metrics
    # stay in play, and a square block-structured one -- the shape a phylogeny or adjacency
    # source produces, and the only one whose feature axis is the asset axis.
    Zr = abs.(randn(StableRNG(11), N, K)) .+ 0.1
    function block_features(rng, N, blocks)
        Z = 0.05 .* abs.(randn(rng, N, N))
        for b in blocks, i in b, j in b
            Z[i, j] += 1.0
        end
        Z .= (Z .+ transpose(Z)) ./ 2
        return Z
    end
    blocks = ([1, 2, 3], [4, 5, 6], [7, 8])
    Zsq = block_features(StableRNG(707), N, blocks)
    # Z3[t, i, j] names its own position, so a mis-indexed observation axis shows up in the
    # values and not merely in the shape.
    Z3 = reshape(Float64.(1:(T * N * K)), T, N, K) ./ 1000
    # The collapse of a bare matrix, driven through the panel that holds it.
    function collapse(Z, sq::Bool, W)
        pnl = if sq
            sqpanel(nx, Z)
        else
            rectpanel([string("c", k) for k in axes(Z, ndims(Z))], Z)
        end
        c = PO.collapse_asset_panel(pnl, W, sq ? nx : nothing)
        return PO.panel_field(c, sq ? "prox" : "beta").vals
    end

    # Long-only weights that sum to one per synthetic asset, and a leveraged/shorting set
    # whose columns do not -- the case an un-normalised collapse inflates.
    Wl = abs.(randn(StableRNG(3), N, k))
    Wl ./= sum(Wl; dims = 1)
    Wlev = randn(StableRNG(5), N, k)
    Wlev[:, 1] .*= 3
    Wlev[1, 2] = -4.0

    @testset "The three collapse shapes" begin
        # Rectangular: only the asset axis contracts, so the label axis survives intact.
        Cr = collapse(Zr, false, Wl)
        @test size(Cr) == (k, K)
        @test Cr ≈ transpose(saw(Wl)) * Zr
        # A long-only, unit-sum weight matrix is already convex, so normalising is exactly
        # the identity here -- which is what makes the leveraged case below the real test.
        @test Cr ≈ transpose(Wl) * Zr

        # Square: two-sided, so the label axis lands on the synthetic universe too.
        Cs = collapse(Zsq, true, Wl)
        @test size(Cs) == (k, k)
        @test Cs ≈ transpose(Cs)
        @test Cs ≈ transpose(saw(Wl)) * Zsq * saw(Wl)
        # The square case is *derived* from the labels, not re-declared, and renaming them
        # after the synthetic assets is what keeps the predicate true one level up.
        csq = PO.collapse_asset_panel(sqpanel(nx, Zsq), Wl, nx)
        fsq = PO.panel_field(csq, "prox")
        @test fsq.labels == ["_$i" for i in 1:k]
        @test isnothing(fsq.groups)
        @test features_are_assets(fsq, ["_$i" for i in 1:k])
        crect = PO.collapse_asset_panel(rectpanel(nf, Zr), Wl, nx)
        @test !features_are_assets(PO.panel_field(crect, "beta"), ["_$i" for i in 1:k])

        # Time-varying: the same arithmetic per observation, observation axis leading.
        C3 = collapse(Z3, false, Wl)
        @test size(C3) == (T, k, K)
        @test all(t -> C3[t, :, :] ≈ transpose(saw(Wl)) * Z3[t, :, :], 1:T)
        Z3sq = Array{Float64}(undef, T, N, N)
        for t in 1:T
            Z3sq[t, :, :] = Zsq
        end
        C3s = collapse(Z3sq, true, Wl)
        @test size(C3s) == (T, k, k)
        @test all(t -> C3s[t, :, :] ≈ Cs, 1:T)

        # A numeric Panel Field stays numeric, and contracts on its one axis.
        num = asset_panel([NumericPanelInput(; name = "mcap", vals = Zr[:, 1])])
        cnum = PO.collapse_asset_panel(num, Wl, nx)
        @test isa(PO.panel_field(cnum, "mcap"), NumericPanelField)
        @test PO.panel_field(cnum, "mcap").vals ≈ transpose(saw(Wl)) * Zr[:, 1]

        # A categorical Panel Field becomes a tensor field of membership fractions: the
        # convex combination of its one-hot block, which is the share of the synthetic
        # asset's weight in each level.
        lvl = ["Fin", "Tech"]
        codes = [1, 2, 1, 2, 1, 2, 1, 2]
        cat = asset_panel([CategoricalPanelInput(; name = "sector", vals = lvl[codes])])
        ccat = PO.collapse_asset_panel(cat, Wl, nx)
        fcat = PO.panel_field(ccat, "sector")
        @test isa(fcat, TensorPanelField)
        @test fcat.axis == "level"
        @test fcat.labels == lvl
        @test size(fcat.vals) == (k, 2)
        @test all(isapprox(sum(fcat.vals[i, :]), 1) for i in 1:k)
        # The value columns of the collapsed panel's Feature Matrix equal the collapse of
        # the original panel's, which is what lets a selector resolve unchanged.
        onehot = Float64[codes[i] == l for i in 1:N, l in 1:2]
        @test fcat.vals ≈ transpose(saw(Wl)) * onehot
        # A selector written for the inner problem resolves on the outer panel.
        @test PO.panel_feature_names(ccat) == ["sector=Fin", "sector=Tech"]

        # A mask collapses as the support of its convex combination.
        msk = trues(N, 1)
        pm = asset_panel([NumericPanelInput(; name = "a", vals = Zr[:, 1])])
        @test isnothing(PO.collapse_panel_mask(nothing, saw(Wl)))
        Wz = copy(Wl)
        Wz[:, 2] .= 0
        m = trues(2, N)
        @test PO.collapse_panel_mask(m, saw(Wz)) == [true false true; true false true]

        # An absent panel stays absent.
        @test isnothing(PO.collapse_asset_panel(nothing, Wl, nx))
    end

    @testset "The collapse is convex, not an un-normalised sum" begin
        #=
        `iv` and `ivpa` are intensive and collapse as convex combinations; features are
        treated the same way. The distinction is invisible for long-only unit-sum weights
        and decisive under leverage or shorting, where a plain weighted sum scales each
        synthetic asset's feature vector by its gross exposure.
        =#
        s = vec(sum(abs.(Wlev); dims = 1))
        @test !all(x -> isapprox(x, 1), s)
        Craw = transpose(abs.(Wlev)) * Zr
        Cvex = collapse(Zr, false, Wlev)
        @test Cvex ≈ Craw ./ s
        # Bounded: every entry of a convex combination lies within the range of the column
        # it combines, whatever the gross exposure. The un-normalised sum does not.
        for j in 1:K
            @test all(minimum(Zr[:, j]) - 1e-12 .<=
                      Cvex[:, j] .<=
                      maximum(Zr[:, j]) + 1e-12)
        end
        @test any(Craw .> maximum(Zr))

        # Under the default metric the normalisation is a mathematical no-op in the
        # rectangular case: it scales one row of the result, and `AngularDist` is invariant
        # to positive row scaling. It is not a no-op in the square case, where the two-sided
        # product rescales the feature columns as well.
        fde = FeatureDistance()
        @test PO.distance(fde, Cvex) ≈ PO.distance(fde, Craw)
        Ssq = collapse(Zsq, true, Wlev)
        Sraw = transpose(abs.(Wlev)) * Zsq * abs.(Wlev)
        @test !isapprox(PO.distance(fde, Ssq), PO.distance(fde, Sraw))
    end

    @testset "A degenerate synthetic asset gets a zero row, not an error" begin
        # The same all-zero weight column already yields a zero returns column, zero `iv`
        # and zero `ivpa` with nothing throwing; the feature matrix must not be the one
        # quantity that hard-fails mid-meta-optimisation.
        Wz = copy(Wl)
        Wz[:, 2] .= 0
        Cz = collapse(Zr, false, Wz)
        @test all(iszero, Cz[2, :])
        @test Cz[[1, 3], :] ≈ collapse(Zr, false, Wl)[[1, 3], :]

        # It then lands on the zero-feature-vector convention the kernel already
        # implements: a finite distance matrix rather than a `NaN` one.
        D = PO.distance(FeatureDistance(), Cz)
        @test all(isfinite, D)
        @test isapprox(D, transpose(D))
        @test iszero(diag(D))
        # Two degenerate assets are declared mutually identical, which is the same
        # convention a zero row gets anywhere else.
        Wzz = copy(Wz)
        Wzz[:, 3] .= 0
        Dzz = PO.distance(FeatureDistance(), collapse(Zr, false, Wzz))
        @test iszero(Dzz[2, 3])
    end

    rd_r = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, ts = ts,
                         pnl = matrix_panel(["z$i" for i in 1:K], Zr))
    rd_sq = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, ts = ts, pnl = sqpanel(nx, Zsq))
    rd_3d = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, ts = ts,
                          pnl = matrix_panel(nf, Z3))

    @testset "prepare_outer_rd collapses, and its arity break is loud" begin
        wi = Wlev
        nb, B, iv, ivpa, pnl, Xb = PO.prepare_outer_rd(rd_r, wi)
        @test PO.panel_feature_names(pnl) == panel_feature_matrix(rd_r.pnl)[1]
        @test panel_feature_matrix(pnl)[2] ≈ collapse(Zr, false, wi)
        @test size(Xb) == (T, k)

        # The square carrier renames its label axis after the synthetic assets, which is
        # what keeps the square case true for the result built from it.
        _, _, _, _, pnl_s, _ = PO.prepare_outer_rd(rd_sq, wi)
        fs = PO.panel_field(pnl_s, "prox")
        @test fs.labels == ["_$i" for i in 1:k]
        @test features_are_assets(fs, ["_$i" for i in 1:k])
        @test fs.vals ≈ collapse(Zsq, true, wi)

        _, _, _, _, pnl_3, _ = PO.prepare_outer_rd(rd_3d, wi)
        @test PO.panel_feature_names(pnl_3) == nf
        @test size(panel_feature_matrix(pnl_3)[2]) == (T, k, K)

        #=
        `predict_outer_returns` is a documented overload point, so extending the tuple is a
        deliberate break. `pnl` is returned *before* `X` precisely so the break bites:
        Julia's destructuring discards trailing values without complaint, so appending it
        would have let a stale overload keep building a feature-less result in silence.
        =#
        stale_nb, stale_B, stale_iv, stale_ivpa, stale_X = PO.prepare_outer_rd(rd_r, wi)
        @test isa(stale_X, AssetPanel)
        @test !isa(stale_X, AbstractMatrix)
    end

    @testset "The outer problem is measured on collapsed features" begin
        # NestedClustered's synthetic assets are its clusters; Stacking's are its inner
        # portfolios. Both reach the outer optimiser through `prepare_outer_rd`.
        mk_st(de) = Stacking(;
                             opti = [plain_hrp(),
                                     HierarchicalEqualRiskContribution(;
                                                                       opt = HierarchicalOptimiser(;
                                                                                                   slv = slv))],
                             opto = HierarchicalRiskParity(; opt = hopt(de)), ex = seq)
        mk_nco(de) = NestedClustered(; cle = ClustersEstimator(; de = FeatureDistance()),
                                     opti = plain_hrp(),
                                     opto = HierarchicalRiskParity(; opt = hopt(de)),
                                     ex = seq)
        # `wi` is the matrix whose columns are the inner solutions laid out on the full
        # universe -- exactly what the outer result was built from. Stacking's inner
        # optimisers all see the whole universe; NestedClustered's see one cluster each.
        stacked_wi(res) = reduce(hcat, [r.w for r in res.resi])
        function clustered_wi(res)
            idx = assignments(res.clr)
            cls = [findall(==(i), idx) for i in 1:(res.clr.k)]
            wi = zeros(N, length(cls))
            for (i, cl) in pairs(cls)
                wi[cl, i] = res.resi[i].w
            end
            return wi
        end
        for (mk, weights_of) in ((mk_st, stacked_wi), (mk_nco, clustered_wi))
            for (rd, Zsrc, sq) in
                ((rd_r, Zr, false), (rd_sq, Zsq, true), (rd_3d, Z3, false))
                ro = RecordingDistance(FeatureDistance())
                res = optimise(mk(ro), rd)
                @test isapprox(sum(res.w), 1)
                @test length(ro.seen) == 1
                Zo = ro.seen[1]
                wi = weights_of(res)
                @test Zo ≈ collapse(Zsrc, sq, wi)
                @test size(Zo, ndims(Zo) - 1) == size(wi, 2)
                @test all(isfinite, Zo)
            end
        end
    end

    @testset "The cross-validation path carries a time-varying collapse" begin
        #=
        Site three is `reconstruct_rd`, inside the fold. Each fold collapses with its own
        weights, so a *static* source becomes genuinely time-varying: constant within a
        fold and different in the next. The row count matches `X` exactly because both are
        built by the same stacking, and the outer default `LastObservation` then reduces it
        to the most recent fold.
        =#
        n_folds = 3
        cvopt = OptimisationCrossValidation(; cv = KFold(; n = n_folds))
        mk_st(de) = Stacking(;
                             opti = [plain_hrp(),
                                     HierarchicalEqualRiskContribution(;
                                                                       opt = HierarchicalOptimiser(;
                                                                                                   slv = slv))],
                             opto = HierarchicalRiskParity(; opt = hopt(de)), cv = cvopt,
                             ex = seq)
        mk_nco(de) = NestedClustered(; cle = ClustersEstimator(; de = FeatureDistance()),
                                     opti = plain_hrp(),
                                     opto = HierarchicalRiskParity(; opt = hopt(de)),
                                     cv = cvopt, ex = seq)
        for mk in (mk_st, mk_nco)
            for (rd, nfeat) in ((rd_r, K), (rd_3d, K))
                ro = RecordingDistance(FeatureDistance())
                res = optimise(mk(ro), rd)
                @test isapprox(sum(res.w), 1)
                Zo = ro.seen[1]
                @test ndims(Zo) == 3
                @test size(Zo, 1) == T
                @test size(Zo, 3) == nfeat
                @test all(isfinite, Zo)
            end

            # A static source is constant within each fold and changes at the boundaries:
            # one distinct feature row per fold, per synthetic asset.
            ro = RecordingDistance(FeatureDistance())
            optimise(mk(ro), rd_r)
            Zo = ro.seen[1]
            for i in axes(Zo, 2)
                @test length(unique(eachrow(Zo[:, i, :]))) == n_folds
            end
        end
    end

    @testset "A square carrier through a fold collapses onto the synthetic universe" begin
        #=
        This testset used to pin the opposite: a square carrier kept the *real* assets as
        its feature axis through a fold, because only one weight vector is in scope inside
        one, and it used to be the fold that collapsed. It is not any more -- the collapse
        happens once at the assembly seam, where every sub-portfolio's weights are in scope
        simultaneously, so the second contraction is available and the outer feature axis is
        the synthetic universe on both paths.

        `Stacking` is the milder half of the defect the recompute closes: it never dropped
        the matrix, it just switched its outer feature axis between the synthetic and the
        real universe depending on whether `cv` was set. That is the assertion below, and
        it is a deliberate behaviour change from `T x N x nassets`.
        =#
        cvopt = OptimisationCrossValidation(; cv = KFold(; n = 3))
        ro = RecordingDistance(FeatureDistance())
        st = Stacking(;
                      opti = [plain_hrp(),
                              HierarchicalEqualRiskContribution(;
                                                                opt = HierarchicalOptimiser(;
                                                                                            slv = slv))],
                      opto = HierarchicalRiskParity(; opt = hopt(ro)), cv = cvopt, ex = seq)
        res = optimise(st, rd_sq)
        @test isapprox(sum(res.w), 1)
        Zo = ro.seen[1]
        @test size(Zo) == (T, 2, 2)
        @test size(Zo, 3) != N
        # Two-sided against a symmetric source, so every observation is symmetric, and the
        # feature axis being the synthetic asset axis is what keeps `features_are_assets`
        # true for a `Stacking` nested inside something that collapses again.
        @test all(t -> Zo[t, :, :] ≈ transpose(Zo[t, :, :]), 1:T)
        @test all(isfinite, Zo)

        #=
        `NestedClustered` is the intersection that used to be dropped outright: its folds
        see cluster-sliced returns, so a fold-side collapse came back with a *different*
        feature axis per cluster and there was nothing to stack them against. The seam has
        the full universe legitimately in scope, so it never faces that mismatch.
        =#
        rn = RecordingDistance(FeatureDistance())
        nco = NestedClustered(; cle = ClustersEstimator(; de = FeatureDistance()),
                              opti = plain_hrp(),
                              opto = HierarchicalRiskParity(; opt = hopt(rn)), cv = cvopt,
                              ex = seq)
        res = optimise(nco, rd_sq)
        @test isapprox(sum(res.w), 1)
        Zn = rn.seen[1]
        @test ndims(Zn) == 3
        @test size(Zn, 1) == T
        @test size(Zn, 2) == size(Zn, 3)
        @test size(Zn, 3) != N
        @test all(t -> Zn[t, :, :] ≈ transpose(Zn[t, :, :]), 1:T)
        @test all(isfinite, Zn)

        # A 3-D square source reaches the same place, and gets genuine per-observation
        # variation inside each fold on top of the per-fold weight variation.
        Z3sq = Array{Float64}(undef, T, N, N)
        for t in 1:T
            Z3sq[t, :, :] = Zsq .* (1 + t / T)
        end
        rd_3dsq = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, ts = ts,
                                pnl = sqpanel(nx, Z3sq))
        r3 = RecordingDistance(FeatureDistance())
        nco3 = NestedClustered(; cle = ClustersEstimator(; de = FeatureDistance()),
                               opti = plain_hrp(),
                               opto = HierarchicalRiskParity(; opt = hopt(r3)), cv = cvopt,
                               ex = seq)
        optimise(nco3, rd_3dsq)
        Z3o = r3.seen[1]
        @test size(Z3o, 1) == T
        @test size(Z3o, 2) == size(Z3o, 3)
        @test length(unique(eachrow(reshape(Z3o, T, :)))) == T
    end

    @testset "cv and non-cv agree: the seam makes the same collapse call" begin
        #=
        The headline property. `cv` is execution control (ADR 0030), so toggling it must
        not change the data the outer problem is measured on. The two paths now share one
        implementation, and this asserts they agree: for every shape, and for both the
        full-universe (`Stacking`) and cluster-sliced (`NestedClustered`) layouts, each
        fold's block of the assembled matrix equals what `prepare_outer_rd` -- the non-`cv`
        path -- returns when handed that same fold's weights and rows.

        `≈`, not bit-for-bit: the seam contracts the whole universe at once where the
        non-`cv` path is called per fold, and gemm reassociation makes the last bits differ.
        =#
        cv = KFold(; n = 3)
        test_idx = PO.split(cv, rd_r).test_idx
        cls = [[1, 2, 3, 4], [5, 6, 7, 8]]
        herc() = HierarchicalEqualRiskContribution(;
                                                   opt = HierarchicalOptimiser(; slv = slv))
        #=
        One prediction vector, reused across all six combinations below. That is itself an
        assertion: `rebuild_returns_result` stacks into *copies* of the first prediction's
        buffers, so it leaves `predictions` untouched and is safe to call repeatedly. It
        used to append into `predictions[1].mrd.X` itself, which silently doubled the
        stacked height on a second call.
        =#
        preds_st = [PO.cross_val_predict(o, rd_r, cv; ex = seq)
                    for o in (plain_hrp(), herc())]
        preds_nc = [PO.cross_val_predict(plain_hrp(), rd_r, cv; cols = cl, ex = seq)
                    for cl in cls]
        nobs_st = [length(p.mrd.X) for p in preds_st]
        for (rd, sq) in ((rd_r, false), (rd_sq, true), (rd_3d, false))
            for (predictions, u) in
                ((preds_st, PO.FullUniverse()), (preds_nc, PO.ClusterUniverse(cls)))
                Ws = [PO.fold_weight_matrix(predictions, u, f, N)
                      for f in eachindex(test_idx)]
                rdo = PO.rebuild_returns_result(rd, predictions, u)
                @test size(panel_feature_matrix(rdo.pnl)[2], 1) == size(rdo.X, 1) == T
                @test size(panel_feature_matrix(rdo.pnl)[2], 2) == length(rdo.nx)
                @test any(f -> PO.features_are_assets(f, rdo.nx), rdo.pnl.pf) == sq
                r = 0
                for (f, rows) in enumerate(test_idx)
                    _, _, _, _, pnl_e, _ = PO.prepare_outer_rd(PO.port_opt_view(rd, rows,
                                                                                :), Ws[f])
                    nz_e, Z_e = panel_feature_matrix(pnl_e)
                    blk = panel_feature_matrix(rdo.pnl)[2][(r + 1):(r + length(rows)), :, :]
                    # A static source has no observation axis of its own, so the non-`cv`
                    # result is the fold's constant; a time-varying one already carries one.
                    expected = if ndims(Z_e) == 3
                        Z_e
                    else
                        permutedims(repeat(Z_e, 1, 1, length(rows)), (3, 1, 2))
                    end
                    @test blk ≈ expected
                    @test panel_feature_matrix(rdo.pnl)[1] == nz_e
                    r += length(rows)
                end
            end
        end
        # The predictions came through all six calls unchanged.
        @test [length(p.mrd.X) for p in preds_st] == nobs_st
    end

    @testset "Fold rows are selected by the clock, not by a cumulative count" begin
        #=
        `IndexWalkForward`'s first test block does not start at row 1 -- its warmup window
        comes first -- so a seam that counted observations instead of locating them would
        pair every fold with the wrong slice of a time-varying feature matrix. `Z3[t, i, j]`
        names its own position, so a mis-indexed observation axis shows up in the values.
        =#
        cvwf = IndexWalkForward(60, 63)
        test_idx = PO.split(cvwf, rd_3d).test_idx
        @test first(first(test_idx)) != 1
        herc() = HierarchicalEqualRiskContribution(;
                                                   opt = HierarchicalOptimiser(; slv = slv))
        predictions = [PO.cross_val_predict(o, rd_3d, cvwf; ex = seq)
                       for o in (plain_hrp(), herc())]
        Ws = [PO.fold_weight_matrix(predictions, PO.FullUniverse(), f, N)
              for f in eachindex(test_idx)]
        rdo = PO.rebuild_returns_result(rd_3d, predictions, PO.FullUniverse())
        # The folds cover fewer rows than the clock has, which is the point: a cumulative
        # count would have started at row 1 and run out before the last fold.
        @test size(panel_feature_matrix(rdo.pnl)[2], 1) == sum(length, test_idx) < T
        r = 0
        for (f, rows) in enumerate(test_idx)
            @test panel_feature_matrix(rdo.pnl)[2][(r + 1):(r + length(rows)), :, :] ≈
                  collapse(Z3[rows, :, :], false, Ws[f])
            r += length(rows)
        end
    end

    @testset "A time-varying feature matrix needs a clock to survive the folds" begin
        #=
        The fold's rows are recovered from its timestamps rather than stored on the
        prediction -- `port_opt_view` slices `ts` with the very `test_idx` the fold was
        built from, so a fold's `ts` *is* its slice of the clock. Without one there is
        nothing to match, and a time-varying feature matrix cannot say which observation
        each fold's observations are. That is refused, loudly and by name.
        =#
        cv = KFold(; n = 3)
        herc() = HierarchicalEqualRiskContribution(;
                                                   opt = HierarchicalOptimiser(; slv = slv))
        rd_3d_nots = ReturnsResult(; nx = nx, X = X, nf = nf, F = F,
                                   pnl = matrix_panel(nf, Z3))
        preds = [PO.cross_val_predict(o, rd_3d_nots, cv; ex = seq)
                 for o in (plain_hrp(), herc())]
        e = try
            PO.rebuild_returns_result(rd_3d_nots, preds, PO.FullUniverse())
        catch err
            err
        end
        @test isa(e, PO.IsNothingError)
        @test occursin("ts", e.msg)

        # The requirement is scoped to the shape that needs it: a static feature matrix has
        # no observation axis to align, so it runs on fold sizes alone and never asks.
        rd_r_nots = ReturnsResult(; nx = nx, X = X, nf = nf, F = F,
                                  pnl = matrix_panel(["z$i" for i in 1:K], Zr))
        preds_r = [PO.cross_val_predict(o, rd_r_nots, cv; ex = seq)
                   for o in (plain_hrp(), herc())]
        @test size(panel_feature_matrix(PO.rebuild_returns_result(rd_r_nots, preds_r,
                                                                  PO.FullUniverse()).pnl)[2]) ==
              (T, 2, K)

        # Recovering by time is only sound on a uniquely-keyed axis, so `ReturnsResult`
        # now refuses a repeated timestamp outright.
        dup_ts = copy(ts)
        dup_ts[5] = dup_ts[4]
        @test_throws ArgumentError ReturnsResult(; nx = nx, X = X, ts = dup_ts)
    end

    @testset "The seam asserts the fold alignment it has always assumed" begin
        #=
        `reshape(X, :, N)` is only meaningful if every sub-portfolio's fold `f` covers the
        same observations, and the per-fold weight matrix is only well defined if it does.
        The invariant predates this map; nothing ever stated it. It matters most on the
        combinatorial path, where each sub-portfolio's `scorer` picks a path independently.
        =#
        p3 = PO.cross_val_predict(plain_hrp(), rd_r, KFold(; n = 3); ex = seq)
        p4 = PO.cross_val_predict(plain_hrp(), rd_r, KFold(; n = 4); ex = seq)
        @test_throws DimensionMismatch PO.rebuild_returns_result(rd_r, [p3, p4],
                                                                 PO.FullUniverse())

        # Same number of folds, different rows in them.
        pw = PO.cross_val_predict(plain_hrp(), rd_r, IndexWalkForward(60, 63); ex = seq)
        pk = PO.cross_val_predict(plain_hrp(), rd_r, KFold(; n = length(pw.pred)); ex = seq)
        @test_throws DimensionMismatch PO.rebuild_returns_result(rd_r, [pw, pk],
                                                                 PO.FullUniverse())
    end

    @testset "The transport carrier is gone" begin
        #=
        `PredictionReturnsResult` used to be the fourth carrier of a feature matrix, and
        the only one nothing ever read from: the fold collapsed, the folds stacked, and the
        seam assembled. Under the recompute that journey has no destination, and keeping it
        would have meant keeping the sole square special case in the collapse -- the one
        site where square structurally cannot be treated like non-square, since a single
        weight vector cannot index both axes.

        This testset used to pin that carrier's validation. It now pins its absence.
        =#
        Xf = collect(1.0:5.0)
        Zf = randn(StableRNG(2), 5, 2)
        @test !hasproperty(PredictionReturnsResult(; nx = ["_1"], X = Xf), :Z)
        @test !hasproperty(PredictionReturnsResult(; nx = ["_1"], X = Xf), :nz)
        @test :Z ∉ fieldnames(PredictionReturnsResult)
        @test :nz ∉ fieldnames(PredictionReturnsResult)
        @test_throws MethodError PredictionReturnsResult(; nx = ["_1"], X = Xf,
                                                         pnl = matrix_panel(["a", "b"], Zf))
        # Nothing is lost: the seam reaches every fold's weights and rows through `pred`.
        pred = PO.cross_val_predict(plain_hrp(), rd_3d, KFold(; n = 3); ex = seq)
        @test length(pred.pred) == 3
        @test all(p -> isa(p.res.w, AbstractVector{<:Real}), pred.pred)
        @test all(p -> !isnothing(p.rd.ts), pred.pred)
        @test PO.fold_row_indices(rd_3d, pred.pred) ==
              PO.split(KFold(; n = 3), rd_3d).test_idx
        # A static carrier has no observation axis to locate, so the recovery is a no-op
        # there -- which is why that shape never asks for a clock at all.
        pred_r = PO.cross_val_predict(plain_hrp(), rd_r, KFold(; n = 3); ex = seq)
        @test PO.fold_row_indices(rd_r, pred_r.pred) == fill(Colon(), 3)
        @test PO.fold_feature_anchors(rd_r, pred_r.pred) ==
              [length(p.rd.X) for p in pred_r.pred]
    end
end
