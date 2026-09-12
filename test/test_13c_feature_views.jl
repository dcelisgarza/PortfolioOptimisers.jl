using PortfolioOptimisers, Test, Clustering, TimeSeries, Dates, StableRNGs, StatsBase,
      LinearAlgebra, Clarabel

const PO = PortfolioOptimisers

#=
The failure this file exists to catch is quiet: a feature matrix that survives a subset or
a fold unsliced still produces a finite, symmetric, plausible distance matrix -- computed
over the wrong universe. Asserting that nothing threw would pass on exactly that bug, so
every test below asserts on the *value* of the matrix that reached the kernel.

`RecordingDistance` is the instrument. It wraps a real `FeatureDistance` and records the
Feature Matrix the routed three-argument methods derive, so a test can compare that matrix
against the slice it should be, rather than inferring it from a weight vector. The two
carriers reach it as `pr` and `rd`, and it derives the matrix the same way the kernel does.
It is deliberately mutable and passes through `port_opt_view` unchanged (the universal
estimator fallback), so one recorder accumulates every fold and every cluster of a run.
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

# An extension author's returns result that forgot to implement `port_opt_view`. The
# library ships no such type on purpose -- the tripwire exists so one cannot exist quietly.
struct UnviewableReturnsResult <: PO.AbstractReturnsResult end

include(joinpath(@__DIR__, "asset_panel_fixture.jl"))
# A square block is one tensor Panel Field whose labels are the asset names, which is
# what makes `features_are_assets` true and the view cut both axes.
function sqpanel(labels, vals)
    return asset_panel([TensorPanelInput(; name = "prox", axis = "asset", labels = labels,
                                         vals = vals)])
end
@testset "Feature matrix under views, folds, meta-optimisers and Pipeline" begin
    # `port_opt_view` is internal, and so is the sequential executor the recorder needs:
    # `push!` under the default `ThreadedEx` would be a race in the test, not in the library.
    port_opt_view = PO.port_opt_view
    seq = PO.FLoops.SequentialEx()
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false))
    hopt(de) = HierarchicalOptimiser(; cle = ClustersEstimator(; de = de), slv = slv)
    plain_hrp() = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; slv = slv))

    T, N, K = 250, 8, 3
    rng = StableRNG(20260728)
    # The asset returns are genuinely driven by the factors, so the regression the derived
    # carrier is built from is well posed and does not warn its way through the file.
    F = randn(rng, T, K) ./ 50
    Bl = randn(rng, N, K)
    X = F * transpose(Bl) .+ randn(rng, T, N) ./ 500
    nx = ["A$i" for i in 1:N]
    ts = collect(Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(T - 1)))
    nf = ["F$i" for i in 1:K]

    # A square, block-structured feature matrix: the shape a phylogeny or adjacency source
    # produces, and the only one whose feature axis is the asset axis.
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
    # Z3[i, j, k] names its own position, so a mis-sliced observation axis is visible in
    # the values and not only in the shape.
    Z3 = reshape(Float64.(1:(T * N * K)), T, N, K) ./ 1000

    rd_sq = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, ts = ts, pnl = sqpanel(nx, Zsq))
    # The same numbers, but the names no longer claim the features are the assets. This is
    # the *only* difference between `rd_sq` and `rd_rect`, which is what makes the pair a
    # controlled experiment on the feature-axis slice.
    rd_rect = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, ts = ts,
                            pnl = matrix_panel(["z$i" for i in 1:N], Zsq))
    rd_3d = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, ts = ts,
                          pnl = matrix_panel(nf, Z3))

    @testset "NestedClustered slices both axes of a square feature matrix" begin
        # The inner optimiser runs per cluster on a column subset. When the features are
        # the assets, missing the feature axis leaves each cluster measured against the
        # full universe's columns -- finite, plausible, wrong.
        function run_nco(rd)
            ri, ro = RecordingDistance(FeatureDistance()),
                     RecordingDistance(FeatureDistance())
            nco = NestedClustered(; cle = ClustersEstimator(; de = ro),
                                  opti = HierarchicalRiskParity(; opt = hopt(ri)),
                                  opto = plain_hrp(), ex = seq)
            return optimise(nco, rd), ri, ro
        end
        res_sq, ri_sq, ro_sq = run_nco(rd_sq)
        res_re, ri_re, ro_re = run_nco(rd_rect)

        idx = assignments(res_sq.clr)
        cls = [findall(==(i), idx) for i in 1:(res_sq.clr.k)]
        @test length(cls) >= 2
        @test all(length(cl) >= 1 for cl in cls)

        # The outer problem is the full universe, so both runs cluster identically: the
        # divergence below is caused by the feature-axis slice and by nothing else.
        @test assignments(res_re.clr) == idx
        @test ro_sq.seen[1] == Zsq
        @test ro_re.seen[1] == Zsq

        # Correct: both axes move with the cluster.
        @test [size(z) for z in ri_sq.seen] == [(length(cl), length(cl)) for cl in cls]
        @test all(ri_sq.seen[i] == Zsq[cls[i], cls[i]] for i in eachindex(cls))
        # Row-only: the columns still point at all eight assets, and the distance kernel
        # accepts it without complaint.
        @test [size(z) for z in ri_re.seen] == [(length(cl), N) for cl in cls]
        @test all(ri_re.seen[i] == Zsq[cls[i], :] for i in eachindex(cls))
        @test all(all(isfinite, PO.distance(FeatureDistance(), z)) for z in ri_re.seen)

        #=
        The two measurements disagree, and the disagreement propagates into the cluster
        each inner optimiser builds. Note that the *distance* always differs while the
        *weights* need not: HierarchicalRiskParity consumes the dendrogram's leaf ordering,
        and a strongly blocked feature matrix can survive the wrong slice with its ordering
        intact. That is exactly why the recorded matrices above, and not the weight vector,
        are this file's primary evidence -- a weight-only test would pass on the bug for
        half the fixtures one might pick.
        =#
        @test all(PO.distance(FeatureDistance(), ri_sq.seen[i]) !=
                  PO.distance(FeatureDistance(), ri_re.seen[i]) for i in eachindex(cls))
        @test any(res_sq.resi[i].clr.res.merges != res_re.resi[i].clr.res.merges
                  for i in eachindex(cls))
        @test !isapprox(res_sq.w, res_re.w)
        @test isapprox(sum(res_sq.w), 1)
        @test isapprox(sum(res_re.w), 1)

        # A rectangular carrier commutes with the asset view: only the asset axis moves,
        # and a time-varying one keeps every observation this arity never indexes.
        ri_3 = RecordingDistance(FeatureDistance())
        nco_3 = NestedClustered(; cle = ClustersEstimator(; de = FeatureDistance()),
                                opti = HierarchicalRiskParity(; opt = hopt(ri_3)),
                                opto = plain_hrp(), ex = seq)
        res_3 = optimise(nco_3, rd_3d)
        idx3 = assignments(res_3.clr)
        cls3 = [findall(==(i), idx3) for i in 1:(res_3.clr.k)]
        @test all(size(z) == (T, length(cls3[i]), K) for (i, z) in pairs(ri_3.seen))
        @test all(ri_3.seen[i] == Z3[:, cls3[i], :] for i in eachindex(cls3))
    end

    @testset "SubsetResampling subsets both axes, and so does a Stacking nested in it" begin
        ri = RecordingDistance(FeatureDistance())
        sr = SubsetResampling(; opt = HierarchicalRiskParity(; opt = hopt(ri)),
                              n_subsets = 4, subset_size = 4, seed = 12345, ex = seq)
        res = optimise(sr, rd_sq)
        cols = [view(res.idx, :, i) for i in axes(res.idx, 2)]
        @test length(ri.seen) == length(cols)
        @test all(size(z) == (4, 4) for z in ri.seen)
        @test all(ri.seen[i] == Zsq[cols[i], cols[i]] for i in eachindex(cols))
        @test isapprox(sum(res.w), 1)

        # `port_opt_view(::Stacking, i, X)` slices the inner optimisers, so a Stacking used
        # as the resampled optimiser inherits the subset rather than reaching past it.
        rs = RecordingDistance(FeatureDistance())
        st = Stacking(; opti = [HierarchicalRiskParity(; opt = hopt(rs))],
                      opto = plain_hrp(), ex = seq)
        sr2 = SubsetResampling(; opt = st, n_subsets = 3, subset_size = 4, seed = 999,
                               ex = seq)
        res2 = optimise(sr2, rd_sq)
        cols2 = [view(res2.idx, :, i) for i in axes(res2.idx, 2)]
        @test all(rs.seen[i] == Zsq[cols2[i], cols2[i]] for i in eachindex(cols2))
    end

    @testset "A meta-optimiser's outer synthetic universe carries collapsed features" begin
        # Stacking's own inner optimisers all see the full universe -- it subsets nothing.
        ri = RecordingDistance(FeatureDistance())
        st = Stacking(;
                      opti = [HierarchicalRiskParity(; opt = hopt(ri)),
                              HierarchicalEqualRiskContribution(;
                                                                opt = HierarchicalOptimiser(;
                                                                                            slv = slv))],
                      opto = plain_hrp(), ex = seq)
        res = optimise(st, rd_sq)
        @test length(ri.seen) == 1
        @test ri.seen[1] == Zsq
        @test isapprox(sum(res.w), 1)

        #=
        The outer problem's assets are sub-portfolios, not the universe, so `Z` cannot ride
        through unchanged -- it is *collapsed* onto them. This testset owns the boundary
        itself: the outer estimator receives a matrix on the synthetic universe rather than
        the error it used to get. `test_13d_feature_collapse.jl` owns the arithmetic.
        =#
        for mk in (de -> Stacking(;
                                  opti = [plain_hrp(),
                                          HierarchicalEqualRiskContribution(;
                                                                            opt = HierarchicalOptimiser(;
                                                                                                        slv = slv))],
                                  opto = HierarchicalRiskParity(; opt = hopt(de)), ex = seq),
                   de -> NestedClustered(; cle = ClustersEstimator(; de = FeatureDistance()),
                                         opti = plain_hrp(),
                                         opto = HierarchicalRiskParity(; opt = hopt(de)),
                                         ex = seq))
            ro = RecordingDistance(FeatureDistance())
            res = optimise(mk(ro), rd_sq)
            @test isapprox(sum(res.w), 1)
            # A square carrier collapses two-sided, so the outer universe is square in its
            # own assets rather than in the original ones.
            @test length(ro.seen) == 1
            @test size(ro.seen[1], 1) == size(ro.seen[1], 2)
            @test size(ro.seen[1], 1) < N
        end
    end

    @testset "Observation splits slice a time-varying feature matrix" begin
        # A 3-tensor's leading axis is the observations, so every fold owes it the same
        # slice it gives `X`. A static carrier has no observation axis and passes through.
        for cv in (KFold(; n = 4), IndexWalkForward(80, 40),
                   CombinatorialCrossValidation(; n_folds = 5, n_test_folds = 2))
            ri = RecordingDistance(FeatureDistance())
            pred = cross_val_predict(HierarchicalRiskParity(; opt = hopt(ri)), rd_3d, cv;
                                     ex = seq)
            sp = split(cv, rd_3d)
            @test length(ri.seen) == length(sp.train_idx)
            @test all(ri.seen[i] == Z3[sp.train_idx[i], :, :]
                      for i in eachindex(sp.train_idx))
            @test all(size(z, 1) == length(sp.train_idx[i]) for (i, z) in pairs(ri.seen))

            rs = RecordingDistance(FeatureDistance())
            cross_val_predict(HierarchicalRiskParity(; opt = hopt(rs)), rd_sq, cv; ex = seq)
            @test all(z == Zsq for z in rs.seen)
        end

        # `train_test_split` is a pair of `port_opt_view`s, so the same rule holds there.
        tr, te = train_test_split(rd_3d; train_size = 150)
        @test panel_feature_matrix(tr.pnl)[2] == Z3[1:size(tr.X, 1), :, :]
        @test panel_feature_matrix(te.pnl)[2] == Z3[(T - size(te.X, 1) + 1):T, :, :]
        @test size(panel_feature_matrix(tr.pnl)[2], 1) +
              size(panel_feature_matrix(te.pnl)[2], 1) == T
    end

    @testset "MultipleRandomised splits observations and assets together" begin
        cv = MultipleRandomised(IndexWalkForward(80, 40); rng = StableRNG(11), seed = 7,
                                n_subsets = 2, subset_size = 5)
        # Time-varying: rows by fold, columns by draw, feature axis untouched.
        ri = RecordingDistance(FeatureDistance())
        cross_val_predict(HierarchicalRiskParity(; opt = hopt(ri)), rd_3d, cv; ex = seq)
        s3 = split(cv, rd_3d)
        @test length(ri.seen) == length(s3.train_idx)
        @test all(ri.seen[i] == Z3[s3.train_idx[i], s3.asset_idx[i], :]
                  for i in eachindex(s3.train_idx))

        # Square: the replication's asset draw moves BOTH axes. This is the same
        # silent-wrongness NestedClustered exposes, one subsetting scheme over.
        rs = RecordingDistance(FeatureDistance())
        cross_val_predict(HierarchicalRiskParity(; opt = hopt(rs)), rd_sq, cv; ex = seq)
        ss = split(cv, rd_sq)
        @test all(size(z) == (cv.subset_size, cv.subset_size) for z in rs.seen)
        @test all(rs.seen[i] == Zsq[ss.asset_idx[i], ss.asset_idx[i]]
                  for i in eachindex(ss.train_idx))
    end

    @testset "Pipeline: preprocessing and the prices conversion run inside the fold" begin
        Tp = 120
        tsp = collect(Date(2021, 1, 1):Day(1):(Date(2021, 1, 1) + Day(Tp - 1)))
        rngp = StableRNG(4242)
        Pv = 100 .+ cumsum(abs.(randn(rngp, Tp, N)) ./ 10; dims = 1)
        Z3p = reshape(Float64.(1:(Tp * N * K)), Tp, N, K) ./ 1000
        Zsqp = block_features(StableRNG(24), N, blocks)

        # Price-level cross-validation: each fold converts its own window, losing the row
        # the percentage change consumes -- and `Z` must lose exactly that row too.
        prc = PricesResult(; X = TimeArray(tsp, Pv, nx), pnl = matrix_panel(nf, Z3p))
        ri = RecordingDistance(FeatureDistance())
        pipe = Pipeline(;
                        steps = (PricesToReturns(),
                                 HierarchicalRiskParity(; opt = hopt(ri))))
        cv = KFold(; n = 3)
        cross_val_predict(pipe, prc, cv; ex = seq)
        sp = split(cv, prc)
        @test length(ri.seen) == length(sp.train_idx)
        @test all(ri.seen[i] == Z3p[sp.train_idx[i][2:end], :, :]
                  for i in eachindex(sp.train_idx))
        @test all(size(z, 1) == length(sp.train_idx[i]) - 1 for (i, z) in pairs(ri.seen))

        # Stateful preprocessing inside the fold: `MissingDataFilter` drops an asset at
        # price level, and a square carrier must lose that asset on both axes.
        Pm = copy(Pv)
        Pm[:, 3] .= NaN
        prm = PricesResult(; X = TimeArray(tsp, Pm, nx), pnl = sqpanel(nx, Zsqp))
        keep = [1, 2, 4, 5, 6, 7, 8]
        rf = RecordingDistance(FeatureDistance())
        pipe_f = Pipeline(;
                          steps = (MissingDataFilter(; col_thr = 0.5), PricesToReturns(),
                                   HierarchicalRiskParity(; opt = hopt(rf))))
        res_f = fit(pipe_f, prm)
        @test res_f.ctx.returns.nx == nx[keep]
        @test panel_feature_matrix(res_f.ctx.returns.pnl)[2] == Zsqp[keep, keep]
        @test rf.seen[1] == Zsqp[keep, keep]

        # A clustering step in the pipeline reaches the same bridge, so it is routed too. It
        # carries the same filter: the conversion deletes no asset (ADR 0133), so the drop
        # that gives this pipeline its seven-asset universe is the filter's.
        rc = RecordingDistance(FeatureDistance())
        pipe_c = Pipeline(;
                          steps = (MissingDataFilter(; col_thr = 0.5), PricesToReturns(),
                                   ClustersEstimator(; de = rc), plain_hrp()))
        fit(pipe_c, prm)
        @test length(rc.seen) == 1
        @test rc.seen[1] == Zsqp[keep, keep]

        # Search cross-validation at price level draws assets *and* windows rows, through
        # `pipeline_asset_view`/`pipeline_data_view` rather than the returns-level arities.
        prs = PricesResult(; X = TimeArray(tsp, Pv, nx), pnl = sqpanel(nx, Zsqp))
        rg = RecordingDistance(FeatureDistance())
        mrs = MultipleRandomised(IndexWalkForward(60, 20); subset_size = 3, n_subsets = 2,
                                 seed = 42)
        grid = ["opt" => [HierarchicalRiskParity(; opt = hopt(rg))]]
        pipe_s = Pipeline(;
                          steps = (PricesToReturns(),
                                   "opt" => HierarchicalRiskParity(; opt = hopt(rg))))
        search_cross_validation(pipe_s,
                                GridSearchCrossValidation(grid; cv = mrs,
                                                          r = ConditionalValueatRisk()),
                                prs)
        ss = split(mrs, prs)
        @test length(rg.seen) == length(ss.train_idx)
        @test all(size(z) == (mrs.subset_size, mrs.subset_size) for z in rg.seen)
        @test all(any(z == Zsqp[c, c] for c in unique(ss.asset_idx)) for z in rg.seen)
    end

    @testset "The carrier's panel slices, and a producer refits" begin
        #=
        Two routes, two behaviours, and the difference is the whole content of `#804`. A
        panel on the *carrier* is data, so a view slices it. A producer on the *distance* is
        configuration, so a view passes it through and the cluster refits it on the
        cluster's own returns. Nothing on either route is a stale slice of a larger matrix.
        =#
        rd_plain = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, ts = ts)

        function run_nco(de_i, de_o, rdx)
            nco = NestedClustered(; pe = FactorPrior(),
                                  cle = ClustersEstimator(; de = de_o),
                                  opti = HierarchicalRiskParity(;
                                                                opt = HierarchicalOptimiser(;
                                                                                            pe = FactorPrior(),
                                                                                            cle = ClustersEstimator(;
                                                                                                                    de = de_i),
                                                                                            slv = slv)),
                                  opto = plain_hrp(), ex = seq)
            res = optimise(nco, rdx)
            idx = assignments(res.clr)
            return res, [findall(==(i), idx) for i in 1:(res.clr.k)]
        end

        # The carrier route: the square panel is sliced on both axes, per cluster.
        ri_d, ro_d = RecordingDistance(FeatureDistance()),
                     RecordingDistance(FeatureDistance())
        res_d, cls_d = run_nco(ri_d, ro_d, rd_sq)
        @test ro_d.seen[1] == Zsq
        @test [size(z) for z in ri_d.seen] == [(length(cl), length(cl)) for cl in cls_d]
        @test all(ri_d.seen[i] == Zsq[cls_d[i], cls_d[i]] for i in eachindex(cls_d))

        # The producer route: a regression producer refits per cluster, so the matrix that
        # reaches the kernel is a fresh estimate whose trailing axis is still the full
        # factor set.
        ape = RegressionPanel()
        ri_p, ro_p = RecordingDistance(FeatureDistance(; ape = ape)),
                     RecordingDistance(FeatureDistance(; ape = ape))
        res_p, cls_p = run_nco(ri_p, ro_p, rd_plain)
        @test [size(z) for z in ri_p.seen] == [(length(cl), K) for cl in cls_p]
        @test all(ri_p.seen[i] == PO.panel_field(PO.asset_panel(ape,
                                            prior(FactorPrior(),
                                                  port_opt_view(rd_plain, cls_p[i])),
                                            nothing, X[:, cls_p[i]]), "loadings").vals
                  for i in eachindex(cls_p))
        @test isapprox(sum(res_p.w), 1)

        # A square *producer* refits per cluster too, and its matrix is square over the
        # cluster -- the shape that would have been wrong had anything been carried through.
        apeq = PhylogenyPanel(; pl = ClustersEstimator())
        ric = RecordingDistance(FeatureDistance(; ape = apeq))
        resc, clsc = run_nco(ric, RecordingDistance(FeatureDistance(; ape = apeq)),
                             rd_plain)
        @test [size(z) for z in ric.seen] == [(length(cl), length(cl)) for cl in clsc]

        # The network source refits per cluster as well, and its outer matrix is the whole
        # universe's.
        apen = PhylogenyPanel(; pl = NetworkEstimator(; alg = KruskalTree()))
        rin = RecordingDistance(FeatureDistance(; ape = apen))
        ron = RecordingDistance(FeatureDistance(; ape = apen))
        resn, clsn = run_nco(rin, ron, rd_plain)
        @test ron.seen[1] ==
              phylogeny_features(Proximity(), NetworkEstimator(; alg = KruskalTree()), X)
        @test [size(z) for z in rin.seen] == [(length(cl), length(cl)) for cl in clsn]
        @test all(rin.seen[i] ==
                  phylogeny_features(Proximity(), NetworkEstimator(; alg = KruskalTree()),
                                     X[:, clsn[i]]) for i in eachindex(clsn))
        @test isapprox(sum(resn.w), 1)

        # The producer itself is never viewed: it is configuration, so the slot survives the
        # round trip through `port_opt_view` unchanged.
        de = FeatureDistance(; ape = apen)
        @test port_opt_view(de, [1, 3, 5]).ape === apen
    end
end
