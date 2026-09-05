using PortfolioOptimisers, Test, Clustering, TimeSeries, Dates, StableRNGs, StatsBase,
      LinearAlgebra, Clarabel

const PO = PortfolioOptimisers

#=
The failure this file exists to catch is quiet: a feature matrix that survives a subset or
a fold unsliced still produces a finite, symmetric, plausible distance matrix -- computed
over the wrong universe. Asserting that nothing threw would pass on exactly that bug, so
every test below asserts on the *value* of the matrix that reached the kernel.

`RecordingDistance` is the instrument. It wraps a real `FeatureDistance` and records the
`Z` handed to it through the routed three-argument methods, so a test can compare that
matrix against the slice it should be, rather than inferring it from a weight vector.
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
function PO.distance(de::RecordingDistance, ce, X; Z = nothing, kwargs...)
    record!(de, Z)
    return PO.distance(de.de, ce, X; Z = Z, kwargs...)
end
function PO.cor_and_dist(de::RecordingDistance, ce, X; Z = nothing, kwargs...)
    record!(de, Z)
    return PO.cor_and_dist(de.de, ce, X; Z = Z, kwargs...)
end
PO.distance(de::RecordingDistance, Z; kwargs...) = PO.distance(de.de, Z; kwargs...)
function PO.cor_and_dist(de::RecordingDistance, Z; kwargs...)
    return PO.cor_and_dist(de.de, Z; kwargs...)
end

# An extension author's returns result that forgot to implement `port_opt_view`. The
# library ships no such type on purpose -- the tripwire exists so one cannot exist quietly.
struct UnviewableReturnsResult <: PO.AbstractReturnsResult end

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

    rd_sq = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, ts = ts,
                          pnl = feature_matrix_panel(nx, Zsq))
    # The same numbers, but the names no longer claim the features are the assets. This is
    # the *only* difference between `rd_sq` and `rd_rect`, which is what makes the pair a
    # controlled experiment on the feature-axis slice.
    rd_rect = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, ts = ts,
                            pnl = feature_matrix_panel(["z$i" for i in 1:N], Zsq))
    rd_3d = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, ts = ts,
                          pnl = feature_matrix_panel(nf, Z3))

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
        prc = PricesResult(; X = TimeArray(tsp, Pv, nx),
                           pnl = feature_matrix_panel(nf, Z3p))
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
        prm = PricesResult(; X = TimeArray(tsp, Pm, nx),
                           pnl = feature_matrix_panel(nx, Zsqp))
        keep = [1, 2, 4, 5, 6, 7, 8]
        rf = RecordingDistance(FeatureDistance())
        pipe_f = Pipeline(;
                          steps = (MissingDataFilter(; col_thr = 0.5), PricesToReturns(),
                                   HierarchicalRiskParity(; opt = hopt(rf))))
        res_f = fit(pipe_f, prm)
        @test res_f.ctx.returns.nx == nx[keep]
        @test panel_feature_matrix(res_f.ctx.returns.pnl)[2] == Zsqp[keep, keep]
        @test rf.seen[1] == Zsqp[keep, keep]

        # A clustering step in the pipeline reaches the same bridge, so it is routed too.
        rc = RecordingDistance(FeatureDistance())
        pipe_c = Pipeline(;
                          steps = (PricesToReturns(), ClustersEstimator(; de = rc),
                                   plain_hrp()))
        fit(pipe_c, prm)
        @test length(rc.seen) == 1
        @test rc.seen[1] == Zsqp[keep, keep]

        # Search cross-validation at price level draws assets *and* windows rows, through
        # `pipeline_asset_view`/`pipeline_data_view` rather than the returns-level arities.
        prs = PricesResult(; X = TimeArray(tsp, Pv, nx),
                           pnl = feature_matrix_panel(nx, Zsqp))
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

    @testset "The two carriers under a view: :data slices, :prior refits" begin
        fpe = FeaturePrior(; pe = FactorPrior(), ze = RegressionFeatures())
        pr_z = prior(fpe, rd_sq)
        # The carriers hold genuinely different matrices, so a test cannot pass by picking
        # the wrong one: the data carrier is the square block matrix, the prior carrier the
        # rectangular factor loadings.
        @test size(panel_feature_matrix(rd_sq.pnl)[2]) == (N, N)
        @test size(panel_feature_matrix(pr_z.pnl)[2]) == (N, K)

        function run_nco(z_src)
            ri, ro = RecordingDistance(FeatureDistance()),
                     RecordingDistance(FeatureDistance())
            nco = NestedClustered(; pe = fpe, cle = ClustersEstimator(; de = ro),
                                  z_src = z_src,
                                  opti = HierarchicalRiskParity(;
                                                                opt = HierarchicalOptimiser(;
                                                                                            pe = fpe,
                                                                                            cle = ClustersEstimator(;
                                                                                                                    de = ri),
                                                                                            slv = slv,
                                                                                            z_src = z_src)),
                                  opto = plain_hrp(), ex = seq)
            res = optimise(nco, rd_sq)
            idx = assignments(res.clr)
            return res, ri, ro, [findall(==(i), idx) for i in 1:(res.clr.k)]
        end
        res_d, ri_d, ro_d, cls_d = run_nco(:data)
        res_p, ri_p, ro_p, cls_p = run_nco(:prior)

        # The selector picks between two populated carriers at the outer level.
        @test ro_d.seen[1] == panel_feature_matrix(rd_sq.pnl)[2]
        @test ro_p.seen[1] == panel_feature_matrix(pr_z.pnl)[2]
        @test assignments(res_d.clr) != assignments(res_p.clr)

        # `:data` slices the carried matrix -- both axes, because it is square.
        @test [size(z) for z in ri_d.seen] == [(length(cl), length(cl)) for cl in cls_d]
        @test all(ri_d.seen[i] == panel_feature_matrix(rd_sq.pnl)[2][cls_d[i], cls_d[i]]
                  for i in eachindex(cls_d))

        # `:prior` does not slice anything: the cluster's prior is refit on the cluster's
        # own returns, so the matrix that reaches the kernel is a fresh estimate whose
        # feature axis is still the full factor set.
        @test [size(z) for z in ri_p.seen] == [(length(cl), K) for cl in cls_p]
        @test all(ri_p.seen[i] ==
                  panel_feature_matrix(prior(fpe, port_opt_view(rd_sq, cls_p[i])).pnl)[2]
                  for i in eachindex(cls_p))

        # A `LowOrderPrior` view slices its own carrier on the asset axis and never on the
        # feature axis -- a square derived `Z` included, since the producer that built it
        # refits rather than being cut down.
        i = [1, 3, 5]
        @test panel_feature_matrix(port_opt_view(pr_z, i).pnl)[2] ==
              panel_feature_matrix(pr_z.pnl)[2][i, :]
        pr_sq = LowOrderPrior(; X = pr_z.X, mu = pr_z.mu, sigma = pr_z.sigma,
                              pnl = feature_matrix_panel(["_z$(k)" for k in axes(Zsq, 2)],
                                                         Zsq))
        @test panel_feature_matrix(port_opt_view(pr_sq, i).pnl)[2] == Zsq[i, :]
    end

    @testset "A square feature producer refits inside a real fold" begin
        #=
        No producer embeds data: `PhylogenyFeatures` holds an *estimator*, so every fold and
        every cluster refits the graph on its own universe. What only a fold can prove is
        that the refit actually reaches through the optimiser's own view and `FeaturePrior`
        to the producer, so a cluster's feature matrix is square over the cluster rather
        than over the universe it came from.
        =#
        rd_plain = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, ts = ts)
        PEX = phylogeny_matrix(NetworkEstimator(; alg = KruskalTree()), rd_plain).X
        @test issymmetric(PEX) && all(iszero, diag(PEX))
        fpe_cle = FeaturePrior(; ze = PhylogenyFeatures(; pl = ClustersEstimator()))
        fpe_conf = FeaturePrior(;
                                ze = PhylogenyFeatures(;
                                                       pl = NetworkEstimator(;
                                                                             alg = KruskalTree())))

        mk_nco(fpe, ri, ro) = NestedClustered(; pe = fpe,
                                              cle = ClustersEstimator(; de = ro),
                                              z_src = :prior,
                                              opti = HierarchicalRiskParity(;
                                                                            opt = HierarchicalOptimiser(;
                                                                                                        pe = fpe,
                                                                                                        cle = ClustersEstimator(;
                                                                                                                                de = ri),
                                                                                                        slv = slv,
                                                                                                        z_src = :prior)),
                                              opto = plain_hrp(), ex = seq)

        # A clustering source refits per cluster too, and its matrix is square over the
        # cluster -- the shape that would have been wrong had anything been carried through.
        ric = RecordingDistance(FeatureDistance())
        resc = optimise(mk_nco(fpe_cle, ric, RecordingDistance(FeatureDistance())),
                        rd_plain)
        idxc = assignments(resc.clr)
        clsc = [findall(==(i), idxc) for i in 1:(resc.clr.k)]
        @test [size(z) for z in ric.seen] == [(length(cl), length(cl)) for cl in clsc]

        # The network source refits per cluster as well.
        ri = RecordingDistance(FeatureDistance())
        ro = RecordingDistance(FeatureDistance())
        res = optimise(mk_nco(fpe_conf, ri, ro), rd_plain)
        idx = assignments(res.clr)
        cls = [findall(==(i), idx) for i in 1:(res.clr.k)]

        @test ro.seen[1] == panel_feature_matrix(prior(fpe_conf, rd_plain).pnl)[2]
        @test [size(z) for z in ri.seen] == [(length(cl), length(cl)) for cl in cls]
        # The reference is the producer viewed the same way the fold views it -- a refit.
        @test all(ri.seen[i] == panel_feature_matrix(prior(port_opt_view(fpe_conf, cls[i]),
                                                           port_opt_view(rd_plain, cls[i])).pnl)[2]
                  for i in eachindex(cls))
        @test isapprox(sum(res.w), 1)

        # A literal matrix in the `ze` slot is the other data-carrying shape, and slices on
        # its asset axis only: its columns are features and stay whole.
        Zlit = abs.(randn(StableRNG(2468), N, 4))
        fpe_lit = FeaturePrior(; ze = Zlit)
        rl = RecordingDistance(FeatureDistance())
        nco_lit = NestedClustered(; pe = fpe_lit,
                                  cle = ClustersEstimator(; de = FeatureDistance()),
                                  z_src = :prior,
                                  opti = HierarchicalRiskParity(;
                                                                opt = HierarchicalOptimiser(;
                                                                                            pe = fpe_lit,
                                                                                            cle = ClustersEstimator(;
                                                                                                                    de = rl),
                                                                                            slv = slv,
                                                                                            z_src = :prior)),
                                  opto = plain_hrp(), ex = seq)
        res_lit = optimise(nco_lit, rd_plain)
        idx_lit = assignments(res_lit.clr)
        cls_lit = [findall(==(i), idx_lit) for i in 1:(res_lit.clr.k)]
        @test [size(z) for z in rl.seen] == [(length(cl), 4) for cl in cls_lit]
        @test all(rl.seen[i] == Zlit[cls_lit[i], :] for i in eachindex(cls_lit))

        # A *time-varying* literal cannot survive an observation fold: folds slice
        # observations before the prior is fit and never touch the estimator, so its
        # leading axis stops matching. The documented answer is "use a producer", and the
        # failure is a construction error rather than a silent misalignment.
        fpe_3d = FeaturePrior(; ze = abs.(randn(StableRNG(1357), T, N, 2)))
        @test_throws DimensionMismatch cross_val_predict(HierarchicalRiskParity(;
                                                                                opt = HierarchicalOptimiser(;
                                                                                                            pe = fpe_3d,
                                                                                                            cle = ClustersEstimator(;
                                                                                                                                    de = FeatureDistance()),
                                                                                                            slv = slv,
                                                                                                            z_src = :prior)),
                                                         rd_plain, KFold(; n = 3); ex = seq)

        # The same chain under a resampling scheme that draws assets and windows rows.
        ri = RecordingDistance(FeatureDistance())
        cv = MultipleRandomised(IndexWalkForward(80, 40); rng = StableRNG(11), seed = 7,
                                n_subsets = 2, subset_size = 5)
        mk_hrp(fpe, de) = HierarchicalRiskParity(;
                                                 opt = HierarchicalOptimiser(; pe = fpe,
                                                                             cle = ClustersEstimator(;
                                                                                                     de = de),
                                                                             slv = slv,
                                                                             z_src = :prior))
        cross_val_predict(mk_hrp(fpe_conf, ri), rd_plain, cv; ex = seq)
        sp = split(cv, rd_plain)
        @test length(ri.seen) == length(sp.train_idx)
        @test all(size(z) == (cv.subset_size, cv.subset_size) for z in ri.seen)
        @test all(ri.seen[i] ==
                  panel_feature_matrix(prior(port_opt_view(fpe_conf, sp.asset_idx[i]),
                                             port_opt_view(rd_plain, sp.train_idx[i],
                                                           sp.asset_idx[i])).pnl)[2]
                  for i in eachindex(sp.train_idx))
    end

    @testset "The port_opt_view tripwires still report accurately" begin
        # A mistyped call must name the call shape rather than be reported as an
        # unimplemented subtype -- `ReturnsResult` does implement `port_opt_view`.
        e = try
            port_opt_view(rd_sq, 1, 2, 3, 4)
        catch err
            err
        end
        @test isa(e, ArgumentError)
        @test occursin("does not accept this call shape", e.msg)
        @test occursin("4 positional index argument(s)", e.msg)

        e = try
            port_opt_view(rd_sq, [1, 2]; assets = [1, 2])
        catch err
            err
        end
        @test isa(e, ArgumentError)
        @test occursin("keyword argument(s) assets", e.msg)
        @test occursin("`factors` is the third positional index", e.msg)

        # An `AbstractReturnsResult` subtype with no method is a missing implementation,
        # not a leaf value to hand back unsubselected.
        e = try
            port_opt_view(UnviewableReturnsResult(), [1, 2])
        catch err
            err
        end
        @test isa(e, ArgumentError)
        @test occursin("does not implement port_opt_view", e.msg)
        @test occursin("silently train on the unsubselected universe", e.msg)
    end
end
