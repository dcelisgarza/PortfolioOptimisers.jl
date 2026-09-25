@testset "The meta-optimisation module states its mathematics" begin
    using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra
    using InteractiveUtils: subtypes

    #=
    The sweep of `src/17_Optimisation/06_Meta/01_Base_MetaOptimisation.jl` (#1052). Each
    testset asserts one claim that the docstrings of that file state, with numbers:

    - `combination_weights` is the closed form of its `# Mathematical definition`, and
      keeps the tilted products when either total is zero.
    - `prepare_outer_rd` collapses `B` as `B W`, `iv` as `V W̃` and a vector `ivpa` as
      `W̃ᵀ a`, and keeps a scalar `ivpa` and a vector `B`.
    - The fold-less outer returns are `X W`, and the final weights are `W v`.
    - `outer_optimisation_finaliser` names every failed stage, in order, and keeps the
      finalised weights.
    - Under `cv`, `rebuild_returns_result` stacks the columns of each prediction, and the
      rows of one fold carry that fold's own normalised weights.
    - `fold_weight_matrix` puts zeros outside each cluster of a `ClusterUniverse`.
    - A fold count or a clock that disagrees raises `DimensionMismatch`, and a
      time-varying panel without a clock raises `IsNothingError`.
    =#

    PO = PortfolioOptimisers
    rng = StableRNG(987654321)
    T, N = 300, 8
    X = randn(rng, T, N) ./ 100 .+ 0.0005
    ts = PO.Dates.Date(2020, 1, 1) .+ PO.Dates.Day.(0:(T - 1))
    iv = 0.1 .+ 0.2 .* rand(rng, T, N)
    ivpa = 0.5 .+ rand(rng, N)
    Bm = randn(rng, T, N) ./ 100
    rd = ReturnsResult(; nx = string.(1:N), X = X, ts = ts, iv = iv, ivpa = ivpa, B = Bm,
                       nb = string.(1:N))
    # The normalised inner weight matrix, written from its definition.
    wtilde(W) = abs.(W) ./ map(x -> iszero(x) ? one(x) : x, sum(abs.(W); dims = 1))
    opti = [InverseVolatility(), EqualWeighted(), InverseVolatility()]
    cv = OptimisationCrossValidation(; cv = KFold(; n = 4))

    @testset "combination_weights is its closed form" begin
        s = [3.0, 1.0, 2.0]
        v = [0.5, 0.3, 0.2]
        @test PO.combination_weights(s, v) ≈ s .* v * sum(v) / sum(s .* v)
        # A zero outer total and a zero tilted total both keep the tilted products.
        @test PO.combination_weights(s, [0.5, -0.3, -0.2]) == s .* [0.5, -0.3, -0.2]
        @test PO.combination_weights([1.0, 5.0, 1.0], [0.5, -0.1, 0.0]) == [0.5, -0.5, 0.0]
    end

    @testset "prepare_outer_rd collapses B, iv and ivpa" begin
        W = randn(StableRNG(3), N, 3)
        W[:, 3] .= 0
        nb, B, ivo, ivpao, pnl, Xb = PO.prepare_outer_rd(rd, W)
        @test nb == ["_b1", "_b2", "_b3"]
        @test B ≈ Bm * W
        @test ivo ≈ iv * wtilde(W)
        @test ivpao ≈ transpose(wtilde(W)) * ivpa
        # A convex combination stays inside the range of its inputs.
        @test all(minimum(iv) - 1e-12 .<= ivo[:, 1:2] .<= maximum(iv) + 1e-12)
        # A column of zeros gives a zero column and a zero entry.
        @test all(iszero, ivo[:, 3])
        @test iszero(ivpao[3])
        @test isnothing(pnl)
        @test size(Xb) == (T, 3)
        @test eltype(Xb) == eltype(X)
        # A scalar adjustment and a benchmark that is not a matrix are kept as they are.
        rds = ReturnsResult(; nx = string.(1:N), X = X, ts = ts, iv = iv, ivpa = 0.7,
                            B = Bm[:, 1])
        nbs, Bs, _, ivpas = PO.prepare_outer_rd(rds, W)
        @test ivpas == 0.7
        @test Bs === rds.B
        @test isnothing(nbs)
    end

    @testset "the fold-less path: outer returns X W, final weights W v" begin
        st = Stacking(; opti = opti, opto = InverseVolatility())
        r = optimise(st, rd)
        W = hcat([x.w for x in r.resi]...)
        pr = r.resi[1].pr
        ro = PO.predict_outer_returns(nothing, st, PO.FullUniverse(), rd, pr, nothing, W,
                                      r.resi)
        @test ro.X ≈ pr.X * W
        @test ro.iv ≈ iv * wtilde(W)
        @test ro.nx == ["_1", "_2", "_3"]
        @test r.w ≈ W * r.reso.w
    end

    @testset "outer_optimisation_finaliser names every failed stage" begin
        st = Stacking(; opti = opti, opto = InverseVolatility())
        r = optimise(st, rd)
        W = hcat([x.w for x in r.resi]...)
        wf = IterativeWeightFinaliser()
        wbad = WeightBounds(; lb = fill(0.2, N), ub = fill(0.2, N))
        rbad = optimise(InverseVolatility(; wb = wbad), rd)
        @test rbad.retcode isa OptimisationFailure
        resi_bad = PO.NonFiniteAllocationOptimisationResult[r.resi[1], rbad, r.resi[3]]

        rc, w = PO.outer_optimisation_finaliser(r.wb, wf, r.resi, OptimisationSuccess(),
                                                r.reso.w, W)
        @test rc isa OptimisationSuccess
        @test w ≈ W * r.reso.w

        rc, w = PO.outer_optimisation_finaliser(r.wb, wf, resi_bad, OptimisationSuccess(),
                                                r.reso.w, W)
        @test rc.res.msg == "opti failed.\n"

        rc, w = PO.outer_optimisation_finaliser(wbad, wf, resi_bad,
                                                OptimisationFailure(; res = "outer"),
                                                r.reso.w, W)
        @test rc.res.msg ==
              "opti failed.\nopto failed.\nweight bounds finalisation failed.\n"
        @test rc.res.opti[2] isa OptimisationFailure
        @test rc.res.opto.res == "outer"
        @test rc.res.wb isa OptimisationFailure
        # The weights of the finalisation are returned when a stage fails too.
        @test w ≈ fill(0.2, N)

        # The frontier method runs the point method once for each point.
        rcs, ws = PO.outer_optimisation_finaliser(r.wb, wf, r.resi,
                                                  [OptimisationSuccess(),
                                                   OptimisationFailure(; res = "outer")],
                                                  [r.reso.w, reverse(r.reso.w)], W)
        @test rcs[1] isa OptimisationSuccess
        @test rcs[2].res.msg == "opto failed.\n"
        @test ws[2] ≈ W * reverse(r.reso.w)

        # The bound accepts only the `WeightBounds` that `weight_bounds_constraints`
        # returns, also for an estimator with no bounds.
        @test_throws MethodError PO.outer_optimisation_finaliser(nothing, wf, r.resi,
                                                                 OptimisationSuccess(),
                                                                 r.reso.w, W)
    end

    @testset "rebuild_returns_result stacks the predictions" begin
        st = Stacking(; opti = opti, opto = InverseVolatility(), cv = cv)
        preds = PO.sub_portfolio_predictions(PO.MultiPeriodPredictionResult, opti,
                                             PO.FullUniverse(), rd, cv.cv, st.ex)
        heights = [length(p.mrd.X) for p in preds]
        ro = PO.rebuild_returns_result(rd, preds, PO.FullUniverse())
        @test ro.X == hcat([p.mrd.X for p in preds]...)
        @test ro.iv == hcat([p.mrd.iv for p in preds]...)
        @test ro.ivpa == [p.mrd.ivpa for p in preds]
        @test ro.nb == ["_b1", "_b2", "_b3"]
        @test size(ro.B) == (T, 3)
        # The method does not change the predictions, so a second call agrees.
        @test PO.rebuild_returns_result(rd, preds, PO.FullUniverse()).X == ro.X
        @test [length(p.mrd.X) for p in preds] == heights

        # The rows of fold 2 carry the normalised weights of fold 2, and `ivpa` the
        # weights of the last fold.
        pred1 = PO.assert_fold_alignment(preds)
        W2 = PO.fold_weight_matrix(preds, PO.FullUniverse(), 2, N)
        rows = findall(in(pred1[2].rd.ts), ts)
        off = length(pred1[1].rd.X)
        @test ro.iv[(off + 1):(off + length(rows)), :] ≈ iv[rows, :] * wtilde(W2)
        W4 = PO.fold_weight_matrix(preds, PO.FullUniverse(), 4, N)
        @test ro.ivpa ≈ vec(transpose(ivpa) * wtilde(W4))

        # The whole meta-optimiser takes the same path.
        @test optimise(st, rd).retcode isa OptimisationSuccess
    end

    @testset "a cluster sub-portfolio sees its own assets" begin
        nco = NestedClustered(; opti = InverseVolatility(), opto = InverseVolatility(),
                              cv = cv)
        rn = optimise(nco, rd)
        idx = PO.assignments(rn.clr)
        cls = [findall(==(i), idx) for i in 1:(rn.clr.k)]
        u = PO.ClusterUniverse(cls)
        @test PO.sub_portfolio_count(u, nco.opti) == length(cls)
        @test PO.sub_portfolio_count(PO.FullUniverse(), opti) == 3
        @test PO.sub_portfolio_view(u, rn.pr, 1).X == rn.pr.X[:, cls[1]]
        @test PO.sub_portfolio_view(PO.FullUniverse(), rn.pr, 1) === rn.pr

        preds = PO.sub_portfolio_predictions(PO.MultiPeriodPredictionResult, nco.opti, u,
                                             rd, cv.cv, nco.ex)
        W2 = PO.fold_weight_matrix(preds, u, 2, N)
        for k in eachindex(cls)
            @test W2[cls[k], k] == preds[k].pred[2].res.w
            @test all(iszero, W2[setdiff(1:N, cls[k]), k])
        end

        # Without folds, the column of a cluster is its weights, placed on the full axis.
        Wn = zeros(N, length(cls))
        for k in eachindex(cls)
            Wn[cls[k], k] = rn.resi[k].w
        end
        ro = PO.predict_outer_returns(nothing, nco, u, rd, rn.pr, nothing, Wn, rn.resi)
        @test ro.X ≈ rn.pr.X * Wn
    end

    @testset "the folds must agree" begin
        preds = PO.sub_portfolio_predictions(PO.MultiPeriodPredictionResult, opti,
                                             PO.FullUniverse(), rd, cv.cv,
                                             PO.FLoops.SequentialEx())
        preds5 = PO.sub_portfolio_predictions(PO.MultiPeriodPredictionResult, opti,
                                              PO.FullUniverse(), rd, KFold(; n = 5),
                                              PO.FLoops.SequentialEx())
        @test_throws DimensionMismatch PO.assert_fold_alignment([preds[1], preds5[2]])
        rd_shift = ReturnsResult(; nx = string.(1:N), X = X, ts = ts .+ PO.Dates.Day(1))
        preds_s = PO.sub_portfolio_predictions(PO.MultiPeriodPredictionResult, opti,
                                               PO.FullUniverse(), rd_shift, cv.cv,
                                               PO.FLoops.SequentialEx())
        @test_throws DimensionMismatch PO.assert_fold_alignment([preds[1], preds_s[2]])
        @test PO.assert_fold_alignment(preds) === preds[1].pred

        rd_nots = ReturnsResult(; nx = string.(1:N), X = X)
        @test_throws PO.IsNothingError PO.fold_row_indices(rd_nots, preds[1].pred)
        err = try
            PO.fold_row_indices(rd_nots, preds[1].pred)
        catch e
            e
        end
        @test occursin("time-varying Asset Panel", sprint(showerror, err))
    end

    @testset "a scheme with no rng is shared" begin
        @test PO.sub_portfolio_cv(cv.cv) === cv.cv
        # No scheme that `OptimisationCrossValidation` accepts has an `rng` field.
        function leaves(S, acc = Any[])
            for s in subtypes(S)
                isabstracttype(s) ? leaves(s, acc) : push!(acc, s)
            end
            return acc
        end
        @test !any(s -> hasfield(Base.unwrap_unionall(s), :rng),
                   leaves(PO.OptimisationCrossValidationEstimator))
    end

    @testset "a categorical field reaches the outer problem as a tensor field" begin
        lvl = ["Fin", "Tech"]
        pnl = asset_panel([CategoricalPanelInput(; name = "sector",
                                                 vals = lvl[[1, 2, 1, 2, 1, 2, 1, 2]])])
        rdc = ReturnsResult(; nx = string.(1:N), X = X, ts = ts, pnl = pnl)
        st = Stacking(; opti = opti, opto = InverseVolatility(), cv = cv)
        preds = PO.sub_portfolio_predictions(PO.MultiPeriodPredictionResult, opti,
                                             PO.FullUniverse(), rdc, cv.cv, st.ex)
        ro = PO.rebuild_returns_result(rdc, preds, PO.FullUniverse())
        f = PO.panel_field(ro.pnl, "sector")
        @test f isa TensorPanelField
        @test size(f.vals) == (T, 3, 2)
        @test all(sum(f.vals; dims = 3) .≈ 1)
        # Neither removed method exists: no path gives either of them an argument.
        @test !hasmethod(PO.panel_field_stack, Tuple{Vector{PO.CategoricalPanelField}})
        @test !hasmethod(PO.fold_asset_panel, Tuple{Nothing, Any, Matrix{Float64}, Any})
    end
end
