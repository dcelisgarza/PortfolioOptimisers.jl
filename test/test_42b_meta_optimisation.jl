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
    - Integer returns give the weights of the same returns as `Float64` (#1363).
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
    cv = KFold(; n = 4)

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
        nb, B, ivo, ivpao, pnl, Xb = PO.prepare_outer_rd(rd, W, RenormaliseActive())
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
        nbs, Bs, _, ivpas = PO.prepare_outer_rd(rds, W, RenormaliseActive())
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
                                             PO.FullUniverse(), rd, cv, st.ex)
        heights = [length(p.mrd.X) for p in preds]
        ro = PO.rebuild_returns_result(rd, preds, PO.FullUniverse(), RenormaliseActive())
        @test ro.X == hcat([p.mrd.X for p in preds]...)
        @test ro.iv == hcat([p.mrd.iv for p in preds]...)
        @test ro.ivpa == [p.mrd.ivpa for p in preds]
        @test ro.nb == ["_b1", "_b2", "_b3"]
        @test size(ro.B) == (T, 3)
        # The method does not change the predictions, so a second call agrees.
        @test PO.rebuild_returns_result(rd, preds, PO.FullUniverse(),
                                        RenormaliseActive()).X == ro.X
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
                                             rd, cv, nco.ex)
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
                                             PO.FullUniverse(), rd, cv,
                                             PO.FLoops.SequentialEx())
        preds5 = PO.sub_portfolio_predictions(PO.MultiPeriodPredictionResult, opti,
                                              PO.FullUniverse(), rd, KFold(; n = 5),
                                              PO.FLoops.SequentialEx())
        @test_throws DimensionMismatch PO.assert_fold_alignment([preds[1], preds5[2]])
        rd_shift = ReturnsResult(; nx = string.(1:N), X = X, ts = ts .+ PO.Dates.Day(1))
        preds_s = PO.sub_portfolio_predictions(PO.MultiPeriodPredictionResult, opti,
                                               PO.FullUniverse(), rd_shift, cv,
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
        @test PO.sub_portfolio_cv(cv) === cv
        # No scheme that the `cv` field of a meta-optimiser accepts has an `rng` field.
        function leaves(S, acc = Any[])
            for s in subtypes(S)
                isabstracttype(s) ? leaves(s, acc) : push!(acc, s)
            end
            return acc
        end
        @test !any(s -> hasfield(Base.unwrap_unionall(s), :rng),
                   leaves(PO.OptimisationCrossValidationEstimator))
    end

    @testset "cv and scorer are flat fields, and a scorer needs a combinatorial scheme" begin
        ccv = CombinatorialCrossValidation(; n_folds = 4, n_test_folds = 2)
        nqp = NearestQuantilePrediction()
        ocv = OnlineIndexWalkForward(100, 50)
        stack(c, s) = Stacking(; opti = opti, opto = InverseVolatility(), cv = c,
                               scorer = s)
        nest(c, s) = NestedClustered(; opti = InverseVolatility(),
                                     opto = InverseVolatility(), cv = c, scorer = s)
        for mk in (stack, nest)
            @test mk(ccv, nqp).scorer === nqp
            @test isnothing(mk(ccv, nothing).scorer)
            @test mk(ocv, nothing).cv === ocv
            # A scorer that the scheme never reads is refused at construction.
            for c in (nothing, cv, ocv)
                err = @test_throws ArgumentError mk(c, nqp)
                @test occursin("CombinatorialCrossValidation", err.value.msg)
            end
            # A split result is not a scheme, and an estimator holds no result.
            @test_throws TypeError mk(split(cv, rd), nothing)
        end
        # The combinatorial path selects each path with the scorer of the meta-optimiser,
        # and `nothing` selects with `NearestQuantilePrediction`.
        calls = Ref(0)
        pick(p) = (calls[] += 1; nqp(p))
        r0 = optimise(stack(ccv, nothing), rd)
        r1 = optimise(stack(ccv, pick), rd)
        @test calls[] == length(opti)
        @test r1.w ≈ r0.w
        @test r1.cv === ccv
        @test r1.scorer === pick
        @test isnothing(r0.scorer)
    end

    @testset "a categorical field reaches the outer problem as a tensor field" begin
        lvl = ["Fin", "Tech"]
        pnl = asset_panel([CategoricalPanelInput(; name = "sector",
                                                 vals = lvl[[1, 2, 1, 2, 1, 2, 1, 2]])])
        rdc = ReturnsResult(; nx = string.(1:N), X = X, ts = ts, pnl = pnl)
        st = Stacking(; opti = opti, opto = InverseVolatility(), cv = cv)
        preds = PO.sub_portfolio_predictions(PO.MultiPeriodPredictionResult, opti,
                                             PO.FullUniverse(), rdc, cv, st.ex)
        ro = PO.rebuild_returns_result(rdc, preds, PO.FullUniverse(), RenormaliseActive())
        f = PO.panel_field(ro.pnl, "sector")
        @test f isa TensorPanelField
        @test size(f.vals) == (T, 3, 2)
        @test all(sum(f.vals; dims = 3) .≈ 1)
        # Neither removed method exists: no path gives either of them an argument.
        @test !hasmethod(PO.panel_field_stack, Tuple{Vector{PO.CategoricalPanelField}})
        @test !hasmethod(PO.fold_asset_panel,
                         Tuple{Nothing, Any, Matrix{Float64}, Any, Any})
    end

    @testset "integer returns give the weights of the same returns as Float64" begin
        #=
        #1363: a weight, a fee and a net return are fractions, so a meta-optimiser that took
        its matrix of inner weights, its weight bounds, its fees or its buffer of outer
        returns from the element type of integer returns threw `InexactError`. Each path
        below threw before the fix.
        =#
        Xi = rand(StableRNG(2), -3:3, 120, 6)
        nx = string.("A", 1:6)
        rdi = ReturnsResult(; nx = nx, X = Xi)
        rdf = ReturnsResult(; nx = nx, X = Matrix{Float64}(Xi))
        sets = UniverseSets(; dict = Dict("nx" => nx))
        fees = FeesEstimator(; l = ["A1" => 0.001], dl = 0.0005)
        kcv = KFold(; n = 3)
        ew = EqualWeighted()
        sti = [EqualWeighted(), InverseVolatility()]
        opts = [NestedClustered(; opti = ew, opto = ew),
                NestedClustered(; opti = ew, opto = ew, cv = kcv),
                NestedClustered(; opti = ew, opto = ew, fees = fees, sets = sets),
                Stacking(; opti = sti, opto = ew),
                Stacking(; opti = sti, opto = ew, cv = kcv),
                Stacking(; opti = sti, opto = ew, fees = fees, sets = sets),
                SubsetResampling(; opt = ew, subset_size = 3, n_subsets = 4, seed = 7),
                SubsetResampling(; opt = ew, subset_size = 3, n_subsets = 4, seed = 7,
                                 fees = fees, sets = sets)]
        for opt in opts
            wi = optimise(opt, rdi).w
            wf = optimise(opt, rdf).w
            @test eltype(wi) == Float64
            @test isapprox(wi, wf; rtol = 0, atol = eps(Float64))
        end
        # The buffer of outer returns holds a net return, a fraction, for integer inputs.
        _, _, _, _, _, Xo = PO.prepare_outer_rd(rdi, [1 0; 0 1; 1 0; 0 1; 1 0; 0 1],
                                                RenormaliseActive())
        @test eltype(Xo) == Float64
        # A type that is not an integer is kept.
        rdr = ReturnsResult(; nx = nx, X = Rational{Int}.(Xi))
        _, _, _, _, _, Xr = PO.prepare_outer_rd(rdr, Rational{Int}.(ones(Int, 6, 2)),
                                                RenormaliseActive())
        @test eltype(Xr) == Rational{Int}
    end
end

@testset "Stacking states its mathematics" begin
    using Test, PortfolioOptimisers, StableRNGs, Clarabel, Statistics

    #=
    The sweep of `src/17_Optimisation/06_Meta/03_Stacking.jl` (#792). Each testset asserts one
    claim that the docstrings of that file state, with numbers:

    - Without `cv`, the synthetic returns are the net returns `X W - F(W)`, the outer weights
      are the answer of `opto` on them, and the stacked weights are `W c`.
    - With `cv`, the synthetic returns are out of sample, `W` is still the full-sample
      matrix, and the outer weights differ from the in-sample ones.
    - An outer efficient frontier gives one weight vector and one return code for each
      point. An inner frontier raises `ArgumentError`.
    - An asset outside the Investable Mask gets a zero weight, and the other weights are
      those of a run without the asset.
    - The constructor, `narrow_optimiser_vector`, the check for `NestedClustered` and the
      time-dependent methods behave as their docstrings state.
    =#

    PO = PortfolioOptimisers
    rng = StableRNG(987654321)
    T, N = 200, 6
    X = randn(rng, T, N) ./ 100 .+ 0.0005
    ts = PO.Dates.Date(2020, 1, 1) .+ PO.Dates.Day.(0:(T - 1))
    rd = ReturnsResult(; nx = string.(1:N), X = X, ts = ts)
    sex = PO.FLoops.SequentialEx()
    ew = EqualWeighted()
    ivo = InverseVolatility()
    opti = [ivo, ew]
    cv = KFold(; n = 4)
    # The closed form of `InverseVolatility` on a returns matrix, the outer map `𝒪` here.
    inverse_volatility(R) = (v = 1 ./ vec(std(R; dims = 1)); v / sum(v))
    inner_weights(res) = hcat([x.w for x in res.resi]...)

    @testset "without cv: R = X W - F(W), v = 𝒪(R), w = W c" begin
        fees = Fees(; l = 0.002)
        scale = [3.0, 1.0]
        res = optimise(Stacking(; opti = opti, opto = ivo, fees = fees, scale = scale,
                                ex = sex), rd)
        W = inner_weights(res)
        @test W[:, 1] ≈ optimise(ivo, rd).w
        @test W[:, 2] ≈ fill(1 / N, N)
        # A proportional long fee charges `l` times the long weight on every observation.
        R = X * W .- transpose(0.002 .* vec(sum(max.(W, 0); dims = 1)))
        v = inverse_volatility(R)
        @test res.reso.w ≈ v
        c = scale .* v * sum(v) / sum(scale .* v)
        @test res.w ≈ W * c
        @test sum(res.w) ≈ 1
        @test res.retcode isa OptimisationSuccess
        # With no Combination Weight, c = v.
        res0 = optimise(Stacking(; opti = opti, opto = ivo, fees = fees, ex = sex), rd)
        @test res0.w ≈ W * v
        # The Combination Weight does not reach the outer problem.
        @test res0.reso.w ≈ res.reso.w
    end

    @testset "with cv: out-of-sample R, full-sample W" begin
        res = optimise(Stacking(; opti = opti, opto = ivo, cv = cv, ex = sex), rd)
        W = inner_weights(res)
        @test W ≈ inner_weights(optimise(Stacking(; opti = opti, opto = ivo, ex = sex), rd))
        rdo = PO.predict_outer_returns(cv, Stacking(; opti = opti, opto = ivo, ex = sex),
                                       PO.FullUniverse(), rd, res.pr, nothing, W, res.resi)
        # Four folds of KFold hold every observation in a test window once.
        @test size(rdo.X) == (T, 2)
        @test res.reso.w ≈ inverse_volatility(rdo.X)
        # The out-of-sample returns give a different answer from the in-sample ones.
        @test !isapprox(res.reso.w, inverse_volatility(X * W))
        @test res.w ≈ W * res.reso.w
        @test res.cv === cv
    end

    @testset "an outer frontier gives one point per weight vector" begin
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     settings = Dict("verbose" => false))
        fr = ArithmeticReturn(; settings = PO.JuMPReturnsSettings(; lb = Frontier(; N = 3)))
        mrf = MeanRisk(; r = Variance(), obj = MinimumRisk(),
                       opt = JuMPOptimiser(; slv = slv, ret = fr))
        res = optimise(Stacking(;
                                opti = [ivo, ew,
                                        MeanRisk(; opt = JuMPOptimiser(; slv = slv))],
                                opto = mrf, ex = sex), rd)
        W = inner_weights(res)
        @test length(res.w) == 3
        @test length(res.retcode) == 3
        @test all(x -> isa(x, OptimisationSuccess), res.retcode)
        @test all(res.w[k] ≈ W * res.reso.w[k] for k in 1:3)
        # `set_retcode` fails one point and keeps every other field.
        rcs = PO.OptimisationReturnCode[res.retcode[1], OptimisationFailure(; res = "drop"),
                                        res.retcode[3]]
        rs = PO.set_retcode(res, rcs)
        @test rs.retcode[2] isa OptimisationFailure
        @test rs.w === res.w
        @test rs.reso === res.reso
        # An inner frontier is refused.
        @test_throws ArgumentError optimise(Stacking(; opti = [ivo, mrf], opto = ivo,
                                                     ex = sex), rd)
    end

    @testset "an asset outside the Investable Mask gets a zero weight" begin
        X2 = copy(X)
        X2[:, 4] .= NaN
        st = Stacking(; opti = opti, opto = ivo, ex = sex)
        res = optimise(st, ReturnsResult(; nx = string.(1:N), X = X2, ts = ts))
        @test res.imsk == [true, true, true, false, true, true]
        @test iszero(res.w[4])
        @test size(inner_weights(res)) == (N - 1, 2)
        keep = [1, 2, 3, 5, 6]
        ref = optimise(st, ReturnsResult(; nx = string.(keep), X = X[:, keep], ts = ts))
        @test res.w[keep] ≈ ref.w
        # The positional constructor does not expand the weights a second time.
        @test PO.StackingResult(res.pr, res.wb, res.fees, res.resi, res.reso, res.cv,
                                res.scorer, res.retcode, res.w, res.imsk, res.fb).w ===
              res.w
    end

    @testset "the constructor validates its arguments" begin
        @test_throws PO.IsEmptyError Stacking(; opti = [], opto = ivo)
        @test_throws ArgumentError Stacking(; opti = [ivo, 1.0], opto = ivo)
        @test_throws PO.IsNothingError Stacking(; opti = opti, opto = ivo,
                                                wb = WeightBoundsEstimator(;
                                                                           lb = Dict("1" =>
                                                                                         0.0)))
        @test_throws PO.IsNothingError Stacking(; opti = opti, opto = ivo,
                                                fees = FeesEstimator(;
                                                                     l = Dict("1" => 0.001)))
        @test_throws ArgumentError Stacking(; opti = TimeDependent([opti, opti], :nearest),
                                            opto = ivo, cv = cv)
        @test_throws ArgumentError Stacking(; opti = opti,
                                            opto = TimeDependent([ivo, ew], :nearest;
                                                                 default = ivo))
        @test_throws DimensionMismatch Stacking(; opti = opti, opto = ivo,
                                                scale = [1.0, 2.0, 3.0])
        @test_throws PO.IsNonFiniteError Stacking(; opti = opti, opto = ivo,
                                                  scale = [1.0, NaN])
        @test PO.stacking_td_defaults() ==
              (; pe = EmpiricalPrior(), opti = PO.NoDefault(), opto = PO.NoDefault(),
               wf = IterativeWeightFinaliser())
    end

    @testset "narrow_optimiser_vector" begin
        typed = [ivo, ivo]
        @test PO.narrow_optimiser_vector(typed) === typed
        td = TimeDependent([ivo, ew], :nearest; default = ew)
        mixed = Any[ivo, td]
        narrowed = PO.narrow_optimiser_vector(mixed)
        @test eltype(narrowed) == Union{typeof(ivo), typeof(td)}
        @test narrowed == mixed
        tdf = TimeDependent([opti, opti]; default = opti)
        @test PO.narrow_optimiser_vector(tdf) === tdf
        @test_throws ArgumentError PO.narrow_optimiser_vector(Any[ivo, "a"])
    end

    @testset "NestedClustered refuses a precomputed result in a Stacking" begin
        res_ew = optimise(ew, rd)
        # A Stacking that runs alone accepts a result in `opti`, also under `cv`.
        @test optimise(Stacking(; opti = [res_ew, ivo], opto = ivo, ex = sex), rd).w isa
              AbstractVector
        @test optimise(Stacking(; opti = [res_ew, ivo], opto = ivo, cv = cv, ex = sex),
                       rd).w isa AbstractVector
        @test_throws ArgumentError NestedClustered(; opti = ivo,
                                                   opto = Stacking(; opti = [res_ew, ew],
                                                                   opto = ivo))
        # A schedule of `opti` is checked entry by entry.
        E = Union{typeof(ew), typeof(ivo), typeof(res_ew)}
        good = TimeDependent([E[ew, ivo], E[ivo, ew]]; default = E[ew, ivo])
        bad = TimeDependent([E[ew, ivo], E[res_ew, ew]]; default = E[ew, ivo])
        @test isnothing(PO.assert_special_nco_requirements(Stacking(; opti = good,
                                                                    opto = ivo)))
        @test_throws ArgumentError PO.assert_special_nco_requirements(Stacking(; opti = bad,
                                                                               opto = ivo))
        @test_throws ArgumentError NestedClustered(;
                                                   opti = Stacking(; opti = bad,
                                                                   opto = ivo), opto = ivo)
        # The external check reads `pe`, `opto`, and `opti` under `cv`.
        @test isnothing(PO.assert_external_optimiser(Stacking(; opti = opti, opto = ivo,
                                                              cv = cv)))
        @test_throws ArgumentError PO.assert_external_optimiser(Stacking(;
                                                                         pe = prior(EmpiricalPrior(),
                                                                                    rd),
                                                                         opti = opti,
                                                                         opto = ivo))
        # The internal check reaches every inner optimiser.
        clr = clusterise(ClustersEstimator(), X)
        hrp = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; cle = clr))
        @test isnothing(PO.assert_internal_optimiser(Stacking(; opti = opti, opto = ivo)))
        @test_throws ArgumentError PO.assert_internal_optimiser(Stacking(;
                                                                         opti = [ivo, hrp],
                                                                         opto = ivo))
    end

    @testset "the time-dependent methods" begin
        st = Stacking(; opti = opti, opto = ivo, ex = sex)
        ctx = PO.TimeDependentContext(; i = 2, n = 2, rd = rd, train_idx = 1:100,
                                      test_idx = 101:120, w_prev = nothing,
                                      path_id = nothing)
        @test !PO.is_time_dependent(st)
        @test PO.update_time_dependent_estimator(st, ctx) === st
        @test !PO.needs_previous_weights(st)
        # An element schedule that binds `:nearest` stays for the inner cross-validation.
        tdel = TimeDependent([ew, ivo], :nearest; default = ew)
        tdwf = TimeDependent([IterativeWeightFinaliser(), EuclideanWeightFinaliser()])
        std = Stacking(; opti = [ivo, tdel], opto = ivo, cv = cv, wf = tdwf, ex = sex)
        @test PO.is_time_dependent(std)
        upd = PO.update_time_dependent_estimator(std, ctx)
        @test upd.wf isa EuclideanWeightFinaliser
        @test upd.opti[2] === tdel
        rst = PO.reset_time_dependent_estimator(std)
        @test rst.wf isa IterativeWeightFinaliser
        @test rst.opti[2] === tdel
        # A field schedule of `opti` resets to its `default`.
        stf = Stacking(; opti = TimeDependent([[ew, ivo], [ivo, ew]]; default = [ew, ivo]),
                       opto = ivo)
        @test PO.is_time_dependent(stf)
        @test PO.reset_time_dependent_estimator(stf).opti == [ew, ivo]
        # `opti`, `opto` and `fb` each pass on a need for previous weights.
        pw = PreviousWeights()
        @test PO.needs_previous_weights(Stacking(; opti = [ivo, pw], opto = ivo))
        @test PO.needs_previous_weights(Stacking(; opti = opti, opto = pw))
        @test PO.needs_previous_weights(Stacking(; opti = opti, opto = ivo, fb = pw))
    end

    @testset "port_opt_view follows its View parameters" begin
        pr = prior(EmpiricalPrior(), rd)
        st = Stacking(; pe = pr, opti = opti, opto = ivo, scale = [2.0, 1.0], cv = cv,
                      ex = sex, fees = Fees(; l = fill(0.001, N)),
                      wb = WeightBounds(; lb = fill(0.0, N), ub = fill(0.5, N)))
        # A matrix of the wrong width: the method reads `pe.X` in its place.
        v = PO.port_opt_view(st, [1, 3, 5], X[:, 1:2])
        @test v.pe.X == X[:, [1, 3, 5]]
        @test v.fees.l == fill(0.001, 3)
        @test v.wb.ub == fill(0.5, 3)
        @test v.scale === st.scale
        @test v.cv === st.cv
        @test v.wf === st.wf
        @test v.opti == st.opti
    end
end

@testset "Nested Clustered Optimisation states its mathematics" begin
    using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Statistics, Clarabel

    #=
    The sweep of `src/17_Optimisation/06_Meta/02_NestedClustered.jl` (#791). Each testset
    asserts one claim that the docstrings of that file state, with numbers:

    - The weights are `W v`: an inverse-volatility oracle written from the definition gives
      the inner and the outer weights, and the synthetic returns `X W` carry the reduced
      moments `W' mu` and `W' Sigma W`.
    - The outer optimiser names a cluster by its synthetic asset, through either shape of
      outer asset sets, and the caller's sets do not change.
    - The weight bounds of the meta-optimiser bind the final weights, and an infeasible
      bound names the finalisation stage.
    - `set_retcode` keeps the expanded weights of a result whose universe dropped an asset.
    - Every refusal reads a precomputed result directly, in a vector and in a
      `TimeDependent` schedule. A schedule of precomputed constraints reached a cluster
      and threw `DimensionMismatch` before the constructor read schedules.
    =#

    PO = PortfolioOptimisers
    rng = StableRNG(987654321)
    T, N = 300, 8
    X = randn(rng, T, N) ./ 100 .+ 0.0005
    nx = string.('A':'H')
    rd = ReturnsResult(; nx = nx, X = X)
    seq = PO.FLoops.SequentialEx()
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false),
                 check_sol = (; allow_local = true, allow_almost = true))
    nco = NestedClustered(; opti = InverseVolatility(), opto = InverseVolatility(),
                          ex = seq)
    r = optimise(nco, rd)
    K = r.clr.k
    cls = [findall(==(k), PO.assignments(r.clr)) for k in 1:K]

    @testset "the weights are W v, and the synthetic returns carry the reduced moments" begin
        @test K >= 2
        @test sort(reduce(vcat, cls)) == 1:N
        W = zeros(N, K)
        for k in 1:K
            W[cls[k], k] = r.resi[k].w
        end
        # Inverse volatility inside each cluster, written from the definition.
        sd = vec(std(X; dims = 1))
        Wo = zeros(N, K)
        for k in 1:K
            v = inv.(sd[cls[k]])
            Wo[cls[k], k] = v / sum(v)
        end
        @test W ≈ Wo
        Xo = X * W
        @test r.reso.pr.X ≈ Xo
        vo = inv.(vec(std(Xo; dims = 1)))
        @test r.reso.w ≈ vo / sum(vo)
        @test r.w ≈ W * r.reso.w
        @test sum(r.w) ≈ 1
        # The sample moments of X W are the reduced moments of the clusters.
        @test vec(mean(Xo; dims = 1)) ≈ transpose(W) * vec(mean(X; dims = 1))
        @test cov(Xo) ≈ transpose(W) * cov(X) * W
        @test r.retcode isa OptimisationSuccess
    end

    @testset "the outer optimiser names a cluster by its synthetic asset" begin
        osets = UniverseSets(; dict = Dict("nx" => ["X", "Y"]))
        # An outer optimiser whose `opt` holds the sets.
        opto = MeanRisk(;
                        opt = JuMPOptimiser(; slv = slv, sets = osets,
                                            lcse = LinearConstraintEstimator(;
                                                                             val = ["_2 >= 0.8"])))
        r1 = optimise(NestedClustered(; opti = InverseVolatility(), opto = opto, ex = seq),
                      rd)
        @test r1.retcode isa OptimisationSuccess
        @test r1.reso.w[2] >= 0.8 - 1e-6
        @test isapprox(r1.reso.w[2], 0.8; atol = 1e-4)
        # An outer optimiser that holds the sets itself.
        opto = InverseVolatility(; sets = osets,
                                 wb = WeightBoundsEstimator(; lb = ["_1" => 0.7],
                                                            ub = nothing))
        r2 = optimise(NestedClustered(; opti = InverseVolatility(), opto = opto, ex = seq),
                      rd)
        @test r2.retcode isa OptimisationSuccess
        @test r2.reso.w[1] ≈ 0.7
        W2 = zeros(N, K)
        for k in 1:K
            W2[cls[k], k] = r2.resi[k].w
        end
        @test r2.w ≈ W2 * r2.reso.w
        # The caller's sets do not change.
        @test osets.dict["nx"] == ["X", "Y"]
        # Sets that already hold the vector of the synthetic names are kept as they are.
        rdo = ReturnsResult(; nx = ["_1", "_2"], X = X[:, 1:2])
        same = UniverseSets(; dict = Dict("nx" => rdo.nx))
        n3 = NestedClustered(; opti = InverseVolatility(),
                             opto = InverseVolatility(; sets = same))
        @test PO._update_asset_sets(n3, rdo) === n3
        @test PO._update_asset_sets(nco, rdo) === nco
    end

    @testset "the weight bounds of the meta-optimiser bind the final weights" begin
        rb = optimise(NestedClustered(; opti = InverseVolatility(),
                                      opto = InverseVolatility(),
                                      wb = WeightBounds(; lb = 0.0, ub = 0.128), ex = seq),
                      rd)
        @test maximum(r.w) > 0.128
        @test rb.retcode isa OptimisationSuccess
        @test maximum(rb.w) ≈ 0.128
        @test sum(rb.w) ≈ 1
        # Eight assets at most 0.12 each cannot sum to one.
        rf = optimise(NestedClustered(; opti = InverseVolatility(),
                                      opto = InverseVolatility(),
                                      wb = WeightBounds(; lb = 0.0, ub = 0.12), ex = seq),
                      rd)
        @test rf.retcode isa OptimisationFailure
        @test rf.retcode.res.msg == "weight bounds finalisation failed.\n"
    end

    @testset "set_retcode keeps the expanded weights" begin
        Xn = copy(X)
        Xn[:, 3] .= NaN
        rn = optimise(nco, ReturnsResult(; nx = nx, X = Xn))
        @test rn.imsk == [true, true, false, true, true, true, true, true]
        @test length(rn.w) == N
        @test iszero(rn.w[3])
        @test sum(rn.w) ≈ 1
        rs = PO.set_retcode(rn, OptimisationFailure())
        @test rs.retcode isa OptimisationFailure
        @test rs.w === rn.w
        @test rs.imsk === rn.imsk
        @test PO.result_investable_mask(rs) === rn.imsk
    end

    @testset "holds_precomputed reads a slot in each shape" begin
        lc = linear_constraints(LinearConstraintEstimator(; val = ["A >= 0.05"]),
                                UniverseSets(; dict = Dict("nx" => nx)))
        lce = LinearConstraintEstimator(; val = ["A >= 0.05"])
        LC = LinearConstraint
        @test PO.holds_precomputed(lc, LC)
        @test !PO.holds_precomputed(lce, LC)
        @test !PO.holds_precomputed(nothing, LC)
        @test PO.holds_precomputed([lce, lc], LC)
        @test !PO.holds_precomputed([lce, lce], LC)
        @test PO.holds_precomputed(TimeDependent([lce, lc]), LC)
        @test PO.holds_precomputed(TimeDependent([lce, lce]; default = lc), LC)
        @test PO.holds_precomputed(TimeDependent([lce, [lce, lc]]), LC)
        @test !PO.holds_precomputed(TimeDependent([lce, lce]), LC)
        # A callable schedule has no value until it runs, so only its default is read.
        @test !PO.holds_precomputed(TimeDependent(ctx -> lc), LC)
        @test PO.holds_precomputed(TimeDependent(ctx -> lce; default = lc), LC)
    end

    @testset "the refusals read a schedule" begin
        lc = linear_constraints(LinearConstraintEstimator(; val = ["A >= 0.05"]),
                                UniverseSets(; dict = Dict("nx" => nx)))
        inner(lcse) = MeanRisk(; opt = JuMPOptimiser(; slv = slv, lcse = lcse))
        @test_throws ArgumentError NestedClustered(; opti = inner(lc),
                                                   opto = InverseVolatility())
        # A schedule of precomputed constraints is refused at construction too.
        @test_throws ArgumentError NestedClustered(; opti = inner(TimeDependent([lc, lc])),
                                                   opto = InverseVolatility())
        @test_throws ArgumentError NestedClustered(;
                                                   opti = inner(TimeDependent([lc, lc];
                                                                              default = lc)),
                                                   opto = InverseVolatility())
        # So is a schedule of clustering results in a nested clustering optimiser.
        hrp = HierarchicalRiskParity(;
                                     opt = HierarchicalOptimiser(;
                                                                 cle = TimeDependent([r.clr,
                                                                                      r.clr])))
        @test_throws ArgumentError NestedClustered(; opti = hrp, opto = InverseVolatility())
        @test_throws ArgumentError NestedClustered(; opti = InverseVolatility(),
                                                   opto = NestedClustered(;
                                                                          cle = TimeDependent([r.clr]),
                                                                          opti = InverseVolatility(),
                                                                          opto = InverseVolatility()))
        # The same optimisers pass with estimators in those slots.
        @test NestedClustered(; opti = HierarchicalRiskParity(),
                              opto = InverseVolatility()) isa NestedClustered
        cv = KFold(; n = 2)
        @test NestedClustered(; opti = InverseVolatility(),
                              opto = NestedClustered(; opti = InverseVolatility(),
                                                     opto = InverseVolatility(), cv = cv)) isa
              NestedClustered
        # A precomputed risk contribution constraint of a variance in a vector of measures.
        mr(rs) = MeanRisk(; r = rs, opt = JuMPOptimiser(; slv = slv))
        @test NestedClustered(; opti = mr([Variance(), ConditionalValueatRisk()]),
                              opto = InverseVolatility()) isa NestedClustered
        @test_throws ArgumentError NestedClustered(;
                                                   opti = mr([Variance(; rc = lc),
                                                              ConditionalValueatRisk()]),
                                                   opto = InverseVolatility())
        # The constructor refuses a stated basis in a schedule of the outer lcse slot.
        F = X[:, 1:2] * [1.0 0.5; 0.2 1.0] .+ randn(StableRNG(5), T, 2) ./ 1000
        rdf = ReturnsResult(; nx = nx, X = X, nf = ["f1", "f2"], F = F)
        rr = regression(StepwiseRegression(), rdf)
        @test PO.stated_constraint_space_basis(TimeDependent([FactorSpace(; re = rr)]))
        @test !PO.stated_constraint_space_basis(TimeDependent([FactorSpace()]))
        # A precomputed regression in a factor risk budget of the outer optimiser.
        rb(rba) = RiskBudgeting(; opt = JuMPOptimiser(; slv = slv), rba = rba)
        @test_throws ArgumentError NestedClustered(; opti = InverseVolatility(),
                                                   opto = rb(FactorRiskBudgeting(; re = rr)))
        @test_throws ArgumentError NestedClustered(; opti = InverseVolatility(),
                                                   opto = rb(TimeDependent([FactorRiskBudgeting(;
                                                                                                re = rr)])))
        @test NestedClustered(; opti = InverseVolatility(),
                              opto = rb(FactorRiskBudgeting())) isa NestedClustered
    end

    @testset "an estimator in wb or fees needs sets" begin
        @test_throws IsNothingError NestedClustered(; opti = InverseVolatility(),
                                                    opto = InverseVolatility(),
                                                    wb = WeightBoundsEstimator(;
                                                                               lb = ["A" =>
                                                                                         0.1],
                                                                               ub = nothing))
        @test_throws IsNothingError NestedClustered(; opti = InverseVolatility(),
                                                    opto = InverseVolatility(),
                                                    fees = FeesEstimator(;
                                                                         l = ["A" => 0.1]))
    end

    @testset "a static optimiser passes through the fold resolution" begin
        ctx = TimeDependentContext(; i = 2, n = 2, rd = rd, train_idx = [1:20, 1:40],
                                   test_idx = [21:40, 41:60])
        @test !PO.is_time_dependent(nco)
        @test PO.update_time_dependent_estimator(nco, ctx) === nco
    end
end
