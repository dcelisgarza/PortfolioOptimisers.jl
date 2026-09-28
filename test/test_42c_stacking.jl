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
    cv = OptimisationCrossValidation(; cv = KFold(; n = 4))
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
                                res.retcode, res.w, res.imsk, res.fb).w === res.w
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
