@testset "Subset Resampling Optimisation" begin
    using PortfolioOptimisers, CSV, Test, TimeSeries, Clarabel, DataFrames, StableRNGs,
          Pajarito, HiGHS, JuMP, Clustering, NearestCorrelationMatrix
    rd = prices_to_returns(price_ingestion(PriceIngestion(),
                                           TimeArray(CSV.File(joinpath(@__DIR__,
                                                                       "./assets/SP500.csv.gz"));
                                                     timestamp = :Date)[(end - 252):end];
                                           F = TimeArray(CSV.File(joinpath(@__DIR__,
                                                                           "./assets/Factors.csv.gz"));
                                                         timestamp = :Date)[(end - 252):end],
                                           B = TimeArray(CSV.File(joinpath(@__DIR__,
                                                                           "./assets/SP500_idx.csv.gz"));
                                                         timestamp = :Date)[(end - 252):end]))
    slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false)),
           Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.95)),
           Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.9)),
           Solver(; name = :clarabel4, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.85)),
           Solver(; name = :clarabel5, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.80)),
           Solver(; name = :clarabel6, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.75)),
           Solver(; name = :clarabel7, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.7)),
           Solver(; name = :clarabel8, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.6,
                                  "max_iter" => 1500, "tol_gap_abs" => 1e-4,
                                  "tol_gap_rel" => 1e-4, "tol_ktratio" => 1e-3,
                                  "tol_feas" => 1e-4, "tol_infeas_abs" => 1e-4,
                                  "tol_infeas_rel" => 1e-4, "reduced_tol_gap_abs" => 1e-4,
                                  "reduced_tol_gap_rel" => 1e-4,
                                  "reduced_tol_ktratio" => 1e-3, "reduced_tol_feas" => 1e-4,
                                  "reduced_tol_infeas_abs" => 1e-4,
                                  "reduced_tol_infeas_rel" => 1e-4))]
    pr = prior(EmpiricalPrior(), rd)
    N = size(rd.X, 2)
    w0 = fill(inv(N), N)
    jopt = JuMPOptimiser(; slv = slv)
    mr = MeanRisk(; opt = jopt)

    opt = SubsetResampling(; subset_size = 1, n_subsets = 21, pe = pr, opt = mr)
    @test_throws ArgumentError optimise(opt, rd)

    opt = SubsetResampling(; subset_size = 1, n_subsets = 20, pe = pr, opt = mr)
    res = optimise(opt, rd)
    @test isapprox(res.w, w0)

    opt = SubsetResampling(; subset_size = 19, n_subsets = 20, pe = pr, opt = mr)
    res = optimise(opt, rd)
    @test isapprox(res.w,
                   [1.9735716605490405e-7, 1.301758617627138e-7, 1.3231751768994914e-6,
                    2.3240305901612208e-7, 0.07565589816687646, 0.007899454085387779,
                    7.324038826272394e-7, 0.3598290471255588, 0.008662230014517738,
                    0.11138846116218364, 4.4789191031505297e-7, 0.17483414950533954,
                    3.49130827153516e-7, 0.09504168533155483, 5.62731575232341e-5,
                    0.02915541034806286, 2.0091277189574918e-7, 9.176942286216526e-7,
                    0.09036630694773816, 0.04710655301037257], rtol = 1e-6)

    mr_res = optimise(mr, rd)
    @test isapprox(res.w, mr_res.w, rtol = 0.05)

    jopt = JuMPOptimiser(; slv = slv,
                         ret = ArithmeticReturn(;
                                                settings = JuMPReturnsSettings(;
                                                                               lb = Frontier(5))))
    mr = MeanRisk(; opt = jopt)

    opt = SubsetResampling(; subset_size = eps(), n_subsets = 20, pe = pr, opt = mr)
    res = optimise(opt, rd)
    @test all(x -> isapprox(x, w0), res.w)

    opt = SubsetResampling(; subset_size = 0.95, n_subsets = 20, pe = pr, opt = mr)
    res = optimise(opt, rd)
    df = CSV.read(joinpath(@__DIR__, "./assets/SubsetResamplingFrontier.csv.gz"), DataFrame)
    success = isapprox(Matrix(df), reduce(hcat, res.w); rtol = 1e-6)
    if !success
        find_tol(Matrix(df), reduce(hcat, res.w))
    end
    @test success

    mr_res = optimise(mr, rd)
    success = isapprox(res.w, mr_res.w; rtol = 0.1)
    if !success
        find_tol(res.w, mr_res.w)
    end
    @test success
end
@testset "The docstrings of 04_SubsetResampling.jl against numbers" begin
    using PortfolioOptimisers, Test, StableRNGs, Clarabel, FLoops, Logging
    seq = FLoops.SequentialEx()
    PO = PortfolioOptimisers
    X = 0.01 .* randn(StableRNG(7), 80, 12) .+ 0.0005
    rd = ReturnsResult(; nx = ["A$i" for i in 1:12], X = X)
    rd6 = ReturnsResult(; nx = rd.nx[1:6], X = X[:, 1:6])
    function subset_average(res, X, w_of)
        N = size(X, 2)
        M = size(res.idx, 2)
        w = zeros(N)
        for m in 1:M
            S = res.idx[:, m]
            w[S] .+= w_of(m, S)
        end
        return w ./ M
    end
    @testset "the average of the subset weights" begin
        sr = SubsetResampling(; subset_size = 0.8, opt = InverseVolatility(), n_subsets = 7,
                              ex = seq, seed = 11)
        res = optimise(sr, rd)
        # `round(0.8 * 12) = 10`.
        @test size(res.idx) == (10, 7)
        @test length(unique(eachcol(res.idx))) == 7
        @test all(issorted, eachcol(res.idx))
        oracle = subset_average(res, X,
                                (m, S) -> optimise(InverseVolatility(),
                                                   ReturnsResult(; nx = rd.nx[S],
                                                                 X = X[:, S])).w)
        @test isapprox(res.w, oracle; rtol = 1e-14)
        @test sum(res.w) ≈ 1
        @test isa(res.retcode, OptimisationSuccess)
        # The generic `optimise` gives the same result when `fb` is `nothing`.
        g = invoke(optimise, Tuple{PO.OptimisationEstimator, Vararg{Any}}, sr, rd)
        @test g.w == res.w && g.idx == res.idx
        # An asset in no subset has zero weight.
        res2 = optimise(SubsetResampling(; opt = InverseVolatility(), subset_size = 2,
                                         n_subsets = 2, ex = seq, seed = 3), rd)
        out = setdiff(1:12, vec(res2.idx))
        @test !isempty(out) && all(iszero, res2.w[out])
    end
    @testset "the efficient frontier averages each point" begin
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("verbose" => false))
        mr = MeanRisk(;
                      opt = JuMPOptimiser(; slv = slv,
                                          ret = ArithmeticReturn(;
                                                                 settings = JuMPReturnsSettings(;
                                                                                                lb = Frontier(3)))))
        res = optimise(SubsetResampling(; opt = mr, subset_size = 4, n_subsets = 3,
                                        ex = seq, seed = 4), rd6)
        @test length(res.w) == 3
        @test all(x -> isa(x, OptimisationSuccess), res.retcode)
        for j in 1:3
            @test isapprox(res.w[j],
                           subset_average(res, X[:, 1:6], (m, S) -> res.ress[m].w[j]);
                           rtol = 1e-14)
        end
    end
    @testset "a point of the frontier reads its own codes, and a fallback stands at each" begin
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("verbose" => false))
        mu = vec(mean(X[:, 1:6]; dims = 1))
        function swept(lb; kwargs...)
            ret = ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = lb))
            mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv, ret = ret), kwargs...)
            return optimise(SubsetResampling(; opt = mr, subset_size = 4, n_subsets = 5,
                                             ex = seq, seed = 4), rd6)
        end
        idx = swept([-1.0, -1.0]).idx
        tops = [maximum(mu[idx[:, i]]) for i in axes(idx, 2)]
        # No subset reaches the second level, so that point fails alone and by name.
        res = swept([-1.0, 1.0])
        @test isa(res.retcode[1], OptimisationSuccess)
        @test res.retcode[2].res.msg == "opti failed.\nweight bounds finalisation failed.\n"
        @test all(x -> isa(x, OptimisationFailure), res.retcode[2].res.opti)
        @test all(isfinite, res.w[1]) && all(isnan, res.w[2][unique(idx)])
        # With a fallback the subset keeps its first point. Its second point still fails.
        rfb = swept([-1.0, 1.0]; fb = EqualWeighted())
        @test rfb.w[1] == res.w[1] && !PO.any_solved(rfb.retcode[2])
        # Levels that a part of the subsets reach. Such a subset answers with a frontier, and
        # the other subsets answer with one portfolio, which stands at each point.
        @test !allequal(tops)
        level = (minimum(tops) + maximum(tops)) / 2
        mix = swept([level, level - 1e-7]; fb = EqualWeighted())
        single = [isa(x.w, PO.VecNum) for x in mix.ress]
        @test single == (tops .< level) && any(single) && !all(single)
        @test PO.fold_solved(mix.retcode) && length(mix.w) == 2
        for j in 1:2
            @test isapprox(mix.w[j],
                           subset_average(mix, X[:, 1:6],
                                          (m, S) -> PO.member_vector(mix.ress[m].w, j));
                           rtol = 1e-14)
        end
        # The subsets must agree on the number of points.
        @test_throws DimensionMismatch PO.subset_resampling_finaliser(6, 5, mix.idx, mix.wb,
                                                                      IterativeWeightFinaliser(),
                                                                      FractionOf(1),
                                                                      mix.ress,
                                                                      [zeros(4)
                                                                       for _ in 1:3])
    end
    @testset "the subset size" begin
        pr5 = prior(EmpiricalPrior(), ReturnsResult(; nx = rd.nx[1:5], X = X[:, 1:5]))
        pr6 = prior(EmpiricalPrior(), rd6)
        # `round` takes the even integer at a tie: `0.5 * 5 = 2.5` gives 2.
        @test PO.get_subset_size(FractionOf(0.5), pr5) == 2
        @test PO.get_subset_size(FractionOf(eps()), pr6) == 1
        @test PO.get_subset_size(pr -> 3, pr6) == 3
        @test PO.get_subset_size(FractionOf(1 // 2), pr6) == 3
        @test PO.get_subset_size(CountOf(4), pr6) == 4
        @test_throws ArgumentError PO.get_subset_size(CountOf(7), pr6)
        # A `Rational` size is a fraction of the assets, as a float is.
        rq = optimise(SubsetResampling(; opt = InverseVolatility(), subset_size = 1 // 2,
                                       ex = seq, seed = 2), rd6)
        rf = optimise(SubsetResampling(; opt = InverseVolatility(), subset_size = 0.5,
                                       ex = seq, seed = 2), rd6)
        @test rq.w == rf.w && size(rq.idx, 1) == 3
        @test_throws DomainError SubsetResampling(; opt = InverseVolatility(),
                                                  subset_size = 3 // 2)
        @test_throws TypeError SubsetResampling(; opt = InverseVolatility(),
                                                subset_size = 0.5 + 0im)
        wf = IndexWalkForward(20, 10)
        @test MultipleRandomised(wf; subset_size = 1 // 2).subset_size ===
              FractionOf(1 // 2)
        @test_throws DomainError MultipleRandomised(wf; subset_size = 3 // 2)
        @test_throws DomainError MultipleRandomised(wf; window_size = 3 // 2)
        @test PO.get_window_size(FractionOf(1 // 2), rd6) == 40
        # Issue #2013: the plain form is read by its type, and the wrapper keeps its own.
        @test MultipleRandomised(wf; subset_size = 2, window_size = 30).window_size ===
              CountOf(30)
        @test PO.get_window_size(CountOf(30), rd6) == 30
        @test_throws DomainError MultipleRandomised(wf; window_size = CountOf(1))
        @test SubsetResampling(; opt = InverseVolatility(), subset_size = 3).subset_size ===
              CountOf(3)
        @test_throws DomainError SubsetResampling(; opt = InverseVolatility(),
                                                  subset_size = 1.0)
        # Issue #2013: a search grid sets each value through the constructor. Julia
        # promotes the plain `[1, 0.5]` to `[1.0, 0.5]`, and the plain `1.0` is refused,
        # where the wrappers keep each value at its own member.
        sr0 = SubsetResampling(; opt = InverseVolatility())
        lens = (PO.Accessors.PropertyLens(:subset_size),)
        @test eltype([1, 0.5]) == Float64
        @test_throws DomainError PO.search_candidate(sr0, lens, ([1, 0.5][1],))
        vals = [CountOf(1), FractionOf(0.5)]
        @test [PO.search_candidate(sr0, lens, (v,)).subset_size for v in vals] == vals
        # R43 of #1817: the literal of a whole grid promotes its pairs as well, to
        # `Pair{String, Vector}`, and each value keeps its member through a full search.
        grid = ["subset_size" => [CountOf(2), FractionOf(0.5)], "n_subsets" => [2, 3]]
        @test eltype(grid) == Pair{String, Vector}
        glenses, gvals = PO.lens_val_grid(grid)
        @test first.(gvals) == [CountOf(2), FractionOf(0.5), CountOf(2), FractionOf(0.5)]
        @test [PO.get_subset_size(PO.search_candidate(sr0, l, v).subset_size, rd6)
               for (l, v) in zip(glenses, gvals)] == [2, 3, 2, 3]
        gscv = GridSearchCrossValidation(grid; cv = KFold(; n = 3),
                                         r = ConditionalValueatRisk())
        res = search_cross_validation(SubsetResampling(; opt = InverseVolatility(),
                                                       ex = seq, seed = 1), gscv, rd6)
        @test size(res.test_scores) == (3, 4)
        @test res.opt.subset_size in (CountOf(2), FractionOf(0.5))
    end
    @testset "the default subset size is a power of the number of assets" begin
        # shen2017 sizes a subset as `n^0.8` in its simulation study and as `n^0.7` in its
        # empirical study: `100^0.8 ≈ 39.8` and `100^0.7 ≈ 25.1`.
        X100 = 0.01 .* randn(StableRNG(3), 60, 100) .+ 0.001
        rd100 = ReturnsResult(; nx = ["A$i" for i in 1:100], X = X100)
        @test PowerSubsetSize() === PowerSubsetSize(; p = 0.8)
        @test PowerSubsetSize()(rd100) == 40
        @test PowerSubsetSize(; p = 0.7)(rd100) == 25
        @test PowerSubsetSize(; p = 1)(rd100) == 100
        @test PowerSubsetSize(; p = 1 // 2)(rd6) == 2
        @test PO.get_subset_size(PowerSubsetSize(), rd6) == 4
        @test_throws DomainError PowerSubsetSize(; p = 0)
        @test_throws DomainError PowerSubsetSize(; p = 1.5)
        sr = SubsetResampling(; opt = InverseVolatility(), ex = seq, seed = 1)
        @test sr.subset_size === PowerSubsetSize()
        @test PO.time_dependent_field_defaults(sr).subset_size === PowerSubsetSize()
        res = optimise(sr, rd100)
        @test isa(res.retcode, OptimisationSuccess) && size(res.idx) == (40, 2)
        @test sum(res.w) ≈ 1
        rp = optimise(SubsetResampling(; opt = InverseVolatility(), subset_size = 40,
                                       ex = seq, seed = 1), rd100)
        @test rp.idx == res.idx && rp.w == res.w
    end
    @testset "the subset draw" begin
        # The combination count overflows a machine integer, and the draw is approximate.
        sub = @test_logs (:warn, r"approximate") match_mode = :any min_level = Logging.Warn PO.sample_unique_assets(100,
                                                                                                                    80,
                                                                                                                    3;
                                                                                                                    rng = StableRNG(1))
        @test size(sub) == (80, 3)
        @test all(c -> issorted(c) && allunique(c) && all(in(1:100), c), eachcol(sub))
        X100 = 0.01 .* randn(StableRNG(3), 60, 100) .+ 0.001
        rd100 = ReturnsResult(; nx = ["A$i" for i in 1:100], X = X100)
        res = optimise(SubsetResampling(; subset_size = 0.8, opt = InverseVolatility(),
                                        ex = seq, seed = 1), rd100)
        @test isa(res.retcode, OptimisationSuccess) && size(res.idx) == (80, 2)
        @test sum(res.w) ≈ 1
        # The exact route draws the same subsets as ranks of the combinations.
        rng = StableRNG(5)
        ranks = PO.StatsBase.sample(rng, 1:binomial(12, 4), 5; replace = false)
        @test PO.sample_unique_assets(12, 4, 5; rng = StableRNG(5)) ==
              reduce(hcat, [PO.combination_by_index(r, 12, 4) for r in ranks])
        @test_throws ArgumentError PO.sample_unique_assets(5, 1, 6)
        @test_throws ArgumentError optimise(SubsetResampling(; opt = InverseVolatility(),
                                                             subset_size = 1, n_subsets = 7,
                                                             ex = seq), rd6)
    end
    @testset "the data types and the investable universe" begin
        Xi = rand(StableRNG(9), -3:5, 40, 6)
        sr = SubsetResampling(; opt = InverseVolatility(), subset_size = 4, n_subsets = 3,
                              ex = seq, seed = 2)
        ri = optimise(sr, ReturnsResult(; nx = rd6.nx, X = Xi))
        rfl = optimise(sr, ReturnsResult(; nx = rd6.nx, X = Float64.(Xi)))
        @test eltype(ri.w) == Float64 && ri.idx == rfl.idx && ri.w == rfl.w
        r32 = optimise(sr, ReturnsResult(; nx = rd6.nx, X = Float32.(X[:, 1:6])))
        @test eltype(r32.w) == Float32
        Xn = copy(X[:, 1:8])
        Xn[:, 3] .= NaN
        rn = optimise(SubsetResampling(; opt = InverseVolatility(), subset_size = 4,
                                       n_subsets = 4, ex = seq, seed = 2),
                      ReturnsResult(; nx = rd.nx[1:8], X = Xn))
        @test length(rn.w) == 8 && iszero(rn.w[3]) && sum(rn.w) ≈ 1
        @test rn.imsk == [true, true, false, true, true, true, true, true]
        @test maximum(rn.idx) <= 7
        @test PO.result_investable_mask(rn) === rn.imsk
    end
    @testset "min_solved averages the subsets that solved" begin
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("verbose" => false))
        mu = vec(mean(X[:, 1:6]; dims = 1))
        function swept(lb, min_solved; kwargs...)
            ret = ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = lb))
            mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv, ret = ret))
            return optimise(SubsetResampling(; opt = mr, subset_size = 4, n_subsets = 5,
                                             ex = seq, seed = 4, min_solved = min_solved,
                                             kwargs...), rd6)
        end
        all5 = swept(-1.0, FractionOf(1))
        @test isnothing(all5.retcode.res)
        # The subsets are the same in each run, so a run with no drop is the same run.
        @test swept(-1.0, CountOf(1)).w == all5.w
        tops = [maximum(mu[all5.idx[:, i]]) for i in axes(all5.idx, 2)]
        @test !allequal(tops)
        level = (minimum(tops) + maximum(tops)) / 2
        ok = tops .> level
        n = count(ok)
        @test 0 < n < 5
        # By default the one failed subset fails the average, and its fallback replaces it.
        @test isa(swept(level, FractionOf(1)).retcode, OptimisationFailure)
        fbr = swept(level, FractionOf(1); fb = EqualWeighted())
        @test fbr.w ≈ fill(1 / 6, 6) && !isnothing(fbr.fb)
        res = @test_logs (:warn, r"meta-optimiser drops") match_mode = :any swept(level,
                                                                                  CountOf(n))
        @test isa(res.retcode, OptimisationSuccess)
        @test res.retcode.res.msg == "opti failed.\n"
        @test PO.any_solved.(res.retcode.res.opti) == ok
        avg = zeros(6)
        for m in findall(ok)
            avg[res.idx[:, m]] .+= res.ress[m].w
        end
        @test res.w ≈ avg / n && sum(res.w) ≈ 1
        # A fraction compares the share, and one more subset than solved is too many.
        @test swept(level, n / 5).w == res.w
        @test isa(swept(level, CountOf(n + 1)).retcode, OptimisationFailure)
        # Each point of a frontier drops its own subsets, and no value accepts none.
        fr = swept([-1.0, level, 1.0], CountOf(1))
        @test [PO.any_solved(rc) for rc in fr.retcode] == [true, true, false]
        @test isnothing(fr.retcode[1].res)
        @test isapprox(fr.w[2], res.w; rtol = 1e-3)
        @test sum(fr.w[1]) ≈ 1 && sum(fr.w[2]) ≈ 1
        @test fr.retcode[3].res.msg == "opti failed.\nweight bounds finalisation failed.\n"
        # The field takes a schedule, and a view keeps it.
        sr = SubsetResampling(; opt = EqualWeighted(),
                              min_solved = TimeDependent([CountOf(1), 0.5]))
        @test PO.reset_time_dependent_estimator(sr).min_solved == FractionOf(1)
        sr = SubsetResampling(; opt = EqualWeighted(), min_solved = 2)
        @test PO.port_opt_view(sr, 1:3, X).min_solved == CountOf(2)
        @test_throws DomainError SubsetResampling(; opt = EqualWeighted(), min_solved = 1.0)
    end
    @testset "the return codes and the fallback chain" begin
        # No weights meet `sum(w) == 1` under an upper bound of 0.1 on six assets.
        sr = SubsetResampling(; opt = InverseVolatility(),
                              wb = WeightBounds(; lb = 0.0, ub = 0.1), subset_size = 4,
                              n_subsets = 3, ex = seq, seed = 4)
        res = optimise(sr, rd6)
        @test isa(res.retcode, OptimisationFailure)
        @test res.retcode.res.msg == "weight bounds finalisation failed.\n"
        @test all(x -> isa(x, OptimisationSuccess), res.retcode.res.opti)
        fail = OptimisationFailure(; res = "x")
        @test PO.subset_resampling_retcode([x.retcode for x in res.ress],
                                           OptimisationSuccess(), false) ==
              OptimisationSuccess()
        bad = PO.set_retcode(res, fail)
        rc = PO.subset_resampling_retcode([bad.retcode, res.ress[2].retcode], fail, true)
        @test isa(rc, OptimisationFailure)
        @test rc.res.msg == "opti failed.\nweight bounds finalisation failed.\n"
        @test rc.res.opti[1] === fail && rc.res.wb === fail
        @test_throws MethodError PO.subset_resampling_finaliser(6, 3, res.idx, nothing,
                                                                sr.wf, sr.min_solved,
                                                                res.ress, 1)
        rs = PO.set_retcode(res, OptimisationSuccess())
        @test rs.retcode == OptimisationSuccess() && rs.w === res.w && rs.idx === res.idx
        fb = Tuple{PO.OptimisationEstimator, PO.OptimisationResult}[(sr, res)]
        rf = PO.factory(res, fb)
        @test rf.fb === fb && rf.w === res.w && rf.retcode === res.retcode
        rfb = optimise(SubsetResampling(; opt = InverseVolatility(),
                                        wb = WeightBounds(; lb = 0.0, ub = 0.1),
                                        subset_size = 4, n_subsets = 3, ex = seq, seed = 4,
                                        fb = InverseVolatility()), rd6)
        @test isa(rfb.retcode, OptimisationSuccess)
        @test isa(rfb.fb[1][2], SubsetResamplingResult)
    end
    @testset "the constructor checks" begin
        @test_throws IsNothingError SubsetResampling(; subset_size = 0.8,
                                                     opt = InverseVolatility(),
                                                     wb = WeightBoundsEstimator(; lb = 0.0))
        @test_throws IsNothingError SubsetResampling(; subset_size = 0.8,
                                                     opt = InverseVolatility(),
                                                     fees = FeesEstimator(; l = 0.001))
        @test_throws DomainError SubsetResampling(; opt = InverseVolatility(),
                                                  subset_size = 0)
        @test_throws DomainError SubsetResampling(; subset_size = 0.8,
                                                  opt = InverseVolatility(), n_subsets = 1)
        @test_throws DomainError SubsetResampling(; subset_size = 0.8,
                                                  opt = InverseVolatility(), max_comb = 0)
    end
    @testset "the schedules" begin
        ctx(i) = TimeDependentContext(; i = i, n = 2, rd = rd6, train_idx = [1:20, 1:40],
                                      test_idx = [21:40, 41:60])
        static = SubsetResampling(; subset_size = 0.8, opt = InverseVolatility())
        @test PO.update_time_dependent_estimator(static, ctx(1)) === static
        @test !PO.is_time_dependent(static)
        @test !PO.needs_previous_weights(static)
        sr = SubsetResampling(; subset_size = 0.8, opt = InverseVolatility(),
                              n_subsets = TimeDependent([3, 4], :outermost))
        @test PO.is_time_dependent(sr)
        @test PO.update_time_dependent_estimator(sr, ctx(1)).n_subsets == 3
        @test PO.update_time_dependent_estimator(sr, ctx(2)).n_subsets == 4
        @test PO.reset_time_dependent_estimator(sr).n_subsets == 2
        sd = SubsetResampling(; subset_size = 0.8, opt = InverseVolatility(),
                              n_subsets = TimeDependent([3, 4], :outermost; default = 5))
        @test PO.reset_time_dependent_estimator(sd).n_subsets == 5
        sf = SubsetResampling(; subset_size = 0.8, opt = InverseVolatility(),
                              fb = TimeDependent([InverseVolatility(), EqualWeighted()],
                                                 :outermost))
        @test isnothing(PO.reset_time_dependent_estimator(sf).fb)
        @test_throws TimeDependentDefaultError PO.reset_time_dependent_estimator(SubsetResampling(;
                                                                                                  subset_size = 0.8,
                                                                                                  opt = TimeDependent([InverseVolatility(),
                                                                                                                       EqualWeighted()],
                                                                                                                      :outermost)))
        @test keys(PO.time_dependent_field_defaults(sr)) ==
              (:pe, :opt, :wf, :min_solved, :subset_size, :n_subsets)
    end
    @testset "the view" begin
        pr6 = prior(EmpiricalPrior(), rd6)
        sr = SubsetResampling(; subset_size = 0.8, pe = pr6, opt = InverseVolatility(),
                              fb = EqualWeighted())
        v = PO.port_opt_view(sr, [2, 4, 5], zeros(3, 3))
        @test v.pe.X == pr6.X[:, [2, 4, 5]]
        @test isa(v.fb, EqualWeighted) && v.subset_size === sr.subset_size
    end
end
