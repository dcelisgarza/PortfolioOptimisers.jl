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
        sr = SubsetResampling(; opt = InverseVolatility(), n_subsets = 7, ex = seq,
                              seed = 11)
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
    @testset "the subset size" begin
        pr5 = prior(EmpiricalPrior(), ReturnsResult(; nx = rd.nx[1:5], X = X[:, 1:5]))
        pr6 = prior(EmpiricalPrior(), rd6)
        # `round` takes the even integer at a tie: `0.5 * 5 = 2.5` gives 2.
        @test PO.get_subset_size(0.5, pr5) == 2
        @test PO.get_subset_size(eps(), pr6) == 1
        @test PO.get_subset_size(pr -> 3, pr6) == 3
        @test PO.get_subset_size(1 // 2, pr6) == 3
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
        @test MultipleRandomised(wf; subset_size = 1 // 2).subset_size == 1 // 2
        @test_throws DomainError MultipleRandomised(wf; subset_size = 3 // 2)
        @test_throws DomainError MultipleRandomised(wf; window_size = 3 // 2)
        @test PO.get_window_size(1 // 2, rd6) == 40
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
        res = optimise(SubsetResampling(; opt = InverseVolatility(), ex = seq, seed = 1),
                       rd100)
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
        @test PO.subset_resampling_retcode(res.ress, OptimisationSuccess()) ==
              OptimisationSuccess()
        bad = PO.set_retcode(res, fail)
        rc = PO.subset_resampling_retcode([bad, res.ress[2]], fail)
        @test rc.res.msg == "opti failed.\nweight bounds finalisation failed.\n"
        @test rc.res.opti[1] === fail && rc.res.wb === fail
        @test_throws MethodError PO.subset_resampling_finaliser(6, 3, res.idx, nothing,
                                                                sr.wf, res.ress,
                                                                res.ress[1].w)
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
        @test_throws IsNothingError SubsetResampling(; opt = InverseVolatility(),
                                                     wb = WeightBoundsEstimator(; lb = 0.0))
        @test_throws IsNothingError SubsetResampling(; opt = InverseVolatility(),
                                                     fees = FeesEstimator(; l = 0.001))
        @test_throws DomainError SubsetResampling(; opt = InverseVolatility(),
                                                  subset_size = 0)
        @test_throws DomainError SubsetResampling(; opt = InverseVolatility(),
                                                  n_subsets = 1)
        @test_throws DomainError SubsetResampling(; opt = InverseVolatility(), max_comb = 0)
    end
    @testset "the schedules" begin
        ctx(i) = TimeDependentContext(; i = i, n = 2, rd = rd6, train_idx = [1:20, 1:40],
                                      test_idx = [21:40, 41:60])
        static = SubsetResampling(; opt = InverseVolatility())
        @test PO.update_time_dependent_estimator(static, ctx(1)) === static
        @test !PO.is_time_dependent(static)
        @test !PO.needs_previous_weights(static)
        sr = SubsetResampling(; opt = InverseVolatility(),
                              n_subsets = TimeDependent([3, 4], :outermost))
        @test PO.is_time_dependent(sr)
        @test PO.update_time_dependent_estimator(sr, ctx(1)).n_subsets == 3
        @test PO.update_time_dependent_estimator(sr, ctx(2)).n_subsets == 4
        @test PO.reset_time_dependent_estimator(sr).n_subsets == 2
        sd = SubsetResampling(; opt = InverseVolatility(),
                              n_subsets = TimeDependent([3, 4], :outermost; default = 5))
        @test PO.reset_time_dependent_estimator(sd).n_subsets == 5
        sf = SubsetResampling(; opt = InverseVolatility(),
                              fb = TimeDependent([InverseVolatility(), EqualWeighted()],
                                                 :outermost))
        @test isnothing(PO.reset_time_dependent_estimator(sf).fb)
        @test_throws TimeDependentDefaultError PO.reset_time_dependent_estimator(SubsetResampling(;
                                                                                                  opt = TimeDependent([InverseVolatility(),
                                                                                                                       EqualWeighted()],
                                                                                                                      :outermost)))
        @test keys(PO.time_dependent_field_defaults(sr)) ==
              (:pe, :opt, :wf, :subset_size, :n_subsets)
    end
    @testset "the view" begin
        pr6 = prior(EmpiricalPrior(), rd6)
        sr = SubsetResampling(; pe = pr6, opt = InverseVolatility(), fb = EqualWeighted())
        v = PO.port_opt_view(sr, [2, 4, 5], zeros(3, 3))
        @test v.pe.X == pr6.X[:, [2, 4, 5]]
        @test isa(v.fb, EqualWeighted) && v.subset_size === sr.subset_size
    end
end
