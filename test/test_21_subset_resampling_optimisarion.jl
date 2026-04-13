@safetestset "Subset Resampling Optimisation" begin
    using PortfolioOptimisers, CSV, Test, TimeSeries, Clarabel, DataFrames, StableRNGs,
          Pajarito, HiGHS, JuMP, Clustering, NearestCorrelationMatrix
    function find_tol(a1, a2; name1 = :lhs, name2 = :rhs)
        for rtol in
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; rtol = rtol)
                println("isapprox($name1, $name2, rtol = $(rtol))")
                break
            end
        end
        for atol in
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; atol = atol)
                println("isapprox($name1, $name2, atol = $(atol))")
                break
            end
        end
    end
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end],
                           TimeArray(CSV.File(joinpath(@__DIR__, "./assets/Factors.csv.gz"));
                                     timestamp = :Date)[(end - 252):end];
                           B = TimeArray(CSV.File(joinpath(@__DIR__,
                                                           "./assets/SP500_idx.csv.gz"));
                                         timestamp = :Date))
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
    @test isapprox(optimise(SubsetResampling(; subset_size = 1, n_subsets = 20, pe = pr,
                                             opt = mr, scale = ones(20)), rd).w, w0)

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
    @test isapprox(optimise(SubsetResampling(; subset_size = 19, n_subsets = 20, pe = pr,
                                             opt = mr, scale = ones(20)), rd).w, res.w)

    mr_res = optimise(mr, rd)
    @test isapprox(res.w, mr_res.w, rtol = 0.05)

    jopt = JuMPOptimiser(; slv = slv, ret = ArithmeticReturn(; lb = Frontier(5)))
    mr = MeanRisk(; opt = jopt)

    opt = SubsetResampling(; subset_size = eps(), n_subsets = 20, pe = pr, opt = mr)
    res = optimise(opt, rd)
    @test all(x -> isapprox(x, w0), res.w)
    @test isapprox(optimise(SubsetResampling(; subset_size = eps(), n_subsets = 20, pe = pr,
                                             opt = mr, scale = ones(20)), rd).w, res.w)

    opt = SubsetResampling(; subset_size = 0.95, n_subsets = 20, pe = pr, opt = mr)
    res = optimise(opt, rd)
    df = CSV.read(joinpath(@__DIR__, "./assets/SubsetResamplingFrontier.csv.gz"), DataFrame)
    success = isapprox(Matrix(df), reduce(hcat, res.w); rtol = 1e-6)
    if !success
        find_tol(Matrix(df), reduce(hcat, res.w))
    end
    @test success
    @test isapprox(optimise(SubsetResampling(; subset_size = 0.95, n_subsets = 20, pe = pr,
                                             opt = mr, scale = ones(20)), rd).w, res.w)

    mr_res = optimise(mr, rd)
    success = isapprox(res.w, mr_res.w; rtol = 0.1)
    if !success
        find_tol(res.w, mr_res.w)
    end
    @test success
end
