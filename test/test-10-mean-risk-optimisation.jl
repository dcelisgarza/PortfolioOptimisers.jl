@safetestset "Mean Risk Optimisation" begin
    using Test, PortfolioOptimisers, DataFrames, CSV, TimeSeries, Clarabel, HiGHS, Pajarito,
          JuMP, StatsBase, StableRNGs, LinearAlgebra
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
                                     timestamp = :Date)[(end - 252):end])
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
    mip_slv = [Solver(; name = :mip1,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip2,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.95)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip3,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.90)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip4,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.85)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip5,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.80)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip6,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.75)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip7,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.7)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip8,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.6,
                                                                                                     "max_iter" => 1500,
                                                                                                     "tol_gap_abs" => 1e-4,
                                                                                                     "tol_gap_rel" => 1e-4,
                                                                                                     "tol_ktratio" => 1e-3,
                                                                                                     "tol_feas" => 1e-4,
                                                                                                     "tol_infeas_abs" => 1e-4,
                                                                                                     "tol_infeas_rel" => 1e-4,
                                                                                                     "reduced_tol_gap_abs" => 1e-4,
                                                                                                     "reduced_tol_gap_rel" => 1e-4,
                                                                                                     "reduced_tol_ktratio" => 1e-3,
                                                                                                     "reduced_tol_feas" => 1e-4,
                                                                                                     "reduced_tol_infeas_abs" => 1e-4,
                                                                                                     "reduced_tol_infeas_rel" => 1e-4)),
                      check_sol = (; allow_local = true, allow_almost = true))]
    sets = AssetSets(;
                     dict = Dict("nx" => rd.nx, "group1" => rd.nx[1:2:end],
                                 "group2" => rd.nx[2:2:end],
                                 "clusters1" => [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
                                                 3, 3, 3, 3, 3, 3],
                                 "clusters2" => [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2,
                                                 3, 1, 2, 3, 1, 2]))
    pr = prior(HighOrderPriorEstimator(), rd)
    clr = clusterise(ClusteringEstimator(), pr)
    w0 = range(; start = inv(size(pr.X, 2)), stop = inv(size(pr.X, 2)),
               length = size(pr.X, 2))
    wp = pweights(range(; start = inv(size(pr.X, 1)), stop = inv(size(pr.X, 1)),
                        length = size(pr.X, 1)))
    ucs1 = sigma_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(),
                                          rng = StableRNG(987654321),
                                          alg = BoxUncertaintySetAlgorithm()), rd.X)
    ucs2 = sigma_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(),
                                          rng = StableRNG(987654321),
                                          alg = EllipseUncertaintySetAlgorithm()), rd.X)
    rf = 4.2 / 100 / 252
    @testset "Mean Risk" begin
        objs = [MinimumRisk(), MaximumUtility(), MaximumRatio(; rf = rf)]
        rets = [ArithmeticReturn(), KellyReturn()]
        rs = [StandardDeviation(), Variance(), UncertaintySetVariance(; ucs = ucs1),
              UncertaintySetVariance(; ucs = ucs2), LowOrderMoment(),
              LowOrderMoment(;
                             alg = LowOrderDeviation(;
                                                     alg = SecondLowerMoment(;
                                                                             alg = SqrtRiskExpr()))),
              LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment())),
              LowOrderMoment(;
                             alg = LowOrderDeviation(;
                                                     alg = SecondCentralMoment(;
                                                                               alg = SqrtRiskExpr()))),
              LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondCentralMoment())),
              LowOrderMoment(; alg = MeanAbsoluteDeviation()), WorstRealisation(), Range(),
              ConditionalValueatRisk(), ConditionalValueatRiskRange(),
              EntropicValueatRisk(), EntropicValueatRiskRange(), RelativisticValueatRisk(),
              RelativisticValueatRiskRange(), MaximumDrawdown(), AverageDrawdown(),
              UlcerIndex(), ConditionalDrawdownatRisk(), EntropicDrawdownatRisk(),
              RelativisticDrawdownatRisk(), SquareRootKurtosis(; N = 2),
              SquareRootKurtosis(), OrderedWeightsArray(; alg = ExactOrderedWeightsArray()),
              OrderedWeightsArray(),
              OrderedWeightsArrayRange(; alg = ExactOrderedWeightsArray()),
              OrderedWeightsArrayRange(), NegativeSkewness(),
              NegativeSkewness(; alg = QuadRiskExpr())]
        df = CSV.read(joinpath(@__DIR__, "./assets/MeanRisk1.csv.gz"), DataFrame)
        i = 1
        for obj in objs, ret in rets, r in rs
            opt = JuMPOptimiser(; pe = pr, slv = slv, ret = ret)
            mr = MeanRisk(; r = r, obj = obj, opt = opt)
            res = optimise!(mr, rd)
            @test isa(res.retcode, OptimisationSuccess)
            rtol = if i ∈ (27, 59, 163, 189)
                5e-4
            elseif i ∈ (123, 187, 190) || Sys.isapple() && i == 60
                5e-3
            elseif i == 126
                1e-3
            else
                1e-4
            end
            success = isapprox(res.w, df[!, i]; rtol = rtol)
            if !success
                println("Counter: $i")
                find_tol(res.w, df[!, i])
            end
            @test success
            if isa(obj, MaximumRatio)
                rk = expected_risk(factory(r, pr, slv), res.w, rd.X)
                rt = expected_return(ret, res.w, pr)

                opt1 = JuMPOptimiser(; pe = pr, slv = slv,
                                     ret = bounds_returns_estimator(ret, rt))
                mr = MeanRisk(; r = r, opt = opt1)
                res = optimise!(mr, rd)
                rt1 = expected_return(ret, res.w, pr)
                @test rt1 >= rt || abs(rt1 - rt) < 1e-10

                mr = MeanRisk(; r = bounds_risk_measure(r, rk), obj = MaximumReturn(),
                              opt = opt)
                res = optimise!(mr, rd)
                rk1 = expected_risk(factory(r, pr, slv), res.w, rd.X)
                if !isa(r, SquareRootKurtosis) ||
                   isa(r, SquareRootKurtosis) && isnothing(r.N)
                    @test rk1 <= rk || abs(rk1 - rk) < 1e-9
                else
                    @test rk1 / rk < 1.07
                end
            end
            i += 1
        end
    end
    @testset "Scalarisers" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
        mr = MeanRisk(; r = r, opt = opt)
        w1 = optimise!(mr, rd).w
        @test isapprox(w1,
                       [1.7074698994991376e-10, 8.433104973224101e-11,
                        1.9860067068611146e-9, 3.1027970850564853e-10, 0.09898828657505643,
                        0.0038933979989648256, 6.636136386259997e-10, 0.35967323582529487,
                        0.012150886312492075, 0.10067191014130493, 2.964806359249512e-10,
                        0.14595634518390374, 3.891455764455336e-10, 0.14289553677354302,
                        5.362751594199594e-10, 0.03009102863550913, 1.1386315235757199e-10,
                        1.1094461786914318e-9, 0.07216633866035838, 0.033513028233384],
                       rtol = 1e-6)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sce = MaxScalariser())
        r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
        mr = MeanRisk(; r = r, opt = opt)
        w2 = optimise!(mr, rd).w

        opt = JuMPOptimiser(; pe = pr, slv = slv, sce = LogSumExpScalariser(; gamma = 1e-3))
        r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
        mr = MeanRisk(; r = r, opt = opt)
        w3 = optimise!(mr, rd).w
        @test isapprox(w3, w1, rtol = 5e-2)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sce = LogSumExpScalariser(; gamma = 1e5))
        r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
        mr = MeanRisk(; r = r, opt = opt)
        w4 = optimise!(mr, rd).w
        @test isapprox(w4, w2, rtol = 1e-4)
    end
    @testset "Arithmetic return uncertainty set" begin
        rng = StableRNG(123456789)
        ucs1 = mu_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(), rng = rng,
                                           alg = BoxUncertaintySetAlgorithm()), pr.X)
        ucs2 = mu_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(), rng = rng,
                                           alg = EllipseUncertaintySetAlgorithm()), pr.X)
        ucss = [ucs1, ucs2]
        objs = [MinimumRisk(), MaximumRatio(; rf = rf), MaximumReturn()]
        df = CSV.read(joinpath(@__DIR__, "./assets/MeanRiskUncertainty.csv.gz"), DataFrame)
        i = 1
        for ucs in ucss
            for obj in objs
                ret = ArithmeticReturn(; ucs = ucs)
                opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
                mre = MeanRisk(; obj = obj, opt = opt)
                res = optimise!(mre)
                @test isa(res.retcode, OptimisationSuccess)
                rtol = 1e-6
                success = isapprox(res.w, df[!, i]; rtol = rtol)
                if !success
                    println("Counter: $i")
                    find_tol(res.w, df[!, i])
                end
                @test success
                i += 1
            end
        end
    end
    @testset "Weight bounds" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                            wb = WeightBoundsEstimator(;
                                                       lb = ["group1" => -1,
                                                             "group2" => 0.1],
                                                       ub = Dict("group1" => -0.1,
                                                                 "group2" => 1)))
        mr = MeanRisk(; opt = opt)
        res1 = optimise!(mr)
        @test isapprox(sum(res1.w), 1)
        @test isapprox(sum(res1.w[res1.w .< zero(eltype(res1.w))]), -1)
        @test isapprox(sum(res1.w[res1.w .>= zero(eltype(res1.w))]), 2)
        @test all(res1.pa.wb.lb[1:2:end] .<= res1.w[1:2:end])
        @test all(abs.(res1.w[1:2:end] .- res1.pa.wb.ub[1:2:end]) .< 5e-10)
        @test all(res1.pa.wb.lb[2:2:end] .<= res1.w[2:2:end] .<= res1.pa.wb.ub[2:2:end])
    end
    @testset "Budget" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                            wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MaximumReturn(), opt = opt)
        res = optimise!(mr)
        @test isapprox(sum(res.w), 1)
        @test isapprox(sum(res.w[res.w .< zero(eltype(res.w))]), -1, rtol = 1e-6)
        @test isapprox(sum(res.w[res.w .>= zero(eltype(res.w))]), 2, rtol = 1e-6)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 0.15, bgt = 0.5,
                            wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mr)
        @test isapprox(sum(res.w), 0.5)
        @test isapprox(sum(res.w[res.w .< zero(eltype(res.w))]), -0.15, rtol = 1e-4)
        @test isapprox(sum(res.w[res.w .>= zero(eltype(res.w))]), 0.65, rtol = 5e-5)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                            sbgt = BudgetRange(; lb = 0.15, ub = 0.15),
                            bgt = BudgetRange(; lb = 0.3, ub = 0.45),
                            wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mr)
        @test 0.1 <= sum(res.w) <= 0.45
        @test isapprox(sum(res.w[res.w .< zero(eltype(res.w))]), -0.15, rtol = 5e-5)
        @test 0.45 <= sum(res.w[res.w .>= zero(eltype(res.w))]) <= 0.60

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = nothing, bgt = 1.7,
                            wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MinimumRisk(), opt = opt)
        res = optimise!(mr)
        @test isapprox(sum(res.w), 1.7)
        @test all(res.w .>= 0)
        @test !haskey(res.model, :sbgt)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1.4, bgt = nothing,
                            wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mr)
        @test isapprox(sum(res.w[res.w .< 0]), -1.4, rtol = 1e-3)
        @test !haskey(res.model, :bgt)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                            sbgt = BudgetRange(; lb = 0.41, ub = 0.63), bgt = 0.87,
                            wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mr)
        @test isapprox(sum(res.w), 0.87)
        @test -0.63 <= sum(res.w[res.w .< zero(eltype(res.w))]) <= -0.41
        @test 0.87 + 0.41 <= sum(res.w[res.w .>= zero(eltype(res.w))]) <= 0.87 + 0.63

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 0.61,
                            bgt = BudgetRange(; lb = 0.4, ub = 0.79),
                            wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MaximumUtility(), opt = opt)
        res = optimise!(mr)
        @test 0.4 <= sum(res.w) <= 0.79
        @test isapprox(sum(res.w[res.w .< zero(eltype(res.w))]), -0.61, rtol = 5e-5)
        @test 0.61 + 0.4 <= sum(res.w[res.w .> zero(eltype(res.w))]) <= 0.61 + 0.79

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                            sbgt = BudgetRange(; lb = 0.41, ub = 0.63), bgt = nothing,
                            wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mr)
        @test -0.63 <= sum(res.w[res.w .< zero(eltype(res.w))]) <= -0.41
        @test !haskey(res.model, :bgt)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = nothing,
                            bgt = BudgetRange(; lb = 0.4, ub = 0.79),
                            wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MaximumUtility(), opt = opt)
        res = optimise!(mr)
        @test 0.4 <= sum(res.w) <= 0.79
        @test !haskey(res.model, :sbgt)
    end
    @testset "Cardinality" begin
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, card = 3)
        mre = MeanRisk(; opt = opt)
        res = optimise!(mre)
        w = res.w
        @test count(w .> 1e-10) <= 3

        opt = JuMPOptimiser(; l2 = 0.1, pe = pr, slv = mip_slv, card = 3)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mre)
        w = res.w
        @test count(w .> 1e-10) <= 3

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, wb = WeightBounds(; lb = -1, ub = 1),
                            sbgt = 1, bgt = 1, card = 7)
        mre = MeanRisk(; opt = opt)
        res = optimise!(mre)
        w = res.w
        @test count(abs.(w) .> 1e-10) <= 7

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                            gcard = [:(XOM + MRK + WMT <= 2), :(group2 == 5)], sets = sets)
        mre = MeanRisk(; opt = opt)
        res = optimise!(mre)
        w = res.w
        @test rd.nx[.!iszero.(vec(res.gcard.A_ineq[1, :]))] == ["MRK", "WMT", "XOM"]
        @test rd.nx[.!iszero.(vec(res.gcard.A_eq[1, :]))] == rd.nx[2:2:end]
        @test count(w[.!iszero.(vec(res.gcard.A_ineq[1, :]))] .> 1e-10) <= 2
        @test count(w[.!iszero.(vec(res.gcard.A_eq[1, :]))] .> 1e-10) == 5

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, wb = WeightBounds(; lb = -1, ub = 1),
                            sbgt = 1, bgt = 1,
                            gcard = [:(XOM + MRK + WMT <= 2), :(group2 == 3)], sets = sets)
        mre = MeanRisk(; opt = opt)
        res = optimise!(mre)
        w = res.w
        @test rd.nx[.!iszero.(vec(res.gcard.A_ineq[1, :]))] == ["MRK", "WMT", "XOM"]
        @test rd.nx[.!iszero.(vec(res.gcard.A_eq[1, :]))] == rd.nx[2:2:end]
        @test count(w[.!iszero.(vec(res.gcard.A_ineq[1, :]))] .> 1e-10) <= 2
        @test count(w[.!iszero.(vec(res.gcard.A_eq[1, :]))] .> 1e-10) == 3

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, scard = 1, smtx = "clusters1",
                            sets = sets)
        mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test sum(.!iszero.([sum(w[res.smtx[i, :]]) for i in axes(res.smtx, 1)])) == 1

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, scard = 2, smtx = "clusters2",
                            sets = sets)
        mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf),
                       opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test sum(.!iszero.([sum(w[res.smtx[i, :]]) for i in axes(res.smtx, 1)])) == 2

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, scard = [1, 1],
                            smtx = ["clusters1", "clusters2"], sets = sets)
        mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        w = res.w

        i = 1
        dict = Dict{Tuple{Int, Int}, Int}()
        clusters3 = Int[]
        for cs in zip(sets.dict["clusters1"], sets.dict["clusters2"])
            if !haskey(dict, cs)
                dict[cs] = i
                i += 1
            end
            push!(clusters3, dict[cs])
        end
        sets.dict["clusters3"] = clusters3
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, scard = 1, smtx = "clusters3",
                            sets = sets)
        mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(), opt = opt)
        @test isapprox(res.w, optimise!(mre, rd).w)

        # opt = JuMPOptimiser(; pe = pr, slv = mip_slv, scard = 1,
        #                     smtx = "clusters1", sets = sets)
        # mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(), opt = opt)
        # res = optimise!(mre, rd)
        # w = res.w
        # [sum(w[res.smtx[i, :]]) for i in axes(res.smtx, 1)]
    end
    @testset "Buy-in threshold" begin
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                            lt = BuyInThresholdEstimator(;
                                                         val = ["WMT" => 0.23,
                                                                "group2" => 0.48]),
                            sets = sets)
        mre = MeanRisk(; opt = opt)
        res = optimise!(mre)
        @test res.w[findfirst(x -> x == "WMT", rd.nx)] >= 0.23
        @test res.w[2:2:end][res.w[2:2:end] .> 1e-9][1] >= 0.48

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, lt = BuyInThreshold(; val = 0.15),
                            sets = sets)
        mre = MeanRisk(; opt = opt)
        res = optimise!(mre)
        res.w[res.w .>= 1e-10] .>= 0.15

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                            lt = BuyInThreshold(; val = fill(0.15, size(pr.X, 2))),
                            sets = sets)
        mre = MeanRisk(; opt = opt)
        @test isapprox(res.w, optimise!(mre).w)

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, wb = WeightBounds(; lb = -1, ub = 1),
                            sbgt = 1, bgt = 1, st = BuyInThreshold(; val = 0.25),
                            lt = BuyInThreshold(; val = 0.4), sets = sets)
        mre = MeanRisk(; opt = opt)
        res = optimise!(mre)
        @test all(res.w[res.w .> 0][res.w[res.w .>= 0] .>= 1e-10] .>= 0.4)
        if Sys.isapple()
            @test all(res.w[res.w .< 0][res.w[res.w .< 0] .<= -1e-10] .<=
                      -0.25 + sqrt(eps()))
        else
            @test all(res.w[res.w .< 0][res.w[res.w .< 0] .<= -1e-10] .<= -0.25)
        end

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, wb = WeightBounds(; lb = -1, ub = 1),
                            sbgt = 1, bgt = 1, st = BuyInThreshold(; val = 0.25),
                            lt = BuyInThreshold(; val = 0.4), sets = sets)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mre)
        @test all(res.w[res.w .> 0][res.w[res.w .>= 0] .>= 1e-10] .>= 0.4)
        @test all(res.w[res.w .< 0][res.w[res.w .< 0] .<= -1e-10] .<= -0.25)
    end
end
