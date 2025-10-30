@safetestset "Prior" begin
    using Test, PortfolioOptimisers, DataFrames, TimeSeries, CSV, StatsBase, Clarabel
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
                                     timestamp = :Date)[(end - 252 * 4):end],
                           TimeArray(CSV.File(joinpath(@__DIR__, "./assets/Factors.csv.gz"));
                                     timestamp = :Date)[(end - 252 * 4):end])
    sets = AssetSets(;
                     dict = Dict("nx" => rd.nx, "group1" => rd.nx[1:2:end],
                                 "group2" => rd.nx[2:2:end],
                                 "clusters1" => [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
                                                 3, 3, 3, 3, 3, 3],
                                 "clusters2" => [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2,
                                                 3, 1, 2, 3, 1, 2]))
    fsets = AssetSets(; dict = Dict("nx" => rd.nf))
    @testset "Empirical Prior" begin
        pes = [EmpiricalPrior(), EmpiricalPrior(; horizon = 252)]
        df = CSV.read(joinpath(@__DIR__, "./assets/EmpiricalPrior.csv.gz"), DataFrame)
        for (i, pe) in enumerate(pes)
            pr = prior(pe, rd)

            mut = reshape(df[1:20, i], size(pr.mu))
            sigmat = reshape(df[21:end, i], size(pr.sigma))

            rtol = 1e-6
            success = isapprox(pr.mu, mut; rtol = rtol)
            if !success
                println("Mu $i fails")
                find_tol(pr.mu, mut)
            end
            @test success

            rtol = 1e-6
            success = isapprox(pr.sigma, sigmat; rtol = rtol)
            if !success
                println("Sigma $i fails")
                find_tol(pr.sigma, sigmat)
            end
            @test success
        end
    end
    @testset "Factor Prior" begin
        pes = [FactorPrior(; rsd = false), FactorPrior(; rsd = true)]
        df = CSV.read(joinpath(@__DIR__, "./assets/FactorPrior1.csv.gz"), DataFrame)
        pr = prior(FactorPrior(; rsd = false), rd)

        df[!, "1"] = [pr.mu; vec(pr.sigma); vec(pr.chol)]

        mut = reshape(df[1:20, 1], size(pr.mu))
        sigmat = reshape(df[21:420, 1], size(pr.sigma))
        cholt = reshape(df[421:end, 1], size(pr.chol))

        rtol = 1e-6
        success = isapprox(pr.mu, mut; rtol = rtol)
        if !success
            println("Mu 1 fails")
            find_tol(pr.mu, mut)
        end
        @test success

        rtol = 1e-6
        success = isapprox(pr.sigma, sigmat; rtol = rtol)
        if !success
            println("Sigma 1 fails")
            find_tol(pr.sigma, sigmat)
        end
        @test success

        rtol = 1e-6
        success = isapprox(pr.chol, cholt; rtol = rtol)
        if !success
            println("Chol 1 fails")
            find_tol(pr.chol, cholt)
        end
        @test success

        df = CSV.read(joinpath(@__DIR__, "./assets/FactorPrior2.csv.gz"), DataFrame)
        pr = prior(FactorPrior(; rsd = true), rd)

        mut = reshape(df[1:20, 1], size(pr.mu))
        sigmat = reshape(df[21:420, 1], size(pr.sigma))
        cholt = reshape(df[421:end, 1], size(pr.chol))

        rtol = 1e-6
        success = isapprox(pr.mu, mut; rtol = rtol)
        if !success
            println("Mu 2 fails")
            find_tol(pr.mu, mut)
        end
        @test success

        rtol = 1e-6
        success = isapprox(pr.sigma, sigmat; rtol = rtol)
        if !success
            println("Sigma 2 fails")
            find_tol(pr.sigma, sigmat)
        end
        @test success

        rtol = 1e-6
        success = isapprox(pr.chol, cholt; rtol = rtol)
        if !success
            println("Chol 2 fails")
            find_tol(pr.chol, cholt)
        end
        @test success
    end
    @testset "High Order Prior" begin
        pr = prior(HighOrderPriorEstimator(), rd)
        @test isapprox(pr.X, rd.X)
        @test isapprox(pr.mu, vec(mean(SimpleExpectedReturns(), rd.X)))
        @test isapprox(pr.sigma, cov(PortfolioOptimisersCovariance(), rd.X))
        @test isapprox(pr.kt, cokurtosis(Cokurtosis(; alg = Full()), rd.X))
        @test all(isapprox.((pr.sk, pr.V), coskewness(Coskewness(; alg = Full()), rd.X)))

        pe = HighOrderPriorEstimator(; kte = Cokurtosis(; alg = Semi()),
                                     ske = Coskewness(; alg = Semi()))
        pr = prior(pe, transpose(rd.X); dims = 2)
        @test isapprox(pr.X, rd.X)
        @test isapprox(pr.mu, vec(mean(SimpleExpectedReturns(), rd.X)))
        @test isapprox(pr.sigma, cov(PortfolioOptimisersCovariance(), rd.X))
        @test isapprox(pr.kt, cokurtosis(Cokurtosis(; alg = Semi()), rd.X))
        @test all(isapprox.((pr.sk, pr.V), coskewness(Coskewness(; alg = Semi()), rd.X)))

        pe1 = FactorPrior(; re = DimensionReductionRegression(;), rsd = true)
        pr1 = prior(pe1, rd)

        pe2 = HighOrderPriorEstimator(; pe = pe1)
        pr2 = prior(pe2, rd)
        @test isa(pe2.me, SimpleExpectedReturns)
        @test isa(pe2.ce, PortfolioOptimisersCovariance)

        @test pr1.X == pr2.X
        @test pr1.mu == pr2.mu
        @test pr1.sigma == pr2.sigma
        @test isapprox(pr2.kt,
                       cokurtosis(Cokurtosis(; alg = Full()), pr2.X;
                                  mean = transpose(pr2.mu)))
        @test all(isapprox.((pr2.sk, pr2.V),
                            coskewness(Coskewness(; alg = Full()), pr2.X;
                                       mean = transpose(pr2.mu))))
    end
    @testset "Vanilla and Bayesian Black Litterman" begin
        df = CSV.read(joinpath(@__DIR__, "./assets/BlackLitterman.csv.gz"), DataFrame)
        pes = [BlackLittermanPrior(; sets = sets, tau = 1 / size(rd.X, 1),
                                   views = LinearConstraintEstimator(;
                                                                     val = ["AAPL == 0.00002",
                                                                            "BAC == CVX",
                                                                            "WMT == group2",
                                                                            "RRC-group1 == 0.0005"])),
               BayesianBlackLittermanPrior(; pe = FactorPrior(; pe = EmpiricalPrior(;)),
                                           sets = fsets, tau = 1 / size(rd.X, 1),
                                           views = LinearConstraintEstimator(;
                                                                             val = ["MTUM == 0.0001",
                                                                                    "QUAL - USMV == -0.0003"])),
               BlackLittermanPrior(; sets = sets, tau = 1 / size(rd.X, 1),
                                   views_conf = [0.05, 0.2, 0.5, 0.9],
                                   views = LinearConstraintEstimator(;
                                                                     val = ["AAPL == 0.00002",
                                                                            "BAC == CVX",
                                                                            "WMT == group2",
                                                                            "RRC-group1 == 0.0005"])),
               BlackLittermanPrior(; sets = sets, tau = 1 / size(rd.X, 1),
                                   views_conf = 0.05,
                                   views = LinearConstraintEstimator(;
                                                                     val = "AAPL == 0.00002"))]
        for (i, pe) in enumerate(pes)
            pr = prior(pe, rd)
            success = isapprox(pr.mu, df[1:20, i]; rtol = 1e-6)
            if !success
                println("Mu $i fails")
                find_tol(pr.mu, df[1:20, i])
            end
            @test success

            success = isapprox(vec(pr.sigma), df[21:420, i]; rtol = 1e-6)
            if !success
                println("Sigma $i fails")
                find_tol(vec(pr.sigma), df[21:420, i])
            end
            @test success
        end

        pr = prior(BlackLittermanPrior(; sets = sets, tau = 1 / size(rd.X, 1),
                                       views = black_litterman_views(LinearConstraintEstimator(;
                                                                                               val = ["AAPL == 0.00002",
                                                                                                      "BAC == CVX",
                                                                                                      "WMT == group2",
                                                                                                      "RRC-group1 == 0.0005"]),
                                                                     sets)), rd)
        @test isapprox(df[!, 1], [pr.mu; vec(pr.sigma)], rtol = 1e-6)
    end
    @testset "Factor Black Litterman" begin
        df = CSV.read(joinpath(@__DIR__, "./assets/FactorBlackLitterman1.csv.gz"),
                      DataFrame)
        pe = FactorBlackLittermanPrior(; pe = EmpiricalPrior(;), rsd = false, sets = fsets,
                                       tau = 1 / size(rd.X, 1),
                                       views = LinearConstraintEstimator(;
                                                                         val = ["MTUM == 0.0001",
                                                                                "QUAL - USMV == -0.0003"]))
        pr = prior(pe, rd)
        success = isapprox(pr.mu, df[1:20, 1]; rtol = 1e-6)
        if !success
            println("Mu $i fails")
            find_tol(pr.mu, df[1:20, i])
        end
        @test success

        success = isapprox(vec(pr.sigma), df[21:420, 1]; rtol = 1e-6)
        if !success
            println("Sigma $i fails")
            find_tol(vec(pr.sigma), df[21:420, i])
        end
        @test success

        success = isapprox(vec(pr.chol), df[421:end, 1]; rtol = 1e-6)
        if !success
            println("Chol $i fails")
            find_tol(vec(pr.chol), df[421:end, i])
        end
        @test success

        df = CSV.read(joinpath(@__DIR__, "./assets/FactorBlackLitterman2.csv.gz"),
                      DataFrame)
        pe = FactorBlackLittermanPrior(; pe = EmpiricalPrior(;), sets = fsets, l = 2,
                                       tau = 1 / size(rd.X, 1),
                                       views = LinearConstraintEstimator(;
                                                                         val = ["MTUM == 0.0001",
                                                                                "QUAL - USMV == -0.0003"]))
        pr = prior(pe, rd)
        success = isapprox(pr.mu, df[1:20, 1]; rtol = 1e-6)
        if !success
            println("Mu $i fails")
            find_tol(pr.mu, df[1:20, i])
        end
        @test success

        success = isapprox(vec(pr.sigma), df[21:420, 1]; rtol = 1e-6)
        if !success
            println("Sigma $i fails")
            find_tol(vec(pr.sigma), df[21:420, i])
        end
        @test success

        success = isapprox(vec(pr.chol), df[421:end, 1]; rtol = 1e-6)
        if !success
            println("Chol $i fails")
            find_tol(vec(pr.chol), df[421:end, i])
        end
        @test success
    end
    @testset "Augmented Black Litterman" begin
        df = CSV.read(joinpath(@__DIR__, "./assets/AugmentedBlackLitterman.csv.gz"),
                      DataFrame)
        pes = [AugmentedBlackLittermanPrior(; a_sets = sets, f_sets = fsets,
                                            tau = 1 / size(rd.X, 1),
                                            a_views = LinearConstraintEstimator(;
                                                                                val = Union{String,
                                                                                            Expr}[:(AAPL ==
                                                                                                    0.00002),
                                                                                                  :(BAC ==
                                                                                                    CVX),
                                                                                                  "WMT == group2",
                                                                                                  "RRC-group1 == 0.0005"]),
                                            f_views = LinearConstraintEstimator(;
                                                                                val = [:(MTUM ==
                                                                                         0.0001),
                                                                                       :(QUAL -
                                                                                         USMV ==
                                                                                         -0.0003)])),
               AugmentedBlackLittermanPrior(; a_sets = sets, f_sets = fsets,
                                            tau = 1 / size(rd.X, 1), l = 2,
                                            a_views = LinearConstraintEstimator(;
                                                                                val = ["AAPL == 0.00002",
                                                                                       "BAC == CVX",
                                                                                       "WMT == group2",
                                                                                       "RRC-group1 == 0.0005"]),
                                            f_views = LinearConstraintEstimator(;
                                                                                val = ["MTUM == 0.0001",
                                                                                       "QUAL - USMV == -0.0003"]))]
        for (i, pe) in enumerate(pes)
            pr = prior(pe, rd)
            success = isapprox(pr.mu, df[1:20, i]; rtol = 1e-6)
            if !success
                println("Mu $i fails")
                find_tol(pr.mu, df[1:20, i])
            end
            @test success

            success = isapprox(vec(pr.sigma), df[21:420, i]; rtol = 1e-6)
            if !success
                println("Sigma $i fails")
                find_tol(vec(pr.sigma), df[21:420, i])
            end
            @test success
        end
    end
    slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = "verbose" => false),
           Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = ["verbose" => false, "max_step_fraction" => 0.95]),
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
    @testset "ExpEntropyPooling" begin
        pr0 = prior(EmpiricalPrior(), rd)
        jopt = JuMPEntropyPooling(; slv = slv)

        mu_views = LinearConstraintEstimator(; val = "AAPL == 0.002")
        pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views), rd)
        @test isapprox(pr.mu[1], 0.002)
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 mu_views = mu_views), rd).w, rtol = 5e-6)

        mu_views = LinearConstraintEstimator(; val = "AAPL >= 0.0025")
        pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views), rd)
        @test pr.mu[1] >= 0.0025
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 mu_views = mu_views), rd).w, rtol = 5e-6)

        mu_views = LinearConstraintEstimator(; val = "AAPL <= 0.001")
        pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views), rd)
        @test pr.mu[1] <= 0.001
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 mu_views = mu_views), rd).w, rtol = 5e-6)

        var_views = LinearConstraintEstimator(; val = "AAPL == 0.03264496113282452")
        pr = prior(EntropyPoolingPrior(; sets = sets, var_views = var_views), rd)
        @test ValueatRisk(; w = pr.w)(rd.X[:, 1]) == ValueatRisk(;)(rd.X[:, 1])
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 var_views = var_views), rd).w)

        var_views = LinearConstraintEstimator(; val = "AAPL >= 1.15*prior(AAPL)")
        pr = prior(EntropyPoolingPrior(; sets = sets, var_views = var_views), rd)
        @test ValueatRisk(; w = pr.w)(rd.X[:, 1]) >= 1.15 * ValueatRisk(;)(rd.X[:, 1])
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 var_views = var_views), rd).w, rtol = 1e-6)

        var_views = LinearConstraintEstimator(; val = "AAPL == 0.12865204867438676")
        pr = prior(EntropyPoolingPrior(; sets = sets, var_views = var_views), rd)
        @test ValueatRisk(; w = pr.w)(rd.X[:, 1]) == WorstRealisation()(rd.X[:, 1])
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, var_views = var_views,
                                                 opt = jopt), rd).w, rtol = 1e-6)

        var_views = LinearConstraintEstimator(; val = ["AAPL == 0.028", "XOM >= 0.027"])
        pr = prior(EntropyPoolingPrior(; sets = sets, var_alpha = 0.07,
                                       var_views = var_views), rd)
        @test isapprox(ValueatRisk(; alpha = 0.07, w = pr.w)(rd.X[:, 1]), 0.028,
                       rtol = 7e-3)
        @test ValueatRisk(; alpha = 0.07, w = pr.w)(rd.X[:, end]) >= 0.027
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 var_alpha = 0.07, var_views = var_views),
                             rd).w, rtol = 1e-4)

        sigma_views = LinearConstraintEstimator(; val = "AAPL == 0.0007")
        pr = prior(EntropyPoolingPrior(; sets = sets, sigma_views = sigma_views), rd)
        r = LowOrderMoment(; w = pr.w, mu = pr.mu[1],
                           alg = SecondMoment(; ve = SimpleVariance(; w = pr.w)))
        @test isapprox(r([1], reshape(pr.X[:, 1], :, 1)), 0.0007, rtol = 1e-3)
        @test isapprox(pr.sigma[1, 1], r([1], reshape(pr.X[:, 1], :, 1)))
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, sigma_views = sigma_views,
                                                 opt = jopt), rd).w, rtol = 1e-2)

        mu_views = LinearConstraintEstimator(; val = "AAPL == 1.7*prior(AAPL)")
        sigma_views = LinearConstraintEstimator(; val = "AAPL == 0.0008")
        pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views,
                                       sigma_views = sigma_views), rd)
        @test isapprox(pr.mu[1], pr0.mu[1] * 1.7)
        @test isapprox(pr.sigma[1, 1], 0.0008, rtol = 1e-3)
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 mu_views = mu_views,
                                                 sigma_views = sigma_views), rd).w,
                       rtol = 5e-4)

        sk_views = LinearConstraintEstimator(; val = "AAPL == prior(AAPL)*2")
        pr = prior(EntropyPoolingPrior(; sets = sets, sk_views = sk_views), rd)
        @test isapprox(Skewness(; w = pr.w, ve = SimpleVariance(; w = pr.w))([1],
                                                                             reshape(pr.X[:,
                                                                                          1],
                                                                                     :, 1)),
                       2 * Skewness()([1], reshape(pr0.X[:, 1], :, 1)), rtol = 2e-3)
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 sk_views = sk_views), rd).w, rtol = 5e-3)

        mu_views = LinearConstraintEstimator(; val = "AAPL==1.5*prior(AAPL)")
        sigma_views = LinearConstraintEstimator(; val = "AAPL==1.3prior(AAPL)")
        sk_views = LinearConstraintEstimator(; val = "AAPL == prior(AAPL)*2")
        pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views,
                                       sigma_views = sigma_views, sk_views = sk_views), rd)
        @test isapprox(pr.mu[1], 1.5 * pr0.mu[1], rtol = 1e-6)
        @test isapprox(pr.sigma[1, 1], 1.3 * pr0.sigma[1, 1], rtol = 5e-3)
        @test isapprox(Skewness(; w = pr.w, ve = SimpleVariance(; w = pr.w))([1],
                                                                             reshape(pr.X[:,
                                                                                          1],
                                                                                     :, 1)),
                       2 * Skewness()([1], reshape(pr0.X[:, 1], :, 1)), rtol = 5e-3)
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views,
                                                 sigma_views = sigma_views,
                                                 sk_views = sk_views), rd).w)

        kt_views = LinearConstraintEstimator(; val = "AAPL == 7.5")
        pr = prior(EntropyPoolingPrior(; sets = sets, kt_views = kt_views), rd)
        @test isapprox(HighOrderMoment(; w = pr.w,
                                       alg = StandardisedHighOrderMoment(;
                                                                         alg = FourthMoment(),
                                                                         ve = SimpleVariance(;
                                                                                             w = pr.w)))([1],
                                                                                                         reshape(pr.X[:,
                                                                                                                      1],
                                                                                                                 :,
                                                                                                                 1)),
                       7.5, rtol = 5e-3)
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 kt_views = kt_views), rd).w, rtol = 1e-2)

        mu_views = LinearConstraintEstimator(; val = "AAPL<=1.5*prior(AAPL)")
        sigma_views = LinearConstraintEstimator(; val = "AAPL==0.7prior(AAPL)")
        kt_views = LinearConstraintEstimator(; val = "AAPL >= prior(AAPL)*0.87")
        pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views,
                                       sigma_views = sigma_views, kt_views = kt_views), rd)
        @test pr.mu[1] <= 1.5 * pr0.mu[1]
        @test isapprox(pr.sigma[1, 1], 0.7 * pr0.sigma[1, 1], rtol = 1e-3)
        @test abs(HighOrderMoment(; w = pr.w,
                                  alg = StandardisedHighOrderMoment(; alg = FourthMoment(),
                                                                    ve = SimpleVariance(;
                                                                                        w = pr.w)))([1],
                                                                                                    reshape(pr.X[:,
                                                                                                                 1],
                                                                                                            :,
                                                                                                            1)) -
                  HighOrderMoment(;
                                  alg = StandardisedHighOrderMoment(; alg = FourthMoment()))([1],
                                                                                             reshape(pr.X[:,
                                                                                                          1],
                                                                                                     :,
                                                                                                     1)) *
                  0.87) <= sqrt(eps())
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 mu_views = mu_views,
                                                 sigma_views = sigma_views,
                                                 kt_views = kt_views), rd).w, rtol = 5e-3)

        rho_views = LinearConstraintEstimator(; val = "(AAPL, XOM) == 0.35")
        pr = prior(EntropyPoolingPrior(; sets = sets, rho_views = rho_views), rd)
        @test isapprox(cov2cor(pr.sigma)[1, end], 0.35, rtol = 5e-6)
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 rho_views = rho_views), rd).w, rtol = 1e-2)

        rho_views = LinearConstraintEstimator(; val = "(AAPL, XOM) == prior(AAPL,XOM)*0.94")
        pr = prior(EntropyPoolingPrior(; sets = sets, rho_views = rho_views), rd)
        @test isapprox(cov2cor(pr.sigma)[1, end], cov2cor(pr0.sigma)[1, end] * 0.94,
                       rtol = 5e-6)
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 rho_views = rho_views), rd).w, rtol = 5e-2)

        pr = prior(HighOrderPriorEstimator(;
                                           pe = EntropyPoolingPrior(;
                                                                    alg = H2_EntropyPooling(),
                                                                    sets = sets,
                                                                    mu_views = LinearConstraintEstimator(;
                                                                                                         val = ["AAPL<=0.92*prior(AAPL)",
                                                                                                                "XOM >= 0.83*prior(XOM)"]),
                                                                    sigma_views = LinearConstraintEstimator(;
                                                                                                            val = ["AAPL==1.2prior(AAPL)",
                                                                                                                   "WMT==1.4prior(WMT)"]),
                                                                    rho_views = LinearConstraintEstimator(;
                                                                                                          val = "(AAPL, XOM) == 0.35"))),
                   rd)
        @test pr.mu[1] <= 0.92 * pr0.mu[1] + sqrt(eps())
        @test pr.mu[end] >= 0.83 * pr0.mu[end] - sqrt(eps())
        @test isapprox(pr.sigma[1, 1], 1.2 * pr0.sigma[1, 1], rtol = 1e-2)
        @test isapprox(pr.sigma[19, 19], 1.4 * pr0.sigma[19, 19], rtol = 5e-3)
        @test isapprox(cov2cor(pr.sigma)[1, end], 0.35, rtol = 1e-3)

        cvar_views = LinearConstraintEstimator(; val = "AAPL == 0.07")
        pr = prior(EntropyPoolingPrior(; sets = sets, cvar_views = cvar_views), rd)
        @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, 1]), 0.07, rtol = 1e-6)
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 cvar_views = cvar_views), rd).w,
                       rtol = 5e-5)

        cvar_views = LinearConstraintEstimator(; val = "AAPL == prior(AAPL)*1.37")
        pr = prior(EntropyPoolingPrior(; sets = sets, cvar_views = cvar_views), rd)
        @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, 1]),
                       ConditionalValueatRisk(;)(rd.X[:, 1]) * 1.37, rtol = 1e-6)
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 cvar_views = cvar_views), rd).w,
                       rtol = 5e-5)

        cvar_views = LinearConstraintEstimator(; val = ["AAPL == 0.053", "XOM==0.045"])
        pr = prior(HighOrderPriorEstimator(;
                                           pe = EntropyPoolingPrior(; sets = sets,
                                                                    alg = H2_EntropyPooling(),
                                                                    cvar_views = cvar_views)),
                   rd)
        @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, 1]), 0.053, rtol = 5e-5)
        @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, end]), 0.045, rtol = 1e-4)
        @test isapprox(pr.w,
                       prior(HighOrderPriorEstimator(;
                                                     pe = EntropyPoolingPrior(; sets = sets,
                                                                              alg = H1_EntropyPooling(),
                                                                              cvar_views = cvar_views)),
                             rd).w, rtol = 5e-3)

        mu_views = LinearConstraintEstimator(;
                                             val = ["AAPL<=0.75*prior(AAPL)",
                                                    "XOM >= 0.4*prior(XOM)"])
        sigma_views = LinearConstraintEstimator(;
                                                val = ["AAPL==0.2prior(AAPL)",
                                                       "WMT==1.4prior(WMT)"])
        rho_views = LinearConstraintEstimator(; val = "(AAPL, XOM) == 0.35")
        kt_views = LinearConstraintEstimator(; val = "AAPL >= prior(AAPL)*0.3")
        sk_views = LinearConstraintEstimator(; val = "WMT == prior(WMT)*1.4")
        pr = prior(HighOrderPriorEstimator(;
                                           pe = EntropyPoolingPrior(;
                                                                    alg = H0_EntropyPooling(),
                                                                    sets = sets,
                                                                    mu_views = mu_views,
                                                                    sigma_views = sigma_views,
                                                                    rho_views = rho_views,
                                                                    kt_views = kt_views,
                                                                    sk_views = sk_views)),
                   rd)
        @test pr.mu[1] <= 0.75 * pr0.mu[1]
        @test pr.mu[end] >= 0.4 * pr0.mu[end]
        @test isapprox(pr.sigma[1, 1], 0.2 * pr0.sigma[1, 1], rtol = 1e-2)
        @test isapprox(pr.sigma[19, 19], 1.4 * pr0.sigma[19, 19], rtol = 5e-3)
        @test !isapprox(cov2cor(pr.sigma)[1, end], 0.35; rtol = 5e-4)
        @test HighOrderMoment(; w = pr.w,
                              alg = StandardisedHighOrderMoment(; alg = FourthMoment(),
                                                                ve = SimpleVariance(;
                                                                                    w = pr.w)))([1],
                                                                                                reshape(pr.X[:,
                                                                                                             1],
                                                                                                        :,
                                                                                                        1)) >=
              HighOrderMoment(; alg = StandardisedHighOrderMoment(; alg = FourthMoment()))([1],
                                                                                           reshape(pr.X[:,
                                                                                                        1],
                                                                                                   :,
                                                                                                   1)) *
              0.3
        @test !isapprox(Skewness(; w = pr.w, ve = SimpleVariance(; w = pr.w))([1],
                                                                              reshape(pr.X[:,
                                                                                           end - 1],
                                                                                      :, 1)),
                        1.4 * Skewness()([1], reshape(pr0.X[:, end - 1], :, 1));
                        rtol = 5e-3)
    end
    @testset "LogEntropyPooling" begin
        pr0 = prior(EmpiricalPrior(), rd)
        opt = OptimEntropyPooling(; alg = LogEntropyPooling())
        jopt = JuMPEntropyPooling(; alg = LogEntropyPooling(), slv = slv)

        mu_views = LinearConstraintEstimator(; val = "AAPL == 0.002")
        pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views, opt = opt), rd)
        @test isapprox(pr.mu[1], 0.002)
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 mu_views = mu_views), rd).w, rtol = 1e-5)

        mu_views = LinearConstraintEstimator(; val = "AAPL >= 0.0025")
        pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views, opt = opt), rd)
        @test pr.mu[1] >= 0.0025
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 mu_views = mu_views), rd).w, rtol = 5e-5)

        mu_views = LinearConstraintEstimator(; val = "AAPL <= 0.001")
        pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views, opt = opt), rd)
        @test pr.mu[1] <= 0.001
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 mu_views = mu_views), rd).w, rtol = 5e-6)

        var_views = LinearConstraintEstimator(; val = "AAPL == 0.03264496113282452")
        pr = prior(EntropyPoolingPrior(; sets = sets, var_views = var_views, opt = opt), rd)
        @test ValueatRisk(; w = pr.w)(rd.X[:, 1]) == ValueatRisk(;)(rd.X[:, 1])
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 var_views = var_views), rd).w)

        var_views = LinearConstraintEstimator(; val = "AAPL >= 1.15*prior(AAPL)")
        pr = prior(EntropyPoolingPrior(; sets = sets, var_views = var_views, opt = opt), rd)
        @test ValueatRisk(; w = pr.w)(rd.X[:, 1]) >= 1.15 * ValueatRisk(;)(rd.X[:, 1])
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 var_views = var_views), rd).w, rtol = 1e-6)

        var_views = LinearConstraintEstimator(; val = "AAPL == 0.12865204867438676")
        pr = prior(EntropyPoolingPrior(; sets = sets, var_views = var_views, opt = opt), rd)
        @test ValueatRisk(; w = pr.w)(rd.X[:, 1]) == WorstRealisation()(rd.X[:, 1])
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, var_views = var_views,
                                                 opt = jopt), rd).w, rtol = 5e-6)

        var_views = LinearConstraintEstimator(; val = ["AAPL == 0.028", "XOM >= 0.027"])
        pr = prior(EntropyPoolingPrior(; sets = sets, var_alpha = 0.07,
                                       var_views = var_views, opt = opt), rd)
        @test isapprox(ValueatRisk(; alpha = 0.07, w = pr.w)(rd.X[:, 1]), 0.028,
                       rtol = 7e-3)
        @test ValueatRisk(; alpha = 0.07, w = pr.w)(rd.X[:, end]) >= 0.027
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 var_alpha = 0.07, var_views = var_views),
                             rd).w, rtol = 1e-4)

        sigma_views = LinearConstraintEstimator(; val = "AAPL == 0.0007")
        pr = prior(EntropyPoolingPrior(; sets = sets, sigma_views = sigma_views, opt = opt),
                   rd)
        r = LowOrderMoment(; w = pr.w, mu = pr.mu[1],
                           alg = SecondMoment(; ve = SimpleVariance(; w = pr.w)))
        @test isapprox(r([1], reshape(pr.X[:, 1], :, 1)), 0.0007, rtol = 1e-3)
        @test isapprox(pr.sigma[1, 1], r([1], reshape(pr.X[:, 1], :, 1)))
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, sigma_views = sigma_views,
                                                 opt = jopt), rd).w, rtol = 1e-2)

        mu_views = LinearConstraintEstimator(; val = "AAPL == 1.7*prior(AAPL)")
        sigma_views = LinearConstraintEstimator(; val = "AAPL == 0.0008")
        pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views,
                                       sigma_views = sigma_views, opt = opt), rd)
        @test isapprox(pr.mu[1], pr0.mu[1] * 1.7)
        @test isapprox(pr.sigma[1, 1], 0.0008, rtol = 1e-3)
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 mu_views = mu_views,
                                                 sigma_views = sigma_views), rd).w,
                       rtol = 5e-4)

        sk_views = LinearConstraintEstimator(; val = "AAPL == prior(AAPL)*2")
        pr = prior(EntropyPoolingPrior(; sets = sets, sk_views = sk_views, opt = opt), rd)
        @test isapprox(Skewness(; w = pr.w, ve = SimpleVariance(; w = pr.w))([1],
                                                                             reshape(pr.X[:,
                                                                                          1],
                                                                                     :, 1)),
                       2 * Skewness()([1], reshape(pr0.X[:, 1], :, 1)), rtol = 2e-3)
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 sk_views = sk_views), rd).w, rtol = 5e-3)

        mu_views = LinearConstraintEstimator(; val = "AAPL==1.5*prior(AAPL)")
        sigma_views = LinearConstraintEstimator(; val = "AAPL==1.3prior(AAPL)")
        sk_views = LinearConstraintEstimator(; val = "AAPL == prior(AAPL)*2")
        pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views,
                                       sigma_views = sigma_views, sk_views = sk_views,
                                       opt = opt), rd)
        @test isapprox(pr.mu[1], 1.5 * pr0.mu[1], rtol = 5e-6)
        @test isapprox(pr.sigma[1, 1], 1.3 * pr0.sigma[1, 1], rtol = 5e-3)
        @test isapprox(Skewness(; w = pr.w, ve = SimpleVariance(; w = pr.w))([1],
                                                                             reshape(pr.X[:,
                                                                                          1],
                                                                                     :, 1)),
                       2 * Skewness()([1], reshape(pr0.X[:, 1], :, 1)), rtol = 5e-3)
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views,
                                                 sigma_views = sigma_views,
                                                 sk_views = sk_views), rd).w, rtol = 1e-6)

        kt_views = LinearConstraintEstimator(; val = "AAPL == 7.5")
        pr = prior(EntropyPoolingPrior(; sets = sets, kt_views = kt_views, opt = opt), rd)
        @test isapprox(HighOrderMoment(; w = pr.w,
                                       alg = StandardisedHighOrderMoment(;
                                                                         alg = FourthMoment(),
                                                                         ve = SimpleVariance(;
                                                                                             w = pr.w)))([1],
                                                                                                         reshape(pr.X[:,
                                                                                                                      1],
                                                                                                                 :,
                                                                                                                 1)),
                       7.5, rtol = 5e-3)
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 kt_views = kt_views), rd).w, rtol = 1e-2)

        mu_views = LinearConstraintEstimator(; val = "AAPL<=1.5*prior(AAPL)")
        sigma_views = LinearConstraintEstimator(; val = "AAPL==0.7prior(AAPL)")
        kt_views = LinearConstraintEstimator(; val = "AAPL >= prior(AAPL)*0.87")
        pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views,
                                       sigma_views = sigma_views, kt_views = kt_views,
                                       opt = opt), rd)
        @test pr.mu[1] <= 1.5 * pr0.mu[1]
        @test isapprox(pr.sigma[1, 1], 0.7 * pr0.sigma[1, 1], rtol = 1e-3)
        @test abs(HighOrderMoment(; w = pr.w,
                                  alg = StandardisedHighOrderMoment(; alg = FourthMoment(),
                                                                    ve = SimpleVariance(;
                                                                                        w = pr.w)))([1],
                                                                                                    reshape(pr.X[:,
                                                                                                                 1],
                                                                                                            :,
                                                                                                            1)) -
                  HighOrderMoment(;
                                  alg = StandardisedHighOrderMoment(; alg = FourthMoment()))([1],
                                                                                             reshape(pr.X[:,
                                                                                                          1],
                                                                                                     :,
                                                                                                     1)) *
                  0.87) <= sqrt(eps())
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 mu_views = mu_views,
                                                 sigma_views = sigma_views,
                                                 kt_views = kt_views), rd).w, rtol = 5e-3)

        rho_views = LinearConstraintEstimator(; val = "(AAPL, XOM) == 0.35")
        pr = prior(EntropyPoolingPrior(; sets = sets, rho_views = rho_views, opt = opt), rd)
        @test isapprox(cov2cor(pr.sigma)[1, end], 0.35, rtol = 5e-6)
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 rho_views = rho_views), rd).w, rtol = 1e-2)

        rho_views = LinearConstraintEstimator(; val = "(AAPL, XOM) == prior(AAPL,XOM)*0.94")
        pr = prior(EntropyPoolingPrior(; sets = sets, rho_views = rho_views, opt = opt), rd)
        @test isapprox(cov2cor(pr.sigma)[1, end], cov2cor(pr0.sigma)[1, end] * 0.94,
                       rtol = 5e-6)
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 rho_views = rho_views), rd).w, rtol = 5e-2)

        pr = prior(HighOrderPriorEstimator(;
                                           pe = EntropyPoolingPrior(;
                                                                    alg = H2_EntropyPooling(),
                                                                    sets = sets,
                                                                    mu_views = LinearConstraintEstimator(;
                                                                                                         val = ["AAPL<=0.92*prior(AAPL)",
                                                                                                                "XOM >= 0.83*prior(XOM)"]),
                                                                    sigma_views = LinearConstraintEstimator(;
                                                                                                            val = ["AAPL==1.2prior(AAPL)",
                                                                                                                   "WMT==1.4prior(WMT)"]),
                                                                    rho_views = LinearConstraintEstimator(;
                                                                                                          val = "(AAPL, XOM) == 0.35"),
                                                                    opt = opt)), rd)
        @test pr.mu[1] <= 0.92 * pr0.mu[1] + sqrt(eps())
        @test pr.mu[end] >= 0.83 * pr0.mu[end] - sqrt(eps())
        @test isapprox(pr.sigma[1, 1], 1.2 * pr0.sigma[1, 1], rtol = 1e-2)
        @test isapprox(pr.sigma[19, 19], 1.4 * pr0.sigma[19, 19], rtol = 5e-3)
        @test isapprox(cov2cor(pr.sigma)[1, end], 0.35, rtol = 1e-3)

        cvar_views = LinearConstraintEstimator(; val = "AAPL == 0.07")
        pr = prior(EntropyPoolingPrior(; sets = sets, cvar_views = cvar_views, opt = opt),
                   rd)
        @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, 1]), 0.07, rtol = 1e-6)
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 cvar_views = cvar_views), rd).w,
                       rtol = 5e-5)

        cvar_views = LinearConstraintEstimator(; val = "AAPL == prior(AAPL)*1.37")
        pr = prior(EntropyPoolingPrior(; sets = sets, cvar_views = cvar_views, opt = opt),
                   rd)
        @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, 1]),
                       ConditionalValueatRisk(;)(rd.X[:, 1]) * 1.37, rtol = 1e-6)
        @test isapprox(pr.w,
                       prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                                 cvar_views = cvar_views), rd).w,
                       rtol = 1e-5)

        cvar_views = LinearConstraintEstimator(; val = ["AAPL == 0.053", "XOM==0.045"])
        pr = prior(HighOrderPriorEstimator(;
                                           pe = EntropyPoolingPrior(; sets = sets,
                                                                    alg = H2_EntropyPooling(),
                                                                    cvar_views = cvar_views,
                                                                    opt = opt)), rd)
        @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, 1]), 0.053, rtol = 5e-5)
        @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, end]), 0.045, rtol = 1e-4)
        @test isapprox(pr.w,
                       prior(HighOrderPriorEstimator(;
                                                     pe = EntropyPoolingPrior(; sets = sets,
                                                                              alg = H1_EntropyPooling(),
                                                                              cvar_views = cvar_views)),
                             rd).w, rtol = 5e-3)

        mu_views = LinearConstraintEstimator(;
                                             val = ["AAPL<=0.75*prior(AAPL)",
                                                    "XOM >= 0.4*prior(XOM)"])
        sigma_views = LinearConstraintEstimator(;
                                                val = ["AAPL==0.2prior(AAPL)",
                                                       "WMT==1.4prior(WMT)"])
        rho_views = LinearConstraintEstimator(; val = "(AAPL, XOM) == 0.35")
        kt_views = LinearConstraintEstimator(; val = "AAPL >= prior(AAPL)*0.3")
        sk_views = LinearConstraintEstimator(; val = "WMT == prior(WMT)*1.4")
        pr = prior(HighOrderPriorEstimator(;
                                           pe = EntropyPoolingPrior(;
                                                                    alg = H0_EntropyPooling(),
                                                                    sets = sets,
                                                                    mu_views = mu_views,
                                                                    sigma_views = sigma_views,
                                                                    rho_views = rho_views,
                                                                    kt_views = kt_views,
                                                                    sk_views = sk_views,
                                                                    opt = opt)), rd)
        @test pr.mu[1] <= 0.75 * pr0.mu[1]
        @test pr.mu[end] >= 0.4 * pr0.mu[end]
        @test isapprox(pr.sigma[1, 1], 0.2 * pr0.sigma[1, 1], rtol = 1e-2)
        @test isapprox(pr.sigma[19, 19], 1.4 * pr0.sigma[19, 19], rtol = 5e-3)
        @test !isapprox(cov2cor(pr.sigma)[1, end], 0.35; rtol = 5e-4)
        @test HighOrderMoment(; w = pr.w,
                              alg = StandardisedHighOrderMoment(; alg = FourthMoment(),
                                                                ve = SimpleVariance(;
                                                                                    w = pr.w)))([1],
                                                                                                reshape(pr.X[:,
                                                                                                             1],
                                                                                                        :,
                                                                                                        1)) >=
              HighOrderMoment(; alg = StandardisedHighOrderMoment(; alg = FourthMoment()))([1],
                                                                                           reshape(pr.X[:,
                                                                                                        1],
                                                                                                   :,
                                                                                                   1)) *
              0.3
        @test !isapprox(Skewness(; w = pr.w, ve = SimpleVariance(; w = pr.w))([1],
                                                                              reshape(pr.X[:,
                                                                                           end - 1],
                                                                                      :, 1)),
                        1.4 * Skewness()([1], reshape(pr0.X[:, end - 1], :, 1));
                        rtol = 5e-3)
    end
    @testset "Opinion pooling" begin
        pr0 = prior(EmpiricalPrior(), rd)
        pr = prior(OpinionPoolingPrior(;
                                       pes = [EntropyPoolingPrior(; sets = sets,
                                                                  mu_views = LinearConstraintEstimator(;
                                                                                                       val = "AAPL == prior(AAPL)*1.5")),
                                              EntropyPoolingPrior(; sets = sets,
                                                                  mu_views = LinearConstraintEstimator(;
                                                                                                       val = "AAPL == prior(AAPL)*2"))],
                                       w = [0.4, 0.4]), rd)
        @test isapprox(pr.mu[1], 0.0022963115075039417, rtol = 1e-6)

        pr = prior(OpinionPoolingPrior(;
                                       pes = [EntropyPoolingPrior(; sets = sets,
                                                                  mu_views = LinearConstraintEstimator(;
                                                                                                       val = "AAPL == prior(AAPL)*1.5")),
                                              EntropyPoolingPrior(; sets = sets,
                                                                  mu_views = LinearConstraintEstimator(;
                                                                                                       val = "AAPL == prior(AAPL)*2"))],
                                       w = [0.5, 0.5], alg = LogarithmicOpinionPooling(),
                                       p = 5), rd)
        @test isapprox(pr.mu[1], 0.002511023272287914, rtol = 1e-6)
    end
end
