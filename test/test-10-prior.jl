@safetestset "Prior" begin
    using Test, PortfolioOptimisers, DataFrames, TimeSeries, CSV, StatsBase,
          CovarianceEstimation
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
        @test (pr2.sk, pr2.V) ==
              coskewness(Coskewness(; alg = Full()), pr2.X; mean = transpose(pr2.mu))
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
    @testset "Factor Black Litterman" begin
        df = CSV.read(joinpath(@__DIR__, "./assets/FactorBlackLitterman.csv.gz"), DataFrame)
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
    end
end
#=
using PortfolioOptimiser
portfolio = Portfolio(;
                      prices = TimeArray(CSV.File(joinpath(@__DIR__,
                                                           "./assets/SP500.csv.gz"));
                                         timestamp = :Date)[(end - 252 * 4):end],
                      f_prices = TimeArray(CSV.File(joinpath(@__DIR__,
                                                             "./assets/Factors.csv.gz"));
                                           timestamp = :Date)[(end - 252 * 4):end])
asset_statistics!(portfolio)
factor_statistics!(portfolio)
views_factors = DataFrame("Enabled" => [true, true], "Factor" => ["MTUM", "USMV"],
                          "Sign" => [">=", "<="], "Value" => [0.0001, 0.0003],
                          "Relative_Factor" => ["", "QUAL"])

f_P, f_Q = factor_views(views_factors, portfolio.loadings)

bl_type = ABLType(; delta = 2, eq = false)
black_litterman_factor_statistics!(portfolio; f_P = f_P, f_Q = f_Q, bl_type = bl_type)
=#
