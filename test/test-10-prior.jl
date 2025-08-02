@safetestset "Prior" begin
    using Test, PortfolioOptimisers, DataFrames, TimeSeries, CSV, StatsBase
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
end
