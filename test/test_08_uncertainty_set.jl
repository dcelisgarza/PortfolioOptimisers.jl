@safetestset "Uncertainty set" begin
    using PortfolioOptimisers, Test, DataFrames, CSV, TimeSeries, StableRNGs, Random
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
    @testset "Box Uncertainty sets" begin
        rng = StableRNG(123456789)
        ues = [DeltaUncertaintySet(;),
               NormalUncertaintySet(; pe = EmpiricalPrior(), rng = rng,
                                    alg = BoxUncertaintySetAlgorithm(), seed = 987654321),
               ARCHUncertaintySet(; alg = BoxUncertaintySetAlgorithm(),
                                  bootstrap = StationaryBootstrap(), seed = 987654321),
               ARCHUncertaintySet(; alg = BoxUncertaintySetAlgorithm(),
                                  bootstrap = MovingBootstrap(), seed = 987654321),
               ARCHUncertaintySet(; alg = BoxUncertaintySetAlgorithm(),
                                  bootstrap = CircularBootstrap(), seed = 987654321)]
        df = CSV.read(joinpath(@__DIR__, "assets/BoxUncertaintySet.csv.gz"), DataFrame)
        for (i, ue) in pairs(ues)
            mu_set1, sigma_set1 = ucs(ue, rd.X)
            mu1 = [mu_set1.lb; mu_set1.ub]
            sigma1 = [vec(sigma_set1.lb); vec(sigma_set1.ub)]

            mu_set2 = mu_ucs(ue, rd.X)
            mu2 = [mu_set2.lb; mu_set2.ub]

            sigma_set2 = sigma_ucs(ue, rd.X)
            sigma2 = [vec(sigma_set2.lb); vec(sigma_set2.ub)]

            success = isapprox(mu1, mu2)
            if !success
                println("Counter: $i")
                find_tol(mu1, mu2)
            end
            @test success

            success = isapprox(sigma1, sigma2)
            if !success
                println("Counter: $i")
                find_tol(sigma1, sigma2)
            end
            @test success

            success = isapprox([mu1; sigma1], df[!, i])
            if !success
                println("Dataframe iteration $i failed")
                find_tol([mu1; sigma1], df[!, i]; name1 = :sigma1, name2 = :sigma2)
            end
            @test success
        end
    end

    @testset "Ellipse Uncertainty sets" begin
        rng = StableRNG(123456789)
        ues = [NormalUncertaintySet(; pe = EmpiricalPrior(), rng = rng, seed = 987654321,
                                    alg = EllipseUncertaintySetAlgorithm(; diagonal = true,
                                                                         method = NormalKUncertaintyAlgorithm())),
               NormalUncertaintySet(; pe = EmpiricalPrior(), rng = rng, seed = 987654321,
                                    alg = EllipseUncertaintySetAlgorithm(; diagonal = false,
                                                                         method = GeneralKUncertaintyAlgorithm())),
               NormalUncertaintySet(; pe = EmpiricalPrior(), rng = rng, seed = 987654321,
                                    alg = EllipseUncertaintySetAlgorithm(; diagonal = true,
                                                                         method = ChiSqKUncertaintyAlgorithm())),
               NormalUncertaintySet(; pe = EmpiricalPrior(), rng = rng, seed = 987654321,
                                    alg = EllipseUncertaintySetAlgorithm(; diagonal = true,
                                                                         method = 10)),
               ARCHUncertaintySet(;
                                  alg = EllipseUncertaintySetAlgorithm(; diagonal = true,
                                                                       method = NormalKUncertaintyAlgorithm()),
                                  seed = 987654321, bootstrap = StationaryBootstrap()),
               ARCHUncertaintySet(;
                                  alg = EllipseUncertaintySetAlgorithm(; diagonal = false,
                                                                       method = GeneralKUncertaintyAlgorithm()),
                                  seed = 987654321, bootstrap = MovingBootstrap()),
               ARCHUncertaintySet(;
                                  alg = EllipseUncertaintySetAlgorithm(; diagonal = true,
                                                                       method = ChiSqKUncertaintyAlgorithm()),
                                  seed = 987654321, bootstrap = CircularBootstrap())]
        df = CSV.read(joinpath(@__DIR__, "assets/EllipseUncertaintySet.csv.gz"), DataFrame)
        for (i, ue) in pairs(ues)
            mu_set1, sigma_set1 = ucs(ue, rd.X)
            mu1 = [vec(mu_set1.sigma); mu_set1.k]
            sigma1 = [vec(sigma_set1.sigma); sigma_set1.k]

            mu_set2 = mu_ucs(ue, rd.X)
            mu2 = [vec(mu_set2.sigma); mu_set2.k]
            sigma_set2 = sigma_ucs(ue, rd.X)
            sigma2 = [vec(sigma_set2.sigma); sigma_set2.k]

            success = isapprox(mu1, mu2; rtol = 1e-6)
            if !success
                println("Mu iteration $i failed")
                find_tol(mu1, mu2)
            end
            @test success

            res = [mu1; sigma1; mu2; sigma2]
            success = isapprox(res, df[!, i]; rtol = 1e-6)
            if !success
                println("Data iteration $i failed")
                find_tol(res, df[!, i])
            end
            @test success
        end
    end
end
