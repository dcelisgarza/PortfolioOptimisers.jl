@testset "Uncertainty set" begin
    using PortfolioOptimisers, Test, DataFrames, CSV, TimeSeries, StableRNGs, Random
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

    @testset "Ellipsoidal Uncertainty sets" begin
        rng = StableRNG(123456789)
        ues = [NormalUncertaintySet(; pe = EmpiricalPrior(), rng = rng, seed = 987654321,
                                    alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                             diagonal = true,
                                                                             method = NormalKUncertaintyAlgorithm())),
               NormalUncertaintySet(; pe = EmpiricalPrior(), rng = rng, seed = 987654321,
                                    alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                             diagonal = false,
                                                                             method = GeneralKUncertaintyAlgorithm())),
               NormalUncertaintySet(; pe = EmpiricalPrior(), rng = rng, seed = 987654321,
                                    alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                             diagonal = true,
                                                                             method = ChiSqKUncertaintyAlgorithm())),
               NormalUncertaintySet(; pe = EmpiricalPrior(), rng = rng, seed = 987654321,
                                    alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                             diagonal = true,
                                                                             method = 10)),
               ARCHUncertaintySet(;
                                  alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                           diagonal = true,
                                                                           method = NormalKUncertaintyAlgorithm()),
                                  seed = 987654321, bootstrap = StationaryBootstrap()),
               ARCHUncertaintySet(;
                                  alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                           diagonal = false,
                                                                           method = GeneralKUncertaintyAlgorithm()),
                                  seed = 987654321, bootstrap = MovingBootstrap()),
               ARCHUncertaintySet(;
                                  alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                           diagonal = true,
                                                                           method = ChiSqKUncertaintyAlgorithm()),
                                  seed = 987654321, bootstrap = CircularBootstrap())]
        df = CSV.read(joinpath(@__DIR__, "assets/EllipsoidalUncertaintySet.csv.gz"),
                      DataFrame)
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
