@safetestset "Uncertainty tests" begin
    using PortfolioOptimisers, Test, Random, StableRNGs, CSV, DataFrames
    @testset "Box Uncertainty sets" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)

        ues = [DeltaUncertaintySetEstimator(;),
               NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(), rng = rng,
                                             class = BoxUncertaintySetClass(),
                                             seed = 987654321),
               ARCHUncertaintySetEstimator(; class = BoxUncertaintySetClass(),
                                           bootstrap = StationaryBootstrap(),
                                           seed = 987654321),
               ARCHUncertaintySetEstimator(; class = BoxUncertaintySetClass(),
                                           bootstrap = MovingBootstrap(), seed = 987654321),
               ARCHUncertaintySetEstimator(; class = BoxUncertaintySetClass(),
                                           bootstrap = CircularBootstrap(),
                                           seed = 987654321)]
        ues_t = CSV.read(joinpath(@__DIR__, "assets/Box-Uncertainty-Sets.csv"), DataFrame)
        for (i, ue) ∈ pairs(ues)
            mu_set1, sigma_set1 = uncertainty_set(ue, transpose(X); dims = 2)
            mu1 = [mu_set1.lo; mu_set1.hi]
            sigma1 = [vec(sigma_set1.lo); vec(sigma_set1.hi)]

            mu_set2 = mu_uncertainty_set(ue, transpose(X); dims = 2)
            mu2 = [mu_set2.lo; mu_set2.hi]

            sigma_set2 = sigma_uncertainty_set(ue, transpose(X); dims = 2)
            sigma2 = [vec(sigma_set2.lo); vec(sigma_set2.hi)]

            res1 = isapprox(mu1, mu2)
            if !res1
                println("Mu iteration $i failed")
                find_tol(mu1, mu2; name1 = :mu1, name2 = :mu2)
            end
            @test res1

            res2 = isapprox(sigma1, sigma2)
            if !res2
                println("Sigma iteration $i failed")
                find_tol(sigma1, sigma2; name1 = :sigma1, name2 = :sigma2)
            end
            @test res2

            res3 = isapprox([mu1; sigma1], ues_t[!, i])
            if !res3
                println("Dataframe iteration $i failed")
                find_tol([mu1; sigma1], ues_t[!, i]; name1 = :sigma1, name2 = :sigma2)
            end
            @test res3
        end
    end
    @testset "Ellipse Uncertainty sets" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 5)
        df = DataFrame()

        ues = [NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(), rng = rng,
                                             class = EllipseUncertaintySetClass(;
                                                                                diagonal = true,
                                                                                method = NormalKUncertaintyMethod()),
                                             seed = 987654321),
               NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(), rng = rng,
                                             class = EllipseUncertaintySetClass(;
                                                                                diagonal = true,
                                                                                method = GeneralKUncertaintyMethod()),
                                             seed = 987654321),
               NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(), rng = rng,
                                             class = EllipseUncertaintySetClass(;
                                                                                diagonal = true,
                                                                                method = ChiSqKUncertaintyMethod()),
                                             seed = 987654321),
               NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(), rng = rng,
                                             class = EllipseUncertaintySetClass(;
                                                                                diagonal = true,
                                                                                method = 10),
                                             seed = 987654321),
               NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(), rng = rng,
                                             class = EllipseUncertaintySetClass(;
                                                                                diagonal = false,
                                                                                method = NormalKUncertaintyMethod()),
                                             seed = 987654321),
               NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(), rng = rng,
                                             class = EllipseUncertaintySetClass(;
                                                                                diagonal = false,
                                                                                method = GeneralKUncertaintyMethod()),
                                             seed = 987654321),
               NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(), rng = rng,
                                             class = EllipseUncertaintySetClass(;
                                                                                diagonal = false,
                                                                                method = ChiSqKUncertaintyMethod()),
                                             seed = 987654321),
               NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(), rng = rng,
                                             class = EllipseUncertaintySetClass(;
                                                                                diagonal = false,
                                                                                method = 10),
                                             seed = 987654321),
               ARCHUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                           class = EllipseUncertaintySetClass(;
                                                                              diagonal = true,
                                                                              method = NormalKUncertaintyMethod()),
                                           seed = 987654321,
                                           bootstrap = StationaryBootstrap()),
               ARCHUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                           class = EllipseUncertaintySetClass(;
                                                                              diagonal = true,
                                                                              method = GeneralKUncertaintyMethod()),
                                           seed = 987654321, bootstrap = MovingBootstrap()),
               ARCHUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                           class = EllipseUncertaintySetClass(;
                                                                              diagonal = true,
                                                                              method = ChiSqKUncertaintyMethod()),
                                           seed = 987654321,
                                           bootstrap = CircularBootstrap()),
               ARCHUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                           class = EllipseUncertaintySetClass(;
                                                                              diagonal = true,
                                                                              method = 10),
                                           seed = 987654321,
                                           bootstrap = StationaryBootstrap()),
               ARCHUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                           class = EllipseUncertaintySetClass(;
                                                                              diagonal = false,
                                                                              method = NormalKUncertaintyMethod()),
                                           seed = 987654321, bootstrap = MovingBootstrap()),
               ARCHUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                           class = EllipseUncertaintySetClass(;
                                                                              diagonal = false,
                                                                              method = GeneralKUncertaintyMethod()),
                                           seed = 987654321,
                                           bootstrap = CircularBootstrap()),
               ARCHUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                           class = EllipseUncertaintySetClass(;
                                                                              diagonal = false,
                                                                              method = ChiSqKUncertaintyMethod()),
                                           seed = 987654321,
                                           bootstrap = StationaryBootstrap()),
               ARCHUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                           class = EllipseUncertaintySetClass(;
                                                                              diagonal = false,
                                                                              method = 10),
                                           seed = 987654321, bootstrap = MovingBootstrap())]
        ues_t = CSV.read(joinpath(@__DIR__, "assets/Ellipse-Uncertainty-Sets.csv"),
                         DataFrame)
        uesigma_t = CSV.read(joinpath(@__DIR__,
                                      "assets/Ellipse-Uncertainty-Sets-Sigma.csv"),
                             DataFrame)
        for (i, ue) ∈ pairs(ues)
            mu_set1, sigma_set1 = uncertainty_set(ue, transpose(X); dims = 2)
            mu1 = [vec(mu_set1.sigma); mu_set1.k]
            sigma1 = [vec(sigma_set1.sigma); sigma_set1.k]

            mu_set2 = mu_uncertainty_set(ue, transpose(X); dims = 2)
            mu2 = [vec(mu_set2.sigma); mu_set2.k]

            sigma_set2 = sigma_uncertainty_set(ue, transpose(X); dims = 2)
            sigma2 = [vec(sigma_set2.sigma); sigma_set2.k]

            res1 = isapprox(mu1, mu2)
            if !res1
                println("Mu iteration $i failed")
                find_tol(mu1, mu2; name1 = :mu1, name2 = :mu2)
            end
            @test res1

            res2 = isapprox(sigma2, uesigma_t[!, i])
            if !res2
                println("Sigma iteration $i failed")
                find_tol(sigma2, uesigma_t; name1 = :sigma1, name2 = :uesigma_t)
            end
            @test res2

            res3 = isapprox([mu1; sigma1], ues_t[!, i])
            if !res3
                println("Dataframe iteration $i failed")
                find_tol([mu1; sigma1], ues_t[!, i]; name1 = :sigma1, name2 = :sigma2)
            end
            @test res3
        end
    end
end
