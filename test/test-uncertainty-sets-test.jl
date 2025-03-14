@safetestset "Uncertainty tests" begin
    using PortfolioOptimisers, Test, Random, StableRNGs, CSV, DataFrames
    function find_tol(a1, a2; name1 = :a1, name2 = :a2)
        for rtol ∈
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; rtol = rtol)
                println("isapprox($name1, $name2, rtol = $(rtol))")
                break
            end
        end
    end
    @testset "No uncertainty sets" begin
        @test isa(uncertainty_set(NoUncertaintySetEstimator()), NoUncertaintySet)
        @test isa(mu_uncertainty_set(NoUncertaintySetEstimator()), NoUncertaintySet)
        @test isa(sigma_uncertainty_set(NoUncertaintySetEstimator()), NoUncertaintySet)
    end
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
            if i == 5 && Sys.iswindows()
                continue
            end
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

            if (i == 13 && (Sys.islinux() || Sys.isapple())) || (i == 5 && Sys.isapple())
                continue
            end
            res2 = isapprox(sigma2, uesigma_t[!, i])
            if !res2
                println("Sigma iteration $i failed")
                find_tol(sigma2, uesigma_t[!, i]; name1 = :sigma1, name2 = :uesigma_t)
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
