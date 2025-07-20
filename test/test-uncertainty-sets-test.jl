#=
@safetestset "Uncertainty tests" begin
    using PortfolioOptimisers, Test, Random, StableRNGs, CSV, DataFrames, TimeSeries
    import PortfolioOptimisers: ucs_factory, ucs_view
    function find_tol(a1, a2; name1 = :a1, name2 = :a2)
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
    Xp = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/asset_prices.csv"));
                   timestamp = :timestamp)
    @testset "No uncertainty sets" begin
        @test isa(ucs(nothing), Nothing)
        @test isa(mu_ucs(nothing), Nothing)
        @test isa(sigma_ucs(nothing), Nothing)
    end
    @testset "Box Uncertainty sets" begin
        rng = StableRNG(123456789)
        rd = prices_to_returns(Xp[(end - 252):end])
        X = rd.X
        ues = [DeltaUncertaintySetEstimator(;),
               NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(), rng = rng,
                                             alg = BoxUncertaintySetAlgorithm(),
                                             seed = 987654321),
               ARCHUncertaintySetEstimator(; alg = BoxUncertaintySetAlgorithm(),
                                           bootstrap = StationaryBootstrap(),
                                           seed = 987654321),
               ARCHUncertaintySetEstimator(; alg = BoxUncertaintySetAlgorithm(),
                                           bootstrap = MovingBootstrap(), seed = 987654321),
               ARCHUncertaintySetEstimator(; alg = BoxUncertaintySetAlgorithm(),
                                           bootstrap = CircularBootstrap(),
                                           seed = 987654321)]
        ues_t = CSV.read(joinpath(@__DIR__, "assets/Box-Uncertainty-Sets.csv"), DataFrame)
        for (i, ue) in enumerate(ues)
            mu_set1, sigma_set1 = ucs(ue, transpose(X); dims = 2)
            mu1 = [mu_set1.lb; mu_set1.ub]
            sigma1 = [vec(sigma_set1.lb); vec(sigma_set1.ub)]

            mu_set2 = mu_ucs(ue, transpose(X); dims = 2)
            mu2 = [mu_set2.lb; mu_set2.ub]

            sigma_set2 = sigma_ucs(ue, transpose(X); dims = 2)
            sigma2 = [vec(sigma_set2.lb); vec(sigma_set2.ub)]

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

            mu_set3, sigma_set3 = ucs((mu_set1, sigma_set1))
            @test mu_set1 === mu_set3
            @test sigma_set1 === sigma_set3

            mu_set4 = mu_ucs(mu_set1)
            @test mu_set1 === mu_set4

            sigma_set4 = sigma_ucs(sigma_set1)
            @test sigma_set1 === sigma_set4
        end
        ucrm1, ucrs1 = ucs(ues[1], X)
        ucrm2, ucrs2 = ucs(ues[1], ReturnsResult(; nx = 1:30, X = X))
        @test ucrm1.lb == ucrm2.lb
        @test ucrm1.ub == ucrm2.ub
        @test ucrs1.lb == ucrs2.lb
        @test ucrs1.ub == ucrs2.ub

        ucrm1 = mu_ucs(ues[1], X)
        ucrm2 = mu_ucs(ues[1], ReturnsResult(; nx = 1:30, X = X))
        @test ucrm1.lb == ucrm2.lb
        @test ucrm1.ub == ucrm2.ub

        ucrs1 = sigma_ucs(ues[1], X)
        ucrs2 = sigma_ucs(ues[1], ReturnsResult(; nx = 1:30, X = X))
        @test ucrs1.lb == ucrs2.lb
        @test ucrs1.ub == ucrs2.ub

        @test isnothing(ucs_factory(nothing, nothing))
        @test ues[1] === ucs_factory(ues[1], ues[2])
        @test ues[1] === ucs_view(ues[1], [3])
        @test ues[1] === ucs_factory(ues[1], ucrm1)
        @test ucrm1 === ucs_factory(ucrm1, ues[2])
        ucrm2 = ucs_view(ucrm1, [3])
        @test ucrm2.lb == view(ucrm1.lb, [3])
        @test ucrm2.ub == view(ucrm1.ub, [3])

        @test ues[2] === ucs_factory(nothing, ues[2])
        @test ues[2] === ucs_view(ues[2], [3])
        @test ucrm1 === ucs_factory(nothing, ucrm1)
        ucrm2 = ucs_view(ucrm1, [3])
        @test ucrm2.lb == view(ucrm1.lb, [3])
        @test ucrm2.ub == view(ucrm1.ub, [3])

        @test ucrs1 === ucs_factory(ucrs1, ues[2])
        ucrs2 = ucs_view(ucrs1, [3])
        @test ucrs2.lb == view(ucrs1.lb, [3], [3])
        @test ucrs2.ub == view(ucrs1.ub, [3], [3])

        @test ues[2] === ucs_factory(nothing, ues[2])
        @test ues[2] === ucs_view(ues[2], [3])
        @test ucrs1 === ucs_factory(nothing, ucrs1)
        ucrs2 = ucs_view(ucrs1, [3])
        @test ucrs2.lb == view(ucrs1.lb, [3], [3])
        @test ucrs2.ub == view(ucrs1.ub, [3], [3])
    end
    @testset "Ellipse Uncertainty sets" begin
        rng = StableRNG(123456789)
        rd = prices_to_returns(Xp[(end - 252):end][:A, :GOOG, :AMZN, :T])
        X = rd.X
        ues = [NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(), rng = rng,
                                             alg = EllipseUncertaintySetAlgorithm(;
                                                                                  diagonal = true,
                                                                                  method = NormalKUncertaintyAlgorithm()),
                                             seed = 987654321),
               NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(), rng = rng,
                                             alg = EllipseUncertaintySetAlgorithm(;
                                                                                  diagonal = true,
                                                                                  method = GeneralKUncertaintyAlgorithm()),
                                             seed = 987654321),
               NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(), rng = rng,
                                             alg = EllipseUncertaintySetAlgorithm(;
                                                                                  diagonal = true,
                                                                                  method = ChiSqKUncertaintyAlgorithm()),
                                             seed = 987654321),
               NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(), rng = rng,
                                             alg = EllipseUncertaintySetAlgorithm(;
                                                                                  diagonal = true,
                                                                                  method = 10),
                                             seed = 987654321),
               NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(), rng = rng,
                                             alg = EllipseUncertaintySetAlgorithm(;
                                                                                  diagonal = false,
                                                                                  method = NormalKUncertaintyAlgorithm()),
                                             seed = 987654321),
               NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(), rng = rng,
                                             alg = EllipseUncertaintySetAlgorithm(;
                                                                                  diagonal = false,
                                                                                  method = GeneralKUncertaintyAlgorithm()),
                                             seed = 987654321),
               NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(), rng = rng,
                                             alg = EllipseUncertaintySetAlgorithm(;
                                                                                  diagonal = false,
                                                                                  method = ChiSqKUncertaintyAlgorithm()),
                                             seed = 987654321),
               NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(), rng = rng,
                                             alg = EllipseUncertaintySetAlgorithm(;
                                                                                  diagonal = false,
                                                                                  method = 10),
                                             seed = 987654321),
               ARCHUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                           alg = EllipseUncertaintySetAlgorithm(;
                                                                                diagonal = true,
                                                                                method = NormalKUncertaintyAlgorithm()),
                                           seed = 987654321,
                                           bootstrap = StationaryBootstrap()),
               ARCHUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                           alg = EllipseUncertaintySetAlgorithm(;
                                                                                diagonal = true,
                                                                                method = GeneralKUncertaintyAlgorithm()),
                                           seed = 987654321, bootstrap = MovingBootstrap()),
               ARCHUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                           alg = EllipseUncertaintySetAlgorithm(;
                                                                                diagonal = true,
                                                                                method = ChiSqKUncertaintyAlgorithm()),
                                           seed = 987654321,
                                           bootstrap = CircularBootstrap()),
               ARCHUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                           alg = EllipseUncertaintySetAlgorithm(;
                                                                                diagonal = true,
                                                                                method = 10),
                                           seed = 987654321,
                                           bootstrap = StationaryBootstrap()),
               ARCHUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                           alg = EllipseUncertaintySetAlgorithm(;
                                                                                diagonal = false,
                                                                                method = NormalKUncertaintyAlgorithm()),
                                           seed = 987654321, bootstrap = MovingBootstrap()),
               ARCHUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                           alg = EllipseUncertaintySetAlgorithm(;
                                                                                diagonal = false,
                                                                                method = GeneralKUncertaintyAlgorithm()),
                                           seed = 987654321,
                                           bootstrap = CircularBootstrap()),
               ARCHUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                           alg = EllipseUncertaintySetAlgorithm(;
                                                                                diagonal = false,
                                                                                method = ChiSqKUncertaintyAlgorithm()),
                                           seed = 987654321,
                                           bootstrap = StationaryBootstrap()),
               ARCHUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                           alg = EllipseUncertaintySetAlgorithm(;
                                                                                diagonal = false,
                                                                                method = 10),
                                           seed = 987654321, bootstrap = MovingBootstrap())]
        ues_t = CSV.read(joinpath(@__DIR__, "assets/Ellipse-Uncertainty-Sets.csv"),
                         DataFrame)
        uesigma_t = CSV.read(joinpath(@__DIR__,
                                      "assets/Ellipse-Uncertainty-Sets-Sigma.csv"),
                             DataFrame)
        for (i, ue) in enumerate(ues)
            mu_set1, sigma_set1 = ucs(ue, transpose(X); dims = 2)
            mu1 = [vec(mu_set1.sigma); mu_set1.k]
            sigma1 = [vec(sigma_set1.sigma); sigma_set1.k]

            mu_set2 = mu_ucs(ue, transpose(X); dims = 2)
            mu2 = [vec(mu_set2.sigma); mu_set2.k]

            sigma_set2 = sigma_ucs(ue, transpose(X); dims = 2)
            sigma2 = [vec(sigma_set2.sigma); sigma_set2.k]

            res1 = isapprox(mu1, mu2)
            if !res1
                println("Mu iteration $i failed")
                find_tol(mu1, mu2; name1 = :mu1, name2 = :mu2)
            end
            @test res1

            rtol = if i == 5
                0.05
            elseif i == 13
                0.01
            else
                1e-6
            end
            res2 = isapprox(sigma2, uesigma_t[!, i]; rtol = rtol)
            if !res2
                println("Sigma iteration $i failed")
                find_tol(sigma2, uesigma_t[!, i]; name1 = :sigma1, name2 = :uesigma_t)
            end
            @test res2

            rtol = if i == 5
                0.005
            elseif i == 13
                0.01
            else
                1e-6
            end
            res3 = isapprox([mu1; sigma1], ues_t[!, i]; rtol = rtol)
            if !res3
                println("Dataframe iteration $i failed")
                find_tol([mu1; sigma1], ues_t[!, i]; name1 = :sigma1, name2 = :sigma2)
            end
            @test res3

            mu_set3, sigma_set3 = ucs((mu_set1, sigma_set1))
            @test mu_set1 === mu_set3
            @test sigma_set1 === sigma_set3

            mu_set4 = mu_ucs(mu_set1)
            @test mu_set1 === mu_set4

            sigma_set4 = sigma_ucs(sigma_set1)
            @test sigma_set1 === sigma_set4
        end
        ucrm1, ucrs1 = ucs(ues[1], X)
        ucrm2, ucrs2 = ucs(ues[1], ReturnsResult(; nx = 1:4, X = X))
        @test ucrm1.sigma == ucrm2.sigma
        @test ucrm1.k == ucrm2.k
        @test ucrs1.sigma == ucrs2.sigma
        @test ucrs1.k == ucrs2.k

        ucrm1 = mu_ucs(ues[1], X)
        ucrm2 = mu_ucs(ues[1], ReturnsResult(; nx = 1:4, X = X))
        @test ucrm1.sigma == ucrm2.sigma
        @test ucrm1.k == ucrm2.k

        ucrs1 = sigma_ucs(ues[1], X)
        ucrs2 = sigma_ucs(ues[1], ReturnsResult(; nx = 1:4, X = X))
        @test ucrs1.sigma == ucrs2.sigma
        @test ucrs1.k == ucrs2.k

        @test isnothing(ucs_factory(nothing, nothing))
        @test isnothing(ucs_view(nothing, 3))
        @test ues[1] === ucs_factory(ues[1], ues[2])
        @test ues[1] === ucs_view(ues[1], [3])
        @test ues[1] === ucs_factory(ues[1], ucrm1)
        @test ues[1] === ucs_view(ues[1], [3])
        @test ucrm1 === ucs_factory(ucrm1, ues[2])
        ucrm2 = ucs_view(ucrm1, [3])
        @test ucrm2.sigma == view(ucrm1.sigma, [3], [3])
        @test ucrm2.k == ucrm1.k

        @test ues[2] === ucs_factory(nothing, ues[2])
        @test ues[2] === ucs_view(ues[2], [3])
        @test ucrm1 === ucs_factory(nothing, ucrm1)
        ucrm2 = ucs_view(ucrm1, [3])
        @test ucrm2.sigma == view(ucrm1.sigma, [3], [3])
        @test ucrm2.k == ucrm1.k

        @test ues[1] === ucs_factory(ues[1], ucrs1)
        @test ues[1] === ucs_view(ues[1], [3, 1])
        @test ucrs1 === ucs_factory(ucrs1, ues[2])
        ucrm2 = ucs_view(ucrs1, [3, 1])
        i = PortfolioOptimisers.fourth_moment_index_factory(floor(Int,
                                                                  sqrt(size(ucrs1.sigma, 1))),
                                                            [3, 1])
        @test ucrm2.sigma == view(ucrs1.sigma, i, i)
        @test ucrm2.k == ucrs1.k

        @test ues[2] === ucs_factory(nothing, ues[2])
        @test ues[2] === ucs_view(ues[2], [3, 1])
        @test ucrs1 === ucs_factory(nothing, ucrs1)
        ucrm2 = ucs_view(ucrs1, [3, 1])
        @test ucrm2.sigma == view(ucrs1.sigma, i, i)
        @test ucrm2.k == ucrs1.k
    end
end
=#
