@safetestset "Prior tests" begin
    using PortfolioOptimisers, StatsBase, Random, StableRNGs, Test, CovarianceEstimation,
          CSV, DataFrames

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
    @testset "Empirical Prior" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 10) * 0.001
        assets = 1:10
        asset_sets = DataFrame(; Asset = assets, Clusters = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        pes = [EmpiricalPriorEstimator(), EmpiricalPriorEstimator(; horizon = 252)]
        pet = CSV.read(joinpath(@__DIR__, "./assets/Empirical-Prior.csv"), DataFrame)
        for i ∈ eachindex(pes)
            pm = prior(pes[i], transpose(X); dims = 2)
            mu_t = reshape(pet[1:10, i], size(pm.mu))
            sigma_t = reshape(pet[11:end, i], size(pm.sigma))
            res1 = isapprox(pm.mu, mu_t)
            if !res1
                println("Test $i fails on mu.")
                find_tol(pm.mu, mu_t; name1 = :er, name2 = :er_t)
            end
            @test res1
            res2 = isapprox(pm.sigma, sigma_t)
            if !res2
                println("Test $i fails on sigma.")
                find_tol(pm.sigma, sigma_t; name1 = :er, name2 = :er_t)
            end
            @test res2
        end
    end
    @testset "Black Litterman Prior" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 10) * 0.001

        assets = 1:10
        asset_sets = DataFrame(; Asset = assets, Clusters = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])

        vc_1 = LinearConstraintAtom(; group = :Asset, name = 2, coef = 1, cnst = 0.003)
        vc_2 = LinearConstraintAtom(; group = [:Asset, :Asset], name = [3, 8],
                                    coef = [1, -1], cnst = -0.001)
        vc_3 = LinearConstraintAtom(; group = [:Clusters, :Asset], name = [3, 9],
                                    coef = [1, -1], cnst = 0.002)
        vc_4 = LinearConstraintAtom(; group = [:Asset, :Clusters], name = [5, 1],
                                    coef = [1, -1], cnst = 0.007)
        vc_5 = LinearConstraintAtom(; group = :Clusters, name = 2, coef = 1, cnst = 0.001)
        views = [vc_1, vc_2, vc_3, vc_4, vc_5]
        pes = [BlackLittermanPriorEstimator(; views = views, asset_sets = asset_sets),
               BlackLittermanPriorEstimator(; views = views, asset_sets = asset_sets,
                                            rf = 0.0001),
               BlackLittermanPriorEstimator(;
                                            pe = EmpiricalPriorEstimator(;
                                                                         me = ExcessExpectedReturns()),
                                            views = views, asset_sets = asset_sets),
               BlackLittermanPriorEstimator(;
                                            pe = EmpiricalPriorEstimator(;
                                                                         me = ExcessExpectedReturns(;
                                                                                                    rf = 0.0001)),
                                            views = views, asset_sets = asset_sets,
                                            rf = 0.0001),
               BlackLittermanPriorEstimator(; views = views, asset_sets = asset_sets,
                                            views_conf = fill(eps(), length(views))),
               BlackLittermanPriorEstimator(; views = views, asset_sets = asset_sets,
                                            views_conf = fill(1.0, length(views)))]
        pet = CSV.read(joinpath(@__DIR__, "./assets/Black-Litterman-Prior.csv"), DataFrame)
        for i ∈ eachindex(pes)
            pm = prior(pes[i], transpose(X); dims = 2)
            mu_t = reshape(pet[1:10, i], size(pm.mu))
            sigma_t = reshape(pet[11:end, i], size(pm.sigma))
            res1 = isapprox(pm.mu, mu_t)
            if !res1
                println("Test $i fails on mu.")
                find_tol(pm.mu, mu_t; name1 = :er, name2 = :er_t)
            end
            @test res1
            res2 = isapprox(pm.sigma, sigma_t)
            if !res2
                println("Test $i fails on sigma.")
                find_tol(pm.sigma, sigma_t; name1 = :er, name2 = :er_t)
            end
            @test res2
        end

        P, Q = views_constraints(LinearConstraintAtom(; group = :Foo, name = 2, coef = 1,
                                                      cnst = 0.003), asset_sets)
        @test isempty(P)
        @test isempty(Q)

        @test_throws ArgumentError views_constraints(LinearConstraintAtom(; group = :Foo,
                                                                          name = 2,
                                                                          coef = 1,
                                                                          cnst = 0.003),
                                                     asset_sets; strict = true)

        P, Q = views_constraints(LinearConstraintAtom(; group = [:Foo], name = [2],
                                                      coef = [1], cnst = 0.003), asset_sets)
        @test isempty(P)
        @test isempty(Q)

        @test_throws ArgumentError views_constraints(LinearConstraintAtom(; group = [:Foo],
                                                                          name = [2],
                                                                          coef = [1],
                                                                          cnst = 0.003),
                                                     asset_sets; strict = true)
    end
    @testset "Factor Model Prior" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10)
        F = X[:, [3, 8]]

        pm1 = prior(FactorModelPriorEstimator(; residuals = false), transpose(X),
                    transpose(F); dims = 2)
        pm1_t = CSV.read(joinpath(@__DIR__, "./assets/Factor-Model-Prior-No-Residuals.csv"),
                         DataFrame)
        X_t = reshape(view(pm1_t, 1:1000, 1), 100, 10)
        mu_t = view(pm1_t, 1001:1010, 1)
        sigma_t = reshape(view(pm1_t, 1011:1110, 1), 10, 10)
        csigma_t = reshape(view(pm1_t, 1111:nrow(pm1_t), 1), :, 10)

        @test isapprox(pm1.X, X_t)
        @test isapprox(pm1.mu, mu_t)
        @test isapprox(pm1.sigma, sigma_t)
        @test isapprox(pm1.chol, csigma_t)

        pm2 = prior(FactorModelPriorEstimator(; residuals = true), transpose(X),
                    transpose(F); dims = 2)
        pm2_t = CSV.read(joinpath(@__DIR__, "./assets/Factor-Model-Prior-Residuals.csv"),
                         DataFrame)
        X_t = reshape(view(pm2_t, 1:1000, 1), 100, 10)
        mu_t = view(pm2_t, 1001:1010, 1)
        sigma_t = reshape(view(pm2_t, 1011:1110, 1), 10, 10)
        csigma_t = reshape(view(pm2_t, 1111:nrow(pm2_t), 1), :, 10)

        @test isapprox(pm2.X, X_t)
        @test isapprox(pm2.mu, mu_t)
        @test isapprox(pm2.sigma, sigma_t)
        @test isapprox(pm2.chol, csigma_t)
    end
end
