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
        sets = DataFrame(; Asset = assets, Clusters = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
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
        sets = DataFrame(; Asset = assets, Clusters = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])

        vc_1 = LinearConstraintAtom(; group = :Asset, name = 2, coef = 1, cnst = 0.003)
        vc_2 = LinearConstraintAtom(; group = [:Asset, :Asset], name = [3, 8],
                                    coef = [1, -1], cnst = -0.001)
        vc_3 = LinearConstraintAtom(; group = [:Clusters, :Asset], name = [3, 9],
                                    coef = [1, -1], cnst = 0.002)
        vc_4 = LinearConstraintAtom(; group = [:Asset, :Clusters], name = [5, 1],
                                    coef = [1, -1], cnst = 0.007)
        vc_5 = LinearConstraintAtom(; group = :Clusters, name = 2, coef = 1, cnst = 0.001)
        views = [vc_1, vc_2, vc_3, vc_4, vc_5]
        pes = [BlackLittermanPriorEstimator(; views = views, sets = sets),
               BlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.0001),
               BlackLittermanPriorEstimator(;
                                            pe = EmpiricalPriorEstimator(;
                                                                         me = ExcessExpectedReturns()),
                                            views = views, sets = sets),
               BlackLittermanPriorEstimator(;
                                            pe = EmpiricalPriorEstimator(;
                                                                         me = ExcessExpectedReturns(;
                                                                                                    rf = 0.0001)),
                                            views = views, sets = sets, rf = 0.0001),
               BlackLittermanPriorEstimator(; views = views, sets = sets,
                                            views_conf = fill(eps(), length(views))),
               BlackLittermanPriorEstimator(; views = views, sets = sets,
                                            views_conf = fill(1.0 - eps(), length(views)))]
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
                                                      cnst = 0.003), sets)
        @test isempty(P)
        @test isempty(Q)

        @test_throws ArgumentError views_constraints(LinearConstraintAtom(; group = :Foo,
                                                                          name = 2,
                                                                          coef = 1,
                                                                          cnst = 0.003),
                                                     sets; strict = true)

        P, Q = views_constraints(LinearConstraintAtom(; group = [:Foo], name = [2],
                                                      coef = [1], cnst = 0.003), sets)
        @test isempty(P)
        @test isempty(Q)

        @test_throws ArgumentError views_constraints(LinearConstraintAtom(; group = [:Foo],
                                                                          name = [2],
                                                                          coef = [1],
                                                                          cnst = 0.003),
                                                     sets; strict = true)

        P, Q = views_constraints(LinearConstraintAtom(; group = :Asset, name = 11, coef = 1,
                                                      cnst = 0.003), sets)
        @test isempty(P)
        @test isempty(Q)

        @test_throws ArgumentError views_constraints(LinearConstraintAtom(; group = :Asset,
                                                                          name = 11,
                                                                          coef = 1,
                                                                          cnst = 0.003),
                                                     sets, strict = true)

        P, Q = views_constraints(LinearConstraintAtom(; group = [:Asset], name = [11],
                                                      coef = [1], cnst = 0.003), sets)
        @test isempty(P)
        @test isempty(Q)

        @test_throws ArgumentError views_constraints(LinearConstraintAtom(;
                                                                          group = [:Asset],
                                                                          name = [11],
                                                                          coef = [1],
                                                                          cnst = 0.003),
                                                     sets, strict = true)
    end
    @testset "Factor Prior" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10)
        F = X[:, [3, 8]]

        pm1 = prior(FactorPriorEstimator(; residuals = false), transpose(X), transpose(F);
                    dims = 2)
        pm1_t = CSV.read(joinpath(@__DIR__, "./assets/Factor-Prior-No-Residuals.csv"),
                         DataFrame)
        X_t = reshape(view(pm1_t, 1:1000, 1), 100, 10)
        mu_t = view(pm1_t, 1001:1010, 1)
        sigma_t = reshape(view(pm1_t, 1011:1110, 1), 10, 10)
        chol_t = reshape(view(pm1_t, 1111:nrow(pm1_t), 1), :, 10)

        @test isapprox(pm1.X, X_t)
        @test isapprox(pm1.mu, mu_t)
        @test isapprox(pm1.sigma, sigma_t)
        @test isapprox(pm1.chol, chol_t)

        pm2 = prior(FactorPriorEstimator(; residuals = true), transpose(X), transpose(F);
                    dims = 2)
        pm2_t = CSV.read(joinpath(@__DIR__, "./assets/Factor-Prior-Residuals.csv"),
                         DataFrame)
        X_t = reshape(view(pm2_t, 1:1000, 1), 100, 10)
        mu_t = view(pm2_t, 1001:1010, 1)
        sigma_t = reshape(view(pm2_t, 1011:1110, 1), 10, 10)
        chol_t = reshape(view(pm2_t, 1111:nrow(pm2_t), 1), :, 10)

        @test isapprox(pm2.X, X_t)
        @test isapprox(pm2.mu, mu_t)
        @test isapprox(pm2.sigma, sigma_t)
        @test isapprox(pm2.chol, chol_t)
    end
    @testset "Bayesian Black Litterman Prior" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10) * 0.001
        F = X[:, [3, 8, 5, 10]]
        assets = 1:10
        sets = DataFrame(:Factor => [1, 2, 3, 4])
        vc_1 = LinearConstraintAtom(; group = :Factor, name = 2, coef = 1, cnst = 0.003)
        vc_2 = LinearConstraintAtom(; group = [:Factor, :Factor], name = [4, 1],
                                    coef = [1, -1], cnst = -0.001)
        vc_3 = LinearConstraintAtom(; group = [:Factor, :Factor], name = [2, 3],
                                    coef = [1, -1], cnst = 0.002)
        views = [vc_1, vc_2, vc_3]
        pes = [BayesianBlackLittermanPriorEstimator(;
                                                    pe = FactorPriorEstimator(;
                                                                              pe = EmpiricalPriorEstimator(;
                                                                                                           me = ExcessExpectedReturns(;
                                                                                                                                      rf = 0.001))),
                                                    mp = DefaultMatrixProcessing(),
                                                    views = views, sets = sets, rf = 0.001),
               BayesianBlackLittermanPriorEstimator(;
                                                    pe = FactorPriorEstimator(;
                                                                              pe = EmpiricalPriorEstimator(;
                                                                                                           me = ExcessExpectedReturns(;
                                                                                                                                      rf = 0.001))),
                                                    mp = DefaultMatrixProcessing(),
                                                    views = views, sets = sets, rf = 0.001,
                                                    views_conf = fill(eps(), length(views))),
               BayesianBlackLittermanPriorEstimator(;
                                                    pe = FactorPriorEstimator(;
                                                                              pe = EmpiricalPriorEstimator(;
                                                                                                           me = ExcessExpectedReturns(;
                                                                                                                                      rf = 0.001))),
                                                    mp = DefaultMatrixProcessing(),
                                                    views = views, sets = sets, rf = 0.001,
                                                    views_conf = fill(1 - sqrt(eps()),
                                                                      length(views)))]
        pet = CSV.read(joinpath(@__DIR__, "./assets/Bayesian-Black-Litterman-Prior.csv"),
                       DataFrame)
        for i ∈ eachindex(pes)
            pm = prior(pes[i], transpose(X), transpose(F); dims = 2)
            X_t = reshape(pet[1:1000, i], size(pm.X))
            mu_t = reshape(pet[1001:1010, i], size(pm.mu))
            sigma_t = reshape(pet[1011:end, i], size(pm.sigma))
            res1 = isapprox(pm.mu, mu_t)
            if !res1
                println("Test $i fails on mu.")
                find_tol(pm.X, X; name1 = :X, name2 = :X_t)
            end
            res2 = isapprox(pm.mu, mu_t)
            if !res2
                println("Test $i fails on mu.")
                find_tol(pm.mu, mu_t; name1 = :mu, name2 = :mu_t)
            end
            @test res2
            res3 = isapprox(pm.sigma, sigma_t)
            if !res3
                println("Test $i fails on sigma.")
                find_tol(pm.sigma, sigma_t; name1 = :sigma, name2 = :sigma_t)
            end
            @test res3
        end
    end
    @testset "Factor Black Litterman Prior" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10) * 0.001
        F = X[:, [3, 8, 5, 10]]

        vc_1 = LinearConstraintAtom(; group = :Factor, name = 2, coef = 1, cnst = 0.003)
        vc_2 = LinearConstraintAtom(; group = [:Factor, :Factor], name = [4, 1],
                                    coef = [1, -1], cnst = -0.001)
        vc_3 = LinearConstraintAtom(; group = [:Factor, :Factor], name = [2, 3],
                                    coef = [1, -1], cnst = 0.002)
        views = [vc_1, vc_2, vc_3]
        sets = DataFrame(:Factor => [1, 2, 3, 4])

        pes = [FactorBlackLittermanPriorEstimator(; views = views, sets = sets,
                                                  residuals = false),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  residuals = false),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, residuals = false),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, w = (1:10) / sum(1:10),
                                                  residuals = false),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets,
                                                  residuals = false,
                                                  views_conf = fill(eps(), length(views)),),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  residuals = false,
                                                  views_conf = fill(eps(), length(views)),),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, residuals = false,
                                                  views_conf = fill(eps(), length(views)),),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, w = (1:10) / sum(1:10),
                                                  residuals = false,
                                                  views_conf = fill(eps(), length(views)),),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets,
                                                  residuals = false,
                                                  views_conf = fill(1 - sqrt(eps()),
                                                                    length(views)),),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  residuals = false,
                                                  views_conf = fill(1 - sqrt(eps()),
                                                                    length(views)),),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, residuals = false,
                                                  views_conf = fill(1 - sqrt(eps()),
                                                                    length(views)),),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, w = (1:10) / sum(1:10),
                                                  residuals = false,
                                                  views_conf = fill(1 - sqrt(eps()),
                                                                    length(views)))]

        pm1_t = CSV.read(joinpath(@__DIR__,
                                  "./assets/Factor-Black-Litterman-Prior-No-Residuals.csv"),
                         DataFrame)

        for (i, pe) ∈ enumerate(pes)
            pm = prior(pe, transpose(X), transpose(F); dims = 2)
            X_t = reshape(view(pm1_t[!, i], 1:1000, 1), 100, 10)
            mu_t = view(pm1_t[!, i], 1001:1010, 1)
            sigma_t = reshape(view(pm1_t[!, i], 1011:1110, 1), 10, 10)
            chol_t = reshape(view(pm1_t[!, i], 1111:nrow(pm1_t), 1), :, 10)

            res1 = isapprox(pm.X, X_t)
            if !res1
                println("Test $i no residuals fails on X.")
                find_tol(pm.X, X_t; name1 = :X, name2 = :X_t)
            end
            @test res1
            res2 = isapprox(pm.mu, mu_t)
            if !res2
                println("Test $i no residuals fails on mu.")
                find_tol(pm.mu, mu_t; name1 = :mu, name2 = :mu_t)
            end
            @test res2
            res3 = isapprox(pm.sigma, sigma_t)
            if !res3
                println("Test $i no residuals fails on sigma.")
                find_tol(pm.sigma, sigma_t; name1 = :sigma, name2 = :sigma_t)
            end
            @test res3
            res4 = isapprox(pm.chol, chol_t)
            if !res4
                println("Test $i no residuals fails on chol.")
                find_tol(pm.chol, chol_t; name1 = :chol, name2 = :chol_t)
            end
            @test res4
        end

        pes = [FactorBlackLittermanPriorEstimator(; views = views, sets = sets,
                                                  residuals = true),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  residuals = true),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, residuals = true),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, w = (1:10) / sum(1:10),
                                                  residuals = true),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets,
                                                  residuals = true,
                                                  views_conf = fill(eps(), length(views)),),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  residuals = true,
                                                  views_conf = fill(eps(), length(views)),),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, residuals = true,
                                                  views_conf = fill(eps(), length(views)),),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, w = (1:10) / sum(1:10),
                                                  residuals = true,
                                                  views_conf = fill(eps(), length(views)),),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets,
                                                  residuals = true,
                                                  views_conf = fill(1 - sqrt(eps()),
                                                                    length(views)),),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  residuals = true,
                                                  views_conf = fill(1 - sqrt(eps()),
                                                                    length(views)),),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, residuals = true,
                                                  views_conf = fill(1 - sqrt(eps()),
                                                                    length(views)),),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, w = (1:10) / sum(1:10),
                                                  residuals = true,
                                                  views_conf = fill(1 - sqrt(eps()),
                                                                    length(views)))]

        pm1_t = CSV.read(joinpath(@__DIR__,
                                  "./assets/Factor-Black-Litterman-Prior-Residuals.csv"),
                         DataFrame)

        for (i, pe) ∈ enumerate(pes)
            pm = prior(pe, transpose(X), transpose(F); dims = 2)
            X_t = reshape(view(pm1_t[!, i], 1:1000, 1), 100, 10)
            mu_t = view(pm1_t[!, i], 1001:1010, 1)
            sigma_t = reshape(view(pm1_t[!, i], 1011:1110, 1), 10, 10)
            chol_t = reshape(view(pm1_t[!, i], 1111:nrow(pm1_t), 1), :, 10)

            res1 = isapprox(pm.X, X_t)
            if !res1
                println("Test $i residuals fails on X.")
                find_tol(pm.X, X_t; name1 = :X, name2 = :X_t)
            end
            @test res1
            res2 = isapprox(pm.mu, mu_t)
            if !res2
                println("Test $i residuals fails on mu.")
                find_tol(pm.mu, mu_t; name1 = :mu, name2 = :mu_t)
            end
            @test res2
            res3 = isapprox(pm.sigma, sigma_t)
            if !res3
                println("Test $i residuals fails on sigma.")
                find_tol(pm.sigma, sigma_t; name1 = :sigma, name2 = :sigma_t)
            end
            @test res3
            res4 = isapprox(pm.chol, chol_t)
            if !res4
                println("Test $i residuals fails on chol.")
                find_tol(pm.chol, chol_t; name1 = :chol, name2 = :chol_t)
            end
            @test res4
        end
    end
    @testset "Augmented Black Litterman Prior" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10) * 0.001
        F = X[:, [3, 8, 5, 10]]

        assets = 1:10
        a_sets = DataFrame(; Asset = assets, Clusters = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        vc_1 = LinearConstraintAtom(; group = :Asset, name = 2, coef = 1, cnst = 0.003)
        vc_2 = LinearConstraintAtom(; group = [:Asset, :Asset], name = [3, 8],
                                    coef = [1, -1], cnst = -0.001)
        vc_3 = LinearConstraintAtom(; group = [:Clusters, :Asset], name = [3, 9],
                                    coef = [1, -1], cnst = 0.002)
        vc_4 = LinearConstraintAtom(; group = [:Asset, :Clusters], name = [5, 1],
                                    coef = [1, -1], cnst = 0.007)
        vc_5 = LinearConstraintAtom(; group = :Clusters, name = 2, coef = 1, cnst = 0.001)
        a_views = [vc_1, vc_2, vc_3, vc_4, vc_5]
        P, Q = views_constraints(a_views, a_sets)

        f_sets = DataFrame(:Factor => [1, 2, 3, 4])
        vc_1 = LinearConstraintAtom(; group = :Factor, name = 2, coef = 1, cnst = 0.003)
        vc_2 = LinearConstraintAtom(; group = [:Factor, :Factor], name = [4, 1],
                                    coef = [1, -1], cnst = -0.001)
        vc_3 = LinearConstraintAtom(; group = [:Factor, :Factor], name = [2, 3],
                                    coef = [1, -1], cnst = 0.002)
        f_views = [vc_1, vc_2, vc_3]
        f_P, f_Q = views_constraints(f_views, f_sets)

        pes = [AugmentedBlackLittermanPriorEstimator(; a_views = a_views, a_sets = a_sets,
                                                     f_views = f_views, f_sets = f_sets),
               AugmentedBlackLittermanPriorEstimator(; a_views = a_views, a_sets = a_sets,
                                                     f_views = f_views, f_sets = f_sets,
                                                     rf = 0.001),
               AugmentedBlackLittermanPriorEstimator(; a_views = a_views, a_sets = a_sets,
                                                     f_views = f_views, f_sets = f_sets,
                                                     rf = 0.001, l = 1),
               AugmentedBlackLittermanPriorEstimator(; a_views = a_views, a_sets = a_sets,
                                                     f_views = f_views, f_sets = f_sets,
                                                     rf = 0.001, l = 1,
                                                     w = (1:10) / sum(1:10)),
               AugmentedBlackLittermanPriorEstimator(; a_views = a_views, a_sets = a_sets,
                                                     f_views = f_views, f_sets = f_sets,
                                                     a_views_conf = fill(eps(),
                                                                         length(a_views)),
                                                     f_views_conf = fill(eps(),
                                                                         length(f_views))),
               AugmentedBlackLittermanPriorEstimator(; a_views = a_views, a_sets = a_sets,
                                                     f_views = f_views, f_sets = f_sets,
                                                     rf = 0.001,
                                                     a_views_conf = fill(eps(),
                                                                         length(a_views)),
                                                     f_views_conf = fill(eps(),
                                                                         length(f_views))),
               AugmentedBlackLittermanPriorEstimator(; a_views = a_views, a_sets = a_sets,
                                                     f_views = f_views, f_sets = f_sets,
                                                     rf = 0.001, l = 1,
                                                     a_views_conf = fill(eps(),
                                                                         length(a_views)),
                                                     f_views_conf = fill(eps(),
                                                                         length(f_views))),
               AugmentedBlackLittermanPriorEstimator(; a_views = a_views, a_sets = a_sets,
                                                     f_views = f_views, f_sets = f_sets,
                                                     rf = 0.001, l = 1,
                                                     w = (1:10) / sum(1:10),
                                                     a_views_conf = fill(eps(),
                                                                         length(a_views)),
                                                     f_views_conf = fill(eps(),
                                                                         length(f_views))),
               AugmentedBlackLittermanPriorEstimator(; a_views = a_views, a_sets = a_sets,
                                                     f_views = f_views, f_sets = f_sets,
                                                     a_views_conf = fill(1 - sqrt(eps()),
                                                                         length(a_views)),
                                                     f_views_conf = fill(1 - sqrt(eps()),
                                                                         length(f_views))),
               AugmentedBlackLittermanPriorEstimator(; a_views = a_views, a_sets = a_sets,
                                                     f_views = f_views, f_sets = f_sets,
                                                     rf = 0.001,
                                                     a_views_conf = fill(1 - sqrt(eps()),
                                                                         length(a_views)),
                                                     f_views_conf = fill(1 - sqrt(eps()),
                                                                         length(f_views))),
               AugmentedBlackLittermanPriorEstimator(; a_views = a_views, a_sets = a_sets,
                                                     f_views = f_views, f_sets = f_sets,
                                                     rf = 0.001, l = 1,
                                                     a_views_conf = fill(1 - sqrt(eps()),
                                                                         length(a_views)),
                                                     f_views_conf = fill(1 - sqrt(eps()),
                                                                         length(f_views))),
               AugmentedBlackLittermanPriorEstimator(; a_views = a_views, a_sets = a_sets,
                                                     f_views = f_views, f_sets = f_sets,
                                                     rf = 0.001, l = 1,
                                                     w = (1:10) / sum(1:10),
                                                     a_views_conf = fill(1 - sqrt(eps()),
                                                                         length(a_views)),
                                                     f_views_conf = fill(1 - sqrt(eps()),
                                                                         length(f_views)))]
        pm_t = CSV.read(joinpath(@__DIR__, "./assets/Augmented-Black-Litterman-Prior.csv"),
                        DataFrame)

        for (i, pe) ∈ enumerate(pes)
            pm = prior(pe, transpose(X), transpose(F); dims = 2)
            X_t = reshape(view(pm_t[!, i], 1:1000, 1), 100, 10)
            mu_t = view(pm_t[!, i], 1001:1010, 1)
            sigma_t = reshape(view(pm_t[!, i], 1011:nrow(pm_t), 1), 10, 10)

            res1 = isapprox(pm.X, X_t)
            if !res1
                println("Test $i fails on X.")
                find_tol(pm.X, X_t; name1 = :X, name2 = :X_t)
            end
            @test res1
            res2 = isapprox(pm.mu, mu_t)
            if !res2
                println("Test $i fails on mu.")
                find_tol(pm.mu, mu_t; name1 = :mu, name2 = :mu_t)
            end
            @test res2
            res3 = isapprox(pm.sigma, sigma_t)
            if !res3
                println("Test $i fails on sigma.")
                find_tol(pm.sigma, sigma_t; name1 = :sigma, name2 = :sigma_t)
            end
            @test res3
        end
    end
end
