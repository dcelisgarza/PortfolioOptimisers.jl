@safetestset "Prior tests" begin
    using PortfolioOptimisers, StatsBase, Random, StableRNGs, Test, CovarianceEstimation,
          CSV, DataFrames, LinearAlgebra, Clarabel

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
    @testset "Type tests" begin
        pe1 = BayesianBlackLittermanPriorEstimator(; tau = 1,
                                                   pe = FactorPriorEstimator(;
                                                                             pe = EmpiricalPriorEstimator(;
                                                                                                          me = JamesSteinExpectedReturns(),
                                                                                                          ce = PortfolioOptimisersCovariance(;
                                                                                                                                             ce = Gerber0Covariance()))))
        @test pe1.tau == 1
        @test isa(pe1.me, JamesSteinExpectedReturns)
        @test isa(pe1.ce.ce, Gerber0Covariance)
        pe2 = BayesianBlackLittermanPriorEstimator(;
                                                   pe = FactorBlackLittermanPriorEstimator(;
                                                                                           pe = EmpiricalPriorEstimator(;
                                                                                                                        me = BodnarOkhrinParolyaExpectedReturns(),
                                                                                                                        ce = PortfolioOptimisersCovariance(;
                                                                                                                                                           ce = Gerber2Covariance()))))
        @test isa(pe2.me, BodnarOkhrinParolyaExpectedReturns)
        @test isa(pe2.ce.ce, Gerber2Covariance)
        pe3 = BayesianBlackLittermanPriorEstimator(;
                                                   pe = AugmentedBlackLittermanPriorEstimator(;
                                                                                              a_pe = EmpiricalPriorEstimator(;
                                                                                                                             me = ExcessExpectedReturns(),
                                                                                                                             ce = PortfolioOptimisersCovariance(;
                                                                                                                                                                ce = Gerber1Covariance()))))
        @test isa(pe3.me, ExcessExpectedReturns)
        @test isa(pe3.ce.ce, Gerber1Covariance)
        pe4 = BayesianBlackLittermanPriorEstimator(;
                                                   pe = BlackLittermanPriorEstimator(;
                                                                                     pe = EmpiricalPriorEstimator(;
                                                                                                                  me = EquilibriumExpectedReturns(),
                                                                                                                  ce = PortfolioOptimisersCovariance(;
                                                                                                                                                     ce = SmythBroby0Covariance()))))
        @test isa(pe4.me, EquilibriumExpectedReturns)
        @test isa(pe4.ce.ce, SmythBroby0Covariance)
        pe5 = BayesianBlackLittermanPriorEstimator(;
                                                   pe = BayesianBlackLittermanPriorEstimator())
        @test isa(pe5.me, EquilibriumExpectedReturns)
        @test isa(pe5.ce.ce, FullCovariance)
        pe6 = BlackLittermanPriorEstimator(;
                                           pe = EmpiricalPriorEstimator(;
                                                                        me = BayesSteinExpectedReturns(),
                                                                        ce = PortfolioOptimisersCovariance(;
                                                                                                           ce = SmythBroby0NormalisedCovariance())))
        @test isa(pe6.me, BayesSteinExpectedReturns)
        @test isa(pe6.ce.ce, SmythBroby0NormalisedCovariance)
        pe7 = BlackLittermanPriorEstimator(; tau = 0.5,
                                           pe = FactorPriorEstimator(;
                                                                     pe = EmpiricalPriorEstimator(;
                                                                                                  me = JamesSteinExpectedReturns(),
                                                                                                  ce = PortfolioOptimisersCovariance(;
                                                                                                                                     ce = Gerber0NormalisedCovariance()))))
        @test pe7.tau == 0.5
        @test isa(pe7.me, JamesSteinExpectedReturns)
        @test isa(pe7.ce.ce, Gerber0NormalisedCovariance)
        pe8 = BlackLittermanPriorEstimator(;
                                           pe = FactorBlackLittermanPriorEstimator(;
                                                                                   pe = EmpiricalPriorEstimator(;
                                                                                                                me = BodnarOkhrinParolyaExpectedReturns(),
                                                                                                                ce = PortfolioOptimisersCovariance(;
                                                                                                                                                   ce = Gerber2NormalisedCovariance()))))
        @test isa(pe8.me, BodnarOkhrinParolyaExpectedReturns)
        @test isa(pe8.ce.ce, Gerber2NormalisedCovariance)
        pe9 = BlackLittermanPriorEstimator(;
                                           pe = AugmentedBlackLittermanPriorEstimator(;
                                                                                      a_pe = EmpiricalPriorEstimator(;
                                                                                                                     me = ExcessExpectedReturns(),
                                                                                                                     ce = PortfolioOptimisersCovariance(;
                                                                                                                                                        ce = Gerber1NormalisedCovariance()))))
        @test isa(pe9.me, ExcessExpectedReturns)
        @test isa(pe9.ce.ce, Gerber1NormalisedCovariance)
        pe10 = BlackLittermanPriorEstimator(; pe = BlackLittermanPriorEstimator())
        @test isa(pe10.me, EquilibriumExpectedReturns)
        @test isa(pe10.ce.ce, FullCovariance)
        pe11 = FactorPriorEstimator(;
                                    pe = EmpiricalPriorEstimator(;
                                                                 me = BayesSteinExpectedReturns(),
                                                                 ce = PortfolioOptimisersCovariance(;
                                                                                                    ce = SmythBroby0NormalisedCovariance())))
        @test isa(pe11.me, BayesSteinExpectedReturns)
        @test isa(pe11.ce.ce, SmythBroby0NormalisedCovariance)

        pe12 = FactorPriorEstimator(;
                                    pe = BlackLittermanPriorEstimator(;
                                                                      pe = EmpiricalPriorEstimator(;
                                                                                                   me = JamesSteinExpectedReturns(),
                                                                                                   ce = PortfolioOptimisersCovariance(;
                                                                                                                                      ce = Gerber2NormalisedCovariance()))))
        @test isa(pe12.me, JamesSteinExpectedReturns)
        @test isa(pe12.ce.ce, Gerber2NormalisedCovariance)
        pe13 = FactorBlackLittermanPriorEstimator(; tau = 0.3,
                                                  pe = EmpiricalPriorEstimator(;
                                                                               me = BayesSteinExpectedReturns(),
                                                                               ce = PortfolioOptimisersCovariance(;
                                                                                                                  ce = SmythBroby0NormalisedCovariance())))
        @test pe13.tau == 0.3
        @test isa(pe13.me, BayesSteinExpectedReturns)
        @test isa(pe13.ce.ce, SmythBroby0NormalisedCovariance)
        pe14 = FactorBlackLittermanPriorEstimator(;
                                                  pe = BlackLittermanPriorEstimator(;
                                                                                    pe = EmpiricalPriorEstimator(;
                                                                                                                 me = JamesSteinExpectedReturns(),
                                                                                                                 ce = PortfolioOptimisersCovariance(;
                                                                                                                                                    ce = Gerber2NormalisedCovariance()))))
        @test isa(pe14.me, JamesSteinExpectedReturns)
        @test isa(pe14.ce.ce, Gerber2NormalisedCovariance)
        pe15 = AugmentedBlackLittermanPriorEstimator(; tau = 0.2,
                                                     a_pe = EmpiricalPriorEstimator(;
                                                                                    me = BayesSteinExpectedReturns(),
                                                                                    ce = PortfolioOptimisersCovariance(;
                                                                                                                       ce = SmythBroby0NormalisedCovariance())))
        @test pe15.tau == 0.2
        @test isa(pe15.me, BayesSteinExpectedReturns)
        @test isa(pe15.ce.ce, SmythBroby0NormalisedCovariance)
        pe16 = AugmentedBlackLittermanPriorEstimator(;
                                                     a_pe = BlackLittermanPriorEstimator(;
                                                                                         pe = EmpiricalPriorEstimator(;
                                                                                                                      me = JamesSteinExpectedReturns(),
                                                                                                                      ce = PortfolioOptimisersCovariance(;
                                                                                                                                                         ce = Gerber2NormalisedCovariance()))))
        @test isa(pe16.me, JamesSteinExpectedReturns)
        @test isa(pe16.ce.ce, Gerber2NormalisedCovariance)
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

        vc_1 = BlackLittermanView(;
                                  A = A_LinearConstraint(; group = :Asset, name = 2,
                                                         coef = 1.0), B = 0.003)
        vc_2 = BlackLittermanView(;
                                  A = A_LinearConstraint(; group = [:Asset, :Asset],
                                                         name = [3, 8], coef = [1.0, -1]),
                                  B = -0.001)
        vc_3 = BlackLittermanView(;
                                  A = A_LinearConstraint(; group = [:Clusters, :Asset],
                                                         name = [3, 9], coef = [1.0, -1]),
                                  B = 0.002)
        vc_4 = BlackLittermanView(;
                                  A = A_LinearConstraint(; group = [:Asset, :Clusters],
                                                         name = [5, 1], coef = [1.0, -1]),
                                  B = 0.007)
        vc_5 = BlackLittermanView(;
                                  A = A_LinearConstraint(; group = :Clusters, name = 2,
                                                         coef = 1.0), B = 0.001)
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
            @test size(pm.X) == size(X)
        end

        @test_throws TypeError views_constraints(BlackLittermanView(;
                                                                    A = A_LinearConstraint(;
                                                                                           group = :Foo,
                                                                                           name = 2,
                                                                                           coef = 1),
                                                                    B = 0.003), sets)

        @test_throws ArgumentError views_constraints(BlackLittermanView(;
                                                                        A = A_LinearConstraint(;
                                                                                               group = :Foo,
                                                                                               name = 2,
                                                                                               coef = 1),
                                                                        B = 0.003), sets;
                                                     strict = true)

        @test_throws TypeError views_constraints(BlackLittermanView(;
                                                                    A = A_LinearConstraint(;
                                                                                           group = [:Foo],
                                                                                           name = [2],
                                                                                           coef = [1]),
                                                                    B = 0.003), sets)

        @test_throws ArgumentError views_constraints(BlackLittermanView(;
                                                                        A = A_LinearConstraint(;
                                                                                               group = [:Foo],
                                                                                               name = [2],
                                                                                               coef = [1]),
                                                                        B = 0.003), sets;
                                                     strict = true)

        @test_throws TypeError views_constraints(BlackLittermanView(;
                                                                    A = A_LinearConstraint(;
                                                                                           group = :Asset,
                                                                                           name = 11,
                                                                                           coef = 1),
                                                                    B = 0.003), sets)

        @test_throws ArgumentError views_constraints(BlackLittermanView(;
                                                                        A = A_LinearConstraint(;
                                                                                               group = :Asset,
                                                                                               name = 11,
                                                                                               coef = 1),
                                                                        B = 0.003), sets,
                                                     strict = true)

        @test_throws TypeError views_constraints(BlackLittermanView(;
                                                                    A = A_LinearConstraint(;
                                                                                           group = [:Asset],
                                                                                           name = [11],
                                                                                           coef = [1]),
                                                                    B = 0.003), sets)

        @test_throws ArgumentError views_constraints(BlackLittermanView(;
                                                                        A = A_LinearConstraint(;
                                                                                               group = [:Asset],
                                                                                               name = [11],
                                                                                               coef = [1]),
                                                                        B = 0.003), sets,
                                                     strict = true)
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
        vc_1 = BlackLittermanView(;
                                  A = A_LinearConstraint(; group = :Factor, name = 2,
                                                         coef = 1), B = 0.003)
        vc_2 = BlackLittermanView(;
                                  A = A_LinearConstraint(; group = [:Factor, :Factor],
                                                         name = [4, 1], coef = [1, -1.0]),
                                  B = -0.001)
        vc_3 = BlackLittermanView(;
                                  A = A_LinearConstraint(; group = [:Factor, :Factor],
                                                         name = [2, 3], coef = [1, -1.0]),
                                  B = 0.002)
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
            @test length(pm.f_mu) ==
                  size(pm.f_sigma, 1) ==
                  size(pm.f_sigma, 2) ==
                  size(pm.loadings.M, 2)
            @test length(pm.mu) == size(pm.loadings.M, 1) == length(pm.loadings.b)
        end
    end
    @testset "Factor Black Litterman Prior" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10) * 0.001
        F = X[:, [3, 8, 5, 10]]

        vc_1 = BlackLittermanView(;
                                  A = A_LinearConstraint(; group = :Factor, name = 2,
                                                         coef = 1), B = 0.003)
        vc_2 = BlackLittermanView(;
                                  A = A_LinearConstraint(; group = [:Factor, :Factor],
                                                         name = [4, 1], coef = [1, -1.0]),
                                  B = -0.001)
        vc_3 = BlackLittermanView(;
                                  A = A_LinearConstraint(; group = [:Factor, :Factor],
                                                         name = [2, 3], coef = [1, -1.0]),
                                  B = 0.002)
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
            @test length(pm.f_mu) ==
                  size(pm.f_sigma, 1) ==
                  size(pm.f_sigma, 2) ==
                  size(pm.loadings.M, 2)
            @test length(pm.mu) == size(pm.loadings.M, 1) == length(pm.loadings.b)
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
        vc_1 = BlackLittermanView(;
                                  A = A_LinearConstraint(; group = :Asset, name = 2,
                                                         coef = 1.0), B = 0.003)
        vc_2 = BlackLittermanView(;
                                  A = A_LinearConstraint(; group = [:Asset, :Asset],
                                                         name = [3, 8], coef = [1, -1.0]),
                                  B = -0.001)
        vc_3 = BlackLittermanView(;
                                  A = A_LinearConstraint(; group = [:Clusters, :Asset],
                                                         name = [3, 9], coef = [1, -1.0]),
                                  B = 0.002)
        vc_4 = BlackLittermanView(;
                                  A = A_LinearConstraint(; group = [:Asset, :Clusters],
                                                         name = [5, 1], coef = [1, -1.0]),
                                  B = 0.007)
        vc_5 = BlackLittermanView(;
                                  A = A_LinearConstraint(; group = :Clusters, name = 2,
                                                         coef = 1.0), B = 0.001)
        a_views = [vc_1, vc_2, vc_3, vc_4, vc_5]

        f_sets = DataFrame(:Factor => [1, 2, 3, 4])
        vc_1 = BlackLittermanView(;
                                  A = A_LinearConstraint(; group = :Factor, name = 2,
                                                         coef = 1.0), B = 0.003)
        vc_2 = BlackLittermanView(;
                                  A = A_LinearConstraint(; group = [:Factor, :Factor],
                                                         name = [4, 1], coef = [1, -1.0]),
                                  B = -0.001)
        vc_3 = BlackLittermanView(;
                                  A = A_LinearConstraint(; group = [:Factor, :Factor],
                                                         name = [2, 3], coef = [1, -1.0]),
                                  B = 0.002)
        f_views = [vc_1, vc_2, vc_3]
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
            @test length(pm.f_mu) ==
                  size(pm.f_sigma, 1) ==
                  size(pm.f_sigma, 2) ==
                  size(pm.loadings.M, 2)
            @test length(pm.mu) == size(pm.loadings.M, 1) == length(pm.loadings.b)
        end
    end
    @testset "High Order Prior" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10)
        pe = HighOrderPriorEstimator()
        pm = prior(pe, transpose(X); dims = 2)
        @test isapprox(pm.X, X)
        @test isapprox(pm.mu, vec(mean(SimpleExpectedReturns(), X)))
        @test isapprox(pm.sigma, cov(PortfolioOptimisersCovariance(), X))
        @test isapprox(pm.kt, cokurtosis(FullCokurtosis(), X))
        @test isapprox(pm.skt, cokurtosis(SemiCokurtosis(), X))
        @test all(isapprox.((pm.sk, pm.V), coskewness(FullCoskewness(), X)))
        @test all(isapprox.((pm.ssk, pm.SV), coskewness(SemiCoskewness(), X)))
        pm = prior(pe, transpose(X); dims = 2, kurt = false, skurt = false, skew = false,
                   sskew = false)
        @test isnothing(pm.kt)
        @test isnothing(pm.skt)
        @test isnothing(pm.sk)
        @test isnothing(pm.V)
        @test isnothing(pm.ssk)
        @test isnothing(pm.SV)
    end

    @testset "Entropy Pooling" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10)
        sets = DataFrame(:Assets => 1:10, :Clusters => [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        views = [EntropyPoolingView(;
                                    A = C0_LinearEntropyPoolingConstraint(; group = :Assets,
                                                                          name = 1),
                                    B = ConstantEntropyPoolingConstraint(; coef = 0.1),
                                    comp = EQ()),
                 EntropyPoolingView(;
                                    A = C0_LinearEntropyPoolingConstraint(; group = :Assets,
                                                                          name = 2),
                                    B = C0_LinearEntropyPoolingConstraint(; group = :Assets,
                                                                          name = 2)),
                 EntropyPoolingView(;
                                    A = C1_LinearEntropyPoolingConstraint(; group = :Assets,
                                                                          name = 2),
                                    B = ConstantEntropyPoolingConstraint(; coef = 0.95),
                                    comp = LEQ()),
                 EntropyPoolingView(;
                                    A = C0_LinearEntropyPoolingConstraint(; group = :Assets,
                                                                          name = 3),
                                    B = C0_LinearEntropyPoolingConstraint(; group = :Assets,
                                                                          name = 3)),
                 EntropyPoolingView(;
                                    A = C1_LinearEntropyPoolingConstraint(; group = :Assets,
                                                                          name = 3),
                                    B = C1_LinearEntropyPoolingConstraint(; group = :Assets,
                                                                          name = 3)),
                 EntropyPoolingView(;
                                    A = C2_LinearEntropyPoolingConstraint(; group = :Assets,
                                                                          name = 3,
                                                                          kind = SkewnessEntropyPoolingView()),
                                    B = ConstantEntropyPoolingConstraint(; coef = -0.2),
                                    comp = GEQ()),
                 EntropyPoolingView(;
                                    A = C0_LinearEntropyPoolingConstraint(; group = :Assets,
                                                                          name = 4),
                                    B = C0_LinearEntropyPoolingConstraint(; group = :Assets,
                                                                          name = 4)),
                 EntropyPoolingView(;
                                    A = C1_LinearEntropyPoolingConstraint(; group = :Assets,
                                                                          name = 4),
                                    B = C1_LinearEntropyPoolingConstraint(; group = :Assets,
                                                                          name = 4)),
                 EntropyPoolingView(;
                                    A = C2_LinearEntropyPoolingConstraint(; group = :Assets,
                                                                          name = 4,
                                                                          kind = KurtosisEntropyPoolingView()),
                                    B = ConstantEntropyPoolingConstraint(; coef = 5.1),
                                    comp = LEQ()),
                 EntropyPoolingView(;
                                    A = C4_LinearEntropyPoolingConstraint(;
                                                                          group1 = :Assets,
                                                                          group2 = :Assets,
                                                                          name1 = 1,
                                                                          name2 = 2),
                                    B = C4_LinearEntropyPoolingConstraint(;
                                                                          group1 = :Assets,
                                                                          group2 = :Assets,
                                                                          name1 = 1,
                                                                          name2 = 2,
                                                                          coef = 0.05),
                                    comp = GEQ())]
        pes = [EntropyPoolingPriorEstimator(; views = views, sets = sets),
               EntropyPoolingPriorEstimator(; views = views, sets = sets,
                                            alg = H1_EntropyPooling()),
               EntropyPoolingPriorEstimator(; views = views, sets = sets,
                                            alg = H2_EntropyPooling()),
               EntropyPoolingPriorEstimator(; views = views, sets = sets,
                                            opt = JuMPEntropyPooling(;
                                                                     solvers = Solver(;
                                                                                      solver = Clarabel.Optimizer,
                                                                                      settings = Dict("verbose" => false)))),
               EntropyPoolingPriorEstimator(; views = views, sets = sets,
                                            alg = H1_EntropyPooling(),
                                            opt = JuMPEntropyPooling(;
                                                                     solvers = Solver(;
                                                                                      solver = Clarabel.Optimizer,
                                                                                      settings = Dict("verbose" => false)))),
               EntropyPoolingPriorEstimator(; views = views, sets = sets,
                                            alg = H2_EntropyPooling(),
                                            opt = JuMPEntropyPooling(;
                                                                     solvers = Solver(;
                                                                                      solver = Clarabel.Optimizer,
                                                                                      settings = Dict("verbose" => false))))]

        res = (0.03400712888845852, 0.031413900180612686, 0.0314139001804876,
               0.03400712973024464, 0.03141390069527771, 0.03141389772080837)

        enss = (0.966564614081624, 0.969074389991833, 0.9690743899919543,
                0.9665646132679833, 0.9690743894930843, 0.9690743923755664)

        for (pe, re_t, ens_t) ∈ zip(pes, res, enss)
            pm = prior(pe, transpose(X); dims = 2)
            res = isapprox(pm.mu[1], 0.1)
            if !res
                println("Test Fails on iteration $i mu")
                find_tol(pm.mu[1], 0.1; name1 = mu, name2 = "== 0.1")
            end
            @test res
            res = diag(pm.sigma)[2] <= 0.95
            if !res
                println("Test Fails on iteration $i sigma")
                find_tol(pm.sigma[1], 0.95; name1 = mu, name2 = "<= 0.95")
            end
            @test res
            res = skewness(pm.X[:, 3], pm.w) >= -0.2
            if !res
                println("Test Fails on iteration $i skewness")
                find_tol(skewness(pm.X[:, 3], pm.w), -0.2; name1 = mu, name2 = ">= -0.2")
            end
            @test res
            res = kurtosis(pm.X[:, 4], pm.w) + 3 <= 5.1
            if !res
                println("Test Fails on iteration $i kurtosis")
                find_tol(kurtosis(pm.X[:, 3], pm.w), 5.1; name1 = mu, name2 = "<= 5.1")
            end
            @test res
            res = cov2cor(pm.sigma)[1, 2] >= 0.05
            if !res
                println("Test Fails on iteration $i correlation")
                find_tol(cov2cor(pm.sigma)[1, 2], 0.05; name1 = mu, name2 = ">= 0.05")
            end
            @test res

            re = relative_entropy(pm.w,
                                  range(; start = inv(100), stop = inv(100), length = 100))
            ens = effective_number_scenarios(pm.w,
                                             range(; start = inv(100), stop = inv(100),
                                                   length = 100))
            res = isapprox(re, re_t, 5e-7)
            if !res
                println("Test Fails on iteration $i re")
                find_tol(re, re_t; name1 = :re, name2 = :re_t)
            end
            res = isapprox(ens, ens_t, 5e-7)
            if !res
                println("Test Fails on iteration $i ens")
                find_tol(ens, ens_t; name1 = :ens, name2 = :ens_t)
            end
            @test ens == exp(-re)
        end
    end
end

function find_tol(a1, a2; name1 = :a1, name2 = :a2)
    for rtol ∈
        [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
         5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0, 1.4e0,
         1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
        if isapprox(a1, a2; rtol = rtol)
            println("isapprox($name1, $name2, rtol = $(rtol))")
            break
        end
    end
end
find_tol(0.031413908040281616, 0.03141389772080837)