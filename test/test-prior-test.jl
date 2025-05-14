@safetestset "Prior tests" begin
    using PortfolioOptimisers, StatsBase, Random, StableRNGs, Test, CovarianceEstimation,
          CSV, DataFrames, LinearAlgebra, Clarabel, SparseArrays
    using PortfolioOptimisers: duplication_matrix, elimination_matrix, summation_matrix,
                               prior_view
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
            pr = prior(pes[i], transpose(X); dims = 2)
            mu_t = reshape(pet[1:10, i], size(pr.mu))
            sigma_t = reshape(pet[11:end, i], size(pr.sigma))
            res1 = isapprox(pr.mu, mu_t)
            if !res1
                println("Test $i fails on mu.")
                find_tol(pr.mu, mu_t; name1 = :er, name2 = :er_t)
            end
            @test res1
            res2 = isapprox(pr.sigma, sigma_t)
            if !res2
                println("Test $i fails on sigma.")
                find_tol(pr.sigma, sigma_t; name1 = :er, name2 = :er_t)
            end
            @test res2
            @test pr === prior(pr)
        end
        pr1 = prior(EmpiricalPriorEstimator(), ReturnsResult(; nx = 1:10, X = X))
        pr2 = prior(EmpiricalPriorEstimator(), X)
        @test pr1.X == pr2.X
        @test pr1.mu == pr2.mu
        @test pr1.sigma == pr2.sigma
        i = [10, 5, 9]
        pes[1] == prior_view(pes[1], i)
        pv1 = prior_view(pr1, i)
        pv1.mu == view(pr1.mu, i)
        pv1.sigma == view(pr1.sigma, i, i)
        pv1.X == view(pr1.X, :, i)
    end
    @testset "Factor Prior" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10)
        F = X[:, [3, 8]]

        pr1 = prior(FactorPriorEstimator(; rsd = false), transpose(X), transpose(F);
                    dims = 2)
        pm1_t = CSV.read(joinpath(@__DIR__, "./assets/Factor-Prior-No-Residuals.csv"),
                         DataFrame)
        X_t = reshape(view(pm1_t, 1:1000, 1), 100, 10)
        mu_t = view(pm1_t, 1001:1010, 1)
        sigma_t = reshape(view(pm1_t, 1011:1110, 1), 10, 10)
        chol_t = reshape(view(pm1_t, 1111:nrow(pm1_t), 1), :, 10)
        @test isapprox(pr1.X, X_t)
        @test isapprox(pr1.mu, mu_t)
        @test isapprox(pr1.sigma, sigma_t)
        @test isapprox(pr1.chol, chol_t)
        @test pr1 === prior(pr1)

        pr2 = prior(FactorPriorEstimator(; rsd = true), transpose(X), transpose(F);
                    dims = 2)
        pm2_t = CSV.read(joinpath(@__DIR__, "./assets/Factor-Prior-Residuals.csv"),
                         DataFrame)
        X_t = reshape(view(pm2_t, 1:1000, 1), 100, 10)
        mu_t = view(pm2_t, 1001:1010, 1)
        sigma_t = reshape(view(pm2_t, 1011:1110, 1), 10, 10)
        chol_t = reshape(view(pm2_t, 1111:nrow(pm2_t), 1), :, 10)
        @test isapprox(pr2.X, X_t)
        @test isapprox(pr2.mu, mu_t)
        @test isapprox(pr2.sigma, sigma_t)
        @test isapprox(pr2.chol, chol_t)

        pr1 = prior(FactorPriorEstimator(; re = StepwiseRegression(; alg = Backward())),
                    ReturnsResult(; nx = 1:10, X = X, nf = 1:2, F = F))
        pr2 = prior(FactorPriorEstimator(; re = StepwiseRegression(; alg = Backward())), X,
                    F)
        @test pr1.X == pr2.X
        @test pr1.mu == pr2.mu
        @test pr1.sigma == pr2.sigma
        @test pr1.chol == pr2.chol
        @test pr1.f_mu == pr2.f_mu
        @test pr1.f_sigma == pr2.f_sigma
        @test pr1.loadings.b == pr2.loadings.b
        @test pr1.loadings.M == pr2.loadings.M

        pe1 = FactorPriorEstimator(; rsd = false)
        ew = eweights(1:10, 0.3)
        pe2 = PortfolioOptimisers.factory(pe1, ew)
        @test pe2.pe.ce.ce.ce.w == ew
        @test pe2.pe.ce.ce.me.w == ew
        @test pe2.pe.me.w === ew
        @test pe2.ve.me.w === ew
        @test pe2.ve.w === ew
        @test !pe2.rsd

        i = [10, 5, 9]
        pe1 === prior_view(pe1, i)
        pr1 = prior(pe1, X, F)
        pv1 = prior_view(pr1, i)
        @test pv1.mu == view(pr1.mu, i)
        @test pv1.sigma == view(pr1.sigma, i, i)
        @test pv1.X == view(pr1.X, :, i)
        @test pv1.f_mu == pr1.f_mu
        @test pv1.f_sigma == pr1.f_sigma
        @test pv1.loadings.b == view(pr1.loadings.b, i)
        @test pv1.loadings.M == view(pr1.loadings.M, i, :)
        @test pv1.chol == view(pr1.chol, :, i)
    end
    @testset "High Order Prior" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10)
        pe = HighOrderPriorEstimator()
        pr = prior(pe, transpose(X); dims = 2)
        @test isapprox(pr.X, X)
        @test isapprox(pr.mu, vec(mean(SimpleExpectedReturns(), X)))
        @test isapprox(pr.sigma, cov(PortfolioOptimisersCovariance(), X))
        @test isapprox(pr.kt, cokurtosis(Cokurtosis(; alg = Full()), X))
        @test all(isapprox.((pr.sk, pr.V), coskewness(Coskewness(; alg = Full()), X)))

        pe = HighOrderPriorEstimator(; kte = Cokurtosis(; alg = Semi()),
                                     ske = Coskewness(; alg = Semi()))
        pr = prior(pe, transpose(X); dims = 2)
        @test isapprox(pr.X, X)
        @test isapprox(pr.mu, vec(mean(SimpleExpectedReturns(), X)))
        @test isapprox(pr.sigma, cov(PortfolioOptimisersCovariance(), X))
        @test isapprox(pr.kt, cokurtosis(Cokurtosis(; alg = Semi()), X))
        @test all(isapprox.((pr.sk, pr.V), coskewness(Coskewness(; alg = Semi()), X)))

        pe1 = HighOrderPriorEstimator()
        ew = eweights(1:10, 0.3)
        pe2 = PortfolioOptimisers.factory(pe1, ew)
        @test pe2.pe.ce.ce.ce.w == ew
        @test pe2.pe.ce.ce.me.w == ew
        @test pe2.pe.me.w === ew
        @test pe2.kte.me.w === ew
        @test pe2.ske.me.w === ew

        pe1 = HighOrderPriorEstimator(; kte = nothing, ske = nothing)
        pe2 = PortfolioOptimisers.factory(pe1, ew)
        @test pe2.pe.ce.ce.ce.w == ew
        @test pe2.pe.ce.ce.me.w == ew
        @test pe2.pe.me.w === ew
        @test isnothing(pe2.kte)
        @test isnothing(pe2.ske)

        pr1 = prior(pe, ReturnsResult(; nx = 1:10, X = X))
        @test isapprox(pr.X, pr1.X)
        @test isapprox(pr.mu, pr1.mu)
        @test isapprox(pr.sigma, pr1.sigma)
        @test isapprox(pr.kt, pr1.kt)
        @test isapprox(pr.sk, pr1.sk)
        @test isapprox(pr.V, pr1.V)

        pr = prior(HighOrderPriorEstimator(; kte = nothing, ske = nothing), transpose(X);
                   dims = 2)
        @test isnothing(pr.kt)
        @test isnothing(pr.sk)
        @test isnothing(pr.V)
        @test pr === prior(pr)

        pr = prior(HighOrderPriorEstimator(; kte = nothing), X)
        prv = prior_view(pr, [10, 5, 9])
        @test isnothing(prv.kt)
        @test isnothing(prv.L2)
        @test isnothing(prv.S2)

        @test duplication_matrix(5) ==
              sparse([1, 2, 6, 3, 11, 4, 16, 5, 21, 7, 8, 12, 9, 17, 10, 22, 13, 14, 18, 15,
                      23, 19, 20, 24, 25],
                     [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8, 9, 9, 10, 11, 11, 12, 12,
                      13, 14, 14, 15],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1], 25, 15)
        @test duplication_matrix(5, false) ==
              sparse([2, 6, 3, 11, 4, 16, 5, 21, 8, 12, 9, 17, 10, 22, 14, 18, 15, 23, 20,
                      24], [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 25, 10)
        @test elimination_matrix(5) ==
              sparse([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                     [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 19, 20, 25],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 15, 25)
        @test elimination_matrix(5, false) ==
              sparse([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 3, 4, 5, 8, 9, 10, 14, 15, 20],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 10, 25)
        @test summation_matrix(5) ==
              sparse([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                     [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 19, 20, 25],
                     [1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 1], 15, 25)
        @test summation_matrix(5, false) ==
              sparse([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 3, 4, 5, 8, 9, 10, 14, 15, 20],
                     [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 10, 25)
    end
    @testset "High Order Factor Prior" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 10) * 0.001
        F = X[:, [3, 8]]

        pe1 = FactorPriorEstimator(; re = DimensionReductionRegression(;), rsd = true)
        pr1 = prior(pe1, transpose(X), transpose(F); dims = 2)

        pe2 = HighOrderPriorEstimator(; pe = pe1)
        pr2 = prior(pe2, transpose(X), transpose(F); dims = 2)
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

        i = [10, 5, 9]
        pe2 == prior_view(pe2, i)
        pr1 = prior(pe2, X, F)
        pv1 = prior_view(pr1, i)
        @test pv1.mu == view(pr1.mu, i)
        @test pv1.sigma == view(pr1.sigma, i, i)
        @test pv1.X == view(pr1.X, :, i)
        @test pv1.f_mu == pr1.f_mu
        @test pv1.f_sigma == pr1.f_sigma
        @test pv1.loadings.b == view(pr1.loadings.b, i)
        @test pv1.loadings.M == view(pr1.loadings.M, i, :)
        @test pv1.chol == view(pr1.chol, :, i)
    end
    @testset "Black Litterman Views" begin
        assets = 1:10
        sets = DataFrame(; Asset = assets, Clusters = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        vc_1 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(; group = :Asset,
                                                                     name = 2, coef = 1.0),
                                            B = 0.003)
        vc_2 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(;
                                                                     group = [:Asset,
                                                                              :Asset],
                                                                     name = [3, 8],
                                                                     coef = [1.0, -1]),
                                            B = -0.001)
        vc_3 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(;
                                                                     group = [:Clusters,
                                                                              :Asset],
                                                                     name = [3, 9],
                                                                     coef = [1.0, -1]),
                                            B = 0.002)
        vc_4 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(;
                                                                     group = [:Asset,
                                                                              :Clusters],
                                                                     name = [5, 1],
                                                                     coef = [1.0, -1]),
                                            B = 0.007)
        vc_5 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(; group = :Clusters,
                                                                     name = 2, coef = 1.0),
                                            B = 0.001)
        views = [vc_1, vc_2, vc_3, vc_4, vc_5]
        blvr = black_litterman_views(views, sets)
        @test isapprox(blvr.P,
                       reshape([0.0, 0.0, 0.0, -0.3333333333333333, 0.0, 1.0, 0.0, 0.0,
                                -0.3333333333333333, 0.0, 0.0, 1.0, 0.25, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.3333333333333333, 0.0, 0.0, 0.25, 1.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0,
                                0.3333333333333333, 0.0, -1.0, 0.0, -0.3333333333333333,
                                0.0, 0.0, 0.0, -0.75, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0],
                               :, 10))
        @test isapprox(blvr.Q, [0.003, -0.001, 0.002, 0.007, 0.001])
        @test blvr === black_litterman_views(blvr)
        @test isnothing(black_litterman_views(nothing))
        @test isnothing(black_litterman_views(BlackLittermanViewsEstimator(;
                                                                           A = LinearConstraintSide(;
                                                                                                    group = nothing,
                                                                                                    name = nothing)),
                                              sets))
        vc = BlackLittermanViewsEstimator(;
                                          A = LinearConstraintSide(; group = :Asset,
                                                                   name = -1, coef = 1.0),
                                          B = 0.003)
        @test isnothing(black_litterman_views(vc, sets))
        @test_throws ArgumentError black_litterman_views(vc, sets, strict = true)

        vc = BlackLittermanViewsEstimator(;
                                          A = LinearConstraintSide(; group = [:Asset],
                                                                   name = [-1],
                                                                   coef = [1.0]), B = 0.003)
        @test isnothing(black_litterman_views(vc, sets))
        @test_throws ArgumentError black_litterman_views(vc, sets, strict = true)

        vc = BlackLittermanViewsEstimator(;
                                          A = LinearConstraintSide(; group = :Foo,
                                                                   name = -1, coef = 1.0),
                                          B = 0.003)
        @test isnothing(black_litterman_views(vc, sets))
        @test_throws ArgumentError black_litterman_views(vc, sets, strict = true)

        vc = BlackLittermanViewsEstimator(;
                                          A = LinearConstraintSide(; group = [:Foo],
                                                                   name = [-1],
                                                                   coef = [1.0]), B = 0.003)
        @test isnothing(black_litterman_views(vc, sets))
        @test_throws ArgumentError black_litterman_views(vc, sets, strict = true)
    end
    @testset "Black Litteman type tests" begin
        pe1 = BayesianBlackLittermanPriorEstimator(; tau = 1,
                                                   views = BlackLittermanViewsEstimator(;
                                                                                        A = LinearConstraintSide(;
                                                                                                                 name = nothing,
                                                                                                                 group = nothing)),
                                                   pe = FactorPriorEstimator(;
                                                                             pe = EmpiricalPriorEstimator(;
                                                                                                          me = ShrunkExpectedReturns(;
                                                                                                                                     alg = JamesStein()),
                                                                                                          ce = PortfolioOptimisersCovariance(;
                                                                                                                                             ce = GerberCovariance(;
                                                                                                                                                                   alg = Gerber0())))))
        @test pe1.tau == 1
        @test isa(pe1.me, ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein{<:GrandMean}})
        @test isa(pe1.ce.ce, GerberCovariance{<:Any, <:Any, <:Any, <:Gerber0})
        pe2 = BayesianBlackLittermanPriorEstimator(;
                                                   views = BlackLittermanViewsEstimator(;
                                                                                        A = LinearConstraintSide(;
                                                                                                                 name = nothing,
                                                                                                                 group = nothing)),
                                                   pe = FactorBlackLittermanPriorEstimator(;
                                                                                           views = BlackLittermanViewsEstimator(;
                                                                                                                                A = LinearConstraintSide(;
                                                                                                                                                         name = nothing,
                                                                                                                                                         group = nothing)),
                                                                                           pe = EmpiricalPriorEstimator(;
                                                                                                                        me = ShrunkExpectedReturns(;
                                                                                                                                                   alg = BodnarOkhrinParolya()),
                                                                                                                        ce = PortfolioOptimisersCovariance(;
                                                                                                                                                           ce = GerberCovariance(;
                                                                                                                                                                                 alg = Gerber2())))))
        @test isa(pe2.me,
                  ShrunkExpectedReturns{<:Any, <:Any, <:BodnarOkhrinParolya{<:GrandMean}})
        @test isa(pe2.ce.ce, GerberCovariance{<:Any, <:Any, <:Any, <:Gerber2})
        pe3 = BayesianBlackLittermanPriorEstimator(;
                                                   views = BlackLittermanViewsEstimator(;
                                                                                        A = LinearConstraintSide(;
                                                                                                                 name = nothing,
                                                                                                                 group = nothing)),
                                                   pe = AugmentedBlackLittermanPriorEstimator(;
                                                                                              a_views = BlackLittermanViewsEstimator(;
                                                                                                                                     A = LinearConstraintSide(;
                                                                                                                                                              name = nothing,
                                                                                                                                                              group = nothing)),
                                                                                              f_views = BlackLittermanViewsEstimator(;
                                                                                                                                     A = LinearConstraintSide(;
                                                                                                                                                              name = nothing,
                                                                                                                                                              group = nothing)),
                                                                                              a_pe = EmpiricalPriorEstimator(;
                                                                                                                             me = ExcessExpectedReturns(),
                                                                                                                             ce = PortfolioOptimisersCovariance(;
                                                                                                                                                                ce = GerberCovariance(;
                                                                                                                                                                                      alg = Gerber1())))))
        @test isa(pe3.me, ExcessExpectedReturns{<:SimpleExpectedReturns{Nothing}, <:Any})
        @test isa(pe3.ce.ce, GerberCovariance{<:Any, <:Any, <:Any, <:Gerber1})
        pe4 = BayesianBlackLittermanPriorEstimator(;
                                                   views = BlackLittermanViewsEstimator(;
                                                                                        A = LinearConstraintSide(;
                                                                                                                 name = nothing,
                                                                                                                 group = nothing)),
                                                   pe = BlackLittermanPriorEstimator(;
                                                                                     views = BlackLittermanViewsEstimator(;
                                                                                                                          A = LinearConstraintSide(;
                                                                                                                                                   name = nothing,
                                                                                                                                                   group = nothing)),
                                                                                     pe = EmpiricalPriorEstimator(;
                                                                                                                  me = EquilibriumExpectedReturns(),
                                                                                                                  ce = PortfolioOptimisersCovariance(;
                                                                                                                                                     ce = SmythBrobyCovariance(;
                                                                                                                                                                               alg = SmythBroby0())))))
        @test isa(pe4.me, EquilibriumExpectedReturns{<:Any, <:Any, <:Any})
        @test isa(pe4.ce.ce,
                  SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:SmythBroby0})
        pe5 = BayesianBlackLittermanPriorEstimator(;
                                                   views = BlackLittermanViewsEstimator(;
                                                                                        A = LinearConstraintSide(;
                                                                                                                 name = nothing,
                                                                                                                 group = nothing)),
                                                   pe = BayesianBlackLittermanPriorEstimator(;
                                                                                             views = BlackLittermanViewsEstimator(;
                                                                                                                                  A = LinearConstraintSide(;
                                                                                                                                                           name = nothing,
                                                                                                                                                           group = nothing))))
        @test isa(pe5.me, EquilibriumExpectedReturns)
        @test isa(pe5.ce.ce, Covariance{<:Any, <:Any, <:Full})
        pe6 = BlackLittermanPriorEstimator(;
                                           views = BlackLittermanViewsEstimator(;
                                                                                A = LinearConstraintSide(;
                                                                                                         name = nothing,
                                                                                                         group = nothing)),
                                           pe = EmpiricalPriorEstimator(;
                                                                        me = ShrunkExpectedReturns(;
                                                                                                   alg = BayesStein()),
                                                                        ce = PortfolioOptimisersCovariance(;
                                                                                                           ce = SmythBrobyCovariance(;
                                                                                                                                     alg = NormalisedSmythBroby0()))))
        @test isa(pe6.me, ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein{<:GrandMean}})
        @test isa(pe6.ce.ce,
                  SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:NormalisedSmythBroby0})
        pe7 = BlackLittermanPriorEstimator(; tau = 0.5,
                                           views = BlackLittermanViewsEstimator(;
                                                                                A = LinearConstraintSide(;
                                                                                                         name = nothing,
                                                                                                         group = nothing)),
                                           pe = FactorPriorEstimator(;
                                                                     pe = EmpiricalPriorEstimator(;
                                                                                                  me = ShrunkExpectedReturns(;
                                                                                                                             alg = JamesStein()),
                                                                                                  ce = PortfolioOptimisersCovariance(;
                                                                                                                                     ce = GerberCovariance(;
                                                                                                                                                           alg = NormalisedGerber0())))))
        @test pe7.tau == 0.5
        @test isa(pe7.me, ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein{<:GrandMean}})
        @test isa(pe7.ce.ce, GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber0})
        pe8 = BlackLittermanPriorEstimator(;
                                           views = BlackLittermanViewsEstimator(;
                                                                                A = LinearConstraintSide(;
                                                                                                         name = nothing,
                                                                                                         group = nothing)),
                                           pe = FactorBlackLittermanPriorEstimator(;
                                                                                   views = BlackLittermanViewsEstimator(;
                                                                                                                        A = LinearConstraintSide(;
                                                                                                                                                 name = nothing,
                                                                                                                                                 group = nothing)),
                                                                                   pe = EmpiricalPriorEstimator(;
                                                                                                                me = ShrunkExpectedReturns(;
                                                                                                                                           alg = BodnarOkhrinParolya()),
                                                                                                                ce = PortfolioOptimisersCovariance(;
                                                                                                                                                   ce = GerberCovariance(;
                                                                                                                                                                         alg = NormalisedGerber2())))))
        @test isa(pe8.me,
                  ShrunkExpectedReturns{<:Any, <:Any, <:BodnarOkhrinParolya{<:GrandMean}})
        @test isa(pe8.ce.ce, GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber2})
        pe9 = BlackLittermanPriorEstimator(;
                                           views = BlackLittermanViewsEstimator(;
                                                                                A = LinearConstraintSide(;
                                                                                                         name = nothing,
                                                                                                         group = nothing)),
                                           pe = AugmentedBlackLittermanPriorEstimator(;
                                                                                      a_views = BlackLittermanViewsEstimator(;
                                                                                                                             A = LinearConstraintSide(;
                                                                                                                                                      name = nothing,
                                                                                                                                                      group = nothing)),
                                                                                      f_views = BlackLittermanViewsEstimator(;
                                                                                                                             A = LinearConstraintSide(;
                                                                                                                                                      name = nothing,
                                                                                                                                                      group = nothing)),
                                                                                      a_pe = EmpiricalPriorEstimator(;
                                                                                                                     me = ExcessExpectedReturns(),
                                                                                                                     ce = PortfolioOptimisersCovariance(;
                                                                                                                                                        ce = GerberCovariance(;
                                                                                                                                                                              alg = NormalisedGerber1())))))
        @test isa(pe9.me, ExcessExpectedReturns)
        @test isa(pe9.ce.ce, GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber1})
        pe10 = BlackLittermanPriorEstimator(;
                                            views = BlackLittermanViewsEstimator(;
                                                                                 A = LinearConstraintSide(;
                                                                                                          name = nothing,
                                                                                                          group = nothing)),
                                            pe = BlackLittermanPriorEstimator(;
                                                                              views = BlackLittermanViewsEstimator(;
                                                                                                                   A = LinearConstraintSide(;
                                                                                                                                            name = nothing,
                                                                                                                                            group = nothing))))
        @test isa(pe10.me, EquilibriumExpectedReturns)
        @test isa(pe5.ce.ce, Covariance{<:Any, <:Any, <:Full})
        pe11 = FactorPriorEstimator(;
                                    pe = EmpiricalPriorEstimator(;
                                                                 me = ShrunkExpectedReturns(;
                                                                                            alg = BayesStein()),
                                                                 ce = PortfolioOptimisersCovariance(;
                                                                                                    ce = SmythBrobyCovariance(;
                                                                                                                              alg = NormalisedSmythBroby0()))))
        @test isa(pe11.me, ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein{<:GrandMean}})
        @test isa(pe11.ce.ce,
                  SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:NormalisedSmythBroby0})
        pe12 = FactorPriorEstimator(;
                                    pe = BlackLittermanPriorEstimator(;
                                                                      views = BlackLittermanViewsEstimator(;
                                                                                                           A = LinearConstraintSide(;
                                                                                                                                    name = nothing,
                                                                                                                                    group = nothing)),
                                                                      pe = EmpiricalPriorEstimator(;
                                                                                                   me = ShrunkExpectedReturns(;
                                                                                                                              alg = JamesStein()),
                                                                                                   ce = PortfolioOptimisersCovariance(;
                                                                                                                                      ce = GerberCovariance(;
                                                                                                                                                            alg = NormalisedGerber2())))))
        @test isa(pe12.me, ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein{<:GrandMean}})
        @test isa(pe12.ce.ce, GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber2})
        pe13 = FactorBlackLittermanPriorEstimator(; tau = 0.3,
                                                  views = BlackLittermanViewsEstimator(;
                                                                                       A = LinearConstraintSide(;
                                                                                                                name = nothing,
                                                                                                                group = nothing)),
                                                  pe = EmpiricalPriorEstimator(;
                                                                               me = ShrunkExpectedReturns(;
                                                                                                          alg = BayesStein()),
                                                                               ce = PortfolioOptimisersCovariance(;
                                                                                                                  ce = SmythBrobyCovariance(;
                                                                                                                                            alg = NormalisedSmythBroby0()))))
        @test pe13.tau == 0.3
        @test isa(pe13.me, ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein{<:GrandMean}})
        @test isa(pe13.ce.ce,
                  SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:NormalisedSmythBroby0})
        pe14 = FactorBlackLittermanPriorEstimator(;
                                                  views = BlackLittermanViewsEstimator(;
                                                                                       A = LinearConstraintSide(;
                                                                                                                name = nothing,
                                                                                                                group = nothing)),
                                                  pe = BlackLittermanPriorEstimator(;
                                                                                    views = BlackLittermanViewsEstimator(;
                                                                                                                         A = LinearConstraintSide(;
                                                                                                                                                  name = nothing,
                                                                                                                                                  group = nothing)),
                                                                                    pe = EmpiricalPriorEstimator(;
                                                                                                                 me = ShrunkExpectedReturns(;
                                                                                                                                            alg = JamesStein()),
                                                                                                                 ce = PortfolioOptimisersCovariance(;
                                                                                                                                                    ce = GerberCovariance(;
                                                                                                                                                                          alg = NormalisedGerber2())))))
        @test isa(pe14.me, ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein{<:GrandMean}})
        @test isa(pe14.ce.ce, GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber2})
        pe15 = AugmentedBlackLittermanPriorEstimator(; tau = 0.2,
                                                     a_views = BlackLittermanViewsEstimator(;
                                                                                            A = LinearConstraintSide(;
                                                                                                                     name = nothing,
                                                                                                                     group = nothing)),
                                                     f_views = BlackLittermanViewsEstimator(;
                                                                                            A = LinearConstraintSide(;
                                                                                                                     name = nothing,
                                                                                                                     group = nothing)),
                                                     a_pe = EmpiricalPriorEstimator(;
                                                                                    me = ShrunkExpectedReturns(;
                                                                                                               alg = BayesStein()),
                                                                                    ce = PortfolioOptimisersCovariance(;
                                                                                                                       ce = SmythBrobyCovariance(;
                                                                                                                                                 alg = NormalisedSmythBroby0()))))
        @test pe15.tau == 0.2
        @test isa(pe15.me, ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein{<:GrandMean}})
        @test isa(pe15.ce.ce,
                  SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:NormalisedSmythBroby0})
        @test isa(pe15.f_me, SimpleExpectedReturns)
        @test isa(pe15.f_ce, PortfolioOptimisersCovariance)
        pe16 = AugmentedBlackLittermanPriorEstimator(;
                                                     a_views = BlackLittermanViewsEstimator(;
                                                                                            A = LinearConstraintSide(;
                                                                                                                     name = nothing,
                                                                                                                     group = nothing)),
                                                     f_views = BlackLittermanViewsEstimator(;
                                                                                            A = LinearConstraintSide(;
                                                                                                                     name = nothing,
                                                                                                                     group = nothing)),
                                                     a_pe = BlackLittermanPriorEstimator(;
                                                                                         views = BlackLittermanViewsEstimator(;
                                                                                                                              A = LinearConstraintSide(;
                                                                                                                                                       name = nothing,
                                                                                                                                                       group = nothing)),
                                                                                         pe = EmpiricalPriorEstimator(;
                                                                                                                      me = ShrunkExpectedReturns(;
                                                                                                                                                 alg = JamesStein()),
                                                                                                                      ce = PortfolioOptimisersCovariance(;
                                                                                                                                                         ce = GerberCovariance(;
                                                                                                                                                                               alg = NormalisedGerber2())))))
        @test isa(pe16.me, ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein{<:GrandMean}})
        @test isa(pe16.ce.ce, GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber2})
    end
    @testset "Black Litterman Prior" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 10) * 0.001

        assets = 1:10
        sets = DataFrame(; Asset = assets, Clusters = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])

        vc_1 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(; group = :Asset,
                                                                     name = 2, coef = 1.0),
                                            B = 0.003)
        vc_2 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(;
                                                                     group = [:Asset,
                                                                              :Asset],
                                                                     name = [3, 8],
                                                                     coef = [1.0, -1]),
                                            B = -0.001)
        vc_3 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(;
                                                                     group = [:Clusters,
                                                                              :Asset],
                                                                     name = [3, 9],
                                                                     coef = [1.0, -1]),
                                            B = 0.002)
        vc_4 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(;
                                                                     group = [:Asset,
                                                                              :Clusters],
                                                                     name = [5, 1],
                                                                     coef = [1.0, -1]),
                                            B = 0.007)
        vc_5 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(; group = :Clusters,
                                                                     name = 2, coef = 1.0),
                                            B = 0.001)
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
            pr = prior(pes[i], transpose(X); dims = 2)
            mu_t = reshape(pet[1:10, i], size(pr.mu))
            sigma_t = reshape(pet[11:end, i], size(pr.sigma))
            res1 = isapprox(pr.mu, mu_t)
            if !res1
                println("Test $i fails on mu.")
                find_tol(pr.mu, mu_t; name1 = :er, name2 = :er_t)
            end
            @test res1
            res2 = isapprox(pr.sigma, sigma_t)
            if !res2
                println("Test $i fails on sigma.")
                find_tol(pr.sigma, sigma_t; name1 = :er, name2 = :er_t)
            end
            @test res2
            @test size(pr.X) == size(X)
            @test pr === prior(pr)
        end

        pe1 = pes[1]
        ew = eweights(1:1000, 0.3)
        pe2 = PortfolioOptimisers.factory(pe1, ew)
        @test pe2.pe.ce.ce.ce.w == ew
        @test pe2.pe.ce.ce.me.w == ew
        @test isnothing(pe2.pe.me.w)

        pr1 = prior(pes[1], ReturnsResult(; nx = 1:10, X = X))
        pr2 = prior(pes[1], X)
        @test pr1.X == pr2.X
        @test pr1.mu == pr2.mu
        @test pr1.sigma == pr2.sigma
        @test_throws ArgumentError black_litterman_views(BlackLittermanViewsEstimator(;
                                                                                      A = LinearConstraintSide(;
                                                                                                               group = :Foo,
                                                                                                               name = 2,
                                                                                                               coef = 1),
                                                                                      B = 0.003),
                                                         sets; strict = true)
        @test_throws ArgumentError black_litterman_views(BlackLittermanViewsEstimator(;
                                                                                      A = LinearConstraintSide(;
                                                                                                               group = [:Foo],
                                                                                                               name = [2],
                                                                                                               coef = [1]),
                                                                                      B = 0.003),
                                                         sets; strict = true)
        @test_throws ArgumentError black_litterman_views(BlackLittermanViewsEstimator(;
                                                                                      A = LinearConstraintSide(;
                                                                                                               group = :Asset,
                                                                                                               name = 11,
                                                                                                               coef = 1),
                                                                                      B = 0.003),
                                                         sets, strict = true)
        @test_throws ArgumentError black_litterman_views(BlackLittermanViewsEstimator(;
                                                                                      A = LinearConstraintSide(;
                                                                                                               group = [:Asset],
                                                                                                               name = [11],
                                                                                                               coef = [1]),
                                                                                      B = 0.003),
                                                         sets, strict = true)
    end
    @testset "Black Litterman Factor Prior" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 10) * 0.001
        F = X[:, [3, 8]]

        assets = 1:10
        sets = DataFrame(; Asset = assets, Clusters = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])

        vc_1 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(; group = :Asset,
                                                                     name = 2, coef = 1.0),
                                            B = 0.003)
        vc_2 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(;
                                                                     group = [:Asset,
                                                                              :Asset],
                                                                     name = [3, 8],
                                                                     coef = [1.0, -1]),
                                            B = -0.001)
        vc_3 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(;
                                                                     group = [:Clusters,
                                                                              :Asset],
                                                                     name = [3, 9],
                                                                     coef = [1.0, -1]),
                                            B = 0.002)
        vc_4 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(;
                                                                     group = [:Asset,
                                                                              :Clusters],
                                                                     name = [5, 1],
                                                                     coef = [1.0, -1]),
                                            B = 0.007)
        vc_5 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(; group = :Clusters,
                                                                     name = 2, coef = 1.0),
                                            B = 0.001)
        views = [vc_1, vc_2, vc_3, vc_4, vc_5]
        pe1 = FactorPriorEstimator(; re = StepwiseRegression(; alg = Backward()),
                                   rsd = true)
        pr1 = prior(pe1, transpose(X), transpose(F); dims = 2)

        pe2 = BlackLittermanPriorEstimator(; pe = pe1, views = views, sets = sets)
        pr2 = prior(pe2, transpose(X), transpose(F); dims = 2)
        df = CSV.read(joinpath(@__DIR__, "./assets/Black-Litterman-Factor-Prior.csv"),
                      DataFrame)
        @test pr1.X == pr2.X
        @test isapprox(pr2.mu, df[1:10, "1"])
        @test isapprox(vec(pr2.sigma), df[11:end, "1"])
    end
    @testset "Bayesian Black Litterman Prior" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10) * 0.001
        F = X[:, [3, 8, 5, 10]]
        assets = 1:10
        sets = DataFrame(:Factor => [1, 2, 3, 4])
        vc_1 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(; group = :Factor,
                                                                     name = 2, coef = 1),
                                            B = 0.003)
        vc_2 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(;
                                                                     group = [:Factor,
                                                                              :Factor],
                                                                     name = [4, 1],
                                                                     coef = [1, -1.0]),
                                            B = -0.001)
        vc_3 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(;
                                                                     group = [:Factor,
                                                                              :Factor],
                                                                     name = [2, 3],
                                                                     coef = [1, -1.0]),
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
            pr = prior(pes[i], transpose(X), transpose(F); dims = 2)
            X_t = reshape(pet[1:1000, i], size(pr.X))
            mu_t = reshape(pet[1001:1010, i], size(pr.mu))
            sigma_t = reshape(pet[1011:end, i], size(pr.sigma))
            res1 = isapprox(pr.mu, mu_t)
            if !res1
                println("Test $i fails on mu.")
                find_tol(pr.X, X; name1 = :X, name2 = :X_t)
            end
            res2 = isapprox(pr.mu, mu_t)
            if !res2
                println("Test $i fails on mu.")
                find_tol(pr.mu, mu_t; name1 = :mu, name2 = :mu_t)
            end
            @test res2
            res3 = isapprox(pr.sigma, sigma_t)
            if !res3
                println("Test $i fails on sigma.")
                find_tol(pr.sigma, sigma_t; name1 = :sigma, name2 = :sigma_t)
            end
            @test res3
            @test length(pr.f_mu) ==
                  size(pr.f_sigma, 1) ==
                  size(pr.f_sigma, 2) ==
                  size(pr.loadings.M, 2)
            @test length(pr.mu) == size(pr.loadings.M, 1) == length(pr.loadings.b)
            @test pr === prior(pr)
        end
        pe1 = pes[1]
        ew = eweights(1:1000, 0.3)
        pe2 = PortfolioOptimisers.factory(pe1, ew)
        @test pe2.pe.ce.ce.ce.w == ew
        @test pe2.pe.ce.ce.me.w == ew
        @test pe2.pe.me.me.w === ew

        pr1 = prior(pes[1], ReturnsResult(; nx = 1:10, X = X, nf = 1:4, F = F))
        pr2 = prior(pes[1], X, F)
        @test pr1.X == pr2.X
        @test pr1.mu == pr2.mu
        @test pr1.sigma == pr2.sigma
        @test pr1.f_mu == pr2.f_mu
        @test pr1.f_sigma == pr2.f_sigma
        @test pr1.loadings.b == pr2.loadings.b
        @test pr1.loadings.M == pr2.loadings.M

        i = [10, 5, 9]
        pes[1] === prior_view(pes[1], i)
        pv1 = prior_view(pr1, i)
        @test pv1.mu == view(pr1.mu, i)
        @test pv1.sigma == view(pr1.sigma, i, i)
        @test pv1.X == view(pr1.X, :, i)
        @test pv1.f_mu == pr1.f_mu
        @test pv1.f_sigma == pr1.f_sigma
        @test pv1.loadings.b == view(pr1.loadings.b, i)
        @test pv1.loadings.M == view(pr1.loadings.M, i, :)
    end
    @testset "Factor Black Litterman Prior" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10) * 0.001
        F = X[:, [3, 8, 5, 10]]

        vc_1 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(; group = :Factor,
                                                                     name = 2, coef = 1),
                                            B = 0.003)
        vc_2 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(;
                                                                     group = [:Factor,
                                                                              :Factor],
                                                                     name = [4, 1],
                                                                     coef = [1, -1.0]),
                                            B = -0.001)
        vc_3 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(;
                                                                     group = [:Factor,
                                                                              :Factor],
                                                                     name = [2, 3],
                                                                     coef = [1, -1.0]),
                                            B = 0.002)
        views = [vc_1, vc_2, vc_3]
        sets = DataFrame(:Factor => [1, 2, 3, 4])

        pes = [FactorBlackLittermanPriorEstimator(; views = views, sets = sets,
                                                  rsd = false),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  rsd = false),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, rsd = false),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, w = (1:10) / sum(1:10),
                                                  rsd = false),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rsd = false,
                                                  views_conf = fill(eps(), length(views))),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  rsd = false,
                                                  views_conf = fill(eps(), length(views))),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, rsd = false,
                                                  views_conf = fill(eps(), length(views))),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, w = (1:10) / sum(1:10),
                                                  rsd = false,
                                                  views_conf = fill(eps(), length(views))),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rsd = false,
                                                  views_conf = fill(1 - sqrt(eps()),
                                                                    length(views))),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  rsd = false,
                                                  views_conf = fill(1 - sqrt(eps()),
                                                                    length(views))),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, rsd = false,
                                                  views_conf = fill(1 - sqrt(eps()),
                                                                    length(views))),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, w = (1:10) / sum(1:10),
                                                  rsd = false,
                                                  views_conf = fill(1 - sqrt(eps()),
                                                                    length(views)))]

        pm1_t = CSV.read(joinpath(@__DIR__,
                                  "./assets/Factor-Black-Litterman-Prior-No-Residuals.csv"),
                         DataFrame)
        for (i, pe) ∈ enumerate(pes)
            pr = prior(pe, transpose(X), transpose(F); dims = 2)
            X_t = reshape(view(pm1_t[!, i], 1:1000, 1), 100, 10)
            mu_t = view(pm1_t[!, i], 1001:1010, 1)
            sigma_t = reshape(view(pm1_t[!, i], 1011:1110, 1), 10, 10)
            chol_t = reshape(view(pm1_t[!, i], 1111:nrow(pm1_t), 1), :, 10)

            res1 = isapprox(pr.X, X_t)
            if !res1
                println("Test $i no rsd fails on X.")
                find_tol(pr.X, X_t; name1 = :X, name2 = :X_t)
            end
            @test res1
            res2 = isapprox(pr.mu, mu_t)
            if !res2
                println("Test $i no rsd fails on mu.")
                find_tol(pr.mu, mu_t; name1 = :mu, name2 = :mu_t)
            end
            @test res2
            res3 = isapprox(pr.sigma, sigma_t)
            if !res3
                println("Test $i no rsd fails on sigma.")
                find_tol(pr.sigma, sigma_t; name1 = :sigma, name2 = :sigma_t)
            end
            @test res3
            res4 = isapprox(pr.chol, chol_t)
            if !res4
                println("Test $i no rsd fails on chol.")
                find_tol(pr.chol, chol_t; name1 = :chol, name2 = :chol_t)
            end
            @test res4
            @test length(pr.f_mu) ==
                  size(pr.f_sigma, 1) ==
                  size(pr.f_sigma, 2) ==
                  size(pr.loadings.M, 2)
            @test length(pr.mu) == size(pr.loadings.M, 1) == length(pr.loadings.b)
            @test pr === prior(pr)
        end
        pe1 = pes[1]
        ew = eweights(1:1000, 0.3)
        pe2 = PortfolioOptimisers.factory(pe1, ew)
        @test pe2.pe.ce.ce.ce.w == ew
        @test pe2.pe.ce.ce.me.w == ew
        @test pe2.pe.me.w === ew
        @test pe2.ve.me.w === ew
        @test pe2.ve.w === ew
        @test !pe2.rsd

        pr1 = prior(pes[1], ReturnsResult(; nx = 1:10, X = X, nf = 1:4, F = F))
        pr2 = prior(pes[1], X, F)
        @test pr1.X == pr2.X
        @test pr1.mu == pr2.mu
        @test pr1.sigma == pr2.sigma
        @test pr1.chol == pr2.chol
        @test pr1.f_mu == pr2.f_mu
        @test pr1.f_sigma == pr2.f_sigma
        @test pr1.loadings.b == pr2.loadings.b
        @test pr1.loadings.M == pr2.loadings.M

        pes = [FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rsd = true),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  rsd = true),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, rsd = true),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, w = (1:10) / sum(1:10),
                                                  rsd = true),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rsd = true,
                                                  views_conf = fill(eps(), length(views))),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  rsd = true,
                                                  views_conf = fill(eps(), length(views))),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, rsd = true,
                                                  views_conf = fill(eps(), length(views))),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, w = (1:10) / sum(1:10), rsd = true,
                                                  views_conf = fill(eps(), length(views))),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rsd = true,
                                                  views_conf = fill(1 - sqrt(eps()),
                                                                    length(views))),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  rsd = true,
                                                  views_conf = fill(1 - sqrt(eps()),
                                                                    length(views))),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, rsd = true,
                                                  views_conf = fill(1 - sqrt(eps()),
                                                                    length(views))),
               FactorBlackLittermanPriorEstimator(; views = views, sets = sets, rf = 0.001,
                                                  l = 1, w = (1:10) / sum(1:10), rsd = true,
                                                  views_conf = fill(1 - sqrt(eps()),
                                                                    length(views)))]

        pm1_t = CSV.read(joinpath(@__DIR__,
                                  "./assets/Factor-Black-Litterman-Prior-Residuals.csv"),
                         DataFrame)

        for (i, pe) ∈ enumerate(pes)
            pr = prior(pe, transpose(X), transpose(F); dims = 2)
            X_t = reshape(view(pm1_t[!, i], 1:1000, 1), 100, 10)
            mu_t = view(pm1_t[!, i], 1001:1010, 1)
            sigma_t = reshape(view(pm1_t[!, i], 1011:1110, 1), 10, 10)
            chol_t = reshape(view(pm1_t[!, i], 1111:nrow(pm1_t), 1), :, 10)

            res1 = isapprox(pr.X, X_t)
            if !res1
                println("Test $i rsd fails on X.")
                find_tol(pr.X, X_t; name1 = :X, name2 = :X_t)
            end
            @test res1
            res2 = isapprox(pr.mu, mu_t)
            if !res2
                println("Test $i rsd fails on mu.")
                find_tol(pr.mu, mu_t; name1 = :mu, name2 = :mu_t)
            end
            @test res2
            res3 = isapprox(pr.sigma, sigma_t)
            if !res3
                println("Test $i rsd fails on sigma.")
                find_tol(pr.sigma, sigma_t; name1 = :sigma, name2 = :sigma_t)
            end
            @test res3
            res4 = isapprox(pr.chol, chol_t)
            if !res4
                println("Test $i rsd fails on chol.")
                find_tol(pr.chol, chol_t; name1 = :chol, name2 = :chol_t)
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
        vc_1 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(; group = :Asset,
                                                                     name = 2, coef = 1.0),
                                            B = 0.003)
        vc_2 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(;
                                                                     group = [:Asset,
                                                                              :Asset],
                                                                     name = [3, 8],
                                                                     coef = [1, -1.0]),
                                            B = -0.001)
        vc_3 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(;
                                                                     group = [:Clusters,
                                                                              :Asset],
                                                                     name = [3, 9],
                                                                     coef = [1, -1.0]),
                                            B = 0.002)
        vc_4 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(;
                                                                     group = [:Asset,
                                                                              :Clusters],
                                                                     name = [5, 1],
                                                                     coef = [1, -1.0]),
                                            B = 0.007)
        vc_5 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(; group = :Clusters,
                                                                     name = 2, coef = 1.0),
                                            B = 0.001)
        a_views = [vc_1, vc_2, vc_3, vc_4, vc_5]

        f_sets = DataFrame(:Factor => [1, 2, 3, 4])
        vc_1 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(; group = :Factor,
                                                                     name = 2, coef = 1.0),
                                            B = 0.003)
        vc_2 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(;
                                                                     group = [:Factor,
                                                                              :Factor],
                                                                     name = [4, 1],
                                                                     coef = [1, -1.0]),
                                            B = -0.001)
        vc_3 = BlackLittermanViewsEstimator(;
                                            A = LinearConstraintSide(;
                                                                     group = [:Factor,
                                                                              :Factor],
                                                                     name = [2, 3],
                                                                     coef = [1, -1.0]),
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
            pr = prior(pe, transpose(X), transpose(F); dims = 2)
            X_t = reshape(view(pm_t[!, i], 1:1000, 1), 100, 10)
            mu_t = view(pm_t[!, i], 1001:1010, 1)
            sigma_t = reshape(view(pm_t[!, i], 1011:nrow(pm_t), 1), 10, 10)

            res1 = isapprox(pr.X, X_t)
            if !res1
                println("Test $i fails on X.")
                find_tol(pr.X, X_t; name1 = :X, name2 = :X_t)
            end
            @test res1
            res2 = isapprox(pr.mu, mu_t)
            if !res2
                println("Test $i fails on mu.")
                find_tol(pr.mu, mu_t; name1 = :mu, name2 = :mu_t)
            end
            @test res2
            res3 = isapprox(pr.sigma, sigma_t)
            if !res3
                println("Test $i fails on sigma.")
                find_tol(pr.sigma, sigma_t; name1 = :sigma, name2 = :sigma_t)
            end
            @test res3
            @test length(pr.f_mu) ==
                  size(pr.f_sigma, 1) ==
                  size(pr.f_sigma, 2) ==
                  size(pr.loadings.M, 2)
            @test length(pr.mu) == size(pr.loadings.M, 1) == length(pr.loadings.b)
            @test pr === prior(pr)
        end
        pe1 = pes[1]
        ew = eweights(1:1000, 0.3)
        pe2 = PortfolioOptimisers.factory(pe1, ew)
        @test pe2.a_pe.ce.ce.ce.w == ew
        @test pe2.a_pe.ce.ce.me.w == ew
        @test pe2.a_pe.me.w === ew
        @test pe2.f_pe.ce.ce.ce.w == ew
        @test pe2.f_pe.ce.ce.me.w == ew
        @test pe2.f_pe.me.w === ew
        @test pe2.ve.me.w === ew
        @test pe2.ve.w === ew

        pr1 = prior(pes[1], ReturnsResult(; nx = 1:10, X = X, nf = 1:4, F = F))
        pr2 = prior(pes[1], X, F)
        @test pr1.X == pr2.X
        @test pr1.mu == pr2.mu
        @test pr1.sigma == pr2.sigma
        @test pr1.f_mu == pr2.f_mu
        @test pr1.f_sigma == pr2.f_sigma
        @test pr1.loadings.b == pr2.loadings.b
        @test pr1.loadings.M == pr2.loadings.M
    end
    @testset "Entropy Pooling Prior" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10)
        sets = DataFrame(:Assets => 1:10, :Clusters => [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        views = [ContinuousEntropyPoolingViewEstimator(;
                                                       A = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = :Assets,
                                                                                                      name = 1),
                                                       B = ConstantEntropyPoolingConstraintEstimator(;
                                                                                                     coef = 0.1),
                                                       comp = EQ()),
                 ContinuousEntropyPoolingViewEstimator(;
                                                       A = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = :Assets,
                                                                                                      name = 2),
                                                       B = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = :Assets,
                                                                                                      name = 2)),
                 ContinuousEntropyPoolingViewEstimator(;
                                                       A = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = :Assets,
                                                                                                      name = 2),
                                                       B = ConstantEntropyPoolingConstraintEstimator(;
                                                                                                     coef = 0.95),
                                                       comp = LEQ()),
                 ContinuousEntropyPoolingViewEstimator(;
                                                       A = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = :Assets,
                                                                                                      name = 3),
                                                       B = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = :Assets,
                                                                                                      name = 3)),
                 ContinuousEntropyPoolingViewEstimator(;
                                                       A = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = :Assets,
                                                                                                      name = 3),
                                                       B = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = :Assets,
                                                                                                      name = 3)),
                 ContinuousEntropyPoolingViewEstimator(;
                                                       A = C2_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = :Assets,
                                                                                                      name = 3,
                                                                                                      kind = SkewnessEntropyPoolingViewAlgorithm()),
                                                       B = ConstantEntropyPoolingConstraintEstimator(;
                                                                                                     coef = -0.25),
                                                       comp = GEQ()),
                 ContinuousEntropyPoolingViewEstimator(;
                                                       A = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = :Assets,
                                                                                                      name = 4),
                                                       B = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = :Assets,
                                                                                                      name = 4)),
                 ContinuousEntropyPoolingViewEstimator(;
                                                       A = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = :Assets,
                                                                                                      name = 4),
                                                       B = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = :Assets,
                                                                                                      name = 4)),
                 ContinuousEntropyPoolingViewEstimator(;
                                                       A = C2_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = :Assets,
                                                                                                      name = 4,
                                                                                                      kind = KurtosisEntropyPoolingAlgorithm()),
                                                       B = ConstantEntropyPoolingConstraintEstimator(;
                                                                                                     coef = 5.1),
                                                       comp = LEQ()),
                 ContinuousEntropyPoolingViewEstimator(;
                                                       A = C4_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group1 = :Assets,
                                                                                                      group2 = :Assets,
                                                                                                      name1 = 10,
                                                                                                      name2 = 3),
                                                       B = C4_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group1 = :Assets,
                                                                                                      group2 = :Assets,
                                                                                                      name1 = 10,
                                                                                                      name2 = 3,
                                                                                                      coef = 0.22),
                                                       comp = GEQ())]
        pes = [EntropyPoolingPriorEstimator(; views = views, sets = sets),
               EntropyPoolingPriorEstimator(; views = views, sets = sets,
                                            alg = H1_EntropyPooling()),
               EntropyPoolingPriorEstimator(; views = views, sets = sets,
                                            alg = H2_EntropyPooling()),
               EntropyPoolingPriorEstimator(; views = views, sets = sets,
                                            opt = JuMPEntropyPoolingEstimator(;
                                                                              slv = Solver(;
                                                                                           solver = Clarabel.Optimizer,
                                                                                           settings = Dict("verbose" => false)))),
               EntropyPoolingPriorEstimator(; views = views, sets = sets,
                                            alg = H1_EntropyPooling(),
                                            opt = JuMPEntropyPoolingEstimator(;
                                                                              slv = Solver(;
                                                                                           solver = Clarabel.Optimizer,
                                                                                           settings = Dict("verbose" => false)))),
               EntropyPoolingPriorEstimator(; views = views, sets = sets,
                                            alg = H2_EntropyPooling(),
                                            opt = JuMPEntropyPoolingEstimator(;
                                                                              slv = Solver(;
                                                                                           solver = Clarabel.Optimizer,
                                                                                           settings = Dict("verbose" => false))))]

        ress = (0.03270155949442489, 0.030586876690884276, 0.03058687450109127,
                0.032701560391104334, 0.03058687741491548, 0.030586877801521424)
        enss = (0.9678273553779563, 0.9698761687748948, 0.9698761708987229,
                0.9678273545101254, 0.9698761680726742, 0.9698761676977143)

        for (i, (pe, re_t, ens_t)) ∈ enumerate(zip(pes, ress, enss))
            pr = prior(pe, transpose(X); dims = 2)
            res = isapprox(pr.mu[1], 0.1; rtol = 5e-8)
            if !res
                println("Test Fails on iteration $i mu")
                find_tol(pr.mu[1], 0.1; name1 = :mu, name2 = "== 0.1")
            end
            @test res
            res = diag(pr.sigma)[2] <= 0.95
            if !res
                println("Test Fails on iteration $i sigma")
                find_tol(pr.sigma[1], 0.95; name1 = :mu, name2 = "<= 0.95")
            end
            @test res
            res = skewness(pr.X[:, 3], pr.w) >= -0.25
            if !res
                println("Test Fails on iteration $i skewness")
                find_tol(skewness(pr.X[:, 3], pr.w), -0.2; name1 = :skewness,
                         name2 = ">= -0.2")
            end
            @test res
            res = kurtosis(pr.X[:, 4], pr.w) + 3 <= 5.1
            if !res
                println("Test Fails on iteration $i kurtosis")
                find_tol(kurtosis(pr.X[:, 3], pr.w), 5.1; name1 = :kurtosis,
                         name2 = "<= 5.1")
            end
            @test res
            res = cov2cor(pr.sigma)[10, 3] >= 0.22
            if !res
                println("Test Fails on iteration $i correlation")
                find_tol(cov2cor(pr.sigma)[10, 3], 0.22; name1 = :covcor, name2 = ">= 0.05")
            end
            @test res

            re = relative_entropy(pr.w,
                                  range(; start = inv(100), stop = inv(100), length = 100))
            ens = effective_number_scenarios(pr.w,
                                             range(; start = inv(100), stop = inv(100),
                                                   length = 100))
            res = isapprox(re, re_t; rtol = 5e-7)
            if !res
                println("Test Fails on iteration $i re")
                find_tol(re, re_t; name1 = :re, name2 = :re_t)
            end
            res = isapprox(ens, ens_t; rtol = 5e-7)
            if !res
                println("Test Fails on iteration $i ens")
                find_tol(ens, ens_t; name1 = :ens, name2 = :ens_t)
            end
            @test ens == exp(-re)
            @test pr === prior(pr)
        end
        pe1 = pes[1]
        ew = eweights(1:1000, 0.3)
        pe2 = PortfolioOptimisers.factory(pe1, ew)
        @test pe2.pe.ce.ce.ce.w == ew
        @test pe2.pe.ce.ce.me.w == ew
        @test pe2.w === ew

        pr1 = prior(pes[1], ReturnsResult(; nx = 1:10, X = X))
        pr2 = prior(pes[1], X)
        @test pr1.X == pr2.X
        @test pr1.mu == pr2.mu
        @test pr1.sigma == pr2.sigma

        views = [ContinuousEntropyPoolingViewEstimator(;
                                                       A = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = [:Assets],
                                                                                                      coef = [1],
                                                                                                      name = [1]),
                                                       B = ConstantEntropyPoolingConstraintEstimator(;
                                                                                                     coef = 0.1),
                                                       comp = EQ()),
                 ContinuousEntropyPoolingViewEstimator(;
                                                       A = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = [:Assets],
                                                                                                      coef = [1],
                                                                                                      name = [2]),
                                                       B = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = [:Assets],
                                                                                                      coef = [1],
                                                                                                      name = [2])),
                 ContinuousEntropyPoolingViewEstimator(;
                                                       A = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = [:Assets],
                                                                                                      coef = [1],
                                                                                                      name = [2]),
                                                       B = ConstantEntropyPoolingConstraintEstimator(;
                                                                                                     coef = 0.95),
                                                       comp = LEQ()),
                 ContinuousEntropyPoolingViewEstimator(;
                                                       A = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = [:Assets],
                                                                                                      coef = [1],
                                                                                                      name = [3]),
                                                       B = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = [:Assets],
                                                                                                      coef = [1],
                                                                                                      name = [3])),
                 ContinuousEntropyPoolingViewEstimator(;
                                                       A = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = [:Assets],
                                                                                                      coef = [1],
                                                                                                      name = [3]),
                                                       B = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = [:Assets],
                                                                                                      coef = [1],
                                                                                                      name = [3])),
                 ContinuousEntropyPoolingViewEstimator(;
                                                       A = C2_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = [:Assets],
                                                                                                      name = [3],
                                                                                                      coef = [1],
                                                                                                      kind = SkewnessEntropyPoolingViewAlgorithm()),
                                                       B = ConstantEntropyPoolingConstraintEstimator(;
                                                                                                     coef = -0.2),
                                                       comp = GEQ()),
                 ContinuousEntropyPoolingViewEstimator(;
                                                       A = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = [:Assets],
                                                                                                      coef = [1],
                                                                                                      name = [4]),
                                                       B = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = [:Assets],
                                                                                                      coef = [1],
                                                                                                      name = [4])),
                 ContinuousEntropyPoolingViewEstimator(;
                                                       A = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = [:Assets],
                                                                                                      coef = [1],
                                                                                                      name = [4]),
                                                       B = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = [:Assets],
                                                                                                      coef = [1],
                                                                                                      name = [4])),
                 ContinuousEntropyPoolingViewEstimator(;
                                                       A = C2_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = [:Assets],
                                                                                                      name = [4],
                                                                                                      coef = [1],
                                                                                                      kind = KurtosisEntropyPoolingAlgorithm()),
                                                       B = ConstantEntropyPoolingConstraintEstimator(;
                                                                                                     coef = 5.1),
                                                       comp = LEQ()),
                 ContinuousEntropyPoolingViewEstimator(;
                                                       A = C4_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group1 = [:Assets],
                                                                                                      group2 = [:Assets],
                                                                                                      name1 = [3],
                                                                                                      name2 = [10],
                                                                                                      coef = [1]),
                                                       B = C4_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group1 = [:Assets],
                                                                                                      group2 = [:Assets],
                                                                                                      name1 = [3],
                                                                                                      name2 = [1],
                                                                                                      coef = [0.22]),
                                                       comp = GEQ())]

        ress = (0.03270155949442489, 0.030586876690884276, 0.03058687450109127,
                0.032701560391104334, 0.03058687741491548, 0.030586877801521424)
        enss = (0.9678273553779563, 0.9698761687748948, 0.9698761708987229,
                0.9678273545101254, 0.9698761680726742, 0.9698761676977143)

        for (i, (pe, re_t, ens_t)) ∈ enumerate(zip(pes, ress, enss))
            pr = prior(pe, transpose(X); dims = 2)
            res = isapprox(pr.mu[1], 0.1; rtol = 5e-8)
            if !res
                println("Test Fails on iteration $i mu")
                find_tol(pr.mu[1], 0.1; name1 = :mu, name2 = "== 0.1")
            end
            @test res
            res = diag(pr.sigma)[2] <= 0.95
            if !res
                println("Test Fails on iteration $i sigma")
                find_tol(pr.sigma[1], 0.95; name1 = :mu, name2 = "<= 0.95")
            end
            @test res
            res = skewness(pr.X[:, 3], pr.w) >= -0.25
            if !res
                println("Test Fails on iteration $i skewness")
                find_tol(skewness(pr.X[:, 3], pr.w), -0.2; name1 = :skewness,
                         name2 = ">= -0.2")
            end
            @test res
            res = kurtosis(pr.X[:, 4], pr.w) + 3 <= 5.1
            if !res
                println("Test Fails on iteration $i kurtosis")
                find_tol(kurtosis(pr.X[:, 3], pr.w), 5.1; name1 = :kurtosis,
                         name2 = "<= 5.1")
            end
            @test res
            res = cov2cor(pr.sigma)[10, 3] >= 0.22
            if !res
                println("Test Fails on iteration $i correlation")
                find_tol(cov2cor(pr.sigma)[10, 3], 0.22; name1 = :covcor, name2 = ">= 0.05")
            end
            @test res

            re = relative_entropy(pr.w,
                                  range(; start = inv(100), stop = inv(100), length = 100))
            ens = effective_number_scenarios(pr.w,
                                             range(; start = inv(100), stop = inv(100),
                                                   length = 100))
            res = isapprox(re, re_t; rtol = 5e-7)
            if !res
                println("Test Fails on iteration $i re")
                find_tol(re, re_t; name1 = :re, name2 = :re_t)
            end
            res = isapprox(ens, ens_t; rtol = 5e-7)
            if !res
                println("Test Fails on iteration $i ens")
                find_tol(ens, ens_t; name1 = :ens, name2 = :ens_t)
            end
            @test ens == exp(-re)
        end

        w = pweights(range(; start = inv(size(X, 1)), stop = inv(size(X, 1)),
                           length = size(X, 1)))
        pm0 = prior(EntropyPoolingPriorEstimator(; views = views, sets = sets, w = w), X)
        pr1 = prior(EntropyPoolingPriorEstimator(; views = views, sets = sets), X)
        @test isapprox(pm0.mu, pr1.mu)
        @test isapprox(pm0.sigma, pr1.sigma)

        pm0 = prior(EntropyPoolingPriorEstimator(; views = views, sets = sets,
                                                 alg = H1_EntropyPooling(), w = w), X)
        pr1 = prior(EntropyPoolingPriorEstimator(; views = views, sets = sets,
                                                 alg = H1_EntropyPooling()), X)
        @test isapprox(pm0.mu, pr1.mu)
        @test isapprox(pm0.sigma, pr1.sigma)

        c = C0_LinearEntropyPoolingConstraintEstimator(; group = [nothing, nothing],
                                                       name = [nothing, nothing],
                                                       coef = [-3, -3.5])
        @test all(isnothing, c.group) && all(isnothing, c.name)
        d = PortfolioOptimisers.freeze_A_view(c)
        @test all(d.coef .== -1)
        c = C0_LinearEntropyPoolingConstraintEstimator(; group = nothing, name = nothing,
                                                       coef = -4)
        @test isnothing(c.group) && isnothing(c.name)
        d = PortfolioOptimisers.freeze_A_view(c)
        @test d.coef == -1

        c = C1_LinearEntropyPoolingConstraintEstimator(; group = [nothing, nothing],
                                                       name = [nothing, nothing],
                                                       coef = [-5, -5.5])
        @test all(isnothing, c.group) && all(isnothing, c.name)
        d = PortfolioOptimisers.freeze_A_view(c)
        @test all(d.coef .== -1)
        c = C1_LinearEntropyPoolingConstraintEstimator(; group = nothing, name = nothing,
                                                       coef = -6)
        @test isnothing(c.group) && isnothing(c.name)
        d = PortfolioOptimisers.freeze_A_view(c)
        @test d.coef == -1

        c = C1_LinearEntropyPoolingConstraintEstimator(; group = [nothing, nothing],
                                                       name = [nothing, nothing],
                                                       coef = [-5, -5.5])
        @test all(isnothing, c.group) && all(isnothing, c.name)
        d = PortfolioOptimisers.freeze_A_view(c)
        @test all(d.coef .== -1)
        c = C1_LinearEntropyPoolingConstraintEstimator(; group = nothing, name = nothing,
                                                       coef = -6)
        @test isnothing(c.group) && isnothing(c.name)
        d = PortfolioOptimisers.freeze_A_view(c)
        @test d.coef == -1

        c = C2_LinearEntropyPoolingConstraintEstimator(; group = [nothing, nothing],
                                                       name = [nothing, nothing],
                                                       coef = [-7, -7.5],
                                                       kind = SkewnessEntropyPoolingViewAlgorithm())
        @test all(isnothing, c.group) && all(isnothing, c.name)
        d = PortfolioOptimisers.freeze_A_view(c)
        @test all(d.coef .== -1)

        c = C2_LinearEntropyPoolingConstraintEstimator(; group = nothing, name = nothing,
                                                       coef = -8,
                                                       kind = SkewnessEntropyPoolingViewAlgorithm())
        @test isnothing(c.group) && isnothing(c.name)
        d = PortfolioOptimisers.freeze_A_view(c)
        @test d.coef == -1

        c = C4_LinearEntropyPoolingConstraintEstimator(; group1 = [nothing, nothing],
                                                       group2 = [nothing, nothing],
                                                       name1 = [nothing, nothing],
                                                       name2 = [nothing, nothing],
                                                       coef = [-9, -9.5])
        @test all(isnothing, c.group1) && all(isnothing, c.name1)
        @test all(isnothing, c.group2) && all(isnothing, c.name2)
        d = PortfolioOptimisers.freeze_A_view(c)
        @test all(d.coef .== -1)

        c = C4_LinearEntropyPoolingConstraintEstimator(; group1 = nothing, name1 = nothing,
                                                       group2 = nothing, name2 = nothing,
                                                       coef = -10)
        @test isnothing(c.group1) && isnothing(c.name1)
        @test isnothing(c.group2) && isnothing(c.name2)
        d = PortfolioOptimisers.freeze_A_view(c)
        @test d.coef == -1

        c = ContinuousEntropyPoolingViewEstimator(;
                                                  A = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                                 group = nothing,
                                                                                                 name = nothing,
                                                                                                 coef = -6),
                                                  B = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                                 group = nothing,
                                                                                                 name = nothing,
                                                                                                 coef = -6))
        @test c == (c[5] = c)
        @test sort(c) == c
        @test sort!(c) == c

        @test_throws AssertionError JuMPEntropyPoolingEstimator(; slv = Solver[])
        pe = EntropyPoolingPriorEstimator(;
                                          views = ContinuousEntropyPoolingViewEstimator(;
                                                                                        A = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                                                                       group = nothing,
                                                                                                                                       name = nothing,
                                                                                                                                       coef = -6),
                                                                                        B = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                                                                       group = nothing,
                                                                                                                                       name = nothing,
                                                                                                                                       coef = -6)))
        @test isa(pe.ce, PortfolioOptimisersCovariance)
        @test isa(pe.me, SimpleExpectedReturns)
    end
    @testset "Entropy Pooling Factor Prior" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10)
        F = X[:, [3, 8]] + randn(rng, 100, 2) * 0.0001
        sets = DataFrame(:Assets => 1:10)
        views = [ContinuousEntropyPoolingViewEstimator(;
                                                       A = C4_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group1 = :Assets,
                                                                                                      group2 = :Assets,
                                                                                                      name1 = 10,
                                                                                                      name2 = 3),
                                                       B = C4_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group1 = :Assets,
                                                                                                      group2 = :Assets,
                                                                                                      name1 = 10,
                                                                                                      name2 = 3,
                                                                                                      coef = 0.217),
                                                       comp = LEQ())]
        pes = [EntropyPoolingPriorEstimator(; pe = FactorPriorEstimator(;), views = views,
                                            sets = sets),
               EntropyPoolingPriorEstimator(; pe = FactorPriorEstimator(;), views = views,
                                            sets = sets, alg = H1_EntropyPooling()),
               EntropyPoolingPriorEstimator(; pe = FactorPriorEstimator(;), views = views,
                                            sets = sets, alg = H2_EntropyPooling())]

        ress = (1.0759615986400306e-15, 1.0759615986400306e-15, 1.0759615986400306e-15)
        enss = (0.9999999999999989, 0.9999999999999989, 0.9999999999999989)
        for (i, (pe, re_t, ens_t)) ∈ enumerate(zip(pes, ress, enss))
            pr = prior(pe, transpose(X), transpose(F); dims = 2)
            res = cov2cor(pr.sigma)[10, 3] <= 0.217
            if !res
                println("Test Fails on iteration $i correlation")
                find_tol(cov2cor(pr.sigma)[10, 3], 0.217; name1 = :covcor,
                         name2 = ">= 0.05")
            end
            @test res

            re = relative_entropy(pr.w,
                                  range(; start = inv(100), stop = inv(100), length = 100))
            ens = effective_number_scenarios(pr.w,
                                             range(; start = inv(100), stop = inv(100),
                                                   length = 100))
            res = isapprox(re, re_t; rtol = 5e-7)
            if !res
                println("Test Fails on iteration $i re")
                find_tol(re, re_t; name1 = :re, name2 = :re_t)
            end
            res = isapprox(ens, ens_t; rtol = 5e-7)
            if !res
                println("Test Fails on iteration $i ens")
                find_tol(ens, ens_t; name1 = :ens, name2 = :ens_t)
            end
            @test ens == exp(-re)
            @test pr === prior(pr)
        end
        i = [10, 5, 9]
        pes[1] === prior_view(pes[1], i)
        pr1 = prior(pes[1], X, F)
        pv1 = prior_view(pr1, i)
        @test pv1.mu == view(pr1.mu, i)
        @test pv1.sigma == view(pr1.sigma, i, i)
        @test pv1.X == view(pr1.X, :, i)
        @test pv1.f_mu == pr1.f_mu
        @test pv1.f_sigma == pr1.f_sigma
        @test pv1.loadings.b == view(pr1.loadings.b, i)
        @test pv1.loadings.M == view(pr1.loadings.M, i, :)
        @test pv1.chol == view(pr1.chol, :, i)
    end
end
