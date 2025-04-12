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
            @test pm === prior(pm)
        end
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
        @test pm1 === prior(pm1)

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
    @testset "High Order Prior" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10)
        pe = HighOrderPriorEstimator()
        pm = prior(pe, transpose(X); dims = 2)
        @test isapprox(pm.X, X)
        @test isapprox(pm.mu, vec(mean(SimpleExpectedReturns(), X)))
        @test isapprox(pm.sigma, cov(PortfolioOptimisersCovariance(), X))
        @test isapprox(pm.kt, cokurtosis(Cokurtosis(; alg = Full()), X))
        @test isapprox(pm.skt, cokurtosis(Cokurtosis(; alg = Semi()), X))
        @test all(isapprox.((pm.sk, pm.V), coskewness(Coskewness(; alg = Full()), X)))
        @test all(isapprox.((pm.ssk, pm.SV), coskewness(Coskewness(; alg = Semi()), X)))
        pm = prior(HighOrderPriorEstimator(; kte = nothing, skte = nothing, ske = nothing,
                                           sske = nothing), transpose(X); dims = 2)
        @test isnothing(pm.kt)
        @test isnothing(pm.skt)
        @test isnothing(pm.sk)
        @test isnothing(pm.V)
        @test isnothing(pm.ssk)
        @test isnothing(pm.SV)
        @test pm === prior(pm)
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
        blvr = black_littterman_views(views, sets)
        @test isapprox(blvr.P,
                       reshape([0.0, 0.0, 0.0, -0.3333333333333333, 0.0, 1.0, 0.0, 0.0,
                                -0.3333333333333333, 0.0, 0.0, 1.0, 0.25, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.3333333333333333, 0.0, 0.0, 0.25, 1.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0,
                                0.3333333333333333, 0.0, -1.0, 0.0, -0.3333333333333333,
                                0.0, 0.0, 0.0, -0.75, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0],
                               :, 10))
        @test isapprox(blvr.Q, [0.003, -0.001, 0.002, 0.007, 0.001])
        @test blvr === black_littterman_views(blvr)
        @test isnothing(black_littterman_views(nothing))
        @test isnothing(black_littterman_views(BlackLittermanViewsEstimator(;
                                                                            A = LinearConstraintSide(;
                                                                                                     group = nothing,
                                                                                                     name = nothing)),
                                               sets))
        vc = BlackLittermanViewsEstimator(;
                                          A = LinearConstraintSide(; group = :Asset,
                                                                   name = -1, coef = 1.0),
                                          B = 0.003)
        @test isnothing(black_littterman_views(vc, sets))
        @test_throws ArgumentError black_littterman_views(vc, sets, strict = true)

        vc = BlackLittermanViewsEstimator(;
                                          A = LinearConstraintSide(; group = [:Asset],
                                                                   name = [-1],
                                                                   coef = [1.0]), B = 0.003)
        @test isnothing(black_littterman_views(vc, sets))
        @test_throws ArgumentError black_littterman_views(vc, sets, strict = true)

        vc = BlackLittermanViewsEstimator(;
                                          A = LinearConstraintSide(; group = :Foo,
                                                                   name = -1, coef = 1.0),
                                          B = 0.003)
        @test isnothing(black_littterman_views(vc, sets))
        @test_throws ArgumentError black_littterman_views(vc, sets, strict = true)

        vc = BlackLittermanViewsEstimator(;
                                          A = LinearConstraintSide(; group = [:Foo],
                                                                   name = [-1],
                                                                   coef = [1.0]), B = 0.003)
        @test isnothing(black_littterman_views(vc, sets))
        @test_throws ArgumentError black_littterman_views(vc, sets, strict = true)
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
        @test isa(pe1.me, ShrunkExpectedReturns{<:JamesStein, <:Any, <:Any, <:GrandMean})
        @test isa(pe1.ce.ce, GerberCovariance{<:Gerber0, <:Any, <:Any, <:Any})
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
                  ShrunkExpectedReturns{<:BodnarOkhrinParolya, <:Any, <:Any, <:GrandMean})
        @test isa(pe2.ce.ce, GerberCovariance{<:Gerber2, <:Any, <:Any, <:Any})
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
        @test isa(pe3.ce.ce, GerberCovariance{<:Gerber1, <:Any, <:Any, <:Any})
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
                  SmythBrobyCovariance{<:SmythBroby0, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:Any})
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
        @test isa(pe5.ce.ce, Covariance{<:Full, <:Any, <:Any})
        pe6 = BlackLittermanPriorEstimator(;
                                           views = BlackLittermanViewsEstimator(;
                                                                                A = LinearConstraintSide(;
                                                                                                         name = nothing,
                                                                                                         group = nothing)),
                                           pe = EmpiricalPriorEstimator(;
                                                                        me = ShrunkExpectedReturns(;
                                                                                                   alg = BayesStein()),
                                                                        ce = PortfolioOptimisersCovariance(;
                                                                                                           ce = NormalisedSmythBrobyCovariance(;
                                                                                                                                               alg = SmythBroby0()))))
        @test isa(pe6.me, ShrunkExpectedReturns{<:BayesStein, <:Any, <:Any, <:GrandMean})
        @test isa(pe6.ce.ce,
                  NormalisedSmythBrobyCovariance{<:SmythBroby0, <:Any, <:Any, <:Any, <:Any,
                                                 <:Any, <:Any, <:Any, <:Any})
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
                                                                                                                                     ce = NormalisedGerberCovariance(;
                                                                                                                                                                     alg = Gerber0())))))
        @test pe7.tau == 0.5
        @test isa(pe7.me, ShrunkExpectedReturns{<:JamesStein, <:Any, <:Any, <:GrandMean})
        @test isa(pe7.ce.ce,
                  NormalisedGerberCovariance{<:Gerber0, <:Any, <:Any, <:Any, <:Any})
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
                                                                                                                                                   ce = NormalisedGerberCovariance(;
                                                                                                                                                                                   alg = Gerber2())))))
        @test isa(pe8.me,
                  ShrunkExpectedReturns{<:BodnarOkhrinParolya, <:Any, <:Any, <:GrandMean})
        @test isa(pe8.ce.ce,
                  NormalisedGerberCovariance{<:Gerber2, <:Any, <:Any, <:Any, <:Any})
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
                                                                                                                                                        ce = NormalisedGerberCovariance(;
                                                                                                                                                                                        alg = Gerber1())))))
        @test isa(pe9.me, ExcessExpectedReturns)
        @test isa(pe9.ce.ce,
                  NormalisedGerberCovariance{<:Gerber1, <:Any, <:Any, <:Any, <:Any})
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
        @test isa(pe5.ce.ce, Covariance{<:Full, <:Any, <:Any})
        pe11 = FactorPriorEstimator(;
                                    pe = EmpiricalPriorEstimator(;
                                                                 me = ShrunkExpectedReturns(;
                                                                                            alg = BayesStein()),
                                                                 ce = PortfolioOptimisersCovariance(;
                                                                                                    ce = NormalisedSmythBrobyCovariance(;
                                                                                                                                        alg = SmythBroby0()))))
        @test isa(pe11.me, ShrunkExpectedReturns{<:BayesStein, <:Any, <:Any, <:GrandMean})
        @test isa(pe11.ce.ce,
                  NormalisedSmythBrobyCovariance{<:SmythBroby0, <:Any, <:Any, <:Any, <:Any,
                                                 <:Any, <:Any, <:Any, <:Any})
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
                                                                                                                                      ce = NormalisedGerberCovariance(;
                                                                                                                                                                      alg = Gerber2())))))
        @test isa(pe12.me, ShrunkExpectedReturns{<:JamesStein, <:Any, <:Any, <:GrandMean})
        @test isa(pe12.ce.ce,
                  NormalisedGerberCovariance{<:Gerber2, <:Any, <:Any, <:Any, <:Any})
        pe13 = FactorBlackLittermanPriorEstimator(; tau = 0.3,
                                                  views = BlackLittermanViewsEstimator(;
                                                                                       A = LinearConstraintSide(;
                                                                                                                name = nothing,
                                                                                                                group = nothing)),
                                                  pe = EmpiricalPriorEstimator(;
                                                                               me = ShrunkExpectedReturns(;
                                                                                                          alg = BayesStein()),
                                                                               ce = PortfolioOptimisersCovariance(;
                                                                                                                  ce = NormalisedSmythBrobyCovariance(;
                                                                                                                                                      alg = SmythBroby0()))))
        @test pe13.tau == 0.3
        @test isa(pe13.me, ShrunkExpectedReturns{<:BayesStein, <:Any, <:Any, <:GrandMean})
        @test isa(pe13.ce.ce,
                  NormalisedSmythBrobyCovariance{<:SmythBroby0, <:Any, <:Any, <:Any, <:Any,
                                                 <:Any, <:Any, <:Any, <:Any})
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
                                                                                                                                                    ce = NormalisedGerberCovariance(;
                                                                                                                                                                                    alg = Gerber2())))))
        @test isa(pe14.me, ShrunkExpectedReturns{<:JamesStein, <:Any, <:Any, <:GrandMean})
        @test isa(pe14.ce.ce,
                  NormalisedGerberCovariance{<:Gerber2, <:Any, <:Any, <:Any, <:Any})
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
                                                                                                                       ce = NormalisedSmythBrobyCovariance(;
                                                                                                                                                           alg = SmythBroby0()))))
        @test pe15.tau == 0.2
        @test isa(pe15.me, ShrunkExpectedReturns{<:BayesStein, <:Any, <:Any, <:GrandMean})
        @test isa(pe15.ce.ce,
                  NormalisedSmythBrobyCovariance{<:SmythBroby0, <:Any, <:Any, <:Any, <:Any,
                                                 <:Any, <:Any, <:Any, <:Any})
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
                                                                                                                                                         ce = NormalisedGerberCovariance(;
                                                                                                                                                                                         alg = Gerber2())))))
        @test isa(pe16.me, ShrunkExpectedReturns{<:JamesStein, <:Any, <:Any, <:GrandMean})
        @test isa(pe16.ce.ce,
                  NormalisedGerberCovariance{<:Gerber2, <:Any, <:Any, <:Any, <:Any})
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
            @test pm === prior(pm)
        end
        @test_throws ArgumentError black_littterman_views(BlackLittermanViewsEstimator(;
                                                                                       A = LinearConstraintSide(;
                                                                                                                group = :Foo,
                                                                                                                name = 2,
                                                                                                                coef = 1),
                                                                                       B = 0.003),
                                                          sets; strict = true)
        @test_throws ArgumentError black_littterman_views(BlackLittermanViewsEstimator(;
                                                                                       A = LinearConstraintSide(;
                                                                                                                group = [:Foo],
                                                                                                                name = [2],
                                                                                                                coef = [1]),
                                                                                       B = 0.003),
                                                          sets; strict = true)
        @test_throws ArgumentError black_littterman_views(BlackLittermanViewsEstimator(;
                                                                                       A = LinearConstraintSide(;
                                                                                                                group = :Asset,
                                                                                                                name = 11,
                                                                                                                coef = 1),
                                                                                       B = 0.003),
                                                          sets, strict = true)
        @test_throws ArgumentError black_littterman_views(BlackLittermanViewsEstimator(;
                                                                                       A = LinearConstraintSide(;
                                                                                                                group = [:Asset],
                                                                                                                name = [11],
                                                                                                                coef = [1]),
                                                                                       B = 0.003),
                                                          sets, strict = true)
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
            @test pm === prior(pm)
        end
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
    @testset "Entropy Pooling" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10)
        sets = DataFrame(:Assets => 1:10, :Clusters => [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        views = [EntropyPoolingViewEstimator(;
                                             A = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = :Assets,
                                                                                            name = 1),
                                             B = ConstantEntropyPoolingConstraintEstimator(;
                                                                                           coef = 0.1),
                                             comp = EQ()),
                 EntropyPoolingViewEstimator(;
                                             A = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = :Assets,
                                                                                            name = 2),
                                             B = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = :Assets,
                                                                                            name = 2)),
                 EntropyPoolingViewEstimator(;
                                             A = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = :Assets,
                                                                                            name = 2),
                                             B = ConstantEntropyPoolingConstraintEstimator(;
                                                                                           coef = 0.95),
                                             comp = LEQ()),
                 EntropyPoolingViewEstimator(;
                                             A = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = :Assets,
                                                                                            name = 3),
                                             B = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = :Assets,
                                                                                            name = 3)),
                 EntropyPoolingViewEstimator(;
                                             A = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = :Assets,
                                                                                            name = 3),
                                             B = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = :Assets,
                                                                                            name = 3)),
                 EntropyPoolingViewEstimator(;
                                             A = C2_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = :Assets,
                                                                                            name = 3,
                                                                                            kind = SkewnessEntropyPoolingViewAlgorithm()),
                                             B = ConstantEntropyPoolingConstraintEstimator(;
                                                                                           coef = -0.25),
                                             comp = GEQ()),
                 EntropyPoolingViewEstimator(;
                                             A = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = :Assets,
                                                                                            name = 4),
                                             B = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = :Assets,
                                                                                            name = 4)),
                 EntropyPoolingViewEstimator(;
                                             A = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = :Assets,
                                                                                            name = 4),
                                             B = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = :Assets,
                                                                                            name = 4)),
                 EntropyPoolingViewEstimator(;
                                             A = C2_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = :Assets,
                                                                                            name = 4,
                                                                                            kind = KurtosisEntropyPoolingAlgorithm()),
                                             B = ConstantEntropyPoolingConstraintEstimator(;
                                                                                           coef = 5.1),
                                             comp = LEQ()),
                 EntropyPoolingViewEstimator(;
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
            pm = prior(pe, transpose(X); dims = 2)
            res = isapprox(pm.mu[1], 0.1; rtol = 5e-8)
            if !res
                println("Test Fails on iteration $i mu")
                find_tol(pm.mu[1], 0.1; name1 = :mu, name2 = "== 0.1")
            end
            @test res
            res = diag(pm.sigma)[2] <= 0.95
            if !res
                println("Test Fails on iteration $i sigma")
                find_tol(pm.sigma[1], 0.95; name1 = :mu, name2 = "<= 0.95")
            end
            @test res
            res = skewness(pm.X[:, 3], pm.w) >= -0.25
            if !res
                println("Test Fails on iteration $i skewness")
                find_tol(skewness(pm.X[:, 3], pm.w), -0.2; name1 = :skewness,
                         name2 = ">= -0.2")
            end
            @test res
            res = kurtosis(pm.X[:, 4], pm.w) + 3 <= 5.1
            if !res
                println("Test Fails on iteration $i kurtosis")
                find_tol(kurtosis(pm.X[:, 3], pm.w), 5.1; name1 = :kurtosis,
                         name2 = "<= 5.1")
            end
            @test res
            res = cov2cor(pm.sigma)[10, 3] >= 0.22
            if !res
                println("Test Fails on iteration $i correlation")
                find_tol(cov2cor(pm.sigma)[10, 3], 0.22; name1 = :covcor, name2 = ">= 0.05")
            end
            @test res

            re = relative_entropy(pm.w,
                                  range(; start = inv(100), stop = inv(100), length = 100))
            ens = effective_number_scenarios(pm.w,
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

        views = [EntropyPoolingViewEstimator(;
                                             A = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = [:Assets],
                                                                                            coef = [1],
                                                                                            name = [1]),
                                             B = ConstantEntropyPoolingConstraintEstimator(;
                                                                                           coef = 0.1),
                                             comp = EQ()),
                 EntropyPoolingViewEstimator(;
                                             A = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = [:Assets],
                                                                                            coef = [1],
                                                                                            name = [2]),
                                             B = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = [:Assets],
                                                                                            coef = [1],
                                                                                            name = [2])),
                 EntropyPoolingViewEstimator(;
                                             A = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = [:Assets],
                                                                                            coef = [1],
                                                                                            name = [2]),
                                             B = ConstantEntropyPoolingConstraintEstimator(;
                                                                                           coef = 0.95),
                                             comp = LEQ()),
                 EntropyPoolingViewEstimator(;
                                             A = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = [:Assets],
                                                                                            coef = [1],
                                                                                            name = [3]),
                                             B = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = [:Assets],
                                                                                            coef = [1],
                                                                                            name = [3])),
                 EntropyPoolingViewEstimator(;
                                             A = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = [:Assets],
                                                                                            coef = [1],
                                                                                            name = [3]),
                                             B = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = [:Assets],
                                                                                            coef = [1],
                                                                                            name = [3])),
                 EntropyPoolingViewEstimator(;
                                             A = C2_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = [:Assets],
                                                                                            name = [3],
                                                                                            coef = [1],
                                                                                            kind = SkewnessEntropyPoolingViewAlgorithm()),
                                             B = ConstantEntropyPoolingConstraintEstimator(;
                                                                                           coef = -0.2),
                                             comp = GEQ()),
                 EntropyPoolingViewEstimator(;
                                             A = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = [:Assets],
                                                                                            coef = [1],
                                                                                            name = [4]),
                                             B = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = [:Assets],
                                                                                            coef = [1],
                                                                                            name = [4])),
                 EntropyPoolingViewEstimator(;
                                             A = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = [:Assets],
                                                                                            coef = [1],
                                                                                            name = [4]),
                                             B = C1_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = [:Assets],
                                                                                            coef = [1],
                                                                                            name = [4])),
                 EntropyPoolingViewEstimator(;
                                             A = C2_LinearEntropyPoolingConstraintEstimator(;
                                                                                            group = [:Assets],
                                                                                            name = [4],
                                                                                            coef = [1],
                                                                                            kind = KurtosisEntropyPoolingAlgorithm()),
                                             B = ConstantEntropyPoolingConstraintEstimator(;
                                                                                           coef = 5.1),
                                             comp = LEQ()),
                 EntropyPoolingViewEstimator(;
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
            pm = prior(pe, transpose(X); dims = 2)
            res = isapprox(pm.mu[1], 0.1; rtol = 5e-8)
            if !res
                println("Test Fails on iteration $i mu")
                find_tol(pm.mu[1], 0.1; name1 = :mu, name2 = "== 0.1")
            end
            @test res
            res = diag(pm.sigma)[2] <= 0.95
            if !res
                println("Test Fails on iteration $i sigma")
                find_tol(pm.sigma[1], 0.95; name1 = :mu, name2 = "<= 0.95")
            end
            @test res
            res = skewness(pm.X[:, 3], pm.w) >= -0.25
            if !res
                println("Test Fails on iteration $i skewness")
                find_tol(skewness(pm.X[:, 3], pm.w), -0.2; name1 = :skewness,
                         name2 = ">= -0.2")
            end
            @test res
            res = kurtosis(pm.X[:, 4], pm.w) + 3 <= 5.1
            if !res
                println("Test Fails on iteration $i kurtosis")
                find_tol(kurtosis(pm.X[:, 3], pm.w), 5.1; name1 = :kurtosis,
                         name2 = "<= 5.1")
            end
            @test res
            res = cov2cor(pm.sigma)[10, 3] >= 0.22
            if !res
                println("Test Fails on iteration $i correlation")
                find_tol(cov2cor(pm.sigma)[10, 3], 0.22; name1 = :covcor, name2 = ">= 0.05")
            end
            @test res

            re = relative_entropy(pm.w,
                                  range(; start = inv(100), stop = inv(100), length = 100))
            ens = effective_number_scenarios(pm.w,
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
        pm1 = prior(EntropyPoolingPriorEstimator(; views = views, sets = sets,), X)
        @test isapprox(pm0.mu, pm1.mu)
        @test isapprox(pm0.sigma, pm1.sigma)

        pm0 = prior(EntropyPoolingPriorEstimator(; views = views, sets = sets,
                                                 alg = H1_EntropyPooling(), w = w), X)
        pm1 = prior(EntropyPoolingPriorEstimator(; views = views, sets = sets,
                                                 alg = H1_EntropyPooling()), X)
        @test isapprox(pm0.mu, pm1.mu)
        @test isapprox(pm0.sigma, pm1.sigma)

        c = C0_LinearEntropyPoolingConstraintEstimator(; group = [nothing, nothing],
                                                       name = [nothing, nothing],
                                                       coef = [-3, -3.5])
        @test all(isnothing.(c.group)) && all(isnothing.(c.name))
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
        @test all(isnothing.(c.group)) && all(isnothing.(c.name))
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
        @test all(isnothing.(c.group)) && all(isnothing.(c.name))
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
        @test all(isnothing.(c.group)) && all(isnothing.(c.name))
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
        @test all(isnothing.(c.group1)) && all(isnothing.(c.name1))
        @test all(isnothing.(c.group2)) && all(isnothing.(c.name2))
        d = PortfolioOptimisers.freeze_A_view(c)
        @test all(d.coef .== -1)

        c = C4_LinearEntropyPoolingConstraintEstimator(; group1 = nothing, name1 = nothing,
                                                       group2 = nothing, name2 = nothing,
                                                       coef = -10)
        @test isnothing(c.group1) && isnothing(c.name1)
        @test isnothing(c.group2) && isnothing(c.name2)
        d = PortfolioOptimisers.freeze_A_view(c)
        @test d.coef == -1

        c = EntropyPoolingViewEstimator(;
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
                                          views = EntropyPoolingViewEstimator(;
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
end
