@safetestset "Moments" begin
    using Test, PortfolioOptimisers, DataFrames, TimeSeries, CSV, CovarianceEstimation,
          StableRNGs, StatsBase, LinearAlgebra, SparseArrays, Distributions
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
    rng = StableRNG(123456789)
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end],
                           TimeArray(CSV.File(joinpath(@__DIR__, "./assets/Factors.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    ew = eweights(1:252, inv(252); scale = true)
    fw = fweights(rand(rng, 252))
    rf = 4.34 / 100 / 252
    @testset "Expected ReturnsResult" begin
        mes = [ShrunkExpectedReturns(; alg = JamesStein()),
               ShrunkExpectedReturns(; alg = JamesStein(; tgt = VolatilityWeighted())),
               ShrunkExpectedReturns(; alg = JamesStein(; tgt = MeanSquaredError())),
               ShrunkExpectedReturns(; alg = JamesStein(),
                                     me = SimpleExpectedReturns(; w = ew)),
               ShrunkExpectedReturns(; alg = BayesStein()),
               ShrunkExpectedReturns(; alg = BayesStein(; tgt = VolatilityWeighted())),
               ShrunkExpectedReturns(; alg = BayesStein(; tgt = MeanSquaredError())),
               ShrunkExpectedReturns(; alg = BayesStein(),
                                     me = SimpleExpectedReturns(; w = ew)),
               ShrunkExpectedReturns(; alg = BodnarOkhrinParolya()),
               ShrunkExpectedReturns(;
                                     alg = BodnarOkhrinParolya(;
                                                               tgt = VolatilityWeighted())),
               ShrunkExpectedReturns(;
                                     alg = BodnarOkhrinParolya(; tgt = MeanSquaredError())),
               ShrunkExpectedReturns(; alg = BodnarOkhrinParolya(),
                                     me = SimpleExpectedReturns(; w = ew)),
               EquilibriumExpectedReturns(), ExcessExpectedReturns(; rf = rf)]
        df = CSV.read(joinpath(@__DIR__, "./assets/expected_returns.csv.gz"), DataFrame)
        for (i, me) in pairs(mes)
            mu = mean(me, rd.X)
            success = isapprox(vec(mu), df[!, i])
            if !success
                println("Counter: $i")
                find_tol(mu, df[!, i])
            end
            @test success
        end
        me0 = StandardDeviationExpectedReturns()
        @test isapprox(mean(me0, rd.X), std(me0.ce, rd.X))
        @test isapprox(mean(me0, rd.X), sqrt.(var(me0.ce, rd.X)))
        me = PortfolioOptimisers.factory(StandardDeviationExpectedReturns(), ew)
        @test me.ce.ce.me.w === ew
        @test me.ce.ce.ce.w === ew

        me0 = ShrunkExpectedReturns(;
                                    ce = PortfolioOptimisersCovariance(;
                                                                       ce = Covariance(;
                                                                                       alg = Semi())),
                                    alg = JamesStein(; tgt = VolatilityWeighted()))
        me = PortfolioOptimisers.factory(me0, ew)
        @test !(me.me === me0.me)
        @test !(me.ce === me0.ce)
        @test me.alg === me0.alg
        @test me.me.w === ew
        @test me.ce.ce.me.w === ew
        @test me.ce.ce.ce.w === ew
        @test me.ce.ce.alg === me0.ce.ce.alg

        me0 = EquilibriumExpectedReturns(;
                                         ce = PortfolioOptimisersCovariance(;
                                                                            ce = Covariance(;
                                                                                            alg = Semi())),
                                         w = [1, 2])
        me = PortfolioOptimisers.factory(me0, ew)
        @test !(me.ce === me0.ce)
        @test me.ce.ce.me.w === ew
        @test me.ce.ce.ce.w === ew
        @test me.w === me0.w
        @test me.l == me0.l

        me0 = ExcessExpectedReturns(; rf = 2)
        me = PortfolioOptimisers.factory(me0, ew)
        @test me.me.w === ew
        @test me.rf == me0.rf

        me0 = factory(WindowedExpectedReturns(; window = 50), ew)
        me = factory(me0.me, ew[(end - 49):end])
        @test mean(me0, rd.X) == mean(me, rd.X[(end - 49):end, :])
    end
    @testset "Covariance Estimators" begin
        ces = [Covariance(; alg = Full()),
               Covariance(; alg = Full(), me = SimpleExpectedReturns(; w = ew),
                          ce = GeneralCovariance(;
                                                 ce = SimpleCovariance(; corrected = false),
                                                 w = ew)),
               Covariance(; alg = Full(),
                          ce = GeneralCovariance(; ce = AnalyticalNonlinearShrinkage())),
               Covariance(; alg = Semi()),
               Covariance(; alg = Semi(), me = SimpleExpectedReturns(; w = ew),
                          ce = GeneralCovariance(;
                                                 ce = SimpleCovariance(; corrected = false),
                                                 w = ew)),
               Covariance(; alg = Semi(), me = SimpleExpectedReturns(; w = fw),
                          ce = GeneralCovariance(; ce = AnalyticalNonlinearShrinkage(),
                                                 w = fw)), SpearmanCovariance(),
               KendallCovariance(), MutualInfoCovariance(),
               MutualInfoCovariance(; bins = Knuth()),
               MutualInfoCovariance(; bins = FreedmanDiaconis()),
               MutualInfoCovariance(; bins = Scott()), MutualInfoCovariance(; bins = 5),
               MutualInfoCovariance(;
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew)),
               DistanceCovariance(), DistanceCovariance(; w = ew),
               LowerTailDependenceCovariance(),
               GerberCovariance(; alg = Gerber0(), me = CustomValueExpectedReturns()),
               GerberCovariance(; alg = Gerber0(), me = CustomValueExpectedReturns(),
                                ve = SimpleVariance(; me = SimpleExpectedReturns(; w = ew),
                                                    corrected = false, w = ew)),
               GerberCovariance(; alg = Gerber0(), me = SimpleExpectedReturns(;)),
               GerberCovariance(; alg = Gerber0(), me = SimpleExpectedReturns(; w = ew),
                                ve = SimpleVariance(; me = SimpleExpectedReturns(; w = ew),
                                                    corrected = false, w = ew)),
               GerberCovariance(; alg = Gerber1(), me = CustomValueExpectedReturns()),
               GerberCovariance(; alg = Gerber1()),
               GerberCovariance(; alg = Gerber2(), me = CustomValueExpectedReturns()),
               GerberCovariance(; alg = Gerber2()),
               SmythBrobyCovariance(; alg = SmythBroby0()),
               SmythBrobyCovariance(; alg = SmythBroby0(),
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew)),
               SmythBrobyCovariance(; alg = SmythBroby1()),
               SmythBrobyCovariance(; alg = SmythBroby2()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber0()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber1()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber2()),
               SmythBrobyCovariance(; alg = SmythBrobyCount0()),
               SmythBrobyCovariance(; alg = SmythBrobyCount1()),
               SmythBrobyCovariance(; alg = SmythBrobyCount2()),
               DenoiseCovariance(; dn = Denoise(; alg = SpectralDenoise())),
               DetoneCovariance(), ProcessedCovariance(; alg = LoGo()),
               GerberIQCovariance(; kind = BasicGerberIQ(), alg = Gerber0()),
               GerberIQCovariance(; kind = FullGerberIQ(), alg = Gerber1(), y = 1 / 252,
                                  t = 100, e = 10, sc = (x, y) -> (min(x, y), min(x, y))),
               GerberIQCovariance(; kind = PartialGerberIQ(), alg = Gerber2(),
                                  y = (x) -> inv(div(size(x, 2), 2)),
                                  t = (x) -> inv(div(size(x, 2), 3)),
                                  e = (x) -> inv(div(size(x, 2), 5)),
                                  sc = (x, y) -> (min(x, y), min(x, y)))]
        df = CSV.read(joinpath(@__DIR__, "./assets/covariance.csv.gz"), DataFrame)
        for (i, ce) in pairs(ces)
            cei = PortfolioOptimisersCovariance(; ce = ce)
            sigma = cov(cei, rd.X'; dims = 2)
            rho = cor(cei, rd.X'; dims = 2)
            @test isapprox(StatsBase.cov2cor(sigma), rho)
            df[!, "$i"] = vec(sigma)
            continue
            success = isapprox(vec(sigma), df[!, i])
            if !success
                println("Counter: $i")
                find_tol(vec(sigma), df[!, i])
            end
            @test success
        end

        ce0 = PortfolioOptimisersCovariance(; ce = CorrelationCovariance())
        @test isapprox(cov(ce0, rd.X), cor(PortfolioOptimisersCovariance(), rd.X))
        @test isapprox(cor(ce0, rd.X), cor(PortfolioOptimisersCovariance(), rd.X))
        ce = PortfolioOptimisers.factory(ce0, ew)
        @test ce.ce.ce.ce.w === ew
        @test ce.ce.ce.me.w === ew

        @test isapprox(df[!, 36],
                       vec(cov(PortfolioOptimisersCovariance(;
                                                             mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                                                   dn = Denoise(;
                                                                                                                alg = SpectralDenoise()))),
                               rd.X)))
        @test isapprox(df[!, 37],
                       vec(cov(PortfolioOptimisersCovariance(;
                                                             mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                                                   dt = Detone())),
                               rd.X)))
        @test isapprox(df[!, 37],
                       vec(cov(PortfolioOptimisersCovariance(;
                                                             mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                                                   order = DenoiseAlgDetone(),
                                                                                                   dt = Detone())),
                               rd.X)))
        @test isapprox(df[!, 37],
                       vec(cov(PortfolioOptimisersCovariance(;
                                                             mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                                                   order = DetoneDenoiseAlg(),
                                                                                                   dt = Detone())),
                               rd.X)))
        @test isapprox(df[!, 37],
                       vec(cov(PortfolioOptimisersCovariance(;
                                                             mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                                                   order = DetoneAlgDenoise(),
                                                                                                   dt = Detone())),
                               rd.X)))
        @test isapprox(df[!, 37],
                       vec(cov(PortfolioOptimisersCovariance(;
                                                             mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                                                   order = AlgDenoiseDetone(),
                                                                                                   dt = Detone())),
                               rd.X)))
        @test isapprox(df[!, 37],
                       vec(cov(PortfolioOptimisersCovariance(;
                                                             mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                                                   order = AlgDetoneDenoise(),
                                                                                                   dt = Detone())),
                               rd.X)))
        @test isapprox(df[!, 38],
                       vec(cov(PortfolioOptimisersCovariance(;
                                                             mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                                                   alg = LoGo())),
                               rd.X)))

        @test isapprox(var(SimpleVariance(; w = ew, corrected = false), rd.X; dims = 1),
                       std(SimpleVariance(; w = ew, corrected = false), rd.X; dims = 1) .^
                       2)
        @test std(SimpleVariance(), rd.X) == std(SimpleVariance(; me = nothing), rd.X)
        @test var(SimpleVariance(), rd.X) == var(SimpleVariance(; me = nothing), rd.X)

        ce0 = PortfolioOptimisersCovariance(;
                                            ce = GerberCovariance(; alg = Gerber2(),
                                                                  me = SimpleExpectedReturns(;),
                                                                  t = 0.1))
        ce = PortfolioOptimisers.factory(ce0, ew)
        @test !(ce.ce.ve === ce0.ce.ve)
        @test !(ce.ce.me === ce0.ce.me)
        @test ce.ce.pdm === ce0.ce.pdm
        @test ce.ce.alg === ce0.ce.alg
        @test ce.ce.t == ce0.ce.t
        @test ce.mp === ce0.mp
        @test ce.ce.ve.w === ew
        @test ce.ce.ve.me.w === ew
        @test ce.mp === ce0.mp
        @test isapprox(cov(GerberCovariance(; alg = Gerber0()), rd.X),
                       cov(GerberCovariance(; alg = Gerber0()), rd.X'; dims = 2))
        @test isapprox(cov(GerberCovariance(; alg = Gerber0(),
                                            me = SimpleExpectedReturns()), rd.X),
                       cov(GerberCovariance(; alg = Gerber0(),
                                            me = SimpleExpectedReturns()), rd.X'; dims = 2))
        @test isapprox(cor(GerberCovariance(; alg = Gerber0()), rd.X),
                       cor(GerberCovariance(; alg = Gerber0()), rd.X'; dims = 2))
        @test isapprox(cor(GerberCovariance(; alg = Gerber0(),
                                            me = SimpleExpectedReturns()), rd.X),
                       cor(GerberCovariance(; alg = Gerber0(),
                                            me = SimpleExpectedReturns()), rd.X'; dims = 2))

        ce0 = PortfolioOptimisersCovariance(;
                                            ce = SmythBrobyCovariance(; alg = SmythBroby2(),
                                                                      me = SimpleExpectedReturns(),
                                                                      c1 = 0.6, c2 = 0.2,
                                                                      c3 = 2.2, n = 3))
        ce = PortfolioOptimisers.factory(ce0, ew)
        @test !(ce.ce.ve === ce0.ce.ve)
        @test !(ce.ce.me === ce0.ce.me)
        @test ce.ce.ve.w === ew
        @test ce.ce.ve.me.w === ew
        @test ce.ce.pdm === ce0.ce.pdm
        @test ce.ce.c1 == ce0.ce.c1
        @test ce.ce.c2 == ce0.ce.c2
        @test ce.ce.c3 == ce0.ce.c3
        @test ce.ce.n == ce0.ce.n
        @test ce.ce.ex === ce0.ce.ex
        @test ce.ce.alg === ce0.ce.alg
        @test isapprox(cov(SmythBrobyCovariance(; alg = SmythBroby0()), rd.X),
                       cov(SmythBrobyCovariance(; alg = SmythBroby0()), rd.X'; dims = 2))
        @test isapprox(cor(SmythBrobyCovariance(; alg = SmythBroby0()), rd.X),
                       cor(SmythBrobyCovariance(; alg = SmythBroby0()), rd.X'; dims = 2))

        ce0 = PortfolioOptimisersCovariance(;
                                            ce = DistanceCovariance(; args = (3,),
                                                                    kwargs = (; foo = 5)))
        ce = PortfolioOptimisers.factory(ce0, ew)
        @test ce.ce.metric === ce0.ce.metric
        @test ce.ce.args === ce0.ce.args
        @test ce.ce.kwargs === ce0.ce.kwargs
        @test ce.ce.ex === ce0.ce.ex
        @test ce.ce.w === ew
        @test isapprox(cov(DistanceCovariance(), rd.X),
                       cov(DistanceCovariance(), rd.X'; dims = 2))
        @test isapprox(cor(DistanceCovariance(), rd.X),
                       cor(DistanceCovariance(), rd.X'; dims = 2))

        ce0 = PortfolioOptimisersCovariance(;
                                            ce = LowerTailDependenceCovariance(;
                                                                               alpha = 0.4))
        ce = PortfolioOptimisers.factory(ce0, ew)
        @test ce.ce.alpha == ce0.ce.alpha
        @test ce.ce.ve.w === ew
        @test ce.ce.ve.me.w === ew
        @test ce.ce.ex === ce0.ce.ex
        @test isapprox(cov(LowerTailDependenceCovariance(), rd.X),
                       cov(LowerTailDependenceCovariance(), rd.X'; dims = 2))
        @test isapprox(cor(LowerTailDependenceCovariance(), rd.X),
                       cor(LowerTailDependenceCovariance(), rd.X'; dims = 2))

        ce0 = PortfolioOptimisersCovariance(; ce = KendallCovariance(;))
        ce = PortfolioOptimisers.factory(ce0, ew)
        @test !(ce.ce.ve === ce0.ce.ve)
        @test ce.ce.ve.w === ew
        @test ce.ce.ve.me.w === ew
        @test isapprox(cov(KendallCovariance(), rd.X),
                       cov(KendallCovariance(), rd.X'; dims = 2))
        @test isapprox(cor(KendallCovariance(), rd.X),
                       cor(KendallCovariance(), rd.X'; dims = 2))

        ce0 = PortfolioOptimisersCovariance(; ce = SpearmanCovariance(;))
        ce = PortfolioOptimisers.factory(ce0, ew)
        @test !(ce.ce.ve === ce0.ce.ve)
        @test ce.ce.ve.w === ew
        @test ce.ce.ve.me.w === ew
        @test isapprox(cov(SpearmanCovariance(), rd.X),
                       cov(SpearmanCovariance(), rd.X'; dims = 2))
        @test isapprox(cor(SpearmanCovariance(), rd.X),
                       cor(SpearmanCovariance(), rd.X'; dims = 2))

        ce0 = PortfolioOptimisersCovariance(;
                                            ce = MutualInfoCovariance(; normalise = false,
                                                                      bins = Knuth()))
        ce = PortfolioOptimisers.factory(ce0, ew)
        @test !(ce.ce.ve === ce0.ce.ve)
        @test ce.ce.ve.w === ew
        @test ce.ce.ve.me.w === ew
        @test ce.ce.bins === ce0.ce.bins
        @test ce.ce.normalise === ce0.ce.normalise
        @test isapprox(cov(MutualInfoCovariance(), rd.X),
                       cov(MutualInfoCovariance(), rd.X'; dims = 2))
        @test isapprox(cor(MutualInfoCovariance(), rd.X),
                       cor(MutualInfoCovariance(), rd.X'; dims = 2))

        ce0 = PortfolioOptimisersCovariance(; ce = DenoiseCovariance())
        ce = PortfolioOptimisers.factory(ce0, ew)
        @test ce.ce.ce.ce.w === ew
        @test ce.ce.ce.me.w === ew
        @test ce.ce.dn == ce0.ce.dn
        @test ce.ce.pdm == ce0.ce.pdm
        @test isapprox(cov(DenoiseCovariance(), rd.X),
                       cov(DenoiseCovariance(), rd.X'; dims = 2))
        @test isapprox(cor(DenoiseCovariance(), rd.X),
                       cor(DenoiseCovariance(), rd.X'; dims = 2))

        ce0 = PortfolioOptimisersCovariance(; ce = DetoneCovariance())
        ce = PortfolioOptimisers.factory(ce0, ew)
        @test ce.ce.ce.ce.w === ew
        @test ce.ce.ce.me.w === ew
        @test ce.ce.dt == ce0.ce.dt
        @test ce.ce.pdm == ce0.ce.pdm
        @test isapprox(cov(DetoneCovariance(), rd.X),
                       cov(DetoneCovariance(), rd.X'; dims = 2))
        @test isapprox(cor(DetoneCovariance(), rd.X),
                       cor(DetoneCovariance(), rd.X'; dims = 2))

        ce0 = PortfolioOptimisersCovariance(; ce = ProcessedCovariance(; alg = LoGo()))
        ce = PortfolioOptimisers.factory(ce0, ew)
        @test ce.ce.ce.ce.w === ew
        @test ce.ce.ce.me.w === ew
        @test ce.ce.alg == ce0.ce.alg
        @test ce.ce.pdm == ce0.ce.pdm
        @test isapprox(cov(ProcessedCovariance(; alg = LoGo()), rd.X),
                       cov(ProcessedCovariance(; alg = LoGo()), rd.X'; dims = 2))
        @test isapprox(cor(ProcessedCovariance(; alg = LoGo()), rd.X),
                       cor(ProcessedCovariance(; alg = LoGo()), rd.X'; dims = 2))

        ce0 = factory(WindowedCovariance(;
                                         ce = PortfolioOptimisersCovariance(;
                                                                            ce = GeneralCovariance(;
                                                                                                   ce = SimpleCovariance(;
                                                                                                                         corrected = false))),
                                         window = 50), ew)
        ce = factory(ce0.ce, ew[(end - 49):end])
        @test isapprox(cov(ce0, rd.X[(end - 49):end, :]), cov(ce, rd.X[(end - 49):end, :]))
        @test isapprox(cor(ce0, rd.X[(end - 49):end, :]), cor(ce, rd.X[(end - 49):end, :]))

        ce0 = factory(WindowedCovariance(;
                                         ce = PortfolioOptimisersCovariance(;
                                                                            ce = GeneralCovariance(;
                                                                                                   ce = SimpleCovariance(;
                                                                                                                         corrected = false)))),
                      ew)
        ce = factory(ce0.ce, ew)
        @test isapprox(cov(ce0, rd.X), cov(ce, rd.X))
        @test isapprox(cor(ce0, rd.X), cor(ce, rd.X))

        ce0 = factory(WindowedVariance(; ce = SimpleVariance(; corrected = false),
                                       window = 50), ew)
        ce = factory(ce0.ce, ew[(end - 49):end])
        @test isapprox(var(ce0, rd.X[(end - 49):end, :]), var(ce, rd.X[(end - 49):end, :]))
        @test isapprox(std(ce0, rd.X[(end - 49):end, :]), std(ce, rd.X[(end - 49):end, :]))
        @test isapprox(var(ce0, rd.X[(end - 49):end, 1]), var(ce, rd.X[(end - 49):end, 1]))
        @test isapprox(std(ce0, rd.X[(end - 49):end, 2]), std(ce, rd.X[(end - 49):end, 2]))

        ce0 = factory(WindowedVariance(; ce = SimpleVariance(; corrected = false)), ew)
        ce = factory(ce0.ce, ew)
        @test isapprox(var(ce0, rd.X), var(ce, rd.X))
        @test isapprox(std(ce0, rd.X), std(ce, rd.X))
        @test isapprox(var(ce0, rd.X[:, 1]), var(ce, rd.X[:, 1]))
        @test isapprox(std(ce0, rd.X[:, 2]), std(ce, rd.X[:, 2]))

        ce0 = factory(WindowedVariance(;
                                       ce = SimpleVariance(; me = nothing,
                                                           corrected = false), window = 50),
                      ew)
        ce = factory(ce0.ce, ew[(end - 49):end])
        @test isapprox(var(ce0, rd.X[(end - 49):end, :]; mean = zeros(1, size(rd.X, 2))),
                       var(ce, rd.X[(end - 49):end, :]; mean = zeros(1, size(rd.X, 2))))
        @test isapprox(std(ce0, rd.X[(end - 49):end, :]; mean = zeros(1, size(rd.X, 2))),
                       std(ce, rd.X[(end - 49):end, :]; mean = zeros(1, size(rd.X, 2))))
        @test isapprox(var(ce0, rd.X[(end - 49):end, 1]; mean = 0),
                       var(ce, rd.X[(end - 49):end, 1]; mean = 0))
        @test isapprox(std(ce0, rd.X[(end - 49):end, 2]; mean = 0),
                       std(ce, rd.X[(end - 49):end, 2]; mean = 0))

        ce0 = factory(WindowedVariance(;
                                       ce = SimpleVariance(; me = nothing,
                                                           corrected = false)), ew)
        ce = factory(ce0.ce, ew)
        @test isapprox(var(ce0, rd.X; mean = zeros(1, size(rd.X, 2))),
                       var(ce, rd.X; mean = zeros(1, size(rd.X, 2))))
        @test isapprox(std(ce0, rd.X; mean = zeros(1, size(rd.X, 2))),
                       std(ce, rd.X; mean = zeros(1, size(rd.X, 2))))
        @test isapprox(var(ce0, rd.X[:, 1]; mean = 0), var(ce, rd.X[:, 1]; mean = 0))
        @test isapprox(std(ce0, rd.X[:, 2]; mean = 0), std(ce, rd.X[:, 2]; mean = 0))

        @test find_uncorrelated_indices(rd.X; t = 0.5) == [4, 6, 12, 16, 17, 19]

        for alg in (Gerber0(), Gerber2())
            ce0 = GerberCovariance(; me = CustomValueExpectedReturns(), alg = alg, t = 0.5)
            for kind in
                (BasicGerberIQ(; n = 1.0, d = 0.5), PartialGerberIQ(; dcp = 0.5, n1 = 1.0),
                 FullGerberIQ(; dcp = 0.5, n1 = 1.0, n4 = 1.0))
                ce1 = GerberIQCovariance(; me = CustomValueExpectedReturns(), c = 0.5,
                                         kind = kind, y = 0, alg = alg,
                                         sc = AssetVolatilityGerberIQScaler())
                res = isapprox(cor(ce0, rd.X), cor(ce1, rd.X))
                if !res
                    println("GerberIQ failed")
                    println(alg)
                    println(kind)
                end
                @test res
            end
        end
    end
    @testset "Regression" begin
        res = [StepwiseRegression(; alg = Forward()),
               StepwiseRegression(; alg = Forward(), crit = AIC()),
               StepwiseRegression(; alg = Forward(), crit = AICC()),
               StepwiseRegression(; alg = Forward(), crit = BIC()),
               StepwiseRegression(; alg = Forward(), crit = RSquared()),
               StepwiseRegression(; alg = Forward(), crit = AdjustedRSquared()),
               StepwiseRegression(; alg = Backward()),
               StepwiseRegression(; alg = Backward(), crit = AIC()),
               StepwiseRegression(; alg = Backward(), crit = AICC()),
               StepwiseRegression(; alg = Backward(), crit = BIC()),
               StepwiseRegression(; alg = Backward(), crit = RSquared()),
               StepwiseRegression(; alg = Backward(), crit = AdjustedRSquared()),
               DimensionReductionRegression(),
               DimensionReductionRegression(; retgt = GeneralisedLinearModel(;)),
               DimensionReductionRegression(; drtgt = PPCA()),
               StepwiseRegression(; crit = PValue(; t = 1e-15)),
               StepwiseRegression(; crit = PValue(; t = 1e-15), alg = Backward())]
        df = CSV.read(joinpath(@__DIR__, "./assets/Regression.csv.gz"), DataFrame)
        for (i, re) in pairs(res)
            rr = regression(re, rd)
            if i == 15
                continue
            end
            lt = [rr.b; vec(rr.M)]
            success = isapprox(lt, df[!, "$i"])
            if !success
                println("Counter: $i")
                find_tol(lt, df[!, "$i"])
            end
            @test success
            res = rr.M === rr.L
            isa(re, StepwiseRegression) ? (@test res) : (@test !res)
        end
    end
    @testset "Coskewness" begin
        skes = [Coskewness(; alg = Full()), Coskewness(; alg = Semi())]
        df = CSV.read(joinpath(@__DIR__, "./assets/coskewness.csv.gz"), DataFrame)
        for (i, ske) in pairs(skes)
            sk, v = coskewness(ske, rd.X'; dims = 2)
            success = isapprox([vec(sk); vec(v)], df[!, i])
            if !success
                println("Counter: $i")
                find_tol([vec(sk); vec(v)], df[!, i])
            end
            @test success
        end
        @test (nothing, nothing) === coskewness(nothing)
        sk0 = Coskewness(; alg = Semi())
        sk = PortfolioOptimisers.factory(sk0, ew)
        @test sk.me.w === ew
        @test sk.alg === sk0.alg

        ske0 = factory(WindowedCoskewness(; window = 50), ew)
        ske = factory(ske0.ske, ew[(end - 49):end])
        sk0, V0 = coskewness(ske0, rd.X[(end - 49):end, :])
        sk, V = coskewness(ske, rd.X[(end - 49):end, :])
        @test isapprox(sk0, sk)
        @test isapprox(V0, V)
    end
    @testset "Cokurtosis" begin
        ktes = [Cokurtosis(; alg = Full()), Cokurtosis(; alg = Semi())]
        df = CSV.read(joinpath(@__DIR__, "./assets/cokurtosis.csv.gz"), DataFrame)
        for (i, kte) in pairs(ktes)
            kt = cokurtosis(kte, rd.X'; dims = 2)
            success = isapprox(vec(kt), df[!, i])
            if !success
                println("Counter: $i")
                find_tol(vec(kt), df[!, i])
            end
            @test success
        end
        @test isnothing(cokurtosis(nothing))
        kt0 = Cokurtosis(; alg = Semi())
        kt = PortfolioOptimisers.factory(kt0, ew)
        @test kt.me.w === ew
        @test kt.alg === kt0.alg

        kte0 = factory(WindowedCokurtosis(; window = 50), ew)
        kte = factory(kte0.ke, ew[(end - 49):end])
        @test isapprox(cokurtosis(kte0, rd.X[(end - 49):end, :]),
                       cokurtosis(kte, rd.X[(end - 49):end, :]))
    end
    @testset "Distance" begin
        des = [Distance(; alg = SimpleAbsoluteDistance()),
               DistanceDistance(; alg = SimpleAbsoluteDistance()),
               Distance(; alg = LogDistance()), DistanceDistance(; alg = LogDistance())]
        desg = [Distance(; power = 1, alg = SimpleAbsoluteDistance()),
                DistanceDistance(; power = 1, alg = SimpleAbsoluteDistance()),
                Distance(; power = 1, alg = LogDistance()),
                DistanceDistance(; power = 1, alg = LogDistance())]
        df = CSV.read(joinpath(@__DIR__, "./assets/distance1.csv.gz"), DataFrame)
        ce = PortfolioOptimisersCovariance()
        for (i, (de, deg)) in enumerate(zip(des, desg))
            d1 = distance(de, ce, rd.X'; dims = 2)
            dg1 = distance(deg, ce, rd.X'; dims = 2)
            d2 = distance(de, cov(ce, rd.X), rd.X)
            dg2 = distance(deg, cov(ce, rd.X), rd.X)
            success = isapprox(vec(d1), df[!, i])
            if !success
                println("Counter: $i")
                find_tol(vec(d1), df[!, i])
            end
            @test success
            @test isapprox(d1, dg1)
            @test isapprox(d2, dg2)
            @test isapprox(d1, d2)
            if isa(de, Distance{<:Any, <:SimpleAbsoluteDistance}) ||
               isa(de, Distance{<:Any, <:LogDistance})
                r, d = cor_and_dist(de, ce, rd.X)
                rg, dg = cor_and_dist(deg, ce, rd.X)
                @test isapprox(r, rg)
                @test isapprox(d, dg)
            end
        end
    end
    @testset "Canonical Distance" begin
        ces = [Covariance(; alg = Full()), SpearmanCovariance(), KendallCovariance(),
               MutualInfoCovariance(), DistanceCovariance(),
               LowerTailDependenceCovariance(),
               GerberCovariance(; me = CustomValueExpectedReturns()),
               SmythBrobyCovariance(), MutualInfoCovariance(; bins = 3)]
        df = CSV.read(joinpath(@__DIR__, "./assets/CanonicalDistance.csv.gz"), DataFrame)
        de = Distance(; alg = CanonicalDistance())
        deg = Distance(; power = 1, alg = CanonicalDistance())
        for (i, ce) in pairs(ces)
            cei = PortfolioOptimisersCovariance(; ce = ce)
            r1, d1 = cor_and_dist(de, cei, rd.X'; dims = 2)
            d2 = cor_and_dist(deg, cei, rd.X'; dims = 2)[2]
            @test isapprox(d1, d2)
            @test isapprox(r1, cor(cei, rd.X))
            d3 = distance(de, cei, rd.X'; dims = 2)
            d4 = if isa(ce, MutualInfoCovariance)
                @test all(isapprox.((r1, d1), cor_and_dist(de, ce, rd.X)))
                @test isapprox(d1, distance(de, ce, rd.X'; dims = 2))
                @test isapprox(d1, distance(de, cei, rd.X'; dims = 2))
                distance(Distance(;
                                  alg = VariationInfoDistance(; bins = ce.bins,
                                                              normalise = ce.normalise)),
                         cov(ce, rd.X'; dims = 2), rd.X'; dims = 2)
            elseif isa(ce, DistanceCovariance)
                distance(Distance(; alg = CorrelationDistance(;)), cov(cei, rd.X), rd.X)
            elseif isa(ce, LowerTailDependenceCovariance)
                distance(Distance(; alg = LogDistance()), cov(cei, rd.X), rd.X)
            else
                distance(de, cov(cei, rd.X), rd.X)
            end
            d5 = distance(deg, ce, rd.X)
            d6 = if isa(ce, MutualInfoCovariance)
                @test all(isapprox.((r1, d1), cor_and_dist(deg, ce, rd.X)))
                @test isapprox(d1, distance(deg, ce, rd.X))
                @test isapprox(d1, distance(deg, cei, rd.X))
                distance(Distance(; power = 1,
                                  alg = VariationInfoDistance(; bins = ce.bins,
                                                              normalise = ce.normalise)),
                         cov(ce, rd.X'; dims = 2), rd.X'; dims = 2)
            elseif isa(ce, DistanceCovariance)
                distance(Distance(; power = 1, alg = CorrelationDistance(;)), cov(ce, rd.X),
                         rd.X)
            elseif isa(ce, LowerTailDependenceCovariance)
                distance(Distance(; power = 1, alg = LogDistance()), cov(ce, rd.X), rd.X)
            else
                distance(deg, cov(ce, rd.X), rd.X)
            end
            success = isapprox(vec(d1), df[!, i])
            if !success
                println("Counter: $i")
                find_tol(vec(d1), df[!, i])
            end
            @test success
            @test isapprox(d1, d2)
            @test isapprox(d3, d1)
            @test isapprox(d4, d1)
            @test isapprox(d5, d1)
            @test isapprox(d6, d1)
        end

        df = CSV.read(joinpath(@__DIR__, "./assets/CanonicalDistanceDistance.csv.gz"),
                      DataFrame)
        de = DistanceDistance(; alg = CanonicalDistance())
        deg = DistanceDistance(; power = 1, alg = CanonicalDistance())
        for (i, ce) in pairs(ces)
            cei = PortfolioOptimisersCovariance(; ce = ce)
            r1, d1 = cor_and_dist(de, cei, rd.X'; dims = 2)
            d2 = cor_and_dist(deg, cei, rd.X'; dims = 2)[2]
            @test isapprox(d1, d2)
            @test isapprox(r1, cor(cei, rd.X))
            d3 = distance(de, cei, rd.X)
            d4 = if isa(ce, MutualInfoCovariance)
                distance(DistanceDistance(;
                                          alg = VariationInfoDistance(; bins = ce.bins,
                                                                      normalise = ce.normalise)),
                         cov(ce, rd.X'; dims = 2), rd.X'; dims = 2)
            elseif isa(ce, DistanceCovariance)
                distance(DistanceDistance(; alg = CorrelationDistance(;)), cov(cei, rd.X),
                         rd.X)
            elseif isa(ce, LowerTailDependenceCovariance)
                distance(DistanceDistance(; alg = LogDistance()), cov(cei, rd.X), rd.X)
            else
                distance(de, cov(cei, rd.X), rd.X)
            end
            d5 = distance(deg, ce, rd.X)
            d6 = if isa(ce, MutualInfoCovariance)
                distance(DistanceDistance(; power = 1,
                                          alg = VariationInfoDistance(; bins = ce.bins,
                                                                      normalise = ce.normalise)),
                         cov(ce, rd.X), rd.X)
            elseif isa(ce, DistanceCovariance)
                distance(DistanceDistance(; power = 1, alg = CorrelationDistance(;)),
                         cov(ce, rd.X), rd.X)
            elseif isa(ce, LowerTailDependenceCovariance)
                distance(DistanceDistance(; power = 1, alg = LogDistance()), cov(ce, rd.X),
                         rd.X)
            else
                distance(deg, cov(ce, rd.X), rd.X)
            end
            success = isapprox(vec(d1), df[!, i])
            if !success
                println("Counter: $i")
                find_tol(vec(d1), df[!, i])
            end
            @test success
            @test isapprox(d1, d2)
            @test isapprox(d3, d1)
            @test isapprox(d4, d1)
            @test isapprox(d5, d1)
            @test isapprox(d6, d1)
        end
    end
end
