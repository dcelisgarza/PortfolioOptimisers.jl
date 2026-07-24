@testset "Moments" begin
    using Test, PortfolioOptimisers, DataFrames, TimeSeries, CSV, CovarianceEstimation,
          StableRNGs, StatsBase, Statistics, LinearAlgebra, SparseArrays, Distributions,
          FLoops
    rng = StableRNG(123456789)
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end],
                           TimeArray(CSV.File(joinpath(@__DIR__, "./assets/Factors.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    ew = eweights(1:size(rd.X, 1), inv(size(rd.X, 1)); scale = true)
    fw = fweights(rand(rng, size(rd.X, 1)))
    pw = pweights(fill(inv(size(rd.X, 1)), size(rd.X, 1)))
    rf = 4.34 / 100 / size(rd.X, 1)
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

        me0 = VarianceExpectedReturns()
        @test isapprox(mean(me0, rd.X), var(me0.ce, rd.X))
        @test isapprox(mean(me0, rd.X), std(me0.ce, rd.X) .^ 2)
        me = PortfolioOptimisers.factory(VarianceExpectedReturns(), ew)
        @test me.ce.ce.me.w === ew
        @test me.ce.ce.ce.w === ew

        me0 = ShrunkExpectedReturns(;
                                    ce = PortfolioOptimisersCovariance(;
                                                                       ce = Covariance(;
                                                                                       alg = SemiMoment())),
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
                                                                                            alg = SemiMoment())),
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

        me0 = factory(WindowedExpectedReturns(; window = 1:50,
                                              me = MedianExpectedReturns()), ew)
        me = factory(me0.me, ew[1:50])
        @test mean(me0, rd.X) ==
              reshape(mean(me, rd.X[1:50, :]'; dims = 2), 1, :) ==
              reduce(hcat, [median(Xi, ew[1:50]) for Xi in eachcol(rd.X[1:50, :])])

        @test mean(MedianExpectedReturns(), rd.X) == median(rd.X; dims = 1)
    end
    @testset "Covariance Estimators" begin
        ces = [Covariance(; alg = FullMoment()),
               Covariance(; alg = FullMoment(), me = SimpleExpectedReturns(; w = ew),
                          ce = GeneralCovariance(;
                                                 ce = SimpleCovariance(; corrected = false),
                                                 w = ew)),
               Covariance(; alg = FullMoment(),
                          ce = GeneralCovariance(; ce = AnalyticalNonlinearShrinkage())),
               Covariance(; alg = SemiMoment()),
               Covariance(; alg = SemiMoment(), me = SimpleExpectedReturns(; w = ew),
                          ce = GeneralCovariance(;
                                                 ce = SimpleCovariance(; corrected = false),
                                                 w = ew)),
               Covariance(; alg = SemiMoment(), me = SimpleExpectedReturns(; w = fw),
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
               GerberIQCovariance(; kind = FullGerberIQ(), alg = Gerber1(),
                                  decay = ExpGerberIQDecay(; y = 1 / 252, e = 110),
                                  sc = (x, y) -> (min(x, y), min(x, y))),
               GerberIQCovariance(; kind = PartialGerberIQ(), alg = Gerber2(),
                                  decay = ExpGerberIQDecay(;
                                                           y = (x) -> inv(div(size(x, 2),
                                                                              2)),
                                                           e = (x) -> inv(div(size(x, 2),
                                                                              3)) +
                                                                      inv(div(size(x, 2),
                                                                              5))),
                                  sc = (x, y) -> (min(x, y), min(x, y)))]
        df = CSV.read(joinpath(@__DIR__, "./assets/covariance.csv.gz"), DataFrame)
        for (i, ce) in pairs(ces)
            cei = PortfolioOptimisersCovariance(; ce = ce)
            sigma = cov(cei, rd.X'; dims = 2)
            rho = cor(cei, rd.X'; dims = 2)
            @test isapprox(StatsBase.cov2cor(sigma), rho)
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
                                                             mp = MatrixProcessing(;
                                                                                   dn = Denoise(;
                                                                                                alg = SpectralDenoise()))),
                               rd.X)))
        @test isapprox(df[!, 37],
                       vec(cov(PortfolioOptimisersCovariance(;
                                                             mp = MatrixProcessing(;
                                                                                   dt = Detone())),
                               rd.X)))
        @test isapprox(df[!, 37],
                       vec(cov(PortfolioOptimisersCovariance(;
                                                             mp = MatrixProcessing(;
                                                                                   order = (:pdm,
                                                                                            :dn,
                                                                                            :alg,
                                                                                            :dt),
                                                                                   dt = Detone())),
                               rd.X)))
        @test isapprox(df[!, 37],
                       vec(cov(PortfolioOptimisersCovariance(;
                                                             mp = MatrixProcessing(;
                                                                                   order = (:pdm,
                                                                                            :dt,
                                                                                            :dn,
                                                                                            :alg),
                                                                                   dt = Detone())),
                               rd.X)))
        @test isapprox(df[!, 37],
                       vec(cov(PortfolioOptimisersCovariance(;
                                                             mp = MatrixProcessing(;
                                                                                   order = (:pdm,
                                                                                            :dt,
                                                                                            :alg,
                                                                                            :dn),
                                                                                   dt = Detone())),
                               rd.X)))
        @test isapprox(df[!, 37],
                       vec(cov(PortfolioOptimisersCovariance(;
                                                             mp = MatrixProcessing(;
                                                                                   order = (:pdm,
                                                                                            :alg,
                                                                                            :dn,
                                                                                            :dt),
                                                                                   dt = Detone())),
                               rd.X)))
        @test isapprox(df[!, 37],
                       vec(cov(PortfolioOptimisersCovariance(;
                                                             mp = MatrixProcessing(;
                                                                                   order = (:pdm,
                                                                                            :alg,
                                                                                            :dt,
                                                                                            :dn),
                                                                                   dt = Detone())),
                               rd.X)))
        @test isapprox(df[!, 38],
                       vec(cov(PortfolioOptimisersCovariance(;
                                                             mp = MatrixProcessing(;
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
        # Regression: the threaded pair-count loop used to fill the tail mask
        # lazily per iteration, so with >1 thread iteration j could read the
        # not-yet-written mask of column i < j. Threaded and sequential
        # executors must agree exactly, and repeated runs must be identical.
        let ltd_seq = cor(LowerTailDependenceCovariance(; ex = FLoops.SequentialEx()), rd.X)
            for _ in 1:5
                @test cor(LowerTailDependenceCovariance(; ex = FLoops.ThreadedEx()),
                          rd.X) == ltd_seq
            end
        end

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
        @test ce.ce.mp.dn == ce0.ce.mp.dn
        @test ce.ce.mp.pdm == ce0.ce.mp.pdm
        @test isapprox(cov(DenoiseCovariance(), rd.X),
                       cov(DenoiseCovariance(), rd.X'; dims = 2))
        @test isapprox(cor(DenoiseCovariance(), rd.X),
                       cor(DenoiseCovariance(), rd.X'; dims = 2))

        ce0 = PortfolioOptimisersCovariance(; ce = DetoneCovariance())
        ce = PortfolioOptimisers.factory(ce0, ew)
        @test ce.ce.ce.ce.w === ew
        @test ce.ce.ce.me.w === ew
        @test ce.ce.mp.dt == ce0.ce.mp.dt
        @test ce.ce.mp.pdm == ce0.ce.mp.pdm
        @test isapprox(cov(DetoneCovariance(), rd.X),
                       cov(DetoneCovariance(), rd.X'; dims = 2))
        @test isapprox(cor(DetoneCovariance(), rd.X),
                       cor(DetoneCovariance(), rd.X'; dims = 2))

        ce0 = PortfolioOptimisersCovariance(; ce = ProcessedCovariance(; alg = LoGo()))
        ce = PortfolioOptimisers.factory(ce0, ew)
        @test ce.ce.ce.ce.w === ew
        @test ce.ce.ce.me.w === ew
        @test ce.ce.mp.alg == ce0.ce.mp.alg
        @test ce.ce.mp.pdm == ce0.ce.mp.pdm
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

        ce0 = factory(WindowedVariance(; ve = SimpleVariance(; corrected = false),
                                       window = 50), ew)
        ce = factory(ce0.ve, ew[(end - 49):end])
        @test isapprox(var(ce0, rd.X[(end - 49):end, :]), var(ce, rd.X[(end - 49):end, :]))
        @test isapprox(std(ce0, rd.X[(end - 49):end, :]), std(ce, rd.X[(end - 49):end, :]))
        @test isapprox(var(ce0, rd.X[(end - 49):end, 1]), var(ce, rd.X[(end - 49):end, 1]))
        @test isapprox(std(ce0, rd.X[(end - 49):end, 2]), std(ce, rd.X[(end - 49):end, 2]))

        ce0 = factory(WindowedVariance(; ve = SimpleVariance(; corrected = false)), ew)
        ce = factory(ce0.ve, ew)
        @test isapprox(var(ce0, rd.X), var(ce, rd.X))
        @test isapprox(std(ce0, rd.X), std(ce, rd.X))
        @test isapprox(var(ce0, rd.X[:, 1]), var(ce, rd.X[:, 1]))
        @test isapprox(std(ce0, rd.X[:, 2]), std(ce, rd.X[:, 2]))

        ce0 = factory(WindowedVariance(;
                                       ve = SimpleVariance(; me = nothing,
                                                           corrected = false), window = 50),
                      ew)
        ce = factory(ce0.ve, ew[(end - 49):end])
        @test isapprox(var(ce0, rd.X[(end - 49):end, :]; mean = zeros(1, size(rd.X, 2))),
                       var(ce, rd.X[(end - 49):end, :]; mean = zeros(1, size(rd.X, 2))))
        @test isapprox(std(ce0, rd.X[(end - 49):end, :]; mean = zeros(1, size(rd.X, 2))),
                       std(ce, rd.X[(end - 49):end, :]; mean = zeros(1, size(rd.X, 2))))
        @test isapprox(var(ce0, rd.X[(end - 49):end, 1]; mean = 0),
                       var(ce, rd.X[(end - 49):end, 1]; mean = 0))
        @test isapprox(std(ce0, rd.X[(end - 49):end, 2]; mean = 0),
                       std(ce, rd.X[(end - 49):end, 2]; mean = 0))

        ce0 = factory(WindowedVariance(;
                                       ve = SimpleVariance(; me = nothing,
                                                           corrected = false)), ew)
        ce = factory(ce0.ve, ew)
        @test isapprox(var(ce0, rd.X; mean = zeros(1, size(rd.X, 2))),
                       var(ce, rd.X; mean = zeros(1, size(rd.X, 2))))
        @test isapprox(std(ce0, rd.X; mean = zeros(1, size(rd.X, 2))),
                       std(ce, rd.X; mean = zeros(1, size(rd.X, 2))))
        @test isapprox(var(ce0, rd.X[:, 1]; mean = 0), var(ce, rd.X[:, 1]; mean = 0))
        @test isapprox(std(ce0, rd.X[:, 2]; mean = 0), std(ce, rd.X[:, 2]; mean = 0))

        @test PortfolioOptimisers.find_uncorrelated_indices(rd.X; t = 0.5) ==
              [4, 6, 12, 16, 17, 19]

        for alg in (Gerber0(), Gerber2())
            ce0 = GerberCovariance(; me = CustomValueExpectedReturns(), alg = alg, t = 0.5)
            for kind in
                (BasicGerberIQ(; n = 1.0, d = 0.5), PartialGerberIQ(; dcp = 0.5, n1 = 1.0),
                 FullGerberIQ(; dp1 = 0.5, n1 = 1.0, n4 = 1.0))
                ce1 = GerberIQCovariance(; me = CustomValueExpectedReturns(), c = 0.5,
                                         kind = kind, decay = ExpGerberIQDecay(; y = 0),
                                         alg = alg, sc = AssetVolatilityGerberIQScaler())
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
    @testset "GerberIQ region templates" begin
        # gerber_iq_weight maps a co-movement (r_i, r_j) to its region's squeezing weight.
        # Sentinel weights n_k = k/100 (and distinct thresholds) make every region identifiable,
        # so these grids verify the templates against the paper's Figures 1(b)/2/3. The default
        # FullGerberIQ has all thresholds equal (empty moderate band), which masks a negative-side
        # dn1/dn2 mix-up; distinct dn1 != dn2 below is what exercises it.
        giqw(ri, rj, kind) = PortfolioOptimisers.gerber_iq_weight(ri, rj, abs(ri), abs(rj),
                                                                  1.0, 1.0, kind)
        # BasicGerberIQ: both >= d -> 1; both < d -> n; mixed -> n^2 (magnitude-only, symmetric).
        basic = BasicGerberIQ(; d = 3.0, n = 0.5)
        for (ri, rj, e) in
            ((3.5, 3.5, 1.0), (-3.5, -3.5, 1.0), (3.5, -3.5, 1.0), (1.5, 1.5, 0.5),
             (-1.5, -1.5, 0.5), (2.5, 0.5, 0.5), (3.5, 1.5, 0.25), (1.5, 3.5, 0.25),
             (-3.5, 1.5, 0.25), (0.5, -3.5, 0.25))
            @test giqw(ri, rj, basic) ≈ e
        end
        # PartialGerberIQ: one representative point per region (Figure 1(b)); check symmetry too.
        partial = PartialGerberIQ(; dcp = 2.0, dcn = 3.0, ddp = 3.0, ddn = 2.0,
                                  Dict(Symbol("n$k") => k / 100 for k in 1:10)...)
        for (pt, k) in
            (((2.5, 2.5), 4), ((2.5, 1.0), 7), ((1.0, 1.0), 1), ((-3.5, -3.5), 5),
             ((-3.5, -1.0), 8), ((-1.5, -1.5), 2), ((3.5, -2.5), 6), ((3.5, -1.0), 9),
             ((1.0, -2.5), 10), ((1.0, -1.0), 3))
            @test giqw(pt[1], pt[2], partial) ≈ k / 100
            @test giqw(pt[2], pt[1], partial) ≈ k / 100
        end
        # FullGerberIQ: the full Figure 2 region grid (rows r_j high->low, cols r_i low->high).
        full = FullGerberIQ(; dp1 = 3.0, dp2 = 2.0, dn1 = 3.0, dn2 = 2.0,
                            Dict(Symbol("n$k") => k / 100 for k in 1:21)...)
        rj_rows = (3.5, 2.5, 1.5, -1.5, -2.5, -3.5)
        ri_cols = (-3.5, -2.5, -1.5, 1.5, 2.5, 3.5)
        fig2 = [13 19 18 15 14 11
                20 6 9 7 4 14
                21 10 3 1 7 15
                16 8 2 3 9 18
                17 5 8 10 6 19
                12 17 16 21 20 13]
        for (r, rj) in enumerate(rj_rows), (c, ri) in enumerate(ri_cols)
            @test round(Int, 100 * giqw(ri, rj, full)) == fig2[r, c]
        end
        # Every region reachable (guards the negative moderate-band dn1/dn2 bug) and symmetric.
        grid = [round(Int, 100 * giqw(ri, rj, full))
                for ri in range(-3.8, 3.8; length = 97),
                    rj in range(-3.8, 3.8; length = 97)]
        @test isempty(setdiff(1:21, unique(grid)))
        @test all(giqw(a, b, full) ≈ giqw(b, a, full)
                  for a in range(-4, 4; length = 31), b in range(-4, 4; length = 31))
    end
    @testset "Regression" begin
        res = [StepwiseRegression(; alg = ForwardSelection()),
               StepwiseRegression(; alg = ForwardSelection(), crit = AIC()),
               StepwiseRegression(; alg = ForwardSelection(), crit = AICC()),
               StepwiseRegression(; alg = ForwardSelection(), crit = BIC()),
               StepwiseRegression(; alg = ForwardSelection(), crit = RSquared()),
               StepwiseRegression(; alg = ForwardSelection(), crit = AdjustedRSquared()),
               StepwiseRegression(; alg = BackwardElimination()),
               StepwiseRegression(; alg = BackwardElimination(), crit = AIC()),
               StepwiseRegression(; alg = BackwardElimination(), crit = AICC()),
               StepwiseRegression(; alg = BackwardElimination(), crit = BIC()),
               StepwiseRegression(; alg = BackwardElimination(), crit = RSquared()),
               StepwiseRegression(; alg = BackwardElimination(), crit = AdjustedRSquared()),
               DimensionReductionRegression(),
               DimensionReductionRegression(; retgt = GeneralisedLinearModel(;)),
               DimensionReductionRegression(; drtgt = PPCA()),
               StepwiseRegression(; crit = PValue(; t = 1e-15)),
               StepwiseRegression(; crit = PValue(; t = 1e-15),
                                  alg = BackwardElimination())]
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
        skes = [Coskewness(; alg = FullMoment()), Coskewness(; alg = SemiMoment())]
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
        sk0 = Coskewness(; alg = SemiMoment())
        sk = PortfolioOptimisers.factory(sk0, ew)
        @test sk.me.w === ew
        @test sk.alg === sk0.alg

        sk, v = coskewness(factory(Coskewness(), pw), rd.X)
        @test isapprox([vec(sk); vec(v)], df[!, 1])

        ske0 = factory(WindowedCoskewness(; window = 50), ew)
        ske = factory(ske0.ske, ew[(end - 49):end])
        sk0, V0 = coskewness(ske0, rd.X[(end - 49):end, :])
        sk, V = coskewness(ske, rd.X[(end - 49):end, :])
        @test isapprox(sk0, sk)
        @test isapprox(V0, V)
    end
    @testset "Cokurtosis" begin
        ktes = [Cokurtosis(; alg = FullMoment()), Cokurtosis(; alg = SemiMoment())]
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
        kt0 = Cokurtosis(; alg = SemiMoment())
        kt = PortfolioOptimisers.factory(kt0, ew)
        @test kt.me.w === ew
        @test kt.alg === kt0.alg

        kt = cokurtosis(factory(Cokurtosis(), pw), rd.X)
        @test isapprox(vec(kt), df[!, 1])

        kte0 = factory(WindowedCokurtosis(; window = 50), ew)
        kte = factory(kte0.kte, ew[(end - 49):end])
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
    @testset "Distance validation and kernel" begin
        # Footgun fix: a non-square rho now throws for EVERY correlation-based
        # algorithm, not only CorrelationDistance.
        nonsquare = rand(3, 4)
        for alg in (SimpleDistance(), SimpleAbsoluteDistance(), LogDistance(),
                    CorrelationDistance())
            @test_throws DimensionMismatch distance(Distance(; alg = alg), nonsquare)
            @test_throws DimensionMismatch distance(Distance(; power = 1, alg = alg),
                                                    nonsquare)
        end
        # Kernel exercised directly through the matrix entry point on a hand-built
        # correlation matrix. Includes a negative entry to hit the abs-guard.
        rho = [1.0 0.5 -0.5
               0.5 1.0 0.0
               -0.5 0.0 1.0]
        # SimpleDistance: sqrt((1 - rho) / 2)
        @test distance(Distance(; alg = SimpleDistance()), rho) ≈ sqrt.((1 .- rho) ./ 2)
        # CorrelationDistance: sqrt(clamp(1 - rho, 0, 1)) — the -0.5 entry clamps to 1.
        @test distance(Distance(; alg = CorrelationDistance()), rho) ≈
              sqrt.(clamp.(1 .- rho, 0, 1))
        # SimpleAbsoluteDistance: abs-guard folds the sign, then sqrt(1 - |rho|).
        @test distance(Distance(; alg = SimpleAbsoluteDistance()), rho) ≈
              sqrt.(1 .- abs.(rho))
        # LogDistance: -log(|rho|); the off-diagonal 0.5 gives -log(0.5).
        @test distance(Distance(; alg = LogDistance()), rho) ≈ -log.(abs.(rho))
        # Power path: SimpleDistance p=2 is even, scale = 1.
        @test distance(Distance(; power = 2, alg = SimpleDistance()), rho) ≈
              sqrt.(clamp.(1 .- rho .^ 2, 0, 1))
        # A covariance matrix (diagonal ≠ 1) is coerced to correlation, and must
        # still be square.
        cov = [4.0 1.0; 1.0 9.0]
        @test distance(Distance(; alg = CorrelationDistance()), cov) ≈
              distance(Distance(; alg = CorrelationDistance()), [1.0 1/6; 1/6 1.0])
        @test_throws DimensionMismatch distance(Distance(; alg = SimpleDistance()),
                                                rand(2, 3))
    end
    @testset "Canonical Distance" begin
        ces = [Covariance(; alg = FullMoment()), SpearmanCovariance(), KendallCovariance(),
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
"""
Records the shape of the `iv` its `cov`/`cor` receive, so a windowed wrapper's `iv`
subsetting can be asserted without depending on a real implied-volatility estimator.
Returns `[size(iv, 1) size(iv, 2); 1 1]`, or a zero matrix when `iv` is `nothing`.
"""
struct IVProbe <: PortfolioOptimisers.AbstractCovarianceEstimator end
function iv_probe_shape(iv)
    return isnothing(iv) ? [0 0; 0 0] : [size(iv, 1) size(iv, 2); 1 1]
end
function Statistics.cov(::IVProbe, ::PortfolioOptimisers.MatNum; iv = nothing, kwargs...)
    return iv_probe_shape(iv)
end
function Statistics.cor(::IVProbe, ::PortfolioOptimisers.MatNum; iv = nothing, kwargs...)
    return iv_probe_shape(iv)
end
@testset "Windowed estimator family" begin
    using Test, PortfolioOptimisers, DataFrames, TimeSeries, CSV, StatsBase, Statistics
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end],
                           TimeArray(CSV.File(joinpath(@__DIR__, "./assets/Factors.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    ew = eweights(1:size(rd.X, 1), inv(size(rd.X, 1)); scale = true)
    win = 1:50

    # Every member of the family carries the same shape: the inner estimator under its
    # conventional field name, `w`, `window` — and nothing else.
    @test propertynames(WindowedExpectedReturns()) == (:me, :w, :window)
    @test propertynames(WindowedCovariance()) == (:ce, :w, :window)
    @test propertynames(WindowedVariance()) == (:ve, :w, :window)
    @test propertynames(WindowedCoskewness()) == (:ske, :w, :window)
    @test propertynames(WindowedCokurtosis()) == (:kte, :w, :window)

    # Each answers a different generic, so each must keep its own supertype (ADR 0039).
    @test WindowedExpectedReturns() isa PortfolioOptimisers.AbstractExpectedReturnsEstimator
    @test WindowedCovariance() isa PortfolioOptimisers.AbstractCovarianceEstimator
    @test WindowedVariance() isa PortfolioOptimisers.AbstractVarianceEstimator
    @test WindowedCoskewness() isa PortfolioOptimisers.CoskewnessEstimator
    @test WindowedCokurtosis() isa PortfolioOptimisers.CokurtosisEstimator

    # `factory` propagates weights into the inner estimator and replaces `w`; `window`
    # passes through untouched. Uniform across the family.
    for (w0, inner) in ((WindowedExpectedReturns(; window = 50), :me),
                        (WindowedCovariance(; window = 50), :ce), (WindowedVariance(; window = 50), :ve),
                        (WindowedCoskewness(; window = 50), :ske),
                        (WindowedCokurtosis(; window = 50), :kte))
        w1 = factory(w0, ew)
        @test w1.w === ew
        @test w1.window == w0.window
        @test getproperty(w1, inner) !== getproperty(w0, inner)
    end

    # `mean` is a *named* keyword on every forwarder except `Statistics.mean`, so it
    # reaches the inner estimator instead of riding in `kwargs...` into
    # `windowed_preamble` (ADR 0039).
    for f in (Statistics.cov, Statistics.cor)
        @test :mean in Base.kwarg_decl(only(methods(f,
                                                    (WindowedCovariance, PortfolioOptimisers.MatNum))))
    end
    @test :mean in Base.kwarg_decl(only(methods(coskewness,
                                                (WindowedCoskewness, PortfolioOptimisers.MatNum))))
    @test :mean in Base.kwarg_decl(only(methods(cokurtosis,
                                                (WindowedCokurtosis, PortfolioOptimisers.MatNum))))
    @test :mean ∉ Base.kwarg_decl(only(methods(Statistics.mean,
                                               (WindowedExpectedReturns, PortfolioOptimisers.MatNum))))

    # Passing `mean` explicitly must agree with letting the inner estimator compute it.
    mu = Statistics.mean(SimpleExpectedReturns(), rd.X[win, :])
    @test isapprox(cor(WindowedCovariance(; window = win), rd.X; mean = mu),
                   cor(WindowedCovariance(; window = win), rd.X))
    sk0, V0 = coskewness(WindowedCoskewness(; window = win), rd.X; mean = mu)
    sk1, V1 = coskewness(WindowedCoskewness(; window = win), rd.X)
    @test isapprox(sk0, sk1)
    @test isapprox(V0, V1)
    @test isapprox(cokurtosis(WindowedCokurtosis(; window = win), rd.X; mean = mu),
                   cokurtosis(WindowedCokurtosis(; window = win), rd.X))

    # `dims = 2` windows the transposed data identically to `dims = 1` on the original.
    Xt = permutedims(rd.X)
    @test isapprox(cov(WindowedCovariance(; window = win), rd.X),
                   cov(WindowedCovariance(; window = win), Xt; dims = 2))
    @test isapprox(vec(var(WindowedVariance(; window = win), rd.X)),
                   vec(var(WindowedVariance(; window = win), Xt; dims = 2)))
    @test isapprox(cokurtosis(WindowedCokurtosis(; window = win), rd.X),
                   cokurtosis(WindowedCokurtosis(; window = win), Xt; dims = 2))

    # An index `window` subsets `iv` to the same rows, so an estimator that consumes
    # implied volatilities sees the window's own, aligned with the windowed returns.
    iv = abs.(rd.X) .+ 0.1
    probe = WindowedCovariance(; ce = IVProbe(), window = win)
    @test cov(probe, rd.X; iv = iv) == [length(win) size(rd.X, 2); 1 1]
    @test cor(probe, rd.X; iv = iv) == [length(win) size(rd.X, 2); 1 1]
    # An Int window resolves to a range, which is also a VecInt, so it subsets `iv` too.
    @test cov(WindowedCovariance(; ce = IVProbe(), window = 50), rd.X; iv = iv) ==
          [50 size(rd.X, 2); 1 1]
    # Only `window = nothing` resolves to a Colon and leaves `iv` whole.
    @test cov(WindowedCovariance(; ce = IVProbe()), rd.X; iv = iv) ==
          [size(rd.X, 1) size(rd.X, 2); 1 1]
    # No `iv` at all still reaches the inner estimator as `nothing`.
    @test cov(probe, rd.X) == [0 0; 0 0]
end
# ---------------------------------------------------------------------------
# @windowed_estimator — the declaration that generates the family above (ADR 0039)
# ---------------------------------------------------------------------------
# A throwaway sixth family member, declared through the same macro the five shipped
# estimators use. Asserting against it keeps the checks on what the *macro* emits rather
# than on what any one shipped type happens to look like today.
module WindowedEstimatorProbe
using Statistics, StatsBase, PortfolioOptimisers
using PortfolioOptimisers: MatNum, VecNum, Option, Int_VecInt, ObsWeights,
                           AbstractVarianceEstimator, arg_dict, field_dict, ret_dict,
                           val_dict, assert_nonempty_nonneg_finite_val, factory_child,
                           windowed_preamble, _wprop, @concrete, @propagatable,
                           @windowed_estimator
import PortfolioOptimisers: factory, port_opt_view
const DocStringExtensions = PortfolioOptimisers.DocStringExtensions
@windowed_estimator ProbeWindowedVariance <: AbstractVarianceEstimator begin
    ve::AbstractVarianceEstimator = SimpleVariance()
    noun = "Variance"
    # `std` deliberately omits `mean`, so an undeclared keyword has no way through.
    forward = [Statistics.var(::MatNum; mean) => :vararr,
               Statistics.std(::VecNum) => :stdnum]
    doctest = """
    julia> 1 + 1
    2
    """
end
end
@testset "@windowed_estimator" begin
    using Test, PortfolioOptimisers, Statistics, StatsBase, StableRNGs
    msg_of(f) =
        try
            f()
            ""
        catch e
            sprint(showerror, e)
        end

    @testset "Declaration parsing" begin
        # The inner estimator line yields the field name, its declared type, and the
        # keyword-constructor default, all unevaluated.
        @test PortfolioOptimisers.windowed_parse_field(:(ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance())) ==
              (:ce, :(StatsBase.CovarianceEstimator), :(PortfolioOptimisersCovariance()))
        # It must be a `field::Type = default` line — nothing looser.
        @test_throws ArgumentError PortfolioOptimisers.windowed_parse_field(:(ve::AbstractVarianceEstimator))
        @test_throws ArgumentError PortfolioOptimisers.windowed_parse_field(:(ve = SimpleVariance()))
        # The field name is also the generated methods' argument name, so it must be a
        # bare symbol...
        @test_throws ArgumentError PortfolioOptimisers.windowed_parse_field(:(a.b::AbstractVarianceEstimator = SimpleVariance()))
        # ...and a `field_dict` key, or the generated field docstring would be empty.
        @test occursin("`vee` is not a `field_dict` key",
                       msg_of(() -> PortfolioOptimisers.windowed_parse_field(:(vee::AbstractVarianceEstimator = SimpleVariance()))))

        # A `forward` entry parses into the generic, its input type, whether it names
        # `mean`, and the `ret_dict` keys documenting the return values.
        @test PortfolioOptimisers.windowed_parse_forward(:(Statistics.cov(::MatNum; mean) =>
                                                               :sigma)) ==
              (:(Statistics.cov), :MatNum, true, [:sigma])
        @test PortfolioOptimisers.windowed_parse_forward(:(Statistics.std(::VecNum) =>
                                                               :stdnum)) ==
              (:(Statistics.std), :VecNum, false, [:stdnum])
        # A tuple return documents each of its values.
        @test PortfolioOptimisers.windowed_parse_forward(:(coskewness(::MatNum; mean) =>
                                                               (:cskew, :cskewV))) ==
              (:coskewness, :MatNum, true, [:cskew, :cskewV])
        # Not a pair, or a left-hand side that is not a call.
        @test_throws ArgumentError PortfolioOptimisers.windowed_parse_forward(:(Statistics.cov(::MatNum)))
        @test_throws ArgumentError PortfolioOptimisers.windowed_parse_forward(:(Statistics.cov =>
                                                                                    :sigma))
        # Exactly one positional type, and it must be one the macro can generate for.
        @test_throws ArgumentError PortfolioOptimisers.windowed_parse_forward(:(Statistics.cov(::MatNum,
                                                                                               ::VecNum) =>
                                                                                    :sigma))
        @test_throws ArgumentError PortfolioOptimisers.windowed_parse_forward(:(Statistics.cov(::AbstractMatrix) =>
                                                                                    :sigma))
        @test_throws ArgumentError PortfolioOptimisers.windowed_parse_forward(:(Statistics.cov(X::MatNum) =>
                                                                                    :sigma))
        # `mean` is the only keyword an entry may name; anything else would have to ride
        # in `kwargs...` and would leak into `windowed_preamble`.
        @test_throws ArgumentError PortfolioOptimisers.windowed_parse_forward(:(Statistics.cov(::MatNum;
                                                                                               dims) =>
                                                                                    :sigma))
        # Return keys are quoted symbols, and must name `ret_dict` entries.
        @test_throws ArgumentError PortfolioOptimisers.windowed_parse_forward(:(Statistics.cov(::MatNum) =>
                                                                                    sigma))
        @test occursin("`sigmaa` is not a `ret_dict` key",
                       msg_of(() -> PortfolioOptimisers.windowed_parse_forward(:(Statistics.cov(::MatNum) =>
                                                                                     :sigmaa))))

        # Every rejection carries the macro's name, so the error points at the declaration
        # rather than at the helper that raised it.
        @test occursin("@windowed_estimator: ",
                       msg_of(() -> PortfolioOptimisers.windowed_estimator_error("boom")))
        @test PortfolioOptimisers.windowed_estimator_check_key(:mu,
                                                               PortfolioOptimisers.ret_dict,
                                                               "ret_dict") === :mu
    end

    @testset "Mistyped-key suggestions" begin
        # The looser Damerau-Levenshtein/0.5 configuration is load-bearing: under the
        # library default a short key like `noun` never matches, so the suggestion would
        # be dead code (ADR 0026 keeps the strict default for asset-name probes).
        @test occursin("did you mean `noun`?",
                       PortfolioOptimisers.windowed_estimator_suggest(:nuon,
                                                                      PortfolioOptimisers.WINDOWED_ESTIMATOR_KEYS))
        @test isempty(PortfolioOptimisers.did_you_mean("nuon",
                                                       ["noun", "forward", "doctest"]))
        # A key too far from any candidate suggests nothing rather than guessing.
        @test isempty(PortfolioOptimisers.windowed_estimator_suggest(:zzzzzzz,
                                                                     PortfolioOptimisers.WINDOWED_ESTIMATOR_KEYS))
    end

    @testset "Generated docstrings and method bodies" begin
        dm = PortfolioOptimisers.windowed_method_doc(:(Statistics.cov), :ce,
                                                     :WindowedCovariance, :MatNum, true,
                                                     [:sigma], "Covariance",
                                                     ["[`sibling`](@ref)"])
        dv = PortfolioOptimisers.windowed_method_doc(:(Statistics.std), :ve,
                                                     :WindowedVariance, :VecNum, false,
                                                     [:stdnum], "Variance", String[])
        # Docstrings are interpolation ASTs, not strings: the dictionary lookups stay live
        # parts of the `DocStr`, exactly as a hand-written `$(arg_dict[:dims])` would.
        @test Meta.isexpr(dm, :string)
        @test filter(x -> !isa(x, String), dm.args) ==
              Any[:(arg_dict[:dims]), :(arg_dict[:oiv]), :(ret_dict[:sigma])]
        # The vector forwarder documents neither `dims` nor `iv` — it takes neither.
        @test filter(x -> !isa(x, String), dv.args) == Any[:(ret_dict[:stdnum])]
        # The summary names the generic, not the type's noun: `std` on a windowed variance
        # estimator computes a standard deviation.
        @test occursin("Compute `Statistics.std` over a rolling or indexed observation window",
                       join(filter(x -> isa(x, String), dv.args)))
        # Siblings cross-link, and a method never links to itself.
        @test occursin("[`sibling`](@ref)", join(filter(x -> isa(x, String), dm.args)))
        @test PortfolioOptimisers.windowed_method_ref(:(Statistics.cov), :ce,
                                                      :WindowedCovariance, :MatNum) ==
              "[`Statistics.cov(ce::WindowedCovariance, X::MatNum)`](@ref)"

        td = PortfolioOptimisers.windowed_type_doc(:WindowedCovariance,
                                                   :AbstractCovarianceEstimator, :ce,
                                                   :(StatsBase.CovarianceEstimator),
                                                   :(PortfolioOptimisersCovariance()),
                                                   "Covariance", "julia> 1 + 1\n2\n",
                                                   ["[`m`](@ref)"])
        @test Meta.isexpr(td, :string)
        # `TYPEDEF`/`FIELDS` must survive as abbreviations; a rendered string would freeze
        # the field list at macro-expansion time.
        @test filter(x -> !isa(x, String), td.args) ==
              Any[:(DocStringExtensions.TYPEDEF), :(DocStringExtensions.FIELDS),
                  :(val_dict[:oow])]

        # The matrix forwarder threads `dims`/`iv` through `windowed_preamble` and names
        # `mean` only when the entry declared it.
        defm = PortfolioOptimisers.windowed_method_def(:(Statistics.cov), :ce,
                                                       :WindowedCovariance, :MatNum, true)
        defv = PortfolioOptimisers.windowed_method_def(:(Statistics.std), :ve,
                                                       :WindowedVariance, :VecNum, false)
        @test Meta.isexpr(defm, :function)
        @test string(defm.args[1]) ==
              "Statistics.cov(ce::WindowedCovariance, X::MatNum; dims::Int = 1, mean = nothing, iv::Option{<:MatNum} = nothing, kwargs...)"
        @test occursin("windowed_preamble(ce.ce, ce.w, ce.window, X; iv = iv, dims = dims, kwargs...)",
                       string(defm.args[2]))
        # The vector forwarder takes the estimator and the data, and nothing else.
        @test string(defv.args[1]) == "Statistics.std(ve::WindowedVariance, X::VecNum; )"
        @test occursin("windowed_preamble(ve.ve, ve.w, ve.window, X)", string(defv.args[2]))
    end

    @testset "Expansion-time rejection" begin
        windowed_decl(head, body) = Expr(:macrocall, Symbol("@windowed_estimator"),
                                         LineNumberNode(@__LINE__), head, body)
        good_head = :(ProbeBad <: AbstractVarianceEstimator)
        good_body = quote
            ve::AbstractVarianceEstimator = SimpleVariance()
            noun = "Variance"
            forward = [Statistics.var(::MatNum; mean) => :vararr]
            doctest = "julia> 1 + 1\n2\n"
        end
        # Expanded in the probe module, where the names the macro emits resolve.
        expand(head, body) = macroexpand(WindowedEstimatorProbe, windowed_decl(head, body))
        bad_msg(head, body) = msg_of(() -> expand(head, body))

        # A well-formed declaration expands to the whole family member: the struct and its
        # constructors, the forwarders, and the export.
        ex = expand(good_head, good_body)
        @test Meta.isexpr(ex, :block)
        @test count(a -> Meta.isexpr(a, :export), ex.args) == 1
        @test ex.args[findfirst(a -> Meta.isexpr(a, :export), ex.args)].args == [:ProbeBad]
        @test occursin("windowed_preamble", string(ex))
        @test occursin("assert_nonempty_nonneg_finite_val", string(ex))

        # Header and body shape. The name must be a bare symbol: it is the type being
        # defined, so a dotted or parametric header is rejected along with a missing `<:`.
        @test occursin("the header must read `Name <: Super`",
                       bad_msg(:ProbeBad, good_body))
        @test occursin("the header must read `Name <: Super`",
                       bad_msg(:(A.B <: AbstractVarianceEstimator), good_body))
        @test occursin("the header must read `Name <: Super`",
                       bad_msg(:(ProbeBad{T} <: AbstractVarianceEstimator), good_body))
        @test occursin("must be a `begin ... end` block", bad_msg(good_head, :(1 + 1)))

        # Body lines are assignments, and exactly one of them declares the inner estimator.
        @test occursin("must be an assignment",
                       bad_msg(good_head,
                               quote
                                   ve::AbstractVarianceEstimator = SimpleVariance()
                                   noun
                               end))
        @test occursin("exactly one `field::Type = default` line",
                       bad_msg(good_head,
                               quote
                                   ve::AbstractVarianceEstimator = SimpleVariance()
                                   me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns()
                               end))

        # A mistyped key is rejected with a suggestion instead of silently producing a
        # malformed docstring or a missing forwarder.
        @test occursin("`nuon` is not a recognised key (did you mean `noun`?)",
                       bad_msg(good_head,
                               quote
                                   ve::AbstractVarianceEstimator = SimpleVariance()
                                   nuon = "Variance"
                               end))

        # Every key is required...
        @test occursin("missing required `noun` declaration",
                       bad_msg(good_head,
                               quote
                                   ve::AbstractVarianceEstimator = SimpleVariance()
                                   forward = [Statistics.var(::MatNum) => :vararr]
                                   doctest = "x"
                               end))
        @test occursin("missing required `field::Type = default` declaration",
                       bad_msg(good_head,
                               quote
                                   noun = "Variance"
                                   forward = [Statistics.var(::MatNum) => :vararr]
                                   doctest = "x"
                               end))
        # ...`noun`/`doctest` must be literals the macro can splice into prose...
        @test occursin("must be string literals",
                       bad_msg(good_head,
                               quote
                                   ve::AbstractVarianceEstimator = SimpleVariance()
                                   noun = string("Vari", "ance")
                                   forward = [Statistics.var(::MatNum) => :vararr]
                                   doctest = "x"
                               end))
        # ...and `forward` must be a non-empty vector, or the type would answer no generic.
        @test occursin("`forward` must be a vector",
                       bad_msg(good_head,
                               quote
                                   ve::AbstractVarianceEstimator = SimpleVariance()
                                   noun = "Variance"
                                   forward = Statistics.var(::MatNum) => :vararr
                                   doctest = "x"
                               end))
        @test occursin("must declare at least one generic",
                       bad_msg(good_head,
                               quote
                                   ve::AbstractVarianceEstimator = SimpleVariance()
                                   noun = "Variance"
                                   forward = []
                                   doctest = "x"
                               end))
    end

    @testset "Generated family member" begin
        rng = StableRNG(987654321)
        X = randn(rng, 60, 4)
        ew = eweights(1:60, inv(60); scale = true)
        pw = pweights(fill(inv(60), 60))
        idx = [2, 7, 11, 40]
        W = WindowedEstimatorProbe.ProbeWindowedVariance

        # One declared field; the macro supplies `w` and `window`, and nothing else.
        @test fieldnames(W) == (:ve, :w, :window)
        @test W <: PortfolioOptimisers.AbstractVarianceEstimator
        # The generated `export` makes the type reachable from the declaring module.
        @test :ProbeWindowedVariance in names(WindowedEstimatorProbe)

        # Keyword-constructor defaults come from the declaration's right-hand side; every
        # field is parametrised (`@concrete`), so the type is concrete.
        w0 = W()
        @test w0.ve isa SimpleVariance
        @test isnothing(w0.w)
        @test isnothing(w0.window)
        @test isconcretetype(typeof(w0))
        @test isconcretetype(typeof(W(; w = ew, window = idx)))

        # The positional constructor validates `w` and `window`, uniformly with the five
        # shipped members.
        @test_throws PortfolioOptimisers.IsEmptyError W(; window = Int[])
        @test_throws DomainError W(; window = -5)
        @test_throws PortfolioOptimisers.IsEmptyError W(; w = pweights(Float64[]))

        # `@fprop`/`@wprop`: `factory` rebinds `w` and recurses into the inner estimator,
        # while `window` rides through untouched.
        w1 = factory(W(; window = 20), ew)
        @test w1.w === ew
        @test w1.ve.w === ew
        @test w1.window == 20
        # `@vprop`: a view recurses into the inner estimator and keeps the window.
        w2 = PortfolioOptimisers.port_opt_view(W(; window = 20), [1, 2])
        @test w2 isa W
        @test w2.window == 20

        # The forwarders window the data, then delegate to the inner estimator.
        @test isapprox(var(W(; window = 20), X), var(SimpleVariance(), X[41:60, :]))
        @test isapprox(var(W(; window = idx), X), var(SimpleVariance(), X[idx, :]))
        @test isapprox(var(W(), X), var(SimpleVariance(), X))
        @test isapprox(std(W(; window = 20), X[:, 1]), std(SimpleVariance(), X[41:60, 1]))
        # `dims = 2` windows the transposed data identically.
        @test isapprox(var(W(; window = 20), permutedims(X); dims = 2),
                       var(SimpleVariance(), permutedims(X[41:60, :]); dims = 2))
        # Observation weights are rebound to the window, not applied whole.
        @test isapprox(var(W(; w = pw, window = 20), X),
                       var(SimpleVariance(; me = SimpleExpectedReturns(; w = pw[41:60]),
                                          w = pw[41:60]), X[41:60, :]))

        # Declaring `mean` emits it as a named keyword, so it reaches the inner estimator
        # instead of riding in `kwargs...` into `windowed_preamble`.
        @test Base.kwarg_decl(only(methods(var, (W, PortfolioOptimisers.MatNum)))) ==
              [:dims, :mean, :iv, Symbol("kwargs...")]
        @test isapprox(var(W(; window = 20), X;
                           mean = mean(SimpleExpectedReturns(), X[41:60, :])),
                       var(W(; window = 20), X))
        # Omitting it emits no keyword at all, so nothing can ride through unnoticed: the
        # call no longer matches the generated forwarder and falls through to a StatsBase
        # fallback that has nothing to compute with.
        @test isempty(Base.kwarg_decl(only(methods(std, (W, PortfolioOptimisers.VecNum)))))
        @test_throws Exception std(W(; window = 20), X[:, 1]; mean = 0.0)

        # Docstrings are generated too, with the dictionary lookups kept live rather than
        # hand-copied, so an `arg_dict`/`ret_dict` edit reaches them.
        tdoc = string(@doc WindowedEstimatorProbe.ProbeWindowedVariance)
        @test occursin("Variance estimator that restricts computation to a rolling or indexed observation window",
                       tdoc)
        @test occursin(PortfolioOptimisers.val_dict[:oow], tdoc)
        @test occursin("julia> 1 + 1", tdoc)
        mdoc = string(Base.Docs.doc(Base.Docs.Binding(Statistics, :var),
                                    Tuple{W, PortfolioOptimisers.MatNum}))
        @test occursin(PortfolioOptimisers.ret_dict[:vararr], mdoc)
        @test occursin(PortfolioOptimisers.arg_dict[:dims], mdoc)
        @test occursin("[`windowed_preamble`](@ref)", mdoc)
        sdoc = string(Base.Docs.doc(Base.Docs.Binding(Statistics, :std),
                                    Tuple{W, PortfolioOptimisers.VecNum}))
        @test occursin(PortfolioOptimisers.ret_dict[:stdnum], sdoc)
        @test !occursin("dims", sdoc)

        # The five shipped members are generated from this same template, which is what
        # keeps them in sync (ADR 0039).
        for T in (WindowedExpectedReturns, WindowedCovariance, WindowedVariance,
                  WindowedCoskewness, WindowedCokurtosis)
            @test occursin("estimator that restricts computation to a rolling or indexed observation window",
                           string(Base.Docs.doc(T)))
        end
    end

    @testset "windowed_preamble" begin
        rng = StableRNG(192837465)
        X = randn(rng, 30, 4)
        iv = abs.(randn(rng, 30, 4)) .+ 1
        ew = eweights(1:30, inv(30); scale = true)
        idx = [2, 5, 9]

        # No window: the data and `iv` pass through whole, and the weights cover the whole
        # sample. Only `window = nothing` resolves to a `Colon`, which is why `iv` is left
        # alone here and subset everywhere else.
        inner, Xw, ivw = PortfolioOptimisers.windowed_preamble(SimpleVariance(), ew,
                                                               nothing, X; iv = iv)
        @test size(Xw) == size(X)
        @test ivw === iv
        @test inner.w == ew
        # An `Int` window resolves to a range over the last observations, which is a
        # `VecInt`, so `iv` and the weights are subset to it too.
        inner, Xw, ivw = PortfolioOptimisers.windowed_preamble(SimpleVariance(), ew, 10, X;
                                                               iv = iv)
        @test Xw == X[21:30, :]
        @test ivw == iv[21:30, :]
        @test inner.w == ew[21:30]
        # An index vector selects exactly those observations.
        inner, Xw, ivw = PortfolioOptimisers.windowed_preamble(SimpleVariance(), ew, idx, X;
                                                               iv = iv)
        @test Xw == X[idx, :]
        @test ivw == iv[idx, :]
        @test inner.w == ew[idx]
        # `dims = 2` windows columns instead of rows, for both `X` and `iv`.
        inner, Xw, ivw = PortfolioOptimisers.windowed_preamble(SimpleVariance(), ew, idx,
                                                               permutedims(X);
                                                               iv = permutedims(iv),
                                                               dims = 2)
        @test Xw == permutedims(X)[:, idx]
        @test ivw == permutedims(iv)[:, idx]
        # Without weights the inner estimator is left unweighted; without `iv` there is
        # nothing to subset.
        inner, Xw, ivw = PortfolioOptimisers.windowed_preamble(SimpleVariance(), nothing,
                                                               idx, X)
        @test isnothing(inner.w)
        @test isnothing(ivw)
        # The vector method windows the series and rebinds the weights the same way.
        inner, xw = PortfolioOptimisers.windowed_preamble(SimpleVariance(), ew, 10, X[:, 1])
        @test xw == X[21:30, 1]
        @test inner.w == ew[21:30]
    end
end
