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
               ShrunkExpectedReturns(; alg = JamesStein(; target = VolatilityWeighted())),
               ShrunkExpectedReturns(; alg = JamesStein(; target = MeanSquaredError())),
               ShrunkExpectedReturns(; alg = JamesStein(),
                                     me = SimpleExpectedReturns(; w = ew)),
               ShrunkExpectedReturns(; alg = BayesStein()),
               ShrunkExpectedReturns(; alg = BayesStein(; target = VolatilityWeighted())),
               ShrunkExpectedReturns(; alg = BayesStein(; target = MeanSquaredError())),
               ShrunkExpectedReturns(; alg = BayesStein(),
                                     me = SimpleExpectedReturns(; w = ew)),
               ShrunkExpectedReturns(; alg = BodnarOkhrinParolya()),
               ShrunkExpectedReturns(;
                                     alg = BodnarOkhrinParolya(;
                                                               target = VolatilityWeighted())),
               ShrunkExpectedReturns(;
                                     alg = BodnarOkhrinParolya(;
                                                               target = MeanSquaredError())),
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
        me0 = ShrunkExpectedReturns(;
                                    ce = PortfolioOptimisersCovariance(;
                                                                       ce = Covariance(;
                                                                                       alg = Semi())),
                                    alg = JamesStein(; target = VolatilityWeighted()))
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
               LowerTailDependenceCovariance(), GerberCovariance(; alg = Gerber0()),
               GerberCovariance(; alg = Gerber0(),
                                ve = SimpleVariance(; me = SimpleExpectedReturns(; w = ew),
                                                    corrected = false, w = ew)),
               GerberCovariance(; alg = StandardisedGerber0()),
               GerberCovariance(;
                                alg = StandardisedGerber0(;
                                                          me = SimpleExpectedReturns(;
                                                                                     w = ew)),
                                ve = SimpleVariance(; me = SimpleExpectedReturns(; w = ew),
                                                    corrected = false, w = ew)),
               GerberCovariance(; alg = Gerber1()),
               GerberCovariance(; alg = StandardisedGerber1()),
               GerberCovariance(; alg = Gerber2()),
               GerberCovariance(; alg = StandardisedGerber2()),
               SmythBrobyCovariance(; alg = SmythBroby0()),
               SmythBrobyCovariance(; alg = SmythBroby0(),
                                    me = SimpleExpectedReturns(; w = ew),
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew)),
               SmythBrobyCovariance(; alg = StandardisedSmythBroby0()),
               SmythBrobyCovariance(; alg = StandardisedSmythBroby0(),
                                    me = SimpleExpectedReturns(; w = ew),
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew)),
               SmythBrobyCovariance(; alg = SmythBroby1()),
               SmythBrobyCovariance(; alg = StandardisedSmythBroby1()),
               SmythBrobyCovariance(; alg = SmythBroby2()),
               SmythBrobyCovariance(; alg = StandardisedSmythBroby2()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber0()),
               SmythBrobyCovariance(; alg = StandardisedSmythBrobyGerber0()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber1()),
               SmythBrobyCovariance(; alg = StandardisedSmythBrobyGerber1()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber2()),
               SmythBrobyCovariance(; alg = StandardisedSmythBrobyGerber2())]
        df = CSV.read(joinpath(@__DIR__, "./assets/covariance.csv.gz"), DataFrame)
        for (i, ce) in pairs(ces)
            cei = PortfolioOptimisersCovariance(; ce = ce)
            sigma = cov(cei, rd.X'; dims = 2)
            rho = cor(cei, rd.X'; dims = 2)
            @test isapprox(cov2cor(sigma), rho)
            success = isapprox(vec(sigma), df[!, i])
            if !success
                println("Counter: $i")
                find_tol(vec(sigma), df[!, i])
            end
            @test success
        end
        @test isapprox(df[!, "40"],
                       vec(cov(PortfolioOptimisersCovariance(;
                                                             mp = DefaultMatrixProcessing(;
                                                                                          alg = LoGo())),
                               rd.X)))
        @test isapprox(var(SimpleVariance(; w = ew, corrected = false), rd.X; dims = 1),
                       std(SimpleVariance(; w = ew, corrected = false), rd.X; dims = 1) .^
                       2)

        ce0 = PortfolioOptimisersCovariance(;
                                            ce = GerberCovariance(; alg = Gerber2(),
                                                                  threshold = 0.1))
        ce = PortfolioOptimisers.factory(ce0, ew)
        @test !(ce.ce.ve === ce0.ce.ve)
        @test ce.ce.pdm === ce0.ce.pdm
        @test ce.ce.alg === ce0.ce.alg
        @test ce.ce.threshold == ce0.ce.threshold
        @test ce.mp === ce0.mp
        @test ce.ce.ve.w === ew
        @test ce.ce.ve.me.w === ew
        @test ce.mp === ce0.mp
        @test isapprox(cov(GerberCovariance(; alg = Gerber0()), rd.X),
                       cov(GerberCovariance(; alg = Gerber0()), rd.X'; dims = 2))
        @test isapprox(cov(GerberCovariance(; alg = StandardisedGerber0()), rd.X),
                       cov(GerberCovariance(; alg = StandardisedGerber0()), rd.X';
                           dims = 2))
        @test isapprox(cor(GerberCovariance(; alg = Gerber0()), rd.X),
                       cor(GerberCovariance(; alg = Gerber0()), rd.X'; dims = 2))
        @test isapprox(cor(GerberCovariance(; alg = StandardisedGerber0()), rd.X),
                       cor(GerberCovariance(; alg = StandardisedGerber0()), rd.X';
                           dims = 2))

        ce0 = PortfolioOptimisersCovariance(;
                                            ce = GerberCovariance(;
                                                                  alg = StandardisedGerber0(),
                                                                  threshold = 0.2))
        ce = PortfolioOptimisers.factory(ce0, ew)
        @test !(ce.ce.ve === ce0.ce.ve)
        @test ce.ce.pdm === ce0.ce.pdm
        @test ce.ce.threshold == ce0.ce.threshold
        @test ce.mp === ce0.mp
        @test ce.ce.ve.w === ew
        @test ce.ce.ve.me.w === ew
        @test ce.ce.alg.me.w === ew

        ce0 = PortfolioOptimisersCovariance(;
                                            ce = SmythBrobyCovariance(; alg = SmythBroby2(),
                                                                      threshold = 0.1,
                                                                      c1 = 0.6, c2 = 0.2,
                                                                      c3 = 2.2, n = 3))
        ce = PortfolioOptimisers.factory(ce0, ew)
        @test !(ce.ce.ve === ce0.ce.ve)
        @test !(ce.ce.me === ce0.ce.me)
        @test ce.ce.ve.w === ew
        @test ce.ce.ve.me.w === ew
        @test ce.ce.me.w === ew
        @test ce.ce.pdm === ce0.ce.pdm
        @test ce.ce.threshold == ce0.ce.threshold
        @test ce.ce.c1 == ce0.ce.c1
        @test ce.ce.c2 == ce0.ce.c2
        @test ce.ce.c3 == ce0.ce.c3
        @test ce.ce.n == ce0.ce.n
        @test ce.ce.threads === ce0.ce.threads
        @test ce.ce.alg === ce0.ce.alg
        @test isapprox(cov(SmythBrobyCovariance(; alg = SmythBroby0()), rd.X),
                       cov(SmythBrobyCovariance(; alg = SmythBroby0()), rd.X'; dims = 2))
        @test isapprox(cov(SmythBrobyCovariance(; alg = StandardisedSmythBroby0()), rd.X),
                       cov(SmythBrobyCovariance(; alg = StandardisedSmythBroby0()), rd.X';
                           dims = 2))
        @test isapprox(cor(SmythBrobyCovariance(; alg = SmythBroby0()), rd.X),
                       cor(SmythBrobyCovariance(; alg = SmythBroby0()), rd.X'; dims = 2))
        @test isapprox(cor(SmythBrobyCovariance(; alg = StandardisedSmythBroby0()), rd.X),
                       cor(SmythBrobyCovariance(; alg = StandardisedSmythBroby0()), rd.X';
                           dims = 2))

        ce0 = PortfolioOptimisersCovariance(;
                                            ce = DistanceCovariance(; args = (3,),
                                                                    kwargs = (; foo = 5)))
        ce = PortfolioOptimisers.factory(ce0, ew)
        @test ce.ce.dist === ce0.ce.dist
        @test ce.ce.args === ce0.ce.args
        @test ce.ce.kwargs === ce0.ce.kwargs
        @test ce.ce.threads === ce0.ce.threads
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
        @test ce.ce.threads === ce0.ce.threads
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
               StepwiseRegression(; crit = PValue(; threshold = 1e-15)),
               StepwiseRegression(; crit = PValue(; threshold = 1e-15), alg = Backward())]
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
               LowerTailDependenceCovariance(), GerberCovariance(), SmythBrobyCovariance()]
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
                @test isapprox(d1, distance(de, ce, rd.X))
                @test isapprox(d1, distance(de, cei, rd.X))
                distance(Distance(;
                                  alg = VariationInfoDistance(; bins = ce.bins,
                                                              normalise = ce.normalise)),
                         cov(ce, rd.X), rd.X)
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
                         cov(ce, rd.X), rd.X)
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
            r1, d1 = cor_and_dist(de, cei, rd.X)
            d2 = cor_and_dist(deg, cei, rd.X)[2]
            @test isapprox(d1, d2)
            @test isapprox(r1, cor(cei, rd.X))

            d3 = distance(de, cei, rd.X)
            d4 = if isa(ce, MutualInfoCovariance)
                distance(DistanceDistance(;
                                          alg = VariationInfoDistance(; bins = ce.bins,
                                                                      normalise = ce.normalise)),
                         cov(ce, rd.X), rd.X)
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
