@safetestset "Moments" begin
    using Test, PortfolioOptimisers, DataFrames, TimeSeries, CSV, CovarianceEstimation,
          StableRNGs, StatsBase, LinearAlgebra
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
               ShrunkExpectedReturns(; alg = JamesStein(; target = MeanSquareError())),
               ShrunkExpectedReturns(; alg = JamesStein(),
                                     me = SimpleExpectedReturns(; w = ew)),
               ShrunkExpectedReturns(; alg = BayesStein()),
               ShrunkExpectedReturns(; alg = BayesStein(; target = VolatilityWeighted())),
               ShrunkExpectedReturns(; alg = BayesStein(; target = MeanSquareError())),
               ShrunkExpectedReturns(; alg = BayesStein(),
                                     me = SimpleExpectedReturns(; w = ew)),
               ShrunkExpectedReturns(; alg = BodnarOkhrinParolya()),
               ShrunkExpectedReturns(;
                                     alg = BodnarOkhrinParolya(;
                                                               target = VolatilityWeighted())),
               ShrunkExpectedReturns(;
                                     alg = BodnarOkhrinParolya(;
                                                               target = MeanSquareError())),
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
    end
    @testset "Covariance Estimators" begin
        ces = [Covariance(; alg = Full()),
               Covariance(; alg = Full(), me = SimpleExpectedReturns(; w = ew),
                          ce = GeneralWeightedCovariance(;
                                                         ce = SimpleCovariance(;
                                                                               corrected = false),
                                                         w = ew)),
               Covariance(; alg = Full(),
                          ce = GeneralWeightedCovariance(;
                                                         ce = AnalyticalNonlinearShrinkage())),
               Covariance(; alg = Semi()),
               Covariance(; alg = Semi(), me = SimpleExpectedReturns(; w = ew),
                          ce = GeneralWeightedCovariance(;
                                                         ce = SimpleCovariance(;
                                                                               corrected = false),
                                                         w = ew)),
               Covariance(; alg = Semi(), me = SimpleExpectedReturns(; w = fw),
                          ce = GeneralWeightedCovariance(;
                                                         ce = AnalyticalNonlinearShrinkage(),
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
               DistanceCovariance(), DistanceCovariance(; w = ew), LTDCovariance(),
               GerberCovariance(; alg = Gerber0()),
               GerberCovariance(; alg = Gerber0(),
                                ve = SimpleVariance(; me = SimpleExpectedReturns(; w = ew),
                                                    corrected = false, w = ew)),
               GerberCovariance(; alg = NormalisedGerber0()),
               GerberCovariance(;
                                alg = NormalisedGerber0(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew)),
                                ve = SimpleVariance(; me = SimpleExpectedReturns(; w = ew),
                                                    corrected = false, w = ew)),
               GerberCovariance(; alg = Gerber1()),
               GerberCovariance(; alg = NormalisedGerber1()),
               GerberCovariance(; alg = Gerber2()),
               GerberCovariance(; alg = NormalisedGerber2()),
               SmythBrobyCovariance(; alg = SmythBroby0()),
               SmythBrobyCovariance(; alg = SmythBroby0(),
                                    me = SimpleExpectedReturns(; w = ew),
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew)),
               SmythBrobyCovariance(; alg = NormalisedSmythBroby0()),
               SmythBrobyCovariance(; alg = NormalisedSmythBroby0(),
                                    me = SimpleExpectedReturns(; w = ew),
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew)),
               SmythBrobyCovariance(; alg = SmythBroby1()),
               SmythBrobyCovariance(; alg = NormalisedSmythBroby1()),
               SmythBrobyCovariance(; alg = SmythBroby2()),
               SmythBrobyCovariance(; alg = NormalisedSmythBroby2()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber0()),
               SmythBrobyCovariance(; alg = NormalisedSmythBrobyGerber0()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber1()),
               SmythBrobyCovariance(; alg = NormalisedSmythBrobyGerber1()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber2()),
               SmythBrobyCovariance(; alg = NormalisedSmythBrobyGerber2())]
        df = CSV.read(joinpath(@__DIR__, "./assets/covariance.csv.gz"), DataFrame)
        for (i, ce) in pairs(ces)
            cei = PortfolioOptimisersCovariance(; ce = ce)
            sigma = cov(cei, rd.X)
            rho = cor(cei, rd.X)
            @test isapprox(cov2cor(sigma), rho)
            success = isapprox(vec(sigma), df[!, i])
            if !success
                println("Counter: $i")
                find_tol(vec(sigma), df[!, i])
            end
            @test success
        end
        sigma = cov(PortfolioOptimisersCovariance(;
                                                  mp = DefaultMatrixProcessing(;
                                                                               alg = LoGo())),
                    rd.X)
        jlogo = sigma \ I
        jlogo[abs.(jlogo) .< 1e-10] .= 0
        @test isapprox(sparse(jlogo),
                       sparse([1, 2, 7, 13, 14, 1, 2, 7, 13, 3, 4, 5, 6, 7, 9, 10, 17, 20,
                               3, 4, 7, 9, 3, 5, 9, 10, 17, 20, 3, 6, 9, 20, 1, 2, 3, 4, 7,
                               9, 10, 13, 14, 16, 18, 19, 8, 10, 11, 12, 14, 15, 16, 3, 4,
                               5, 6, 7, 9, 10, 18, 20, 3, 5, 7, 8, 9, 10, 14, 16, 18, 8, 11,
                               12, 14, 15, 8, 11, 12, 14, 15, 16, 1, 2, 7, 13, 14, 18, 1, 7,
                               8, 10, 11, 12, 13, 14, 16, 18, 19, 8, 11, 12, 15, 7, 8, 10,
                               12, 14, 16, 19, 3, 5, 17, 20, 7, 9, 10, 13, 14, 18, 7, 14,
                               16, 19, 3, 5, 6, 9, 17, 20],
                              [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4,
                               4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
                               7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                               10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12,
                               12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14,
                               14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16,
                               16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19,
                               19, 19, 20, 20, 20, 20, 20, 20],
                              [6121.2007513367535, -696.2580934634964, -607.6539848628298,
                               -3772.389241410706, -770.5235947962645, -696.2580934634962,
                               1461.5118283412462, -187.97605425600062, -888.8462163576756,
                               13905.755188577927, -657.5280018919394, 174.72122355946658,
                               -662.4153169577866, 380.1177965702748, -12578.731835036182,
                               -78.8948158034948, -197.14366367284828, -1311.2792466076562,
                               -657.528001891939, 2501.2837895782036, -1757.125753258157,
                               39.82670275504742, 174.7212235594701, 7682.448549016267,
                               -1725.9863133739323, -756.5668021679467, -84.02317044610857,
                               -5755.5717690338215, -662.4153169577858, 2341.1472234024145,
                               -815.3767607600333, -693.8739983414891, -607.6539848628299,
                               -187.97605425600062, 380.1177965702772, -1757.1257532581562,
                               7787.663993740247, -1182.1560713991157, 305.65444832008274,
                               -1544.5009190294306, -1712.2054513956934,
                               -187.95996241810565, -547.0228044852099, -708.1129872834789,
                               14942.130900768876, -1588.3882395252074, -1261.3886184984456,
                               -2468.703569362591, -1972.678037244616, -2333.413224623252,
                               -3021.370248668776, -12578.73183503618, 39.82670275504751,
                               -1725.9863133739264, -815.3767607600316, -1182.1560713991123,
                               17680.56930599364, -1257.6237721563375, -852.8892127248357,
                               1168.097384551503, -78.89481580349671, -756.5668021679495,
                               305.65444832007773, -1588.3882395252094, -1257.623772156334,
                               12816.919855942262, -6063.037344488557, -1376.9781836886555,
                               -360.01800431502625, -1261.388618498445, 4236.228013944008,
                               -1324.9506019349958, -1000.8724648926739, -888.4385543853767,
                               -2468.703569362593, -1324.9506019349958, 7739.180313860352,
                               -510.3532086791768, -1195.1827156524448, -661.2792580814379,
                               -3772.3892414107045, -888.8462163576754, -1544.500919029432,
                               8718.647353117387, -1212.967630979414, -939.9082299748511,
                               -770.5235947962638, -1712.2054513956869, -1972.678037244611,
                               -6063.037344488563, -1000.8724648926732, -510.35320867917676,
                               -1212.9676309794124, 21199.803395777668, -6091.3538739134665,
                               -1307.845662198386, -2121.180452774497, -2333.413224623253,
                               -888.4385543853762, -1195.182715652445, 5264.84397321922,
                               -187.95996241810627, -3021.370248668777, -1376.9781836886568,
                               -661.27925808144, -6091.35387391346, 14004.94079595229,
                               -1902.165355145698, -197.14366367284822, -84.02317044610845,
                               584.6987862571475, -352.7043724988662, -547.0228044852101,
                               -852.8892127248356, -360.0180043150252, -939.9082299748502,
                               -1307.8456621983885, 4527.0407375588375, -708.1129872834813,
                               -2121.180452774496, -1902.1653551456986, 7177.812799670624,
                               -1311.2792466076617, -5755.571769033821, -693.8739983414905,
                               1168.0973845515114, -352.704372498866, 8116.582242189527],
                              20, 20))
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
               DimensionReductionRegression(; drtgt = PPCA())]
        df = CSV.read(joinpath(@__DIR__, "./assets/Regression.csv.gz"), DataFrame)
        for (i, re) in pairs(res)
            rr = regression(re, rd.X, rd.F)
            if i == 14
                continue
            end
            lt = [rr.b; vec(rr.M)]
            success = isapprox(lt, df[!, i])
            if !success
                println("Counter: $i")
                find_tol(lt, df[!, i])
            end
            @test success
        end
    end
    @testset "Coskewness" begin
        skes = [Coskewness(; alg = Full()), Coskewness(; alg = Semi())]
        df = CSV.read(joinpath(@__DIR__, "./assets/coskewness.csv.gz"), DataFrame)
        for (i, ske) in pairs(skes)
            sk, v = coskewness(ske, rd.X)
            success = isapprox([vec(sk); vec(v)], df[!, i])
            if !success
                println("Counter: $i")
                find_tol([vec(sk); vec(v)], df[!, i])
            end
            @test success
        end
    end
    @testset "Cokurtosis" begin
        ktes = [Cokurtosis(; alg = Full()), Cokurtosis(; alg = Semi())]
        df = CSV.read(joinpath(@__DIR__, "./assets/cokurtosis.csv.gz"), DataFrame)
        for (i, kte) in pairs(ktes)
            kt = cokurtosis(kte, rd.X)
            success = isapprox(vec(kt), df[!, i])
            if !success
                println("Counter: $i")
                find_tol(vec(kt), df[!, i])
            end
            @test success
        end
    end
    @testset "Distance" begin
        des = [Distance(; alg = SimpleAbsoluteDistance()),
               DistanceDistance(; alg = SimpleAbsoluteDistance()),
               Distance(; alg = LogDistance()), DistanceDistance(; alg = LogDistance())]
        desg = [GeneralDistance(; alg = SimpleAbsoluteDistance()),
                GeneralDistanceDistance(; alg = SimpleAbsoluteDistance()),
                GeneralDistance(; alg = LogDistance()),
                GeneralDistanceDistance(; alg = LogDistance())]
        df = CSV.read(joinpath(@__DIR__, "./assets/distance1.csv.gz"), DataFrame)
        ce = PortfolioOptimisersCovariance()
        for (i, (de, deg)) in enumerate(zip(des, desg))
            d1 = distance(de, ce, rd.X)
            dg1 = distance(deg, ce, rd.X)
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
            if isa(de, Distance{<:SimpleAbsoluteDistance}) ||
               isa(de, Distance{<:LogDistance}) ||
               isa(de, GeneralDistance{<:Any, <:SimpleAbsoluteDistance}) ||
               isa(de, GeneralDistance{<:Any, <:LogDistance})
                r, d = cor_and_dist(de, ce, rd.X)
                rg, dg = cor_and_dist(deg, ce, rd.X)
                @test isapprox(r, rg)
                @test isapprox(d, dg)
            end
        end
    end
    @testset "Canonical Distance" begin
        ces = [Covariance(; alg = Full()), SpearmanCovariance(), KendallCovariance(),
               MutualInfoCovariance(), DistanceCovariance(), LTDCovariance(),
               GerberCovariance(), SmythBrobyCovariance()]
        df = CSV.read(joinpath(@__DIR__, "./assets/CanonicalDistance.csv.gz"), DataFrame)
        de = Distance(; alg = CanonicalDistance())
        deg = GeneralDistance(; alg = CanonicalDistance())
        for (i, ce) in pairs(ces)
            cei = PortfolioOptimisersCovariance(; ce = ce)
            r1, d1 = cor_and_dist(de, cei, rd.X)
            d2 = cor_and_dist(deg, cei, rd.X)[2]
            @test isapprox(d1, d2)
            @test isapprox(r1, cor(cei, rd.X))
            d3 = distance(de, cei, rd.X)
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
            elseif isa(ce, LTDCovariance)
                distance(Distance(; alg = LogDistance()), cov(cei, rd.X), rd.X)
            else
                distance(de, cov(cei, rd.X), rd.X)
            end
            d5 = distance(deg, ce, rd.X)
            d6 = if isa(ce, MutualInfoCovariance)
                @test all(isapprox.((r1, d1), cor_and_dist(deg, ce, rd.X)))
                @test isapprox(d1, distance(deg, ce, rd.X))
                @test isapprox(d1, distance(deg, cei, rd.X))
                distance(GeneralDistance(;
                                         alg = VariationInfoDistance(; bins = ce.bins,
                                                                     normalise = ce.normalise)),
                         cov(ce, rd.X), rd.X)
            elseif isa(ce, DistanceCovariance)
                distance(GeneralDistance(; alg = CorrelationDistance(;)), cov(ce, rd.X),
                         rd.X)
            elseif isa(ce, LTDCovariance)
                distance(GeneralDistance(; alg = LogDistance()), cov(ce, rd.X), rd.X)
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
        deg = GeneralDistanceDistance(; alg = CanonicalDistance())
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
            elseif isa(ce, LTDCovariance)
                distance(DistanceDistance(; alg = LogDistance()), cov(cei, rd.X), rd.X)
            else
                distance(de, cov(cei, rd.X), rd.X)
            end
            d5 = distance(deg, ce, rd.X)
            d6 = if isa(ce, MutualInfoCovariance)
                distance(GeneralDistanceDistance(;
                                                 alg = VariationInfoDistance(;
                                                                             bins = ce.bins,
                                                                             normalise = ce.normalise)),
                         cov(ce, rd.X), rd.X)
            elseif isa(ce, DistanceCovariance)
                distance(GeneralDistanceDistance(; alg = CorrelationDistance(;)),
                         cov(ce, rd.X), rd.X)
            elseif isa(ce, LTDCovariance)
                distance(GeneralDistanceDistance(; alg = LogDistance()), cov(ce, rd.X),
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
