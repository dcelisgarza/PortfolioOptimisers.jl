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
    #=
    Issue #461, under child map #417 of the map of maps #404.

    Every claim these tests pin is a sentence of one of the five expected-returns docstrings
    that #461 swept. The numbers are hand-computed here rather than taken from a stored
    fixture, so a claim and its check can be read side by side.
    =#
    @testset "Expected returns estimators, swept claims" begin
        # --- equilibrium_mu -------------------------------------------------
        # `mu = l * sigma * w`, hand-computed on a three-asset sample.
        S461 = [4.0 1.0 0.5; 1.0 9.0 2.0; 0.5 2.0 16.0]
        w461 = [0.2, 0.3, 0.5]
        l461 = 2.5
        @test PortfolioOptimisers.equilibrium_mu(l461, S461, w461) ==
              [l461 * sum(S461[i, j] * w461[j] for j in 1:3) for i in 1:3]
        # `w = nothing` selects the equal-weight fallback, not a zero vector.
        @test PortfolioOptimisers.equilibrium_mu(l461, S461, nothing) ≈
              [l461 * sum(S461[i, j] for j in 1:3) / 3 for i in 1:3]
        # `sigma` is a block, so the result has length `size(sigma, 1)`, not `size(sigma, 2)`.
        B461 = [1.0 2.0 3.0; 4.0 5.0 6.0]
        @test PortfolioOptimisers.equilibrium_mu(2.0, B461, w461) ==
              [2.0 * sum(B461[i, j] * w461[j] for j in 1:3) for i in 1:2]
        @test length(PortfolioOptimisers.equilibrium_mu(2.0, B461, w461)) == size(B461, 1)
        # The length check reads `size(sigma, 2)`, and its message names both numbers.
        @test_throws DimensionMismatch("length(w) (2) must match the number of assets, size(sigma, 2) (3)") PortfolioOptimisers.equilibrium_mu(1.0,
                                                                                                                                               S461,
                                                                                                                                               [1.0,
                                                                                                                                                2.0])

        # --- EquilibriumExpectedReturns -------------------------------------
        # Every observation, so that `ew`, `fw` and `pw` line up with the rows.
        X461 = rd.X[:, 1:3]
        me461 = EquilibriumExpectedReturns(; w = w461, l = 3.0)
        @test mean(me461, X461) ==
              PortfolioOptimisers.equilibrium_mu(3.0, cov(me461.ce, X461), w461)
        # The result is a plain vector for both values of `dims`, not a `1 x N` matrix.
        @test isa(mean(me461, X461), Vector)
        @test isa(mean(EquilibriumExpectedReturns(), X461; dims = 2), Vector)
        @test length(mean(EquilibriumExpectedReturns(), X461)) == size(X461, 2)
        @test length(mean(EquilibriumExpectedReturns(), X461; dims = 2)) == size(X461, 1)
        # `dims` is checked by the covariance estimator, not by the method.
        @test_throws DomainError mean(me461, X461; dims = 3)
        # The observation weights reach the result through `ce`.
        @test mean(factory(me461, pw), X461) ==
              PortfolioOptimisers.equilibrium_mu(3.0, cov(factory(me461, pw).ce, X461),
                                                 w461)
        @test_throws PortfolioOptimisers.IsEmptyError EquilibriumExpectedReturns(;
                                                                                 w = Float64[])
        # `l` scales the whole vector, and `l == 0` gives a zero mean.
        @test mean(EquilibriumExpectedReturns(; w = w461, l = 6.0), X461) ≈
              2 * mean(me461, X461)
        @test all(iszero, mean(EquilibriumExpectedReturns(; w = w461, l = 0), X461))
        # `nothing` weights are the equal-weight vector, not a zero vector.
        @test mean(EquilibriumExpectedReturns(; l = 3.0), X461) ≈
              mean(EquilibriumExpectedReturns(; w = fill(inv(3), 3), l = 3.0), X461)

        # --- ExcessExpectedReturns ------------------------------------------
        exc461 = ExcessExpectedReturns(; me = SimpleExpectedReturns(), rf = 0.01)
        @test mean(exc461, X461) == mean(SimpleExpectedReturns(), X461) .- 0.01
        # The shape follows the nested estimator, because the subtraction is elementwise.
        @test size(mean(exc461, X461)) == size(mean(SimpleExpectedReturns(), X461))
        @test size(mean(exc461, X461; dims = 2)) ==
              size(mean(SimpleExpectedReturns(), X461; dims = 2))
        @test mean(factory(exc461, ew), X461) ==
              mean(factory(SimpleExpectedReturns(), ew), X461) .- 0.01
        @test_throws DomainError mean(exc461, X461; dims = 3)
        @test_throws PortfolioOptimisers.IsNonFiniteError ExcessExpectedReturns(; rf = Inf)
        @test_throws PortfolioOptimisers.IsNonFiniteError ExcessExpectedReturns(; rf = NaN)

        # --- StandardDeviationExpectedReturns and VarianceExpectedReturns ----
        # One is the elementwise square of the other, on the same data and the same `ce`.
        sd461 = StandardDeviationExpectedReturns()
        vr461 = VarianceExpectedReturns()
        @test mean(vr461, X461) ≈ mean(sd461, X461) .^ 2
        @test mean(sd461, X461) ≈ sqrt.(mean(vr461, X461))
        @test size(mean(sd461, X461)) == (1, size(X461, 2))
        @test size(mean(sd461, X461; dims = 2)) == (size(X461, 1), 1)
        @test size(mean(vr461, X461)) == (1, size(X461, 2))
        # Every choice inside `ce` reaches the result, so the weighted result differs.
        @test mean(factory(vr461, pw), X461) ≈ mean(factory(sd461, pw), X461) .^ 2
        @test !isapprox(mean(factory(sd461, fw), X461), mean(sd461, X461))
        @test_throws DomainError mean(sd461, X461; dims = 3)
        @test_throws DomainError mean(vr461, X461; dims = 3)

        # --- MedianExpectedReturns ------------------------------------------
        #=
        `StatsBase.median(v, w)` is the 0.5 quantile of the weighted sample. It
        interpolates between two order statistics, and `w_(1)` in its definition is the
        weight of the SMALLEST value, not the weight of the first observation.
        =#
        v461 = [5.0, 1.0, 3.0, 7.0]
        wq461 = StatsBase.Weights([0.4, 0.1, 0.2, 0.3])
        p461 = sortperm(v461)
        vs461, ws461 = v461[p461], wq461[p461]
        S_461 = cumsum(ws461)
        h461 = 0.5 * (sum(ws461) - ws461[1]) + ws461[1]     # w_(1) is the weight of 1.0
        k461 = findlast(<=(h461), S_461)
        hand461 = vs461[k461] +
                  (h461 - S_461[k461]) / (S_461[k461 + 1] - S_461[k461]) *
                  (vs461[k461 + 1] - vs461[k461])
        @test median(v461, wq461) == hand461 == 4.25
        # It interpolates: the answer is not one of the observed values.
        @test hand461 ∉ v461
        # Reading `w[1]` instead of the weight of the smallest value gives 5.0, not 4.25.
        h_naive461 = 0.5 * (sum(wq461) - wq461[1]) + wq461[1]
        kn461 = findlast(<=(h_naive461), S_461)
        naive461 = vs461[kn461] +
                   (h_naive461 - S_461[kn461]) / (S_461[kn461 + 1] - S_461[kn461]) *
                   (vs461[kn461 + 1] - vs461[kn461])
        @test naive461 == 5.0
        @test naive461 != hand461
        # The estimator is that quantile, column by column.
        Xq461 = [5.0 10.0; 1.0 20.0; 3.0 30.0; 7.0 40.0]
        mew461 = MedianExpectedReturns(; w = wq461)
        @test mean(mew461, Xq461) ==
              reshape([median(view(Xq461, :, j), wq461) for j in 1:2], 1, :)
        @test mean(mew461, Xq461)[1, 1] == 4.25
        @test size(mean(mew461, Xq461)) == (1, size(Xq461, 2))
        @test size(mean(mew461, permutedims(Xq461); dims = 2)) == (size(Xq461, 2), 1)
        # Under equal weights it reduces to the ordinary median.
        @test mean(MedianExpectedReturns(; w = StatsBase.Weights(fill(0.25, 4))), Xq461) ==
              mean(MedianExpectedReturns(), Xq461) ==
              median(Xq461; dims = 1)
        # The weighted branch checks `dims` through `dims_oriented`, not `assert_dims`.
        @test_throws DomainError mean(mew461, Xq461; dims = 3)
        @test_throws DomainError mean(MedianExpectedReturns(), Xq461; dims = 3)
        @test_throws PortfolioOptimisers.IsEmptyError MedianExpectedReturns(;
                                                                            w = StatsBase.Weights(Float64[]))

        # --- CustomValueExpectedReturns -------------------------------------
        # The assertion is the whole contract, so drive every rejection and assert the
        # message, not only the type.
        @test_throws ArgumentError("val must be a vector of numbers, one element per asset. Got\nval => Float64") PortfolioOptimisers.assert_custom_expected_returns_val(1.0,
                                                                                                                                                                         3)
        @test_throws ArgumentError("val must be a vector of numbers, one element per asset. Got\nval => typeof(sum)") PortfolioOptimisers.assert_custom_expected_returns_val(sum,
                                                                                                                                                                             3)
        @test_throws DimensionMismatch("length(val) (2) must match the number of assets (3)") PortfolioOptimisers.assert_custom_expected_returns_val([1.0,
                                                                                                                                                      2.0],
                                                                                                                                                     3)
        @test isnothing(PortfolioOptimisers.assert_custom_expected_returns_val([1.0, 2.0,
                                                                                3.0], 3))
        # `val_sym` reaches the message of the callable branch.
        @test_throws DimensionMismatch("length(val(X; dims = 1, kwargs...)) (2) must match the number of assets (3)") PortfolioOptimisers.assert_custom_expected_returns_val([1.0,
                                                                                                                                                                              2.0],
                                                                                                                                                                             3,
                                                                                                                                                                             "val(X; dims = 1, kwargs...)")
        # The scalar branch broadcasts and inserts the reduced dimension.
        @test mean(CustomValueExpectedReturns(; val = 0.5), X461) ==
              fill(0.5, 1, size(X461, 2))
        @test mean(CustomValueExpectedReturns(; val = 0.5), X461; dims = 2) ==
              fill(0.5, size(X461, 1), 1)
        # The vector branch takes the stored vector and inserts the reduced dimension.
        @test mean(CustomValueExpectedReturns(; val = [0.1, 0.2, 0.3]), X461) ==
              [0.1 0.2 0.3]
        @test_throws DimensionMismatch mean(CustomValueExpectedReturns(; val = [0.1, 0.2]),
                                            X461)
        # The callable branch returns the vector unchanged, and inserts no dimension.
        cb461 = mean(CustomValueExpectedReturns(;
                                                val = (X; dims = 1, kwargs...) -> fill(0.7,
                                                                                       size(X,
                                                                                            setdiff((1,
                                                                                                     2),
                                                                                                    (dims,))[1]))),
                     X461)
        @test isa(cb461, Vector)
        @test cb461 == fill(0.7, size(X461, 2))
        @test_throws DimensionMismatch mean(CustomValueExpectedReturns(;
                                                                       val = (X; kwargs...) -> [0.1,
                                                                                                0.2]),
                                            X461)
        @test_throws ArgumentError mean(CustomValueExpectedReturns(;
                                                                   val = (X; kwargs...) -> 0.1),
                                        X461)
        @test_throws DomainError mean(CustomValueExpectedReturns(), X461; dims = 3)
        @test_throws PortfolioOptimisers.IsEmptyError CustomValueExpectedReturns(;
                                                                                 val = Float64[])

        #=
        `assert_nonempty` labelled its bullet `!isempty(sym)` and reported the value of
        `isempty(sym)`, so the failure printed `!isempty(w) => true` while raising because
        `!isempty(w)` was false. The label and the value now name the same expression.
        =#
        @test_throws PortfolioOptimisers.IsEmptyError("!isempty(w) must hold. Got\n!isempty(w) => false") PortfolioOptimisers.assert_nonempty(Float64[],
                                                                                                                                              :w)
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
        # The `SimpleVariance{Nothing}` matrix methods must read their own `w`, not the `w`
        # of the `SimpleExpectedReturns` they build to resolve the mean.
        vew = SimpleVariance(; me = nothing, w = ew, corrected = false)
        @test isapprox(var(vew, rd.X; dims = 1), std(vew, rd.X; dims = 1) .^ 2)
        @test !isapprox(var(vew, rd.X; dims = 1),
                        var(SimpleVariance(; me = nothing, corrected = false), rd.X;
                            dims = 1))
        #=
        ADR 0088, #490. `w` weights the whole estimate, so the centre carries the same
        weights as the deviations. The divisor block below pins the numbers on three
        observations. These four pin the routes: the standard deviation as well as the
        variance, the estimator that `factory` builds, the `SimpleVariance{Nothing}` method
        that builds its own centring estimator, and the unweighted path that did not move.
        =#
        let v490 = rd.X[:, 1], X490 = reshape(rd.X[:, 1], :, 1), aw490 = aweights(ew)
            ve490 = SimpleVariance(; w = aw490)
            @test isapprox(std(ve490, X490; dims = 1)[1], std(ve490, v490))
            @test isapprox(var(factory(SimpleVariance(), aw490), X490; dims = 1)[1],
                           var(ve490, v490))
            @test isapprox(var(SimpleVariance(; me = nothing, w = aw490), X490; dims = 1)[1],
                           var(ve490, v490))
            @test isapprox(var(SimpleVariance(), X490; dims = 1)[1],
                           var(SimpleVariance(), v490))
        end
        #=
        ADR 0088, #492. The rule of #490 covers the whole moment layer. `Covariance` gained
        a `@wprop w` of its own, and `Coskewness` and `Cokurtosis` now read the `w` they
        already held. For each of the four a hand-built weighted estimator therefore answers
        what `factory` builds, under both moment algorithms.

        Option 2 of #492 is what these pin. `w` weights the centre through `me` and the
        deviations through `ce`, so one field weights the whole estimate. Weights that live
        inside `ce` alone still centre on the unweighted mean, because no verb reads the
        weights of an arbitrary `StatsBase.CovarianceEstimator`; the `mean` keyword reaches
        that combination, and the last comparison of the block pins the pair.
        =#
        let aw492 = aweights(ew),
            aw492b = aweights(reverse(collect(ew))),
            mu492 = mean(SimpleExpectedReturns(), rd.X; dims = 1)

            for malg in (FullMoment(), SemiMoment())
                @test isapprox(cov(Covariance(; alg = malg, w = aw492), rd.X),
                               cov(factory(Covariance(; alg = malg), aw492), rd.X))
                @test isapprox(cor(Covariance(; alg = malg, w = aw492), rd.X),
                               cor(factory(Covariance(; alg = malg), aw492), rd.X))
                @test isapprox(coskewness(Coskewness(; alg = malg, w = aw492), rd.X)[1],
                               coskewness(factory(Coskewness(; alg = malg), aw492), rd.X)[1])
                @test isapprox(cokurtosis(Cokurtosis(; alg = malg, w = aw492), rd.X),
                               cokurtosis(factory(Cokurtosis(; alg = malg), aw492), rd.X))
            end
            # The weighted centre is not the unweighted one, so each fix is measurable.
            @test !isapprox(cov(Covariance(; w = aw492), rd.X),
                            cov(Covariance(; w = aw492), rd.X; mean = mu492))
            @test !isapprox(coskewness(Coskewness(; w = aw492), rd.X)[1],
                            coskewness(Coskewness(; w = aw492), rd.X; mean = mu492)[1])
            @test !isapprox(cokurtosis(Cokurtosis(; w = aw492), rd.X),
                            cokurtosis(Cokurtosis(; w = aw492), rd.X; mean = mu492))
            # `w` wins over the weights that `me` and `ce` carry, which is what `factory`
            # does on every other path.
            @test isapprox(cov(Covariance(; me = SimpleExpectedReturns(; w = aw492b),
                                          ce = GeneralCovariance(; w = aw492b), w = aw492),
                               rd.X), cov(Covariance(; w = aw492), rd.X))
            @test isapprox(coskewness(Coskewness(; me = SimpleExpectedReturns(; w = aw492b),
                                                 w = aw492), rd.X)[1],
                           coskewness(Coskewness(; w = aw492), rd.X)[1])
            # The unweighted path did not move.
            @test isapprox(cov(Covariance(), rd.X), Statistics.cov(rd.X))
            # The `mean` keyword still reaches an unweighted centre with weighted
            # deviations, which is what weights inside `ce` alone give.
            @test isapprox(cov(Covariance(; w = aw492), rd.X; mean = mu492),
                           cov(Covariance(; ce = GeneralCovariance(; w = aw492)), rd.X))
        end

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

        # A constant column never crosses its own threshold, so every Gerber denominator
        # vanishes on it: `Gerber0` divides by `(U + D)' * (U + D)`, `Gerber1` by
        # `T .- N' * N`, and `Gerber2` by an unclamped `sqrt.(diag(H))`. Routing the three
        # through `comovement_ratio` / `standardise_comovement!` guards all of them to
        # zero, which is what the Smyth-Broby family already returned. Before the routing
        # the Gerber estimators returned a NaN row here. See ADR 0022.
        # The column is zero away from the diagonal, where the guarded zero is the right
        # answer, and one on the diagonal, which `comovement_unit_diagonal!` writes. These
        # compare the reductions with `pdm = nothing` to read the guard directly.
        Xq = randn(StableRNG(987654321), 200, 4)
        Xq[:, 3] .= 0.02
        for (galg, salg) in
            ((Gerber0(), SmythBrobyCount0()), (Gerber1(), SmythBrobyCount1()),
             (Gerber2(), SmythBrobyCount2()))
            rho = cor(GerberCovariance(; alg = galg, pdm = nothing), Xq)
            sbrho = cor(SmythBrobyCovariance(; alg = salg, pdm = nothing), Xq)
            @test all(isfinite, rho)
            @test all(isfinite, sbrho)
            @test all(iszero, view(rho, [1, 2, 4], 3))
            @test isone(rho[3, 3])
            @test isone(sbrho[3, 3])
            # The threshold crossings are unchanged away from the degenerate asset.
            @test isapprox(rho[1:2, 1:2],
                           cor(GerberCovariance(; alg = galg, pdm = nothing),
                               Xq[:, [1, 2, 4]])[1:2, 1:2])
        end

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
    @testset "Gerber statistic (#454)" begin
        # ---- the counting rule, hand-counted -----------------------------------------
        # Two assets, eight observations, each column already centred exactly, so
        # `demean_returns` is a no-op and the crossings are the ones written here.
        #
        #   row | asset 1 | asset 2 | the pair is
        #   ----+---------+---------+-------------------
        #     1 |   up    |   up    | concordant
        #     2 |   up    |  down   | discordant
        #     3 |  down   |  down   | concordant
        #     4 |   up    | neutral | exactly one crossed
        #     5 | neutral |   up    | exactly one crossed
        #     6 | neutral | neutral | neither crossed
        #     7 |  down   |   up    | discordant
        #     8 |  down   |  down   | concordant
        #
        # So nconc = 3, ndisc = 2, nneut = 2 and both are neutral once. Asset 1 crosses on
        # rows 1, 2, 3, 4, 7, 8 and asset 2 on rows 1, 2, 3, 5, 7, 8, so each diagonal
        # concordant count is 6.
        Xh = [2.0 2.0; 2.0 -2.0; -2.0 -2.0; 2.0 0.0; 0.0 2.0; 0.0 0.0; -2.0 2.0; -2.0 -2.0]
        @test all(iszero, sum(Xh; dims = 1))
        ceh = GerberCovariance(; t = 0.5, pdm = nothing)
        sdh = std(ceh.ve, Xh; dims = 1)
        Uh, Dh = PortfolioOptimisers.gerber_updown(ceh, Xh, sdh)
        @test Uh == Bool[1 1; 1 0; 0 0; 1 0; 0 1; 0 0; 0 1; 0 0]
        @test Dh == Bool[0 0; 0 1; 1 1; 0 0; 0 0; 0 0; 1 0; 1 1]
        # Every threshold makes the two bands disjoint.
        @test !any(Uh .& Dh)
        UmDh = Uh - Dh
        UpDh = Uh + Dh
        nconch, ndisch = PortfolioOptimisers.concordance_counts(transpose(UmDh) * UmDh,
                                                                transpose(UpDh) * UpDh)
        @test nconch == [6.0 3.0; 3.0 6.0]
        @test ndisch == [0.0 2.0; 2.0 0.0]
        # The split is exact: it returns the difference and the sum it was given.
        @test nconch .- ndisch == transpose(UmDh) * UmDh
        @test nconch .+ ndisch == transpose(UpDh) * UpDh

        # ---- the three tags give three different matrices ------------------------------
        rh = Dict(a => cor(GerberCovariance(; t = 0.5, pdm = nothing, alg = a), Xh)
                  for a in (Gerber0(), Gerber1(), Gerber2()))
        # Gerber0 divides by nconc + ndisc = 5, Gerber1 by nconc + ndisc + nneut = 7, and
        # Gerber2 by sqrt(6 * 6) = 6.
        @test isapprox(rh[Gerber0()][1, 2], 1 / 5)
        @test isapprox(rh[Gerber1()][1, 2], 1 / 7)
        @test isapprox(rh[Gerber2()][1, 2], 1 / 6)
        # Every diagonal is unit: an asset is concordant with itself at every crossing.
        for a in (Gerber0(), Gerber1(), Gerber2())
            @test isapprox(diag(rh[a]), ones(2))
        end

        Xg = randn(StableRNG(454454454), 40, 12)
        mg = [cor(GerberCovariance(; alg = a, pdm = nothing), Xg)
              for a in (Gerber0(), Gerber1(), Gerber2())]
        @test !isapprox(mg[1], mg[2])
        @test !isapprox(mg[1], mg[3])
        @test !isapprox(mg[2], mg[3])
        # Gerber1's denominator carries nneut on top of Gerber0's, so it is never smaller
        # and the statistic is never larger in magnitude.
        @test all(abs.(mg[2]) .<= abs.(mg[1]) .+ sqrt(eps()))

        # ---- the threshold's effect at both degenerate ends ----------------------------
        # A threshold above every observation: nothing crosses, every denominator
        # vanishes, and the guard in `comovement_ratio` / `standardise_comovement!` returns
        # zero rather than a NaN. The diagonal is the one entry that is a definition rather
        # than a measurement, so `comovement_unit_diagonal!` writes one onto it and the
        # matrix stays a correlation matrix. #495, ADR 0093.
        for a in (Gerber0(), Gerber1(), Gerber2())
            rbig = cor(GerberCovariance(; t = 1e3, pdm = nothing, alg = a), Xg)
            @test all(isfinite, rbig)
            @test isapprox(rbig, I(12))
            # `posdef!` divided by the square root of the old zero diagonal and raised
            # `ArgumentError: matrix contains Infs or NaNs`. It now takes the identity.
            @test isapprox(cor(GerberCovariance(; t = 1e3, alg = a), Xg), I(12))
        end
        # A zero threshold: every observation whose centred return is not exactly zero
        # crosses, so no observation is neutral and Gerber0 and Gerber1 coincide.
        r0lo = cor(GerberCovariance(; t = 0.0, pdm = nothing, alg = Gerber0()), Xg)
        r1lo = cor(GerberCovariance(; t = 0.0, pdm = nothing, alg = Gerber1()), Xg)
        @test isapprox(r0lo, r1lo)
        @test isapprox(diag(r0lo), ones(12))
        # `t = 0` puts the band edge at zero, where `>= 0` and `<= 0` both hold at an
        # exactly zero return. The sign test in `gerber_updown` makes such a return
        # neutral instead of both up and down, so the two bands stay disjoint. `Xh`
        # carries three such returns. Its bands at `t = 0` are the bands at `t = 0.5`,
        # because every crossing of `Xh` clears both edges, so the three matrices and the
        # unit diagonal carry over unchanged. #491, ADR 0090.
        ceh0 = GerberCovariance(; t = 0.0, pdm = nothing)
        Uh0, Dh0 = PortfolioOptimisers.gerber_updown(ceh0, Xh, sdh)
        @test Uh0 == Uh
        @test Dh0 == Dh
        @test !any(Uh0 .& Dh0)
        # Rows 4, 5 and 6 carry the exactly zero returns, and each one is neutral.
        @test !any(Uh0[4:6, :] .& Dh0[4:6, :])
        @test Uh0[6, :] == Dh0[6, :] == Bool[0, 0]
        for a in (Gerber0(), Gerber1(), Gerber2())
            rh0 = cor(GerberCovariance(; t = 0.0, pdm = nothing, alg = a), Xh)
            @test isapprox(rh0, rh[a])
            @test isapprox(diag(rh0), ones(2))
        end

        # ---- the cor and cov pair -------------------------------------------------------
        sdg = vec(std(GerberCovariance().ve, Xg; dims = 1))
        for a in (Gerber0(), Gerber1(), Gerber2())
            ceg = GerberCovariance(; alg = a, pdm = nothing)
            rho = cor(ceg, Xg)
            sigma = cov(ceg, Xg)
            @test isapprox(sigma, rho .* (sdg * transpose(sdg)))
            @test isapprox(diag(sigma), sdg .^ 2)
            @test isapprox(StatsBase.cov2cor(sigma), rho)
        end

        # ---- positive definiteness -------------------------------------------------------
        # A Gerber matrix is a matrix of pairwise votes, so it is not positive definite in
        # general. `Gerber0` is indefinite on this sample, and `pdm` repairs it.
        rho_raw = cor(GerberCovariance(; alg = Gerber0(), pdm = nothing), Xg)
        @test minimum(eigvals(Symmetric(rho_raw))) < 0
        rho_pdm = cor(GerberCovariance(; alg = Gerber0(), pdm = Posdef()), Xg)
        @test minimum(eigvals(Symmetric(rho_pdm))) >= 0
        @test !isapprox(rho_raw, rho_pdm)
    end
    @testset "Gerber IQ zero threshold (#498)" begin
        # `c = 0` puts the noise threshold at zero. The gate `abs(x) < 0` then drops
        # nothing, and the closed comparison `abs(x) >= 0` holds for a return of exactly
        # zero. Such a return crossed on both axes but carried no sign, so neither sign
        # test fired and it fell through to the neutral accumulator. `Gerber1` divides by
        # that accumulator, and its diagonal was then not one. `iq_crossed` gives this
        # family the rule ADR 0090 gave `GerberCovariance`: a return of exactly zero never
        # crosses. #498.

        # ---- the crossing predicate ------------------------------------------------------
        # The sign test binds only at a zero threshold. For every positive `c` the
        # predicate is the closed comparison it replaces, so no code path at a positive
        # threshold can move.
        for c in (0.5, 1.0, 2.0), x in (-2.0, -0.5, 0.0, 0.5, 2.0)
            @test PortfolioOptimisers.iq_crossed(x, abs(x), c) == (abs(x) >= c)
        end
        for x in (-2.0, -0.5, 0.5, 2.0)
            @test PortfolioOptimisers.iq_crossed(x, abs(x), 0.0)
        end
        @test !PortfolioOptimisers.iq_crossed(0.0, 0.0, 0.0)

        # ---- the sample of the ticket ----------------------------------------------------
        # Column 1 is centred exactly, so rows 2 and 5 carry an exactly zero return for
        # asset 1. Column 2 carries none.
        Xq = [1.0 2.0; 0.0 0.0; -1.0 3.0; 2.0 -1.0; 0.0 1.0; -2.0 -2.0]
        ceq(a) = GerberIQCovariance(; c = 0.0, kind = BasicGerberIQ(; d = 0.0, n = 1.0),
                                    sc = AssetVolatilityGerberIQScaler(),
                                    decay = ExpGerberIQDecay(; e = 0.0, y = 0.0),
                                    pdm = nothing, alg = a)
        for a in (Gerber0(), Gerber1(), Gerber2())
            @test isapprox(diag(cor(ceq(a), Xq)), ones(2))
        end
        # ---- the classification at a zero threshold --------------------------------------
        # The neutral accumulator is not emptied, it is corrected: it holds the
        # observations on which exactly one asset crossed, which is what its own docstring
        # says. Drive the kernel directly at `c = 0`, with unit standard deviations so the
        # pair's scaled thresholds are zero.
        polq = PortfolioOptimisers.GerberIQKernel(Gerber1(),
                                                  BasicGerberIQ(; d = 0.0, n = 1.0),
                                                  ExpGerberIQDecay(; e = 0.0, y = 0.0),
                                                  AssetVolatilityGerberIQScaler(), 0.0,
                                                  [1.0, 1.0])
        stq = PortfolioOptimisers.comovement_pair_state(polq, 1, 2)
        @test stq.ci == stq.cj == 0.0
        accq = (pos = 0.0, neg = 0.0, nn = 0.0, cpos = 0, cneg = 0, cnn = 0)
        # Neither asset crosses, so the observation leaves the pair entirely.
        @test PortfolioOptimisers.comovement_step(polq, accq, stq, 0.0, 0.0, 4, 1) == accq
        # Exactly one asset crosses, so the observation is neutral.
        accn = PortfolioOptimisers.comovement_step(polq, accq, stq, 0.0, 1.0, 4, 1)
        @test accn.nn > 0
        @test accn.pos == accn.neg == 0
        # Both assets cross, so their product carries a sign and the observation is
        # concordant or discordant. It can no longer reach the neutral accumulator.
        accp = PortfolioOptimisers.comovement_step(polq, accq, stq, 1.0, 1.0, 4, 1)
        @test accp.pos > 0
        @test accp.neg == accp.nn == 0
        accd = PortfolioOptimisers.comovement_step(polq, accq, stq, 1.0, -1.0, 4, 1)
        @test accd.neg > 0
        @test accd.pos == accd.nn == 0

        # ---- the reduction to the Gerber statistic ----------------------------------------
        # Every weight one, no decay and the per-asset volatility scaling make the Gerber
        # IQ statistic the Gerber statistic. The reduction holds at `c = 0` as it holds at
        # a positive threshold, and it is what the defect broke: `Gerber1` disagreed at a
        # zero threshold while the other two branches, which read no neutral count, agreed.
        Yq = randn(StableRNG(987654321), 60, 5)
        Yq[3, 2] = 0.0
        for t in (0.0, 0.5), a in (Gerber0(), Gerber1(), Gerber2())
            cgq = GerberCovariance(; t = t, pdm = nothing, alg = a)
            ciq = GerberIQCovariance(; c = t, kind = BasicGerberIQ(; d = t, n = 1.0),
                                     sc = AssetVolatilityGerberIQScaler(),
                                     decay = ExpGerberIQDecay(; e = 0.0, y = 0.0),
                                     pdm = nothing, alg = a)
            @test isapprox(cor(cgq, Yq), cor(ciq, Yq))
        end
    end
    @testset "Co-movement unit diagonal (#495)" begin
        # An asset whose every return sits inside its own noise zone crosses no threshold.
        # `comovement_ratio` then finds a zero denominator for every pair that asset
        # belongs to and returns the guarded zero. That zero is right off the diagonal,
        # where the pair has no measured co-movement, and wrong on the diagonal, where a
        # correlation is one by definition. `posdef!` divided by the square root of that
        # zero and raised `ArgumentError: matrix contains Infs or NaNs`, naming neither the
        # asset nor the cause. `comovement_unit_diagonal!` writes the diagonal after the
        # reduction and before the repair. #495, ADR 0093.

        # A sample whose sixth asset crosses `t = 1.5` on no observation.
        rngq = StableRNG(20260824)
        Xq = nothing
        for _ in 1:400
            Xt = randn(rngq, 14, 6)
            Xtc = Xt .- mean(Xt; dims = 1)
            sdt = vec(std(Xtc; dims = 1, corrected = true))
            if any(i -> count(k -> abs(Xtc[k, i]) >= 1.5 * sdt[i], axes(Xtc, 1)) == 0,
                   axes(Xtc, 2))
                Xq = Xt
                break
            end
        end
        @test !isnothing(Xq)
        Xqc = Xq .- mean(Xq; dims = 1)
        sdq = vec(std(Xqc; dims = 1, corrected = true))
        crossings = [count(k -> abs(Xqc[k, i]) >= 1.5 * sdq[i], axes(Xqc, 1))
                     for i in axes(Xqc, 2)]
        @test crossings == [1, 1, 3, 2, 3, 0]

        galgs = (Gerber0(), Gerber1(), Gerber2())
        for a in galgs
            # The quiet asset gets a zero row: it has no measured co-movement with anyone.
            # Its diagonal entry is one, so the matrix is a correlation matrix.
            for ce in (GerberCovariance(; t = 1.5, alg = a, pdm = nothing),
                       GerberIQCovariance(; c = 1.5, alg = a, pdm = nothing))
                rq = cor(ce, Xq)
                @test isapprox(diag(rq), ones(6))
                @test all(iszero, rq[1:5, 6])
                @test all(iszero, rq[6, 1:5])
            end
            # The default `pdm` raised before the fix. It now answers a repaired matrix.
            @test isposdef(cor(GerberCovariance(; t = 1.5, alg = a), Xq))
            @test isposdef(cor(GerberIQCovariance(; c = 1.5, alg = a), Xq))
        end

        # The Smyth-Broby family reaches the same zero through the confusion zone. `c1`
        # here excludes every observation of the FIRST asset and none of any other, so its
        # diagonal pair qualifies nothing while its off-diagonal pairs still vote.
        ratios = [maximum(abs, Xq[:, i]) / std(Xq[:, i]) for i in axes(Xq, 2)]
        @test count(<(1.885), ratios) == 1
        @test ratios[1] < 1.885
        sbalgs = (SmythBroby0(), SmythBroby1(), SmythBroby2(), SmythBrobyGerber0(),
                  SmythBrobyGerber1(), SmythBrobyGerber2(), SmythBrobyCount0(),
                  SmythBrobyCount1(), SmythBrobyCount2())
        for a in sbalgs
            rq = cor(SmythBrobyCovariance(; c1 = 1.885, alg = a, pdm = nothing), Xq)
            @test isapprox(diag(rq), ones(6))
            @test isposdef(cor(SmythBrobyCovariance(; c1 = 1.885, alg = a), Xq))
        end

        # ---- the write is a no-op wherever the asset votes at least once ----------------
        # A sample with no quiet asset answers bit for bit what it answered before, so
        # applying the function to the answer changes nothing at all. The `2` markers are
        # the reason the write is guarded rather than unconditional: their diagonal is one
        # to within a unit in the last place, not exactly one.
        Xu = randn(StableRNG(495495495), 60, 8)
        for a in galgs
            for ce in (GerberCovariance(; alg = a, pdm = nothing),
                       GerberIQCovariance(; alg = a, pdm = nothing))
                ru = cor(ce, Xu)
                @test isapprox(diag(ru), ones(8))
                rc = copy(ru)
                PortfolioOptimisers.comovement_unit_diagonal!(rc)
                @test rc == ru
            end
        end
        for a in sbalgs
            ru = cor(SmythBrobyCovariance(; alg = a, pdm = nothing), Xu)
            @test isapprox(diag(ru), ones(8))
            rc = copy(ru)
            PortfolioOptimisers.comovement_unit_diagonal!(rc)
            @test rc == ru
        end
        # The `0` and `1` markers divide a value by itself, so their diagonal is EXACTLY
        # one. The `2` markers divide by the square root of that value twice, and on this
        # sample the rounding shows. An unconditional write would move `posdef!` onto its
        # other branch for such a matrix.
        for a in (Gerber0(), Gerber1())
            @test diag(cor(GerberCovariance(; alg = a, pdm = nothing), Xu)) == ones(8)
        end
        @test diag(cor(GerberCovariance(; alg = Gerber2(), pdm = nothing), Xu)) != ones(8)
        @test isapprox(diag(cor(GerberCovariance(; alg = Gerber2(), pdm = nothing), Xu)),
                       ones(8))

        # ---- the function writes a zero diagonal entry and nothing else -----------------
        # The `iszero` guard is load-bearing, not decoration. `posdef!` reads its diagonal
        # with an exact `isone` test, and a `Gerber2` diagonal is one to within a unit in
        # the last place rather than exactly one. Writing an exact one there flips that
        # test, drops the `cov2cor!` clamp, and moves a pinned covariance on a sample with
        # no degenerate asset. So the entry that already reads as one is left alone.
        rw = reshape(collect(1.0:9.0), 3, 3)
        rw[2, 2] = 0.0
        rw[3, 3] = nextfloat(1.0)
        rb = copy(rw)
        @test isnothing(PortfolioOptimisers.comovement_unit_diagonal!(rw))
        @test rw[1, 1] == rb[1, 1]
        @test rw[2, 2] == 1.0
        @test rw[3, 3] === nextfloat(1.0)
        @test [rw[i, j] for i in 1:3 for j in 1:3 if i != j] ==
              [rb[i, j] for i in 1:3 for j in 1:3 if i != j]

        # ---- the covariance path keeps its variance diagonal ---------------------------
        # `cor2cov!` scaled the old zero diagonal to the variance anyway, so the covariance
        # answer does not move. It now follows from a unit correlation diagonal.
        sdqr = vec(std(GerberCovariance().ve, Xq; dims = 1))
        for a in galgs
            @test isapprox(diag(cov(GerberCovariance(; t = 1.5, alg = a, pdm = nothing),
                                    Xq)), sdqr .^ 2)
        end
    end
    @testset "Smyth-Broby zero indecision zone (#499)" begin
        # `c2 = 0` puts the indecision zone at zero. The gate `ari < 0` then rejects
        # nothing, and the closed comparison `ari >= 0` holds for a centred return of
        # exactly zero. Such a return crossed on both axes but carried no sign, so neither
        # sign test fired and it fell through to the neutral accumulator. The three `*1`
        # markers divide by that accumulator, and their diagonal was then not one.
        # `sb_crossed` gives this family the rule ADR 0090 gave `GerberCovariance` and
        # `GerberIQCovariance`: a centred return of exactly zero never crosses. #499.

        # ---- the crossing predicate ------------------------------------------------------
        # The sign test binds only at a zero threshold. For every positive `c2` the
        # predicate is the closed comparison it replaces, so no code path at a positive
        # threshold can move.
        for c in (0.5, 1.0, 2.0), r in (-2.0, -0.5, 0.0, 0.5, 2.0)
            @test PortfolioOptimisers.sb_crossed(r, abs(r), c) == (abs(r) >= c)
        end
        for r in (-2.0, -0.5, 0.5, 2.0)
            @test PortfolioOptimisers.sb_crossed(r, abs(r), 0.0)
        end
        @test !PortfolioOptimisers.sb_crossed(0.0, 0.0, 0.0)

        # ---- the sample of the ticket ----------------------------------------------------
        # Both column means are exactly zero, so rows 3 and 4 carry an exactly zero centred
        # return for asset 1. `c3` is wide enough to admit every observation.
        Xs = [2.0 2.0; -2.0 -2.0; 0.0 1.0; 0.0 -1.0]
        ces(a) = SmythBrobyCovariance(; alg = a, c1 = 0.0, c2 = 0.0, c3 = 1e6,
                                      pdm = nothing,
                                      me = SimpleExpectedReturns(; w = nothing))
        salgs = (SmythBroby0(), SmythBroby1(), SmythBroby2(), SmythBrobyGerber0(),
                 SmythBrobyGerber1(), SmythBrobyGerber2(), SmythBrobyCount0(),
                 SmythBrobyCount1(), SmythBrobyCount2())
        for a in salgs
            @test isapprox(diag(cor(ces(a), Xs)), ones(2))
        end

        # ---- the classification at a zero indecision zone --------------------------------
        # The neutral accumulator is not emptied, it is corrected: it holds the
        # observations on which exactly one asset crossed, which is what its own docstring
        # says. Drive the kernel directly at `c2 = 0`, with zero means and unit standard
        # deviations, so the centred standardised return is the raw one.
        pols = PortfolioOptimisers.SmythBrobyKernel(SmythBroby1(), [0.0, 0.0], [1.0, 1.0],
                                                    0.0, 0.0, 1e6, 2.0)
        sts = PortfolioOptimisers.comovement_pair_state(pols, 1, 2)
        @test sts.c1i == sts.c1j == 0.0
        @test sts.mui == sts.muj == 0.0
        accs = (pos = 0.0, neg = 0.0, nn = 0.0, cpos = 0, cneg = 0, cnn = 0)
        # Neither asset crosses, so the observation leaves the pair entirely.
        @test PortfolioOptimisers.comovement_step(pols, accs, sts, 0.0, 0.0, 4, 1) == accs
        # Exactly one asset crosses, so the observation is neutral.
        accsn = PortfolioOptimisers.comovement_step(pols, accs, sts, 0.0, 1.0, 4, 1)
        @test accsn.nn > 0
        @test accsn.pos == accsn.neg == 0
        # Both assets cross, so their product carries a sign and the observation is
        # concordant or discordant. It can no longer reach the neutral accumulator.
        accsp = PortfolioOptimisers.comovement_step(pols, accs, sts, 1.0, 1.0, 4, 1)
        @test accsp.pos > 0
        @test accsp.neg == accsp.nn == 0
        accsd = PortfolioOptimisers.comovement_step(pols, accs, sts, 1.0, -1.0, 4, 1)
        @test accsd.neg > 0
        @test accsd.pos == accsd.nn == 0

        # ---- the confusion zone keeps its own meaning ------------------------------------
        # The rule binds on the centred quantity alone. `c1` reads the raw, uncentred
        # return, so `c1 = 0` still means "no confusion zone" and rejects nothing, even an
        # observation on which both raw returns are exactly zero. Give the pair a non-zero
        # mean, so the two raw zeros centre to a return that does cross.
        polc = PortfolioOptimisers.SmythBrobyKernel(SmythBroby1(), [1.0, 1.0], [1.0, 1.0],
                                                    0.0, 0.0, 1e6, 2.0)
        stc = PortfolioOptimisers.comovement_pair_state(polc, 1, 2)
        @test stc.c1i == stc.c1j == 0.0
        accc = PortfolioOptimisers.comovement_step(polc, accs, stc, 0.0, 0.0, 4, 1)
        @test accc.pos > 0
        @test accc.neg == accc.nn == 0

        # ---- a positive threshold keeps its diagonal too ---------------------------------
        # The predicate loop above is the proof that no path at a positive `c2` moves: it
        # is the closed comparison it replaces. This pins the contract the rule protects,
        # on the same sample and at a positive threshold. Both column sums of `Xs` are
        # exactly zero, so the two zero entries of column 1 centre to exactly zero.
        for c2 in (0.25, 0.5, 1.0), a in salgs
            ce1 = SmythBrobyCovariance(; alg = a, c1 = 0.0, c2 = c2, c3 = 1e6,
                                       pdm = nothing,
                                       me = SimpleExpectedReturns(; w = nothing))
            @test isapprox(diag(cor(ce1, Xs)), ones(2))
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
               StepwiseRegression(; alg = ForwardSelection(), crit = :aic),
               StepwiseRegression(; alg = ForwardSelection(), crit = :aicc),
               StepwiseRegression(; alg = ForwardSelection(), crit = :bic),
               StepwiseRegression(; alg = ForwardSelection(), crit = :r2),
               StepwiseRegression(; alg = ForwardSelection(), crit = :adjr2),
               StepwiseRegression(; alg = BackwardElimination()),
               StepwiseRegression(; alg = BackwardElimination(), crit = :aic),
               StepwiseRegression(; alg = BackwardElimination(), crit = :aicc),
               StepwiseRegression(; alg = BackwardElimination(), crit = :bic),
               StepwiseRegression(; alg = BackwardElimination(), crit = :r2),
               StepwiseRegression(; alg = BackwardElimination(), crit = :adjr2),
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
    @testset "Regression criteria under a GeneralisedLinearModel target (#399)" begin
        # Every criterion must score a fitted GeneralisedLinearModel. `:r2` and `:adjr2`
        # used to throw a MethodError, because GLM defines them for a LinearModel only.
        for crit in PortfolioOptimisers.STEPWISE_REGRESSION_CRITERIA,
            alg in (ForwardSelection(), BackwardElimination())

            re = StepwiseRegression(; crit = crit, alg = alg,
                                    tgt = GeneralisedLinearModel())
            @test isa(regression(re, rd), Regression)
        end
        # The default variant is :devianceratio, which matches the classical R² of the
        # LinearModel path for the default Normal() response.
        for crit in PortfolioOptimisers.MAX_VAL_STEPWISE_REGRESSION_CRITERIA
            r1 = regression(StepwiseRegression(; crit = crit,
                                               tgt = GeneralisedLinearModel()), rd)
            r2 = regression(StepwiseRegression(; crit = crit,
                                               tgt = GeneralisedLinearModel(;
                                                                            variant = :devianceratio)),
                            rd)
            @test isapprox(r1.M, r2.M)
        end
        # Every variant of the wider set is accepted by :r2.
        for variant in PortfolioOptimisers.PSEUDO_R2_VARIANTS
            re = StepwiseRegression(; crit = :r2,
                                    tgt = GeneralisedLinearModel(; variant = variant))
            @test isa(regression(re, rd), Regression)
        end
        # :adjr2 accepts the narrower set only.
        for variant in PortfolioOptimisers.ADJUSTED_PSEUDO_R2_VARIANTS
            re = StepwiseRegression(; crit = :adjr2,
                                    tgt = GeneralisedLinearModel(; variant = variant))
            @test isa(regression(re, rd), Regression)
        end
        for variant in setdiff(PortfolioOptimisers.PSEUDO_R2_VARIANTS,
                               PortfolioOptimisers.ADJUSTED_PSEUDO_R2_VARIANTS)
            @test_throws ArgumentError StepwiseRegression(; crit = :adjr2,
                                                          tgt = GeneralisedLinearModel(;
                                                                                       variant = variant))
        end
        # The criterion symbol and the variant symbol are both checked at construction.
        @test_throws ArgumentError StepwiseRegression(; crit = :nonesuch)
        @test_throws ArgumentError GeneralisedLinearModel(; variant = :nonesuch)
        # The criterion is stored as a Val, so the estimator stays type stable.
        @test isa(StepwiseRegression(; crit = :aic).crit,
                  PortfolioOptimisers.MinValStepwiseRegressionCriterion)
        @test isa(StepwiseRegression(; crit = :r2).crit,
                  PortfolioOptimisers.MaxValStepwiseRegressionCriterion)
        # factory carries the variant through.
        @test PortfolioOptimisers.factory(GeneralisedLinearModel(; variant = :McFadden),
                                          pw).variant == :McFadden
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
    @testset "Every distance algorithm has an exactly zero diagonal" begin
        #=
        A distance matrix with a non-zero diagonal is not a distance matrix: `SimpleWeightedGraph`
        reads it as a self-loop and `PhylogenyResult` rejects it outright. Two sources were
        measured, and neither was a rounding speck that could be left alone.

        `variation_info` estimated `VI(X, X)` instead of pinning it. It is zero by definition,
        but the histogram estimate of `I(X; X)` does not reproduce the estimate of `H(X)` bit
        for bit, so it left roughly 1e-16 on 7 of 12 assets.

        `ShrunkDenoise` -- the *default* denoise algorithm -- reconstructs from eigen components
        and, unlike `SpectralDenoise` and `FixedDenoise`, does not route through `cov2cor`, so
        the correlation diagonal came out at `1 ± 1.5e-15`. That one matters more than it looks:
        the correlation kernels take `sqrt(1 - rho[i, i])`, which **amplifies** 1.1e-16 into
        7.45e-9 -- a real self-loop weight, not noise.
        =#
        Xd = rd.X
        for a in (SimpleDistance(), SimpleAbsoluteDistance(), LogDistance(),
                  CorrelationDistance(), CanonicalDistance(), VariationInfoDistance()),
            c in (Covariance(),
                  PortfolioOptimisersCovariance(;
                                                mp = PortfolioOptimisers.MatrixProcessing(;
                                                                                          dn = Denoise())),
                  PortfolioOptimisersCovariance(;
                                                mp = PortfolioOptimisers.MatrixProcessing(;
                                                                                          dt = Detone())))

            @test all(iszero, diag(distance(Distance(; alg = a), c, Xd)))
        end
        # The correlation diagonal is one by definition, and `Denoise` now leaves it exact.
        cdn = PortfolioOptimisersCovariance(;
                                            mp = PortfolioOptimisers.MatrixProcessing(;
                                                                                      dn = Denoise()))
        rdn = cor(cdn, Xd)
        @test all(isone, diag(rdn))
        @test isposdef(rdn)
        # `variation_info`'s diagonal is pinned, not estimated -- under both normalisations.
        @test all(iszero, diag(PortfolioOptimisers.variation_info(Xd)))
        @test all(iszero, diag(PortfolioOptimisers.variation_info(Xd, Knuth(), false)))
        # `mutual_variation_info` pins its VI diagonal but keeps its MI one: `I(X; X) = H(X)`
        # is a real value there, unlike VI's zero.
        mm, vm = PortfolioOptimisers.mutual_variation_info(Xd)
        @test all(iszero, diag(vm))
        @test all(!iszero, diag(mm))
    end
    @testset "The distance sweep of #448" begin
        #=
        Every claim the two distance files make, pinned by running it. The four closed forms,
        the two identities that relate them, the `_absguard` boundary, the clamp, the
        `_as_correlation` round trip, every `CanonicalDistance` redirect, and the agreement of
        `cor_and_dist` with `distance`.
        =#
        rngd = StableRNG(987654321)
        Xs = randn(rngd, 200, 5)
        rhos = cor(Xs)
        # --- the four closed forms, against the formula written by hand ---
        hS(r, p) =
            if isnothing(p)
                sqrt(clamp((1 - r) / 2, 0, 1))
            else
                sqrt(clamp((1 - r^p) * (isodd(p) ? 0.5 : 1.0), 0, 1))
            end
        hSA(r, p) =
            if isnothing(p)
                sqrt(clamp(1 - abs(r), 0, 1))
            else
                sqrt(clamp(1 - abs(r)^p, 0, 1))
            end
        hL(r, p) = isnothing(p) ? max(-log(abs(r)), 0) : max(-log(abs(r)^p), 0)
        hC(r, p) = isnothing(p) ? sqrt(clamp(1 - r, 0, 1)) : sqrt(clamp(1 - r^p, 0, 1))
        for (a, h) in
            ((SimpleDistance(), hS), (SimpleAbsoluteDistance(), hSA), (LogDistance(), hL),
             (CorrelationDistance(), hC)), p in (nothing, 1, 2, 3, 4)

            R = isa(a, Union{SimpleAbsoluteDistance, LogDistance}) ? abs.(rhos) : rhos
            @test distance(Distance(; power = p, alg = a), rhos) ==
                  [h(R[i, j], p) for i in axes(R, 1), j in axes(R, 2)]
        end
        # --- `power = 1` is the neutral position of the knob, for every algorithm ---
        for a in (SimpleDistance(), SimpleAbsoluteDistance(), LogDistance(),
                  CorrelationDistance())
            @test distance(Distance(; power = 1, alg = a), rhos) ==
                  distance(Distance(; alg = a), rhos)
        end
        # --- `SimpleDistance`'s scale is 1 / (1 - min rho^p) over |rho| <= 1 ---
        for p in 1:6
            b = [-1.0, 0.0, 1.0] .^ p
            sc = 1 / (1 - minimum(b))
            @test distance(Distance(; power = p, alg = SimpleDistance()), rhos) ==
                  sqrt.(clamp.(sc .* (1 .- rhos .^ p), 0, 1))
        end
        # --- the two identities, on a non-negative correlation where they hold ---
        rhop = abs.(rhos)
        DS = distance(Distance(; alg = SimpleDistance()), rhop)
        DC = distance(Distance(; alg = CorrelationDistance()), rhop)
        DA = distance(Distance(; alg = SimpleAbsoluteDistance()), rhop)
        @test isapprox(DC, sqrt(2) .* DS)
        @test DC == DA
        # --- and their failure on the negative half: `CorrelationDistance` saturates at 1 ---
        DCs = distance(Distance(; alg = CorrelationDistance()), rhos)
        neg = rhos .< 0
        @test any(neg)
        @test all(isone, DCs[neg])
        @test all(!isone, sqrt.(1 .- rhos[neg]))
        # --- `_absguard` is an allocation guard, not a branch in the mathematics ---
        nonneg = [1.0 0.5; 0.5 1.0]
        withzero = [1.0 0.0; 0.0 1.0]
        signed = [1.0 -0.5; -0.5 1.0]
        @test PortfolioOptimisers._absguard(nonneg) === nonneg
        @test PortfolioOptimisers._absguard(withzero) === withzero
        @test PortfolioOptimisers._absguard(signed) !== signed
        for M in (nonneg, withzero, signed, rhos, [1.0 -0.0; -0.0 1.0])
            @test PortfolioOptimisers._absguard(M) == abs.(M)
        end
        # --- `LogDistance` at an exactly-zero correlation is `Inf`, and `max` does not hide it ---
        @test distance(Distance(; alg = LogDistance()), withzero)[1, 2] == Inf
        @test distance(Distance(; power = 3, alg = LogDistance()), withzero)[1, 2] == Inf
        # --- the guard reads the magnitude: -0.5 and +0.5 give the same distance ---
        @test distance(Distance(; alg = LogDistance()), signed) ==
              distance(Distance(; alg = LogDistance()), nonneg)
        # --- the clamp keeps a correlation from round-off a real number ---
        for alg in (SimpleDistance(), SimpleAbsoluteDistance(), LogDistance(),
                    CorrelationDistance())
            for M in ([1.0 1+1e-12; 1+1e-12 1.0], [1.0 -1-1e-12; -1-1e-12 1.0])
                D = distance(Distance(; alg = alg), M)
                @test all(isfinite, D)
                @test all(>=(0), D)
            end
        end
        # --- `_as_correlation` round-trips, mutates nothing, and passes a correlation through ---
        sdv = [2.0, 0.5, 3.0, 1.5, 0.25]
        Sig = StatsBase.cor2cov(rhos, sdv)
        Sigkeep = copy(Sig)
        @test isapprox(PortfolioOptimisers._as_correlation(Sig), rhos)
        @test Sig == Sigkeep
        @test PortfolioOptimisers._as_correlation(rhos) === rhos
        for a in (SimpleDistance(), SimpleAbsoluteDistance(), LogDistance(),
                  CorrelationDistance()), p in (nothing, 2, 3)

            @test isapprox(distance(Distance(; power = p, alg = a), Sig),
                           distance(Distance(; power = p, alg = a), rhos))
        end
        # --- the element type survives the transform, `SimpleDistance` included ---
        rho32 = Float32.(rhos)
        for a in (SimpleDistance(), SimpleAbsoluteDistance(), LogDistance(),
                  CorrelationDistance()), p in (nothing, 2, 3)

            @test eltype(distance(Distance(; power = p, alg = a), rho32)) === Float32
        end
        @test eltype(distance(Distance(; alg = SimpleDistance()), [1 0; 0 1])) === Float64
        # --- every `CanonicalDistance` redirect, wrapped and bare ---
        cemi = MutualInfoCovariance(; bins = 7, normalise = false)
        celtd = LowerTailDependenceCovariance()
        cedc = DistanceCovariance()
        cepl = Covariance()
        can = Distance(; alg = CanonicalDistance())
        for (ce, expect) in ((cemi, VariationInfoDistance(; bins = 7, normalise = false)),
                             (PortfolioOptimisersCovariance(; ce = cemi),
                              VariationInfoDistance(; bins = 7, normalise = false)), (celtd, LogDistance()),
                             (PortfolioOptimisersCovariance(; ce = celtd), LogDistance()),
                             (cedc, CorrelationDistance()),
                             (PortfolioOptimisersCovariance(; ce = cedc), CorrelationDistance()),
                             (cepl, SimpleDistance()),
                             (PortfolioOptimisersCovariance(; ce = cepl), SimpleDistance()))
            @test distance(can, ce, Xs) == distance(Distance(; alg = expect), ce, Xs)
        end
        # --- the `bins` and `normalise` copy is load-bearing ---
        Ddefault = distance(Distance(; alg = VariationInfoDistance()), cemi, Xs)
        @test !isapprox(distance(can, cemi, Xs), Ddefault)
        @test !isapprox(distance(can, PortfolioOptimisersCovariance(; ce = cemi), Xs),
                        Ddefault)
        # --- `cor_and_dist` returns the `D` that `distance` returns, on every route ---
        for ce in (cepl, cemi, PortfolioOptimisersCovariance(; ce = cemi), celtd,
                   PortfolioOptimisersCovariance(; ce = celtd), cedc,
                   PortfolioOptimisersCovariance(; ce = cedc),
                   PortfolioOptimisersCovariance(; ce = cepl)),
            de in (Distance(; alg = SimpleDistance()),
                   Distance(; power = 3, alg = SimpleAbsoluteDistance()), can,
                   Distance(; power = 2, alg = CanonicalDistance()))

            r, D = cor_and_dist(de, ce, Xs)
            @test D == distance(de, ce, Xs)
            @test r == cor(ce, Xs)
        end
        # --- `dims` on the variation-of-information route, and the `power` it raises ---
        devi = Distance(; alg = VariationInfoDistance(; bins = 7, normalise = false))
        devi3 = Distance(; power = 3,
                         alg = VariationInfoDistance(; bins = 7, normalise = false))
        D1 = distance(devi, nothing, Xs; dims = 1)
        @test D1 == distance(devi, nothing, permutedims(Xs); dims = 2)
        @test D1 == PortfolioOptimisers.variation_info(Xs, 7, false)
        @test distance(devi3, nothing, Xs; dims = 1) == D1 .^ 3
        @test isapprox(distance(Distance(), cepl, Xs; dims = 1),
                       distance(Distance(), cepl, permutedims(Xs); dims = 2))
        @test_throws DomainError distance(devi, nothing, Xs; dims = 3)
        @test_throws DomainError cor_and_dist(devi, cepl, Xs; dims = 3)
        @test_throws DomainError distance(Distance(), cepl, Xs; dims = 3)
        @test_throws DomainError cor_and_dist(Distance(), cepl, Xs; dims = 3)
        # --- the constructor refuses a power below one ---
        @test_throws DomainError Distance(; power = 0)
        @test_throws DomainError Distance(; power = -2)
    end
    @testset "The distance-of-distances sweep of #449" begin
        #=
        Every claim `src/09_Distance/03_DistanceDistance.jl` makes, pinned by running it:
        both entry points against the two steps computed by hand, the agreement of
        `cor_and_dist` with `distance`, the `CanonicalDistance` redirect inside step 1, the
        `power = 1` identity, and the `args`/`kwargs` fields that no test had reached.
        =#
        Dist = PortfolioOptimisers.Distances
        rngdd = StableRNG(987)
        Xd = randn(rngdd, 300, 6)
        ced = PortfolioOptimisersCovariance()
        Cd = cor(ced, Xd)
        # --- step 1 then step 2, by hand, on both entry points ---
        for (power, alg) in
            ((nothing, SimpleDistance()), (2, SimpleAbsoluteDistance()), (3, LogDistance()),
             (nothing, CorrelationDistance()))
            dd = DistanceDistance(; power = power, alg = alg)
            base = Distance(; power = power, alg = alg)
            @test distance(dd, ced, Xd) ==
                  Dist.pairwise(Dist.Euclidean(), distance(base, ced, Xd))
            @test distance(dd, Cd) == Dist.pairwise(Dist.Euclidean(), distance(base, Cd))
            # `cor_and_dist` returns the base estimator's own correlation beside the
            # matrix `distance` returns for the same arguments.
            rhod, Dd = cor_and_dist(dd, ced, Xd)
            @test Dd == distance(dd, ced, Xd)
            @test rhod == cor(ced, Xd)
        end
        # --- the `metric` field reaches `pairwise`, and changes the answer ---
        Db = distance(Distance(), Cd)
        ddm = DistanceDistance(; metric = Dist.Minkowski(3.0))
        @test distance(ddm, Cd) == Dist.pairwise(Dist.Minkowski(3.0), Db)
        @test distance(ddm, Cd) != distance(DistanceDistance(), Cd)
        #=
        `args` and `kwargs` are fields of the estimator and they reach `pairwise` itself.
        The positional `args...` and `kwargs...` of the method go the other way, to the
        base distance of step 1, which ignores the positional ones.
        =#
        for dims in (1, 2)
            ddk = DistanceDistance(; kwargs = (; dims = dims))
            @test distance(ddk, Cd) == Dist.pairwise(Dist.Euclidean(), Db; dims = dims)
        end
        # A base distance matrix is symmetric, so which axis `pairwise` reads as the
        # observation does not change the answer.
        @test issymmetric(Db)
        @test Dist.pairwise(Dist.Euclidean(), Db; dims = 1) ==
              Dist.pairwise(Dist.Euclidean(), Db; dims = 2)
        # `args` supplies the second matrix `pairwise` takes, where the axis does matter.
        Bd = Db[:, 1:3]
        dda = DistanceDistance(; args = (Bd,), kwargs = (; dims = 2))
        @test distance(dda, Cd) == Dist.pairwise(Dist.Euclidean(), Db, Bd; dims = 2)
        @test size(distance(dda, Cd)) == (6, 3)
        # --- the `CanonicalDistance` redirect happens inside step 1 ---
        cemi = MutualInfoCovariance()
        ddc = DistanceDistance(; alg = CanonicalDistance())
        @test distance(ddc, cemi, Xd) == Dist.pairwise(Dist.Euclidean(),
                                                       distance(Distance(; alg = CanonicalDistance()), cemi, Xd))
        # There is no covariance estimator on the matrix route, so the redirect table
        # cannot select a row and its `SimpleDistance` fallback is taken.
        @test distance(ddc, Cd) == distance(DistanceDistance(; alg = SimpleDistance()), Cd)
        # --- `power = 1` reproduces the base distance, so it equals `power = nothing` ---
        @test distance(DistanceDistance(; power = 1), Cd) ==
              distance(DistanceDistance(; power = nothing), Cd)
        @test distance(DistanceDistance(; power = 2), Cd) !=
              distance(DistanceDistance(; power = nothing), Cd)
        # --- the constructor refuses a power below one ---
        @test_throws DomainError DistanceDistance(; power = 0)
        @test_throws DomainError DistanceDistance(; power = -2)
        #=
        The Euclidean norm of two columns of a bounded distance matrix is not itself
        bounded by one, which is why `ComplementSimilarity` is out of domain against this
        estimator's own default.
        =#
        @test maximum(distance(DistanceDistance(), Cd)) > 1
    end
    @testset "The covariance base sweep of #453" begin
        #=
        Every claim the covariance base and its four wrappers make, pinned by running it.
        The variance divisor under each weight type, the semi-moment target and its
        divisor, the composition order of the wrappers, the correlation the
        `CorrelationCovariance` answers with, and the tie policy of
        `find_uncorrelated_indices`.
        =#
        # --- the variance divisor: `corrected` and the weight type fix `c`, not `ve` ---
        vs = [1.0, 2.0, 4.0]
        Tv = length(vs)
        muv = sum(vs) / Tv
        d2 = sum((vs .- muv) .^ 2)
        wv = [0.2, 0.3, 0.5]
        muw = sum(wv .* vs) / sum(wv)
        d2w = sum(wv .* (vs .- muw) .^ 2)
        @test var(SimpleVariance(), vs) ≈ d2 / (Tv - 1)
        @test var(SimpleVariance(; corrected = false), vs) ≈ d2 / Tv
        @test var(SimpleVariance(; w = Weights(wv), corrected = false), vs) ≈ d2w / sum(wv)
        @test var(SimpleVariance(; w = aweights(wv)), vs) ≈
              d2w / (sum(wv) - sum(wv .^ 2) / sum(wv))
        @test var(SimpleVariance(; w = pweights(wv)), vs) ≈ d2w / (sum(wv) - sum(wv) / Tv)
        let fwv = fweights([1.0, 2.0, 3.0])
            m = sum(fwv .* vs) / sum(fwv)
            @test var(SimpleVariance(; w = fwv), vs) ≈
                  sum(fwv .* (vs .- m) .^ 2) / (sum(fwv) - 1)
        end
        # A plain `Weights` carries no correction, so `corrected = true` is refused.
        @test_throws ArgumentError var(SimpleVariance(; w = Weights(wv)), vs)
        #=
        The vector methods leave the centring to `Statistics`, so a weighted vector is
        centred on its weighted mean. The matrix methods take the centre from `ve.me`
        under the same `ve.w`, so the two paths answer the same number. Spelling the
        weights into `me` changes nothing. See #490 and ADR 0088.
        =#
        let Xc = reshape(vs, Tv, 1), aw = aweights(wv), c = sum(wv .^ 2) / sum(wv)
            @test var(SimpleVariance(; w = aw), Xc; dims = 1)[1] ≈
                  sum(wv .* (vs .- muw) .^ 2) / (sum(wv) - c)
            @test var(SimpleVariance(; w = aw), Xc; dims = 1)[1] ≈
                  var(SimpleVariance(; w = aw), vs)
            @test var(SimpleVariance(; me = SimpleExpectedReturns(; w = aw), w = aw), Xc;
                      dims = 1)[1] ≈ var(SimpleVariance(; w = aw), vs)
            # The `mean` keyword still reaches the unweighted centre.
            @test var(SimpleVariance(; w = aw), Xc; dims = 1, mean = [muv])[1] ≈
                  sum(wv .* (vs .- muv) .^ 2) / (sum(wv) - c)
        end
        # --- the semi-moment target is the mean, and the divisor stays `T - 1` ---
        Xsm = [0.01 0.02; 0.03 0.04; 0.02 0.03]
        let mu = mean(Xsm; dims = 1), Y = min.(Xsm .- mu, 0.0)
            @test cov(Covariance(; alg = SemiMoment()), Xsm) == Y'Y / (size(Xsm, 1) - 1)
        end
        # --- the fallback variance and standard deviation read the covariance diagonal ---
        @test vec(var(Covariance(), Xsm)) == diag(cov(Covariance(), Xsm))
        @test vec(std(Covariance(), Xsm)) == sqrt.(diag(cov(Covariance(), Xsm)))
        @test size(var(Covariance(), Xsm; dims = 1)) == (1, 2)
        @test size(var(Covariance(), permutedims(Xsm); dims = 2)) == (2, 1)
        # --- the wrappers compose `ce` and then `mp`, in the order their row declares ---
        rng453 = StableRNG(11223344)
        X453 = randn(rng453, 6, 10)
        mp_dn = MatrixProcessing(; pdm = Posdef(), dn = Denoise(), order = (:pdm, :dn))
        mp_dt = MatrixProcessing(; pdm = Posdef(), dt = Detone(), order = (:pdm, :dt))
        mp_al = MatrixProcessing(; pdm = Posdef(), alg = nothing, order = (:pdm, :alg))
        function processed(mp, X)
            s = Matrix(cov(Covariance(), X))
            matrix_processing!(mp, s, X)
            return s
        end
        @test cov(PortfolioOptimisersCovariance(; ce = Covariance(), mp = mp_dn), X453) ==
              processed(mp_dn, X453)
        @test cov(DenoiseCovariance(), X453) == processed(mp_dn, X453)
        @test cov(DetoneCovariance(), X453) == processed(mp_dt, X453)
        @test cov(ProcessedCovariance(), X453) == processed(mp_al, X453)
        #=
        The post-processing really runs: the raw covariance of a random matrix is dense, and
        the denoised one is not. Whether the two `order` permutations differ numerically is a
        property of the data, not of the wrapper, so it is not asserted here.
        =#
        @test processed(mp_dn, X453) != Matrix(cov(Covariance(), X453))
        # --- `CorrelationCovariance` answers both verbs with the correlation ---
        ccv = CorrelationCovariance(; ce = Covariance())
        @test cov(ccv, X453) == cor(Covariance(), X453)
        @test cor(ccv, X453) == cov(ccv, X453)
        @test all(isone, diag(cov(ccv, X453)))
        # --- `find_uncorrelated_indices`: the drop score, the tie policy, the guard ---
        rng454 = StableRNG(11)
        B453 = randn(rng454, 60, 3)
        W453 = hcat(B453[:, 1], B453[:, 1] .+ 1e-9 .* randn(rng454, 60), B453[:, 2],
                    B453[:, 3])
        @test PortfolioOptimisers.find_uncorrelated_indices(W453; t = 0.95) == [2, 3, 4]
        @test PortfolioOptimisers.find_uncorrelated_indices(W453; t = 0.95,
                                                            scores = [0.0, 9.0, 0.0, 0.0]) ==
              [1, 3, 4]
        @test PortfolioOptimisers.find_uncorrelated_indices(W453; t = 0.95,
                                                            scores = [9.0, 0.0, 0.0, 0.0]) ==
              [2, 3, 4]
        # Two columns that score equally leave no survivor.
        @test PortfolioOptimisers.find_uncorrelated_indices(hcat(B453[:, 1], B453[:, 1],
                                                                 B453[:, 2]); t = 0.95) ==
              [3]
        @test_throws DimensionMismatch PortfolioOptimisers.find_uncorrelated_indices(W453;
                                                                                     scores = [1.0,
                                                                                               2.0])
    end
    @testset "The information-theoretic sweep of #459" begin
        #=
        Every claim the histogram and mutual information files make, pinned by running it.
        The four bin rules against their own closed forms, the two direct routes to a bin
        count, the two extremes of `mutual_info`, the metric axioms of `variation_info`,
        the identity that relates `intrinsic_mutual_info` to `mutual_info`, and the shape
        of `MutualInfoCovariance`.
        =#
        PO = PortfolioOptimisers
        SF = PortfolioOptimisers.SpecialFunctions
        rngh = StableRNG(987654321)
        xh = randn(rngh, 200)
        yh = randn(rngh, 200)
        Th = length(xh)
        # --- Scott's rule against its closed form ---
        # dx = sigma * (24 sqrt(pi) / n)^(1/3), with the UNCORRECTED standard deviation.
        sc = PO.bin_width(Scott(), xh)
        @test isapprox(sc, std(xh; corrected = false) * (24 * sqrt(pi) / Th)^(1 / 3))
        # The corrected standard deviation is a different number, so the flag is not free.
        @test !isapprox(sc, std(xh; corrected = true) * (24 * sqrt(pi) / Th)^(1 / 3))
        # --- Freedman-Diaconis against its closed form ---
        # dx = 2 IQR(x) / n^(1/3).
        q25, q75 = quantile(xh, [0.25, 0.75])
        fd = PO.bin_width(FreedmanDiaconis(), xh)
        @test isapprox(fd, 2 * (q75 - q25) / Th^(1 / 3))
        #=
        The two rules differ in their constant AND in their dispersion measure, so a check
        that does not separate them checks neither. On this sample Scott is the wider.
        =#
        @test !isapprox(sc, fd; rtol = 0.1)
        @test sc > fd
        # --- Knuth's rule maximises its own posterior ---
        # `bin_width` returns `range / M`, so recover M and score its neighbours.
        function knuth_f(x, M)
            n = length(x)
            xl, xu = extrema(x)
            rx = xu - xl
            nk = zeros(Int, M)
            for xi in x
                nk[min(floor(Int, (xi - xl) / rx * M) + 1, M)] += 1
            end
            return n * log(M) + SF.loggamma(M / 2) - M * SF.loggamma(0.5) -
                   SF.loggamma(n + M / 2) + sum(SF.loggamma, nk .+ 0.5)
        end
        kdx = PO.bin_width(Knuth(), xh)
        Mk = round(Int, (maximum(xh) - minimum(xh)) / kdx)
        @test Mk >= 1
        @test knuth_f(xh, Mk) >= knuth_f(xh, Mk - 1)
        @test knuth_f(xh, Mk) >= knuth_f(xh, Mk + 1)
        # The three width rules are genuinely different selections on one sample.
        @test length(unique([sc, fd, kdx])) == 3
        # --- Hacine-Gharbi-Ravier, both closed forms ---
        ch = cor(xh, yh)
        @test !isone(abs(ch))
        @test PO.calc_num_bins(HacineGharbiRavier(), xh, yh, 1, 2, Th) ==
              round(Int, sqrt(1 + sqrt(1 + 24 * Th / (1 - ch^2))) / sqrt(2))
        zk = cbrt(8 + 324 * Th + 12 * sqrt(36 * Th + 729 * Th^2))
        @test PO.calc_num_bins(HacineGharbiRavier(), xh, xh, 1, 1, Th) ==
              round(Int, zk / 6 + 2 / (3 * zk) + 1 / 3)
        #=
        The bi-histogram form divides by `1 - rho^2`, so it is singular at BOTH ends of the
        correlation range. A perfectly anti-correlated pair used to reach that branch and
        raise `InexactError: Int64(Inf)`; it now takes the univariate form, as `rho = 1`
        does. Fixed under #459.
        =#
        @test cor(xh, -xh) == -1
        @test PO.calc_num_bins(HacineGharbiRavier(), xh, -xh, 1, 2, Th) ==
              PO.calc_num_bins(HacineGharbiRavier(), xh, xh, 1, 1, Th)
        @test !isnan(PO.mutual_info(hcat(xh, -xh))[1, 2])
        @test size(PO.variation_info(hcat(xh, -xh))) == (2, 2)
        # --- the two direct routes to a bin count ---
        # An `Integer` is returned unchanged and reads no other argument.
        @test PO.calc_num_bins(11, xh, yh, 1, 2, Th) == 11
        @test PO.calc_num_bins(11) == 11
        @test 11 isa PO.Int_Bin
        @test HacineGharbiRavier() isa PO.Int_Bin
        #=
        A `BinWidthBins` divides the range by the width. The self pair takes one variable's
        count; an off-diagonal pair takes the LARGER of the two.
        =#
        k1 = (maximum(xh) - minimum(xh)) / sc
        k2 = (maximum(yh) - minimum(yh)) / PO.bin_width(Scott(), yh)
        @test PO.calc_num_bins(Scott(), xh, yh, 1, 1, Th) == round(Int, k1)
        @test PO.calc_num_bins(Scott(), xh, yh, 1, 2, Th) == round(Int, max(k1, k2))
        # --- `calc_hist_data` shape and units ---
        nb = 8
        exh, eyh, hxy = PO.calc_hist_data(xh, yh, nb)
        @test size(hxy) == (nb, nb)
        # The marginals are normalised before the entropy; the joint histogram is not.
        wmx = fit(Histogram, xh,
                  range(minimum(xh), nextfloat(maximum(xh)); length = nb + 1)).weights
        @test exh ≈ entropy(wmx ./ sum(wmx))
        @test sum(hxy) == length(xh)
        #=
        ADR 0089, #493. The edges are widened by the spacing at the value, not by the
        machine epsilon of the type. `eps(Float64)` is half of one unit in the last place at
        a magnitude of two or more, so the old upper edge rounded back onto the maximum and
        the exclusive edge binned the largest observation out. `nextfloat` is a whole unit
        at every magnitude, so nothing is lost. The testset below carries the rest.
        =#
        xbig = xh .+ 10
        @test maximum(xbig) + eps(Float64) == maximum(xbig)
        @test sum(PO.calc_hist_data(xbig, xbig, nb)[3]) == length(xbig)
        # --- `intrinsic_mutual_info` is the unnormalised, unclamped estimate ---
        Zh = hcat(xh, yh)
        @test PO.intrinsic_mutual_info(hxy) ≈ PO.mutual_info(Zh, nb, false)[1, 2]
        @test PO.mutual_info(Zh, nb, true)[1, 2] ≈
              PO.intrinsic_mutual_info(hxy) / min(exh, eyh)
        # A single bin on either axis carries no information.
        @test iszero(PO.intrinsic_mutual_info(PO.calc_hist_data(xh, yh, 1)[3]))
        @test iszero(PO.intrinsic_mutual_info(ones(1, 4)))
        @test iszero(PO.intrinsic_mutual_info(ones(4, 1)))
        # --- `mutual_info` at its two extremes ---
        rngm = StableRNG(123456789)
        Xm = randn(rngm, 500, 3)
        Mun = PO.mutual_info(Xm, HacineGharbiRavier(), false)
        # A variable against itself gives its own entropy.
        for j in 1:3
            nbj = PO.calc_num_bins(HacineGharbiRavier(), view(Xm, :, j), view(Xm, :, j), j,
                                   j, size(Xm, 1))
            @test Mun[j, j] ≈ PO.calc_hist_data(view(Xm, :, j), view(Xm, :, j), nbj)[1]
        end
        # Independent variables give about zero -- a small positive finite-sample bias.
        for j in 1:3, i in 1:(j - 1)
            @test 0 <= Mun[j, i] < 0.1
        end
        # Normalised, the diagonal is one and every entry is bounded by [0, 1].
        Mno = PO.mutual_info(Xm, HacineGharbiRavier(), true)
        @test all(x -> isapprox(x, 1), diag(Mno))
        @test all(x -> 0 <= x <= 1 + eps(), Mno)
        @test issymmetric(Mno)
        # --- `variation_info` is a metric ---
        rngv = StableRNG(555)
        Zv = randn(rngv, 600, 6)
        Zv[:, 2] .= 0.8 .* Zv[:, 1] .+ 0.6 .* Zv[:, 2]
        Zv[:, 3] .= 0.5 .* Zv[:, 2] .+ 0.9 .* Zv[:, 3]
        for nrm in (true, false)
            Vv = PO.variation_info(Zv, 8, nrm)
            # Identity of indiscernibles, pinned rather than estimated.
            @test all(iszero, diag(Vv))
            # Non-negativity and symmetry.
            @test all(>=(0), Vv)
            @test issymmetric(Vv)
            # The triangle inequality, over every ordered triple.
            for a in axes(Vv, 1), b in axes(Vv, 1), c in axes(Vv, 1)
                @test Vv[a, c] <= Vv[a, b] + Vv[b, c] + 1e-12
            end
        end
        # The normalised form divides by the joint entropy, which bounds it to [0, 1].
        Vn = PO.variation_info(Zv, 8, true)
        @test all(x -> 0 <= x <= 1, Vn)
        # `mutual_info` is NOT a metric: its diagonal is the entropy, not zero.
        @test all(!iszero, diag(PO.mutual_info(Zv, 8, false)))
        # The unnormalised form is the closed form, term by term.
        Vu = PO.variation_info(Zv, 8, false)
        for j in axes(Zv, 2), i in 1:(j - 1)
            ex2, ey2, h2 = PO.calc_hist_data(view(Zv, :, j), view(Zv, :, i), 8)
            mi2 = PO.intrinsic_mutual_info(h2)
            @test Vu[j, i] ≈ ex2 + ey2 - 2 * mi2
            @test Vn[j, i] ≈ (ex2 + ey2 - 2 * mi2) / (ex2 + ey2 - mi2)
        end
        # --- `mutual_variation_info` returns exactly the two matrices ---
        mmv, vmv = PO.mutual_variation_info(Zv, 8, true)
        @test mmv ≈ PO.mutual_info(Zv, 8, true)
        @test vmv ≈ Vn
        @test all(iszero, diag(vmv))
        @test all(x -> isapprox(x, 1), diag(mmv))
        # --- `MutualInfoCovariance` ---
        Xc = randn(StableRNG(4242), 400, 4)
        vc = vec(var(SimpleVariance(), Xc))
        sc4 = sqrt.(vc)
        for nrm in (true, false)
            cec = MutualInfoCovariance(; bins = 8, normalise = nrm)
            Rc = cor(cec, Xc)
            Cc = cov(cec, Xc)
            @test issymmetric(Rc)
            @test issymmetric(Cc)
            @test Rc ≈ PO.mutual_info(Xc, 8, nrm)
            #=
            `cor2cov!` writes the variance onto the diagonal whatever the correlation
            carries there, so the covariance diagonal is the variance under BOTH flags.
            =#
            @test isapprox(diag(Cc), vc)
            # Off the diagonal the covariance is the correlation scaled by the two sigmas.
            for j in 1:4, i in 1:(j - 1)
                @test Cc[j, i] ≈ Rc[j, i] * sc4[j] * sc4[i]
            end
            # Mutual information is non-negative, so no pair is ever reported as opposed.
            @test all(>=(0), Cc)
        end
        # Normalised, the correlation diagonal is one; unnormalised, it is the entropy.
        @test all(x -> isapprox(x, 1), diag(cor(MutualInfoCovariance(; bins = 8), Xc)))
        @test all(>(1), diag(cor(MutualInfoCovariance(; bins = 8, normalise = false), Xc)))
        # `dims = 2` transposes before the estimate rather than after.
        @test cor(MutualInfoCovariance(; bins = 8), Xc) ≈
              cor(MutualInfoCovariance(; bins = 8), Xc'; dims = 2)
        @test_throws DomainError cor(MutualInfoCovariance(; bins = 8), Xc; dims = 3)
    end
    @testset "A histogram edge is widened by the spacing at the value (#493)" begin
        #=
        ADR 0089. `calc_hist_data` widened both edges by `eps(eltype(x))`, the epsilon of
        the type rather than the spacing at the value. At a magnitude of two or more that is
        half a unit in the last place, so `maximum(x) + eps(Float64)` rounded back onto the
        maximum. A `StatsBase.Histogram` bin is closed on the left, so its upper edge is
        exclusive, and the largest observation was binned out.
        =#
        Z493 = randn(StableRNG(555), 600, 6)
        Tobs493 = size(Z493, 1)
        #=
        The rounding that caused the defect is a property of the value, not of the fix, so
        it is still there. `eps(Float64)` is exactly half a unit in the last place at a
        magnitude between two and four, and a half rounds to even, so the sum returns the
        value it started from. `nextfloat` is a whole unit and always moves.
        =#
        @test 3.5 + eps(Float64) == 3.5
        @test nextfloat(3.5) > 3.5
        # Three of these six columns carry a maximum that the old widening rounded away.
        @test count(k -> maximum(view(Z493, :, k)) + eps(Float64) ==
                         maximum(view(Z493, :, k)), axes(Z493, 2)) >= 1
        # Every observation of every pair is binned, under every bin rule.
        for bins in (HacineGharbiRavier(), Knuth(), Scott(), FreedmanDiaconis(), 8)
            for j in axes(Z493, 2), i in 1:j
                xj = view(Z493, :, j)
                xi = view(Z493, :, i)
                nb = PortfolioOptimisers.calc_num_bins(bins, xj, xi, j, i, Tobs493)
                ex, ey, hxy = PortfolioOptimisers.calc_hist_data(xj, xi, nb)
                @test sum(hxy) == Tobs493
                @test isfinite(ex)
                @test isfinite(ey)
            end
        end
        #=
        The pair this data loses most on. It binned 598 of 600, and its mutual information
        was 0.03695594625897657, which is 7.4 % below the value below. #493's own table is
        a different matrix -- the same seed with two columns made correlated -- so its
        numbers are not these. These are the plain sample, which is what this pins.
        =#
        ex493, ey493, hxy493 = PortfolioOptimisers.calc_hist_data(view(Z493, :, 2),
                                                                  view(Z493, :, 5), 8)
        @test sum(hxy493) == 600
        @test isapprox(ex493, 1.683319735731528)
        @test isapprox(ey493, 1.5597176994273774)
        @test isapprox(PortfolioOptimisers.intrinsic_mutual_info(hxy493),
                       0.03969760755165243)
        #=
        A constant column at a magnitude where the old widening rounded away closed both
        edges onto one value. The histogram was all zero, `hx / sum(hx)` was a vector of
        `NaN`, and the entropy was `NaN`. The entropy of a constant variable is zero.
        =#
        c493 = fill(3.5, 20)
        exc, eyc, hxyc = PortfolioOptimisers.calc_hist_data(c493, c493, 8)
        @test iszero(exc)
        @test iszero(eyc)
        @test sum(hxyc) == 20
        #=
        The library's own pinned numbers do not move, and this is the reason. Every return
        of the test data is small enough that `eps(Float64)` is many units in the last place
        there, so the widening that ADR 0089 replaced already did what it was written to do.
        =#
        @test !any(k -> maximum(view(rd.X, :, k)) + eps(Float64) ==
                        maximum(view(rd.X, :, k)), axes(rd.X, 2))
        for bins in (HacineGharbiRavier(), Knuth(), Scott(), FreedmanDiaconis(), 5, 7)
            for j in axes(rd.X, 2), i in 1:j
                xj = view(rd.X, :, j)
                xi = view(rd.X, :, i)
                nb = PortfolioOptimisers.calc_num_bins(bins, xj, xi, j, i, size(rd.X, 1))
                @test sum(PortfolioOptimisers.calc_hist_data(xj, xi, nb)[3]) ==
                      size(rd.X, 1)
            end
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
@testset "CustomValueExpectedReturns callable validation" begin
    X = randn(StableRNG(987654321), 50, 4)
    scalar_val = (X; kwargs...) -> 0.123
    short_val = (X; kwargs...) -> [0.1, 0.2]
    per_asset_val = function (X; dims::Int = 1, kwargs...)
        return fill(0.5, size(X, setdiff((1, 2), (dims,))[1]))
    end
    # The callable branch validates what the callable returns, as the vector branch
    # validates the stored field.
    @test_throws ArgumentError mean(CustomValueExpectedReturns(; val = scalar_val), X)
    @test_throws DimensionMismatch mean(CustomValueExpectedReturns(; val = short_val), X)
    # A callable that returns one value per asset is accepted, along both dims.
    @test mean(CustomValueExpectedReturns(; val = per_asset_val), X) == fill(0.5, 4)
    @test mean(CustomValueExpectedReturns(; val = per_asset_val), X; dims = 2) ==
          fill(0.5, 50)
    # The scalar and vector branches are unchanged.
    @test mean(CustomValueExpectedReturns(; val = 0.02), X) == fill(0.02, 1, 4)
    @test mean(CustomValueExpectedReturns(; val = [0.1, 0.2, 0.3, 0.4]), X) ==
          [0.1 0.2 0.3 0.4]
    @test_throws DimensionMismatch mean(CustomValueExpectedReturns(; val = [0.1, 0.2]), X)
    @testset "The shrunk expected returns sweep of #460" begin
        # Every claim of `src/08_Moments/16_ShrunkExpectedReturns.jl` is checked by running
        # it. The three targets are checked against a hand-computed vector, the three
        # intensities against their own closed form, and the degeneracies the docstrings
        # name are pinned so that a later change cannot turn a documented `NaN` into a
        # silent number or the other way round.
        rng460 = StableRNG(987654321)
        X460 = randn(rng460, 40, 5) ./ 100
        tgts460 = (GrandMean(), VolatilityWeighted(), MeanSquaredError())
        algs460 = (JamesStein, BayesStein, BodnarOkhrinParolya)

        # The sample moments the file's own default estimators produce. Every hand
        # computation below starts here, so a change of default moves the check with the
        # code.
        me460 = ShrunkExpectedReturns()
        mu460 = mean(me460.me, X460; dims = 1)
        sigma460 = cov(me460.ce, X460; dims = 1)
        isigma460 = sigma460 \ I
        T460, N460 = size(X460)

        @testset "each target against a hand-computed vector" begin
            # `target_mean` returns a constant range, never a dense vector, so the length
            # and the constancy are part of the contract.
            for tgt in tgts460
                b = PortfolioOptimisers.target_mean(tgt, mu460, sigma460, isigma460;
                                                    T = T460)
                @test b isa AbstractRange
                @test length(b) == N460
                @test all(==(first(b)), b)
            end
            # `GrandMean` is the unweighted mean of `mu`.
            @test collect(PortfolioOptimisers.target_mean(GrandMean(), mu460, sigma460)) ==
                  fill(mean(mu460), N460)
            # `VolatilityWeighted` is `1'inv(S)mu / 1'inv(S)1`.
            vw460 = sum(isigma460 * vec(mu460)) / sum(isigma460)
            @test isapprox(collect(PortfolioOptimisers.target_mean(VolatilityWeighted(),
                                                                   mu460, sigma460,
                                                                   isigma460)),
                           fill(vw460, N460))
            # The same value comes back when the caller passes no inverse, so the internal
            # solve and the supplied inverse agree.
            @test isapprox(collect(PortfolioOptimisers.target_mean(VolatilityWeighted(),
                                                                   mu460, sigma460)),
                           collect(PortfolioOptimisers.target_mean(VolatilityWeighted(),
                                                                   mu460, sigma460,
                                                                   isigma460)))
            # `MeanSquaredError` is the trace over `T`, and it reads neither `mu` nor the
            # inverse.
            @test collect(PortfolioOptimisers.target_mean(MeanSquaredError(), mu460,
                                                          sigma460; T = T460)) ==
                  fill(tr(sigma460) / T460, N460)
        end

        @testset "each intensity against its own closed form" begin
            # Each of the three closed forms is written out separately here. A copied
            # denominator is exactly what a shared helper would hide.
            evals460 = eigvals(sigma460)
            for tgt in tgts460
                b = vec(collect(PortfolioOptimisers.target_mean(tgt, mu460, sigma460,
                                                                isigma460; T = T460)))
                vm = vec(mu460)
                mb = vm - b

                a_js = (N460 * mean(evals460) - 2 * maximum(evals460)) /
                       (T460 * dot(mb, mb))
                @test isapprox(vec(mean(ShrunkExpectedReturns(;
                                                              alg = JamesStein(; tgt = tgt)),
                                        X460)), (1 - a_js) * vm + a_js * b)

                a_bs = (N460 + 2) / ((N460 + 2) + T460 * dot(mb, isigma460, mb))
                @test isapprox(vec(mean(ShrunkExpectedReturns(;
                                                              alg = BayesStein(; tgt = tgt)),
                                        X460)), (1 - a_bs) * vm + a_bs * b)

                u = dot(vm, isigma460, vm)
                v = dot(b, isigma460, b)
                w = dot(vm, isigma460, b)
                a_bop = ((u - N460 / (T460 - N460)) * v - w^2) / (u * v - w^2)
                b_bop = (1 - a_bop) * w / u
                @test isapprox(vec(mean(ShrunkExpectedReturns(;
                                                              alg = BodnarOkhrinParolya(;
                                                                                        tgt = tgt)),
                                        X460)), a_bop * vm + b_bop * b)
            end
        end

        @testset "`dims = 2` agrees with `dims = 1`" begin
            # The transposed branch of all three methods. It was the whole of this file's
            # uncovered code before #460.
            Xt460 = permutedims(X460)
            for alg in algs460, tgt in tgts460
                me = ShrunkExpectedReturns(; alg = alg(; tgt = tgt))
                r1 = mean(me, X460; dims = 1)
                r2 = mean(me, Xt460; dims = 2)
                @test size(r1) == (1, N460)
                @test size(r2) == (N460, 1)
                @test isapprox(vec(r1), vec(r2))
            end
        end

        @testset "the Bayes-Stein intensity is the only convex one" begin
            # `alpha` is `(N+2) / ((N+2) + T q)` with `q` a non-negative quadratic form, so
            # it lies in `(0, 1]` and the result lies between the sample mean and the
            # target, componentwise. The docstring says so; this drives it.
            for tgt in tgts460
                b = vec(collect(PortfolioOptimisers.target_mean(tgt, mu460, sigma460,
                                                                isigma460; T = T460)))
                vm = vec(mu460)
                mb = vm - b
                a_bs = (N460 + 2) / ((N460 + 2) + T460 * dot(mb, isigma460, mb))
                @test 0 < a_bs <= 1
                r = vec(mean(ShrunkExpectedReturns(; alg = BayesStein(; tgt = tgt)), X460))
                @test all(min.(vm, b) .- sqrt(eps()) .<= r .<= max.(vm, b) .+ sqrt(eps()))
            end
            # The other two are not convex on this sample, which is why the docstrings warn
            # that nothing clamps them.
            evals460 = eigvals(sigma460)
            b_gm = vec(collect(PortfolioOptimisers.target_mean(GrandMean(), mu460,
                                                               sigma460)))
            mb_gm = vec(mu460) - b_gm
            a_js = (N460 * mean(evals460) - 2 * maximum(evals460)) /
                   (T460 * dot(mb_gm, mb_gm))
            @test 0 < a_js < 1
            u = dot(vec(mu460), isigma460, vec(mu460))
            v = dot(b_gm, isigma460, b_gm)
            w = dot(vec(mu460), isigma460, b_gm)
            a_bop = ((u - N460 / (T460 - N460)) * v - w^2) / (u * v - w^2)
            @test a_bop < 0
        end

        @testset "an intensity of one returns the target" begin
            # When the target equals the sample mean the Bayes-Stein quadratic form is zero
            # and `alpha` is exactly one. Columns that share a mean drive it: the grand mean
            # is then the sample mean itself.
            rng_eq = StableRNG(24680)
            Xeq = randn(rng_eq, 60, 4) ./ 100
            Xeq .-= mean(Xeq; dims = 1)
            me_eq = ShrunkExpectedReturns(; alg = BayesStein())
            mu_eq = mean(me_eq.me, Xeq; dims = 1)
            b_eq = collect(PortfolioOptimisers.target_mean(GrandMean(), mu_eq,
                                                           cov(me_eq.ce, Xeq; dims = 1)))
            @test isapprox(vec(mu_eq), b_eq; atol = 1e-14)
            @test isapprox(vec(mean(me_eq, Xeq)), b_eq; atol = 1e-14)
        end

        @testset "the Bodnar-Okhrin-Parolya coefficient does not read the target scale" begin
            # Every target of this file is a multiple of the vector of ones, so the
            # multiplier cancels in `alpha` and survives in `beta * b`. The coefficient is
            # therefore one number per sample, and the three results still differ.
            u = dot(vec(mu460), isigma460, vec(mu460))
            alphas = map(tgts460) do tgt
                b = vec(collect(PortfolioOptimisers.target_mean(tgt, mu460, sigma460,
                                                                isigma460; T = T460)))
                v = dot(b, isigma460, b)
                w = dot(vec(mu460), isigma460, b)
                return ((u - N460 / (T460 - N460)) * v - w^2) / (u * v - w^2)
            end
            @test all(a -> isapprox(a, first(alphas)), alphas)
            results = map(tgts460) do tgt
                return vec(mean(ShrunkExpectedReturns(;
                                                      alg = BodnarOkhrinParolya(;
                                                                                tgt = tgt)),
                                X460))
            end
            @test !isapprox(results[1], results[2])
            @test !isapprox(results[1], results[3])
        end

        @testset "the James-Stein intensity is negative for two assets or fewer" begin
            # `N * mean(evals)` is the trace, so it never exceeds `2 * maximum(evals)` when
            # `N <= 2`. The blend then extrapolates away from the target.
            for n in (1, 2)
                Xn = X460[:, 1:n]
                sn = cov(me460.ce, Xn; dims = 1)
                en = eigvals(sn)
                @test n * mean(en) - 2 * maximum(en) <= 0
            end
        end

        @testset "the degenerate samples the docstrings name" begin
            # #497, ADR 0092, turned the three degeneracies below from a silent `NaN` or
            # `Inf` into a `DomainError`. Each case names the guard that raises it.
            #
            # One asset. `GrandMean` and `VolatilityWeighted` both reduce to the sample
            # mean, so the James-Stein denominator is exactly zero.
            X1 = X460[:, 1:1]
            @test_throws DomainError mean(ShrunkExpectedReturns(; alg = JamesStein()), X1)
            @test_throws DomainError mean(ShrunkExpectedReturns(;
                                                                alg = JamesStein(;
                                                                                 tgt = VolatilityWeighted())),
                                          X1)
            # `MeanSquaredError` never reads the sample mean, so it stays finite there.
            @test all(isfinite,
                      mean(ShrunkExpectedReturns(;
                                                 alg = JamesStein(;
                                                                  tgt = MeanSquaredError())),
                           X1))
            # At one asset the Cauchy-Schwarz gap `u v - w^2` is exactly zero, so
            # Bodnar-Okhrin-Parolya raises under every target. `T > N` holds here, so the
            # gap guard is the one that fires, and its message names the gap.
            for tgt in tgts460
                @test_throws DomainError mean(ShrunkExpectedReturns(;
                                                                    alg = BodnarOkhrinParolya(;
                                                                                              tgt = tgt)),
                                              X1)
                err = try
                    mean(ShrunkExpectedReturns(; alg = BodnarOkhrinParolya(; tgt = tgt)),
                         X1)
                    nothing
                catch e
                    e
                end
                @test occursin("Cauchy-Schwarz gap", err.msg)
            end
            # Bayes-Stein has a strictly positive denominator, so it survives one asset.
            for tgt in tgts460
                @test all(isfinite,
                          mean(ShrunkExpectedReturns(; alg = BayesStein(; tgt = tgt)), X1))
            end
            # A square returns matrix. `N / (T - N)` is undefined, so the `T > N` guard
            # raises before the coefficients are formed.
            Xsq = X460[1:N460, :]
            for tgt in tgts460
                @test_throws DomainError mean(ShrunkExpectedReturns(;
                                                                    alg = BodnarOkhrinParolya(;
                                                                                              tgt = tgt)),
                                              Xsq)
            end
            # Two observations and five assets. The covariance estimator repairs the
            # singular matrix, so James-Stein and Bayes-Stein return a finite vector.
            X2 = X460[1:2, :]
            for alg in (JamesStein, BayesStein), tgt in tgts460
                @test all(isfinite,
                          mean(ShrunkExpectedReturns(; alg = alg(; tgt = tgt)), X2))
            end
            # `T < N` makes `N / (T - N)` negative, which silently flipped the sign of the
            # coefficient before #497. The same guard refuses it.
            for tgt in tgts460
                @test_throws DomainError mean(ShrunkExpectedReturns(;
                                                                    alg = BodnarOkhrinParolya(;
                                                                                              tgt = tgt)),
                                              X2)
            end
        end
    end
end
