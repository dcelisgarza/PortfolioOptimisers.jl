@safetestset "Risk Measure Tests" begin
    using PortfolioOptimisers, Test, Random, StableRNGs, CSV, DataFrames, StatsBase,
          Clarabel, LinearAlgebra
    import PortfolioOptimisers: risk_measure_factory, risk_measure_view, ucs_view,
                                nothing_scalar_array_view, prior_view,
                                fourth_moment_index_factory,
                                nothing_scalar_array_view_odd_order, __coskewness
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
    @testset "Variance and Moment risk measures" begin
        rng = StableRNG(123456789)
        X = randn(rng, 500, 20)
        ew = eweights(1:500, 1 / 500; scale = true)
        w = rand(rng, 20)
        w ./= sum(w)
        i = [20, 3, 9]
        wv = nothing_scalar_array_view(w, i)
        slv = Solver(; name = :Clarabel, solver = Clarabel.Optimizer,
                     settings = Dict("verbose" => false))
        ucs1 = DeltaUncertaintySetEstimator(;)
        ucs2 = NormalUncertaintySetEstimator(;)
        settings = RiskMeasureSettings(; rke = false, scale = -1, ub = 3)
        formulation = RSOC()
        ucs1view = ucs_view(ucs1, i)
        ucs2view = ucs_view(ucs2, i)
        rc = LinearConstraint(; A = LinearConstraintSide(; group = :A, name = :B), B = 0)

        rs = BrownianDistanceVariance(; settings = settings,
                                      formulation = IneqBrownianDistanceVariance())
        r = risk_measure_factory(rs)
        @test r === rs
        @test isapprox(expected_risk(r, w, X), 0.2018426182783387)

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
        sets = DataFrame(:Assets => 1:20)
        pes = [HighOrderPriorEstimator(),
               HighOrderPriorEstimator(;
                                       pe = EntropyPoolingPriorEstimator(; views = views,
                                                                         sets = sets,
                                                                         alg = H1_EntropyPooling())),
               EntropyPoolingPriorEstimator(; views = views, sets = sets,
                                            alg = H1_EntropyPooling())]
        mu = rand(rng, 20)
        muv = nothing_scalar_array_view(mu, i)
        zerovec = fill(0.0, 20)
        zerovecv = nothing_scalar_array_view(zerovec, i)
        for pe ∈ pes
            pr1 = prior(pe, X)
            sigma = pr1.sigma * 1.5
            prv = prior_view(pr1, i)
            sigmav = nothing_scalar_array_view(sigma, i)
            rs = [Variance(; settings = settings, formulation = formulation, sigma = sigma,
                           rc = rc), Variance(;)]
            r = risk_measure_factory(rs, Ref(pr1), Ref(slv), Ref(ucs2))
            @test r[1].settings === settings
            @test r[1].formulation === formulation
            @test r[1].sigma === sigma
            @test r[1].rc === rc
            @test isapprox(expected_risk(r[1], w, X), dot(w, sigma, w))

            @test r[2].sigma === pr1.sigma
            @test isapprox(expected_risk(r[2], w, X), dot(w, pr1.sigma, w))

            sigma = pr1.sigma * 2
            sigmav = nothing_scalar_array_view(sigma, i)
            rs = [StandardDeviation(; settings = settings, sigma = sigma),
                  StandardDeviation(;)]
            r = risk_measure_factory(rs, Ref(pr1), Ref(slv), Ref(ucs2))
            @test r[1].settings === settings
            @test r[1].sigma === sigma
            @test isapprox(expected_risk(r[1], w, X), sqrt(dot(w, sigma, w)))

            @test r[2].sigma === pr1.sigma
            @test isapprox(expected_risk(r[2], w, X), sqrt(dot(w, pr1.sigma, w)))

            sigma = pr1.sigma * 2.5
            sigmav = nothing_scalar_array_view(sigma, i)
            rs = [UncertaintySetVariance(; settings = settings, ucs = ucs1, sigma = sigma),
                  UncertaintySetVariance(;)]
            r = risk_measure_factory(rs, Ref(pr1), Ref(slv), Ref(ucs2))
            @test r[1].settings === settings
            @test r[1].ucs === ucs1
            @test r[1].sigma === sigma
            @test isapprox(expected_risk(r[1], w, X), dot(w, sigma, w))

            @test r[2].sigma === pr1.sigma
            @test r[2].ucs === ucs2
            @test isapprox(expected_risk(r[2], w, X), dot(w, pr1.sigma, w))

            formulation = Quad()
            rs = [LowOrderMoment(; settings = settings, w = ew, mu = mu),
                  LowOrderMoment(; mu = 0), LowOrderMoment(; mu = zerovec),
                  LowOrderMoment(; alg = SemiDeviation(; ddof = 2)),
                  LowOrderMoment(;
                                 alg = SecondLowerMoment(; ddof = 3,
                                                         formulation = formulation)),
                  LowOrderMoment(; alg = MeanAbsoluteDeviation()),
                  LowOrderMoment(; alg = MeanAbsoluteDeviation(; w = ew))]
            r = risk_measure_factory(rs, Ref(pr1), Ref(slv), Ref(ucs2))
            @test r[1].settings === settings
            @test r[1].w === ew
            @test r[1].mu === mu
            val = X * w .- dot(w, mu)
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(r[1], w, X), -sum(val) / size(X, 1))
            val = prv.X * wv .- dot(wv, muv)
            val = val[val .<= zero(eltype(val))]

            @test r[2].mu == 0
            val = X * w
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(r[2], w, X), -sum(val) / size(X, 1))
            val = prv.X * wv
            val = val[val .<= zero(eltype(val))]

            @test isa(r[3].alg, FirstLowerMoment)
            @test r[3].mu === zerovec
            val = X * w .- dot(w, zerovec)
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(r[3], w, X), -sum(val) / size(X, 1))
            val = prv.X * wv .- dot(wv, zerovecv)
            val = val[val .<= zero(eltype(val))]

            @test r[4].alg.ddof == 2
            @test r[4].mu === pr1.mu
            val = X * w .- dot(w, pr1.mu)
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(r[4], w, X),
                           sqrt(dot(val, val) / (size(X, 1) - r[4].alg.ddof)))
            val = prv.X * wv .- dot(wv, prv.mu)
            val = val[val .<= zero(eltype(val))]

            @test r[5].alg.ddof == 3
            @test r[5].alg.formulation === formulation
            @test r[5].mu === pr1.mu
            val = X * w .- dot(w, pr1.mu)
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(r[5], w, X),
                           dot(val, val) / (size(X, 1) - r[5].alg.ddof))
            val = prv.X * wv .- dot(wv, prv.mu)
            val = val[val .<= zero(eltype(val))]

            if isa(pr1,
                   HighOrderPriorResult{<:LowOrderPriorResult, <:Any, <:Any, <:Any, <:Any})
                @test isnothing(r[6].alg.w)
            elseif isa(pr1,
                       HighOrderPriorResult{<:EntropyPoolingPriorResult, <:Any, <:Any,
                                            <:Any, <:Any})
                @test r[6].alg.w === pr1.pr.w
            end
            @test r[6].mu === pr1.mu
            val = X * w .- dot(w, pr1.mu)
            @test isapprox(expected_risk(r[6], w, X), mean(abs.(val)))
            val = prv.X * wv .- dot(wv, prv.mu)

            if isa(pr1,
                   HighOrderPriorResult{<:LowOrderPriorResult, <:Any, <:Any, <:Any, <:Any})
                @test r[7].alg.w === ew
            elseif isa(pr1,
                       HighOrderPriorResult{<:EntropyPoolingPriorResult, <:Any, <:Any,
                                            <:Any, <:Any})
                @test r[7].alg.w === pr1.pr.w
            else
                @test r[7].alg.w === pr1.w
            end
            @test r[7].mu === pr1.mu
            val = X * w .- dot(w, pr1.mu)
            @test isapprox(expected_risk(r[7], w, X),
                           mean(abs.(val),
                                if isa(pr1,
                                       HighOrderPriorResult{<:LowOrderPriorResult, <:Any,
                                                            <:Any, <:Any, <:Any})
                                    ew
                                elseif isa(pr1,
                                           HighOrderPriorResult{<:EntropyPoolingPriorResult,
                                                                <:Any, <:Any, <:Any, <:Any})
                                    pr1.pr.w
                                else
                                    pr1.w
                                end))
            val = prv.X * wv .- dot(wv, prv.mu)

            rs = [HighOrderMoment(; settings = settings, w = ew, mu = mu),
                  HighOrderMoment(; alg = FourthLowerMoment(), mu = 0),
                  HighOrderMoment(; alg = FourthCentralMoment(), mu = zerovec),
                  HighOrderMoment(; alg = HighOrderDeviation(; alg = ThirdLowerMoment())),
                  HighOrderMoment(;
                                  alg = HighOrderDeviation(; alg = FourthLowerMoment(),
                                                           ve = SimpleVariance(;
                                                                               corrected = false,
                                                                               me = nothing,
                                                                               w = ew))),
                  HighOrderMoment(;
                                  alg = HighOrderDeviation(; alg = FourthCentralMoment()))]
            r = risk_measure_factory(rs, Ref(pr1), Ref(slv), Ref(ucs2))
            @test r[1].settings === settings
            @test r[1].w === ew
            @test r[1].mu === mu
            val = X * w .- dot(w, mu)
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(r[1], w, X), -sum(val .^ 3) / size(X, 1))
            val = prv.X * wv .- dot(wv, muv)
            val = val[val .<= zero(eltype(val))]

            @test r[2].mu === 0
            val = X * w
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(r[2], w, X), sum(val .^ 4) / size(X, 1))
            val = prv.X * wv
            val = val[val .<= zero(eltype(val))]

            @test isa(r[3].alg, FourthCentralMoment)
            @test r[3].mu === zerovec
            val = X * w .- dot(w, zerovec)
            @test isapprox(expected_risk(r[3], w, X), sum(val .^ 4) / size(X, 1))
            val = prv.X * wv .- dot(wv, zerovecv)

            @test r[4].mu === pr1.mu
            val = X * w .- dot(w, pr1.mu)
            val = val[val .<= zero(eltype(val))]
            s = std(r[4].alg.ve, val; mean = zero(eltype(val)))
            @test isapprox(expected_risk(r[4], w, X), -sum(val .^ 3) / size(X, 1) / s^3)
            val = prv.X * wv .- dot(wv, prv.mu)
            val = val[val .<= zero(eltype(val))]

            @test r[5].mu === pr1.mu
            if isa(pr1,
                   HighOrderPriorResult{<:LowOrderPriorResult, <:Any, <:Any, <:Any, <:Any})
                @test r[5].alg.ve.w === ew
            elseif isa(pr1,
                       HighOrderPriorResult{<:EntropyPoolingPriorResult, <:Any, <:Any,
                                            <:Any, <:Any})
                @test r[5].alg.ve.w === pr1.pr.w
            else
                @test r[5].alg.ve.w === pr1.w
            end
            @test isa(r[5].alg, HighOrderDeviation)
            @test isa(r[5].alg.alg, FourthLowerMoment)
            val = X * w .- dot(w, pr1.mu)
            val = val[val .<= zero(eltype(val))]
            s = std(r[5].alg.ve, val; mean = zero(eltype(val)))
            @test isapprox(expected_risk(r[5], w, X), sum(val .^ 4) / size(X, 1) / s^4)
            val = prv.X * wv .- dot(wv, prv.mu)
            val = val[val .<= zero(eltype(val))]

            @test r[6].mu === pr1.mu
            if isa(pr1,
                   HighOrderPriorResult{<:LowOrderPriorResult, <:Any, <:Any, <:Any, <:Any})
                @test isnothing(r[6].alg.ve.w)
            elseif isa(pr1,
                       HighOrderPriorResult{<:EntropyPoolingPriorResult, <:Any, <:Any,
                                            <:Any, <:Any})
                @test r[6].alg.ve.w === pr1.pr.w
            else
                @test r[6].alg.ve.w === pr1.w
            end
            @test isa(r[6].alg, HighOrderDeviation)
            @test isa(r[6].alg.alg, FourthCentralMoment)
            x = X * w
            val = x .- dot(w, pr1.mu)
            s = std(r[6].alg.ve, x)
            @test isapprox(expected_risk(r[6], w, X), sum(val .^ 4) / size(X, 1) / s^4)
            x = prv.X * wv
            val = x .- dot(wv, prv.mu)
        end
    end
    @testset "Square root kurtosis and negative skewness" begin
        rng = StableRNG(123456789)
        X = randn(rng, 500, 20)
        ew = eweights(1:500, 1 / 500; scale = true)
        w = rand(rng, 20)
        w ./= sum(w)
        i = [20, 3, 9]
        wv = nothing_scalar_array_view(w, i)
        slv = Solver(; name = :Clarabel, solver = Clarabel.Optimizer,
                     settings = Dict("verbose" => false))
        ucs1 = DeltaUncertaintySetEstimator(;)
        ucs2 = NormalUncertaintySetEstimator(;)
        settings = RiskMeasureSettings(; rke = false, scale = -1, ub = 3)
        formulation = RSOC()
        ucs1view = ucs_view(ucs1, i)
        ucs2view = ucs_view(ucs2, i)
        rc = LinearConstraint(; A = LinearConstraintSide(; group = :A, name = :B), B = 0)

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
        sets = DataFrame(:Assets => 1:20)
        pes = [HighOrderPriorEstimator(),
               HighOrderPriorEstimator(;
                                       pe = EntropyPoolingPriorEstimator(; views = views,
                                                                         sets = sets,
                                                                         alg = H1_EntropyPooling())),
               EntropyPoolingPriorEstimator(; views = views, sets = sets,
                                            alg = H1_EntropyPooling()),
               EmpiricalPriorEstimator()]
        mu = rand(rng, 20)
        muv = nothing_scalar_array_view(mu, i)
        zerovec = fill(0.0, 20)
        zerovecv = nothing_scalar_array_view(zerovec, i)
        for pe ∈ pes
            pr1 = prior(pe, X)
            sigma = pr1.sigma * 1.5
            prv = prior_view(pr1, i)
            sigmav = nothing_scalar_array_view(sigma, i)
            kt = cokurtosis(Cokurtosis(), X; mean = transpose(pr1.mu))
            idx = fourth_moment_index_factory(size(X, 2), i)
            ktv = nothing_scalar_array_view(kt, idx)
            rs = [SquareRootKurtosis(; settings = settings, w = ew, mu = mu,
                                     kt = if isa(pr1, HighOrderPriorResult)
                                         pr1.kt
                                     else
                                         kt
                                     end),
                  SquareRootKurtosis(; mu = 0, kt = if isa(pr1, HighOrderPriorResult)
                                         nothing
                                     else
                                         kt
                                     end),
                  SquareRootKurtosis(; mu = zerovec,
                                     kt = if isa(pr1, HighOrderPriorResult)
                                         nothing
                                     else
                                         kt
                                     end),
                  SquareRootKurtosis(; alg = Semi(),
                                     kt = if isa(pr1, HighOrderPriorResult)
                                         nothing
                                     else
                                         kt
                                     end)]
            r = risk_measure_factory(rs, Ref(pr1), Ref(slv), Ref(ucs2))
            @test r[1].settings === settings
            @test r[1].w === ew
            @test r[1].mu === mu
            if isa(pr1, HighOrderPriorResult)
                @test r[1].kt === pr1.kt
            else
                @test r[1].kt === kt
            end
            val = X * w .- dot(w, mu)
            @test isapprox(expected_risk(r[1], w, X), sqrt(sum(val .^ 4) / size(X, 1)))
            @test rv[1].settings === settings
            @test rv[1].w === ew
            @test rv[1].mu === muv
            if isa(pr1, HighOrderPriorResult)
                @test rv[1].kt == prv.kt
            else
                @test rv[1].kt == ktv
            end
            val = prv.X * wv .- dot(wv, muv)
            @test isapprox(expected_risk(rv[1], wv, prv.X),
                           sqrt(sum(val .^ 4) / size(X, 1)))

            if isa(pr1,
                   HighOrderPriorResult{<:LowOrderPriorResult, <:Any, <:Any, <:Any, <:Any}) ||
               isa(pr1, LowOrderPriorResult)
                @test isnothing(r[2].w)
            elseif isa(pr1,
                       HighOrderPriorResult{<:EntropyPoolingPriorResult, <:Any, <:Any,
                                            <:Any, <:Any})
                @test r[2].w === pr1.pr.w
            else
                @test r[2].w === pr1.w
            end
            @test r[2].mu == 0
            if isa(pr1, HighOrderPriorResult)
                @test r[2].kt === pr1.kt
            else
                @test r[2].kt == kt
            end
            val = X * w
            @test isapprox(expected_risk(r[2], w, X), sqrt(sum(val .^ 4) / size(X, 1)))
            if isa(prv,
                   HighOrderPriorResult{<:LowOrderPriorResult, <:Any, <:Any, <:Any, <:Any}) ||
               isa(pr1, LowOrderPriorResult)
                @test isnothing(rv[2].w)
            elseif isa(prv,
                       HighOrderPriorResult{<:EntropyPoolingPriorResult, <:Any, <:Any,
                                            <:Any, <:Any})
                @test rv[2].w === prv.pr.w
            else
                @test rv[2].w === prv.w
            end
            @test rv[2].mu == 0
            if isa(pr1, HighOrderPriorEstimator)
                @test rv[2].kt == prv.kt
            else
                @test rv[2].kt == ktv
            end
            val = prv.X * wv
            @test isapprox(expected_risk(rv[2], wv, prv.X),
                           sqrt(sum(val .^ 4) / size(X, 1)))

            if isa(pr1,
                   HighOrderPriorResult{<:LowOrderPriorResult, <:Any, <:Any, <:Any, <:Any}) ||
               isa(pr1, LowOrderPriorResult)
                @test isnothing(r[3].w)
            elseif isa(pr1,
                       HighOrderPriorResult{<:EntropyPoolingPriorResult, <:Any, <:Any,
                                            <:Any, <:Any})
                @test r[3].w === pr1.pr.w
            else
                @test r[3].w === pr1.w
            end
            @test r[3].mu === zerovec
            if isa(pr1, HighOrderPriorEstimator)
                @test r[3].kt == pr1.kt
            else
                @test r[3].kt == kt
            end
            val = X * w .- dot(w, zerovec)
            @test isapprox(expected_risk(r[3], w, X), sqrt(sum(val .^ 4) / size(X, 1)))
            if isa(pr1,
                   HighOrderPriorResult{<:LowOrderPriorResult, <:Any, <:Any, <:Any, <:Any}) ||
               isa(pr1, LowOrderPriorResult)
                @test isnothing(rv[3].w)
            elseif isa(pr1,
                       HighOrderPriorResult{<:EntropyPoolingPriorResult, <:Any, <:Any,
                                            <:Any, <:Any})
                @test rv[3].w === pr1.pr.w
            else
                @test rv[3].w === pr1.w
            end
            @test rv[3].mu === zerovecv
            if isa(pr1, HighOrderPriorEstimator)
                @test rv[3].kt == prv.kt
            else
                @test rv[3].kt == ktv
            end
            val = prv.X * wv .- dot(wv, zerovecv)
            @test isapprox(expected_risk(rv[3], wv, prv.X),
                           sqrt(sum(val .^ 4) / size(X, 1)))

            @test r[4].alg == Semi()
            @test r[4].mu === pr1.mu
            if isa(pr1, HighOrderPriorEstimator)
                @test r[4].kt == pr1.kt
            else
                @test r[4].kt == kt
            end
            val = X * w .- dot(w, pr1.mu)
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(r[4], w, X), sqrt(sum(val .^ 4) / size(X, 1)))
            @test rv[4].alg == Semi()
            @test rv[4].mu === prv.mu
            if isa(pr1, HighOrderPriorEstimator)
                @test rv[4].kt == prv.kt
            else
                @test rv[4].kt == ktv
            end
            val = prv.X * wv .- dot(wv, prv.mu)
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(rv[4], wv, prv.X),
                           sqrt(sum(val .^ 4) / size(X, 1)))

            ske = Coskewness()
            sk, V = coskewness(Coskewness(), X)
            skv = nothing_scalar_array_view_odd_order(sk, i, idx)
            Vv = __coskewness(skv, view(pr1.X, :, i), ske.mp)
            rs = [NegativeSkewness(; settings = settings, alg = QuadraticNegativeSkewness(),
                                   sk = if isa(pr1, HighOrderPriorEstimator)
                                       pr1.sk
                                   else
                                       sk
                                   end, V = if isa(pr1, HighOrderPriorEstimator)
                                       pr1.V
                                   else
                                       V
                                   end),
                  NegativeSkewness(; sk = if isa(pr1, HighOrderPriorEstimator)
                                       pr1.sk
                                   else
                                       sk
                                   end, V = if isa(pr1, HighOrderPriorEstimator)
                                       pr1.V
                                   else
                                       V
                                   end), NegativeSkewness(;)]
            r = risk_measure_factory(rs[1:2], Ref(pr1), Ref(slv), Ref(ucs2))

            @test r[1].settings === settings
            @test r[1].alg === QuadraticNegativeSkewness()
            if isa(pr1, HighOrderPriorEstimator)
                @test r[1].sk === pr1.sk
                @test r[1].V === pr1.V
                @test isapprox(expected_risk(r[1], w, X), dot(w, pr1.V, w))
            else
                @test r[1].sk == sk
                @test r[1].V == V
                @test isapprox(expected_risk(r[1], w, X), dot(w, V, w))
            end
            @test rv[1].settings === settings
            @test rv[1].alg === QuadraticNegativeSkewness()
            if isa(pr1, HighOrderPriorEstimator)
                @test rv[1].sk == prv.sk
                @test rv[1].V == prv.V
                @test isapprox(expected_risk(rv[1], wv, prv.X), dot(wv, prv.V, wv))
            else
                @test rv[1].sk == skv
                @test rv[1].V == Vv
                @test isapprox(expected_risk(rv[1], wv, prv.X), dot(wv, Vv, wv))
            end
            if isa(pr1, HighOrderPriorEstimator)
                @test r[2].sk === pr1.sk
                @test r[2].V === pr1.V
                @test isapprox(expected_risk(r[2], w, X), sqrt(dot(w, pr1.V, w)))
            else
                @test r[2].sk == sk
                @test r[2].V == V
                @test isapprox(expected_risk(r[2], w, X), sqrt(dot(w, V, w)))
            end
            if isa(pr1, HighOrderPriorEstimator)
                @test rv[2].sk == prv.sk
                @test rv[2].V == prv.V
                @test isapprox(expected_risk(rv[2], wv, prv.X), sqrt(dot(wv, prv.V, wv)))
            else
                @test rv[2].sk == skv
                @test rv[2].V == Vv
                @test isapprox(expected_risk(rv[2], wv, prv.X), sqrt(dot(wv, Vv, wv)))
            end
            if isa(pr1, HighOrderPriorResult)
                r = risk_measure_factory(rs[3], pr1, slv, ucs2)
                @test r.sk === pr1.sk
                @test r.V === pr1.V
            else
                @test_throws ArgumentError risk_measure_factory(rs[3], pr1, slv, ucs2)
                @test_throws ArgumentError risk_measure_view(rs[3], pr1, i, slv, ucs2)
            end
        end
    end
    @testset "X at risk" begin
        rng = StableRNG(987654321)
        X = randn(rng, 500, 20) * 0.01
        w = rand(rng, 20)
        w ./= sum(w)
        slv = [Solver(; name = :Clarabel, solver = Clarabel.Optimizer,
                      settings = Dict("verbose" => false))]
        settings = HierarchicalRiskMeasureSettings(; scale = 2)
        rs1 = [ValueatRisk(; settings = settings, alpha = 0.1),
               ValueatRiskRange(; settings = settings, alpha = 0.2, beta = 0.3),
               DrawdownatRisk(; settings = settings, alpha = 0.4),
               RelativeDrawdownatRisk(; settings = settings, alpha = 0.5)]
        r1 = risk_measure_factory(rs1)
        @test all(rs1 .=== r1)
        @test all(rs1 .=== rv1)
        er1 = expected_risk.(r1, Ref(w), Ref(X))
        res = isapprox(er1,
                       [0.002979408952834433, 0.0033497521220081376, 0.01747290349846975,
                        0.011423759846772574])
        if !res
            find_tol(er1,
                     [0.002979408952834433, 0.0033497521220081376, 0.01747290349846975,
                      0.011423759846772574]; name1 = :er1, name2 = :res)
        end
        @test res

        settings = RiskMeasureSettings(; rke = false, scale = 2, ub = 3)
        hcsettings = HierarchicalRiskMeasureSettings(; scale = -2)
        rs2 = [ConditionalValueatRisk(; settings = settings, alpha = 0.1),
               DistributionallyRobustConditionalValueatRisk(; alpha = 0.1),
               ConditionalValueatRiskRange(; settings = settings, alpha = 0.2, beta = 0.3),
               ConditionalDrawdownatRisk(; settings = settings, alpha = 0.4),
               RelativeConditionalDrawdownatRisk(; settings = hcsettings, alpha = 0.5)]
        r2 = risk_measure_factory(rs2)
        @test all(rs2 .=== r2)
        @test all(rs2 .=== rv2)
        er2 = expected_risk.(r2, Ref(w), Ref(X))
        res = isapprox(er2,
                       [0.004336619383058753, 0.004336619383058753, 0.002433632636088897,
                        0.030517667443021403, 0.012855891046985323])
        if !res
            find_tol(er1,
                     [0.004336619383058753, 0.004336619383058753, 0.002433632636088897,
                      0.030517667443021403, 0.012855891046985323]; name1 = :er1,
                     name2 = :res)
        end
        @test res

        rs3 = [EntropicValueatRisk(; settings = settings, alpha = 0.1, slv = slv),
               EntropicValueatRiskRange(; settings = settings, alpha = 0.2, beta = 0.3,
                                        slv = slv),
               EntropicDrawdownatRisk(; settings = settings, alpha = 0.4),
               RelativeEntropicDrawdownatRisk(; settings = hcsettings, alpha = 0.5)]
        r3 = risk_measure_factory(rs3, nothing, Ref(slv))
        @test all(rs3[1:2] .=== r3[1:2])
        @test r3[3].slv === slv
        @test r3[4].slv === slv
        @test all(r3 .=== rv3)
        er3 = expected_risk.(r3, Ref(w), Ref(X))
        res = isapprox(er3,
                       [0.0053317677739742235, 0.007936600442013136, 0.034345738932130394,
                        0.03172723567187419]; rtol = 1e-6)
        if !res
            find_tol(er1,
                     [0.0053317677739742235, 0.007936600442013136, 0.034345738932130394,
                      0.03172723567187419]; name1 = :er1, name2 = :res)
        end
        @test res

        rs4 = [RelativisticValueatRisk(; settings = settings, alpha = 0.1, kappa = 0.4,
                                       slv = slv),
               RelativisticValueatRiskRange(; settings = settings, alpha = 0.02,
                                            kappa_a = 0.2, beta = 0.05, kappa_b = 0.2,
                                            slv = slv),
               RelativisticDrawdownatRisk(; settings = settings, alpha = 0.4, kappa = 0.35),
               RelativeRelativisticDrawdownatRisk(; settings = hcsettings, alpha = 0.5,
                                                  kappa = 0.25)]
        r4 = risk_measure_factory(rs4, nothing, Ref(slv))
        @test all(rs4[1:2] .=== r4[1:2])
        @test r4[3].slv === slv
        @test r4[4].slv === slv
        @test all(r4 .=== rv4)
        er4 = expected_risk.(r4, Ref(w), Ref(X))
        res = isapprox(er4,
                       [0.00686226055400148, 0.01232999076425514, 0.03696131888913639,
                        0.03325026478122334]; rtol = 5e-7)
        if !res
            find_tol(er1,
                     [0.00686226055400148, 0.01232999076425514, 0.03696131888913639,
                      0.03325026478122334]; name1 = :er1, name2 = :res)
        end
        @test res
    end
    @testset "OWA" begin
        rng = StableRNG(987456321)
        X = randn(rng, 500, 20) * 0.01
        w = rand(rng, 20)
        w ./= sum(w)
        settings = RiskMeasureSettings(; rke = false, scale = 2, ub = 3)
        rs1 = [OrderedWeightsArray(; settings = settings),
               OrderedWeightsArray(; formulation = ExactOrderedWeightsArray(),
                                   w = owa_tg(500; alpha_i = 0.001, alpha = 0.06,
                                              a_sim = 70)), OrderedWeightsArray(),
               OrderedWeightsArray(;
                                   w = owa_tg(500; alpha_i = 0.001, alpha = 0.06,
                                              a_sim = 70)),
               OrderedWeightsArray(; w = owa_tgrg(500))]
        r1 = risk_measure_factory(rs1)
        @test all(rs1 .=== r1)
        @test all(rs1 .=== rv1)
        er1 = expected_risk.(r1, Ref(w), Ref(X))
        @test er1[1] == er1[3]
        @test er1[2] == er1[4]
        @test isapprox(er1,
                       [0.0027927028241994376, 0.005092766618669591, 0.0027927028241994376,
                        0.005092766618669591, 0.010867592331463578])
    end
    @testset "Average drawdown" begin
        rng = StableRNG(123456789)
        X = randn(rng, 500, 10)
        ew = eweights(1:500, 1 / 500; scale = true)
        w = rand(rng, 10)
        w ./= sum(w)
        views = [ContinuousEntropyPoolingViewEstimator(;
                                                       A = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = :Assets,
                                                                                                      coef = 1,
                                                                                                      name = 1),
                                                       B = ConstantEntropyPoolingConstraintEstimator(;
                                                                                                     coef = 0.02),
                                                       comp = EQ())]
        sets = DataFrame(:Assets => 1:10)
        pes = [EmpiricalPriorEstimator(),
               EntropyPoolingPriorEstimator(; views = views, sets = sets,
                                            alg = H1_EntropyPooling()),
               HighOrderPriorEstimator(;
                                       pe = EntropyPoolingPriorEstimator(; views = views,
                                                                         sets = sets,
                                                                         alg = H1_EntropyPooling()))]
        settings = RiskMeasureSettings(; rke = false, scale = 2, ub = 3)
        for pe ∈ pes
            pr1 = prior(pe, X)
            rs = [AverageDrawdown(; settings = settings), RelativeAverageDrawdown(;),
                  AverageDrawdown(; w = ew), RelativeAverageDrawdown(; w = ew)]
            r = risk_measure_factory(rs, Ref(pr1))
            if isa(pr1, LowOrderPriorResult)
                @test all(r .=== rs)
                @test all(rv .=== rs)
                @test isapprox(expected_risk.(r, Ref(w), Ref(X)),
                               [3.751365586265874, 0.9811305224406326, 3.36092436959623,
                                0.9888287864252111])
            elseif isa(pr1,
                       HighOrderPriorResult{<:EntropyPoolingPriorResult, <:Any, <:Any,
                                            <:Any, <:Any})
                r[1].w === pr1.pr.w
                r[2].w === pr1.pr.w
                r[3].w === ew
                r[4].w === ew
                @test all(rv .=== r)
                @test isapprox(expected_risk.(r, Ref(w), Ref(X)),
                               [3.7517665162794507, 0.9811101606850575, 3.36092436959623,
                                0.9888287864252111])
            else
                r[1].w === pr1.w
                r[2].w === pr1.w
                r[3].w === ew
                r[4].w === ew
                @test all(rv .=== r)
                @test isapprox(expected_risk.(r, Ref(w), Ref(X)),
                               [3.7517665162794507, 0.9811101606850575, 3.36092436959623,
                                0.9888287864252111])
            end
        end
    end
    @testset "Settings only risk measures" begin
        rng = StableRNG(123456789)
        X = randn(rng, 500, 10)
        w = rand(rng, 10)
        w ./= sum(w)
        pe = EmpiricalPriorEstimator()
        settings = RiskMeasureSettings(; rke = false, scale = 2, ub = 3)
        hcsettings = HierarchicalRiskMeasureSettings(; scale = -2)
        pr1 = prior(pe, X)
        rs = [UlcerIndex(; settings = settings),
              RelativeUlcerIndex(; settings = hcsettings), MaximumDrawdown(),
              RelativeMaximumDrawdown(), WorstRealisation(), Range(), EqualRiskMeasure()]
        r = risk_measure_factory(rs)
        @test all(rs .=== r)
        @test all(rs .=== rv)
        x = X * w
        lb, ub = extrema(x)
        @test isapprox(expected_risk.(r, Ref(w), Ref(X)),
                       [4.729197396913109, 0.9852613709401206, 10.012825798312244,
                        1.0000278890665872, -lb, ub - lb, inv(length(w))])
    end
    @testset "Tracking and tn" begin
        rng = StableRNG(123456789)
        X = randn(rng, 500, 10)
        w1 = 1:10
        w2 = 10:-1:1
        w1 /= sum(w1)
        w2 /= sum(w2)
        i = [8, 5, 3]
        w1v = nothing_scalar_array_view(w1, i)
        w2v = nothing_scalar_array_view(w2, i)
        Xv = view(X, :, i)
        settings = RiskMeasureSettings(; rke = false, scale = 5, ub = 3)
        rs = [TurnoverRiskMeasure(; settings = settings, w = w2),
              TrackingRiskMeasure(; tracking = WeightsTracking(; w = w2)),
              TrackingRiskMeasure(; settings = settings,
                                  tracking = ReturnsTracking(; w = X * w2))]
        r = risk_measure_factory(rs)

        @test all(rs .=== r)
        @test rv[1].w == rv[2].tracking.w == view(w2, i)
        @test rs[3] === r[3] === rv[3]

        er1 = expected_risk.(r, Ref(w2), Ref(X))
        @test all(iszero, er1)

        er2 = expected_risk.(r, Ref(w1), Ref(X))
        @test isapprox(er2,
                       [norm(w1 - w2, 1), norm((X * w1 - X * w2) / (sqrt(size(X, 1) - 1))),
                        norm((X * w1 - X * w2) / (sqrt(size(X, 1) - 1)))])

        er3 = expected_risk.(rv, Ref(w2v), Ref(Xv))
        @test all(iszero, er3[1:2])
        @test isapprox(er3[3], norm((Xv * w2v - X * w2) / (sqrt(size(X, 1) - 1))))

        er4 = expected_risk.(rv, Ref(w1v), Ref(Xv))
        @test isapprox(er4,
                       [norm(w1v - w2v, 1),
                        norm((Xv * w1v - Xv * w2v) / (sqrt(size(X, 1) - 1))),
                        norm((Xv * w1v - X * w2) / (sqrt(size(X, 1) - 1)))])
    end
    @testset "No optimisation risk measures" begin
        rng = StableRNG(123456789)
        X = randn(rng, 500, 20)
        ew = eweights(1:500, 1 / 500; scale = true)
        w = rand(rng, 20)
        w ./= sum(w)
        i = [20, 3, 9]
        wv = nothing_scalar_array_view(w, i)
        views = [ContinuousEntropyPoolingViewEstimator(;
                                                       A = C0_LinearEntropyPoolingConstraintEstimator(;
                                                                                                      group = :Assets,
                                                                                                      coef = 1,
                                                                                                      name = 1),
                                                       B = ConstantEntropyPoolingConstraintEstimator(;
                                                                                                     coef = 0.02),
                                                       comp = EQ())]
        sets = DataFrame(:Assets => 1:20)
        pes = [EmpiricalPriorEstimator(),
               HighOrderPriorEstimator(;
                                       pe = EntropyPoolingPriorEstimator(; views = views,
                                                                         sets = sets,
                                                                         alg = H1_EntropyPooling())),
               EntropyPoolingPriorEstimator(; views = views, sets = sets,
                                            alg = H1_EntropyPooling())]
        mu = rand(rng, 20)
        muv = nothing_scalar_array_view(mu, i)
        for pe ∈ pes
            pr1 = prior(pe, X)
            rs = [MeanReturn(;), MeanReturn(; w = ew), ThirdCentralMoment(),
                  ThirdCentralMoment(; mu = mu, w = ew), Skewness(),
                  Skewness(;
                           ve = SimpleVariance(; corrected = false,
                                               me = SimpleExpectedReturns(; w = ew),
                                               w = ew), w = ew, mu = mu),
                  ThirdCentralMoment(; mu = 0), Skewness(; mu = 0)]
            r = risk_measure_factory(rs, Ref(pr1))
            er = expected_risk.(r, Ref(w), Ref(X))
            if isa(pr1, LowOrderPriorResult)
                @test isnothing(r[1].w)
                @test isnothing(rv[1].w)
                @test isnothing(r[3].w)
                @test isnothing(rv[3].w)
                @test isnothing(r[5].w)
                @test isnothing(rv[5].w)
                @test isnothing(r[5].ve.me.w)
                @test isnothing(rv[5].ve.me.w)
                @test isapprox(er,
                               [0.005409516986537154, 0.0076144516706446825,
                                0.0013290846602404623, -0.19156209155128498,
                                0.08735509114197064, -1.2778663258289709,
                                0.002323701237582384, 0.1526176806960154])
            elseif isa(pr1, HighOrderPriorResult)
                @test r[1].w === pr1.pr.w
                @test rv[1].w === pr1.pr.w
                @test r[3].w === pr1.pr.w
                @test rv[3].w === pr1.pr.w
                @test r[5].w === pr1.pr.w
                @test rv[5].w === pr1.pr.w
                @test r[5].ve.me.w === pr1.pr.w
                @test rv[5].ve.me.w === pr1.pr.w
                @test isapprox(er,
                               [0.005211295487597505, 0.0076144516706446825,
                                0.0013655247071769562, -0.19156209155128498,
                                0.08979373959490125, -1.2778663258289709,
                                0.002323701237582384, 0.15269965197093327])
            else
                @test r[1].w === pr1.w
                @test rv[1].w === pr1.w
                @test r[3].w === pr1.w
                @test rv[3].w === pr1.w
                @test r[5].w === pr1.w
                @test rv[5].w === pr1.w
                @test r[5].ve.me.w === pr1.w
                @test rv[5].ve.me.w === pr1.w
                @test isapprox(er,
                               [0.005211295487597505, 0.0076144516706446825,
                                0.0013655247071769562, -0.19156209155128498,
                                0.08979373959490125, -1.2778663258289709,
                                0.002323701237582384, 0.15269965197093327])
            end
            @test r[2].w === ew
            @test rv[2].w === ew
            @test r[4].w === ew
            @test rv[4].w === ew
            @test r[4].mu === mu
            @test rv[4].mu === muv
            @test r[6].ve.me.w === ew
            @test rv[6].ve.me.w === ew
            @test r[7].mu == 0
            @test rv[7].mu == 0
            @test r[8].mu == 0
            @test rv[8].mu == 0
        end
    end
end
