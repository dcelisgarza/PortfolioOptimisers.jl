@safetestset "Risk Measure Tests" begin
    using PortfolioOptimisers, Test, Random, StableRNGs, CSV, DataFrames, StatsBase,
          Clarabel, LinearAlgebra
    import PortfolioOptimisers: risk_measure_factory, risk_measure_view, ucs_view,
                                nothing_scalar_array_view, prior_view
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
        ucs1view = ucs_view(ucs1, ucs2, i)
        ucs2view = ucs_view(nothing, ucs2, i)
        rc = LinearConstraint(; A = LinearConstraintSide(; group = :A, name = :B), B = 0)

        views = [EntropyPoolingViewEstimator(;
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
            rv = risk_measure_view(rs, Ref(pr1), Ref(i), Ref(slv), Ref(ucs2))
            @test r[1].settings === settings
            @test r[1].formulation === formulation
            @test r[1].sigma === sigma
            @test r[1].rc === rc
            @test isapprox(expected_risk(r[1], w, X), dot(w, sigma, w))
            @test rv[1].settings === settings
            @test rv[1].formulation === formulation
            @test rv[1].sigma === sigmav
            @test rv[1].rc === rc
            @test isapprox(expected_risk(rv[1], wv, prv.X), dot(wv, sigmav, wv))

            @test r[2].sigma === pr1.sigma
            @test isapprox(expected_risk(r[2], w, X), dot(w, pr1.sigma, w))
            @test rv[2].sigma === prv.sigma
            @test isapprox(expected_risk(rv[2], wv, prv.X), dot(wv, prv.sigma, wv))

            sigma = pr1.sigma * 2
            sigmav = nothing_scalar_array_view(sigma, i)
            rs = [StandardDeviation(; settings = settings, sigma = sigma),
                  StandardDeviation(;)]
            r = risk_measure_factory(rs, Ref(pr1), Ref(slv), Ref(ucs2))
            rv = risk_measure_view(rs, Ref(pr1), Ref(i), Ref(slv), Ref(ucs2))
            @test r[1].settings === settings
            @test r[1].sigma === sigma
            @test isapprox(expected_risk(r[1], w, X), sqrt(dot(w, sigma, w)))
            @test rv[1].settings === settings
            @test rv[1].sigma === sigmav
            @test isapprox(expected_risk(rv[1], wv, prv.X), sqrt(dot(wv, sigmav, wv)))

            @test r[2].sigma === pr1.sigma
            @test isapprox(expected_risk(r[2], w, X), sqrt(dot(w, pr1.sigma, w)))
            @test rv[2].sigma === prv.sigma
            @test isapprox(expected_risk(rv[2], wv, prv.X), sqrt(dot(wv, prv.sigma, wv)))

            sigma = pr1.sigma * 2.5
            sigmav = nothing_scalar_array_view(sigma, i)
            rs = [UncertaintySetVariance(; settings = settings, ucs = ucs1, sigma = sigma),
                  UncertaintySetVariance(;)]
            r = risk_measure_factory(rs, Ref(pr1), Ref(slv), Ref(ucs2))
            rv = risk_measure_view(rs, Ref(pr1), Ref(i), Ref(slv), Ref(ucs2))
            @test r[1].settings === settings
            @test r[1].ucs === ucs1
            @test r[1].sigma === sigma
            @test isapprox(expected_risk(r[1], w, X), dot(w, sigma, w))
            @test rv[1].settings === settings
            @test rv[1].ucs === ucs1view
            @test rv[1].sigma === sigmav
            @test isapprox(expected_risk(rv[1], wv, prv.X), dot(wv, sigmav, wv))

            @test r[2].sigma === pr1.sigma
            @test r[2].ucs === ucs2
            @test isapprox(expected_risk(r[2], w, X), dot(w, pr1.sigma, w))
            @test rv[2].sigma === prv.sigma
            @test rv[2].ucs === ucs2view
            @test isapprox(expected_risk(rv[2], wv, prv.X), dot(wv, prv.sigma, wv))

            formulation = Quad()
            rs = [LowOrderMoment(; settings = settings, w = ew, mu = mu),
                  LowOrderMoment(; mu = 0), LowOrderMoment(; mu = zerovec),
                  LowOrderMoment(; alg = SemiDeviation(; ddof = 2)),
                  LowOrderMoment(;
                                 alg = SemiVariance(; ddof = 3, formulation = formulation)),
                  LowOrderMoment(; alg = MeanAbsoluteDeviation()),
                  LowOrderMoment(; alg = MeanAbsoluteDeviation(; w = ew))]
            r = risk_measure_factory(rs, Ref(pr1), Ref(slv), Ref(ucs2))
            rv = risk_measure_view(rs, Ref(pr1), Ref(i), Ref(slv), Ref(ucs2))
            @test r[1].settings === settings
            @test r[1].w === ew
            @test r[1].mu === mu
            val = X * w .- dot(w, mu)
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(r[1], w, X), -sum(val) / size(X, 1))
            @test rv[1].settings === settings
            @test rv[1].w === ew
            @test rv[1].mu === muv
            val = prv.X * wv .- dot(wv, muv)
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(rv[1], wv, prv.X), -sum(val) / size(prv.X, 1))

            @test r[2].mu == 0
            val = X * w
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(r[2], w, X), -sum(val) / size(X, 1))
            @test rv[2].mu == 0
            val = prv.X * wv
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(rv[2], wv, prv.X), -sum(val) / size(prv.X, 1))

            @test isa(r[3].alg, FirstLowerMoment)
            @test r[3].mu === zerovec
            val = X * w .- dot(w, zerovec)
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(r[3], w, X), -sum(val) / size(X, 1))
            @test rv[3].mu === zerovecv
            val = prv.X * wv .- dot(wv, zerovecv)
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(rv[3], wv, prv.X), -sum(val) / size(prv.X, 1))

            @test r[4].alg.ddof == 2
            @test r[4].mu === pr1.mu
            val = X * w .- dot(w, pr1.mu)
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(r[4], w, X),
                           sqrt(dot(val, val) / (size(X, 1) - r[4].alg.ddof)))
            @test isa(rv[4].alg, SemiDeviation)
            @test rv[4].alg.ddof == 2
            @test rv[4].mu === prv.mu
            val = prv.X * wv .- dot(wv, prv.mu)
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(rv[4], wv, prv.X),
                           sqrt(dot(val, val) / (size(prv.X, 1) - rv[4].alg.ddof)))

            @test r[5].alg.ddof == 3
            @test r[5].alg.formulation === formulation
            @test r[5].mu === pr1.mu
            val = X * w .- dot(w, pr1.mu)
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(r[5], w, X),
                           dot(val, val) / (size(X, 1) - r[5].alg.ddof))
            @test isa(rv[5].alg, SemiVariance)
            @test rv[5].alg.ddof == 3
            @test rv[5].alg.formulation === formulation
            @test rv[5].mu === prv.mu
            val = prv.X * wv .- dot(wv, prv.mu)
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(rv[5], wv, prv.X),
                           dot(val, val) / (size(prv.X, 1) - rv[5].alg.ddof))

            if isa(pr1,
                   HighOrderPriorResult{<:EmpiricalPriorResult, <:Any, <:Any, <:Any, <:Any})
                @test isnothing(r[6].alg.w)
            elseif isa(pr1,
                       HighOrderPriorResult{<:EntropyPoolingPriorResult, <:Any, <:Any,
                                            <:Any, <:Any})
                @test r[6].alg.w === pr1.pr.w
            end
            @test r[6].mu === pr1.mu
            val = X * w .- dot(w, pr1.mu)
            @test isapprox(expected_risk(r[6], w, X), mean(abs.(val)))
            if isa(pr1,
                   HighOrderPriorResult{<:EmpiricalPriorResult, <:Any, <:Any, <:Any, <:Any})
                @test isnothing(rv[6].alg.w)
            elseif isa(pr1,
                       HighOrderPriorResult{<:EntropyPoolingPriorResult, <:Any, <:Any,
                                            <:Any, <:Any})
                @test rv[6].alg.w === prv.pr.w
            else
                @test rv[6].alg.w === prv.w
            end
            @test rv[6].mu === prv.mu
            val = prv.X * wv .- dot(wv, prv.mu)
            @test isapprox(expected_risk(rv[6], wv, prv.X), mean(abs.(val)))

            if isa(pr1,
                   HighOrderPriorResult{<:EmpiricalPriorResult, <:Any, <:Any, <:Any, <:Any})
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
                                       HighOrderPriorResult{<:EmpiricalPriorResult, <:Any,
                                                            <:Any, <:Any, <:Any})
                                    ew
                                elseif isa(pr1,
                                           HighOrderPriorResult{<:EntropyPoolingPriorResult,
                                                                <:Any, <:Any, <:Any, <:Any})
                                    pr1.pr.w
                                else
                                    pr1.w
                                end))
            if isa(pr1,
                   HighOrderPriorResult{<:EmpiricalPriorResult, <:Any, <:Any, <:Any, <:Any})
                @test rv[7].alg.w === ew
            elseif isa(pr1,
                       HighOrderPriorResult{<:EntropyPoolingPriorResult, <:Any, <:Any,
                                            <:Any, <:Any})
                @test rv[7].alg.w === prv.pr.w
            else
                @test rv[7].alg.w === prv.w
            end
            @test rv[7].mu === prv.mu
            val = prv.X * wv .- dot(wv, prv.mu)
            @test isapprox(expected_risk(rv[7], wv, prv.X),
                           mean(abs.(val),
                                if isa(pr1,
                                       HighOrderPriorResult{<:EmpiricalPriorResult, <:Any,
                                                            <:Any, <:Any, <:Any})
                                    ew
                                elseif isa(pr1,
                                           HighOrderPriorResult{<:EntropyPoolingPriorResult,
                                                                <:Any, <:Any, <:Any, <:Any})
                                    prv.pr.w
                                else
                                    prv.w
                                end))

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
            rv = risk_measure_view(rs, Ref(pr1), Ref(i), Ref(slv), Ref(ucs2))
            @test r[1].settings === settings
            @test r[1].w === ew
            @test r[1].mu === mu
            val = X * w .- dot(w, mu)
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(r[1], w, X), -sum(val .^ 3) / size(X, 1))
            @test rv[1].settings === settings
            @test rv[1].w === ew
            @test rv[1].mu === muv
            val = prv.X * wv .- dot(wv, muv)
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(rv[1], wv, prv.X), -sum(val .^ 3) / size(prv.X, 1))

            @test r[2].mu === 0
            val = X * w
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(r[2], w, X), sum(val .^ 4) / size(X, 1))
            @test rv[2].mu === 0
            val = prv.X * wv
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(rv[2], wv, prv.X), sum(val .^ 4) / size(X, 1))

            @test isa(r[3].alg, FourthCentralMoment)
            @test r[3].mu === zerovec
            val = X * w .- dot(w, zerovec)
            @test isapprox(expected_risk(r[3], w, X), sum(val .^ 4) / size(X, 1))
            @test isa(rv[3].alg, FourthCentralMoment)
            @test rv[3].mu === zerovecv
            val = prv.X * wv .- dot(wv, zerovecv)
            @test isapprox(expected_risk(rv[3], wv, prv.X), sum(val .^ 4) / size(prv.X, 1))

            @test r[4].mu === pr1.mu
            val = X * w .- dot(w, pr1.mu)
            val = val[val .<= zero(eltype(val))]
            s = std(r[4].alg.ve, val; mean = zero(eltype(val)))
            @test isapprox(expected_risk(r[4], w, X), -sum(val .^ 3) / size(X, 1) / s^3)
            @test rv[4].mu === prv.mu
            val = prv.X * wv .- dot(wv, prv.mu)
            val = val[val .<= zero(eltype(val))]
            s = std(rv[4].alg.ve, val; mean = zero(eltype(val)))
            @test isapprox(expected_risk(rv[4], wv, prv.X),
                           -sum(val .^ 3) / size(prv.X, 1) / s^3)

            @test r[5].mu === pr1.mu
            if isa(pr1,
                   HighOrderPriorResult{<:EmpiricalPriorResult, <:Any, <:Any, <:Any, <:Any})
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
            @test rv[5].mu === prv.mu
            if isa(pr1,
                   HighOrderPriorResult{<:EmpiricalPriorResult, <:Any, <:Any, <:Any, <:Any})
                @test rv[5].alg.ve.w === ew
            elseif isa(pr1,
                       HighOrderPriorResult{<:EntropyPoolingPriorResult, <:Any, <:Any,
                                            <:Any, <:Any})
                @test rv[5].alg.ve.w === prv.pr.w
            else
                @test rv[5].alg.ve.w === prv.w
            end
            @test isa(rv[5].alg, HighOrderDeviation)
            @test isa(rv[5].alg.alg, FourthLowerMoment)
            val = prv.X * wv .- dot(wv, prv.mu)
            val = val[val .<= zero(eltype(val))]
            s = std(rv[5].alg.ve, val; mean = zero(eltype(val)))
            @test isapprox(expected_risk(rv[5], wv, prv.X),
                           sum(val .^ 4) / size(prv.X, 1) / s^4)

            @test r[6].mu === pr1.mu
            if isa(pr1,
                   HighOrderPriorResult{<:EmpiricalPriorResult, <:Any, <:Any, <:Any, <:Any})
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
            @test rv[6].mu === prv.mu
            if isa(pr1,
                   HighOrderPriorResult{<:EmpiricalPriorResult, <:Any, <:Any, <:Any, <:Any})
                @test isnothing(rv[6].alg.ve.w)
            elseif isa(pr1,
                       HighOrderPriorResult{<:EntropyPoolingPriorResult, <:Any, <:Any,
                                            <:Any, <:Any})
                @test rv[6].alg.ve.w === prv.pr.w
            else
                @test rv[6].alg.ve.w === prv.w
            end
            @test isa(rv[6].alg, HighOrderDeviation)
            @test isa(rv[6].alg.alg, FourthCentralMoment)
            x = prv.X * wv
            val = x .- dot(wv, prv.mu)
            s = std(rv[6].alg.ve, x)
            @test isapprox(expected_risk(rv[6], wv, prv.X),
                           sum(val .^ 4) / size(prv.X, 1) / s^4)
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
        ucs1view = ucs_view(ucs1, ucs2, i)
        ucs2view = ucs_view(nothing, ucs2, i)
        rc = LinearConstraint(; A = LinearConstraintSide(; group = :A, name = :B), B = 0)

        views = [EntropyPoolingViewEstimator(;
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
                                                                         alg = H1_EntropyPooling()))]
        mu = rand(rng, 20)
        muv = nothing_scalar_array_view(mu, i)
        zerovec = fill(0.0, 20)
        zerovecv = nothing_scalar_array_view(zerovec, i)
        for pe ∈ pes
            pr1 = prior(pe, X)
            sigma = pr1.sigma * 1.5
            prv = prior_view(pr1, i)
            sigmav = nothing_scalar_array_view(sigma, i)
            rs = [SquareRootKurtosis(; settings = settings, w = ew, mu = mu, kt = pr1.kt),
                  SquareRootKurtosis(; mu = 0), SquareRootKurtosis(; mu = zerovec),
                  SquareRootKurtosis(; alg = Semi())]
            r = risk_measure_factory(rs, Ref(pr1), Ref(slv), Ref(ucs2))
            rv = risk_measure_view(rs, Ref(pr1), Ref(i), Ref(slv), Ref(ucs2))
            @test r[1].settings === settings
            @test r[1].w === ew
            @test r[1].mu === mu
            @test r[1].kt === pr1.kt
            val = X * w .- dot(w, mu)
            @test isapprox(expected_risk(r[1], w, X), sqrt(sum(val .^ 4) / size(X, 1)))
            @test rv[1].settings === settings
            @test rv[1].w === ew
            @test rv[1].mu === muv
            @test rv[1].kt == prv.kt
            val = prv.X * wv .- dot(wv, muv)
            @test isapprox(expected_risk(rv[1], wv, prv.X),
                           sqrt(sum(val .^ 4) / size(X, 1)))

            @test isnothing(r[2].w)
            @test r[2].mu == 0
            @test r[2].kt === pr1.kt
            val = X * w
            @test isapprox(expected_risk(r[2], w, X), sqrt(sum(val .^ 4) / size(X, 1)))
            @test isnothing(rv[2].w)
            @test rv[2].mu == 0
            @test rv[2].kt == prv.kt
            val = prv.X * wv
            @test isapprox(expected_risk(rv[2], wv, prv.X),
                           sqrt(sum(val .^ 4) / size(X, 1)))

            @test isnothing(r[3].w)
            @test r[3].mu === zerovec
            @test r[3].kt === pr1.kt
            val = X * w .- dot(w, zerovec)
            @test isapprox(expected_risk(r[3], w, X), sqrt(sum(val .^ 4) / size(X, 1)))
            @test isnothing(rv[3].w)
            @test rv[3].mu === zerovecv
            @test rv[3].kt == prv.kt
            val = prv.X * wv .- dot(wv, zerovecv)
            @test isapprox(expected_risk(rv[3], wv, prv.X),
                           sqrt(sum(val .^ 4) / size(X, 1)))

            @test r[4].alg == Semi()
            @test r[4].mu === pr1.mu
            @test r[4].kt === pr1.kt
            val = X * w .- dot(w, pr1.mu)
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(r[4], w, X), sqrt(sum(val .^ 4) / size(X, 1)))
            @test rv[4].alg == Semi()
            @test rv[4].mu === prv.mu
            @test rv[4].kt == prv.kt
            val = prv.X * wv .- dot(wv, prv.mu)
            val = val[val .<= zero(eltype(val))]
            @test isapprox(expected_risk(rv[4], wv, prv.X),
                           sqrt(sum(val .^ 4) / size(X, 1)))

            rs = [NegativeSkewness(; settings = settings, alg = QuadraticNegativeSkewness(),
                                   sk = pr1.sk, V = pr1.V), NegativeSkewness(;)]
            r = risk_measure_factory(rs, Ref(pr1), Ref(slv), Ref(ucs2))
            rv = risk_measure_view(rs, Ref(pr1), Ref(i), Ref(slv), Ref(ucs2))

            @test r[1].settings === settings
            @test r[1].alg === QuadraticNegativeSkewness()
            @test r[1].sk === pr1.sk
            @test r[1].V === pr1.V
            @test isapprox(expected_risk(r[1], w, X), dot(w, pr1.V, w))
            @test rv[1].settings === settings
            @test rv[1].alg === QuadraticNegativeSkewness()
            @test rv[1].sk == prv.sk
            @test rv[1].V == prv.V
            @test isapprox(expected_risk(rv[1], wv, prv.X), dot(wv, prv.V, wv))

            @test r[2].alg == LinearNegativeSkewness()
            @test r[2].sk === pr1.sk
            @test r[2].V === pr1.V
            @test isapprox(expected_risk(r[2], w, X), sqrt(dot(w, pr1.V, w)))
            @test rv[2].alg === LinearNegativeSkewness()
            @test rv[2].sk == prv.sk
            @test rv[2].V == prv.V
            @test isapprox(expected_risk(rv[2], wv, prv.X), sqrt(dot(wv, prv.V, wv)))
        end
    end
end
