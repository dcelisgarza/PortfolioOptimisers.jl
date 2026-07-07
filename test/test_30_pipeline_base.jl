@testset "Pipeline base" begin
    using Test, PortfolioOptimisers, TimeSeries, Dates, StableRNGs

    function make_prices(; T = 5, N = 3)
        rng = StableRNG(123456789)
        ts = Date(2020, 1, 1):Day(1):Date(2020, 1, T)
        X = TimeArray(ts, 100 .+ rand(rng, T, N), string.("A", 1:N))
        F = TimeArray(ts, 50 .+ rand(rng, T, 2), ["f1", "f2"])
        B = TimeArray(ts, 100 .+ rand(rng, T, 1), ["BM"])
        iv = TimeArray(ts, rand(rng, T, N), string.("A", 1:N))
        return X, F, B, iv
    end

    @testset "PricesResult validation" begin
        X, F, B, iv = make_prices()
        pr = PricesResult(; X = X)
        @test pr.X === X
        @test isnothing(pr.F)
        @test isnothing(pr.B)
        @test isnothing(pr.iv)
        @test isnothing(pr.ivpa)

        pr = PricesResult(; X = X, F = F, B = B, iv = iv, ivpa = 0.5)
        @test pr.F === F
        @test pr.B === B
        @test pr.iv === iv
        @test pr.ivpa == 0.5

        # per-asset ivpa vector
        pr = PricesResult(; X = X, iv = iv, ivpa = fill(0.5, 3))
        @test length(pr.ivpa) == 3

        # empty X
        Xe = TimeArray(Date[], zeros(0, 2), ["A", "B"])
        @test_throws PortfolioOptimisers.IsEmptyError PricesResult(; X = Xe)

        # benchmark column count must be 1 or match X
        Bbad = TimeArray(timestamp(X), rand(StableRNG(1), 5, 2), ["b1", "b2"])
        @test_throws DimensionMismatch PricesResult(; X = X, B = Bbad)

        # iv column count must match X
        ivbad = TimeArray(timestamp(X), rand(StableRNG(2), 5, 2), ["A1", "A2"])
        @test_throws DimensionMismatch PricesResult(; X = X, iv = ivbad)

        # iv must be non-negative
        ivneg = TimeArray(timestamp(X), -rand(StableRNG(3), 5, 3), ["A1", "A2", "A3"])
        @test_throws DomainError PricesResult(; X = X, iv = ivneg)

        # ivpa must be strictly positive and finite
        @test_throws DomainError PricesResult(; X = X, iv = iv, ivpa = 0.0)
        @test_throws DomainError PricesResult(; X = X, iv = iv, ivpa = Inf)

        # vector ivpa length must match asset count
        @test_throws DimensionMismatch PricesResult(; X = X, iv = iv,
                                                    ivpa = fill(0.5, 2))
    end

    @testset "prices_view" begin
        X, F, B, iv = make_prices()
        pr = PricesResult(; X = X, F = F, B = B, iv = iv, ivpa = fill(0.5, 3))

        # Colon is a passthrough
        @test PortfolioOptimisers.prices_view(pr, :) === pr

        # integer window
        pv = PortfolioOptimisers.prices_view(pr, 2:4)
        @test timestamp(pv.X) == timestamp(X)[2:4]
        @test values(pv.X) == values(X)[2:4, :]
        @test timestamp(pv.F) == timestamp(X)[2:4]
        @test values(pv.F) == values(F)[2:4, :]
        @test values(pv.B) == values(B)[2:4, :]
        @test values(pv.iv) == values(iv)[2:4, :]
        @test pv.ivpa === pr.ivpa

        # timestamp window
        window = timestamp(X)[[1, 3, 5]]
        pv = PortfolioOptimisers.prices_view(pr, window)
        @test timestamp(pv.X) == window
        @test values(pv.X) == values(X)[[1, 3, 5], :]

        # other series align to the master clock by timestamp: rows missing from a
        # series are dropped from that series, not invented
        Fshort = TimeArray(timestamp(X)[1:3], values(F)[1:3, :], ["f1", "f2"])
        pr2 = PricesResult(; X = X, F = Fshort)
        pv2 = PortfolioOptimisers.prices_view(pr2, 2:5)
        @test timestamp(pv2.X) == timestamp(X)[2:5]
        @test timestamp(pv2.F) == timestamp(X)[2:3]

        # optional series stay nothing
        pr3 = PricesResult(; X = X)
        pv3 = PortfolioOptimisers.prices_view(pr3, 2:3)
        @test isnothing(pv3.F)
        @test isnothing(pv3.B)
        @test isnothing(pv3.iv)
        @test isnothing(pv3.ivpa)
    end

    @testset "slot traits" begin
        # estimator families map to slots by dispatch
        @test PortfolioOptimisers.pipe_writes(EmpiricalPrior()) == :prior
        @test PortfolioOptimisers.pipe_reads(EmpiricalPrior()) == (:returns,)
        @test PortfolioOptimisers.pipe_writes(ClustersEstimator()) == :phylogeny
        @test PortfolioOptimisers.pipe_reads(ClustersEstimator()) == (:returns,)
        @test PortfolioOptimisers.pipe_writes(NormalUncertaintySet()) == :uncertainty
        @test PortfolioOptimisers.pipe_reads(NormalUncertaintySet()) == (:returns,)
        @test PortfolioOptimisers.pipe_writes(MeanRisk()) == :opt
        @test PortfolioOptimisers.pipe_reads(MeanRisk()) == (:returns,)

        # estimators without a steppable family are rejected with guidance
        @test_throws ArgumentError PortfolioOptimisers.pipe_writes(Covariance())
        @test_throws ArgumentError PortfolioOptimisers.pipe_reads(Covariance())
    end

    @testset "PipelineStep" begin
        ps = PipelineStep(; est = NormalUncertaintySet(), reads = (:returns,),
                          writes = :uncertainty, target = :mu)
        @test PortfolioOptimisers.pipe_writes(ps) == :uncertainty
        @test PortfolioOptimisers.pipe_reads(ps) == (:returns,)
        @test ps.target == :mu
        @test ps.est isa NormalUncertaintySet

        # reads defaults to empty, target to nothing
        ps = PipelineStep(; est = EmpiricalPrior(), writes = :prior)
        @test PortfolioOptimisers.pipe_reads(ps) == ()
        @test isnothing(ps.target)

        # callables are steppable when wrapped
        ps = PipelineStep(; est = x -> x, reads = (:prices,), writes = :prices)
        @test PortfolioOptimisers.pipe_writes(ps) == :prices

        # slot names are validated
        @test_throws ArgumentError PipelineStep(; est = EmpiricalPrior(),
                                                writes = :nonsense)
        @test_throws ArgumentError PipelineStep(; est = EmpiricalPrior(),
                                                reads = (:nonsense,), writes = :prior)
    end

    @testset "PipelineContext" begin
        ctx = PortfolioOptimisers.PipelineContext()
        @test all(s -> isnothing(getproperty(ctx, s)),
                  PortfolioOptimisers.PIPELINE_SLOTS)

        rd = ReturnsResult(; nx = ["a", "b"], X = [0.1 -0.2; -0.1 0.2; 0.05 0.1])
        pr = prior(EmpiricalPrior(), rd)
        ctx = PortfolioOptimisers.PipelineContext(; returns = rd, prior = pr)
        @test ctx.returns === rd
        @test ctx.prior === pr

        # slots are typed
        @test_throws MethodError PortfolioOptimisers.PipelineContext(; returns = pr)
    end
end
