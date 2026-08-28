# The drawdown series builders used to seed their running peak by mutating the caller's
# vector -- `pushfirst!(x, 0)`, compute, `popfirst!(x)`. That worked only because every
# caller handed them a freshly allocated `Vector` from `calc_net_returns`. Any other
# `AbstractVector` (a column view, a range) raised a `MethodError`, and an exception thrown
# between the two calls would have left the caller's vector one element longer than it
# started. The peak is now carried in a scalar, so `x` is read and never written.
@testset "Drawdown series are pure" begin
    using Test, PortfolioOptimisers, Random

    PO = PortfolioOptimisers
    Random.seed!(987654321)

    # the previous, mutating implementations, kept here as the behavioural reference
    function ref_absolute(x)
        x = copy(x)
        pushfirst!(x, zero(eltype(x)))
        cs = cumsum(x)
        peak = typemin(eltype(x))
        dd = similar(cs)
        for (idx, i) in pairs(cs)
            peak = ifelse(i > peak, i, peak)
            dd[idx] = i - peak
        end
        popfirst!(dd)
        return dd
    end
    function ref_relative(x)
        x = copy(x)
        pushfirst!(x, zero(eltype(x)))
        cs = cumprod(x .+ one(eltype(x)))
        peak = typemin(eltype(x))
        dd = similar(cs)
        for (idx, i) in pairs(cs)
            peak = ifelse(i > peak, i, peak)
            dd[idx] = i / peak - one(peak)
        end
        popfirst!(dd)
        return dd
    end

    @testset "identical to the previous implementation" begin
        for _ in 1:200
            x = 0.05 .* randn(rand(1:50))
            @test PO.absolute_drawdown_vec(x) ≈ ref_absolute(x)
            @test PO.relative_drawdown_vec(x) ≈ ref_relative(x)
        end
    end

    @testset "the input is not written" begin
        x = 0.05 .* randn(40)
        before = copy(x)
        PO.absolute_drawdown_vec(x)
        PO.relative_drawdown_vec(x)
        MaximumDrawdown()(x)
        @test x == before
        @test length(x) == 40
    end

    @testset "any AbstractVector works" begin
        X = 0.05 .* randn(30, 3)
        col = view(X, :, 2)
        @test PO.absolute_drawdown_vec(col) ≈ PO.absolute_drawdown_vec(X[:, 2])
        @test PO.relative_drawdown_vec(col) ≈ PO.relative_drawdown_vec(X[:, 2])
        @test MaximumDrawdown()(col) ≈ MaximumDrawdown()(X[:, 2])

        # a range is immutable; the old implementation could not take one at all
        r = 0.01:0.01:0.05
        @test PO.absolute_drawdown_vec(r) == zeros(5)   # monotone rise, no drawdown
        @test length(PO.relative_drawdown_vec(r)) == 5
    end

    @testset "the peak starts at the initial portfolio value" begin
        # a first observation below zero is already a drawdown, not the reference point
        @test PO.absolute_drawdown_vec([-0.1, 0.0]) ≈ [-0.1, -0.1]
        @test PO.relative_drawdown_vec([-0.1, 0.0]) ≈ [-0.1, -0.1]

        # and the drawdown series is never positive
        x = 0.05 .* randn(100)
        @test all(<=(0), PO.absolute_drawdown_vec(x))
        @test all(<=(0), PO.relative_drawdown_vec(x))
    end

    # `drawdowns` used to re-inline the two array builders and drop their `init`, so its
    # running peak started at the first observation. The JuMP path reads
    # `absolute_drawdown_arr` and the reporting path reads `drawdowns`, so the constraint
    # and the reported series disagreed on a portfolio that is under water at t = 1.
    @testset "the exported drawdowns agrees with the array builders" begin
        @test drawdowns([-0.1, 0.0]) ≈ [-0.1, -0.1]
        @test drawdowns([-0.1, 0.0], true) ≈ [-0.1, -0.1]
        @test all(<=(0), drawdowns(0.05 .* randn(100)))
        @test all(<=(0), drawdowns(0.05 .* randn(100), true))

        for _ in 1:50
            X = 0.05 .* randn(rand(2:40), rand(1:5))
            for dims in (1, 2)
                @test drawdowns(X; dims = dims) ≈ PO.absolute_drawdown_arr(X; dims = dims)
                @test drawdowns(X, true; dims = dims) ≈
                      PO.relative_drawdown_arr(X; dims = dims)
            end
            cXa = PO.absolute_cumulative_returns(X; dims = 1)
            cXr = PO.relative_cumulative_returns(X; dims = 1)
            @test drawdowns(cXa; cX = true) ≈ PO.absolute_drawdown_arr(cXa; cX = true)
            @test drawdowns(cXr, true; cX = true) ≈ PO.relative_drawdown_arr(cXr; cX = true)
        end

        # column by column, the array builder is the vector builder
        X = 0.05 .* randn(30, 4)
        for j in axes(X, 2)
            @test drawdowns(X)[:, j] ≈ PO.absolute_drawdown_vec(X[:, j])
            @test drawdowns(X, true)[:, j] ≈ PO.relative_drawdown_vec(X[:, j])
        end
    end
end

# The drawdown half of `src/17_NetReturnsDrawdowns.jl`, swept under issue #547. The
# running-peak convention, the `cX` short circuit and the `dims` axis are checked with
# numbers rather than read off the source.
@testset "The drawdown conventions" begin
    using Test, PortfolioOptimisers

    PO = PortfolioOptimisers

    @testset "the running peak starts at the initial capital" begin
        # A series under water from the first period reports a drawdown there. Were the
        # peak seeded at the first observation, both answers would open at `0.0`.
        @test drawdowns([-0.1, 0.05]) ≈ [-0.1, -0.05]
        @test drawdowns([-0.1, 0.05]) != zeros(2)
        @test PO.absolute_drawdown_arr([-0.1, 0.05]) ≈ [-0.1, -0.05]

        # the compound path is seeded at one and gives the matching relative answer
        @test drawdowns([-0.1, 0.05], true) ≈ [-0.09999999999999998, -0.05499999999999994]
        @test PO.relative_drawdown_arr([-0.1, 0.05]) ≈ drawdowns([-0.1, 0.05], true)

        # and the two agree to first order on a small return
        small = [-1e-6, 5e-7]
        @test maximum(abs, drawdowns(small) - drawdowns(small, true)) < 1e-12
    end

    @testset "cX = true skips the cumulation" begin
        x = [0.01, -0.02, 0.03, -0.015]
        @test drawdowns(PO.absolute_cumulative_returns(x); cX = true) == drawdowns(x)
        @test drawdowns(PO.relative_cumulative_returns(x), true; cX = true) ==
              drawdowns(x, true)

        # feeding the raw series with `cX = true` is a different answer, so the flag is read
        @test drawdowns(x; cX = true) != drawdowns(x)
    end

    @testset "dims = 2 transposes the answer of dims = 1" begin
        # A `dims` bug hides on a square matrix, so the fixture is 3x2.
        M = [0.1 -0.2; 0.0 0.0; 0.05 0.05]
        @test drawdowns(M; dims = 2) == permutedims(drawdowns(permutedims(M); dims = 1))
        @test drawdowns(M, true; dims = 2) ==
              permutedims(drawdowns(permutedims(M), true; dims = 1))
        @test cumulative_returns(M; dims = 2) ==
              permutedims(cumulative_returns(permutedims(M); dims = 1))
        @test cumulative_returns(M, true; dims = 2) ==
              permutedims(cumulative_returns(permutedims(M), true; dims = 1))

        # and the two axes genuinely disagree on this fixture
        @test drawdowns(M; dims = 1) != drawdowns(M; dims = 2)
        @test drawdowns(M; dims = 1) ≈ [0.0 -0.2; 0.0 -0.2; 0.0 -0.15]
        @test drawdowns(M; dims = 2) ≈ [0.0 -0.2; 0.0 0.0; 0.0 0.0]
    end

    @testset "cumulative_returns names the helper each branch reaches" begin
        x = [0.01, -0.02, 0.03, -0.015]
        @test cumulative_returns(x) == PO.absolute_cumulative_returns(x)
        @test cumulative_returns(x, true) == PO.relative_cumulative_returns(x)
        @test PO.absolute_cumulative_returns(x) == cumsum(x)
        @test PO.relative_cumulative_returns(x) == cumprod(1 .+ x)
    end
end
