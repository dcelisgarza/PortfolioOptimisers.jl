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
