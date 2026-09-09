#=
Check `calc_turnover` in `src/17_NetReturnsDrawdowns.jl`. Issue #937.

The verb is the library's first value-level turnover. `plot_turnover`
(`ext/PortfolioOptimisersPlotsExt.jl`) computed the same sum inline, which made it the one
place in the extension that computed a statistic rather than drawing one, and
`forecast_portfolio` needed the same number. The inline expression is kept below as the
behavioural reference, so the refactor is pinned to what the plot drew before it.

TWO CONVENTIONS SHAPE THE PROBES.

1. THE FIRST OBSERVATION IS `NaN`, NOT ZERO. A path that opens a book from cash trades its
   whole gross, and one that inherits a book trades nothing; the series cannot tell them
   apart, so it declines to answer. The answer therefore has one entry per observation of
   the path and the caller drops the leading gap, which is what `plot_turnover` does.

2. THE TWO METHODS ARE THE SAME NUMBER ON THE TWO SHAPES. A matrix carries one observation
   per row and a vector of vectors carries one per element, and the library holds a weight
   path in both shapes, so the two are asserted against each other rather than each against
   its own expectation.
=#
@testset "Value-level turnover" begin
    using Test, PortfolioOptimisers, StableRNGs

    PO = PortfolioOptimisers

    # The expression `plot_turnover` carried inline before #937, kept as the reference.
    ref_turnover(w) = [sum(abs, w[t] .- w[t - 1]) for t in 2:length(w)]

    @testset "The sum is the one-way absolute weight change" begin
        w = [0.5 0.5; 0.25 0.75; 0.25 0.75; -0.5 1.5]
        tn = calc_turnover(w)
        @test length(tn) == size(w, 1)
        @test isnan(tn[1])
        @test tn[2] ≈ 0.5
        @test tn[3] == 0
        @test tn[4] ≈ 1.5
    end

    @testset "A single observation carries only the leading gap" begin
        @test isequal(calc_turnover([0.4 0.6]), [NaN])
        @test isequal(calc_turnover([[0.4, 0.6]]), [NaN])
    end

    @testset "The matrix and the vector-of-vectors methods agree" begin
        rng = StableRNG(987654321)
        K, N = 7, 5
        w_series = [randn(rng, N) for _ in 1:K]
        W = permutedims(reduce(hcat, w_series))
        tn_v = calc_turnover(w_series)
        tn_m = calc_turnover(W)
        @test isequal(tn_v, tn_m)
        @test isnan(tn_v[1])
        @test tn_v[2:end] ≈ ref_turnover(w_series)
    end

    @testset "The bound is `<:Number`, so a complex path answers a real turnover" begin
        # `abs` of a complex weight is real, so nothing in the body orders the data and the
        # bound is wider than the `<:Real` the numeric kernels of the library carry.
        w = [1.0 + 0.0im 0.0 + 1.0im; 0.0 + 0.0im 0.0 + 0.0im]
        tn = calc_turnover(w)
        @test eltype(tn) === Float64
        @test tn[2] ≈ 2
    end

    @testset "A non-finite weight poisons its own observation and the next" begin
        # No guard: the verb reports the sum it computed. A `NaN` weight makes the two
        # observations that read it non-finite and leaves every other one alone.
        w = [0.5 0.5; NaN 0.5; 0.25 0.75; 0.25 0.75]
        tn = calc_turnover(w)
        @test isnan(tn[1])
        @test isnan(tn[2])
        @test isnan(tn[3])
        @test tn[4] == 0
    end
end
