#=
Unit tests for the `scalarise` reduction combinator and its slot-wise helpers.
These are the shared reducers behind `hrp_scalarised_risk`,
`herc_scalarised_risk_i!`/`_o!` and `near_optimal_centering_risks`; testing them
directly means the four reduction rules (sum, max, min, log-sum-exp) are checked
without needing an optimiser, prior, or solver.
=#
const PO = PortfolioOptimisers

@testset "scalarise scalars" begin
    xs = [1.0, 3.0, 2.0]
    @test PO.scalarise(identity, SumScalariser(), xs) == 6.0
    @test PO.scalarise(identity, MaxScalariser(), xs) == 3.0
    @test PO.scalarise(identity, MinScalariser(), xs) == 1.0
    gamma = 2.0
    lse = PO.scalarise(identity, LogSumExpScalariser(; gamma = gamma), xs)
    @test isapprox(lse, log(sum(exp.(gamma * xs))) / gamma)
    # Single element: every scalariser is the identity.
    for sca in (SumScalariser(), MaxScalariser(), MinScalariser(),
                LogSumExpScalariser(; gamma = gamma))
        @test isapprox(PO.scalarise(identity, sca, [4.2]), 4.2)
    end
end
@testset "scalarise arrays" begin
    xs = [1.0, 2.0]
    @test PO.scalarise(x -> [x, 2 * x], SumScalariser(), xs) == [3.0, 6.0]
    @test PO.scalarise(x -> [x, 3.0 - x], MaxScalariser(), xs) == [2.0, 2.0]
    @test PO.scalarise(x -> [x, 3.0 - x], MinScalariser(), xs) == [1.0, 1.0]
    gamma = 1.5
    lse = PO.scalarise(x -> [x, 3.0 - x], LogSumExpScalariser(; gamma = gamma), xs)
    @test isapprox(lse,
                   [log(sum(exp.(gamma * [1.0, 2.0]))) / gamma,
                    log(sum(exp.(gamma * [2.0, 1.0]))) / gamma])
end
@testset "scalarise tuples with mixed slots" begin
    # The NearOptimalCentering frontier shape: (scalar, vector, scalar).
    xs = [1.0, 2.0]
    f = x -> (x, [x, 3.0 - x], -x)
    s = PO.scalarise(f, SumScalariser(), xs)
    @test s[1] == 3.0 && s[2] == [3.0, 3.0] && s[3] == -3.0
    m = PO.scalarise(f, MaxScalariser(), xs)
    @test m[1] == 2.0 && m[2] == [2.0, 2.0] && m[3] == -1.0
    gamma = 2.0
    l = PO.scalarise(f, LogSumExpScalariser(; gamma = gamma), xs)
    @test isapprox(l[1], log(sum(exp.(gamma * xs))) / gamma)
    @test isapprox(l[2],
                   [log(sum(exp.(gamma * [1.0, 2.0]))) / gamma,
                    log(sum(exp.(gamma * [2.0, 1.0]))) / gamma])
    @test isapprox(l[3], log(sum(exp.(-gamma * xs))) / gamma)
end
@testset "scalarise keyed max/min" begin
    # `by` switches max/min from slot-wise to winner-take-all: the whole f
    # result with the extremal key is returned, as used by HRP (by = sum on the
    # (lrisk, rrisk) pair) and HERC's inner max/min (by = first on (key, value)).
    xs = [1.0, 3.0, 2.0]
    @test PO.scalarise(x -> (x, 10.0 * x), MaxScalariser(), xs; by = first) == (3.0, 30.0)
    @test PO.scalarise(x -> (x, 10.0 * x), MinScalariser(), xs; by = first) == (1.0, 10.0)
    @test PO.scalarise(x -> (x, 2.0 * x), MaxScalariser(), xs; by = sum) == (3.0, 6.0)
    # All keys tie: the earliest element wins.
    @test PO.scalarise(x -> (x, 10.0 - x), MaxScalariser(), xs; by = sum) == (1.0, 9.0)
    @test PO.scalarise(x -> (x, 10.0 - x), MinScalariser(), xs; by = sum) == (1.0, 9.0)
    # `by` is accepted and ignored by the accumulating scalarisers.
    @test PO.scalarise(identity, SumScalariser(), xs; by = sum) == 6.0
    gamma = 2.0
    @test isapprox(PO.scalarise(identity, LogSumExpScalariser(; gamma = gamma), xs;
                                by = sum), log(sum(exp.(gamma * xs))) / gamma)
end
@testset "scalarise helpers" begin
    @test PO.scalarise_combine(+, 1.0, 2.0) == 3.0
    @test PO.scalarise_combine(max, [1.0, 4.0], [2.0, 3.0]) == [2.0, 4.0]
    @test PO.scalarise_combine(+, (1.0, [1.0, 2.0]), (2.0, [3.0, 4.0])) == (3.0, [4.0, 6.0])
    @test PO.scalarise_map(x -> 2 * x, (1.0, [1.0, 2.0])) == (2.0, [2.0, 4.0])
    @test isapprox(PO.scalarise_logsumexp([1.0, 2.0]), log(exp(1.0) + exp(2.0)))
    @test isapprox(PO.scalarise_logsumexp([[1.0, 2.0], [2.0, 1.0]]),
                   [log(exp(1.0) + exp(2.0)), log(exp(2.0) + exp(1.0))])
    t = PO.scalarise_logsumexp([(1.0, [1.0, 2.0]), (2.0, [2.0, 1.0])])
    @test isapprox(t[1], log(exp(1.0) + exp(2.0)))
    @test isapprox(t[2], [log(exp(1.0) + exp(2.0)), log(exp(2.0) + exp(1.0))])
end
