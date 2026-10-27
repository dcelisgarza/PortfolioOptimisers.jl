using Clarabel
const PO = PortfolioOptimisers

# The sweep of `src/17_Optimisation/05_JuMP/04_MeanRisk.jl`. Each objective of `MeanRisk` is
# checked against its closed form on a budget-only problem, and each frontier sweep against
# the efficient set of markowitz1952: page 82 defines it as the minimum variance for a given
# expected return or more, and the maximum expected return for a given variance or less.
# Page 87 states that it runs from the portfolio of minimum variance to the portfolio of
# maximum expected return, which are the two corner solves of the sweep.

const MRS_X = let rng = StableRNG(686)
    T, N = 300, 5
    F = randn(rng, T, 2)
    0.0008 .+ F * (0.01 .* randn(rng, 2, N)) .+ 0.01 .* randn(rng, T, N) .+
    (0.0004 .* (1:N))'
end
const MRS_PR = prior(EmpiricalPrior(), MRS_X)
const MRS_SLV = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                       check_sol = (; allow_local = true, allow_almost = true),
                       settings = Dict("verbose" => false, "tol_gap_abs" => 1e-11,
                                       "tol_gap_rel" => 1e-11, "tol_feas" => 1e-11))
mrs_opt(; kwargs...) = JuMPOptimiser(; pe = MRS_PR, slv = MRS_SLV, kwargs...)
# No weight bound and a short budget that never binds: the budget row alone constrains the
# weights, so each objective has a closed form.
function mrs_free(; kwargs...)
    return mrs_opt(; wb = WeightBounds(; lb = -Inf, ub = Inf), sbgt = 1000.0, kwargs...)
end
mrs_floor(lb) = ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = lb))

@testset "MeanRisk sweep: each objective agrees with its closed form" begin
    mu, S = MRS_PR.mu, MRS_PR.sigma
    Si = inv(S)
    o = ones(length(mu))
    # Minimum variance: Sigma^-1 1 / (1' Sigma^-1 1).
    w = optimise(MeanRisk(; opt = mrs_free())).w
    @test isapprox(w, Si * o / (o' * Si * o); atol = 5e-5)
    # Utility mu'w - l w'Sigma w: Sigma^-1 (mu - gamma 1) / (2 l), with gamma set by the
    # budget. `l` multiplies the whole variance, with no factor of one half.
    l = 3.0
    w = optimise(MeanRisk(; opt = mrs_free(), obj = MaximumUtility(; l = l))).w
    g = (o' * Si * mu - 2l) / (o' * Si * o)
    @test isapprox(w, Si * (mu .- g) / (2l); atol = 5e-5)
    @test mu' * w - l * w' * S * w >=
          mu' * (Si * o / (o' * Si * o)) - l * (1 / (o' * Si * o))
    # Ratio (mu'w - rf) / sqrt(w'Sigma w): the tangency portfolio
    # Sigma^-1 (mu - rf 1) / (1' Sigma^-1 (mu - rf 1)).
    for rf in (1e-4, 1.2e-3)
        w = optimise(MeanRisk(; opt = mrs_free(), obj = MaximumRatio(; rf = rf))).w
        wt = Si * (mu .- rf) / (o' * Si * (mu .- rf))
        @test isapprox(w, wt; atol = 5e-5)
    end
    # Maximum return on a long-only book: all of the weight on the asset of the largest mean.
    w = optimise(MeanRisk(; opt = mrs_opt(), obj = MaximumReturn())).w
    @test isapprox(w, Float64.(eachindex(mu) .== argmax(mu)); atol = 1e-8)
end

@testset "MeanRisk sweep: a return frontier is the minimum variance at each floor" begin
    mu = MRS_PR.mu
    wmin = optimise(MeanRisk(; opt = mrs_opt())).w
    wmax = optimise(MeanRisk(; opt = mrs_opt(), obj = MaximumReturn())).w
    res = optimise(MeanRisk(; opt = mrs_opt(; ret = mrs_floor(Frontier(; N = 5)))))
    @test length(res.w) == 5
    @test all(c -> isa(c, PO.OptimisationSuccess), res.retcode)
    # The floors run evenly from the return of the minimum-risk corner to the maximum return.
    levels = range(mu' * wmin, mu' * wmax; length = 5)
    @test isapprox([mu' * w for w in res.w], levels; atol = 1e-7)
    @test isapprox(res.w[1], wmin; atol = 5e-5)
    @test isapprox(res.w[end], wmax; atol = 1e-8)
    # Each point equals an independent solve at the same scalar floor.
    for (w, e) in zip(res.w, levels)
        wi = optimise(MeanRisk(; opt = mrs_opt(; ret = mrs_floor(e)))).w
        @test isapprox(w, wi; atol = 5e-5)
    end
end

@testset "MeanRisk sweep: a risk frontier is the maximum return at each ceiling" begin
    mu, S = MRS_PR.mu, MRS_PR.sigma
    sd(w) = sqrt(w' * S * w)
    wmin = optimise(MeanRisk(; opt = mrs_opt())).w
    wmax = optimise(MeanRisk(; opt = mrs_opt(), obj = MaximumReturn())).w
    r = Variance(; settings = RiskMeasureSettings(; ub = Frontier(; N = 5)))
    res = optimise(MeanRisk(; opt = mrs_opt(), r = r, obj = MaximumReturn()))
    @test length(res.w) == 5
    # The default variance takes a SquareRootBound, so the ceilings are evenly spaced in
    # standard deviation, and not in variance.
    levels = range(sd(wmin), sd(wmax); length = 5)
    @test isapprox(sd.(res.w), levels; atol = 1e-8)
    @test !isapprox(sd.(res.w) .^ 2, range(sd(wmin)^2, sd(wmax)^2; length = 5); rtol = 1e-2)
    for (w, s) in zip(res.w, levels)
        # The caller writes a scalar bound in the units of the measure, the variance.
        rs = Variance(; settings = RiskMeasureSettings(; ub = s^2))
        wi = optimise(MeanRisk(; opt = mrs_opt(), r = rs, obj = MaximumReturn())).w
        @test isapprox(w, wi; atol = 5e-5)
        # The two definitions of page 82 give one curve: the minimum variance for the return
        # of this point is the variance of this point.
        wb = optimise(MeanRisk(; opt = mrs_opt(; ret = mrs_floor(mu' * w)))).w
        @test isapprox(sd(wb), sd(w); rtol = 1e-6)
    end
end

@testset "MeanRisk sweep: vector floors give the closed-form frontier" begin
    mu, S = MRS_PR.mu, MRS_PR.sigma
    Si = inv(S)
    o = ones(length(mu))
    A, B, C = o' * Si * o, o' * Si * mu, mu' * Si * mu
    D = A * C - B^2
    es = B / A .+ [5e-4, 1e-3, 2e-3]
    res = optimise(MeanRisk(; opt = mrs_free(; ret = mrs_floor(es))))
    # The minimum variance at mean e, above the mean of the minimum-variance portfolio, is
    # (A e^2 - 2 B e + C) / D, at the weights Sigma^-1 ((C - e B) 1 + (e A - B) mu) / D.
    for (w, e) in zip(res.w, es)
        @test isapprox(w' * S * w, (A * e^2 - 2B * e + C) / D; rtol = 1e-6)
        @test isapprox(w, Si * ((C - e * B) * o + (e * A - B) * mu) / D; atol = 5e-5)
    end
end

@testset "MeanRisk sweep: a risk span transforms, scales and sorts its two ends" begin
    X = MRS_X
    pr = MRS_PR
    rs = factory(Variance(), pr)
    w_lo = optimise(MeanRisk(; opt = mrs_opt())).w
    w_hi = optimise(MeanRisk(; opt = mrs_opt(), obj = MaximumReturn())).w
    v_lo, v_hi = expected_risk(rs, w_lo, X), expected_risk(rs, w_hi, X)
    @test v_lo < v_hi
    for (bound, g) in
        ((LinearBound(), identity), (SquareRootBound(), sqrt), (SquaredBound(), x -> x^2))
        front = PO._Frontier(; N = 4, factor = 2, bound = bound)
        registry = [(:a_ub_var, :a_ub) => (:EXPR, front, true, 1)]
        entry = PO._rebuild_risk_frontier(pr, nothing, rs, registry, w_lo, w_hi, 1)
        @test collect(entry.second[2]) ≈ range(2 * g(v_lo), 2 * g(v_hi); length = 4)
        # A corner pair in the other order gives the same ascending span. A measure of a
        # vector can have more risk at the minimum-risk corner, because that solve
        # minimises the scalarised risk of every measure.
        swapped = PO._rebuild_risk_frontier(pr, nothing, rs, registry, w_hi, w_lo, 1)
        @test collect(swapped.second[2]) == collect(entry.second[2])
        @test issorted(swapped.second[2])
    end
end

@testset "MeanRisk sweep: one measure owns every registry entry" begin
    front = Frontier(; N = 2)
    registry = [(:a_ub_var, :a_ub) => (:EXPR_ONE, [0.1, 0.2], true, 1),
                (:b_ub_var, :b_ub) => (:EXPR_TWO, front, true, 1),
                (:c_ub_var, :c_ub) => (:EXPR_THREE, front, true, 1)]
    r = Variance()
    owners = PO.frontier_owner_measures(r, registry, [2, 3])
    @test length(owners) == 2
    @test all(x -> x === r, owners)
    rs = [Variance(), ConditionalValueatRisk(), StandardDeviation()]
    registry = [(:b_ub_var, :b_ub) => (:EXPR_TWO, front, true, 2),
                (:c_ub_var, :c_ub) => (:EXPR_THREE, front, true, 3)]
    @test collect(PO.frontier_owner_measures(rs, registry, [1, 2])) == rs[2:3]
    @test collect(PO.frontier_owner_measures(rs, registry, [2])) == rs[3:3]
end

@testset "MeanRisk sweep: one measure with two bounds keeps both" begin
    pr = prior(HighOrderPriorEstimator(), MRS_X)
    opt = JuMPOptimiser(; pe = pr, slv = MRS_SLV)
    kt = Kurtosis(; settings = RiskMeasureSettings(; ub = Frontier(; N = 2)))
    # A resolved vector at position 1 and a Frontier at position 2: the rebuild of position 1
    # read the vector as a Frontier and threw a FieldError.
    r = VarianceSkewKurtosis(;
                             vr = Variance(;
                                           settings = RiskMeasureSettings(;
                                                                          ub = [1e-4, 2e-4])),
                             kt = kt)
    res = optimise(MeanRisk(; opt = opt, r = r, obj = MaximumReturn()))
    @test length(res.w) == 4
    # Two Frontier bounds of two points each: the sweep is their product. The rebuild kept
    # only the first entry, so the sweep had two points.
    r = VarianceSkewKurtosis(;
                             vr = Variance(;
                                           settings = RiskMeasureSettings(;
                                                                          ub = Frontier(;
                                                                                        N = 2))),
                             kt = kt)
    res = optimise(MeanRisk(; opt = opt, r = r, obj = MaximumReturn()))
    @test length(res.w) == 4
end

@testset "MeanRisk sweep: a Frontier needs two points" begin
    # `range(a, b; length = 1)` throws when `a != b`, so every sweep of one point failed
    # at the solve with an error that named neither the frontier nor `N`.
    @test_throws DomainError Frontier(; N = 1)
    @test_throws DomainError Frontier(; N = 0)
    @test Frontier(; N = 2).N == 2
end
