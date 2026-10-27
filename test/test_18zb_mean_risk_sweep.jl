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
        registry = [(:a_ub_var, :a_ub) => (:EXPR, front, true, rs)]
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

# The variance and the fourth moment of `VarianceSkewKurtosis` at `w`, read off the matrices
# that the model reads. These are the units of the `vr` and `kt` expressions.
function mrs_vsk_terms(pr, w)
    v, _, k = PO.variance_skew_kurtosis_terms(factory(VarianceSkewKurtosis(), pr), w)
    return v, k
end
function mrs_vsk_vr(ub)
    return VarianceSkewKurtosis(;
                                vr = Variance(; settings = RiskMeasureSettings(; ub = ub)))
end
function mrs_vsk_kt(ub; alg2 = SOCRiskExpr())
    return VarianceSkewKurtosis(;
                                kt = Kurtosis(; settings = RiskMeasureSettings(; ub = ub),
                                              alg = alg2))
end

@testset "MeanRisk sweep: a sweep on a term of VarianceSkewKurtosis spans that term" begin
    pr = prior(HighOrderPriorEstimator(), MRS_X)
    opt = JuMPOptimiser(; pe = pr, slv = MRS_SLV)
    wmin = optimise(MeanRisk(; opt = opt, r = VarianceSkewKurtosis())).w
    wmax = optimise(MeanRisk(; opt = opt, r = VarianceSkewKurtosis(),
                             obj = MaximumReturn())).w
    (v_lo, k_lo), (v_hi, k_hi) = mrs_vsk_terms(pr, wmin), mrs_vsk_terms(pr, wmax)
    tol = 1e-4
    # The variance term takes a LinearBound and the kurtosis term a SquaredBound, so each span
    # runs in the units of its expression. A span from the risk of the whole measure put the
    # first kurtosis level below the kurtosis of every portfolio, so that solve failed, and it
    # stopped the sweep short of the maximum-return corner. Under a quadratic formulation the
    # kurtosis term takes a LinearBound, because its functor gives the fourth moment itself.
    for (r, j, lo, hi, owner) in ((mrs_vsk_vr(Frontier(; N = 5)), 1, v_lo, v_hi, Variance),
                                  (mrs_vsk_kt(Frontier(; N = 5)), 2, k_lo, k_hi, Kurtosis),
                                  (mrs_vsk_kt(Frontier(; N = 5); alg2 = RSOCRiskExpr()), 2, k_lo, k_hi, Kurtosis))
        for rr in (r, [r])
            res = optimise(MeanRisk(; opt = opt, r = rr, obj = MaximumReturn()))
            @test all(c -> isa(c, PO.OptimisationSuccess), res.retcode)
            @test only(res.model[:risk_frontier]).second[4] isa owner
            got = [mrs_vsk_terms(pr, w)[j] for w in res.w]
            @test isapprox(got, range(lo, hi; length = 5); rtol = tol)
        end
    end
    # Two terms swept together form the product of their levels. The variance entry
    # registers first, so its level changes fastest.
    r = VarianceSkewKurtosis(;
                             vr = Variance(;
                                           settings = RiskMeasureSettings(;
                                                                          ub = Frontier(;
                                                                                        N = 3))),
                             kt = Kurtosis(;
                                           settings = RiskMeasureSettings(;
                                                                          ub = Frontier(;
                                                                                        N = 3))))
    res = optimise(MeanRisk(; opt = opt, r = r, obj = MaximumReturn()))
    @test length(res.w) == 9
    @test all(c -> isa(c, PO.OptimisationSuccess), res.retcode)
    lv, lk = range(v_lo, v_hi; length = 3), range(k_lo, k_hi; length = 3)
    # The tolerance is a share of the largest level of each span. At the first point both
    # levels are the terms of the minimum-risk corner, so that corner is the only feasible
    # portfolio, and the solve meets the ceilings to its absolute tolerance, which is a large
    # share of a kurtosis level near 1e-9.
    av, ak = tol * lv[end], tol * lk[end]
    for (j, w) in enumerate(res.w)
        a, b = mod1(j, 3), cld(j, 3)
        v, k = mrs_vsk_terms(pr, w)
        @test v <= lv[a] + av && k <= lk[b] + ak
        # The maximum-return objective moves the weights until one of the two ceilings holds.
        @test isapprox(v, lv[a]; atol = av) || isapprox(k, lk[b]; atol = ak)
    end
    @test isapprox(res.w[end], wmax; atol = 1e-6)
end

@testset "NearOptimalCentering sweep: a sweep on a term of VarianceSkewKurtosis spans that term" begin
    pr = prior(HighOrderPriorEstimator(), MRS_X)
    opt = JuMPOptimiser(; pe = pr, slv = MRS_SLV)
    wmin = optimise(MeanRisk(; opt = opt, r = VarianceSkewKurtosis())).w
    wmax = optimise(MeanRisk(; opt = opt, r = VarianceSkewKurtosis(),
                             obj = MaximumReturn())).w
    k_lo, k_hi = mrs_vsk_terms(pr, wmin)[2], mrs_vsk_terms(pr, wmax)[2]
    function noc(ub)
        return NearOptimalCentering(; r = mrs_vsk_kt(ub), obj = MaximumReturn(), opt = opt,
                                    alg = ConstrainedNearOptimalCentering())
    end
    nf = noc(Frontier(; N = 3))
    res = optimise(nf)
    # The span from the whole measure asked for less kurtosis than any portfolio has, so the
    # first point failed.
    @test all(c -> isa(c, PO.OptimisationSuccess), res.retcode)
    entry = only(res.model[:risk_frontier])
    @test entry.second[4] isa Kurtosis
    # The levels are the span of the term between the corners, which NOC solves as MeanRisk
    # does. A near-optimal centre lies inside its region and not on its bound, so the test
    # checks the levels and not the kurtosis of each portfolio.
    span = PO.rebuild_risk_frontier(nf, pr, nothing, [entry], wmin, wmax, [1])
    @test collect(only(span).second[2]) ≈ range(k_lo, k_hi; length = 3) rtol = 1e-10
end

@testset "MeanRisk sweep: a sweep on the skewness floor of VarianceSkewKurtosis spans m3" begin
    pr = prior(HighOrderPriorEstimator(), MRS_X)
    opt = JuMPOptimiser(; pe = pr, slv = MRS_SLV)
    wmin = optimise(MeanRisk(; opt = opt, r = VarianceSkewKurtosis())).w
    wmax = optimise(MeanRisk(; opt = opt, r = VarianceSkewKurtosis(),
                             obj = MaximumReturn())).w
    # The floor bounds the third moment that the model reads off the coskewness matrix. The
    # span took the risk of the whole measure, and the functor of `Skewness` gives the
    # standardised skewness, so neither is in the units of the floor.
    m3(w) = PO.portfolio_third_moment(pr.sk, w)
    r = VarianceSkewKurtosis(;
                             sk = Skewness(;
                                           settings = MaxRiskMeasureSettings(;
                                                                             lb = Frontier(;
                                                                                           N = 5))))
    res = optimise(MeanRisk(; opt = opt, r = r, obj = MaximumReturn()))
    @test all(c -> isa(c, PO.OptimisationSuccess), res.retcode)
    entry = only(res.model[:risk_frontier])
    @test entry.second[4] isa Skewness
    owner = factory(entry.second[4], pr, MRS_SLV)
    span = PO._rebuild_risk_frontier(pr, nothing, owner, [entry], wmin, wmax, 1)
    @test collect(span.second[2]) ≈ range(m3(wmax), m3(wmin); length = 5) rtol = 1e-12
    # NearOptimalCentering takes the same span.
    noc = NearOptimalCentering(; r = r, obj = MaximumReturn(), opt = opt,
                               alg = ConstrainedNearOptimalCentering())
    span = PO.rebuild_risk_frontier(noc, pr, nothing, [entry], wmin, wmax, [1])
    @test collect(only(span).second[2]) ≈ range(m3(wmax), m3(wmin); length = 5) rtol = 1e-12
end

@testset "MeanRisk sweep: a span reads the comoment matrix that the model reads" begin
    pr = prior(HighOrderPriorEstimator(), MRS_X)
    w = collect(range(0.1, 0.3; length = 5))
    w ./= sum(w)
    m3, m4 = PO.portfolio_third_moment(pr.sk, w), PO.portfolio_fourth_moment(pr.kt, w)
    @test m3 ≈ dot(w, pr.sk, kron(w, w))
    @test m4 ≈ dot(kron(w, w), pr.kt, kron(w, w))
    # The span of a kurtosis owner is the value of its functor, in the units of its
    # formulation. The functor reads the matrix of the model, so a stated matrix moves the
    # span with it (#1846).
    for (alg2, g) in ((SOCRiskExpr(), sqrt), (RSOCRiskExpr(), identity))
        kt = factory(Kurtosis(; alg = alg2), pr)
        @test expected_risk(kt, w, MRS_X) ≈ g(m4)
        kt = factory(Kurtosis(; alg = alg2, kt = 2 .* pr.kt), pr)
        @test expected_risk(kt, w, MRS_X) ≈ g(2 * m4)
    end
    # The functor of Skewness is the moment that the floor bounds.
    sk = factory(Skewness(; sk = 2 .* pr.sk), pr)
    @test expected_risk(sk, w, MRS_X) ≈ 2 * m3
end

@testset "MeanRisk sweep: a sweep on the whole of VarianceSkewKurtosis meets each level" begin
    pr = prior(HighOrderPriorEstimator(), MRS_X)
    opt = JuMPOptimiser(; pe = pr, slv = MRS_SLV)
    wmin = optimise(MeanRisk(; opt = opt, r = VarianceSkewKurtosis())).w
    wmax = optimise(MeanRisk(; opt = opt, r = VarianceSkewKurtosis(),
                             obj = MaximumReturn())).w
    # At a unit skewness scale the relaxation is tight between the corners, so the bound on
    # the weighted sum holds the functor, which reads the matrices of the model.
    r = VarianceSkewKurtosis(; settings = RiskMeasureSettings(; ub = Frontier(; N = 5)))
    res = optimise(MeanRisk(; opt = opt, r = r, obj = MaximumReturn()))
    @test all(c -> isa(c, PO.OptimisationSuccess), res.retcode)
    fr = factory(r, pr)
    levels = range(fr(wmin, MRS_X), fr(wmax, MRS_X); length = 5)
    @test isapprox([fr(w, MRS_X) for w in res.w], levels; rtol = 1e-6)
end

@testset "Frontier sweep: a sweep of one point gives the portfolio of its scalar bound" begin
    pr = prior(HighOrderPriorEstimator(), MRS_X)
    opt = JuMPOptimiser(; pe = pr, slv = MRS_SLV)
    wmin = optimise(MeanRisk(; opt = opt, r = VarianceSkewKurtosis())).w
    wmax = optimise(MeanRisk(; opt = opt, r = VarianceSkewKurtosis(),
                             obj = MaximumReturn())).w
    (v_lo, k_lo), (v_hi, k_hi) = mrs_vsk_terms(pr, wmin), mrs_vsk_terms(pr, wmax)
    noc(r) = NearOptimalCentering(; r = r, obj = MaximumReturn(), opt = opt,
                                  alg = ConstrainedNearOptimalCentering())
    mr(r) = MeanRisk(; r = r, obj = MaximumReturn(), opt = opt)
    # A sweep writes its bound into a parameter, which the solver receives as a variable
    # fixed by an equality row. In the units of its point, a bound near 1e-7 broke the
    # kurtosis ceiling of a constrained NOC by 0.22 %, and the weights moved by 7e-4 from
    # those of the scalar ceiling. The anchors of the barrier, raw parameters too, moved the
    # centre by 3e-3. The kurtosis term takes a SquaredBound, so its bound is the root of the
    # level.
    for (r, j, level, ub) in ((mrs_vsk_kt, 2, (k_lo + k_hi) / 2, sqrt((k_lo + k_hi) / 2)),
                              (mrs_vsk_vr, 1, (v_lo + v_hi) / 2, (v_lo + v_hi) / 2)),
        head in (noc, mr)

        a = optimise(head(r(ub)))
        b = optimise(head(r([ub])))
        @test isa(a.retcode, PO.OptimisationSuccess)
        @test isa(only(b.retcode), PO.OptimisationSuccess)
        @test mrs_vsk_terms(pr, only(b.w))[j] <= level * (1 + 1e-5)
        @test isapprox(only(b.w), a.w; atol = 1e-5)
    end
    # A return floor of a constrained NOC. The floor holds, and the centre is the centre of the
    # scalar floor.
    rl = (dot(pr.mu, wmin) + dot(pr.mu, wmax)) / 2
    function noc_ret(lb)
        return NearOptimalCentering(; r = StandardDeviation(), obj = MinimumRisk(),
                                    opt = JuMPOptimiser(; pe = pr, slv = MRS_SLV,
                                                        ret = mrs_floor(lb)),
                                    alg = ConstrainedNearOptimalCentering())
    end
    a = optimise(noc_ret(rl))
    b = optimise(noc_ret([rl]))
    @test isa(only(b.retcode), PO.OptimisationSuccess)
    @test dot(pr.mu, only(b.w)) >= rl * (1 - 1e-8)
    @test isapprox(only(b.w), a.w; atol = 1e-5)
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
