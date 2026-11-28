using Clarabel, JuMP
using Distributions: Normal
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
# A cost on the first asset, `c * w[1]`, and a term that records the objective it gets.
struct MrsFirstCost <: PO.CustomJuMPObjective
    c::Float64
end
function PO.add_custom_objective_term!(model::JuMP.Model, ::Any, t::MrsFirstCost, ::Any,
                                       ::Any)
    w = PO.get_w(model)
    PO.add_to_objective_penalty!(model, JuMP.@expression(model, t.c * w[1]))
    return nothing
end
struct MrsObjectiveSpy <: PO.CustomJuMPObjective
    log::Vector{PO.ObjectiveFunction}
end
function PO.add_custom_objective_term!(::JuMP.Model, obj, t::MrsObjectiveSpy, ::Any, ::Any)
    push!(t.log, obj)
    return nothing
end

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

@testset "MeanRisk sweep: a custom objective term counts once in each solve (#2065)" begin
    # The corners and the sweep set the objective of one model. The Objective Penalty keeps
    # every term that it gets, so a term added for each objective counted three times at the
    # sweep. The penalty `l1` went in once, in the assembly, and the term goes in there too.
    rf = Variance(; settings = RiskMeasureSettings(; ub = Frontier(; N = 3)))
    res = optimise(MeanRisk(; r = rf, obj = MaximumReturn(),
                            opt = mrs_opt(; cobj = MrsFirstCost(1.0), l1 = 0.5)))
    m = res.model
    @test JuMP.coefficient(m[:op], PO.get_w(m)[1]) == 1.0
    @test JuMP.coefficient(m[:op], m[:t_l1]) == 0.5
    # The hook runs once, with the declared objective, and not with the objective of a corner.
    log = PO.ObjectiveFunction[]
    optimise(MeanRisk(; r = rf, obj = MaximumReturn(),
                      opt = mrs_opt(; cobj = MrsObjectiveSpy(log))))
    @test length(log) == 1 && only(log) isa MaximumReturn
    # A sweep on both axes calls the hook once too.
    empty!(log)
    optimise(MeanRisk(;
                      r = Variance(;
                                   settings = RiskMeasureSettings(; ub = Frontier(; N = 2))),
                      opt = mrs_opt(; ret = mrs_floor(Frontier(; N = 2)),
                                    cobj = MrsObjectiveSpy(log))))
    @test length(log) == 1 && only(log) isa MinimumRisk
    # Each corner and each point of a return sweep counts the term once. So the floors run
    # from the return of the minimum-risk solve with the term to the return of the
    # maximum-return solve with the term, and each point equals a plain solve with the term at
    # its floor. The minimum-variance book holds none of the first asset, so a cost on it moves
    # nothing. A reward of the order of the variance moves the weights, so a term that counts
    # three times gives other weights.
    mu = MRS_PR.mu
    t = MrsFirstCost(-1e-4)
    res = optimise(MeanRisk(;
                            opt = mrs_opt(; ret = mrs_floor(Frontier(; N = 3)), cobj = t)))
    @test all(c -> isa(c, PO.OptimisationSuccess), res.retcode)
    wmin = optimise(MeanRisk(; opt = mrs_opt(; cobj = t))).w
    wmax = optimise(MeanRisk(; opt = mrs_opt(; cobj = t), obj = MaximumReturn())).w
    levels = range(mu' * wmin, mu' * wmax; length = 3)
    @test isapprox([mu' * w for w in res.w], levels; atol = 1e-7)
    for (w, e) in zip(res.w, levels)
        wi = optimise(MeanRisk(; opt = mrs_opt(; ret = mrs_floor(e), cobj = t))).w
        @test isapprox(w, wi; atol = 5e-5)
    end
    e = levels[2]
    w1 = optimise(MeanRisk(; opt = mrs_opt(; ret = mrs_floor(e), cobj = t))).w
    w3 = optimise(MeanRisk(;
                           opt = mrs_opt(; ret = mrs_floor(e), cobj = MrsFirstCost(-3e-4)))).w
    w0 = optimise(MeanRisk(; opt = mrs_opt(; ret = mrs_floor(e)))).w
    @test !isapprox(w1, w3; atol = 1e-3) && !isapprox(w1, w0; atol = 1e-3)
end

@testset "MeanRisk sweep: a Frontier under MaximumRatio spans the corners of the plain weights" begin
    # #1848: the corners were solved on the homogenised model. The minimum-risk solve gave
    # the tangency portfolio, and the maximum-return solve was unbounded in k for rf > 0 and
    # any feasible point for rf = 0. The span must run between the two corners of the plain
    # weights. The rows of the sweep need a parametric solver (#1842), so this test reads the
    # resolved span from the model, which the sweep writes before its first point.
    mu, S = MRS_PR.mu, MRS_PR.sigma
    sd(w) = sqrt(w' * S * w)
    wmin = optimise(MeanRisk(; opt = mrs_opt())).w
    wmax = optimise(MeanRisk(; opt = mrs_opt(), obj = MaximumReturn())).w
    risk_levels = range(sd(wmin), sd(wmax); length = 5)
    ret_levels = range(mu' * wmin, mu' * wmax; length = 5)
    r = Variance(; settings = RiskMeasureSettings(; ub = Frontier(; N = 5)))
    # rf = 0, a rate below the means (the return form), and a rate above every mean (the
    # risk form).
    for rf in (0.0, 1e-4, 2 * maximum(mu))
        res = optimise(MeanRisk(; opt = mrs_opt(), r = r, obj = MaximumRatio(; rf = rf)))
        @test length(res.w) == 5
        @test isapprox(collect(only(res.model[:risk_frontier]).second[2]), risk_levels;
                       rtol = 1e-6)
        res = optimise(MeanRisk(; opt = mrs_opt(; ret = mrs_floor(Frontier(; N = 5))),
                                obj = MaximumRatio(; rf = rf)))
        @test length(res.w) == 5
        @test isapprox(collect(only(res.model[:ret_frontier]).second[2]), ret_levels;
                       rtol = 1e-6)
    end
end

@testset "MeanRisk sweep: the corners of a MaximumRatio Frontier hold no phylogeny penalty" begin
    # The corner model of a ratio takes MinimumRisk. With a semidefinite phylogeny on the
    # weights of the variance, MinimumRisk omits the penalty p tr(W), and MaximumReturn puts
    # it into both corner solves. On this data the penalty moved the maximum-return corner
    # to a return of 0.00195, below the return of the minimum-risk corner, 0.00209. A
    # long-only maximum return is one asset, so it meets every phylogeny row.
    mu = MRS_PR.mu
    ple = SemiDefinitePhylogenyEstimator(; pl = NetworkEstimator())
    wmin = optimise(MeanRisk(; opt = mrs_opt(; ple = ple))).w
    res = optimise(MeanRisk(;
                            opt = mrs_opt(; ple = ple, ret = mrs_floor(Frontier(; N = 3))),
                            obj = MaximumRatio(; rf = 1e-4)))
    @test isapprox(collect(only(res.model[:ret_frontier]).second[2]),
                   range(mu' * wmin, maximum(mu); length = 3); rtol = 1e-6)
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
        front = Frontier(4, 2, bound)
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
    # A second moment rebuilds the `Frontier` of the caller with the positional constructor
    # (#2071): it keeps `N`, takes the inverse of its factor, and names the bound of its
    # formulation, the square root unless the expression is a second-order cone.
    front = Frontier(; N = 5)
    soc = PO.second_moment_bound_val(SOCRiskExpr(), front, 4)
    sq = PO.second_moment_bound_val(QuadRiskExpr(), front, 4)
    @test (soc.N, soc.factor, soc.bound) == (5, 0.25, LinearBound())
    @test (sq.N, sq.factor, sq.bound) == (5, 0.25, SquareRootBound())
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

@testset "A scalar ceiling of the size of a fourth moment holds under a constrained NOC" begin
    # Each bound row divides its expression by the size of the bound. The undivided Kurtosis
    # ceiling has coefficients near the cokurtosis, which the equilibration of Clarabel scales
    # up only to its cap. At the default settings the centring then stopped at ALMOST_SOLVED
    # and broke the ceiling by 1.45 %, 0.63 % and 0.085 % at the three levels below.
    pr = prior(HighOrderPriorEstimator(), MRS_X)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false))
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    rk = factory(Kurtosis(), pr)
    lo = expected_risk(rk, optimise(MeanRisk(; opt = opt, r = Kurtosis())).w, pr)
    hi = expected_risk(rk, optimise(MeanRisk(; opt = opt, obj = MaximumReturn())).w, pr)
    for f in (0.25, 0.5, 0.75)
        ub = lo + f * (hi - lo)
        res = optimise(NearOptimalCentering(;
                                            r = Kurtosis(;
                                                         settings = RiskMeasureSettings(;
                                                                                        ub = ub)),
                                            obj = MaximumReturn(), opt = opt,
                                            alg = ConstrainedNearOptimalCentering()))
        @test isa(res.retcode, PO.OptimisationSuccess)
        @test expected_risk(rk, res.w, pr) <= ub * (1 + 1e-5)
    end
end

@testset "An EntropicValueatRisk ceiling holds under a constrained NOC (#1905)" begin
    # The anchor cones of the barrier divide their slack by the width of the near optimal
    # region. In the units of the risk, the 50 % level failed, and the 75 % level stopped at
    # ALMOST_SOLVED, which allow_almost accepts, and broke the ceiling by 1.3 %. On the CI
    # hosts the first solver can stop at INSUFFICIENT_PROGRESS, so fallback solvers with a
    # shorter step follow it, as in test18_setup.jl. The last two take a shorter step still,
    # and tighter tolerances.
    pr = prior(HighOrderPriorEstimator(), MRS_X)
    tight = Dict("tol_gap_abs" => 1e-10, "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10)
    slv = [Solver(; name = Symbol(:clarabel, i), solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = merge(Dict("verbose" => false, "max_step_fraction" => msf),
                                   msf < 0.8 ? tight : empty(tight)))
           for (i, msf) in enumerate((0.99, 0.95, 0.9, 0.85, 0.75, 0.65))]
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    ev(; kwargs...) = EntropicValueatRisk(; slv = slv, kwargs...)
    rk = factory(ev(), pr)
    lo = expected_risk(rk, optimise(MeanRisk(; opt = opt, r = ev())).w, pr)
    hi = expected_risk(rk, optimise(MeanRisk(; opt = opt, obj = MaximumReturn())).w, pr)
    for f in (0.25, 0.5, 0.75)
        ub = lo + f * (hi - lo)
        res = optimise(NearOptimalCentering(;
                                            r = ev(;
                                                   settings = RiskMeasureSettings(;
                                                                                  ub = ub)),
                                            obj = MaximumReturn(), opt = opt,
                                            alg = ConstrainedNearOptimalCentering()))
        @test isa(res.retcode, PO.OptimisationSuccess)
        # A solve that stops at ALMOST_OPTIMAL holds the ceiling to 1.5 %. On one CI host kind
        # the 75 % level overshoots the ceiling by 1.116 %, over the old bound of 1 % (runs
        # 43866733484, 43876066388, 43877015997, 43891346201 and 43899887696, Issue #2118).
        # An earlier run overshot it by 0.44 % (runs 41047211489 and 41050705423, G1 of
        # #1817).
        almost = PO.JuMP.termination_status(res.model) == PO.JuMP.MOI.ALMOST_OPTIMAL
        # The form prints the relative overshoot when it fails, and its bound shows the status.
        @test expected_risk(rk, res.w, pr) / ub - 1 <= (almost ? 1.5e-2 : 1e-5)
    end
end

@testset "An anchor cone keeps its slack in the units of its bound when the region is degenerate" begin
    # The width of the region divides the slack of the cone.
    @test PO.anchor_cone_scale(5e-4, 0.03) == 5e-4
    @test PO.anchor_cone_scale([2e-4, -5e-4], [0.01, 0.03]) == 5e-4
    # A width of the order of the rounding error, as when the three anchors are one
    # portfolio, gives way to one. A division by the bound made the rounding error of the
    # slack 137 times larger, and the constrained NOC of test_02:1216 stopped at
    # NUMERICAL_ERROR under the flags of CI (R21 of #1817).
    @test PO.anchor_cone_scale(-1e-13, 0.0073) === 1.0
    @test PO.anchor_cone_scale(0.0, [0.01, -0.02]) === 1.0
    # The comparison has no float tolerance of its own, so a Rational keeps its type.
    @test PO.anchor_cone_scale(1 // 10, 1 // 2) === 1 // 10
    @test PO.anchor_cone_scale(0 // 1, 1 // 2) === 1 // 1
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

# A range of a user's own, whose tails keep the bounds that the caller gives.
struct MrsRange{T1, T2, T3} <: PO.RangeRiskMeasure
    settings::T1
    loss::T2
    gain::T3
end
PO.risk_input_kind(::MrsRange) = PO.NetReturnsInput()
PO.range_tails(r::MrsRange) = PO.RangeTails(r.loss, r.gain)
PO.range_risk_key(::MrsRange) = :mrs_range_risk_

@testset "MeanRisk sweep: a Frontier on a tail of a range spans that tail" begin
    # The sweep took the span of the whole range, so the levels were too high and most points
    # were the portfolio of maximum return (#2319).
    front = RiskMeasureSettings(; rke = false, ub = Frontier(; N = 4))
    off = RiskMeasureSettings(; rke = false)
    cvar(s) = ConditionalValueatRisk(; settings = s)
    # A law with a mean far from zero, so that the two tails read different z-scores.
    vdist(s) = ValueatRisk(; settings = s,
                           alg = DistributionValueatRisk(; dist = Normal(0.1, 2.0)))
    gvar(l, g) = GenericValueatRiskRange(; loss = l, gain = g)
    user(l, g) = MrsRange(RiskMeasureSettings(), l, g)
    loss_value(r) = w -> expected_risk(r, w, MRS_PR)
    gain_value(r) = w -> expected_risk(PO.GainTail(r), w, MRS_PR)
    # Two returns tails, two tails that share a cone, a tail that reads the weights beside one
    # that reads the returns, and a range of a user.
    cases = ((gvar, cvar, cvar, :loss), (gvar, cvar, cvar, :gain),
             (gvar, vdist, vdist, :loss), (gvar, vdist, vdist, :gain),
             (gvar, cvar, vdist, :gain), (user, cvar, cvar, :gain))
    @testset "case $i" for (i, (mk, l, g, side)) in enumerate(cases)
        free = mk(l(off), g(off))
        rng, value = if side === :loss
            mk(l(front), g(off)), loss_value(l(off))
        else
            mk(l(off), g(front)), gain_value(g(off))
        end
        wmin = optimise(MeanRisk(; r = free, opt = mrs_opt())).w
        wmax = optimise(MeanRisk(; r = free, opt = mrs_opt(), obj = MaximumReturn())).w
        levels = range(minmax(value(wmin), value(wmax))...; length = 4)
        # The levels of the whole range are far above the levels of the tail.
        @test first(levels) < last(levels)
        @test last(levels) < 0.9 * (loss_value(l(off))(wmax) + gain_value(g(off))(wmax))
        res = optimise(MeanRisk(; r = rng, opt = mrs_opt(), obj = MaximumReturn()))
        @test all(c -> isa(c, PO.OptimisationSuccess), res.retcode)
        entry = only(res.model[:risk_frontier])
        @test isa(entry.second[4], PO.GainTail) == (side === :gain)
        # The maximum return meets each ceiling, so the tail of each point is its level.
        @test isapprox([value(w) for w in res.w], levels; rtol = 1e-4)
        # The corner solves of `NearOptimalCentering` clear the bounds of the tails with
        # `no_bounds_risk_measure`, which the range of a user does not define.
        if mk === user
            continue
        end
        noc = NearOptimalCentering(; r = rng, opt = mrs_opt(), obj = MaximumReturn(),
                                   alg = ConstrainedNearOptimalCentering())
        res = optimise(noc)
        # The centring model rebuilds the entry by the same owner, so it has the same levels.
        registry = res.model[:risk_frontier]
        @test isa(only(registry).second[4], PO.GainTail) == (side === :gain)
        rebuilt = PO.rebuild_risk_frontier(noc, MRS_PR, nothing, registry, wmin, wmax, [1])
        @test isapprox(only(rebuilt).second[2], levels; rtol = 1e-12)
        # The centre is inside the region, so each point that solves lies at its level or
        # below it. Clarabel stops at SLOW_PROGRESS on some points of the centring model, also
        # for a frontier on a lone conditional value at risk.
        @test length(res.w) == 4
        @test all(((w, e),) -> !all(isfinite, w) || value(w) <= e * (1 + 1e-4),
                  zip(res.w, levels))
    end
    # A sweep on the range keeps the span of the range beside a sweep on each tail.
    r = GenericValueatRiskRange(; settings = RiskMeasureSettings(; ub = Frontier(; N = 2)),
                                loss = cvar(front), gain = cvar(front))
    res = optimise(MeanRisk(; r = r, opt = mrs_opt(), obj = MaximumReturn()))
    owners = [e.second[4] for e in res.model[:risk_frontier]]
    @test isa(owners[1], ConditionalValueatRisk)
    @test isa(owners[2], PO.GainTail)
    @test isa(owners[3], GenericValueatRiskRange)
end

@testset "MeanRisk sweep: a Frontier needs two points" begin
    # `range(a, b; length = 1)` throws when `a != b`, so every sweep of one point failed
    # at the solve with an error that named neither the frontier nor `N`.
    @test_throws DomainError Frontier(; N = 1)
    @test_throws DomainError Frontier(; N = 0)
    @test Frontier(; N = 2).N == 2
end

@testset "MeanRisk sweep: the fallback runs only when no point solved" begin
    ok, bad = OptimisationSuccess(), OptimisationFailure(; res = "x")
    @test PO.any_solved(ok) && !PO.any_solved(bad)
    @test PO.any_solved([bad, ok]) && !PO.any_solved([bad, bad])
    @test !PO.any_solved(OptimisationFailure[])
    @test PO.member_retcode(ok, 3) === ok && PO.member_retcode([bad, ok], 2) === ok
    rd = ReturnsResult(; nx = string.(1:5), X = MRS_X, ts = Date(2020) .+ Day.(1:300),
                       iv = abs.(MRS_X) .+ 0.1, ivpa = 1.2)
    top = maximum(MRS_PR.mu)
    function swept(lb, fb = EqualWeighted())
        return MeanRisk(; opt = JuMPOptimiser(; slv = MRS_SLV, ret = mrs_floor(lb)),
                        fb = fb)
    end
    # Every point solved. The chain once ran here, and the fallback replaced the sweep.
    for lb in (Frontier(; N = 3), [0.0, 0.5 * top])
        res = optimise(swept(lb), rd)
        @test isa(res, MeanRiskResult) && isnothing(res.fb)
        @test res.w == optimise(swept(lb, nothing), rd).w
    end
    # An infeasible end keeps the other points, and its own code states the failure.
    res = optimise(swept([0.0, 0.5 * top, 1.5 * top]), rd)
    @test isa(res, MeanRiskResult) && isnothing(res.fb)
    @test isa.(res.retcode, OptimisationSuccess) == [true, true, false]
    @test all(isnan, res.w[3])
    # No point solved.
    res = optimise(swept([1.5 * top, 2 * top]), rd)
    @test isa(res, NaiveOptimisationResult) && length(res.fb) == 1
    @test !PO.any_solved(res.fb[1][2].retcode)
    # The chain stops at the first fallback with a solved point.
    res = optimise(swept([1.5 * top], swept([2 * top, 0.0])), rd)
    @test isa(res, MeanRiskResult) && length(res.fb) == 1
    @test isa.(res.retcode, OptimisationSuccess) == [false, true]

    # A fold whose every point failed holds the one portfolio of the fallback. Its series
    # stands at every member of the path. A fall of the returns in the rows of one training
    # window alone puts that fold first, and then last.
    train = [1:150, 51:200, 101:250]
    for (fell, k) in ((1:50, 1), (201:250, 3)), wd in (nothing, SelfFinancingDrift())
        X = copy(MRS_X)
        X[fell, :] .-= 0.02
        tops = [maximum(mean(X[t, :]; dims = 1)) for t in train]
        level = (minimum(tops) + maximum(tops)) / 2
        opt = swept([level, 1.0001 * level])
        p = cross_val_predict(opt,
                              ReturnsResult(; nx = rd.nx, X = X, ts = rd.ts, iv = rd.iv,
                                            ivpa = 1.2), IndexWalkForward(150, 50; wd = wd))
        reached = [f != k for f in 1:3]
        @test [isa(f.res, MeanRiskResult) for f in p.pred] == reached
        @test length(p.mrd.X) == 2 && all(x -> length(x) == 150, p.mrd.X)
        for (f, r) in enumerate(reached), j in 1:2
            rows = (50 * (f - 1) + 1):(50 * f)
            x = p.pred[f].rd.X
            @test p.mrd.X[j][rows] == (r ? x[j] : x)
            iv = p.pred[f].rd.iv
            @test p.mrd.iv[j][rows] == (r ? iv[j] : iv)
        end
        @test length(p.mrd.ivpa) == 2 && all(==(1.2), p.mrd.ivpa)
    end
    # Folds whose series have two types still make one series.
    prd(x) = PO.PredictionReturnsResult(; X = x)
    @test PO.mapreduce_RetMtx([prd([0.1, 0.2]), prd(view([0.3, 0.4], 1:1))]) ==
          [0.1, 0.2, 0.3]
    # A path of sweeps alone keeps one series, and one implied volatility, for each member.
    p = cross_val_predict(swept([0.0, 1e-6], nothing), rd, IndexWalkForward(150, 50))
    @test p.mrd.X[2] == reduce(vcat, f.rd.X[2] for f in p.pred)
    @test p.mrd.iv[2] == reduce(vcat, f.rd.iv[2] for f in p.pred)

    # One failed vector cannot hold the members of the fold before, so it keeps its `NaN`.
    wn = fill(NaN, 5)
    @test PO.held_start_weights(bad, wn, [fill(0.2, 5), fill(0.2, 5)]) === wn

    # A solved sweep is a solved member, and a path of sweeps has no rank.
    cv = CombinatorialCrossValidation(; n_folds = 4, n_test_folds = 2)
    pop = cross_val_predict(swept([0.0, 1e-6], nothing), rd, cv)
    @test all(PO.member_solved, pop.pred)
    f = pop.pred[1].pred[1]
    @test !PO.member_solved(PredictionResult(; res = PO.set_retcode(f.res, [ok, bad]),
                                             rd = f.rd))
    r = ConditionalValueatRisk()
    @test_throws ArgumentError PO.sort_by_measure(SumScalariser(), pop, r)
    @test_throws ArgumentError PO.quantile_by_measure(SumScalariser(), pop, r, 0.5)
end

# A head fits the set of an `UncertaintySetVariance` once, so an estimator gives the
# optimisation that its own fitted set gives (#2318).
const MRS_RD = ReturnsResult(; nx = string.("A", 1:5), X = MRS_X)
# Each estimator fits without a draw, or with a seed, so its set is the set of the head.
const MRS_UES = (DeltaUncertaintySet(), DeltaUncertaintySet(; pe = nothing),
                 NormalUncertaintySet(; seed = 2318,
                                      alg = EllipsoidalUncertaintySetAlgorithm()))
mrs_usv(u; kwargs...) = UncertaintySetVariance(; ucs = u, kwargs...)

@testset "A Frontier on an UncertaintySetVariance spans the worst case of its estimator (#2318)" begin
    st = RiskMeasureSettings(; ub = Frontier(; N = 4))
    for ue in MRS_UES
        set = sigma_ucs(ue, MRS_RD, MRS_PR)
        res, ref = [optimise(MeanRisk(; r = mrs_usv(u; settings = st),
                                      obj = MaximumReturn(), opt = mrs_opt()), MRS_RD)
                    for u in (ue, set)]
        @test all(rc -> isa(rc, OptimisationSuccess), res.retcode)
        @test all(isapprox.(res.w, ref.w; rtol = 1e-6))
        # The rows of a ratio sweep need a parametric solver, so the span is read from the
        # model, which holds it before the first point.
        spans = map((ue, set)) do u
            x = optimise(MeanRisk(; r = mrs_usv(u; settings = st), obj = MaximumRatio(),
                                  opt = mrs_opt()), MRS_RD)
            return collect(only(x.model[:risk_frontier]).second[2])
        end
        @test isapprox(spans[1], spans[2]; rtol = 1e-8)
        # The top of the span is the worst case of the last point of the sweep.
        @test isapprox(spans[1][end], PO.ucs_variance(set, MRS_PR.sigma, res.w[end]);
                       rtol = 1e-3)
        res, ref = [optimise(NearOptimalCentering(; r = mrs_usv(u; settings = st),
                                                  opt = mrs_opt(),
                                                  alg = ConstrainedNearOptimalCentering()),
                             MRS_RD) for u in (ue, set)]
        @test isa.(res.noc_retcode, OptimisationSuccess) ==
              isa.(ref.noc_retcode, OptimisationSuccess)
        for (a, b, rc) in zip(res.w, ref.w, ref.noc_retcode)
            @test !isa(rc, OptimisationSuccess) || isapprox(a, b; rtol = 1e-5)
        end
    end
end

@testset "The anchors, the trackers and the Result read the set of an estimator (#2318)" begin
    w0 = fill(0.2, 5)
    wt = WeightsTracking(; w = w0)
    for ue in MRS_UES
        set = sigma_ucs(ue, MRS_RD, MRS_PR)
        wc(w) = PO.ucs_variance(set, MRS_PR.sigma, w)
        # The Result holds the fitted set.
        res = optimise(MeanRisk(; r = mrs_usv(ue), opt = mrs_opt()), MRS_RD)
        @test isapprox(expected_risk(res.r, res.w, res.pr), wc(res.w); rtol = 1e-12)
        @test expected_risk(res.r, res.w, res.pr) > dot(res.w, MRS_PR.sigma, res.w)
        # The anchors of a centring.
        for alg in (UnconstrainedNearOptimalCentering(), ConstrainedNearOptimalCentering())
            noc = NearOptimalCentering(; alg = alg, r = mrs_usv(ue), opt = mrs_opt())
            s = PO.near_optimal_centering_setup(noc, MRS_RD)
            @test isapprox(s.rk_opt - s.rk_delta, wc(s.w_opt); rtol = 1e-10)
        end
        # The benchmark risk of a dependent tracker, in the objective and as a constraint.
        for alg in (DependentVariableTracking(), IndependentVariableTracking())
            rtr(u) = RiskTrackingRiskMeasure(; tr = wt, r = mrs_usv(u), alg = alg)
            res, ref = [optimise(MeanRisk(; r = rtr(u), opt = mrs_opt()), MRS_RD)
                        for u in (ue, set)]
            @test isapprox(expected_risk(res.r, res.w, res.pr),
                           expected_risk(ref.r, ref.w, ref.pr); atol = 1e-8)
            te = RiskTrackingError(; tr = wt, r = mrs_usv(ue), alg = alg, err = 2e-6)
            res = optimise(MeanRisk(; obj = MaximumReturn(), opt = mrs_opt(; tr = te)),
                           MRS_RD)
            @test isa(res.retcode, OptimisationSuccess)
            risk = if isa(alg, DependentVariableTracking)
                wc(res.w) - wc(w0)
            else
                wc(res.w - w0)
            end
            # The solver can lower the slack of the lifted matrix, so the value level of the
            # ellipsoid is at or above the model expression.
            if isa(set, BoxUncertaintySet)
                @test isapprox(risk, 2e-6; rtol = 1e-3)
            end
        end
    end
end

@testset "NearOptimalCentering views its risk measures at the Investable Mask (#2320)" begin
    Xn = copy(MRS_X)
    Xn[1:100, 3] .= NaN
    rdn = ReturnsResult(; nx = string.("A", 1:5), X = Xn)
    keep = [1, 2, 4, 5]
    rdk = ReturnsResult(; nx = rdn.nx[keep], X = Xn[:, keep])
    opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = MRS_SLV)
    sds(i) = StandardDeviation(; sigma = MRS_PR.sigma[i, i])
    for (r, rk) in ((sds(1:5), sds(keep)), map(x -> (x, x), mrs_usv.(MRS_UES))...)
        for alg in (UnconstrainedNearOptimalCentering(), ConstrainedNearOptimalCentering())
            res = optimise(NearOptimalCentering(; r = r, opt = opt, alg = alg), rdn)
            ref = optimise(NearOptimalCentering(; r = rk, opt = opt, alg = alg), rdk)
            @test isa(res.retcode, OptimisationSuccess) ==
                  isa(ref.retcode, OptimisationSuccess)
            @test iszero(res.w[3])
            @test isapprox(res.w[keep], ref.w; rtol = 1e-5)
        end
    end
end

@testset "A RiskTrackingError resolves the Deferred Quantity of its measure" begin
    wt = WeightsTracking(; w = fill(0.2, 5))
    for alg in (DependentVariableTracking(), IndependentVariableTracking())
        ws = map((PortfolioOptimisersCovariance(), MRS_PR.sigma)) do sigma
            te = RiskTrackingError(; tr = wt, r = Variance(; sigma = sigma), alg = alg,
                                   err = 2e-6)
            res = optimise(MeanRisk(; obj = MaximumReturn(), opt = mrs_opt(; tr = te)),
                           MRS_RD)
            @test isa(res.retcode, OptimisationSuccess)
            return res.w
        end
        @test isapprox(ws[1], ws[2]; rtol = 1e-6)
    end
end
