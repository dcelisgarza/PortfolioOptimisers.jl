# The sweep of the higher moment measures (#599): `Kurtosis`, `NegativeSkewness`, `Skewness` and
# `VarianceSkewKurtosis` against hand computations, the orientation of the two skewness
# measures, the negative spectral coskewness against its definition, the views against a refit
# on the subset, the composite against its three terms, the semidefinite relaxation against the
# value it bounds, and the copies without bounds of the two composites whose children hold
# bounds.
using Clarabel, JuMP, LinearAlgebra, Statistics, StatsBase, StableRNGs

const PO = PortfolioOptimisers

# Three assets with a positive skew, from squared normal shocks, and a weight vector.
const _X599 = let rng = StableRNG(599)
    (randn(rng, 200, 3) .^ 2 .- 1) .* [0.01 0.012 0.008] .+ 0.002 .* randn(rng, 200, 3)
end
const _w599 = [0.5, 0.3, 0.2]
const _pr599 = prior(HighOrderPriorEstimator(), _X599)
const _slv599 = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                       settings = Dict("verbose" => false, "tol_gap_abs" => 1e-10,
                                       "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10),
                       check_sol = (; allow_local = true, allow_almost = true))

_m3_599(pr, w) = dot(w, pr.sk, kron(w, w))
_m4_599(pr, w) = dot(kron(w, w), pr.kt, kron(w, w))

# The negative spectral coskewness of the definition: minus the sum of the negative spectral
# parts of the slices of the coskewness matrix.
function _V599(sk)
    n = size(sk, 1)
    V = zeros(n, n)
    for i in 1:n
        E = eigen(Symmetric(sk[:, ((i - 1) * n + 1):(i * n)]))
        V .-= E.vectors * Diagonal(min.(E.values, 0)) * transpose(E.vectors)
    end
    return V
end

function _solve599(r, obj = MinimumRisk(); pr = _pr599)
    return optimise(MeanRisk(; r = r, obj = obj,
                             opt = JuMPOptimiser(; pe = pr, slv = _slv599)))
end

@testset "Higher moment risk measures" begin
    X, w, pr = _X599, _w599, _pr599
    x = X * w
    T = length(x)
    d = x .- mean(x)
    ow = pweights(collect(range(0.5, 1.5; length = T)))
    dw = x .- mean(x, ow)
    @testset "Kurtosis is the fourth moment of its cokurtosis, or its square root" begin
        m4 = mean(d .^ 4)
        @test factory(Kurtosis(), pr)(w) ≈ sqrt(m4)
        for alg in (QuadRiskExpr(), SquaredSOCRiskExpr(), RSOCRiskExpr())
            @test factory(Kurtosis(; alg = alg), pr)(w) ≈ m4
        end
        # The divisor is `T`, and the moment is neither standardised nor an excess.
        @test !isapprox(m4, mean(d .^ 4) / var(x; corrected = false)^2)
        # Equation 4 of `pkurt`: the Kronecker form agrees with the series.
        @test _m4_599(pr, w) ≈ m4 rtol = 1e-10
        # The functor reads the matrix it holds, so a stated matrix gives its own moment.
        @test Kurtosis(; kt = 2 .* pr.kt)(w) ≈ sqrt(2 * m4)
        # The values of the returns are on the moment measures (#1846).
        semi = mean(min.(d, 0) .^ 4)
        @test LowOrderMoment(; alg = EvenMoment())(w, X) ≈ sqrt(m4)
        @test HighOrderMoment(; alg = FourthMoment())(w, X) ≈ m4
        @test LowOrderMoment(; alg = EvenMoment(; alg = SemiMoment()))(w, X) ≈ sqrt(semi)
        @test HighOrderMoment(; alg = FourthMoment(; alg = SemiMoment()))(w, X) ≈ semi
        @test HighOrderMoment(; w = ow, alg = FourthMoment())(w, X) ≈ mean(dw .^ 4, ow)
        @test HighOrderMoment(; w = ow, alg = FourthMoment(; alg = SemiMoment()))(w, X) ≈
              mean(min.(dw, 0) .^ 4, ow)
        dc = x .- dot(w, pr.mu)
        @test factory(HighOrderMoment(; w = ow, alg = FourthMoment()), pr)(w, X) ≈
              mean(dc .^ 4, ow)
    end
    @testset "Kurtosis and Skewness read their matrices, not the returns" begin
        @test PO.risk_input_kind(Kurtosis()) isa PO.WeightsInput
        @test PO.risk_input_kind(Skewness()) isa PO.WeightsInput
        @test !PO.supports_precomputed_returns(Kurtosis(; kt = pr.kt))
        @test !PO.supports_precomputed_returns(Skewness(; sk = pr.sk))
        # The fields that the functors read from the returns are gone.
        @test_throws MethodError Kurtosis(; mu = 0.1)
        @test_throws MethodError Kurtosis(; w = ow)
        @test_throws MethodError Kurtosis(; alg1 = SemiMoment())
        @test_throws MethodError Kurtosis(; pe = EmpiricalPrior())
        @test_throws MethodError Skewness(; mu = 0.1)
        @test_throws MethodError Skewness(; w = ow)
        @test_throws MethodError Skewness(; pe = EmpiricalPrior())
        @test_throws DimensionMismatch Kurtosis(; kt = pr.kt[:, 1:4])
        @test_throws PO.IsEmptyError Kurtosis(; kt = zeros(0, 0))
        @test_throws DimensionMismatch Skewness(; sk = pr.sk[:, 1:4])
    end
    @testset "A deferred co-moment resolves on its own" begin
        kd = factory(Kurtosis(; kt = Cokurtosis()), pr)
        @test kd.kt ≈ pr.kt
        sd = factory(Skewness(; sk = Coskewness()), pr)
        @test sd.sk ≈ pr.sk
        # A prior estimator in the slot gives the matrix of its own prior.
        @test factory(Kurtosis(; kt = HighOrderPriorEstimator()), pr).kt ≈ pr.kt
        @test factory(Skewness(; sk = HighOrderPriorEstimator()), pr).sk ≈ pr.sk
        @test_throws DomainError Kurtosis(; N = 0)
        @test_throws ArgumentError NegativeSkewness(; sk = Coskewness(), V = pr.V)
        prs = prior(HighOrderPriorEstimator(), X[:, [1, 3]])
        @test PO.port_opt_view(Kurtosis(; kt = pr.kt), [1, 3]).kt ≈ prs.kt atol = 1e-18
    end
    @testset "A low order prior gives no cokurtosis" begin
        prl = prior(EmpiricalPrior(), X)
        kl = factory(Kurtosis(), prl)
        @test isnothing(kl.kt)
        @test isnothing(factory(Skewness(), prl).sk)
        @test_throws PO.IsNothingError expected_risk(kl, w, X)
        # A deferred co-moment needs only the returns, so it resolves here too.
        @test factory(Kurtosis(; kt = Cokurtosis()), prl).kt ≈ pr.kt
        @test factory(Skewness(; sk = Coskewness()), prl).sk ≈ pr.sk
        @test_throws ArgumentError _solve599(Kurtosis(); pr = prl)
    end
    @testset "The model and the functor read the same kt" begin
        full = _solve599(Kurtosis())
        @test JuMP.value(full.model[:risk]) ≈ factory(Kurtosis(), pr)(full.w) rtol = 1e-5
        @test JuMP.value(full.model[:risk]) ≈ expected_risk(Kurtosis(), full.w, pr) rtol = 1e-5
        # A lower semi-cokurtosis gives the heuristic of `pkurt`, which is not the semi-moment
        # of the portfolio returns.
        prs = prior(HighOrderPriorEstimator(; kte = Cokurtosis(; alg = SemiMoment())), X)
        hs = _solve599(Kurtosis(); pr = prs)
        @test JuMP.value(hs.model[:risk]) ≈ factory(Kurtosis(), prs)(hs.w) rtol = 1e-5
        @test !isapprox(JuMP.value(hs.model[:risk]),
                        LowOrderMoment(; alg = EvenMoment(; alg = SemiMoment()))(hs.w, X);
                        rtol = 1e-2)
        # A deferred semi-cokurtosis in `kt` fits the same tensor against the default prior,
        # so it gives the model of the semi prior (#1818).
        ds = _solve599(Kurtosis(; kt = Cokurtosis(; alg = SemiMoment())))
        @test ds.w ≈ hs.w rtol = 1e-5
        @test JuMP.value(ds.model[:risk]) ≈ JuMP.value(hs.model[:risk]) rtol = 1e-6
        # A deferred cokurtosis that carries observation weights weights the model and the
        # functor alike.
        dk = Kurtosis(; kt = Cokurtosis(; w = ow))
        dks = _solve599(dk)
        @test JuMP.value(dks.model[:risk]) ≈ factory(dk, pr)(dks.w) rtol = 1e-6
        @test !isapprox(dks.w, full.w; atol = 1e-2)
    end
    @testset "The views agree with a refit on the subset" begin
        idx = [1, 3]
        prs = prior(HighOrderPriorEstimator(), X[:, idx])
        kv = PO.port_opt_view(Kurtosis(; kt = pr.kt), idx)
        @test kv.kt ≈ prs.kt atol = 1e-18
        sv = PO.port_opt_view(Skewness(; sk = pr.sk), idx)
        @test sv.sk ≈ prs.sk atol = 1e-18
        nv = PO.port_opt_view(NegativeSkewness(; sk = pr.sk, V = pr.V), idx, X)
        @test nv.V ≈ prs.V rtol = 1e-12
        # A Deferred Quantity crosses the view unresolved, and a measure with no matrix
        # returns as itself.
        @test PO.port_opt_view(Kurtosis(; kt = Cokurtosis()), idx).kt isa Cokurtosis
        @test PO.port_opt_view(Skewness(; sk = Coskewness()), idx).sk isa Coskewness
        k0, s0 = Kurtosis(), Skewness()
        @test PO.port_opt_view(k0, idx) === k0
        @test PO.port_opt_view(s0, idx) === s0
    end
    @testset "NegativeSkewness is the negative quadratic skewness" begin
        @test pr.V ≈ _V599(pr.sk) rtol = 1e-12
        @test isposdef(pr.V + 1e-12 * I)
        soc = factory(NegativeSkewness(), pr)(w)
        @test soc ≈ sqrt(dot(w, pr.V, w))
        for alg in (QuadRiskExpr(), SquaredSOCRiskExpr())
            @test factory(NegativeSkewness(; alg = alg), pr)(w) ≈ soc^2
        end
        # Smaller is better: the mirrored data have the opposite skew and a larger measure.
        prn = prior(HighOrderPriorEstimator(), -X)
        @test factory(Skewness(), pr)(w) > 0
        @test factory(Skewness(), prn)(w) < 0
        @test factory(NegativeSkewness(), prn)(w) > 2 * soc
        @test !PO.bigger_is_better(NegativeSkewness())
    end
    @testset "Skewness is the third moment of its coskewness" begin
        # Equation 1 of `sdpmom`: the moment of the coskewness agrees with the series.
        @test factory(Skewness(), pr)(w) ≈ mean(d .^ 3) rtol = 1e-10
        @test _m3_599(pr, w) ≈ mean(d .^ 3) rtol = 1e-10
        @test PO.bigger_is_better(Skewness())
    end
    @testset "ThirdCentralMoment divides the third moment by T and the variance by T - 1" begin
        # The standardised value is the old value of the functor of `Skewness` (#1846).
        @test ThirdCentralMoment()(w, X) ≈ mean(d .^ 3)
        g1 = mean(d .^ 3) / std(d; corrected = false)^3
        sk = ThirdCentralMoment(; ve = SimpleVariance())(w, X)
        @test sk ≈ mean(d .^ 3) / std(d; corrected = true)^3
        @test sk ≈ ((T - 1) / T)^(3 / 2) * g1
        @test sk ≈ ((T - 1) / T)^(3 / 2) * StatsBase.skewness(x)
        @test ThirdCentralMoment(; ve = SimpleVariance(; me = nothing, corrected = false))(x) ≈
              g1
        @test ThirdCentralMoment(; w = ow, ve = SimpleVariance())(w, X) ≈
              mean(dw .^ 3, ow) / std(dw, ow; corrected = true, mean = 0)^3
        @test PO.bigger_is_better(ThirdCentralMoment(; ve = SimpleVariance()))
    end
    @testset "The bound traits of Skewness rebuild a MaxRiskMeasureSettings" begin
        s = Skewness(; settings = MaxRiskMeasureSettings(; scale = 0.5, lb = 1e-7))
        b = PO.bounds_risk_measure(s, 2e-7)
        @test (b.settings.lb, b.settings.scale, b.settings.rke) == (2e-7, 0.5, true)
        n = PO.no_bounds_risk_measure(s)
        @test (n.settings.lb, n.settings.scale, n.settings.rke) == (nothing, 0.5, true)
        e = PO.no_risk_expr_risk_measure(s)
        @test (e.settings.lb, e.settings.scale, e.settings.rke) == (1e-7, 0.5, false)
        ne = PO.no_bounds_no_risk_expr_risk_measure(s)
        @test (ne.settings.lb, ne.settings.scale, ne.settings.rke) == (nothing, 1, false)
    end
    @testset "VarianceSkewKurtosis is the weighted sum of three raw moments" begin
        r = VarianceSkewKurtosis(;
                                 vr = Variance(;
                                               settings = RiskMeasureSettings(;
                                                                              scale = 2.0)),
                                 sk = Skewness(;
                                               settings = MaxRiskMeasureSettings(;
                                                                                 scale = 0.5)),
                                 kt = Kurtosis(;
                                               settings = RiskMeasureSettings(;
                                                                              scale = 3.0)))
        rf = factory(r, pr)
        hand = 2.0 * var(x) - 0.5 * mean(d .^ 3) + 3.0 * mean(d .^ 4)
        @test rf(w, X) ≈ hand rtol = 1e-10
        @test rf(w, X) ≈
              2.0 * dot(w, pr.sigma, w) - 0.5 * _m3_599(pr, w) + 3.0 * _m4_599(pr, w)
        @test !PO.supports_precomputed_returns(r)
        # A call that dispatch cannot fold runs the method, which a coverage run counts (#1912).
        dyn = Ref{Any}(r)
        @test !PO.supports_precomputed_returns(dyn[])
        # The functors of the children give the raw terms, the kurtosis as its square root.
        _, s3, k4 = PO.variance_skew_kurtosis_terms(rf, w)
        @test 0.5 * rf.sk(w) ≈ s3
        @test 3.0 * rf.kt(w)^2 ≈ k4
    end
    @testset "The semidefinite relaxation of VarianceSkewKurtosis" begin
        sol = _solve599(VarianceSkewKurtosis())
        m, ws = sol.model, sol.w
        # The variance and kurtosis blocks lie at or above their moments.
        @test JuMP.value(m[:vr_risk_1]) >= dot(ws, pr.sigma, ws) - 1e-12
        @test JuMP.value(m[:kt_risk_1]) >= _m4_599(pr, ws) - 1e-12
        @test JuMP.value(m[:risk]) ≈ factory(VarianceSkewKurtosis(), pr)(ws, X) rtol = 1e-3
        # A floor holds the relaxed third moment, and the weights miss it.
        lb = 1.2 * _m3_599(pr, ws)
        sl = _solve599(VarianceSkewKurtosis(;
                                            sk = Skewness(;
                                                          settings = MaxRiskMeasureSettings(;
                                                                                            lb = lb))))
        @test JuMP.value(sl.model[:sk_risk_1]) >= lb * (1 - 1e-6)
        @test _m3_599(pr, sl.w) < lb
        # Every single asset has a third moment above the floor, so the floor is feasible.
        @test all(i -> _m3_599(pr, Float64.(1:3 .== i)) > lb, 1:3)
        # A ceiling on a block that lies above its moment holds the weights.
        u = 0.95 * sqrt(_m4_599(pr, ws))
        su = _solve599(VarianceSkewKurtosis(;
                                            kt = Kurtosis(;
                                                          settings = RiskMeasureSettings(;
                                                                                         ub = u))),
                       MaximumReturn())
        @test sqrt(_m4_599(pr, su.w)) <= u * (1 + 1e-6)
        @test haskey(su.model, :kt_risk_1_ub)
    end
    @testset "A copy without bounds clears the bounds of the children" begin
        rb = VarianceSkewKurtosis(; settings = RiskMeasureSettings(; ub = 1.0),
                                  vr = Variance(;
                                                settings = RiskMeasureSettings(; ub = 1e-6,
                                                                               scale = 2.0)),
                                  sk = Skewness(;
                                                settings = MaxRiskMeasureSettings(;
                                                                                  lb = 1e-7,
                                                                                  scale = 0.5)),
                                  kt = Kurtosis(;
                                                settings = RiskMeasureSettings(; ub = 1e-3)))
        nb = PO.no_bounds_risk_measure(rb)
        @test isnothing(nb.settings.ub)
        @test isnothing(nb.vr.settings.ub)
        @test isnothing(nb.sk.settings.lb)
        @test isnothing(nb.kt.settings.ub)
        @test (nb.vr.settings.scale, nb.sk.settings.scale) == (2.0, 0.5)
        @test nb.settings.rke
        @test !nb.vr.settings.rke && !nb.sk.settings.rke && !nb.kt.settings.rke
        ne = PO.no_bounds_no_risk_expr_risk_measure(rb)
        @test !ne.settings.rke
        @test isnothing(ne.vr.settings.ub)
        @test ne.vr.settings.scale == 2.0
        # Without the child ceiling the maximum-return corner holds the asset of the largest
        # mean alone.
        imax = argmax(pr.mu)
        corner = _solve599(nb, MaximumReturn())
        @test corner.w[imax] > 1 - 1e-6
        g = GenericValueatRiskRange(; settings = RiskMeasureSettings(; ub = 1.0),
                                    loss = ConditionalValueatRisk(;
                                                                  settings = RiskMeasureSettings(;
                                                                                                 ub = 0.01)),
                                    gain = ConditionalValueatRisk(;
                                                                  settings = RiskMeasureSettings(;
                                                                                                 ub = 0.02)))
        gn = PO.no_bounds_risk_measure(g)
        @test isnothing(gn.settings.ub)
        @test isnothing(gn.loss.settings.ub) && isnothing(gn.gain.settings.ub)
        gne = PO.no_bounds_no_risk_expr_risk_measure(g)
        @test !gne.settings.rke
        @test isnothing(gne.loss.settings.ub) && isnothing(gne.gain.settings.ub)
    end
end
