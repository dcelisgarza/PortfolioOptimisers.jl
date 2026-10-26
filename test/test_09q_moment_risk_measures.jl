# The sweep of `src/16_RiskMeasures/03_MomentRiskMeasures.jl` (#598): every moment against a
# hand computation, the Moment Target and the divisor of each measure, the standardised moments
# against the unstandardised ones, the validation of `EvenMoment` and of `mu`, the helpers of
# the file, and the model of every low-order moment against its functor. The model of
# `MeanAbsoluteDeviation` measured `2 E[max(tau - r, 0)]`, which is the mean absolute deviation
# only when the target is the mean of the net series, so a stated target or a prior mean that
# is not the sample mean gave a model that disagreed with the functor.
using Clarabel, JuMP, LinearAlgebra, Statistics, StatsBase, StableRNGs

const PO = PortfolioOptimisers

# A short series with a mean of 0.002, so each hand computation is short.
const _x598 = [0.02, -0.01, 0.03, -0.04, 0.01]
const _X598 = reshape(_x598, :, 1)
const _d598 = _x598 .- mean(_x598)
const _dm598 = min.(_d598, 0)
const _lw598 = pweights([1.0, 2.0, 1.0, 3.0, 1.0])

const _Xm598 = 0.01 .* randn(StableRNG(598), 120, 4) .+ [0.001 0.0005 0.0012 0.0003]
const _rd598 = ReturnsResult(; nx = ["a", "b", "c", "d"], X = _Xm598)
const _pr598 = prior(EmpiricalPrior(), _rd598)
# Tight tolerances: at the default ones the epigraph of a second or an even moment sits up to
# 3e-4 above the functor, relative, and at these ones within 2e-6.
const _slv598 = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                       settings = Dict("verbose" => false, "tol_gap_abs" => 1e-10,
                                       "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10),
                       check_sol = (; allow_local = true, allow_almost = true))

# The value the model holds in `risk` and the value of the functor at the optimal weights.
function _model_and_value598(r, pr; obj = MinimumRisk(), fees = nothing)
    opt = JuMPOptimiser(; pe = pr, slv = _slv598, fees = fees)
    sol = optimise(MeanRisk(; r = r, obj = obj, opt = opt), _rd598)
    @test isa(sol.retcode, OptimisationSuccess)
    return JuMP.value(sol.model[:risk]),
           expected_risk(factory(r, pr, _slv598), sol.w, pr.X, fees), sol.w
end

@testset "Moment risk measures" begin
    @testset "Each moment against a hand computation" begin
        x = _x598
        T = length(x)
        @test LowOrderMoment(; alg = FirstLowerMoment())(x) ≈ mean(max.(mean(x) .- x, 0))
        @test LowOrderMoment(; alg = FirstLowerMoment())(x) ≈ 0.0108
        @test LowOrderMoment(; alg = MeanAbsoluteDeviation())(x) ≈ mean(abs.(_d598))
        @test LowOrderMoment(; alg = MeanAbsoluteDeviation())(x) ≈ 0.0216
        @test HighOrderMoment(; alg = ThirdLowerMoment())(x) ≈ -mean(_dm598 .^ 3)
        @test HighOrderMoment(; alg = FourthMoment())(x) ≈ mean(_d598 .^ 4)
        @test HighOrderMoment(; alg = FourthMoment(; alg = SemiMoment()))(x) ≈
              mean(_dm598 .^ 4)
        @test LowOrderMoment(; alg = EvenMoment())(x) ≈ sqrt(mean(_d598 .^ 4))
        @test LowOrderMoment(; alg = EvenMoment(; alg = SemiMoment()))(x) ≈
              sqrt(mean(_dm598 .^ 4))
        @test LowOrderMoment(; alg = EvenMoment(; p = 3, ddof = 1))(x) ≈
              (sum(_d598 .^ 6) / (T - 1))^(1 / 3)
        # At p = 2 and ddof = 0 the even moment is the square root of the fourth moment.
        @test LowOrderMoment(; alg = EvenMoment())(x) ≈
              sqrt(HighOrderMoment(; alg = FourthMoment())(x))
        # The two forms of the functor give one value.
        @test LowOrderMoment(; alg = FirstLowerMoment())(x) ≈
              LowOrderMoment(; alg = FirstLowerMoment())([1.0], _X598)
    end

    @testset "The mean absolute deviation against the first lower moment" begin
        x = _x598
        flm(mu) = LowOrderMoment(; mu = mu, alg = FirstLowerMoment())(x)
        mad(mu) = LowOrderMoment(; mu = mu, alg = MeanAbsoluteDeviation())(x)
        # Against the mean the two-sided measure is twice the one-sided one.
        @test mad(nothing) ≈ 2 * flm(nothing)
        # Against another target the identity keeps the gap between the mean and the target.
        for tau in (0.0, 0.004, -0.01)
            @test mad(tau) ≈ mean(abs.(x .- tau))
            @test flm(tau) ≈ mean(max.(tau .- x, 0))
            @test mad(tau) ≈ 2 * flm(tau) + mean(x) - tau
        end
        @test mad(0.0) ≈ 0.022
    end

    @testset "The Moment Target moves the deviations, not the reduction" begin
        x = _x598
        X = _X598
        w = [1.0]
        r(mu) = LowOrderMoment(; mu = mu, alg = FirstLowerMoment())
        @test r(0.005)(w, X) ≈ mean(max.(0.005 .- x, 0))
        @test r([0.005])(w, X) ≈ mean(max.(0.005 .- x, 0))
        @test r(VecScalar(; v = [0.005], s = 0.001))(w, X) ≈ mean(max.(0.006 .- x, 0))
        # Observation weights weigh the target and the mean of the reduction alike.
        rw = LowOrderMoment(; w = _lw598, alg = FirstLowerMoment())
        @test rw(w, X) ≈ mean(max.(mean(x, _lw598) .- x, 0), _lw598)
        # The five methods of `calc_moment_target`.
        w2 = [0.6, 0.4]
        @test PO.calc_moment_target(LowOrderMoment(), nothing, x) ≈ mean(x)
        @test PO.calc_moment_target(LowOrderMoment(; w = _lw598), nothing, x) ≈
              mean(x, _lw598)
        @test PO.calc_moment_target(LowOrderMoment(; mu = [0.1, 0.2]), w2, nothing) ≈ 0.14
        @test PO.calc_moment_target(HighOrderMoment(;
                                                    mu = VecScalar(; v = [0.1, 0.2],
                                                                   s = 0.05)), w2,
                                    nothing) ≈ 0.19
        @test PO.calc_moment_target(HighOrderMoment(; mu = 0.3), nothing, nothing) == 0.3
    end

    @testset "A per-asset target is net of the mean fee per period" begin
        X2 = [0.02 0.01; -0.01 0.0; 0.03 -0.02; -0.04 0.01; 0.01 0.02]
        w2 = [0.6, 0.4]
        fees = Fees(; l = 0.002)
        nx = PO.calc_net_returns(w2, X2, fees)
        dv = PO.calc_deviations_vec(LowOrderMoment(; mu = [0.001, 0.002]), w2, X2, fees)
        # The fee of a long book that sums to one is 0.002 per period.
        @test dv ≈ nx .- (dot(w2, [0.001, 0.002]) - 0.002)
        # A number and the mean of the series take no fee.
        @test PO.calc_deviations_vec(LowOrderMoment(; mu = 0.001), w2, X2, fees) ≈
              nx .- 0.001
        @test PO.calc_deviations_vec(LowOrderMoment(), w2, X2, fees) ≈ nx .- mean(nx)
        # The single-argument form reads no asset weights, so a per-asset target refuses it.
        @test PO.calc_deviations_vec(LowOrderMoment(), _x598) ≈ _d598
        @test_throws MethodError PO.calc_deviations_vec(LowOrderMoment(; mu = [0.1]), _x598)
        @test PO.supports_precomputed_returns(LowOrderMoment())
        @test PO.supports_precomputed_returns(HighOrderMoment(; mu = 0.1))
        @test !PO.supports_precomputed_returns(LowOrderMoment(; mu = [0.1]))
        @test !PO.supports_precomputed_returns(HighOrderMoment(;
                                                               mu = VecScalar(; v = [0.1],
                                                                              s = 0.0)))
        @test !PO.supports_precomputed_returns(LowOrderMoment(;
                                                              mu = SimpleExpectedReturns()))
    end

    @testset "The divisor of the second moment is the divisor of ve" begin
        x = _x598
        T = length(x)
        unc = SimpleVariance(; me = nothing, corrected = false)
        @test LowOrderMoment(; alg = SecondMoment())(x) ≈ sum(_d598 .^ 2) / (T - 1)
        @test LowOrderMoment(; alg = SecondMoment(; ve = unc))(x) ≈ sum(_d598 .^ 2) / T
        @test LowOrderMoment(; alg = SecondMoment(; alg1 = SemiMoment()))(x) ≈
              sum(_dm598 .^ 2) / (T - 1)
        @test LowOrderMoment(; alg = SecondMoment(; alg2 = SOCRiskExpr()))(x) ≈
              sqrt(sum(_d598 .^ 2) / (T - 1))
        # `ve` reads the deviations at a mean of zero, so its `me` has no effect.
        ve_me = SimpleVariance(; me = SimpleExpectedReturns())
        @test LowOrderMoment(; alg = SecondMoment(; ve = ve_me))(x) ≈
              LowOrderMoment(; alg = SecondMoment())(x)
        # Against a stated target it is the second moment about that target.
        @test LowOrderMoment(; mu = 0.05, alg = SecondMoment())(x) ≈
              sum((x .- 0.05) .^ 2) / (T - 1)
        # With observation weights the divisor is `varcorrection` of the weights.
        r = LowOrderMoment(; w = _lw598, alg = SecondMoment())
        @test r.alg.ve.w === _lw598
        dw = x .- mean(x, _lw598)
        @test r(x) ≈ StatsBase.varcorrection(_lw598, true) * sum(_lw598 .* dw .^ 2)
    end

    @testset "A standardised moment divides by the divisor of ve" begin
        x = _x598
        T = length(x)
        unc = SimpleVariance(; me = nothing, corrected = false)
        num = -mean(_dm598 .^ 3)
        r = HighOrderMoment(; alg = StandardisedHighOrderMoment())
        ru = HighOrderMoment(; alg = StandardisedHighOrderMoment(; ve = unc))
        @test r(x) ≈ num / (sum(_dm598 .^ 2) / (T - 1))^1.5
        @test ru(x) ≈ num / mean(_dm598 .^ 2)^1.5
        @test r(x) ≈ 1.4555036361053537
        @test ru(x) ≈ 2.034128169893555
        # The unstandardised value is the numerator alone.
        @test HighOrderMoment(; alg = ThirdLowerMoment())(x) ≈ num
        fm = FourthMoment()
        sfm = FourthMoment(; alg = SemiMoment())
        @test HighOrderMoment(; alg = StandardisedHighOrderMoment(; alg = fm))(x) ≈
              mean(_d598 .^ 4) / (sum(_d598 .^ 2) / (T - 1))^2
        @test HighOrderMoment(; alg = StandardisedHighOrderMoment(; alg = sfm))(x) ≈
              mean(_dm598 .^ 4) / (sum(_dm598 .^ 2) / (T - 1))^2
        # The weights of the measure reach the estimator of the denominator.
        rw = HighOrderMoment(; w = _lw598, alg = StandardisedHighOrderMoment())
        @test rw.alg.ve.w === _lw598
        dmw = min.(x .- mean(x, _lw598), 0)
        @test rw(x) ≈
              -mean(dmw .^ 3, _lw598) / var(dmw, _lw598; mean = 0, corrected = true)^1.5
    end

    @testset "EvenMoment takes an even order and a divisor T_d" begin
        x = _x598
        T = length(x)
        @test_throws DomainError EvenMoment(; p = 1)
        @test_throws DomainError EvenMoment(; ddof = -1)
        msg = try
            EvenMoment(; p = 1)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("`2p`", msg)
        # The order 2p is even for every integer p, so no odd order can be stated.
        @test all(p -> iseven(2 * EvenMoment(; p = p).p), 2:6)
        # Observation weights scale the divisor by their sum.
        r = LowOrderMoment(; w = _lw598, alg = EvenMoment(; ddof = 1))
        dw = x .- mean(x, _lw598)
        @test r(x) ≈ (sum(_lw598 .* dw .^ 4) / ((T - 1) / T * sum(_lw598)))^(1 / 2)
        rs = LowOrderMoment(; w = _lw598, alg = EvenMoment(; alg = SemiMoment()))
        @test rs(x) ≈ (sum(_lw598 .* min.(dw, 0) .^ 4) / sum(_lw598))^(1 / 2)
        # The degenerate divisors.
        @test LowOrderMoment(; alg = EvenMoment(; ddof = T))(x) == Inf
        @test_throws DomainError LowOrderMoment(; alg = EvenMoment(; ddof = T + 1))(x)
    end

    @testset "The validation of mu and w" begin
        for M in (LowOrderMoment, HighOrderMoment)
            @test_throws PO.IsEmptyError M(; mu = Float64[])
            @test_throws PO.IsNonFiniteError M(; mu = NaN)
            @test_throws PO.IsNonFiniteError M(; mu = [Inf])
            @test_throws DomainError M(; w = pweights([-1.0, 1, 1, 1, 1]))
        end
        @test_throws DomainError VecScalar(; v = [NaN], s = 0.0)
    end

    @testset "factory, the views and the deferred slot" begin
        pr = _pr598
        # A member that holds no estimator returns unchanged.
        @test factory(FirstLowerMoment(), _lw598) === FirstLowerMoment()
        @test factory(EvenMoment(), _lw598) isa EvenMoment
        s = factory(StandardisedHighOrderMoment(), _lw598)
        @test s.ve.w === _lw598
        @test s.alg isa ThirdLowerMoment
        for M in (LowOrderMoment, HighOrderMoment)
            # A measure that states nothing takes the mean and the weights of the prior.
            r = factory(M(), pr)
            @test r.mu === pr.mu
            @test r.w === pr.w
            # A stated target and stated weights stay.
            ws = pweights(fill(1.0, 120))
            r = factory(M(; mu = 0.01, w = ws), pr)
            @test r.mu == 0.01
            @test r.w === ws
            # A deferred target resolves against the prior.
            r = factory(M(; mu = SimpleExpectedReturns()), pr)
            @test r.mu ≈ vec(mean(_Xm598; dims = 1))
            @test PO.resolve_deferred_quantities(M(; mu = 0.01), pr).mu == 0.01
            # A view slices a per-asset target and keeps every other slot.
            v = PO.port_opt_view(M(; mu = [0.1, 0.2, 0.3, 0.4], w = ws), [2, 4])
            @test v.mu == [0.2, 0.4]
            @test v.w === ws
            @test PO.port_opt_view(M(; mu = 0.01), [2, 4]).mu == 0.01
            @test PO.port_opt_view(M(), [2, 4]).mu === nothing
            dq = PO.port_opt_view(M(; mu = SimpleExpectedReturns()), [2, 4]).mu
            @test dq isa SimpleExpectedReturns
        end
    end

    @testset "moment_risk overwrites the deviations of a full fourth moment" begin
        val = copy(_d598)
        PO.moment_risk(HighOrderMoment(; alg = FourthMoment()), val)
        @test val ≈ _d598 .^ 4
        val = copy(_d598)
        PO.moment_risk(HighOrderMoment(; alg = FourthMoment(; alg = SemiMoment())), val)
        @test val == _d598
    end

    @testset "The model of each low-order moment agrees with its functor" begin
        algs = (FirstLowerMoment(), MeanAbsoluteDeviation(), SecondMoment(),
                SecondMoment(; alg1 = SemiMoment()), SecondMoment(; alg2 = SOCRiskExpr()),
                EvenMoment(), EvenMoment(; alg = SemiMoment()))
        for alg in algs, mu in (nothing, 0.0, fill(0.004, 4))
            m, v, _ = _model_and_value598(LowOrderMoment(; mu = mu, alg = alg), _pr598)
            @test isapprox(m, v; rtol = 1e-5, atol = 1e-9)
        end
    end

    @testset "A zero-weight loss does not move the even moment model" begin
        # Every asset loses 0.3 on the third observation, which carries no weight. Its power
        # cone rows do not change the optimum, but at the default tolerances the solver
        # stalled on them before they were dropped, as in the EVaR and power-norm programmes.
        X0 = copy(_Xm598)
        X0[3, :] .= -0.3
        rd0 = ReturnsResult(; nx = _rd598.nx, X = X0)
        rd1 = ReturnsResult(; nx = _rd598.nx, X = X0[setdiff(1:120, 3), :])
        w0 = pweights([1.0; 1.0; 0.0; ones(117)])
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     settings = Dict("verbose" => false),
                     check_sol = (; allow_local = true, allow_almost = true))
        for (p, alg) in ((2, FullMoment()), (2, SemiMoment()), (3, FullMoment()))
            e = EvenMoment(; p = p, alg = alg)
            s0 = optimise(MeanRisk(; r = LowOrderMoment(; w = w0, mu = 0.0, alg = e),
                                   opt = JuMPOptimiser(; pe = prior(EmpiricalPrior(), rd0),
                                                       slv = slv)), rd0)
            s1 = optimise(MeanRisk(; r = LowOrderMoment(; mu = 0.0, alg = e),
                                   opt = JuMPOptimiser(; pe = prior(EmpiricalPrior(), rd1),
                                                       slv = slv)), rd1)
            @test isapprox(s0.w, s1.w; atol = 1e-8)
            @test isapprox(JuMP.value(s0.model[:risk]), JuMP.value(s1.model[:risk]);
                           rtol = 1e-8)
        end
    end

    @testset "The model of the mean absolute deviation holds for every target" begin
        M(; kw...) = LowOrderMoment(; alg = MeanAbsoluteDeviation(), kw...)
        ws = pweights(collect(range(0.5, 1.5; length = 120)))
        cases = ((M(; mu = 0.0), _pr598, nothing),
                 (M(; mu = VecScalar(; v = fill(0.001, 4), s = 0.0005)), _pr598, nothing),
                 (M(; w = ws), _pr598, nothing), (M(; w = ws, mu = 0.0), _pr598, nothing),
                 (M(; mu = 0.0), _pr598, Fees(; l = 0.002)))
        for (r, pr, fees) in cases
            m, v, _ = _model_and_value598(r, pr; fees = fees)
            @test isapprox(m, v; rtol = 1e-6, atol = 1e-9)
        end
        # A prior mean that is not the sample mean moves the default target off the mean.
        prjs = prior(EmpiricalPrior(; me = ShrunkExpectedReturns(; alg = JamesStein())),
                     _rd598)
        w1 = [1.0, 0.0, 0.0, 0.0]
        @test abs(mean(_Xm598 * w1) - dot(w1, prjs.mu)) > 1e-4
        m, v, _ = _model_and_value598(M(), prjs)
        @test isapprox(m, v; rtol = 1e-6, atol = 1e-9)
        # The bound acts on the mean absolute deviation itself.
        ub = 0.006
        r = M(; mu = 0.0, settings = RiskMeasureSettings(; ub = ub))
        _, v, _ = _model_and_value598(r, _pr598; obj = MaximumReturn())
        @test v <= ub + 1e-7
        @test v >= ub - 1e-4
    end
end
