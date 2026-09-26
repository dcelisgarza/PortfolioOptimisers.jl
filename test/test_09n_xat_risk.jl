# The quantile measures of `src/16_RiskMeasures/06_XatRisk/01_XatRisk.jl`, checked against the
# forms their docstrings state: every empirical functor against an independent solve of the
# mixed-integer programme of Cajas, Equation 7.51, the parametric z-scores against the quantiles
# of the unit-variance distributions, the models against the functors, and the validation of
# the constructors.
using Clarabel, Distributions, HiGHS, JuMP

# The minimum of the mixed-integer programme over a loss series `l`, with a big-M constant that
# fits the losses and a tight integrality tolerance, so that no row is left loose.
function mip_quantile(l, alpha, s = 1e-5, p = ones(length(l)))
    T = length(l)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    set_attribute(m, "mip_feasibility_tolerance", 1e-9)
    set_attribute(m, "mip_rel_gap", 0.0)
    b = 2 * (maximum(l) - minimum(l)) + 1
    @variable(m, r)
    @variable(m, z[1:T], Bin)
    @constraint(m, dot(p, z) <= (alpha - s) * sum(p))
    @constraint(m, r .>= l .- b .* z)
    @objective(m, Min, r)
    optimize!(m)
    return value(r)
end
# The solver returns an order statistic to a rounding error, and neighbouring order statistics
# of the samples below differ by far more than this tolerance.
same(a, b) = isapprox(a, b; atol = 1e-12)
# The drawdown series of the definitions, with the running peaks starting at c_0 = 0 and C_0 = 1.
abs_dd(x) = cumsum(x) .- accumulate(max, cumsum(x); init = zero(eltype(x)))
rel_dd(x) = cumprod(1 .+ x) ./ accumulate(max, cumprod(1 .+ x); init = one(eltype(x))) .- 1

@testset "Quantile measures" begin
    @testset "The z-scores are the quantiles of the unit-variance distributions" begin
        PO = PortfolioOptimisers
        L = Laplace(0.0, 1 / sqrt(2))
        t5 = TDist(5)
        for alpha in (0.01, 0.05, 0.3, 0.5, 0.7, 0.95)
            @test isapprox(PO.compute_value_at_risk_z(Laplace(), alpha),
                           cquantile(L, alpha); atol = 1e-14)
            @test isapprox(PO.compute_value_at_risk_cz(Laplace(), alpha),
                           quantile(L, alpha); atol = 1e-14)
            # `dist` reads as the standardised shape, so its own parameters do not move it.
            @test PO.compute_value_at_risk_z(Laplace(3.0, 2.0), alpha) ==
                  PO.compute_value_at_risk_z(Laplace(), alpha)
            @test PO.compute_value_at_risk_z(Normal(), alpha) == cquantile(Normal(), alpha)
            @test PO.compute_value_at_risk_cz(Normal(), alpha) == quantile(Normal(), alpha)
            @test isapprox(PO.compute_value_at_risk_z(t5, alpha),
                           cquantile(t5, alpha) * sqrt(3 / 5))
            @test isapprox(PO.compute_value_at_risk_cz(t5, alpha),
                           quantile(t5, alpha) * sqrt(3 / 5))
        end
        # The three distributions are symmetric, so the two tails are mirror images.
        for dist in (Normal(), t5, Laplace()), alpha in (0.05, 0.3)
            @test isapprox(PO.compute_value_at_risk_cz(dist, alpha),
                           -PO.compute_value_at_risk_z(dist, alpha))
        end
        # A variance needs more than two degrees of freedom.
        @test_throws DomainError PO.compute_value_at_risk_z(TDist(2), 0.05)
        @test_throws DomainError PO.compute_value_at_risk_cz(TDist(2), 0.05)
    end
    @testset "Each empirical functor is the minimum of the mixed-integer programme" begin
        rng = StableRNG(1046)
        for T in (20, 37, 100), alpha in (0.05, 0.07, 0.29, 0.5, 0.9)
            x = 0.02 .* randn(rng, T) .+ 0.001
            x0 = copy(x)
            p = rand(rng, T)
            p[1] = 0.0
            pw = pweights(p)
            p0 = copy(p)
            @test same(ValueatRisk(; alpha = alpha)(x), mip_quantile(-x, alpha))
            @test same(ValueatRisk(; alpha = alpha, w = pw)(x),
                       mip_quantile(-x, alpha, 1e-5, p))
            # Unit weights select the same order statistic as no weights.
            @test ValueatRisk(; alpha = alpha, w = pweights(ones(T)))(x) ==
                  ValueatRisk(; alpha = alpha)(x)
            # The slack is read off the formulation.
            @test same(ValueatRisk(; alpha = alpha, alg = MIPValueatRisk(; s = 0.015))(x),
                       mip_quantile(-x, alpha, 0.015))
            for beta in (alpha, 0.2)
                @test same(ValueatRiskRange(; alpha = alpha, beta = beta)(x),
                           mip_quantile(-x, alpha) + mip_quantile(x, beta))
                @test same(ValueatRiskRange(; alpha = alpha, beta = beta, w = pw)(x),
                           mip_quantile(-x, alpha, 1e-5, p) +
                           mip_quantile(x, beta, 1e-5, p))
            end
            @test same(DrawdownatRisk(; alpha = alpha)(x), mip_quantile(-abs_dd(x), alpha))
            @test same(DrawdownatRisk(; alpha = alpha, w = pw)(x),
                       mip_quantile(-abs_dd(x), alpha, 1e-5, p))
            @test same(DrawdownatRisk(; alpha = alpha, s = 0.015)(x),
                       mip_quantile(-abs_dd(x), alpha, 0.015))
            @test same(RelativeDrawdownatRisk(; alpha = alpha)(x),
                       mip_quantile(-rel_dd(x), alpha))
            @test same(RelativeDrawdownatRisk(; alpha = alpha, w = pw)(x),
                       mip_quantile(-rel_dd(x), alpha, 1e-5, p))
            # No functor writes its input or the weights it holds.
            @test x == x0
            @test p == p0
        end
    end
    @testset "Without weights the index is the ceiling of alpha T" begin
        # `alpha * T` rounds above the integer at 0.07 * 100, and the slack absorbs it, so every
        # level on a grid of hundredths selects the order statistic that its decimal value names.
        x = collect(range(-0.05, 0.05; length = 100)) .+ 0.0001 .* sin.(1:100)
        s = sort(x)
        for j in 1:99
            @test ValueatRisk(; alpha = j / 100)(x) == -s[j]
        end
        @test 0.07 * 100 > 7
        # A fractional alpha T takes the next order statistic.
        x = collect(range(-0.05, 0.05; length = 37))
        @test ValueatRisk(; alpha = 0.1)(x) == -sort(x)[4]
        # The absolute and relative drawdowns follow their definitions.
        x = [0.01, -0.02, 0.005, -0.03, 0.04, -0.01]
        @test PortfolioOptimisers.absolute_drawdown_vec(x) ≈ abs_dd(x)
        @test PortfolioOptimisers.relative_drawdown_vec(x) ≈ rel_dd(x)
        @test PortfolioOptimisers.absolute_drawdown_vec(1:3) == zeros(Int, 3)
    end
    @testset "The defaults of the big-M constant and the slack" begin
        PO = PortfolioOptimisers
        # A `nothing` constant passes on, for the builder to derive from the data.
        @test PO.mip_var_bounds(nothing, nothing) === (nothing, 1e-5)
        @test PO.mip_var_bounds(2.0, nothing) == (2.0, 1e-5)
        @test PO.mip_var_bounds(nothing, 0.01) === (nothing, 0.01)
    end
    @testset "The bound on the gross exposure" begin
        PO = PortfolioOptimisers
        function g(wb; bgt = nothing, sbgt = nothing, gbgt = nothing)
            return PO.gross_exposure_bound(wb, bgt, sbgt, gbgt, 4)
        end
        # Long-only weights have a gross exposure equal to their sum.
        @test g(WeightBounds(); bgt = 1.0) == 1.0
        @test g(WeightBounds(); bgt = BudgetRange(; lb = 0.8, ub = 1.2)) == 1.2
        # Without a budget the bounds alone bound it, asset by asset.
        @test g(WeightBounds()) == 4.0
        @test g(WeightBounds(; lb = 0.0, ub = [0.1, 0.2, 0.3, 0.4])) == 1.0
        # With shorts the long parts sum to at most bgt + sbgt and the short parts to sbgt.
        @test g(WeightBounds(; lb = -1.0, ub = 1.0); bgt = 1.0, sbgt = 0.5) == 2.0
        @test g(WeightBounds(; lb = -1.0, ub = 1.0); bgt = 1.0) == 4.0
        @test g(WeightBounds(; lb = -1.0, ub = 1.0); gbgt = 1.5) == 1.5
        @test g(WeightBounds(; lb = -1.0, ub = 1.0);
                bgt = BudgetRange(; lb = nothing, ub = 1.0),
                sbgt = BudgetRange(; lb = nothing, ub = 0.2)) == 1.4
        # No finite bound gives none.
        @test g(WeightBounds(; lb = -Inf, ub = Inf); bgt = 1.0) == Inf
        @test g(WeightBounds(; lb = nothing, ub = 1.0); bgt = 1.0) == Inf
        # Two weight builds on one model constrain the same weights, so the smaller bound holds.
        m = JuMP.Model()
        @test PO.set_gross_exposure_bound!(m, 2.0) == 2.0
        @test PO.set_gross_exposure_bound!(m, 3.0) == 2.0
        @test PO.set_gross_exposure_bound!(m, 1.5) == 1.5
        @test m[:w_gross_ub] == 1.5
    end
    @testset "The derived big-M constant keeps the programme exact at the default tolerance" begin
        PO = PortfolioOptimisers
        rng = StableRNG(1046)
        X = 0.01 .* randn(rng, 100, 4) .+ 0.0005
        pr = prior(EmpiricalPrior(), X)
        opts = Dict("log_to_console" => false, "mip_rel_gap" => 0.0)
        slv = Solver(; name = :highs, solver = HiGHS.Optimizer, settings = opts)
        tslv = Solver(; name = :highs, solver = HiGHS.Optimizer,
                      settings = merge(opts, Dict("mip_feasibility_tolerance" => 1e-9)))
        # With b = 1000 the default tolerance of 1e-6 left these three about 1e-4 apart (#1323).
        for r in (ValueatRisk(; alpha = 0.29), DrawdownatRisk(; alpha = 0.29),
                  ValueatRiskRange(; alpha = 0.29, beta = 0.1))
            sol = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                    opt = JuMPOptimiser(; pe = pr, slv = slv)))
            tsol = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                     opt = JuMPOptimiser(; pe = pr, slv = tslv)))
            @test isapprox(JuMP.value(sol.model[:risk]), expected_risk(r, sol.w, X);
                           atol = 1e-12)
            @test isapprox(JuMP.value(sol.model[:risk]), JuMP.value(tsol.model[:risk]);
                           atol = 1e-12)
        end
        # The constant is the gross bound times the largest spread per unit of weight.
        c = [zeros(1, 4); cumsum(X; dims = 1)]
        d_ret = maximum(maximum(X; dims = 1) - minimum(X; dims = 1))
        d_dd = maximum(maximum(c; dims = 1) - minimum(c; dims = 1))
        bm(model, alg, b = nothing) = PO.mip_big_m(model, b, 1e-5, alg, pr)
        ret, dd = PO.NetReturnsRiskSeries(), PO.DrawdownRiskSeries()
        mr(opt; obj = MinimumRisk(), r = ValueatRisk()) = optimise(MeanRisk(; r = r,
                                                                            obj = obj,
                                                                            opt = opt)).model
        m = mr(JuMPOptimiser(; pe = pr, slv = slv))
        @test m[:w_gross_ub] == 1.0
        @test bm(m, ret) == d_ret
        @test bm(m, dd) ≈ d_dd
        m = mr(JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1),
                             sbgt = 0.5))
        @test m[:w_gross_ub] == 2.0
        @test bm(m, ret) == 2 * d_ret
        # A stated constant is kept, and it must exceed the slack.
        @test bm(m, ret, 0.5) == 0.5
        @test_throws DomainError bm(m, ret, 1e-6)
        # A free scale of the weights has no bound, so the constant is Cajas's 1000.
        m = mr(JuMPOptimiser(; pe = pr, slv = slv); obj = MaximumRatio())
        @test bm(m, ret) == 1e3
        # A per period fee leaves the spread of the returns but not of the drawdowns.
        m = mr(JuMPOptimiser(; pe = pr, slv = slv, fees = Fees(; l = 1e-4)))
        @test PO.shared_has(m, :fees)
        @test bm(m, ret) == d_ret
        @test bm(m, dd) == 1e3
        # A fixed fee is one-time. On the first observation it changes the spread of the
        # returns, and amortised it charges every observation the same.
        m = mr(JuMPOptimiser(; pe = pr, slv = slv, fees = Fees(; fl = 1e-4)))
        @test PO.shared_has(m, :one_time_fees)
        @test !PO.shared_has(m, :fees)
        @test bm(m, ret) == 1e3
        # A drawdown adds up the fee of every period, so a fixed fee alone changes its
        # spread on either clock.
        @test bm(m, dd) == 1e3
        m = mr(JuMPOptimiser(; pe = pr, slv = slv,
                             fees = Fees(; fl = 1e-4, fa = AmortisedFees())))
        @test PO.shared_has(m, :one_time_fees)
        @test bm(m, ret) == d_ret
        @test bm(m, dd) == 1e3
        # A build on shifted weights does not read the bound of the head's weights.
        @test PO.mip_big_m(mr(JuMPOptimiser(; pe = pr, slv = slv)), nothing, 1e-5, ret, pr;
                           prefix = :shifted_) == 1e3
    end
    @testset "The constructors validate b, s and the weights" begin
        @test_throws DomainError MIPValueatRisk(; b = -1.0)
        @test_throws DomainError MIPValueatRisk(; s = 0.0)
        @test_throws DomainError MIPValueatRisk(; b = 1e-6, s = 1e-5)
        @test MIPValueatRisk(; b = 10.0, s = 1e-3).b == 10.0
        @test_throws DomainError DrawdownatRisk(; b = Inf)
        @test_throws DomainError DrawdownatRisk(; s = -1e-5)
        @test_throws DomainError DrawdownatRisk(; b = 1e-6, s = 1e-5)
        @test DrawdownatRisk(; b = 10.0, s = 1e-3).s == 1e-3
        for M in (ValueatRisk, ValueatRiskRange, DrawdownatRisk, RelativeDrawdownatRisk)
            @test_throws DomainError M(; w = pweights([1.0, -1.0]))
            @test_throws DomainError M(; alpha = 1.0)
        end
    end
    @testset "The models and the functors report one number" begin
        rng = StableRNG(1046)
        T, N = 40, 3
        X = 0.01 .* randn(rng, T, N) .+ 0.0005
        pr = prior(EmpiricalPrior(), X)
        # A tight integrality tolerance, so that the big-M constant leaves no row loose.
        slv = Solver(; name = :highs, solver = HiGHS.Optimizer,
                     settings = Dict("log_to_console" => false, "mip_rel_gap" => 0.0,
                                     "mip_feasibility_tolerance" => 1e-9))
        for alpha in (0.05, 0.1, 0.3),
            r in (ValueatRisk(; alpha = alpha), DrawdownatRisk(; alpha = alpha),
                  ValueatRiskRange(; alpha = alpha, beta = 0.2))

            sol = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                    opt = JuMPOptimiser(; pe = pr, slv = slv)))
            @test isapprox(JuMP.value(sol.model[:risk]), expected_risk(r, sol.w, X);
                           atol = 1e-7)
        end
        cslv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                      settings = Dict("verbose" => false))
        for dist in (Normal(), TDist(5), Laplace()), alpha in (0.05, 0.2)
            alg = DistributionValueatRisk(; dist = dist)
            for r in (ValueatRisk(; alpha = alpha, alg = alg),
                      ValueatRiskRange(; alpha = alpha, beta = 0.1, alg = alg))
                sol = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                        opt = JuMPOptimiser(; pe = pr, slv = cslv)))
                @test isapprox(JuMP.value(sol.model[:risk]),
                               expected_risk(factory(r, pr), sol.w, pr); rtol = 1e-6)
            end
        end
        # A negative coefficient of the standard deviation leaves the minimisation unbounded.
        for r in (ValueatRisk(; alpha = 0.7, alg = DistributionValueatRisk()),
                  ValueatRiskRange(; alpha = 0.3, beta = 0.8, alg = DistributionValueatRisk()))
            sol = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                    opt = JuMPOptimiser(; pe = pr, slv = cslv)))
            @test sol.retcode isa OptimisationFailure
            @test JuMP.termination_status(sol.model) == JuMP.DUAL_INFEASIBLE
        end
        # The parametric range is the spread of the two z-scores over one standard deviation,
        # and for a symmetric distribution the gain z-score is minus the loss z-score.
        w = fill(1 / N, N)
        sd = sqrt(dot(w, pr.sigma, w))
        r = factory(ValueatRiskRange(; alpha = 0.05, beta = 0.05,
                                     alg = DistributionValueatRisk(; dist = Laplace())), pr)
        @test isapprox(r(w), 2 * cquantile(Laplace(0.0, 1 / sqrt(2)), 0.05) * sd)
        # The parametric measures read the weights, not a return series.
        @test !PortfolioOptimisers.supports_precomputed_returns(r)
        @test !PortfolioOptimisers.supports_precomputed_returns(factory(ValueatRisk(;
                                                                                    alg = DistributionValueatRisk()),
                                                                        pr))
    end
end
