# The quantile measures of `src/16_RiskMeasures/06_XatRisk/01_XatRisk.jl`, checked against the
# forms their docstrings state: every empirical functor against the definition of Cajas,
# Equation 7.50, and against an independent solve of the mixed-integer programme of Equation
# 7.51 with its slack added, the parametric z-scores against the quantiles of the unit-variance
# distributions, the models against the functors, and the validation of the constructors. The
# checks of #600 add the parametric quantile against the empirical one, the two drawdown paths
# in exact arithmetic, and the range as the sum of its tails when the two levels overlap.
using Clarabel, Distributions, HiGHS, JuMP

# The minimum of the mixed-integer programme over a loss series `l`, with a big-M constant that
# fits the losses and a tight integrality tolerance, so that no row is left loose. The solver
# accepts an indicator within its tolerance of zero as zero, which lowers `r` by up to `b` times
# that tolerance, so the minimum is read as the largest loss that no indicator exempts.
function mip_quantile(l, alpha, s = 1e-9, p = ones(length(l)))
    T = length(l)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    set_attribute(m, "mip_feasibility_tolerance", 1e-9)
    set_attribute(m, "mip_rel_gap", 0.0)
    b = 2 * (maximum(l) - minimum(l)) + 1
    @variable(m, r)
    @variable(m, z[1:T], Bin)
    @constraint(m, dot(p, z) <= (alpha + s) * sum(p))
    @constraint(m, r .>= l .- b .* z)
    @objective(m, Min, r)
    optimize!(m)
    return maximum(l[value.(z) .< 0.5])
end
# Equation 7.50, `-inf{x : F(x) > alpha}`, with `F` the weighted distribution function of the
# returns `x`, read off the sorted sample in exact rational arithmetic, so no slack is needed.
function upper_quantile(x, alpha, p = ones(Int, length(x)))
    o = sortperm(x)
    q = rationalize.(BigInt, p[o])
    F = cumsum(q) ./ sum(q)
    return x[o[findfirst(>(rationalize(BigInt, alpha)), F)]]
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
            # The functor is the definition, Equation 7.50, and the minimum of the programme.
            @test ValueatRisk(; alpha = alpha)(x) == -upper_quantile(x, alpha)
            @test ValueatRisk(; alpha = alpha, w = pw)(x) == -upper_quantile(x, alpha, p)
            @test same(ValueatRisk(; alpha = alpha)(x), mip_quantile(-x, alpha))
            @test same(ValueatRisk(; alpha = alpha, w = pw)(x),
                       mip_quantile(-x, alpha, 1e-9, p))
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
                           mip_quantile(-x, alpha, 1e-9, p) +
                           mip_quantile(x, beta, 1e-9, p))
            end
            @test same(DrawdownatRisk(; alpha = alpha)(x),
                       -upper_quantile(abs_dd(x), alpha))
            @test same(DrawdownatRisk(; alpha = alpha)(x), mip_quantile(-abs_dd(x), alpha))
            @test same(DrawdownatRisk(; alpha = alpha, w = pw)(x),
                       mip_quantile(-abs_dd(x), alpha, 1e-9, p))
            @test same(DrawdownatRisk(; alpha = alpha, s = 0.015)(x),
                       mip_quantile(-abs_dd(x), alpha, 0.015))
            @test same(RelativeDrawdownatRisk(; alpha = alpha)(x),
                       mip_quantile(-rel_dd(x), alpha))
            @test same(RelativeDrawdownatRisk(; alpha = alpha, w = pw)(x),
                       mip_quantile(-rel_dd(x), alpha, 1e-9, p))
            # The CVaR is the VaR plus the mean excess over it, so minus the VaR minimises the
            # objective of Rockafellar and Uryasev. The losses above the VaR hold alpha T of
            # the weight at an integer alpha T, so their mean is the CVaR (#1364).
            v = ValueatRisk(; alpha = alpha)(x)
            @test ConditionalValueatRisk(; alpha = alpha)(x) ≈
                  v + sum(max.(-x .- v, 0)) / (alpha * T)
            vw = ValueatRisk(; alpha = alpha, w = pw)(x)
            @test ConditionalValueatRisk(; alpha = alpha, w = pw)(x) ≈
                  vw + dot(p, max.(-x .- vw, 0)) / (alpha * sum(p))
            vd = DrawdownatRisk(; alpha = alpha)(x)
            @test ConditionalDrawdownatRisk(; alpha = alpha)(x) ≈
                  vd + sum(max.(-abs_dd(x) .- vd, 0)) / (alpha * T)
            # The mean excess is never negative, so the tail mean is never below the
            # quantile at the same level (#601).
            @test v <= ConditionalValueatRisk(; alpha = alpha)(x)
            @test vw <= ConditionalValueatRisk(; alpha = alpha, w = pw)(x)
            @test vd <= ConditionalDrawdownatRisk(; alpha = alpha)(x)
            @test RelativeDrawdownatRisk(; alpha = alpha)(x) <=
                  RelativeConditionalDrawdownatRisk(; alpha = alpha)(x)
            if isinteger(round(alpha * T; digits = 8))
                @test ConditionalValueatRisk(; alpha = alpha)(x) ≈ mean(-x[-x .> v])
            end
            # No functor writes its input or the weights it holds.
            @test x == x0
            @test p == p0
        end
    end
    @testset "Without weights the index is the floor of alpha T plus one" begin
        # `alpha * T` rounds below the integer at 0.29 * 100, and the slack absorbs it, so every
        # level on a grid of hundredths selects the order statistic after the one that its
        # decimal value names.
        x = collect(range(-0.05, 0.05; length = 100)) .+ 0.0001 .* sin.(1:100)
        s = sort(x)
        for j in 1:99
            @test ValueatRisk(; alpha = j / 100)(x) == -s[j + 1]
        end
        @test 0.29 * 100 < 29
        @test 0.07 * 100 > 7
        # A fractional alpha T takes the next order statistic.
        x = collect(range(-0.05, 0.05; length = 37))
        @test ValueatRisk(; alpha = 0.1)(x) == -sort(x)[4]
        # At alpha T = 1 the measure is the second largest loss, so it stays below the tail
        # mean, which is the largest loss (#1364).
        x = collect(range(-0.05, 0.05; length = 100))
        @test ValueatRisk(; alpha = 0.01)(x) == -sort(x)[2]
        @test ValueatRisk(; alpha = 0.01)(x) < ConditionalValueatRisk(; alpha = 0.01)(x)
        # A slack of the old default moves the index at a non-integer alpha T once s T >= 1.
        x = collect(range(-1.0, 1.0; length = 100_000))
        @test ValueatRisk(; alpha = 0.050005)(x) == -sort(x)[5001]
        @test ValueatRisk(; alpha = 0.050005, alg = MIPValueatRisk(; s = 1e-5))(x) ==
              -sort(x)[5002]
        # The absolute and relative drawdowns follow their definitions.
        x = [0.01, -0.02, 0.005, -0.03, 0.04, -0.01]
        @test PortfolioOptimisers.absolute_drawdown_vec(x) ≈ abs_dd(x)
        @test PortfolioOptimisers.relative_drawdown_vec(x) ≈ rel_dd(x)
        @test PortfolioOptimisers.absolute_drawdown_vec(1:3) == zeros(Int, 3)
    end
    @testset "The defaults of the big-M constant and the slack" begin
        PO = PortfolioOptimisers
        # A `nothing` constant passes on, for the builder to derive from the data.
        @test PO.mip_var_bounds(nothing, nothing) === (nothing, 1e-9)
        @test PO.mip_var_bounds(2.0, nothing) == (2.0, 1e-9)
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
        # On 100 rows the levels exempt the 28 and 9 observations of that measurement. A range
        # that exempts 10 gain observations, as beta = 0.1 does on 100 rows, takes HiGHS twice
        # as long (#1364). On the 40 rows of the range the levels exempt 11 and 3.
        #
        # The range solves a draw of 40 rows from the same seed. On the 100 rows its two solves
        # took about 490 s (#1634), and on 40 rows each takes about 1 s. The draw still guards
        # #1323. Measured: `MIPValueatRisk(; b = 1000)` at the default tolerance leaves the model
        # 7.6e-4 below the functor and 3.5e-4 below the solve at the tight tolerance, and the
        # derived constant leaves 1.8e-17 and 1.0e-17.
        X40 = 0.01 .* randn(StableRNG(1046), 40, 4) .+ 0.0005
        pr40 = prior(EmpiricalPrior(), X40)
        for (r, Xr, prr) in ((ValueatRisk(; alpha = 0.285), X, pr),
                             (DrawdownatRisk(; alpha = 0.285), X, pr),
                             (ValueatRiskRange(; alpha = 0.285, beta = 0.095), X40, pr40))
            sol = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                    opt = JuMPOptimiser(; pe = prr, slv = slv)))
            tsol = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                     opt = JuMPOptimiser(; pe = prr, slv = tslv)))
            @test isapprox(JuMP.value(sol.model[:risk]), expected_risk(r, sol.w, Xr);
                           atol = 1e-12)
            @test isapprox(JuMP.value(sol.model[:risk]), JuMP.value(tsol.model[:risk]);
                           atol = 1e-12)
        end
        # The control: the old constant misses on the draw of 40 rows, so the draw still guards
        # #1323.
        r = ValueatRiskRange(; alpha = 0.285, beta = 0.095,
                             alg = MIPValueatRisk(; b = 1000))
        sol = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                opt = JuMPOptimiser(; pe = pr40, slv = slv)))
        @test abs(JuMP.value(sol.model[:risk]) - expected_risk(r, sol.w, X40)) > 1e-5
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
    @testset "The gain tail of a generic range reads the law of the negated loss (#1822)" begin
        PO = PortfolioOptimisers
        # The rebuild negates the mean of a `Normal`, and every other law passes through.
        @test PO.gain_tail_distribution(Normal(0.1, 2.0)) == Normal(-0.1, 2.0)
        for dist in (Normal(), Normal(0.0, 2.0), TDist(5), Laplace(), Laplace(0.1, 2.0))
            @test PO.gain_tail_distribution(dist) === dist
        end
        # The loss quantile of the reflected law is the negated gain quantile of the law.
        for dist in (Normal(), Normal(0.1, 2.0), TDist(5), Laplace(), Laplace(0.1, 2.0)),
            alpha in (0.05, 0.2)

            @test isapprox(PO.compute_value_at_risk_z(PO.gain_tail_distribution(dist),
                                                      alpha),
                           -PO.compute_value_at_risk_cz(dist, alpha))
        end
        # The reproduction of #1822. A `Normal` with a nonzero mean made the two ranges differ
        # by twice the mean times the standard deviation of the portfolio.
        rng = StableRNG(5)
        X = 0.01 .* randn(rng, 300, 6) .+ range(-0.0005, 0.002; length = 6)'
        pr = prior(EmpiricalPrior(), X)
        w = fill(1 / 6, 6)
        sd = sqrt(dot(w, pr.sigma, w))
        for dist in (Normal(), Normal(0.1, 2.0), Laplace(), Laplace(0.1, 2.0), TDist(5)),
            (alpha, beta) in ((0.05, 0.05), (0.05, 0.2))

            alg = DistributionValueatRisk(; dist = dist)
            gen = factory(GenericValueatRiskRange(;
                                                  loss = ValueatRisk(; alpha = alpha,
                                                                     alg = alg),
                                                  gain = ValueatRisk(; alpha = beta,
                                                                     alg = alg)), pr)
            var = factory(ValueatRiskRange(; alpha = alpha, beta = beta, alg = alg), pr)
            @test isapprox(expected_risk(gen, w, pr), expected_risk(var, w, pr))
        end
        # The gain tail of the non-centred law is `dot(mu, w) - zc * sd`, the gain tail of
        # `ValueatRiskRange`, not `dot(mu, w) + z * sd`.
        dist = Normal(0.1, 2.0)
        vd = ValueatRisk(; alg = DistributionValueatRisk(; dist = dist))
        g = factory(GenericValueatRiskRange(; loss = ConditionalValueatRisk(), gain = vd),
                    pr)
        zc = PO.compute_value_at_risk_cz(dist, 0.05)
        @test isapprox(g(w, pr.X),
                       ConditionalValueatRisk()(pr.X * w) + dot(pr.mu, w) - zc * sd)
        # The rebuild of the gain tail keeps a symmetric law and every returns tail as they are.
        vs = factory(ValueatRisk(; alg = DistributionValueatRisk()), pr)
        @test PO.gain_tail_measure(vs) === vs
        cv = ConditionalValueatRisk()
        @test PO.gain_tail_measure(cv) === cv
        @test PO.gain_tail_measure(vd).alg.dist == Normal(-0.1, 2.0)
        # The model builds the same gain tail, so its risk equals the functor and the range.
        cslv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                      settings = Dict("verbose" => false))
        opt = JuMPOptimiser(; pe = pr, slv = cslv)
        for r in (GenericValueatRiskRange(; loss = vd, gain = vd),
                  GenericValueatRiskRange(; loss = ConditionalValueatRisk(), gain = vd))
            sol = optimise(MeanRisk(; r = r, obj = MinimumRisk(), opt = opt))
            @test isa(sol.retcode, OptimisationSuccess)
            @test isapprox(JuMP.value(sol.model[:risk]),
                           expected_risk(factory(r, pr), sol.w, pr); rtol = 1e-6)
        end
        sol = optimise(MeanRisk(; r = GenericValueatRiskRange(; loss = vd, gain = vd),
                                obj = MinimumRisk(), opt = opt))
        @test isapprox(JuMP.value(sol.model[:risk]),
                       expected_risk(factory(ValueatRiskRange(; alg = vd.alg), pr), sol.w,
                                     pr); rtol = 1e-6)
    end
    @testset "Two parametric tails of a generic range share one cone (#1901)" begin
        rng = StableRNG(5)
        X = 0.01 .* randn(rng, 300, 6) .+ range(-0.0005, 0.002; length = 6)'
        pr = prior(EmpiricalPrior(), X)
        cslv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                      settings = Dict("verbose" => false))
        opt = JuMPOptimiser(; pe = pr, slv = cslv)
        nsoc(m) = JuMP.num_constraints(m, Vector{JuMP.AffExpr}, JuMP.MOI.SecondOrderCone)
        # A mean far from zero gives one tail a negative coefficient. With a cone for each
        # tail the minimisation was unbounded, and one cone makes it the minimum of the range.
        for dist in (Normal(2.0, 1.0), Normal(-2.0, 1.0), Normal(), TDist(5))
            vd = ValueatRisk(; alg = DistributionValueatRisk(; dist = dist))
            r = GenericValueatRiskRange(; loss = vd, gain = vd)
            sol = optimise(MeanRisk(; r = r, obj = MinimumRisk(), opt = opt))
            @test isa(sol.retcode, OptimisationSuccess)
            @test nsoc(sol.model) == 1
            @test isapprox(JuMP.value(sol.model[:risk]),
                           expected_risk(factory(r, pr), sol.w, pr); rtol = 1e-6)
            sv = optimise(MeanRisk(; r = ValueatRiskRange(; alg = vd.alg),
                                   obj = MinimumRisk(), opt = opt))
            @test isapprox(JuMP.value(sol.model[:risk]), JuMP.value(sv.model[:risk]);
                           rtol = 1e-6)
        end
        # Tails that read different moments keep a cone each.
        dist = Normal(0.5, 1.0)
        a1 = DistributionValueatRisk(; dist = dist)
        a2 = DistributionValueatRisk(; dist = dist, sigma = 2 .* pr.sigma)
        for (alg, n) in
            ((a1, 2), (DistributionValueatRisk(; dist = dist, sigma = 2 .* pr.sigma), 1))
            r = GenericValueatRiskRange(; loss = ValueatRisk(; alpha = 0.05, alg = alg),
                                        gain = ValueatRisk(; alpha = 0.1, alg = a2))
            sol = optimise(MeanRisk(; r = r, obj = MinimumRisk(), opt = opt))
            @test isa(sol.retcode, OptimisationSuccess)
            @test nsoc(sol.model) == n
            @test isapprox(JuMP.value(sol.model[:risk]),
                           expected_risk(factory(r, pr), sol.w, pr); rtol = 1e-6)
        end
        # The bound of a tail still bounds the value of that tail.
        vd = ValueatRisk(; alg = DistributionValueatRisk(; dist = Normal(2.0, 1.0)))
        r = GenericValueatRiskRange(; loss = vd, gain = vd)
        sol = optimise(MeanRisk(; r = r, obj = MaximumReturn(), opt = opt))
        ub = expected_risk(factory(vd, pr), sol.w, pr) / 2
        vb = ValueatRisk(; alg = vd.alg, settings = RiskMeasureSettings(; ub = ub))
        sol = optimise(MeanRisk(; r = GenericValueatRiskRange(; loss = vb, gain = vd),
                                obj = MaximumReturn(), opt = opt))
        @test isa(sol.retcode, OptimisationSuccess)
        @test isapprox(expected_risk(factory(vd, pr), sol.w, pr), ub; rtol = 1e-6)
    end
    @testset "The parametric quantile against the empirical one" begin
        # The parametric functor of one asset at unit weight, with the sample moments.
        function par(x, alpha)
            r = ValueatRisk(; alpha = alpha, alg = DistributionValueatRisk())
            return factory(r, prior(EmpiricalPrior(), reshape(x, :, 1)))([1.0])
        end
        emp(x, alpha) = ValueatRisk(; alpha = alpha)(x)
        rng = StableRNG(600)
        T = 200_000
        # On a normal sample the two estimands agree to the sampling error.
        xn = 0.01 .* randn(rng, T) .+ 0.0005
        for alpha in (0.01, 0.05, 0.1, 0.3)
            @test isapprox(par(xn, alpha), emp(xn, alpha); rtol = 2e-2)
        end
        # An exponential loss tail is heavier than the normal one, so at a small alpha the
        # parametric quantile understates the loss. The mirrored sample has a light loss tail,
        # so the parametric quantile overstates it.
        xl = 0.01 .* (1 .- rand(rng, Exponential(1.0), T))
        for alpha in (0.01, 0.05)
            @test par(xl, alpha) < 0.9 * emp(xl, alpha)
            @test par(-xl, alpha) > 1.5 * emp(-xl, alpha)
        end
        # At a larger alpha the order reverses.
        @test par(xl, 0.3) > emp(xl, 0.3)
    end
    @testset "The two drawdown paths in exact arithmetic" begin
        PO = PortfolioOptimisers
        # c = (1/10, -1/10, -1/20, 1/20, -1/20) with the peak 1/10 from the first period, and
        # C = (11/10, 22/25, 231/250, 2541/2500, 22869/25000) with the peak 11/10.
        x = [1 // 10, -2 // 10, 1 // 20, 1 // 10, -1 // 10]
        @test PO.absolute_drawdown_vec(x) == [0, -1 // 5, -3 // 20, -1 // 20, -3 // 20]
        @test PO.relative_drawdown_vec(x) ==
              [0, -1 // 5, -4 // 25, -19 // 250, -421 // 2500]
        @test eltype(PO.relative_drawdown_vec(x)) == Rational{Int}
        # At alpha = 0.2 and T = 5 the index is 2, the second deepest drawdown.
        @test DrawdownatRisk(; alpha = 0.2)(Float64.(x)) ≈ 0.15
        @test RelativeDrawdownatRisk(; alpha = 0.2)(Float64.(x)) ≈ 0.1684
        # A path that never falls has no drawdown on either path.
        up = [1 // 100, 0 // 1, 2 // 100]
        @test PO.absolute_drawdown_vec(up) == zeros(Rational{Int}, 3)
        @test PO.relative_drawdown_vec(up) == zeros(Rational{Int}, 3)
        @test iszero(DrawdownatRisk()(Float64.(up)))
        @test iszero(RelativeDrawdownatRisk()(Float64.(up)))
        # The peaks start at c_0 = 0 and C_0 = 1, so a fall in the first period is a drawdown.
        down = [-1 // 10, 1 // 20]
        @test PO.absolute_drawdown_vec(down) == [-1 // 10, -1 // 20]
        @test PO.relative_drawdown_vec(down) == [-1 // 10, -11 // 200]
    end
    @testset "The range is the sum of its two tails, and overlapping levels make it negative" begin
        PO = PortfolioOptimisers
        x = [0.03, -0.01, 0.02, -0.04, 0.05, 0.0, -0.02, 0.01, -0.03, 0.04]
        xs = sort(x)
        pw = pweights(collect(1.0:10.0))
        sum_of_tails(r, x) = (t = PO.range_tails(r); t.loss(x) + t.gain(-x))
        # Without weights the range is x_(T + 1 - k_beta) - x_(k_alpha), with T = 10.
        for (alpha, beta, v) in
            ((0.05, 0.2, xs[8] - xs[1]), (0.45, 0.45, xs[6] - xs[5]), (0.5, 0.45, 0.0),
             (0.5, 0.5, xs[5] - xs[6]), (0.6, 0.6, xs[4] - xs[7]))
            r = ValueatRiskRange(; alpha = alpha, beta = beta)
            rw = ValueatRiskRange(; alpha = alpha, beta = beta, w = pw)
            @test r(x) == v
            @test r(x) == sum_of_tails(r, x)
            @test rw(x) == sum_of_tails(rw, x)
        end
        # With weights k_alpha = 5 and k_beta = 6 at alpha = beta = 0.5, so k_alpha + k_beta
        # is T + 1 and the two tails select the same return.
        @test ValueatRiskRange(; alpha = 0.5, w = pw)(x) == 0.0
        # The model of overlapping levels is still the order statistic of the functor.
        rng = StableRNG(600)
        X = 0.01 .* randn(rng, 20, 3) .+ 0.0005
        pr = prior(EmpiricalPrior(), X)
        slv = Solver(; name = :highs, solver = HiGHS.Optimizer,
                     settings = Dict("log_to_console" => false, "mip_rel_gap" => 0.0,
                                     "mip_feasibility_tolerance" => 1e-9))
        for (alpha, beta) in ((0.6, 0.6), (0.3, 0.8))
            r = ValueatRiskRange(; alpha = alpha, beta = beta)
            sol = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                    opt = JuMPOptimiser(; pe = pr, slv = slv)))
            @test JuMP.value(sol.model[:risk]) < 0
            @test isapprox(JuMP.value(sol.model[:risk]), expected_risk(r, sol.w, X);
                           atol = 1e-12)
        end
    end
end
