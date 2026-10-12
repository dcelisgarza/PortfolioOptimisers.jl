# The quantile measures of `src/16_RiskMeasures/06_XatRisk/01_XatRisk.jl`, checked against the
# forms their docstrings state: every empirical functor against the definition of Cajas,
# Equation 7.50, and against an independent solve of the mixed-integer programme of Equation
# 7.51 with its slack added, the parametric z-scores against the quantiles of the unit-variance
# distributions, the models against the functors, and the validation of the constructors.
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

# The range case of the big-M testset of `test_09n_xat_risk.jl`. Its two solves take most of
# the time of that file, so they run in a file of their own.
@testset "Quantile measures" begin
    @testset "The derived big-M constant keeps the programme exact at the default tolerance, the range" begin
        PO = PortfolioOptimisers
        rng = StableRNG(1046)
        X = 0.01 .* randn(rng, 100, 4) .+ 0.0005
        pr = prior(EmpiricalPrior(), X)
        opts = Dict("log_to_console" => false, "mip_rel_gap" => 0.0)
        slv = Solver(; name = :highs, solver = HiGHS.Optimizer, settings = opts)
        tslv = Solver(; name = :highs, solver = HiGHS.Optimizer,
                      settings = merge(opts, Dict("mip_feasibility_tolerance" => 1e-9)))
        # With b = 1000 the default tolerance of 1e-6 left these three about 1e-4 apart (#1323).
        # The levels exempt the 28 and 9 observations of that measurement. A range that exempts
        # 10 gain observations, as beta = 0.1 now does, takes HiGHS twice as long (#1364).
        r = ValueatRiskRange(; alpha = 0.285, beta = 0.095)
        let
            sol = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                    opt = JuMPOptimiser(; pe = pr, slv = slv)))
            tsol = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                     opt = JuMPOptimiser(; pe = pr, slv = tslv)))
            @test isapprox(JuMP.value(sol.model[:risk]), expected_risk(r, sol.w, X);
                           atol = 1e-12)
            @test isapprox(JuMP.value(sol.model[:risk]), JuMP.value(tsol.model[:risk]);
                           atol = 1e-12)
        end
    end
end
