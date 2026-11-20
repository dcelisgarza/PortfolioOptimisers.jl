# The sweep of the Worst-Case Distribution (#1836). Each distribution is checked against a
# reference that shares no code with the library: the closed forms of the EVaR and of the
# PNVaR in `BigFloat`, and the dual programme of the RLVaR. The gradient of a measure is
# checked against a `BigFloat` finite difference of its closed form, which has no solver noise.
# Two defects: the risk contributions of a `GenericValueatRiskRange` differenced two solves, and
# the gradient of a tracking measure of a `RiskRatio` raised under fees.
using Clarabel, JuMP, LinearAlgebra, StableRNGs, StatsBase

const PO = PortfolioOptimisers

const _X1836 = randn(StableRNG(1836), 120, 4) * 0.01 .+ 0.0004
const _w1836 = [0.4, 0.3, 0.2, 0.1]
const _x1836 = _X1836 * _w1836
const _pr1836 = prior(EmpiricalPrior(), _X1836)
const _slv1836 = Solver(; solver = Clarabel.Optimizer, settings = Dict("verbose" => false))
const _pe1836 = fill(inv(120), 120)

# The minimum of a convex function of one variable, by golden-section search in `BigFloat`.
function _golden1836(f, a, b)
    g = (sqrt(big(5)) - 1) / 2
    for _ in 1:200
        c = b - g * (b - a)
        d = a + g * (b - a)
        if f(c) < f(d)
            b = d
        else
            a = c
        end
    end
    return (a + b) / 2
end
# EVaR = min_z z ln(sum p exp(L / z) / alpha), and its distribution is p exp(L / z*) normalised.
function _evar1836(x, alpha, p)
    L = -big.(x)
    p = big.(p)
    f(lz) = (z = exp(lz); m = maximum(L / z);
             z * (m + log(sum(p .* exp.(L / z .- m))) - log(big(alpha))))
    lz = _golden1836(f, big(-20), big(5))
    e = p .* exp.(L / exp(lz) .- maximum(L / exp(lz)))
    return f(lz), e / sum(e)
end
# PNVaR = min_eta eta + (sum p (L - eta)_+^p)^(1/p) / alpha, and its distribution is
# p (L - eta*)_+^(p - 1) normalised.
function _pnvar1836(x, alpha, pw, p)
    L = -big.(x)
    p = big.(p)
    pw = big(pw)
    f(eta) = eta + (sum(p .* max.(L .- eta, 0) .^ pw))^(1 / pw) / big(alpha)
    eta = _golden1836(f, minimum(L) - 1, maximum(L))
    e = p .* max.(L .- eta, 0) .^ (pw - 1)
    return f(eta), e / sum(e)
end
# The RLVaR is the largest expected loss over the Kaniadakis ball, solved here for `q` as its
# own power cone model, as in `test_18v`.
function _rlvar1836(x, alpha, kappa, p)
    T = length(x)
    c = p * T
    u = inv(alpha * T)
    lnk = (u^kappa - u^(-kappa)) / (2 * kappa)
    m = JuMP.Model(Clarabel.Optimizer)
    JuMP.set_silent(m)
    JuMP.@variable(m, q[1:T] >= 0)
    JuMP.@variable(m, nu[1:T])
    JuMP.@variable(m, tau[1:T])
    JuMP.@constraint(m, sum(q) == 1)
    JuMP.@constraint(m, [i = 1:T],
                     [nu[i], c[i], q[i]] in JuMP.MOI.PowerCone(inv(1 + kappa)))
    JuMP.@constraint(m, [i = 1:T], [q[i], c[i], tau[i]] in JuMP.MOI.PowerCone(1 - kappa))
    JuMP.@constraint(m, sum(nu - tau) / (2 * kappa) <= lnk)
    JuMP.@objective(m, Max, -dot(q, x))
    JuMP.optimize!(m)
    return JuMP.value.(q)
end
# The two-sided finite difference of a closed form in `BigFloat`.
function _bfd1836(f, w)
    h = big"1e-30"
    wb = big.(w)
    return [(f(wb + h * e) - f(wb - h * e)) / (2h)
            for e in eachcol(Matrix{BigFloat}(I, 4, 4))]
end
function _dd1836(x)
    return (c = [zero(eltype(x)); cumsum(x)];
            [c[t] - maximum(view(c, 1:t)) for t in 2:length(c)])
end

@testset "Each Worst-Case Distribution matches an independent reference" begin
    x = _x1836
    ow = StatsBase.Weights([isodd(t) ? 1.0 + t / 120 : 0.5 for t in 1:120])
    # Every third observation has weight zero, so its row leaves the programme.
    ow0 = StatsBase.Weights([t % 3 == 0 ? 0.0 : 1.0 for t in 1:120])
    for (o, p) in ((nothing, _pe1836), (ow, ow / sum(ow)), (ow0, ow0 / sum(ow0)))
        v, q = _evar1836(x, 0.05, p)
        @test isapprox(PO.ERM_worst_case(x, _slv1836, 0.05, o), q; rtol = 5e-4)
        # The value of a solve meets the closed form to the gap of the solver, about 1e-8.
        @test isapprox(PO.ERM(x, _slv1836, 0.05, o), v; rtol = 1e-5)
        v, q = _pnvar1836(x, 0.2, 1.5, p)
        @test isapprox(PO.PRM_worst_case(x, _slv1836, 0.2, 1.5, o), q; rtol = 5e-4)
        @test isapprox(PO.PRM(x, _slv1836, 0.2, 1.5, o), v; rtol = 1e-5)
        q = _rlvar1836(x, 0.05, 0.3, p)
        @test isapprox(PO.RRM_worst_case(x, _slv1836, 0.05, 0.3, o), q; rtol = 1e-4)
    end
    # The largest loss of positive weight takes the mass. The second observation has the
    # largest loss of all, but weight zero.
    @test PO.worst_loss_distribution([-0.2, -0.3, 0.1, -0.2], [1, 0, 2, 3]) ==
          [0.25, 0, 0, 0.75]
end

@testset "The gradient is the expected loss under the Worst-Case Distribution" begin
    X = big.(_X1836)
    for (r, f) in ((EntropicValueatRisk(), v -> _evar1836(X * v, 0.05, _pe1836)[1]),
                   (PowerNormValueatRisk(; alpha = 0.2, p = 1.5),
                    v -> _pnvar1836(X * v, 0.2, 1.5, _pe1836)[1]),
                   (EntropicDrawdownatRisk(), v -> _evar1836(_dd1836(X * v), 0.05, _pe1836)[1]),
                   (EntropicValueatRiskRange(; alpha = 0.05, beta = 0.1),
                    v -> _evar1836(X * v, 0.05, _pe1836)[1] + _evar1836(-X * v, 0.1, _pe1836)[1]))
        r = factory(r, _pr1836, _slv1836)
        @test isapprox(risk_gradient(r, _w1836, _pr1836), Float64.(_bfd1836(f, _w1836));
                       rtol = 1e-4)
        q = PO.worst_case_distribution(r, _w1836, _X1836, nothing)
        @test isapprox(PO.linearised_risk(q, r, _w1836, _X1836, nothing),
                       expected_risk(r, _w1836, _X1836); rtol = 1e-6)
    end
end

@testset "A generic range differentiates the distribution of each tail" begin
    ratio(r) = sum(risk_contribution(r, _w1836, _pr1836)) /
               expected_risk(r, _w1836, _pr1836)
    # Before the fix, these sums were 0.971 and 0.994 of the risk.
    for r in (GenericValueatRiskRange(; loss = EntropicValueatRisk(),
                                      gain = EntropicValueatRisk(; alpha = 0.1)),
              GenericValueatRiskRange(; loss = PowerNormValueatRisk(; alpha = 0.2, p = 1.5),
                                      gain = ConditionalValueatRisk()))
        @test isapprox(ratio(factory(r, _pr1836, _slv1836)), 1; rtol = 1e-6)
    end
    # A tail with no solve keeps its own value, so two such tails keep the finite difference of
    # the range itself.
    r = GenericValueatRiskRange()
    @test PO.worst_case_distribution(r, _x1836) == (nothing, nothing)
    @test risk_gradient(r, _w1836, _pr1836) ==
          PO.finite_difference_gradient(v -> expected_risk(r, v, _pr1836), _w1836, 1e-6)
end

@testset "A tracking measure of a composite reads the distributions under fees" begin
    fees = Fees(; l = fill(0.0004, 4))
    wb = fill(0.25, 4)
    wb2 = [0.4, 0.2, 0.2, 0.2]
    ev = factory(EntropicValueatRisk(), _pr1836, _slv1836)
    sd = factory(StandardDeviation(), _pr1836)
    # Before the fix, the ratio raised an `ArgumentError` that named `risk_input_kind`.
    for inner in (RiskRatio(; r1 = ev, r2 = sd),
                  RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb2), r = ev),
                  RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb2), r = ev,
                                          alg = DependentVariableTracking()))
        t = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb), r = inner)
        q = PO.worst_case_distribution(t, _w1836, _X1836, fees)
        @test q[1] isa Tuple
        @test isapprox(PO.linearised_risk(q, t, _w1836, _X1836, fees),
                       expected_risk(t, _w1836, _X1836, fees); rtol = 1e-6)
        gfd = PO.finite_difference_gradient(v -> expected_risk(t, v, _pr1836, fees), _w1836,
                                            1e-3)
        @test isapprox(risk_gradient(t, _w1836, _pr1836, fees), gfd; rtol = 5e-3)
    end
end
