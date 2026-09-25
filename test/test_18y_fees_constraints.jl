using Test, PortfolioOptimisers, JuMP, Clarabel, HiGHS, StableRNGs, LinearAlgebra

# The fee builders of a JuMP model. Each testset solves a model and reads its entries, or the
# weights, against the closed form that the docstring of the builder states. The fixture is a
# small synthetic panel, so a MIP solve takes well under a second.

const PO = PortfolioOptimisers

X_fc = 0.01 * randn(StableRNG(42), 80, 6) .+ 0.001 * (1:6)'
pr_fc = prior(EmpiricalPrior(), X_fc)
T_fc, N_fc = size(X_fc)
hslv_fc = Solver(; name = :highs, solver = HiGHS.Optimizer,
                 settings = "log_to_console" => false,
                 check_sol = (; allow_local = true, allow_almost = true))
cslv_fc = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = "verbose" => false,
                 check_sol = (; allow_local = true, allow_almost = true))
function solve_fc(obj; r = ConditionalValueatRisk(), slv = hslv_fc, kw...)
    opt = JuMPOptimiser(; pe = pr_fc, slv = slv, kw...)
    return optimise(MeanRisk(; r = r, obj = obj, opt = opt))
end
entry_names_fc(res) = Set(keys(JuMP.object_dictionary(res.model)))
# CVaR of a return series, from its linear programme.
function cvar_fc(r, alpha)
    m = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(m)
    JuMP.@variable(m, t)
    JuMP.@variable(m, z[1:length(r)] >= 0)
    JuMP.@constraint(m, z .>= -r .- t)
    JuMP.@objective(m, Min, t + sum(z) / (alpha * length(r)))
    JuMP.optimize!(m)
    return JuMP.objective_value(m)
end

@testset "A fixed fee scales with k, and the ratio reaches its optimum" begin
    c = 0.002
    fees = Fees(; fl = c)
    alpha = ConditionalValueatRisk().alpha
    e1 = [1.0; zeros(T_fc - 1)]
    # The ratio the model states: the expected return pays the one-off fee spread over the
    # fit, and the CVaR reads the net series, whose first observation pays it whole.
    function ratio(w)
        C = c * count(>(1e-6), w)
        return (dot(pr_fc.mu, w) - C / T_fc) / cvar_fc(X_fc * w .- C .* e1, alpha)
    end
    # The optimum over every support. On a fixed support the fee is a constant, so each
    # support is one Charnes-Cooper linear programme.
    best = -Inf
    for mask in 1:(2 ^ N_fc - 1)
        S = [(mask >> (i - 1)) & 1 == 1 for i in 1:N_fc]
        C = c * sum(S)
        m = JuMP.Model(HiGHS.Optimizer)
        JuMP.set_silent(m)
        JuMP.@variable(m, y[1:N_fc] >= 0)
        JuMP.@variable(m, k >= 0)
        JuMP.@variable(m, t)
        JuMP.@variable(m, z[1:T_fc] >= 0)
        JuMP.@constraint(m, [i in 1:N_fc; !S[i]], y[i] == 0)
        JuMP.@constraint(m, sum(y) == k)
        JuMP.@constraint(m, z .>= -(X_fc * y .- C * k .* e1) .- t)
        JuMP.@constraint(m, t + sum(z) / (alpha * T_fc) <= 1)
        JuMP.@objective(m, Max, dot(pr_fc.mu, y) - C / T_fc * k)
        JuMP.optimize!(m)
        if JuMP.termination_status(m) == JuMP.MOI.OPTIMAL
            best = max(best, JuMP.objective_value(m))
        end
    end
    ws = map((1e-3, 1e-1, 1.0)) do ohf
        res = solve_fc(MaximumRatio(; ohf = ohf); fees = fees)
        @test isa(res.retcode, OptimisationSuccess)
        m = res.model
        # The fee is the rate times the gate, and the gate is k on each held position.
        k = JuMP.value(m[:k])
        @test isapprox(JuMP.value(m[:ffl]), c * k * sum(round.(JuMP.value.(m[:ib])));
                       rtol = 1e-6)
        @test isapprox(ratio(res.w), best; rtol = 1e-6)
        return res.w
    end
    # The weights do not depend on `ohf`.
    @test isapprox(ws[1], ws[2]; atol = 1e-6)
    @test isapprox(ws[1], ws[3]; atol = 1e-6)
    # The long-short bundle charges each side on its own gate.
    res = solve_fc(MaximumRatio(; ohf = 0.1); fees = Fees(; fl = 1e-4, fs = 2e-4),
                   wb = WeightBounds(; lb = -1.0, ub = 1.0), sbgt = 0.5)
    @test isa(res.retcode, OptimisationSuccess)
    m = res.model
    k = JuMP.value(m[:k])
    @test isapprox(JuMP.value(m[:ffl]), 1e-4 * k * sum(round.(JuMP.value.(m[:ilb])));
                   rtol = 1e-6)
    @test isapprox(JuMP.value(m[:ffs]), 2e-4 * k * sum(round.(JuMP.value.(m[:isb])));
                   rtol = 1e-6)
    # Under a constant k the gate is the binary itself.
    res = solve_fc(MinimumRisk(); fees = fees)
    @test isapprox(JuMP.value(res.model[:ffl]),
                   c * sum(round.(JuMP.value.(res.model[:ib]))))
end

@testset "A fixed fee alone reaches the net return series" begin
    for (fa, cf) in ((nothing, t -> t == 1 ? 1.0 : 0.0), (AmortisedFees(), t -> 1 / T_fc))
        res = solve_fc(MinimumRisk(); fees = Fees(; fl = 0.002, fa = fa))
        m = res.model
        @test !haskey(m, :fees)
        net = JuMP.value.(m[:net_X])
        gross = X_fc * JuMP.value.(m[:w])
        f_o = JuMP.value(m[:one_time_fees])
        @test f_o > 0
        @test all(isapprox(net[t], gross[t] - cf(t) * f_o; atol = 1e-12) for t in 1:T_fc)
        # A zero per period fee registers `:fees`, and it leaves the weights alone.
        res0 = solve_fc(MinimumRisk(); fees = Fees(; fl = 0.002, fa = fa, l = 0.0))
        @test haskey(res0.model, :fees)
        @test isapprox(res0.w, res.w; atol = 1e-6)
    end
end

@testset "The turnover fee is an epigraph, tight when the fee is read" begin
    wt = fill(1 / N_fc, N_fc)
    fees = Fees(; tn = Turnover(; w = wt, val = 0.01))
    res = solve_fc(MaximumReturn(); r = Variance(), slv = cslv_fc, fees = fees)
    @test isa(res.retcode, OptimisationSuccess)
    m = res.model
    @test all(n -> n in entry_names_fc(res), (:t_ftn, :x_ftn, :ftn, :cftn, :fees))
    x = JuMP.value.(m[:w]) .- wt .* JuMP.value(m[:k])
    @test isapprox(JuMP.value.(m[:x_ftn]), x; atol = 1e-10)
    @test isapprox(JuMP.value.(m[:t_ftn]), abs.(x); atol = 1e-7)
    @test isapprox(JuMP.value(m[:ftn]), 0.01 * sum(abs, x); atol = 1e-8)
    # A variance does not read the net series, so under a minimum risk objective nothing
    # pulls on the fee: a larger epigraph entry leaves the objective and the weights.
    res = solve_fc(MinimumRisk(); r = Variance(), slv = cslv_fc, fees = fees)
    m = res.model
    obj = JuMP.objective_value(m)
    w = JuMP.value.(m[:w])
    JuMP.set_lower_bound(m[:t_ftn][1], 1.0)
    JuMP.optimize!(m)
    @test isapprox(JuMP.objective_value(m), obj; rtol = 1e-5)
    @test isapprox(JuMP.value.(m[:w]), w; atol = 1e-4)
    @test JuMP.value(m[:ftn]) > 0.01 * sum(abs, w .- wt) + 1e-3
end

@testset "The proportional fees charge the parts of the weights" begin
    fees = Fees(; l = 0.001, s = 0.002)
    # Long only: the long part is the weights, and the model registers no short part.
    res = solve_fc(MaximumReturn(); slv = cslv_fc, fees = fees)
    @test haskey(res.model, :fl)
    @test !haskey(res.model, :fs)
    @test isapprox(JuMP.value(res.model[:fl]), 0.001 * sum(res.w); atol = 1e-9)
    # Long-short: the budget pins the sums of the parts. Without the pin the minimum risk
    # portfolio holds no short position and still pays both fees in full; with the pin the
    # fees agree with the weights.
    kw = (; wb = WeightBounds(; lb = -1.0, ub = 1.0), sbgt = 0.5, fees = fees)
    for (xbgt, gap_l, gap_s) in ((false, 0.0005, 0.001), (true, 0.0, 0.0))
        res = solve_fc(MinimumRisk(); slv = xbgt ? hslv_fc : cslv_fc, xbgt = xbgt, kw...)
        @test isa(res.retcode, OptimisationSuccess)
        m = res.model
        w = JuMP.value.(m[:w])
        @test isapprox(JuMP.value(m[:fl]) - 0.001 * sum(max.(w, 0)), gap_l; atol = 1e-8)
        @test isapprox(JuMP.value(m[:fs]) - 0.002 * sum(max.(-w, 0)), gap_s; atol = 1e-8)
    end
end

@testset "The liquidation charges are constants times k" begin
    lq = Turnover(; w = [0.1, -0.2, 0.0], val = [0.01, 0.02, 0.03])
    for k in (1, 2.5)
        m = JuMP.Model()
        m[:k] = k
        PO.set_liquidation_fees!(m, lq)
        PO.set_fixed_liquidation_fees!(m, lq, (; atol = 1e-8))
        # The proportional charge prices the traded amount, the fixed one each position sold.
        @test JuMP.value(m[:flq_prop]) ≈ (0.01 * 0.1 + 0.02 * 0.2) * k
        @test JuMP.value(m[:flq_fixed]) ≈ (0.01 + 0.02) * k
        @test JuMP.value(m[:fees]) ≈ JuMP.value(m[:flq_prop])
        @test JuMP.value(m[:one_time_fees]) ≈ JuMP.value(m[:flq_fixed])
        # A second term adds to the registered expression in place.
        JuMP.@variable(m, x)
        PO.add_to_fees!(m, 2x)
        PO.add_to_one_time_fees!(m, 3x)
        @test JuMP.coefficient(m[:fees], x) == 2
        @test JuMP.coefficient(m[:one_time_fees], x) == 3
        @test JuMP.constant(m[:fees]) ≈ (0.01 * 0.1 + 0.02 * 0.2) * k
    end
    m = JuMP.Model()
    @test isnothing(PO.set_liquidation_fees!(m, nothing))
    @test isnothing(PO.set_fixed_liquidation_fees!(m, nothing, (;)))
    @test !haskey(m, :fees)
    @test !haskey(m, :one_time_fees)
end

@testset "A fee refuses a rate of +Inf on a turnover carrier" begin
    tn = Turnover(; w = fill(1 / N_fc, N_fc), val = Inf)
    @test_throws DomainError Fees(; tn = tn)
    @test_throws DomainError Fees(; lq = tn)
    @test_throws DomainError Fees(; flq = tn)
    @test Fees(; tn = Turnover(; w = fill(1 / N_fc, N_fc), val = 0.01)).tn.val == 0.01
end
