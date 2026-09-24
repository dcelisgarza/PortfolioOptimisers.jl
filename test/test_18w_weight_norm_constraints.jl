using Test, PortfolioOptimisers, JuMP, Clarabel, StableRNGs, LinearAlgebra

# The weight norm ceilings and the dual norm epigraph. Each testset solves a model and reads
# the reported weights, or the model entries, against the closed form that the docstring of the
# builder states. The fixture is a small synthetic panel so the file runs in seconds.

const PO = PortfolioOptimisers

X_wn = 0.01 * randn(StableRNG(42), 150, 8) .+ 0.001 * (1:8)'
pr_wn = prior(EmpiricalPrior(), X_wn)
slv_wn = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                check_sol = (; allow_local = true, allow_almost = true),
                settings = "verbose" => false)
function solve_wn(head, kw; bgt = 1)
    opt = JuMPOptimiser(; pe = pr_wn, slv = slv_wn, bgt = bgt, kw...)
    est = head === :rb ? RiskBudgeting(; opt = opt) : MeanRisk(; obj = head, opt = opt)
    return optimise(est)
end
entry_names(res) = Set(keys(JuMP.object_dictionary(res.model)))

@testset "Weight norm ceilings bind on the reported weights" begin
    cases = ((; l2c = 0.4), x -> norm(x, 2), 0.4, (:l2c, :cl2c_soc, :cl2c)),
            ((; lpc = LpRegularisation(; p = 3, val = 0.3)), x -> norm(x, 3), 0.3,
             (:t_lpc_1, :r_lpc_1, :clpc_1, :cslpc_1, :clpc_bnd_1)),
            ((; linfc = 0.2), x -> norm(x, Inf), 0.2, (:t_linfc, :clinfc_nic, :clinfc))
    for (kw, f, val, names) in cases
        # Without the ceiling both objectives sit above it, so the ceiling binds.
        for obj in (MaximumReturn(), MaximumRatio())
            @test f(solve_wn(obj, (;)).w) > val + 1e-3
            res = solve_wn(obj, kw)
            @test isa(res.retcode, OptimisationSuccess)
            # MaximumRatio stops about 1.5e-5 below the ceiling, on the feasible side.
            @test f(res.w) <= val * (1 + 1e-6)
            @test isapprox(f(res.w), val; rtol = 1e-4)
            @test all(n -> n in entry_names(res), names)
        end
        # Under MaximumRatio `k` is not one, and the ceiling still holds on `w / k`.
        @test !isapprox(JuMP.value(solve_wn(MaximumRatio(), kw).model[:k]), 1; atol = 1e-2)
    end
end

@testset "The budget does not scale a weight norm ceiling" begin
    res = solve_wn(MaximumReturn(), (; l2c = 0.4); bgt = 0.5)
    @test isapprox(sum(res.w), 0.5; rtol = 1e-6)
    @test isapprox(norm(res.w, 2), 0.4; rtol = 1e-5)
end

@testset "RiskBudgeting: a weight norm ceiling holds on the renormalised weights" begin
    res = solve_wn(:rb, (; linfc = 0.14))
    @test JuMP.value(res.model[:k]) > 2
    @test isapprox(sum(res.w), 1; rtol = 1e-6)
    @test isapprox(norm(res.w, Inf), 0.14; rtol = 1e-5)
    # At the floor `N^(1/p - 1)` the only admitted weights are the equal weights.
    res = solve_wn(:rb, (; lpc = LpRegularisation(; p = 3, val = 8^(1 / 3 - 1))))
    @test isapprox(res.w, fill(1 / 8, 8); rtol = 1e-4)
end

@testset "A weight norm ceiling below the floor is infeasible" begin
    # Eight weights that sum to one have a 2-norm of at least `1 / sqrt(8) ≈ 0.354`.
    @test isa(solve_wn(MinimumRisk(), (; l2c = 0.34)).retcode, OptimisationFailure)
    @test isa(solve_wn(MinimumRisk(), (; linfc = 0.12)).retcode, OptimisationFailure)
end

@testset "A weight norm ceiling states a floor on the effective number of assets" begin
    m = 5
    res = solve_wn(MaximumReturn(), (; l2c = inv(sqrt(m))))
    @test number_effective_assets(res.w) >= m - 1e-6
    res = solve_wn(MaximumReturn(),
                   (; lpc = LpRegularisation(; p = 3, val = m^(1 / 3 - 1))))
    @test sum(abs.(res.w) .^ 3)^inv(1 - 3) >= m - 1e-6
    res = solve_wn(MaximumReturn(), (; linfc = 1 / m))
    @test inv(norm(res.w, Inf)) >= m - 1e-6
    @test count(>(1e-6), abs.(res.w)) >= m
end

@testset "Weight norm ceilings refuse a value that is not positive and finite" begin
    model = JuMP.Model()
    PO.set_model_scales!(model, 1.0, 1.0)
    for (f, name) in ((PO.set_weight_norm_2_constraints!, "l2c"),
                      (PO.set_weight_norm_inf_constraints!, "linfc")),
        val in (0.0, -1.0, Inf)

        err = try
            f(model, val)
            nothing
        catch e
            e
        end
        @test isa(err, DomainError)
        @test occursin(name, sprint(showerror, err))
    end
    # Any other argument adds nothing to the model.
    @test isnothing(PO.set_weight_norm_2_constraints!(model, nothing))
    @test isnothing(PO.set_weight_norm_p_constraints!(model, nothing))
    @test isnothing(PO.set_weight_norm_inf_constraints!(model, nothing))
    @test isempty(JuMP.all_variables(model))
end

@testset "The dual norm epigraph reaches the dual norm on every route" begin
    xv = [0.3, -0.7, 0.05, 1.2, -0.4]
    for p in (2, Inf, 1, 3, 1.5)
        model = JuMP.Model(Clarabel.Optimizer)
        JuMP.set_silent(model)
        # A constraint scale other than one leaves the feasible set unchanged.
        PO.set_model_scales!(model, 3.0, 1.0)
        JuMP.@variable(model, x[1:length(xv)])
        JuMP.fix.(x, xv)
        t = PO.norm_ball_dual_norm_epigraph!(model, :probe_, 1, x, p)
        JuMP.@objective(model, Min, t)
        JuMP.optimize!(model)
        q = PO.dual_norm_order(p)
        @test isapprox(JuMP.value(t), norm(xv, q); rtol = 1e-6)
        @test haskey(model, :probe_t_nbucs_1)
        @test haskey(model, :probe_nbucs_cone_1)
        power = !(q in (1, 2, Inf))
        @test haskey(model, :probe_r_nbucs_1) == power
        @test haskey(model, :probe_nbucs_cone_sum_1) == power
    end
end
