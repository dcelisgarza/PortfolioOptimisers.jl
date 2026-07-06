@testset "Optimisation failure" begin
    using PortfolioOptimisers, JuMP, Test, Clarabel
    @testset "optimise_JuMP_model!" begin
        model = JuMP.Model()
        JuMP.set_optimizer(model, Clarabel.Optimizer)
        res = PortfolioOptimisers.optimise_JuMP_model!(JuMP.Model(),
                                                       [Solver(; name = :Clarabel,
                                                               solver = Clarabel.Optimizer,
                                                               settings = Dict("verbose" =>
                                                                                   false)),
                                                        Solver(; name = :Nothing,
                                                               solver = nothing,
                                                               settings = Dict("verbose" =>
                                                                                   false))])
        @test !res.success
        @test haskey(res.trials[:Clarabel], :optimize!)
        @test haskey(res.trials[:Nothing], :set_optimizer)
        res = PortfolioOptimisers.optimise_JuMP_model!(JuMP.Model(), Solver[])
        @test !res.success
    end
end

@testset "Solver fallback chain robustness" begin
    using PortfolioOptimisers, JuMP, Test, Clarabel
    function testmodel(lb)
        model = JuMP.Model()
        JuMP.@variable(model, w[1:3])
        JuMP.@variable(model, k)
        JuMP.@constraint(model, sum(w) == 1)
        JuMP.@constraint(model, w .>= lb)
        JuMP.@constraint(model, k == 1)
        JuMP.@objective(model, Min, sum(w .^ 2))
        return model
    end
    @testset "clean termination without primal does not abort the chain" begin
        # `w .>= 0.9` with `sum(w) == 1` is infeasible: solvers terminate without a
        # primal, which used to crash the unguarded `JuMP.value` reads.
        slv = [Solver(; name = :c1, solver = Clarabel.Optimizer,
                      settings = Dict("verbose" => false)),
               Solver(; name = :c2, solver = Clarabel.Optimizer,
                      check_sol = (; allow_local = true, allow_almost = true),
                      settings = Dict("verbose" => false))]
        mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        logger = SimpleLogger()
        with_logger(logger) do
            @test_logs (:warn, r"Failed to solve") PortfolioOptimisers.optimise_JuMP_model!(testmodel(0.9),
                                                                                            mr)
        end
        retcode, sol = PortfolioOptimisers.optimise_JuMP_model!(testmodel(0.9), mr)
        @test isa(retcode, OptimisationFailure)
        @test all(isnan, sol.w)
        # Both solvers must have been tried and their diagnostics merged: the
        # assertion error must not be overwritten by the solution summary.
        for name in (:c1, :c2)
            trial = retcode.res[name]
            @test haskey(trial, :assert_is_solved_and_feasible)
            @test haskey(trial, :err)
            @test haskey(trial, :settings)
        end
    end
    @testset "chain advances past a failing solver to a successful one" begin
        slv = [Solver(; name = :limited, solver = Clarabel.Optimizer,
                      settings = Dict("verbose" => false, "max_iter" => 1)),
               Solver(; name = :ok, solver = Clarabel.Optimizer,
                      check_sol = (; allow_local = true, allow_almost = true),
                      settings = Dict("verbose" => false))]
        mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        retcode, sol = PortfolioOptimisers.optimise_JuMP_model!(testmodel(0.0), mr)
        @test isa(retcode, OptimisationSuccess)
        @test haskey(retcode.res, :limited)
        @test haskey(retcode.res[:limited], :assert_is_solved_and_feasible)
        @test isapprox(sol.w, fill(inv(3), 3); rtol = 1e-6)
    end
end

@testset "Risk upper bound with unsupported optimiser" begin
    using PortfolioOptimisers, JuMP, Test, Clarabel
    slv = Solver(; name = :c1, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false))
    frc = FactorRiskContribution(; opt = JuMPOptimiser(; slv = slv))
    model = JuMP.Model()
    # A user-set `settings.ub` must warn instead of being silently dropped.
    logger = SimpleLogger()
    with_logger(logger) do
        @test_logs (:warn, r"Risk upper bound") PortfolioOptimisers.set_risk_upper_bound!(model,
                                                                                          frc,
                                                                                          1.0,
                                                                                          1.0,
                                                                                          :risk)
    end
    # No bound requested: no-op.
    @test isnothing(PortfolioOptimisers.set_risk_upper_bound!(model, frc, 1.0, nothing,
                                                              :risk))
end
