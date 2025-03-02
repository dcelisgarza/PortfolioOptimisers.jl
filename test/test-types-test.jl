@safetestset "Types tests" begin
    using PortfolioOptimisers, Test, Clarabel, HiGHS
    @testset "Solver" begin
        solvers1 = Solver(; solver = Clarabel.Optimizer,
                          settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        solvers2 = Solver(; solver = Clarabel.Optimizer,
                          settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        solvers3 = Solver(; solver = HiGHS.Optimizer,
                          settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        @test isequal(solvers1, solvers2)
        @test !isequal(solvers1, solvers3)
    end
end
