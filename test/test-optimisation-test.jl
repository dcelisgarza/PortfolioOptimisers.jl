#=
@safetestset "Optimisation failure" begin
    using PortfolioOptimisers, JuMP, Test, Clarabel
    @testset "optimise_JuMP_model!" begin
        model = JuMP.Model()
        set_optimizer(model, Clarabel.Optimizer)
        success, solvers_tried = PortfolioOptimisers.optimise_JuMP_model!(JuMP.Model(),
                                                                          Solver(;
                                                                                 name = :Clarabel,
                                                                                 solver = Clarabel.Optimizer,
                                                                                 settings = Dict("verbose" => false)))
        @test !success
        @test haskey(solvers_tried[:Clarabel], :jump_error)
    end
end
=#
