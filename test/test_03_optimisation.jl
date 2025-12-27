@safetestset "Optimisation failure" begin
    using PortfolioOptimisers, JuMP, Test, Clarabel
    @testset "optimise_JuMP_model!" begin
        model = JuMP.Model()
        JuMP.set_optimizer(model, Clarabel.Optimizer)
        res = PortfolioOptimisers.optimise_JuMP_model!(JuMP.Model(),
                                                       [Solver(; name = :Clarabel,
                                                               solver = Clarabel.Optimizer,
                                                               settings = Dict("verbose" => false)),
                                                        Solver(; name = :Nothing,
                                                               solver = nothing,
                                                               settings = Dict("verbose" => false))])
        @test !res.success
        @test haskey(res.trials[:Clarabel], :optimize!)
        @test haskey(res.trials[:Nothing], :set_optimizer)
        res = PortfolioOptimisers.optimise_JuMP_model!(JuMP.Model(), Solver[])
        @test !res.success
    end
end
