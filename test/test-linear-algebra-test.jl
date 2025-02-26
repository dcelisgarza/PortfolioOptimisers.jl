@safetestset "Linear Algebra" begin
    @testset "Operators" begin
        using PortfolioOptimisers, StatsBase, Random, StableRNGs, Test
        rng = StableRNG(987654321)
        X1 = rand(rng, 10)
        X2 = rand(rng, 10)
        res1 = X1 * transpose(X2)
        res2 = PortfolioOptimisers.:⊗(X1, X2)
        res3 = PortfolioOptimisers.:⊗(transpose(X1), transpose(X2))
        res4 = PortfolioOptimisers.outer_prod(X1, X2)
        res5 = PortfolioOptimisers.outer_prod(transpose(X1), transpose(X2))
        res6 = PortfolioOptimisers.:⊗(X2, X1)
        res7 = PortfolioOptimisers.outer_prod(X2, X1)

        @test isapprox(res1, res2)
        @test isapprox(res1, res3)
        @test isapprox(res1, res4)
        @test isapprox(res1, res5)
        @test !isapprox(res1, res6)
        @test !isapprox(res1, res7)
    end
end
