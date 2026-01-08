@safetestset "Tools" begin
    using PortfolioOptimisers, Statistics, StatsBase, Test, LinearAlgebra
    A = [0, 1, 1, 2, 3, 4, 5, 8]
    B = [2, 3, 3, 5, 8, 10, 15, 21]
    C = 13
    W = fweights([3, 5, 9, 8, 12, 3, 5, 9])
    @test PortfolioOptimisers.:⊙(C, A) == C * A
    @test PortfolioOptimisers.:⊘(A, C) == A / C
    @test PortfolioOptimisers.:⊘(C, C) == C / C
    @test PortfolioOptimisers.:⊕(A, B) == A + B
    @test PortfolioOptimisers.:⊕(C, B) == C .+ B
    @test PortfolioOptimisers.dot_scalar(C, A) == C * sum(A)
    @test PortfolioOptimisers.dot_scalar(A, C) == C * sum(A)
    @test PortfolioOptimisers.vec_to_real_measure(StdValue(), A) == std(A)
    @test PortfolioOptimisers.vec_to_real_measure(VarValue(), A) == var(A)
    @test PortfolioOptimisers.vec_to_real_measure(SumValue(), A) == sum(A)
    @test PortfolioOptimisers.vec_to_real_measure(ProdValue(), A) == prod(A)
    @test PortfolioOptimisers.vec_to_real_measure(ModeValue(), A) == mode(A)
    @test PortfolioOptimisers.vec_to_real_measure(MedianValue(), A) == median(A)
    @test PortfolioOptimisers.vec_to_real_measure(StandardisedValue(), A) ==
          mean(A) / std(A)
    msv = factory(StandardisedValue(), W)
    @test msv.mv.w === W
    @test msv.sv.w === W
    vv = factory(VarValue(), W)
    @test vv.w === W
    vv = factory(VarValue(), W)
    @test vv.w === W
    mdv = factory(MedianValue(), W)
    @test mdv.w === W
end
