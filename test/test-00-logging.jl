@safetestset "Logging tests" begin
    using Test, PortfolioOptimisers
    @testset "Errors" begin
        @test_throws IsNothingError ReturnsResult(nx = nothing, X = rand(3, 4))
        @test_throws IsNothingError ReturnsResult(nx = [1, 2, 3], X = nothing)
        @test_throws IsEmptyError ReturnsResult(nx = [], X = rand(3, 4))
        @test_throws IsEmptyError ReturnsResult(nx = [1, 2, 3], X = Matrix(undef, 0, 0))
        @test_throws DimensionMismatch ReturnsResult(nx = [1, 2, 3], X = rand(3, 4))
        @test_throws IsEmptyError ReturnsResult(ts = [])
        @test_throws IsNothingEmptyError ReturnsResult(ts = [1, 2, 3])
        @test_throws IsEmptyError ReturnsResult(
            nx = [1, 2, 3],
            X = rand(2, 3),
            iv = Matrix(undef, 0, 0),
        )
        @test_throws DimensionMismatch ReturnsResult(;
            nx = [1, 2, 3],
            X = rand(2, 3),
            iv = rand(3, 3),
        )
        @test_throws DomainError ReturnsResult(;
            nx = [1, 2, 3],
            X = rand(3, 3),
            iv = -rand(3, 3),
        )
        @test_throws DomainError ReturnsResult(;
            nx = [1, 2, 3],
            X = rand(3, 3),
            iv = rand(3, 3),
            ivpa = 0,
        )
        @test_throws IsEmptyError ReturnsResult(;
            nx = [1, 2, 3],
            X = rand(3, 3),
            iv = rand(3, 3),
            ivpa = [],
        )
        @test_throws DimensionMismatch ReturnsResult(;
            nx = [1, 2, 3],
            X = rand(3, 3),
            iv = rand(3, 3),
            ivpa = [1],
        )
        @test_throws DomainError ReturnsResult(;
            nx = [1, 2, 3],
            X = rand(3, 3),
            iv = rand(3, 3),
            ivpa = [1, 0, Inf],
        )
        @test_throws DomainError ReturnsResult(;
            nx = [1, 2, 3],
            X = rand(3, 3),
            iv = rand(3, 3),
            ivpa = [1, 0, -1],
        )
    end
end
