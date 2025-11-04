@safetestset "Struct tests" begin
    using Test, PortfolioOptimisers
    @testset "ReturnsResult" begin
        X = rand(3, 4)
        F = rand(3, 2)
        iv = rand(3, 4)
        ivpa = rand(4)
        nx = 1:4
        nf = 1:2
        ts = 1:3
        @test_throws IsNothingError ReturnsResult(; nx = nothing, X = X)
        @test_throws IsNothingError ReturnsResult(; nx = nx)
        @test_throws IsEmptyError ReturnsResult(; nx = [], X = X)
        @test_throws IsEmptyError ReturnsResult(; nx = nx, X = Matrix{Float64}(undef, 0, 0))
        @test_throws DimensionMismatch ReturnsResult(; nx = nx, X = rand(3, 5))

        @test_throws DimensionMismatch ReturnsResult(; nx = nx, X = X, nf = nf,
                                                     F = rand(4, 2))

        @test_throws IsEmptyError ReturnsResult(; ts = [])
        @test_throws IsNothingError ReturnsResult(; ts = ts)
        @test_throws DimensionMismatch ReturnsResult(; ts = ts, nx = nx, X = rand(2, 4))
        @test_throws DimensionMismatch ReturnsResult(; ts = ts, nf = nf, F = rand(2, 2))

        @test_throws IsEmptyError ReturnsResult(; ts = ts, nx = nx, X = X, nf = nf, F = F,
                                                iv = Matrix{Float64}(undef, 0, 0))
        @test_throws DomainError ReturnsResult(; ts = ts, nx = nx, X = X, nf = nf, F = F,
                                               iv = [Inf Inf])
        @test_throws DomainError ReturnsResult(; ts = ts, nx = nx, X = X, nf = nf, F = F,
                                               iv = [0 -1])

        @test_throws DomainError ReturnsResult(; ts = ts, nx = nx, X = X, nf = nf, F = F,
                                               iv = iv, ivpa = 0)
        @test_throws IsEmptyError ReturnsResult(; ts = ts, nx = nx, X = X, nf = nf, F = F,
                                                iv = iv, ivpa = Float64[])
        @test_throws DomainError ReturnsResult(; ts = ts, nx = nx, X = X, nf = nf, F = F,
                                               iv = iv, ivpa = [0])
        @test_throws DomainError ReturnsResult(; ts = ts, nx = nx, X = X, nf = nf, F = F,
                                               iv = iv, ivpa = Inf)
        @test_throws DomainError ReturnsResult(; ts = ts, nx = nx, X = X, nf = nf, F = F,
                                               iv = iv, ivpa = [Inf; Inf])
        @test_throws DimensionMismatch ReturnsResult(; ts = ts, nx = nx, X = X, nf = nf,
                                                     F = F, iv = iv, ivpa = [1])

        rr = ReturnsResult(; ts = ts, nx = nx, X = X, nf = nf, F = F, iv = iv, ivpa = ivpa)
        @test rr.ts === ts
        @test rr.nx === nx
        @test rr.X === X
        @test rr.nf === nf
        @test rr.F === F
        @test rr.iv === iv
        @test rr.ivpa === ivpa

        rr = ReturnsResult(; ts = ts, nx = nx, X = X, nf = nf, F = F, iv = iv, ivpa = 1)
        @test rr.ivpa == 1
    end
end
