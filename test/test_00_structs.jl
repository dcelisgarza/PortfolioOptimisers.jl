@safetestset "Struct tests" begin
    using Test, PortfolioOptimisers, AverageShiftedHistograms
    @testset "VecScalar" begin
        @test_throws IsEmptyError VecScalar(; v = Float64[], s = 1)
        @test_throws DomainError VecScalar(; v = [1.0, Inf, 3.0], s = 1)
        @test_throws DomainError VecScalar(; v = [1.0, -2.0, 3.0], s = Inf)
        vs = VecScalar(; v = [1.0, -2.0, 3.0], s = 2)
        @test vs.v == [1.0, -2.0, 3.0]
        @test vs.s == 2
    end
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

        rr = ReturnsResult()
        @test isnothing(rr.ts)
        @test isnothing(rr.nx)
        @test isnothing(rr.X)
        @test isnothing(rr.nf)
        @test isnothing(rr.F)
        @test isnothing(rr.iv)
        @test isnothing(rr.ivpa)
    end
    @testset "Denoise" begin
        @test_throws DomainError ShrunkDenoise(; alpha = -eps())
        @test_throws DomainError ShrunkDenoise(; alpha = 1.0 + eps())
        sd = ShrunkDenoise(; alpha = 0.5)
        @test sd.alpha == 0.5

        @test_throws DomainError Denoise(; m = 1)
        @test_throws DomainError Denoise(; n = 1)

        de = Denoise()
        @test de.alg == ShrunkDenoise()
        @test de.args == Tuple{}()
        @test de.kwargs == NamedTuple{}()
        @test de.kernel === AverageShiftedHistograms.Kernels.gaussian
        @test de.m == 10
        @test de.n == 1000

        de = Denoise(; alg = FixedDenoise(), args = (1,), kwargs = (foo = 2,),
                     kernel = AverageShiftedHistograms.Kernels.logistic, m = 5, n = 20)
        @test de.alg == FixedDenoise()
        @test de.args == (1,)
        @test de.kwargs == (foo = 2,)
        @test de.kernel === AverageShiftedHistograms.Kernels.logistic
        @test de.m == 5
        @test de.n == 20
    end
    @testset "Detone" begin
        @test_throws DomainError Detone(; n = 0)
        dt = Detone()
        @test dt.n == 1
        dt = Detone(; n = 5)
        @test dt.n == 5
    end
    @testset "Solver" begin
        @test_throws IsEmptyError Solver(; settings = Dict())
        @test_throws IsEmptyError Solver(; settings = Pair[])

        s = Solver()
        @test s.name == ""
        @test isnothing(s.solver)
        @test isnothing(s.settings)
        @test s.check_sol == NamedTuple{}()
        @test s.add_bridges == true

        s = Solver(; name = "MySolver", solver = :X, settings = Dict(:param => 1),
                   check_sol = (foo = 2,), add_bridges = false)
        @test s.name == "MySolver"
        @test s.solver == :X
        @test s.settings == Dict(:param => 1)
        @test s.check_sol == (foo = 2,)
        @test s.add_bridges == false
    end
    @testset "JuMPResult" begin
        trials = Dict(:try => false)
        @test_logs (:warn, "Model could not be solved satisfactorily.\n$trials") JuMPResult(;
                                                                                            trials = Dict(:try => false),
                                                                                            success = false)
        res = JuMPResult(; trials = trials, success = true)
        @test res.trials === trials
        @test res.success == true
    end
    @testset "OWA" begin
        @test_throws DomainError NormalisedConstantRelativeRiskAversion(; g = 0)
        @test_throws DomainError NormalisedConstantRelativeRiskAversion(; g = 1)
        ncrra = NormalisedConstantRelativeRiskAversion()
        @test ncrra.g == 0.5

        ncrra = NormalisedConstantRelativeRiskAversion(; g = 0.25)
        @test ncrra.g == 0.25
    end
end
