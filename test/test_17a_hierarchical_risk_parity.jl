include(joinpath(@__DIR__, "test17_setup.jl"))

@testset "HierarchicalRiskParity" begin
    df = CSV.read(joinpath(@__DIR__, "./assets/HierarchicalRiskParity1.csv.gz"), DataFrame)
    for (i, r) in pairs(rs)
        res = optimise(HierarchicalRiskParity(; r = r, opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        success = isapprox(res.w, df[!, i]; rtol = 5e-7)
        if !success
            println("Counter: $i")
            find_tol(res.w, df[!, i])
        end
        @test success
    end
end
@testset "HierarchicalRiskParity vector rm" begin
    sces = [SumScalariser(), MaxScalariser(), LogSumExpScalariser(; gamma = 1.2e2),
            LogSumExpScalariser(; gamma = 1e6), MinScalariser()]
    df = CSV.read(joinpath(@__DIR__, "./assets/HierarchicalRiskParity2.csv.gz"), DataFrame)
    for (i, sca) in pairs(sces)
        res = optimise(HierarchicalRiskParity(;
                                              r = [ConditionalValueatRisk(),
                                                   Variance(;
                                                            settings = RiskMeasureSettings(;
                                                                                           scale = 2e2))],
                                              opt = opt, sca = sca))
        @test isa(res.retcode, OptimisationSuccess)
        success = isapprox(res.w, df[!, i])
        if !success
            println("Counter: $i")
            find_tol(res.w, df[!, i])
        end
        @test success
    end
end
