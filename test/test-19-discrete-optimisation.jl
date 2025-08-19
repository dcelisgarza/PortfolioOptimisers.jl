@safetestset "Finite allocation" begin
    using PortfolioOptimisers, Clarabel, HiGHS, Test, CSV, TimeSeries, DataFrames,
          LinearAlgebra, StatsBase
    X = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
    rd = prices_to_returns(X)
    slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false),
                 check_sol = (; allow_local = true, allow_almost = true))
    mip_slv = Solver(; name = :highs1, solver = HiGHS.Optimizer,
                     settings = Dict("log_to_console" => false),
                     check_sol = (; allow_local = true, allow_almost = true))
    da = DiscreteAllocation(; slv = mip_slv)
    ga = GreedyAllocation(; unit = 0.3, kwargs = (sigdigits = 1,))
    mr = MeanRisk(; obj = MaximumRatio(; rf = 4.2 / 252 / 100),
                  opt = JuMPOptimiser(; sbgt = 1, bgt = 0.5,
                                      wb = WeightBounds(; lb = -1, ub = 1), slv = slv))
    res = optimise!(mr, rd)

    res_da = optimise!(da, res.w, vec(values(X[end])), 4206.9)
    @test isapprox(sum(res_da.cost), 4206.9 * 0.5, rtol = 5e-3)
    @test isapprox(sum(res.w[res.w .< 0]), -1, rtol = 1e-4)
    @test isapprox(res_da.shares .* vec(values(X[end])), res_da.cost)
    @test isapprox(rmsd(res.w, res_da.w), 0.0838, rtol = 5e-4)

    res_ga = optimise!(ga, res.w, vec(values(X[end])), 4206.9)
    @test isapprox(sum(res_ga.cost), 4206.9 * 0.5, rtol = 4e-2)
    @test isapprox(sum(res.w[res.w .< 0]), -1, rtol = 1e-4)
    @test isapprox(res_ga.shares .* vec(values(X[end])), res_ga.cost)
    @test isapprox(rmsd(res.w, res_ga.w), 0.0568, rtol = 2e-3)
    @test all(isapprox.(mod.(round.(mod.(abs.(res_ga.shares), 1), sigdigits = 1), ga.unit),
                        0, atol = 1e-10))
end
