# @safetestset "Cross Validation" begin
using Test, PortfolioOptimisers, DataFrames, TimeSeries, CSV, Clarabel, Dates
rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                 timestamp = :Date)[(end - 252 * 4):end],
                       TimeArray(CSV.File(joinpath(@__DIR__, "./assets/Factors.csv.gz"));
                                 timestamp = :Date)[(end - 252 * 4):end])
# @testset "KFold" begin
T = size(rd.X, 1)
n = 5
cv = KFold(; n = n)
train, test = split(cv, rd)
N = n_splits(cv, rd)
@test length(train) == length(test) == N == cv.n - 1

lengths = fill(div(T, cv.n), cv.n)
lengths[1:(mod(T, cv.n))] .+= one(eltype(lengths))
@test length.(train) == lengths[1:4]
@test length.(test) == lengths[2:end]
# end
@testset "Walk forward" begin
    cv = IndexWalkForward(127, 171; reduce_test = false)
    train, test = split(cv, rd)
    N = n_splits(cv, rd)
    @test length(train) == length(test) == N
    @test all(x -> length(x) == 127, train)
    @test all(x -> length(x) == 171, test)
    @test train == UnitRange{Int64}[1:127, 172:298, 343:469, 514:640, 685:811]
    @test test == UnitRange{Int64}[128:298, 299:469, 470:640, 641:811, 812:982]

    cv = IndexWalkForward(137, 111; reduce_test = true, purged_size = 13)
    train, test = split(cv, rd)
    N = n_splits(cv, rd)
    @test length(train) == length(test) == N
    @test all(x -> length(x) == 137, train)
    @test all(x -> length(x) == 111, test[1:(end - 1)])
    @test train ==
          UnitRange{Int64}[1:137, 112:248, 223:359, 334:470, 445:581, 556:692, 667:803,
                           778:914]
    @test test ==
          UnitRange{Int64}[151:261, 262:372, 373:483, 484:594, 595:705, 706:816, 817:927,
                           928:1008]

    cv = IndexWalkForward(137, 111; reduce_test = true, purged_size = 13,
                          expend_train = true)
    train, test = split(cv, rd)
    N = n_splits(cv, rd)
    @test length(train) == length(test) == N
    for (i, t) in enumerate(train)
        @test length(t) == 137 + (i - 1) * 111
    end
    @test all(x -> length(x) == 111, test[1:(end - 1)])
    @test train == UnitRange{Int64}[1:137, 1:248, 1:359, 1:470, 1:581, 1:692, 1:803, 1:914]
    @test test ==
          UnitRange{Int64}[151:261, 262:372, 373:483, 484:594, 595:705, 706:816, 817:927,
                           928:1008]

    function ldm(x)
        val = lastdayofmonth.(x)
        if val[end] > x[end]
            val = val[1:(end - 1)]
        end
        return val
    end
    cv = DateWalkForward(12, 3; period = Month(1), adjuster = ldm)
    train, test = split(cv, rd)
    N = n_splits(cv, rd)
    @test length(train) == length(test) == N
    @test all(x -> length(x) in (252, 253), train)
    @test all(x -> length(x) in (61, 62, 63, 64), test)
    @test train ==
          UnitRange{Int64}[2:253, 64:315, 127:378, 190:442, 254:506, 316:567, 379:630,
                           443:694, 507:758, 568:820, 631:882]
    @test test ==
          UnitRange{Int64}[254:315, 316:378, 379:442, 443:506, 507:567, 568:630, 631:694,
                           695:758, 759:820, 821:882, 883:946]

    cv = DateWalkForward(12, 3; period = Month(1), adjuster = ldm, period_offset = Week(3))
    train, test = split(cv, rd)
    N = n_splits(cv, rd)
    @test length(train) == length(test) == N
    @test all(x -> length(x) in (251, 252, 253), train)
    @test all(x -> length(x) in (62, 63, 65), test)
    @test train ==
          UnitRange{Int64}[16:266, 78:329, 141:392, 205:457, 267:519, 330:581, 393:644,
                           458:709, 520:772, 582:834, 645:896]
    @test test ==
          UnitRange{Int64}[267:329, 330:392, 393:457, 458:519, 520:581, 582:644, 645:709,
                           710:772, 773:834, 835:896, 897:961]

    cv = DateWalkForward(12, 3; period = Month(1), adjuster = ldm, previous = true)
    train, test = split(cv, rd)
    N = n_splits(cv, rd)
    @test length(train) == length(test) == N
    @test all(x -> length(x) in (252, 253), train)
    @test all(x -> length(x) in (61, 62, 63, 64), test)
    @test train ==
          UnitRange{Int64}[2:253, 63:315, 126:378, 190:442, 254:506, 316:567, 379:630,
                           443:694, 507:758, 568:820, 631:882]
    @test test ==
          UnitRange{Int64}[254:315, 316:378, 379:442, 443:506, 507:567, 568:630, 631:694,
                           695:758, 759:820, 821:882, 883:946]

    cv = DateWalkForward(12, 3; period = Month(1), adjuster = ldm, previous = true,
                         purged_size = 17, reduce_test = true)
    train, test = split(cv, rd)
    N = n_splits(cv, rd)
    @test length(train) == length(test) == N
    @test all(x -> length(x) in (252 - 17, 253 - 17), train)
    @test all(x -> length(x) in (61, 62, 63, 64), test)
    @test train ==
          UnitRange{Int64}[2:236, 63:298, 126:361, 190:425, 254:489, 316:550, 379:613,
                           443:677, 507:741, 568:803, 631:865, 695:929]
    @test test ==
          UnitRange{Int64}[254:315, 316:378, 379:442, 443:506, 507:567, 568:630, 631:694,
                           695:758, 759:820, 821:882, 883:946, 947:1008]

    cv = DateWalkForward(12, 3; period = Month(1), adjuster = ldm, previous = true,
                         purged_size = 17, reduce_test = true, expend_train = true)
    train, test = split(cv, rd)
    N = n_splits(cv, rd)
    @test length(train) == length(test) == N
    l = 0
    for (i, t) in enumerate(train)
        l += i == 1 ? 0 : length(test[i - 1])
        @test length(t) in (252 - 17 + l, 253 - 17 + l)
    end
    @test all(x -> length(x) in (61, 62, 63, 64), test)
    @test train ==
          UnitRange{Int64}[1:236, 1:298, 1:361, 1:425, 1:489, 1:550, 1:613, 1:677, 1:741,
                           1:803, 1:865, 1:929]
    @test test ==
          UnitRange{Int64}[254:315, 316:378, 379:442, 443:506, 507:567, 568:630, 631:694,
                           695:758, 759:820, 821:882, 883:946, 947:1008]

    cv = DateWalkForward(Day(23), 13; period = Month(1), adjuster = ldm)
    train, test = split(cv, rd)
    N = n_splits(cv, rd)
    @test length(train) == length(test) == N
    @test all(x -> length(x) in (16, 17), train)
    @test all(x -> length(x) in (272, 273, 274), test)
    @test train == UnitRange{Int64}[7:22, 279:294, 551:567]
    @test test == UnitRange{Int64}[23:294, 295:567, 568:841]

    cv = DateWalkForward(Day(23), 13; period = Month(1), adjuster = ldm, previous = true,
                         purged_size = 17, reduce_test = true, expend_train = true)
    train, test = split(cv, rd)
    N = n_splits(cv, rd)
    @test length(train) == length(test) == N
    @test all(x -> length(x) in (16, 17), train)
    @test all(x -> length(x) in (272, 273, 274), test)
    @test train == UnitRange{Int64}[7:22, 279:294, 551:567]
    @test test == UnitRange{Int64}[23:294, 295:567, 568:841]

    cv = DateWalkForward(Day(23), 13; period = Month(1), adjuster = ldm, purged_size = 17,
                         period_offset = Week(2))
    train, test = split(cv, rd)
    N = n_splits(cv, rd)
    @test length(train) == length(test) == N
    @test all(x -> length(x) in (16, 17), train)
    @test all(x -> length(x) in (272 - 17, 274 - 16), test)
    @test train == UnitRange{Int64}[16:32, 288:304, 561:576]
    @test test == UnitRange{Int64}[33:287, 305:559, 577:834]
end
@testset "Cross val predict" begin
    slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = "verbose" => false),
           Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = ["verbose" => false, "max_step_fraction" => 0.95]),
           Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.9)),
           Solver(; name = :clarabel4, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.85)),
           Solver(; name = :clarabel5, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.80)),
           Solver(; name = :clarabel6, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.75)),
           Solver(; name = :clarabel7, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.7)),
           Solver(; name = :clarabel8, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.6,
                                  "max_iter" => 1500, "tol_gap_abs" => 1e-4,
                                  "tol_gap_rel" => 1e-4, "tol_ktratio" => 1e-3,
                                  "tol_feas" => 1e-4, "tol_infeas_abs" => 1e-4,
                                  "tol_infeas_rel" => 1e-4, "reduced_tol_gap_abs" => 1e-4,
                                  "reduced_tol_gap_rel" => 1e-4,
                                  "reduced_tol_ktratio" => 1e-3, "reduced_tol_feas" => 1e-4,
                                  "reduced_tol_infeas_abs" => 1e-4,
                                  "reduced_tol_infeas_rel" => 1e-4))]
    mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
    cross_val_predict(mr, rd)
end
# end
