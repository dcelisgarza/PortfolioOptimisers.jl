#=
Zoned timestamps (issue #2161).

A `ZonedDateTime` of TimeZones.jl passes through `prices_to_returns` and the `ReturnsResult`
unchanged, because `VecDate` takes any `Dates.AbstractTime`. This file runs one zoned clock
through every consumer of the timestamps that the issue found: the views of the two carriers,
a walk-forward split and the Brinson attribution. The clock crosses the change to daylight
saving time in New York on 2020-03-08, so a reader that drops the zone or steps by absolute
time gives a different answer.
=#
using Dates, TimeZones

@testset "Zoned timestamps reach the views, a walk-forward split and the Brinson attribution (#2161)" begin
    PV = PortfolioOptimisers.port_opt_view
    tz = tz"America/New_York"
    # Ten daily prices at noon local time, from 2020-03-03 to 2020-03-12.
    ts = ZonedDateTime(2020, 3, 3, 12, tz) .+ Day.(0:9)
    P = TimeArray(ts, 100.0 .+ (1:10) * [1.0 2.0 3.0], [:a, :b, :c])

    @testset "the returns keep the zone" begin
        rd = prices_to_returns(P)
        @test rd.ts isa Vector{ZonedDateTime}
        @test rd.ts == ts[2:end]
        @test all(t -> TimeZones.timezone(t) == tz, rd.ts)
        # The UTC offset changes at 2020-03-08, so the zone is not a fixed offset.
        @test endswith(string(rd.ts[1]), "-05:00")
        @test endswith(string(rd.ts[end]), "-04:00")
    end

    @testset "a view of a ReturnsResult reads a zoned timestamp and a vector of them" begin
        rd = prices_to_returns(P)
        v = PV(rd, rd.ts[[2, 4]], :)
        @test v.X == rd.X[[2, 4], :]
        @test v.ts == rd.ts[[2, 4]]
        # One timestamp selects one observation, and the view keeps its matrix shape.
        s = PV(rd, rd.ts[5], :)
        @test size(s.X) == (1, 3)
        @test s.X == rd.X[5:5, :]
        @test s.ts == rd.ts[5:5]
        @test PV(rd, rd.ts[5], ["b"]).X == rd.X[5:5, 2:2]
        # A timestamp that the clock does not hold is refused, and the message names it.
        off = rd.ts[5] + Hour(1)
        err = @test_throws ArgumentError PV(rd, off, :)
        @test occursin(string(off), err.value.msg)
        # A naive timestamp does not compare with a zoned one. The refusal names the index
        # and both types, and not a bare error about promotion.
        err = @test_throws ArgumentError PV(rd, [DateTime(2020, 3, 5, 12)], :)
        @test occursin("DateTime", err.value.msg) &&
              occursin("ZonedDateTime", err.value.msg)
        @test occursin("i holds", err.value.msg)
        @test_throws ArgumentError PV(rd, DateTime(2020, 3, 5, 12), :)
        # A timestamp never selects an asset.
        err = @test_throws ArgumentError PV(rd, :, rd.ts[2])
        @test occursin("observation index", err.value.msg)
        @test_throws ArgumentError PV(rd, rd.ts[2])
        # A carrier with no timestamps has nothing to match one timestamp against.
        bare = ReturnsResult(; nx = rd.nx, X = rd.X)
        err = @test_throws ArgumentError PV(bare, rd.ts[2], :)
        @test occursin("no timestamps", err.value.msg)
    end

    @testset "a view of a PricesResult reads zoned timestamps" begin
        pr = PricesResult(; X = P)
        v = PV(pr, ts[[2, 3]], :)
        @test TimeSeries.timestamp(v.X) == ts[[2, 3]]
        @test values(v.X) == values(P)[[2, 3], :]
        @test values(PV(pr, ts[[2, 3]], ["c"]).X) == values(P)[[2, 3], 3:3]
        err = @test_throws ArgumentError PV(pr, [DateTime(2020, 3, 5, 12)], :)
        @test occursin("ZonedDateTime", err.value.msg)
        @test_throws ArgumentError PV(pr, [DateTime(2020, 3, 5, 12)], ["a"])
    end

    @testset "a Date selects the observations at midnight of DateTime timestamps" begin
        # Dates promotes a Date and a DateTime, so the two compare across their types. The
        # PricesResult view gave a bare MethodError for this call before #2161.
        tsd = DateTime(2020, 3, 3) .+ Day.(0:9)
        Pd = TimeArray(tsd, values(P), [:a, :b, :c])
        prd = PricesResult(; X = Pd)
        @test values(PV(prd, [Date(2020, 3, 4), Date(2020, 3, 6)], :).X) ==
              values(Pd)[[2, 4], :]
        @test values(PV(prd, [Date(2020, 3, 4)], ["a"]).X) == values(Pd)[2:2, 1:1]
        rdd = prices_to_returns(Pd)
        @test PV(rdd, Date(2020, 3, 6), :).X == rdd.X[3:3, :]
    end

    @testset "a walk-forward split of zoned timestamps keeps the wall clock under a calendar period" begin
        # Thirty daily observations at noon local time, across the change on 2020-03-08.
        tsz = ZonedDateTime(2020, 2, 25, 12, tz) .+ Day.(0:29)
        tsn = DateTime(2020, 2, 25, 12) .+ Day.(0:29)
        X = reshape(sin.(1:90) ./ 100, 30, 3)
        rdz = ReturnsResult(; nx = ["a", "b", "c"], X = X, ts = tsz)
        rdn = ReturnsResult(; nx = ["a", "b", "c"], X = X, ts = tsn)
        # A calendar period steps the wall clock, so the zoned split equals the naive split
        # of the same wall clock.
        sz = split(DateWalkForward(5, 3; period = Day(1)), rdz)
        sn = split(DateWalkForward(5, 3; period = Day(1)), rdn)
        @test sz.train_idx == sn.train_idx
        @test sz.test_idx == sn.test_idx
        # A fixed period steps absolute time. After the change each range date is at 13:00,
        # one hour after its observation, and `previous = false` maps it to the next day. So
        # the folds after the change differ, as the docstring of `DateWalkForward` states.
        sh = split(DateWalkForward(5, 3; period = Hour(24)), rdz)
        @test sh.test_idx[1:2] == sz.test_idx[1:2]
        @test sh.test_idx != sz.test_idx
        @test sh.test_idx[3] == 12:15
    end

    @testset "the Brinson attribution reads zoned bounds" begin
        ac = DataFrame(; Asset = ["a", "b", "c"], Class = ["x", "x", "y"])
        w = [0.5, 0.3, 0.2]
        wb = fill(1 / 3, 3)
        # The bounds 2020-03-04 and 2020-03-10 lie on both sides of the change.
        df = brinson_attribution(P, w, wb, ac, :Class, ts[2], ts[8])
        ret = vec(values(P)[8, :] ./ values(P)[2, :] .- 1)
        @test isapprox(df[4, "Total"], dot(w, ret) - dot(wb, ret))
        # Each bound applies on its own.
        @test brinson_attribution(P, w, wb, ac, :Class, ts[2]) ==
              brinson_attribution(P, w, wb, ac, :Class, ts[2], ts[end])
        @test brinson_attribution(P, w, wb, ac, :Class, nothing, ts[8]) ==
              brinson_attribution(P, w, wb, ac, :Class, ts[1], ts[8])
        @test brinson_attribution(P, w, wb, ac, :Class, ts[2]) !=
              brinson_attribution(P, w, wb, ac, :Class)
        # A bound between two observations takes the observations inside it.
        @test brinson_attribution(P, w, wb, ac, :Class, ts[1] + Hour(1), ts[8] + Hour(1)) ==
              df
        # A naive bound does not compare with zoned prices.
        err = @test_throws ArgumentError brinson_attribution(P, w, wb, ac, :Class,
                                                             DateTime(2020, 3, 4),
                                                             DateTime(2020, 3, 10))
        @test occursin("date0", err.value.msg) && occursin("ZonedDateTime", err.value.msg)
        err = @test_throws ArgumentError brinson_attribution(P, w, wb, ac, :Class, nothing,
                                                             DateTime(2020, 3, 10))
        @test occursin("date1", err.value.msg)
        # A window with no observation in it is refused, and the message names both bounds.
        lo, hi = ts[2] + Hour(1), ts[2] + Hour(2)
        err = @test_throws ArgumentError brinson_attribution(P, w, wb, ac, :Class, lo, hi)
        @test occursin(string(lo), err.value.msg) && occursin(string(hi), err.value.msg)
    end
end
