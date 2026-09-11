#=
The ingestion layer's emission, issue #973 of map #955.

`PriceIngestion` assembles raw price series into the carrier the conversion reads, and the
conversion hands the returns carrier an `AssetPanel` stating the universe. What the file pins
is the set of statements ADR 0129, 0131 and 0132 make about that path, each one asserted
rather than assumed:

  - The layer emits **one carrier**, and it **always** carries a panel. A gapless price table
    is not a special case that skips one; it is a panel whose masks are all true. So
    `pnl === nothing` on a returns carrier means one thing only: the carrier was not built by
    the layer.

  - Saying so costs nothing. A gapless ingestion's two masks are one `AllTrueMask`, which
    stores no cell, and a gapped one's active mask is the projected `ListingSpan`, two
    integers per asset. Both are ordinary `AbstractMatrix{Bool}`s to every reader.

  - `emsk` is a **snapshot** taken after the conversion, not a view over `X`. A value-level
    rewrite of the returns therefore does not move the estimation universe under a fold that
    was already scored against it.

  - The alignment guarantee has **split**. The asset axis is fixed before the split, so
    `assert_universe_aligned` narrows to a provenance check: `nx` equality plus panel-presence
    parity, reaching only a carrier built outside the layer or a step that changes the asset
    set.

  - An `AssetPanel` with **no Panel Field** is the layer's common case. A caller holding only
    prices has no market capitalisation and no sector, and the panel states a universe and
    nothing else.

The closing test is the destination itself: a caller holding a gapped price table reaches a
walk-forward fold with no hand-built `ReturnsResult` and no hand-built mask.
=#
using Dates, Clarabel

# The fixture carries all three span cases in one table, because the Span Rule reads position.
# `a` is priced throughout, so no window's universe is ever empty.
T59, N59 = 40, 4
ts59 = Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(T59 - 1))
nx59 = ["a", "b", "c", "d"]
P59 = 100.0 .+ cumsum(randn(StableRNG(973), T59, N59) ./ 10; dims = 1)
P59[1:8, 2] .= NaN             # b lists at observation 9: a LEADING run of gaps
P59[33:40, 3] .= NaN           # c delists at observation 33: a TRAILING run of gaps
P59[18:20, 4] .= NaN           # d is suspended and prices again: an INTERIOR gap
X59 = TimeArray(collect(ts59), P59, nx59)

# The same table with every gap spelled the other way, which is what a wide table built from a
# tidy one leaves.
Pm59 = Matrix{Union{Missing, Float64}}(P59)
Pm59[.!isfinite.(P59)] .= missing
Xm59 = TimeArray(collect(ts59), Pm59, nx59)

# A table with no gap at all, the case the layer must still emit a panel for.
Pg59 = 100.0 .+ cumsum(randn(StableRNG(9731), T59, N59) ./ 10; dims = 1)
Xg59 = TimeArray(collect(ts59), Pg59, nx59)

ptr59 = PricesToReturns()
rd59 = prices_to_returns(ptr59, price_ingestion(PriceIngestion(), X59))

@testset "PriceIngestion assembles the span-carrying price carrier" begin
    pr = price_ingestion(PriceIngestion(), X59)
    @test isa(pr, PricesResult)
    @test isa(pr.span, PortfolioOptimisers.ListingSpan)
    @test size(pr.span) == (T59, N59)

    # The Span Rule: leading and trailing runs fall outside the span, an interior one inside.
    expected = trues(T59, N59)
    expected[1:8, 2] .= false
    expected[33:40, 3] .= false
    @test Matrix(pr.span) == expected
    @test pr.span.first == [1, 9, 1, 1]
    @test pr.span.last == [40, 40, 32, 40]

    @testset "the two absent-price conventions unify, and unify to NaN" begin
        prm = price_ingestion(PriceIngestion(), Xm59)
        # A `missing` source and a `NaN` source give the same carrier and the same span.
        @test isa(values(prm.X), Matrix{Float64})
        @test Matrix(prm.span) == Matrix(pr.span)
        @test isequal(values(prm.X), values(pr.X))
        # The unification is what no deletion step reads, so the gaps survive the conversion.
        @test count(!isfinite, values(prm.X)) == count(!isfinite, P59)
        # A float table is already spelled that way and is returned untouched.
        @test PortfolioOptimisers.unify_gaps(X59) === X59
        @test !isa(PortfolioOptimisers.unify_gaps(Xm59), typeof(Xm59))
    end

    @testset "a caller's declaration replaces the Span Rule's answer outright" begin
        # A listing calendar that says `c` is listed to the end, against the gaps.
        own = trues(T59, N59)
        own[1:8, 2] .= false
        prd = price_ingestion(PriceIngestion(; span = own), X59)
        @test prd.span === own
        @test Matrix(prd.span) != Matrix(pr.span)
    end

    @testset "the join and the collapse move the clock, so they run here" begin
        # A factor priced on every second day. ADR 0135: the asset table states the clock,
        # so the default join keeps every asset observation and pads the factor at the
        # twenty it is silent at, and the layer names the padding.
        F = TimeArray(collect(ts59)[1:2:end], reshape(P59[1:2:end, 1], :, 1), ["f"])
        prj = @test_logs (:warn, r"`F` at 20 of 40 observations, in every column \(f\)") price_ingestion(PriceIngestion(),
                                                                                                         X59;
                                                                                                         F = F)
        @test TimeSeries.timestamp(prj.X) == TimeSeries.timestamp(prj.F)
        @test TimeSeries.timestamp(prj.X) == collect(ts59)
        @test size(values(prj.X)) == (T59, N59)
        @test size(prj.span) == (T59, N59)
        # Half the factor rows are gaps the join padded, and the asset span is unchanged.
        @test count(!isfinite, values(prj.F)) == T59 - length(1:2:T59)
        @test Matrix(prj.span) == Matrix(pr.span)
        # `strict` refuses with the same report.
        err = try
            price_ingestion(PriceIngestion(; strict = true), X59; F = F)
        catch e
            e
        end
        @test isa(err, ArgumentError)
        @test occursin("`F` at 20 of 40 observations", err.msg)

        # A shared benchmark joins on the same clock, and it is one column whatever the
        # asset count is. It covers the asset clock, so it pads nothing and reports nothing:
        # the factor is the only line.
        B = TimeArray(collect(ts59), reshape(P59[:, 1], :, 1), ["bm"])
        prb = @test_logs (:warn,
                          r"universe: `F` at 20 of 40 observations, in every column \(f\)\. Under") price_ingestion(PriceIngestion(),
                                                                                                                    X59;
                                                                                                                    F = F,
                                                                                                                    B = B)
        @test TimeSeries.timestamp(prb.B) == TimeSeries.timestamp(prb.X)
        @test size(values(prb.B), 2) == 1
        @test Matrix(prb.span) == Matrix(pr.span)
        @test all(isfinite, values(prb.B))
        # A covariate on the asset clock pads nothing, warns nothing, and passes `strict`.
        prq = @test_logs price_ingestion(PriceIngestion(; strict = true), X59; B = B)
        @test all(isfinite, values(prq.B))

        # `:outer` and `:inner` stay reachable. The union pads the ASSET table too, at the
        # observations only a covariate has, and the report names it — the Span Rule reads
        # those rows as listings and delistings, which is why it is not the default.
        tsl = collect(Date(2019, 12, 22):Day(1):(Date(2020, 1, 1) + Day(T59 - 1)))
        Bl = TimeArray(tsl, reshape(collect(200.0:(200.0 + length(tsl) - 1)), :, 1), ["bm"])
        pro = @test_logs (:warn,
                          r"`X` at 10 of 50 observations, in every column \(a, b, c, d\)") price_ingestion(PriceIngestion(;
                                                                                                                          join_method = :outer),
                                                                                                           X59;
                                                                                                           B = Bl)
        @test TimeSeries.timestamp(pro.X) == tsl
        @test count(!isfinite, values(pro.X)) == count(!isfinite, P59) + 10 * N59
        @test all(isfinite, values(pro.B))
        @test_throws ArgumentError price_ingestion(PriceIngestion(; join_method = :outer,
                                                                  strict = true), X59;
                                                   B = Bl)
        # The intersection drops the covariate's extra rows, pads nothing and reports
        # nothing.
        pri = @test_logs price_ingestion(PriceIngestion(; join_method = :inner,
                                                        strict = true), X59; B = Bl)
        @test TimeSeries.timestamp(pri.X) == collect(ts59)
        # And the default, `:left`, gives the same clock with the benchmark cut to it.
        prl = @test_logs price_ingestion(PriceIngestion(; strict = true), X59; B = Bl)
        @test TimeSeries.timestamp(prl.X) == collect(ts59)
        @test values(prl.B) == values(pri.B)

        # A weekly collapse renumbers every observation, and the span is derived after it.
        prc = price_ingestion(PriceIngestion(; collapse_args = (Dates.week, first, last)),
                              X59)
        @test size(values(prc.X), 1) < T59
        @test size(prc.span) == size(values(prc.X))
    end

    @testset "implied volatilities are carried and clock-aligned, never converted" begin
        iv = TimeArray(collect(ts59), fill(0.2, T59, N59), nx59)
        pri = @test_logs price_ingestion(PriceIngestion(; strict = true), X59; iv = iv,
                                         ivpa = 1.5)
        @test TimeSeries.timestamp(pri.iv) == TimeSeries.timestamp(pri.X)
        @test all(values(pri.iv) .== 0.2)
        @test pri.ivpa == 1.5
        # ADR 0135: an implied volatility series is a carried series like the factors and
        # the benchmark. Silent at an observation of the emitted clock, it is padded `NaN`
        # there and the padding is named, rather than the ingestion being refused.
        ivs = iv[collect(ts59)[1:(T59 - 1)]]
        prs = @test_logs (:warn,
                          r"`iv` at 1 of 40 observations, in every column \(a, b, c, d\)") price_ingestion(PriceIngestion(),
                                                                                                           X59;
                                                                                                           iv = ivs)
        @test TimeSeries.timestamp(prs.iv) == collect(ts59)
        @test all(isnan, values(prs.iv)[T59, :])
        @test all(values(prs.iv)[1:(T59 - 1), :] .== 0.2)
        @test_throws ArgumentError price_ingestion(PriceIngestion(; strict = true), X59;
                                                   iv = ivs)
        # An implied volatility on a wider clock than the assets' is cut to theirs, and
        # that is not padding.
        ivw = TimeArray(collect(Date(2019, 12, 25):Day(1):(Date(2020, 1, 1) + Day(T59 - 1))),
                        fill(0.2, T59 + 7, N59), nx59)
        prw = @test_logs price_ingestion(PriceIngestion(; strict = true), X59; iv = ivw)
        @test TimeSeries.timestamp(prw.iv) == collect(ts59)
        @test all(values(prw.iv) .== 0.2)
    end

    @testset "ADR 0135's measured refusal stops firing" begin
        # A five-day panel, a five-day implied volatility on its clock, and a fifteen-day
        # benchmark. Under a symmetric join the benchmark moved the clock out from under
        # the implied volatilities, and the ingestion refused with `10 of the 15 emitted
        # observations are absent from iv`. Nothing is wrong with the data.
        ts5 = Date(2020, 1, 1):Day(1):Date(2020, 1, 5)
        X5 = TimeArray(ts5, [100.0 50.0; 101 51; 102 52; 103 53; 104 54], ["A", "B"])
        iv5 = TimeArray(ts5, fill(0.2, 5, 2), ["A", "B"])
        ts15 = Date(2019, 12, 27):Day(1):Date(2020, 1, 10)
        B15 = TimeArray(ts15, collect(200.0:214.0), ["BM"])
        pr5 = @test_logs price_ingestion(PriceIngestion(; strict = true), X5; iv = iv5,
                                         B = B15)
        @test size(values(pr5.X)) == (5, 2)
        @test TimeSeries.timestamp(pr5.X) == collect(ts5)
        @test all(isfinite, values(pr5.X))
        @test all(isfinite, values(pr5.B))
        @test all(values(pr5.iv) .== 0.2)
        # The invariant the default path gains: the emitted clock is the asset table's.
        @test TimeSeries.timestamp(pr5.X) == TimeSeries.timestamp(X5)
        rd5 = prices_to_returns(pr5)
        @test size(rd5.X) == (4, 2)
        @test all(isfinite, rd5.X)
        @test all(isfinite, rd5.B)
    end

    @testset "the value type is derived from the series, and a type that cannot spell an absence is refused by name" begin
        ts5 = Date(2020, 1, 1):Day(1):Date(2020, 1, 5)
        P5 = [100 50; 101 51; 102 52; 103 53; 104 54]
        F5 = TimeArray(ts5, reshape(collect(1.0:5.0), :, 1), ["f"])

        # A `Float32` panel stays `Float32`, and so does its span-carrying conversion.
        X32 = TimeArray(ts5, Float32.(P5), ["A", "B"])
        pr32 = price_ingestion(PriceIngestion(), X32)
        @test eltype(values(pr32.X)) === Float32
        @test values(pr32.X) == values(X32)
        @test eltype(prices_to_returns(pr32).X) === Float32

        # `Float32` beside `Float64` promotes and joins, where `TimeSeries.merge` alone
        # raises a `MethodError` on the two value types.
        @test_throws MethodError TimeSeries.merge(X32, F5; method = :left)
        prm = price_ingestion(PriceIngestion(), X32; F = F5)
        @test eltype(values(prm.X)) === Float64
        @test eltype(values(prm.F)) === Float64
        @test values(prm.X) == Float64.(P5)

        # An integer panel takes the floating-point type that represents it, derived from
        # the division a return performs rather than named.
        Xi = TimeArray(ts5, P5, ["A", "B"])
        pri = price_ingestion(PriceIngestion(), Xi)
        @test eltype(values(pri.X)) === typeof(one(Int) / one(Int))
        @test values(pri.X) == P5
        @test PortfolioOptimisers.absence_type(Int) === Float64
        @test PortfolioOptimisers.absence_type(Float32) === Float32
        @test PortfolioOptimisers.absence_type(BigInt) === BigFloat
        @test PortfolioOptimisers.absence_type(Rational{Int}) === Rational{Int}

        # A `Rational` panel with no gap is carried as it is: nothing is absent, so nothing
        # is spelled. One holding a gap is refused by name, and so is one a join must pad.
        Xr = TimeArray(ts5, Rational{Int}.(P5), ["A", "B"])
        prr = price_ingestion(PriceIngestion(), Xr)
        @test eltype(values(prr.X)) === Rational{Int}
        Prm = Matrix{Union{Missing, Rational{Int}}}(Rational{Int}.(P5))
        Prm[1, 2] = missing
        Xrm = TimeArray(ts5, Prm, ["A", "B"])
        err = try
            price_ingestion(PriceIngestion(), Xrm)
        catch e
            e
        end
        @test isa(err, DomainError)
        @test err.val === Rational{Int}
        @test occursin("cannot carry one", err.msg)
        Fr = TimeArray(ts5[1:3], Rational{Int}.(reshape(1:3, :, 1)), ["f"])
        @test_throws DomainError price_ingestion(PriceIngestion(), Xr; F = Fr)
        # An inner join pads nothing, so it asks for no absence.
        @test eltype(values(price_ingestion(PriceIngestion(; join_method = :inner), Xr;
                                            F = Fr).X)) === Rational{Int}

        # The single-argument `unify_gaps` derives the same target from the series alone,
        # which is what a carrier built by hand gets at the conversion.
        @test PortfolioOptimisers.unify_gaps(X32) === X32
        @test eltype(values(PortfolioOptimisers.unify_gaps(Xi))) === Float64
        Pm = Matrix{Union{Missing, Int}}(P5)
        Pm[2, 1] = missing
        um = PortfolioOptimisers.unify_gaps(TimeArray(ts5, Pm, ["A", "B"]))
        @test eltype(values(um)) === Float64
        @test isnan(values(um)[2, 1])
        @test values(um)[1, 1] == 100.0
        @test_throws DomainError PortfolioOptimisers.absent_value(Rational{Int})
        @test_throws DomainError PortfolioOptimisers.absent_value(Int)
        @test isnan(PortfolioOptimisers.absent_value(Float32))
        @test PortfolioOptimisers.series_value_type(nothing) === Union{}
        @test PortfolioOptimisers.series_value_type(Xrm) === Rational{Int}
    end

    @testset "a PricesResult is re-ingested through the same verb" begin
        pr2 = price_ingestion(PriceIngestion(), PricesResult(; X = X59))
        @test Matrix(pr2.span) == Matrix(pr.span)
        @test_throws PortfolioOptimisers.IsEmptyError price_ingestion(PriceIngestion(),
                                                                      X59[Date[]])
    end
end

# Issue #990. A column name says which series a column came from, and both doors refuse a
# name that cannot do that job any more. The refusal is one function,
# `assert_distinct_series_names`, called by `price_ingestion` before it merges and by
# `prices_to_returns` before it does.
@testset "A shared column name is refused at both doors" begin
    one59(name) = TimeArray(collect(ts59), reshape(fill(10.0, T59), T59, 1), [name])
    cae = PortfolioOptimisers.ConflictingArgumentError

    # `b` is an asset name, so this factor table collides with the asset table.
    Fb = TimeArray(collect(ts59),
                   50.0 .+ cumsum(randn(StableRNG(9732), T59, 2) ./ 10; dims = 1),
                   ["f", "b"])
    Bc = one59("c")            # `c` is an asset name too
    Fg, Bg = one59("g"), one59("g")    # the factor and the benchmark collide with each other
    Ff = one59("f")            # no collision with anything

    # What the collision did before the refusal, asserted rather than recalled: `merge`
    # renames the SECOND of two columns that share a name, so `M[colnames(F)]` reads `X`'s
    # column and the factor block silently carried an asset's prices.
    M = TimeSeries.merge(X59, Fb; method = :outer)
    @test TimeSeries.colnames(M) == [:a, :b, :c, :d, :f, :b_1]
    @test isequal(values(M[TimeSeries.colnames(Fb)])[:, 2], values(X59[:b]))

    # All three pairs, at the ingestion door.
    @test_throws cae price_ingestion(PriceIngestion(), X59; F = Fb)
    @test_throws cae price_ingestion(PriceIngestion(), X59; B = Bc)
    @test_throws cae price_ingestion(PriceIngestion(), X59; F = Fg, B = Bg)

    # And at the conversion's own door, which is the form issue #990 reported. The
    # conversion takes a carrier, so the collision reaches it on one: a hand-built
    # `PricesResult` is the one route that does not pass the ingestion door first.
    @test_throws cae prices_to_returns(PricesResult(; X = X59, F = Fb))
    @test_throws cae prices_to_returns(PricesResult(; X = X59, B = Bc))
    @test_throws cae prices_to_returns(PricesResult(; X = X59, F = Fg, B = Bg))

    # The error names the shared column, because a caller has to know which one to rename.
    err = try
        price_ingestion(PriceIngestion(), X59; F = Fb)
    catch e
        e
    end
    @test occursin("[:b]", err.msg)
    @test occursin("`X`", err.msg)
    @test occursin("`F`", err.msg)

    # The clock's own name is the layer's. A series carrying it takes the place of the column
    # the `DataFrames.DataFrame` conversion writes, and the block holding it reads dates as
    # prices.
    Xts = TimeArray(collect(ts59), P59, ["timestamp", "b", "c", "d"])
    @test_throws cae price_ingestion(PriceIngestion(), Xts)
    @test_throws cae prices_to_returns(Xts)
    @test_throws cae PortfolioOptimisers.assert_distinct_series_names(X59,
                                                                      one59("timestamp"))
    @test_throws cae PortfolioOptimisers.assert_distinct_series_names(X59, nothing,
                                                                      one59("timestamp"))

    # A name list is read off a table, and an absent table names nothing.
    @test PortfolioOptimisers.series_names(nothing) == Symbol[]
    @test PortfolioOptimisers.series_names(X59) == Symbol.(nx59)
    @test isnothing(PortfolioOptimisers.assert_distinct_series_names(X59, Ff, one59("bmk")))
    @test isnothing(PortfolioOptimisers.assert_distinct_series_names(X59))

    # A table's own duplicates cannot reach the door: every `TimeArray` constructor renames
    # them, so the refusal has nothing left to say about them.
    @test TimeSeries.colnames(TimeArray(collect(ts59), P59, ["a", "a", "a", "a"])) ==
          [:a, :a_1, :a_2, :a_3]

    # The names the refusal guarantees are what lets the conversion name its blocks outright.
    # The clock is the `timestamp` column, typed, and never a `Vector{Any}` of interleaved
    # dates and prices.
    rd = prices_to_returns(price_ingestion(PriceIngestion(), X59; F = Ff, B = one59("bmk")))
    @test isa(rd.ts, Vector{Date})
    @test rd.nx == nx59
    @test rd.nf == ["f"]
    @test rd.nb == ["bmk"]
    @test size(rd.X) == (T59 - 1, N59)
end

@testset "The layer emits one carrier, and it always carries a panel" begin
    pr = price_ingestion(PriceIngestion(), X59)
    rd = prices_to_returns(ptr59, pr)
    @test isa(rd, ReturnsResult)
    @test !isnothing(rd.pnl)
    @test rd.nx == nx59

    # `padding = false`, so the returns clock is one observation shorter than the prices.
    m = T59 - 1
    @test size(rd.X) == (m, N59)

    @testset "the active mask is the projected span, and the estimation mask follows the values" begin
        # A return consumes the earlier price of its pair, so the unpadded projection is
        # [first, last - 1]: b's inception emits no Held Gap, and c's delisting is booked one
        # observation earlier than its last price.
        amsk = falses(m, N59)
        amsk[:, 1] .= true
        amsk[9:m, 2] .= true
        amsk[1:31, 3] .= true
        amsk[:, 4] .= true
        @test Matrix(rd.pnl.amsk) == amsk
        # The estimation mask is the active mask intersected with finiteness, by construction.
        @test Matrix(rd.pnl.emsk) == amsk .& isfinite.(rd.X)
        # d's interior gap of k = 3 prices leaves k + 1 = 4 non-finite returns, and those four
        # are active but not estimable: the Held Gap.
        @test count(view(amsk, :, 4) .& .!view(Matrix(rd.pnl.emsk), :, 4)) == 4
    end

    @testset "the compressions: two integers per asset, and none at all" begin
        # A gapped ingestion keeps ADR 0129's exact compression across the clock crossing.
        @test isa(rd.pnl.amsk, PortfolioOptimisers.ListingSpan)
        # A gapless one says "all of them" in O(1) for both masks.
        rdg = prices_to_returns(ptr59, price_ingestion(PriceIngestion(), Xg59))
        @test isa(rdg.pnl.amsk, PortfolioOptimisers.AllTrueMask)
        @test rdg.pnl.emsk === rdg.pnl.amsk
        @test all(rdg.pnl.amsk)
        @test size(rdg.pnl.amsk) == (m, N59)
        # Both are ordinary matrices of booleans to every reader.
        @test isa(rdg.pnl.amsk, AbstractMatrix{Bool})
        @test Base.IndexStyle(typeof(rdg.pnl.amsk)) === IndexCartesian()
        @test sprint(show, rdg.pnl.amsk) == "AllTrueMask($m × $N59)"
        @test sprint(show, MIME"text/plain"(), rdg.pnl.amsk) == "AllTrueMask($m × $N59)"
        @test_throws DomainError PortfolioOptimisers.AllTrueMask(-1, 2)
        @test_throws BoundsError rdg.pnl.amsk[m + 1, 1]
    end

    @testset "the masks slice in step with the returns under port_opt_view" begin
        v = PortfolioOptimisers.port_opt_view(rd, 10:20, [1, 4])
        @test size(v.X) == (11, 2)
        @test size(v.pnl.amsk) == (11, 2)
        @test Matrix(v.pnl.amsk) == Matrix(rd.pnl.amsk)[10:20, [1, 4]]
        @test Matrix(v.pnl.emsk) == Matrix(rd.pnl.emsk)[10:20, [1, 4]]
        # A price carrier's span slices the same way, and it is viewed rather than re-derived:
        # `c`'s delisting still reads as a delisting inside a window that ends before it.
        pv = PortfolioOptimisers.port_opt_view(pr, 1:20, 1:N59)
        @test size(pv.span) == (20, N59)
        @test Matrix(pv.span) == Matrix(pr.span)[1:20, :]
        @test all(view(Matrix(pv.span), :, 3))
        # A row cut can split an interval in half, which no interval can say, so the cut span
        # is an ordinary view of booleans rather than a Listing Span.
        @test !isa(pv.span, PortfolioOptimisers.ListingSpan)
        # Keeping the whole clock in order leaves every interval intact, so the two integers
        # per asset survive the cut.
        pw = PortfolioOptimisers.port_opt_view(pr, 1:T59, [1, 3])
        @test isa(pw.span, PortfolioOptimisers.ListingSpan)
        @test Matrix(pw.span) == Matrix(pr.span)[:, [1, 3]]

        # A caller's own declaration is any matrix of booleans, and it slices the same way.
        own = trues(T59, N59)
        own[1:8, 2] .= false
        pd = PortfolioOptimisers.port_opt_view(price_ingestion(PriceIngestion(; span = own),
                                                               X59), 5:25, [2, 4])
        @test Matrix(pd.span) == own[5:25, [2, 4]]

        # A price-level filter that drops rows and columns cuts the span with them, and one
        # that only fills leaves it alone.
        mdf = fit_preprocessing(MissingDataFilter(; row_thr = 0.2), pr)
        pf = apply_preprocessing(mdf, pr)
        @test size(pf.span) == size(values(pf.X))
        pgf = fit_preprocessing(PriceGapFill(), pr)
        @test apply_preprocessing(pgf, pr).span === pr.span
    end

    @testset "the estimation mask is a snapshot, not a view over the returns" begin
        # A value-level rewrite of every return leaves both masks exactly where they were.
        before_a = Matrix(rd.pnl.amsk)
        before_e = Matrix(rd.pnl.emsk)
        rd2 = ReturnsResult(; nx = rd.nx, X = fill(0.01, size(rd.X)), ts = rd.ts,
                            pnl = rd.pnl)
        @test Matrix(rd2.pnl.amsk) == before_a
        @test Matrix(rd2.pnl.emsk) == before_e
        # The snapshot still records the Held Gap the rewritten values no longer show.
        @test !all(rd2.pnl.emsk)
        @test all(isfinite, rd2.X)
    end

    @testset "a caller's Panel Fields are kept and only the masks are the layer's" begin
        # A static panel joins at the panel's observation count, lazily lifted.
        stat = AssetPanel(; pf = [NumericPanelField(; name = "mcap", vals = ones(N59))])
        prs = price_ingestion(PriceIngestion(), X59; pnl = stat)
        rds = prices_to_returns(ptr59, prs)
        @test length(rds.pnl.pf) == 1
        @test PortfolioOptimisers.panel_field(rds.pnl, "mcap") isa NumericPanelField
        @test size(PortfolioOptimisers.panel_field(rds.pnl, "mcap").vals) == (m, N59)
        @test Matrix(rds.pnl.amsk) == Matrix(rd.pnl.amsk)
        @test !PortfolioOptimisers.panel_is_static(rds.pnl)

        # A time-varying panel keeps its fields, and its masks are replaced rather than kept:
        # a declared estimation mask marking a non-finite return estimable cannot survive.
        tv = AssetPanel(; pf = [NumericPanelField(; name = "mcap", vals = ones(T59, N59))],
                        amsk = trues(T59, N59), emsk = trues(T59, N59))
        prt = price_ingestion(PriceIngestion(), X59; pnl = tv)
        rdt = prices_to_returns(ptr59, prt)
        @test length(rdt.pnl.pf) == 1
        @test !all(rdt.pnl.emsk)
        @test Matrix(rdt.pnl.emsk) == Matrix(rd.pnl.emsk)
    end
end

@testset "The layer's span bounds the layer's fill" begin
    # `PriceGapFill` asks the carrier for its Listing Span rather than deriving one, and a
    # carrier the layer built answers. That is what keeps a fill off a window's edge: a
    # window-local derivation reads `d`'s suspension at the edge as an inception there.
    pr = price_ingestion(PriceIngestion(), X59)
    @test PortfolioOptimisers.carrier_listing_span(pr) === pr.span
    @test isnothing(PortfolioOptimisers.carrier_listing_span(PricesResult(; X = X59)))

    res = fit_preprocessing(PriceGapFill(), pr)
    pf = apply_preprocessing(res, pr)
    # The fill states a price and not a listing, so the span rides through untouched and the
    # Span Rule reads the same listing off the filled panel as off the raw one.
    @test pf.span === pr.span
    @test Matrix(listing_span(values(pf.X))) == Matrix(pr.span)
    # `d`'s interior gap is filled, and neither `b`'s inception nor `c`'s delisting is.
    @test all(isfinite, view(values(pf.X), 18:20, 4))
    @test all(!isfinite, view(values(pf.X), 1:8, 2))
    @test all(!isfinite, view(values(pf.X), 33:40, 3))

    # A filled Held Gap is estimable, because a caller who filled has said the asset traded.
    rdf = prices_to_returns(ptr59, pf)
    @test all(view(Matrix(rdf.pnl.emsk), :, 4))
    @test !all(view(Matrix(rd59.pnl.emsk), :, 4))
end

@testset "An Asset Panel with no Panel Field states a universe and nothing else" begin
    a = trues(5, 3)
    a[1, 1] = false
    e = copy(a)
    e[2, 2] = false
    pnl = AssetPanel(; amsk = a, emsk = e)
    @test isempty(pnl.pf)
    @test PortfolioOptimisers.panel_axes(pnl) == (5, 3)
    @test !PortfolioOptimisers.panel_is_static(pnl)

    # It slices like any other panel, and the slice is still field-less.
    v = PortfolioOptimisers.port_opt_view(pnl, 2:4, [1, 3])
    @test isempty(v.pf)
    @test size(v.amsk) == (3, 2)

    # It derives a Feature Matrix with no column, because it holds no feature data.
    nz, Z = PortfolioOptimisers.panel_feature_matrix(pnl)
    @test isempty(nz)
    @test size(Z) == (5, 3, 0)
    @test isempty(PortfolioOptimisers.panel_feature_names(pnl))

    # A panel with neither a field nor a mask carries nothing at all and is refused.
    @test_throws PortfolioOptimisers.IsEmptyError AssetPanel()
    @test_throws PortfolioOptimisers.IsEmptyError AssetPanel(;
                                                             pf = PortfolioOptimisers.AbstractPanelField[])
    # A static panel still needs a field: nothing else would state its asset axis.
    @test PortfolioOptimisers.panel_axes([NumericPanelField(; name = "m", vals = ones(3))],
                                         nothing) == (3,)
end

@testset "A carrier the layer did not build states no universe" begin
    # No span and no gap: nothing to state, and no panel is emitted. This is what keeps
    # `pnl === nothing` meaning one thing only.
    @test isnothing(prices_to_returns(ptr59, PricesResult(; X = Xg59)).pnl)

    # No span and gaps: still no universe, and still no panel. ADR 0133 deleted the
    # window-local derivation rather than rewording it -- a delisting straddling the window
    # end reads there as an asset that was never listed, so the branch answered a question it
    # could not answer correctly. The gaps are carried either way, and with no panel the
    # Coverage Universe reads finiteness alone.
    bare = PricesResult(; X = X59)
    rdw = prices_to_returns(ptr59, bare)
    @test isnothing(rdw.pnl)
    @test size(rdw.X) == (T59 - 1, N59)
    @test any(!isfinite, rdw.X)
    # The rule is total: two methods, and neither of them warns or refuses.
    @test PortfolioOptimisers.returns_universe_masks(nothing, rdw.X) === (nothing, nothing)
    @test length(methods(PortfolioOptimisers.returns_universe_masks)) == 2

    # The contradiction `assert_span_convertible` reported cannot be written any more: there
    # is no keyword that would delete the gaps a span describes.
    @test !isdefined(PortfolioOptimisers, :assert_span_convertible)
    pr = price_ingestion(PriceIngestion(), X59)
    rdn = prices_to_returns(PricesToReturns(), pr)
    @test !all(rdn.pnl.emsk)

    # A span that does not fit the price clock is refused outright.
    @test_throws DimensionMismatch PricesResult(; X = X59, span = trues(T59 + 1, N59))
    @test_throws DimensionMismatch PricesResult(; X = X59, span = trues(T59, N59 + 1))
    # The conversion checks the shape again, because a `span` reaches it on a carrier that
    # a caller may have rebuilt around a different price table.
    @test_throws DimensionMismatch PortfolioOptimisers.assert_span_shape(trues(T59,
                                                                               N59 + 1),
                                                                         T59, N59)
end

@testset "assert_universe_aligned narrows to a provenance check" begin
    pr = price_ingestion(PriceIngestion(), X59)
    pipe = Pipeline(; steps = (ptr59, EmpiricalPrior(), EqualWeighted()))
    res = fit(pipe, pr)
    rd = res.ctx.returns
    @test !isnothing(rd.pnl)
    @test PortfolioOptimisers.assert_universe_aligned(res, rd) === nothing

    # A window carrying a different asset set is still refused by name.
    other = ReturnsResult(; nx = nx59[1:3], X = rd.X[:, 1:3])
    @test_throws ArgumentError PortfolioOptimisers.assert_universe_aligned(res, other)

    # And so is one that states no universe where the fitted context states one, which is the
    # half the check gained: the two did not come from one ingestion.
    stripped = ReturnsResult(; nx = rd.nx, X = rd.X, ts = rd.ts)
    @test_throws ArgumentError PortfolioOptimisers.assert_universe_aligned(res, stripped)

    # A fitted context with no returns slot has nothing to compare against, so the check
    # is silent rather than guessing.
    price_only = fit(Pipeline(; steps = (MissingDataFilter(),)), pr)
    @test isnothing(price_only.ctx.returns)
    @test PortfolioOptimisers.assert_universe_aligned(price_only, rd) === nothing
end

@testset "A gapped price table reaches a walk-forward fold with nothing hand-built" begin
    # The destination of map #955: no hand-built ReturnsResult, and no hand-built mask.
    pr = price_ingestion(PriceIngestion(), X59)
    pipe = Pipeline(; steps = (ptr59, EmpiricalPrior(), EqualWeighted()))
    iwf = IndexWalkForward(20, 8)
    p = cross_val_predict(pipe, pr, iwf)
    @test length(p.pred) >= 2
    rd = prices_to_returns(ptr59, pr)
    cv = split(iwf, pr)
    for (k, f) in pairs(p.pred)
        # Reduce-and-expand answers on the full universe, and the panel the layer emitted is
        # what drove the reduction: no mask was written by hand anywhere on this path.
        @test length(f.res.w) == N59
        @test all(isfinite, f.res.w)
        @test !isnothing(f.res.pr.pnl)

        # The fold's Investable Mask is the Coverage Universe the ingested panel states, and
        # every asset outside it holds exactly zero rather than a small number.
        tr = cv.train_idx[k]
        rows = tr[1]:(last(tr) - 1)
        cm = PortfolioOptimisers.coverage_mask(view(rd.X, rows, :),
                                               PortfolioOptimisers.port_opt_view(rd.pnl,
                                                                                 rows,
                                                                                 1:N59))
        @test f.res.imsk == cm
        @test all(iszero, view(f.res.w, .!cm))
        @test isapprox(sum(f.res.w), 1; rtol = 1e-8)
    end

    # The universe each fold states is the panel-wide one cut to the window, never one
    # re-derived from it: `c` is still listed inside a window that ends before its delisting.
    for tr in split(iwf, rd).train_idx
        v = PortfolioOptimisers.port_opt_view(rd, tr, 1:N59)
        @test Matrix(v.pnl.amsk) == Matrix(rd.pnl.amsk)[tr, :]
    end
end

@testset "The conversion deletes nothing, and the universe is stated rather than survived" begin
    # Issue #985, ADR 0133. The evidence the ticket asks to reproduce, re-measured here.
    # A 40 x 4 table, one asset delisted after observation 30 and one suspended for three
    # observations, and no leading gap: the question is what the conversion does with them,
    # not what the Span Rule reads.
    T, N = 40, 4
    ts = collect(Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(T - 1)))
    nx = ["a", "b", "c", "d"]
    P = 100.0 .+ cumsum(randn(StableRNG(985), T, N) ./ 10; dims = 1)
    P[31:40, 3] .= NaN          # c delists after observation 30
    P[18:20, 4] .= NaN          # d is suspended for three observations
    X = TimeArray(ts, P, nx)
    slv985 = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                    settings = Dict("verbose" => false),
                    check_sol = (; allow_local = true, allow_almost = true))

    # The conversion keeps the whole clock and the whole universe, and the gaps land where
    # the arithmetic puts them: 10 + 1 non-finite returns for the delisting and 3 + 1 for the
    # suspension, 14 in all.
    pr = price_ingestion(PriceIngestion(), X)
    rd = prices_to_returns(PricesToReturns(), pr)
    @test size(rd.X) == (T - 1, N)
    @test count(!isfinite, rd.X) == 14
    @test !isnothing(rd.pnl)

    # The universe the panel states excludes exactly the two assets the window cannot
    # estimate, and they hold zero rather than a small number.
    res = optimise(MeanRisk(; obj = MinimumRisk(),
                            opt = JuMPOptimiser(; pe = prior(EmpiricalPrior(), rd),
                                                slv = slv985)))
    @test res.imsk == Bool[1, 1, 0, 0]
    @test isapprox(res.w, [0.5217, 0.4783, 0.0, 0.0]; atol = 1e-4)
    @test iszero(res.w[3]) && iszero(res.w[4])

    # What the deleted path did, reproduced by hand: delete every observation row that holds
    # a gap and the table is 26 of 39 rows, and the two dead names take 43% of the book
    # because nothing is left to say they are dead. Ingestion is the only door, so the
    # universe is still stated -- and what it states is the falsehood the deletion
    # manufactured: every asset listed and estimable at every surviving observation.
    kept = [t for t in 1:T if all(isfinite, view(P, t, :))]
    rdd = prices_to_returns(TimeArray(ts[kept], P[kept, :], nx))
    @test size(rdd.X, 1) == 26
    @test all(rdd.pnl.amsk)
    @test all(rdd.pnl.emsk)
    resd = optimise(MeanRisk(; obj = MinimumRisk(),
                             opt = JuMPOptimiser(; pe = prior(EmpiricalPrior(), rdd),
                                                 slv = slv985)))
    @test resd.w[3] + resd.w[4] > 0.4

    # The two spellings are one gap, and with `impute_method` deleted there is no path left
    # on which they could differ.
    Pm = Matrix{Union{Missing, Float64}}(P)
    Pm[.!isfinite.(P)] .= missing
    rdm = prices_to_returns(PricesToReturns(),
                            price_ingestion(PriceIngestion(), TimeArray(ts, Pm, nx)))
    @test isequal(rdm.X, rd.X)
    @test Matrix(rdm.pnl.emsk) == Matrix(rd.pnl.emsk)
end

# Map #955, ADR 0133. The conversion takes a carrier, so the implied volatilities reach it as a
# field rather than as a keyword, and the one thing it does with them is project them onto the
# returns clock: a return costs the first observation unless `padding` keeps it.
@testset "the carrier's implied volatilities are cut to the returns clock" begin
    iv59 = TimeArray(collect(ts59), fill(0.2, T59, N59), nx59)
    pri = price_ingestion(PriceIngestion(), X59; iv = iv59, ivpa = 1.5)

    rd = prices_to_returns(pri)
    @test size(rd.iv) == (T59 - 1, N59)
    @test all(rd.iv .== 0.2)
    @test rd.ivpa == 1.5

    # Under `padding` the returns keep the price clock, so the implied volatilities do too.
    @test size(prices_to_returns(pri; padding = true).iv) == (T59, N59)

    # A per-asset adjustment is checked against the asset axis rather than carried blindly.
    prv = price_ingestion(PriceIngestion(), X59; iv = iv59, ivpa = fill(1.5, N59))
    @test length(prices_to_returns(prv).ivpa) == N59

    # A carrier built by hand can state implied volatilities that do not cover the returns
    # clock, and the conversion refuses rather than padding: only the door pads.
    short = PricesResult(; X = X59, iv = iv59[collect(ts59)[1:(T59 - 2)]])
    @test_throws ArgumentError prices_to_returns(short)
end

# Map #955, ADR 0135, issue #1002. An absence is carried on every axis the layer touches: an
# implied volatility the source is silent on is padded `NaN` at the door, both carriers admit
# one, and the estimator that reads the series is what excludes the asset — #995 built that
# narrowing on the keyword path, and this is the carried path reaching it.
@testset "an absent implied volatility is carried to the estimator that excludes it" begin
    using Statistics
    Tg, Ng = 12, 3
    tsg = Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(Tg - 1))
    nxg = ["A", "B", "C"]
    Xg = TimeArray(collect(tsg), 100.0 .+ cumsum(randn(StableRNG(1002), Tg, Ng); dims = 1),
                   nxg)
    ivfull = fill(0.2, Tg, Ng)

    # `C`'s implied volatility starts three days after the assets. The source spells that
    # as three `NaN` cells on the full clock — a gap the source states rather than an
    # observation the layer pads, so nothing is reported — and the returns carrier holds
    # the two that survive the conversion.
    ivc = TimeArray(collect(tsg)[4:end], ivfull[4:end, 3:3], nxg[3:3])
    ivab = TimeArray(collect(tsg), ivfull[:, 1:2], nxg[1:2])
    ivr = TimeSeries.merge(ivab, ivc; method = :left)
    @test findall(isnan, values(ivr)[:, 3]) == [1, 2, 3]
    prg = @test_logs price_ingestion(PriceIngestion(; strict = true), Xg; iv = ivr,
                                     ivpa = 1.2)
    @test TimeSeries.timestamp(prg.iv) == collect(tsg)
    @test findall(isnan, values(prg.iv)[:, 3]) == [1, 2, 3]
    @test all(isfinite, values(prg.iv)[:, 1:2])
    rdg = prices_to_returns(prg)
    @test size(rdg.iv) == (Tg - 1, Ng)
    @test findall(isnan, rdg.iv[:, 3]) == [1, 2]
    @test rdg.ivpa == 1.2

    # The estimator narrows its Coverage Universe to the columns whose implied
    # volatilities are complete: `C` takes the `NaN` row and column an absent return takes,
    # and the surviving block is the fit on `A` and `B` alone.
    ce = ImpliedVolatility(; alg = ImpliedVolatilityPremium())
    for f in (cov, cor)
        m = f(ce, rdg.X, rdg.pnl; iv = rdg.iv, ivpa = rdg.ivpa)
        @test all(isnan, view(m, 3, :))
        @test all(isnan, view(m, :, 3))
        @test all(isfinite, view(m, 1:2, 1:2))
        @test view(m, 1:2, 1:2) == f(ce, rdg.X[:, 1:2]; iv = rdg.iv[:, 1:2], ivpa = 1.2)
    end
    # And the same carrier reaches a prior, which is the path a Pipeline takes.
    pr = prior(EmpiricalPrior(; ce = ce), rdg)
    @test all(isnan, view(pr.sigma, 3, :))
    @test all(isfinite, view(pr.sigma, 1:2, 1:2))

    # A complete implied-volatility surface pads nothing and fits every column, so the
    # carried path and the hand-built path agree entry for entry.
    prf = @test_logs price_ingestion(PriceIngestion(; strict = true), Xg;
                                     iv = TimeArray(collect(tsg), ivfull, nxg), ivpa = 1.2)
    rdf = prices_to_returns(prf)
    @test all(isfinite, cov(ce, rdf.X, rdf.pnl; iv = rdf.iv, ivpa = rdf.ivpa))

    # The carriers' guard is non-negative WHERE A VALUE IS PRESENT. A `NaN` is an absence
    # and passes; so does a `missing`, the spelling a hand-built carrier may hold, which
    # the conversion unifies to `NaN` before any reader sees it; a negative value is
    # refused by both carriers, and an infinite one is a present value and passes.
    ivneg = copy(ivfull)
    ivneg[5, 2] = -0.1
    @test_throws DomainError PricesResult(; X = Xg,
                                          iv = TimeArray(collect(tsg), ivneg, nxg))
    @test_throws DomainError ReturnsResult(; nx = nxg, X = rdg.X, ts = rdg.ts,
                                           iv = ivneg[2:end, :], ivpa = 1.2)
    ivm = Matrix{Union{Missing, Float64}}(ivfull)
    ivm[6, 1] = missing
    prmis = PricesResult(; X = Xg, iv = TimeArray(collect(tsg), ivm, nxg), ivpa = 1.2)
    rdm = prices_to_returns(prmis)
    @test eltype(rdm.iv) === Float64
    @test findall(isnan, rdm.iv[:, 1]) == [5]
    ivmn = copy(ivm)
    ivmn[7, 3] = -1.0
    @test_throws DomainError PricesResult(; X = Xg, iv = TimeArray(collect(tsg), ivmn, nxg))
    # An infinity is neither a volatility nor the marker of an absence, so it is refused
    # on the same rule as an infinite return, at every carrier that reads the surface.
    ivinf = copy(ivfull)
    ivinf[5, 2] = Inf
    @test_throws DomainError PricesResult(; X = Xg,
                                          iv = TimeArray(collect(tsg), ivinf, nxg))
    @test_throws DomainError ReturnsResult(; nx = nxg, X = rdg.X, ts = rdg.ts,
                                           iv = ivinf[2:end, :], ivpa = 1.2)
    @test_throws DomainError PortfolioOptimisers.assert_nonneg_where_present([-1.0, NaN],
                                                                             :iv)
    @test_throws DomainError PortfolioOptimisers.assert_nonneg_where_present([Inf, NaN],
                                                                             :iv)
    @test_throws DomainError PortfolioOptimisers.assert_nonneg_where_present([-Inf, NaN],
                                                                             :iv)
    @test isnothing(PortfolioOptimisers.assert_nonneg_where_present([missing, NaN, 0.0,
                                                                     1.0], :iv))
    # A wrong-shaped surface reports its shape, before any value is read.
    @test_throws DimensionMismatch ReturnsResult(; nx = nxg, X = rdg.X, ts = rdg.ts,
                                                 iv = ivinf, ivpa = 1.2)
    # An empty surface is still refused, before the sign is read.
    @test_throws PortfolioOptimisers.IsEmptyError ReturnsResult(; nx = nxg, X = rdg.X,
                                                                ts = rdg.ts,
                                                                iv = zeros(0, Ng),
                                                                ivpa = 1.2)
end
