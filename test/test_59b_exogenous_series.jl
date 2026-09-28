#=
The Exogenous Series (#1366, ADR 0184): named series that belong to no asset. A `PricesResult`
holds their levels, and `prices_to_returns` converts them with the return rule of the assets, so a
`ReturnsResult` holds their returns. Every view and every rebuild carries them, and they never add
or drop an observation of the assets.
=#
using Dates, Statistics

@testset "The Exogenous Series" begin
    PO = PortfolioOptimisers
    cae = PO.ConflictingArgumentError
    ts = Date(2020, 1, 1):Day(1):Date(2020, 1, 10)
    rng = StableRNG(1_366)
    Xp = TimeArray(ts, 100 .+ cumsum(randn(rng, 10, 3); dims = 1), ["A", "B", "C"])
    lv = 1 .+ 0.01 .* cumsum(randn(rng, 10, 2); dims = 1)
    E = TimeArray(ts, lv, ["EUR", "JPY"])
    simple(L) = L[2:end, :] ./ L[1:(end - 1), :] .- 1
    logr(L) = log.(L[2:end, :] ./ L[1:(end - 1), :])

    @testset "A PricesResult holds levels, and a ReturnsResult holds their returns" begin
        pr = PricesResult(; X = Xp, E = E)
        rs = prices_to_returns(pr)
        @test rs.ne == ["EUR", "JPY"]
        @test rs.E ≈ simple(lv)
        @test size(rs.E, 1) == size(rs.X, 1)
        @test prices_to_returns(pr; ret_method = :log).E ≈ logr(lv)
        # The padding of the assets reaches E too, so the two keep one row count.
        rp = prices_to_returns(pr; padding = true)
        @test size(rp.E) == (10, 2)
        @test all(isnan, rp.E[1, :])
        # Without E the conversion is unchanged.
        @test isnothing(prices_to_returns(PricesResult(; X = Xp)).E)
    end

    @testset "The ingestion aligns E by timestamp and never moves the clock of the assets" begin
        # E is silent on the 4th, and states a day the assets do not.
        tsE = [ts[1:3]; ts[5:10]; Date(2021, 1, 1)]
        Eg = TimeArray(tsE, vcat(lv[[1, 2, 3, 5, 6, 7, 8, 9, 10], :], [9.0 9.0]),
                       ["EUR", "JPY"])
        pr = @test_logs (:warn, r"`E` at 1 of 10 observations") min_level = Logging.Warn price_ingestion(PriceIngestion(),
                                                                                                         Xp;
                                                                                                         E = Eg)
        @test TimeSeries.timestamp(pr.E) == TimeSeries.timestamp(pr.X)
        @test all(isnan, values(pr.E)[4, :])
        rd = prices_to_returns(pr)
        @test size(rd.X, 1) == 9
        # A gap reaches the two returns that read it, and no other row.
        @test findall(isnan, rd.E[:, 1]) == [3, 4]
        @test isfinite(rd.E[5, 1])
        @test_throws ArgumentError price_ingestion(PriceIngestion(; strict = true), Xp;
                                                   E = Eg)
        # A missing level is an absence, as a missing price is.
        Em = TimeArray(ts,
                       Union{Missing, Float64}[lv[1:4, :]; missing missing; lv[6:10, :]],
                       ["EUR", "JPY"])
        rm = prices_to_returns(price_ingestion(PriceIngestion(), Xp; E = Em))
        @test findall(isnan, rm.E[:, 2]) == [4, 5]
        # The ingestion of a PricesResult carries its E.
        @test price_ingestion(PriceIngestion(), PricesResult(; X = Xp, E = E)).E == E
    end

    @testset "A name of E belongs to no other series" begin
        @test_throws cae price_ingestion(PriceIngestion(), Xp;
                                         E = TimeArray(ts, lv, ["A", "JPY"]))
        @test_throws cae prices_to_returns(PricesResult(; X = Xp,
                                                        F = TimeArray(ts, lv, ["EUR", "f"]),
                                                        E = E))
        # A PricesResult built by hand states one clock, and E must state it.
        @test_throws cae prices_to_returns(PricesResult(; X = Xp,
                                                        E = TimeArray(ts[2:10], lv[2:10, :],
                                                                      ["EUR", "JPY"])))
        @test_throws PO.IsEmptyError PricesResult(; X = Xp, E = E[[Date(2030, 1, 1)]])
    end

    @testset "The online conversion gives the batch E" begin
        tsE = [ts[1:3]; ts[5:10]]
        Eg = TimeArray(tsE, lv[[1, 2, 3, 5, 6, 7, 8, 9, 10], :], ["EUR", "JPY"])
        for pr in
            (PricesResult(; X = Xp, E = E), price_ingestion(PriceIngestion(), Xp; E = Eg)),
            alg in (nothing, CatchUpGapReturn()), m in (:simple, :log)

            ptr = PricesToReturns(; ret_method = m, gap_return_alg = alg)
            s1, r1 = PO.partial_fit_transform(ptr, PO.port_opt_view(pr, 1:4))
            s2, r2 = PO.partial_fit_transform(s1, PO.port_opt_view(pr, 5:10))
            @test isequal(vcat(r1.E, r2.E), prices_to_returns(ptr, pr).E)
            @test r2.ne == ["EUR", "JPY"]
        end
    end

    @testset "A view cuts E by rows, and an asset view passes it through" begin
        rd = prices_to_returns(PricesResult(; X = Xp, E = E))
        v = PO.port_opt_view(rd, 2:5, [1, 3])
        @test v.E == rd.E[2:5, :]
        @test v.ne == rd.ne
        va = PO.port_opt_view(rd, [2])
        @test va.E === rd.E
        pv = PO.port_opt_view(PricesResult(; X = Xp, E = E), 3:6, [2])
        @test values(pv.E) == lv[3:6, :]
        @test TimeSeries.colnames(pv.E) == [:EUR, :JPY]
    end

    @testset "The constructors check E against the other series" begin
        X = randn(StableRNG(2), 5, 2)
        @test_throws PO.IsNothingError ReturnsResult(; nx = ["a", "b"], X = X,
                                                     E = ones(5, 1))
        @test_throws DimensionMismatch ReturnsResult(; nx = ["a", "b"], X = X, ne = ["u"],
                                                     E = ones(4, 1))
        @test_throws DimensionMismatch ReturnsResult(; nx = ["a", "b"], X = X,
                                                     ne = ["u", "v"], E = ones(5, 1))
        @test_throws DimensionMismatch ReturnsResult(; nf = ["f"], F = ones(5, 1),
                                                     ne = ["u"], E = ones(4, 1))
        @test_throws DimensionMismatch ReturnsResult(; nx = ["a", "b"], X = X,
                                                     ts = collect(ts[1:5]), ne = ["u"],
                                                     E = ones(4, 1))
        rd = ReturnsResult(; nx = ["a", "b"], X = X, ne = ["u"], E = [1.0; NaN; 3; 4; 5;;])
        @test isnan(rd.E[2])
    end

    @testset "Every rebuild of the returns data carries E" begin
        X = randn(StableRNG(3), 6, 4) ./ 100
        Ed = randn(StableRNG(4), 6, 1) ./ 100
        rd = ReturnsResult(; nx = ["a", "b", "c", "d"], X = X, nb = ["m"], B = X[:, 1],
                           ne = ["u"], E = Ed)
        @test returns_result_picker(rd, true).E === Ed
        rdv = PO.vcat_observations(PO.port_opt_view(rd, 1:3, :),
                                   PO.port_opt_view(rd, 4:6, :))
        @test rdv.E == Ed
        @test_throws ArgumentError PO.vcat_observations(PO.port_opt_view(rd, 1:3, :),
                                                        ReturnsResult(; nx = rd.nx,
                                                                      X = X[4:6, :],
                                                                      nb = ["m"],
                                                                      B = X[4:6, 1],
                                                                      ne = ["v"],
                                                                      E = Ed[4:6, :]))
        # The fold context of an optimiser keeps E, and gives it back.
        st = PO.partial_fit!(PO.ReturnsBufferState(), PO.port_opt_view(rd, 1:3, :);
                             own_returns = true)
        st = PO.partial_fit!(st, PO.port_opt_view(rd, 4:6, :); own_returns = true)
        @test PO.returns_result(st, st.X).E == Ed
        @test PO.returns_result(copy(st), st.X).ne == ["u"]
        @test PO.port_opt_view(st, [1, 2]).E.n == 6
        @test_throws ArgumentError PO.partial_fit!(st,
                                                   ReturnsResult(; nx = rd.nx,
                                                                 X = X[1:1, :], nb = ["m"],
                                                                 B = X[1:1, 1]);
                                                   own_returns = true)
        # The outer problem of a meta-optimiser carries E over its synthetic assets.
        pr = prior(EmpiricalPrior(), rd)
        cls = [[1, 2], [3, 4]]
        res = [optimise(EqualWeighted(), PO.port_opt_view(rd, cl)) for cl in cls]
        W = [0.5 0.0; 0.5 0.0; 0.0 0.5; 0.0 0.5]
        rdo = PO.predict_outer_returns(nothing, nothing, PO.ClusterUniverse(cls), rd, pr,
                                       nothing, W, res)
        @test rdo.E === Ed
        @test rdo.ne == ["u"]
        # A prediction and its multi-period stack carry E.
        pred = PO.PredictionReturnsResult(; nx = ["p"], X = X[:, 1], ne = ["u"], E = Ed)
        @test pred.E === Ed
        @test_throws DimensionMismatch PO.PredictionReturnsResult(; nx = ["p"], X = X[:, 1],
                                                                  ne = ["u"],
                                                                  E = Ed[1:5, :])
        # A population carries one series per member, and E rides beside each of them.
        popu = PO.PredictionReturnsResult(; nx = ["p"], X = [X[:, 1], X[:, 2]], ne = ["u"],
                                          E = Ed, ts = collect(ts[1:6]))
        @test popu.E === Ed
        @test_throws DimensionMismatch PO.PredictionReturnsResult(; nx = ["p"],
                                                                  X = [X[:, 1], X[1:5, 2]],
                                                                  ne = ["u"], E = Ed)
        @test_throws DimensionMismatch PO.PredictionReturnsResult(; nf = ["f"], F = Ed,
                                                                  ne = ["u"], E = Ed,
                                                                  ts = collect(ts[1:5]))
    end

    @testset "currency_excess_index builds the textbook currency excess return" begin
        rngc = StableRNG(5)
        S = 1 .+ 0.01 .* cumsum(randn(rngc, 10, 2); dims = 1)
        rc = 0.001 .* rand(rngc, 10, 2)
        rb = 0.0005 .* rand(rngc, 10)
        K = cumprod(1 .+ rc; dims = 1)
        Kb = cumprod(1 .+ rb)
        fx = TimeArray(ts, S, ["EUR", "JPY"])
        # The cash indices come in another column order; the helper reads them by name.
        cash = TimeArray(ts, K[:, [2, 1]], ["JPY", "EUR"])
        base = TimeArray(ts, reshape(Kb, :, 1), ["USD"])
        I = currency_excess_index(fx, cash, base)
        @test TimeSeries.colnames(I) == [:EUR, :JPY]
        RS = S[2:end, :] ./ S[1:(end - 1), :] .- 1
        @test simple(values(I)) ≈ (1 .+ RS) .* (1 .+ rc[2:end, :]) ./ (1 .+ rb[2:end]) .- 1
        # With log returns the split of a base-currency excess return is exact.
        RL = 0.01 .* randn(rngc, 9)
        RB = (1 .+ RL) .* (1 .+ RS[:, 1]) .- 1
        @test log.(1 .+ RL) .- log.(1 .+ rc[2:end, 1]) .+ logr(values(I))[:, 1] ≈
              log.(1 .+ RB) .- log.(1 .+ rb[2:end])
        # It converts as one more level series.
        rd = prices_to_returns(PricesResult(; X = Xp, E = I); ret_method = :log)
        @test rd.E ≈ logr(values(I))
        @test_throws cae currency_excess_index(fx, cash[ts[2:10]], base)
        @test_throws cae currency_excess_index(fx, cash, base[ts[2:10]])
        @test_throws ArgumentError currency_excess_index(fx, cash[:JPY], base)
        @test_throws DimensionMismatch currency_excess_index(fx, cash, cash)
    end
end
