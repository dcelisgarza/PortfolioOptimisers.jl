# Two half-implemented estimators, used to reach the dispatch-error stubs of the
# preprocessing interface. Neither implements `fit_preprocessing` or `apply_preprocessing`,
# which is the mistake the stubs exist to name.
struct UnimplementedPreprocessing <: PortfolioOptimisers.AbstractPreprocessingEstimator end
struct UnimplementedPreprocessingResult <: PortfolioOptimisers.AbstractPreprocessingResult end
# A price-level carrier that implements no `port_opt_view`, used to reach the fallback of
# the family.
struct BarePricesResult <: PortfolioOptimisers.AbstractPricesResult end
# A missing-data rule that keeps the last asset, used to prove that a new rule plugs in
# through `fit_universe`.
struct KeepLastAsset <: PortfolioOptimisers.AbstractMissingDataFilterAlgorithm end
function PortfolioOptimisers.fit_universe(::KeepLastAsset, ::MissingDataFilter,
                                          pr::PricesResult)
    return [size(values(pr.X), 2)]
end
# A Gap Return algorithm that ignores the invariant entirely and answers the whole column with
# one sentinel. `apply_gap_return` reads back only the writable cells, so the invariant is the
# driver's and not the algorithm's, and this is what proves it.
struct RogueGapReturn <: PortfolioOptimisers.AbstractGapReturnAlgorithm end
function PortfolioOptimisers.gap_return(::RogueGapReturn, ::AbstractVector,
                                        r::AbstractVector, ::Symbol)
    return fill(-99.0, length(r))
end
# A Gap Return algorithm that drops the last cell, used to reach the length check of
# `apply_gap_return`.
struct ShortGapReturn <: PortfolioOptimisers.AbstractGapReturnAlgorithm end
function PortfolioOptimisers.gap_return(::ShortGapReturn, ::AbstractVector,
                                        r::AbstractVector, ::Symbol)
    return r[1:(end - 1)]
end
include(joinpath(@__DIR__, "asset_panel_fixture.jl"))
@testset "Tools tests" begin
    using Test, PortfolioOptimisers, DataFrames, TimeSeries, Dates, Random, StableRNGs, CSV,
          Statistics, LinearAlgebra

    @testset "prices_to_returns" begin
        rng = StableRNG(987654321)
        Px = rand(rng, 101, 20)
        Fx = rand(rng, 101, 5)

        dfx = DataFrame(Px, :auto)
        dfx.all_missing = fill(NaN, nrow(dfx))
        dfx[!, :date] = (today() - Day(100)):Day(1):today()
        Px = TimeArray(dfx; timestamp = :date)

        dfy = DataFrame(Fx, [:fx1, :fx2, :fx3, :fx4, :fx5])
        # Named apart from the asset table's gap column on purpose, and now refused outright
        # if it were not: a name shared by `X` and `F` cannot say which series a column came
        # from, so both doors raise a `ConflictingArgumentError`. Issue #990; the released
        # thresholds hid it by deleting both columns, and
        # `test_59_ingestion_layer.jl` pins the refusal.
        dfy.all_missing_f = fill(NaN, nrow(dfy))
        dfy[!, :date] = (today() - Day(100)):Day(1):today()
        Py = TimeArray(dfy; timestamp = :date)

        # The `all_missing` column of each table used to be deleted by
        # `missing_row_percent`, and the oracles below were recorded on the twenty assets
        # and five factors that survived it. The conversion now deletes nothing, so the
        # gap column arrives as a column of gaps and the oracle covers the columns beside
        # it, bit for bit. #1414 recorded both files again from `(p_t - p_{t-1}) / p_{t-1}`,
        # which is the correctly rounded return on every cell of this fixture.
        rd = prices_to_returns(price_ingestion(PriceIngestion(), Px; F = Py))
        ts1 = rd.ts
        X1 = rd.X
        F1 = rd.F
        @test rd.nx[end] == "all_missing"
        @test rd.nf[end] == "all_missing_f"
        @test all(!isfinite, view(X1, :, 21))
        @test all(!isfinite, view(F1, :, 6))

        df = CSV.read(joinpath(@__DIR__, "assets/prices_to_returns_X_F.csv.gz"), DataFrame)
        # The oracle stacks the two settings the two calls used to differ in; they always
        # agreed, so one call answers both halves.
        oracle = Matrix(df)
        @test oracle[1:100, :] == oracle[101:200, :]
        @test hcat(view(X1, :, 1:20), view(F1, :, 1:5)) == view(oracle, 1:100, :)

        rd = prices_to_returns(Px)
        ts3 = rd.ts
        X3 = rd.X
        @test isnothing(rd.F)
        @test all(!isfinite, view(X3, :, 21))

        df = CSV.read(joinpath(@__DIR__, "assets/prices_to_returns_X.csv.gz"), DataFrame)
        oracle = Matrix(df)
        @test oracle[1:100, :] == oracle[101:200, :]
        @test view(X3, :, 1:20) == view(oracle, 1:100, :)

        @test dfy[2:end, :date] == ts1 == ts3
    end

    @testset "feature matrix through prices, returns and views" begin
        # The carried feature matrix is canonically assets-major: `assets × features` when
        # static, `observations × assets × features` when time-varying. `port_opt_view` has
        # no `dims` keyword to declare an orientation with, so the layout is fixed and the
        # constructor rejects a transposed one instead of letting it reach a fold.
        port_opt_view = PortfolioOptimisers.port_opt_view

        ts0 = Date(2020, 1, 1):Day(1):Date(2020, 1, 10)
        rng = StableRNG(24681357)
        Pv = 100 .+ cumsum(rand(rng, 10, 3); dims = 1)
        Pv[:, 2] .= NaN                      # asset "B" is entirely missing
        Px = TimeArray(collect(ts0), Pv, ["A", "B", "C"])

        Zs = Float64[1 2; 3 4; 5 6]          # assets × features
        # Z3[i, j, k] == i + 10(j - 1) + 30(k - 1), so every entry names its own position.
        Z3 = reshape(Float64.(1:60), 10, 3, 2)
        nz = ["f1", "f2"]

        @testset "construction validation" begin
            X = rand(rng, 5, 3)
            nx = ["A", "B", "C"]
            # A transposed non-square Z is caught, which is the whole point of fixing the
            # layout rather than inferring it.
            @test_throws DimensionMismatch ReturnsResult(; nx = nx, X = X,
                                                         pnl = matrix_panel(nz,
                                                                            rand(rng, 2, 3)))
            @test_throws DimensionMismatch ReturnsResult(; nx = nx, X = X,
                                                         pnl = matrix_panel(nz,
                                                                            rand(rng, 4, 2)))
            @test_throws ArgumentError ReturnsResult(; nx = nx, X = X,
                                                     pnl = matrix_panel(["f", "f"],
                                                                        rand(rng, 3, 2)))
            # A blank never reaches a carrier: `NaN` is a blank, and the default `NoPanelFill`
            # refuses it at the build rather than letting it reach a metric that would map it
            # to a plausible wrong distance.
            @test_throws ArgumentError ReturnsResult(; nx = nx, X = X,
                                                     pnl = matrix_panel(nz,
                                                                        [1.0 NaN;
                                                                         2.0 3.0;
                                                                         4.0 5.0]))
            @test_throws PortfolioOptimisers.IsNonFiniteError ReturnsResult(; nx = nx,
                                                                            X = X,
                                                                            pnl = matrix_panel(nz,
                                                                                               [                                  1.0 Inf;
                                                                                                2.0 3.0;
                                                                                                4.0 5.0]))
            # A time-varying Z binds its leading axis to the observations, and needs both
            # an asset anchor and an observation anchor to bind to.
            @test_throws DimensionMismatch ReturnsResult(; nx = nx, X = X,
                                                         pnl = matrix_panel(nz,
                                                                            rand(rng, 4, 3,
                                                                                 2)))
            @test_throws PortfolioOptimisers.IsNothingError ReturnsResult(; nx = nx,
                                                                          pnl = matrix_panel(nz,
                                                                                             rand(rng,
                                                                                                  5,
                                                                                                  3,
                                                                                                  2)))
            @test_throws PortfolioOptimisers.IsNothingError ReturnsResult(;
                                                                          pnl = matrix_panel(nz,
                                                                                             rand(rng,
                                                                                                  3,
                                                                                                  2)))
            # The price level binds the same two axes to X's columns and rows.
            @test_throws DimensionMismatch PricesResult(; X = Px,
                                                        pnl = matrix_panel(nz,
                                                                           rand(rng, 2, 2)))
            @test_throws DimensionMismatch PricesResult(; X = Px,
                                                        pnl = matrix_panel(nz,
                                                                           rand(rng, 9, 3,
                                                                                2)))
        end

        @testset "both-or-neither evidence states the truth" begin
            # The label reads `!isnothing(x)`, so the value under it must be the value of
            # `!isnothing(x)`. Issue #544: the two lines interpolated `isnothing(x)`, so
            # each `Got` block stated the opposite of the fact that raised the error, and a
            # test on the exception type alone passed with the values inverted.
            nx = ["A", "B", "C"]
            msg = sprint(showerror,
                         try
                             ReturnsResult(; X = [0.01 0.02 0.03; 0.04 0.05 0.06])
                         catch e
                             e
                         end)
            @test occursin("nx cannot be nothing if X is not `nothing`", msg)
            @test occursin("!isnothing(nx) => false", msg)
            @test occursin("!isnothing(X) => true", msg)
            msg = sprint(showerror, try
                             ReturnsResult(; nx = nx)
                         catch e
                             e
                         end)
            @test occursin("X cannot be nothing if nx is not `nothing`", msg)
            @test occursin("!isnothing(nx) => true", msg)
            @test occursin("!isnothing(X) => false", msg)
        end

        @testset "ReturnsResult views" begin
            X = rand(rng, 5, 3)
            nx = ["A", "B", "C"]
            rds = ReturnsResult(; nx = nx, X = X, pnl = matrix_panel(nz, Zs))
            rd3 = ReturnsResult(; nx = nx, X = X,
                                pnl = matrix_panel(nz, reshape(Float64.(1:30), 5, 3, 2)))

            # Two-argument arity: assets only. The static shape has no observation axis;
            # the time-varying one keeps all of its rows.
            vs = port_opt_view(rds, [1, 3])
            @test vs.nx == ["A", "C"]
            @test panel_feature_matrix(vs.pnl)[2] == Zs[[1, 3], :]
            @test panel_feature_matrix(vs.pnl)[1] == nz
            v3 = port_opt_view(rd3, [1, 3])
            @test size(panel_feature_matrix(v3.pnl)[2]) == (5, 2, 2)
            @test panel_feature_matrix(v3.pnl)[2] ==
                  panel_feature_matrix(rd3.pnl)[2][:, [1, 3], :]

            # Four-argument arity: observations first, assets second. A static Z ignores the
            # observation index, exactly as ivpa does.
            vs4 = port_opt_view(rds, 2:4, [1, 3])
            @test panel_feature_matrix(vs4.pnl)[2] == Zs[[1, 3], :]
            v34 = port_opt_view(rd3, 2:4, [1, 3])
            @test size(panel_feature_matrix(v34.pnl)[2]) == (3, 2, 2)
            @test panel_feature_matrix(v34.pnl)[2] ==
                  panel_feature_matrix(rd3.pnl)[2][2:4, [1, 3], :]
            # The third positional index selects factors and must not touch Z.
            rdf = ReturnsResult(; nx = nx, X = X, nf = ["F1", "F2"], F = rand(rng, 5, 2),
                                pnl = matrix_panel(nz, Zs))
            @test panel_feature_matrix(port_opt_view(rdf, 2:4, [1, 3], 1:1).pnl)[2] ==
                  Zs[[1, 3], :]

            # Views compose: a fold of a subset is a subset of the fold.
            @test panel_feature_matrix(port_opt_view(port_opt_view(rd3, [1, 3]), 2:4,
                                                     [1, 2]).pnl)[2] ==
                  panel_feature_matrix(rd3.pnl)[2][2:4, [1, 3], :]
        end

        @testset "features are assets" begin
            # A square phylogeny or adjacency matrix reused as a feature source has features
            # that ARE assets, so subselecting assets must slice the feature axis too --
            # otherwise the columns keep pointing at the full universe.
            X = rand(rng, 5, 3)
            nx = ["A", "B", "C"]
            Zsq = Float64[1 2 3; 4 5 6; 7 8 9]
            sqp(labels, vals) = asset_panel([TensorPanelInput(; name = "prox",
                                                              axis = "asset",
                                                              labels = labels, vals = vals)])
            rdsq = ReturnsResult(; nx = nx, X = X, pnl = sqp(nx, Zsq))
            vsq = port_opt_view(rdsq, [1, 3])
            @test PortfolioOptimisers.panel_field(vsq.pnl, "prox").vals ==
                  Zsq[[1, 3], [1, 3]]
            @test PortfolioOptimisers.panel_field(vsq.pnl, "prox").labels ==
                  ["A", "C"] ==
                  vsq.nx
            # And the rule survives a second view, because the labels and the asset names are
            # sliced by the same indices and stay equal.
            @test PortfolioOptimisers.features_are_assets(PortfolioOptimisers.panel_field(vsq.pnl,
                                                                                          "prox"),
                                                          vsq.nx)
            @test PortfolioOptimisers.panel_field(port_opt_view(vsq, [2]).pnl, "prox").vals ==
                  Zsq[[3], [3]]

            Zsq3 = reshape(Float64.(1:45), 5, 3, 3)
            rdsq3 = ReturnsResult(; nx = nx, X = X, pnl = sqp(nx, Zsq3))
            vsq3 = port_opt_view(rdsq3, 2:4, [1, 3])
            @test PortfolioOptimisers.panel_field(vsq3.pnl, "prox").vals ==
                  Zsq3[2:4, [1, 3], [1, 3]]
            @test PortfolioOptimisers.panel_field(vsq3.pnl, "prox").labels == ["A", "C"]

            # Equal lengths alone are not a claim that the axes mean the same thing.
            rdns = ReturnsResult(; nx = nx, X = X, pnl = sqp(["f1", "f2", "f3"], Zsq))
            @test !PortfolioOptimisers.features_are_assets(PortfolioOptimisers.panel_field(rdns.pnl,
                                                                                           "prox"),
                                                           nx)
            @test PortfolioOptimisers.panel_field(port_opt_view(rdns, [1, 3]).pnl,
                                                  "prox").vals == Zsq[[1, 3], :]
        end

        @testset "prices_to_returns carries Z across" begin
            prs = PricesResult(; X = Px, pnl = matrix_panel(nz, Zs))
            pr3 = PricesResult(; X = Px, pnl = matrix_panel(nz, Z3))

            # The conversion itself deletes nothing (ADR 0133), so asset "B" keeps its
            # column and its features even though it is missing throughout.
            kept = prices_to_returns(PricesToReturns(), prs)
            @test kept.nx == ["A", "B", "C"]
            @test panel_feature_matrix(kept.pnl)[2] == Zs
            @test all(!isfinite, view(kept.X, :, 2))

            # The static shape keeps its whole asset axis, because no keyword can cut one:
            # the conversion has nothing that drops a column. Through the layer it also
            # gains an observation axis: the carrier states a Listing Span, so the panel
            # gains the two masks, and `attach_universe_masks` lifts a static field onto
            # the clock they live on rather than leaving the two shapes side by side.
            rrs = prices_to_returns(price_ingestion(PriceIngestion(), Px;
                                                    pnl = matrix_panel(nz, Zs)))
            @test rrs.nx == ["A", "B", "C"]
            @test size(panel_feature_matrix(rrs.pnl)[2]) == (9, 3, 2)
            @test all(panel_feature_matrix(rrs.pnl)[2][t, :, :] == Zs
                      for t in 1:length(rrs.ts))
            @test panel_feature_matrix(rrs.pnl)[1] == nz

            # A carrier built outside the layer states no span, so it states no universe,
            # and a static field stays static: there is no observation axis to lift onto.
            rrs_bare = prices_to_returns(PricesResult(; X = Px, pnl = matrix_panel(nz, Zs)))
            @test panel_feature_matrix(rrs_bare.pnl)[2] == Zs

            # The observation axis is the one the conversion does cut, and it cuts it by
            # exactly the observation the percentage change costs: rows 2:10, every asset.
            rr3 = prices_to_returns(price_ingestion(PriceIngestion(), Px;
                                                    pnl = matrix_panel(nz, Z3)))
            @test size(panel_feature_matrix(rr3.pnl)[2]) == (9, 3, 2)
            @test panel_feature_matrix(rr3.pnl)[2] == Z3[2:10, :, :]
            @test length(rr3.ts) ==
                  size(panel_feature_matrix(rr3.pnl)[2], 1) ==
                  size(rr3.X, 1)

            # Under collapse_args the aggregated period takes the features of the last row of
            # its group -- last-observation semantics. The collapse is the
            # step that renumbers the observation, so `price_ingestion` is what projects the
            # panel onto the clock it emits; the conversion moves no clock and cuts only the
            # observation `padding` costs.
            rc = prices_to_returns(price_ingestion(PriceIngestion(;
                                                                  collapse_args = (week,
                                                                                   last)),
                                                   Px; pnl = matrix_panel(nz, Z3)))
            @test rc.ts == [Date(2020, 1, 10)]
            @test panel_feature_matrix(rc.pnl)[2] == Z3[[10], :, :]

            # A table that is a gap throughout is still a universe: every asset keeps its
            # column, its features and its place, and every return is non-finite. Nothing
            # is left with nothing to bind to, which is why the conversion no longer
            # carries a refusal for an emptied asset axis.
            Pall = TimeArray(collect(ts0), fill(NaN, 10, 3), ["A", "B", "C"])
            Fok = TimeArray(collect(ts0),
                            100 .+ cumsum(rand(StableRNG(7), 10, 2); dims = 1),
                            ["F1", "F2"])
            rall = prices_to_returns(price_ingestion(PriceIngestion(), Pall; F = Fok,
                                                     pnl = matrix_panel(nz, Zs)))
            @test rall.nx == ["A", "B", "C"]
            @test all(!isfinite, rall.X)
            @test all(panel_feature_matrix(rall.pnl)[2][t, :, :] == Zs
                      for t in 1:length(rall.ts))
            rnof = prices_to_returns(Pall)
            @test rnof.nx == ["A", "B", "C"]
            @test all(!isfinite, rnof.X)

            # A surviving timestamp absent from the price clock cannot be mapped back to a
            # row of Z, and must throw rather than pair assets with another period.
            p3 = matrix_panel(nz, Z3)
            ps = matrix_panel(nz, Zs)
            @test_throws ArgumentError PortfolioOptimisers.feature_row_indices(p3,
                                                                               [Date(2019,
                                                                                     1, 1)],
                                                                               collect(ts0))
            @test PortfolioOptimisers.feature_row_indices(p3, ts0[3:5], collect(ts0)) == 3:5
            # The shapes with no observation axis cost nothing.
            @test PortfolioOptimisers.feature_row_indices(ps, ts0[3:5], collect(ts0)) ===
                  Colon()
            @test PortfolioOptimisers.feature_row_indices(nothing, ts0[3:5],
                                                          collect(ts0)) === Colon()
        end

        @testset "PricesResult views and preprocessing" begin
            pr3 = PricesResult(; X = Px, pnl = matrix_panel(nz, Z3))
            prs = PricesResult(; X = Px, pnl = matrix_panel(nz, Zs))

            # Timestamp windows slice a time-varying Z positionally against the price clock.
            @test panel_feature_matrix(port_opt_view(pr3, 3:6, [1, 3]).pnl)[2] ==
                  Z3[3:6, [1, 3], :]
            @test panel_feature_matrix(port_opt_view(pr3, 3:6).pnl)[2] == Z3[3:6, :, :]
            @test panel_feature_matrix(port_opt_view(prs, 3:6, [1, 3]).pnl)[2] ==
                  Zs[[1, 3], :]
            # A window that touches neither of a static Z's axes leaves it alone entirely.
            @test panel_feature_matrix(port_opt_view(prs, 3:6).pnl)[2] == Zs
            @test port_opt_view(pr3, :, :) === pr3

            # MissingDataFilter drops assets at price level, so it owes Z the same slice.
            mdf = fit_preprocessing(MissingDataFilter(; col_thr = 0.5), pr3)
            pm = apply_preprocessing(mdf, pr3)
            @test string.(TimeSeries.colnames(pm.X)) == ["A", "C"]
            @test panel_feature_matrix(pm.pnl)[2] == Z3[:, [1, 3], :]
            # A fill changes no shape, so Z passes through untouched. The carrier states
            # a span, which is what bounds the fill (ADR 0130).
            pr3s = PricesResult(; X = Px, pnl = matrix_panel(nz, Z3),
                                span = trues(size(Pv)))
            pim = apply_preprocessing(fit_preprocessing(PriceGapFill(), pr3s), pr3s)
            @test panel_feature_matrix(pim.pnl)[2] == Z3
            @test panel_feature_matrix(pim.pnl)[1] == nz

            # train_test_split is a pair of port_opt_views, so it splits Z's observations.
            rr3 = prices_to_returns(PricesToReturns(), pr3)
            tr, te = train_test_split(rr3; train_size = 5)
            @test size(panel_feature_matrix(tr.pnl)[2], 1) == size(tr.X, 1)
            @test size(panel_feature_matrix(te.pnl)[2], 1) == size(te.X, 1)
            @test panel_feature_matrix(tr.pnl)[2] ==
                  panel_feature_matrix(rr3.pnl)[2][1:size(panel_feature_matrix(tr.pnl)[2],
                                                          1), :, :]
        end

        @testset "other ReturnsResult constructors thread Z" begin
            # The online step of a prior at `x_src = ExcessSource()` folds X - B in a rebuilt ReturnsResult;
            # it must not lose Z on the way.
            rdb = ReturnsResult(; nx = ["A", "B"], X = rand(rng, 4, 2), nb = ["bm"],
                                B = rand(rng, 4), pnl = matrix_panel(nz, rand(rng, 2, 2)))
            rp = PortfolioOptimisers.prior_step_returns(rdb, ExcessSource())
            @test panel_feature_matrix(rp.pnl)[2] == panel_feature_matrix(rdb.pnl)[2]
            @test panel_feature_matrix(rp.pnl)[1] == panel_feature_matrix(rdb.pnl)[1]
            @test rp.X == rdb.X .- rdb.B && rp.B === rdb.B
            @test PortfolioOptimisers.prior_step_returns(rdb, nothing) === rdb

            # AssetSelectorResult subselects through port_opt_view, so it is carried for free.
            rds = ReturnsResult(; nx = ["A", "B", "C"], X = rand(rng, 5, 3),
                                pnl = matrix_panel(nz, Zs))
            sel = apply_preprocessing(PortfolioOptimisers.AssetSelectorResult(["A", "C"]),
                                      rds)
            @test panel_feature_matrix(sel.pnl)[2] == Zs[[1, 3], :]
        end
    end

    @testset "prices_to_returns closed form" begin
        # A two-row example, computed by hand. The simple branch is the formula the
        # docstring states, bit for bit (#1414). The log branch is `log1p` of it (#1519).
        ts = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 3))
        P = TimeArray(ts, [100.0 50.0; 110.0 45.0; 121.0 54.0], [:a, :b])
        simple = [(110-100)/100 (45-50)/50; (121-110)/110 (54-45)/45]
        logret = [log(110 / 100) log(45 / 50); log(121 / 110) log(54 / 45)]

        rs = prices_to_returns(P)
        @test rs.X == simple
        @test rs.nx == ["a", "b"]
        @test rs.ts == ts[2:3]

        rl = prices_to_returns(P; ret_method = :log)
        @test rl.X ≈ logret
        @test rl.X == log1p.(simple)

        # `padding` keeps the first observation and fills its return with `NaN`, so the
        # returns keep the length of the price clock.
        rp = prices_to_returns(P; padding = true)
        @test rp.ts == ts
        @test all(isnan, rp.X[1, :])
        @test rp.X[2:3, :] == simple

        # #1414: the difference of two prices within a factor of two of each other is exact,
        # so the division is the one rounding and the return is correctly rounded. The jump
        # from 100.3 to 1000.7 rounds twice, and stays within one ulp. The old route,
        # `expm1` of a log difference, matched no cell of this example exactly.
        pq = [100.0, 100.1, 99.95, 100.3, 1000.7, 999.9]
        tq = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 6))
        rq = vec(prices_to_returns(TimeArray(tq, reshape(pq, :, 1), [:a])).X)
        exact = big.(pq[2:end]) ./ big.(pq[1:(end - 1)]) .- 1
        @test rq == (pq[2:end] .- pq[1:(end - 1)]) ./ pq[1:(end - 1)]
        @test rq[[1, 2, 3, 5]] == Float64.(exact[[1, 2, 3, 5]])
        @test abs(rq[4] - exact[4]) <= eps(rq[4])
        @test rq != expm1.(log.(pq[2:end]) .- log.(pq[1:(end - 1)]))

        # #1519: the log branch is `log1p` of the simple return. The simple return has one
        # rounding, and `log1p` adds about one ulp, so every log return is within one ulp of
        # the exact one. The difference of logarithms cancels on two close prices: each
        # logarithm carries an absolute error of about `eps * |log p|`, so the error grows
        # with the price level and not with the return.
        lq = vec(prices_to_returns(TimeArray(tq, reshape(pq, :, 1), [:a]);
                                   ret_method = :log).X)
        lexact = log.(big.(pq[2:end]) ./ big.(pq[1:(end - 1)]))
        @test lq == log1p.(rq)
        @test all(abs.(lq .- lexact) .<= eps.(lq))
        # Prices near 1e5 and a return of 1e-9: the old difference of logarithms is
        # thousands of ulps off, and the fix stays within one ulp.
        ph = [1.0e5, 1.0e5 * (1 + 1e-9), 1.0e5 * (1 + 2e-9)]
        th = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 3))
        lh = vec(prices_to_returns(TimeArray(th, reshape(ph, :, 1), [:a]);
                                   ret_method = :log).X)
        hexact = log.(big.(ph[2:end]) ./ big.(ph[1:(end - 1)]))
        hold = log.(ph[2:end]) .- log.(ph[1:(end - 1)])
        @test all(abs.(lh .- hexact) .<= eps.(lh))
        @test all(abs.(hold .- hexact) .> 1000 .* eps.(lh))

        # A `NaN` price is carried, not deleted. The clock keeps its row, and the two
        # returns of `a` that read the absent price are the ones left non-finite: nothing
        # spans the gap, because a return is the change between two consecutive observations.
        Pm = TimeArray(ts, [100.0 50.0; NaN 45.0; 121.0 54.0], [:a, :b])
        rm = prices_to_returns(Pm)
        @test rm.ts == ts[2:3]
        @test all(!isfinite, view(rm.X, :, 1))
        @test rm.X[:, 2] ≈ [45 / 50 - 1, 54 / 45 - 1]

        # An elementwise map is not a keyword of the conversion: it changes a price rather
        # than the arithmetic that turns two prices into a return. ADR 0129 mints no step
        # for it either (#1001): nothing outside this file ever used it, and a caller maps
        # the table they hold before the door in one line. A common scale factor leaves a
        # return unchanged; a shift does not.
        @test prices_to_returns(map((t, v) -> (t, 2 .* v), P)).X ≈ simple
        @test prices_to_returns(map((t, v) -> (t, v .+ 100.0), P)).X ≈
              [210/200-1 145/150-1; 221/210-1 154/145-1]
        @test_throws MethodError prices_to_returns(P; map_func = (t, v) -> (t, 2 .* v))

        # A price has no sign, so both branches refuse a negative price. The quotient of
        # the simple branch takes no logarithm, so an explicit check refuses it (#1414),
        # names the column, and skips an absent price.
        neg_px = TimeArray(ts, [100.0 50.0; -110.0 45.0; 121.0 54.0], [:a, :b])
        @test_throws DomainError prices_to_returns(neg_px)
        @test_throws DomainError prices_to_returns(neg_px; ret_method = :log)
        neg_err = try
            prices_to_returns(neg_px)
        catch err
            err
        end
        @test neg_err.val == ["a"]
        neg_f = TimeArray(ts, reshape([101.0, -102.0, 103.0], :, 1), [:f1])
        @test_throws DomainError prices_to_returns(price_ingestion(PriceIngestion(), P;
                                                                   F = neg_f))
        @test prices_to_returns(TimeArray(ts, [100.0 -0.0; 110.0 45.0; 121.0 54.0],
                                          [:a, :b])).X[1, 2] == Inf
        # A zero price is an infinity rather than a throw, on either branch.
        zero_px = TimeArray(ts, [100.0 50.0; 0.0 45.0; 121.0 54.0], [:a, :b])
        @test prices_to_returns(zero_px).X[:, 1] == [-1.0, Inf]
        @test prices_to_returns(zero_px; ret_method = :log).X[:, 1] == [-Inf, Inf]

        # An asset that is a gap for most of the window keeps its column beside a factor
        # that is not, and the result carries both. The asset group is never `nothing`,
        # because the conversion removes no column.
        ts5 = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 5))
        Pa = TimeArray(ts5, reshape([100.0, NaN, NaN, NaN, 104.0], 5, 1), [:a])
        Pf = TimeArray(ts5, reshape(Float64.(101:105), 5, 1), [:f1])
        rdrop = prices_to_returns(price_ingestion(PriceIngestion(), Pa; F = Pf))
        @test rdrop.nx == ["a"]
        @test findall(!isfinite, view(rdrop.X, :, 1)) == [1, 2, 3, 4]
        @test rdrop.nf == ["f1"]
    end

    @testset "port_opt_view carries a benchmark of either width" begin
        port_opt_view = PortfolioOptimisers.port_opt_view
        ts = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 5))
        Xv = Float64.(reshape(1:20, 5, 4))
        X = TimeArray(ts, Xv, [:a1, :a2, :a3, :a4])
        Bwide = TimeArray(ts, Float64.(reshape(101:120, 5, 4)), [:b1, :b2, :b3, :b4])
        Bnarrow = TimeArray(ts, reshape(Float64.(201:205), 5, 1), [:bm])
        i = ts[[2, 4]]
        j = [1, 3]

        # One benchmark column per asset: the asset index selects the same columns of B.
        vw = port_opt_view(PricesResult(; X = X, B = Bwide), i, j)
        @test string.(TimeSeries.colnames(vw.B)) == ["b1", "b3"]
        @test values(vw.B) == values(Bwide)[[2, 4], j]
        @test values(vw.X) == Xv[[2, 4], j]

        # One shared benchmark column: `j` addresses the assets, so B keeps its single
        # column. Indexing it by `j` would read past the one column it has.
        vn = port_opt_view(PricesResult(; X = X, B = Bnarrow), i, j)
        @test string.(TimeSeries.colnames(vn.B)) == ["bm"]
        @test values(vn.B) == values(Bnarrow)[[2, 4], :]

        # A time-varying feature matrix cannot be sliced without the surviving timestamps,
        # because its observation axis is parallel to the price clock positionally.
        Z3 = reshape(Float64.(1:40), 5, 4, 2)
        @test_throws ArgumentError PortfolioOptimisers.feature_row_indices(matrix_panel(["f1",
                                                                                         "f2"],
                                                                                        Z3),
                                                                           nothing, ts)
    end

    @testset "PricesResult and its views agree with their docstrings" begin
        port_opt_view = PortfolioOptimisers.port_opt_view
        ts = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 5))
        Xv = [100.0 200 300; 101 201 301; 102 202 302; 103 203 303; 104 204 304]
        X = TimeArray(ts, Xv, [:A, :B, :C])
        iv = TimeArray(ts, repeat([0.1 0.2 0.3], 5), [:A, :B, :C])
        B = TimeArray(ts, 10 .* Xv, [:bA, :bB, :bC])
        span = trues(5, 3)
        span[1, 3] = false
        ivpa = [1.0, 2.0, 3.0]
        pr = PricesResult(; X = X, iv = iv, B = B, ivpa = ivpa, span = span)

        # The TimeArray fields are copies, the other fields can share memory with the
        # parent, and two Colons return the parent itself.
        v = port_opt_view(pr, 1:2, 1:2)
        @test values(v.X) == Xv[1:2, 1:2]
        v.ivpa[1] = 9.0
        @test pr.ivpa[1] == 9.0
        pr.ivpa[1] = 1.0
        @test port_opt_view(pr, :, :) === pr

        # Rows come back in clock order whatever the order of `i`, and a timestamp that
        # `X` does not hold selects no row.
        @test timestamp(port_opt_view(pr, [3, 1]).X) == ts[[1, 3]]
        va = port_opt_view(pr, [ts[2], Date(2021, 1, 1)])
        @test timestamp(va.X) == ts[[2]]
        @test size(va.span) == (1, 3)
        @test_throws PortfolioOptimisers.IsEmptyError port_opt_view(pr, [Date(2021, 1, 1)])

        # `j` keeps the order it gives, on every field with an asset axis.
        vr = port_opt_view(pr, :, [3, 1])
        @test values(vr.X)[1, :] == [300.0, 100.0]
        @test values(vr.iv)[1, :] == [0.3, 0.1]
        @test string.(TimeSeries.colnames(vr.B)) == ["bC", "bA"]
        @test vr.ivpa == [3.0, 1.0]
        @test vr.span[1, :] == [false, true]

        # A call shape that no method takes names itself, not a leaf helper.
        msg(f) =
            try
                f()
                ""
            catch e
                e isa ArgumentError ? sprint(showerror, e) : string(typeof(e))
            end
        m1 = msg(() -> port_opt_view(pr, 2))
        @test occursin("PricesResult with the index argument type(s) (Int64)", m1)
        @test occursin("port_opt_view(pr, observations, assets)", m1)
        @test occursin("(UnitRange{Int64}, UnitRange{Int64}, Colon)",
                       msg(() -> port_opt_view(pr, 1:2, 1:2, :)))
        @test occursin("keyword argument(s) foo",
                       msg(() -> port_opt_view(pr, 1:2; foo = 1)))
        @test occursin("BarePricesResult",
                       msg(() -> port_opt_view(BarePricesResult(), 1:2)))
        # The fallback takes no call that a method of PricesResult takes.
        @test !any(p -> p[1].name === :port_opt_view,
                   Test.detect_ambiguities(PortfolioOptimisers))

        # The constructor accepts an absent implied volatility, NaN or missing, and refuses
        # a present value that is negative or infinite.
        ivn = TimeArray(ts, [fill(NaN, 5) fill(0.2, 5) fill(0.3, 5)], [:A, :B, :C])
        @test PricesResult(; X = X, iv = ivn) isa PricesResult
        ivmv = Matrix{Union{Missing, Float64}}(fill(0.1, 5, 3))
        ivmv[1, 1] = missing
        ivm = TimeArray(ts, ivmv, [:A, :B, :C])
        @test PricesResult(; X = X, iv = ivm) isa PricesResult
        for bad in (Inf, -0.1)
            ivb = TimeArray(ts, [fill(bad, 5) fill(0.2, 5) fill(0.3, 5)], [:A, :B, :C])
            @test_throws DomainError PricesResult(; X = X, iv = ivb)
        end
        @test PortfolioOptimisers.assert_nonneg_where_present([0.0, -0.0, NaN, 1], :iv) ===
              nothing
        # The other raises of the Validation list.
        @test_throws PortfolioOptimisers.IsEmptyError PricesResult(; X = X,
                                                                   ivpa = Float64[])
        @test_throws DomainError PricesResult(; X = X, ivpa = [1.0, -1.0, 1.0])
        @test_throws DimensionMismatch PricesResult(; X = X, ivpa = [1.0, 1.0])
        @test_throws DimensionMismatch PricesResult(; X = X, B = B[:bA, :bB])
        @test_throws DimensionMismatch PricesResult(; X = X, span = trues(5, 2))
        # A repeated timestamp is refused, because a window of `port_opt_view` finds each
        # row by its timestamp, and it would take the first of the two rows.
        tsd = [ts[1], ts[1], ts[2], ts[3], ts[3]]
        Xd = TimeArray(tsd, values(X), colnames(X))
        @test_throws "Got 2 repeated timestamp(s), and the first is $(ts[1])" PricesResult(;
                                                                                           X = Xd)
    end

    @testset "the preprocessing interface refuses a half-implemented estimator" begin
        pr = PricesResult(;
                          X = TimeArray(collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 3)),
                                        Float64.(reshape(1:6, 3, 2)), [:a, :b]))
        @test_throws ArgumentError fit_preprocessing(UnimplementedPreprocessing(), pr)
        @test_throws ArgumentError apply_preprocessing(UnimplementedPreprocessing(), pr)
        @test_throws ArgumentError apply_preprocessing(UnimplementedPreprocessingResult(),
                                                       pr)
    end

    @testset "PriceGapFill fits on the training window and never refits" begin
        # Map #955, ADR 0130. `missing` is the other spelling of an absent price, and the
        # fill accepts it through `is_missing_value` exactly as `NaN`; the top-level testsets
        # of this file from "The fill is bounded by the Listing Span under both conventions"
        # on drive the conventions and the bound on `NaN`, so what is pinned here is the
        # `missing` path and the fit/apply replay across a train/test split.
        ts = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 6))
        # `a` is never observed, `b` is observed everywhere, and `c` is observed in the
        # training window alone.
        vals = Union{Float64, Missing}[missing 1.0 10.0
                                       missing 3.0 12.0
                                       missing 5.0 14.0
                                       missing 7.0 missing
                                       missing 9.0 missing
                                       missing 11.0 missing]
        # A caller's own listing calendar says every asset is listed throughout, which is
        # ADR 0129's route to a fill bounded by nothing but the panel itself.
        pr = PricesResult(; X = TimeArray(ts, vals, [:a, :b, :c]), span = trues(6, 3))
        tr = PortfolioOptimisers.port_opt_view(pr, ts[1:3], :)
        te = PortfolioOptimisers.port_opt_view(pr, ts[4:6], :)

        res = fit_preprocessing(PriceGapFill(; fill = MedianValue()), tr)
        # An asset with no observed price in the training window gets no fill value, so it
        # takes no entry in the result and is left untouched at apply time.
        @test string.(res.nx) == ["b", "c"]
        @test res.v == [3.0, 12.0]

        applied = apply_preprocessing(res, te)
        # `c` has no observed price in the test window at all, so its three missing entries
        # can only take the training median. This is the contract the fit/apply split exists
        # for: the fill value never comes from the window being transformed.
        @test values(applied.X)[:, 3] == fill(12.0, 3)
        @test all(ismissing, values(applied.X)[:, 1])
        @test values(applied.X)[:, 2] == [7.0, 9.0, 11.0]

        # A fitted name the window does not carry is skipped rather than searched for, so a
        # narrowed universe still applies.
        narrow = PricesResult(;
                              X = TimeArray(ts[4:6],
                                            reshape(Union{Float64, Missing}[7.0, missing,
                                                                            11.0], 3, 1),
                                            [:b]), span = trues(3, 1))
        @test vec(values(apply_preprocessing(res, narrow).X)) == [7.0, 3.0, 11.0]
    end

    @testset "the missing-data path on both axes" begin
        is_missing_value = PortfolioOptimisers.is_missing_value

        # `missing` and `NaN` are the two conventions for an absent price, and one predicate
        # accepts both. A non-number is never missing.
        @test is_missing_value(missing)
        @test is_missing_value(NaN)
        @test !is_missing_value(1.0)
        @test !is_missing_value("a")

        # `MissingDataFilter` splits the two axes across the fit/apply seam: `col_thr`
        # selects the universe at fit time and `row_thr` drops rows at apply time.
        ts = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 5))
        vals = Union{Float64, Missing}[missing 1.0 10.0
                                       missing missing 11.0
                                       missing 3.0 12.0
                                       missing 4.0 13.0
                                       missing 5.0 14.0]
        pr = PricesResult(; X = TimeArray(ts, vals, [:a, :b, :c]))

        # `a` is missing in 5 of 5 rows and `b` in 1 of 5, so a threshold of 0.5 keeps both
        # `b` and `c`.
        fitted = fit_preprocessing(MissingDataFilter(; col_thr = 0.5, row_thr = 1.0), pr)
        @test string.(fitted.nx) == ["b", "c"]
        @test fitted.row_thr == 1.0
        # `row_thr = 1.0` admits a row in which every surviving asset is missing, so the
        # window keeps all five rows.
        @test TimeSeries.timestamp(apply_preprocessing(fitted, pr).X) == ts

        # Two surviving assets, so `row_thr = 0.4` admits at most 0.8 missing entries per
        # row: the second row, which holds one, goes.
        tight = fit_preprocessing(MissingDataFilter(; col_thr = 0.5, row_thr = 0.4), pr)
        applied = apply_preprocessing(tight, pr)
        @test TimeSeries.timestamp(applied.X) == ts[[1, 3, 4, 5]]
        @test values(applied.X) == [1.0 10.0; 3.0 12.0; 4.0 13.0; 5.0 14.0]

        # The carrier states one clock, so a dropped row leaves it for every series the
        # carrier holds: the factor, benchmark and implied-volatility series are read at the
        # surviving timestamps, as `port_opt_view` reads them, and the conversion — which
        # refuses a covariate stating a clock the assets no longer do — runs. A per-asset
        # benchmark follows the kept columns too; a shared one has nothing to follow.
        F = TimeArray(ts, [1.0 2.0; 1.1 2.1; 1.2 2.2; 1.3 2.3; 1.4 2.4], [:f1, :f2])
        B1 = TimeArray(ts, [5.0, 5.1, 5.2, 5.3, 5.4], [:bm])
        B3 = TimeArray(ts,
                       [5.0 6.0 7.0; 5.1 6.1 7.1; 5.2 6.2 7.2; 5.3 6.3 7.3; 5.4 6.4 7.4],
                       [:ba, :bb, :bc])
        iv = TimeArray(ts, fill(0.2, 5, 3), [:a, :b, :c])
        prc = PricesResult(; X = pr.X, F = F, B = B1, iv = iv, ivpa = [1.0, 2.0, 3.0])
        apc = apply_preprocessing(fit_preprocessing(MissingDataFilter(; col_thr = 0.5,
                                                                      row_thr = 0.4), prc),
                                  prc)
        @test TimeSeries.timestamp(apc.F) == ts[[1, 3, 4, 5]]
        @test values(apc.F) == values(F)[[1, 3, 4, 5], :]
        @test TimeSeries.timestamp(apc.B) == ts[[1, 3, 4, 5]]
        @test values(apc.B) == values(B1)[[1, 3, 4, 5]]
        @test TimeSeries.timestamp(apc.iv) == ts[[1, 3, 4, 5]]
        @test size(values(apc.iv)) == (4, 2)
        @test apc.ivpa == [2.0, 3.0]
        rdc = prices_to_returns(apc)
        @test size(rdc.X) == (3, 2)
        @test size(rdc.F) == (3, 2)
        @test length(rdc.B) == 3
        @test size(rdc.iv) == (3, 2)
        prb = PricesResult(; X = pr.X, B = B3)
        apb = apply_preprocessing(fit_preprocessing(MissingDataFilter(; col_thr = 0.5,
                                                                      row_thr = 0.4), prb),
                                  prb)
        @test TimeSeries.colnames(apb.B) == [:bb, :bc]
        @test values(apb.B) == values(B3)[[1, 3, 4, 5], 2:3]
        @test size(prices_to_returns(apb).B) == (3, 2)
        # ADR 0133's own composition of the dense matrix, on a carrier with a covariate:
        # a filter at `row_thr = 0` and the conversion, one after the other.
        dense = apply_preprocessing(fit_preprocessing(MissingDataFilter(; col_thr = 0.5,
                                                                        row_thr = 0.0),
                                                      prc), prc)
        @test TimeSeries.timestamp(dense.X) == ts[[1, 3, 4, 5]]
        @test size(apply_preprocessing(PricesToReturns(), dense).X) == (3, 2)

        # A universe that keeps nothing is refused rather than returned empty.
        allmissing = PricesResult(;
                                  X = TimeArray(ts,
                                                Union{Float64, Missing}[missing missing
                                                                        missing missing
                                                                        missing missing
                                                                        missing missing
                                                                        missing missing],
                                                [:a, :b]))
        @test_throws PortfolioOptimisers.IsEmptyError fit_preprocessing(MissingDataFilter(;
                                                                                          col_thr = 0.5),
                                                                        allmissing)
        # A window that carries none of the fitted universe is refused, and the message names
        # the first fitted asset it lacks.
        other = PricesResult(; X = TimeArray(ts, Float64.(reshape(1:5, 5, 1)), [:zz]))
        @test_throws ArgumentError apply_preprocessing(fitted, other)
    end

    @testset "MissingDataFilter agrees with its docstrings" begin
        share_at_most = PortfolioOptimisers.share_at_most
        # A share is k / n in the type of the threshold. `100 * 0.29` is 28.999999999999996,
        # so a test of k <= n * thr dropped 29 missing entries of 100 at 0.29.
        @test all(k -> share_at_most(k, 100, k / 100), 0:100)
        @test all(k -> share_at_most(k, 100, Float32(k / 100)), 0:100)
        @test all(k -> !share_at_most(k + 1, 100, k / 100), 0:99)
        @test share_at_most(1, 3, 1 // 3) && !share_at_most(2, 3, 1 // 3)
        @test share_at_most(0, 7, 0) && !share_at_most(1, 7, 0) && share_at_most(7, 7, 1)

        ts100 = collect(Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(99)))
        # A row with 29 of 100 assets missing has the share 0.29, so row_thr = 0.29 keeps it
        # and row_thr = 0.28 drops it.
        v = fill(1.0, 100, 100)
        v[1, 1:29] .= NaN
        pr = PricesResult(; X = TimeArray(ts100, v, Symbol.("a", 1:100)))
        keep29 = apply_preprocessing(fit_preprocessing(MissingDataFilter(; row_thr = 0.29),
                                                       pr), pr)
        @test size(values(keep29.X), 1) == 100
        drop28 = apply_preprocessing(fit_preprocessing(MissingDataFilter(; row_thr = 0.28),
                                                       pr), pr)
        @test TimeSeries.timestamp(drop28.X) == ts100[2:end]

        # A column with 29 of 100 observations missing survives col_thr = 0.29 in Float64,
        # Float32 and Rational, in the batch fit and in the online read-out alike.
        w = fill(1.0, 100, 2)
        w[1:29, 1] .= NaN
        pc = PricesResult(; X = TimeArray(ts100, w, [:a, :b]))
        for thr in (0.29, 0.29f0, 29 // 100)
            mdf = MissingDataFilter(; col_thr = thr)
            @test fit_preprocessing(mdf, pc).nx == [:a, :b]
            stepped, _ = PortfolioOptimisers.partial_fit_transform(mdf, pc)
            @test fit_preprocessing(stepped).nx == [:a, :b]
        end
        for thr in (0.28, 0.28f0, 28 // 100)
            @test fit_preprocessing(MissingDataFilter(; col_thr = thr), pc).nx == [:b]
        end

        # The apply step replays the fitted universe by name, in the fitted order, and
        # refuses a window that lacks a fitted asset.
        t5 = ts100[1:5]
        p3 = PricesResult(;
                          X = TimeArray(t5,
                                        [1.0 2.0 3.0; 1.1 2.1 3.1; 1.2 2.2 3.2;
                                         1.3 2.3 3.3; 1.4 2.4 3.4], [:a, :b, :c]))
        r3 = fit_preprocessing(MissingDataFilter(), p3)
        swapped = PricesResult(; X = p3.X[[:c, :a, :b]])
        out = apply_preprocessing(r3, swapped)
        @test TimeSeries.colnames(out.X) == [:a, :b, :c]
        @test values(out.X) == values(p3.X)
        partial = PricesResult(; X = p3.X[[:a, :c]])
        e = try
            apply_preprocessing(r3, partial)
        catch err
            err
        end
        @test e isa ArgumentError
        @test occursin("variable `b` not in asset universe", e.msg)

        # A vector ivpa follows the kept columns when the carrier holds no iv series.
        x = [NaN 1.0; NaN 2.0; NaN 3.0; 1.0 4.0; 2.0 5.0]
        pv = PricesResult(; X = TimeArray(t5, x, [:a, :b]), ivpa = [1.0, 2.0])
        av = apply_preprocessing(fit_preprocessing(MissingDataFilter(; col_thr = 0.5), pv),
                                 pv)
        @test TimeSeries.colnames(av.X) == [:b]
        @test av.ivpa == [2.0]
        @test size(prices_to_returns(av).X) == (4, 1)

        # A window whose every observation the row filter drops is refused by name.
        allgap = PricesResult(; X = TimeArray(t5, fill(NaN, 5, 2), [:a, :b]))
        e = try
            apply_preprocessing(fit_preprocessing(MissingDataFilter(; row_thr = 0.0), pv),
                                allgap)
        catch err
            err
        end
        @test e isa PortfolioOptimisers.IsEmptyError
        @test occursin("row_thr = 0.0 drops every observation", e.msg)

        # Both thresholds are shares in [0, 1].
        @test_throws DomainError MissingDataFilter(; col_thr = -0.1)
        @test_throws DomainError MissingDataFilter(; row_thr = 1.1)
    end

    @testset "MissingDataFilter fits its universe to a fixed point (#1758)" begin
        @test MissingDataFilter().alg isa FixedPointFilter
        ts10 = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 10))
        # `c` misses 4 of 10 observations, which col_thr = 0.4 admits. The row filter drops
        # the first three, where `a` and `b` miss, and over the 7 that stay `c` misses 4.
        M = [NaN NaN 70; NaN NaN 71; NaN NaN 72; 100 50 NaN; 101 51 NaN; 102 52 NaN;
             103 53 NaN; 104 54 77; 105 55 78; 106 56 79]
        pr = PricesResult(; X = TimeArray(ts10, M, [:a, :b, :c]))
        fp = MissingDataFilter(; col_thr = 0.4, row_thr = 1 // 3)
        sp = MissingDataFilter(; col_thr = 0.4, row_thr = 1 // 3, alg = SinglePassFilter())
        rsp = fit_preprocessing(sp, pr)
        @test rsp.nx == [:a, :b, :c]
        psp = apply_preprocessing(rsp, pr)
        @test count(isnan, values(psp.X)[:, 3]) == 4
        @test size(values(psp.X), 1) == 7
        # The single pass is not a fixed point: a second fit drops `c`.
        @test fit_preprocessing(sp, psp).nx == [:a, :b]
        rfp = fit_preprocessing(fp, pr)
        @test rfp.nx == [:a, :b]
        pfp = apply_preprocessing(rfp, pr)
        @test TimeSeries.timestamp(pfp.X) == ts10[4:10]
        # At the fixed point a second fit changes nothing, and every kept asset and every
        # kept observation meets its threshold on the rows that the apply step keeps.
        @test fit_preprocessing(fp, pfp).nx == rfp.nx
        gaps = isnan.(values(pfp.X))
        @test all(vec(count(gaps; dims = 1)) .// size(gaps, 1) .<= 4 // 10)
        @test all(vec(count(gaps; dims = 2)) .// size(gaps, 2) .<= 1 // 3)

        # The rows come back from the whole window on each pass. Row 7 misses `a` and `c`,
        # 2 of 4, so the first pass drops it. Over the 6 rows that stay `c` misses 3, more
        # than 0.4, so `c` drops. Over `a`, `b` and `d` row 7 misses 1 of 3, which
        # row_thr = 1/3 admits, so the fixed point keeps it.
        Mr = [NaN NaN 70 80; NaN NaN 71 81; NaN NaN 72 82; 100 50 NaN 83; 101 51 NaN 84;
              102 52 NaN 85; NaN 53 NaN 86; 104 54 77 87; 105 55 78 88; 106 56 79 89]
        prr = PricesResult(; X = TimeArray(ts10, Mr, [:a, :b, :c, :d]))
        rr = fit_preprocessing(MissingDataFilter(; col_thr = 0.4, row_thr = 1 // 3), prr)
        @test rr.nx == [:a, :b, :d]
        @test TimeSeries.timestamp(apply_preprocessing(rr, prr).X) == ts10[4:10]
        rs = fit_preprocessing(MissingDataFilter(; col_thr = 0.4, row_thr = 1 // 3,
                                                 alg = SinglePassFilter()), prr)
        @test rs.nx == [:a, :b, :c, :d]
        @test TimeSeries.timestamp(apply_preprocessing(rs, prr).X) ==
              ts10[[4, 5, 6, 8, 9, 10]]
        t4 = ts10[1:4]

        # At row_thr = 1 the first pass is the fixed point, and the two rules agree.
        for thr in (0.3, 0.4, 0.5, 1.0)
            @test fit_preprocessing(MissingDataFilter(; col_thr = thr), pr).nx ==
                  fit_preprocessing(MissingDataFilter(; col_thr = thr,
                                                      alg = SinglePassFilter()), pr).nx
        end

        # The fixed point runs the row filter in the fit, so a row filter that keeps no
        # observation is refused at the fit. The single pass refuses only at the apply.
        Mz = [1.0 NaN; NaN 2.0; NaN 3.0; 4.0 NaN]
        prz = PricesResult(; X = TimeArray(t4, Mz, [:a, :b]))
        @test_throws PortfolioOptimisers.IsEmptyError fit_preprocessing(MissingDataFilter(;
                                                                                          col_thr = 0.5,
                                                                                          row_thr = 0.0),
                                                                        prz)
        rz = fit_preprocessing(MissingDataFilter(; col_thr = 0.5, row_thr = 0.0,
                                                 alg = SinglePassFilter()), prz)
        @test rz.nx == [:a, :b]
        @test_throws PortfolioOptimisers.IsEmptyError apply_preprocessing(rz, prz)

        # An asset with no price stays at col_thr = 1, so a row filter below 1 drops every
        # observation. The fit refuses, and the message names the asset and the way out.
        pe = PricesResult(;
                          X = TimeArray(t4,
                                        [1.0 2.0 NaN; 1.1 2.1 NaN; 1.2 2.2 NaN;
                                         1.3 2.3 NaN], [:a, :b, :c]))
        e = try
            fit_preprocessing(MissingDataFilter(; row_thr = 0.0), pe)
        catch err
            err
        end
        @test e isa PortfolioOptimisers.IsEmptyError
        @test occursin("row_thr = 0.0 drops every observation", e.msg)
        @test occursin("The assets c have no observation", e.msg)
        @test occursin("col_thr below 1", e.msg)
        @test fit_preprocessing(MissingDataFilter(; col_thr = 0.99, row_thr = 0.0), pe).nx ==
              [:a, :b]

        # A factor with a gap at every observation misses every asset everywhere, and no
        # filter drops a series, so the message says to fill it or remove it.
        Ff = TimeArray(t4, [1.0 NaN; 1.1 NaN; 1.2 NaN; 1.3 NaN], [:f1, :f2])
        pf = PricesResult(;
                          X = TimeArray(t4, [1.0 2.0; 1.1 2.1; 1.2 2.2; 1.3 2.3], [:a, :b]),
                          F = Ff)
        e = try
            apply_preprocessing(fit_preprocessing(MissingDataFilter(; row_thr = 0.5), pf),
                                pf)
        catch err
            err
        end
        @test e isa PortfolioOptimisers.IsEmptyError
        @test occursin("gap at every observation", e.msg)
        @test !occursin("have no observation", e.msg)

        # A new rule plugs in through `fit_universe`.
        @test fit_preprocessing(MissingDataFilter(; alg = KeepLastAsset()), pr).nx == [:c]
    end

    @testset "prices_to_returns filters neither axis" begin
        # Issue #473 was a defect in the interaction of the conversion's two threshold
        # keywords: the column filter counted the missing entries over the table as it was
        # before the row filter ran, and divided by the surviving row total. Map #955 and
        # ADR 0133 removed both keywords rather than the arithmetic -- deleting a row or a
        # column is a Universe Policy, and a stateless conversion cannot fit one -- so the
        # interaction the defect lived in no longer exists. `MissingDataFilter` owns both
        # axes, and `test_30_pipeline_base.jl` covers the split across its fit/apply seam.
        ts4 = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 4))
        # Row 1 alone is incomplete. `A` and `B` are missing there, and `C` is complete.
        X = TimeArray(ts4, [NaN NaN 1.0; 2.0 2.0 2.0; 3.0 3.0 3.0; 4.0 4.0 4.0],
                      [:A, :B, :C])
        rd473 = prices_to_returns(X)
        @test rd473.nx == ["A", "B", "C"]
        @test rd473.ts == ts4[2:end]
        @test findall(!isfinite, view(rd473.X, :, 1)) == [1]

        # There is no keyword left that would cut either axis, on the verb or the estimator.
        @test_throws MethodError prices_to_returns(X; missing_col_percent = 0.5)
        @test_throws MethodError prices_to_returns(X; missing_row_percent = 0.3)
        @test_throws MethodError prices_to_returns(X; missing_row_percent = nothing)
        @test :missing_col_percent ∉ fieldnames(PricesToReturns)
        @test :missing_row_percent ∉ fieldnames(PricesToReturns)

        # A column that is a gap over most of the window is kept whole, where the released
        # thresholds would have deleted it.
        ts5 = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 5))
        Y = TimeArray(ts5, [NaN NaN 1.0; NaN 2.0 2.0; NaN 3.0 3.0; 4.0 4.0 4.0
                            5.0 5.0 5.0], [:A, :B, :C])
        rdY = prices_to_returns(Y)
        @test rdY.nx == ["A", "B", "C"]
        @test size(rdY.X) == (4, 3)
        @test findall(!isfinite, view(rdY.X, :, 1)) == [1, 2, 3]
    end

    @testset "the conversion carries every gap, and there is no flag" begin
        # Map #955, ADR 0133. An absent price has one spelling, `NaN`, and the conversion
        # carries it into the returns: it deletes no observation and no asset. There is no
        # keyword that says otherwise, which is what makes the gap-carrying path the only
        # path.
        ts6 = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 6))
        # `A` is priced throughout, `B` has no price until observation 3, `C` loses its
        # price at observation 5.
        Z = TimeArray(ts6,
                      [10.0 NaN 30.0; 11.0 NaN 31.0; 12.0 20.0 32.0
                       13.0 21.0 33.0; 14.0 22.0 NaN; 15.0 23.0 NaN], [:A, :B, :C])

        # The whole clock and the whole universe survive, with no keyword given.
        got = prices_to_returns(Z)
        @test got.nx == ["A", "B", "C"]
        @test size(got.X) == (5, 3)

        # There is no flag: the field is gone from the estimator, and the keyword from the
        # verb.
        @test :nan_to_missing ∉ fieldnames(PricesToReturns)
        @test_throws MethodError prices_to_returns(Z; nan_to_missing = false)
        @test_throws MethodError PricesToReturns(; nan_to_missing = false)
        @test :impute_method ∉ fieldnames(PricesToReturns)
        @test_throws MethodError prices_to_returns(Z; impute_method = nothing)

        # The gap does not spread. A run of `k` gapped prices makes exactly the `k + 1`
        # returns that read one of them, and every later return of that column is finite.
        @test all(isfinite, view(got.X, :, 1))                 # A is never gapped
        @test findall(!isfinite, view(got.X, :, 2)) == [1, 2]   # B: prices 1:2 gapped
        @test findall(!isfinite, view(got.X, :, 3)) == [4, 5]   # C: prices 5:6 gapped
        # The finite entries are the ordinary returns, unchanged by the gap beside them.
        @test got.X[3, 2] ≈ 21.0 / 20.0 - 1
        @test got.X[3, 3] ≈ 33.0 / 32.0 - 1

        # A column that is gapped throughout keeps its place in the universe rather than
        # vanishing from it, which is what moves the universe between windows.
        Zdead = TimeArray(ts6, [10.0 NaN; 11.0 NaN; 12.0 NaN; 13.0 NaN; 14.0 NaN; 15.0 NaN],
                          [:A, :B])
        dead = prices_to_returns(Zdead)
        @test dead.nx == ["A", "B"]
        @test all(!isfinite, view(dead.X, :, 2))

        # Deleting is still expressible, and it is a fitted step's job rather than the
        # conversion's own: `B` and `C` each hold two gaps over six rows and go when
        # `MissingDataFilter`'s tolerated fraction is tightened below a third.
        mdfZ = PortfolioOptimisers.fit_preprocessing(MissingDataFilter(; col_thr = 0.1),
                                                     PricesResult(; X = Z))
        @test mdfZ.nx == [:A]

        # The estimator runs the same path, and carries no flag to change it.
        pr6 = PricesResult(; X = Z)
        @test size(apply_preprocessing(PricesToReturns(), pr6).X) == (5, 3)

        # A source spells an absent price either way, and the one unification makes them one
        # gap. A wide table built from a tidy one holds `missing`, and it must reach the
        # returns as the same gap a `NaN` does.
        Zm = TimeArray(ts6,
                       [10.0 missing 30.0; 11.0 missing 31.0; 12.0 20.0 32.0
                        13.0 21.0 33.0; 14.0 22.0 missing; 15.0 23.0 missing], [:A, :B, :C])
        @test eltype(values(Zm)) == Union{Missing, Float64}
        mgot = prices_to_returns(Zm)
        @test mgot.nx == ["A", "B", "C"]
        @test size(mgot.X) == (5, 3)
        @test findall(!isfinite, view(mgot.X, :, 2)) == [1, 2]
        @test findall(!isfinite, view(mgot.X, :, 3)) == [4, 5]
        # The two sources agree entry for entry. With `impute_method` deleted there is no
        # path left on which the two spellings could behave differently.
        @test isequal(mgot.X, got.X)

        # The ergonomics this buys: ragged per-asset histories, outer-joined and converted
        # without losing an observation. `TimeSeries.merge` pads a `Float64` array with `NaN`.
        Aw = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 6), [10.0, 11, 12, 13, 14, 15],
                       ["A"])
        Bw = TimeArray(Date(2020, 1, 3):Day(1):Date(2020, 1, 5), [20.0, 21, 22], ["B"])
        joined = merge(Aw, Bw; method = :outer)
        @test isnan(values(joined)[1, 2])
        wide = prices_to_returns(joined)
        @test wide.nx == ["A", "B"]
        @test size(wide.X) == (5, 2)
        @test all(isfinite, view(wide.X, :, 1))
        @test findall(!isfinite, view(wide.X, :, 2)) == [1, 2, 5]

        # ADR 0135: the asset table states the clock. A benchmark or factor series whose
        # history is longer than the asset slice is cut to the asset clock under the default
        # join, so "the benchmark on my asset clock" is what a caller gets by not asking.
        Xa = TimeArray(ts6,
                       [10.0 30.0; 11.0 31.0; 12.0 32.0
                        13.0 33.0; 14.0 34.0; 15.0 35.0], [:A, :C])
        tsb = collect(Date(2019, 12, 25):Day(1):Date(2020, 1, 6))
        Bb = TimeArray(tsb, collect(200.0:(200.0 + length(tsb) - 1)), ["BM"])
        for got in (prices_to_returns(price_ingestion(PriceIngestion(), Xa; B = Bb)),
                    prices_to_returns(price_ingestion(PriceIngestion(), Xa; B = Bb[ts6])),
                    prices_to_returns(price_ingestion(PriceIngestion(; join_method = :inner), Xa;
                                                      B = Bb)))
            @test size(got.X) == (5, 2)
            @test all(isfinite, got.X)
            @test all(isfinite, got.B)
            @test got.ts == collect(ts6)[2:end]
        end

        # The union stays reachable, and it is what the OUTER join is: symmetric, so a
        # longer benchmark expands the observation clock and pads the ASSETS at the
        # observations only it has. The padding is real data absence, carried like any
        # other gap, and the layer names it.
        wideb = @test_logs (:warn, r"`X` at 7 of 13 observations") prices_to_returns(price_ingestion(PriceIngestion(;
                                                                                                                    join_method = :outer),
                                                                                                     Xa;
                                                                                                     B = Bb))
        @test size(wideb.X, 1) == length(tsb) - 1
        @test count(!isfinite, wideb.X) == 2 * (length(tsb) - length(ts6))
        @test all(isfinite, wideb.B)
    end
    @testset "the Gap Return family writes only the cells a gap left non-finite" begin
        # One complete column, and one carrying all three gap positions at once: an inception
        # gap, an interior suspension of k = 2, and a delisting. The span rule reads the
        # position, so the one column exercises every clause of the invariant.
        gts = Date(2020, 1, 1):Day(1):Date(2020, 1, 8)
        pA = [10.0, 11, 12, 13, 14, 15, 16, 17]
        pB = [NaN, 20.0, 21, NaN, NaN, 24, 25, NaN]
        Zg = TimeArray(gts, hcat(pA, pB), ["A", "B"])

        @testset "the writable set is the invariant, held once" begin
            base = prices_to_returns(Zg)
            rB = view(base.X, :, 2)
            # A return reads two consecutive prices, so the k = 2 suspension leaves
            # k + 1 = 3 non-finite returns, and the inception and the delisting one each.
            @test findall(!isfinite, rB) == [1, 3, 4, 5, 7]
            @test findall(!isfinite, view(base.X, :, 1)) == Int[]

            wB = PortfolioOptimisers.gap_return_writable(pB, collect(rB))
            # The Listing Span of B is prices 2 to 7, which projects to returns 2 to 6, so
            # the inception cell (1) and the delisting cell (7) are outside it. Only the
            # suspension's three cells are writable.
            @test findall(wB) == [3, 4, 5]
            # A complete column admits nothing, and neither does a column with no price.
            @test !any(PortfolioOptimisers.gap_return_writable(pA,
                                                               collect(view(base.X, :, 1))))
            @test !any(PortfolioOptimisers.gap_return_writable(fill(NaN, 8), fill(NaN, 7)))

            # Under `padding` the clocks line up and the row every asset is padded on is
            # return 1, which no column admits: it has no earlier observed price.
            padded = prices_to_returns(Zg; padding = true)
            @test size(padded.X, 1) == 8
            @test all(!isfinite, view(padded.X, 1, :))
            wpad = PortfolioOptimisers.gap_return_writable(pB,
                                                           collect(view(padded.X, :, 2)))
            @test !wpad[1]
            @test findall(wpad) == [4, 5, 6]
            @test !any(PortfolioOptimisers.gap_return_writable(pA,
                                                               collect(view(padded.X, :, 1))))
        end

        @testset "CatchUpGapReturn books the move on the observation that ends the gap" begin
            base = prices_to_returns(Zg)
            got = prices_to_returns(Zg; gap_return_alg = CatchUpGapReturn())
            rB = view(got.X, :, 2)
            # The Held Gap shortens from k + 1 = 3 to k = 2: the two unpriced observations
            # stay non-finite and the re-pricing observation carries the whole move. The
            # inception cell (1) and the delisting cell (7) are untouched.
            @test findall(!isfinite, rB) == [1, 3, 4, 7]
            @test rB[5] == (24 - 21) / 21
            # No return is written before the first price, and none after the last.
            @test !isfinite(rB[1])
            @test !isfinite(rB[7])
            # A cell computed from two observed prices is frozen, bit for bit, and so is
            # every cell of a column that carries no gap.
            @test rB[2] === base.X[2, 2]
            @test rB[6] === base.X[6, 2]
            @test view(got.X, :, 1) == view(base.X, :, 1)

            # The log branch reads the same pair of prices through the same arithmetic.
            glog = prices_to_returns(Zg; ret_method = :log,
                                     gap_return_alg = CatchUpGapReturn())
            @test glog.X[5, 2] == log1p((24.0 - 21.0) / 21.0)
            @test PortfolioOptimisers.gap_return_value(:log, 24.0, 21.0) ==
                  log1p(PortfolioOptimisers.gap_return_value(:simple, 24.0, 21.0))

            # Under `padding` the padded row stays untouched for every asset.
            gpad = prices_to_returns(Zg; padding = true,
                                     gap_return_alg = CatchUpGapReturn())
            @test all(!isfinite, view(gpad.X, 1, :))
            @test findall(!isfinite, view(gpad.X, :, 2)) == [1, 2, 4, 5, 8]
            @test gpad.X[6, 2] ≈ 24 / 21 - 1
        end

        @testset "the driver freezes what the algorithm has no licence to write" begin
            base = prices_to_returns(Zg)
            rogue = prices_to_returns(Zg; gap_return_alg = RogueGapReturn())
            # `RogueGapReturn` answers every cell with -99.0. Only the three writable cells
            # of B take it; everything else is exactly what the default rule computed.
            @test findall(==(-99.0), rogue.X) == CartesianIndex.([3, 4, 5], 2)
            @test view(rogue.X, :, 1) == view(base.X, :, 1)
            @test isequal(view(rogue.X, [1, 2, 6, 7], 2), view(base.X, [1, 2, 6, 7], 2))
        end

        @testset "the default rule is bit-identical, and a contradicted algorithm informs" begin
            base = prices_to_returns(Zg)
            @test isequal(prices_to_returns(Zg; gap_return_alg = nothing).X, base.X)

            # A gapless table admits no cell, so the writable set is provably empty. That is
            # an `@info`, not a refusal and not a warning: the returns it computed are
            # correct, they are simply the default's.
            clean = TimeArray(gts, hcat(pA, pA .+ 1), ["A", "B"])
            got = @test_logs (:info,) match_mode=:any prices_to_returns(clean;
                                                                        gap_return_alg = CatchUpGapReturn())
            @test isequal(got.X, prices_to_returns(clean).X)
        end

        @testset "PricesToReturns carries the algorithm" begin
            @test isnothing(PricesToReturns().gap_return_alg)
            ptr = PricesToReturns(; gap_return_alg = CatchUpGapReturn())
            @test isa(ptr.gap_return_alg, CatchUpGapReturn)
            pg = PricesResult(; X = Zg)
            @test isequal(apply_preprocessing(ptr, pg).X,
                          prices_to_returns(Zg; gap_return_alg = CatchUpGapReturn()).X)
            @test fit_preprocessing(ptr, pg) === ptr
        end

        @testset "a zero price is not a gap, so its returns keep their values" begin
            # #894. A zero price gives -1 (simple) or -Inf (log) on its own observation and
            # Inf on the next. Both returns read two observed prices, so no algorithm can
            # write them. The writable set once admitted every non-finite cell in the span,
            # and `RogueGapReturn` overwrote the Inf with -99.
            zts = Date(2020, 1, 1):Day(1):Date(2020, 1, 4)
            pz = [10.0, 0.0, 12.0, 13.0]
            Zz = TimeArray(zts, hcat(pz, [1.0, 2, 3, 4]), ["A", "B"])
            base = prices_to_returns(Zz)
            @test base.X[1:2, 1] == [-1.0, Inf]
            @test prices_to_returns(Zz; ret_method = :log).X[1:2, 1] == [-Inf, Inf]
            @test !any(PortfolioOptimisers.gap_return_writable(pz, base.X[:, 1]))
            rogue = @test_logs (:info,) match_mode=:any prices_to_returns(Zz;
                                                                          gap_return_alg = RogueGapReturn())
            @test isequal(rogue.X, base.X)

            # A zero price beside a gap. Only the two returns that read the gapped price
            # are writable, and the -1 onto the zero price keeps its value.
            pzg = [10.0, 0.0, NaN, 12.0]
            Zzg = TimeArray(zts, hcat(pzg, [1.0, 2, 3, 4]), ["A", "B"])
            rzg = prices_to_returns(Zzg).X[:, 1]
            @test findall(PortfolioOptimisers.gap_return_writable(pzg, rzg)) == [2, 3]
            @test isequal(prices_to_returns(Zzg; gap_return_alg = RogueGapReturn()).X[:, 1],
                          [-1.0, -99.0, -99.0])
        end

        @testset "the catch-up value keeps the element type of the prices" begin
            p32 = Float32[100, NaN, NaN, 110, 121, 133.1]
            Z32 = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 6),
                            hcat(p32, Float32[1, 2, 3, 4, 5, 6]), ["A", "B"])
            r32 = prices_to_returns(Z32; gap_return_alg = CatchUpGapReturn())
            @test eltype(r32.X) == Float32
            @test findall(!isfinite, view(r32.X, :, 1)) == [1, 2]
            @test r32.X[3, 1] ===
                  PortfolioOptimisers.gap_return_value(:simple, 110.0f0, 100.0f0)
            @test r32.X[3, 1] ≈ 110.0f0 / 100.0f0 - 1
        end

        @testset "an algorithm that returns the wrong length is refused" begin
            @test_throws DimensionMismatch prices_to_returns(Zg;
                                                             gap_return_alg = ShortGapReturn())
        end
    end
    @testset "prices_to_returns validation" begin
        vts = Date(2020, 1, 1):Day(1):Date(2020, 1, 4)
        Xv = TimeArray(vts, [10.0 1.0; 11.0 2.0; 12.0 3.0; 13.0 4.0], ["A", "B"])
        @test_throws ArgumentError prices_to_returns(Xv; ret_method = :foo)
        @test_throws ArgumentError PricesToReturns(; ret_method = :foo)
        # The implied volatilities must cover the returns clock.
        ivv = TimeArray(Date(2021, 1, 1):Day(1):Date(2021, 1, 4), fill(0.2, 4, 2),
                        ["A", "B"])
        @test_throws ArgumentError prices_to_returns(PricesResult(; X = Xv, iv = ivv))
    end
    @testset "the conversion takes a carrier, and a bare table runs the layer" begin
        # Map #955, ADR 0133. A keyword survives on the conversion if and only if it changes
        # the arithmetic of a return, so the estimator carries three such fields and the
        # verb takes three keywords beside the carrier it converts. The fourth field is
        # the partial-fit state ADR 0106 admits, which no keyword of the verb names.
        @test fieldnames(PricesToReturns) ==
              (:ret_method, :padding, :gap_return_alg, :cache)
        tsc = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 5))
        Xc = TimeArray(tsc, [10.0 30.0; 11.0 31.0; 12.0 32.0; 13.0 33.0; 14.0 34.0],
                       [:A, :B])
        Fc = TimeArray(tsc, reshape(collect(50.0:54.0), 5, 1), [:F1])

        # Every datum the conversion reads is already a field of `PricesResult`, so naming
        # one as a keyword would be a second way to say what the carrier says. None of them
        # is a keyword any more, on the verb or on the estimator.
        for kw in
            (:F, :B, :iv, :ivpa, :pnl, :span, :join_method, :collapse_args, :map_func)
            @test_throws MethodError prices_to_returns(Xc; (kw => nothing,)...)
        end
        for f in (:collapse_args, :map_func, :join_method)
            @test f ∉ fieldnames(PricesToReturns)
        end
        # The second positional argument goes with them: a factor block reaches the
        # conversion on the carrier, which is where the layer puts it.
        @test_throws MethodError prices_to_returns(Xc, Fc)

        # The bare call is exactly the two steps, and it is the layer's own path rather than
        # a way around it, so it emits the `AssetPanel` the layer always emits. A carrier
        # built by hand states no span, and therefore no universe.
        pr = price_ingestion(PriceIngestion(), Xc)
        routed = prices_to_returns(Xc)
        @test isequal(routed.X, prices_to_returns(pr).X)
        @test routed.ts == tsc[2:end]
        @test isa(routed.pnl, AssetPanel)
        @test isnothing(prices_to_returns(PricesResult(; X = Xc)).pnl)

        # The three surviving keywords reach the carrier form unchanged.
        @test prices_to_returns(pr; ret_method = :log).X ≈
              prices_to_returns(Xc; ret_method = :log).X
        @test size(prices_to_returns(pr; padding = true).X, 1) == length(tsc)

        # A carrier whose covariate series states a clock of its own is refused by name
        # rather than joined onto the asset clock: a join adds or drops an observation, and
        # `price_ingestion` owns every clock move.
        @test_throws ConflictingArgumentError prices_to_returns(PricesResult(; X = Xc,
                                                                             F = Fc[tsc[1:4]]))
        @test_throws ConflictingArgumentError prices_to_returns(PricesResult(; X = Xc,
                                                                             B = Fc[tsc[1:4]]))
    end
    @testset "the carrier views recover rows by timestamp and name the derived columns" begin
        ts = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 6))
        # The rows come back in the order of the selection, not sorted, as a Vector{Int}.
        rows = PortfolioOptimisers.matched_row_indices(ts[[5, 2, 4]], ts)
        @test rows == [5, 2, 4]
        @test rows isa Vector{Int}
        @test PortfolioOptimisers.matched_row_indices(Date[], ts) == Int[]
        # A timestamp that appears twice in the clock matches its first position.
        @test PortfolioOptimisers.matched_row_indices([ts[2]], ts[[1, 2, 2, 3]]) == [2]
        # Both messages name the carriers that hold an axis on a clock, not the feature axis.
        err = try
            PortfolioOptimisers.matched_row_indices([Date(2019, 1, 1)], ts)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Asset Panel or Listing Span", err.msg)
        @test occursin("2019-01-01", err.msg)
        err = try
            PortfolioOptimisers.matched_row_indices(nothing, ts)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Asset Panel or Listing Span", err.msg)

        # A panel with its two masks and no Panel Field has an observation axis.
        pm = AssetPanel(; pf = PortfolioOptimisers.AbstractPanelField[], amsk = trues(6, 3),
                        emsk = trues(6, 3))
        @test !PortfolioOptimisers.panel_is_static(pm)
        @test PortfolioOptimisers.feature_row_indices(pm, ts[[6, 1]], ts) == [6, 1]
        @test PortfolioOptimisers.panel_feature_names(pm) == String[]

        # The names follow the column order of the derived Feature Matrix, observed masks
        # after the values of their own field.
        pnl = AssetPanel(;
                         pf = [NumericPanelField(; name = "mcap", vals = [1.0, 2.0],
                                                 omsk = [true, false]),
                               TensorPanelField(; name = "beta", axis = "f",
                                                labels = ["a", "b"],
                                                vals = [1.0 2.0; 3.0 4.0],
                                                omsk = trues(2, 2))])
        @test PortfolioOptimisers.panel_feature_names(pnl) ==
              ["mcap", "mcap::observed", "beta=a", "beta=b", "beta=a::observed",
               "beta=b::observed"]
        @test PortfolioOptimisers.panel_feature_names(pnl) == panel_feature_matrix(pnl)[1]

        # The view cuts the label axis of a square tensor field only when it has the names.
        @test isnothing(PortfolioOptimisers.asset_panel_view(nothing, 1:2, 1:2, nothing))
        sq = AssetPanel(;
                        pf = [TensorPanelField(; name = "prox", axis = "asset",
                                               labels = ["A", "B", "C"],
                                               vals = [1.0 2.0 3.0; 4.0 5.0 6.0;
                                                       7.0 8.0 9.0])])
        @test panel_feature_matrix(PortfolioOptimisers.asset_panel_view(sq, :, [2, 3],
                                                                        ["A", "B", "C"])) ==
              (["prox=B", "prox=C"], [5.0 6.0; 8.0 9.0])
        @test panel_feature_matrix(PortfolioOptimisers.asset_panel_view(sq, :, [2, 3],
                                                                        nothing)) ==
              (["prox=A", "prox=B", "prox=C"], [4.0 5.0 6.0; 7.0 8.0 9.0])
        # A time-varying panel is cut on both axes, and a static one ignores the rows.
        Z3 = reshape(Float64.(1:36), 6, 3, 2)
        tv = PortfolioOptimisers.asset_panel_view(matrix_panel(["f1", "f2"], Z3), [2, 5],
                                                  [1, 3], nothing)
        @test panel_feature_matrix(tv)[2] == Z3[[2, 5], [1, 3], :]
        st = PortfolioOptimisers.asset_panel_view(matrix_panel(["f1", "f2"], Z3[1, :, :]),
                                                  [2, 5], [1, 3], nothing)
        @test panel_feature_matrix(st)[2] == Z3[1, [1, 3], :]
    end
    @testset "the preprocessing family: its levels, its missing values and its stubs" begin
        PO = PortfolioOptimisers
        # `missing` and a `NaN` of any number type are missing. An infinity, a zero, a
        # rational infinity and a value that is not a number are not, and the answer is a Bool.
        for v in (missing, NaN, NaN32, big(NaN), complex(NaN, 0.0))
            @test PO.is_missing_value(v) === true
        end
        for v in (Inf, -Inf, 0.0, 1, 1 // 0, nothing, "a", Date(2020, 1, 1), :x)
            @test PO.is_missing_value(v) === false
        end

        # An estimator that reads and writes one level subtypes that level. The two that
        # change the level or work at both subtype the root alone.
        @test PriceGapFill <: PO.AbstractPricesPreprocessingEstimator
        @test MissingDataFilter <: PO.AbstractPricesPreprocessingEstimator
        @test PO.AbstractAssetSelector <: PO.AbstractReturnsPreprocessingEstimator
        for T in (PricesToReturns, TrainTestSplit)
            @test T <: PO.AbstractPreprocessingEstimator
            @test !(T <: PO.AbstractPricesPreprocessingEstimator)
            @test !(T <: PO.AbstractReturnsPreprocessingEstimator)
        end
        @test PriceGapFillResult <: PO.AbstractPricesPreprocessingResult
        @test MissingDataFilterResult <: PO.AbstractPricesPreprocessingResult
        @test AssetSelectorResult <: PO.AbstractReturnsPreprocessingResult
        # A holdout applies to no other window, so its result is not a preprocessing result.
        @test !(TrainTestSplitResult <: PO.AbstractPreprocessingResult)

        pr = PricesResult(;
                          X = TimeArray(collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 4)),
                                        [1.0 2.0; 1.1 2.2; 1.2 2.1; 1.3 2.3], [:a, :b]))
        # A stateless estimator is its own fitted object, and the conversion changes the level.
        ptr = PricesToReturns()
        @test fit_preprocessing(ptr, pr) === ptr
        rd = apply_preprocessing(ptr, pr)
        @test rd isa ReturnsResult
        @test fit_preprocessing(TrainTestSplit(), pr) isa TrainTestSplitResult

        # A pipeline runs a price-level estimator on the prices slot alone, and a fitted
        # result of one level passes a window of the other level through with no change.
        ctx = PO.PipelineContext(; prices = pr)
        res, ctx′ = PO.run_step(MissingDataFilter(; col_thr = 0.0), ctx)
        @test res isa MissingDataFilterResult
        @test ctx′.prices isa PricesResult
        @test isnothing(ctx′.returns)
        @test PO.apply_fitted_step(res, rd) === rd
        @test PO.apply_fitted_step(AssetSelectorResult([:a]), pr) === pr
        # A direct subtype of the root with no step method of its own is refused.
        @test_throws "is not steppable" PO.run_step(UnimplementedPreprocessing(), ctx)

        # Both stubs name the type that does not implement the method.
        msg(f) =
            try
                f()
            catch e
                e isa ArgumentError ? e.msg : rethrow()
            end
        @test occursin("UnimplementedPreprocessing subtypes AbstractPreprocessingEstimator",
                       msg(() -> fit_preprocessing(UnimplementedPreprocessing(), pr)))
        @test occursin("UnimplementedPreprocessingResult subtypes AbstractPreprocessingEstimator or AbstractPreprocessingResult",
                       msg(() -> apply_preprocessing(UnimplementedPreprocessingResult(), pr)))
    end
end

#=
Check `src/03_InputData/14_CrossSectionalTransforms.jl` against the rules its docstrings
state, and against the oracle that the map of issue #643 ports. Issue #716.

THREE FACTS SHAPE THE PROBES.

1. THE STORED MATRICES COME FROM THE ORACLE. Every `ORACLE_*` matrix below was produced
   by the oracle's own five transformers, driven on the inputs written beside them. `The five
   members reproduce the oracle` compares each one cell by cell, including the position of
   every `NaN`, and states the measured worst cell beside its tolerance.

2. A SECOND, INDEPENDENT DERIVATION STANDS BESIDE THEM. `naive_winsorise`,
   `naive_tanh_shrink`, `naive_standardise`, `naive_percentile_rank` and
   `naive_gaussian_rank` re-derive the same quantities in plain Julia, one observation at a
   time, sharing no code with the implementation: the rank counting is a double loop rather
   than a sorted search, and the inverse normal comes from `Distributions` rather than from
   `erfinv`. So no testset compares the file against itself.

3. THE ESTIMATION SET IS THE ONE RULE EVERY MEMBER SHARES. It is the finite cells carrying a
   positive weight when `w` is given, and the finite cells otherwise. Three testsets pin its
   three consequences separately: an observation with an empty set returns all `NaN`, a cell
   outside the set is still transformed against it, and a group too small to estimate from
   takes the whole observation's statistics.
=#

using Statistics, Distributions
include(joinpath(@__DIR__, "parity_harness.jl"))

# --- the second derivation ---------------------------------------------------------------
#
# Written one observation at a time from the definitions in the docstrings. It shares no code
# with `src/03_InputData/14_CrossSectionalTransforms.jl`.

function naive_masks(x, wrow)
    fin = isfinite.(x)
    est = isnothing(wrow) ? fin : fin .& (wrow .> 0)
    return fin, est
end
function naive_centre(x, wrow, est)
    idx = findall(est)
    return if isnothing(wrow)
        sum(x[idx]) / length(idx)
    else
        sum(wrow[i] * x[i] for i in idx) / sum(wrow[i] for i in idx)
    end
end
function naive_scale(x, est, mu)
    idx = findall(est)
    return length(idx) < 2 ? 0.0 : sqrt(sum((x[i] - mu)^2 for i in idx) / (length(idx) - 1))
end
function naive_winsorise(X, w, low, high)
    Y = fill(NaN, size(X))
    for t in axes(X, 1)
        wrow = isnothing(w) ? nothing : view(w, t, :)
        _, est = naive_masks(view(X, t, :), wrow)
        if !any(est)
            continue
        end
        v = X[t, findall(est)]
        qlo, qhi = quantile(v, low), quantile(v, high)
        Y[t, :] .= clamp.(view(X, t, :), qlo, qhi)
    end
    return Y
end
function naive_tanh_shrink(X, w, knee)
    Y = fill(NaN, size(X))
    for t in axes(X, 1)
        wrow = isnothing(w) ? nothing : view(w, t, :)
        _, est = naive_masks(view(X, t, :), wrow)
        if !any(est)
            continue
        end
        v = X[t, findall(est)]
        m = median(v)
        s = 1.4826022185056018 * median(abs.(v .- m))
        h = knee * s
        Y[t, :] .= s > 1e-12 ? m .+ h .* tanh.((view(X, t, :) .- m) ./ h) : view(X, t, :)
    end
    return Y
end
function naive_standardise(X, w)
    Y = fill(NaN, size(X))
    for t in axes(X, 1)
        wrow = isnothing(w) ? nothing : view(w, t, :)
        fin, est = naive_masks(view(X, t, :), wrow)
        if !any(est)
            continue
        end
        mu = naive_centre(view(X, t, :), wrow, est)
        s = naive_scale(view(X, t, :), est, mu)
        for i in axes(X, 2)
            Y[t, i] = !fin[i] ? NaN : (s > 1e-12 ? (X[t, i] - mu) / s : 0.0)
        end
    end
    return Y
end
function naive_percentile_rank(X, w)
    P = fill(NaN, size(X))
    for t in axes(X, 1)
        wrow = isnothing(w) ? nothing : view(w, t, :)
        fin, est = naive_masks(view(X, t, :), wrow)
        n = count(est)
        if iszero(n)
            continue
        end
        for i in axes(X, 2)
            if !fin[i]
                continue
            end
            nlt = count(j -> est[j] && X[t, j] < X[t, i], axes(X, 2))
            nle = count(j -> est[j] && X[t, j] <= X[t, i], axes(X, 2))
            P[t, i] = clamp((nlt + nle) / (2n), 0.5 / n, 1 - 0.5 / n)
        end
    end
    return P
end
function naive_gaussian_rank(X, w, scale)
    P = naive_percentile_rank(X, w)
    Q = [isnan(p) ? NaN : quantile(Normal(), p) for p in P]
    Y = fill(NaN, size(X))
    for t in axes(X, 1)
        wrow = isnothing(w) ? nothing : view(w, t, :)
        fin, est = naive_masks(view(X, t, :), wrow)
        if !any(est)
            continue
        end
        mu = naive_centre(view(Q, t, :), wrow, est)
        C = [fin[i] ? Q[t, i] - mu : NaN for i in axes(X, 2)]
        if !scale
            Y[t, :] .= C
            continue
        end
        s = naive_scale(C, est, 0.0)
        for i in axes(X, 2)
            Y[t, i] = !fin[i] ? NaN : (s > 1e-12 ? C[i] / s : 0.0)
        end
    end
    return Y
end

# Cell-by-cell agreement, with every `NaN` in the same place.
function same_matrix(A, B; atol = 1e-12)
    if !(size(A) == size(B))
        return false
    end
    if !(all(isnan.(A) .== isnan.(B)))
        return false
    end
    return all(isnan(a) || isapprox(a, b; atol = atol) for (a, b) in zip(A, B))
end

# Cell-by-cell agreement with a stored oracle matrix, with every `NaN` in the same place.
# Measured on 2026-10-07: 11 of the 18 cases are bit-equal, and the worst cell is maxrel
# 3.1e-15 (maxabs 3.3e-16), on `GAUSSIAN_WG`. The Gaussian rank goes through an inverse normal
# that the oracle computes by another routine, so its cells differ by a few ulps, and a small
# cell near the median shows that as a larger relative difference. The ranks, the
# standardisations and the clips differ by 2e-16 at most. No cell needs an absolute tolerance.
oracle_matrix(A, B, name) = parity_compare(A, B; rtol = 1e-14, name).ok

# --- the inputs --------------------------------------------------------------------------

# The four-asset panel the oracle documents its own transformers on. The
# `NaN`s are the blanks, and the two weight matrices are the same estimation universe read
# two ways: `DOC_MASK_W` selects, and `DOC_W` selects and weights.
const DOC_X = [1.0 NaN 3.0 4.0; 4.0 3.0 2.0 1.0; 10.0 20.0 NaN 40.0]
const DOC_MASK_W = [1.0 0.0 1.0 1.0; 1.0 0.0 1.0 1.0; 1.0 1.0 0.0 1.0]
const DOC_W = [3.0 0.0 1.0 2.0; 4.0 0.0 2.0 3.0; 2.0 3.0 0.0 5.0]
const DOC_G = [0 0 1 1; 0 0 1 1; 0 0 1 1]

# A nine-observation panel that carries every awkward observation at once: row 3 has an empty
# estimation set, row 4 is constant, row 5 holds a three-way tie, row 6 holds a single
# estimation asset, row 7 holds finite cells outside the estimation set, row 8 holds two
# assets with no group, and row 9 puts every asset in one group.
const RAND_X = [0.9167957292444923 -0.2346415855536521 NaN -1.8344670329243553 1.7512914063517457 -1.915160074432138 -2.83204471683961;
                NaN 0.8278383772321528 3.0584612838608565 -3.7948411456558597 0.9802467543908795 NaN -1.891796716385608;
                3.9521462606205207 -1.7992302110956562 0.28066323584990494 2.0201470488940534 1.4558208123822998 3.3299888687167467 2.5909454750713525;
                2.5 2.5 2.5 2.5 2.5 2.5 2.5;
                0.4833674175087648 0.4833674175087648 0.4833674175087648 0.830940810334497 6.5726769365446875 4.984498338092746 -0.9726551899575635;
                5.3468383172369585 -4.029556022202094 0.2317386451977349 -0.15643354451165647 1.9531666627800435 1.3469968901429301 0.26289521569212204;
                -0.3065184759687538 -0.7164225092092993 0.0910216470675862 0.4321792560280313 -2.10594472643264 3.637122847535857 1.8817284105300176;
                6.428499233323535 3.2375237670223345 1.5083789238399274 2.6099672161637946 3.4229991281360674 -1.291445788814194 2.3210754150873134;
                1.8056965631636879 -1.8203448982661925 5.891948972017129 0.7467212510705343 2.0126134625870966 4.904709938139171 0.6976501658294585]
const RAND_W = [0.5293353138499124 0.8254881420992161 0.193431752815708 0.5175582505024171 0.607300723965417 0.32042470656143174 0.4682345584677836;
                0.6035528976738073 0.8573179041511593 0.6076849715570043 0.6068553966090401 0.35383753035283527 0.33009970445355086 0.04293537002890879;
                0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                0.12291250798791231 0.34824939185828174 0.8683775294494073 0.13242557607534244 0.606589419258639 0.24523033475580702 0.4302263422780156;
                0.1765644206371929 0.18539480983631107 0.0340164908541416 0.3181686270804682 0.8953686587304909 0.5562418460520835 0.8154129546588738;
                0.0 0.0 0.0 1.0 0.0 0.0 0.0;
                0.0 0.0 0.9177651090153105 0.33878329894824777 0.45698619943304253 0.19143230229673658 0.9975335267830137;
                0.5023892905062024 0.0784366938959522 0.12574614208271728 0.9674584440088674 0.9785237030989982 0.014975039536811674 0.32338743459952624;
                0.9644706890855015 0.00914155886312662 0.949220954680857 0.9973446825549453 0.06995279607591554 0.055587314029686974 0.5168602940358353]
const RAND_G = [1 0 1 2 1 2 0;
                0 0 1 2 0 2 2;
                2 1 2 2 1 0 2;
                2 2 2 0 1 0 1;
                0 1 0 1 0 0 2;
                2 2 1 0 2 0 0;
                2 1 0 2 2 2 1;
                -1 -1 2 0 1 0 1;
                0 0 0 0 0 0 0]

# --- the oracle --------------------------------------------------------------------------
#
# Produced by the oracle on the inputs above.

const ORACLE_STD_VECTOR = [-0.806225774829855 -0.558156305651438 -0.062017367294604234 1.4263994477758974]
const ORACLE_STANDARDISE = [-1.0910894511799618 NaN 0.2182178902359925 0.8728715609439697;
                               1.161895003862225 0.3872983346207417 -0.3872983346207417 -1.161895003862225;
                               -0.8728715609439694 -0.2182178902359923 NaN 1.091089451179962]
const ORACLE_STANDARDISE_WG = [-0.5545432546903087 NaN -0.6218206328680191 1.1427251984694726;
                                  0.6225458561187999 -0.1532420568907816 0.5035012041966656 -1.165728610956177;
                                  -1.3373607497456683 0.20821245125832238 NaN 0.4100168291432739]
const ORACLE_WINSORISE = [1.4 NaN 3.0 3.8; 3.7 3.0 2.0 1.3; 12.0 20.0 NaN 36.0]
const ORACLE_WINSORISE_W = [1.4 NaN 3.0 3.8; 3.6 3.0 2.0 1.2; 12.0 20.0 NaN 36.0]
const ORACLE_TANH = [1.1247186586948457 NaN 3.0 3.983484358943085;
                        3.945606186343971 2.9979044051347525 2.0020955948652475 1.0543938136560287;
                        10.165156410569153 20.0 NaN 38.75281341305154]
const ORACLE_TANH_W = [1.1247186586948457 NaN 3.0 3.983484358943085;
                          3.875281341305154 2.983484358943085 2.0 1.0165156410569152;
                          10.165156410569153 20.0 NaN 38.75281341305154]
const ORACLE_GAUSSIAN = [-1.0 NaN 0.0 1.0;
                            1.1803020040387493 0.32693604766393136 -0.32693604766393136 -1.1803020040387493;
                            -1.0 0.0 NaN 1.0]
const ORACLE_GAUSSIAN_WG = [-0.6791367019533091 NaN -0.3454139090430476 1.1914120074514876;
                               0.6985779156934256 0.08635889733764598 0.36442411553058796 -1.1743866312782931;
                               -1.3330544887858795 0.13413753156684322 NaN 0.4527392765742459]
const ORACLE_PERCENTILE = [0.16666666666666666 NaN 0.5 0.8333333333333334;
                              0.875 0.625 0.375 0.125;
                              0.16666666666666666 0.5 NaN 0.8333333333333334]
const ORACLE_PERCENTILE_WG = [0.16666666666666666 NaN 0.25 0.75;
                                 0.8333333333333334 0.6666666666666666 0.75 0.25;
                                 0.25 0.75 NaN 0.8333333333333334]
const ORACLE_GAUSSIAN_N2 = [-0.7071067811865475 0.7071067811865475]
const ORACLE_PERCENTILE_N2 = [0.25 0.75]
const ORACLE_RAND_STANDARDISE_WG = [-0.9732008167705817 0.636947972680258 NaN 0.6795691885714986 0.8482610664787932 -1.0976576809784284 -1.1229265099574306;
                                       NaN -0.8756374006965427 1.080447936381937 -0.4495311256884143 0.9139345921394546 NaN 1.0141679977191214;
                                       NaN NaN NaN NaN NaN NaN NaN;
                                       0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                                       -1.0712573956327813 -0.6485309453126881 -1.0712573956327813 1.0636569223729115 0.8428459233594862 0.34362053221355515 -1.1508225008148423;
                                       0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                                       -0.24327457408267078 -1.0076442814090247 -0.5015503879445279 0.08454059373357573 -1.0418139396690067 1.5068126216329571 0.6208377684732352;
                                       1.1738887597746162 -0.2805141217053441 -1.0686346412704182 -0.17219720648544495 0.1654182691247276 -1.3213157849182775 -1.2642889406776139;
                                       -0.25239203459984605 -1.60893090238727 1.2763164137624106 -0.6485654336528176 -0.17498232501554672 0.9069802835781051 -0.6669234240735282]
const ORACLE_RAND_GAUSSIAN_WG = [-1.0696547877373648 0.7228452820190501 NaN 0.7228452820190501 0.7228452820190501 -1.0696547877373648 -1.0696547877373648;
                                    NaN -0.757003048462599 1.3634123286129871 -0.757003048462599 0.7053366541669216 NaN 0.7053366541669216;
                                    NaN NaN NaN NaN NaN NaN NaN;
                                    0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                                    -0.699638117711332 -0.699638117711332 -0.699638117711332 0.7280556914298366 1.2316822089792752 0.3514410777490456 -1.5365228692952078;
                                    0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                                    -0.4557525008713562 -0.98085346447882 -0.5754619697240673 0.09469446620253628 -1.1416197117323017 1.331008644137374 0.7648509021291396;
                                    0.887209798643473 -0.2205969096129938 -1.3874840205140937 0.09022163677638026 0.09022163677638026 -1.2694100956729926 -1.2694100956729926;
                                    -0.18782517131387136 -1.6451994006771897 1.2695490580494466 -0.5519677213087968 0.17631737868105415 0.5995670603242557 -0.9752174029519983]
const ORACLE_RAND_PERCENTILE_G = [0.25 0.75 NaN 0.75 0.75 0.25 0.25;
                                     NaN 0.25 0.9 0.25 0.75 NaN 0.75;
                                     0.875 0.25 0.125 0.375 0.75 0.7857142857142857 0.625;
                                     0.5 0.5 0.5 0.5 0.5 0.5 0.5;
                                     0.25 0.25 0.25 0.75 0.875 0.625 0.07142857142857142;
                                     0.8333333333333334 0.16666666666666666 0.35714285714285715 0.16666666666666666 0.5 0.8333333333333334 0.5;
                                     0.375 0.25 0.5 0.625 0.125 0.875 0.75;
                                     0.9285714285714286 0.6428571428571429 0.21428571428571427 0.75 0.75 0.25 0.25;
                                     0.5 0.07142857142857142 0.9285714285714286 0.35714285714285715 0.6428571428571429 0.7857142857142857 0.21428571428571427]
const ORACLE_RAND_TANH_W = [0.8498457315787247 -0.2394168817853104 NaN -1.829691736692697 1.5644530718248375 -1.908798591751048 -2.7793873412298233;
                               NaN 0.8278383772321528 3.021621116601644 -3.487027965653941 0.9802347671245752 NaN -1.8256649614232;
                               NaN NaN NaN NaN NaN NaN NaN;
                               2.5 2.5 2.5 2.5 2.5 2.5 2.5;
                               0.4833674175087648 0.4833674175087648 0.4833674175087648 0.825200412937653 2.028135099480493 2.0201871796463102 -0.6545444604955515;
                               5.3468383172369585 -4.029556022202094 0.2317386451977349 -0.15643354451165647 1.9531666627800435 1.3469968901429301 0.26289521569212204;
                               -0.30330298712025716 -0.7044233300567011 0.09133970051814871 0.4321792560280313 -1.9824758962839528 3.396856856400813 1.8577881769999078;
                               5.445467122458685 3.2312988916503693 1.5412344260974788 2.6099672161637946 3.409571384910116 -0.25688731337487525 2.32168842877521;
                               1.8056965631636879 -1.2822718982051673 5.1570281236189555 0.7627235664124943 2.0124919691460788 4.55194756701089 0.7159502624022387]
const ORACLE_RAND_WINSORISE_W = [0.9167957292444923 -0.2346415855536521 NaN -1.8344670329243553 1.7095666224963832 -1.915160074432138 -2.7862004847192363;
                                    NaN 0.8278383772321528 2.9753327026820573 -3.7187193684850497 0.9802467543908795 NaN -1.891796716385608;
                                    NaN NaN NaN NaN NaN NaN NaN;
                                    2.5 2.5 2.5 2.5 2.5 2.5 2.5;
                                    0.4833674175087648 0.4833674175087648 0.4833674175087648 0.830940810334497 6.4773862206375705 4.984498338092746 -0.8852938335095838;
                                    -0.15643354451165647 -0.15643354451165647 -0.15643354451165647 -0.15643354451165647 -0.15643354451165647 -0.15643354451165647 -0.15643354451165647;
                                    -0.3065184759687538 -0.7164225092092993 0.0910216470675862 0.4321792560280313 -2.0180660714926306 3.566907070055623 1.8817284105300176;
                                    6.248169227012285 3.2375237670223345 1.5083789238399274 2.6099672161637946 3.4229991281360674 -1.1234563060549467 2.3210754150873134;
                                    1.8056965631636879 -1.6692651944204535 5.8327146299844514 0.7467212510705343 2.0126134625870966 4.904709938139171 0.6976501658294585]

@testset "The five members reproduce the oracle" begin
    # The three-observation panel, read with no weights, with a mask, and with a weight and a
    # classification.
    @test oracle_matrix(cross_sectional_transform(CrossSectionalWinsoriser(; low = 0.1,
                                                                           high = 0.9),
                                                  DOC_X), ORACLE_WINSORISE, "WINSORISE")
    @test oracle_matrix(cross_sectional_transform(CrossSectionalWinsoriser(; low = 0.1,
                                                                           high = 0.9),
                                                  DOC_X; w = DOC_MASK_W),
                        ORACLE_WINSORISE_W, "WINSORISE_W")
    @test oracle_matrix(cross_sectional_transform(CrossSectionalTanhShrinker(), DOC_X),
                        ORACLE_TANH, "TANH")
    @test oracle_matrix(cross_sectional_transform(CrossSectionalTanhShrinker(), DOC_X;
                                                  w = DOC_MASK_W), ORACLE_TANH_W, "TANH_W")
    @test oracle_matrix(cross_sectional_transform(CrossSectionalStandardiser(), DOC_X),
                        ORACLE_STANDARDISE, "STANDARDISE")
    @test oracle_matrix(cross_sectional_transform(CrossSectionalStandardiser(;
                                                                             min_group_size = 2),
                                                  DOC_X; w = DOC_W, groups = DOC_G),
                        ORACLE_STANDARDISE_WG, "STANDARDISE_WG")
    @test oracle_matrix(cross_sectional_transform(CrossSectionalGaussianRank(), DOC_X),
                        ORACLE_GAUSSIAN, "GAUSSIAN")
    @test oracle_matrix(cross_sectional_transform(CrossSectionalGaussianRank(;
                                                                             min_group_size = 2),
                                                  DOC_X; w = DOC_W, groups = DOC_G),
                        ORACLE_GAUSSIAN_WG, "GAUSSIAN_WG")
    @test oracle_matrix(cross_sectional_transform(CrossSectionalPercentileRank(), DOC_X),
                        ORACLE_PERCENTILE, "PERCENTILE")
    @test oracle_matrix(cross_sectional_transform(CrossSectionalPercentileRank(;
                                                                               min_group_size = 2),
                                                  DOC_X; w = DOC_MASK_W, groups = DOC_G),
                        ORACLE_PERCENTILE_WG, "PERCENTILE_WG")
    # The census's own four-value cross-section, and the two-asset cross-section that pins
    # the smallest rank a member can carry.
    @test oracle_matrix(cross_sectional_transform(CrossSectionalStandardiser(),
                                                  [1.0 2.0 4.0 10.0]), ORACLE_STD_VECTOR,
                        "STD_VECTOR")
    @test oracle_matrix(cross_sectional_transform(CrossSectionalGaussianRank(), [1.0 2.0]),
                        ORACLE_GAUSSIAN_N2, "GAUSSIAN_N2")
    @test oracle_matrix(cross_sectional_transform(CrossSectionalPercentileRank(),
                                                  [1.0 2.0]), ORACLE_PERCENTILE_N2,
                        "PERCENTILE_N2")
    # The nine-observation panel, which carries every awkward observation at once.
    @test oracle_matrix(cross_sectional_transform(CrossSectionalWinsoriser(), RAND_X;
                                                  w = RAND_W), ORACLE_RAND_WINSORISE_W,
                        "RAND_WINSORISE_W")
    @test oracle_matrix(cross_sectional_transform(CrossSectionalTanhShrinker(), RAND_X;
                                                  w = RAND_W), ORACLE_RAND_TANH_W,
                        "RAND_TANH_W")
    @test oracle_matrix(cross_sectional_transform(CrossSectionalStandardiser(;
                                                                             min_group_size = 2),
                                                  RAND_X; w = RAND_W, groups = RAND_G),
                        ORACLE_RAND_STANDARDISE_WG, "RAND_STANDARDISE_WG")
    @test oracle_matrix(cross_sectional_transform(CrossSectionalGaussianRank(;
                                                                             min_group_size = 2),
                                                  RAND_X; w = RAND_W, groups = RAND_G),
                        ORACLE_RAND_GAUSSIAN_WG, "RAND_GAUSSIAN_WG")
    @test oracle_matrix(cross_sectional_transform(CrossSectionalPercentileRank(;
                                                                               min_group_size = 2),
                                                  RAND_X; groups = RAND_G),
                        ORACLE_RAND_PERCENTILE_G, "RAND_PERCENTILE_G")
end

@testset "An independent derivation agrees with the port" begin
    for w in (nothing, RAND_W)
        @test same_matrix(cross_sectional_transform(CrossSectionalWinsoriser(), RAND_X;
                                                    w = w),
                          naive_winsorise(RAND_X, w, 0.01, 0.99))
        @test same_matrix(cross_sectional_transform(CrossSectionalTanhShrinker(), RAND_X;
                                                    w = w),
                          naive_tanh_shrink(RAND_X, w, 3.0))
        @test same_matrix(cross_sectional_transform(CrossSectionalStandardiser(), RAND_X;
                                                    w = w), naive_standardise(RAND_X, w))
        @test same_matrix(cross_sectional_transform(CrossSectionalPercentileRank(), RAND_X;
                                                    w = w),
                          naive_percentile_rank(RAND_X, w))
        @test same_matrix(cross_sectional_transform(CrossSectionalGaussianRank(), RAND_X;
                                                    w = w),
                          naive_gaussian_rank(RAND_X, w, true))
        @test same_matrix(cross_sectional_transform(CrossSectionalGaussianRank(;
                                                                               scale = false),
                                                    RAND_X; w = w),
                          naive_gaussian_rank(RAND_X, w, false))
    end
    # A group large enough to hold the whole observation reproduces the ungrouped answer,
    # because the fallback then never fires and the re-centring is idempotent.
    @test same_matrix(cross_sectional_transform(CrossSectionalPercentileRank(;
                                                                             min_group_size = 1),
                                                RAND_X; groups = fill(4, size(RAND_X))),
                      naive_percentile_rank(RAND_X, nothing))
end

@testset "The estimation set is the finite cells that carry a positive weight" begin
    # A weight of zero removes an asset from every statistic. Winsorising with the mask must
    # therefore differ from winsorising without it exactly where the excluded asset would
    # have moved a percentile.
    Y0 = cross_sectional_transform(CrossSectionalWinsoriser(; low = 0.1, high = 0.9), DOC_X)
    Y1 = cross_sectional_transform(CrossSectionalWinsoriser(; low = 0.1, high = 0.9), DOC_X;
                                   w = DOC_MASK_W)
    @test Y0[2, :] != Y1[2, :]
    @test Y0[1, [1, 3, 4]] == Y1[1, [1, 3, 4]]
    # A weight that is positive everywhere selects every finite cell, so it recovers the
    # unweighted answer for a member that reads the weights as a mask alone.
    @test same_matrix(cross_sectional_transform(CrossSectionalWinsoriser(), RAND_X;
                                                w = ones(size(RAND_X))),
                      cross_sectional_transform(CrossSectionalWinsoriser(), RAND_X))
    @test same_matrix(cross_sectional_transform(CrossSectionalPercentileRank(), RAND_X;
                                                w = ones(size(RAND_X))),
                      cross_sectional_transform(CrossSectionalPercentileRank(), RAND_X))
    # The standardiser reads them as a weight as well, so a non-constant weight moves the
    # centre and the unweighted answer is not recovered.
    @test !same_matrix(cross_sectional_transform(CrossSectionalStandardiser(), RAND_X;
                                                 w = RAND_W),
                       cross_sectional_transform(CrossSectionalStandardiser(), RAND_X))
    @test same_matrix(cross_sectional_transform(CrossSectionalStandardiser(), RAND_X;
                                                w = fill(2.5, size(RAND_X))),
                      cross_sectional_transform(CrossSectionalStandardiser(), RAND_X))
end

@testset "An observation with an empty estimation set returns all NaN" begin
    # Row 3 of the nine-observation panel carries a zero weight at every asset, so no
    # member has anything to transform it against, and no cell of it survives.
    for ct in (CrossSectionalWinsoriser(), CrossSectionalTanhShrinker(),
               CrossSectionalStandardiser(; min_group_size = 2),
               CrossSectionalGaussianRank(; min_group_size = 2),
               CrossSectionalPercentileRank(; min_group_size = 2))
        Y = cross_sectional_transform(ct, RAND_X; w = RAND_W, groups = RAND_G)
        @test all(isnan, Y[3, :])
        @test !all(isnan, Y[4, :])
    end
    # A whole matrix of blanks is the same rule read at every observation.
    @test all(isnan,
              cross_sectional_transform(CrossSectionalStandardiser(), fill(NaN, 2, 3)))
end

@testset "A cell outside the estimation set is still transformed against it" begin
    # Row 7 of the nine-observation panel holds two finite assets of zero weight. They carry
    # a value in the output, and that value is scored against the other five.
    Y = cross_sectional_transform(CrossSectionalStandardiser(), RAND_X; w = RAND_W)
    @test all(isfinite, Y[7, [1, 2]])
    idx = [3, 4, 5, 6, 7]
    mu = sum(RAND_W[7, i] * RAND_X[7, i] for i in idx) / sum(RAND_W[7, i] for i in idx)
    s = sqrt(sum((RAND_X[7, i] - mu)^2 for i in idx) / (length(idx) - 1))
    @test Y[7, 1] ≈ (RAND_X[7, 1] - mu) / s
    # The excluded assets are scored, and they do not move the score of the included ones.
    Y2 = cross_sectional_transform(CrossSectionalStandardiser(), RAND_X[:, idx];
                                   w = RAND_W[:, idx])
    @test Y[7, idx] ≈ Y2[7, :]
end

@testset "A small group and a missing group take the whole observation's statistics" begin
    # Row 8 labels two assets `CS_MISSING_GROUP`. Raising the threshold above every group
    # size sends every asset to the fallback, which is the ungrouped answer up to the final
    # re-centring, and the ranks are then exactly the ungrouped ranks.
    @test same_matrix(cross_sectional_transform(CrossSectionalPercentileRank(;
                                                                             min_group_size = 8),
                                                RAND_X; groups = RAND_G),
                      cross_sectional_transform(CrossSectionalPercentileRank(), RAND_X))
    # A matrix labelled entirely missing is the same fallback read at every cell.
    @test same_matrix(cross_sectional_transform(CrossSectionalPercentileRank(;
                                                                             min_group_size = 1),
                                                RAND_X;
                                                groups = fill(PortfolioOptimisers.CS_MISSING_GROUP,
                                                              size(RAND_X))),
                      cross_sectional_transform(CrossSectionalPercentileRank(), RAND_X))
    # The two assets of row 8 that carry no group are ranked against the whole observation,
    # while their neighbours in a standing group are not.
    P = cross_sectional_transform(CrossSectionalPercentileRank(; min_group_size = 2),
                                  RAND_X; groups = RAND_G)
    Pg = cross_sectional_transform(CrossSectionalPercentileRank(), RAND_X)
    @test P[8, 1] == Pg[8, 1]
    @test P[8, 2] == Pg[8, 2]
    @test P[8, 5] != Pg[8, 5]
end

@testset "The two outlier members accept groups and ignore them" begin
    for ct in (CrossSectionalWinsoriser(), CrossSectionalTanhShrinker())
        @test same_matrix(cross_sectional_transform(ct, RAND_X; groups = RAND_G),
                          cross_sectional_transform(ct, RAND_X))
        @test same_matrix(cross_sectional_transform(ct, RAND_X; w = RAND_W,
                                                    groups = RAND_G),
                          cross_sectional_transform(ct, RAND_X; w = RAND_W))
    end
end

@testset "The three scoring members re-centre and re-scale the whole observation" begin
    # After the grouped pass every observation that carries dispersion has a weighted centre
    # of zero and a unit equal-weighted scale over its own estimation set.
    for ct in (CrossSectionalStandardiser(; min_group_size = 2),
               CrossSectionalGaussianRank(; min_group_size = 2))
        Y = cross_sectional_transform(ct, RAND_X; w = RAND_W, groups = RAND_G)
        for t in (1, 2, 5, 7, 8, 9)
            idx = [i for i in axes(RAND_X, 2) if isfinite(RAND_X[t, i]) && RAND_W[t, i] > 0]
            mu = sum(RAND_W[t, i] * Y[t, i] for i in idx) / sum(RAND_W[t, i] for i in idx)
            @test isapprox(mu, 0.0; atol = 1e-12)
            s = sqrt(sum(Y[t, i]^2 for i in idx) / (length(idx) - 1))
            @test isapprox(s, 1.0; atol = 1e-12)
        end
    end
    # `scale = false` recentres and stops, so the scale is whatever the ranks give.
    Y = cross_sectional_transform(CrossSectionalGaussianRank(; scale = false), RAND_X)
    idx = [i for i in axes(RAND_X, 2) if isfinite(RAND_X[1, i])]
    @test isapprox(sum(Y[1, i] for i in idx) / length(idx), 0.0; atol = 1e-12)
    @test !isapprox(sqrt(sum(Y[1, i]^2 for i in idx) / (length(idx) - 1)), 1.0; atol = 1e-6)
end

@testset "An observation with no dispersion scores zero, and one asset carries no scale" begin
    # Row 4 of the nine-observation panel is constant, so its scale is zero and every finite
    # cell scores zero rather than dividing by it.
    @test all(iszero, cross_sectional_transform(CrossSectionalStandardiser(), RAND_X)[4, :])
    # The tanh shrinker has no scale either, so it returns that observation unchanged.
    @test cross_sectional_transform(CrossSectionalTanhShrinker(), RAND_X)[4, :] ==
          RAND_X[4, :]
    # Row 6 carries a single estimation asset, so the divisor of the scale is zero and the
    # same convention applies.
    @test all(iszero,
              cross_sectional_transform(CrossSectionalStandardiser(), RAND_X; w = RAND_W)[6,
                                                                                          :])
    # A percentile rank still exists on both, because it needs an order and not a scale.
    @test all(==(0.5),
              cross_sectional_transform(CrossSectionalPercentileRank(), RAND_X)[4, :])
end

@testset "The output element type follows the inputs" begin
    @test eltype(cross_sectional_transform(CrossSectionalPercentileRank(),
                                           Float32[1.0 2.0 3.0])) === Float32
    @test eltype(cross_sectional_transform(CrossSectionalStandardiser(), [1 2 3; 4 5 6])) ===
          Float64
    @test eltype(cross_sectional_transform(CrossSectionalWinsoriser(; low = 0.1f0,
                                                                    high = 0.9f0),
                                           Float32[1.0 2.0 3.0])) === Float32
    # A weight matrix widens the answer when it is wider than the data.
    @test eltype(cross_sectional_transform(CrossSectionalStandardiser(),
                                           Float32[1.0 2.0 3.0];
                                           w = [1.0, 1.0, 1.0]' .* 1.0)) === Float64
end

@testset "Constructors refuse what their validation states" begin
    @test_throws DomainError CrossSectionalWinsoriser(; low = 0.9, high = 0.1)
    @test_throws DomainError CrossSectionalWinsoriser(; low = -0.1, high = 0.9)
    @test_throws DomainError CrossSectionalWinsoriser(; low = 0.1, high = 1.1)
    @test_throws DomainError CrossSectionalTanhShrinker(; knee = 0.0)
    @test_throws DomainError CrossSectionalTanhShrinker(; knee = Inf)
    @test_throws DomainError CrossSectionalTanhShrinker(; atol = -1.0)
    @test_throws DomainError CrossSectionalTanhShrinker(; atol = NaN)
    @test_throws DomainError CrossSectionalStandardiser(; min_group_size = 0)
    @test_throws DomainError CrossSectionalStandardiser(; atol = -1.0)
    @test_throws DomainError CrossSectionalStandardiser(; atol = Inf)
    @test_throws DomainError CrossSectionalGaussianRank(; min_group_size = 0)
    @test_throws DomainError CrossSectionalGaussianRank(; atol = -1.0)
    @test_throws DomainError CrossSectionalGaussianRank(; atol = Inf)
    @test_throws DomainError CrossSectionalPercentileRank(; min_group_size = 0)
    # The keyword constructors reach the same objects the positional ones do.
    @test CrossSectionalWinsoriser().low == 0.01
    @test CrossSectionalTanhShrinker().knee == 3.0
    @test CrossSectionalStandardiser().min_group_size == 8
    @test CrossSectionalGaussianRank().scale
    @test CrossSectionalPercentileRank().min_group_size == 8
end

@testset "The verb refuses a malformed X, w or groups" begin
    ct = CrossSectionalStandardiser()
    @test_throws PortfolioOptimisers.IsEmptyError cross_sectional_transform(ct,
                                                                            Matrix{Float64}(undef,
                                                                                            0,
                                                                                            0))
    @test_throws DomainError cross_sectional_transform(ct, [1.0 Inf; 2.0 3.0])
    @test_throws DomainError cross_sectional_transform(ct, [1.0 -Inf; 2.0 3.0])
    @test_throws DimensionMismatch cross_sectional_transform(ct, [1.0 2.0];
                                                             w = [1.0 2.0 3.0])
    @test_throws DomainError cross_sectional_transform(ct, [1.0 2.0]; w = [1.0 -1.0])
    @test_throws PortfolioOptimisers.IsNonFiniteError cross_sectional_transform(ct,
                                                                                [1.0 2.0];
                                                                                w = [1.0 NaN])
    @test_throws DimensionMismatch cross_sectional_transform(ct, [1.0 2.0];
                                                             groups = [1 2 3])
    @test_throws DomainError cross_sectional_transform(ct, [1.0 2.0]; groups = [0 -2])
    # `CS_MISSING_GROUP` itself is admitted: it is the label that says an asset has no group.
    @test size(cross_sectional_transform(ct, [1.0 2.0];
                                         groups = [0 PortfolioOptimisers.CS_MISSING_GROUP])) ==
          (1, 2)
end

@testset "cross_sectional_groups reads a one-hot block" begin
    B = reshape([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 1, 2, 3)
    @test cross_sectional_groups(B) == [1 PortfolioOptimisers.CS_MISSING_GROUP]
    B2 = zeros(2, 3, 2)
    B2[1, 1, 1] = 1.0
    B2[1, 2, 2] = 1.0
    B2[2, 3, 1] = 1.0
    @test cross_sectional_groups(B2) == [1 2 PortfolioOptimisers.CS_MISSING_GROUP;
                                         PortfolioOptimisers.CS_MISSING_GROUP PortfolioOptimisers.CS_MISSING_GROUP 1]
    # The derived labels drive the verb, which is the route the exposure build takes.
    G = cross_sectional_groups(B2)
    @test size(cross_sectional_transform(CrossSectionalPercentileRank(; min_group_size = 1),
                                         [1.0 2.0 3.0; 4.0 5.0 6.0]; groups = G)) == (2, 3)
end

@testset "cross_sectional_groups reads an Asset Panel's categorical field" begin
    # A categorical Panel Field stores its codes, so a group label is read rather than
    # recovered from a one-hot block, and every cell carries one.
    pnl = AssetPanel(;
                     pf = [CategoricalPanelField(; name = "sector", levels = ["a", "b"],
                                                 codes = [1 2 1; 2 1 1]),
                           NumericPanelField(; name = "size", vals = ones(2, 3))],
                     amsk = trues(2, 3), emsk = trues(2, 3))
    @test cross_sectional_groups(pnl, "sector") == [1 2 1; 2 1 1]
    @test_throws ArgumentError cross_sectional_groups(pnl, "size")
    @test_throws KeyError cross_sectional_groups(pnl, "country")
    # A cell a fill policy wrote is not a membership: it reads as CS_MISSING_GROUP, which is
    # what OneHotExposure and descriptor_field_values answer for the same cell.
    filled = asset_panel([CategoricalPanelInput(; name = "sector",
                                                vals = ["a" "b" missing; missing "a" "a"],
                                                alg = ForwardPanelFill(; val = "b"))];
                         amsk = trues(2, 3), emsk = trues(2, 3))
    @test cross_sectional_groups(filled, "sector") ==
          [1 2 PortfolioOptimisers.CS_MISSING_GROUP;
           PortfolioOptimisers.CS_MISSING_GROUP 1 1]
    # A field that admits no blank carries no mask, and its codes are the labels whole.
    whole = asset_panel([CategoricalPanelInput(; name = "sector",
                                               vals = ["a" "b" "b"; "b" "a" "a"])];
                        amsk = trues(2, 3), emsk = trues(2, 3))
    @test cross_sectional_groups(whole, "sector") == [1 2 2; 2 1 1]
    # The verb reads one label per observation and asset, so a static panel has none.
    stat = AssetPanel(;
                      pf = [CategoricalPanelField(; name = "sector", levels = ["a", "b"],
                                                  codes = [1, 2, 1])])
    @test_throws DimensionMismatch cross_sectional_groups(stat, "sector")
end

@testset "The internal helpers answer on their own" begin
    # An estimation set whose weights are all zero is excluded by the estimation set's own
    # definition, so the guard is reached only by a direct call. It answers zero rather than
    # dividing by nothing.
    @test PortfolioOptimisers.cross_sectional_weighted_mean([1.0 2.0], [0.0 0.0], 1,
                                                            [1, 2]) == 0.0
    @test PortfolioOptimisers.cross_sectional_weighted_mean([1.0 3.0], [1.0 3.0], 1,
                                                            [1, 2]) == 2.5
    @test PortfolioOptimisers.cross_sectional_weighted_mean([1.0 3.0], nothing, 1,
                                                            [1, 2]) == 2.0
    @test PortfolioOptimisers.cross_sectional_equal_std([1.0 3.0], 1, [1], 1.0) == 0.0
    @test PortfolioOptimisers.cross_sectional_equal_std([1.0 3.0], 1, [1, 2], 2.0) ==
          sqrt(2.0)
    @test PortfolioOptimisers.cross_sectional_indices([true false true], 1) == [1, 3]
    @test PortfolioOptimisers.cross_sectional_stat([1.0, 2.0], 2) == 2.0
    @test PortfolioOptimisers.cross_sectional_stat(3.0, 2) == 3.0
    @test PortfolioOptimisers.cross_sectional_weight_type(nothing) === Bool
    @test PortfolioOptimisers.cross_sectional_weight_type([1.0f0 2.0f0]) === Float32
    @test PortfolioOptimisers.cross_sectional_estimation_mask([true true], nothing) ==
          [true true]
    @test PortfolioOptimisers.cross_sectional_estimation_mask([true true], [1.0 0.0]) ==
          [true false]
    @test PortfolioOptimisers.cross_sectional_row_groups([true true true], [1 1 2], 1) ==
          Dict(1 => [1, 2], 2 => [3])
    @test PortfolioOptimisers.cross_sectional_row_groups([true true true],
                                                         [1 PortfolioOptimisers.CS_MISSING_GROUP 2],
                                                         1) == Dict(1 => [1], 2 => [3])
end

#=
The Span Rule and the estimation-mask rule (issue #961, map #955).

The ingestion layer seams at a Listing Span: `listing_span` reads the raw price gaps and states
the interval each asset is listed over, on the price clock; `universe_masks` projects that
interval onto the returns clock and intersects it with finiteness, giving an Asset Panel's two
universe masks. ADR 0129 fixes the pieces, ADR 0131 fixes the projection as `[first + 1, last]`,
and ADR 0132 fixes that the estimation mask is never the caller's to state.

The fixture below carries one column per row of the Span Rule's table, plus the four corner cases
the ticket names: a column that is a gap throughout, an interior gap of length one, a leading gap
that runs to the last observation but one, and a panel that is active throughout.

Two expectations of the ticket were refined after it was written, and the tests below follow the
ADRs rather than the ticket text:

  - Under padding the leading observation is active for **nobody**, because a return consumes the
    earlier price of its pair (ADR 0131). So a panel active throughout is all-`true` on the
    unpadded clock, and all-`true` below the leading row on the padded one.
  - There is no both-supplied override form. A caller's declaration replaces the active mask
    outright, and the estimation mask is always re-derived (ADR 0132), which is what makes
    `emsk ⊆ amsk` hold by construction.
=#
function span_961_prices()
    # observations × assets, one column per case:
    #  1 active throughout      2 leading gap (not yet listed)   3 trailing gap (delisted)
    #  4 interior gap, k = 1    5 gap throughout                 6 priced on the last row alone
    #  7 interior gap, k = 2    8 leading and trailing gap
    return [10.0 NaN 30.0 40.0 NaN NaN 70.0 NaN
            11.0 NaN 31.0 41.0 NaN NaN 71.0 NaN
            12.0 20.0 32.0 NaN NaN NaN NaN 82.0
            13.0 21.0 33.0 43.0 NaN NaN NaN 83.0
            14.0 22.0 NaN 44.0 NaN NaN 74.0 84.0
            15.0 23.0 NaN 45.0 NaN 60.0 75.0 NaN]
end
function span_961_returns(X::AbstractMatrix, padding::Bool)
    T = size(X, 1)
    R = X[2:T, :] ./ X[1:(T - 1), :] .- 1
    return padding ? vcat(fill(NaN, 1, size(X, 2)), R) : R
end
function span_961_amsk_padded()
    return Bool[0 0 0 0 0 0 0 0
                1 0 1 1 0 0 1 0
                1 0 1 1 0 0 1 0
                1 1 1 1 0 0 1 1
                1 1 0 1 0 0 1 1
                1 1 0 1 0 0 1 0]
end
function span_961_emsk_padded()
    return Bool[0 0 0 0 0 0 0 0
                1 0 1 1 0 0 1 0
                1 0 1 0 0 0 0 0
                1 1 1 0 0 0 0 1
                1 1 0 1 0 0 0 1
                1 1 0 1 0 0 1 0]
end
@testset "The Span Rule derives a Listing Span from a price panel" begin
    X = span_961_prices()
    span = listing_span(X)
    @test isa(span, PortfolioOptimisers.ListingSpan)
    @test size(span) == size(X)
    # One row of the Span Rule's table per column, and the corner cases beside them.
    @test span.first == [1, 3, 1, 1, 1, 6, 1, 3]
    @test span.last == [6, 6, 4, 6, 0, 6, 6, 5]
    # A leading run of gaps is an asset not yet listed; a trailing run is a delisting; an interior
    # gap leaves the asset listed, whatever its length.
    @test Matrix(span) == Bool[1 0 1 1 0 0 1 0
                               1 0 1 1 0 0 1 0
                               1 1 1 1 0 0 1 1
                               1 1 1 1 0 0 1 1
                               1 1 0 1 0 0 1 1
                               1 1 0 1 0 1 1 0]
    # A column that is a gap throughout is the empty interval, and a column priced on one
    # observation alone is a leading run that ends on the last row but one.
    @test span.first[5] > span.last[5]
    @test span.first[6] == span.last[6] == 6
    # An absent price is spelled `missing` or non-finite, and the rule unifies both.
    Xm = Matrix{Union{Missing, Float64}}(X)
    Xm[.!isfinite.(X)] .= missing
    spanm = listing_span(Xm)
    @test spanm.first == span.first
    @test spanm.last == span.last
    # A panel with no gap at all is active throughout, on the price clock.
    @test Matrix(listing_span(ones(4, 3))) == trues(4, 3)
    @test_throws PortfolioOptimisers.IsEmptyError listing_span(zeros(0, 3))
end
@testset "A Listing Span is a lazy matrix of booleans" begin
    span = listing_span(span_961_prices())
    @test isa(span, AbstractMatrix{Bool})
    @test Base.IndexStyle(typeof(span)) == IndexCartesian()
    @test span[1, 1]
    @test !span[1, 2]
    @test_throws BoundsError span[7, 1]
    @test_throws BoundsError span[1, 9]
    @test sprint(show, span) == "ListingSpan(6 × 8)"
    @test sprint(show, MIME"text/plain"(), span) == "ListingSpan(6 × 8)"
    @test_throws DimensionMismatch PortfolioOptimisers.ListingSpan([1, 2], [3], 3)
    @test_throws DomainError PortfolioOptimisers.ListingSpan([1], [2], -1)
end
@testset "universe_masks projects the span and intersects it with finiteness" begin
    X = span_961_prices()
    span = listing_span(X)
    amsk_exp = span_961_amsk_padded()
    emsk_exp = span_961_emsk_padded()
    # The padded convention: the clocks align row for row, and the leading observation carries a
    # non-finite return for everybody, so it is active for nobody.
    Rp = span_961_returns(X, true)
    amsk, emsk = universe_masks(span, Rp)
    @test size(Rp, 1) == size(X, 1)
    @test size(amsk) == size(emsk) == size(Rp)
    @test isa(amsk, PortfolioOptimisers.ListingSpan)
    @test isa(emsk, BitMatrix)
    @test Matrix(amsk) == amsk_exp
    @test Matrix(emsk) == emsk_exp
    # The compression survives the crossing: `[first + 1, last]`.
    @test amsk.first == span.first .+ 1
    @test amsk.last == span.last
    # Without padding the returns clock is one observation shorter, and the masks are the padded
    # ones with the leading row dropped.
    Ru = span_961_returns(X, false)
    amsku, emsku = universe_masks(span, Ru)
    @test size(Ru, 1) == size(X, 1) - 1
    @test size(amsku) == size(emsku) == size(Ru)
    @test Matrix(amsku) == amsk_exp[2:end, :]
    @test Matrix(emsku) == emsk_exp[2:end, :]
    @test amsku.first == span.first
    @test amsku.last == span.last .- 1
    # The estimation mask is the active mask intersected with the finiteness of the returns, and
    # it is a subset of it by construction.
    @test Matrix(emsk) == Matrix(amsk) .& isfinite.(Rp)
    @test all(Matrix(emsk) .<= Matrix(amsk))
    # An interior gap is the one thing the two masks disagree about: a Held Gap of `k + 1`
    # observations for a run of `k` gapped prices, and never at an inception or a delisting.
    @test findall(Matrix(amsk) .& .!Matrix(emsk)) ==
          CartesianIndex.([(3, 4), (4, 4), (3, 7), (4, 7), (5, 7)])
end
@testset "universe_masks refuses a statement that does not fit the returns clock" begin
    X = span_961_prices()
    span = listing_span(X)
    R = span_961_returns(X, true)
    @test_throws DimensionMismatch universe_masks(span, R[:, 1:3])
    @test_throws DimensionMismatch universe_masks(span, R[1:3, :])
    @test_throws DimensionMismatch universe_masks(span, vcat(R, R))
end
@testset "A panel active throughout gives all-true masks" begin
    X = 100.0 .+ collect(1.0:6.0) * ones(1, 3)
    span = listing_span(X)
    # Without padding both masks are all-true. Under padding the leading row is active for
    # nobody, because a return consumes the earlier price of its pair (ADR 0131).
    amsku, emsku = universe_masks(span, span_961_returns(X, false))
    @test Matrix(amsku) == trues(5, 3)
    @test Matrix(emsku) == trues(5, 3)
    amskp, emskp = universe_masks(span, span_961_returns(X, true))
    @test Matrix(amskp) == vcat(falses(1, 3), trues(5, 3))
    @test Matrix(emskp) == vcat(falses(1, 3), trues(5, 3))
end
@testset "A caller's declaration replaces the active mask outright" begin
    X = span_961_prices()
    span = listing_span(X)
    R = span_961_returns(X, true)
    # A declaration enters at the same point as the derived span, under the same
    # `AbstractMatrix{Bool}` bound. Here it says the delisted column is listed throughout, which
    # the Span Rule would deny.
    decl = Matrix(span)
    decl[:, 3] .= true
    amsk, emsk = universe_masks(decl, R)
    @test isa(amsk, BitMatrix)
    @test amsk[:, 3] == Bool[0, 1, 1, 1, 1, 1]
    # The derived answer is not merged in: it is replaced.
    @test amsk[:, 3] != span_961_amsk_padded()[:, 3]
    # The estimation mask is never the caller's to state, and is re-derived from the values, so
    # the subset invariant holds by construction rather than by refusal.
    @test Matrix(emsk) == Matrix(amsk) .& isfinite.(R)
    @test all(Matrix(emsk) .<= Matrix(amsk))
    @test emsk[:, 3] == Bool[0, 1, 1, 1, 0, 0]
    # A declaration that leaves and rejoins books no return across its absence.
    gappy = trues(6, 1)
    gappy[3:4] .= false
    amskg, _ = universe_masks(gappy, R[:, 1:1])
    @test vec(amskg) == Bool[0, 1, 0, 0, 0, 1]
    # Applied to a Listing Span's own cells, the cell-by-cell rule reproduces the compressed one,
    # under both conventions.
    @test PortfolioOptimisers.project_span(Matrix(span), 6) == span_961_amsk_padded()
    @test PortfolioOptimisers.project_span(Matrix(span), 5) ==
          span_961_amsk_padded()[2:end, :]
end
@testset "The two masks round trip through an Asset Panel" begin
    X = span_961_prices()
    span = listing_span(X)
    amsk, emsk = universe_masks(span, span_961_returns(X, true))
    ax = size(amsk)
    @test isnothing(PortfolioOptimisers.assert_panel_masks(ax, amsk, emsk))
    pnl = AssetPanel(;
                     pf = [NumericPanelField(; name = "mcap",
                                             vals = reshape(collect(1.0:48.0), 6, 8))],
                     amsk = amsk, emsk = emsk)
    @test !PortfolioOptimisers.panel_is_static(pnl)
    # The compressed active mask rides into the panel as it stands, under the public
    # `AbstractMatrix{Bool}` bound.
    @test isa(pnl.amsk, PortfolioOptimisers.ListingSpan)
    @test pnl.amsk == amsk
    @test pnl.emsk == emsk
    # The invariant is checked rather than coerced, so a pair that breaks it is refused.
    bad = trues(size(emsk)...)
    @test_throws ArgumentError PortfolioOptimisers.assert_panel_masks(ax, amsk, bad)
end
using Dates
#=
The docstrings of `03_ListingSpan.jl` against numbers. Each unit states a closed form, and the
tests below compute it cell by cell and compare it with the unit.

A window of a longer clock gives a `ListingSpan` whose bounds lie outside the window, because
`listing_span_view` moves them by the rows the window drops in front. The padded projection moved
such an opening bound to `first + 1 <= 1`, so the first row of the returns clock was active for an
asset listed before the window. The same window as a `Matrix{Bool}` leaves that row inactive. The
projection now clamps the opening bound to the clock first.
=#
function span_974_priced(x)
    return !ismissing(x) && isfinite(x)
end
function span_974_oracle(X::AbstractMatrix)
    T, N = size(X)
    a = falses(T, N)
    for i in 1:N
        ts = findall(t -> span_974_priced(X[t, i]), 1:T)
        isempty(ts) || (a[ts[1]:ts[end], i] .= true)
    end
    return a
end
function span_974_project(a::AbstractMatrix{Bool}, m::Integer)
    n, N = size(a)
    o = n - m
    return [t + o - 1 >= 1 && a[t + o - 1, i] && a[t + o, i] for t in 1:m, i in 1:N]
end
@testset "The docstrings of 03_ListingSpan.jl against numbers" begin
    rng = StableRNG(974)
    @testset "listing_span and project_span equal their closed forms" begin
        for _ in 1:500
            T = rand(rng, 1:8)
            N = rand(rng, 1:5)
            X = Matrix{Union{Missing, Float64}}(rand(rng, T, N) .+ 1)
            for k in eachindex(X)
                r = rand(rng)
                X[k] = r < 0.15 ? NaN : r < 0.25 ? missing : r < 0.3 ? Inf : X[k]
            end
            span = listing_span(X)
            a = span_974_oracle(X)
            @test Matrix(span) == a
            for m in (T, T - 1)
                @test Matrix(PortfolioOptimisers.project_span(span, m)) ==
                      PortfolioOptimisers.project_span(a, m) ==
                      span_974_project(a, m)
            end
        end
    end
    @testset "A window's bounds outside the clock project as the matrix they hold" begin
        for _ in 1:500
            n = rand(rng, 1:8)
            N = rand(rng, 1:4)
            f = rand(rng, -4:(n + 3), N)
            l = rand(rng, -4:(n + 3), N)
            span = PortfolioOptimisers.ListingSpan(f, l, n)
            for m in (n, n - 1)
                o = n - m
                p = PortfolioOptimisers.project_span(span, m)
                @test p.first == max.(f, 1) .+ (1 - o)
                @test p.last == l .- o
                @test Matrix(p) ==
                      PortfolioOptimisers.project_span(Matrix(span), m) ==
                      span_974_project(Matrix(span), m)
            end
        end
    end
    @testset "A fold window of a derived span leaves the padded first row inactive" begin
        ts = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 8))
        P = [1.0 2.0; 1.1 2.1; 1.2 2.2; 1.3 NaN; 1.4 2.4; 1.5 2.5; 1.6 2.6; 1.7 2.7]
        X = TimeArray(ts, P, [:A, :B])
        pr = PricesResult(; X = X, span = listing_span(P))
        prm = PricesResult(; X = X, span = Matrix(listing_span(P)))
        pv = PortfolioOptimisers.port_opt_view(pr, 3:8)
        @test pv.span isa PortfolioOptimisers.ListingSpan
        @test pv.span.first == [-1, -1]
        for padding in (true, false)
            rr = prices_to_returns(pv; padding = padding)
            rm = prices_to_returns(PortfolioOptimisers.port_opt_view(prm, 3:8);
                                   padding = padding)
            @test Matrix(rr.pnl.amsk) == Matrix(rm.pnl.amsk)
            @test rr.pnl.emsk == rm.pnl.emsk
        end
        rr = prices_to_returns(pv; padding = true)
        @test !any(rr.pnl.amsk[1, :])
        @test Matrix(rr.pnl.amsk) == Bool[0 0; 1 1; 1 1; 1 1; 1 1; 1 1]
        @test rr.pnl.emsk == Bool[0 0; 1 0; 1 0; 1 1; 1 1; 1 1]
    end
    @testset "An interior gap next to an end of the listing is a Held Gap" begin
        # Asset 1 has a gap after its first price, and asset 2 a gap before its last one.
        P = [1.0 NaN; NaN 2.0; 1.2 2.1; 1.3 NaN; 1.4 2.3; 1.5 NaN]
        amsk, emsk = universe_masks(listing_span(P), span_961_returns(P, true))
        @test Matrix(amsk) == Bool[0 0; 1 0; 1 1; 1 1; 1 1; 1 0]
        @test emsk == Bool[0 0; 0 0; 0 1; 1 0; 1 0; 1 0]
        # The active mask opens on the first return that reads two listed prices, which
        # is not finite here, so the asset is held with no return.
        @test amsk[2, 1] && !emsk[2, 1]
        @test amsk[5, 2] && !emsk[5, 2]
        @test all(emsk .<= Matrix(amsk))
    end
end

#=
The span-bounded price fill (issue #970, map #955).

ADR 0130 gives the ingestion layer exactly one fill, off unless a caller asks for it. It exists to
state a price convention across a suspension rather than to remove a gap, it is bounded by the
Listing Span so it touches Held Gaps alone, and it is fitted on a training window and replayed.

`CarriedPrice` states the Held Price convention and conserves wealth across a gap; a
`Num_VecToScaM` reduction states a constant and manufactures two moves the market never printed.
Both are asserted below, so the difference between them is pinned rather than assumed.

The span the fill is bounded by is the carrier's, always. A carrier that states none bounds the
fill by nothing at all and no price is written, because the window cannot supply one: a suspension
straddling its edge reads there as an inception or a delisting, so a window-local derivation fills
the wrong cells rather than fewer of them (ADR 0133). The carriers below therefore state a span, as
`price_ingestion` builds them, and the span-less case is asserted in its own testset. The seed path
— a window that opens inside a gap — is driven through `gap_fill_span` and `gap_fill_column!` with
a stated span, which is the same public `AbstractMatrix{Bool}` bound a caller's listing calendar
enters at.
=#
function pgf_970_prices()
    # observations × assets, one column per case:
    #  A interior gap, k = 2   B leading gap (not yet listed)
    #  C trailing gap (delisted)   D a gap throughout
    return [100.0 NaN 300.0 NaN
            101.0 NaN 301.0 NaN
            102.0 200.0 302.0 NaN
            NaN 201.0 303.0 NaN
            NaN 202.0 304.0 NaN
            120.0 203.0 NaN NaN
            121.0 204.0 NaN NaN
            122.0 205.0 NaN NaN]
end
function pgf_970_carrier(X::AbstractMatrix)
    ts = Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(size(X, 1) - 1))
    # The carrier states the Span Rule's reading of the whole panel, which is what
    # `price_ingestion` puts in the field. Nothing below relies on a window deriving one.
    return PricesResult(; X = TimeArray(collect(ts), X, ["A", "B", "C", "D"]),
                        span = listing_span(X))
end
function pgf_970_quiet_apply(res, pr)
    # A carrier that states a span reports nothing, so this is a belt-and-braces silence: the
    # one carrier that reports is the span-less one, and it has its own testset.
    return Logging.with_logger(Logging.NullLogger()) do
        return apply_preprocessing(res, pr)
    end
end
function pgf_970_returns(X::AbstractMatrix)
    T = size(X, 1)
    return X[2:T, :] ./ X[1:(T - 1), :] .- 1
end
@testset "A Held Price conserves wealth across a gap; a constant manufactures two moves" begin
    X = pgf_970_prices()
    pr = pgf_970_carrier(X)
    res = fit_preprocessing(PriceGapFill(), pr)
    @test isa(res, PriceGapFillResult)
    @test isa(res.fill, CarriedPrice)
    # A column with no observed price gets no fitted value and no entry, so it is untouched.
    @test res.nx == [:A, :B, :C]
    # C is delisted after its last price, so the window ends outside its listing and C gets
    # no seed (#1330). It keeps its entry.
    @test isequal(res.v, [122.0, 205.0, missing])
    Xc = values(pgf_970_quiet_apply(res, pr).X)
    # The k = 2 interior gap takes the last priced observation, and nothing else moved.
    @test Xc[4:5, 1] == [102.0, 102.0]
    @test Xc[[1, 2, 3, 6, 7, 8], 1] == X[[1, 2, 3, 6, 7, 8], 1]
    Rc = pgf_970_returns(Xc)
    # `k` zero returns inside the gap, and the whole move on the observation that ends it.
    @test Rc[3:4, 1] == [0.0, 0.0]
    @test Rc[5, 1] ≈ 120.0 / 102.0 - 1
    # Wealth is conserved: the compounded product over the gap is exactly the price ratio.
    @test prod(1 .+ Rc[3:5, 1]) ≈ 120.0 / 102.0
    # The same gap under a per-asset reduction: two fabricated moves instead of one real one.
    resm = fit_preprocessing(PriceGapFill(; fill = MedianValue()), pr)
    @test resm.fill === MedianValue()
    @test resm.v[1] ≈ 111.0
    Xm = values(pgf_970_quiet_apply(resm, pr).X)
    @test Xm[4:5, 1] ≈ [111.0, 111.0]
    Rm = pgf_970_returns(Xm)
    @test Rm[3, 1] ≈ 111.0 / 102.0 - 1
    @test Rm[4, 1] == 0.0
    @test Rm[5, 1] ≈ 120.0 / 111.0 - 1
    # Wealth is conserved over the run either way — the conventions differ in where the move
    # lands, not in how much of it there is.
    @test prod(1 .+ Rm[3:5, 1]) ≈ 120.0 / 102.0
    # A plain number and a plain function reach the same slot.
    resn = fit_preprocessing(PriceGapFill(; fill = 99.0), pr)
    @test values(pgf_970_quiet_apply(resn, pr).X)[4:5, 1] == [99.0, 99.0]
    resf = fit_preprocessing(PriceGapFill(; fill = minimum), pr)
    @test values(pgf_970_quiet_apply(resf, pr).X)[4:5, 1] == [100.0, 100.0]
end
@testset "The fill is bounded by the Listing Span under both conventions" begin
    X = pgf_970_prices()
    pr = pgf_970_carrier(X)
    for fill in (CarriedPrice(), MedianValue())
        res = fit_preprocessing(PriceGapFill(; fill = fill), pr)
        Xf = values(pgf_970_quiet_apply(res, pr).X)
        # A leading run is an asset not yet listed and a trailing run is a delisting. Neither is
        # inside the span, so neither is written whatever the convention states.
        @test all(isnan, Xf[1:2, 2])
        @test all(isnan, Xf[6:8, 3])
        # A column with no observed price at all is untouched.
        @test all(isnan, Xf[:, 4])
        # The span the Span Rule reads off the filled panel is the one it read off the raw panel,
        # so the fill cannot erase a delisting or invent a listing.
        span_raw = listing_span(X)
        span_filled = listing_span(Xf)
        @test span_filled.first == span_raw.first
        @test span_filled.last == span_raw.last
        # The active mask still reports the leading and the trailing run inactive after the fill.
        amsk, emsk = universe_masks(span_filled, pgf_970_returns(Xf))
        @test !any(Matrix(amsk)[1:2, 2])
        @test !any(Matrix(amsk)[5:7, 3])
        # A filled cell is finite in the returns, so its estimation mask entry is true — the
        # design ADR 0130 states, not an oversight — and the subset invariant still holds.
        @test all(emsk[3:5, 1])
        @test all(Matrix(emsk) .<= Matrix(amsk))
        @test isnothing(PortfolioOptimisers.assert_panel_masks(size(amsk), amsk, emsk))
    end
end
@testset "A window that opens inside a gap fills from the fitted seed" begin
    X = pgf_970_prices()
    # Fit on the head of the panel, then replay on a tail whose first observations are a gap. The
    # tail states a span of its own — the public `AbstractMatrix{Bool}` bound, which is where a
    # caller's listing calendar enters — saying the asset is listed throughout it.
    res = fit_preprocessing(PriceGapFill(), pgf_970_carrier(X[1:3, :]))
    @test res.v[1] == 102.0
    # The result records where the training window ended, which is what tells a replay
    # whether the seed precedes the window it is written onto.
    @test res.te == Date(2020, 1, 3)
    tail = copy(X[4:8, :])
    # A deliberately different in-window price: were the fill reading the window rather than the
    # seed, the two gapped observations would take 120.0 and not 102.0.
    @test tail[3, 1] == 120.0
    span = trues(size(tail))
    @test PortfolioOptimisers.gap_fill_span(span, tail, false) === span
    # Every observation of the tail follows the training window, so the seed applies from
    # its first row.
    PortfolioOptimisers.gap_fill_column!(res.fill, tail, span, 1, res.v[1], 1)
    @test tail[1:2, 1] == [102.0, 102.0]
    @test tail[3:5, 1] == X[6:8, 1]
    # Once the window observes a price of its own, the convention tracks it: a later gap fills
    # from the most recent observed price rather than from the seed.
    later = [NaN, 130.0, NaN, 140.0, NaN]
    PortfolioOptimisers.gap_fill_column!(CarriedPrice(), reshape(later, 5, 1), trues(5, 1),
                                         1, 102.0, 1)
    @test later == [102.0, 130.0, 130.0, 140.0, 140.0]
    # A reduction carries nothing, so every gap of the column takes the one fitted value.
    flat = [NaN, 130.0, NaN, 140.0, NaN]
    PortfolioOptimisers.gap_fill_column!(MedianValue(), reshape(flat, 5, 1), trues(5, 1), 1,
                                         111.0, 1)
    @test flat == [111.0, 130.0, 111.0, 140.0, 111.0]
    # The bound is read before the price, so a stated span that excludes an observation keeps the
    # fill out of it whatever the convention says.
    bounded = [NaN, NaN, 130.0, NaN, NaN]
    span_b = reshape(Bool[0, 1, 1, 1, 0], 5, 1)
    PortfolioOptimisers.gap_fill_column!(CarriedPrice(), reshape(bounded, 5, 1), span_b, 1,
                                         102.0, 1)
    @test isnan(bounded[1])
    @test isnan(bounded[5])
    # Row 1 is an absence from the listing, so the seed, a price from before it, does not
    # reach row 2: the return at row 3 would be a move across the absence.
    @test isnan(bounded[2])
    @test bounded[3:4] == [130.0, 130.0]
end
@testset "The seed is written only after the training window (#1068)" begin
    # Issue #1068. The seed is the last observed training price, so on a window that follows
    # the training window it precedes every row it is written onto. On the training window
    # itself, which a Pipeline transforms with the step it just fitted, it is a price from the
    # window's end, and a gap that opens the window inside an asset's suspension must not take
    # it: the first return the conversion would compute is a move the market never printed,
    # computed with a price observations in the future.
    ts = Date(2020, 1, 1):Day(1):Date(2020, 1, 12)
    P = [100.0 NaN 50.0 10.0; 101.0 NaN 51.0 10.1; 102.0 NaN 52.0 10.2; 103.0 20.0 53.0 10.3
         NaN 20.5 54.0 10.4; NaN 21.0 55.0 10.5; 106.0 21.5 56.0 10.6; 107.0 22.0 57.0 10.7
         108.0 22.5 58.0 10.8; 109.0 23.0 NaN 10.9; 110.0 23.5 NaN 11.0;
         111.0 24.0 NaN 11.1]
    pr = price_ingestion(PriceIngestion(), TimeArray(collect(ts), P, ["A", "B", "C", "D"]))
    # A training window that opens inside A's suspension: the span is `true` there and the
    # price is `NaN`.
    w = PortfolioOptimisers.port_opt_view(pr, 5:12)
    @test all(Matrix(w.span)[1:2, 1])
    @test all(isnan, values(w.X)[1:2, 1])
    res = fit_preprocessing(PriceGapFill(), w)
    a = findfirst(==(:A), res.nx)
    @test res.v[a] == 111.0
    @test res.te == Date(2020, 1, 12)
    # On the training window the two leading cells stay a Held Gap: no price precedes them.
    # Every later cell of the column is observed, and the other columns are untouched.
    Xw = values(apply_preprocessing(res, w).X)
    @test all(isnan, Xw[1:2, 1])
    @test Xw[3:8, 1] == P[7:12, 1]
    @test isequal(Xw[:, 2:4], P[5:12, 2:4])
    # The returns the conversion computes for A then start at its first observed price, and
    # the -4.5 % move from 111 to 106 the seed manufactured is not among them.
    rd = prices_to_returns(PricesToReturns(), apply_preprocessing(res, w))
    ra = Matrix(rd.X)[:, 1]
    @test all(isnan, ra[1:2])
    @test ra[3] ≈ 107.0 / 106.0 - 1
    @test !any(x -> isfinite(x) && x ≈ 106.0 / 111.0 - 1, ra)
    # A window that follows the training window is seeded exactly as ADR 0130 describes: its
    # leading in-span gap takes the last training price.
    head = fit_preprocessing(PriceGapFill(), PortfolioOptimisers.port_opt_view(pr, 1:4))
    @test head.te == Date(2020, 1, 4)
    @test head.v[findfirst(==(:A), head.nx)] == 103.0
    Xt = values(apply_preprocessing(head, w).X)
    @test Xt[1:2, 1] == [103.0, 103.0]
    @test Xt[3:8, 1] == P[7:12, 1]
    # A window that overlaps the training window interpolates: the rows at or before its end
    # are not seeded, and the rows after it are. Fitted on 1:5, applied to 4:8, the gap at row
    # 5 (at the training end) stays, and the gap at row 6 (after it) takes the seed.
    mid = fit_preprocessing(PriceGapFill(), PortfolioOptimisers.port_opt_view(pr, 1:5))
    @test mid.te == Date(2020, 1, 5)
    @test mid.v[findfirst(==(:A), mid.nx)] == 103.0
    ov = PortfolioOptimisers.port_opt_view(pr, 5:8)
    Xo = values(apply_preprocessing(mid, ov).X)
    @test isnan(Xo[1, 1])
    @test Xo[2, 1] == 103.0
    @test Xo[3:4, 1] == P[7:8, 1]
    # A gap the window's own earlier price precedes fills from that price whether or not the
    # seed applies: fitted and applied on 1:12, A's suspension takes 103 and C's trailing run
    # is outside the span and untouched.
    full = fit_preprocessing(PriceGapFill(), pr)
    Xf = values(apply_preprocessing(full, pr).X)
    @test Xf[5:6, 1] == [103.0, 103.0]
    @test all(isnan, Xf[10:12, 3])
    # The reduction arm is unaffected: replaying a fitted statistic on the window it was
    # fitted on is the meaning of fitted state, so the leading in-span gap takes the median.
    resm = fit_preprocessing(PriceGapFill(; fill = MedianValue()), w)
    Xm = values(apply_preprocessing(resm, w).X)
    @test Xm[1:2, 1] == fill(resm.v[findfirst(==(:A), resm.nx)], 2)
    # The walk itself: before `t0` a gap no observed price precedes stays a gap, from `t0` on
    # the seed is the carry, and an observed price replaces it either side of `t0`.
    col = [NaN, NaN, 130.0, NaN, NaN, NaN]
    PortfolioOptimisers.gap_fill_column!(CarriedPrice(), reshape(col, 6, 1), trues(6, 1), 1,
                                         102.0, 5)
    @test all(isnan, col[1:2])
    @test col[3:6] == [130.0, 130.0, 130.0, 130.0]
    col2 = [NaN, NaN, NaN, NaN, 130.0, NaN]
    PortfolioOptimisers.gap_fill_column!(CarriedPrice(), reshape(col2, 6, 1), trues(6, 1),
                                         1, 102.0, 3)
    @test all(isnan, col2[1:2])
    @test col2[3:6] == [102.0, 102.0, 130.0, 130.0]
    col3 = [NaN, NaN, NaN]
    PortfolioOptimisers.gap_fill_column!(CarriedPrice(), reshape(col3, 3, 1), trues(3, 1),
                                         1, 102.0, 4)
    @test all(isnan, col3)
end
@testset "A carrier that states no Listing Span fills nothing, and refuses under strict" begin
    X = pgf_970_prices()
    ts = Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(size(X, 1) - 1))
    bare = PricesResult(; X = TimeArray(collect(ts), copy(X), ["A", "B", "C", "D"]))
    @test isnothing(PortfolioOptimisers.prices_listing_span(bare))
    res = fit_preprocessing(PriceGapFill(), bare)
    @test !res.strict
    out = @test_logs (:warn,) match_mode = :any apply_preprocessing(res, bare)
    # An all-`false` span bounds the fill by nothing, so not one cell is written: the window
    # cannot answer the question, and answering it wrongly is worse than not answering it.
    @test isequal(values(out.X), X)
    @test PortfolioOptimisers.gap_fill_span(nothing, X, false) == falses(size(X))
    strict = fit_preprocessing(PriceGapFill(; strict = true), bare)
    @test strict.strict
    @test_throws ArgumentError apply_preprocessing(strict, bare)
    # A window that holds no gap has nothing to fill and nothing to get wrong, so a carrier
    # stating no span is silent there and the values pass through untouched.
    cleanX = 100.0 .+ collect(1.0:8.0) * ones(1, 4)
    clean = PricesResult(; X = TimeArray(collect(ts), cleanX, ["A", "B", "C", "D"]))
    quiet = fit_preprocessing(PriceGapFill(; strict = true), clean)
    cout = @test_logs apply_preprocessing(quiet, clean)
    @test values(cout.X) == values(clean.X)
    # The carrier is not mutated by an apply, and every field but `X` passes through.
    pr = pgf_970_carrier(X)
    @test all(isnan, values(pr.X)[4:5, 1])
    filled = pgf_970_quiet_apply(fit_preprocessing(PriceGapFill(), pr), pr)
    @test TimeSeries.timestamp(filled.X) == TimeSeries.timestamp(pr.X)
    @test TimeSeries.colnames(filled.X) == TimeSeries.colnames(pr.X)
    @test isnothing(filled.F) && isnothing(filled.B) && isnothing(filled.pnl)
    # A stated span must fit the window it bounds.
    @test_throws DimensionMismatch PortfolioOptimisers.gap_fill_span(trues(3, 4), X, false)
end
@testset "A fitted universe is replayed by name" begin
    X = pgf_970_prices()
    res = fit_preprocessing(PriceGapFill(), pgf_970_carrier(X))
    # A window that carries only some of the fitted names fills those and skips the rest.
    ts = Date(2021, 1, 1):Day(1):(Date(2021, 1, 1) + Day(size(X, 1) - 1))
    sub = PricesResult(; X = TimeArray(collect(ts), X[:, [1, 4]], ["A", "D"]),
                       span = listing_span(X[:, [1, 4]]))
    out = pgf_970_quiet_apply(res, sub)
    @test TimeSeries.colnames(out.X) == [:A, :D]
    @test values(out.X)[4:5, 1] == [102.0, 102.0]
    @test all(isnan, values(out.X)[:, 2])
    # A window whose names the fit never saw is left alone entirely.
    other = PricesResult(; X = TimeArray(collect(ts), X[:, 1:1], ["Z"]),
                         span = listing_span(X[:, 1:1]))
    @test isequal(values(pgf_970_quiet_apply(res, other).X), values(other.X))
end
#=
The docstrings of `08_PriceGapFill.jl` state the fill as a closed form: a gap inside the listing
takes the last observed price of its own listed run, or the seed from `t0` on while no absence
precedes it, or the fitted value under a statistic. The references below restate that form cell
by cell, and the fill must equal them on random panels and random spans.
=#
function pgf_978_is_gap(x)
    return ismissing(x) || (isa(x, Number) && isnan(x))
end
function pgf_978_held_price(X::AbstractMatrix, span::AbstractMatrix{Bool}, i::Integer,
                            v::Union{Missing, Number}, t0::Integer)
    Y = copy(X)
    for t in axes(X, 1)
        if !(span[t, i] && pgf_978_is_gap(X[t, i]))
            continue
        end
        a = findlast(u -> !span[u, i], 1:(t - 1))
        r = isnothing(a) ? 1 : a + 1
        s = findlast(u -> !pgf_978_is_gap(X[u, i]), r:(t - 1))
        if !isnothing(s)
            Y[t, i] = X[r + s - 1, i]
        elseif t >= t0 && r == 1 && !ismissing(v)
            Y[t, i] = v
        end
    end
    return Y
end
function pgf_978_constant(X::AbstractMatrix, span::AbstractMatrix{Bool}, i::Integer,
                          v::Number)
    Y = copy(X)
    for t in axes(X, 1)
        if span[t, i] && pgf_978_is_gap(X[t, i])
            Y[t, i] = v
        end
    end
    return Y
end
@testset "The docstrings of 08_PriceGapFill.jl against numbers" begin
    rng = StableRNG(978)
    for trial in 1:1500
        nobs, nas = rand(rng, 1:9), rand(rng, 1:3)
        P = Matrix{Union{Missing, Float64}}(100 .+ 10 .* rand(rng, nobs, nas))
        for k in eachindex(P)
            if rand(rng) < 0.3
                P[k] = isodd(trial) && rand(rng) < 0.5 ? missing : NaN
            end
        end
        # A Span Rule span, or a caller's span that can leave the listing and join again.
        span = rand(rng) < 0.5 ? rand(rng, Bool, nobs, nas) : listing_span(P)
        for i in 1:nas
            v = 1000.0 + i
            t0 = rand(rng, 1:(nobs + 1))
            # A `missing` seed, which the fit gives after an absence, is never written.
            held = PortfolioOptimisers.gap_fill_column!(CarriedPrice(), copy(P), span, i,
                                                        missing, t0)
            @test isequal(held, pgf_978_held_price(P, span, i, missing, t0))
            held = PortfolioOptimisers.gap_fill_column!(CarriedPrice(), copy(P), span, i, v,
                                                        t0)
            @test isequal(held, pgf_978_held_price(P, span, i, v, t0))
            con = PortfolioOptimisers.gap_fill_column!(MedianValue(), copy(P), span, i, v,
                                                       t0)
            @test isequal(con, pgf_978_constant(P, span, i, v))
        end
    end
    # A caller's span takes the asset out of the listing at row 3 and back in at row 4. The
    # gap at row 4 takes neither the price before the absence nor the seed, so the return at
    # row 5 is not a move across the absence. The fill leaves the returns as they are.
    P = reshape([10.0, 11.0, 50.0, NaN, 12.0, 13.0], :, 1)
    span = reshape(Bool[1, 1, 0, 1, 1, 1], :, 1)
    @test isequal(vec(PortfolioOptimisers.gap_fill_column!(CarriedPrice(), copy(P), span, 1,
                                                           9.0, 1)),
                  [10.0, 11.0, 50.0, NaN, 12.0, 13.0])
    ts = Date(2020, 1, 1):Day(1):Date(2020, 1, 6)
    pr = PricesResult(; X = TimeArray(collect(ts), P, [:A]), span = span)
    filled = pgf_970_quiet_apply(fit_preprocessing(PriceGapFill(), pr), pr)
    @test isequal(values(prices_to_returns(filled).X), values(prices_to_returns(pr).X))
    # Without the absence, the same gap takes the price before it.
    @test vec(PortfolioOptimisers.gap_fill_column!(CarriedPrice(), copy(P), trues(6, 1), 1,
                                                   9.0, 1))[4] == 50.0
    # The fitted value: the last observed training price, or the reduction of the observed
    # training prices. An asset with no observed price has no fitted value.
    P = [100.0 NaN 5.0; NaN NaN 6.0; 103.0 NaN NaN]
    ts = Date(2020, 1, 1):Day(1):Date(2020, 1, 3)
    pr = PricesResult(; X = TimeArray(collect(ts), P, [:a, :b, :c]), span = listing_span(P))
    held = fit_preprocessing(PriceGapFill(), pr)
    @test held.nx == [:a, :c]
    # The window ends after the delisting of `c`, so its carried seed is retired (#1330).
    @test isequal(held.v, [103.0, missing])
    @test held.te == Date(2020, 1, 3)
    med = fit_preprocessing(PriceGapFill(; fill = MedianValue()), pr)
    @test med.v == [median([100.0, 103.0]), median([5.0, 6.0])]
    @test PortfolioOptimisers.gap_fill_seed(CarriedPrice(), [1.0, 2.0, 7.0], true) == 7.0
    @test PortfolioOptimisers.gap_fill_seed(MeanValue(), [1.0, 2.0, 6.0], true) == 3.0
    # An absence after the last price retires a carried seed, and does not touch a statistic.
    @test ismissing(PortfolioOptimisers.gap_fill_seed(CarriedPrice(), [1.0, 2.0, 7.0],
                                                      false))
    @test PortfolioOptimisers.gap_fill_seed(MeanValue(), [1.0, 2.0, 6.0], false) == 3.0
    # The Held Price conserves wealth: across p0, _, _, p3 the returns are 0, 0, p3/p0 - 1.
    P = reshape([100.0, NaN, NaN, 130.0], :, 1)
    ts = Date(2020, 1, 1):Day(1):Date(2020, 1, 4)
    pr = PricesResult(; X = TimeArray(collect(ts), P, [:A]), span = listing_span(P))
    rr = prices_to_returns(pgf_970_quiet_apply(fit_preprocessing(PriceGapFill(), pr), pr))
    @test vec(values(rr.X)) ≈ [0.0, 0.0, 0.3]
    @test prod(1 .+ vec(values(rr.X))) ≈ 130.0 / 100.0
    # The replay start: one past the end on the training window, the first row of a later
    # window, and the row after the end of training on a window that overlaps it.
    ts = Date(2020, 1, 1):Day(1):Date(2020, 1, 8)
    P = reshape([100.0, 101.0, 102.0, NaN, NaN, 105.0, NaN, 107.0], :, 1)
    full = PricesResult(; X = TimeArray(collect(ts), P, [:A]), span = listing_span(P))
    res = fit_preprocessing(PriceGapFill(), PortfolioOptimisers.port_opt_view(full, 1:3))
    later = PortfolioOptimisers.port_opt_view(full, 4:8)
    @test vec(values(pgf_970_quiet_apply(res, later).X)) ==
          [102.0, 102.0, 105.0, 105.0, 107.0]
    overlap = PortfolioOptimisers.port_opt_view(full, 2:6)
    @test vec(values(pgf_970_quiet_apply(res, overlap).X)) ==
          [101.0, 102.0, 102.0, 102.0, 105.0]
    # A Float32 panel and a panel with `missing` keep their element types.
    P32 = Float32[1 NaN; NaN 2; 3 4]
    ts = Date(2020, 1, 1):Day(1):Date(2020, 1, 3)
    pr32 = PricesResult(; X = TimeArray(collect(ts), P32, [:a, :b]),
                        span = listing_span(P32))
    for fill in (CarriedPrice(), MedianValue())
        res32 = fit_preprocessing(PriceGapFill(; fill = fill), pr32)
        @test eltype(res32.v) == Float32
        @test eltype(values(pgf_970_quiet_apply(res32, pr32).X)) == Float32
    end
    Pm = Union{Missing, Float64}[1.0 missing; missing 2.0; 3.0 4.0]
    prm = PricesResult(; X = TimeArray(collect(ts), Pm, [:a, :b]), span = listing_span(Pm))
    outm = pgf_970_quiet_apply(fit_preprocessing(PriceGapFill(), prm), prm)
    @test isequal(values(outm.X), Union{Missing, Float64}[1.0 missing; 1.0 2.0; 3.0 4.0])
    # A statistic fill has no online form.
    @test !PortfolioOptimisers.supports_partial_fit(PriceGapFill(; fill = MedianValue()))
    @test PortfolioOptimisers.supports_partial_fit(PriceGapFill())
    @test_throws ArgumentError PortfolioOptimisers.partial_fit_transform(PriceGapFill(;
                                                                                      fill = MedianValue()),
                                                                         pr32)
end
#=
Issue #1330. `ff27cc22b8` (#978) stopped the carry at an absence inside the window, but a window
cannot see an absence before its first row. When the listing breaks after the last observed
training price, the fit must give no seed, or a later window books a return across the absence.
The asset keeps its entry, with a `missing` seed, so a later window still fills its gaps from its
own prices. The checks below pin the case from the issue, then pin on random caller's spans that
a later window fills as the same rows of the full clock do, and that the online form agrees.
=#
@testset "A seed does not cross an absence at the end of the training window (#1330)" begin
    ts = Date(2020, 1, 1):Day(1):Date(2020, 1, 6)
    P = reshape([10.0, 11.0, NaN, NaN, 12.0, 13.0], :, 1)
    span = reshape(Bool[1, 1, 0, 1, 1, 1], :, 1)
    full = PricesResult(; X = TimeArray(collect(ts), P, [:A]), span = span)
    res = fit_preprocessing(PriceGapFill(), PortfolioOptimisers.port_opt_view(full, 1:3))
    # The asset stays in the result, but row 3 is an absence after its last price.
    @test res.nx == [:A]
    @test isequal(res.v, [missing])
    later = pgf_970_quiet_apply(res, PortfolioOptimisers.port_opt_view(full, 4:6))
    @test isequal(vec(values(later.X)), [NaN, 12.0, 13.0])
    @test isequal(vec(values(pgf_970_quiet_apply(res, full).X)), vec(P))
    # The last price lies before an absence inside the training window, and the window ends
    # inside the listing again: the seed would still cross the absence.
    res4 = fit_preprocessing(PriceGapFill(), PortfolioOptimisers.port_opt_view(full, 1:4))
    @test isequal(res4.v, [missing])
    # With no absence after the last price, the seed stays.
    res2 = fit_preprocessing(PriceGapFill(), PortfolioOptimisers.port_opt_view(full, 1:2))
    @test res2.v == [11.0]
    # A statistic carries no price, so an absence does not retire it.
    @test fit_preprocessing(PriceGapFill(; fill = MedianValue()),
                            PortfolioOptimisers.port_opt_view(full, 1:3)).v == [10.5]
    # A carrier that states no span states no absence, so the seed stays.
    nospan = PricesResult(; X = TimeArray(collect(ts)[1:3], P[1:3, :], [:A]))
    @test fit_preprocessing(PriceGapFill(), nospan).v == [11.0]
    @test PortfolioOptimisers.gap_fill_open(span, 1, 2) === false
    @test PortfolioOptimisers.gap_fill_open(span, 1, 4) === true
    @test PortfolioOptimisers.gap_fill_open(nothing, 1, 2) === true
    # The online form retires the carried price at the end of the block that holds the absence.
    est = PortfolioOptimisers.partial_fit!(PriceGapFill(),
                                           PortfolioOptimisers.port_opt_view(full, 1:3))
    @test est.cache.held == [false]
    blk = PortfolioOptimisers.partial_fit_transform(est,
                                                    PortfolioOptimisers.port_opt_view(full,
                                                                                      4:6))[2]
    @test isequal(vec(values(blk.X)), [NaN, 12.0, 13.0])
    @test isequal(fit_preprocessing(est).v, [missing])
    rng = StableRNG(1330)
    for trial in 1:800
        nobs, nas = rand(rng, 2:10), rand(rng, 1:3)
        X = 100 .+ 10 .* rand(rng, nobs, nas)
        X[rand(rng, nobs, nas) .< 0.35] .= NaN
        sp = rand(rng) < 0.8 ? rand(rng, nobs, nas) .< 0.75 : listing_span(X)
        ts = Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(nobs - 1))
        full = PricesResult(; X = TimeArray(collect(ts), X, [:a, :b, :c][1:nas]), span = sp)
        k = rand(rng, 1:(nobs - 1))
        res = fit_preprocessing(PriceGapFill(),
                                PortfolioOptimisers.port_opt_view(full, 1:k))
        tail = pgf_970_quiet_apply(res,
                                   PortfolioOptimisers.port_opt_view(full, (k + 1):nobs))
        whole = pgf_970_quiet_apply(res, full)
        @test isequal(values(tail.X), values(whole.X)[(k + 1):nobs, :])
        # Two blocks fold to the batch fit over the whole clock, and two halves merge to it.
        a = PortfolioOptimisers.partial_fit_transform(PriceGapFill(),
                                                      PortfolioOptimisers.port_opt_view(full,
                                                                                        1:k))
        b = PortfolioOptimisers.partial_fit_transform(a[1],
                                                      PortfolioOptimisers.port_opt_view(full,
                                                                                        (k + 1):nobs))
        bres = fit_preprocessing(PriceGapFill(), full)
        @test isequal(vcat(values(a[2].X), values(b[2].X)),
                      values(pgf_970_quiet_apply(bres, full).X))
        @test isequal(fit_preprocessing(b[1]).nx, bres.nx)
        @test isequal(fit_preprocessing(b[1]).v, bres.v)
        h2 = PortfolioOptimisers.partial_fit!(PriceGapFill(),
                                              PortfolioOptimisers.port_opt_view(full,
                                                                                (k + 1):nobs))
        m = PortfolioOptimisers.merge_states(a[1].cache, h2.cache)
        @test isequal(m.v, b[1].cache.v)
        @test m.held == b[1].cache.held
    end
end

#=
The concatenation of Asset Panels along the observation axis.

`vcat` joins the parts of one universe. A panel split into two views and joined again is the
panel it came from, for each kind of Panel Field, with and without an observed mask. A part
whose schema differs from the first part is refused, one refusal per testset entry. A lazy
lift and an all-true mask stay lazy when every part holds the same one.
=#
const PO = PortfolioOptimisers

function concat_test_panel(T::Integer, N::Integer; rng = StableRNG(1401))
    num = NumericPanelField(; name = "mcap", vals = rand(rng, T, N),
                            omsk = rand(rng, T, N) .> 0.2)
    cat = CategoricalPanelField(; name = "sector", levels = ["Tech", "Energy", "Retail"],
                                codes = rand(rng, 1:3, T, N))
    ten = TensorPanelField(; name = "beta", axis = "factor",
                           labels = ["size", "value", "mom"],
                           groups = ["style", "style", "momentum"],
                           vals = randn(rng, T, N, 3), omsk = rand(rng, T, N, 3) .> 0.1)
    amsk = rand(rng, T, N) .> 0.1
    emsk = amsk .& (rand(rng, T, N) .> 0.2)
    return AssetPanel(; pf = [num, cat, ten], amsk = amsk, emsk = emsk)
end

function same_panel(a::AssetPanel, b::AssetPanel)
    return isequal(panel_feature_matrix(a), panel_feature_matrix(b)) &&
           isequal(Matrix(a.amsk), Matrix(b.amsk)) &&
           isequal(Matrix(a.emsk), Matrix(b.emsk))
end

function flat_omsk(f)
    if isnothing(f.omsk)
        return nothing
    end
    return Matrix(reshape(f.omsk, size(f.omsk, 1), :))
end

function tv_panel(pf; T::Integer, N::Integer)
    return AssetPanel(; pf = pf, amsk = trues(T, N), emsk = trues(T, N))
end

@testset "A split panel joins back to itself, for every kind of Panel Field" begin
    T, N = 9, 4
    pnl = concat_test_panel(T, N)
    parts = [PO.port_opt_view(pnl, r, :) for r in (1:3, 4:4, 5:T)]
    joined = vcat(parts...)
    @test same_panel(joined, pnl)
    @test PO.panel_axes(joined) == (T, N)
    @test [typeof(f).name.wrapper for f in joined.pf] ==
          [NumericPanelField, CategoricalPanelField, TensorPanelField]
    # Each field keeps its values and its observed mask cell for cell.
    for (f, g) in zip(joined.pf, pnl.pf)
        @test f.name == g.name
        @test isequal(flat_omsk(f), flat_omsk(g))
    end
    @test panel_field(joined, "sector").levels == ["Tech", "Energy", "Retail"]
    @test panel_field(joined, "sector").codes == panel_field(pnl, "sector").codes
    @test panel_field(joined, "beta").groups == ["style", "style", "momentum"]
    @test panel_field(joined, "beta").axis == "factor"
    @test panel_field(joined, "beta").vals == panel_field(pnl, "beta").vals
    # `reduce` gives the same panel, and so does a concatenation of one part.
    @test same_panel(reduce(vcat, parts), pnl)
    @test same_panel(vcat(pnl), pnl)
end

@testset "A part with no observed mask joins a part with one as observed cells" begin
    a = tv_panel([NumericPanelField(; name = "x", vals = [1.0 2.0; 3.0 4.0])]; T = 2, N = 2)
    b = tv_panel([NumericPanelField(; name = "x", vals = [5.0 6.0], omsk = [true false])];
                 T = 1, N = 2)
    c = vcat(a, b)
    @test c.pf[1].vals == [1.0 2.0; 3.0 4.0; 5.0 6.0]
    @test c.pf[1].omsk == [true true; true true; true false]
    # With no mask in any part, the result has none.
    @test isnothing(vcat(a, a).pf[1].omsk)
end

@testset "The element types promote as vcat promotes them" begin
    a = tv_panel([NumericPanelField(; name = "x", vals = Float32[1 2])]; T = 1, N = 2)
    b = tv_panel([NumericPanelField(; name = "x", vals = [3.0 4.0])]; T = 1, N = 2)
    @test eltype(vcat(a, b).pf[1].vals) === Float64
    @test eltype(vcat(a, a).pf[1].vals) === Float32
end

@testset "A panel with no Panel Field joins its masks" begin
    a = AssetPanel(; amsk = [true true; false true], emsk = [true false; false true])
    b = AssetPanel(; amsk = [true false], emsk = [true false])
    c = vcat(a, b)
    @test isempty(c.pf)
    @test c.pf === a.pf
    @test Matrix(c.amsk) == [true true; false true; true false]
    @test Matrix(c.emsk) == [true false; false true; true false]
end

@testset "A lazy lift and an all-true mask stay lazy" begin
    s = NumericPanelField(; name = "size", vals = [1.0, 2.0, 3.0])
    lifted(n) = AssetPanel(; pf = [PO.panel_field_lift(s, n)], amsk = PO.AllTrueMask(n, 3),
                           emsk = PO.AllTrueMask(n, 3))
    c = vcat(lifted(2), lifted(5))
    @test isa(c.pf[1].vals, PO.RepeatedLeading)
    @test size(c.pf[1].vals) == (7, 3)
    @test c.pf[1].vals == repeat([1.0 2.0 3.0], 7)
    @test isa(c.amsk, PO.AllTrueMask) && isa(c.emsk, PO.AllTrueMask)
    @test size(c.amsk) == (7, 3)
    # A lift of an equal array in a new object is the same lift.
    t = NumericPanelField(; name = "size", vals = [1.0, 2.0, 3.0])
    d = vcat(lifted(2),
             AssetPanel(; pf = [PO.panel_field_lift(t, 1)], amsk = PO.AllTrueMask(1, 3),
                        emsk = PO.AllTrueMask(1, 3)))
    @test isa(d.pf[1].vals, PO.RepeatedLeading)
    # Equal lifts of two element types stay lazy, and the element types promote as `vcat`
    # promotes them.
    lift(v, n) = AssetPanel(;
                            pf = [PO.panel_field_lift(NumericPanelField(; name = "size",
                                                                        vals = v), n)],
                            amsk = PO.AllTrueMask(n, 3), emsk = PO.AllTrueMask(n, 3))
    for (x, y) in ((Float32[1, 2, 3], [1.0, 2.0, 3.0]), ([1, 2, 3], Float32[1, 2, 3]))
        for (p, q) in ((x, y), (y, x))
            f = only(vcat(lift(p, 2), lift(q, 1)).pf).vals
            @test isa(f, PO.RepeatedLeading)
            @test eltype(f) === eltype(vcat(repeat(p', 2), q'))
            @test f == repeat([1.0 2.0 3.0], 3)
        end
    end
    # A lift of another array makes the field time-varying, and a mask with a false cell
    # makes the mask dense.
    u = NumericPanelField(; name = "size", vals = [4.0, 5.0, 6.0])
    e = vcat(lifted(2),
             AssetPanel(; pf = [PO.panel_field_lift(u, 1)], amsk = trues(1, 3),
                        emsk = [true false true]))
    @test !isa(e.pf[1].vals, PO.RepeatedLeading)
    @test e.pf[1].vals == [1.0 2.0 3.0; 1.0 2.0 3.0; 4.0 5.0 6.0]
    @test !isa(e.amsk, PO.AllTrueMask)
    @test Matrix(e.emsk) == [true true true; true true true; true false true]
end

@testset "Equal static panels join to the first one" begin
    mk() = AssetPanel(;
                      pf = [NumericPanelField(; name = "mcap", vals = [1.0, 2.0]),
                            CategoricalPanelField(; name = "sector", levels = ["A", "B"],
                                                  codes = [2, 1])])
    a = mk()
    @test vcat(a, mk(), mk()) === a
    b = AssetPanel(;
                   pf = [NumericPanelField(; name = "mcap", vals = [1.0, 3.0]),
                         CategoricalPanelField(; name = "sector", levels = ["A", "B"],
                                               codes = [2, 1])])
    @test_throws "part 3 holds values, observed masks or placeholder masks that differ" vcat(a,
                                                                                             mk(),
                                                                                             b)
    # The Feature Matrix holds no placeholder mask, so a placeholder `0.0` and a datum `0.0`
    # derived one Feature Matrix, and the answer was the first part, whichever it was (R7 of
    # #1630).
    p = AssetPanel(;
                   pf = [NumericPanelField(; name = "x", vals = [0.0, 1.0],
                                           omsk = [false, true], pmsk = [true, false])])
    d = AssetPanel(;
                   pf = [NumericPanelField(; name = "x", vals = [0.0, 1.0],
                                           omsk = [false, true])])
    @test_throws "part 2 holds values, observed masks or placeholder masks that differ" vcat(p,
                                                                                             d)
    @test_throws "part 2 holds values, observed masks or placeholder masks that differ" vcat(d,
                                                                                             p)
    @test vcat(p, p) === p
    # A missing mask reads as the time-varying concatenation reads it: no observed mask is
    # `true` in each cell, and no placeholder mask is `false` in each cell (#1769).
    v = [1.0, 2.0, 3.0]
    nump(; kw...) = AssetPanel(; pf = [NumericPanelField(; name = "mcap", vals = v, kw...)])
    o = [true, false, true]
    for (x, y) in ((nump(), nump(; omsk = trues(3))),
                   (nump(; omsk = o), nump(; omsk = o, pmsk = falses(3))),
                   (nump(), nump(; omsk = trues(3), pmsk = falses(3))))
        @test vcat(x, y) === x
        @test vcat(y, x) === y
    end
    msg = "part 2 holds values, observed masks or placeholder masks that differ"
    @test_throws msg vcat(nump(), nump(; omsk = o))
    @test_throws msg vcat(nump(; omsk = o, pmsk = [false, true, false]), nump(; omsk = o))
    @test_throws msg vcat(nump(; omsk = o), nump(; omsk = [true, true, false]))
    # The codes of a categorical field and the values of a tensor field are compared.
    catp(c; kw...) = AssetPanel(;
                                pf = [CategoricalPanelField(; name = "sector",
                                                            levels = ["A", "B"], codes = c,
                                                            kw...)])
    @test vcat(catp([1, 2]), catp([1, 2]; omsk = trues(2))) isa AssetPanel
    @test_throws msg vcat(catp([1, 2]), catp([2, 1]))
    tenp(x; kw...) = AssetPanel(;
                                pf = [TensorPanelField(; name = "beta", axis = "factor",
                                                       labels = ["size", "value"], vals = x,
                                                       kw...)])
    @test vcat(tenp(ones(2, 2)),
               tenp(ones(2, 2); omsk = trues(2, 2), pmsk = falses(2, 2))) isa AssetPanel
    @test_throws msg vcat(tenp(ones(2, 2)), tenp([1.0 1.0; 1.0 2.0]))
end

@testset "A part whose schema differs is refused" begin
    T, N = 2, 3
    base = concat_test_panel(T, N)
    num(name = "mcap"; n = N) = NumericPanelField(; name = name, vals = ones(T, n))
    function catf(levels = ["Tech", "Energy", "Retail"]; n = N)
        return CategoricalPanelField(; name = "sector", levels = levels,
                                     codes = ones(Int, T, n))
    end
    function ten(; axis = "factor", labels = ["size", "value", "mom"],
                 groups = ["style", "style", "momentum"], n = N)
        return TensorPanelField(; name = "beta", axis = axis, labels = labels,
                                groups = groups, vals = ones(T, n, 3))
    end
    part(pf; n = N) = tv_panel(pf; T = T, N = n)
    # The schema of `base` itself joins.
    @test PO.panel_axes(vcat(base, part([num(), catf(), ten()]))) == (2T, N)

    static = AssetPanel(; pf = [NumericPanelField(; name = "mcap", vals = ones(N))])
    @test_throws "the first part is time-varying and part 2 is static" vcat(base, static)
    @test_throws "the first part is static and part 2 is time-varying" vcat(static, base)
    narrow = part([num(; n = 2), catf(; n = 2), ten(; n = 2)]; n = 2)
    @test_throws DimensionMismatch vcat(base, narrow)
    @test_throws "got 2 assets in part 2 against 3" vcat(base, narrow)
    msg = "the same Panel Fields in the same order"
    @test_throws msg vcat(base, part([num(), catf()]))
    @test_throws msg vcat(base, part([catf(), num(), ten()]))
    @test_throws msg vcat(base, part([num("size"), catf(), ten()]))
    kind = NumericPanelField(; name = "sector", vals = ones(T, N))
    msg = "is a `NumericPanelField` in part 2 and a `CategoricalPanelField`"
    @test_throws msg vcat(base, part([num(), kind, ten()]))
    msg = "the codes of the two parts name different categories"
    @test_throws msg vcat(base, part([num(), catf(["Energy", "Tech", "Retail"]), ten()]))
    msg = "the tensor Panel Field \"beta\""
    @test_throws msg vcat(base, part([num(), catf(), ten(; axis = "loading")]))
    @test_throws msg vcat(base,
                          part([num(), catf(), ten(; labels = ["size", "mom", "value"])]))
    @test_throws msg vcat(base, part([num(), catf(), ten(; groups = ["a", "b", "c"])]))
    @test_throws msg vcat(base, part([num(), catf(), ten(; groups = nothing)]))
    # The message names the position of the part.
    @test_throws "in part 3" vcat(base, base,
                                  part([num(), catf(), ten(; axis = "loading")]))
end

@testset "Two blocks of returns data join their panels with every Panel Field" begin
    T, N = 8, 4
    pnl = concat_test_panel(T, N)
    rd = ReturnsResult(; nx = ["A", "B", "C", "D"], X = randn(StableRNG(7), T, N),
                       pnl = pnl)
    joined = PO.vcat_observations(PO.port_opt_view(rd, 1:5, :),
                                  PO.port_opt_view(rd, 6:T, :))
    @test joined.X == rd.X
    @test same_panel(joined.pnl, pnl)
end

#=
The missing-share report, the info report and the active-mask alignment of an Asset Panel (#1400).

`describe` counts the cells that the builder filled, `panel_info` prints the dimensions, the mask
coverage and the level coverage, and `panel_align_active` moves the start of each asset to the
first active observation where each named Panel Field is observed. Each is checked by hand on a
small panel, and against the oracle on the parity fixture of map #1375 with an observed mask
that has leading gaps, a field that is never observed, a categorical field with unobserved
cells, and a tensor field.
=#
const PO = PortfolioOptimisers

function summary_small_panel()
    mcap = NumericPanelField(; name = "mcap",
                             vals = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0],
                             omsk = Bool[0 1 1; 1 1 0; 1 0 0])
    sector = CategoricalPanelField(; name = "sector", levels = ["Tech", "Energy"],
                                   codes = [1 2 1; 1 2 1; 1 2 1],
                                   omsk = Bool[1 1 1; 1 1 0; 1 1 1])
    beta = TensorPanelField(; name = "beta", axis = "factor", labels = ["size", "value"],
                            vals = zeros(3, 3, 2),
                            omsk = cat(Bool[1 1 1; 1 1 1; 1 1 1], Bool[0 1 1; 1 1 1; 1 1 0];
                                       dims = 3))
    amsk = Bool[1 1 0; 1 1 1; 1 1 1]
    emsk = Bool[1 0 0; 1 1 1; 1 1 1]
    return AssetPanel(; pf = [mcap, sector, beta], amsk = amsk, emsk = emsk)
end

# The parity fixture with observed masks that the alignment acts on. Asset 1 starts
# `book_equity` late; asset 2 lists late and starts it later still; asset 4 misses it at its
# relisting, which is not a leading cell; asset 12 never reports it. `sales_ttm` is observed on
# asset 7 at the first observation only and then from observation 10, while `book_equity`
# starts on it at observation 5, so the order of the two fields matters to a one-pass rule.
# `industry` is not observed on asset 5 for its first ten observations, and the tensor field
# misses one label on asset 3 for its first three.
function summary_parity_panel()
    fx = parity_small_panel()
    pnl = fx.rd.pnl
    T, N = size(pnl.amsk)
    book = panel_field(pnl, "book_equity")
    bo = copy(book.omsk)
    bo[1:6, 1] .= false
    bo[21:26, 2] .= false
    bo[41:43, 4] .= false
    bo[:, 12] .= false
    bo[1:4, 7] .= false
    sales = panel_field(pnl, "sales_ttm")
    so = copy(sales.omsk)
    so[2:9, 7] .= false
    so[1:2, 1] .= false
    ind = panel_field(pnl, "industry")
    io_ = trues(T, N)
    io_[1:10, 5] .= false
    rng = StableRNG(1400)
    eo = trues(T, N, 3)
    eo[1:3, 3, 2] .= false
    eo[60:62, 8, 1] .= false
    expo = TensorPanelField(; name = "exposure", axis = "factor",
                            labels = ["size", "value", "mom"], vals = randn(rng, T, N, 3),
                            omsk = eo)
    pf = [NumericPanelField(; name = "book_equity", vals = book.vals, omsk = bo),
          NumericPanelField(; name = "sales_ttm", vals = sales.vals, omsk = so),
          panel_field(pnl, "adj_close"),
          CategoricalPanelField(; name = "industry", levels = ind.levels, codes = ind.codes,
                                omsk = io_), panel_field(pnl, "currency"), expo]
    return AssetPanel(; pf = pf, amsk = pnl.amsk, emsk = pnl.emsk)
end

const SUMMARY_ALIGN_CASES = ["Book" => ["book_equity"],
                             "BookSales" => ["book_equity", "sales_ttm"],
                             "SalesBook" => ["sales_ttm", "book_equity"],
                             "Industry" => ["industry"]]

@testset "describe counts the filled cells of each Panel Field" begin
    pnl = summary_small_panel()
    df = describe(pnl)
    @test df.field == ["mcap", "sector", "beta"]
    @test df.kind == [:numeric, :categorical, :tensor]
    @test df.cells == [9, 9, 18]
    @test df.missing ≈ [4 / 9, 1 / 9, 2 / 18] rtol = 1e-15
    @test df.active_cells == [8, 8, 16]
    # mcap: active and missing at (1, 1), (2, 3), (3, 2) and (3, 3); (1, 3) is inactive.
    @test df.active_missing ≈ [4 / 8, 1 / 8, 2 / 16] rtol = 1e-15
    # Asset 3 is active at observations 2 and 3, and mcap is missing at both.
    @test df.assets_missing == [1, 0, 0]
    by = describe(pnl; by = "sector")
    @test by.field == ["mcap", "mcap", "beta", "beta"]
    @test by.level == ["Tech", "Energy", "Tech", "Energy"]
    # Tech: (1, 1), (2, 1), (3, 1), (3, 3); (2, 3) is not observed and (1, 3) is inactive.
    @test by.cells == [4, 3, 8, 6]
    @test by.missing ≈ [2 / 4, 1 / 3, 2 / 8, 0 / 6] rtol = 1e-15
    @test_throws ArgumentError describe(pnl; by = "mcap")
    @test_throws KeyError describe(pnl; by = "industry")
end

@testset "describe on a static panel, an empty level and a panel with no Panel Field" begin
    pnl = AssetPanel(;
                     pf = [NumericPanelField(; name = "mcap", vals = [1.0, 2.0, 3.0],
                                             omsk = Bool[1, 0, 1]),
                           CategoricalPanelField(; name = "sector",
                                                 levels = ["Tech", "Energy", "Retail"],
                                                 codes = [1, 2, 1])])
    df = describe(pnl)
    @test df.cells == [3, 3]
    @test df.active_cells == [3, 3]
    @test df.missing == df.active_missing
    @test df.assets_missing == [1, 0]
    by = describe(pnl; by = "sector")
    @test by.cells == [2, 1, 0]
    # A share over no cell is NaN.
    @test isnan(by.missing[3])
    @test by.missing[1:2] == [0.0, 1.0]
    empty = AssetPanel(; amsk = trues(2, 2), emsk = trues(2, 2))
    @test size(describe(empty)) == (0, 7)
end

@testset "panel_info prints the dimensions, the masks, the fields and the levels" begin
    pnl = summary_small_panel()
    s = sprint(io -> panel_info(io, pnl; ts = ["t1", "t2", "t3"]))
    @test occursin("observations : 3 (t1 to t3)", s)
    @test occursin("Panel Fields : 3", s)
    @test occursin("missing      : 19.4% of the cells, 21.9% of the active cells", s)
    @test occursin("in mask          : 8 / 9 cells (88.9%)", s)
    @test occursin("in mask          : 7 / 9 cells (77.8%)", s)
    @test occursin("sector: 2 levels", s)
    @test occursin("< 10    : 2 levels (Tech, Energy)", s)
    @test !occursin("equal to the active mask", s)
    @test_throws DimensionMismatch panel_info(devnull, pnl; ts = ["t1"])
    @test isnothing(panel_info(devnull, pnl))
    # The method with no stream prints to `stdout`, so the report goes to the test log.
    @test isnothing(panel_info(pnl))
    same = AssetPanel(; pf = pnl.pf, amsk = pnl.amsk, emsk = pnl.amsk)
    @test occursin("estimation mask: equal to the active mask", sprint(panel_info, same))
    static = AssetPanel(; pf = [NumericPanelField(; name = "mcap", vals = [1.0, 2.0])])
    s = sprint(io -> panel_info(io, static; ts = [1]))
    @test occursin("observations : none, the panel is static", s)
    @test occursin("active mask: none, the panel is static", s)
    @test !occursin("categorical", s)
    # A group of more than six levels prints four names and the count of the others.
    levels = ["L$k" for k in 1:8]
    many = AssetPanel(;
                      pf = [CategoricalPanelField(; name = "g", levels = levels,
                                                  codes = [1:8;])])
    @test occursin("< 10    : 8 levels (L1, L2, L3, L4, … +4 more)",
                   sprint(panel_info, many))
    # A panel with no row prints no spread.
    norow = AssetPanel(; amsk = falses(0, 2), emsk = falses(0, 2))
    s = sprint(panel_info, norow)
    @test !occursin("assets per obs.", s)
    @test !occursin("duration", s)
end

@testset "panel_info prints the median of an even count of assets or observations" begin
    # Better than the oracle (R80 of #1416). The median of an even count of numbers is the mean
    # of its two middle numbers. The oracle takes that median and prints its integer part.
    # The four observations hold 1, 2, 3 and 4 assets, and the four assets last 4, 3, 2 and 1
    # observations. Each middle pair is 2 and 3, so each median is (2 + 3) / 2 = 2.5. The
    # oracle prints 2.
    amsk = Bool[1 0 0 0; 1 1 0 0; 1 1 1 0; 1 1 1 1]
    c = PO.panel_mask_coverage(amsk)
    @test c.per_observation == (; min = 1, median = 2.5, max = 4)
    @test c.durations == (; min = 1, median = 2.5, max = 4)
    s = sprint(panel_info, AssetPanel(; amsk = amsk, emsk = amsk))
    @test occursin("assets per obs.  : min 1, median 2.5, max 4", s)
    @test occursin("duration (obs.)  : min 1, median 2.5, max 4", s)
end

@testset "panel_align_active moves the start of each asset to its first observed cell" begin
    pnl = summary_small_panel()
    res = panel_align_active(pnl, "mcap")
    # Asset 1 starts at 2; asset 2 is observed at its first cell; asset 3 is never observed
    # while active, so it leaves the universe.
    @test res.n == 1 + 2
    @test res.pnl.amsk == Bool[0 1 0; 1 1 0; 1 1 0]
    @test res.pnl.emsk == res.pnl.emsk .& res.pnl.amsk
    @test res.pnl.emsk == Bool[0 0 0; 1 1 0; 1 1 0]
    @test res.pnl.pf === pnl.pf
    # The tensor field is observed at a pair when each label is. Its gap at (3, 3) follows
    # the first active cell of asset 3, which is observed, so it stays.
    res = panel_align_active(pnl, ["beta"])
    @test res.n == 1
    @test res.pnl.amsk == Bool[0 1 0; 1 1 1; 1 1 1]
    # The order of the fields does not change the result, and a second call removes nothing.
    a = panel_align_active(pnl, ["mcap", "beta", "sector"])
    b = panel_align_active(pnl, ["sector", "beta", "mcap"])
    @test a.pnl.amsk == b.pnl.amsk
    @test a.n == b.n
    @test panel_align_active(a.pnl, ["mcap", "beta", "sector"]).n == 0
    # No field, or a field that cannot blank, removes nothing.
    @test panel_align_active(pnl, String[]).n == 0
    lifted = AssetPanel(;
                        pf = [NumericPanelField(; name = "x",
                                                vals = PO.RepeatedLeading([1.0, 2.0, 3.0],
                                                                          3))],
                        amsk = pnl.amsk, emsk = pnl.emsk)
    @test panel_align_active(lifted, "x").n == 0
    @test_throws KeyError panel_align_active(pnl, "nope")
    static = AssetPanel(; pf = [NumericPanelField(; name = "mcap", vals = [1.0, 2.0])])
    @test_throws ArgumentError panel_align_active(static, "mcap")
end

@testset "describe equals the oracle on the parity fixture" begin
    pnl = summary_parity_panel()
    df = describe(pnl)
    # The oracle gives each share as a percentage; the stored file holds it over 100.
    ref = parity_load("AssetPanelSummary", "Small", "Describe")
    r = parity_compare([df.missing df.active_missing], ref; name = "describe")
    @test r.ok
    # The oracle's rows are level by level; ours are field by field.
    by = describe(pnl; by = "industry")
    ref = parity_load("AssetPanelSummary", "Small", "DescribeBy")
    names = [f.name for f in pnl.pf]
    lv = panel_field(pnl, "industry").levels
    ord = [findfirst(i -> by.field[i] == names[Int(row[1]) + 1] &&
                          by.level[i] == lv[Int(row[2]) + 1], eachindex(by.field))
           for row in eachrow(ref)]
    @test by.cells[ord] == Int.(ref[:, 3])
    r = parity_compare(by.missing[ord], ref[:, 4]; name = "describe by industry")
    @test r.ok
end

@testset "the numbers of panel_info equal the oracle on the parity fixture" begin
    pnl = summary_parity_panel()
    for (msk, name) in ((pnl.amsk, "InfoActive"), (pnl.emsk, "InfoEstimation"))
        c = PO.panel_mask_coverage(msk)
        ref = Int.(vec(parity_load("AssetPanelSummary", "Small", name)))
        # The oracle truncates a median to an integer. The testset above pins our median.
        @test [c.in_mask, c.cells, c.per_observation.min,
               floor(Int, c.per_observation.median), c.per_observation.max, c.assets, c.N,
               floor(Int, c.durations.median), c.durations.min, c.durations.max] == ref
    end
    # The oracle reports the fully missing assets of a numeric or categorical field.
    df = describe(pnl)
    ref = parity_load("AssetPanelSummary", "Small", "InfoFullyMissing")
    @test df.assets_missing[Int.(ref[:, 1]) .+ 1] == Int.(ref[:, 2])
    # The level groups: each row is a field, a level and a group, all zero-based.
    ref = Int.(parity_load("AssetPanelSummary", "Small", "InfoLevels"))
    s = sprint(panel_info, pnl)
    groups = ["< 10", "10 - 19", "20 - 49", ">= 50"]
    for row in eachrow(ref)
        f = pnl.pf[row[1] + 1]
        line = only(filter(l -> occursin("    $(rpad(groups[row[3] + 1], 8)):", l),
                           split(s[findfirst("  $(f.name): ", s)[1]:end], '\n')[2:5]))
        @test occursin(f.levels[row[2] + 1], line)
    end
end

@testset "panel_align_active equals the oracle on the parity fixture" begin
    pnl = summary_parity_panel()
    for (case, fs) in SUMMARY_ALIGN_CASES
        res = panel_align_active(pnl, fs)
        n = Int.(vec(parity_load("AssetPanelSummary", "Align$case", "N")))
        # The oracle's second call; one call of ours gives it.
        amsk = parity_load("AssetPanelSummary", "Align$case", "ActiveMask2")
        @test res.pnl.amsk == (amsk .== 1)
        @test res.n == sum(n)
        @test res.pnl.emsk ==
              (parity_load("AssetPanelSummary", "Align$case", "EstimationMask") .== 1) .&
              res.pnl.amsk
        first = parity_load("AssetPanelSummary", "Align$case", "ActiveMask") .== 1
        if case == "SalesBook"
            # Better: the oracle's first call walks the fields in turn, so it stops at
            # observation 5 on asset 7, where `sales_ttm` is not observed, and its second
            # call moves on to observation 10. The rule of the library is its fixed point.
            @test findfirst(first[:, 7]) == 5
            @test findfirst(res.pnl.amsk[:, 7]) == 10
            @test n[2] > 0
        else
            # The oracle's second call aligns nothing more, so its first call is the result.
            @test n[2] == 0
            @test first == res.pnl.amsk
        end
    end
    # Better: the oracle refuses a tensor field; the library aligns asset 3 past the three
    # observations where the label "value" is not observed. The gap of asset 8 is not a
    # leading one, so it stays.
    res = panel_align_active(pnl, "exposure")
    @test res.n == 3
    @test findfirst(res.pnl.amsk[:, 3]) == 4
end
