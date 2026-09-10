# Two half-implemented estimators, used to reach the dispatch-error stubs of the
# preprocessing interface. Neither implements `fit_preprocessing` or `apply_preprocessing`,
# which is the mistake the stubs exist to name.
struct UnimplementedPreprocessing <: PortfolioOptimisers.AbstractPreprocessingEstimator end
struct UnimplementedPreprocessingResult <: PortfolioOptimisers.AbstractPreprocessingResult end
# A Gap Return algorithm that ignores the invariant entirely and answers the whole column with
# one sentinel. `apply_gap_return` reads back only the writable cells, so the invariant is the
# driver's and not the algorithm's, and this is what proves it.
struct RogueGapReturn <: PortfolioOptimisers.AbstractGapReturnAlgorithm end
function PortfolioOptimisers.gap_return(::RogueGapReturn, ::AbstractVector,
                                        r::AbstractVector, ::Symbol)
    return fill(-99.0, length(r))
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
        # it, bit for bit.
        rd = prices_to_returns(Px, Py)
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
            # the conversion has nothing that drops a column.
            rrs = prices_to_returns(Px; pnl = matrix_panel(nz, Zs))
            @test rrs.nx == ["A", "B", "C"]
            @test panel_feature_matrix(rrs.pnl)[2] == Zs
            @test panel_feature_matrix(rrs.pnl)[1] == nz

            # The observation axis is the one the conversion does cut, and it cuts it by
            # exactly the observation the percentage change costs: rows 2:10, every asset.
            rr3 = prices_to_returns(Px; pnl = matrix_panel(nz, Z3))
            @test size(panel_feature_matrix(rr3.pnl)[2]) == (9, 3, 2)
            @test panel_feature_matrix(rr3.pnl)[2] == Z3[2:10, :, :]
            @test length(rr3.ts) ==
                  size(panel_feature_matrix(rr3.pnl)[2], 1) ==
                  size(rr3.X, 1)

            # Under collapse_args the aggregated period takes the features of the row at its
            # representative timestamp -- last-observation semantics.
            rc = prices_to_returns(Px; pnl = matrix_panel(nz, Z3),
                                   collapse_args = (week, last))
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
            rall = prices_to_returns(Pall, Fok; pnl = matrix_panel(nz, Zs))
            @test rall.nx == ["A", "B", "C"]
            @test all(!isfinite, rall.X)
            @test panel_feature_matrix(rall.pnl)[2] == Zs
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
            # returns_result_picker folds B into X; it must not lose Z on the way.
            rdb = ReturnsResult(; nx = ["A", "B"], X = rand(rng, 4, 2), nb = ["bm"],
                                B = rand(rng, 4), pnl = matrix_panel(nz, rand(rng, 2, 2)))
            rp = returns_result_picker(rdb, true)
            @test panel_feature_matrix(rp.pnl)[2] == panel_feature_matrix(rdb.pnl)[2]
            @test panel_feature_matrix(rp.pnl)[1] == panel_feature_matrix(rdb.pnl)[1]
            @test isnothing(returns_result_picker(rdb, false).B) == false

            # AssetSelectorResult subselects through port_opt_view, so it is carried for free.
            rds = ReturnsResult(; nx = ["A", "B", "C"], X = rand(rng, 5, 3),
                                pnl = matrix_panel(nz, Zs))
            sel = apply_preprocessing(PortfolioOptimisers.AssetSelectorResult(["A", "C"]),
                                      rds)
            @test panel_feature_matrix(sel.pnl)[2] == Zs[[1, 3], :]
        end
    end

    @testset "prices_to_returns closed form" begin
        # A two-row example, computed by hand. `TimeSeries.percentchange` routes the simple
        # branch through `expm1(ln P_t - ln P_{t-1})` rather than the quotient the docstring
        # states, so the two agree to floating point and not to the last bit.
        ts = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 3))
        P = TimeArray(ts, [100.0 50.0; 110.0 45.0; 121.0 54.0], [:a, :b])
        simple = [110/100-1 45/50-1; 121/110-1 54/45-1]
        logret = [log(110 / 100) log(45 / 50); log(121 / 110) log(54 / 45)]

        rs = prices_to_returns(P)
        @test rs.X ≈ simple
        @test rs.X ==
              expm1.(log.([110.0 45.0; 121.0 54.0]) .- log.([100.0 50.0; 110.0 45.0]))
        @test rs.nx == ["a", "b"]
        @test rs.ts == ts[2:3]

        rl = prices_to_returns(P; ret_method = :log)
        @test rl.X ≈ logret
        @test rl.X == log.([110.0 45.0; 121.0 54.0]) .- log.([100.0 50.0; 110.0 45.0])

        # `padding` keeps the first observation and fills its return with `NaN`, so the
        # returns keep the length of the price clock.
        rp = prices_to_returns(P; padding = true)
        @test rp.ts == ts
        @test all(isnan, rp.X[1, :])
        @test rp.X[2:3, :] ≈ simple

        # A `NaN` price is carried, not deleted. The clock keeps its row, and the two
        # returns of `a` that read the absent price are the ones left non-finite: nothing
        # spans the gap, because a return is the change between two consecutive observations.
        Pm = TimeArray(ts, [100.0 50.0; NaN 45.0; 121.0 54.0], [:a, :b])
        rm = prices_to_returns(Pm)
        @test rm.ts == ts[2:3]
        @test all(!isfinite, view(rm.X, :, 1))
        @test rm.X[:, 2] ≈ [45 / 50 - 1, 54 / 45 - 1]

        # `map_func` is applied to every row of the merged table, before the collapse. A
        # common scale factor leaves a return unchanged; a shift does not.
        @test prices_to_returns(P; map_func = (t, v) -> (t, 2 .* v)).X ≈ simple
        @test prices_to_returns(P; map_func = (t, v) -> (t, v .+ 100.0)).X ≈
              [210/200-1 145/150-1; 221/210-1 154/145-1]

        # Both branches run through a logarithm, so both need a positive price. The simple
        # branch throws too, although the closed form it documents is defined there.
        @test_throws DomainError prices_to_returns(TimeArray(ts,
                                                             [100.0 50.0; -110.0 45.0;
                                                              121.0 54.0], [:a, :b]))
        @test_throws DomainError prices_to_returns(TimeArray(ts,
                                                             [100.0 50.0; -110.0 45.0;
                                                              121.0 54.0], [:a, :b]);
                                                   ret_method = :log)
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
        rdrop = prices_to_returns(Pa, Pf)
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
        # fill accepts it through `is_missing_value` exactly as `NaN`; `test_06g` drives the
        # conventions and the bound on `NaN`, so what is pinned here is the `missing` path
        # and the fit/apply replay across a train/test split.
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
        find_complete_indices = PortfolioOptimisers.find_complete_indices
        is_missing_value = PortfolioOptimisers.is_missing_value

        # `missing` and `NaN` are the two conventions for an absent price, and one predicate
        # accepts both. A non-number is never missing.
        @test is_missing_value(missing)
        @test is_missing_value(NaN)
        @test !is_missing_value(1.0)
        @test !is_missing_value("a")

        # `dims = 1` reports the complete columns, `dims = 2` the complete rows. One entry
        # is enough to remove the whole column or row.
        Xm = [1.0 2.0 NaN; 4.0 missing 6.0]
        @test find_complete_indices(Xm) == [1]
        @test find_complete_indices(Xm; dims = 2) == Int[]
        @test find_complete_indices([1.0 2.0; 3.0 4.0]) == [1, 2]
        @test find_complete_indices([1.0 2.0; 3.0 4.0]; dims = 2) == [1, 2]

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
        # A window that carries none of the fitted universe is refused the same way.
        other = PricesResult(; X = TimeArray(ts, Float64.(reshape(1:5, 5, 1)), [:zz]))
        @test_throws PortfolioOptimisers.IsEmptyError apply_preprocessing(fitted, other)
    end

    @testset "prices_to_returns filters neither axis" begin
        # Issue #473 was a defect in the interaction of the conversion's two threshold
        # keywords: the column filter counted the missing entries over the table as it was
        # before the row filter ran, and divided by the surviving row total. Map #955 and
        # ADR 0133 removed both keywords rather than the arithmetic -- deleting a row or a
        # column is a Universe Policy, and a stateless conversion cannot fit one -- so the
        # interaction the defect lived in no longer exists. `MissingDataFilter` owns both
        # axes, and `test_31` covers the split across its fit/apply seam.
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

        # And the cost of that, which the deletion used to hide: an OUTER join is symmetric,
        # so a benchmark or factor series whose history is longer than the asset slice
        # expands the observation clock to the UNION rather than onto the asset clock. The
        # padding is real data absence and is carried like any other gap.
        Xa = TimeArray(ts6,
                       [10.0 30.0; 11.0 31.0; 12.0 32.0
                        13.0 33.0; 14.0 34.0; 15.0 35.0], [:A, :C])
        tsb = collect(Date(2019, 12, 25):Day(1):Date(2020, 1, 6))
        Bb = TimeArray(tsb, collect(200.0:(200.0 + length(tsb) - 1)), ["BM"])
        wideb = prices_to_returns(Xa; B = Bb)
        @test size(wideb.X, 1) == length(tsb) - 1
        @test count(!isfinite, wideb.X) == 2 * (length(tsb) - length(ts6))
        @test all(isfinite, wideb.B)

        # A caller who means "the benchmark on my asset clock" says so, by slicing it or by
        # asking for an inner join. Both give the asset clock back, gapless.
        for got in (prices_to_returns(Xa; B = Bb[ts6]),
                    prices_to_returns(Xa; B = Bb, join_method = :inner))
            @test size(got.X) == (5, 2)
            @test all(isfinite, got.X)
        end
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
            # `percentchange` reads two consecutive prices, so the k = 2 suspension leaves
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
            @test rB[5] ≈ 24 / 21 - 1
            @test rB[5] == expm1(log(24.0) - log(21.0))
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
            @test glog.X[5, 2] == log(24.0) - log(21.0)

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
    end
end
