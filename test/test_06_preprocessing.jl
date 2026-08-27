# Two half-implemented estimators, used to reach the dispatch-error stubs of the
# preprocessing interface. Neither implements `fit_preprocessing` or `apply_preprocessing`,
# which is the mistake the stubs exist to name.
struct UnimplementedPreprocessing <: PortfolioOptimisers.AbstractPreprocessingEstimator end
struct UnimplementedPreprocessingResult <: PortfolioOptimisers.AbstractPreprocessingResult end
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
        dfy.all_missing = fill(NaN, nrow(dfy))
        dfy[!, :date] = (today() - Day(100)):Day(1):today()
        Py = TimeArray(dfy; timestamp = :date)

        rd = prices_to_returns(Px, Py; missing_col_percent = 0.1, missing_row_percent = 0.5)
        ts1 = rd.ts
        X1 = rd.X
        F1 = rd.F
        rd = prices_to_returns(Px, Py; missing_col_percent = 0.1,
                               missing_row_percent = nothing)
        ts2 = rd.ts
        X2 = rd.X
        F2 = rd.F

        df = CSV.read(joinpath(@__DIR__, "assets/prices_to_returns_X_F.csv.gz"), DataFrame)
        @test hcat(vcat(X1, X2), vcat(F1, F2)) == Matrix(df)

        rd = prices_to_returns(Px; missing_col_percent = 0.1, missing_row_percent = 0.5)
        ts3 = rd.ts
        X3 = rd.X
        F3 = rd.F
        @test isnothing(F3)

        rd = prices_to_returns(Px; missing_col_percent = 0.1, missing_row_percent = nothing)
        ts4 = rd.ts
        X4 = rd.X
        F4 = rd.F
        @test isnothing(F4)

        df = CSV.read(joinpath(@__DIR__, "assets/prices_to_returns_X.csv.gz"), DataFrame)
        @test vcat(X3, X4) == Matrix(df)

        @test dfy[2:end, :date] == ts1 == ts2 == ts3 == ts4
    end

    @testset "impute_method without Impute loaded" begin
        # `Impute` is a weak dependency (ADR 0042) and is deliberately absent from the test
        # environment, so this testset exercises the not-loaded half of the seam.
        @test isnothing(Base.get_extension(PortfolioOptimisers,
                                           :PortfolioOptimisersImputeExt))

        rng = StableRNG(123456789)
        dfx = DataFrame(rand(rng, 21, 4), :auto)
        dfx[!, :date] = (today() - Day(20)):Day(1):today()
        Px = TimeArray(dfx; timestamp = :date)

        # The default path must not need `Impute`, whether the keyword is omitted or given
        # explicitly as `nothing`.
        rd = prices_to_returns(Px)
        @test prices_to_returns(Px; impute_method = nothing).X == rd.X
        @test PortfolioOptimisers.apply_impute_method(dfx, nothing) === dfx

        # Anything else is an ArgumentError naming both the missing `using Impute` and the
        # unrelated `Imputer` estimator the name collides with.
        for bad in (Imputer(), :LOCF, "LOCF")
            err = try
                prices_to_returns(Px; impute_method = bad)
                nothing
            catch e
                e
            end
            @test isa(err, ArgumentError)
            @test occursin("using Impute", err.msg)
            @test occursin("`Impute` is not loaded", err.msg)
            @test occursin("Imputer", err.msg)
        end
        @test_throws ArgumentError PortfolioOptimisers.apply_impute_method(dfx, Imputer())
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
            @test_throws DimensionMismatch ReturnsResult(; nx = nx, X = X, nz = nz,
                                                         Z = rand(rng, 2, 3))
            @test_throws DimensionMismatch ReturnsResult(; nx = nx, X = X, nz = nz,
                                                         Z = rand(rng, 3, 3))
            # Both-or-neither, and both directions of the omission.
            @test_throws PortfolioOptimisers.IsNothingError ReturnsResult(; nx = nx, X = X,
                                                                          nz = nz)
            @test_throws PortfolioOptimisers.IsNothingError ReturnsResult(; nx = nx, X = X,
                                                                          Z = rand(rng, 3,
                                                                                   2))
            @test_throws ArgumentError ReturnsResult(; nx = nx, X = X, nz = ["f", "f"],
                                                     Z = rand(rng, 3, 2))
            # A feature matrix is never imputed, so a non-finite entry is rejected here
            # rather than reaching a metric that would map it to a plausible wrong distance.
            @test_throws PortfolioOptimisers.IsNonFiniteError ReturnsResult(; nx = nx,
                                                                            X = X, nz = nz,
                                                                            Z = [1.0 NaN;
                                                                                 2.0 3.0;
                                                                                 4.0 5.0])
            @test_throws PortfolioOptimisers.IsNonFiniteError ReturnsResult(; nx = nx,
                                                                            X = X, nz = nz,
                                                                            Z = [1.0 Inf;
                                                                                 2.0 3.0;
                                                                                 4.0 5.0])
            # A time-varying Z binds its leading axis to the observations, and needs both
            # an asset anchor and an observation anchor to bind to.
            @test_throws DimensionMismatch ReturnsResult(; nx = nx, X = X, nz = nz,
                                                         Z = rand(rng, 4, 3, 2))
            @test_throws PortfolioOptimisers.IsNothingError ReturnsResult(; nx = nx,
                                                                          nz = nz,
                                                                          Z = rand(rng, 5,
                                                                                   3, 2))
            @test_throws PortfolioOptimisers.IsNothingError ReturnsResult(; nz = nz,
                                                                          Z = rand(rng, 3,
                                                                                   2))
            # The price level binds the same two axes to X's columns and rows.
            @test_throws DimensionMismatch PricesResult(; X = Px, nz = nz,
                                                        Z = rand(rng, 2, 2))
            @test_throws DimensionMismatch PricesResult(; X = Px, nz = nz,
                                                        Z = rand(rng, 9, 3, 2))
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
            rds = ReturnsResult(; nx = nx, X = X, nz = nz, Z = Zs)
            rd3 = ReturnsResult(; nx = nx, X = X, nz = nz,
                                Z = reshape(Float64.(1:30), 5, 3, 2))

            # Two-argument arity: assets only. The static shape has no observation axis;
            # the time-varying one keeps all of its rows.
            vs = port_opt_view(rds, [1, 3])
            @test vs.nx == ["A", "C"]
            @test vs.Z == Zs[[1, 3], :]
            @test vs.nz == nz
            v3 = port_opt_view(rd3, [1, 3])
            @test size(v3.Z) == (5, 2, 2)
            @test v3.Z == rd3.Z[:, [1, 3], :]

            # Four-argument arity: observations first, assets second. A static Z ignores the
            # observation index, exactly as ivpa does.
            vs4 = port_opt_view(rds, 2:4, [1, 3])
            @test vs4.Z == Zs[[1, 3], :]
            v34 = port_opt_view(rd3, 2:4, [1, 3])
            @test size(v34.Z) == (3, 2, 2)
            @test v34.Z == rd3.Z[2:4, [1, 3], :]
            # The third positional index selects factors and must not touch Z.
            rdf = ReturnsResult(; nx = nx, X = X, nf = ["F1", "F2"], F = rand(rng, 5, 2),
                                nz = nz, Z = Zs)
            @test port_opt_view(rdf, 2:4, [1, 3], 1:1).Z == Zs[[1, 3], :]

            # Views compose: a fold of a subset is a subset of the fold.
            @test port_opt_view(port_opt_view(rd3, [1, 3]), 2:4, [1, 2]).Z ==
                  rd3.Z[2:4, [1, 3], :]
        end

        @testset "features are assets" begin
            # A square phylogeny or adjacency matrix reused as a feature source has features
            # that ARE assets, so subselecting assets must slice the feature axis too --
            # otherwise the columns keep pointing at the full universe.
            X = rand(rng, 5, 3)
            nx = ["A", "B", "C"]
            Zsq = Float64[1 2 3; 4 5 6; 7 8 9]
            rdsq = ReturnsResult(; nx = nx, X = X, nz = nx, Z = Zsq)
            vsq = port_opt_view(rdsq, [1, 3])
            @test vsq.Z == Zsq[[1, 3], [1, 3]]
            @test vsq.nz == ["A", "C"] == vsq.nx
            # And the rule survives a second view, because both name vectors are sliced by
            # the same indices and stay equal.
            @test PortfolioOptimisers.features_are_assets(vsq.nz, vsq.nx)
            @test port_opt_view(vsq, [2]).Z == Zsq[[3], [3]]

            Zsq3 = reshape(Float64.(1:45), 5, 3, 3)
            rdsq3 = ReturnsResult(; nx = nx, X = X, nz = nx, Z = Zsq3)
            vsq3 = port_opt_view(rdsq3, 2:4, [1, 3])
            @test vsq3.Z == Zsq3[2:4, [1, 3], [1, 3]]
            @test vsq3.nz == ["A", "C"]

            # Equal lengths alone are not a claim that the axes mean the same thing.
            @test !PortfolioOptimisers.features_are_assets(["f1", "f2", "f3"], nx)
            @test port_opt_view(ReturnsResult(; nx = nx, X = X, nz = ["f1", "f2", "f3"],
                                              Z = Zsq), [1, 3]).Z == Zsq[[1, 3], :]
        end

        @testset "prices_to_returns carries Z across" begin
            prs = PricesResult(; X = Px, nz = nz, Z = Zs)
            pr3 = PricesResult(; X = Px, nz = nz, Z = Z3)

            # Asset "B" is entirely missing and is dropped; it must take its features with
            # it, or the two matrices desynchronise silently.
            rrs = prices_to_returns(PricesToReturns(), prs)
            @test rrs.nx == ["A", "C"]
            @test rrs.Z == Zs[[1, 3], :]
            @test rrs.nz == nz

            # The time-varying shape drops the same asset AND the observation lost to the
            # percentage change, so rows 2:10 and assets 1 and 3 survive.
            rr3 = prices_to_returns(PricesToReturns(), pr3)
            @test size(rr3.Z) == (9, 2, 2)
            @test rr3.Z == Z3[2:10, [1, 3], :]
            @test length(rr3.ts) == size(rr3.Z, 1) == size(rr3.X, 1)

            # Under collapse_args the aggregated period takes the features of the row at its
            # representative timestamp -- last-observation semantics.
            rc = prices_to_returns(PricesToReturns(; collapse_args = (week, last)), pr3)
            @test rc.ts == [Date(2020, 1, 10)]
            @test rc.Z == Z3[[10], [1, 3], :]

            # Dropping every asset leaves Z with nothing to bind to. Reachable only when
            # something else survives the conversion -- all-missing prices with no factors
            # already die earlier, inside the TimeArray reconstruction, with or without Z.
            Pall = TimeArray(collect(ts0), fill(NaN, 10, 3), ["A", "B", "C"])
            Fok = TimeArray(collect(ts0),
                            100 .+ cumsum(rand(StableRNG(7), 10, 2); dims = 1),
                            ["F1", "F2"])
            @test_throws PortfolioOptimisers.IsEmptyError prices_to_returns(Pall, Fok;
                                                                            nz = nz, Z = Zs)
            @test_throws ArgumentError prices_to_returns(Pall)

            # A surviving timestamp absent from the price clock cannot be mapped back to a
            # row of Z, and must throw rather than pair assets with another period.
            @test_throws ArgumentError PortfolioOptimisers.feature_row_indices(Z3,
                                                                               [Date(2019,
                                                                                     1, 1)],
                                                                               collect(ts0))
            @test PortfolioOptimisers.feature_row_indices(Z3, ts0[3:5], collect(ts0)) == 3:5
            # The shapes with no observation axis cost nothing.
            @test PortfolioOptimisers.feature_row_indices(Zs, ts0[3:5], collect(ts0)) ===
                  Colon()
            @test PortfolioOptimisers.feature_row_indices(nothing, ts0[3:5],
                                                          collect(ts0)) === Colon()
        end

        @testset "PricesResult views and preprocessing" begin
            pr3 = PricesResult(; X = Px, nz = nz, Z = Z3)
            prs = PricesResult(; X = Px, nz = nz, Z = Zs)

            # Timestamp windows slice a time-varying Z positionally against the price clock.
            @test port_opt_view(pr3, 3:6, [1, 3]).Z == Z3[3:6, [1, 3], :]
            @test port_opt_view(pr3, 3:6).Z == Z3[3:6, :, :]
            @test port_opt_view(prs, 3:6, [1, 3]).Z == Zs[[1, 3], :]
            # A window that touches neither of a static Z's axes leaves it alone entirely.
            @test port_opt_view(prs, 3:6).Z === Zs
            @test port_opt_view(pr3, :, :) === pr3

            # MissingDataFilter drops assets at price level, so it owes Z the same slice.
            mdf = fit_preprocessing(MissingDataFilter(; col_thr = 0.5), pr3)
            pm = apply_preprocessing(mdf, pr3)
            @test string.(TimeSeries.colnames(pm.X)) == ["A", "C"]
            @test pm.Z == Z3[:, [1, 3], :]
            # Imputer changes no shape, so Z passes through untouched.
            pim = apply_preprocessing(fit_preprocessing(Imputer(), pr3), pr3)
            @test pim.Z === Z3
            @test pim.nz === nz

            # train_test_split is a pair of port_opt_views, so it splits Z's observations.
            rr3 = prices_to_returns(PricesToReturns(), pr3)
            tr, te = train_test_split(rr3; train_size = 5)
            @test size(tr.Z, 1) == size(tr.X, 1)
            @test size(te.Z, 1) == size(te.X, 1)
            @test tr.Z == rr3.Z[1:size(tr.Z, 1), :, :]
        end

        @testset "other ReturnsResult constructors thread Z" begin
            # returns_result_picker folds B into X; it must not lose Z on the way.
            rdb = ReturnsResult(; nx = ["A", "B"], X = rand(rng, 4, 2), nb = ["bm"],
                                B = rand(rng, 4), nz = nz, Z = rand(rng, 2, 2))
            rp = returns_result_picker(rdb, true)
            @test rp.Z === rdb.Z
            @test rp.nz === rdb.nz
            @test isnothing(returns_result_picker(rdb, false).B) == false

            # AssetSelectorResult subselects through port_opt_view, so it is carried for free.
            rds = ReturnsResult(; nx = ["A", "B", "C"], X = rand(rng, 5, 3), nz = nz,
                                Z = Zs)
            sel = apply_preprocessing(PortfolioOptimisers.AssetSelectorResult(["A", "C"]),
                                      rds)
            @test sel.Z == Zs[[1, 3], :]
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

        # A `NaN` price becomes `missing`, and its row is dropped before the conversion. The
        # surviving returns therefore span the gap: the 01-03 return is measured against
        # 01-01, not against the row that went.
        Pm = TimeArray(ts, [100.0 50.0; NaN 45.0; 121.0 54.0], [:a, :b])
        rm = prices_to_returns(Pm)
        @test rm.ts == [ts[3]]
        @test rm.X ≈ [121/100-1 54/50-1]

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

        # Every asset column can go while a factor column survives. The result then carries
        # no asset data at all, rather than an empty matrix.
        ts5 = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 5))
        Pa = TimeArray(ts5, reshape([100.0, NaN, NaN, NaN, 104.0], 5, 1), [:a])
        Pf = TimeArray(ts5, reshape(Float64.(101:105), 5, 1), [:f1])
        rdrop = prices_to_returns(Pa, Pf; missing_row_percent = 0.3)
        @test isnothing(rdrop.nx)
        @test isnothing(rdrop.X)
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
        @test_throws ArgumentError PortfolioOptimisers.feature_row_indices(Z3, nothing, ts)
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

    @testset "Imputer fits on the training window and never refits" begin
        ts = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 6))
        # `a` is never observed, `b` is observed everywhere, and `c` is observed in the
        # training window alone.
        vals = Union{Float64, Missing}[missing 1.0 10.0
                                       missing 3.0 12.0
                                       missing 5.0 14.0
                                       missing 7.0 missing
                                       missing 9.0 missing
                                       missing 11.0 missing]
        pr = PricesResult(; X = TimeArray(ts, vals, [:a, :b, :c]))
        tr = PortfolioOptimisers.port_opt_view(pr, ts[1:3], :)
        te = PortfolioOptimisers.port_opt_view(pr, ts[4:6], :)

        res = fit_preprocessing(Imputer(), tr)
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
                                            Union{Float64, Missing}[7.0, missing, 11.0],
                                            [:b]))
        @test values(apply_preprocessing(res, narrow).X) == [7.0, 3.0, 11.0]
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

    @testset "the column filter of prices_to_returns reads the surviving rows" begin
        # Issue #473. The column filter used to count the missing entries over the table as
        # it was before the row filter ran, and to divide that count by the surviving row
        # total. A column was then dropped for missing entries that sat only in rows that
        # were already gone.
        ts4 = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 4))
        # Row 1 alone is incomplete. `A` and `B` are missing there, and `C` is complete.
        X = TimeArray(ts4, [NaN NaN 1.0; 2.0 2.0 2.0; 3.0 3.0 3.0; 4.0 4.0 4.0],
                      [:A, :B, :C])

        # `missing_col_percent = 1.0` admits row 1, so every column keeps its one missing
        # entry and the threshold of `0.3 * 4` rows admits it.
        @test prices_to_returns(X; missing_col_percent = 1.0, missing_row_percent = 0.3).nx ==
              ["A", "B", "C"]
        # `missing_col_percent = 0.5` drops row 1. `A` and `B` then hold no missing entry at
        # all, so tightening the row filter must not cost a column.
        @test prices_to_returns(X; missing_col_percent = 0.5, missing_row_percent = 0.3).nx ==
              ["A", "B", "C"]

        # The column filter still drops a column whose missing entries survive the row
        # filter. `A` is missing in rows 1, 2 and 3, and only row 1 goes.
        ts5 = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 5))
        Y = TimeArray(ts5, [NaN NaN 1.0; NaN 2.0 2.0; NaN 3.0 3.0; 4.0 4.0 4.0
                            5.0 5.0 5.0], [:A, :B, :C])
        # Four rows survive, so the threshold is `0.2 * 4 = 0.8`. `A` holds two missing
        # entries over those rows and goes; `B`'s one missing entry went with row 1.
        @test prices_to_returns(Y; missing_col_percent = 0.5, missing_row_percent = 0.2).nx ==
              ["B", "C"]
        # The `nothing` branch reads the same counts. Over the surviving rows they are
        # `[2, 0, 0]`, so the mode is `0` and the same two columns survive.
        @test prices_to_returns(Y; missing_col_percent = 0.5,
                                missing_row_percent = nothing).nx == ["B", "C"]
    end
end
