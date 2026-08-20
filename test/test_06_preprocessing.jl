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
end
