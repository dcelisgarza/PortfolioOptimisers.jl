#=
The parity harness of map #1375 (#1376): the fixtures, the exchange by CSV, the comparison and the
storage that every parity ticket of the map reads from `parity_harness.jl`.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))

@testset "The parity harness (#1376)" begin
    PO = PortfolioOptimisers
    fx = parity_small_panel()
    rd = fx.rd
    at = fx.at

    @testset "The small panel holds every condition of the fixture" begin
        @test size(rd.X) == (80, 12)
        # A late listing, a delisting, and a delisting that lists again.
        @test !fx.amsk[at.late[2] - 1, at.late[1]] &&
              all(fx.amsk[at.late[2]:end, at.late[1]])
        @test all(fx.amsk[1:at.delist[2], at.delist[1]]) &&
              !any(fx.amsk[(at.delist[2] + 1):end, at.delist[1]])
        i, a, b = at.relist
        @test fx.amsk[a - 1, i] && !any(fx.amsk[a:(b - 1), i]) && all(fx.amsk[b:end, i])
        # A holiday gap inside an active stretch.
        @test fx.amsk[at.holiday...] && isnan(rd.X[at.holiday...])
        @test all(isfinite,
                  rd.X[fx.amsk .& .!(CartesianIndices(rd.X) .== CartesianIndex(at.holiday))])
        @test all(isnan, rd.X[.!fx.amsk])
        # The estimation mask is a strict subset of the active mask.
        @test all(fx.emsk .<= fx.amsk) && fx.emsk != fx.amsk
        @test all(fx.amsk[at.outside_estimation...]) &&
              !any(fx.emsk[at.outside_estimation...])
        # A zero and a negative denominator, a negative volume and a negative short interest.
        vals(name) = PO.panel_field(rd.pnl, name).vals
        @test iszero(vals("book_equity")[at.zero_denominator...])
        @test all(<(0), vals("book_equity")[at.negative_denominator...])
        @test vals("adj_volume")[at.negative_volume...] < 0
        @test vals("short_interest")[at.negative_short_interest...] < 0
        # Ties in one cross-section.
        t, js = at.ties
        @test allequal(vals("style1")[t, js])
        # The level "Utilities" is empty on the active universe after the delisting.
        f = PO.panel_field(rd.pnl, "industry")
        k = findfirst(==(at.empty_level[1]), f.levels)
        @test findall(==(k), f.codes[1, :]) == [at.empty_level[2]]
        @test !any(((f.codes .== k) .& fx.amsk)[(at.delist[2] + 1):end, :])
        # The Exogenous Series: three currencies and one macro series.
        @test rd.ne == ["EUR", "JPY", "USD", "MACRO"]
        @test rd.X[2, 1] ≈ fx.loc[2, 1] + fx.R[2, fx.code[1]] rtol = 1e-15
        lx = parity_large_panel()
        @test size(lx.rd.X) == (250, 40)
        @test lx.emsk != lx.amsk && lx.rd.ne == rd.ne
    end

    @testset "The exchange writes every file the oracle side reads" begin
        dir = parity_write(mktempdir(), rd)
        X = parity_read(joinpath(dir, "returns.csv"))
        @test isequal(X, rd.X)
        @test parity_read(joinpath(dir, "active_mask.csv")) == fx.amsk
        @test parity_read(joinpath(dir, "estimation_mask.csv")) == fx.emsk
        @test readlines(joinpath(dir, "assets.csv")) == rd.nx
        kinds = split.(readlines(joinpath(dir, "fields.csv")), ',')
        @test length(kinds) == length(rd.pnl.pf)
        @test count(k -> k[2] == "categorical", kinds) == 2
        # A numeric cell outside the active mask, or not observed, is NaN.
        V = parity_read(joinpath(dir, "field_adj_close.csv"))
        @test all(isnan, V[.!fx.amsk]) && isnan(V[at.holiday...])
        @test V[fx.amsk .& .!isnan.(V)] ==
              PO.panel_field(rd.pnl, "adj_close").vals[fx.amsk .& .!isnan.(V)]
        # Under `filled = true` the holiday holds the value of the fill policy, the last close.
        Vf = parity_read(joinpath(parity_write(mktempdir(), rd; filled = true),
                                  "field_adj_close.csv"))
        @test Vf[at.holiday...] == V[at.holiday[1] - 1, at.holiday[2]]
        @test all(isnan, Vf[.!fx.amsk])
        @test isequal(parity_read(joinpath(dir, "field_book_equity.csv")),
                      ifelse.(fx.amsk, PO.panel_field(rd.pnl, "book_equity").vals, NaN))
        # A categorical cell is a zero-based code, and -1 outside the active mask.
        C = parity_read(joinpath(dir, "field_industry.csv"))
        f = PO.panel_field(rd.pnl, "industry")
        @test C[fx.amsk] == f.codes[fx.amsk] .- 1
        @test all(==(-1), C[.!fx.amsk])
        @test readlines(joinpath(dir, "levels_industry.csv")) == f.levels
        E = CSV.read(joinpath(dir, "exogenous.csv"), DataFrame)
        @test names(E) == rd.ne && Matrix(E) == rd.E
    end

    @testset "The comparison checks the pattern and the relative difference" begin
        a = [1.0 NaN; -2.0 0.0]
        @test parity_compare(a, copy(a); name = "equal") ==
              (; ok = true, pattern = true, maxrel = 0.0, maxscaled = 0.0, maxabs = 0.0)
        r = parity_compare(a, a .* (1 + 1e-13); name = "1e-13")
        @test r.ok
        @test r.maxrel ≈ 1e-13 rtol = 1e-3
        r = parity_compare(a, a .* (1 + 1e-11); quiet = true)
        @test !r.ok && r.pattern
        @test parity_compare(a, a .* (1 + 1e-11); rtol = 1e-10).ok
        # A zero against a tiny value is a relative difference of one, which only `atol` passes.
        b = [1.0 NaN; -2.0 1e-18]
        @test !parity_compare(a, b; quiet = true).ok
        @test parity_compare(a, b; atol = 1e-16).ok
        # Under `scale = :array` a cell is measured against the largest entry, 2 here.
        r = parity_compare(a, b; scale = :array)
        @test r.ok && r.maxrel == 1 && r.maxscaled == 5e-19
        @test !parity_compare(a, [1.0 NaN; -2.0 3e-12]; scale = :array, quiet = true).ok
        # The non-finite cells must agree exactly.
        r = parity_compare(a, [1.0 2.0; -2.0 0.0]; quiet = true)
        @test !r.ok && !r.pattern
        @test !parity_compare([Inf], [-Inf]; quiet = true).ok &&
              parity_compare([Inf], [Inf]).ok
        @test parity_compare(a, a[:, 1]; quiet = true) ==
              (; ok = false, pattern = false, maxrel = Inf, maxscaled = Inf, maxabs = Inf)
        # A check that fails prints one line under its name, and `quiet = true` prints none,
        # so a negative check that passes adds no output.
        out = mktemp() do path, io
            redirect_stdout(io) do
                parity_compare([1.0], [2.0]; name = "loud")
                parity_compare([1.0], [2.0]; name = "hush", quiet = true)
                parity_compare([1.0], [1.0]; name = "pass")
                parity_compare(a, a[:, 1]; name = "shape", quiet = true)
                return nothing
            end
            close(io)
            return read(path, String)
        end
        @test startswith(out, "parity loud: maxrel = 0.5") && count('\n', out) == 1
    end

    @testset "The storage round-trips a stored output exactly" begin
        dir = mktempdir()
        p = parity_asset("CrossSectionalFactorPrior", "Small", "Sigma"; dir = dir)
        @test basename(p) == "Parity_CrossSectionalFactorPrior_Small_Sigma.csv.gz"
        @test dirname(parity_asset("U", "C", "O")) == joinpath(@__DIR__, "assets")
        A = randn(StableRNG(1376), 5, 3)
        A[2, 2] = NaN
        parity_store(p, A)
        @test isequal(parity_load("CrossSectionalFactorPrior", "Small", "Sigma"; dir = dir),
                      A)
        v = [0.1, 1e-300, -3.0]
        parity_store(parity_asset("U", "C", "Mu"; dir = dir), v)
        @test parity_load("U", "C", "Mu"; dir = dir) == reshape(v, :, 1)
        # A plain output of the oracle side carries no header.
        q = joinpath(dir, "out.csv")
        write(q, "1.5,NaN\n-2.0e-7,3.0\n")
        @test isequal(parity_read(q), [1.5 NaN; -2.0e-7 3.0])
    end
end
