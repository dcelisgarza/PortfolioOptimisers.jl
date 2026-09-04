#=
Check `src/08_Moments/42_FactorExposures/08_Base_Exposure.jl`, `09_CompositeExposure.jl`,
`10_DerivedExposure.jl`, `11_OneHotExposure.jl` and `12_ConstantExposure.jl` against the
contract their docstrings state, and against the reference implementation the map of issue
#643 ports. Issue #721.

FOUR CONVENTIONS SHAPE THE PROBES.

1. AN INACTIVE CELL IS `NaN`, on every member. The Descriptors carry the convention into
   the composite and the derived member; the one-hot member and the constant member write
   it themselves. The hand panels below carry a value on an inactive cell on purpose.

2. THE COMBINATION IS FINITE-AWARE. A Descriptor that is `NaN` on a cell contributes
   neither its score nor its weight there, so the surviving weight is the coverage of the
   cell. `min_coverage` is a threshold on that weight, not on a count of Descriptors.

3. A BENCHMARK WEIGHT IS A SELECTOR. `cross_sectional_transform` refuses a `NaN` weight,
   so the read zeroes every unobserved and inactive cell, which puts it outside the
   estimation set of its observation rather than inside it with an unknown weight.

4. THE STORED CASES ARE THE REFERENCE IMPLEMENTATION'S OWN OUTPUT.
   `assets/CompositeExposure.csv.gz` and `assets/DerivedExposure.csv.gz` were produced by
   the reference implementation's `FixedWeightedFactor` and `DerivedFactor`, driven on the
   synthetic panel the last testset rebuilds, with the same two raw Descriptors, the same
   weights, the same coverage threshold, the same two transforms and the same grouping.
   The reference implementation's other two members were diffed the same way and agree to
   the last bit, so they are checked here against their closed forms rather than stored.

ONE DELIBERATE DEVIATION. The reference implementation's constant exposure returns one on
every cell, including a cell outside the active universe. The library writes `NaN` there,
because the root's contract makes every Factor Exposure `NaN` on an inactive cell. The
regression drops those cells on both sides, so no fitted quantity moves.
=#
include(joinpath(@__DIR__, "test06c_setup.jl"))

# A small hand panel carrying the benchmark weights beside the fields. Every numeric field
# takes a forward fill, so each earns an observed-mask column and a raw `NaN` reads back as
# `NaN` rather than as the fill value.
function exposure_hand_panel(fields::AbstractVector{<:Pair{String, <:AbstractMatrix}};
                             amsk::AbstractMatrix{Bool} = trues(size(fields[1][2])...),
                             emsk::AbstractMatrix{Bool} = amsk,
                             bw::Union{Nothing, <:AbstractMatrix} = nothing,
                             sectors::Union{Nothing, <:AbstractMatrix} = nothing,
                             sector_alg::PortfolioOptimisers.AbstractPanelFillAlgorithm = NoPanelFill())
    inputs = Any[NumericPanelInput(; name = n, vals = v,
                                   alg = ForwardPanelFill(; val = 0.0))
                 for (n, v) in fields]
    if !isnothing(bw)
        push!(inputs,
              NumericPanelInput(; name = "benchmark_weights", vals = bw,
                                alg = ForwardPanelFill(; val = 0.0)))
    end
    if !isnothing(sectors)
        push!(inputs,
              CategoricalPanelInput(; name = "sector", vals = sectors, alg = sector_alg))
    end
    res = asset_panel(identity.(inputs); amsk = amsk, emsk = emsk)
    T, N = size(amsk)
    return ReturnsResult(; nx = ["A" * string(i) for i in 1:N], X = zeros(T, N), res...)
end

@testset "Exposure constructors and their refusals" begin
    @testset "Every member carries the reference implementation's own defaults" begin
        xc = CompositeExposure(; descriptors = [Passthrough(; field = "a")])
        @test isa(xc.outlier, CrossSectionalWinsoriser)
        @test isa(xc.scoring, CrossSectionalStandardiser)
        @test isnothing(xc.weights)
        @test isnothing(xc.group)
        @test iszero(xc.min_coverage)
        @test xc.bw == "benchmark_weights"
        @test xc.family == "style"
        xd = DerivedExposure(; source = "size", f = abs2)
        @test isnothing(xd.outlier)
        @test isa(xd.scoring, CrossSectionalStandardiser)
        @test xd.bw == "benchmark_weights"
        @test xd.family == "style"
        @test ConstantExposure().family == "market"
        @test OneHotExposure(; field = "sector", family = "sector").family == "sector"
    end
    @testset "Every member is an Exposure Estimator" begin
        for xe in (CompositeExposure(; descriptors = [Passthrough(; field = "a")]),
                   DerivedExposure(; source = "size", f = abs2),
                   OneHotExposure(; field = "sector", family = "sector"), ConstantExposure())
            @test isa(xe, PortfolioOptimisers.AbstractExposureEstimator)
            @test isa(xe, PortfolioOptimisers.AbstractEstimator)
        end
    end
    @testset "The constructors refuse what the docstrings say they refuse" begin
        @test_throws PortfolioOptimisers.IsEmptyError CompositeExposure(;
                                                                        descriptors = PortfolioOptimisers.AbstractDescriptorEstimator[])
        @test_throws DimensionMismatch CompositeExposure(;
                                                         descriptors = [Passthrough(;
                                                                                    field = "a")],
                                                         weights = [0.5, 0.5])
        @test_throws DomainError CompositeExposure(;
                                                   descriptors = [Passthrough(;
                                                                              field = "a"),
                                                                  Passthrough(;
                                                                              field = "b")],
                                                   weights = [-0.5, 1.5])
        @test_throws DomainError CompositeExposure(;
                                                   descriptors = [Passthrough(;
                                                                              field = "a"),
                                                                  Passthrough(;
                                                                              field = "b")],
                                                   weights = [0.5, 0.4])
        @test_throws DomainError CompositeExposure(;
                                                   descriptors = [Passthrough(;
                                                                              field = "a")],
                                                   min_coverage = 1.5)
        @test_throws DomainError CompositeExposure(;
                                                   descriptors = [Passthrough(;
                                                                              field = "a")],
                                                   min_coverage = NaN)
        @test_throws PortfolioOptimisers.IsEmptyError CompositeExposure(;
                                                                        descriptors = [Passthrough(;
                                                                                                   field = "a")],
                                                                        family = "")
        @test_throws PortfolioOptimisers.IsEmptyError CompositeExposure(;
                                                                        descriptors = [Passthrough(;
                                                                                                   field = "a")],
                                                                        bw = "")
        @test_throws PortfolioOptimisers.IsEmptyError CompositeExposure(;
                                                                        descriptors = [Passthrough(;
                                                                                                   field = "a")],
                                                                        group = "")
        @test_throws PortfolioOptimisers.IsEmptyError DerivedExposure(; source = "",
                                                                      f = abs2)
        @test_throws PortfolioOptimisers.IsEmptyError DerivedExposure(; source = "s",
                                                                      f = abs2, family = "")
        @test_throws PortfolioOptimisers.IsEmptyError OneHotExposure(; field = "",
                                                                     family = "sector")
        @test_throws PortfolioOptimisers.IsEmptyError OneHotExposure(; field = "sector",
                                                                     family = "")
        @test_throws PortfolioOptimisers.IsEmptyError ConstantExposure(; family = "")
    end
    @testset "The equal-weight composite is the mean of its Descriptors" begin
        @test PortfolioOptimisers.composite_weights(nothing, 4) == fill(0.25, 4)
        @test PortfolioOptimisers.composite_weights([0.4, 0.6], 2) == [0.4, 0.6]
    end
end

@testset "The shared reads of the exposure base" begin
    rd = exposure_hand_panel(["a" => [1.0 2.0; 3.0 4.0]]; amsk = [true true; true false],
                             bw = [1.0 2.0; NaN 4.0])
    @testset "A benchmark weight is zero where it is unobserved or inactive" begin
        W = PortfolioOptimisers.exposure_benchmark_weights(rd, "benchmark_weights")
        @test W == [1.0 2.0; 0.0 0.0]
        @test all(isfinite, W)
    end
    @testset "A member that names no grouping field partitions nothing" begin
        @test isnothing(PortfolioOptimisers.exposure_group_labels(rd, nothing))
    end
    @testset "A nothing transform slot is the identity" begin
        X = [1.0 2.0; 3.0 4.0]
        @test PortfolioOptimisers.exposure_transform(nothing, X, nothing, nothing) === X
    end
    @testset "The active fill covers both ranks, and checks the shape" begin
        L2 = [1.0 2.0; 3.0 4.0]
        PortfolioOptimisers.exposure_active_fill!(L2, rd.pnl)
        @test isequal(L2, [1.0 2.0; 3.0 NaN])
        L3 = ones(2, 2, 3)
        PortfolioOptimisers.exposure_active_fill!(L3, rd.pnl)
        @test all(isnan, L3[2, 2, :])
        @test all(isone, L3[1, :, :])
        @test_throws DimensionMismatch PortfolioOptimisers.exposure_active_fill!(ones(3, 3),
                                                                                 rd.pnl)
        @test_throws DimensionMismatch PortfolioOptimisers.exposure_active_fill!(ones(3, 3,
                                                                                      2),
                                                                                 rd.pnl)
    end
end

@testset "The composite on a hand panel" begin
    a = [1.0 2.0; 3.0 4.0]
    b = [5.0 6.0; 7.0 8.0]
    rd = exposure_hand_panel(["a" => a, "b" => b]; bw = ones(2, 2))
    des = [Passthrough(; field = "a"), Passthrough(; field = "b")]
    @testset "Fixed weights combine the Descriptors when both slots are nothing" begin
        xe = CompositeExposure(; descriptors = des, weights = [0.25, 0.75],
                               outlier = nothing, scoring = nothing)
        @test factor_exposure(xe, rd) ≈ 0.25 .* a .+ 0.75 .* b
    end
    @testset "The equal-weight default is the plain mean" begin
        xe = CompositeExposure(; descriptors = des, outlier = nothing, scoring = nothing)
        @test factor_exposure(xe, rd) ≈ 0.5 .* a .+ 0.5 .* b
    end
    @testset "A single Descriptor takes no second scoring pass" begin
        xe = CompositeExposure(; descriptors = [Passthrough(; field = "a")],
                               outlier = nothing, scoring = nothing)
        @test factor_exposure(xe, rd) ≈ a
    end
    @testset "The coverage threshold is on weight, not on a count" begin
        # The second Descriptor is blank on the first cell, so that cell carries a
        # surviving weight of 0.4, which is the weight of the Descriptor that remains.
        bn = copy(b)
        bn[1, 1] = NaN
        rdn = exposure_hand_panel(["a" => a, "b" => bn]; bw = ones(2, 2))
        xhi = CompositeExposure(; descriptors = des, weights = [0.4, 0.6],
                                min_coverage = 0.5, outlier = nothing, scoring = nothing)
        L = factor_exposure(xhi, rdn)
        @test isnan(L[1, 1])
        @test L[2, 1] ≈ 0.4 * a[2, 1] + 0.6 * bn[2, 1]
        xlo = CompositeExposure(; descriptors = des, weights = [0.4, 0.6],
                                min_coverage = 0.0, outlier = nothing, scoring = nothing)
        @test factor_exposure(xlo, rdn)[1, 1] ≈ a[1, 1]
        # A threshold at exactly the surviving weight admits the cell.
        xeq = CompositeExposure(; descriptors = des, weights = [0.4, 0.6],
                                min_coverage = 0.4, outlier = nothing, scoring = nothing)
        @test factor_exposure(xeq, rdn)[1, 1] ≈ a[1, 1]
    end
    @testset "A cell no Descriptor reaches is NaN, whatever the threshold" begin
        an = copy(a)
        bn = copy(b)
        an[1, 2] = NaN
        bn[1, 2] = NaN
        rdn = exposure_hand_panel(["a" => an, "b" => bn]; bw = ones(2, 2))
        xe = CompositeExposure(; descriptors = des, min_coverage = 0.0, outlier = nothing,
                               scoring = nothing)
        @test isnan(factor_exposure(xe, rdn)[1, 2])
    end
    @testset "An inactive cell is NaN, whatever the Panel Fields hold" begin
        rdi = exposure_hand_panel(["a" => a, "b" => b]; amsk = [true true; true false],
                                  bw = ones(2, 2))
        xe = CompositeExposure(; descriptors = des, outlier = nothing, scoring = nothing)
        L = factor_exposure(xe, rdi)
        @test isnan(L[2, 2])
        @test L[1, 1] ≈ 0.5 * (a[1, 1] + b[1, 1])
    end
    @testset "Both slots run on every Descriptor, and the composite is scored again" begin
        ct = CrossSectionalStandardiser(; min_group_size = 1)
        xe = CompositeExposure(; descriptors = des, outlier = nothing, scoring = ct)
        Sa = cross_sectional_transform(ct, a; w = ones(2, 2))
        Sb = cross_sectional_transform(ct, b; w = ones(2, 2))
        expected = cross_sectional_transform(ct, 0.5 .* Sa .+ 0.5 .* Sb; w = ones(2, 2))
        @test factor_exposure(xe, rd) ≈ expected
    end
    @testset "The grouping field reaches the transforms" begin
        rdg = exposure_hand_panel(["a" => [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0]];
                                  bw = ones(2, 4),
                                  sectors = ["t" "t" "b" "b"; "t" "t" "b" "b"])
        ct = CrossSectionalStandardiser(; min_group_size = 2)
        xe = CompositeExposure(; descriptors = [Passthrough(; field = "a")],
                               outlier = nothing, scoring = ct, group = "sector")
        G = cross_sectional_groups(rdg.pnl, rdg.Z, "sector")
        expected = cross_sectional_transform(ct,
                                             PortfolioOptimisers.panel_field_values(rdg,
                                                                                    "a");
                                             w = ones(2, 4), groups = G)
        @test factor_exposure(xe, rdg) ≈ expected
    end
    @testset "The estimator holds no data: two carriers, one estimator" begin
        xe = CompositeExposure(; descriptors = des, outlier = nothing, scoring = nothing)
        rd2 = exposure_hand_panel(["a" => 2 .* a, "b" => 2 .* b]; bw = ones(2, 2))
        @test factor_exposure(xe, rd) ≈ 0.5 .* (a .+ b)
        @test factor_exposure(xe, rd2) ≈ a .+ b
    end
end

@testset "The derived exposure on a hand panel" begin
    rd = exposure_hand_panel(["a" => [1.0 2.0; 3.0 4.0]]; bw = ones(2, 2))
    xs = [1.0 2.0; 3.0 4.0]
    @testset "The function runs, then the two slots" begin
        xe = DerivedExposure(; source = "size", f = x -> x .^ 2, scoring = nothing)
        @test factor_exposure(xe, rd, xs) ≈ xs .^ 2
        ct = CrossSectionalStandardiser(; min_group_size = 1)
        xs2 = DerivedExposure(; source = "size", f = x -> x .^ 2, scoring = ct)
        @test factor_exposure(xs2, rd, xs) ≈
              cross_sectional_transform(ct, xs .^ 2; w = ones(2, 2))
    end
    @testset "The two-argument method refuses, naming the source" begin
        xe = DerivedExposure(; source = "size", f = abs2)
        @test_throws ArgumentError factor_exposure(xe, rd)
    end
    @testset "A source of the wrong shape, and a function that changes it, are refused" begin
        xe = DerivedExposure(; source = "size", f = x -> x .^ 2, scoring = nothing)
        @test_throws DimensionMismatch factor_exposure(xe, rd, ones(3, 3))
        xf = DerivedExposure(; source = "size", f = x -> x[:, 1:1], scoring = nothing)
        @test_throws DimensionMismatch factor_exposure(xf, rd, xs)
    end
end

@testset "The one-hot exposure on a hand panel" begin
    rd = exposure_hand_panel(["a" => [1.0 2.0; 3.0 4.0]]; amsk = [true true; true false],
                             sectors = ["tech" "banks"; "tech" "tech"])
    xe = OneHotExposure(; field = "sector", family = "sector")
    @testset "One factor per level, named by the expansion form" begin
        @test PortfolioOptimisers.one_hot_exposure_names(xe, rd) ==
              ["sector=banks", "sector=tech"]
        L = factor_exposure(xe, rd)
        @test size(L) == (2, 2, 2)
        @test L[1, 1, :] == [0.0, 1.0]
        @test L[1, 2, :] == [1.0, 0.0]
        @test L[2, 1, :] == [0.0, 1.0]
    end
    @testset "An inactive cell is NaN across every level" begin
        @test all(isnan, factor_exposure(xe, rd)[2, 2, :])
    end
    @testset "An asset that sets no level is NaN across every level" begin
        rdb = exposure_hand_panel(["a" => [1.0 2.0; 3.0 4.0]];
                                  sectors = ["tech" "banks"; "tech" "tech"])
        Z = Array{Float64, 3}(rdb.Z)
        Z[1, 1, :] .= 0.0
        rdz = ReturnsResult(; nx = rdb.nx, X = rdb.X, nz = rdb.nz, Z = Z, pnl = rdb.pnl)
        @test all(isnan, factor_exposure(xe, rdz)[1, 1, :])
    end
    @testset "A level the fill wrote is NaN, not a classification the asset carried" begin
        # The field can blank, so it earns an observed-mask column. The forward fill
        # resolves the blank to a level, and the read must undo that resolution.
        rdf = exposure_hand_panel(["a" => [1.0 2.0; 3.0 4.0]];
                                  sectors = ["tech" missing; "tech" "banks"],
                                  sector_alg = ForwardPanelFill(; val = "tech"))
        L = factor_exposure(xe, rdf)
        @test all(isnan, L[1, 2, :])
        @test L[1, 1, :] == [0.0, 1.0]
        @test L[2, 2, :] == [1.0, 0.0]
    end
    @testset "A numeric Panel Field, and a carrier with no Asset Panel, are refused" begin
        @test_throws ArgumentError factor_exposure(OneHotExposure(; field = "a",
                                                                  family = "f"), rd)
        rdn = ReturnsResult(; nx = ["A1", "A2"], X = zeros(2, 2))
        @test_throws PortfolioOptimisers.IsNothingError factor_exposure(xe, rdn)
    end
end

@testset "The constant exposure on a hand panel" begin
    rd = exposure_hand_panel(["a" => [1.0 2.0; 3.0 4.0]]; amsk = [true true; true false])
    @testset "The ones column, NaN where the asset is not listed" begin
        L = factor_exposure(ConstantExposure(), rd)
        @test isequal(L, [1.0 1.0; 1.0 NaN])
    end
    @testset "A carrier with no Asset Panel is refused" begin
        rdn = ReturnsResult(; nx = ["A1", "A2"], X = zeros(2, 2))
        @test_throws PortfolioOptimisers.IsNothingError factor_exposure(ConstantExposure(),
                                                                        rdn)
    end
end

@testset "The members reproduce the reference implementation" begin
    sp = synthetic_asset_panel(; n_assets = 20, n_observations = 60, n_industries = 4,
                               late_listing_proba = 0.3, delisting_proba = 0.3,
                               missing_ratio = 0.08, rng = StableRNG(987654321))
    rd = sp.rd
    ct_out = CrossSectionalWinsoriser()
    ct_sco = CrossSectionalStandardiser(; min_group_size = 2)
    xc = CompositeExposure(;
                           descriptors = [Passthrough(; field = "book_equity"),
                                          Passthrough(; field = "market_cap")],
                           weights = [0.4, 0.6], min_coverage = 0.5, outlier = ct_out,
                           scoring = ct_sco, group = "industry", bw = "market_cap")
    Lc = factor_exposure(xc, rd)
    xd = DerivedExposure(; source = "style", f = x -> x .^ 3, scoring = ct_sco,
                         group = "industry", bw = "market_cap")
    Ld = factor_exposure(xd, rd, Lc)
    @testset "The composite matches the stored case cell by cell" begin
        E = Matrix(CSV.read(joinpath(@__DIR__, "assets/CompositeExposure.csv.gz"),
                            DataFrame))
        @test size(Lc) == size(E)
        @test isequal(isnan.(Lc), isnan.(E))
        @test Lc[isfinite.(E)] ≈ E[isfinite.(E)]
    end
    @testset "The derived exposure matches the stored case cell by cell" begin
        E = Matrix(CSV.read(joinpath(@__DIR__, "assets/DerivedExposure.csv.gz"), DataFrame))
        @test size(Ld) == size(E)
        @test isequal(isnan.(Ld), isnan.(E))
        @test Ld[isfinite.(E)] ≈ E[isfinite.(E)]
    end
    @testset "The one-hot block is the classification, and the constant is the ones" begin
        Lo = factor_exposure(OneHotExposure(; field = "industry", family = "industry"), rd)
        G = cross_sectional_groups(rd.pnl, rd.Z, "industry")
        amsk = rd.pnl.amsk
        @test size(Lo) == (size(amsk)..., 4)
        for k in CartesianIndices(amsk)
            if amsk[k]
                @test Lo[k, :] == [l == G[k] ? 1.0 : 0.0 for l in 1:4]
            else
                @test all(isnan, Lo[k, :])
            end
        end
        Lk = factor_exposure(ConstantExposure(), rd)
        @test isequal(isnan.(Lk), .!amsk)
        @test all(isone, Lk[amsk])
    end
end
