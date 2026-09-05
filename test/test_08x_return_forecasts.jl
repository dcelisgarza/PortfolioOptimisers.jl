#=
Check `src/08_Moments/45_ReturnForecasts/01_Base_ReturnForecast.jl`, `02_DescriptorScores.jl`,
`03_CustomValueReturnForecast.jl` and `04_FixedWeightedReturnForecast.jl` against the contract
their docstrings state, and against the reference implementation the map of issue #643 ports.
Issue #737.

FOUR CONVENTIONS SHAPE THE PROBES.

1. THE CROSS-SECTIONAL WEIGHT IS THE ESTIMATION MASK, not a benchmark weight. A Descriptor
   score is standardised over the estimation universe of its observation, so the family
   carries no `bw` field and a static Asset Panel is refused outright.

2. THE COMBINATION IS FINITE-AWARE, AND THE WEIGHTS ARE SIGNED. A Descriptor that is `NaN`
   on a cell contributes neither its score nor its absolute weight there. `min_coverage` is
   a threshold on the surviving **absolute** weight, and the weights are normalised by their
   absolute sum, so that threshold is a share of one.

3. `mu` IS ALWAYS IN RETURN UNITS. The Forecast Unit says what the Descriptors forecast, and
   the conversion is a method on the tag: the Sharpe unit multiplies the whole history by the
   idiosyncratic volatility of the same observation, so its last row is what `mu` reads.

4. THE STORED CASES ARE THE REFERENCE IMPLEMENTATION'S OWN OUTPUT.
   `assets/FixedWeightedReturnForecast1.csv.gz` and `assets/FixedWeightedReturnForecast2.csv.gz`
   were produced by the reference implementation's fixed-weighted alpha estimator, driven on
   the synthetic panel the last testset rebuilds, with the same two raw Descriptors, the same
   weights, the same coverage threshold, the same two transforms, the same grouping and, for
   the second case, the same Neutralisation and the same Forecast Unit. The factor-model block
   is assembled by hand from the Factor Exposures of issue #721, because the prior of issue
   #725 has not landed.
=#
include(joinpath(@__DIR__, "test06c_setup.jl"))

# A small hand panel. Every numeric field takes a forward fill, so each earns an observed-mask
# column and a raw `NaN` reads back as `NaN` rather than as the fill value.
function forecast_hand_panel(fields::AbstractVector{<:Pair{String, <:AbstractMatrix}};
                             amsk::AbstractMatrix{Bool} = trues(size(fields[1][2])...),
                             emsk::AbstractMatrix{Bool} = amsk)
    inp = [NumericPanelInput(; name = f[1], vals = Matrix{Float64}(f[2]),
                             alg = ForwardPanelFill()) for f in fields]
    pnl = asset_panel(inp; amsk = amsk, emsk = emsk)
    nx = ["A$i" for i in axes(fields[1][2], 2)]
    return ReturnsResult(; nx = nx, X = zeros(size(fields[1][2])...), pnl = pnl)
end

function forecast_hand_block(N::Integer; vs = nothing, Ms = nothing, nf = nothing,
                             fam = nothing)
    return CrossSectionalFactorModel(; M = reshape(fill(1.0, N), N, 1), b = zeros(N),
                                     vs = vs, Ms = Ms, nf = nf, fam = fam)
end

@testset "The Forecast Unit converts, and the estimation mask weighs" begin
    PO = PortfolioOptimisers
    F = [1.0 2.0; 3.0 4.0]
    vs = [0.04 0.25; 0.09 0.01]

    @testset "The return unit returns the forecast unchanged" begin
        @test PO.forecast_return_units(IdiosyncraticReturnUnit(), F, nothing) === F
        @test PO.forecast_return_units(IdiosyncraticReturnUnit(), F, vs) === F
    end

    @testset "The Sharpe unit multiplies by the idiosyncratic volatility" begin
        @test PO.forecast_return_units(IdiosyncraticSharpeUnit(), F, vs) ≈ F .* sqrt.(vs)
        @test isnan(PO.forecast_return_units(IdiosyncraticSharpeUnit(), F,
                                             [NaN 0.25; 0.09 0.01])[1, 1])
    end

    @testset "The Sharpe unit refuses an absent and a mismatched variance history" begin
        @test_throws PO.IsNothingError PO.forecast_return_units(IdiosyncraticSharpeUnit(),
                                                                F, nothing)
        @test_throws DimensionMismatch PO.forecast_return_units(IdiosyncraticSharpeUnit(),
                                                                F, [0.04 0.25])
    end

    @testset "The weights are the estimation mask, and a static panel is refused" begin
        rd = forecast_hand_panel(["a" => [1.0 2.0; 3.0 4.0]];
                                 amsk = [true true; true false],
                                 emsk = [true false; true false])
        @test PO.return_forecast_weights(rd) == [1.0 0.0; 1.0 0.0]
        spnl = asset_panel([NumericPanelInput(; name = "a", vals = [1.0, 2.0])])
        srd = ReturnsResult(; nx = ["A1", "A2"], X = zeros(2, 2), pnl = spnl)
        @test_throws PO.IsNothingError PO.return_forecast_weights(srd)
        rdn = ReturnsResult(; nx = ["A1", "A2"], X = zeros(2, 2))
        @test_throws PO.IsNothingError PO.return_forecast_weights(rdn)
    end
end

@testset "DescriptorScores states the recipe and refuses a name that is not one" begin
    PO = PortfolioOptimisers

    @testset "The defaults are the two transforms and no Neutralisation" begin
        ds = DescriptorScores(; descriptors = [Passthrough(; field = "a")])
        @test isa(ds.outlier, CrossSectionalWinsoriser)
        @test isa(ds.scoring, CrossSectionalStandardiser)
        @test isnothing(ds.neutralise)
        @test isnothing(ds.group)
    end

    @testset "The constructor refuses an empty recipe" begin
        @test_throws PO.IsEmptyError DescriptorScores(;
                                                      descriptors = PO.AbstractDescriptorEstimator[])
        @test_throws PO.IsEmptyError DescriptorScores(;
                                                      descriptors = [Passthrough(;
                                                                                 field = "a")],
                                                      neutralise = "")
        @test_throws PO.IsEmptyError DescriptorScores(;
                                                      descriptors = [Passthrough(;
                                                                                 field = "a")],
                                                      neutralise = String[])
        @test_throws PO.IsEmptyError DescriptorScores(;
                                                      descriptors = [Passthrough(;
                                                                                 field = "a")],
                                                      neutralise = ["style", ""])
        @test_throws PO.IsEmptyError DescriptorScores(;
                                                      descriptors = [Passthrough(;
                                                                                 field = "a")],
                                                      group = "")
    end

    @testset "One name and a list of names are both accepted" begin
        @test PO.assert_neutralisation_names("style") === nothing
        @test PO.assert_neutralisation_names(["style", "industry"]) === nothing
    end
end

@testset "The scores are the Descriptors, transformed and stacked" begin
    PO = PortfolioOptimisers
    a = [1.0 2.0; 3.0 4.0]
    b = [5.0 6.0; 7.0 8.0]
    rd = forecast_hand_panel(["a" => a, "b" => b])
    csfm = forecast_hand_block(2)
    ds = DescriptorScores(;
                          descriptors = [Passthrough(; field = "a"),
                                         Passthrough(; field = "b")], outlier = nothing,
                          scoring = nothing)
    S = descriptor_scores(ds, rd, csfm)

    @testset "The third axis is the Descriptor axis, in the written order" begin
        @test size(S) == (2, 2, 2)
        @test S[:, :, 1] == a
        @test S[:, :, 2] == b
    end

    @testset "A block with no factor axis refuses a Neutralisation" begin
        dsn = DescriptorScores(; descriptors = [Passthrough(; field = "a")],
                               neutralise = "style", outlier = nothing, scoring = nothing)
        @test_throws PO.IsNothingError descriptor_scores(dsn, rd, csfm)
        Ms = reshape([1.0 1.0; 1.0 1.0], 2, 2, 1)
        @test_throws PO.IsNothingError descriptor_scores(dsn, rd,
                                                         forecast_hand_block(2; Ms = Ms))
        @test_throws PO.IsNothingError descriptor_scores(dsn, rd,
                                                         forecast_hand_block(2; Ms = Ms,
                                                                             nf = ["style"]))
        @test_throws ArgumentError descriptor_scores(dsn, rd,
                                                     forecast_hand_block(2; Ms = Ms,
                                                                         nf = ["size"],
                                                                         fam = ["value"]))
    end

    @testset "A Neutralisation against a factor leaves the residual of that fit" begin
        # The one target is the score itself, and the fit carries no intercept, so every
        # residual is zero.
        Ms = reshape(Float64.(a), 2, 2, 1)
        blk = forecast_hand_block(2; Ms = Ms, nf = ["style"], fam = ["style"])
        dsn = DescriptorScores(; descriptors = [Passthrough(; field = "a")],
                               neutralise = ["style"], outlier = nothing, scoring = nothing)
        Sn = descriptor_scores(dsn, rd, blk)
        @test all(abs.(Sn) .< 1e-10)
    end

    @testset "A group name partitions each observation" begin
        pf = [NumericPanelInput(; name = "a", vals = a, alg = ForwardPanelFill()),
              CategoricalPanelInput(; name = "g", vals = ["x" "y"; "x" "y"])]
        pnl = asset_panel(pf; amsk = trues(2, 2), emsk = trues(2, 2))
        rdg = ReturnsResult(; nx = ["A1", "A2"], X = zeros(2, 2), pnl = pnl)
        dsg = DescriptorScores(; descriptors = [Passthrough(; field = "a")],
                               outlier = nothing,
                               scoring = CrossSectionalStandardiser(; min_group_size = 2),
                               group = "g")
        @test size(descriptor_scores(dsg, rdg, csfm)) == (2, 2, 1)
    end
end

@testset "The stated member carries the caller's forecast" begin
    PO = PortfolioOptimisers
    rd = forecast_hand_panel(["a" => [1.0 2.0; 3.0 4.0]])
    csfm = forecast_hand_block(2)

    @testset "The forecast is returned, and there is no history" begin
        rf = return_forecast(CustomValueReturnForecast(; mu = [0.01, NaN]), rd, csfm)
        @test isa(rf, CustomValueReturnForecastResult)
        @test isequal(rf.mu, [0.01, NaN])
        @test isnothing(rf.hist)
    end

    @testset "An empty forecast and a wrong length are refused" begin
        @test_throws PO.IsEmptyError CustomValueReturnForecast(; mu = Float64[])
        @test_throws PO.IsEmptyError CustomValueReturnForecastResult(; mu = Float64[])
        @test_throws DimensionMismatch return_forecast(CustomValueReturnForecast(;
                                                                                 mu = [0.01]),
                                                       rd, csfm)
    end
end

@testset "The fixed weighted member combines the scores under signed weights" begin
    PO = PortfolioOptimisers
    a = [1.0 2.0; 3.0 4.0]
    b = [5.0 6.0; 7.0 8.0]
    rd = forecast_hand_panel(["a" => a, "b" => b])
    csfm = forecast_hand_block(2)
    ds = DescriptorScores(;
                          descriptors = [Passthrough(; field = "a"),
                                         Passthrough(; field = "b")], outlier = nothing,
                          scoring = nothing)

    @testset "The weights normalise by their absolute sum" begin
        @test PO.signed_composite_weights(nothing, 4) == fill(0.25, 4)
        @test PO.signed_composite_weights([2.0, -2.0], 2) == [0.5, -0.5]
        @test PO.assert_signed_composite_weights(nothing, 3) === nothing
        @test_throws DimensionMismatch PO.assert_signed_composite_weights([1.0], 2)
        @test_throws PO.IsNonFiniteError PO.assert_signed_composite_weights([1.0, NaN], 2)
        @test_throws DomainError PO.assert_signed_composite_weights([1.0, -1.0, 0.0] .* 0.0,
                                                                    3)
    end

    @testset "Two Descriptors at plus and minus one give half their difference" begin
        rf = return_forecast(FixedWeightedReturnForecast(; scores = ds, scale = 1.0,
                                                         weights = [1.0, -1.0]), rd, csfm)
        @test rf.hist ≈ (a .- b) ./ 2
        @test rf.mu ≈ vec((a .- b)[end, :] ./ 2)
        @test rf.weights == [0.5, -0.5]
        @test isa(rf, FixedWeightedReturnForecastResult)
    end

    @testset "The scale multiplies the composite" begin
        rf = return_forecast(FixedWeightedReturnForecast(; scores = ds, scale = 3.0,
                                                         weights = [1.0, -1.0]), rd, csfm)
        @test rf.hist ≈ 3 .* (a .- b) ./ 2
    end

    @testset "A missing score leaves the coverage below the threshold" begin
        rdm = forecast_hand_panel(["a" => [NaN 2.0; 3.0 4.0], "b" => b])
        dsm = DescriptorScores(;
                               descriptors = [Passthrough(; field = "a"),
                                              Passthrough(; field = "b")],
                               outlier = nothing, scoring = nothing)
        r6 = return_forecast(FixedWeightedReturnForecast(; scores = dsm, scale = 1.0,
                                                         weights = [1.0, -1.0],
                                                         min_coverage = 0.6), rdm, csfm)
        @test isnan(r6.hist[1, 1])
        @test r6.hist[2, 1] ≈ (a[2, 1] - b[2, 1]) / 2
        r0 = return_forecast(FixedWeightedReturnForecast(; scores = dsm, scale = 1.0,
                                                         weights = [1.0, -1.0],
                                                         min_coverage = 0.0), rdm, csfm)
        @test r0.hist[1, 1] ≈ -b[1, 1]
    end

    @testset "One Descriptor takes no second scoring pass" begin
        ds1 = DescriptorScores(; descriptors = [Passthrough(; field = "a")],
                               outlier = nothing,
                               scoring = CrossSectionalStandardiser(; min_group_size = 2))
        rf = return_forecast(FixedWeightedReturnForecast(; scores = ds1, scale = 1.0), rd,
                             csfm)
        S = descriptor_scores(ds1, rd, csfm)
        @test rf.hist ≈ S[:, :, 1]
    end

    @testset "The Sharpe unit converts the whole history" begin
        vs = [0.04 0.25; 0.09 0.01]
        blk = forecast_hand_block(2; vs = vs)
        rf = return_forecast(FixedWeightedReturnForecast(; scores = ds, scale = 2.0,
                                                         weights = [1.0, -1.0],
                                                         unit = IdiosyncraticSharpeUnit()),
                             rd, blk)
        @test rf.hist ≈ 2 .* ((a .- b) ./ 2) .* sqrt.(vs)
        @test rf.mu ≈ rf.hist[end, :]
        @test_throws PO.IsNothingError return_forecast(FixedWeightedReturnForecast(;
                                                                                   scores = ds,
                                                                                   scale = 2.0,
                                                                                   unit = IdiosyncraticSharpeUnit()),
                                                       rd, csfm)
    end

    @testset "The constructor refuses a scale and a coverage that state nothing" begin
        @test_throws DomainError FixedWeightedReturnForecast(; scores = ds, scale = 0.0)
        @test_throws DomainError FixedWeightedReturnForecast(; scores = ds, scale = -1.0)
        @test_throws DomainError FixedWeightedReturnForecast(; scores = ds, scale = NaN)
        @test_throws DomainError FixedWeightedReturnForecast(; scores = ds, scale = 1.0,
                                                             min_coverage = 1.5)
        @test_throws DomainError FixedWeightedReturnForecast(; scores = ds, scale = 1.0,
                                                             min_coverage = NaN)
    end

    @testset "The Result checks its own shapes" begin
        @test_throws PO.IsEmptyError FixedWeightedReturnForecastResult(; mu = Float64[],
                                                                       hist = [1.0 2.0],
                                                                       weights = [1.0])
        @test_throws PO.IsEmptyError FixedWeightedReturnForecastResult(; mu = [1.0, 2.0],
                                                                       hist = Matrix{Float64}(undef,
                                                                                              0,
                                                                                              0),
                                                                       weights = [1.0])
        @test_throws PO.IsEmptyError FixedWeightedReturnForecastResult(; mu = [1.0, 2.0],
                                                                       hist = [1.0 2.0],
                                                                       weights = Float64[])
        @test_throws DimensionMismatch FixedWeightedReturnForecastResult(; mu = [1.0],
                                                                         hist = [1.0 2.0],
                                                                         weights = [1.0])
    end

    @testset "A signed weight accumulates its absolute value" begin
        num = zeros(1, 2)
        den = zeros(1, 2)
        PO.signed_composite_accumulate!(num, den, reshape([1.0, NaN], 1, 2, 1), [-0.5])
        @test num == [-0.5 0.0]
        @test den == [0.5 0.0]
    end
end

@testset "The member reproduces the reference implementation" begin
    sp = synthetic_asset_panel(; n_assets = 20, n_observations = 60, n_industries = 4,
                               late_listing_proba = 0.3, delisting_proba = 0.3,
                               missing_ratio = 0.08, rng = StableRNG(987654321))
    rd = sp.rd
    pnl = rd.pnl
    T, N = size(pnl.amsk)
    ct_out = CrossSectionalWinsoriser()
    ct_sco = CrossSectionalStandardiser(; min_group_size = 2)
    xc = CompositeExposure(;
                           descriptors = [Passthrough(; field = "book_equity"),
                                          Passthrough(; field = "market_cap")],
                           weights = [0.4, 0.6], min_coverage = 0.5, outlier = ct_out,
                           scoring = ct_sco, group = "industry", bw = "market_cap")
    Lo = factor_exposure(OneHotExposure(; field = "industry", family = "industry"), rd)
    K = 1 + size(Lo, 3)
    Ms = Array{Float64, 3}(undef, T, N, K)
    Ms[:, :, 1] = factor_exposure(xc, rd)
    for k in 1:size(Lo, 3)
        Ms[:, :, k + 1] = Lo[:, :, k]
    end
    nf = ["style"; ["ind$k" for k in 1:size(Lo, 3)]]
    fam = ["style"; fill("industry", size(Lo, 3))]
    vs = [pnl.amsk[t, i] ? 0.0004 * (1.5 + sin(0.3 * t + 0.7 * i)) : NaN
          for t in 1:T, i in 1:N]
    csfm = CrossSectionalFactorModel(; M = Ms[end, :, :], b = zeros(N), Ms = Ms, vs = vs,
                                     nf = nf, fam = fam)
    descriptors = [Passthrough(; field = "book_equity"),
                   Passthrough(; field = "market_cap")]

    @testset "The signed composite matches the stored case cell by cell" begin
        ds = DescriptorScores(; descriptors = descriptors, outlier = ct_out,
                              scoring = ct_sco, group = "industry")
        H = return_forecast(FixedWeightedReturnForecast(; scores = ds, scale = 0.02,
                                                        weights = [0.4, -0.6],
                                                        min_coverage = 0.5), rd, csfm).hist
        E = Matrix(CSV.read(joinpath(@__DIR__,
                                     "assets/FixedWeightedReturnForecast1.csv.gz"),
                            DataFrame))
        @test size(H) == size(E)
        @test isequal(isnan.(H), isnan.(E))
        @test H[isfinite.(E)] ≈ E[isfinite.(E)]
    end

    @testset "The neutralised Sharpe case matches the stored case cell by cell" begin
        ds = DescriptorScores(; descriptors = descriptors, neutralise = ["industry"],
                              outlier = ct_out, scoring = ct_sco, group = "industry")
        rf = return_forecast(FixedWeightedReturnForecast(; scores = ds, scale = 0.03,
                                                         unit = IdiosyncraticSharpeUnit()),
                             rd, csfm)
        E = Matrix(CSV.read(joinpath(@__DIR__,
                                     "assets/FixedWeightedReturnForecast2.csv.gz"),
                            DataFrame))
        @test size(rf.hist) == size(E)
        @test isequal(isnan.(rf.hist), isnan.(E))
        @test rf.hist[isfinite.(E)] ≈ E[isfinite.(E)]
        @test isequal(rf.mu, rf.hist[end, :])
    end
end
