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
    S = descriptor_scores(ds, rd, csfm).S

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
        Sn = descriptor_scores(dsn, rd, blk).S
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
        @test size(descriptor_scores(dsg, rdg, csfm).S) == (2, 2, 1)
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
        S = descriptor_scores(ds1, rd, csfm).S
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

#=
Issue #738 continues the file: the two members that FIT their Descriptor weights.

FOUR MORE CONVENTIONS SHAPE THESE PROBES.

5. THE TARGET MATURES. The target of observation `t` is the mean of the idiosyncratic
   returns over `t + lag` to `t + lag + horizon - 1`, so the last `lag + horizon - 1`
   observations state no target and the fit never sees them. The forecast at `t` reads the
   coefficient row `t - (lag + horizon - 1)`, which is why the first rows of a history are
   `NaN` and why `mu` equals the last row of it.

6. THE FORECAST UNIT CHOOSES THE PAIR. In the return unit the target is the forward return
   and the weights are the inverse idiosyncratic variance. In the Sharpe unit the target
   carries the division and the weights are the estimation mask alone.

7. THE STORED CASES ARE THE REFERENCE IMPLEMENTATION'S OWN OUTPUT, on the same synthetic
   panel the file already rebuilds, with the idiosyncratic returns and variances drawn from
   the two closed forms the last testset writes and exported to both sides.

8. TWO DELIBERATE DEPARTURES FROM THE REFERENCE, both settled by issue #655.
   - `cv = nothing` calibrates IN SAMPLE. The reference's own `cv=None` still splits into
     five folds, and `cv = KFold(; n = 5)` reproduces that to a relative 4.8e-13.
   - `min_obs` gates the publication of a coefficient. The reference publishes from the
     first fitted observation, which is `min_obs = 1`, and every stored case sets it.
=#

function forecast_fit_panel(a::AbstractMatrix, b::AbstractMatrix, eps::AbstractMatrix,
                            vs::AbstractMatrix; emsk = trues(size(a)...))
    T, N = size(a)
    inp = [NumericPanelInput(; name = "a", vals = Matrix{Float64}(a),
                             alg = ForwardPanelFill()),
           NumericPanelInput(; name = "b", vals = Matrix{Float64}(b),
                             alg = ForwardPanelFill())]
    pnl = asset_panel(inp; amsk = trues(T, N), emsk = emsk)
    rd = ReturnsResult(; nx = ["A$i" for i in 1:N], X = zeros(T, N), pnl = pnl)
    csr = CrossSectionalRegression(; f = zeros(T, 1), eps = Matrix{Float64}(eps),
                                   n = fill(N, T))
    csfm = CrossSectionalFactorModel(; M = reshape(fill(1.0, N), N, 1), b = zeros(N),
                                     csr = csr, vs = Matrix{Float64}(vs))
    ds = DescriptorScores(;
                          descriptors = [Passthrough(; field = "a"),
                                         Passthrough(; field = "b")], outlier = nothing,
                          scoring = nothing)
    return rd, csfm, ds
end

@testset "The forward target matures, and the unit converts it" begin
    PO = PortfolioOptimisers

    @testset "The forward mean is the mean of its window, and skips a NaN" begin
        @test isequal(PO.forward_mean_returns([1.0; 2.0; NaN; 4.0; 5.0;;], 2, 1),
                      [2.0; 4.0; 4.5; NaN; NaN;;])
        @test isequal(PO.forward_mean_returns([1.0; 2.0; 3.0;;], 1, 1), [2.0; 3.0; NaN;;])
        @test isequal(PO.forward_mean_returns([1.0; 2.0; 3.0;;], 1, 2), [3.0; NaN; NaN;;])
        @test isequal(PO.forward_mean_returns([1.0; NaN; NaN;;], 2, 1), [NaN; NaN; NaN;;])
        @test all(isnan, PO.forward_mean_returns([1.0; 2.0;;], 5, 1))
    end

    @testset "The unit converts the target, and the return unit reads no variance" begin
        y = [0.2 1.0]
        @test PO.forecast_unit_target(IdiosyncraticReturnUnit(), y, nothing) === y
        @test PO.forecast_unit_target(IdiosyncraticReturnUnit(), y, [0.04 0.25]) === y
        @test PO.forecast_unit_target(IdiosyncraticSharpeUnit(), y, [0.04 0.25]) ≈ [1.0 2.0]
    end

    @testset "The two block reads refuse what they need and do not have" begin
        blk = forecast_hand_block(2)
        @test_throws PO.IsNothingError PO.forecast_idiosyncratic_returns(blk)
        @test_throws PO.IsNothingError PO.forecast_idiosyncratic_variances(blk)
        csr = CrossSectionalRegression(; f = zeros(2, 1), eps = [0.1 0.2; 0.3 0.4],
                                       n = [2, 2])
        blk2 = CrossSectionalFactorModel(; M = reshape([1.0, 1.0], 2, 1), b = zeros(2),
                                         csr = csr, vs = [0.1 0.2; 0.3 0.0])
        @test PO.forecast_idiosyncratic_returns(blk2) == [0.1 0.2; 0.3 0.4]
        @test_throws DomainError PO.forecast_idiosyncratic_variances(blk2)
        blk3 = CrossSectionalFactorModel(; M = reshape([1.0, 1.0], 2, 1), b = zeros(2),
                                         csr = csr, vs = [0.1 0.2; 0.3 NaN])
        @test isequal(PO.forecast_idiosyncratic_variances(blk3), [0.1 0.2; 0.3 NaN])
    end
end

@testset "The exponentially weighted member fits its Descriptor weights" begin
    PO = PortfolioOptimisers
    a = [1.0 2.0 3.0; 2.0 1.0 4.0]
    b = [4.0 1.0 2.0; 1.0 3.0 2.0]
    eps = [0.01 -0.02 0.03; -0.01 0.02 0.01]
    vs = [0.04 0.09 0.01; 0.02 0.05 0.03]
    rd, csfm, ds = forecast_fit_panel(a, b, eps, vs)

    @testset "One trainable observation is the plain weighted least squares" begin
        rf = return_forecast(ExpWeightedReturnForecast(; scores = ds, decay = 0.5,
                                                       min_obs = 1, ridge = 0.0), rd, csfm)
        S1 = [a[1, :] b[1, :]]
        w = inv.(vs[1, :])
        w ./= sum(w) / length(w)
        W = LinearAlgebra.Diagonal(w)
        coef = (transpose(S1) * W * S1) \ (transpose(S1) * W * eps[2, :])
        @test rf.coef ≈ coef
        @test rf.n == 1
        @test all(isnan, rf.hist[1, :])
        @test rf.hist[2, :] ≈ [a[2, :] b[2, :]] * coef
        @test isequal(rf.mu, rf.hist[end, :])
        @test isa(rf, ExpWeightedReturnForecastResult)
    end

    @testset "The accumulators are the state the recursion carries" begin
        rf = return_forecast(ExpWeightedReturnForecast(; scores = ds, decay = 0.5,
                                                       min_obs = 1, ridge = 0.0), rd, csfm)
        S1 = [a[1, :] b[1, :]]
        w = inv.(vs[1, :])
        w ./= sum(w) / length(w)
        Sw = S1 .* sqrt.(w)
        yw = eps[2, :] .* sqrt.(w)
        @test rf.A ≈ 0.5 * transpose(Sw) * Sw
        @test rf.c ≈ 0.5 * transpose(Sw) * yw
        @test size(rf.A) == (2, 2)
        @test length(rf.c) == 2
    end

    @testset "Without normalisation the weights enter as they stand" begin
        rf = return_forecast(ExpWeightedReturnForecast(; scores = ds, decay = 0.5,
                                                       min_obs = 1, ridge = 0.0,
                                                       normalise = false), rd, csfm)
        S1 = [a[1, :] b[1, :]]
        W = LinearAlgebra.Diagonal(inv.(vs[1, :]))
        @test rf.coef ≈ (transpose(S1) * W * S1) \ (transpose(S1) * W * eps[2, :])
    end

    @testset "The Sharpe unit divides the target and weighs by the mask alone" begin
        rf = return_forecast(ExpWeightedReturnForecast(; scores = ds, decay = 0.5,
                                                       min_obs = 1, ridge = 0.0,
                                                       unit = IdiosyncraticSharpeUnit()),
                             rd, csfm)
        S1 = [a[1, :] b[1, :]]
        # The target of an observation is divided by ITS OWN idiosyncratic volatility, not by
        # the volatility of the observation the forward return comes from.
        y = eps[2, :] ./ sqrt.(vs[1, :])
        coef = (transpose(S1) * S1) \ (transpose(S1) * y)
        @test rf.coef ≈ coef
        @test rf.hist[2, :] ≈ ([a[2, :] b[2, :]] * coef) .* sqrt.(vs[2, :])
    end

    @testset "Three observations follow the hand-written recursion" begin
        a3 = [1.0 2.0 3.0; 2.0 1.0 4.0; 3.0 2.0 1.0; 1.0 4.0 2.0]
        b3 = [4.0 1.0 2.0; 1.0 3.0 2.0; 2.0 1.0 3.0; 3.0 2.0 1.0]
        e3 = [0.01 -0.02 0.03; -0.01 0.02 0.01; 0.02 0.01 -0.03; 0.00 0.03 0.02]
        v3 = [0.04 0.09 0.01; 0.02 0.05 0.03; 0.06 0.01 0.02; 0.03 0.04 0.05]
        rd3, blk3, ds3 = forecast_fit_panel(a3, b3, e3, v3)
        lambda = 0.4
        rf = return_forecast(ExpWeightedReturnForecast(; scores = ds3, decay = lambda,
                                                       min_obs = 1, ridge = 0.0), rd3, blk3)
        A = zeros(2, 2)
        c = zeros(2)
        coefs = Matrix{Float64}(undef, 3, 2)
        for t in 1:3
            St = [a3[t, :] b3[t, :]]
            w = inv.(v3[t, :])
            w ./= sum(w) / length(w)
            Sw = St .* sqrt.(w)
            yw = e3[t + 1, :] .* sqrt.(w)
            A .= lambda .* A .+ (1 - lambda) .* (transpose(Sw) * Sw)
            c .= lambda .* c .+ (1 - lambda) .* (transpose(Sw) * yw)
            coefs[t, :] = A \ c
        end
        @test rf.A ≈ A
        @test rf.c ≈ c
        @test rf.coef ≈ coefs[3, :]
        @test rf.n == 3
        for t in 2:4
            @test rf.hist[t, :] ≈ [a3[t, :] b3[t, :]] * coefs[t - 1, :]
        end
        @test all(isnan, rf.hist[1, :])
    end

    @testset "min_obs holds the coefficients back until the warm-up ends" begin
        a3 = [1.0 2.0 3.0; 2.0 1.0 4.0; 3.0 2.0 1.0; 1.0 4.0 2.0]
        b3 = [4.0 1.0 2.0; 1.0 3.0 2.0; 2.0 1.0 3.0; 3.0 2.0 1.0]
        e3 = [0.01 -0.02 0.03; -0.01 0.02 0.01; 0.02 0.01 -0.03; 0.00 0.03 0.02]
        v3 = fill(0.02, 4, 3)
        rd3, blk3, ds3 = forecast_fit_panel(a3, b3, e3, v3)
        rf = return_forecast(ExpWeightedReturnForecast(; scores = ds3, decay = 0.5,
                                                       min_obs = 3), rd3, blk3)
        @test all(isnan, rf.hist[1:3, :])
        @test all(isfinite, rf.hist[4, :])
        rf9 = return_forecast(ExpWeightedReturnForecast(; scores = ds3, decay = 0.5,
                                                        min_obs = 9), rd3, blk3)
        @test all(isnan, rf9.hist)
        @test all(isnan, rf9.mu)
        # `min_obs` gates the PUBLICATION of a forecast, not the state of the recursion: the
        # Result still carries the coefficients an update seam would resume from.
        @test all(isfinite, rf9.coef)
        @test rf9.n == 3
    end

    @testset "A masked asset and a horizon that eats the sample" begin
        emsk = [true false true; true true true]
        rd2, blk2, ds2 = forecast_fit_panel(a, b, eps, vs; emsk = emsk)
        rf = return_forecast(ExpWeightedReturnForecast(; scores = ds2, decay = 0.5,
                                                       min_obs = 1, ridge = 0.0), rd2, blk2)
        S1 = [a[1, [1, 3]] b[1, [1, 3]]]
        w = inv.(vs[1, [1, 3]])
        w ./= sum(w) / length(w)
        W = LinearAlgebra.Diagonal(w)
        @test rf.coef ≈ (transpose(S1) * W * S1) \ (transpose(S1) * W * eps[2, [1, 3]])
        rfh = return_forecast(ExpWeightedReturnForecast(; scores = ds, decay = 0.5,
                                                        min_obs = 1, horizon = 4), rd, csfm)
        @test all(isnan, rfh.hist)
        @test rfh.n == 0
    end

    @testset "The helpers state the weights, the mask, the solve and the history" begin
        @test PO.ew_forecast_weights(IdiosyncraticReturnUnit(), [1.0 1.0], [0.5 0.25]) ==
              [2.0 4.0]
        w = [1.0 0.0]
        @test PO.ew_forecast_weights(IdiosyncraticSharpeUnit(), w, [0.5 0.25]) === w
        S = reshape([1.0, NaN, 2.0, 3.0], 1, 2, 2)
        @test PO.ew_forecast_valid(S, [0.1 0.2], [0.3 0.4], [1.0 1.0]) == [true false]
        @test PO.ew_forecast_valid(S, [0.1 0.2], [0.3 0.4], [0.0 1.0]) == [false false]
        @test PO.ew_forecast_valid(S, [NaN 0.2], [0.3 0.4], [1.0 1.0]) == [false false]
        @test PO.ew_forecast_valid(S, [0.1 0.2], [NaN 0.4], [1.0 1.0]) == [false false]
        @test PO.ew_forecast_solve([1.0 1.0; 1.0 1.0], [2.0, 2.0], 0.0, 1) ≈ [1.0, 1.0]
        @test PO.ew_forecast_solve([2.0 0.0; 0.0 4.0], [2.0, 4.0], 0.0, 1) ≈ [1.0, 1.0]
        @test PO.ew_forecast_solve([2.0 0.0; 0.0 2.0], [2.0, 2.0], 1.0, 1) ≈ [0.5, 0.5]
        H = PO.ew_forecast_history(reshape([1.0, 2.0, 3.0, 4.0], 2, 2, 1),
                                   reshape([5.0], 1, 1), 1)
        @test isequal(H, [NaN NaN; 10.0 20.0])
    end

    @testset "The estimator and the Result refuse what states no fit" begin
        @test_throws DomainError ExpWeightedReturnForecast(; scores = ds, decay = 0.0)
        @test_throws DomainError ExpWeightedReturnForecast(; scores = ds, decay = 1.0)
        @test_throws DomainError ExpWeightedReturnForecast(; scores = ds, decay = 0.5,
                                                           min_obs = 0)
        @test_throws DomainError ExpWeightedReturnForecast(; scores = ds, decay = 0.5,
                                                           ridge = -1.0)
        @test_throws DomainError ExpWeightedReturnForecast(; scores = ds, decay = 0.5,
                                                           ridge = NaN)
        @test_throws DomainError ExpWeightedReturnForecast(; scores = ds, decay = 0.5,
                                                           horizon = 0)
        @test_throws DomainError ExpWeightedReturnForecast(; scores = ds, decay = 0.5,
                                                           lag = 0)
        @test_throws DomainError ExpWeightedReturnForecast(; scores = ds, decay = 0.5,
                                                           scale = 0.0)
        @test ExpWeightedReturnForecast(; scores = ds, half_life = 2).decay ≈
              PO.half_life_decay(2)
        @test ExpWeightedReturnForecast(; scores = ds, half_life = 2).min_obs == 2

        @test_throws PO.IsEmptyError ExpWeightedReturnForecastResult(; mu = Float64[],
                                                                     hist = [1.0 2.0],
                                                                     coef = [1.0],
                                                                     A = fill(1.0, 1, 1),
                                                                     c = [1.0], n = 1)
        @test_throws PO.IsEmptyError ExpWeightedReturnForecastResult(; mu = [1.0, 2.0],
                                                                     hist = Matrix{Float64}(undef,
                                                                                            0,
                                                                                            0),
                                                                     coef = [1.0],
                                                                     A = fill(1.0, 1, 1),
                                                                     c = [1.0], n = 1)
        @test_throws PO.IsEmptyError ExpWeightedReturnForecastResult(; mu = [1.0, 2.0],
                                                                     hist = [1.0 2.0],
                                                                     coef = Float64[],
                                                                     A = fill(1.0, 1, 1),
                                                                     c = [1.0], n = 1)
        @test_throws DimensionMismatch ExpWeightedReturnForecastResult(; mu = [1.0],
                                                                       hist = [1.0 2.0],
                                                                       coef = [1.0],
                                                                       A = fill(1.0, 1, 1),
                                                                       c = [1.0], n = 1)
        @test_throws DimensionMismatch ExpWeightedReturnForecastResult(; mu = [1.0, 2.0],
                                                                       hist = [1.0 2.0],
                                                                       coef = [1.0],
                                                                       A = fill(1.0, 2, 2),
                                                                       c = [1.0], n = 1)
        @test_throws DimensionMismatch ExpWeightedReturnForecastResult(; mu = [1.0, 2.0],
                                                                       hist = [1.0 2.0],
                                                                       coef = [1.0],
                                                                       A = fill(1.0, 1, 1),
                                                                       c = [1.0, 2.0],
                                                                       n = 1)
        @test_throws DomainError ExpWeightedReturnForecastResult(; mu = [1.0, 2.0],
                                                                 hist = [1.0 2.0],
                                                                 coef = [1.0],
                                                                 A = fill(1.0, 1, 1),
                                                                 c = [1.0], n = -1)
    end
end

@testset "The target member fits a regression over every observation and asset" begin
    PO = PortfolioOptimisers
    a = [1.0 2.0 3.0; 2.0 1.0 4.0; 3.0 2.0 1.0; 1.0 4.0 2.0]
    b = [4.0 1.0 2.0; 1.0 3.0 2.0; 2.0 1.0 3.0; 3.0 2.0 1.0]
    eps = [0.01 -0.02 0.03; -0.01 0.02 0.01; 0.02 0.01 -0.03; 0.00 0.03 0.02]
    vs = [0.04 0.09 0.01; 0.02 0.05 0.03; 0.06 0.01 0.02; 0.03 0.04 0.05]
    rd, csfm, ds = forecast_fit_panel(a, b, eps, vs)

    @testset "Every valid pair is one sample of an ordinary least squares" begin
        rf = return_forecast(TargetReturnForecast(; scores = ds, target_outlier = nothing,
                                                  calibrate = false), rd, csfm)
        Sf = [vec(transpose(a[1:3, :])) vec(transpose(b[1:3, :]))]
        yf = vec(transpose(eps[2:4, :]))
        coef = Sf \ yf
        @test rf.mu ≈ [a[4, :] b[4, :]] * coef
        @test isnothing(rf.hist)
        @test isnan(rf.calib)
        @test isa(rf, TargetReturnForecastResult)
        @test PortfolioOptimisers.StatsAPI.coef(rf.model) ≈ coef
    end

    @testset "A longer horizon drops the observations whose target has not matured" begin
        rf = return_forecast(TargetReturnForecast(; scores = ds, target_outlier = nothing,
                                                  calibrate = false, horizon = 2), rd, csfm)
        Sf = [vec(transpose(a[1:2, :])) vec(transpose(b[1:2, :]))]
        yf = vec(transpose((eps[2:3, :] .+ eps[3:4, :]) ./ 2))
        @test rf.mu ≈ [a[4, :] b[4, :]] * (Sf \ yf)
    end

    @testset "The scale multiplies, and the calibration states its own coefficient" begin
        rf1 = return_forecast(TargetReturnForecast(; scores = ds, target_outlier = nothing,
                                                   calibrate = false), rd, csfm)
        rf = return_forecast(TargetReturnForecast(; scores = ds, target_outlier = nothing,
                                                  calibrate = false, scale = 3.0), rd, csfm)
        @test rf.mu ≈ 3 .* rf1.mu
        rfc = return_forecast(TargetReturnForecast(; scores = ds, target_outlier = nothing,
                                                   decay = 0.5, min_obs = 1), rd, csfm)
        @test isfinite(rfc.calib)
        @test rfc.mu ≈ rfc.calib .* rf1.mu
        rfw = return_forecast(TargetReturnForecast(; scores = ds, target_outlier = nothing,
                                                   decay = 0.5, min_obs = 9), rd, csfm)
        @test isnan(rfw.calib)
        @test all(isnan, rfw.mu)
    end

    @testset "The Sharpe unit divides the target and multiplies the forecast" begin
        rf = return_forecast(TargetReturnForecast(; scores = ds, target_outlier = nothing,
                                                  calibrate = false,
                                                  unit = IdiosyncraticSharpeUnit()), rd,
                             csfm)
        Sf = [vec(transpose(a[1:3, :])) vec(transpose(b[1:3, :]))]
        yf = vec(transpose(eps[2:4, :] ./ sqrt.(vs[1:3, :])))
        @test rf.mu ≈ ([a[4, :] b[4, :]] * (Sf \ yf)) .* sqrt.(vs[4, :])
    end

    @testset "An out of fold calibration needs two samples per fold" begin
        rf = return_forecast(TargetReturnForecast(; scores = ds, target_outlier = nothing,
                                                  decay = 0.5, min_obs = 1,
                                                  cv = KFold(; n = 3)), rd, csfm)
        @test isfinite(rf.calib)
        @test_throws ArgumentError return_forecast(TargetReturnForecast(; scores = ds,
                                                                        target_outlier = nothing,
                                                                        decay = 0.5,
                                                                        min_obs = 1,
                                                                        cv = KFold(; n = 7)),
                                                   rd, csfm)
    end

    @testset "A sample set with no valid pair fits nothing" begin
        rf = return_forecast(TargetReturnForecast(; scores = ds, target_outlier = nothing,
                                                  calibrate = false, horizon = 9), rd, csfm)
        @test isnothing(rf.model)
        @test all(isnan, rf.mu)
        @test isnan(rf.calib)
    end

    @testset "The helpers state the variances, the layout, the latest row and the scale" begin
        @test isnothing(PO.target_forecast_variances(IdiosyncraticReturnUnit(), csfm,
                                                     false))
        @test PO.target_forecast_variances(IdiosyncraticReturnUnit(), csfm, true) == vs
        @test PO.target_forecast_variances(IdiosyncraticSharpeUnit(), csfm, false) == vs
        @test PO.target_forecast_scatter([1.0, 2.0, 3.0, 4.0], 2, 2) == [1.0 2.0; 3.0 4.0]
        @test isnothing(PO.target_forecast_latest_variances(nothing))
        @test PO.target_forecast_latest_variances([1.0 2.0; 3.0 4.0]) == [3.0 4.0]
        @test all(isnan, PO.target_forecast_latest(nothing, reshape([1.0, 2.0], 1, 2, 1)))
        @test PO.target_forecast_multiplier(false, NaN) == 1
        @test PO.target_forecast_multiplier(true, 2.5) == 2.5
        S = reshape([1.0, NaN, 2.0, 3.0], 1, 2, 2)
        Sf, yf, ok = PO.target_forecast_samples(S, [0.1 0.2], [1.0 1.0], 1)
        @test ok == [true, false]
        @test isequal(Sf, [1.0 2.0; NaN 3.0])
        @test yf == [0.1, 0.2]
        @test isnothing(PO.target_forecast_fit(TargetReturnForecast(; scores = ds), Sf, yf,
                                               falses(2)))
    end

    @testset "The estimator and the Result refuse what states no fit" begin
        @test_throws DomainError TargetReturnForecast(; scores = ds, horizon = 0)
        @test_throws DomainError TargetReturnForecast(; scores = ds, lag = 0)
        @test_throws DomainError TargetReturnForecast(; scores = ds, scale = 0.0)
        @test_throws DomainError TargetReturnForecast(; scores = ds, decay = 1.0)
        @test_throws DomainError TargetReturnForecast(; scores = ds, min_obs = 0)
        @test TargetReturnForecast(; scores = ds, half_life = 2).min_obs == 2
        @test isa(TargetReturnForecast(; scores = ds).target_outlier,
                  CrossSectionalWinsoriser)
        @test isnothing(TargetReturnForecast(; scores = ds).target_scoring)
        @test_throws PO.IsEmptyError TargetReturnForecastResult(; mu = Float64[])
        r = TargetReturnForecastResult(; mu = [1.0])
        @test isnothing(r.hist)
        @test isnothing(r.model)
        @test isnan(r.calib)
    end
end

@testset "The two fitted members reproduce the reference implementation" begin
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
    eps = [if pnl.amsk[t, i]
               0.01 * sin(0.7 * t + 0.29 * i) + 0.004 * cos(0.11 * t * i)
           else
               NaN
           end
           for t in 1:T, i in 1:N]
    csr = CrossSectionalRegression(; f = zeros(T, K), eps = eps, n = fill(N, T))
    csfm = CrossSectionalFactorModel(; M = Ms[end, :, :], b = zeros(N), csr = csr, Ms = Ms,
                                     vs = vs, nf = nf, fam = fam)
    descriptors = [Passthrough(; field = "book_equity"),
                   Passthrough(; field = "market_cap")]
    ds1 = DescriptorScores(; descriptors = descriptors, outlier = ct_out, scoring = ct_sco,
                           group = "industry")
    stored(name) = Matrix(CSV.read(joinpath(@__DIR__, "assets/$name.csv.gz"), DataFrame))

    @testset "The return unit history matches the stored case cell by cell" begin
        rf = return_forecast(ExpWeightedReturnForecast(; scores = ds1, half_life = 10.0,
                                                       min_obs = 1, horizon = 2, lag = 1,
                                                       scale = 1.5), rd, csfm)
        E = stored("ExpWeightedReturnForecast1")
        @test size(rf.hist) == size(E)
        @test isequal(isnan.(rf.hist), isnan.(E))
        @test rf.hist[isfinite.(E)] ≈ E[isfinite.(E)]
        @test isequal(rf.mu, rf.hist[end, :])
    end

    @testset "The neutralised Sharpe history matches the stored case cell by cell" begin
        ds2 = DescriptorScores(; descriptors = descriptors, neutralise = ["industry"],
                               outlier = ct_out, scoring = ct_sco, group = "industry")
        rf = return_forecast(ExpWeightedReturnForecast(; scores = ds2, half_life = 10.0,
                                                       min_obs = 1, horizon = 3, lag = 2,
                                                       unit = IdiosyncraticSharpeUnit()),
                             rd, csfm)
        E = stored("ExpWeightedReturnForecast2")
        @test size(rf.hist) == size(E)
        @test isequal(isnan.(rf.hist), isnan.(E))
        m = isfinite.(E) .& (abs.(E) .> 1e-9)
        @test rf.hist[m] ≈ E[m]
    end

    @testset "The uncalibrated target forecast matches the stored case" begin
        rf = return_forecast(TargetReturnForecast(; scores = ds1, horizon = 2, lag = 1,
                                                  calibrate = false), rd, csfm)
        E = vec(stored("TargetReturnForecast1"))
        @test isequal(isnan.(rf.mu), isnan.(E))
        @test rf.mu[isfinite.(E)] ≈ E[isfinite.(E)]
    end

    @testset "The out of fold calibrated forecast matches the stored case" begin
        rf = return_forecast(TargetReturnForecast(; scores = ds1, horizon = 2, lag = 1,
                                                  half_life = 10.0, min_obs = 1,
                                                  cv = KFold(; n = 3)), rd, csfm)
        E = vec(stored("TargetReturnForecast2"))
        @test isequal(isnan.(rf.mu), isnan.(E))
        @test rf.mu[isfinite.(E)] ≈ E[isfinite.(E)]
    end
end

@testset "The calibration skips an observation that states no slope" begin
    a = [1.0 2.0 3.0; 2.0 1.0 4.0; 3.0 2.0 1.0; 1.0 4.0 2.0]
    b = [4.0 1.0 2.0; 1.0 3.0 2.0; 2.0 1.0 3.0; 3.0 2.0 1.0]
    vs = [0.04 0.09 0.01; 0.02 0.05 0.03; 0.06 0.01 0.02; 0.03 0.04 0.05]

    @testset "An observation with one entering asset states no slope" begin
        eps = [0.01 -0.02 0.03; -0.01 0.02 0.01; 0.02 0.01 -0.03; 0.00 0.03 0.02]
        emsk = [true false false; true true true; true true true; true true true]
        rd, csfm, ds = forecast_fit_panel(a, b, eps, vs; emsk = emsk)
        rf = return_forecast(TargetReturnForecast(; scores = ds, target_outlier = nothing,
                                                  decay = 0.5, min_obs = 1), rd, csfm)
        # The first observation carries one asset, so it advances no accumulator; the rest do.
        @test isfinite(rf.calib)
    end

    @testset "A prediction that is identically zero states no slope either" begin
        eps = zeros(4, 3)
        rd, csfm, ds = forecast_fit_panel(a, b, eps, vs)
        rf = return_forecast(TargetReturnForecast(; scores = ds, target_outlier = nothing,
                                                  decay = 0.5, min_obs = 1), rd, csfm)
        # Every coefficient is zero, so the normal accumulator never leaves zero.
        @test isnan(rf.calib)
        @test all(isnan, rf.mu)
    end
end

#=
Issue #835 finishes the file: the estimator scores the WHOLE carrier and answers on the
BLOCK's rows. ADR 0112.

THREE MORE CONVENTIONS SHAPE THESE PROBES.

 9. THE BLOCK IS A SUFFIX OF THE CARRIER, FOUND BY SIZE. The prior drops the leading
    observations its own Descriptors warm up over, so the block's histories are the trailing
    rows of the carrier and `return_forecast_rows` finds them from the two observation
    counts alone. A carrier of exactly the block's length gives the whole range, and a
    carrier shorter than the block is refused.

10. THE DESCRIPTORS OF THE FORECAST WARM UP OVER THE WHOLE CARRIER. A Descriptor with a
    warm-up would otherwise warm up a second time inside the block's window, which is the
    one design the reference implementation cannot express. Every member then cuts to the
    block's rows, so `hist` still lines up with `vs`, with `csr.eps` and with `pr.o_X`.

11. THE TARGET MEMBER KEEPS THE BOUNDARY BAND. Under `whole_history = true` the block's
    idiosyncratic returns are placed into the rows they were fitted on, so a signal row
    before the block whose forward window reaches into the block is a training row. There
    are exactly `lag + horizon - 1` such rows. Under `false` the fit trains on the block's
    rows alone, and the reference implementation has no such mode.

    `assets/FixedWeightedReturnForecast3.csv.gz`, `assets/TargetReturnForecast3.csv.gz` and
    `assets/TargetReturnForecast4.csv.gz` are the reference implementation's own output on
    the panel this testset rebuilds, with the block padded back onto the whole observation
    axis as the reference's own prior pads it. Its momentum Descriptor carries a warm-up,
    which is what the earlier stored cases cannot see. The forecast agrees to a relative
    1.3e-14 and the calibration coefficient BIT FOR BIT.
=#

@testset "The estimator scores the whole carrier and answers on the block's rows" begin
    sp = synthetic_asset_panel(; n_assets = 20, n_observations = 60, n_industries = 4,
                               late_listing_proba = 0.3, delisting_proba = 0.3,
                               missing_ratio = 0.08, rng = StableRNG(987654321))
    rd = sp.rd
    amsk = rd.pnl.amsk
    T, N = size(amsk)
    Tb = 40
    rows = (T - Tb + 1):T
    gap = 2
    vsw = [amsk[t, i] ? 0.0004 * (1.5 + sin(0.3 * t + 0.7 * i)) : NaN
           for t in 1:T, i in 1:N]
    epsw = [amsk[t, i] ? 0.01 * sin(0.7 * t + 0.29 * i) + 0.004 * cos(0.11 * t * i) : NaN
            for t in 1:T, i in 1:N]
    vs = vsw[rows, :]
    eps = epsw[rows, :]
    csr = CrossSectionalRegression(; f = zeros(Tb, 1), eps = eps, n = fill(N, Tb))
    csfm = CrossSectionalFactorModel(; M = ones(N, 1), b = zeros(N), csr = csr, vs = vs)
    ds = DescriptorScores(;
                          descriptors = [EWMomentum(; half_life = 8.0, skip = 2),
                                         Passthrough(; field = "book_equity")],
                          outlier = CrossSectionalWinsoriser(),
                          scoring = CrossSectionalStandardiser(; min_group_size = 2),
                          group = "industry")

    @testset "The block is the trailing rows, and a shorter carrier is refused" begin
        @test PO.return_forecast_rows(rd, csfm) == rows
        # A carrier of exactly the block's length is the whole of it, which is the call a
        # caller makes when it hands the already narrowed carrier.
        @test PO.return_forecast_rows(PO.port_opt_view(rd, rows, :), csfm) == 1:Tb
        @test_throws DimensionMismatch PO.return_forecast_rows(PO.port_opt_view(rd,
                                                                                (first(rows) + 1):T,
                                                                                :), csfm)
        # A block that carries no history states no window.
        bare = CrossSectionalFactorModel(; M = ones(N, 1), b = zeros(N))
        @test PO.return_forecast_rows(rd, bare) == 1:T
        @test isnothing(PO.return_forecast_block_observations(bare))
        @test PO.return_forecast_block_observations(csfm) == Tb
    end

    @testset "The Descriptors warm up over the carrier, not inside the block" begin
        sc = descriptor_scores(ds, rd, csfm)
        S = sc.S
        @test size(S) == (T, N, 2)
        @test sc.rows == rows
        # The momentum reads a skip of two, so its first two rows read nothing at all.
        @test all(isnan, view(S, 1:2, :, 1))
        # The same recipe on the block alone warms up a second time, so it states fewer
        # scores over the very same rows.
        Sb = descriptor_scores(ds, PO.port_opt_view(rd, rows, :), csfm).S
        @test size(Sb) == (Tb, N, 2)
        @test count(isfinite, S[rows, :, 1]) > count(isfinite, Sb[:, :, 1])
    end

    @testset "A Neutralisation writes NaN on the rows before the block" begin
        # The block states an exposure only on its own rows, so a score before them has
        # nothing to neutralise against. That is also why a Neutralisation drops the
        # boundary band of the target member: the band's scores leave this step `NaN`.
        Ms = reshape(descriptor(Passthrough(; field = "market_cap"), rd)[rows, :], Tb, N, 1)
        blk = CrossSectionalFactorModel(; M = ones(N, 1), b = zeros(N), csr = csr, vs = vs,
                                        Ms = Ms, nf = ["style"], fam = ["style"])
        dsn = DescriptorScores(; descriptors = [Passthrough(; field = "book_equity")],
                               neutralise = ["style"], outlier = nothing, scoring = nothing)
        sc = descriptor_scores(dsn, rd, blk)
        @test sc.rows == rows
        @test all(isnan, view(sc.S, 1:(first(rows) - 1), :, 1))
        @test any(isfinite, view(sc.S, rows, :, 1))
    end

    @testset "The fixed weighted member matches the stored case on the block's rows" begin
        rf = return_forecast(FixedWeightedReturnForecast(; scores = ds, scale = 0.02,
                                                         weights = [0.4, -0.6],
                                                         min_coverage = 0.5), rd, csfm)
        E = Matrix(CSV.read(joinpath(@__DIR__,
                                     "assets/FixedWeightedReturnForecast3.csv.gz"),
                            DataFrame))
        @test size(E) == (T, N)
        @test size(rf.hist) == (Tb, N)
        B = E[rows, :]
        @test isequal(isnan.(rf.hist), isnan.(B))
        @test rf.hist[isfinite.(B)] ≈ B[isfinite.(B)]
        @test isequal(rf.mu, rf.hist[end, :])
    end

    @testset "Every training row before the block lies in the boundary band" begin
        S = descriptor_scores(ds, rd, csfm).S
        w = PO.return_forecast_weights(rd)
        al = PO.target_forecast_alignment(true, S, eps, nothing, w,
                                          PO.exposure_group_labels(rd, "industry"), rows)
        @test size(al.eps) == (T, N)
        @test all(isnan, view(al.eps, 1:(first(rows) - 1), :))
        fwd = PO.forward_mean_returns(al.eps, 2, 1)
        # A signal row states a target only where its forward window reaches into the block,
        # which is the band of `lag + horizon - 1` rows before it and nothing earlier.
        @test all(isnan, view(fwd, 1:(first(rows) - gap - 1), :))
        @test any(isfinite, view(fwd, (first(rows) - gap):(first(rows) - 1), :))
        # The cut alignment states the block's rows and nothing before them.
        cut = PO.target_forecast_alignment(false, S, eps, nothing, w, nothing, rows)
        @test size(cut.S) == (Tb, N, 2)
        @test isequal(cut.eps, eps)
        @test isnothing(cut.groups)
    end

    @testset "The target member in return units matches the stored case" begin
        rf = return_forecast(TargetReturnForecast(; scores = ds, horizon = 2, lag = 1,
                                                  calibrate = false), rd, csfm)
        E = vec(Matrix(CSV.read(joinpath(@__DIR__, "assets/TargetReturnForecast3.csv.gz"),
                                DataFrame)))
        @test isequal(isnan.(rf.mu), isnan.(E))
        @test rf.mu[isfinite.(E)] ≈ E[isfinite.(E)]
        # The block's rows alone lose the boundary band, so the fit is a different one.
        rb = return_forecast(TargetReturnForecast(; scores = ds, horizon = 2, lag = 1,
                                                  calibrate = false, whole_history = false),
                             rd, csfm)
        @test isequal(isnan.(rb.mu), isnan.(E))
        @test !isapprox(rb.mu[isfinite.(E)], E[isfinite.(E)])
    end

    @testset "The calibration reads the variance at the signal row" begin
        rf = return_forecast(TargetReturnForecast(; scores = ds, horizon = 2, lag = 1,
                                                  calibrate = true, half_life = 10.0,
                                                  min_obs = 1, cv = KFold(; n = 5)), rd,
                             csfm)
        E = vec(Matrix(CSV.read(joinpath(@__DIR__, "assets/TargetReturnForecast4.csv.gz"),
                                DataFrame)))
        # The reference implementation's own coefficient, which the padded rows never enter.
        @test rf.calib ≈ -0.7730488894268933
        @test rf.mu[isfinite.(E)] ≈ E[isfinite.(E)]
    end
end
