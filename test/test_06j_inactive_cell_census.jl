#=
The census of #1411: every consumer of an Asset Panel ignores what its inactive cells hold.

A numeric Panel Field keeps a finite value in every cell, the inactive ones included (ADR 0102),
and a consumer reads the universe masks to leave those cells out. A consumer that forgets the
masks reads the stored value and gives a plausible answer, where the usual convention of a missing
cell outside the universe would give `NaN`. This file finds such a consumer.

**The poison.** `census_poison` copies a `ReturnsResult` and changes every inactive cell of every
Panel Field: a numeric or a tensor value becomes `1e6`, a category code moves to the next level,
and the observed mask becomes `true`. The last change matters: the builder marks each inactive
cell of the fixture as unobserved, and a reader that honours the observed mask would then never
see the poison. Each case runs one consumer on the clean and on the poisoned copy, and needs the
two answers equal cell by cell, `NaN` pattern included.

**The census.** Two lists are built by reflection, not by hand:

  - every function of the package with a method whose signature names an `AssetPanel`, found by
    walking the signature types;
  - every concrete Descriptor Estimator, exposure estimator, forecast unit and forecast target,
    each of which reads the Asset Panel that a `ReturnsResult` carries.

Each name on the first list has a case below or an exemption that states why no cell can reach its
answer. Each type on the second list has a case. A new consumer fails the census until it gets
one, so it is checked the day it lands.

Two consumers read the inactive cells, and each needs a decision of the maintainer; their cases
are `@test_broken`, and the issue each names holds the reproduction: `FeatureDistance` (#1450)
and the panel collapse of a meta-optimiser (#1451).
=#
include(joinpath(@__DIR__, "parity_harness.jl"))

const CENSUS_PO = PortfolioOptimisers

# The poison: every inactive cell of every Panel Field changes, and reads as observed.
function census_poison_cells!(V::AbstractArray, amsk::AbstractMatrix{Bool}, f)
    tail = ntuple(Returns(Colon()), ndims(V) - ndims(amsk))
    for k in CartesianIndices(amsk)
        if !amsk[k]
            w = view(V, Tuple(k)..., tail...)
            w .= f.(w)
        end
    end
    return V
end
census_poison_mask(::Nothing, ::Any) = nothing
census_poison_mask(m, amsk) = census_poison_cells!(Array(m), amsk, Returns(true))
function census_poison_field(f::NumericPanelField, amsk)
    return NumericPanelField(f.name,
                             census_poison_cells!(Array(f.vals), amsk, Returns(1e6)),
                             census_poison_mask(f.omsk, amsk))
end
function census_poison_field(f::CategoricalPanelField, amsk)
    nl = length(f.levels)
    return CategoricalPanelField(f.name, f.levels,
                                 census_poison_cells!(Array(f.codes), amsk,
                                                      c -> mod1(c + 1, nl)),
                                 census_poison_mask(f.omsk, amsk))
end
function census_poison_field(f::TensorPanelField, amsk)
    return TensorPanelField(f.name, f.axis, f.labels, f.groups,
                            census_poison_cells!(Array(f.vals), amsk, Returns(1e6)),
                            census_poison_mask(f.omsk, amsk))
end
function census_poison(rd::ReturnsResult)
    pnl = rd.pnl
    pf = [census_poison_field(f, pnl.amsk) for f in pnl.pf]
    return ReturnsResult(; nx = rd.nx, X = rd.X, nf = rd.nf, F = rd.F, nb = rd.nb, B = rd.B,
                         ne = rd.ne, E = rd.E, ts = rd.ts, iv = rd.iv, ivpa = rd.ivpa,
                         pnl = AssetPanel(pf, pnl.amsk, pnl.emsk))
end

# Two answers are equal when every array they hold is equal cell by cell, `NaN` included. A
# Result has no `==` of its own, so the comparison walks its fields.
census_equal(a, b) = isequal(a, b)
census_equal(a::Number, b::Number) = isequal(a, b)
census_equal(a::AbstractString, b::AbstractString) = a == b
census_equal(a::Symbol, b::Symbol) = a === b
function census_equal(a::AbstractArray, b::AbstractArray)
    return size(a) == size(b) && all(census_equal(x, y) for (x, y) in zip(a, b))
end
function census_equal(a::Union{Tuple, NamedTuple}, b::Union{Tuple, NamedTuple})
    return typeof(a) == typeof(b) && all(census_equal(x, y) for (x, y) in zip(a, b))
end
function census_equal(a::T, b::T) where {T}
    if !isstructtype(T) || fieldcount(T) == 0 || a isa Function
        return isequal(a, b)
    end
    return all(census_equal(getfield(a, k), getfield(b, k)) for k in 1:fieldcount(T))
end

# The signature census: a method names an Asset Panel when one of its argument types is an
# `AssetPanel`, or holds one inside a `Union`, a type parameter or a bound.
census_mentions(::Any, ::Int = 0) = false
census_mentions(t::TypeVar, d::Int = 0) = census_mentions(t.ub, d)
census_mentions(t::UnionAll, d::Int = 0) = census_mentions(Base.unwrap_unionall(t), d)
census_mentions(t::Union, d::Int = 0) = census_mentions(t.a, d) || census_mentions(t.b, d)
function census_mentions(t::DataType, d::Int = 0)
    if t <: AssetPanel
        return true
    end
    if d > 4
        return false
    end
    return any(p -> census_mentions(p, d + 1), t.parameters)
end
function panel_signature_census()
    out = Set{Symbol}()
    for n in names(CENSUS_PO; all = true)
        # A name that starts with `#` is a generated keyword body; its function is listed.
        if startswith(string(n), '#')
            continue
        end
        if !(isdefined(CENSUS_PO, n))
            continue
        end
        f = getfield(CENSUS_PO, n)
        if !(f isa Function)
            continue
        end
        for m in methods(f)
            if !(m.module === CENSUS_PO)
                continue
            end
            if any(census_mentions, Base.unwrap_unionall(m.sig).parameters[2:end])
                push!(out, nameof(f))
                break
            end
        end
    end
    return out
end
function panel_family_census()
    fams = (CENSUS_PO.AbstractDescriptorEstimator, CENSUS_PO.AbstractExposureEstimator,
            CENSUS_PO.AbstractForecastUnit, CENSUS_PO.AbstractForecastTarget)
    return Set{Symbol}(nameof(T) for F in fams
                       for T in CENSUS_PO.traverse_concrete_subtypes(F))
end

# The functions of the signature census that no case runs, and why no inactive cell reaches
# their answer.
const CENSUS_EXEMPT = Dict{Symbol, String}(
                                           # The reads that state a policy, which test_06i covers.
                                           :panel_field => "a lookup: it returns the Panel Field itself",
                                           :panel_field_values => "the read at a stated policy, which test_06i pins",
                                           # Checks, shapes and names: no cell value enters the answer.
                                           :assert_asset_panel_supplied => "a check of presence",
                                           :assert_panel_concat => "a check of axes, names and static fields before a vcat",
                                           :check_asset_panel => "a check of the axes",
                                           :panel_axes => "reads the axes",
                                           :panel_is_static => "reads whether the masks exist",
                                           :panel_feature_names => "reads the names",
                                           :panel_column_label => "reads the names",
                                           :feature_labels => "reads the names",
                                           :collapse_rows => "reads the axes",
                                           # Transport: a cut, a view or a move keeps each inactive cell as it was, and the consumer
                                           # that reads the result is the one a case runs.
                                           :port_opt_view => "a view",
                                           :asset_panel_view => "a view",
                                           :fold_asset_panel => "a cut to the rows of a fold",
                                           :windowed_panel => "a cut to the rows of a window",
                                           :feature_row_indices => "reads the clock",
                                           :project_panel_clock => "moves a panel onto a new clock",
                                           :attach_universe_masks => "replaces the masks",
                                           # An export returns the stored cells by design, as the default read does.
                                           :panel_dataframe => "an export of the stored cells",
                                           :panel_frame_field => "a step of panel_dataframe",
                                           :panel_frame_fields => "a step of panel_dataframe",
                                           :panel_frame_long => "a step of panel_dataframe",
                                           :panel_frame_wide => "a step of panel_dataframe",
                                           # A report whose share of all cells counts every cell by design; its other
                                           # numbers are the cases of describe and panel_info_levels below.
                                           :panel_info => "a report of every cell",
                                           :panel_info_header => "a step of panel_info",
                                           :panel_info_fields => "a step of panel_info",
                                           # Steps whose public caller has a case below.
                                           :descriptor_active_fill! => "a step of every Descriptor",
                                           :ew_active_returns => "a step of the exponentially weighted Descriptors",
                                           :exposure_active_fill! => "a step of every exposure",
                                           :exposure_weight_fill! => "a step of the benchmark weights of an exposure",
                                           :coverage_panel_moment => "a step of the moments with a Coverage Policy",
                                           :coverage_series_frame => "a step of variance_series",
                                           :coverage_variance_series => "a step of variance_series",
                                           :windowed_series_row => "a step of variance_series on a windowed estimator",
                                           :windowed_variance_series => "a step of variance_series on a windowed estimator",
                                           :prior_forecast_location => "a step of forecast_location on a prior",
                                           :ep_prior => "passes the panel to the prior it wraps",
                                           # The stack of the Feature Matrix returns the stored cells, and FeatureDistance reads it.
                                           :panel_feature_matrix => "the stack FeatureDistance reads, whose case is below (#1450)",
                                           :feature_matrix => "the stack FeatureDistance reads, whose case is below (#1450)",
                                           :select_fields => "a step of feature_matrix",
                                           :select_fields_push! => "a step of feature_matrix")

# The fixture: the small parity panel, with the negative volume and short interest made
# positive so the turnover and the days to cover compute, a benchmark weight for the composite
# exposure under a name the prior does not write, and a tensor field so the poison reaches a label axis.
function census_fixture()
    fx = parity_small_panel()
    rd = fx.rd
    pnl = rd.pnl
    T, N = size(rd.X)
    rng = StableRNG(1411)
    pf = CENSUS_PO.AbstractPanelField[]
    for f in pnl.pf
        if f.name in ("adj_volume", "short_interest")
            push!(pf, NumericPanelField(f.name, abs.(f.vals), f.omsk))
        else
            push!(pf, f)
        end
    end
    push!(pf, NumericPanelField(; name = "bench", vals = rand(rng, T, N)))
    push!(pf,
          TensorPanelField(; name = "loadings", axis = "factor", labels = ["x", "y"],
                           vals = randn(rng, T, N, 2)))
    rdc = ReturnsResult(; nx = rd.nx, X = rd.X, ne = rd.ne, E = rd.E,
                        pnl = AssetPanel(pf, pnl.amsk, pnl.emsk))
    return rdc, census_poison(rdc)
end

@testset "Every consumer of an Asset Panel ignores its inactive cells (#1411)" begin
    rd, rdp = census_fixture()
    X = rd.X
    macro_s = rd.E[:, findfirst(==("MACRO"), rd.ne)]
    cvg = CoveragePolicy()
    D(de; kw...) = r -> descriptor(de, r; kw...)
    L(xe) = r -> factor_exposure(xe, r)
    ds3 = [BookToPrice(), Passthrough(; field = "style2"), LogMarketCap()]
    pass(f) = CompositeExposure(; descriptors = [Passthrough(; field = f)],
                                outlier = nothing, scoring = nothing, family = "style")
    factors = ["market" => ConstantExposure(), "style1" => pass("style1"),
               "style2" => pass("style2")]
    pe = EmpiricalPrior(; me = ExpWeightedExpectedReturns(),
                        ce = RegimeAdjustedExpWeightedCovariance(; centred = true))
    ve = RegimeAdjustedExpWeightedVariance(; centred = true, min_val = 0.0)
    cspe(; kw...) = CrossSectionalFactorPrior(; factors = factors, pe = pe, ve = ve,
                                              minra = 5, kw...)
    fit(r; kw...) = prior(cspe(; kw...), r)
    scores = DescriptorScores(;
                              descriptors = [Passthrough(; field = "net_income_ttm"),
                                             Passthrough(; field = "sales_ttm"),
                                             EWMomentum(; half_life = 5, skip = 3)])
    rf(e) = r -> return_forecast(e, r, fit(r).rr)
    fe(target) = r -> forecast_evaluation(FixedWeightedReturnForecast(; scores = scores,
                                                                      scale = 0.02), r,
                                          fit(r).rr; target = target)
    # Each case: the census names it covers, a label, and the consumer.
    cases = Any[
                # The Descriptor Estimators.
                ((:PanelFieldRatio,), "BookToPrice", D(BookToPrice())),
                ((:PanelFieldRatio,), "a ratio of a sum",
                 D(PanelFieldRatio(; num = ["net_income_ttm" => 1, "sales_ttm" => 1],
                                   den = "total_assets"))),
                ((:PanelFieldLog,), "LogMarketCap", D(LogMarketCap())),
                ((:Passthrough,), "Passthrough", D(Passthrough(; field = "style2"))),
                ((:GrowthRate,), "lag 30 across a listing",
                 D(GrowthRate(; field = "sales_ttm", lag = 30))),
                ((:ChangeToScale,), "lag 30 across a listing",
                 D(ChangeToScale(; field = "net_income_ttm", scale = "total_assets",
                                 lag = 30))),
                ((:ChangeInIntensity,), "lag 30 across a listing",
                 D(ChangeInIntensity(; field = "sales_ttm", scale = "total_assets",
                                     lag = 30))),
                ((:EWMean,), "EWMomentum", D(EWMomentum(; half_life = 5, skip = 3))),
                ((:EWVolumeRatio,), "a volume ratio",
                 D(EWVolumeRatio(; num = "adj_volume", den = "adj_shares_outstanding",
                                 decay = 0.8, min_obs = 4))),
                ((:EWVolumeRatio,), "EWAmihudIlliquidity", D(EWAmihudIlliquidity())),
                ((:DaysToCover,), "DaysToCover", D(DaysToCover())),
                ((:EWVolatility,), "EWVolatility", D(EWVolatility())),
                ((:EWVolatility,), "EWDownsideVolatility", D(EWDownsideVolatility())),
                ((:EWResidualVolatility,), "EWResidualVolatility",
                 D(EWResidualVolatility())),
                ((:EWBeta,), "EWMarketBeta", D(EWMarketBeta())),
                ((:EWBeta, :cross_sectional_groups), "a group prior",
                 D(EWMarketBeta(; half_life = 8, group = "industry", min_group_size = 3))),
                ((:EWBeta,), "aggregated in groups",
                 D(EWMarketBeta(; half_life = 4, agg_obs = 3, group = "industry",
                                min_group_size = 3))),
                ((:EWDownsideBeta,), "EWDownsideBeta", D(EWDownsideBeta())),
                ((:EWMacroSensitivity,), "a series",
                 D(EWMacroSensitivity(; series = "MACRO"))),
                ((:EWMacroSensitivity,), "a reference",
                 D(EWMacroSensitivity(); ref = macro_s)),
                ((:RollingLogReturn,), "RollingLogReturn",
                 D(RollingLogReturn(; window = 7, skip = 2))),
                ((:RollingMax,), "RollingMax", D(RollingMax(; window = 5))),
                # The exposures.
                ((:ConstantExposure,), "ConstantExposure", L(ConstantExposure())),
                ((:OneHotExposure,), "OneHotExposure",
                 L(OneHotExposure(; field = "industry", family = "industry"))),
                ((:CompositeExposure,), "on the benchmark weights",
                 L(CompositeExposure(; descriptors = ds3, weights = [0.5, 0.3, 0.2],
                                     min_coverage = 0.6, bw = "bench"))),
                ((:CompositeExposure, :cross_sectional_groups), "scored in groups",
                 L(CompositeExposure(; descriptors = ds3, weights = [0.5, 0.3, 0.2],
                                     min_coverage = 0.6, bw = "bench",
                                     scoring = CrossSectionalGaussianRank(;
                                                                          min_group_size = 3),
                                     group = "industry"))),
                ((:CurrencyExposure,), "CurrencyExposure", L(CurrencyExposure())),
                ((:ObservedExposure,), "ObservedExposure",
                 L(ObservedExposure(;
                                    xe = CompositeExposure(;
                                                           descriptors = [Passthrough(;
                                                                                      field = "style1")],
                                                           outlier = nothing,
                                                           scoring = nothing, bw = "bench"),
                                    series = "MACRO", family = "macro"))),
                ((:DerivedExposure, :prior), "a derived exposure inside the prior",
                 r -> prior(CrossSectionalFactorPrior(;
                                                      factors = ["market" =>
                                                                     ConstantExposure(),
                                                                 "c1" => pass("style1"),
                                                                 "d1" => DerivedExposure(;
                                                                                         source = "c1",
                                                                                         f = x -> x .^
                                                                                                  2)],
                                                      pe = pe, ve = ve, minra = 5), r)),
                # The prior and what reads its block.
                ((:prior,), "CrossSectionalFactorPrior", r -> fit(r)),
                ((:prior,), "with a one-hot family",
                 r -> fit(r;
                          factors = [factors;
                                     "industry" => OneHotExposure(; field = "industry",
                                                                  family = "industry")])),
                ((:prior,), "EmpiricalPrior on a ReturnsResult",
                 r -> prior(EmpiricalPrior(), r)),
                ((:IdiosyncraticReturnUnit,), "a fixed-weight forecast",
                 rf(FixedWeightedReturnForecast(; scores = scores, scale = 0.02))),
                ((:IdiosyncraticSharpeUnit,), "a Sharpe forecast",
                 rf(FixedWeightedReturnForecast(; scores = scores, scale = 0.02,
                                                unit = IdiosyncraticSharpeUnit()))),
                ((), "an exponentially weighted forecast",
                 rf(ExpWeightedReturnForecast(; scores = scores))),
                ((:IdiosyncraticTarget,), "the evaluation", fe(IdiosyncraticTarget())),
                ((:AssetReturnTarget,), "the evaluation", fe(AssetReturnTarget())),
                ((:PanelFieldTarget,), "the evaluation",
                 fe(PanelFieldTarget(; name = "market_cap"))),
                # The summary of a panel. The share of all cells counts every cell by design,
                # so the case of describe reads its active columns and its levels.
                ((:panel_align_active,), "the alignment",
                 r -> (a = panel_align_active(r.pnl,
                                              ["book_equity", "industry", "loadings"]);
                       (a.pnl.amsk, a.pnl.emsk, a.n))),
                ((:describe, :panel_active_cells), "the active columns and the levels",
                 r -> (d = describe(r.pnl);
                       (d.active_cells, d.active_missing, d.assets_missing,
                        Matrix(describe(r.pnl; by = "industry")[:, [:cells, :missing]])))),
                ((:panel_info_levels,), "the level groups",
                 r -> sprint(io -> CENSUS_PO.panel_info_levels(io, r.pnl,
                                                               panel_field(r.pnl,
                                                                           "industry"),
                                                               r.pnl.amsk))),
                # The moments that read the masks of a panel.
                ((:mean,), "mean",
                 r -> mean(SimpleExpectedReturns(; cvg = cvg), r.X, r.pnl)),
                ((:cov,), "cov", r -> cov(Covariance(; cvg = cvg), r.X, r.pnl)),
                ((:cor,), "cor", r -> cor(Covariance(; cvg = cvg), r.X, r.pnl)),
                ((:var,), "var", r -> var(SimpleVariance(; cvg = cvg), r.X, r.pnl)),
                ((:std,), "std", r -> std(SimpleVariance(; cvg = cvg), r.X, r.pnl)),
                ((:variance_series,), "variance_series",
                 r -> CENSUS_PO.variance_series(SimpleVariance(; cvg = cvg), r.X, r.pnl)),
                ((:coskewness,), "coskewness", r -> coskewness(Coskewness(), r.X, r.pnl)),
                ((:cokurtosis,), "cokurtosis", r -> cokurtosis(Cokurtosis(), r.X, r.pnl)),
                ((:forecast_location,), "forecast_location",
                 r -> CENSUS_PO.forecast_location(Covariance(), r.X, r.pnl)),
                ((:coverage_mask, :coverage_reduction), "the Coverage Universe",
                 r -> CENSUS_PO.coverage_reduction(r.X, r.pnl)),
                ((:cross_sectional_panel_masks, :panel_moment_masks, :last_active_mask),
                 "the masks",
                 r -> (CENSUS_PO.cross_sectional_panel_masks(r.pnl),
                       CENSUS_PO.panel_moment_masks(r.pnl),
                       CENSUS_PO.last_active_mask(r.pnl)))]
    # Each case whose consumer reads an inactive cell, and the issue that holds the decision.
    broken = Any[((:FeatureDistance,), "FeatureDistance at the last observation (#1450)",
                  r -> distance(FeatureDistance(; sel = ["market_cap", "industry"]),
                                nothing, r.X; rd = r)),
                 ((:collapse_asset_panel,), "the panel of a meta-optimiser (#1451)",
                  r -> CENSUS_PO.collapse_asset_panel(r.pnl,
                                                      [fill(1 / 6, 6) zeros(6);
                                                       zeros(6) fill(1 / 6, 6)], r.nx))]
    @testset "The census names each consumer once" begin
        covered = Set{Symbol}(n for c in [cases; broken] for n in c[1])
        sig = panel_signature_census()
        @test isempty(setdiff(sig, covered, keys(CENSUS_EXEMPT)))
        @test isempty(setdiff(keys(CENSUS_EXEMPT), sig))
        @test isempty(intersect(covered, keys(CENSUS_EXEMPT)))
        @test isempty(setdiff(panel_family_census(), covered))
    end
    @testset "$(join(names, ", ")): $(label)" for (names, label, f) in cases
        @test census_equal(f(rd), f(rdp))
    end
    @testset "$(join(names, ", ")): $(label)" for (names, label, f) in broken
        @test_broken census_equal(f(rd), f(rdp))
    end
end
