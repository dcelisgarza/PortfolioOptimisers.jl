#=
The shared parts of the censuses of the consumers of an Asset Panel: test_06j poisons the
inactive cells (#1411), and test_06m poisons the active unobserved cells (#1508). Each census
copies the fixture, poisons the copy, runs every case on both, and needs equal answers.
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
# An inactive cell that reads as observed holds no placeholder (#1631).
census_poison_placeholder(::Nothing, ::Any) = nothing
census_poison_placeholder(p, amsk) = census_poison_cells!(Array(p), amsk, Returns(false))
function census_poison_field(f::NumericPanelField, amsk, v)
    return NumericPanelField(f.name, census_poison_cells!(Array(f.vals), amsk, Returns(v)),
                             census_poison_mask(f.omsk, amsk),
                             census_poison_placeholder(f.pmsk, amsk))
end
function census_poison_field(f::CategoricalPanelField, amsk, ::Any)
    nl = length(f.levels)
    return CategoricalPanelField(f.name, f.levels,
                                 census_poison_cells!(Array(f.codes), amsk,
                                                      c -> mod1(c + 1, nl)),
                                 census_poison_mask(f.omsk, amsk),
                                 census_poison_placeholder(f.pmsk, amsk))
end
function census_poison_field(f::TensorPanelField, amsk, v)
    return TensorPanelField(f.name, f.axis, f.labels, f.groups,
                            census_poison_cells!(Array(f.vals), amsk, Returns(v)),
                            census_poison_mask(f.omsk, amsk),
                            census_poison_placeholder(f.pmsk, amsk))
end
census_poison_rates(::Nothing, ::Any) = nothing
function census_poison_rates(iv::AbstractMatrix, amsk)
    return census_poison_cells!(Array(iv), amsk, Returns(1e6))
end
census_poison_rates(ivpa::AbstractVector, amsk) = ifelse.(amsk[end, :], ivpa, 1e6)
# `v` is the value a numeric or a tensor cell takes. The rates keep `1e6`, which is positive.
function census_poison(rd::ReturnsResult; v = 1e6)
    pnl = rd.pnl
    pf = [census_poison_field(f, pnl.amsk, v) for f in pnl.pf]
    return ReturnsResult(; nx = rd.nx, X = rd.X, nf = rd.nf, F = rd.F, nb = rd.nb, B = rd.B,
                         ne = rd.ne, E = rd.E, ts = rd.ts,
                         iv = census_poison_rates(rd.iv, pnl.amsk),
                         ivpa = census_poison_rates(rd.ivpa, pnl.amsk),
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
    # A test file defines its own members, `CarryRuleUserDescriptor` of `test_12zg` for one,
    # and they stay in the session after it. The census reads the members of the library.
    return Set{Symbol}(nameof(T) for F in fams
                       for T in CENSUS_PO.traverse_concrete_subtypes(F)
                       if parentmodule(T) === CENSUS_PO)
end

# The functions of the signature census that no case runs, and why no inactive cell reaches
# their answer.
const CENSUS_EXEMPT = Dict{Symbol, String}(
                                           # The reads that state a policy, which
                                           # test_06b_asset_panel.jl covers.
                                           :panel_field => "a lookup: it returns the Panel Field itself",
                                           :panel_field_values => "the read at a stated policy, which test_06b_asset_panel.jl pins",
                                           # Checks, shapes and names: no cell value enters the answer.
                                           :assert_asset_panel_supplied => "a check of presence",
                                           :assert_panel_concat => "a check of axes, names and static fields before a vcat",
                                           :check_asset_panel => "a check of the axes",
                                           :panel_axes => "reads the axes",
                                           :panel_is_static => "reads whether the masks exist",
                                           :attribution_active_mask => "reads the active mask of the panel",
                                           :panel_feature_names => "reads the names",
                                           :panel_column_label => "reads the names",
                                           :feature_labels => "reads the names",
                                           :collapse_rows => "reads the axes and the active mask",
                                           :entry_activity => "a step of feature_readable_mask, whose case is below",
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
                                           :panel_frame_axes => "names the axes",
                                           :panel_manifest => "reads the names, the kinds and the levels, and no cell value",
                                           # A report whose share of all cells counts every cell by design; its other
                                           # numbers are the cases of describe and panel_info_levels below.
                                           :panel_info => "a report of every cell",
                                           :panel_info_header => "a step of panel_info",
                                           :panel_info_fields => "a step of panel_info",
                                           # Steps whose public caller has a case below.
                                           :descriptor_active_fill! => "a step of every Descriptor",
                                           :ew_active_returns => "a step of the exponentially weighted Descriptors",
                                           :ew_volatility_variance => "a step of the exponentially weighted volatility Descriptors",
                                           :exposure_active_fill! => "a step of every exposure",
                                           :exposure_weight_fill! => "a step of the benchmark weights of an exposure",
                                           :coverage_panel_moment => "a step of the moments with a Coverage Policy",
                                           :coverage_series_frame => "a step of variance_series",
                                           :coverage_variance_series => "a step of variance_series",
                                           :windowed_series_row => "a step of variance_series on a windowed estimator",
                                           :windowed_variance_series => "a step of variance_series on a windowed estimator",
                                           :prior_forecast_location => "a step of forecast_location on a prior",
                                           # Steps of the fit of the Cross-Sectional Factor Prior, whose `prior` has a case below.
                                           :cross_sectional_benchmark_stage => "a step of prior on a CrossSectionalFactorPrior",
                                           :cross_sectional_exposure_series => "a step of prior on a CrossSectionalFactorPrior",
                                           :cross_sectional_exposure_stage => "a step of prior on a CrossSectionalFactorPrior",
                                           # The online step of a prior: it moves the cells of a step into a buffer or a carry,
                                           # and the call with no data runs the batch prior that a case covers.
                                           :buffer_prior => "the batch prior over the rows of a sample buffer",
                                           :refit_step_kwargs => "passes the masks and the Panel Fields of a step to the refit",
                                           :step_active_kwargs => "passes the active mask of a step",
                                           :step_panel_fields => "passes or refuses the Panel Fields of a step",
                                           :assert_carry_step_panel => "a check of the masks and the fields that a carry honours",
                                           :ep_prior => "passes the panel to the prior it wraps",
                                           :read_member => "passes the panel of a buffer to the batch moment, whose cases are below",
                                           :carried_location => "passes the panel of a carry to forecast_location, whose case is below",
                                           # The steps of the priors that rebuild a prior or read a co-moment: each passes the
                                           # panel on, as ep_prior does (R7 of #1630).
                                           :scenario_prior => "passes the panel to the refit of the prior it wraps",
                                           :scenario_factory => "a step of scenario_prior: passes the panel to the prior it rebuilds",
                                           :prior_probability_factory => "passes the panel to the prior it rebuilds under the prior probabilities",
                                           :opinion_pooling_weights => "passes the panel to the prior of each opinion",
                                           :scenario_rows => "a cut to the rows of the scenarios",
                                           :read_comoment => "passes the panel to cokurtosis or coskewness, whose cases are below",
                                           # The refit and the pin of a prior that reweights its scenarios (G9 of #1630).
                                           :ep_wrapped_prior => "a step of ep_prior: passes the panel to the prior it wraps",
                                           :scenario_count => "counts the rows the nested prior answers, through prior on the panel",
                                           :opinion_pooling_scenarios => "passes the panel to the prior of pe1",
                                           :pin_prior_choice => "passes the panel to the prior that the pin of the Choice Rule fits",
                                           :sample_buffer_fit_kwargs => "the panel picks the keywords of a buffer, and no cell is read",
                                           :cross_sectional_regression_stage => "a step of prior on a CrossSectionalFactorPrior",
                                           :cross_sectional_factor_scenarios => "the regression stage of prior on a CrossSectionalFactorPrior",
                                           # Masks: the active mask and the estimation mask, and no cell value.
                                           :coverage_keyword_reduction => "a step of the moments on a panel: the active mask, through coverage_mask, whose case is below",
                                           :coverage_mask_source => "chooses the active mask coverage_mask reads, whose case is below",
                                           :benchmark_estimation_mask => "narrows the estimation mask to the finite benchmark returns",
                                           # An export of every stored cell. FeatureDistance reads the stack of feature_matrix
                                           # beside the readable mask.
                                           :panel_feature_matrix => "an export of every stored cell beside its observed-mask column",
                                           :feature_matrix => "the stack FeatureDistance reads, whose cases are below",
                                           :select_fields => "a step of feature_matrix",
                                           :select_fields_push! => "a step of feature_matrix",
                                           :feature_stack => "a step of feature_matrix",
                                           :feature_stack_eltype => "reads the types and the masks, and no cell value",
                                           # The mask of the cells FeatureDistance reads: the masks alone, and no cell value.
                                           :feature_data_cells => "reads the placeholder masks",
                                           :feature_window_mask => "a step of FeatureDistance, whose cases are below",
                                           :readable_window_rows => "reads the axes and the masks")

# The fixture: the small parity panel, with the negative volume and short interest made
# positive so the turnover and the days to cover compute, a benchmark weight for the composite
# exposure under a name the prior does not write, and a tensor field so the poison reaches a label axis.
# `blank` adds blanks to the clean copy, and `poison` makes the poisoned copy from it.
function census_fixture(poison = census_poison, blank = identity)
    fx = parity_small_panel()
    rd = fx.rd
    pnl = rd.pnl
    T, N = size(rd.X)
    rng = StableRNG(1411)
    pf = CENSUS_PO.AbstractPanelField[]
    for f in pnl.pf
        if f.name in ("adj_volume", "short_interest")
            push!(pf, NumericPanelField(f.name, abs.(f.vals), f.omsk, f.pmsk))
        else
            push!(pf, f)
        end
    end
    push!(pf, NumericPanelField(; name = "bench", vals = rand(rng, T, N)))
    push!(pf,
          TensorPanelField(; name = "loadings", axis = "factor", labels = ["x", "y"],
                           vals = randn(rng, T, N, 2)))
    rdc = blank(ReturnsResult(; nx = rd.nx, X = rd.X, ne = rd.ne, E = rd.E,
                              pnl = AssetPanel(pf, pnl.amsk, pnl.emsk)))
    return rdc, poison(rdc)
end
# The fixture of the panel collapse: the census fixture with a static input lifted over the
# observations, a square field whose labels are the asset names, the implied volatilities, and
# a clock, which the cross-validated path needs to find the rows of each fold.
function census_collapse_fixture(rd::ReturnsResult, poison = census_poison,
                                 blank = identity)
    pnl = rd.pnl
    T, N = size(rd.X)
    rng = StableRNG(1456)
    pf = CENSUS_PO.AbstractPanelField[pnl.pf;
                                      NumericPanelField(; name = "lift",
                                                        vals = CENSUS_PO.RepeatedLeading(rand(rng,
                                                                                              N),
                                                                                         T));
                                      TensorPanelField(; name = "adjacency", axis = "asset",
                                                       labels = rd.nx,
                                                       vals = rand(rng, T, N, N))]
    ts = CENSUS_PO.Dates.Date(2020, 1, 1) .+ CENSUS_PO.Dates.Day.(0:(T - 1))
    rdc = blank(ReturnsResult(; nx = rd.nx, X = rd.X, ne = rd.ne, E = rd.E, ts = ts,
                              iv = 0.1 .+ rand(rng, T, N), ivpa = 0.5 .+ rand(rng, N),
                              pnl = AssetPanel(pf, pnl.amsk, pnl.emsk)))
    return rdc, poison(rdc)
end
# Each case: the census names it covers, a label, and the consumer. `rd` gives the series of
# the macro sensitivity, which the poison leaves as it is.
function census_cases(rd::ReturnsResult)
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
                        ce = RegimeAdjustedExpWeightedCovariance(; centring = PreCentred()))
    ve = RegimeAdjustedExpWeightedVariance(; centring = PreCentred(), min_val = 0.0)
    cspe(; kw...) = CrossSectionalFactorPrior(; lambda = 1, factors = factors, pe = pe,
                                              ve = ve, minra = 5, kw...)
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
                # The carry fold reads a rolling return off its carried state (#1583).
                ((:CarriedDescriptor, :descriptor_step, :descriptor_carry,
                  :cross_sectional_descriptor_carry), "folded from a carried state",
                 r -> descriptor(last(only(CENSUS_PO.cross_sectional_descriptor_carry(["m" =>
                                                                                           RollingLogReturn(;
                                                                                                            window = 7,
                                                                                                            skip = 2)],
                                                                                      r,
                                                                                      size(r.X,
                                                                                           1)).xv)),
                                 r)),
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
                 r -> prior(CrossSectionalFactorPrior(; lambda = 1,
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
    # FeatureDistance reads each asset and each pair at its own active rows (#1454). The
    # last row holds every asset but the delisted asset 3, and rows 1 to 20 beside rows 61 to 80
    # give the pair (2, 3) no shared active row.
    fsel = ["style1", "style2"]
    fd(alg; kwargs...) = FeatureDistance(; sel = fsel, alg = alg, kwargs...)
    # The assets readable at the last row: active, and observed in each selected field.
    function last_view(r)
        keep = CENSUS_PO.feature_readable_mask(fd(LastRow()), nothing, r)
        return CENSUS_PO.port_opt_view(r, isnothing(keep) ? Colon() : findall(keep))
    end
    gap_view(r) = CENSUS_PO.port_opt_view(r, vcat(1:20, 61:80), :)
    fdist(de, r) = distance(de, nothing, r.X; rd = r)
    fmsg(f) =
        try
            f()
        catch e
            sprint(showerror, e)
        end
    euc = CENSUS_PO.Distances.Euclidean()
    fb = FeatureFallback()
    append!(cases,
            Any[((:FeatureDistance,),
                 "FeatureDistance under LastRow, on the assets it reads",
                 r -> fdist(fd(LastRow()), last_view(r))),
                ((:FeatureDistance,),
                 "FeatureDistance under LastRow refuses the delisted asset",
                 r -> fmsg(() -> fdist(fd(LastRow()), r))),
                ((:FeatureDistance,), "FeatureDistance under LastActiveRow",
                 r -> fdist(fd(LastActiveRow()), r)),
                ((:FeatureDistance,), "FeatureDistance under AggregateFeatures",
                 r -> fdist(fd(AggregateFeatures()), r)),
                ((:FeatureDistance,), "FeatureDistance under a weighted median",
                 r -> fdist(fd(AggregateFeatures(; alg = MedianCollapse(),
                                                 w = CENSUS_PO.StatsBase.eweights(size(r.X,
                                                                                       1),
                                                                                  0.05))),
                            r)),
                ((:FeatureDistance,), "FeatureDistance under AggregateDistances",
                 r -> fdist(fd(AggregateDistances()), r)),
                ((:FeatureDistance,), "FeatureDistance under StackObservations",
                 r -> fdist(fd(StackObservations(); metric = euc), r)),
                ((:FeatureDistance,), "an empty pair under RefusePair",
                 r -> fmsg(() -> fdist(fd(AggregateDistances()), gap_view(r)))),
                ((:FeatureDistance,), "an empty pair under FeatureFallback",
                 r -> (fdist(fd(AggregateDistances(; pair = fb)), gap_view(r)),
                       fdist(fd(StackObservations(; pair = fb); metric = euc), gap_view(r)))),
                ((:FeatureDistance, :feature_readable_mask),
                 "an empty pair under DropFewerRows, at the entry of a fit",
                 r -> CENSUS_PO.feature_readable_mask(fd(StackObservations(;
                                                                           pair = DropFewerRows())),
                                                      nothing, gap_view(r)))])
    for alg in CENSUS_PCOLS
        push!(cases,
              ((:collapse_asset_panel,), "the panel of a meta-optimiser under $(alg)",
               r -> CENSUS_PO.collapse_asset_panel(r.pnl, CENSUS_WC, r.nx, alg)))
    end
    return cases
end
# The checks of FeatureDistance read only the cells that its collapse reads (#1734). The census
# fixture makes `adj_volume` and `short_interest` non-negative, so each case is in the domain of
# its metric on the clean copy, and a negative poison leaves the domain at every inactive cell.
# Each collapse runs under a metric that needs non-negative features. ChiSqDist has a rescale
# under StackObservations, and BrayCurtis and Jaccard have none.
function census_domain_cases(rd::ReturnsResult)
    fsel = ["adj_volume", "short_interest"]
    fd(alg, metric) = FeatureDistance(; sel = fsel, alg = alg, metric = metric)
    fdist(de, r) = distance(de, nothing, r.X; rd = r)
    fmsg(f) =
        try
            f()
        catch e
            sprint(showerror, e)
        end
    gap_view(r) = CENSUS_PO.port_opt_view(r, vcat(1:20, 61:80), :)
    chi = CENSUS_PO.Distances.ChiSqDist()
    bray = CENSUS_PO.Distances.BrayCurtis()
    jac = CENSUS_PO.Distances.Jaccard()
    fb = FeatureFallback()
    return Any[("LastRow refuses the delisted asset",
                r -> fmsg(() -> fdist(fd(LastRow(), chi), r))),
               ("LastActiveRow", r -> fdist(fd(LastActiveRow(), chi), r)),
               ("AggregateFeatures", r -> fdist(fd(AggregateFeatures(), bray), r)),
               ("a median under AggregateFeatures",
                r -> fdist(fd(AggregateFeatures(; alg = MedianCollapse()), jac), r)),
               ("AggregateDistances", r -> fdist(fd(AggregateDistances(), jac), r)),
               ("StackObservations, with a rescale",
                r -> fdist(fd(StackObservations(), chi), r)),
               ("StackObservations, with no rescale",
                r -> fdist(fd(StackObservations(), bray), r)),
               ("an empty pair under RefusePair",
                r -> fmsg(() -> fdist(fd(StackObservations(), jac), gap_view(r)))),
               ("an empty pair under FeatureFallback",
                r -> (fdist(fd(AggregateDistances(; pair = fb), jac), gap_view(r)),
                      fdist(fd(StackObservations(; pair = fb), chi), gap_view(r))))]
end
# The panel collapse of a meta-optimiser weighs the active members of each observation (#1456).
# The census fixture holds numeric, categorical and rectangular tensor fields; the collapse
# fixture adds a lifted field, a square field and the implied volatilities. Two sub-portfolios
# hold assets 1 to 6 and 7 to 12, so the first loses asset 2 before it lists, asset 3 after it
# delists and asset 4 in its gap. The third holds asset 2 alone, so it is inactive on rows 1 to
# 20. The first cluster of the cross-validated case holds assets 2 and 3, of which one alone is
# active on rows 1 to 20 and on rows 61 to 80.
const CENSUS_WC = [fill(1 / 6, 6) zeros(6) [0.0; 1.0; zeros(4)];
                   zeros(6) fill(1 / 6, 6) zeros(6)]
const CENSUS_PCOLS = (RenormaliseActive(), InactiveAsCash())
# The cases of the panel collapse of a meta-optimiser, on the collapse fixture `rdk`.
function census_collapse_cases(rdk::ReturnsResult)
    ucl = CENSUS_PO.ClusterUniverse([[2, 3], [1; 4:12]])
    cvg1456 = CoveragePolicy()
    pe1456 = EmpiricalPrior(; me = SimpleExpectedReturns(; cvg = cvg1456),
                            ce = Covariance(; cvg = cvg1456))
    # The predictions read the returns alone, which the poison leaves as they are, so one set
    # serves the clean and the poisoned fixture.
    preds1456 = [cross_val_predict(InverseVolatility(; pe = pe1456), rdk, KFold(; n = 4);
                                   cols = cl, ex = CENSUS_PO.FLoops.SequentialEx())
                 for cl in ucl.cls]
    collapse_cases = Any[]
    for alg in CENSUS_PCOLS
        append!(collapse_cases,
                Any[((:collapse_asset_panel,), "a lifted and a square field under $(alg)",
                     r -> CENSUS_PO.collapse_asset_panel(r.pnl, CENSUS_WC, r.nx, alg)),
                    ((), "iv and ivpa of the outer problem under $(alg)",
                     r -> CENSUS_PO.prepare_outer_rd(r, CENSUS_WC, alg)[3:5]),
                    ((), "the cross-validated path under $(alg)",
                     r -> (o = CENSUS_PO.rebuild_returns_result(r, preds1456, ucl, alg);
                           (o.pnl, o.iv, o.ivpa)))])
    end
    # The rates of one fold, which `reconstruct_rd` collapses with the weights of a
    # sub-portfolio. The first sub-portfolio holds asset 3, which delists before the last row,
    # so the poison reaches both `iv` and `ivpa` (R7 of #1630).
    push!(collapse_cases,
          ((:collapse_fold_rate,), "the rates of a fold",
           r -> (CENSUS_PO.collapse_fold_rate(r.iv, CENSUS_WC[:, 1], r.pnl),
                 CENSUS_PO.collapse_fold_rate(r.ivpa, CENSUS_WC[:, 1], r.pnl))))
    return collapse_cases
end
# The census of the names: each consumer that the reflection finds has a case or an exemption,
# and no name has both.
function census_names_test(cases, collapse_cases, exempt::AbstractDict)
    covered = Set{Symbol}(n for c in [cases; collapse_cases] for n in c[1])
    sig = panel_signature_census()
    @test isempty(setdiff(sig, covered, keys(exempt)))
    @test isempty(setdiff(keys(exempt), sig))
    @test isempty(intersect(covered, keys(exempt)))
    @test isempty(setdiff(panel_family_census(), covered))
    return nothing
end
