#=
The outputs of every configuration that reads the Excess Frame, pinned as fixed values before
`x_src` takes over `brt` (#1679, map #1678).

The values were recorded on `dev` with today's `brt`. A migration build changes the helpers in
the section "The doors of the Excess Frame" and never the pins: the helpers are the only lines
of this file that spell `brt`. A migration closes only when every pin passes unchanged.

Each case returns a vector of numbers. A case marked `:solved` reads a conic solve and compares
to `atol = 5e-5`; a case marked `:exact` reads closed-form algebra and compares to `rtol = 1e-8`.
The pins hold ten significant digits.

Some outputs do not read the frame, and their pins record them as they are: the weights of a
meta head (its `brt` reaches only its own prior and clustering), SPOLC and OLDEM under a vector
benchmark, and the two scores of #1664, which read the benchmark of a prediction directly.

Not pinned: the refusals (no benchmark, a data benchmark under `brt`, a frame mismatch in a
Pipeline), which the decision tickets of the map can change, and a vector benchmark gap under
`x_src = :data`, which a head with no fallback refuses today (#1688).
=#
@testset "The pinned outputs of the Excess Frame (#1679)" begin
    using Test, PortfolioOptimisers, StableRNGs, Dates, LinearAlgebra, Logging, Clarabel
    po = PortfolioOptimisers

    # One panel: six assets driven by two factors, and a benchmark that tracks the first
    # factor. `rdv` carries a vector benchmark, `rdm` a matrix benchmark, one column per asset.
    T, N = 120, 6
    rng = StableRNG(1679)
    F = 0.01 .* randn(rng, T, 2)
    L = 0.5 .+ rand(rng, N, 2)
    X = F * transpose(L) .+ 0.008 .* randn(rng, T, N) .+ 0.0005
    Bv = 0.9 .* F[:, 1] .+ 0.002 .* randn(rng, T)
    Bm = Bv .+ 0.003 .* randn(rng, T, N)
    nx = ["A$i" for i in 1:N]
    nf = ["F1", "F2"]
    ts = Date(2020, 1, 1) .+ Day.(0:(T - 1))
    rdv = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, nb = ["BM"], B = Bv, ts = ts)
    rdm = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, nb = nx .* "_b", B = Bm, ts = ts)
    panels = (("vector B", rdv), ("matrix B", rdm))
    # The gaps of #1672: a gap in a vector benchmark drops the observation, and a gap in a
    # matrix benchmark is a gap in the excess return of its asset.
    Bvg = copy(Bv)
    Bvg[[10, 47]] .= NaN
    rdvg = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, nb = ["BM"], B = Bvg, ts = ts)
    Bmg = copy(Bm)
    Bmg[20, 3] = NaN
    rdmg = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, nb = nx .* "_b", B = Bmg,
                         ts = ts,
                         pnl = AssetPanel(; amsk = trues(T, N), emsk = trues(T, N)))

    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false, "tol_gap_abs" => 1e-10,
                                 "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10))
    rows(r, i) = po.port_opt_view(r, i, :)
    stepall(opt, r) = foldl((o, i) -> partial_fit!(o, rows(r, i:i)), 1:size(r.X, 1);
                            init = opt)
    function stepped(pipe, r, blocks)
        pipe = po.update_online_estimator(pipe)
        for b in blocks
            pipe = partial_fit!(pipe, rows(r, b))
        end
        return pipe
    end
    quiet(f) = with_logger(f, NullLogger())
    finite(v) = filter(isfinite, vec(v))
    sets = UniverseSets(; dict = Dict("nx" => nx, "nf" => nf))
    exposure() = ExposureConstraintEstimator(;
                                             lce = LinearConstraintEstimator(;
                                                                             val = "F1 <= 0.3"),
                                             space = FactorSpace(;
                                                                 re = DimensionReductionRegression()))
    cv = IndexWalkForward(60, 5; purged_size = 2, expand_train = true)
    ocv = OnlineIndexWalkForward(60, 5; purged_size = 2)
    cfe_columns(a) = vcat(finite(a.qlike), finite(a.mahalanobis_ratio), finite(a.frobenius))

    # --- The doors of the Excess Frame: the only lines of this file that spell `brt`. ---
    excess_jump(; kw...) = JuMPOptimiser(; slv = slv, brt = true, kw...)
    excess_hierarchical(; kw...) = HierarchicalOptimiser(; brt = true, kw...)
    excess_inverse_volatility() = InverseVolatility(; brt = true)
    excess_nested(; kw...) = NestedClustered(; brt = true, kw...)
    excess_stacking(; kw...) = Stacking(; brt = true, kw...)
    excess_subsets(; kw...) = SubsetResampling(; brt = true, kw...)
    excess_prior(pe, r) = prior(pe, r; brt = true)
    excess_ucs(uc, r) = ucs(uc, r; brt = true)
    excess_exposure(ece, s, r) = linear_constraints(ece, s; rd = r, brt = true)
    excess_shared_view(r) = po.shared_series_view(r, true, false)
    excess_step(; kw...) = PipelineStep(; brt = true, kw...)
    excess_cfe(est, r, c) = covariance_forecast_evaluation(est, r, c; brt = true,
                                                           store_forecasts = true)
    excess_selection(; kw...) = OnlinePortfolioSelection(; brt = true, kw...)
    excess_risk_loss(; kw...) = RiskLoss(; brt = true, kw...)
    excess_programme_set(; kw...) = ProgrammeAllocationSet(; slv = slv, brt = true, kw...)
    excess_loss_control(; kw...) = ShortTermLossControlPortfolio(; slv = slv, brt = true,
                                                                 kw...)
    excess_ensemble(; kw...) = LowDimensionEnsemblePortfolio(; N = N, slv = slv, brt = true,
                                                             kw...)

    # --- The cases: `(key, tol, f)`, where `f()` gives the vector that the pin holds. ---
    cases = Tuple{String, Symbol, Function}[]
    case!(key, tol, f) = push!(cases, (key, tol, f))
    w(opt, r) = optimise(opt, r).w

    for (b, r) in panels
        # The JuMP heads.
        case!("MeanRisk, $b", :solved, () -> w(MeanRisk(; opt = excess_jump()), r))
        case!("MeanRisk, a return uncertainty set with its own prior, $b", :solved,
              () -> w(MeanRisk(; obj = MaximumRatio(),
                               opt = excess_jump(;
                                                 ret = ArithmeticReturn(;
                                                                        ucs = NormalUncertaintySet(;
                                                                                                   pe = EmpiricalPrior(),
                                                                                                   rng = StableRNG(1651),
                                                                                                   alg = BoxUncertaintySetAlgorithm())))),
                      r))
        case!("MeanRisk, UncertaintySetVariance with its own prior, $b", :solved,
              () -> w(MeanRisk(;
                               r = UncertaintySetVariance(;
                                                          ucs = DeltaUncertaintySet(;
                                                                                    pe = EmpiricalPrior())),
                               opt = excess_jump()), r))
        case!("MeanRisk, an exposure constraint that refits the loadings, $b", :solved,
              () -> w(MeanRisk(; opt = excess_jump(; lcse = exposure(), sets = sets)), r))
        for xs in (:prior, :data)
            case!("MeanRisk, phylogeny and centrality constraints, FactorPrior, x_src = :$xs, $b",
                  :solved,
                  () -> w(MeanRisk(;
                                   opt = excess_jump(; pe = FactorPrior(), x_src = xs,
                                                     ple = SemiDefinitePhylogenyEstimator(),
                                                     cte = CentralityConstraint())), r))
        end
        case!("RiskBudgeting, $b", :solved,
              () -> w(RiskBudgeting(; opt = excess_jump()), r))
        case!("RiskBudgeting, factor risk budgeting, $b", :solved,
              () -> w(RiskBudgeting(;
                                    rba = FactorRiskBudgeting(;
                                                              re = DimensionReductionRegression()),
                                    opt = excess_jump()), r))
        case!("RelaxedRiskBudgeting, $b", :solved,
              () -> w(RelaxedRiskBudgeting(; opt = excess_jump()), r))
        case!("RelaxedRiskBudgeting, factor risk budgeting, $b", :solved,
              () -> w(RelaxedRiskBudgeting(;
                                           rba = FactorRiskBudgeting(;
                                                                     re = DimensionReductionRegression()),
                                           opt = excess_jump()), r))
        case!("FactorRiskContribution, $b", :solved,
              () -> w(FactorRiskContribution(; opt = excess_jump()), r))
        case!("NearOptimalCentering, $b", :solved,
              () -> w(NearOptimalCentering(; opt = excess_jump()), r))
        case!("NearOptimalCentering, UncertaintySetVariance with its own prior, $b",
              :solved,
              () -> w(NearOptimalCentering(;
                                           r = UncertaintySetVariance(;
                                                                      ucs = DeltaUncertaintySet(;
                                                                                                pe = EmpiricalPrior())),
                                           opt = excess_jump()), r))

        # The hierarchical heads. A FactorPrior gives a prior `X` that differs from the
        # data `X - B`, so the two sources differ. The online step folds no FactorPrior, so
        # the online cases below compare to the heads over an EmpiricalPrior.
        for (pn, mkpe, xs) in
            (("FactorPrior", FactorPrior, :prior), ("FactorPrior", FactorPrior, :data),
             ("EmpiricalPrior", EmpiricalPrior, :data))
            ho() = excess_hierarchical(; pe = mkpe(), x_src = xs)
            case!("HierarchicalRiskParity, $pn, x_src = :$xs, $b", :exact,
                  () -> w(HierarchicalRiskParity(; opt = ho()), r))
            case!("SchurComplementHierarchicalRiskParity, $pn, x_src = :$xs, $b", :exact,
                  () -> w(SchurComplementHierarchicalRiskParity(; opt = ho(),
                                                                params = SchurComplementParams(;
                                                                                               gamma = 0.5)),
                          r))
            case!("HierarchicalEqualRiskContribution, $pn, x_src = :$xs, $b", :exact,
                  () -> w(HierarchicalEqualRiskContribution(; opt = ho()), r))
        end
        case!("InverseVolatility, $b", :exact, () -> w(excess_inverse_volatility(), r))

        # The meta heads. Their own `brt` changes their own prior and clustering, and no
        # inner or outer optimiser, so the prior and the clustering are pinned beside the
        # weights.
        inner() = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        nco() = excess_nested(; opti = inner(), opto = inner())
        case!("NestedClustered, the weights, $b", :solved, () -> w(nco(), r))
        case!("NestedClustered, the prior and the distances, $b", :exact,
              () -> (res = optimise(nco(), r); vcat(res.pr.mu, vec(res.clr.D))))
        case!("NestedClustered, inner and outer in the Excess Frame, $b", :solved,
              () -> w(excess_nested(; opti = MeanRisk(; opt = excess_jump()),
                                    opto = MeanRisk(; opt = excess_jump())), r))
        sto() = excess_stacking(; opti = [inner(), InverseVolatility()], opto = inner())
        case!("Stacking, the weights, $b", :solved, () -> w(sto(), r))
        case!("Stacking, the prior, $b", :exact, () -> optimise(sto(), r).pr.mu)
        sro() = excess_subsets(; opt = inner(), seed = 3, n_subsets = 3, subset_size = 4)
        case!("SubsetResampling, the weights, $b", :solved, () -> w(sro(), r))
        case!("SubsetResampling, the prior, $b", :exact, () -> optimise(sro(), r).pr.mu)
        case!("SubsetResampling, inner in the Excess Frame, $b", :solved,
              () -> w(SubsetResampling(; opt = MeanRisk(; opt = excess_jump()), seed = 3,
                                       n_subsets = 3, subset_size = 4), r))

        # The verbs.
        case!("prior, EmpiricalPrior, $b", :exact,
              () -> (p = excess_prior(EmpiricalPrior(), r); vcat(p.mu, vec(p.sigma))))
        case!("prior, FactorPrior, $b", :exact,
              () -> (p = excess_prior(FactorPrior(), r); vcat(p.mu, vec(p.sigma))))
        case!("ucs, NormalUncertaintySet box, $b", :exact,
              () -> (u = excess_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(),
                                                         rng = StableRNG(1679),
                                                         alg = BoxUncertaintySetAlgorithm()),
                                    r); vcat(u[1].lb, u[1].ub, vec(u[2].lb), vec(u[2].ub))))
        case!("linear_constraints, an exposure that refits the loadings, $b", :exact,
              () -> vec(excess_exposure(exposure(), sets, r).ineq.A))
        case!("covariance_forecast_evaluation, Covariance, batch, $b", :exact,
              () -> cfe_columns(excess_cfe(Covariance(), r, cv)))
        case!("covariance_forecast_evaluation, Covariance, online, $b", :exact,
              () -> cfe_columns(excess_cfe(Covariance(), r, ocv)))
        case!("covariance_forecast_evaluation, Online(Covariance()), online, $b", :exact,
              () -> cfe_columns(excess_cfe(po.Online(Covariance()), r, ocv)))
        case!("covariance_forecast_portfolio of an evaluation in the Excess Frame, $b",
              :exact,
              () -> (p = covariance_forecast_portfolio(excess_cfe(Covariance(), r, cv), r,
                                                       [fill(1 / N, N)]);
                     vcat(finite(p.standardised_return), finite(p.portfolio_qlike))))

        # The Pipeline steps that read the returns.
        pstep() = excess_step(; est = EmpiricalPrior(), writes = :prior)
        jop(; kw...) = excess_jump(; pe = EmpiricalPrior(), kw...)
        pipe(steps...) = Pipeline(; steps = (pstep(), steps...))
        case!("Pipeline, a prior step, $b", :solved,
              () -> fit(pipe(MeanRisk(; opt = jop())), r).w)
        case!("Pipeline, an uncertainty step, $b", :solved,
              () -> fit(pipe(excess_step(; est = DeltaUncertaintySet(), reads = (:returns,),
                                         writes = :uncertainty, target = :sigma),
                             MeanRisk(; r = UncertaintySetVariance(), opt = jop())), r).w)
        case!("Pipeline, a clustering step, $b", :exact,
              () -> fit(pipe(excess_step(; est = ClustersEstimator(), writes = :phylogeny),
                             HierarchicalRiskParity(;
                                                    opt = excess_hierarchical(;
                                                                              x_src = :data))),
                        r).w)
        case!("Pipeline, a network step, $b", :exact,
              () -> Float64.(vec(fit(pipe(excess_step(; est = NetworkEstimator(),
                                                      writes = :phylogeny),
                                          MeanRisk(; opt = jop())), r).ctx.phylogeny.X)))
        case!("Pipeline, a phylogeny constraint step, $b", :solved,
              () -> fit(pipe(excess_step(; est = SemiDefinitePhylogenyEstimator(),
                                         writes = :constraints),
                             MeanRisk(; opt = jop(; x_src = :data))), r).w)
        case!("Pipeline, a centrality constraint step, $b", :solved,
              () -> fit(pipe(excess_step(; est = CentralityConstraint(),
                                         writes = :constraints),
                             MeanRisk(; opt = jop(; x_src = :data))), r).w)
        case!("Pipeline, an exposure step, $b", :solved,
              () -> fit(pipe(excess_step(; est = exposure(), writes = :constraints),
                             MeanRisk(; opt = jop())), r).w)
        case!("Pipeline, a prior step, online, $b", :solved,
              () -> fit(stepped(pipe(MeanRisk(; opt = jop())), r,
                                [1:60, 61:61, 62:100, 101:T])).w)

        # The online step of each head that holds a prior equals its batch fit, so each
        # online case compares to the pin of its batch case.
        for (key, opt) in (("MeanRisk", MeanRisk(; opt = excess_jump())),
                           ("RiskBudgeting", RiskBudgeting(; opt = excess_jump())),
                           ("RelaxedRiskBudgeting", RelaxedRiskBudgeting(; opt = excess_jump())),
                           ("NearOptimalCentering", NearOptimalCentering(; opt = excess_jump())),
                           ("HierarchicalRiskParity, EmpiricalPrior, x_src = :data",
                            HierarchicalRiskParity(; opt = excess_hierarchical(; x_src = :data))),
                           ("SchurComplementHierarchicalRiskParity, EmpiricalPrior, x_src = :data",
                            SchurComplementHierarchicalRiskParity(;
                                                                  opt = excess_hierarchical(; x_src = :data),
                                                                  params = SchurComplementParams(;
                                                                                                 gamma = 0.5))),
                           ("HierarchicalEqualRiskContribution, EmpiricalPrior, x_src = :data",
                            HierarchicalEqualRiskContribution(; opt = excess_hierarchical(; x_src = :data))),
                           ("InverseVolatility", excess_inverse_volatility()),
                           ("NestedClustered, the weights", nco()), ("Stacking, the weights", sto()),
                           ("SubsetResampling, the weights", sro()))
            tol = if occursin("Hierarchical", key) || key == "InverseVolatility"
                :exact
            else
                :solved
            end
            case!("online: $key, $b", tol, () -> optimise(stepall(opt, r)).w)
        end

        # The online selection head (#1662) and the readers of its rows (#1663).
        sel(alg; kw...) = quiet(() -> optimise(excess_selection(; alg = alg, kw...), r).w)
        for (name, alg) in (("ExponentiatedGradient", ExponentiatedGradient()),
                            ("PassiveAggressiveMeanReversion", PassiveAggressiveMeanReversion(; eps = 1.0)),
                            ("ForecastReversion", ForecastReversion(; eps = 1.002)),
                            ("ForecastReversion, MovingAverage scale",
                             ForecastReversion(; eps = 1.002, scale = MovingAverage())),
                            ("AntiCorrelation", AntiCorrelation(; window = 5)))
            case!("OnlinePortfolioSelection, $name, $b", :exact, () -> sel(alg))
        end
        case!("online: OnlinePortfolioSelection, ExponentiatedGradient, $b", :exact,
              () -> quiet(() -> optimise(stepall(excess_selection(;
                                                                  alg = ExponentiatedGradient()),
                                                 r)).w))
        r24 = rows(r, 1:24)
        lead() = MeanRisk(; opt = excess_jump(; pe = EmpiricalPrior()))
        case!("FollowTheLeader, a CorrelationMatch selector, $b", :solved,
              () -> quiet(() -> optimise(excess_selection(;
                                                          alg = FollowTheLeader(;
                                                                                sel = CorrelationMatch())),
                                         r24).w))
        case!("FollowTheLeader, a leader in the Excess Frame, $b", :solved,
              () -> quiet(() -> optimise(OnlinePortfolioSelection(;
                                                                  alg = FollowTheLeader(;
                                                                                        opt = lead())),
                                         r24).w))
        case!("ShortTermLossControlPortfolio, $b", :solved,
              () -> quiet(() -> optimise(OnlinePortfolioSelection(;
                                                                  alg = excess_loss_control(;
                                                                                            gamma = 5.0)),
                                         r24).w))
        case!("LowDimensionEnsemblePortfolio, $b", :solved,
              () -> quiet(() -> optimise(OnlinePortfolioSelection(;
                                                                  alg = excess_ensemble()),
                                         r24).w))
        md() = MirrorDescent(; eta = 50.0, obj = excess_risk_loss(; window = 8))
        case!("MirrorDescent, a RiskLoss in the Excess Frame, $b", :exact,
              () -> quiet(() -> optimise(OnlinePortfolioSelection(; alg = md()), r).w))
        case!("MirrorDescent, a RiskLoss and the head in the Excess Frame, $b", :exact,
              () -> sel(md()))
        vset() = excess_programme_set(;
                                      r = Variance(;
                                                   settings = RiskMeasureSettings(;
                                                                                  ub = 2e-5)))
        case!("ProgrammeAllocationSet, a Variance ceiling, $b", :solved,
              () -> quiet(() -> optimise(OnlinePortfolioSelection(;
                                                                  alg = PassiveAggressiveMeanReversion(),
                                                                  set = vset()), r).w))
    end

    # The cross-sectional factor prior reads Descriptors, so it fits on the synthetic Asset
    # Panel of `test06c_setup.jl`. Its `mu` is zero, so the pin reads `sigma`.
    include(joinpath(@__DIR__, "test06c_setup.jl"))
    rc = synthetic_asset_panel(; n_assets = 30, n_observations = 150, n_industries = 3,
                               rng = StableRNG(1679)).rd
    Tc, Nc = size(rc.X)
    rngc = StableRNG(16790)
    Bcv = 0.01 .* randn(rngc, Tc)
    Bcm = 0.01 .* randn(rngc, Tc, Nc)
    cse() = CrossSectionalFactorPrior(;
                                      factors = ["market" => ConstantExposure(),
                                                 "industry" =>
                                                     OneHotExposure(; field = "industry",
                                                                    family = "industry"),
                                                 "size" => CompositeExposure(;
                                                                             descriptors = [LogMarketCap()],
                                                                             family = "style"),
                                                 "value" => CompositeExposure(;
                                                                              descriptors = [BookToPrice()],
                                                                              family = "style")],
                                      minra = 10)
    for (b, r) in (("vector B",
                    ReturnsResult(; nx = rc.nx, X = rc.X, ts = rc.ts, pnl = rc.pnl, nb = ["BM"], B = Bcv)),
                   ("matrix B",
                    ReturnsResult(; nx = rc.nx, X = rc.X, ts = rc.ts, pnl = rc.pnl,
                                  nb = rc.nx .* "_b", B = Bcm)))
        case!("prior, CrossSectionalFactorPrior, $b", :exact,
              () -> (p = excess_prior(cse(), r);
                     vcat(finite(diag(p.sigma)), finite(p.sigma[:, 1]))))
    end

    # The gaps of #1672.
    case!("a vector benchmark gap, prior, EmpiricalPrior", :exact,
          () -> (p = excess_prior(EmpiricalPrior(), rdvg); vcat(p.mu, vec(p.sigma))))
    case!("a vector benchmark gap, MeanRisk", :solved,
          () -> w(MeanRisk(; opt = excess_jump()), rdvg))
    case!("a vector benchmark gap, InverseVolatility", :exact,
          () -> w(excess_inverse_volatility(), rdvg))
    # Under `x_src = :data` a head with no fallback refuses the gap rows today (#1688), so
    # the hierarchical gap case reads the prior.
    case!("a vector benchmark gap, HierarchicalRiskParity, FactorPrior", :exact,
          () -> w(HierarchicalRiskParity(;
                                         opt = excess_hierarchical(; pe = FactorPrior(),
                                                                   x_src = :prior)), rdvg))
    case!("a vector benchmark gap, covariance_forecast_evaluation", :exact,
          () -> cfe_columns(excess_cfe(Covariance(), rdvg, cv)))
    case!("a matrix benchmark gap, prior, EmpiricalPrior", :exact,
          () -> (p = excess_prior(EmpiricalPrior(), rdmg);
                 vcat(finite(p.mu), finite(p.sigma))))
    case!("a matrix benchmark gap, the estimation mask", :exact,
          () -> Float64.(vec(excess_shared_view(rdmg).pnl.emsk)))
    case!("a matrix benchmark gap, MeanRisk", :solved,
          () -> w(MeanRisk(; opt = excess_jump()), rdmg))

    # The scores of #1664 read the benchmark of a prediction directly, with no `brt`.
    pred = cross_val_predict(MeanRisk(; opt = JuMPOptimiser(; slv = slv)), rdv,
                             IndexWalkForward(60, 10))
    case!("log_wealth_regret against the benchmark of the prediction", :solved,
          () -> (g = log_wealth_regret(pred); vcat(g.regret, finite(g.cumulative))))
    case!("performance_summary against the benchmark of the prediction", :solved,
          () -> (s = performance_summary(pred);
                 [s.excess_ret, s.tracking_error, s.information_ratio]))

    # The pins, recorded on `dev` at 201f29aa71 with today's `brt`. Change no value here.
    pins = Dict{String, Vector{Float64}}("MeanRisk, vector B" =>
                                             [0.3287372866, 3.319536659e-10,
                                              3.912052867e-10, 0.04058699229, 0.3935155297,
                                              0.2371601907],
                                         "MeanRisk, a return uncertainty set with its own prior, vector B" =>
                                             [0.9999996823, 1.099491482e-8, 2.949772993e-8,
                                              4.784463977e-8, 5.003597039e-8,
                                              1.793049332e-7],
                                         "MeanRisk, UncertaintySetVariance with its own prior, vector B" =>
                                             [0.3287372499, 5.600148667e-9, 7.269424054e-9,
                                              0.04058643413, 0.3935139877, 0.2371623153],
                                         "MeanRisk, an exposure constraint that refits the loadings, vector B" =>
                                             [0.3287357135, 1.133894535e-9, 1.343111109e-9,
                                              0.04059275998, 0.3934997946, 0.2371717295],
                                         "MeanRisk, phylogeny and centrality constraints, FactorPrior, x_src = :prior, vector B" =>
                                             [1.838042377e-13, 7.734177406e-14,
                                              7.741382821000001e-14, 1.833570331e-13,
                                              0.5423951185, 0.4576048815],
                                         "MeanRisk, phylogeny and centrality constraints, FactorPrior, x_src = :data, vector B" =>
                                             [0.334377101, 2.39071586e-13, 2.389104104e-13,
                                              0.020183911, 0.4118975431, 0.2335414448],
                                         "RiskBudgeting, vector B" =>
                                             [0.1959464752, 0.1277776521, 0.1298987312,
                                              0.1644391147, 0.1933093917, 0.1886286352],
                                         "RiskBudgeting, factor risk budgeting, vector B" =>
                                             [0.06957160774, 0.1474773023, 0.1756225195,
                                              0.2397053405, 0.218803977, 0.1488192529],
                                         "RelaxedRiskBudgeting, vector B" =>
                                             [0.1959458882, 0.1277790137, 0.1298976814,
                                              0.1644440384, 0.1933059206, 0.1886274578],
                                         "RelaxedRiskBudgeting, factor risk budgeting, vector B" =>
                                             [0.08325446812, 0.1723760066, 0.2076171989,
                                              0.2200300939, 0.1962010749, 0.1205211576],
                                         "FactorRiskContribution, vector B" =>
                                             [0.207583186, 0.2474113979, 0.1345040755,
                                              0.08680065616, 0.05415810581, 0.2695425786],
                                         "NearOptimalCentering, vector B" =>
                                             [0.3404287974, 0.005515696667, 0.009179416698,
                                              0.07632233989, 0.3229231792, 0.2456305702],
                                         "NearOptimalCentering, UncertaintySetVariance with its own prior, vector B" =>
                                             [0.3408808025, 0.005890788688, 0.009932516269,
                                              0.0780622264, 0.3191127832, 0.2461208829],
                                         "HierarchicalRiskParity, FactorPrior, x_src = :prior, vector B" =>
                                             [0.1699538838, 0.09220594111, 0.1593791025,
                                              0.1397965844, 0.179787456, 0.2588770321],
                                         "SchurComplementHierarchicalRiskParity, FactorPrior, x_src = :prior, vector B" =>
                                             [0.1699538838, 0.09220594111, 0.1593791025,
                                              0.1397965844, 0.179787456, 0.2588770321],
                                         "HierarchicalEqualRiskContribution, FactorPrior, x_src = :prior, vector B" =>
                                             [0.2113372863, 0.114657888, 0.1087168431,
                                              0.1619579252, 0.2082883747, 0.1950416826],
                                         "HierarchicalRiskParity, FactorPrior, x_src = :data, vector B" =>
                                             [0.1699538838, 0.09220594111, 0.08854242411,
                                              0.2207841007, 0.1696366182, 0.2588770321],
                                         "SchurComplementHierarchicalRiskParity, FactorPrior, x_src = :data, vector B" =>
                                             [0.1699538838, 0.09220594111, 0.08854242411,
                                              0.2207841007, 0.1696366182, 0.2588770321],
                                         "HierarchicalEqualRiskContribution, FactorPrior, x_src = :data, vector B" =>
                                             [0.287832233, 0.08513160552, 0.08203143042,
                                              0.122204066, 0.1571623386, 0.2656383264],
                                         "HierarchicalRiskParity, EmpiricalPrior, x_src = :data, vector B" =>
                                             [0.1684002444, 0.09136303729, 0.0876927059,
                                              0.2216548237, 0.1680086604, 0.2628805283],
                                         "SchurComplementHierarchicalRiskParity, EmpiricalPrior, x_src = :data, vector B" =>
                                             [0.1684002444, 0.09136303729, 0.0876927059,
                                              0.2216548237, 0.1680086604, 0.2628805283],
                                         "HierarchicalEqualRiskContribution, EmpiricalPrior, x_src = :data, vector B" =>
                                             [0.2904242075, 0.08418138032, 0.08111580888,
                                              0.120840044, 0.155408118, 0.2680304413],
                                         "InverseVolatility, vector B" =>
                                             [0.1885526161, 0.1388821618, 0.1363299258,
                                              0.1663963839, 0.1887014761, 0.1811374362],
                                         "NestedClustered, the weights, vector B" =>
                                             [0.03140174028, 3.729550388e-10,
                                              6.451774831e-10, 0.2469824813, 0.7094828767,
                                              0.01213290077],
                                         "NestedClustered, the prior and the distances, vector B" =>
                                             [0.001012201381, -0.0006027746974,
                                              0.0008963215387, 0.0007380867656,
                                              0.000449171902, 0.0009682674454, 0.0,
                                              0.3714541746, 0.4481212291, 0.4761601685,
                                              0.4949376915, 0.4118176029, 0.3714541746, 0.0,
                                              0.3499047219, 0.3649244423, 0.4044660319,
                                              0.3794761579, 0.4481212291, 0.3499047219, 0.0,
                                              0.3486434816, 0.364792154, 0.4486413792,
                                              0.4761601685, 0.3649244423, 0.3486434816, 0.0,
                                              0.3934091267, 0.4736642662, 0.4949376915,
                                              0.4044660319, 0.364792154, 0.3934091267, 0.0,
                                              0.4958007842, 0.4118176029, 0.3794761579,
                                              0.4486413792, 0.4736642662, 0.4958007842,
                                              0.0],
                                         "NestedClustered, inner and outer in the Excess Frame, vector B" =>
                                             [0.3081876531, 1.986387645e-9, 9.74309012e-10,
                                              0.1352065919, 0.3135005518, 0.2431052003],
                                         "Stacking, the weights, vector B" =>
                                             [0.08404551048, 1.559091316e-8, 1.748746997e-8,
                                              0.2200745503, 0.6958798875, 1.860813468e-8],
                                         "Stacking, the prior, vector B" =>
                                             [0.001012201381, -0.0006027746974,
                                              0.0008963215387, 0.0007380867656,
                                              0.000449171902, 0.0009682674454],
                                         "SubsetResampling, the weights, vector B" =>
                                             [0.1049182668, 1.523658768e-9, 4.811154957e-9,
                                              0.08609119214, 0.8089905283, 6.440907131e-9],
                                         "SubsetResampling, the prior, vector B" =>
                                             [0.001012201381, -0.0006027746974,
                                              0.0008963215387, 0.0007380867656,
                                              0.000449171902, 0.0009682674454],
                                         "SubsetResampling, inner in the Excess Frame, vector B" =>
                                             [0.2786504914, 2.09134823e-9, 3.547118102e-9,
                                              0.03380476557, 0.4640577631, 0.2234869742],
                                         "prior, EmpiricalPrior, vector B" =>
                                             [0.001012201381, -0.0006027746974,
                                              0.0008963215387, 0.0007380867656,
                                              0.000449171902, 0.0009682674454,
                                              0.0001714052204, 0.0001684902807,
                                              0.0001418530415, 0.0001061541678,
                                              8.736026735e-5, 0.0001179034887,
                                              0.0001684902807, 0.0003159339036,
                                              0.0002430385506, 0.0001934611826,
                                              0.0001564453718, 0.0001724693198,
                                              0.0001418530415, 0.0002430385506,
                                              0.0003278738444, 0.0002033246246,
                                              0.0001738328963, 0.0001474298345,
                                              0.0001061541678, 0.0001934611826,
                                              0.0002033246246, 0.0002200905529,
                                              0.0001340008583, 0.0001114583767,
                                              8.736026735e-5, 0.0001564453718,
                                              0.0001738328963, 0.0001340008583,
                                              0.0001711348959, 9.063161741e-5,
                                              0.0001179034887, 0.0001724693198,
                                              0.0001474298345, 0.0001114583767,
                                              9.063161741e-5, 0.0001857260133],
                                         "prior, FactorPrior, vector B" =>
                                             [0.001012201381, -0.0006027746974,
                                              0.0008963215387, 0.0007380867656,
                                              0.000449171902, 0.0009682674454,
                                              0.0001714052204, 0.0001585029666,
                                              0.0001411526633, 0.0001096202337,
                                              8.859166015e-5, 0.0001217021886,
                                              0.0001585029666, 0.0003159339036,
                                              0.0002392425273, 0.0001890696811,
                                              0.0001557638247, 0.000177814897,
                                              0.0001411526633, 0.0002392425273,
                                              0.0003278738444, 0.0001999377179,
                                              0.0001681232111, 0.000151153608,
                                              0.0001096202337, 0.0001890696811,
                                              0.0001999377179, 0.0002200905529,
                                              0.0001366160846, 0.0001159064398,
                                              8.859166015e-5, 0.0001557638247,
                                              0.0001681232111, 0.0001366160846,
                                              0.0001711348959, 9.233097416e-5,
                                              0.0001217021886, 0.000177814897,
                                              0.000151153608, 0.0001159064398,
                                              9.233097416e-5, 0.0001857260133],
                                         "ucs, NormalUncertaintySet box, vector B" =>
                                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004684890948,
                                              0.006360416864, 0.00647949039, 0.005308699765,
                                              0.004681195201, 0.004876675206,
                                              0.0001311493529, 0.000119828776,
                                              9.528313176e-5, 6.88835762e-5, 5.542149653e-5,
                                              8.207670019e-5, 0.000119828776,
                                              0.0002386999066, 0.0001741633227,
                                              0.0001372321404, 0.0001080547184,
                                              0.0001216602788, 9.528313176e-5,
                                              0.0001741633227, 0.0002503365929,
                                              0.000145697764, 0.0001224312482,
                                              9.799999231e-5, 6.88835762e-5,
                                              0.0001372321404, 0.000145697764,
                                              0.0001668566111, 9.350945154e-5,
                                              7.103225859e-5, 5.542149653e-5,
                                              0.0001080547184, 0.0001224312482,
                                              9.350945154e-5, 0.0001304864305,
                                              5.552763059e-5, 8.207670019e-5,
                                              0.0001216602788, 9.799999231e-5,
                                              7.103225859e-5, 5.552763059e-5,
                                              0.0001411695787, 0.0002167454371,
                                              0.0002208803595, 0.0001928957947,
                                              0.0001475098043, 0.0001227764351,
                                              0.0001579006167, 0.0002208803595,
                                              0.0004005540081, 0.0003197213357,
                                              0.0002555187425, 0.000209081387,
                                              0.0002285285642, 0.0001928957947,
                                              0.0003197213357, 0.0004124955696,
                                              0.0002676573916, 0.0002297279228,
                                              0.0001992570069, 0.0001475098043,
                                              0.0002555187425, 0.0002676573916,
                                              0.0002783946706, 0.000176307083,
                                              0.0001549629763, 0.0001227764351,
                                              0.000209081387, 0.0002297279228,
                                              0.000176307083, 0.0002160076768,
                                              0.0001274363902, 0.0001579006167,
                                              0.0002285285642, 0.0001992570069,
                                              0.0001549629763, 0.0001274363902,
                                              0.000237779075],
                                         "linear_constraints, an exposure that refits the loadings, vector B" =>
                                             [0.412051131, 0.2899077671, -0.1770035364,
                                              -0.2269887201, -0.2645304741, 0.6260823645],
                                         "covariance_forecast_evaluation, Covariance, batch, vector B" =>
                                             [-242.3856323, -250.5891304, -243.2843965,
                                              -221.9677536, -250.0551611, -217.9555594,
                                              -255.1452389, -228.796259, -252.9630087,
                                              -238.4838398, -237.4462915, -231.9723828,
                                              1.122803095, 0.8512345208, 1.097989414,
                                              1.819271816, 0.8657051469, 1.914474714,
                                              0.6833587633, 1.528782496, 0.7296575335,
                                              1.199200021, 1.243192708, 1.410490803,
                                              2.899609515e-7, 2.478684035e-7,
                                              5.638036469e-7, 2.207347325e-6,
                                              9.413484715e-7, 2.644732967e-7,
                                              3.912425715e-7, 3.610684988e-7,
                                              2.402723625e-7, 2.615168517e-7,
                                              1.276454988e-7, 3.836676135e-7],
                                         "covariance_forecast_evaluation, Covariance, online, vector B" =>
                                             [-242.3856323, -250.5891304, -243.2843965,
                                              -221.9677536, -250.0551611, -217.9555594,
                                              -255.1452389, -228.796259, -252.9630087,
                                              -238.4838398, -237.4462915, -231.9723828,
                                              1.122803095, 0.8512345208, 1.097989414,
                                              1.819271816, 0.8657051469, 1.914474714,
                                              0.6833587633, 1.528782496, 0.7296575335,
                                              1.199200021, 1.243192708, 1.410490803,
                                              2.899609515e-7, 2.478684035e-7,
                                              5.638036469e-7, 2.207347325e-6,
                                              9.413484715e-7, 2.644732967e-7,
                                              3.912425715e-7, 3.610684988e-7,
                                              2.402723625e-7, 2.615168517e-7,
                                              1.276454988e-7, 3.836676135e-7],
                                         "covariance_forecast_evaluation, Online(Covariance()), online, vector B" =>
                                             [-242.3856323, -250.5891304, -243.2843965,
                                              -221.9677536, -250.0551611, -217.9555594,
                                              -255.1452389, -228.796259, -252.9630087,
                                              -238.4838398, -237.4462915, -231.9723828,
                                              1.122803095, 0.8512345208, 1.097989414,
                                              1.819271816, 0.8657051469, 1.914474714,
                                              0.6833587633, 1.528782496, 0.7296575335,
                                              1.199200021, 1.243192708, 1.410490803,
                                              2.899609515e-7, 2.478684035e-7,
                                              5.638036469e-7, 2.207347325e-6,
                                              9.413484715e-7, 2.644732967e-7,
                                              3.912425715e-7, 3.610684988e-7,
                                              2.402723625e-7, 2.615168517e-7,
                                              1.276454988e-7, 3.836676135e-7],
                                         "covariance_forecast_portfolio of an evaluation in the Excess Frame, vector B" =>
                                             [0.07457043807, -0.2621665358, -0.2581053294,
                                              -0.7542242034, -0.07183801109, 0.735929701,
                                              0.48909526, -0.9552440579, -0.2234693869,
                                              0.1556560285, -0.7072268619, -1.081514998,
                                              -6.291675368, -6.270710631, -6.679468718,
                                              -4.853414646, -6.945833541, -6.368371812,
                                              -6.582564948, -6.15827643, -5.769517553,
                                              -6.546483638, -6.113157784, -5.734214057],
                                         "Pipeline, a prior step, vector B" =>
                                             [0.3287372866, 3.319536659e-10,
                                              3.912052867e-10, 0.04058699229, 0.3935155297,
                                              0.2371601907],
                                         "Pipeline, an uncertainty step, vector B" =>
                                             [0.3287372499, 5.600148667e-9, 7.269424054e-9,
                                              0.04058643413, 0.3935139877, 0.2371623153],
                                         "Pipeline, a clustering step, vector B" =>
                                             [0.1684002444, 0.09136303729, 0.0876927059,
                                              0.2216548237, 0.1680086604, 0.2628805283],
                                         "Pipeline, a network step, vector B" =>
                                             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
                                              0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
                                              0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                              0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                         "Pipeline, a phylogeny constraint step, vector B" =>
                                             [0.3287577898, 4.551182741e-8, 5.854248923e-8,
                                              0.04069290395, 0.3933940985, 0.2371551038],
                                         "Pipeline, a centrality constraint step, vector B" =>
                                             [0.3287427428, 1.334304035e-13,
                                              1.332526795e-13, 0.04055003461, 0.393534464,
                                              0.2371727586],
                                         "Pipeline, an exposure step, vector B" =>
                                             [0.3287357135, 1.133894535e-9, 1.343111109e-9,
                                              0.04059275998, 0.3934997946, 0.2371717295],
                                         "Pipeline, a prior step, online, vector B" =>
                                             [0.3287372883, 3.308639341e-10,
                                              3.899304665e-10, 0.04058698885, 0.3935155333,
                                              0.2371601888],
                                         "OnlinePortfolioSelection, ExponentiatedGradient, vector B" =>
                                             [0.1671326498, 0.1654466298, 0.166943002,
                                              0.1668290094, 0.1665660035, 0.1670827054],
                                         "OnlinePortfolioSelection, PassiveAggressiveMeanReversion, vector B" =>
                                             [0.07186765987, 0.0, 0.0, 0.2644542128,
                                              0.1324693378, 0.5312087895],
                                         "OnlinePortfolioSelection, ForecastReversion, vector B" =>
                                             [0.01534199048, 0.0, 0.0, 0.2771929626,
                                              0.07466325771, 0.6328017893],
                                         "OnlinePortfolioSelection, ForecastReversion, MovingAverage scale, vector B" =>
                                             [0.01755234355, 0.0, 0.0, 0.2765527286,
                                              0.07276420206, 0.6331307258],
                                         "OnlinePortfolioSelection, AntiCorrelation, vector B" =>
                                             [0.0, 0.0, 0.0, 0.0, 0.528279352, 0.471720648],
                                         "FollowTheLeader, a CorrelationMatch selector, vector B" =>
                                             [0.9999999999, 1.116755609e-22,
                                              1.138549487e-10, 6.029418839e-27,
                                              4.785989272e-20, 3.2532975590000005e-53],
                                         "FollowTheLeader, a leader in the Excess Frame, vector B" =>
                                             [0.4684829236, 8.219640665e-10,
                                              9.163308156e-10, 3.289614068e-9, 0.4063345637,
                                              0.1251825077],
                                         "ShortTermLossControlPortfolio, vector B" =>
                                             [0.9999999988, 3.846234644e-10, 8.29036088e-11,
                                              1.172809475e-10, 5.591127309e-10,
                                              8.543748909e-11],
                                         "LowDimensionEnsemblePortfolio, vector B" =>
                                             [-1.942891899e-12, 1.364400359e-12,
                                              8.495370302e-12, 2.389569142e-12,
                                              1.776080645e-11, 1.0],
                                         "MirrorDescent, a RiskLoss in the Excess Frame, vector B" =>
                                             [0.2149310043, 0.08735320804, 0.0978208309,
                                              0.1623066459, 0.223089752, 0.2144985588],
                                         "MirrorDescent, a RiskLoss and the head in the Excess Frame, vector B" =>
                                             [0.2149310043, 0.08735320804, 0.0978208309,
                                              0.1623066459, 0.223089752, 0.2144985588],
                                         "ProgrammeAllocationSet, a Variance ceiling, vector B" =>
                                             [2.094365003e-13, 2.059264098e-13,
                                              0.2265169425, 3.782446104e-13, 0.7734830575,
                                              8.007670099e-13],
                                         "MeanRisk, matrix B" =>
                                             [0.3454040729, 4.36098608e-9, 5.205926575e-9,
                                              0.05423798722, 0.3819628689, 0.2183950614],
                                         "MeanRisk, a return uncertainty set with its own prior, matrix B" =>
                                             [1.473966722e-5, 1.865807594e-7,
                                              5.782473249e-7, 0.9999825572, 1.197101656e-6,
                                              7.411978286e-7],
                                         "MeanRisk, UncertaintySetVariance with its own prior, matrix B" =>
                                             [0.3454140657, 7.693915524e-9, 1.110989512e-8,
                                              0.05423886655, 0.3819702837, 0.2183767653],
                                         "MeanRisk, an exposure constraint that refits the loadings, matrix B" =>
                                             [0.3454038789, 3.743064965e-9, 4.407746939e-9,
                                              0.05423856425, 0.3819589897, 0.218398559],
                                         "MeanRisk, phylogeny and centrality constraints, FactorPrior, x_src = :prior, matrix B" =>
                                             [3.811507371e-13, 1.932730863e-13,
                                              1.929568449e-13, 3.71625318e-13, 0.5552267313,
                                              0.4447732687],
                                         "MeanRisk, phylogeny and centrality constraints, FactorPrior, x_src = :data, matrix B" =>
                                             [0.3471449262, 5.893397008e-13,
                                              5.900951669e-13, 0.05488952615, 0.3866046928,
                                              0.2113608548],
                                         "RiskBudgeting, matrix B" =>
                                             [0.1964362107, 0.1301073926, 0.1318659302,
                                              0.1629432129, 0.1927954613, 0.1858517923],
                                         "RiskBudgeting, factor risk budgeting, matrix B" =>
                                             [0.06580066183, 0.1315481018, 0.2060227752,
                                              0.2484351396, 0.2052415496, 0.1429517721],
                                         "RelaxedRiskBudgeting, matrix B" =>
                                             [0.1964380406, 0.1301061835, 0.1318628487,
                                              0.1629463522, 0.1927909102, 0.1858556649],
                                         "RelaxedRiskBudgeting, factor risk budgeting, matrix B" =>
                                             [0.08013971927, 0.1630202704, 0.2254261403,
                                              0.2273539126, 0.1898478341, 0.1142121233],
                                         "FactorRiskContribution, matrix B" =>
                                             [0.188243276, 0.22463333, 0.1684680627,
                                              0.09297874526, 0.06444192237, 0.2612346637],
                                         "NearOptimalCentering, matrix B" =>
                                             [0.3384393752, 0.01231026699, 0.02060860669,
                                              0.1249921958, 0.3145278403, 0.189121715],
                                         "NearOptimalCentering, UncertaintySetVariance with its own prior, matrix B" =>
                                             [0.3368559707, 0.0139104018, 0.02376142842,
                                              0.131429248, 0.3071888937, 0.1868540574],
                                         "HierarchicalRiskParity, FactorPrior, x_src = :prior, matrix B" =>
                                             [0.1739015918, 0.09997130252, 0.1614113034,
                                              0.1401897388, 0.1780843862, 0.2464416773],
                                         "SchurComplementHierarchicalRiskParity, FactorPrior, x_src = :prior, matrix B" =>
                                             [0.1739015918, 0.09997130252, 0.1614113034,
                                              0.1401897388, 0.1780843862, 0.2464416773],
                                         "HierarchicalEqualRiskContribution, FactorPrior, x_src = :prior, matrix B" =>
                                             [0.2894013005, 0.09099431548, 0.08544624156,
                                              0.1230243546, 0.156279032, 0.2548547559],
                                         "HierarchicalRiskParity, FactorPrior, x_src = :data, matrix B" =>
                                             [0.1739015918, 0.09997130252, 0.1614113034,
                                              0.1401897388, 0.1780843862, 0.2464416773],
                                         "SchurComplementHierarchicalRiskParity, FactorPrior, x_src = :data, matrix B" =>
                                             [0.1739015918, 0.09997130252, 0.1614113034,
                                              0.1401897388, 0.1780843862, 0.2464416773],
                                         "HierarchicalEqualRiskContribution, FactorPrior, x_src = :data, matrix B" =>
                                             [0.2894013005, 0.09099431548, 0.08544624156,
                                              0.1230243546, 0.156279032, 0.2548547559],
                                         "HierarchicalRiskParity, EmpiricalPrior, x_src = :data, matrix B" =>
                                             [0.1716304928, 0.09866570937, 0.1604037817,
                                              0.1397897381, 0.1775762614, 0.2519340167],
                                         "SchurComplementHierarchicalRiskParity, EmpiricalPrior, x_src = :data, matrix B" =>
                                             [0.1716304928, 0.09866570937, 0.1604037817,
                                              0.1397897381, 0.1775762614, 0.2519340167],
                                         "HierarchicalEqualRiskContribution, EmpiricalPrior, x_src = :data, matrix B" =>
                                             [0.2932230114, 0.08955930859, 0.08409872942,
                                              0.1210842246, 0.1538144661, 0.2582202599],
                                         "InverseVolatility, matrix B" =>
                                             [0.1884448388, 0.1428795731, 0.1384552713,
                                              0.1661340148, 0.1872463536, 0.1768399484],
                                         "NestedClustered, the weights, matrix B" =>
                                             [0.03140174033, 3.729550387e-10,
                                              6.451774831e-10, 0.2469824812, 0.7094828766,
                                              0.01213290079],
                                         "NestedClustered, the prior and the distances, matrix B" =>
                                             [0.0008638618957, -0.0004297788067,
                                              0.001056166882, 0.001203214206,
                                              0.000607004792, 0.0005236166467, 0.0,
                                              0.3920619692, 0.4641138338, 0.488267683,
                                              0.5179622509, 0.441117128, 0.3920619692, 0.0,
                                              0.3686860413, 0.3787035686, 0.4165206349,
                                              0.4048104339, 0.4641138338, 0.3686860413, 0.0,
                                              0.3691478155, 0.3961371483, 0.4721444741,
                                              0.488267683, 0.3787035686, 0.3691478155, 0.0,
                                              0.4142607476, 0.4908969431, 0.5179622509,
                                              0.4165206349, 0.3961371483, 0.4142607476, 0.0,
                                              0.5164683389, 0.441117128, 0.4048104339,
                                              0.4721444741, 0.4908969431, 0.5164683389,
                                              0.0],
                                         "NestedClustered, inner and outer in the Excess Frame, matrix B" =>
                                             [0.3192895549, 1.28387963e-9, 1.133220587e-10,
                                              0.1482377963, 0.3024980935, 0.229974554],
                                         "Stacking, the weights, matrix B" =>
                                             [0.08404551048, 1.559096477e-8, 1.748752592e-8,
                                              0.2200745503, 0.6958798875, 1.860819108e-8],
                                         "Stacking, the prior, matrix B" =>
                                             [0.0008638618957, -0.0004297788067,
                                              0.001056166882, 0.001203214206,
                                              0.000607004792, 0.0005236166467],
                                         "SubsetResampling, the weights, matrix B" =>
                                             [0.1049182668, 1.523658768e-9, 4.811154957e-9,
                                              0.08609119214, 0.8089905283, 6.440907131e-9],
                                         "SubsetResampling, the prior, matrix B" =>
                                             [0.0008638618957, -0.0004297788067,
                                              0.001056166882, 0.001203214206,
                                              0.000607004792, 0.0005236166467],
                                         "SubsetResampling, inner in the Excess Frame, matrix B" =>
                                             [0.2880279248, 4.465391529e-9, 4.49941286e-9,
                                              0.04459237974, 0.4573058375, 0.210073849],
                                         "prior, EmpiricalPrior, matrix B" =>
                                             [0.0008638618957, -0.0004297788067,
                                              0.001056166882, 0.001203214206,
                                              0.000607004792, 0.0005236166467,
                                              0.0001802549832, 0.000164652407,
                                              0.000139644683, 0.0001069724279,
                                              8.407028162e-5, 0.000117330919,
                                              0.000164652407, 0.0003135562678,
                                              0.0002356089284, 0.0001923173382,
                                              0.0001562426117, 0.000170310177,
                                              0.000139644683, 0.0002356089284,
                                              0.0003339156578, 0.000202440176,
                                              0.0001694152236, 0.0001448773323,
                                              0.0001069724279, 0.0001923173382,
                                              0.000202440176, 0.0002319202409,
                                              0.0001351453673, 0.0001128705003,
                                              8.407028162e-5, 0.0001562426117,
                                              0.0001694152236, 0.0001351453673,
                                              0.0001825698406, 9.018475709e-5,
                                              0.000117330919, 0.000170310177,
                                              0.0001448773323, 0.0001128705003,
                                              9.018475709e-5, 0.000204689241],
                                         "prior, FactorPrior, matrix B" =>
                                             [0.0008638618957, -0.0004297788067,
                                              0.001056166882, 0.001203214206,
                                              0.000607004792, 0.0005236166467,
                                              0.0001802549832, 0.0001504225102,
                                              0.0001435732882, 0.0001091142325,
                                              8.763594897e-5, 0.0001220688156,
                                              0.0001504225102, 0.0003135562678,
                                              0.0002351660184, 0.0001861681503,
                                              0.0001516958555, 0.0001748841573,
                                              0.0001435732882, 0.0002351660184,
                                              0.0003339156578, 0.0001969512071,
                                              0.0001620395905, 0.0001610908676,
                                              0.0001091142325, 0.0001861681503,
                                              0.0001969512071, 0.0002319202409,
                                              0.0001363129037, 0.0001179782554,
                                              8.763594897e-5, 0.0001516958555,
                                              0.0001620395905, 0.0001363129037,
                                              0.0001825698406, 9.34561883e-5,
                                              0.0001220688156, 0.0001748841573,
                                              0.0001610908676, 0.0001179782554,
                                              9.34561883e-5, 0.000204689241],
                                         "ucs, NormalUncertaintySet box, matrix B" =>
                                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004804310895,
                                              0.00633643825, 0.006538917466, 0.005449501675,
                                              0.00483506127, 0.005119587517,
                                              0.0001379206791, 0.0001151871165,
                                              9.225052381e-5, 6.816611265e-5,
                                              5.089871498e-5, 8.01295343e-5,
                                              0.0001151871165, 0.000237049816,
                                              0.0001672354176, 0.0001350909119,
                                              0.0001067374762, 0.0001184534272,
                                              9.225052381e-5, 0.0001672354176,
                                              0.0002542283973, 0.0001439085116,
                                              0.0001166666588, 9.251633773e-5,
                                              6.816611265e-5, 0.0001350909119,
                                              0.0001439085116, 0.00017564787,
                                              9.317373036e-5, 6.983154643e-5,
                                              5.089871498e-5, 0.0001067374762,
                                              0.0001166666588, 9.317373036e-5,
                                              0.0001390822942, 5.324297221e-5,
                                              8.01295343e-5, 0.0001184534272,
                                              9.251633773e-5, 6.983154643e-5,
                                              5.324297221e-5, 0.0001557429131,
                                              0.0002279361447, 0.0002178645493,
                                              0.0001920006436, 0.0001497105437,
                                              0.0001207907785, 0.0001594668051,
                                              0.0002178645493, 0.0003977808192,
                                              0.0003125432162, 0.0002562337042,
                                              0.0002101918805, 0.0002280630858,
                                              0.0001920006436, 0.0003125432162,
                                              0.0004199828363, 0.0002677231864,
                                              0.0002250445953, 0.0001987324802,
                                              0.0001497105437, 0.0002562337042,
                                              0.0002677231864, 0.0002933947181,
                                              0.0001800199919, 0.0001597230744,
                                              0.0001207907785, 0.0002101918805,
                                              0.0002250445953, 0.0001800199919,
                                              0.0002302659443, 0.0001305088181,
                                              0.0001594668051, 0.0002280630858,
                                              0.0001987324802, 0.0001597230744,
                                              0.0001305088181, 0.0002626780537],
                                         "linear_constraints, an exposure that refits the loadings, matrix B" =>
                                             [0.4082081976, 0.2812401874, -0.171487239,
                                              -0.2048466642, -0.2243309381, 0.6895361585],
                                         "covariance_forecast_evaluation, Covariance, batch, matrix B" =>
                                             [-241.7952815, -243.5518468, -245.0361358,
                                              -218.0537771, -249.0675572, -223.0834928,
                                              -252.5964568, -228.4014988, -249.3823094,
                                              -238.0355838, -228.1897156, -217.7207305,
                                              0.9872240175, 0.9312990038, 0.8885115292,
                                              1.805097715, 0.75787597, 1.60699585,
                                              0.6315824551, 1.423923775, 0.7311777843,
                                              1.098371095, 1.440634256, 1.767391482,
                                              2.913056414e-7, 3.01293522e-7, 5.103213339e-7,
                                              2.658645712e-6, 9.295543916e-7,
                                              2.497088395e-7, 3.466034008e-7,
                                              4.279163757e-7, 3.688425233e-7,
                                              1.998948554e-7, 1.760853612e-7,
                                              7.757862841e-7],
                                         "covariance_forecast_evaluation, Covariance, online, matrix B" =>
                                             [-241.7952815, -243.5518468, -245.0361358,
                                              -218.0537771, -249.0675572, -223.0834928,
                                              -252.5964568, -228.4014988, -249.3823094,
                                              -238.0355838, -228.1897156, -217.7207305,
                                              0.9872240175, 0.9312990038, 0.8885115292,
                                              1.805097715, 0.75787597, 1.60699585,
                                              0.6315824551, 1.423923775, 0.7311777843,
                                              1.098371095, 1.440634256, 1.767391482,
                                              2.913056414e-7, 3.01293522e-7, 5.103213339e-7,
                                              2.658645712e-6, 9.295543916e-7,
                                              2.497088395e-7, 3.466034008e-7,
                                              4.279163757e-7, 3.688425233e-7,
                                              1.998948554e-7, 1.760853612e-7,
                                              7.757862841e-7],
                                         "covariance_forecast_evaluation, Online(Covariance()), online, matrix B" =>
                                             [-241.7952815, -243.5518468, -245.0361358,
                                              -218.0537771, -249.0675572, -223.0834928,
                                              -252.5964568, -228.4014988, -249.3823094,
                                              -238.0355838, -228.1897156, -217.7207305,
                                              0.9872240175, 0.9312990038, 0.8885115292,
                                              1.805097715, 0.75787597, 1.60699585,
                                              0.6315824551, 1.423923775, 0.7311777843,
                                              1.098371095, 1.440634256, 1.767391482,
                                              2.913056414e-7, 3.01293522e-7, 5.103213339e-7,
                                              2.658645712e-6, 9.295543916e-7,
                                              2.497088395e-7, 3.466034008e-7,
                                              4.279163757e-7, 3.688425233e-7,
                                              1.998948554e-7, 1.760853612e-7,
                                              7.757862841e-7],
                                         "covariance_forecast_portfolio of an evaluation in the Excess Frame, matrix B" =>
                                             [0.07506872465, -0.2025913117, -0.2443771523,
                                              -0.8550053601, -0.1275599978, 0.7129832487,
                                              0.3830428857, -1.195464577, -0.1799315842,
                                              0.1773473281, -0.7805826624, -1.184963676,
                                              -6.334251223, -6.385979863, -6.681264916,
                                              -4.739015107, -6.987956067, -6.362162555,
                                              -6.582436742, -6.127032947, -5.730518908,
                                              -6.46067759, -6.03122, -5.579834271],
                                         "Pipeline, a prior step, matrix B" =>
                                             [0.3454040729, 4.36098608e-9, 5.205926575e-9,
                                              0.05423798722, 0.3819628689, 0.2183950614],
                                         "Pipeline, an uncertainty step, matrix B" =>
                                             [0.3454140657, 7.693915524e-9, 1.110989512e-8,
                                              0.05423886655, 0.3819702837, 0.2183767653],
                                         "Pipeline, a clustering step, matrix B" =>
                                             [0.1716304928, 0.09866570937, 0.1604037817,
                                              0.1397897381, 0.1775762614, 0.2519340167],
                                         "Pipeline, a network step, matrix B" =>
                                             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
                                              0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
                                              0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                              0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                         "Pipeline, a phylogeny constraint step, matrix B" =>
                                             [0.3453798918, 9.710522418e-9, 8.14809282e-9,
                                              0.05430333166, 0.381941648, 0.2183751106],
                                         "Pipeline, a centrality constraint step, matrix B" =>
                                             [0.3454000682, 3.907313942e-13,
                                              3.903227633e-13, 0.05424503702, 0.3819579999,
                                              0.2183968948],
                                         "Pipeline, an exposure step, matrix B" =>
                                             [0.3454038789, 3.743064965e-9, 4.407746939e-9,
                                              0.05423856425, 0.3819589897, 0.218398559],
                                         "Pipeline, a prior step, online, matrix B" =>
                                             [0.3454040729, 4.360986502e-9, 5.205927079e-9,
                                              0.05423798722, 0.3819628689, 0.2183950614],
                                         "OnlinePortfolioSelection, ExponentiatedGradient, matrix B" =>
                                             [0.1669236648, 0.1655613659, 0.1670445633,
                                              0.1672324656, 0.1666628298, 0.1665751106],
                                         "OnlinePortfolioSelection, PassiveAggressiveMeanReversion, matrix B" =>
                                             [0.09918273931, 0.0, 0.0, 0.1100938152,
                                              0.3643926579, 0.4263307875],
                                         "OnlinePortfolioSelection, ForecastReversion, matrix B" =>
                                             [0.0, 0.0, 0.0, 0.2478700426, 0.1265508763,
                                              0.6255790811],
                                         "OnlinePortfolioSelection, ForecastReversion, MovingAverage scale, matrix B" =>
                                             [0.0, 0.0, 0.0, 0.2479326417, 0.1268695603,
                                              0.6251977979],
                                         "OnlinePortfolioSelection, AntiCorrelation, matrix B" =>
                                             [0.0, 0.0, 0.0, 0.0, 1.0, 7.418412301e-68],
                                         "FollowTheLeader, a CorrelationMatch selector, matrix B" =>
                                             [0.9999999999, 1.116755609e-22,
                                              1.138549487e-10, 6.029418839e-27,
                                              4.785989272e-20, 3.2532975590000005e-53],
                                         "FollowTheLeader, a leader in the Excess Frame, matrix B" =>
                                             [0.5045298174, 2.536398072e-9, 2.24567967e-9,
                                              9.034619119e-9, 0.3325849712, 0.1628851976],
                                         "ShortTermLossControlPortfolio, matrix B" =>
                                             [8.569227343e-9, 0.9999999911, 5.242781647e-11,
                                              7.552484815e-11, 1.106746801e-10,
                                              4.723727451e-11],
                                         "LowDimensionEnsemblePortfolio, matrix B" =>
                                             [4.132691672e-12, 3.021340223e-12,
                                              5.68213602e-12, 6.559851294e-12, 1.0,
                                              2.180664128e-11],
                                         "MirrorDescent, a RiskLoss in the Excess Frame, matrix B" =>
                                             [0.2140170663, 0.0887016282, 0.09885494154,
                                              0.1607563846, 0.2285949211, 0.2090750582],
                                         "MirrorDescent, a RiskLoss and the head in the Excess Frame, matrix B" =>
                                             [0.2140170663, 0.0887016282, 0.09885494154,
                                              0.1607563846, 0.2285949211, 0.2090750582],
                                         "ProgrammeAllocationSet, a Variance ceiling, matrix B" =>
                                             [3.556735054e-12, 3.738607992e-12, 0.156935427,
                                              7.438787603e-12, 0.843064573,
                                              1.238477654e-11],
                                         "prior, CrossSectionalFactorPrior, vector B" =>
                                             [0.0004917174546, 0.0002342595329,
                                              0.000306925665, 0.0006236643089,
                                              0.0003556508868, 0.0005075339824,
                                              0.0003334417616, 0.000301691597,
                                              0.0002052219097, 0.0005163947746,
                                              0.0004496710408, 0.0003573533869,
                                              0.0004481104537, 0.0002146903691,
                                              0.0005613360944, 0.0003472287567,
                                              0.0006280467355, 0.0004016483042,
                                              0.0003576206033, 0.0002466390792,
                                              0.0003143428491, 0.000393212676,
                                              0.0004781973342, 0.0005785469738,
                                              0.0004528268306, 0.0003808037497,
                                              0.0002459132641, 0.0003158000251,
                                              0.0003105043541, 0.0003700743657,
                                              0.0004917174546, 0.0002387595672,
                                              0.0002280799756, 0.0003829373015,
                                              0.0002181021643, 0.0002626276704,
                                              0.0003061709922, 0.0002823166839,
                                              0.0001943824034, 0.0003752713503,
                                              0.0003151107483, 0.0003455631226,
                                              0.0003123922889, 0.0001673641552,
                                              0.0002733861913, 0.0002401749362,
                                              0.0002756667617, 0.0002942755503,
                                              0.0002629975909, 0.000233728174,
                                              0.0002952342246, 0.0002924653985,
                                              0.0003948082004, 0.0002737583495,
                                              0.0003441623448, 0.0002208805899,
                                              0.0002545932043, 0.0003124122585,
                                              0.0002136092715, 0.0003195878083],
                                         "prior, CrossSectionalFactorPrior, matrix B" =>
                                             [0.0004898399521, 0.0002496281091,
                                              0.0003352624175, 0.0006472263995,
                                              0.0003765939033, 0.0005548111001,
                                              0.00038155362, 0.0003333689006,
                                              0.0002187555211, 0.0005826771872,
                                              0.0004739239315, 0.0003771290636,
                                              0.0004882124198, 0.0002116795187,
                                              0.0006608645545, 0.0003889282232,
                                              0.0006578931912, 0.0004364640812,
                                              0.0004142972626, 0.0002579279271,
                                              0.0003713996646, 0.0004578178964,
                                              0.000524569526, 0.0006276807374,
                                              0.0004395737223, 0.0004692653145,
                                              0.0002775407024, 0.0004004496733,
                                              0.0003368522611, 0.0004248877601,
                                              0.0004898399521, 0.0001490816869,
                                              0.0001706437799, 0.0003053392256,
                                              0.000136326433, 0.0002013758263,
                                              0.0002300686923, 0.0001977054121,
                                              0.0001019390219, 0.0003065906047,
                                              0.0002272830634, 0.0002753099415,
                                              0.0002373505784, 9.662510621e-5,
                                              0.0001881280558, 0.0001758240638,
                                              0.0002143442148, 0.0002165036258,
                                              0.0001983848085, 0.0001561954627,
                                              0.0002050603866, 0.0002164728848,
                                              0.0003191651684, 0.0002196530018,
                                              0.0002651588572, 0.0001673276213,
                                              0.0001667596898, 0.0002543907068,
                                              0.0001486627902, 0.0002355771329],
                                         "a vector benchmark gap, prior, EmpiricalPrior" =>
                                             [0.0008852612208, -0.0007843846159,
                                              0.0005603852398, 0.0005178401479,
                                              0.0003620027835, 0.000888319728,
                                              0.0001687099532, 0.0001660529675,
                                              0.0001423813879, 0.0001023573511,
                                              8.390361357e-5, 0.0001165745252,
                                              0.0001660529675, 0.0003160300311,
                                              0.000244078169, 0.0001910415642,
                                              0.0001545511452, 0.0001722358963,
                                              0.0001423813879, 0.000244078169,
                                              0.0003265488484, 0.0002028995277,
                                              0.0001756619992, 0.0001487263935,
                                              0.0001023573511, 0.0001910415642,
                                              0.0002028995277, 0.0002176136073,
                                              0.0001315217938, 0.0001099967909,
                                              8.390361357e-5, 0.0001545511452,
                                              0.0001756619992, 0.0001315217938,
                                              0.0001696604466, 8.924598473e-5,
                                              0.0001165745252, 0.0001722358963,
                                              0.0001487263935, 0.0001099967909,
                                              8.924598473e-5, 0.0001869109893],
                                         "a vector benchmark gap, MeanRisk" =>
                                             [0.3420650301, 9.899643624e-10, 1.080674006e-9,
                                              0.04752948512, 0.3917816672, 0.2186238154],
                                         "a vector benchmark gap, InverseVolatility" =>
                                             [0.1894951465, 0.1384536045, 0.1362054123,
                                              0.1668496959, 0.188963594, 0.1800325469],
                                         "a vector benchmark gap, HierarchicalRiskParity, FactorPrior" =>
                                             [0.1725984247, 0.09214020597, 0.1586536586,
                                              0.1404489137, 0.1801456696, 0.2560131274],
                                         "a vector benchmark gap, covariance_forecast_evaluation" =>
                                             [-241.7315419, -257.2059513, -233.6965311,
                                              -229.8227483, -250.8981474, -217.0927741,
                                              -242.6139347, -236.4825978, -251.5227979,
                                              -235.0087894, -243.3181941, 1.165684456,
                                              0.6453505463, 1.445819635, 1.571949385,
                                              0.8259181679, 1.965912603, 1.073326273,
                                              1.295957064, 0.7681954939, 1.332218221,
                                              1.047284669, 2.305116135e-7, 3.44445421e-7,
                                              3.70902202e-7, 1.514884844e-6, 9.251863285e-7,
                                              2.086897635e-7, 3.734820062e-7,
                                              1.662649439e-7, 3.313887556e-7, 1.43355389e-7,
                                              1.62512698e-7],
                                         "a matrix benchmark gap, prior, EmpiricalPrior" =>
                                             [0.0008638618957, -0.0004297788067,
                                              0.001203214206, 0.000607004792,
                                              0.0005236166467, 0.0001802549832,
                                              0.000164652407, 0.0001069724279,
                                              8.407028162e-5, 0.000117330919,
                                              0.000164652407, 0.0003135562678,
                                              0.0001923173382, 0.0001562426117,
                                              0.000170310177, 0.0001069724279,
                                              0.0001923173382, 0.0002319202409,
                                              0.0001351453673, 0.0001128705003,
                                              8.407028162e-5, 0.0001562426117,
                                              0.0001351453673, 0.0001825698406,
                                              9.018475709e-5, 0.000117330919,
                                              0.000170310177, 0.0001128705003,
                                              9.018475709e-5, 0.000204689241],
                                         "a matrix benchmark gap, the estimation mask" =>
                                             [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                         "a matrix benchmark gap, MeanRisk" =>
                                             [0.3454030465, 1.648350465e-9, 0.0,
                                              0.05423982531, 0.3819612489, 0.2183958777],
                                         "log_wealth_regret against the benchmark of the prediction" =>
                                             [-0.0006243783057, -0.01427246434,
                                              -0.02528121735, -0.01950885463,
                                              -0.03127841763, -0.02119499223,
                                              -0.03361118094, -0.008788647344,
                                              -0.002840885427, -0.005826175967,
                                              -0.02245069138, -0.02020249994,
                                              -0.02514147229, -0.01416699216,
                                              -0.01481058705, -0.01515130475,
                                              -0.03418696473, -0.0205131451, -0.0459020907,
                                              -0.04309675136, -0.04112597938,
                                              -0.03909236215, -0.03050029158,
                                              -0.03387796973, -0.02742588164,
                                              -0.03351931077, -0.02336446209,
                                              -0.02433144056, -0.03743397153,
                                              -0.04030762068, -0.04761820717,
                                              -0.04285195724, -0.06446643508,
                                              -0.05861144608, -0.06305628163,
                                              -0.07252012054, -0.04824334858,
                                              -0.03246282109, -0.04136379627,
                                              -0.05453730633, -0.0571675328, -0.03484445847,
                                              -0.03394760033, -0.03717487974,
                                              -0.03335279231, -0.04605261157,
                                              -0.05038913702, -0.06300842155,
                                              -0.07035325704, -0.06628781681,
                                              -0.07610032797, -0.06326893946,
                                              -0.05022861196, -0.04277822572,
                                              -0.05855377308, -0.04963515188,
                                              -0.06241863471, -0.06364377267,
                                              -0.02523374924, -0.0218668839,
                                              -0.0006243783057],
                                         "performance_summary against the benchmark of the prediction" =>
                                             [7.035220273e-5, 0.01240283443,
                                              0.005672268152])

    @testset "$key" for (key, tol, f) in cases
        pin = pins[replace(key, r"^online: " => "")]
        v = f()
        @test length(v) == length(pin)
        if tol === :solved
            @test isapprox(v, pin; atol = 5e-5)
        else
            @test isapprox(v, pin; rtol = 1e-8)
        end
    end
end
