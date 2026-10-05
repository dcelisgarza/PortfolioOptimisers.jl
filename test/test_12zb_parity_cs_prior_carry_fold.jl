#=
The carry fold of the Cross-Sectional Factor Prior (#1471, map #1375, ADR 0193).

An unwrapped prior folds as a carry: its first step seeds a `CrossSectionalCarryState`, and each
step computes the exposures, the regression and the idiosyncratic variance of the new
observations alone, from the panel rows that its Descriptors read. The factor prior and the
idiosyncratic variance fold, and the Return Forecast and the idiosyncratic correlation refit at the
call with no data. A batch choice that moves, or a factor that comes alive, fits every carried observation
again.

A step folds the member states in place, as every fold does, so each stream reads the prior out
right after its step.

Every `Parity_CrossSectionalFactorPrior_CarrySeed*` file is an output of the oracle's online update
over the stream of #1468, rows 1-90, 91-170 and 171-250 of the exchange that
`parity_write(dir, grid_fixture(parity_large_panel()).rd; filled = true)` writes. Its factor prior
is an exponentially weighted mean and covariance of half-life 20 and `min_observations = 5`, whose
window of 60 rows cuts the factor returns of the first call alone. Each file stacks the three steps
vertically, as the `Online*` files of test_12za do: one row of `mu`, one 40 x 40 block of `sigma`,
or the factor returns of every row fitted so far. The cases are `SeedStyle` (`Base` with
`families = ["style" => nothing]`) and `SeedIndustry` (`FamOne`). The pinned choice with no window
is checked against the `Online*Fold*` files of test_12za.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))
include(joinpath(@__DIR__, "parity_grid.jl"))
include(joinpath(@__DIR__, "test06c_setup.jl"))

@testset "The carry fold of the Cross-Sectional Factor Prior (#1471)" begin
    po = PortfolioOptimisers
    fx = parity_large_panel()
    rd = grid_fixture(fx)
    N = size(rd.X, 2)
    edges = (0, 90, 170, 250)
    rows(r, i) = po.port_opt_view(r, i, :)
    # Equal cell by cell, a `NaN` equal to a `NaN`.
    same(a, b) = size(a) == size(b) && all(((x, y),) -> isequal(x, y) || x == y, zip(a, b))
    function relerr(a, b)
        return maximum(abs, filter(isfinite, a - b); init = 0.0) /
               maximum(abs, filter(isfinite, b); init = 0.0)
    end
    # The prior after each step of a stream, and its call with no data at once.
    function stream(pe, r = rd, e = edges)
        out = []
        for k in 1:(length(e) - 1)
            pe = partial_fit!(pe, rows(r, (e[k] + 1):e[k + 1]))
            pr = try
                prior(pe)
            catch err
                err
            end
            push!(out, (; pe = pe, pr = pr))
        end
        return out
    end
    batch(pe, k, r = rd, e = edges) = prior(pe, rows(r, 1:e[k + 1]))
    function message(f)
        return try
            f()
            ""
        catch err
            sprint(showerror, err)
        end
    end
    function agrees(x, b)
        return same(x.mu, b.mu) &&
               same(x.sigma, b.sigma) &&
               same(x.X, b.X) &&
               same(x.o_X, b.o_X) &&
               same(x.fpr.X, b.fpr.X) &&
               same(x.rr.csr.f, b.rr.csr.f) &&
               same(x.rr.vs, b.rr.vs) &&
               same(x.rr.Ms, b.rr.Ms) &&
               same(x.rr.rw, b.rr.rw) &&
               isequal(x.rr.edof, b.rr.edof)
    end
    dropped(pr) = pr.rr.nf[po.dropped_factor_indices(pr.rr.fcb)]
    style = (; grid_config("Base", rd)..., families = ["style" => nothing])
    industry = grid_config("FamOne", rd)

    @testset "The carry fold equals the batch fit at each step" begin
        # Measured: every configuration gives a difference of exactly zero, because each
        # output of an observation reads no later observation, and the step runs the code of
        # the batch fit on the new observations.
        for name in
            ("Base", "Lag2", "Blend", "Neutralised", "FamStated", "FamTwo", "FcFixed",
             "FcCustom")
            pe = CrossSectionalFactorPrior(; grid_config(name, rd)...)
            for (k, x) in enumerate(stream(pe))
                @test agrees(x.pr, batch(pe, k))
            end
        end
        # One observation at a time after the first fit.
        pe = CrossSectionalFactorPrior(; style...)
        e = (0, 90, 91, 92, 93, 250)
        for (k, x) in enumerate(stream(pe, rd, e))
            @test agrees(x.pr, batch(pe, k, rd, e))
        end
    end

    @testset "The carried panel rows" begin
        # The Passthrough exposures read one row and the lag is one, so two rows are kept.
        pe = CrossSectionalFactorPrior(; style...)
        x = last(stream(pe))
        @test po.cross_sectional_carry_rows(pe) == 2 == size(x.pe.cache.win.X, 1)
        @test size(po.sample_buffer(po.returns_buffer(x.pe.cache))) == size(rd.X)
        # A Return Forecast that reads the panel keeps every row, a custom one keeps the
        # rows of the exposures.
        fixed = CrossSectionalFactorPrior(; grid_config("FcFixed", rd)...)
        @test isnothing(po.cross_sectional_carry_rows(fixed))
        @test size(last(stream(fixed)).pe.cache.win.X, 1) == 250
        @test po.cross_sectional_carry_rows(CrossSectionalFactorPrior(;
                                                                      grid_config("FcCustom",
                                                                                  rd)...)) ==
              2
        # A finite look-back keeps its last rows, an exponentially weighted Descriptor keeps
        # every row. The synthetic panel lists, delists and leaves gaps.
        syn = synthetic_asset_panel(; n_assets = 60, n_observations = 160, n_industries = 4,
                                    rng = StableRNG(1471)).rd
        bounded = ["market" => ConstantExposure(),
                   "reversal" => CompositeExposure(; descriptors = [Reversal()]),
                   "value" => CompositeExposure(; descriptors = [BookToPrice()]),
                   "industry" => OneHotExposure(; field = "industry", family = "industry")]
        ew = vcat(bounded, ["beta" => CompositeExposure(; descriptors = [EWMarketBeta()])])
        # The first block lies inside the Descriptor warm-up, and the next ones are too short
        # for the factor prior, so the fold refuses where the batch fit refuses, with the same
        # type of error. Inside the warm-up the batch fit refuses in its warm-up and the fold at
        # its call with no data, so the two messages differ there.
        e = (0, 10, 40, 41, 80, 81, 120, 160)
        for (factors, kept) in ((bounded, 22), (ew, 160))
            pe = CrossSectionalFactorPrior(; factors = factors,
                                           families = ["industry" => nothing], minra = 5,
                                           pe = GRID_PE, ve = GRID_VE)
            for (k, x) in enumerate(stream(pe, syn, e))
                b = try
                    batch(pe, k, syn, e)
                catch err
                    err
                end
                if isa(b, LowOrderPrior)
                    # Measured 1.3e-15 on the bounded case: a rolling return over the cut
                    # rows is a difference of cumulative sums from another first row (#1470).
                    # Exactly zero on the unbounded case.
                    @test relerr(x.pr.mu, b.mu) < 1e-14 &&
                          relerr(x.pr.sigma, b.sigma) < 1e-14
                    @test isequal(isnan.(x.pr.sigma), isnan.(b.sigma))
                else
                    @test typeof(x.pr) == typeof(b)
                end
            end
            @test size(last(stream(pe, syn, e)).pe.cache.win.X, 1) == kept
        end
    end

    @testset "The Choice Rule" begin
        # A batch choice that moves fits every observation again, and equals the batch fit.
        sb = stream(CrossSectionalFactorPrior(; style...))
        @test [dropped(x.pr) for x in sb] == [["style1"], ["style2"], ["style2"]]
        @test all(k -> agrees(sb[k].pr, batch(CrossSectionalFactorPrior(; style...), k)),
                  1:3)
        # A pinned choice is recorded in the state, and `families` stays as it is. Its
        # call with no data is the batch fit with the member stated.
        sp = stream(CrossSectionalFactorPrior(; style..., choice = PinnedChoice()))
        @test all(x -> x.pe.families == ["style" => nothing], sp)
        @test all(x -> x.pe.cache.families == ["style" => "style1"], sp)
        stated = CrossSectionalFactorPrior(; style..., families = ["style" => "style1"])
        @test all(k -> agrees(sp[k].pr, batch(stated, k)), 1:3)
        # A factor that is empty at the first fit and comes alive fits every observation
        # again.
        rdz = deepcopy(rd)
        po.panel_field(rdz.pnl, "style1").vals[1:120, :] .= 0.0
        pe = CrossSectionalFactorPrior(; grid_config("Base", rdz)...)
        sz = stream(pe, rdz)
        @test [x.pe.cache.lv for x in sz] == [[true, false, true], trues(3), trues(3)]
        @test all(k -> agrees(sz[k].pr, batch(pe, k, rdz)), 1:3)
    end

    @testset "Folds the members that fold, and refits the rest" begin
        # A rolling window does not fold, so every step fits every observation again over
        # the carried histories, the second pass of the weights too.
        wv = WindowedVariance(;
                              ve = ExpWeightedVariance(; decay = 2.0^(-1 / 20),
                                                       min_obs = 5), window = 60)
        for cfg in (style, grid_config("Blend", rd))
            pe = CrossSectionalFactorPrior(; cfg..., ve = wv)
            @test all(((k, x),) -> agrees(x.pr, batch(pe, k)), enumerate(stream(pe)))
        end
        # A factor prior that does not fold refits over the carried factor returns.
        pe = CrossSectionalFactorPrior(; style..., pe = EntropyPoolingPrior(; pe = GRID_PE))
        sx = stream(pe)
        @test all(x -> isa(x.pe.cache.pe, EntropyPoolingPrior), sx)
        @test all(k -> agrees(sx[k].pr, batch(pe, k)), 1:3)
    end

    @testset "Parity with the oracle's online update" begin
        function check(out, unit_case, src)
            load(o) = parity_load("CrossSectionalFactorPrior", unit_case, "$(src)$(o)")
            mu, S, F = load("Mu"), load("Sigma"), load("FactorReturns")
            r0 = 0
            for k in 1:3
                pr = out[k].pr
                n = edges[k + 1] - 1
                @test parity_compare(pr.mu, mu[k, :]; name = "$(unit_case) mu $(k)").ok
                @test parity_compare(pr.sigma, S[((k - 1) * N + 1):(k * N), :];
                                     scale = :array, name = "$(unit_case) sigma $(k)").ok
                @test parity_compare(pr.fpr.X, F[(r0 + 1):(r0 + n), :]; scale = :array,
                                     name = "$(unit_case) factor returns $(k)").ok
                r0 += n
            end
        end
        pinned(cfg) = stream(CrossSectionalFactorPrior(; cfg..., choice = PinnedChoice()))
        # The pinned choice reproduces the oracle's online update, which pins the dropped
        # member at its first call. Measured over the three steps: mu maxrel 1.7e-14, sigma
        # maxscaled 4.2e-13, factor returns maxscaled 6.7e-16 on `Style`; mu maxrel 5.3e-13,
        # sigma maxscaled 6.5e-13, factor returns maxscaled 2.9e-15 on `Industry`.
        check(pinned(style), "OnlineStyle", "Fold")
        check(pinned(industry), "OnlineIndustry", "Fold")
        # A seed window on the factor prior cuts the factor returns of the first fit to the
        # last 60, and every later step folds every row, as the oracle's does. Measured: mu
        # maxrel 7.2e-15, sigma maxscaled 3.7e-13, factor returns maxscaled 6.7e-16 on `Style`;
        # mu maxrel 1.4e-13, sigma maxscaled 7.3e-13, factor returns maxscaled 2.9e-15 on
        # `Industry`.
        decay = 2.0^(-1 / 20)
        seed = EmpiricalPrior(;
                              me = WindowedExpectedReturns(;
                                                           me = ExpWeightedExpectedReturns(;
                                                                                           decay = decay,
                                                                                           min_obs = 5),
                                                           window = 60,
                                                           rule = SeedWindow()),
                              ce = WindowedCovariance(;
                                                      ce = ExpWeightedCovariance(;
                                                                                 decay = decay,
                                                                                 min_obs = 5,
                                                                                 centred = true),
                                                      window = 60, rule = SeedWindow()))
        ss = pinned((; style..., pe = seed))
        check(ss, "CarrySeedStyle", "Fold")
        check(pinned((; industry..., pe = seed)), "CarrySeedIndustry", "Fold")
        # The first fit is the batch fit, and the window of the first fit alone moves the
        # later calls with no data away from it: sigma by 19 % of its largest entry at the last step.
        spe = CrossSectionalFactorPrior(; style..., pe = seed)
        @test agrees(ss[1].pr, batch(spe, 1))
        @test relerr(ss[3].pr.sigma, batch(spe, 3).sigma) > 0.1
    end

    @testset "On the online step of an optimiser" begin
        pe = CrossSectionalFactorPrior(; style...)
        o = po.update_online_estimator(InverseVolatility(; pe = pe))
        for k in 1:3
            o = partial_fit!(o, rows(rd, (edges[k] + 1):edges[k + 1]))
            bo, brd = po.batch_from_state(o)
            @test isa(o.pe.cache, po.CrossSectionalCarryState)
            @test agrees(bo.pe, batch(pe, k))
            @test isequal(brd.X, rd.X[1:edges[k + 1], :])
        end
        @test optimise(o).w == optimise(InverseVolatility(; pe = pe), rd).w
        # `partial_fit` copies the state, so the earlier prior answers as it did.
        e1 = partial_fit!(pe, rows(rd, 1:90))
        p1 = prior(e1)
        e2 = partial_fit(e1, rows(rd, 91:170))
        @test agrees(prior(e1), p1) && agrees(prior(e2), batch(pe, 2))
    end

    @testset "Refusals" begin
        pe = CrossSectionalFactorPrior(; style...)
        m = message(() -> prior(partial_fit!(pe, rows(rd, 1:2))))
        @test occursin("holds 2 observation(s) after the Descriptor warm-up", m)
        @test occursin("lag + 2 = 3", m)
        m = message(() -> partial_fit!(pe, ReturnsResult(; nx = rd.nx, X = rd.X[1:5, :])))
        @test occursin("the step carries no panel", m)
        e = partial_fit!(pe, rows(rd, 1:90))
        m = message(() -> po.port_opt_view(e, 1:5))
        @test occursin("not a slice of the ones it carries", m)
        m = message(() -> po.merge_states(e.cache, partial_fit!(pe, rows(rd, 91:170)).cache))
        @test occursin("cannot merge two states fitted on disjoint blocks", m)
        ccy = CrossSectionalFactorPrior(; grid_config("Currency", rd)...)
        m = message(() -> partial_fit!(ccy, rows(rd, 1:90)))
        @test occursin("does not record the Exogenous Series that an observed factor or a macro sensitivity reads",
                       m)
        @test occursin("the matrix form of `partial_fit!` carries none",
                       message(() -> partial_fit!(pe, rd.X)))
    end
end
