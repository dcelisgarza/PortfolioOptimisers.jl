#=
The Cross-Sectional Factor Prior on the online step (#1468, map #1375, ADR 0193).

The prior refits over a sample buffer that records the returns, both masks and the Panel Fields,
so its call with no data equals the batch fit over the rows of the buffer. Its Choice Rule says how the
automatic dropped member of a Factor Family behaves on that seam: `BatchChoice()` chooses again at
each call with no data, and `PinnedChoice()` keeps the choice of the first fit.

Every `Parity_CrossSectionalFactorPrior_Online*` file is an output of the oracle's online update
over one stream of batches, rows 1-90, 91-170 and 171-250 of the exchange that
`parity_write(dir, grid_fixture(parity_large_panel()).rd; filled = true)` writes. The panel holds
Panel Fields, and its estimation mask differs from its active mask on asset 6. Each file stacks
the three steps vertically: one row of `mu`, one 40 x 40 block of `sigma`, or the factor returns
of every row fitted so far. `Fold*` is the oracle's online update after each batch, and `Prefix*`
is its batch fit over the rows seen so far. The two cases are the grid configurations of
`parity_grid.jl`:

  - `Style`: `Base` with `families = ["style" => nothing]`. The oracle's update pins `style1` over
    the first batch, and its batch fit over 170 and 250 rows drops `style2`.
  - `Industry`: `FamOne`. Every fit drops `industry=Software`, so `Fold` and `Prefix` agree, and
    only `Fold` is stored.
  - `CurrencyStyle` (#1478): `Currency` with `families = ["style" => nothing]`. The oracle's update
    reads the Currency Excess Returns of each batch, and it pins `style1` as in `Style`. Only `Fold`
    is stored.

The refit records the Exogenous Series that an observed factor or a macro sensitivity reads
(#1478, ADR 0193), so those priors take the online step too.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))
include(joinpath(@__DIR__, "parity_grid.jl"))

@testset "The Cross-Sectional Factor Prior on the online step (#1468)" begin
    po = PortfolioOptimisers
    fx = parity_large_panel()
    rd = grid_fixture(fx)
    N = size(rd.X, 2)
    edges = (0, 90, 170, 250)
    rows(r, i) = po.port_opt_view(r, i, :)
    online(est; kwargs...) = po.update_online_estimator(Online(est; kwargs...))
    # The prior after each step of the stream, and its call with no data.
    function stream(pe)
        e = online(pe)
        out = []
        for k in 1:3
            e = partial_fit!(e, rows(rd, (edges[k] + 1):edges[k + 1]))
            push!(out, (; pe = e, pr = prior(e)))
        end
        return out
    end
    batch(pe, k) = prior(pe, rows(rd, 1:edges[k + 1]))
    dropped(pr) = pr.rr.nf[po.dropped_factor_indices(pr.rr.fcb)]
    function message(f)
        return try
            f()
            ""
        catch err
            err.msg
        end
    end
    style = (; grid_config("Base", rd)..., families = ["style" => nothing])
    industry = grid_config("FamOne", rd)
    sb = stream(CrossSectionalFactorPrior(; style...))
    sp = stream(CrossSectionalFactorPrior(; style..., choice = PinnedChoice()))
    ib = stream(CrossSectionalFactorPrior(; industry...))
    ip = stream(CrossSectionalFactorPrior(; industry..., choice = PinnedChoice()))
    sbatch = [batch(CrossSectionalFactorPrior(; style...), k) for k in 1:3]

    @testset "The refit equals the batch fit over the same rows" begin
        for k in 1:3
            pr, b = sb[k].pr, sbatch[k]
            @test isequal(pr.mu, b.mu) && isequal(pr.sigma, b.sigma)
            @test isequal(pr.fpr.X, b.fpr.X) && isequal(pr.rr.M, b.rr.M)
        end
        # A cap windows the whole fit: the call with no data is the batch fit over the last rows.
        e = online(CrossSectionalFactorPrior(; style...); max_history = 120)
        for k in 1:3
            e = partial_fit!(e, rows(rd, (edges[k] + 1):edges[k + 1]))
        end
        b = prior(CrossSectionalFactorPrior(; style...), rows(rd, 131:250))
        @test isequal(prior(e).mu, b.mu) && isequal(prior(e).sigma, b.sigma)
        # A view of a stepped prior slices its buffer to the selected assets.
        v = po.port_opt_view(sb[1].pe, 1:20)
        @test size(po.sample_buffer(v.cache)) == (90, 20)
        # The buffer is running state, and it does not print.
        s = sprint(show, sb[1].pe)
        @test !occursin("cache", s) && occursin("choice", s)
    end

    @testset "On the online step of an optimiser" begin
        # The step forwards each block to the prior, and the call with no data of the optimiser reads
        # the prior out of its buffer. A time-varying panel with Panel Fields and a narrower
        # estimation universe reaches the prior, and the weights equal the batch weights.
        for choice in (BatchChoice(), PinnedChoice())
            pe = CrossSectionalFactorPrior(; style..., choice = choice)
            o = po.update_online_estimator(InverseVolatility(; pe = Online(pe)))
            for k in 1:3
                o = partial_fit!(o, rows(rd, (edges[k] + 1):edges[k + 1]))
                bo, brd = po.batch_from_state(o)
                ref = choice isa BatchChoice ? sb[k].pr : sp[k].pr
                @test isequal(bo.pe.mu, ref.mu) && isequal(bo.pe.sigma, ref.sigma)
                @test isequal(brd.X, rd.X[1:edges[k + 1], :])
            end
            if choice isa BatchChoice
                @test isequal(optimise(o).w, optimise(InverseVolatility(; pe = pe), rd).w)
            end
        end
    end

    @testset "The Choice Rule" begin
        # In a batch fit the two rules agree.
        for k in 1:3
            pp = batch(CrossSectionalFactorPrior(; style..., choice = PinnedChoice()), k)
            @test isequal(pp.sigma, sbatch[k].sigma) && isequal(pp.mu, sbatch[k].mu)
        end
        # A batch choice chooses again at each call with no data, and leaves `families` alone.
        @test [dropped(x.pr) for x in sb] == [["style1"], ["style2"], ["style2"]]
        @test all(x -> x.pe.families == ["style" => nothing], sb)
        # A pinned choice writes the choice of the first fit into `families`, and every
        # later call with no data drops it.
        @test all(x -> x.pe.families == ["style" => "style1"], sp)
        @test [dropped(x.pr) for x in sp] == [["style1"], ["style1"], ["style1"]]
        @test isequal(sp[1].pr.sigma, sb[1].pr.sigma)
        # Pinned, the call with no data is the batch fit with the member stated.
        stated = CrossSectionalFactorPrior(; style..., families = ["style" => "style1"])
        @test isequal(sp[3].pr.sigma, batch(stated, 3).sigma)
        # The choice moves on this stream, so the two rules differ after the first step.
        @test maximum(abs, filter(isfinite, sp[3].pr.sigma - sb[3].pr.sigma)) > 1e-6
        # A first block that the fit refuses pins nothing. The next block pins the choice of
        # the fit over every row the buffer holds, the first fit.
        e = online(CrossSectionalFactorPrior(; style..., choice = PinnedChoice()))
        e = partial_fit!(e, rows(rd, 1:2))
        @test e.families == ["style" => nothing]
        @test_throws ArgumentError prior(e)
        e = partial_fit!(e, rows(rd, 3:90))
        @test e.families == ["style" => "style1"]
        @test isequal(prior(e).sigma, sp[1].pr.sigma)
        # A stated member is not pinned again, and the step leaves `families` as it is.
        e = online(CrossSectionalFactorPrior(; style..., choice = PinnedChoice(),
                                             families = ["style" => "style2"]))
        e = partial_fit!(e, rows(rd, 1:90))
        @test e.families == ["style" => "style2"]
        # So does a prior with no constrained family.
        e = online(CrossSectionalFactorPrior(; grid_config("Base", rd)...,
                                             choice = PinnedChoice()))
        @test isnothing(partial_fit!(e, rows(rd, 1:90)).families)
        # The industry choice does not move, so the two rules agree at every step.
        @test all(x -> x.pe.families == ["industry" => "industry=Software"], ip)
        @test all(k -> isequal(ip[k].pr.sigma, ib[k].pr.sigma), 1:3)
    end

    @testset "Refusals" begin
        pe = CrossSectionalFactorPrior(; style...)
        # An unwrapped prior takes the other route, the carry fold of #1471, which
        # test_12zb_parity_cs_prior_carry_fold.jl tests.
        @test isa(partial_fit!(pe, rows(rd, 1:90)).cache, po.CrossSectionalCarryState)
        m = message(() -> partial_fit!(online(pe), rd.X))
        @test occursin("the matrix form of `partial_fit!` carries none", m)
    end

    @testset "The refit records the Exogenous Series (#1478)" begin
        # An observed factor reads its return from the Exogenous Series, and an
        # `EWMacroSensitivity` that names a series reads its reference return there. The
        # buffer records every column of the series, so each of the three refits with no data
        # equals the batch fit over the same rows, bit for bit.
        base = grid_config("Base", rd)
        msens = "msens" => CompositeExposure(;
                                             descriptors = [EWMacroSensitivity(; series = "MACRO",
                                                                               half_life = 10)],
                                             outlier = nothing, scoring = nothing, family = "msens")
        cfgs = (; Currency = grid_config("Currency", rd), Macro = grid_config("Macro", rd),
                Sensitivity = (; base..., factors = [base.factors; msens]))
        for (name, cfg) in pairs(cfgs)
            pe = CrossSectionalFactorPrior(; cfg...)
            @test po.reads_exogenous_series(pe)
            out = stream(pe)
            for k in 1:3
                b = batch(pe, k)
                @test isequal(out[k].pr.mu, b.mu) && isequal(out[k].pr.sigma, b.sigma)
                @test isequal(out[k].pr.fpr.X, b.fpr.X)
            end
            st = out[3].pe.cache
            @test st.ne == rd.ne && isequal(po.exogenous_buffer_kwargs(st).E, rd.E)
            # A cap windows the series with the rows.
            e = online(pe; max_history = 120)
            for k in 1:3
                e = partial_fit!(e, rows(rd, (edges[k] + 1):edges[k + 1]))
            end
            @test isequal(prior(e).sigma, prior(pe, rows(rd, 131:250)).sigma)
        end
        # The predicate answers per type, through the factor list, the Return Forecast
        # Estimator and the wrapper. A macro sensitivity that names no series reads none.
        @test !po.reads_exogenous_series(CrossSectionalFactorPrior(; style...))
        @test !po.reads_exogenous_series(EWMacroSensitivity())
        @test po.reads_exogenous_series(Online(CrossSectionalFactorPrior(; base...,
                                                                         factors = [base.factors;
                                                                                    msens])))
        ds = DescriptorScores(; descriptors = [EWMacroSensitivity(; series = "MACRO")])
        @test po.reads_exogenous_series(CrossSectionalFactorPrior(; style...,
                                                                  rfe = FixedWeightedReturnForecast(;
                                                                                                    scores = ds,
                                                                                                    scale = 0.02)))
        # The same prior inside an optimiser: the prior's buffer owns the series, the Fold
        # Context keeps no copy and reads it back, and the weights equal the batch weights.
        ccy = CrossSectionalFactorPrior(; grid_config("Currency", rd)...)
        o = po.update_online_estimator(InverseVolatility(; pe = Online(ccy)))
        for k in 1:3
            o = partial_fit!(o, rows(rd, (edges[k] + 1):edges[k + 1]))
        end
        @test isnothing(o.cache.E) && isnothing(o.cache.ne)
        bo, brd = po.batch_from_state(o)
        @test brd.ne == rd.ne && isequal(brd.E, rd.E)
        @test isequal(bo.pe.sigma, prior(ccy, rd).sigma)
        @test isequal(optimise(o).w, optimise(InverseVolatility(; pe = ccy), rd).w)
        # A prior that reads no series leaves it to the Fold Context.
        o = po.update_online_estimator(InverseVolatility(;
                                                         pe = Online(CrossSectionalFactorPrior(;
                                                                                               style...))))
        o = partial_fit!(o, rows(rd, 1:90))
        @test isnothing(o.pe.cache.E) && o.cache.ne == rd.ne
        @test isequal(po.batch_from_state(o)[2].E, rd.E[1:90, :])
        # Refusals. A step with no series, and a step with other names.
        nor = ReturnsResult(; nx = rd.nx, X = rd.X, pnl = rd.pnl)
        m = message(() -> partial_fit!(online(ccy), rows(nor, 1:90)))
        @test occursin("this step carries no `rd.E`", m)
        e = partial_fit!(online(ccy), rows(rd, 1:90))
        m = message(() -> partial_fit!(e, rows(nor, 91:170)))
        @test occursin("this step carries no `rd.E`", m)
        ren = ReturnsResult(; nx = rd.nx, X = rd.X, ne = [rd.ne[1:(end - 1)]; "OTHER"],
                            E = rd.E, pnl = rd.pnl)
        m = message(() -> partial_fit!(e, rows(ren, 91:170)))
        @test occursin("a later block must carry the same names", m)
        # A non-finite value is refused on a row that the fit reads, and accepted on a row
        # that it does not: row 1 has no lagged exposure, so the regression never reads it.
        for (row, refused) in ((200, true), (1, false))
            E = copy(rd.E)
            E[row, 1] = NaN
            rn = ReturnsResult(; nx = rd.nx, X = rd.X, ne = rd.ne, E = E, pnl = rd.pnl)
            e = online(ccy)
            for k in 1:3
                e = partial_fit!(e, rows(rn, (edges[k] + 1):edges[k + 1]))
            end
            if refused
                @test_throws IsNonFiniteError prior(e)
                @test_throws IsNonFiniteError prior(ccy, rn)
            else
                @test isequal(prior(e).sigma, prior(ccy, rn).sigma)
            end
        end
        # The carry fold does not record the series yet, so it refuses a tree that reads it,
        # a macro sensitivity included.
        for cfg in (cfgs.Currency, cfgs.Sensitivity)
            m = message(() -> partial_fit!(CrossSectionalFactorPrior(; cfg...),
                                           rows(rd, 1:90)))
            @test occursin("the carry fold of a Cross-Sectional Factor Prior does not record the Exogenous Series",
                           m)
        end
    end

    @testset "Parity with the oracle's online update" begin
        load(case, o) = parity_load("CrossSectionalFactorPrior", "Online$(case)", o)
        # The prior of each step against the stored block of that step. `mu` compares cell by
        # cell; `sigma` and the factor returns compare against their largest entry, because
        # their small entries come from a cancellation (#1376), and the industry level
        # "Utilities" is empty after its only asset delists, so its factor return is a
        # round-off zero on both sides.
        function check(out, case, src)
            mu = load(case, "$(src)Mu")
            S = load(case, "$(src)Sigma")
            F = load(case, "$(src)FactorReturns")
            r0 = 0
            for k in 1:3
                pr = out[k].pr
                n = edges[k + 1] - 1
                @test parity_compare(pr.mu, mu[k, :]; name = "$(case) $(src) mu $(k)").ok
                @test parity_compare(pr.sigma, S[((k - 1) * N + 1):(k * N), :];
                                     scale = :array, name = "$(case) $(src) sigma $(k)").ok
                @test parity_compare(pr.fpr.X, F[(r0 + 1):(r0 + n), :]; scale = :array,
                                     name = "$(case) $(src) factor returns $(k)").ok
                r0 += n
            end
        end
        # The pinned choice reproduces the oracle's online update, which pins the dropped
        # member at its first call. Measured over the three steps: mu maxrel 1.7e-14, sigma
        # maxscaled 3.3e-13, factor returns maxscaled 6.7e-16 on `Style`; mu maxrel 5.3e-13,
        # sigma maxscaled 6.5e-13, factor returns maxscaled 2.9e-15 on `Industry`.
        check(sp, "Style", "Fold")
        check(ip, "Industry", "Fold")
        # The pinned choice under currency factors (#1478): the buffer records the Exogenous
        # Series, and the update pins `style1` over the first batch while the batch fit over
        # 170 and 250 rows drops `style2`. Measured over the three steps: mu maxrel 2.3e-13,
        # sigma maxscaled 4.2e-13, factor returns maxscaled 4.5e-16.
        cpe = CrossSectionalFactorPrior(; grid_config("Currency", rd)...,
                                        families = ["style" => nothing],
                                        choice = PinnedChoice())
        cp = stream(cpe)
        @test all(x -> x.pe.families == ["style" => "style1"], cp)
        check(cp, "CurrencyStyle", "Fold")
        # Deliberate difference: the batch choice is the oracle's batch fit over the rows
        # seen so far, not its online update. The automatic dropped member is a function of
        # the whole sample: the member with the largest sum of absolute benchmark-weighted
        # exposures. Under the batch choice the call with no data is the same function of the same
        # rows as the batch fit, whatever the split of the stream. The factor returns of the
        # full basis are the same under either member, but the factor prior fits the reduced
        # factor returns, whose time-varying ratios depend on the member, so its moments move
        # with the member: here sigma by 7.4e-5 at the second step. `PinnedChoice()` is the
        # oracle's rule, one keyword away. Measured against the oracle's batch fit: mu maxrel
        # 2.7e-14, sigma maxscaled 4.1e-13, factor returns maxscaled 6.7e-16.
        check(sb, "Style", "Prefix")
        @test !parity_compare(sb[2].pr.sigma, load("Style", "FoldSigma")[(N + 1):(2N), :];
                              scale = :array, name = "Style batch vs fold sigma 2").ok
    end
end
