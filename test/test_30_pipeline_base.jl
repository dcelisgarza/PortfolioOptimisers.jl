include(joinpath(@__DIR__, "asset_panel_fixture.jl"))
@testset "Pipeline base" begin
    using Test, PortfolioOptimisers, TimeSeries, Dates, StableRNGs

    function make_prices(; T = 5, N = 3)
        rng = StableRNG(123456789)
        ts = Date(2020, 1, 1):Day(1):Date(2020, 1, T)
        X = TimeArray(ts, 100 .+ rand(rng, T, N), string.("A", 1:N))
        F = TimeArray(ts, 50 .+ rand(rng, T, 2), ["f1", "f2"])
        B = TimeArray(ts, 100 .+ rand(rng, T, 1), ["BM"])
        iv = TimeArray(ts, rand(rng, T, N), string.("A", 1:N))
        return X, F, B, iv
    end

    @testset "PricesResult validation" begin
        X, F, B, iv = make_prices()
        pr = PricesResult(; X = X)
        @test pr.X === X
        @test isnothing(pr.F)
        @test isnothing(pr.B)
        @test isnothing(pr.iv)
        @test isnothing(pr.ivpa)

        pr = PricesResult(; X = X, F = F, B = B, iv = iv, ivpa = 0.5)
        @test pr.F === F
        @test pr.B === B
        @test pr.iv === iv
        @test pr.ivpa == 0.5

        # per-asset ivpa vector
        pr = PricesResult(; X = X, iv = iv, ivpa = fill(0.5, 3))
        @test length(pr.ivpa) == 3

        # empty X
        Xe = TimeArray(Date[], zeros(0, 2), ["A", "B"])
        @test_throws PortfolioOptimisers.IsEmptyError PricesResult(; X = Xe)

        # benchmark column count must be 1 or match X
        Bbad = TimeArray(timestamp(X), rand(StableRNG(1), 5, 2), ["b1", "b2"])
        @test_throws DimensionMismatch PricesResult(; X = X, B = Bbad)

        # iv column count must match X
        ivbad = TimeArray(timestamp(X), rand(StableRNG(2), 5, 2), ["A1", "A2"])
        @test_throws DimensionMismatch PricesResult(; X = X, iv = ivbad)

        # iv must be non-negative
        ivneg = TimeArray(timestamp(X), -rand(StableRNG(3), 5, 3), ["A1", "A2", "A3"])
        @test_throws DomainError PricesResult(; X = X, iv = ivneg)

        # ivpa must be strictly positive and finite
        @test_throws DomainError PricesResult(; X = X, iv = iv, ivpa = 0.0)
        @test_throws DomainError PricesResult(; X = X, iv = iv, ivpa = Inf)

        # vector ivpa length must match asset count
        @test_throws DimensionMismatch PricesResult(; X = X, iv = iv, ivpa = fill(0.5, 2))
    end

    @testset "port_opt_view" begin
        X, F, B, iv = make_prices()
        pr = PricesResult(; X = X, F = F, B = B, iv = iv, ivpa = fill(0.5, 3))

        # Colon is a passthrough
        @test PortfolioOptimisers.port_opt_view(pr, :) === pr

        # integer window
        pv = PortfolioOptimisers.port_opt_view(pr, 2:4)
        @test timestamp(pv.X) == timestamp(X)[2:4]
        @test values(pv.X) == values(X)[2:4, :]
        @test timestamp(pv.F) == timestamp(X)[2:4]
        @test values(pv.F) == values(F)[2:4, :]
        @test values(pv.B) == values(B)[2:4, :]
        @test values(pv.iv) == values(iv)[2:4, :]
        @test pv.ivpa === pr.ivpa

        # timestamp window
        window = timestamp(X)[[1, 3, 5]]
        pv = PortfolioOptimisers.port_opt_view(pr, window)
        @test timestamp(pv.X) == window
        @test values(pv.X) == values(X)[[1, 3, 5], :]

        # other series align to the master clock by timestamp: rows missing from a
        # series are dropped from that series, not invented
        Fshort = TimeArray(timestamp(X)[1:3], values(F)[1:3, :], ["f1", "f2"])
        pr2 = PricesResult(; X = X, F = Fshort)
        pv2 = PortfolioOptimisers.port_opt_view(pr2, 2:5)
        @test timestamp(pv2.X) == timestamp(X)[2:5]
        @test timestamp(pv2.F) == timestamp(X)[2:3]

        # optional series stay nothing
        pr3 = PricesResult(; X = X)
        pv3 = PortfolioOptimisers.port_opt_view(pr3, 2:3)
        @test isnothing(pv3.F)
        @test isnothing(pv3.B)
        @test isnothing(pv3.iv)
        @test isnothing(pv3.ivpa)
    end

    @testset "slot traits" begin
        # estimator families map to slots by dispatch
        @test PortfolioOptimisers.pipe_writes(EmpiricalPrior()) == :prior
        @test PortfolioOptimisers.pipe_reads(EmpiricalPrior()) == (:returns,)
        @test PortfolioOptimisers.pipe_writes(ClustersEstimator()) == :phylogeny
        @test PortfolioOptimisers.pipe_reads(ClustersEstimator()) == (:returns,)
        @test PortfolioOptimisers.pipe_writes(NormalUncertaintySet()) == :uncertainty
        @test PortfolioOptimisers.pipe_reads(NormalUncertaintySet()) == (:returns,)
        @test PortfolioOptimisers.pipe_writes(EqualWeighted()) == :opt
        @test PortfolioOptimisers.pipe_reads(EqualWeighted()) == (:returns,)

        #=
        Constraint estimators read :returns, from which the minimal UniverseSets is
        built. An ExposureConstraintEstimator is the one exception: it also reads
        :prior, because the basis it re-bases through is the prior's loadings. It is
        a declaration, not a preference — the reads are checked at construction, so
        the same pipeline without a prior step is rejected before it runs (see the
        fit tests).
        =#
        ece = ExposureConstraintEstimator(;
                                          lce = LinearConstraintEstimator(;
                                                                          val = "f1 <= 0.3"),
                                          space = FactorSpace())
        @test PortfolioOptimisers.pipe_writes(ece) == :constraints
        @test PortfolioOptimisers.pipe_reads(ece) == (:returns, :prior)
        # the estimator it decorates keeps the family default
        @test PortfolioOptimisers.pipe_writes(LinearConstraintEstimator(;
                                                                        val = "A1 <= 0.3")) ==
              :constraints
        @test PortfolioOptimisers.pipe_reads(LinearConstraintEstimator(; val = "A1 <= 0.3")) ==
              (:returns,)

        # estimators without a steppable family are rejected with guidance
        @test_throws ArgumentError PortfolioOptimisers.pipe_writes(Covariance())
        @test_throws ArgumentError PortfolioOptimisers.pipe_reads(Covariance())
    end

    @testset "slot invalidation" begin
        # a step that rewrites :returns after a returns-derived slot was written
        # leaves that slot computed on a stale asset universe
        @test_throws ArgumentError Pipeline(;
                                            steps = (PricesToReturns(), EmpiricalPrior(),
                                                     MissingDataFilter(),
                                                     PricesToReturns()))
        @test_throws ArgumentError Pipeline(;
                                            steps = (PricesToReturns(), ClustersEstimator(),
                                                     PricesToReturns()))

        # a prices-level step after the prices-to-returns conversion leaves :returns stale
        @test_throws ArgumentError Pipeline(;
                                            steps = (PricesToReturns(),
                                                     MissingDataFilter()))

        # the invalidation is declared by the step's writes, so a wrapped callable is caught too
        @test_throws ArgumentError Pipeline(;
                                            steps = (PricesToReturns(), EmpiricalPrior(),
                                                     PipelineStep(;
                                                                  est = ctx -> ctx.returns,
                                                                  reads = (:returns,),
                                                                  writes = :returns)))

        # slots filled by the pipeline input are not "written", so the canonical
        # prices-level ordering is unaffected
        @test Pipeline(;
                       steps = (MissingDataFilter(), PriceGapFill(), PricesToReturns(),
                                EmpiricalPrior(), EqualWeighted())) isa Pipeline

        # writing a derived slot invalidates nothing
        @test Pipeline(;
                       steps = (PricesToReturns(), EmpiricalPrior(), ClustersEstimator(),
                                EqualWeighted())) isa Pipeline

        # the invalidation table is derived from slot order + data slots; pin the whole
        # thing so a future PIPELINE_SLOTS edit cannot silently change it
        @test PortfolioOptimisers.PIPELINE_DATA_SLOTS == (:prices, :returns)
        @test PortfolioOptimisers.PIPELINE_INVALIDATES ==
              (prices = (:returns, :prior, :phylogeny, :uncertainty, :constraints),
               returns = (:prior, :phylogeny, :uncertainty, :constraints))
        # only data slots are keys; every other slot (incl. terminal :opt) invalidates nothing
        @test keys(PortfolioOptimisers.PIPELINE_INVALIDATES) ==
              PortfolioOptimisers.PIPELINE_DATA_SLOTS
        @test all(s -> get(PortfolioOptimisers.PIPELINE_INVALIDATES, s, ()) == (),
                  (:prior, :phylogeny, :uncertainty, :constraints, :opt))
    end

    @testset "optimiser must be last" begin
        # an optimisation step writes the terminal :opt slot; nothing may run after it
        @test_throws ArgumentError Pipeline(;
                                            steps = (PricesToReturns(), EqualWeighted(),
                                                     EmpiricalPrior()))
        # caught for a wrapped/optimiser-position step too
        @test_throws ArgumentError Pipeline(; steps = (EqualWeighted(), EqualWeighted()))
        # a non-terminal optimiser hidden in a nested pipeline is caught by its own construction
        @test_throws ArgumentError Pipeline(;
                                            steps = (Pipeline(;
                                                              steps = (PricesToReturns(),
                                                                       EqualWeighted(),
                                                                       EmpiricalPrior())),))
        # optimiser last is fine; no optimiser at all is fine (prior-only pipeline)
        @test Pipeline(; steps = (PricesToReturns(), EmpiricalPrior(), EqualWeighted())) isa
              Pipeline
        @test Pipeline(; steps = (PricesToReturns(), EmpiricalPrior())) isa Pipeline
    end

    @testset "step names" begin
        PO = PortfolioOptimisers
        # A slot that one step writes names the step; a repeated slot takes its position.
        # The count includes the step with an explicit name.
        @test Pipeline(; steps = (EmpiricalPrior(), EmpiricalPrior(), EqualWeighted())).names ==
              ("prior_1", "prior_2", "opt")
        @test Pipeline(; steps = ("mine" => EmpiricalPrior(), EmpiricalPrior())).names ==
              ("mine", "prior_2")
        # An explicit name can collide with a generated one, and the error names it.
        err = try
            Pipeline(;
                     steps = ("prior_2" => EmpiricalPrior(), EmpiricalPrior(),
                              EmpiricalPrior()))
        catch e
            e
        end
        @test err isa ArgumentError && occursin("\"prior_2\"", err.msg)
        @test isnothing(PO.first_duplicate(()))
        @test isnothing(PO.first_duplicate(("a", "b")))
        @test PO.first_duplicate(("a", "b", "a", "b")) == "a"
    end

    @testset "PipelineStep" begin
        ps = PipelineStep(; est = NormalUncertaintySet(), reads = (:returns,),
                          writes = :uncertainty, target = :mu)
        @test PortfolioOptimisers.pipe_writes(ps) == :uncertainty
        @test PortfolioOptimisers.pipe_reads(ps) == (:returns,)
        @test ps.target == :mu
        @test ps.est isa NormalUncertaintySet

        # reads defaults to empty, target to nothing
        ps = PipelineStep(; est = EmpiricalPrior(), writes = :prior)
        @test PortfolioOptimisers.pipe_reads(ps) == ()
        @test isnothing(ps.target)

        # callables are steppable when wrapped
        ps = PipelineStep(; est = x -> x, reads = (:prices,), writes = :prices)
        @test PortfolioOptimisers.pipe_writes(ps) == :prices

        # slot names are validated
        @test_throws ArgumentError PipelineStep(; est = EmpiricalPrior(),
                                                writes = :nonsense)
        @test_throws ArgumentError PipelineStep(; est = EmpiricalPrior(),
                                                reads = (:nonsense,), writes = :prior)

        # The routing annotation is allowlisted here too, so a mistyped target is refused
        # where it is written rather than by the step that reads it, mid-pipeline.
        @test_throws ArgumentError PipelineStep(; est = NormalUncertaintySet(),
                                                reads = (:returns,), writes = :uncertainty,
                                                target = :nonsense)
        for t in PortfolioOptimisers.PIPELINE_STEP_TARGETS
            @test PipelineStep(; est = NormalUncertaintySet(), reads = (:returns,),
                               writes = :uncertainty, target = t).target === t
        end
    end

    @testset "PipelineContext" begin
        ctx = PortfolioOptimisers.PipelineContext()
        @test all(s -> isnothing(getproperty(ctx, s)), PortfolioOptimisers.PIPELINE_SLOTS)

        rd = ReturnsResult(; nx = ["a", "b"], X = [0.1 -0.2; -0.1 0.2; 0.05 0.1])
        pr = prior(EmpiricalPrior(), rd)
        ctx = PortfolioOptimisers.PipelineContext(; returns = rd, prior = pr)
        @test ctx.returns === rd
        @test ctx.prior === pr

        # slots are typed (keyword type mismatch throws a TypeError)
        @test_throws TypeError PortfolioOptimisers.PipelineContext(; returns = pr)
    end
    @testset "a price-starting Pipeline keeps the Asset Panel" begin
        # The panel enters as part of the :prices Data Slot, and every step between it and
        # the :returns slot owes it the slice that step's own axes take. MissingDataFilter
        # drops an asset at price level and PricesToReturns drops the first observation to
        # the percentage change, so a panel that reaches :returns intact has survived both.
        # The filter is what drops the asset: the conversion itself deletes nothing (ADR
        # 0133), so `col_thr` has to say so.
        ts = Date(2020, 1, 1):Day(1):Date(2020, 1, 10)
        Pv = 100 .+ cumsum(rand(StableRNG(987654321), 10, 3); dims = 1)
        Pv[:, 2] .= NaN                       # asset "A2" is entirely missing
        Px = TimeArray(collect(ts), Pv, ["A1", "A2", "A3"])
        # Z3[i, j, k] == i + 10(j - 1) + 30(k - 1), so every entry names its own position.
        Z3 = reshape(Float64.(1:60), 10, 3, 2)
        nz = ["f1", "f2"]
        pipe = Pipeline(; steps = (MissingDataFilter(; col_thr = 0.5), PricesToReturns()))

        rd = fit(pipe, PricesResult(; X = Px, pnl = matrix_panel(nz, Z3))).ctx.returns
        @test !isnothing(rd.pnl)
        @test rd.nx == ["A1", "A3"]
        @test panel_feature_matrix(rd.pnl)[2] == Z3[2:10, [1, 3], :]
        @test panel_feature_matrix(rd.pnl)[1] == nz
        # Both universe masks are the carrier's own two axes.
        @test size(rd.pnl.amsk) == size(rd.pnl.emsk) == size(rd.X)

        # A static panel has no observation axis, so only its asset axis is sliced, and it
        # still carries no mask.
        Zs = Float64[1 2; 3 4; 5 6]
        rds = fit(pipe, PricesResult(; X = Px, pnl = matrix_panel(nz, Zs))).ctx.returns
        @test panel_feature_matrix(rds.pnl)[2] == Zs[[1, 3], :]
        @test isnothing(rds.pnl.amsk)
        @test isnothing(rds.pnl.emsk)
    end
end

@testset "Pipeline preprocessing steps" begin
    using Test, PortfolioOptimisers, TimeSeries, Dates, StableRNGs, Statistics

    function make_prices(; T = 10, N = 4)
        rng = StableRNG(123456789)
        ts = Date(2020, 1, 1):Day(1):Date(2020, 1, T)
        X = TimeArray(ts, 100 .+ rand(rng, T, N), string.("A", 1:N))
        return X, ts
    end

    @testset "PricesToReturns" begin
        @test_throws ArgumentError PricesToReturns(; ret_method = :nonsense)

        X, _ = make_prices()
        pr = PricesResult(; X = X)
        ptr = PricesToReturns()

        # stateless: the fitted object is the estimator itself
        @test PortfolioOptimisers.fit_preprocessing(ptr, pr) === ptr

        # apply matches the underlying function
        rr = PortfolioOptimisers.apply_preprocessing(ptr, pr)
        rr_ref = prices_to_returns(X)
        @test rr.X == rr_ref.X
        @test rr.nx == rr_ref.nx
        @test rr.ts == rr_ref.ts

        # ret_method is honoured
        rr_log = PortfolioOptimisers.apply_preprocessing(PricesToReturns(;
                                                                         ret_method = :log),
                                                         pr)
        @test rr_log.X != rr.X
        @test rr_log.X ≈ log1p.(rr.X)

        # run_step reads :prices, writes :returns
        ctx = PortfolioOptimisers.PipelineContext(; prices = pr)
        fitted, ctx2 = PortfolioOptimisers.run_step(ptr, ctx)
        @test fitted === ptr
        @test ctx2.returns.X == rr.X
        @test ctx2.prices === pr

        # requires :prices
        @test_throws PortfolioOptimisers.IsNothingError PortfolioOptimisers.run_step(ptr,
                                                                                     PortfolioOptimisers.PipelineContext())
    end

    @testset "MissingDataFilter" begin
        # The domain is `[0, 1]`: zero is the tightest policy the estimator can state, and
        # it is a spelling rather than an arithmetic accident (ADR 0133).
        @test_throws DomainError MissingDataFilter(; col_thr = -0.1)
        @test_throws DomainError MissingDataFilter(; col_thr = 1.5)
        @test_throws DomainError MissingDataFilter(; row_thr = -0.1)
        @test_throws DomainError MissingDataFilter(; row_thr = 1.5)
        @test MissingDataFilter(; col_thr = 0.0, row_thr = 0.0) isa MissingDataFilter

        X, ts = make_prices()
        vals = copy(values(X))
        vals[1:6, 2] .= NaN      # A2: 60% missing -> dropped at col_thr = 0.5
        vals[1, 3] = NaN         # A3: 10% missing -> kept
        Xm = TimeArray(ts, vals, string.("A", 1:4))
        pr = PricesResult(; X = Xm)

        mdf = MissingDataFilter(; col_thr = 0.5, row_thr = 0.5)
        res = PortfolioOptimisers.fit_preprocessing(mdf, pr)
        @test res.nx == [:A1, :A3, :A4]

        # apply subsets the universe; row 1 has 1/3 missing <= row_thr so it stays
        pv = PortfolioOptimisers.apply_preprocessing(res, pr)
        @test TimeSeries.colnames(pv.X) == [:A1, :A3, :A4]
        @test length(TimeSeries.timestamp(pv.X)) == 10

        # rows above the row threshold are dropped from the window being transformed
        res_strict = PortfolioOptimisers.fit_preprocessing(MissingDataFilter(;
                                                                             col_thr = 0.5,
                                                                             row_thr = 0.2),
                                                           pr)
        pv_strict = PortfolioOptimisers.apply_preprocessing(res_strict, pr)
        @test length(TimeSeries.timestamp(pv_strict.X)) == 9
        @test first(TimeSeries.timestamp(pv_strict.X)) == ts[2]

        # the universe is fitted state: a clean test window is still subset to it
        Xc, _ = make_prices()
        pr_clean = PricesResult(; X = Xc)
        pv_clean = PortfolioOptimisers.apply_preprocessing(res, pr_clean)
        @test TimeSeries.colnames(pv_clean.X) == [:A1, :A3, :A4]

        # iv and vector ivpa follow the universe
        iv = TimeArray(ts, rand(StableRNG(4), 10, 4), string.("A", 1:4))
        pr_iv = PricesResult(; X = Xm, iv = iv, ivpa = [0.1, 0.2, 0.3, 0.4])
        pv_iv = PortfolioOptimisers.apply_preprocessing(res, pr_iv)
        @test TimeSeries.colnames(pv_iv.iv) == [:A1, :A3, :A4]
        @test pv_iv.ivpa == [0.1, 0.3, 0.4]

        # dropping every asset is an error at fit time
        Xall = TimeArray(ts, fill(NaN, 10, 2), ["A1", "A2"])
        @test_throws PortfolioOptimisers.IsEmptyError PortfolioOptimisers.fit_preprocessing(MissingDataFilter(;
                                                                                                              col_thr = 0.5),
                                                                                            PricesResult(;
                                                                                                         X = Xall))

        # run_step reads and writes :prices
        ctx = PortfolioOptimisers.PipelineContext(; prices = pr)
        fitted, ctx2 = PortfolioOptimisers.run_step(mdf, ctx)
        @test fitted isa MissingDataFilterResult
        @test TimeSeries.colnames(ctx2.prices.X) == [:A1, :A3, :A4]
    end

    @testset "MissingDataFilter at the zero threshold" begin
        # Map #955, ADR 0133. `prices_to_returns` deletes nothing, so this estimator is the
        # only filter there is, and `0.0` is the tightest policy it can state: no gap is
        # tolerated. It is what `dropmissing!` used to do inside the conversion, said by
        # name and split correctly across the fit/apply seam.
        ts = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 6))
        # `A1` is priced throughout; `A2` and `A5` are listed from observation 3; `A3` is
        # suspended at observation 4; `A4` is delisted after observation 4. The two
        # inceptions make the leading rows the gappiest, so a row threshold between them
        # and the rest opens the window without touching the delisting.
        vals = [10.0 NaN 30.0 40.0 NaN
                11.0 NaN 31.0 41.0 NaN
                12.0 20.0 32.0 42.0 50.0
                13.0 21.0 NaN 43.0 51.0
                14.0 22.0 34.0 NaN 52.0
                15.0 23.0 35.0 NaN 53.0]
        Xg = TimeArray(ts, vals, string.("A", 1:5))
        pr = PricesResult(; X = Xg, span = listing_span(vals))

        # `col_thr = 0.0` keeps the columns with no gap at all, and nothing else. Only `A1`
        # is priced at every observation of the training window.
        zc = PortfolioOptimisers.fit_preprocessing(MissingDataFilter(; col_thr = 0.0), pr)
        @test zc.nx == [:A1]
        # A column holding one gap in six is above zero, so the boundary is `count == 0`
        # and not a small fraction: `A3` needs the threshold raised to its own fraction.
        @test PortfolioOptimisers.fit_preprocessing(MissingDataFilter(; col_thr = 1 / 6),
                                                    pr).nx == [:A1, :A3]

        # `row_thr = 0.0` keeps the observations at which every surviving asset is priced.
        # Over the whole universe that is observation 3 alone.
        zr = PortfolioOptimisers.fit_preprocessing(MissingDataFilter(; col_thr = 1.0,
                                                                     row_thr = 0.0), pr)
        pv = PortfolioOptimisers.apply_preprocessing(zr, pr)
        @test TimeSeries.timestamp(pv.X) == [ts[3]]
        @test !any(PortfolioOptimisers.is_missing_value, values(pv.X))
        # A row holding one gap in five is above zero, so the row boundary is `count == 0`
        # too, and it takes `1/5` to admit one.
        loose = PortfolioOptimisers.fit_preprocessing(MissingDataFilter(; col_thr = 1.0,
                                                                        row_thr = 1 / 5),
                                                      pr)
        @test TimeSeries.timestamp(PortfolioOptimisers.apply_preprocessing(loose, pr).X) ==
              ts[[3, 4, 5, 6]]

        # A **column** drop leaves every row where it was, so the Span Rule reads the same
        # `first` and `last` off the filtered panel as off the raw one for every asset the
        # filter kept. This is the reading the ticket asked for, and it holds on the axis
        # the fitted universe cuts.
        raw = listing_span(vals)
        conly = PortfolioOptimisers.fit_preprocessing(MissingDataFilter(; col_thr = 1 / 6,
                                                                        row_thr = 1.0), pr)
        cpr = PortfolioOptimisers.apply_preprocessing(conly, pr)
        @test conly.nx == [:A1, :A3]
        cut = listing_span(values(cpr.X))
        cols = Vector{Int}(indexin(conly.nx, TimeSeries.colnames(pr.X)))
        @test cut.first == raw.first[cols]
        @test cut.last == raw.last[cols]

        # A **row** drop does move an observation, so re-deriving the rule off the filtered
        # panel is *not* the same reading — which is exactly why the estimator never
        # re-derives. The span it hands on is the carrier's, viewed at the surviving rows
        # and columns, so every surviving observation says of every kept asset what it said
        # before the filter ran.
        for thr in (0.0, 1 / 5, 2 / 5, 1.0)
            res = PortfolioOptimisers.fit_preprocessing(MissingDataFilter(; col_thr = 1.0,
                                                                          row_thr = thr),
                                                        pr)
            fpr = PortfolioOptimisers.apply_preprocessing(res, pr)
            rows = Vector{Int}(indexin(TimeSeries.timestamp(fpr.X), ts))
            @test fpr.span == view(pr.span, rows, 1:length(res.nx))
        end

        # And that is what keeps a delisting readable. At `row_thr = 1/5` the window opens
        # at observation 3, and `A4`'s two trailing gaps are still outside its span.
        f4 = PortfolioOptimisers.apply_preprocessing(loose, pr)
        @test TimeSeries.timestamp(f4.X) == ts[[3, 4, 5, 6]]
        @test f4.span[:, 4] == [true, true, false, false]
        # The inception `A2` and `A5` carry is inside the surviving window on both
        # readings, so nothing about them turns on the view here.
        @test f4.span[:, 2] == [true, true, true, true]

        # The case that proves a re-derivation would be wrong rather than merely different:
        # a row drop can take away every priced observation an asset has inside its
        # listing. `B2` is priced at observations 2 and 5 and suspended between them, and
        # `row_thr = 0.0` keeps observations 3 and 4 alone. The carried span still reports
        # it listed there -- a Held Gap -- where the rule read off the surviving prices
        # would report an asset that was never listed at all.
        ts2 = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 6))
        v2 = [10.0 NaN
              11.0 20.0
              12.0 NaN
              13.0 NaN
              14.0 23.0
              NaN NaN]
        pr2 = PricesResult(; X = TimeArray(ts2, v2, ["B1", "B2"]), span = listing_span(v2))
        # Observations 3 and 4 hold a gap in `B2` alone, so no row survives `row_thr = 0`;
        # `1/2` keeps every row whose gap count is at most one, which is 1 through 5.
        r2 = PortfolioOptimisers.fit_preprocessing(MissingDataFilter(; col_thr = 1.0,
                                                                     row_thr = 1 / 2), pr2)
        f2 = PortfolioOptimisers.apply_preprocessing(r2, pr2)
        @test TimeSeries.timestamp(f2.X) == ts2[1:5]
        # The carried span says B2 is listed from observation 2 to observation 5.
        @test f2.span[:, 2] == [false, true, true, true, true]
        # A re-derivation would agree here, because the priced observations survived. Now
        # take them away: keep observations 3 and 4 by hand and compare the two readings.
        kept_rows = [3, 4]
        @test view(pr2.span, kept_rows, :)[:, 2] == [true, true]
        rederived = listing_span(v2[kept_rows, :])
        @test rederived.last[2] < rederived.first[2]
    end

    @testset "PriceGapFill" begin
        # Map #955, ADR 0130. The fill role holds one estimator, and it is bounded by the
        # Listing Span, so the carrier states one — `test_06_preprocessing.jl` owns the
        # conventions, the bound and the replay; what is asserted here is the step's place in
        # a pipeline.
        X, ts = make_prices()
        vals = copy(values(X))
        vals[2, 1] = NaN
        Xm = TimeArray(ts, vals, string.("A", 1:4))
        pr = PricesResult(; X = Xm, span = trues(size(vals)))

        pgf = PriceGapFill(; fill = MedianValue())
        res = PortfolioOptimisers.fit_preprocessing(pgf, pr)
        @test res.nx == [:A1, :A2, :A3, :A4]
        train_med = median([x for x in vals[:, 1] if !isnan(x)])
        @test res.v[1] == train_med

        pv = PortfolioOptimisers.apply_preprocessing(res, pr)
        @test values(pv.X)[2, 1] == train_med
        @test !any(isnan, values(pv.X))

        # leakage regression: a test window is filled with TRAIN statistics,
        # not with its own
        Xt, _ = make_prices()
        tvals = copy(values(Xt)) .+ 50.0
        tvals[5, 1] = NaN
        Xtm = TimeArray(ts, tvals, string.("A", 1:4))
        pv_test = PortfolioOptimisers.apply_preprocessing(res,
                                                          PricesResult(; X = Xtm,
                                                                       span = trues(size(tvals))))
        test_med = median([x for x in tvals[:, 1] if !isnan(x)])
        @test values(pv_test.X)[5, 1] == train_med
        @test values(pv_test.X)[5, 1] != test_med

        # the convention is configurable
        res_mean = PortfolioOptimisers.fit_preprocessing(PriceGapFill(; fill = MeanValue()),
                                                         pr)
        train_mean = mean([x for x in vals[:, 1] if !isnan(x)])
        @test res_mean.v[1] ≈ train_mean

        # assets with no observed values get no fill value and pass through
        vals_empty = copy(values(X))
        vals_empty[:, 2] .= NaN
        Xe = TimeArray(ts, vals_empty, string.("A", 1:4))
        pr_e = PricesResult(; X = Xe, span = trues(size(vals_empty)))
        res_e = PortfolioOptimisers.fit_preprocessing(pgf, pr_e)
        @test res_e.nx == [:A1, :A3, :A4]
        pv_e = PortfolioOptimisers.apply_preprocessing(res_e, pr_e)
        @test all(isnan, values(pv_e.X)[:, 2])

        # run_step reads and writes :prices
        ctx = PortfolioOptimisers.PipelineContext(; prices = pr)
        fitted, ctx2 = PortfolioOptimisers.run_step(pgf, ctx)
        @test fitted isa PriceGapFillResult
        @test !any(isnan, values(ctx2.prices.X))
    end

    @testset "run_step for existing estimator families" begin
        rng = StableRNG(987654321)
        rd = ReturnsResult(; nx = string.("A", 1:5), X = randn(rng, 60, 5) / 100)
        ctx = PortfolioOptimisers.PipelineContext(; returns = rd)

        # prior step
        fitted, ctx2 = PortfolioOptimisers.run_step(EmpiricalPrior(), ctx)
        @test fitted === ctx2.prior
        @test ctx2.prior.X == rd.X
        @test length(ctx2.prior.mu) == 5

        # phylogeny step
        fitted, ctx3 = PortfolioOptimisers.run_step(ClustersEstimator(), ctx2)
        @test fitted === ctx3.phylogeny
        @test !isnothing(ctx3.phylogeny)
        @test ctx3.prior === ctx2.prior

        # optimisation step (naive, no solver required)
        fitted, ctx4 = PortfolioOptimisers.run_step(EqualWeighted(), ctx3)
        @test fitted === ctx4.opt
        @test fitted.w ≈ fill(0.2, 5)

        # steps require their slots
        @test_throws PortfolioOptimisers.IsNothingError PortfolioOptimisers.run_step(EmpiricalPrior(),
                                                                                     PortfolioOptimisers.PipelineContext())

        # non-steppable estimators are rejected with guidance
        @test_throws ArgumentError PortfolioOptimisers.run_step(Covariance(), ctx)
    end

    @testset "PipelineStep execution" begin
        rng = StableRNG(987654321)
        rd = ReturnsResult(; nx = string.("A", 1:5), X = randn(rng, 60, 5) / 100)
        ctx = PortfolioOptimisers.PipelineContext(; returns = rd)

        # wrapped callable: receives the context, its value lands in the declared slot
        ps = PipelineStep(; est = c -> prior(EmpiricalPrior(), c.returns),
                          reads = (:returns,), writes = :prior)
        fitted, ctx2 = PortfolioOptimisers.run_step(ps, ctx)
        @test fitted === ctx2.prior
        @test ctx2.prior.X == rd.X

        # declared reads are enforced
        @test_throws PortfolioOptimisers.IsNothingError PortfolioOptimisers.run_step(ps,
                                                                                     PortfolioOptimisers.PipelineContext())

        # wrapped estimators delegate to their family's run_step
        ps2 = PipelineStep(; est = EmpiricalPrior(), reads = (:returns,), writes = :prior)
        fitted2, ctx3 = PortfolioOptimisers.run_step(ps2, ctx)
        @test fitted2.mu ≈ fitted.mu
    end
end

#=
Issue #776. A stub of the prior arm of the ucs triple, for the Pipeline step that serves it.
It carries no `pe`, so a call that reached the returns-data base would raise instead of
quietly answering, and the sets it builds are read off `pr.mu` and `pr.sigma`, so a probe can
tell which prior the step saw.
=#
struct Ticket776PipelinePriorUCS <: PortfolioOptimisers.AbstractPriorUncertaintySetEstimator end
function PortfolioOptimisers.mu_ucs(::Ticket776PipelinePriorUCS,
                                    pr::PortfolioOptimisers.AbstractPriorResult; kwargs...)
    return BoxUncertaintySet(; lb = pr.mu .- 1, ub = pr.mu .+ 1, val = pr.mu)
end
function PortfolioOptimisers.sigma_ucs(::Ticket776PipelinePriorUCS,
                                       pr::PortfolioOptimisers.AbstractPriorResult;
                                       kwargs...)
    return BoxUncertaintySet(; lb = pr.sigma .- 1, ub = pr.sigma .+ 1, val = pr.sigma)
end
function PortfolioOptimisers.ucs(ue::Ticket776PipelinePriorUCS,
                                 pr::PortfolioOptimisers.AbstractPriorResult; kwargs...)
    return PortfolioOptimisers.mu_ucs(ue, pr), PortfolioOptimisers.sigma_ucs(ue, pr)
end

@testset "Pipeline fit" begin
    using Test, PortfolioOptimisers, TimeSeries, Dates, StableRNGs, Statistics, Clarabel

    function make_prices(; T = 60, N = 5)
        rng = StableRNG(123456789)
        ts = range(Date(2020, 1, 1); step = Day(1), length = T)
        return TimeArray(ts, 100 .+ cumsum(randn(rng, T, N) / 10; dims = 1),
                         string.("A", 1:N))
    end
    function make_returns(; T = 60, N = 5)
        rng = StableRNG(987654321)
        return ReturnsResult(; nx = string.("A", 1:N), X = randn(rng, T, N) / 100)
    end
    # assets built from a known 5×2 loading matrix, so a factor mandate has something
    # to bite on and the regression recovers a well-conditioned `M`
    function make_factor_returns(; T = 60, N = 5)
        rng = StableRNG(987654321)
        F = randn(rng, T, 2) / 100
        M = [1.0 0.0; 0.5 0.5; 0.0 1.0; 0.8 0.2; 0.2 0.8]
        return ReturnsResult(; nx = string.("A", 1:N),
                             X = F * transpose(M) + randn(rng, T, N) / 1000,
                             nf = ["MTUM", "VLUE"], F = F)
    end
    jopt() = JuMPOptimiser(;
                           slv = Solver(; name = :Clarabel, solver = Clarabel.Optimizer,
                                        settings = Dict("verbose" => false)))

    @testset "construction and naming" begin
        pipe = Pipeline(; steps = (PricesToReturns(), EmpiricalPrior(), EqualWeighted()))
        @test pipe.names == ("returns", "prior", "opt")

        # repeated slots are suffixed in order; explicit names pass through
        pipe = Pipeline(;
                        steps = (MissingDataFilter(), "gap_fill" => PriceGapFill(),
                                 PricesToReturns()))
        @test pipe.names == ("prices_1", "gap_fill", "returns")
        pipe = Pipeline(; steps = (MissingDataFilter(), PriceGapFill()))
        @test pipe.names == ("prices_1", "prices_2")

        # steps cannot be empty
        @test_throws PortfolioOptimisers.IsEmptyError Pipeline(; steps = ())

        # duplicate names are rejected
        @test_throws ArgumentError Pipeline(;
                                            steps = ("a" => PriceGapFill(),
                                                     "a" => MissingDataFilter()))

        # non-steppable estimators are rejected at construction
        @test_throws ArgumentError Pipeline(; steps = (Covariance(),))

        # declared reads must be writable by an earlier step or the input
        ps = PipelineStep(; est = c -> c.prior, reads = (:prior,), writes = :prior)
        @test_throws ArgumentError Pipeline(; steps = (ps,))
        pipe = Pipeline(; steps = (EmpiricalPrior(), ps))
        @test pipe.names == ("prior_1", "prior_2")
    end

    @testset "fit at returns level" begin
        rd = make_returns()
        pipe = Pipeline(; steps = (EmpiricalPrior(), EqualWeighted()))
        res = fit(pipe, rd)
        @test res isa PipelineResult
        @test res["prior"] === res.results[1]
        @test res.ctx.prior.X == rd.X
        @test res.ctx.returns === rd
        @test res.w ≈ fill(0.2, 5)
        @test res["opt"].w ≈ fill(0.2, 5)
        @test_throws ArgumentError res["nope"]

        # a pipeline without an optimisation step has no weights
        res = fit(Pipeline(; steps = (EmpiricalPrior(),)), rd)
        @test isnothing(res.ctx.opt)
        @test_throws PortfolioOptimisers.PropertyPathError res.w
    end

    @testset "fit at prices level with injection" begin
        X = make_prices()
        pr = PricesResult(; X = X)
        pipe = Pipeline(;
                        steps = (MissingDataFilter(; col_thr = 0.5), PriceGapFill(),
                                 PricesToReturns(), EmpiricalPrior(),
                                 HierarchicalRiskParity()))
        res = fit(pipe, pr)
        @test size(res.ctx.returns.X) == (59, 5)
        @test length(res.w) == 5
        @test sum(res.w) ≈ 1

        # the shared prior is injected: identical to manually configuring the optimiser
        rd = prices_to_returns(X)
        pr_manual = prior(EmpiricalPrior(), rd)
        w_manual = optimise(HierarchicalRiskParity(;
                                                   opt = HierarchicalOptimiser(;
                                                                               pe = pr_manual)),
                            rd).w
        @test res.w ≈ w_manual
    end

    @testset "inject_context: prior and phylogeny" begin
        rd = make_returns()
        pr = prior(EmpiricalPrior(), rd)
        cl = clusterise(ClustersEstimator(), rd)
        ctx = PortfolioOptimisers.PipelineContext(; returns = rd, prior = pr,
                                                  phylogeny = cl)

        # hierarchical: pe and cle both overridden
        hrp2 = PortfolioOptimisers.inject_context(HierarchicalRiskParity(), ctx)
        @test hrp2.opt.pe === pr
        @test hrp2.opt.cle === cl

        # JuMP: pe overridden, phylogeny ignored (enters via constraint results)
        mr2 = PortfolioOptimisers.inject_context(MeanRisk(; opt = jopt()), ctx)
        @test mr2.opt.pe === pr

        # PipelineStep-wrapped optimisers are injected too
        ps = PipelineStep(; est = HierarchicalRiskParity(), reads = (:returns,),
                          writes = :opt)
        ps2 = PortfolioOptimisers.maybe_inject_step(ps, ctx)
        @test ps2.est.opt.pe === pr

        # naive optimisers have nothing to inject and ignore prior/phylogeny
        ew2 = PortfolioOptimisers.inject_context(EqualWeighted(), ctx)
        @test ew2 === EqualWeighted()
    end

    @testset "a prior step reads the source of its own prior (#1659, #1702)" begin
        # `inject_context` puts the Prior Result of a prior step into the optimiser
        # unchanged. The prior states its own source, so the step fits in the frame that
        # its `x_src` names, as the same prior does in the field `pe` of the optimiser.
        rd0 = make_returns()
        rd = ReturnsResult(; nx = rd0.nx, X = rd0.X, nb = ["BM"],
                           B = randn(StableRNG(1659), size(rd0.X, 1)) / 100)
        step(src) = PipelineStep(; est = EmpiricalPrior(; x_src = src), writes = :prior)
        pe(src) = EmpiricalPrior(; x_src = src)
        jo(src) = JuMPOptimiser(; pe = pe(src),
                                slv = Solver(; name = :Clarabel,
                                             solver = Clarabel.Optimizer,
                                             settings = Dict("verbose" => false)))
        ho(src) = HierarchicalOptimiser(; pe = pe(src),
                                        cle = ClustersEstimator(; x_src = src))
        cases = (src -> MeanRisk(; opt = jo(src)),
                 src -> PipelineStep(; est = MeanRisk(; opt = jo(src)), reads = (:returns,),
                                     writes = :opt),
                 src -> HierarchicalRiskParity(; opt = ho(src)),
                 src -> InverseVolatility(; pe = pe(src)),
                 src -> NestedClustered(; pe = pe(src),
                                        cle = ClustersEstimator(; x_src = src),
                                        opti = InverseVolatility(),
                                        opto = InverseVolatility()))
        for mk in cases, src in (nothing, ExcessSource())
            # A step in the frame of the optimiser's own prior equals the optimiser with no
            # prior step.
            opt = mk(src)
            pipe = @test_logs min_level = Logging.Warn Pipeline(; steps = (step(src), opt))
            res = fit(pipe, rd)
            ref = fit(Pipeline(; steps = (opt,)), rd)
            # The Prior Result of the step records its source, so the outer problem of a
            # meta-optimiser reads it in its frame (#1705).
            @test isapprox(res.w, ref.w; atol = 1e-6)
            @test res.ctx.prior.mu ≈ prior(pe(src), rd).mu
            # The returns data keeps its raw `X` and its `B`.
            @test res.ctx.returns === rd
        end
        # The prior of the step replaces the prior of the optimiser, so the reader at
        # `nothing` in `pe` leaves the group, and a mix that the caller stated is silent.
        @test_logs min_level = Logging.Warn Pipeline(;
                                                     steps = (step(ExcessSource()),
                                                              cases[1](nothing)))
        hdata = HierarchicalRiskParity(;
                                       opt = HierarchicalOptimiser(;
                                                                   cle = ClustersEstimator(;
                                                                                           x_src = DataSource())))
        @test_logs min_level = Logging.Warn Pipeline(;
                                                     steps = (step(ExcessSource()), hdata))
        # A reader that states no source beside a step of the Excess Frame warns once, with
        # its path from the pipeline, and the pipeline builds.
        hraw = HierarchicalRiskParity(; opt = HierarchicalOptimiser())
        @test_logs (:warn, r"^`Pipeline` holds a reader .*: `steps\[2\]\.opt\.cle`\.") Pipeline(;
                                                                                                steps = (step(ExcessSource()),
                                                                                                         hraw))
        hexc = HierarchicalRiskParity(;
                                      opt = HierarchicalOptimiser(;
                                                                  cle = ClustersEstimator(;
                                                                                          x_src = ExcessSource())))
        @test_logs (:warn, r": `steps\[1\]`\.") Pipeline(; steps = (EmpiricalPrior(), hexc))
        @test_logs (:warn, r": `steps\[1\]\.est`\.") Pipeline(;
                                                              steps = (step(nothing), hexc))
        # The prior of a nested pipeline is the prior of its last step.
        @test_logs (:warn, r": `steps\[1\]\.steps\[1\]\.est`\.") Pipeline(;
                                                                          steps = (Pipeline(;
                                                                                            steps = (step(nothing),)),
                                                                                   hexc))
        @test_logs min_level = Logging.Warn Pipeline(;
                                                     steps = (Pipeline(;
                                                                       steps = (step(ExcessSource()),)),
                                                              hexc))
        # The pipeline did not read the benchmark itself: the batch fit equals a prior on
        # the excess returns.
        rdx = ReturnsResult(; nx = rd.nx,
                            X = benchmark_relative_returns(rd, ExcessSource()))
        @test isapprox(fit(Pipeline(; steps = (step(ExcessSource()), InverseVolatility())),
                           rd).w, optimise(InverseVolatility(), rdx).w)
        # An optimiser that reads no prior, and a callable prior step, state no source.
        @test Pipeline(; steps = (step(ExcessSource()), EqualWeighted())) isa Pipeline
        callable = PipelineStep(; est = ctx -> prior(EmpiricalPrior(), ctx.returns),
                                reads = (:returns,), writes = :prior)
        @test Pipeline(; steps = (callable, cases[1](ExcessSource()))) isa Pipeline
        # The wrapper holds no source: `brt` is gone with no deprecated keyword (ADR 0213).
        @test !hasfield(PipelineStep, :brt)
        @test_throws MethodError PipelineStep(; est = EmpiricalPrior(), writes = :prior,
                                              brt = true)
        # `ExcessSource()` on returns with no benchmark is refused by the step, as by
        # `prior`.
        @test_throws ArgumentError fit(Pipeline(;
                                                steps = (step(ExcessSource()),
                                                         InverseVolatility())), rd0)
    end

    @testset "a prior step in the Excess Frame refuses a data benchmark (#1692, #1702)" begin
        # The optimiser receives the Prior Result of the step, so its own constructor
        # cannot see the source of the step. The `Pipeline` constructor refuses the pair,
        # because the tracking series would be `(X - B) * w - B`.
        slv = Solver(; name = :Clarabel, solver = Clarabel.Optimizer,
                     settings = Dict("verbose" => false))
        step(src) = PipelineStep(; est = EmpiricalPrior(; x_src = src), writes = :prior)
        dbt = DataBenchmarkTracking()
        opts = ((MeanRisk(;
                          opt = JuMPOptimiser(; slv = slv,
                                              tr = TrackingError(; tr = dbt, err = 0.01))),
                 "`opt.tr.tr`"),
                (MeanRisk(; r = TrackingRiskMeasure(; tr = dbt),
                          opt = JuMPOptimiser(; slv = slv)), "`r.tr`"),
                (MeanRisk(; r = [Variance(), TrackingRiskMeasure(; tr = dbt)],
                          opt = JuMPOptimiser(; slv = slv)), "`r[2].tr`"))
        for (opt, path) in opts
            err = try
                Pipeline(; steps = (step(ExcessSource()), opt))
                nothing
            catch e
                e
            end
            @test isa(err, ArgumentError)
            @test occursin("step 1 of 2", err.msg) && occursin(path, err.msg)
            # The prior of a nested pipeline and a wrapped optimiser are seen through.
            @test_throws ArgumentError Pipeline(;
                                                steps = (Pipeline(;
                                                                  steps = (step(ExcessSource()),)),
                                                         PipelineStep(; est = opt,
                                                                      reads = (:returns,),
                                                                      writes = :opt)))
            # A raw prior step, and a callable prior step, which states no source, build.
            for st in (step(nothing), step(DataSource()), EmpiricalPrior(),
                       PipelineStep(; est = ctx -> prior(EmpiricalPrior(), ctx.returns),
                                    reads = (:returns,), writes = :prior))
                @test Pipeline(; steps = (st, opt)) isa Pipeline
            end
            # The last prior step is the one whose Prior Result the optimiser receives.
            @test Pipeline(; steps = (step(ExcessSource()), step(nothing), opt)) isa
                  Pipeline
        end
        # A plain measure beside the excess prior step builds.
        @test Pipeline(;
                       steps = (step(ExcessSource()),
                                MeanRisk(;
                                         r = TrackingRiskMeasure(;
                                                                 tr = WeightsTracking(;
                                                                                      w = fill(0.2,
                                                                                               5))),
                                         opt = JuMPOptimiser(; slv = slv)))) isa Pipeline
    end

    @testset "every step that reads the returns reads its own source (#1670, #1702)" begin
        # An uncertainty, clustering, phylogeny, centrality or exposure step reads the
        # returns from the source of its own reader, and `inject_context` puts its value
        # into the optimiser. A step equals the same estimator in the field of the
        # optimiser.
        rng = StableRNG(987654321)
        F = randn(rng, 60, 2) / 100
        M = [1.0 0.0; 0.5 0.5; 0.0 1.0; 0.8 0.2; 0.2 0.8]
        rd = ReturnsResult(; nx = string.("A", 1:5),
                           X = F * transpose(M) + randn(rng, 60, 5) / 100,
                           nf = ["MTUM", "VLUE"], F = F, nb = ["BM"],
                           B = randn(StableRNG(1670), 60) / 100)
        pe(src) = EmpiricalPrior(; x_src = src)
        pstep(src) = PipelineStep(; est = pe(src), writes = :prior)
        sets = UniverseSets(; dict = Dict("nx" => rd.nx, "nf" => rd.nf))
        ue(src) = DeltaUncertaintySet(; pe = pe(src))
        ple(src) = SemiDefinitePhylogenyEstimator(; x_src = src)
        cte(src) = CentralityConstraint(; x_src = src)
        lce = LinearConstraintEstimator(; val = "MTUM <= 0.3")
        ece(src) = ExposureConstraintEstimator(; lce = lce,
                                               space = FactorSpace(;
                                                                   re = StepwiseRegression(;
                                                                                           x_src = src)))
        jo(src; kw...) = JuMPOptimiser(; pe = pe(src), slv = jopt().slv, kw...)
        ho(src; kw...) = HierarchicalOptimiser(; pe = pe(src), kw...)
        # (the step, the optimiser that receives it, the optimiser with the estimator)
        cases = (src -> (PipelineStep(; est = ue(src), reads = (:returns,),
                                      writes = :uncertainty, target = :sigma),
                         MeanRisk(; r = UncertaintySetVariance(), opt = jo(src)),
                         MeanRisk(; r = UncertaintySetVariance(; ucs = ue(src)),
                                  opt = jo(src))),
                 src -> (PipelineStep(; est = ClustersEstimator(; x_src = src),
                                      writes = :phylogeny),
                         HierarchicalRiskParity(; opt = ho(src)),
                         HierarchicalRiskParity(;
                                                opt = ho(src;
                                                         cle = ClustersEstimator(;
                                                                                 x_src = src)))),
                 src -> (PipelineStep(; est = ple(src), writes = :constraints),
                         MeanRisk(; opt = jo(src)),
                         MeanRisk(; opt = jo(src; ple = ple(src)))),
                 src -> (PipelineStep(; est = cte(src), writes = :constraints),
                         MeanRisk(; opt = jo(src)),
                         MeanRisk(; opt = jo(src; cte = cte(src)))),
                 src -> (PipelineStep(; est = ece(src), writes = :constraints),
                         MeanRisk(; opt = jo(src)),
                         MeanRisk(; opt = jo(src; lcse = ece(src), sets = sets))))
        for mk in cases
            for src in (nothing, ExcessSource())
                st, opt, ref = mk(src)
                # The value of the step replaces the slot that holds the default reader of
                # the optimiser, so the group of a matched step is silent.
                pipe = @test_logs min_level = Logging.Warn Pipeline(;
                                                                    steps = (pstep(src), st,
                                                                             opt))
                res = fit(pipe, rd)
                @test isapprox(res.w, optimise(ref, rd).w; atol = 1e-6)
            end
            # A step that states no source beside a prior of the Excess Frame warns, with
            # its path, and the pipeline builds.
            opt = mk(ExcessSource())[2]
            @test_logs (:warn, r"`steps\[2\]\.est") Pipeline(;
                                                             steps = (pstep(ExcessSource()),
                                                                      mk(nothing)[1], opt))
        end
        # The value of each step is the verb under the source of its reader.
        res = fit(Pipeline(; steps = (pstep(ExcessSource()), cases[1](ExcessSource())[1])),
                  rd)
        @test res.ctx.uncertainty.sigma.ub == sigma_ucs(ue(ExcessSource()), rd).ub
        @test res.ctx.uncertainty.sigma.ub != sigma_ucs(ue(nothing), rd).ub
        # A bare step that states no source warns in the same way.
        opt = cases[3](ExcessSource())[2]
        @test_logs (:warn, r": `steps\[2\]`\.") Pipeline(;
                                                         steps = (pstep(ExcessSource()),
                                                                  ple(nothing), opt))
        # A network step writes a phylogeny matrix that no optimiser receives, so it reads
        # its own source and stays out of the group.
        ne = PipelineStep(; est = NetworkEstimator(; x_src = ExcessSource()),
                          writes = :phylogeny)
        res = fit(Pipeline(; steps = (ne,)), rd)
        @test res.ctx.phylogeny.X ==
              phylogeny_matrix(NetworkEstimator(; x_src = ExcessSource()), rd; rd = rd).X
        hexc = HierarchicalRiskParity(;
                                      opt = ho(ExcessSource();
                                               cle = ClustersEstimator(;
                                                                       x_src = ExcessSource())))
        @test_logs min_level = Logging.Warn Pipeline(; steps = (NetworkEstimator(), hexc))
        # An optimiser with no field for `:cle` ignores a clustering, so the clustering
        # stays out of the group.
        @test_logs min_level = Logging.Warn Pipeline(;
                                                     steps = (ClustersEstimator(),
                                                              MeanRisk(;
                                                                       opt = jo(ExcessSource()))))
    end

    @testset "uncertainty steps and routing" begin
        rd = make_returns()
        ps_mu = PipelineStep(; est = DeltaUncertaintySet(), reads = (:returns,),
                             writes = :uncertainty, target = :mu)
        ps_sigma = PipelineStep(; est = DeltaUncertaintySet(), reads = (:returns,),
                                writes = :uncertainty, target = :sigma)
        pipe = Pipeline(; steps = (ps_mu, ps_sigma))
        @test pipe.names == ("uncertainty_1", "uncertainty_2")
        res = fit(pipe, rd)
        unc = res.ctx.uncertainty
        @test unc isa PortfolioOptimisers.PipelineUncertaintySets
        mu_ref = mu_ucs(DeltaUncertaintySet(), rd.X)
        sigma_ref = sigma_ucs(DeltaUncertaintySet(), rd.X)
        @test unc.mu.lb == mu_ref.lb
        @test unc.mu.ub == mu_ref.ub
        @test unc.sigma.lb == sigma_ref.lb
        @test unc.sigma.ub == sigma_ref.ub

        # routing: mu -> ret.ucs, sigma -> UncertaintySetVariance.ucs
        mr = MeanRisk(; opt = jopt(), r = UncertaintySetVariance())
        mr2 = PortfolioOptimisers.inject_context(mr, res.ctx)
        @test mr2.opt.ret.ucs === unc.mu
        @test mr2.r.ucs === unc.sigma

        # vector risk measures are searched for UncertaintySetVariance
        mr = MeanRisk(; opt = jopt(), r = [Variance(), UncertaintySetVariance()])
        mr2 = PortfolioOptimisers.inject_context(mr, res.ctx)
        @test mr2.r[2].ucs === unc.sigma
        @test mr2.r[1] === mr.r[1]

        # target = :both derives both halves from a single `ucs` call
        ps_both = PipelineStep(; est = DeltaUncertaintySet(), reads = (:returns,),
                               writes = :uncertainty, target = :both)
        res_both = fit(Pipeline(; steps = (ps_both,)), rd)
        unc_both = res_both.ctx.uncertainty
        @test unc_both isa PortfolioOptimisers.PipelineUncertaintySets
        @test unc_both.mu.lb == mu_ref.lb
        @test unc_both.mu.ub == mu_ref.ub
        @test unc_both.sigma.lb == sigma_ref.lb
        @test unc_both.sigma.ub == sigma_ref.ub

        # ... and agrees with the two narrowed steps run separately
        @test unc_both.mu.ub == unc.mu.ub
        @test unc_both.sigma.ub == unc.sigma.ub

        # the fitted object of a :both step is the pair itself
        @test res_both.results[1] === unc_both

        # :both requires an optimiser that can take *both* halves
        mr_both = PortfolioOptimisers.inject_context(MeanRisk(; opt = jopt(),
                                                              r = UncertaintySetVariance()),
                                                     res_both.ctx)
        @test mr_both.opt.ret.ucs === unc_both.mu
        @test mr_both.r.ucs === unc_both.sigma
        @test_throws ArgumentError PortfolioOptimisers.inject_context(MeanRisk(;
                                                                               opt = jopt(),
                                                                               r = Variance()),
                                                                      res_both.ctx)

        # bare uncertainty steps must declare a target
        ctx = PortfolioOptimisers.PipelineContext(; returns = rd)
        @test_throws ArgumentError PortfolioOptimisers.run_step(DeltaUncertaintySet(), ctx)
        ps_bad = PipelineStep(; est = DeltaUncertaintySet(), writes = :uncertainty)
        @test_throws ArgumentError PortfolioOptimisers.run_step(ps_bad, ctx)

        # An unknown target is rejected at construction, beside the allowlist `writes`
        # already gets, rather than by the step run after every earlier step has been fitted.
        @test_throws ArgumentError PipelineStep(; est = DeltaUncertaintySet(),
                                                reads = (:returns,), writes = :uncertainty,
                                                target = :nonsense)
        # `run_uncertainty_step` keeps its own check, because an unwrapped estimator reaches
        # it having declared no target at all.
        @test_throws ArgumentError PortfolioOptimisers.run_uncertainty_step(DeltaUncertaintySet(),
                                                                            :nonsense, ctx)

        # Issue #776. An estimator that reads the prior result serves the same three targets,
        # and the slot is the difference: it requires `:prior`, never `:returns`.
        @testset "an uncertainty step that reads the prior" begin
            ue776 = Ticket776PipelinePriorUCS()
            pr776 = prior(EmpiricalPrior(), rd)
            ctx776 = PortfolioOptimisers.PipelineContext(; returns = rd, prior = pr776)
            for (target, half) in ((:mu, :mu), (:sigma, :sigma))
                res776, ctx2 = PortfolioOptimisers.run_uncertainty_step(ue776, target,
                                                                        ctx776)
                # The set was fitted from the pipeline's own prior.
                ref = target == :mu ? pr776.mu : pr776.sigma
                @test res776.val == ref
                @test getproperty(ctx2.uncertainty, half).val == ref
                # A narrowed step leaves the other half alone.
                other = target == :mu ? :sigma : :mu
                @test isnothing(getproperty(ctx2.uncertainty, other))
            end
            # `:both` derives both halves from one read.
            pair776, ctx_both = PortfolioOptimisers.run_uncertainty_step(ue776, :both,
                                                                         ctx776)
            @test pair776 isa PortfolioOptimisers.PipelineUncertaintySets
            @test pair776.mu.val == pr776.mu
            @test pair776.sigma.val == pr776.sigma
            @test ctx_both.uncertainty === pair776
            # The two narrowed steps compose into the same pair.
            _, ctx_mu = PortfolioOptimisers.run_uncertainty_step(ue776, :mu, ctx776)
            _, ctx_two = PortfolioOptimisers.run_uncertainty_step(ue776, :sigma, ctx_mu)
            @test ctx_two.uncertainty.mu.val == pair776.mu.val
            @test ctx_two.uncertainty.sigma.val == pair776.sigma.val
            # The `:prior` slot is required, and the returns alone do not satisfy it.
            ctx_no_prior = PortfolioOptimisers.PipelineContext(; returns = rd)
            for target in (:mu, :sigma, :both)
                @test_throws PortfolioOptimisers.IsNothingError PortfolioOptimisers.run_uncertainty_step(ue776,
                                                                                                         target,
                                                                                                         ctx_no_prior)
            end
            # The returns are not read at all, so a context that carries the prior alone runs.
            ctx_prior_only = PortfolioOptimisers.PipelineContext(; prior = pr776)
            @test PortfolioOptimisers.run_uncertainty_step(ue776, :mu, ctx_prior_only)[1].val ==
                  pr776.mu
            # The target check is the returns-data method's, unchanged.
            @test_throws ArgumentError PortfolioOptimisers.run_uncertainty_step(ue776,
                                                                                :nonsense,
                                                                                ctx776)
            # A prior step ahead of it fills the slot, so the whole pipeline runs.
            ps776 = PipelineStep(; est = ue776, reads = (:prior,), writes = :uncertainty,
                                 target = :both)
            res_pipe = fit(Pipeline(; steps = (EmpiricalPrior(), ps776)), rd)
            @test res_pipe.ctx.uncertainty.mu.val == res_pipe.ctx.prior.mu
            @test res_pipe.ctx.uncertainty.sigma.val == res_pipe.ctx.prior.sigma

            # Issue #1015, ADR 0138. A returns-data estimator with `pe = nothing` reads the
            # prior slot on the same terms, decided by `reads_prior_result` rather than by
            # the root it subtypes.
            ue1015 = DeltaUncertaintySet(; pe = nothing)
            @test PortfolioOptimisers.reads_prior_result(ue1015)
            res1015, ctx1015 = PortfolioOptimisers.run_uncertainty_step(ue1015, :mu, ctx776)
            @test res1015.val === pr776.mu
            @test ctx1015.uncertainty.mu.val === pr776.mu
            @test PortfolioOptimisers.run_uncertainty_step(ue1015, :mu, ctx_prior_only)[1].val ==
                  pr776.mu
            @test_throws PortfolioOptimisers.IsNothingError PortfolioOptimisers.run_uncertainty_step(ue1015,
                                                                                                     :both,
                                                                                                     ctx_no_prior)
            # The same set with its own prior still reads the returns slot.
            @test PortfolioOptimisers.run_uncertainty_step(DeltaUncertaintySet(), :mu,
                                                           ctx_no_prior)[1].val == pr776.mu
            ps1015 = PipelineStep(; est = ue1015, reads = (:prior,), writes = :uncertainty,
                                  target = :both)
            res_pipe = fit(Pipeline(; steps = (EmpiricalPrior(), ps1015)), rd)
            @test res_pipe.ctx.uncertainty.mu.val === res_pipe.ctx.prior.mu
            @test res_pipe.ctx.uncertainty.sigma.val === res_pipe.ctx.prior.sigma
        end

        # unroutable targets fail loudly
        @test_throws ArgumentError fit(Pipeline(; steps = (ps_sigma, EqualWeighted())), rd)
        @test_throws ArgumentError fit(Pipeline(;
                                                steps = (ps_mu, HierarchicalRiskParity())),
                                       rd)
        # sigma with no UncertaintySetVariance in r
        @test_throws ArgumentError fit(Pipeline(;
                                                steps = (ps_sigma,
                                                         HierarchicalRiskParity())), rd)
    end

    @testset "constraint steps and routing" begin
        rd = make_returns()
        pipe = Pipeline(;
                        steps = (WeightBoundsEstimator(; lb = nothing, ub = 0.3),
                                 LinearConstraintEstimator(; val = "A1 <= 0.25"),
                                 EmpiricalPrior()))
        @test pipe.names == ("constraints_1", "constraints_2", "prior")
        res = fit(pipe, rd)
        cons = res.ctx.constraints
        @test cons isa Vector{PortfolioOptimisers.AbstractConstraintResult}
        @test cons[1] isa WeightBounds
        @test cons[2] isa LinearConstraint

        # routing into a JuMP optimiser
        mr2 = PortfolioOptimisers.inject_context(MeanRisk(; opt = jopt()), res.ctx)
        @test mr2.opt.wb === cons[1]
        @test mr2.opt.lcse === cons[2]

        # phylogeny-constraint estimators are steppable: they compute their own
        # phylogeny from :returns and route to the JuMP optimiser's `ple` field
        pipe_ph = Pipeline(; steps = (SemiDefinitePhylogenyEstimator(), EmpiricalPrior()))
        @test pipe_ph.names == ("constraints", "prior")
        res_ph = fit(pipe_ph, rd)
        @test res_ph.ctx.constraints isa SemiDefinitePhylogeny
        mr_ph = PortfolioOptimisers.inject_context(MeanRisk(; opt = jopt()), res_ph.ctx)
        @test mr_ph.opt.ple === res_ph.ctx.constraints

        # hierarchical optimisers accept weight bounds but not linear constraints
        ctx_wb = PortfolioOptimisers.PipelineContext(; returns = rd, constraints = cons[1])
        hrp2 = PortfolioOptimisers.inject_context(HierarchicalRiskParity(), ctx_wb)
        @test hrp2.opt.wb === cons[1]
        @test_throws ArgumentError PortfolioOptimisers.inject_context(HierarchicalRiskParity(),
                                                                      res.ctx)

        # unroutable constraint results fail loudly
        rb = PortfolioOptimisers.risk_budget_constraints(nothing; N = 5)
        ctx_rb = PortfolioOptimisers.PipelineContext(; returns = rd, constraints = rb)
        @test_throws ArgumentError PortfolioOptimisers.inject_context(MeanRisk(;
                                                                               opt = jopt()),
                                                                      ctx_rb)

        # naive optimisers carry an asset-dimensioned `wb` of their own, so a computed
        # weight bound reaches them directly rather than being rejected
        ew2 = PortfolioOptimisers.inject_context(EqualWeighted(), ctx_wb)
        @test ew2.wb === cons[1]
        # they still reject a target they have no field for
        ctx_lc = PortfolioOptimisers.PipelineContext(; returns = rd, constraints = cons[2])
        @test_throws ArgumentError PortfolioOptimisers.inject_context(EqualWeighted(),
                                                                      ctx_lc)
    end

    @testset "exposure constraint steps" begin
        #=
        A factor mandate as a bare pipeline step. The returns carry a factor block, so
        `pipeline_asset_sets` declares the `nf` axis and the step's names resolve
        against it; the basis is the pipeline prior's loadings, and what lands in the
        `constraints` slot is an ordinary asset-space LinearConstraint.
        =#
        rd = make_factor_returns()
        ece = ExposureConstraintEstimator(;
                                          lce = LinearConstraintEstimator(;
                                                                          val = "MTUM <= 0.3"),
                                          space = FactorSpace())
        pipe = Pipeline(; steps = (FactorPrior(), ece, MeanRisk(; opt = jopt())))
        @test pipe.names == ("prior", "constraints", "opt")
        res = fit(pipe, rd)
        lcr = res.ctx.constraints
        @test lcr isa LinearConstraint
        # projected into asset space: one row, one column per asset, not per factor
        @test size(lcr.ineq.A) == (1, length(rd.nx))

        #=
        The criterion is that the step buys nothing but convenience: generating the
        same constraint by hand against the same prior and passing it through `lcse`
        must give the same weights. It does, bit for bit, because the step calls the
        same generator with the same basis.
        =#
        pr = prior(FactorPrior(), rd)
        sets = UniverseSets(; dict = Dict("nx" => rd.nx, "nf" => rd.nf))
        lcr_direct = linear_constraints(ece, sets; rr = pr.rr)
        w_direct = optimise(MeanRisk(;
                                     opt = JuMPOptimiser(; pe = pr, lcse = lcr_direct,
                                                         slv = jopt().slv)), rd).w
        @test res.w == w_direct
        # and the mandate actually binds
        @test (transpose(pr.rr.M)*res.w)[1] ≈ 0.3 atol = 1e-6

        #=
        The `:prior` read is a construction-time declaration, so a pipeline with no
        prior step is rejected before any data is touched — the reads check fires,
        not `require_slot`.
        =#
        @test_throws ArgumentError Pipeline(; steps = (ece, MeanRisk(; opt = jopt())))

        #=
        A prior that carries no regression is a different failure: the slot is
        populated, so the pipeline constructs, and generation throws the
        missing-loadings error. It is deliberately not governed by `strict`.
        =#
        pipe_nolo = Pipeline(; steps = (EmpiricalPrior(), ece, MeanRisk(; opt = jopt())))
        @test_throws PortfolioOptimisers.IsNothingError fit(pipe_nolo, rd)
        # same error when the returns carry no factor block at all
        rd_nof = ReturnsResult(; nx = rd.nx, X = rd.X)
        @test_throws PortfolioOptimisers.IsNothingError fit(pipe_nolo, rd_nof)

        #=
        A wrapped vector produces sibling constraints rather than a nested one, so
        `constraint_targets` sees them alongside every other step's result and packs
        them into `lcse` as a vector.
        =#
        ece_vec = ExposureConstraintEstimator(;
                                              lce = [LinearConstraintEstimator(;
                                                                               val = "MTUM <= 0.3"),
                                                     LinearConstraintEstimator(;
                                                                               val = "VLUE >= 0.1")],
                                              space = FactorSpace())
        res_vec = fit(Pipeline(;
                               steps = (FactorPrior(), ece_vec, MeanRisk(; opt = jopt()))),
                      rd)
        cons_vec = res_vec.ctx.constraints
        @test cons_vec isa Vector{PortfolioOptimisers.AbstractConstraintResult}
        @test length(cons_vec) == 2
        @test all(c -> isa(c, LinearConstraint), cons_vec)
        f_w = transpose(pr.rr.M) * res_vec.w
        @test f_w[1] <= 0.3 + 1e-6
        @test f_w[2] >= 0.1 - 1e-6

        #=
        A row whose names all resolve away leaves the slot untouched instead of
        crashing: generation already decided the condition was recoverable and
        returned nothing, and the slot's job is to carry constraints, not to
        re-diagnose that.

        One warning, not two: an unresolved name takes its whole row (ADR 0125), so
        the row never reaches the empty-row report that used to say the same thing a
        second time.
        =#
        ece_none = ExposureConstraintEstimator(;
                                               lce = LinearConstraintEstimator(;
                                                                               val = "NOPE <= 0.3"),
                                               space = FactorSpace())
        res_none = (@test_logs (:warn,) fit(Pipeline(;
                                                     steps = (FactorPrior(), ece_none,
                                                              MeanRisk(; opt = jopt()))),
                                            rd))
        @test isnothing(res_none.ctx.constraints)
        @test isapprox(sum(res_none.w), 1)
        #=
        The same step in front of an optimiser with no `lcse` field is refused when the
        Pipeline is built, not when it runs. The step declares that it writes :lcse and
        nothing on an EqualWeighted can receive one; that this particular row happened to
        resolve away is a run-time accident, and the next row would not.
        =#
        @test_throws ArgumentError Pipeline(;
                                            steps = (FactorPrior(), ece_none,
                                                     EqualWeighted()))

        #=
        Two ordinary linear-constraint steps route into `lcse` as a vector of
        precomputed constraints, which the optimiser must pass through untouched.
        This is not exposure-specific — it was reachable before there was a
        re-basis — but the wrapped-vector step above lands on the same seam.
        =#
        res_two = fit(Pipeline(;
                               steps = (LinearConstraintEstimator(; val = "A1 <= 0.3"),
                                        LinearConstraintEstimator(; val = "A2 >= 0.1"),
                                        FactorPrior(), MeanRisk(; opt = jopt()))), rd)
        @test res_two.w[1] <= 0.3 + 1e-6
        @test res_two.w[2] >= 0.1 - 1e-6
    end

    @testset "nested pipelines and guards" begin
        X = make_prices()
        pr = PricesResult(; X = X)
        sub = Pipeline(; steps = (MissingDataFilter(), PricesToReturns()))
        @test PortfolioOptimisers.pipe_reads(sub) == (:prices,)
        @test PortfolioOptimisers.pipe_writes(sub) == :returns
        pipe = Pipeline(; steps = (sub, EmpiricalPrior(), EqualWeighted()))
        @test pipe.names == ("returns", "prior", "opt")
        res = fit(pipe, pr)
        @test res.results[1] isa PipelineResult
        @test size(res.ctx.returns.X) == (59, 5)
        @test res.w ≈ fill(0.2, 5)

        # a pipeline is not an optimiser and cannot be wrapped in one
        @test_throws ArgumentError optimise(pipe, prices_to_returns(X))
    end

    @testset "a nested pipeline changes no Data Slot that the outer pipeline misses" begin
        rng = StableRNG(20260928)
        rd = ReturnsResult(; nx = string.("A", 1:4), X = randn(rng, 60, 4) / 100)
        sel = ScoreSelector(; score = MeanReturn(), rule = RankRule(; best = 2))
        #=
        The selector inside cut the prior to two assets, and the outer returns kept four.
        The fit gave four weights over a two-asset prior, and the prediction then failed
        with an error that blamed the ingestion of the data.
        =#
        @test_throws ArgumentError Pipeline(;
                                            steps = (Pipeline(;
                                                              steps = (sel,
                                                                       EmpiricalPrior())),
                                                     EqualWeighted()))
        @test_throws ArgumentError Pipeline(;
                                            steps = (PipelineStep(;
                                                                  est = Pipeline(;
                                                                                 steps = (sel,
                                                                                          EmpiricalPrior())),
                                                                  writes = :prior),
                                                     EqualWeighted()))
        # The same steps in the outer pipeline describe one universe.
        flat = fit(Pipeline(; steps = (sel, EmpiricalPrior(), EqualWeighted())), rd)
        @test length(flat.ctx.returns.nx) == 2
        @test flat.w ≈ fill(0.5, 2)
        # A nested pipeline that reads returns receives the returns slot.
        nested = fit(Pipeline(;
                              steps = (Pipeline(; steps = (EmpiricalPrior(),)),
                                       EqualWeighted())), rd)
        @test nested.ctx.prior.mu ≈ vec(mean(rd.X; dims = 1))
        @test nested.ctx.prior.mu ≈ prior(EmpiricalPrior(), rd).mu
        @test nested.w ≈ fill(0.25, 4)
        # Returns input leaves the prices slot empty, so a price step cannot run on it.
        @test_throws PortfolioOptimisers.IsNothingError fit(Pipeline(;
                                                                     steps = (MissingDataFilter(),
                                                                              EmpiricalPrior())),
                                                            rd)
    end
end

using Test, PortfolioOptimisers, TimeSeries, Dates, StableRNGs

# a selector that never implements select_assets, for the erroring-fallback test
struct UnimplementedSelector <: PortfolioOptimisers.AbstractAssetSelector end
# a selector whose mask is one entry short, for the length check of the funnel
struct ShortMaskSelector <: PortfolioOptimisers.AbstractAssetSelector end
function PortfolioOptimisers.select_assets(::ShortMaskSelector,
                                           rd::PortfolioOptimisers.AbstractReturnsResult)
    return trues(size(rd.X, 2) - 1)
end

@testset "Asset selection" begin
    PO = PortfolioOptimisers

    # A: var 0.0233, B: constant, C: var 0.07, D: 3.3e-19
    XZ = [0.1 0.0 -0.2 0.0
          -0.1 0.0 0.3 1e-9
          0.2 0.0 -0.1 0.0]
    RDZ = ReturnsResult(; nx = ["A", "B", "C", "D"], X = XZ)

    @testset "rule construction validation" begin
        @test_throws PO.IsNothingError ThresholdRule()
        @test_throws DomainError ThresholdRule(; lo = 1.0, hi = 0.0)

        @test_throws PO.IsNothingError RankRule()
        @test_throws DomainError RankRule(; best = -1)
        @test_throws DomainError RankRule(; best = 0, worst = 0)
        @test_throws ArgumentError RankRule(; best = 1, action = :nonsense)

        @test_throws PO.IsNothingError QuantileRule()
        @test_throws DomainError QuantileRule(; best = 0.0)
        @test_throws DomainError QuantileRule(; worst = 1.0)
        @test_throws ArgumentError QuantileRule(; best = 0.5, action = :nonsense)
    end

    @testset "rule semantics" begin
        s = [1.0, 2.0, 3.0, 4.0, 5.0]

        # ordinal rules read orientation from bigger_is_better
        @test findall(PO.rule_keep(RankRule(; best = 2), s, false)) == [1, 2]
        @test findall(PO.rule_keep(RankRule(; best = 2), s, true)) == [4, 5]
        @test findall(PO.rule_keep(RankRule(; worst = 2), s, false)) == [4, 5]

        # action complements the whole selection
        @test findall(PO.rule_keep(RankRule(; worst = 1, action = :drop), s, false)) ==
              [1, 2, 3, 4]

        # both tails at once; :drop keeps the middle
        @test findall(PO.rule_keep(RankRule(; best = 1, worst = 1), s, false)) == [1, 5]
        @test findall(PO.rule_keep(RankRule(; best = 1, worst = 1, action = :drop), s,
                                   false)) == [2, 3, 4]

        # counts saturate rather than throwing, so a grid search over k survives
        @test all(PO.rule_keep(RankRule(; best = 99), s, false))

        # quantile bounds are fractions of the universe
        @test findall(PO.rule_keep(QuantileRule(; best = 0.4), s, false)) == [1, 2]
        @test findall(PO.rule_keep(QuantileRule(; worst = 0.2, action = :drop), s, false)) ==
              [1, 2, 3, 4]

        # thresholds are literal: they ignore bigger_is_better
        @test findall(PO.rule_keep(ThresholdRule(; lo = 1.5, hi = 4.5), s, false)) ==
              [2, 3, 4]
        @test PO.rule_keep(ThresholdRule(; lo = 1.5), s, false) ==
              PO.rule_keep(ThresholdRule(; lo = 1.5), s, true)

        # bounds are exclusive
        @test findall(PO.rule_keep(ThresholdRule(; lo = 1.0), s, false)) == [2, 3, 4, 5]
    end

    @testset "tie policy: excluded, never split" begin
        # a tied block straddling the cut is dropped entirely, so fewer than k survive
        @test findall(PO.rule_keep(RankRule(; best = 2), [1.0, 2.0, 2.0, 3.0], false)) ==
              [1]

        # a tied block that fits within k is kept whole
        @test findall(PO.rule_keep(RankRule(; best = 2), [1.0, 1.0, 3.0, 4.0], false)) ==
              [1, 2]

        # nothing is distinguishable, so nothing is selected
        @test !any(PO.rule_keep(RankRule(; best = 2), [7.0, 7.0, 7.0], false))

        # and an empty selection fails closed at fit time
        rd = ReturnsResult(; nx = ["A", "B", "C"], X = [1.0 1.0 1.0; -1.0 -1.0 -1.0])
        @test_throws PO.IsEmptyError fit_preprocessing(ScoreSelector(; score = SCM(),
                                                                     rule = RankRule(;
                                                                                     best = 2)),
                                                       rd)
    end

    @testset "ScoreSelector" begin
        # Variance is a WeightsInput measure: it cannot score one asset's return series
        @test_throws ArgumentError ScoreSelector(; score = Variance(),
                                                 rule = ThresholdRule(; lo = 0.0))
        @test_throws ArgumentError ScoreSelector(; score = StandardDeviation(),
                                                 rule = ThresholdRule(; lo = 0.0))
        err = try
            ScoreSelector(; score = Variance(), rule = ThresholdRule(; lo = 0.0))
        catch e
            e
        end
        @test occursin("SCM()", err.msg)

        # SCM scores each column's variance
        @test PO.asset_scores(SCM(), XZ) ≈
              [0.02333333333333333, 0.0, 0.07, 3.3333333333333335e-19]

        # keep the two highest-variance assets
        sel = ScoreSelector(; score = SCM(), rule = RankRule(; best = 2, action = :drop))
        @test fit_preprocessing(sel, RDZ).nx == ["A", "C"]

        # MeanReturn is maximised, so :best is the highest mean
        rd = ReturnsResult(; nx = ["A", "B", "C"], X = [0.1 0.2 0.3; 0.1 0.2 0.3])
        @test fit_preprocessing(ScoreSelector(; score = MeanReturn(),
                                              rule = RankRule(; best = 1)), rd).nx == ["C"]
        @test fit_preprocessing(ScoreSelector(; score = MeanReturn(),
                                              rule = RankRule(; worst = 1)), rd).nx == ["A"]

        # non-finite scores are rejected rather than sorted arbitrarily
        @test_throws DomainError PO.asset_scores(MaximumDrawdown(),
                                                 [NaN 0.1; NaN 0.2; NaN -0.1])
    end

    @testset "ZeroVarianceFilter alias" begin
        @test ZeroVarianceFilter() isa ScoreSelector
        @test fit_preprocessing(ZeroVarianceFilter(), RDZ).nx == ["A", "C"]

        # the bound is exclusive, so tol = 0 still drops an exactly-constant asset
        @test fit_preprocessing(ZeroVarianceFilter(; tol = 0.0), RDZ).nx == ["A", "C", "D"]
        @test_throws DomainError ZeroVarianceFilter(; tol = -1.0)
    end

    @testset "CompleteAssetSelector" begin
        rd = ReturnsResult(; nx = ["A", "B"], X = [0.1 0.2; 0.3 NaN])
        @test fit_preprocessing(CompleteAssetSelector(), rd).nx == ["A"]
        @test_throws PO.IsEmptyError fit_preprocessing(CompleteAssetSelector(),
                                                       ReturnsResult(; nx = ["A"],
                                                                     X = [0.1; NaN;;]))
    end

    @testset "fit/apply contract" begin
        res = fit_preprocessing(ZeroVarianceFilter(), RDZ)
        @test res isa AssetSelectorResult
        @test res isa PortfolioOptimisers.AbstractReturnsPreprocessingResult

        # apply sub-selects by name, in fitted order, even when the window reorders columns
        shuffled = ReturnsResult(; nx = ["C", "D", "B", "A"], X = XZ[:, [3, 4, 2, 1]])
        @test collect(apply_preprocessing(res, shuffled).nx) == ["A", "C"]

        # a fitted asset absent from the window throws rather than shrinking the universe
        partial = ReturnsResult(; nx = ["A", "B"], X = XZ[:, 1:2])
        @test_throws ArgumentError apply_preprocessing(res, partial)
        # the message names the missing asset and the window's size, never a whole universe (#1038)
        err = try
            apply_preprocessing(res, partial)
        catch e
            e
        end
        @test err.msg == PO.unknown_variable_msg("C", partial.nx, :nx;
                                                 consequence = "the window must contain the whole fitted universe")
        @test occursin("`C`", err.msg)
        @test occursin("2 assets", err.msg)
        @test !occursin(r"\bA\b", err.msg)
        @test !occursin(r"\bB\b", err.msg)

        # extra assets in the window are simply not selected
        extra = ReturnsResult(; nx = ["A", "B", "C", "D", "E"],
                              X = hcat(XZ, [1.0; 2.0; 3.0]))
        @test collect(apply_preprocessing(res, extra).nx) == ["A", "C"]

        # the erroring fallback names the missing method
        @test_throws ArgumentError fit_preprocessing(UnimplementedSelector(), RDZ)
    end

    @testset "tail_mask boundaries" begin
        # the tied block straddles the cut, so ONE asset is admitted, not two and not three
        s = [3.0, 2.0, 2.0, 1.0]
        @test findall(PO.tail_mask(s, 2, true, :best)) == [1]
        @test count(PO.tail_mask(s, 2, true, :best)) == 1

        # the tied block fits inside k, so both members are admitted
        @test findall(PO.tail_mask([3.0, 3.0, 1.0], 2, true, :best)) == [1, 2]

        # the count saturates at both ends
        @test !any(PO.tail_mask(s, 0, true, :best))
        @test !any(PO.tail_mask(s, -1, true, :best))
        @test all(PO.tail_mask(s, 4, true, :best))
        @test all(PO.tail_mask(s, 9, true, :best))

        # bib decides which raw end is `:best`, so the two ends swap under it
        r = [1.0, 2.0, 3.0, 4.0]
        @test PO.tail_mask(r, 2, true, :best) == PO.tail_mask(r, 2, false, :worst)
        @test PO.tail_mask(r, 2, false, :best) == PO.tail_mask(r, 2, true, :worst)
        @test findall(PO.tail_mask(r, 2, true, :best)) == [3, 4]
        @test findall(PO.tail_mask(r, 2, false, :best)) == [1, 2]
        # and the two `:best` masks are complements, because k = 2 halves a 4-asset universe
        @test PO.tail_mask(r, 2, true, :best) == .!PO.tail_mask(r, 2, false, :best)
    end

    @testset "rule boundaries" begin
        s = [1.0, 2.0, 3.0, 4.0, 5.0]

        # ThresholdRule compares strictly, so a score exactly on a bound is DROPPED
        @test findall(PO.rule_keep(ThresholdRule(; lo = 2.0), s, false)) == [3, 4, 5]
        @test findall(PO.rule_keep(ThresholdRule(; hi = 4.0), s, false)) == [1, 2, 3]

        # an omitted bound disables that side alone
        @test findall(PO.rule_keep(ThresholdRule(; hi = 3.0), s, false)) == [1, 2]

        # QuantileRule rounds an exact half UP: 0.05 of ten assets is one, not none
        s10 = collect(1.0:10.0)
        @test count(PO.rule_keep(QuantileRule(; best = 0.05), s10, false)) == 1
        @test count(PO.rule_keep(QuantileRule(; best = 0.04), s10, false)) == 0

        # `:drop` is the exact complement of `:keep` on the same rule and scores
        @test PO.rule_keep(RankRule(; best = 2, action = :drop), s, false) ==
              .!PO.rule_keep(RankRule(; best = 2), s, false)
        @test PO.rule_keep(QuantileRule(; best = 0.4, worst = 0.2, action = :drop), s,
                           false) ==
              .!PO.rule_keep(QuantileRule(; best = 0.4, worst = 0.2), s, false)
    end

    @testset "selection rule validators" begin
        @test_throws PO.IsNothingError PO.assert_tail_counts(nothing, nothing, :RankRule)
        @test_throws DomainError PO.assert_tail_counts(-1, nothing, :RankRule)
        @test_throws DomainError PO.assert_tail_counts(nothing, -1, :RankRule)
        @test_throws DomainError PO.assert_tail_counts(0, 0, :RankRule)
        # a zero on one side is legal: the rule still takes an asset from the other
        @test isnothing(PO.assert_tail_counts(0, 2, :RankRule))
        @test findall(PO.rule_keep(RankRule(; best = 0, worst = 2), [1.0, 2.0, 3.0, 4.0],
                                   false)) == [3, 4]

        @test_throws ArgumentError PO.assert_selection_action(:nonsense)
        @test isnothing(PO.assert_selection_action(:keep))
        @test isnothing(PO.assert_selection_action(:drop))
    end

    @testset "asset_scores and assert_scoreable" begin
        # the score of column i is `score(X[:, i])`, computed one column at a time
        @test PO.asset_scores(SCM(), XZ) == [SCM()(view(XZ, :, i)) for i in axes(XZ, 2)]

        # a NaN score is rejected: Skewness divides by a zero standard deviation
        Xc = [0.1 0.0; -0.1 0.0; 0.2 0.0]
        @test isnan(Skewness()(view(Xc, :, 2)))
        @test_throws DomainError PO.asset_scores(Skewness(), Xc)
        # a drawdown measure on a constant column is finite, so it is not that case
        @test isfinite(MaximumDrawdown()(view(Xc, :, 2)))

        # the message names the wrapper type, and only the two variance measures get the hint
        for m in (Variance(), StandardDeviation())
            e = try
                PO.assert_scoreable(m)
            catch err
                err
            end
            @test occursin(string(Base.typename(typeof(m)).wrapper), e.msg)
            @test occursin("SCM()", e.msg)
        end
        e = try
            PO.assert_scoreable(EqualRisk())
        catch err
            err
        end
        @test occursin("EqualRisk", e.msg)
        @test !occursin("SCM()", e.msg)
    end

    @testset "CompleteAssetSelector drops columns, never observations" begin
        X = [0.1 0.2 NaN 0.4
             0.3 0.2 0.1 0.5
             0.2 0.1 0.2 0.6]
        rd = ReturnsResult(; nx = ["A", "B", "C", "D"], X = X)
        res = fit_preprocessing(CompleteAssetSelector(), rd)
        @test res.nx == ["A", "B", "D"]
        @test size(apply_preprocessing(res, rd).X, 1) == size(X, 1)

        # a `missing` never reaches the selector: a returns carrier binds X to numbers
        @test_throws TypeError ReturnsResult(; nx = ["A", "B"],
                                             X = Union{Missing, Float64}[0.1 0.2
                                                                         0.3 missing])
    end

    #=
    Issue #860, ADR 0120: the pre-selection funnel reduces to the Coverage Universe before
    every selector, so a selector ranks among the assets that are live throughout the
    training window and never among one that is not yet listed, is delisted, or is missing a
    quote.

    The oracle is the hand-reduced panel: the same selector on the window with the gapped
    columns removed by hand must keep the same names.
    =#
    @testset "the funnel reduces to the Coverage Universe" begin
        rngc = StableRNG(20260907)
        Tc = 40
        # A is clean and volatile, B is clean and quiet, C, D and E each leave the universe
        # one of the three ways: an interior gap, a leading gap, a trailing gap.
        Xc = randn(rngc, Tc, 5) ./ 100
        Xc[:, 2] ./= 10                     # B is the quiet one, so a rank rule can order them
        Xgap = copy(Xc)
        Xgap[20, 3] = NaN                   # C: an interior gap, a missed quote
        Xgap[1:5, 4] .= NaN                 # D: a leading gap, not yet listed
        Xgap[(end - 4):end, 5] .= NaN       # E: a trailing gap, delisted
        nxc = ["A", "B", "C", "D", "E"]
        rdgap = ReturnsResult(; nx = nxc, X = Xgap)
        # the hand-reduced oracle: the two columns that are live throughout
        rdlive = ReturnsResult(; nx = ["A", "B"], X = Xc[:, 1:2])

        for sel in (ScoreSelector(; score = SCM(), rule = RankRule(; best = 1)),
                    CompleteAssetSelector(),
                    RedundancySelector(; alg = PairwiseCorrelation(; t = 0.9)))
            keep = fit_preprocessing(sel, rdgap).nx
            @test keep == fit_preprocessing(sel, rdlive).nx
            @test isempty(intersect(keep, ["C", "D", "E"]))
        end

        # A complete window gives the result of today: the reduction is the identity on it.
        # `bigger_is_better(SCM())` is false, so `best` is the lowest variance, which is B.
        @test fit_preprocessing(ScoreSelector(; score = SCM(), rule = RankRule(; best = 1)),
                                rdlive).nx == ["B"]

        # The panel's active mask is read too, so a stale finite price during an inactive
        # spell leaves the asset out. Every return here is finite.
        amsk = trues(Tc, 5)
        amsk[10, 2] = false
        pnl = AssetPanel(; pf = [NumericPanelField(; name = "mcap", vals = ones(Tc, 5))],
                         amsk = amsk, emsk = amsk)
        rdmsk = ReturnsResult(; nx = nxc, X = Xc, pnl = pnl)
        @test all(isfinite, Xc)
        @test fit_preprocessing(CompleteAssetSelector(), rdmsk).nx == ["A", "C", "D", "E"]

        # An all-dead window is refused where the mask is derived.
        dead = ReturnsResult(; nx = ["A", "B"], X = [0.1 NaN; NaN 0.2])
        @test_throws PO.IsEmptyError fit_preprocessing(CompleteAssetSelector(), dead)
        @test_throws PO.IsEmptyError fit_preprocessing(ScoreSelector(; score = SCM(),
                                                                     rule = RankRule(;
                                                                                     best = 1)),
                                                       dead)

        # The reduction happens once, in the funnel, so a selector that reads values sees a
        # clean block and `asset_scores` keeps its refusal for a live column.
        @test_throws DomainError PO.asset_scores(MaximumDrawdown(), [NaN 0.1; NaN 0.2])
    end

    @testset "the fitted universe replays" begin
        Xtr = [0.10 0.01 0.001; -0.10 -0.01 -0.001; 0.05 0.02 0.002]
        Xte = [0.001 0.02 0.10; -0.001 -0.02 -0.10; 0.002 0.01 0.05]
        rdtr = ReturnsResult(; nx = ["A", "B", "C"], X = Xtr)
        rdte = ReturnsResult(; nx = ["A", "B", "C"], X = Xte)
        sel = ScoreSelector(; score = SCM(), rule = RankRule(; best = 2))

        res = fit_preprocessing(sel, rdtr)
        @test res.nx == ["B", "C"]
        # the test window's own scores would choose a different universe
        @test fit_preprocessing(sel, rdte).nx == ["A", "B"]
        # the replay keeps the fitted one, which is what makes the selector safe in CV
        @test collect(apply_preprocessing(res, rdte).nx) == ["B", "C"]
    end

    @testset "the base of the family agrees with its docstrings" begin
        # CompleteAssetSelector keeps the Coverage Universe, which drops an Inf column too
        Xi = [1.0 Inf; 2.0 3.0]
        @test fit_preprocessing(CompleteAssetSelector(),
                                ReturnsResult(; nx = ["A", "B"], X = Xi)).nx == ["A"]

        # the fitted names keep the column order of the training window, not a sorted one
        rd = ReturnsResult(; nx = ["C", "A", "B"],
                           X = [0.1 0.2 0.0; -0.1 0.3 0.0; 0.2 -0.1 0.0])
        res = fit_preprocessing(ZeroVarianceFilter(), rd)
        @test res.nx == ["C", "A"]
        # and the replay reorders the window's columns into that order
        w = ReturnsResult(; nx = ["B", "A", "C", "D"],
                          X = [0.0 0.2 0.1 1.0; 0.0 0.3 -0.1 2.0; 0.0 -0.1 0.2 3.0])
        aw = apply_preprocessing(res, w)
        @test collect(aw.nx) == ["C", "A"]
        @test aw.X == w.X[:, [3, 2]]

        # a mask of the wrong length is refused, and the message states both lengths
        e = try
            fit_preprocessing(ShortMaskSelector(), rd)
        catch err
            err
        end
        @test e isa DimensionMismatch
        @test occursin("length 2 for the 3 asset columns", e.msg)

        # the unimplemented fallback names the subtype
        e = try
            fit_preprocessing(UnimplementedSelector(), rd)
        catch err
            err
        end
        @test occursin("UnimplementedSelector", e.msg)

        # a carrier with no asset axis matches no method of the reduction
        prr = PO.PredictionReturnsResult(; nx = ["P"], X = [0.1, -0.2, 0.3])
        @test_throws MethodError fit_preprocessing(CompleteAssetSelector(), prr)
    end

    @testset "pipeline integration" begin
        rng = StableRNG(1234)
        T, N = 120, 6
        ts = collect(Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(T - 1)))
        Xp = 100 .+ cumsum(0.01 .* randn(rng, T, N); dims = 1)
        Xp[:, 3] .= 100.0                      # C is constant
        pr = PricesResult(; X = TimeArray(ts, Xp, ["A", "B", "C", "D", "E", "F"]))

        # selectors are steppable with no pipeline-side code
        @test PortfolioOptimisers.pipe_writes(ZeroVarianceFilter()) == :returns
        @test PortfolioOptimisers.pipe_reads(ZeroVarianceFilter()) == (:returns,)

        pipe = Pipeline(;
                        steps = (MissingDataFilter(), PriceGapFill(), PricesToReturns(),
                                 ZeroVarianceFilter(), EmpiricalPrior(), EqualWeighted()))
        res = fit(pipe, pr)
        @test res.ctx.returns.nx == ["A", "B", "D", "E", "F"]
        @test length(res.ctx.opt.w) == 5

        # predict replays the fitted universe on an unseen window
        @test predict(res, pr, 61:120) isa Any

        # a selector after a returns-derived step is rejected at construction
        @test_throws ArgumentError Pipeline(;
                                            steps = (PricesToReturns(), EmpiricalPrior(),
                                                     ZeroVarianceFilter(), EqualWeighted()))
    end
end

using Test, PortfolioOptimisers, StableRNGs, Statistics, LinearAlgebra

# an algorithm that implements no redundancy_keep, for the erroring-fallback test
struct UnimplementedRedundancy <: PortfolioOptimisers.AbstractRedundancyAlgorithm end

@testset "Redundancy selection" begin
    POx = PortfolioOptimisers

    # A and B are perfectly correlated with each other and uncorrelated with C.
    # A has the lower variance, so a variance score keeps A over B.
    rd_pair = ReturnsResult(; nx = ["A", "B", "C"],
                            X = [0.010 0.020 -0.05
                                 -0.010 -0.020 0.07
                                 0.005 0.010 -0.02
                                 0.002 0.004 0.09])

    @testset "construction validation" begin
        @test_throws DomainError PairwiseCorrelation(; t = 1.5)
        @test_throws DomainError CorrelationComponents(; t = -1.5)

        # ClusterGroups has no fallback survivor rule, so it needs a score
        @test_throws POx.IsNothingError RedundancySelector(; alg = ClusterGroups())
        @test RedundancySelector(; alg = ClusterGroups(), score = SCM()) isa
              RedundancySelector

        # the correlation algorithms do have one
        @test RedundancySelector(; alg = PairwiseCorrelation()) isa RedundancySelector
        @test RedundancySelector(; alg = CorrelationComponents()) isa RedundancySelector

        # an unscoreable measure is rejected exactly as it is for ScoreSelector
        @test_throws ArgumentError RedundancySelector(; alg = PairwiseCorrelation(),
                                                      score = Variance())

        # the default is greedy pairwise pruning
        @test RedundancySelector().alg isa PairwiseCorrelation
        @test isnothing(RedundancySelector().score)
    end

    @testset "score orients the survivor" begin
        # lower variance is better, so A survives the A/B pair
        sel = RedundancySelector(; alg = PairwiseCorrelation(; t = 0.99), score = SCM())
        @test fit_preprocessing(sel, rd_pair).nx == ["A", "C"]

        # MeanReturn is maximised, so the higher-mean asset of the pair survives
        rd = ReturnsResult(; nx = ["A", "B", "C"],
                           X = [0.010 0.020 -0.05
                                -0.010 -0.020 0.07
                                0.005 0.010 -0.02
                                0.002 0.004 0.09])
        selm = RedundancySelector(; alg = PairwiseCorrelation(; t = 0.99),
                                  score = MeanReturn())
        kept = fit_preprocessing(selm, rd).nx
        @test "C" in kept
        @test length(kept) == 2
        @test kept != ["A", "C"]   # B has the larger mean, so B survives, not A
    end

    @testset "drop_scores flips with bigger_is_better" begin
        s = [1.0, 2.0, 3.0]
        @test POx.drop_scores(s, false) == s        # lower is better -> higher is drop-worthy
        @test POx.drop_scores(s, true) == -s        # higher is better -> lower is drop-worthy
    end

    @testset "greedy does not chain; components do" begin
        # rho(A,B) high, rho(B,C) high, rho(A,C) low
        rho = [1.0 0.97 0.10
               0.97 1.0 0.97
               0.10 0.97 1.0]

        # union-find over the strict lower triangle: one transitive blob
        comps = POx.correlation_components(rho, 0.95)
        @test length(comps) == 1
        @test sort(only(comps)) == [1, 2, 3]

        # below any edge threshold every asset is its own singleton
        @test length(POx.correlation_components(rho, 0.99)) == 3
    end

    @testset "groups_argbest and the trust-neither tie policy" begin
        # one survivor per group: the best-scoring member
        @test findall(POx.groups_argbest([[1, 2], [3]], [5.0, 1.0, 9.0], false)) == [2, 3]
        @test findall(POx.groups_argbest([[1, 2], [3]], [5.0, 1.0, 9.0], true)) == [1, 3]

        # a group whose best score is tied keeps nobody
        @test findall(POx.groups_argbest([[1, 2], [3]], [4.0, 4.0, 9.0], false)) == [3]

        # a singleton is always unambiguous
        @test findall(POx.groups_argbest([[1], [2]], [4.0, 4.0], false)) == [1, 2]
    end

    @testset "identical columns leave no survivor" begin
        # A and B are the same series: neither the correlation summary nor a variance score
        # can tell them apart, so both are discarded
        rd = ReturnsResult(; nx = ["A", "B", "C"],
                           X = [0.10 0.10 -0.05
                                -0.10 -0.10 0.07
                                0.05 0.05 -0.02
                                0.02 0.02 0.09])
        @test fit_preprocessing(RedundancySelector(; alg = PairwiseCorrelation(; t = 0.99),
                                                   score = SCM()), rd).nx == ["C"]
        @test fit_preprocessing(RedundancySelector(;
                                                   alg = CorrelationComponents(; t = 0.99),
                                                   score = SCM()), rd).nx == ["C"]
    end

    @testset "find_uncorrelated_indices gains a scores keyword" begin
        rng = StableRNG(42)
        X = randn(rng, 200, 6)
        X[:, 2] = X[:, 1] .+ 0.01 .* randn(rng, 200)   # 2 duplicates 1

        # default behaviour is unchanged
        base = POx.find_uncorrelated_indices(X; t = 0.9)
        @test 1 in base || 2 in base
        @test !(1 in base && 2 in base)

        # supplying scores overrides the survivor rule: higher score is dropped
        scores = [10.0, 1.0, 0.0, 0.0, 0.0, 0.0]      # asset 1 is drop-worthy
        withs = POx.find_uncorrelated_indices(X; t = 0.9, scores = scores)
        @test 2 in withs
        @test !(1 in withs)

        # and it is length-checked
        @test_throws DimensionMismatch POx.find_uncorrelated_indices(X; t = 0.9,
                                                                     scores = [1.0, 2.0])
    end

    @testset "ClusterGroups keeps one representative per cluster" begin
        rng = StableRNG(7)
        # two tight blocks of three assets each
        f1 = randn(rng, 150)
        f2 = randn(rng, 150)
        X = hcat(f1 .+ 0.01 .* randn(rng, 150, 3), f2 .+ 0.01 .* randn(rng, 150, 3))
        rd = ReturnsResult(; nx = string.('A':'F'), X = X)

        sel = RedundancySelector(; alg = ClusterGroups(), score = SCM())
        kept = fit_preprocessing(sel, rd).nx
        @test length(kept) <= 6
        @test !isempty(kept)
        # one survivor per cluster, so far fewer than the full universe
        @test length(kept) < 6
    end

    @testset "requires_score and the erroring fallback" begin
        # the generic method answers `true`, so a new algorithm is asked for a score
        @test POx.requires_score(UnimplementedRedundancy())
        @test POx.requires_score(ClusterGroups())
        @test !POx.requires_score(PairwiseCorrelation())
        @test !POx.requires_score(CorrelationComponents())

        # ClusterGroups needs a score, and the message says so
        err = try
            RedundancySelector(; alg = ClusterGroups())
        catch e
            e
        end
        @test err isa POx.IsNothingError
        msg = sprint(showerror, err)
        @test occursin("ClusterGroups", msg)
        @test occursin("give the RedundancySelector a score", msg)

        # an algorithm implementing no redundancy_keep hits the family's erroring fallback
        rd = ReturnsResult(; nx = ["A", "B"], X = [0.1 0.2; -0.1 0.3; 0.2 0.1])
        @test_throws ArgumentError POx.redundancy_keep(UnimplementedRedundancy(), rd,
                                                       [1.0, 2.0], false)
        e2 = try
            POx.redundancy_keep(UnimplementedRedundancy(), rd, nothing, false)
        catch e
            e
        end
        @test occursin("does not implement redundancy_keep", e2.msg)
        @test occursin("requires_score", e2.msg)
    end

    @testset "correlation_components partitions the universe" begin
        rho = [1.0 0.97 0.10 0.05
               0.97 1.0 0.12 0.02
               0.10 0.12 1.0 0.01
               0.05 0.02 0.01 1.0]
        comps = POx.correlation_components(rho, 0.95)
        # every index lands in exactly one component
        @test sort(vcat(comps...)) == 1:4
        @test sort(sort.(comps)) == [[1, 2], [3], [4]]
        # an asset with no over-threshold partner is a singleton
        @test count(g -> length(g) == 1, comps) == 2

        # `absolute` joins a strongly NEGATIVE pair, and the raw comparison does not
        rneg = [1.0 -0.97; -0.97 1.0]
        @test length(POx.correlation_components(rneg, 0.95)) == 2
        @test length(POx.correlation_components(abs.(rneg), 0.95)) == 1

        # thresholds outside a correlation's range are rejected; both endpoints are legal
        @test isnothing(POx.assert_correlation_threshold(-1.0))
        @test isnothing(POx.assert_correlation_threshold(1.0))
        @test_throws DomainError POx.assert_correlation_threshold(-1.0001)
        @test_throws DomainError POx.assert_correlation_threshold(1.0001)
    end

    @testset "transitivity: components chain, pairwise pruning does not" begin
        # the docstring's own chain, and the only kind that is realisable: two edges at rho
        # force the third above rho1*rho2 - sqrt((1 - rho1^2)(1 - rho2^2))
        rho = [1.0 0.80 0.32
               0.80 1.0 0.81
               0.32 0.81 1.0]
        @test minimum(eigvals(rho)) > 0      # a correlation matrix, not just a table

        rng = StableRNG(20260828)
        Z = Matrix(qr(randn(rng, 500, 3)).Q)[:, 1:3]
        Z = (Z .- mean(Z; dims = 1)) ./ std(Z; dims = 1)
        X = Z * cholesky(rho).L'
        rd = ReturnsResult(; nx = ["A", "B", "C"], X = X)
        @test isapprox(cor(X), rho; atol = 5e-3)

        # union-find reads the chain as ONE component of three
        comps = POx.correlation_components(cor(X), 0.7)
        @test length(comps) == 1
        @test sort(only(comps)) == [1, 2, 3]

        # so the components algorithm keeps one asset, and greedy pruning keeps more
        cc = POx.select_assets(RedundancySelector(; alg = CorrelationComponents(; t = 0.7)),
                               rd)
        pw = POx.select_assets(RedundancySelector(; alg = PairwiseCorrelation(; t = 0.7)),
                               rd)
        @test count(cc) == 1
        @test count(pw) == 2
        @test count(pw) > count(cc)
        # greedy drops the middle asset and keeps both ends, as PairwiseCorrelation states
        @test findall(pw) == [1, 3]
        @test findall(cc) == [1]
    end

    @testset "the fallback survivor, and a score that overrides it" begin
        rng = StableRNG(11)
        f = randn(rng, 400)
        h = randn(rng, 400)
        # A is tied to both f and h, but scaled small; B is tied to f alone, at full scale
        A = 0.1 .* (f .+ 0.05 .* randn(rng, 400) .+ 0.30 .* h)
        B = f .+ 0.05 .* randn(rng, 400)
        X = hcat(A, B, h)
        rd = ReturnsResult(; nx = ["A", "B", "C"], X = X)

        rho = cor(X)
        summ = [POx.vec_to_real_measure(MeanValue(), x) for x in eachcol(rho)]
        scm = POx.asset_scores(SCM(), X)
        # A and B are one component, and the two rules disagree inside it
        @test sort(sort.(POx.correlation_components(rho, 0.9))) == [[1, 2], [3]]
        @test summ[2] < summ[1]      # B is the least redundant of the pair
        @test scm[1] < scm[2]        # A has the lower variance

        # with no score the least redundant member survives
        @test findall(POx.select_assets(RedundancySelector(;
                                                           alg = CorrelationComponents(;
                                                                                       t = 0.9)),
                                        rd)) == [2, 3]
        # supplying a score overrides that choice
        @test findall(POx.select_assets(RedundancySelector(;
                                                           alg = CorrelationComponents(;
                                                                                       t = 0.9),
                                                           score = SCM()), rd)) == [1, 3]

        # the same nothing-score path through greedy pruning
        keep = POx.select_assets(RedundancySelector(; alg = PairwiseCorrelation(; t = 0.9)),
                                 rd)
        @test count(keep) == 2
        @test 3 in findall(keep)

        # `absolute` reaches the components algorithm too
        Xneg = hcat(f, -f .+ 0.01 .* randn(rng, 400), h)
        rdneg = ReturnsResult(; nx = ["A", "B", "C"], X = Xneg)
        @test count(POx.select_assets(RedundancySelector(;
                                                         alg = CorrelationComponents(;
                                                                                     t = 0.9,
                                                                                     absolute = true)),
                                      rdneg)) == 2
        @test count(POx.select_assets(RedundancySelector(;
                                                         alg = CorrelationComponents(;
                                                                                     t = 0.9,
                                                                                     absolute = false)),
                                      rdneg)) == 3
    end

    @testset "the drop-score sign reaches find_uncorrelated_indices" begin
        rng = StableRNG(5)
        f = randn(rng, 300)
        A = 0.01 .* f .+ 0.0002 .* randn(rng, 300)
        B = 0.01 .* f .+ 0.0002 .* randn(rng, 300) .+ 0.02   # B has the far larger mean
        X = hcat(A, B, 0.01 .* randn(rng, 300))
        rd = ReturnsResult(; nx = ["A", "B", "C"], X = X)

        means = POx.asset_scores(MeanReturn(), X)
        @test PortfolioOptimisers.bigger_is_better(MeanReturn())
        @test means[2] > means[1]
        @test cor(A, B) > 0.95

        # a bigger-is-better measure must keep the HIGH-scoring asset of the pair
        @test findall(POx.select_assets(RedundancySelector(;
                                                           alg = PairwiseCorrelation(;
                                                                                     t = 0.95),
                                                           score = MeanReturn()), rd)) ==
              [2, 3]
        # and a lower-is-better one keeps the low-scoring asset
        @test POx.asset_scores(SCM(), X)[1] < POx.asset_scores(SCM(), X)[2]
        @test findall(POx.select_assets(RedundancySelector(;
                                                           alg = PairwiseCorrelation(;
                                                                                     t = 0.95),
                                                           score = SCM()), rd)) == [1, 3]
    end

    @testset "groups_argbest skips an empty group" begin
        # an empty group contributes nothing and does not stop the walk
        @test findall(POx.groups_argbest([Int[], [1, 2]], [1.0, 2.0], false)) == [1]
        @test findall(POx.groups_argbest([[1, 2], Int[]], [1.0, 2.0], true)) == [2]
        @test !any(POx.groups_argbest([Int[]], [1.0, 2.0], false))
    end

    @testset "pipeline integration" begin
        @test PortfolioOptimisers.pipe_writes(RedundancySelector()) == :returns
        @test PortfolioOptimisers.pipe_reads(RedundancySelector()) == (:returns,)

        rng = StableRNG(99)
        f = randn(rng, 200)
        X = hcat(0.01 .* f, 0.01 .* f .+ 1e-6 .* randn(rng, 200),
                 0.01 .* randn(rng, 200, 3))
        rd = ReturnsResult(; nx = string.('A':'E'), X = X)

        pipe = Pipeline(;
                        steps = (RedundancySelector(; alg = PairwiseCorrelation(; t = 0.95),
                                                    score = SCM()), EmpiricalPrior(),
                                 EqualWeighted()))
        res = fit(pipe, rd)
        @test length(res.ctx.returns.nx) < 5
        @test length(res.ctx.opt.w) == length(res.ctx.returns.nx)

        # replaying the fitted universe on an unseen window
        @test predict(res, rd) isa Any
    end
end

@testset "Train/test split" begin
    using Test, PortfolioOptimisers, TimeSeries, Dates, StableRNGs

    safe_index = PortfolioOptimisers.safe_index

    function make_prices(; T = 100, N = 4)
        rng = StableRNG(987654321)
        ts = Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(T - 1))
        X = TimeArray(collect(ts), 100 .+ cumsum(rand(rng, T, N); dims = 1),
                      string.("A", 1:N))
        return PricesResult(; X = X)
    end
    function make_returns(; T = 100, N = 4)
        rng = StableRNG(987654321)
        ts = collect(Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(T - 1)))
        return ReturnsResult(; nx = string.("A", 1:N), X = 0.01 .* randn(rng, T, N),
                             ts = ts)
    end

    @testset "safe_index sizing" begin
        # neither size given: 75/25, and the two windows partition the data
        @test safe_index(nothing, nothing, 100) == (1:75, 76:100)

        # one side given: the other is its complement, counts and fractions alike
        @test safe_index(80, nothing, 100) == (1:80, 81:100)
        @test safe_index(0.8, nothing, 100) == (1:80, 81:100)
        @test safe_index(nothing, 20, 100) == (1:80, 81:100)
        @test safe_index(nothing, 0.2, 100) == (1:80, 81:100)

        # both given and summing to less than N: the middle rows are embargoed
        @test safe_index(0.6, 0.2, 100) == (1:60, 81:100)
        @test safe_index(60, 20, 100) == (1:60, 81:100)

        # both given and exactly covering the data: no embargo
        @test safe_index(70, 30, 100) == (1:70, 71:100)

        # overlapping windows are rejected
        @test_throws ArgumentError safe_index(0.9, 0.2, 100)
        @test_throws ArgumentError safe_index(90, 20, 100)

        # a window that would be empty is rejected rather than silently returned
        @test_throws ArgumentError safe_index(100, nothing, 100)
        @test_throws ArgumentError safe_index(nothing, 100, 100)

        # counts saturate, fractions must be a genuine fraction
        @test_throws ArgumentError safe_index(150, nothing, 100)  # saturates, empties test
        @test_throws DomainError safe_index(0.0, nothing, 100)
        @test_throws DomainError safe_index(1.0, nothing, 100)
        @test_throws DomainError safe_index(nothing, 1.5, 100)
        @test_throws DomainError safe_index(0, nothing, 100)
        @test_throws DomainError safe_index(nothing, -5, 100)
    end

    @testset "split_count boundaries" begin
        split_count = PortfolioOptimisers.split_count

        # A count saturates at the observation total.
        @test split_count(3, 10, :train_size) == 3
        @test split_count(10, 10, :train_size) == 10
        @test split_count(15, 10, :train_size) == 10

        # A fraction that lands exactly on an integer takes that integer, and one that
        # floors below one row is clamped up to one, so rounding alone never empties a
        # window.
        @test split_count(0.5, 10, :train_size) == 5
        @test split_count(0.01, 10, :train_size) == 1
        @test split_count(0.9999, 10, :train_size) == 9

        # A count must be positive, and a fraction must lie strictly inside (0, 1).
        @test_throws DomainError split_count(0, 10, :train_size)
        @test_throws DomainError split_count(-1, 10, :train_size)
        @test_throws DomainError split_count(0.0, 10, :train_size)
        @test_throws DomainError split_count(1.0, 10, :train_size)
    end

    @testset "train_test_split free function" begin
        pr = make_prices()
        rd = make_returns()

        tr, te = train_test_split(rd; test_size = 0.2)
        @test size(tr.X, 1) == 80
        @test size(te.X, 1) == 20
        @test tr.X == rd.X[1:80, :]
        @test te.X == rd.X[81:100, :]
        @test tr.nx == rd.nx == te.nx

        trp, tep = train_test_split(pr; test_size = 0.2)
        @test size(values(trp.X), 1) == 80
        @test size(values(tep.X), 1) == 20
        @test timestamp(tep.X) == timestamp(pr.X)[81:100]

        # the previously-broken branches now yield a non-empty test window
        tr2, te2 = train_test_split(rd)
        @test size(tr2.X, 1) == 75
        @test size(te2.X, 1) == 25
        tr3, te3 = train_test_split(rd; train_size = 60)
        @test size(tr3.X, 1) == 60
        @test size(te3.X, 1) == 40

        # embargo: the rows between the windows belong to neither
        tr4, te4 = train_test_split(rd; train_size = 0.6, test_size = 0.2)
        @test size(tr4.X, 1) == 60
        @test size(te4.X, 1) == 20
        @test te4.X == rd.X[81:100, :]
    end

    @testset "TrainTestSplit estimator" begin
        pr = make_prices()
        rd = make_returns()

        tts = TrainTestSplit(; test_size = 0.2)
        @test TTS === TrainTestSplit

        # the estimator form of the free function returns the same fitted result
        tsr = train_test_split(tts, rd)
        @test isa(tsr, TrainTestSplitResult)
        @test size(tsr.train.X, 1) == 80
        @test size(tsr.test.X, 1) == 20
        @test tsr.test.X == rd.X[81:100, :]
        @test size(values(train_test_split(tts, pr).test.X), 1) == 20

        # fitting cuts both windows; applying is a pass-through
        res = PortfolioOptimisers.fit_preprocessing(tts, rd)
        @test isa(res, TrainTestSplitResult)
        @test size(res.train.X, 1) == 80
        @test size(res.test.X, 1) == 20
        @test PortfolioOptimisers.apply_preprocessing(res, rd) === rd

        # an unfitted split is a pass-through too, so a step that never saw
        # `fit_preprocessing` hands the window on whole rather than cutting it silently
        @test PortfolioOptimisers.apply_preprocessing(tts, rd) === rd
        @test PortfolioOptimisers.apply_preprocessing(tts, pr) === pr

        # it splits whichever data level it is handed
        resp = PortfolioOptimisers.fit_preprocessing(tts, pr)
        @test size(values(resp.train.X), 1) == 80
        @test size(values(resp.test.X), 1) == 20

        # the step writes the sentinel slot and requires no slot up front
        @test PortfolioOptimisers.pipe_writes(tts) == :split
        @test PortfolioOptimisers.pipe_reads(tts) == ()

        # run_step narrows the data slot the input filled, at either level
        ctx = PortfolioOptimisers.PipelineContext(; prices = pr)
        fitted, ctx2 = PortfolioOptimisers.run_step(tts, ctx)
        @test isa(fitted, TrainTestSplitResult)
        @test size(values(ctx2.prices.X), 1) == 80
        @test isnothing(ctx2.returns)

        ctxr = PortfolioOptimisers.PipelineContext(; returns = rd)
        fittedr, ctxr2 = PortfolioOptimisers.run_step(tts, ctxr)
        @test size(fittedr.test.X, 1) == 20
        @test size(ctxr2.returns.X, 1) == 80
    end

    @testset "position rules" begin
        tts = TrainTestSplit(; test_size = 0.2)

        # first step: legal, and auto-named "split"
        pipe = Pipeline(;
                        steps = (tts, PricesToReturns(), EmpiricalPrior(), EqualWeighted()))
        @test pipe.names == ("split", "returns", "prior", "opt")

        # anywhere else: rejected, because an earlier fitted step would have seen the
        # held-out rows
        @test_throws ArgumentError Pipeline(;
                                            steps = (MissingDataFilter(), tts,
                                                     PricesToReturns(), EqualWeighted()))
        @test_throws ArgumentError Pipeline(;
                                            steps = (PricesToReturns(), tts,
                                                     EqualWeighted()))

        # nested inside another pipeline: rejected
        inner = Pipeline(; steps = (tts, PricesToReturns()))
        @test_throws ArgumentError Pipeline(; steps = (inner, EqualWeighted()))

        @test PortfolioOptimisers.has_split(pipe)
        @test !PortfolioOptimisers.has_split(Pipeline(;
                                                      steps = (PricesToReturns(),
                                                               EqualWeighted())))
    end

    @testset "fit, predict and the holdout payoff" begin
        pr = make_prices()

        pipe = Pipeline(;
                        steps = (TrainTestSplit(; test_size = 0.2), MissingDataFilter(),
                                 PriceGapFill(), PricesToReturns(), EmpiricalPrior(),
                                 EqualWeighted()))
        res = fit(pipe, pr)

        # every downstream step saw the training window alone
        @test size(res.ctx.returns.X, 1) == 79   # 80 prices → 79 returns
        @test size(values(res["split"].train.X), 1) == 80
        @test size(values(res["split"].test.X), 1) == 20

        # fit_predict evaluates on the held-out window, not on the training data
        pred = fit_predict(pipe, pr)
        @test length(pred.rd.X) == 19            # 20 held-out prices → 19 returns

        # without a split, fit_predict stays in-sample
        pipe_ns = Pipeline(;
                           steps = (MissingDataFilter(), PriceGapFill(), PricesToReturns(),
                                    EmpiricalPrior(), EqualWeighted()))
        pred_ns = fit_predict(pipe_ns, pr)
        @test length(pred_ns.rd.X) == 99

        # replaying a fitted split on unseen data is a pass-through, so predicting on
        # genuinely new observations still works
        pr_future = make_prices(; T = 130)
        pred_future = predict(res, pr_future, 101:130)
        @test length(pred_future.rd.X) == 29
    end

    @testset "the split agrees with its docstrings" begin
        split_count = PortfolioOptimisers.split_count

        # a fraction rounds down to whole rows in exact arithmetic: 0.29 * 100 is
        # 28.999999999999996, and a floor of that product lost the 29th row
        @test all(k -> split_count(k / 100, 100, :train_size) == k, 1:99)
        @test all(k -> split_count(Float32(k / 100), 100, :train_size) == k, 1:99)
        @test all(k -> split_count(k / 7, 7, :train_size) == k, 1:6)
        @test split_count(0.295, 100, :train_size) == 29
        @test split_count(0.305, 100, :train_size) == 30
        @test safe_index(0.29, nothing, 100) == (1:29, 30:100)

        # a fraction is any Real in (0, 1), so an exact Rational resolves exactly
        @test split_count(29 // 100, 100, :train_size) == 29
        @test split_count(1 // 3, 10, :train_size) == 3
        @test_throws DomainError split_count(3 // 2, 10, :train_size)
        rd10 = ReturnsResult(; nx = ["A"], X = reshape(collect(0.1:0.1:1.0), 10, 1))
        tr, te = train_test_split(rd10; test_size = 1 // 5)
        @test (size(tr.X, 1), size(te.X, 1)) == (8, 2)

        # with no size the training window takes floor(0.75 N) rows, and one row
        # leaves no test window
        @test safe_index(nothing, nothing, 4) == (1:3, 4:4)
        @test safe_index(nothing, nothing, 2) == (1:1, 2:2)
        @test_throws ArgumentError safe_index(nothing, nothing, 1)

        # the overlap message states both lengths and the row count
        e = try
            safe_index(8, 3, 10)
        catch err
            err
        end
        @test occursin("8 training and 3 test observations exceed the 10 available", e.msg)

        # at the returns level every time-axis array is a view of the input, sliced by row
        rng = StableRNG(1)
        ts = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 10))
        rd = ReturnsResult(; nx = ["A", "B"], X = randn(rng, 10, 2), nf = ["f"],
                           F = randn(rng, 10, 1), ts = ts)
        tr, te = train_test_split(rd; test_size = 3)
        @test tr.X isa SubArray && tr.F isa SubArray && tr.ts isa SubArray
        @test parent(te.X) === rd.X
        @test tr.ts == ts[1:7]
        @test te.ts == ts[8:10]
        @test te.F == rd.F[8:10, :]

        # at the price level the windows are copies, so a write to one leaves the input
        pr = PricesResult(;
                          X = TimeArray(ts, 100 .+ cumsum(rand(rng, 10, 2); dims = 1),
                                        ["A", "B"]))
        p81 = values(pr.X)[8, 1]
        trp, tep = train_test_split(pr; test_size = 3)
        @test timestamp(trp.X) == ts[1:7]
        @test timestamp(tep.X) == ts[8:10]
        values(tep.X)[1, 1] = -1.0
        @test values(pr.X)[8, 1] == p81
    end

    @testset "one evaluation protocol per call" begin
        pr = make_prices()
        pipe = Pipeline(;
                        steps = (TrainTestSplit(; test_size = 0.2), PricesToReturns(),
                                 EqualWeighted()))

        # a pipeline carrying a holdout cannot also be cross-validated
        @test_throws ArgumentError PortfolioOptimisers.assert_no_holdout(pipe)

        gscv = GridSearchCrossValidation(Dict("returns" =>
                                                  [PricesToReturns(; ret_method = :simple),
                                                   PricesToReturns(; ret_method = :log)]);
                                         cv = KFold(; n = 2), r = Variance())
        @test_throws ArgumentError search_cross_validation(pipe, gscv, pr)
        @test_throws ArgumentError cross_val_predict(pipe, pr, KFold(; n = 2))
    end
end

@testset "Pipeline routing targets" begin
    using Test, PortfolioOptimisers, InteractiveUtils

    #=
    Solver-free by construction: every assertion here is a struct rebuild, so the seam
    between the Pipeline and the optimisers is exercised without building a JuMP model.
    The `Solver` below is a placeholder needed only to construct a `JuMPOptimiser`.
    =#
    PO = PortfolioOptimisers
    slv = Solver(; name = :placeholder, solver = nothing)
    jo() = JuMPOptimiser(; slv = slv)

    optimisers() = ("JuMPOptimiser" => jo(),
                    "HierarchicalOptimiser" => HierarchicalOptimiser(),
                    "MeanRisk" => MeanRisk(; opt = jo()),
                    "RiskBudgeting" => RiskBudgeting(; opt = jo()),
                    "NearOptimalCentering" => NearOptimalCentering(; opt = jo()),
                    "FactorRiskContribution" => FactorRiskContribution(; opt = jo()),
                    "RelaxedRiskBudgeting" => RelaxedRiskBudgeting(; opt = jo()),
                    "HierarchicalRiskParity" => HierarchicalRiskParity(),
                    "NestedClustered" =>
                        NestedClustered(; opti = EqualWeighted(), opto = EqualWeighted()),
                    "Stacking" =>
                        Stacking(; opti = [EqualWeighted()], opto = EqualWeighted()),
                    "SubsetResampling" => SubsetResampling(; opt = EqualWeighted()),
                    "EqualWeighted" => EqualWeighted(),
                    "InverseVolatility" => InverseVolatility(),
                    "RandomWeighted" => RandomWeighted())

    @testset "accepted targets are exactly the expected table" begin
        #=
        The routing rule is derived, not declared: a target lands in the like-named field
        of whichever optimiser has one. This table is therefore a lock on the *consequences*
        of the field layouts, and it fails in both directions — adding a field named after a
        target silently enrols that optimiser in routing, and removing one silently
        withdraws it. Either way the change should be deliberate, so it should land here.

        Reading the rows: the two configurations route into their own fields; the six
        estimators that hold a configuration forward to it (and the five carrying risk
        measures additionally accept :sigma_ucs into their own `r`); the naive and meta
        optimisers carry the target fields themselves. `:rkb` is the one target that is
        neither derived nor universal — it lands at `rba.rkb`, so only the two optimisers
        carrying a risk-budgeting algorithm accept it.
        =#
        jump = [:pe, :wb, :lcse, :cte, :ple, :lt, :st, :slt, :sst, :sglt, :sgst, :smtx,
                :sgmtx]
        expected = Dict("JuMPOptimiser" => [jump; :mu_ucs],
                        "HierarchicalOptimiser" => [:pe, :cle, :wb],
                        "MeanRisk" => [jump; :mu_ucs; :sigma_ucs],
                        "RiskBudgeting" => [jump; :rkb; :mu_ucs; :sigma_ucs],
                        "NearOptimalCentering" => [jump; :mu_ucs; :sigma_ucs],
                        "FactorRiskContribution" => [jump; :mu_ucs; :sigma_ucs],
                        # No `r` field, so no covariance uncertainty set.
                        "RelaxedRiskBudgeting" => [jump; :rkb; :mu_ucs],
                        "HierarchicalRiskParity" => [:pe, :cle, :wb, :sigma_ucs],
                        # Meta-optimisers carry `pe`/`cle`/`wb` themselves.
                        "NestedClustered" => [:pe, :cle, :wb], "Stacking" => [:pe, :wb],
                        "SubsetResampling" => [:pe, :wb], "EqualWeighted" => [:wb],
                        "InverseVolatility" => [:pe, :wb], "RandomWeighted" => [:wb])
        for (name, opt) in optimisers()
            accepted = [t
                        for t in PO.PIPELINE_ROUTING_TARGETS
                        if PO.pipe_accepts(opt, Val(t))]
            @test accepted == expected[name]
        end
        @test length(expected) == length(optimisers())
    end

    @testset "delegation, declared rather than probed" begin
        #=
        `MeanRisk` has no `pe` field of its own, so the derived hasfield rule alone would
        reject `:pe`; it accepts only because it declares where its configuration lives.
        =#
        mr = MeanRisk(; opt = jo())
        @test !hasfield(typeof(mr), :pe)
        @test PO.pipe_accepts(mr, Val(:pe))
        @test PO.pipe_config_field(mr) === :opt
        @test PO.pipe_config_field(EqualWeighted()) === nothing
        #=
        `SubsetResampling` has a field literally named `opt`, holding an inner *estimator*
        rather than a configuration. Declaring rather than probing for `:opt` is what keeps
        that name collision from being mistaken for a configuration to route through.
        =#
        sr = SubsetResampling(; opt = EqualWeighted())
        @test hasfield(typeof(sr), :opt)
        @test PO.pipe_config_field(sr) === nothing
    end

    @testset "pipe_route places the value and changes nothing else" begin
        wb = WeightBounds(; lb = fill(0.1, 3), ub = fill(0.9, 3))
        for (name, opt) in optimisers()
            PO.pipe_accepts(opt, Val(:wb)) || continue
            routed = PO.pipe_route(opt, Val(:wb), wb)
            # Type parameters change with the new field value; the wrapper must not.
            @test Base.typename(typeof(routed)).wrapper ===
                  Base.typename(typeof(opt)).wrapper
            target = if isnothing(PO.pipe_config_field(opt))
                routed
            else
                getfield(routed, PO.pipe_config_field(opt))
            end
            @test target.wb === wb
        end
        # Delegation rebuilds through the configuration rather than replacing it.
        mr = MeanRisk(; opt = jo())
        routed = PO.pipe_route(mr, Val(:wb), wb)
        @test routed.opt.wb === wb
        @test routed.r === mr.r
        @test routed.obj === mr.obj
    end

    @testset "unroutable targets: optional pass by, the rest fail closed" begin
        @test PO.PIPELINE_OPTIONAL_TARGETS == (:pe, :cle)
        @test all(t -> t in PO.PIPELINE_ROUTING_TARGETS, PO.PIPELINE_OPTIONAL_TARGETS)

        ew = EqualWeighted()
        # `EqualWeighted` has no prior to override and computes nothing from one.
        @test PO.pipe_route(ew, Val(:pe), nothing) === ew
        # A `JuMPOptimiser` takes phylogeny as constraint results, never as a structure.
        j = jo()
        @test PO.pipe_route(j, Val(:cle), nothing) === j
        # Everything else would silently change the solved portfolio, so it throws.
        for t in
            (:lcse, :cte, :ple, :lt, :st, :slt, :sst, :sglt, :sgst, :smtx, :sgmtx, :rkb,
             :mu_ucs, :sigma_ucs)
            @test_throws ArgumentError PO.pipe_route(ew, Val(t), nothing)
        end
        @test_throws ArgumentError PO.pipe_route(HierarchicalOptimiser(), Val(:lcse),
                                                 nothing)
    end

    @testset "the two policy-carrying targets" begin
        #=
        `:mu_ucs` and `:sigma_ucs` name no field: they carry the validation the seam would
        otherwise lose. Both fail closed rather than dropping a computed set.
        =#
        mu_ucs = MuUncertaintySetClass()
        sigma_ucs = SigmaUncertaintySetClass()
        # Only an ArithmeticReturn can carry a bound on expected returns.
        @test PO.pipe_route(jo(), Val(:mu_ucs), mu_ucs).ret.ucs === mu_ucs
        @test_throws ArgumentError PO.pipe_route(JuMPOptimiser(; slv = slv,
                                                               ret = LogarithmicReturn()),
                                                 Val(:mu_ucs), mu_ucs)
        # The covariance half lands in the estimator's own risk measures, not its config.
        routed = PO.pipe_route(MeanRisk(; opt = jo(), r = UncertaintySetVariance()),
                               Val(:sigma_ucs), sigma_ucs)
        @test routed.r.ucs === sigma_ucs
        # An `r` field with no UncertaintySetVariance has nowhere to put the set.
        @test_throws ArgumentError PO.pipe_route(MeanRisk(; opt = jo(), r = Variance()),
                                                 Val(:sigma_ucs), sigma_ucs)
    end

    @testset "unroutable uncertainty is rejected at construction" begin
        #=
        Without this the failure surfaces at injection, which under cross_val_predict is
        after the first fold has already fitted every earlier step.
        =#
        ucs_step(t) = PipelineStep(; est = NormalUncertaintySet(), reads = (:returns,),
                                   writes = :uncertainty, target = t)

        # HierarchicalRiskParity can take a covariance set into its `r`, but has nowhere
        # to put a mean set.
        @test_throws ArgumentError Pipeline(;
                                            steps = (EmpiricalPrior(), ucs_step(:mu),
                                                     HierarchicalRiskParity()))
        @test_throws ArgumentError Pipeline(;
                                            steps = (EmpiricalPrior(), ucs_step(:both),
                                                     HierarchicalRiskParity()))
        hrp = Pipeline(;
                       steps = (EmpiricalPrior(), ucs_step(:sigma),
                                HierarchicalRiskParity()))
        @test length(hrp.steps) == 3

        # A JuMP-based optimiser accepts both halves.
        mr = Pipeline(;
                      steps = (EmpiricalPrior(), ucs_step(:both),
                               MeanRisk(; opt = jo(), r = UncertaintySetVariance())))
        @test length(mr.steps) == 3

        #=
        The check is structural: it establishes the optimiser family can receive the
        target, not that this configuration accepts the value. A JuMPOptimiser carrying a
        non-ArithmeticReturn still constructs, and still fails at injection.
        =#
        @test PO.pipe_accepts(MeanRisk(;
                                       opt = JuMPOptimiser(; slv = slv,
                                                           ret = LogarithmicReturn())),
                              Val(:mu_ucs))

        # Targets are known statically for uncertainty steps and for constraint steps.
        @test PO.pipe_required_targets(ucs_step(:both)) == (:mu_ucs, :sigma_ucs)
        @test PO.pipe_required_targets(ucs_step(:sigma)) == (:sigma_ucs,)
        @test PO.pipe_required_targets(EmpiricalPrior()) == ()
        @test PO.pipe_required_targets(WeightBoundsEstimator()) == (:wb,)
        @test PO.pipe_required_targets(CentralityConstraint()) == (:cte,)
    end

    @testset "each step declares the slot and the target that it writes" begin
        ucs_step(t) = PipelineStep(; est = NormalUncertaintySet(), reads = (:returns,),
                                   writes = :uncertainty, target = t)
        mr = MeanRisk(; opt = jo(), r = UncertaintySetVariance())
        # A constraint target on an uncertainty step computed both halves, as :both does.
        @test_throws ArgumentError Pipeline(; steps = (EmpiricalPrior(), ucs_step(:lt), mr))
        @test_throws ArgumentError PO.run_uncertainty_step(NormalUncertaintySet(), :lt,
                                                           PO.PipelineContext())
        # With no target, the step failed only when the fit reached it.
        @test_throws ArgumentError Pipeline(;
                                            steps = (EmpiricalPrior(), ucs_step(nothing),
                                                     mr))
        @test_throws ArgumentError Pipeline(;
                                            steps = (EmpiricalPrior(),
                                                     NormalUncertaintySet(), mr))
        # A family writes its own slot when it runs, so `writes` must name that slot.
        @test_throws ArgumentError Pipeline(;
                                            steps = (PipelineStep(;
                                                                  est = WeightBoundsEstimator(;
                                                                                              lb = 0.1),
                                                                  reads = (:returns,),
                                                                  writes = :phylogeny),
                                                     EqualWeighted()))
        @test_throws ArgumentError Pipeline(;
                                            steps = (EmpiricalPrior(),
                                                     PipelineStep(;
                                                                  est = NormalUncertaintySet(),
                                                                  reads = (:returns,),
                                                                  writes = :prior,
                                                                  target = :mu), mr))
        @test Pipeline(;
                       steps = (PipelineStep(; est = EmpiricalPrior(), reads = (:returns,),
                                             writes = :prior), EqualWeighted())).names ==
              ("prior", "opt")
        @test Pipeline(; steps = (EmpiricalPrior(), ucs_step(:both), mr)).names ==
              ("prior", "uncertainty", "opt")
        # A callable that writes :uncertainty names no target that the constructor knows.
        @test PO.pipe_required_targets(PipelineStep(; est = ctx -> nothing,
                                                    writes = :uncertainty)) == ()
    end

    @testset "constraint slot fans out by carried target, then by result type" begin
        wb = WeightBounds(; lb = fill(0.1, 3), ub = fill(0.9, 3))
        lc = LinearConstraint(;
                              ineq = PartialLinearConstraint(; A = [1.0 0.0 0.0],
                                                             B = [0.5]))
        thr = Threshold(0.05)
        rkb = PO.risk_budget_constraints(nothing; N = 5)
        @test PO.constraint_targets(nothing) == []
        @test PO.constraint_targets(wb) == [:wb => wb]
        # A RiskBudget names exactly one field, so its type alone places it.
        @test PO.constraint_targets(rkb) == [:rkb => rkb]
        #=
        A Threshold names six fields, so its type cannot place it. A step declares the
        target and the declaration travels with the value.
        =#
        @test_throws ArgumentError PO.constraint_targets(thr)
        @test PO.constraint_targets(PO.TargetedConstraint(:sglt, thr)) == [:sglt => thr]
        #=
        An accumulating target packs; a single-valued one refuses the second write rather
        than letting it overwrite the first.
        =#
        pair = PO.constraint_targets(PO.AbstractConstraintResult[lc, lc])
        @test only(pair).first === :lcse
        @test only(pair).second == [lc, lc]
        @test_throws ArgumentError PO.constraint_targets(PO.AbstractConstraintResult[wb,
                                                                                     wb])
        @test_throws ArgumentError PO.constraint_targets(PO.AbstractConstraintResult[PO.TargetedConstraint(:lt,
                                                                                                           thr),
                                                                                     PO.TargetedConstraint(:lt,
                                                                                                           thr)])
        # A result of a type no target exists for is still rejected at the fan-out.
        @test_throws ArgumentError PO.constraint_targets(PartialLinearConstraint(;
                                                                                 A = [1.0 0.0 0.0],
                                                                                 B = [0.5]))
    end

    @testset "a target accumulates exactly when its field takes a vector of results" begin
        #=
        `PIPELINE_ACCUMULATING_TARGETS` is a claim about the *receiving fields*, so it is
        checked against them rather than restated. Each row puts a two-element vector of
        computed results in one target and asserts the optimiser takes it; the scenario and
        group rows carry the membership matrices their lengths are validated against, which
        is the check that makes packing them safe.
        =#
        wb = WeightBounds(; lb = fill(0.1, 3), ub = fill(0.9, 3))
        lc = LinearConstraint(;
                              ineq = PartialLinearConstraint(; A = [1.0 0.0 0.0],
                                                             B = [0.5]))
        thr = Threshold(0.05)
        ple = SemiDefinitePhylogeny(; A = [0 1 0; 1 0 0; 0 0 0], p = 0.05)
        mtx = [1 0 0; 0 1 1]
        packed = (:lcse => (; lcse = [lc, lc]), :ple => (; ple = [ple, ple]),
                  :slt => (; slt = [thr, thr], smtx = [mtx, mtx]),
                  :sst => (; sst = [thr, thr], smtx = [mtx, mtx]),
                  :sglt => (; sglt = [thr, thr], sgmtx = [mtx, mtx]),
                  :sgst => (; sgst = [thr, thr], sgmtx = [mtx, mtx]),
                  :smtx => (; smtx = [mtx, mtx]), :sgmtx => (; sgmtx = [mtx, mtx]))
        for (target, kwargs) in packed
            @test JuMPOptimiser(; slv = slv, kwargs...) isa JuMPOptimiser
            @test target in PO.PIPELINE_ACCUMULATING_TARGETS
        end
        #=
        `:cte` accumulates by *folding*, not packing, so it is the one member whose field
        does not take a vector of results. Its vector form is a vector of estimators, and
        generation appends their rows into one constraint — which is the value the fold
        produces, so that is what the field must accept.
        =#
        @test :cte in PO.PIPELINE_ACCUMULATING_TARGETS
        @test_throws TypeError JuMPOptimiser(; slv = slv, cte = [lc, lc])
        @test JuMPOptimiser(; slv = slv, cte = PO.merge_linear_constraints([lc, lc])) isa
              JuMPOptimiser
        @test Set([first.(packed)...; :cte]) == Set(PO.PIPELINE_ACCUMULATING_TARGETS)

        # The complement: fields that hold exactly one value, and refuse a vector.
        single = (:wb => (; wb = [wb, wb]), :lt => (; lt = [thr, thr]),
                  :st => (; st = [thr, thr]))
        for (target, kwargs) in single
            @test_throws TypeError JuMPOptimiser(; slv = slv, kwargs...)
            @test !(target in PO.PIPELINE_ACCUMULATING_TARGETS)
        end
        # `:rkb` lives on the risk-budgeting algorithm, and is single-valued there.
        rkb = PO.risk_budget_constraints(nothing; N = 3)
        @test_throws TypeError AssetRiskBudgeting(; rkb = [rkb, rkb])
        @test !(:rkb in PO.PIPELINE_ACCUMULATING_TARGETS)

        #=
        The fan-out follows the table in both directions: two writes combine for an
        accumulating target and throw for every other one.
        =#
        for target in PO.PIPELINE_ACCUMULATING_TARGETS
            if target === :cte
                continue
            end
            two = PO.AbstractConstraintResult[PO.TargetedConstraint(target, thr),
                                              PO.TargetedConstraint(target, thr)]
            @test only(PO.constraint_targets(two)).second == [thr, thr]
        end
        for target in (:wb, :lt, :st, :rkb)
            two = PO.AbstractConstraintResult[PO.TargetedConstraint(target, thr),
                                              PO.TargetedConstraint(target, thr)]
            @test_throws ArgumentError PO.constraint_targets(two)
        end
        # Two writes to `:cte` fold into one constraint carrying both blocks of rows.
        two_cte = PO.AbstractConstraintResult[PO.TargetedConstraint(:cte, lc),
                                              PO.TargetedConstraint(:cte, lc)]
        folded = only(PO.constraint_targets(two_cte)).second
        @test folded isa LinearConstraint
        @test folded.ineq.A == [lc.ineq.A; lc.ineq.A]
        @test folded.ineq.B == [lc.ineq.B; lc.ineq.B]
        # A value that is not a LinearConstraint cannot be folded, and says so.
        @test_throws ArgumentError PO.constraint_targets(PO.AbstractConstraintResult[PO.TargetedConstraint(:cte,
                                                                                                           thr),
                                                                                     PO.TargetedConstraint(:cte,
                                                                                                           thr)])
    end

    @testset "n centrality steps equal one cte field holding n estimators" begin
        #=
        The equivalence the fold exists to preserve. `centrality_constraints` over a vector
        of estimators appends every row into one result, so a pipeline running one step per
        estimator must reach the optimiser with that same constraint — otherwise the two
        ways of writing the same mandate would build different models.
        =#
        rd = ReturnsResult(; nx = string.("A", 1:5),
                           X = randn(StableRNG(987654321), 60, 5) / 100)
        cc1 = CentralityConstraint(; B = MinValue(), comp = <=)
        cc2 = CentralityConstraint(; B = MaxValue(), comp = >=)
        cc3 = CentralityConstraint(; B = MinValue(), comp = ==)
        manual = PO.centrality_constraints([cc1, cc2, cc3], rd)

        ctx = PO.PipelineContext(; returns = rd)
        for cc in (cc1, cc2, cc3)
            _, ctx = PO.run_step(cc, ctx)
        end
        target, stepped = only(PO.constraint_targets(ctx.constraints))
        @test target === :cte
        @test stepped isa LinearConstraint
        @test stepped.ineq.A == manual.ineq.A
        @test stepped.ineq.B == manual.ineq.B
        @test stepped.eq.A == manual.eq.A
        @test stepped.eq.B == manual.eq.B
        #=
        And the merged constraint is what the optimiser actually receives. The fold builds
        a fresh constraint per call, so this is an equality of rows rather than of objects
        — unlike a packed target, which routes the very results the steps produced.
        =#
        routed = PO.inject_context(MeanRisk(; opt = jo()), ctx)
        @test routed.opt.cte.ineq.A == manual.ineq.A
        @test routed.opt.cte.ineq.B == manual.ineq.B
        @test routed.opt.cte.eq.A == manual.eq.A
        @test routed.opt.cte.eq.B == manual.eq.B
    end

    @testset "every constraint family declares where its step lands" begin
        #=
        The finding this locks: a family with a run_step but no routing target computed a
        result the injection seam could not place, so the Pipeline fitted every earlier
        step and then always threw — under cross_val_predict, after the first fold. One
        declaration now drives the step, the construction check and the fan-out, so a new
        family cannot be half-wired.
        =#
        function concrete_subtypes(T)
            out = Any[]
            for S in subtypes(T)
                isabstracttype(S) ? append!(out, concrete_subtypes(S)) : push!(out, S)
            end
            return out
        end
        families = [S
                    for S in concrete_subtypes(PO.AbstractConstraintEstimator)
                    if parentmodule(S) === PO && !(S <: PO.JuMPConstraintEstimator)]
        @test !isempty(families)
        #=
        Asked of the type rather than of an instance: a family declares by defining a
        method more specific than the empty fallback, and that is exactly what the
        fallback's own signature tests for.
        =#
        fallback = which(PO.pipe_constraint_targets, Tuple{PO.AbstractConstraintEstimator})
        for S in families
            @test which(PO.pipe_constraint_targets, Tuple{S}) !== fallback
        end
        # Every declared target is a real one, and every choice is a legal annotation.
        @test all(t -> t in PO.PIPELINE_ROUTING_TARGETS, PO.PIPELINE_THRESHOLD_TARGETS)
        @test all(t -> t in PO.PIPELINE_STEP_TARGETS, PO.PIPELINE_THRESHOLD_TARGETS)
        @test all(t -> t in PO.PIPELINE_ROUTING_TARGETS,
                  PO.PIPELINE_ASSET_SETS_MATRIX_TARGETS)
        @test all(t -> t in PO.PIPELINE_STEP_TARGETS, PO.PIPELINE_ASSET_SETS_MATRIX_TARGETS)
        @test all(t -> t in PO.PIPELINE_ROUTING_TARGETS, PO.PIPELINE_ACCUMULATING_TARGETS)
    end

    @testset "each family computes a value and pairs it with its target" begin
        #=
        The end-to-end half of the finding: a family with a step but no target computed a
        result and then always threw at injection. Running each family's step here and
        routing what it produced is what shows the two halves meet.
        =#
        rd = ReturnsResult(; nx = string.("A", 1:5),
                           X = randn(StableRNG(987654321), 60, 5) / 100)
        ctx0 = PO.PipelineContext(; returns = rd)
        steps = ["WeightBoundsEstimator" =>
                     (WeightBoundsEstimator(; lb = 0.05, ub = 0.5), nothing, :wb),
                 "LinearConstraintEstimator" =>
                     (LinearConstraintEstimator(; val = "A1 <= 0.3"), nothing, :lcse),
                 "ThresholdEstimator" => (ThresholdEstimator(; val = 0.05), :st, :st),
                 # The scalar resolves to `RiskBudget(1.0)`, which `test_01_structs.jl`
                 # asserts. This row asserts the routing target only.
                 "RiskBudgetEstimator" => (RiskBudgetEstimator(; val = 0.2), nothing, :rkb),
                 "CentralityConstraint" => (CentralityConstraint(), nothing, :cte),
                 "IntegerPhylogenyEstimator" =>
                     (IntegerPhylogenyEstimator(; B = 1), nothing, :ple)]
        for (name, (est, declared, target)) in steps
            step = if isnothing(declared)
                est
            else
                PipelineStep(; est = est, reads = (:returns,), writes = :constraints,
                             target = declared)
            end
            _, ctx = PO.run_step(step, ctx0)
            @test PO.constraint_targets(ctx.constraints)[1].first === target
        end
        # An unwrapped uncertainty step still refuses; the constraint rule did not widen it.
        @test_throws ArgumentError PO.run_step(NormalUncertaintySet(), ctx0)
    end

    @testset "a constraint step is refused when the optimiser cannot receive it" begin
        thr_step(t) = PipelineStep(; est = ThresholdEstimator(; val = 0.05),
                                   reads = (:returns,), writes = :constraints, target = t)
        #=
        A ThresholdEstimator names six fields, so an unwrapped step cannot be placed and
        the Pipeline says so at construction rather than at injection.
        =#
        @test_throws ArgumentError Pipeline(;
                                            steps = (ThresholdEstimator(; val = 0.05),
                                                     MeanRisk(; opt = jo())))
        # A target that belongs to another family is refused where it is written.
        @test_throws ArgumentError Pipeline(;
                                            steps = (thr_step(:smtx),
                                                     MeanRisk(; opt = jo())))
        # Declared correctly, it constructs; in front of an optimiser with no such field,
        # it does not.
        @test length(Pipeline(; steps = (thr_step(:lt), MeanRisk(; opt = jo()))).steps) == 2
        @test_throws ArgumentError Pipeline(; steps = (thr_step(:lt), EqualWeighted()))
        #=
        A risk budget names one field, so it needs no annotation — and only the two
        optimisers carrying a risk-budgeting algorithm can receive it.
        =#
        rkb_est = RiskBudgetEstimator(; val = 0.2)
        @test length(Pipeline(; steps = (rkb_est, RiskBudgeting(; opt = jo()))).steps) == 2
        @test_throws ArgumentError Pipeline(; steps = (rkb_est, MeanRisk(; opt = jo())))
        #=
        A CentralityConstraint produces a LinearConstraint, which would fan out to :lcse
        by type; the family declares :cte instead, so the constraint keeps its own model
        keys rather than being merged into the general linear block.
        =#
        @test PO.pipe_constraint_targets(CentralityConstraint()) == (:cte,)
        @test length(Pipeline(; steps = (CentralityConstraint(), MeanRisk(; opt = jo()))).steps) ==
              2
        # A JuMP-model constraint is configuration, not a step, and is refused as one.
        @test_throws ArgumentError Pipeline(;
                                            steps = (BudgetRange(; lb = 0.9, ub = 1.1),
                                                     MeanRisk(; opt = jo())))
    end
end

@testset "Meta-optimiser scalar weight bounds are asset-dimensioned" begin
    using Test, PortfolioOptimisers, StableRNGs

    #=
    A scalar weight bound means "this bound applies to every asset". Meta-optimisers used to
    expand it to the wrong length: `Stacking` passed `N = Ni` (the number of inner
    optimisers) and `NestedClustered` passed `N = clr.k` (the number of clusters) to
    `weight_bounds_constraints`. But `outer_optimisation_finaliser` maps the outer weights
    back through `wi` into asset space *before* applying the bound, so the bound must be
    asset-length.

    That produced two failures, and the first is why these tests assert on weights rather
    than only on lengths:

      1. Silently wrong answers. `opt_weight_bounds` guards with
         `map((x, y) -> x > y, lb, w)`, and `map` truncates to its shorter argument, so only
         the first `Ni` (or `clr.k`) assets were ever checked. A bound violated by a later
         asset was neither detected nor enforced, and the optimiser returned success.
      2. An opaque crash. Once the truncated guard *did* see a violation, the clamping loop
         reached a real broadcast (`w .< ub .&& w .> lb`) and threw `DimensionMismatch`,
         naming neither the bound nor the optimiser.

    The default bounds are `nothing`, which expands to ±Inf and can never be violated, so
    the early return fired before either path — which is why no existing test caught this.
    =#
    N = 5
    rng = StableRNG(20260721)
    X = randn(rng, 250, N) ./ 100
    #= Assets 4 and 5 get much lower volatility, so InverseVolatility overweights them and
    the largest weights land *beyond* the truncation point. Without that the truncated guard
    would have caught the violation by luck and the regression would not reproduce. =#
    X[:, 5] ./= 8
    X[:, 4] ./= 4
    rd = ReturnsResult(; nx = string.("A", 1:N), X = X)

    ub = 0.30
    lb = 0.15

    @testset "Stacking" begin
        Ni = 2
        unbounded = optimise(Stacking(; opti = [EqualWeighted(), InverseVolatility()],
                                      opto = EqualWeighted()), rd)
        # Guard: the test only exercises truncation if the violation is past index Ni.
        @test argmax(unbounded.w) > Ni
        @test maximum(unbounded.w) > ub

        res = optimise(Stacking(; opti = [EqualWeighted(), InverseVolatility()],
                                opto = EqualWeighted(),
                                wb = WeightBounds(; lb = 0.0, ub = ub)), rd)
        # Pre-fix: unenforced past index Ni, so this returned maximum(w) ≈ 0.366.
        @test all(res.w .<= ub + 1e-10)
        @test length(res.wb.ub) == N
        @test length(res.wb.lb) == N
        @test isapprox(sum(res.w), 1.0)
        @test isa(res.retcode, OptimisationSuccess)

        # Violated within the first Ni: pre-fix this reached the clamping broadcast and threw.
        resl = optimise(Stacking(; opti = [EqualWeighted(), InverseVolatility()],
                                 opto = EqualWeighted(),
                                 wb = WeightBounds(; lb = lb, ub = 1.0)), rd)
        @test all(resl.w .>= lb - 1e-10)
        @test isapprox(sum(resl.w), 1.0)
        @test isa(resl.retcode, OptimisationSuccess)
    end

    @testset "NestedClustered" begin
        unbounded = optimise(NestedClustered(; opti = InverseVolatility(),
                                             opto = EqualWeighted()), rd)
        k = unbounded.clr.k
        # Guard: the same reasoning, with clr.k as the truncation point.
        @test k < N
        @test argmax(unbounded.w) > k
        @test maximum(unbounded.w) > ub

        res = optimise(NestedClustered(; opti = InverseVolatility(), opto = EqualWeighted(),
                                       wb = WeightBounds(; lb = 0.0, ub = ub)), rd)
        @test all(res.w .<= ub + 1e-10)
        @test length(res.wb.ub) == N
        @test length(res.wb.lb) == N
        @test isapprox(sum(res.w), 1.0)
        @test isa(res.retcode, OptimisationSuccess)
    end

    @testset "vector bounds are unaffected" begin
        #=
        `weight_bounds_constraints_side(wb::VecNum, args...) = wb` ignores N entirely, so
        vector bounds never had the defect. This pins that the fix did not change them —
        and covers everything a Pipeline routes in, which is always asset-length.
        =#
        vec_res = optimise(Stacking(; opti = [EqualWeighted(), InverseVolatility()],
                                    opto = EqualWeighted(),
                                    wb = WeightBounds(; lb = fill(0.0, N),
                                                      ub = fill(ub, N))), rd)
        sca_res = optimise(Stacking(; opti = [EqualWeighted(), InverseVolatility()],
                                    opto = EqualWeighted(),
                                    wb = WeightBounds(; lb = 0.0, ub = ub)), rd)
        # A scalar bound must now be exactly equivalent to the vector spelling of it.
        @test isapprox(vec_res.w, sca_res.w)
    end
end

@testset "Weight bound resolution keeps its sign and its length" begin
    using Test, PortfolioOptimisers, StableRNGs

    #=
    Two defects that issue #517 found while it swept
    `src/09_ConstraintGeneration/05_WeightBoundsConstraintGeneration.jl`.

      1. An infinite scalar bound lost its sign. `weight_bounds_constraints_side(wb::Number,
         N, val)` filled with `val` whenever `isinf(wb)`, and `val` is the free bound of the
         side: `-Inf` below and `Inf` above. So a lower bound of `+Inf`, which admits no
         weight at all, resolved to `-Inf`, which admits every weight. The pair
         `WeightBounds(; lb = Inf, ub = Inf)` passes validation, because `Inf <= Inf`, so a
         caller reached it. The branch exists because `range(Inf, Inf; length = 3)` collects
         to `NaN`, and it is kept with `wb` in place of `val`.
      2. A vector bound of the wrong length passed through unchecked. Every reader compares a
         bound with the weights through `map`, which truncates to its shorter argument, so
         each asset past the end of the bound went unchecked and the optimiser reported
         success on a portfolio that broke the bound. Measured before the fix, with five
         assets and a two-entry `ub` of `0.3`: asset five took `0.646`.
    =#
    @testset "an infinite scalar bound keeps its sign" begin
        both_up = weight_bounds_constraints(WeightBounds(; lb = Inf, ub = Inf); N = 3)
        # Pre-fix: `lb` came back as `[-Inf, -Inf, -Inf]`.
        @test all(isequal(Inf), both_up.lb)
        @test all(isequal(Inf), both_up.ub)

        both_down = weight_bounds_constraints(WeightBounds(; lb = -Inf, ub = -Inf); N = 3)
        # Pre-fix: `ub` came back as `[Inf, Inf, Inf]`.
        @test all(isequal(-Inf), both_down.lb)
        @test all(isequal(-Inf), both_down.ub)

        # The free box is unchanged, and it is the case every default takes.
        free = weight_bounds_constraints(WeightBounds(; lb = -Inf, ub = Inf); N = 3)
        @test all(isequal(-Inf), free.lb)
        @test all(isequal(Inf), free.ub)
        @test !any(isnan, free.lb)
        @test !any(isnan, free.ub)

        # A finite scalar still expands to a constant range rather than to an `Array`.
        finite = weight_bounds_constraints(WeightBounds(0.0, 1.0); N = 3)
        @test finite.lb == zeros(3)
        @test finite.ub == ones(3)
        @test !isa(finite.lb, Array)
    end

    @testset "a vector bound must match the asset count" begin
        @test_throws DimensionMismatch weight_bounds_constraints(WeightBounds([0.0, 0.0],
                                                                              [0.3, 0.3]);
                                                                 N = 5)
        # The mixed pair takes the other method, and it is guarded too.
        @test_throws DimensionMismatch weight_bounds_constraints(WeightBounds([0.0, 0.0],
                                                                              0.3); N = 5)
        @test_throws DimensionMismatch PortfolioOptimisers.weight_bounds_constraints_side([0.1,
                                                                                           0.2],
                                                                                          5,
                                                                                          -Inf)
        # A matching length passes, and so does a caller that names no asset count.
        matched = weight_bounds_constraints(WeightBounds([0.0, 0.0], [0.3, 0.3]); N = 2)
        @test matched.lb == [0.0, 0.0]
        @test weight_bounds_constraints(WeightBounds([0.0, 0.0], [0.3, 0.3])).ub ==
              [0.3, 0.3]
        @test PortfolioOptimisers.weight_bounds_constraints_side([0.1, 0.2]) == [0.1, 0.2]
    end

    @testset "the truncated guard is now a raise" begin
        #= The end-to-end shape of defect 2. Asset five carries the largest weight, so a
        two-entry bound never reaches it. =#
        N = 5
        rng = StableRNG(20260826)
        X = randn(rng, 200, N) ./ 100
        X[:, 5] ./= 8
        rd = ReturnsResult(; nx = string.("A", 1:N), X = X)

        full = optimise(InverseVolatility(; wb = WeightBounds(fill(0.0, N), fill(0.3, N))),
                        rd)
        @test all(full.w .<= 0.3 + 1e-10)
        @test isa(full.retcode, OptimisationSuccess)

        # Pre-fix this returned `OptimisationSuccess` with `w[5] ≈ 0.646`.
        @test_throws DimensionMismatch optimise(InverseVolatility(;
                                                                  wb = WeightBounds([0.0,
                                                                                     0.0],
                                                                                    [0.3,
                                                                                     0.3])),
                                                rd)
    end
end

@testset "validate_bounds dispatches every method it declares" begin
    using Test, PortfolioOptimisers

    #=
    Seven methods, and the pair of argument types alone decides which one runs. Two of the
    seven exist to catch a `nothing` on one side, and the catch-all closes the family with no
    check at all. Each block below selects one method and drives its raises.
    =#
    vb = PortfolioOptimisers.validate_bounds

    @testset "(Number, Number)" begin
        @test isnothing(vb(0.0, 1.0))
        @test_throws DomainError vb(0.3, 0.2)
    end

    @testset "(VecNum, Number)" begin
        @test isnothing(vb([0.1, 0.2], 0.5))
        @test_throws PortfolioOptimisers.IsEmptyError vb(Float64[], 0.5)
        @test_throws DomainError vb([0.1, 0.9], 0.5)
    end

    @testset "(Number, VecNum)" begin
        @test isnothing(vb(0.05, [0.1, 0.2]))
        @test_throws PortfolioOptimisers.IsEmptyError vb(0.5, Float64[])
        @test_throws DomainError vb(0.5, [0.9, 0.1])
    end

    @testset "(VecNum, VecNum)" begin
        @test isnothing(vb([0.1, 0.2], [0.3, 0.4]))
        @test_throws PortfolioOptimisers.IsEmptyError vb(Float64[], [0.1])
        @test_throws PortfolioOptimisers.IsEmptyError vb([0.1], Float64[])
        # The length check runs before the comparison, so `map` never truncates a pair here.
        @test_throws DimensionMismatch vb([0.1, 0.2], [0.3])
        @test_throws DomainError vb([0.1, 0.9], [0.3, 0.5])
    end

    @testset "(VecNum, Any) and (Any, VecNum)" begin
        @test isnothing(vb([0.1, 0.2], nothing))
        @test isnothing(vb(nothing, [0.1, 0.2]))
        @test_throws PortfolioOptimisers.IsEmptyError vb(Float64[], nothing)
        @test_throws PortfolioOptimisers.IsEmptyError vb(nothing, Float64[])
    end

    @testset "the catch-all checks nothing" begin
        @test isnothing(vb(nothing, nothing))
        @test isnothing(vb(0.9, nothing))
        @test isnothing(vb(nothing, 0.1))
    end

    @testset "the estimator route validates the resolved pair" begin
        sets = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"]))
        #= The estimator's own constructor compares a `Dict` with a number with nothing,
        because neither side is resolved yet. `weight_bounds_constraints` builds a
        `WeightBounds` from the resolved pair, and that constructor compares them. =#
        loose = WeightBoundsEstimator(; lb = Dict("A" => 0.9), ub = 0.5)
        @test_throws DomainError weight_bounds_constraints(loose, sets)

        # `dlb` and `dub` reach every asset the estimator does not name.
        named = weight_bounds_constraints(WeightBoundsEstimator(; lb = Dict("A" => 0.1),
                                                                ub = Dict("A" => 0.5),
                                                                dlb = 0.02, dub = 0.7),
                                          sets)
        @test named.lb == [0.1, 0.02, 0.02]
        @test named.ub == [0.5, 0.7, 0.7]

        # A `nothing` side survives the resolution as `nothing`.
        @test isnothing(weight_bounds_constraints(WeightBoundsEstimator(; lb = nothing,
                                                                        ub = 1.0), sets).lb)
        @test isnothing(weight_bounds_constraints(WeightBoundsEstimator(; lb = 0.0,
                                                                        ub = nothing),
                                                  sets).ub)
    end

    @testset "the estimator compares two vector bounds" begin
        #= The branch the estimator's constructor takes when both sides are numbers or
        vectors of numbers. A `Dict`, a `Pair` or an algorithmic rule never reaches it.
        The constructor states no check of its own here: it hands the pair to
        `validate_bounds`, which owns the length check and the entry-by-entry
        comparison. =#
        both = WeightBoundsEstimator(; lb = [0.0, 0.1], ub = [0.8, 0.9])
        @test both.lb == [0.0, 0.1]
        @test both.ub == [0.8, 0.9]
        @test_throws DimensionMismatch WeightBoundsEstimator(; lb = [0.0, 0.1], ub = [0.8])
        @test_throws DomainError WeightBoundsEstimator(; lb = [0.9, 0.1], ub = [0.8, 0.9])

        # A scalar pair reaches the same branch, and `validate_bounds` compares it too.
        @test_throws DomainError WeightBoundsEstimator(; lb = 0.9, ub = 0.5)
        # `dlb` and `dub` are compared with each other whenever both are given.
        @test_throws DomainError WeightBoundsEstimator(; dlb = 0.9, dub = 0.5)

        #= The constructor's own empty guards name the side they refuse, the way the
        `validate_bounds` siblings do. A bare exception type leaves that to ArgCheck's
        generated text, and the two shapes then read differently for one rule. =#
        err_lb = try
            WeightBoundsEstimator(; lb = Float64[])
            nothing
        catch e
            e
        end
        @test isa(err_lb, PortfolioOptimisers.IsEmptyError)
        @test err_lb.msg == "lb cannot be empty"
        err_ub = try
            WeightBoundsEstimator(; ub = Float64[])
            nothing
        catch e
            e
        end
        @test isa(err_ub, PortfolioOptimisers.IsEmptyError)
        @test err_ub.msg == "ub cannot be empty"
    end

    @testset "the default asset count builds empty bounds and is refused" begin
        @test_throws PortfolioOptimisers.IsEmptyError weight_bounds_constraints(WeightBounds(0.0,
                                                                                             1.0))
        @test_throws PortfolioOptimisers.IsEmptyError weight_bounds_constraints(nothing)
    end
end
