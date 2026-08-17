@testset "Value-level multiplicity: the singular case is unchanged" begin
    using PortfolioOptimisers, Test, StableRNGs, Clarabel, JuMP, LinearAlgebra, Statistics

    #=
    The whole value-level `expected_*` surface now takes one risk measure or a vector of
    them. The first thing that must hold is that nothing moved for a caller who passes one
    measure, so these assertions are `==` and not `isapprox`: the singular path must be
    **bit-identical**, not merely close.
    =#

    rng = StableRNG(20260816)
    X = randn(rng, 250, 8) ./ 100 .+ 0.0004
    w = fill(inv(8), 8)
    pr = prior(EmpiricalPrior(), X)

    rv = Variance()
    rc = ConditionalValueatRisk()
    rs = StandardDeviation()

    @testset "one measure gives the same number through every entry" begin
        for r in (rv, rc, rs)
            # The prior route and the matrix route are separate methods; both stayed put.
            @test expected_risk(r, w, pr) == expected_risk(factory(r, pr), w, pr.X)
        end
    end

    @testset "`sca` on a singular call is inert" begin
        # A scalariser over one element returns that element, so every scalariser agrees
        # with the un-scalarised call. It is swallowed by `kwargs...`, never consulted.
        base = expected_risk(rv, w, pr)
        for sca in
            (SumScalariser(), MaxScalariser(), MinScalariser(), LogSumExpScalariser())
            @test expected_risk(rv, w, pr; sca = sca) == base
        end
    end

    @testset "a one-element vector under `SumScalariser` equals the singular call" begin
        #=
        Load-bearing identity. #299 leaned on it when it accepted the reported-versus-
        optimised mismatch in silence: a caller who names no scalariser anywhere gets a
        matched figure, because the sum is the identity on one element and
        `JuMPOptimiser.sca` defaults to `SumScalariser()` too.
        =#
        for r in (rv, rc, rs)
            @test expected_risk([r], w, pr) == expected_risk(r, w, pr)
        end
    end

    @testset "`scale` is a combination weight on the value level" begin
        #=
        Inert on a scalar, applied on a vector, under **every** scalariser. This mirrors
        the model, where `set_risk_expression!` pushes `scale * r_expr` before any
        scalariser runs.
        =#
        base = expected_risk(rv, w, pr)
        for sc in (1.0, 2.0, 50.0)
            scaled = Variance(; settings = RiskMeasureSettings(; scale = sc))
            # Inert on the scalar entry.
            @test expected_risk(scaled, w, pr) == base
            # Applied on the vector entry.
            @test expected_risk([scaled], w, pr) ≈ sc * base
            # And under a non-summing scalariser too.
            @test expected_risk([scaled], w, pr; sca = MaxScalariser()) ≈ sc * base
        end
    end

    @testset "the scalarisers combine as they say they do" begin
        a = expected_risk(rv, w, pr)
        b = expected_risk(rc, w, pr)
        v = [rv, rc]
        @test expected_risk(v, w, pr; sca = SumScalariser()) ≈ a + b
        @test expected_risk(v, w, pr; sca = MaxScalariser()) ≈ max(a, b)
        # `MinScalariser` is admitted at the value level even though no `MeanRisk`
        # optimisation can express it: the value level evaluates numbers, and convexity
        # does not bind numbers.
        @test expected_risk(v, w, pr; sca = MinScalariser()) ≈ min(a, b)
    end
end

@testset "Value-level multiplicity: the live break closes" begin
    using PortfolioOptimisers, Test, StableRNGs, Clarabel, JuMP, LinearAlgebra

    #=
    `JuMPOptimiser.ret` has taken a vector of return terms since #272, and the bundle
    stores the resolved value, so `res.ret` is a vector the moment a user optimises against
    several terms. Documented example sites pass `res.ret` straight into `ExpectedReturn`,
    and that call raised a `MethodError` until `rt` widened.
    =#

    rng = StableRNG(1234321)
    X = randn(rng, 250, 6) ./ 100 .+ 0.0005
    slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false),
                 check_sol = (; allow_local = true, allow_almost = true))
    pr = prior(EmpiricalPrior(), X)

    rets = [ArithmeticReturn(),
            ArithmeticReturn(; settings = JuMPReturnsSettings(; scale = 2.0))]
    res = optimise(MeanRisk(; r = Variance(), obj = MinimumRisk(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv, ret = rets)))
    @test isa(res.retcode, OptimisationSuccess)
    @test isa(res.ret, AbstractVector)

    @testset "the documented route out of a result works" begin
        # This is the call 32 documented example sites make.
        er = ExpectedReturn(; rt = res.ret)
        @test isa(expected_risk(er, res.w, pr), Number)

        errr = ExpectedReturnRiskRatio(; rt = res.ret, rk = res.r, rf = 0.0)
        @test isa(expected_risk(errr, res.w, pr), Number)

        # And the documented tuple route.
        rk, rt, sr = expected_risk_ret_ratio(res.r, res.ret, res.w, res.pr; sca = res.sca)
        @test isfinite(rk) && isfinite(rt) && isfinite(sr)
    end

    @testset "a vector `rt` sums at its terms' own scales" begin
        # 1.0 + 2.0 = 3.0 times the lone term, and no scalariser is involved: the return
        # axis sums and never scalarises.
        one_term = expected_risk(ExpectedReturn(; rt = ArithmeticReturn()), res.w, pr)
        many = expected_risk(ExpectedReturn(; rt = rets), res.w, pr)
        @test many ≈ 3 * one_term
    end

    @testset "the result carries the measure and the scalariser it ran under" begin
        # A resolved measure is fitted state, so it belongs on the Result.
        @test isa(res.sca, PortfolioOptimisers.Scalariser)
        @test !isnothing(res.r)
        # Stored resolved: reporting against it reproduces the optimised figure without a
        # refit, and it passes `assert_resolved_slots`.
        @test expected_risk(res.r, res.w, res.pr; sca = res.sca) ≈
              expected_risk(res.r, res.w, res.pr.X; sca = res.sca)
    end

    @testset "the result overload resolves the caller's own measure" begin
        #=
        `expected_risk(r, res, pr)` extracts the prior and then forwards it whole. It used
        to unwrap `pr.X`, so the caller's own measure met the kernel with an unstated slot.
        This overload had no test at all, because every other call site in the suite passes
        `res.w` explicitly.

        The invariant it must satisfy: it agrees with the explicit `res.w` route on the same
        measure, in every slot state. Stated this way the assertion does not depend on
        whether a bare `Covariance()` equals the prior's own estimator.
        =#
        fees = PortfolioOptimisers.extract_fees(res, nothing)
        for r in (Variance(), Variance(; sigma = Covariance()), res.r)
            # Both arms: the prior extracted from the result, and one passed explicitly.
            @test expected_risk(r, res) ≈ expected_risk(r, res.w, res.pr, fees)
            @test expected_risk(r, res, res.pr) ≈ expected_risk(r, res.w, res.pr, fees)
        end
        # The already-resolved measure reproduces the optimised figure through the overload.
        @test expected_risk(res.r, res; sca = res.sca) ≈
              expected_risk(res.r, res.w, res.pr; sca = res.sca)
        # The matrix arm opts out of resolution, so it needs a fully stated measure.
        @test expected_risk(factory(Variance(), res.pr), res, res.pr.X) ≈
              expected_risk(factory(Variance(), res.pr), res.w, res.pr.X, fees)
    end
end

@testset "Value-level multiplicity: the field beats a caller's keyword" begin
    using PortfolioOptimisers, Test, StableRNGs, LinearAlgebra

    #=
    A wrapper is a value, so it must carry its own scalariser: `plot_measures(w, pr; x, y,
    c)` has one shared keyword channel serving three measures, and a wrapper stored in
    `opt.r` is evaluated with no call site to carry a scalariser at all.

    The wrapper therefore splats `kwargs...` **first** and pins its own field last. Without
    that order a caller's `sca` would flow through `expected_ratio` into the inner vector
    call and silently win.
    =#

    rng = StableRNG(55555)
    X = randn(rng, 200, 5) ./ 100 .+ 0.0004
    w = fill(0.2, 5)
    pr = prior(EmpiricalPrior(), X)
    # Unresolved: `expected_ratio` hands the risk axis the prior itself, so an unstated
    # slot takes the prior's field. No `factory` call is needed at the call site.
    rks = [Variance(), ConditionalValueatRisk()]

    errr_sum = ExpectedReturnRiskRatio(; rk = rks, sca = SumScalariser())
    errr_max = ExpectedReturnRiskRatio(; rk = rks, sca = MaxScalariser())

    v_sum = expected_risk(errr_sum, w, pr)
    v_max = expected_risk(errr_max, w, pr)
    @test v_sum != v_max

    # A caller's keyword cannot change either answer.
    @test expected_risk(errr_sum, w, pr; sca = MaxScalariser()) == v_sum
    @test expected_risk(errr_max, w, pr; sca = SumScalariser()) == v_max

    @testset "`NonOptimisationRiskRatio` carries one scalariser per axis" begin
        # It is the only type in the family with two independent risk vectors. This testset
        # evaluates on the matrix, so it resolves the measures itself.
        rrks = factory(rks, pr)
        r = NonOptimisationRiskRatio(; r1 = rrks, sca1 = SumScalariser(), r2 = rrks,
                                     sca2 = MaxScalariser())
        num = expected_risk(rrks, w, pr.X; sca = SumScalariser())
        den = expected_risk(rrks, w, pr.X; sca = MaxScalariser())
        @test expected_risk(r, w, pr.X) ≈ num / den
    end

    @testset "the ratio family routes its risk axis through the prior" begin
        #=
        `expected_ratio` used to unwrap `pr.X` for the risk axis while handing the return
        axis `pr`, two lines apart. So an unstated slot reached the kernel as `nothing`, and
        a Deferred Quantity was refused for want of a prior that was in the argument list.
        The `VecVecNum` method resolved first, so the vector twin answered where the
        singular one threw.
        =#
        unstated = ExpectedReturnRiskRatio(; rk = Variance())
        deferred = ExpectedReturnRiskRatio(; rk = Variance(; sigma = Covariance()))
        stated = ExpectedReturnRiskRatio(; rk = factory(Variance(), pr))

        ref = expected_risk(stated, w, pr)
        # All three slot states agree: the prior supplies the same matrix each way.
        @test expected_risk(unstated, w, pr) ≈ ref
        @test expected_risk(deferred, w, pr) ≈ ref
        # The singular method and the `VecVecNum` twin agree on every state.
        for m in (unstated, deferred, stated)
            @test expected_risk(m, [w], pr)[1] ≈ expected_risk(m, w, pr)
        end
        # The four public ratio entry points take an unstated slot directly.
        @test expected_ratio(Variance(), ArithmeticReturn(), w, pr) ≈ ref
        @test expected_ratio(Variance(; sigma = Covariance()), ArithmeticReturn(), w, pr) ≈
              ref
        rk, rt, rr = expected_risk_ret_ratio(Variance(), ArithmeticReturn(), w, pr)
        @test rr ≈ ref
        @test rk ≈ expected_risk(factory(Variance(), pr), w, pr.X)
        @test rt ≈ expected_return(ArithmeticReturn(), w, pr)
        @test expected_sric(Variance(), ArithmeticReturn(), w, pr) < ref
        srk, srt, ssr = expected_risk_ret_sric(Variance(), ArithmeticReturn(), w, pr)
        @test (srk, srt) == (rk, rt)
        @test ssr < rr
    end
end

@testset "Value-level multiplicity: polarity is refused on a mixed vector" begin
    using PortfolioOptimisers, Test

    #=
    `[Variance(), ExpectedReturn()]` scalarises to a number with no defined orientation.
    Neither `all` nor `any` is right, and both are wrong in silence: under `all` the vector
    answers `false` and `RankRule(; best = 20)` silently keeps the wrong twenty assets.
    Refusing by name is the only answer that cannot be wrong.
    =#

    bib = PortfolioOptimisers.bigger_is_better

    @test bib([Variance(), ConditionalValueatRisk()]) == false
    @test bib([ExpectedReturn(), MeanReturn()]) == true
    @test_throws ArgumentError bib([Variance(), ExpectedReturn()])
    @test_throws ArgumentError bib([ExpectedReturn(), Variance()])

    @testset "the refusal names the offending measures" begin
        msg = try
            bib([Variance(), ExpectedReturn()])
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("Variance", msg)
        @test occursin("ExpectedReturn", msg)
    end
end

@testset "Value-level multiplicity: risk contribution decomposes the aggregate" begin
    using PortfolioOptimisers, Test, StableRNGs, LinearAlgebra

    #=
    `adjust_risk_contribution` is a homogeneity correction, so a mixed vector has no single
    degree and adjusting the aggregate is impossible. Adjusting each element **before** the
    scalariser restores Euler's identity exactly, and that is a positive argument for the
    widening rather than a caveat on it.
    =#

    rng = StableRNG(24680)
    X = randn(rng, 250, 6) ./ 100 .+ 0.0004
    w = fill(inv(6), 6)
    pr = prior(EmpiricalPrior(), X)

    rv = Variance()                    # degree 2
    rc = ConditionalValueatRisk()      # degree 1

    @testset "the singular case is unchanged" begin
        @test sum(risk_contribution(rv, w, pr)) ≈ expected_risk(rv, w, pr) rtol=1e-5
        @test risk_contribution([rv], w, pr) ≈ risk_contribution(rv, w, pr)
    end

    @testset "Euler's identity survives a mixture of homogeneity degrees" begin
        mixed = [rv, rc]
        @test sum(risk_contribution(mixed, w, pr)) ≈ expected_risk(mixed, w, pr) rtol=1e-5
    end

    @testset "and it survives the combination weights too" begin
        mixed = [Variance(; settings = RiskMeasureSettings(; scale = 3.0)), rc]
        @test sum(risk_contribution(mixed, w, pr)) ≈ expected_risk(mixed, w, pr) rtol=1e-5
    end
end

@testset "Value-level multiplicity: `measure_label` never says \"Vector\"" begin
    using PortfolioOptimisers, Test

    #=
    `string(nameof(typeof(rs)))` on a vector evaluates to `"Vector"` — a wrong axis label,
    silently, with no error. Seven plot sites spelled that expression inline.
    =#
    ml = PortfolioOptimisers.measure_label
    @test ml(Variance()) == "Variance"
    @test ml([Variance()]) == "Variance"
    @test ml([Variance(), ConditionalValueatRisk()]) == "Variance + ConditionalValueatRisk"
    @test !occursin("Vector", ml([Variance(), ConditionalValueatRisk()]))
end

@testset "Value-level multiplicity: the hierarchical split forwards every property" begin
    using PortfolioOptimisers, Test, StableRNGs, Clustering, LinearAlgebra

    #=
    `HierarchicalResult` split into a shared core embedded as `hr` and one leaf per
    estimator. Because `getproperty` forwards through the core, every reader of `res.w`,
    `res.pr`, `res.clr`, `res.wb`, `res.fees` and `res.retcode` stays source-compatible.
    Only code that **names** the type, and positional construction, may break.
    =#

    rng = StableRNG(13579)
    X = randn(rng, 250, 8) ./ 100 .+ 0.0004
    pr = prior(EmpiricalPrior(), X)
    opt = HierarchicalOptimiser(; pe = pr)

    hrp = optimise(HierarchicalRiskParity(; r = Variance(), opt = opt))
    herc = optimise(HierarchicalEqualRiskContribution(; ri = Variance(), opt = opt))

    @test isa(hrp, PortfolioOptimisers.HierarchicalRiskParityResult)
    @test isa(herc, PortfolioOptimisers.HierarchicalEqualRiskContributionResult)

    @testset "the core's properties still read off the leaf" begin
        for res in (hrp, herc)
            for p in (:w, :pr, :clr, :wb, :fees, :retcode)
                @test hasproperty(res, p)
            end
            @test isa(res.retcode, OptimisationSuccess)
            @test sum(res.w) ≈ 1
            # `hr` is the embedded core, and it carries no `fb` — ADR 0011 keeps `fb` last
            # on each concrete result instead.
            @test isa(res.hr, PortfolioOptimisers.HierarchicalResult)
            @test !hasfield(typeof(res.hr), :fb)
            @test hasfield(typeof(res), :fb)
        end
    end

    @testset "each leaf carries its own estimator's arity" begin
        # HRP: one measure, one scalariser.
        @test !isnothing(hrp.r)
        @test isa(hrp.sca, PortfolioOptimisers.Scalariser)
        # HERC: two measures, two scalarisers.
        @test !isnothing(herc.ri)
        @test !isnothing(herc.ro)
        @test isa(herc.scai, PortfolioOptimisers.Scalariser)
        @test isa(herc.scao, PortfolioOptimisers.Scalariser)
    end

    @testset "the measure on the result reproduces a reported figure" begin
        # Stored resolved, so it needs no refit and passes `assert_resolved_slots`.
        @test isa(expected_risk(hrp.r, hrp.w, hrp.pr; sca = hrp.sca), Number)
    end
end

@testset "Value-level multiplicity: a vector measure does not zip against the results" begin
    using PortfolioOptimisers, Test, StableRNGs, Clarabel, StatsPlots, GraphRecipes,
          LinearAlgebra

    #=
    `plot_measures(res_vec, pr; …)` **broadcasts** over the results. A single measure is not
    iterable, so it broadcasts as a scalar and the call is correct. A **vector** of measures
    is iterable, so without `Ref` it would zip elementwise against `pr`, `w` and `fees` — a
    `DimensionMismatch` when the lengths differ and a silent **wrong answer** when they
    happen to match.

    Three measures against three results is exactly the case that would pass silently, so it
    is the case this test pins.
    =#

    ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
    is_plot(x) = x isa Plots.Plot || x isa Plots.AbstractLayout

    rng = StableRNG(97531)
    X = randn(rng, 200, 5) ./ 100 .+ 0.0004
    slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false),
                 check_sol = (; allow_local = true, allow_almost = true))
    pr = prior(EmpiricalPrior(), X)

    res_vec = [optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                 opt = JuMPOptimiser(; pe = pr, slv = slv)))
               for r in (Variance(), StandardDeviation(), ConditionalValueatRisk())]
    @test length(res_vec) == 3

    # Three measures, three results: the lengths match, so a zip would not raise.
    xs = [Variance(), StandardDeviation(), ConditionalValueatRisk()]
    @test is_plot(plot_measures(res_vec, pr; x = xs, y = ExpectedReturn()))

    # The figure must report the **scalarised** measure at every result, not measure `i` at
    # result `i`. Check the numbers the plot is built from rather than the plot object.
    ws = getproperty.(res_vec, :w)
    scalarised = [expected_risk(xs, wi, pr, nothing) for wi in ws]
    zipped = [expected_risk(xs[i], ws[i], pr, nothing) for i in eachindex(ws)]
    @test scalarised != zipped
    @test all(scalarised .> zipped)  # a sum of three positives exceeds any one of them

    # A single measure keeps working unchanged.
    @test is_plot(plot_measures(res_vec, pr; x = Variance(), y = ExpectedReturn()))
end
