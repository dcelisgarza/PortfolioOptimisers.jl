#=
Check `src/08_Moments/22_StepwiseRegression.jl` and
`src/08_Moments/23_DimensionReductionRegression.jl` against the mathematics and the procedures
their docstrings state. Issue #464 of child map 3 (#417) under the map of maps (#404).

#404's condition 2 is "checked with real numbers. Not read -- run". Every testset below pins a
CLAIM the two files make, computed a second way.

FIVE FACTS SHAPE THE PROBES.

1. BOTH SELECTION HELPERS SEARCH THE WHOLE `value` VECTOR, not the live set. The docstrings
   argue that this is still correct: an entry of a factor that is no longer live holds the
   score that moved it, `t` is that same score, and `t` only moves one way, so a score
   strictly beyond `t` can only belong to a live factor. `The whole-vector search` runs the
   library against a reference search that scans the live set alone, and
   `The selection helpers in isolation` feeds each helper a `value` vector whose global
   extremum IS the stale entry -- the exact case the argument is about.

2. THE TWO DIRECTIONS DIFFER IN WHETHER THE SELECTION CAN EMPTY. Forward selection starts at
   `regression_threshold`, the worst score the criterion can take, so its first addition
   always happens. Backward elimination starts at the score of the FULL model, so a criterion
   that rewards every removal removes every factor. `The empty selection` pins both, and pins
   that the `Regression` such a run produces is an intercept-only model and still usable.

3. `included` IS IN INSERTION ORDER, NOT SORTED. Forward selection pushes each winner onto the
   end, so `:r2` on the sample below returns `[1, 3, 4, 5, 2]`. `regression` then writes the
   fitted coefficients into the columns that order names, and `Insertion order` checks the
   whole row against a plain least-squares fit rather than trusting the permutation.

4. THE STANDARDISATION AND THE RECOVERY MUST READ THE SAME STATISTICS. That was issue #398.
   `The round trip` builds the matched pair and the mismatched pair from the same sample and
   compares the two predictions; matched agrees to 4.4e-16 and mismatched parts by 2.1e-3.

5. `PPCA` CANNOT RETAIN EVERY COMPONENT. `MultivariateStats` caps a latent-variable model at
   one less than the input dimension. The fit at the full width SUCCEEDS and
   `MultivariateStats.projection` then raises, so the failure surfaces inside
   `prep_dim_red_reg`. `The retained width` pins the cap and the raise.
=#

const PO = PortfolioOptimisers
const StatsAPI = PortfolioOptimisers.StatsAPI
const MVS = PortfolioOptimisers.MultivariateStats

# Reference forward search that scans the EXCLUDED set only, never the whole `value` vector.
function ref_forward_selection(crit, tgt, y, F; ismin::Bool)
    T, N = size(F)
    ovec = ones(T)
    cf = PO.regression_criterion_func(crit, tgt)
    included = Int[]
    excluded = collect(1:N)
    t = ismin ? Inf : -Inf
    while true
        best, bestval = 0, ismin ? Inf : -Inf
        for i in excluded
            s = cf(StatsAPI.fit(tgt, [ovec F[:, [included; i]]], y))
            if (ismin && s < bestval) || (!ismin && s > bestval)
                best, bestval = i, s
            end
        end
        if (ismin && bestval < t) || (!ismin && bestval > t)
            t = bestval
            push!(included, best)
            deleteat!(excluded, findfirst(==(best), excluded))
        else
            break
        end
        isempty(excluded) && break
    end
    return included
end

# Reference backward search that scans the INCLUDED set only.
function ref_backward_elimination(crit, tgt, y, F; ismin::Bool)
    T, N = size(F)
    ovec = ones(T)
    cf = PO.regression_criterion_func(crit, tgt)
    included = collect(1:N)
    t = cf(StatsAPI.fit(tgt, [ovec F], y))
    while true
        best, bestval = 0, ismin ? Inf : -Inf
        for (i, _) in pairs(included)
            factors = copy(included)
            popat!(factors, i)
            f1 = isempty(factors) ? reshape(ovec, :, 1) : [ovec F[:, factors]]
            s = cf(StatsAPI.fit(tgt, f1, y))
            if (ismin && s < bestval) || (!ismin && s > bestval)
                best, bestval = i, s
            end
        end
        if iszero(best)
            break
        end
        if (ismin && bestval < t) || (!ismin && bestval > t)
            t = bestval
            popat!(included, best)
        else
            break
        end
        isempty(included) && break
    end
    return included
end

@testset "Stepwise and dimension reduction regression" begin
    using Test, PortfolioOptimisers, StableRNGs, StatsBase, LinearAlgebra, Statistics

    rng = StableRNG(987654321)
    T, N = 200, 5
    F = randn(rng, T, N)
    y = 0.8 .* F[:, 1] .+ 0.3 .* F[:, 3] .+ 0.1 .* randn(rng, T)
    ynoise = randn(rng, T)
    X = hcat(y, ynoise)
    ovec = ones(T)
    tgt = PO.LinearModel()

    @testset "The whole-vector search" begin
        # Both helpers call `findmin`/`findmax` over every entry of `value`, and the
        # docstrings claim the answer is still the best LIVE factor. The reference searches
        # scan the live set alone, so an agreement over all five criteria and both directions
        # is the claim itself.
        for c in PO.STEPWISE_REGRESSION_CRITERIA
            ismin = c in PO.MIN_VAL_STEPWISE_REGRESSION_CRITERIA
            lib_f = PO._regression(StepwiseRegression(; crit = c, alg = ForwardSelection()),
                                   y, F)
            @test collect(lib_f) == ref_forward_selection(Val(c), tgt, y, F; ismin = ismin)

            lib_b = PO._regression(StepwiseRegression(; crit = c,
                                                      alg = BackwardElimination()), y, F)
            @test sort(collect(lib_b)) ==
                  sort(ref_backward_elimination(Val(c), tgt, y, F; ismin = ismin))
        end
    end

    @testset "The selection helpers in isolation" begin
        # FORWARD, MINIMISED. `value[1]` is the score that selected factor 1, so it equals
        # the current `t`. It is also the global minimum here, which is exactly the case the
        # docstring's argument covers: the strict `val < t` test rejects it, and nothing
        # moves.
        value = [10.0, 12.0, 15.0]
        excluded, included = [2, 3], [1]
        @test PO.get_forward_reg_incl_excl!(Val(:aic), value, excluded, included, 10.0) ==
              10.0
        @test excluded == [2, 3]
        @test included == [1]

        # The same shape with a live factor that does beat `t`: it moves, and `t` becomes its
        # score.
        value = [10.0, 2.0, 15.0]
        excluded, included = [2, 3], [1]
        @test PO.get_forward_reg_incl_excl!(Val(:aic), value, excluded, included, 10.0) ==
              2.0
        @test excluded == [3]
        @test included == [1, 2]

        # FORWARD, MAXIMISED. `t` rises, so a stale entry is at or BELOW it.
        value = [0.9, 0.8, 0.7]
        excluded, included = [2, 3], [1]
        @test PO.get_forward_reg_incl_excl!(Val(:r2), value, excluded, included, 0.9) == 0.9
        @test excluded == [2, 3]
        @test included == [1]

        value = [0.9, 0.95, 0.7]
        excluded, included = [2, 3], [1]
        @test PO.get_forward_reg_incl_excl!(Val(:r2), value, excluded, included, 0.9) ==
              0.95
        @test excluded == [3]
        @test included == [1, 2]

        # BACKWARD, MINIMISED. `value[2]` is the score that removed factor 2 and it equals
        # `t`; it is the global minimum, and the strict test rejects it.
        value = [5.0, 3.0, 4.0]
        included = [1, 3]
        @test PO.get_backward_reg_incl!(Val(:aic), value, included, 3.0) == 3.0
        @test included == [1, 3]

        value = [5.0, 3.0, 4.0]
        included = [1, 2, 3]
        @test PO.get_backward_reg_incl!(Val(:aic), value, included, 6.0) == 3.0
        @test included == [1, 3]

        # BACKWARD, MAXIMISED.
        value = [0.5, 0.9, 0.6]
        included = [1, 3]
        @test PO.get_backward_reg_incl!(Val(:r2), value, included, 0.9) == 0.9
        @test included == [1, 3]

        value = [0.5, 0.9, 0.6]
        included = [1, 2, 3]
        @test PO.get_backward_reg_incl!(Val(:r2), value, included, 0.4) == 0.9
        @test included == [1, 3]
    end

    @testset "The backward reading of best" begin
        # `value[j]` is the score of the model that OMITS `j`, so the code takes the same
        # extremum the criterion does: the LOWEST value under `:aic`. The docstring quotes
        # these six numbers.
        cf = PO.regression_criterion_func(Val(:aic), tgt)
        full = cf(StatsAPI.fit(tgt, [ovec F], y))
        vals = [cf(StatsAPI.fit(tgt, [ovec F[:, setdiff(1:N, j)]], y)) for j in 1:N]
        @test isapprox(full, -371.52; atol = 5e-3)
        @test isapprox(vals, [487.92, -372.72, 82.79, -369.15, -372.44]; atol = 5e-3)

        # Factor 2 is the lowest, and it is the one removed first. Factor 1 is the highest --
        # dropping it costs the most -- and it survives.
        @test argmin(vals) == 2
        @test argmax(vals) == 1
        @test sort(collect(PO._regression(StepwiseRegression(; crit = :aic,
                                                             alg = BackwardElimination()),
                                          y, F))) == [1, 3, 4]
    end

    @testset "The empty selection" begin
        # FORWARD never empties: `t` starts at the worst score the criterion can take, so the
        # first addition always happens. `regression_threshold` is that starting score.
        @test PO.regression_threshold(Val(:aic)) == Inf
        @test PO.regression_threshold(Val(:r2)) == -Inf
        rz = StableRNG(4321)
        Fz, yz = randn(rz, 50, 3), randn(rz, 50)
        for c in PO.STEPWISE_REGRESSION_CRITERIA
            @test !isempty(PO._regression(StepwiseRegression(; crit = c,
                                                             alg = ForwardSelection()), yz,
                                          Fz))
        end

        # BACKWARD can empty: it starts at the score of the full model, and on a response the
        # factors do not explain every removal improves `:aic`.
        rn = StableRNG(11)
        Fn, yn = randn(rn, 40, 4), randn(rn, 40)
        @test isempty(PO._regression(StepwiseRegression(; crit = :aic,
                                                        alg = BackwardElimination()), yn,
                                     Fn))

        # The `Regression` such a run produces is an intercept-only model, and it is usable:
        # the intercept is the response mean and the whole loadings row is zero.
        reg = regression(StepwiseRegression(; crit = :aic, alg = BackwardElimination()),
                         reshape(yn, :, 1), Fn)
        @test isapprox(reg.b[1], mean(yn))
        @test all(iszero, reg.M)

        # A p-value search never empties in either direction, because the helper adds one.
        for alg in (ForwardSelection(), BackwardElimination())
            inc = @test_logs (:warn,) match_mode = :any PO._regression(StepwiseRegression(;
                                                                                          crit = PValue(;
                                                                                                        t = 1e-12),
                                                                                          alg = alg),
                                                                       yn, Fn)
            @test length(inc) == 1
        end
    end

    @testset "add_best_factor_after_pval_failure!" begin
        # An empty `included` gains exactly the factor of smallest single-factor p-value, and
        # the call warns because that factor failed the threshold by construction.
        pv = [StatsAPI.coeftable(StatsAPI.fit(tgt, [ovec F[:, [i]]], ynoise)).cols[4][2]
              for i in 1:N]
        included = Int[]
        @test_logs (:warn,) PO.add_best_factor_after_pval_failure!(tgt, included, F, ynoise)
        @test included == [argmin(pv)]

        # A non-empty `included` is left alone, and nothing is logged.
        included = [3]
        @test_logs PO.add_best_factor_after_pval_failure!(tgt, included, F, ynoise)
        @test included == [3]
    end

    @testset "Insertion order" begin
        # `:r2` never falls when a factor is added, so forward selection takes all five --
        # in the order it added them, which is not sorted.
        inc = PO._regression(StepwiseRegression(; crit = :r2, alg = ForwardSelection()), y,
                             F)
        @test collect(inc) == [1, 3, 4, 5, 2]
        @test !issorted(inc)

        # `regression` writes each coefficient into the column that order names. With every
        # factor selected the row must equal a plain least-squares fit on all five.
        reg = regression(StepwiseRegression(; crit = :r2, alg = ForwardSelection()),
                         reshape(y, :, 1), F)
        ols = StatsAPI.coef(StatsAPI.fit(tgt, [ovec F], y))
        @test isapprox(vec(reg.M), ols[2:end])
        @test isapprox(reg.b[1], ols[1])
    end

    @testset "The unselected factors are exact zeros" begin
        reg = regression(StepwiseRegression(; crit = :bic, alg = ForwardSelection()), X, F)
        inc1 = PO._regression(StepwiseRegression(; crit = :bic, alg = ForwardSelection()),
                              y, F)
        @test count(iszero, reg.M[1, :]) == N - length(inc1)
        for j in setdiff(1:N, inc1)
            @test reg.M[1, j] === zero(eltype(reg.M))
        end

        # `L` is left unset, so it reads back as `M` through the result's `swap(L, M)` rule.
        @test isnothing(getfield(reg, :L))
        @test reg.L === reg.M
        @test size(reg.L, 2) == N
    end

    @testset "Forward and backward part on a collinear design" begin
        # Two factors that carry almost the same information: forward takes one of them and
        # backward keeps the other, so the two directions land on different sets.
        r = StableRNG(1)
        T2 = 60
        Z = randn(r, T2, 2)
        Fc = hcat(Z[:, 1], Z[:, 1] .+ 0.1 .* randn(r, T2), Z[:, 2],
                  Z[:, 1] .+ Z[:, 2] .+ 0.2 .* randn(r, T2), randn(r, T2))
        yc = Z[:, 1] .+ 0.5 .* Z[:, 2] .+ 0.5 .* randn(r, T2)
        f = sort(collect(PO._regression(StepwiseRegression(; crit = :aic,
                                                           alg = ForwardSelection()), yc,
                                        Fc)))
        b = sort(collect(PO._regression(StepwiseRegression(; crit = :aic,
                                                           alg = BackwardElimination()), yc,
                                        Fc)))
        @test f == [2, 4]
        @test b == [2, 3]
        @test f != b

        # `:r2` rewards every addition and no removal, so the two directions agree there.
        f2 = sort(collect(PO._regression(StepwiseRegression(; crit = :r2,
                                                            alg = ForwardSelection()), yc,
                                         Fc)))
        b2 = sort(collect(PO._regression(StepwiseRegression(; crit = :r2,
                                                            alg = BackwardElimination()),
                                         yc, Fc)))
        @test f2 == b2 == collect(1:5)
    end

    @testset "The p-value threshold" begin
        # `t` is an OPEN unit interval: the two ends raise.
        @test_throws DomainError PValue(; t = 0.0)
        @test_throws DomainError PValue(; t = 1.0)

        # A threshold near one admits every factor; one near zero admits only the strongest.
        for alg in (ForwardSelection(), BackwardElimination())
            @test sort(collect(PO._regression(StepwiseRegression(;
                                                                 crit = PValue(; t = 0.999),
                                                                 alg = alg), y, F))) ==
                  collect(1:N)
            @test sort(collect(PO._regression(StepwiseRegression(;
                                                                 crit = PValue(; t = 1e-12),
                                                                 alg = alg), y, F))) ==
                  [1, 3]
        end

        # Backward returns the `1:size(F, 2)` range itself when the full model already passes,
        # and a `Vector` once it removes anything.
        re = StepwiseRegression(; crit = PValue(; t = 0.999), alg = BackwardElimination())
        @test PO._regression(re, y, F) === 1:N
        re = StepwiseRegression(; crit = PValue(; t = 1e-12), alg = BackwardElimination())
        @test isa(PO._regression(re, y, F), Vector{Int})
    end

    @testset "The retained width" begin
        # `PCA()` retains every component of a full-rank factor matrix and reduces nothing.
        # `PPCA()` retains one fewer, because `MultivariateStats` caps a latent-variable
        # model at one less than the input dimension.
        for n in (2, 3, 5, 7)
            r = StableRNG(1234 + n)
            Fn = randn(r, 300, n)
            Fs = permutedims((Fn .- mean(Fn; dims = 1)) ./ std(Fn; dims = 1))
            @test size(MVS.projection(StatsAPI.fit(PCA(), Fs)), 2) == n
            @test size(MVS.projection(StatsAPI.fit(PPCA(), Fs)), 2) == n - 1
        end

        # `kwargs` is the only place the width is set.
        r = StableRNG(99)
        Fn = randn(r, 300, 5)
        Fs = permutedims((Fn .- mean(Fn; dims = 1)) ./ std(Fn; dims = 1))
        @test size(MVS.projection(StatsAPI.fit(PCA(; kwargs = (; pratio = 0.8)), Fs)), 2) ==
              4
        @test size(MVS.projection(StatsAPI.fit(PCA(; kwargs = (; maxoutdim = 2)), Fs)),
                   2) == 2
        @test size(MVS.projection(StatsAPI.fit(PPCA(; kwargs = (; maxoutdim = 4)), Fs)),
                   2) == 4

        # The cap is hard. The FIT at the full width succeeds, and `projection` raises, so a
        # caller meets the failure inside `prep_dim_red_reg` rather than at construction.
        model = StatsAPI.fit(PPCA(; kwargs = (; maxoutdim = 5)), Fs)
        @test_throws ArgumentError MVS.projection(model)
        @test_throws ArgumentError PO.prep_dim_red_reg(DimensionReductionRegression(;
                                                                                    drtgt = PPCA(;
                                                                                                 kwargs = (;
                                                                                                           maxoutdim = 5))),
                                                       Fn)
    end

    @testset "prep_dim_red_reg reads its statistics from ve" begin
        re = DimensionReductionRegression()
        x1, Vp, mu, sigma = PO.prep_dim_red_reg(re, F)
        @test size(x1) == (T, N + 1)
        @test all(isone, view(x1, :, 1))
        @test size(Vp) == (N, N)
        @test isapprox(mu, vec(mean(F; dims = 1)))
        @test isapprox(sigma, vec(std(F; dims = 1)))

        # `factory` writes the incoming weights into `ve`, and both statistics follow.
        w = pweights(range(0.5, 1.5; length = T))
        rew = factory(DimensionReductionRegression(), w)
        _, _, muw, sigmaw = PO.prep_dim_red_reg(rew, F)
        @test isapprox(muw, vec(mean(F, w; dims = 1)))
        @test isapprox(sigmaw, [std(F[:, j], w; corrected = true) for j in 1:N])
        @test !isapprox(sigmaw, vec(std(F; dims = 1)))

        # A constant factor cannot divide by zero: its standard deviation is floored at `eps`.
        Fc = hcat(F[:, 1], fill(2.5, T))
        _, _, _, sigmac = PO.prep_dim_red_reg(re, Fc)
        @test sigmac[2] == eps(eltype(sigmac))
        @test all(isfinite, PO.prep_dim_red_reg(re, Fc)[1])
    end

    @testset "The round trip" begin
        # The recovery divides by the scale that standardised the factors, so a prediction
        # built from the recovered coefficients reproduces the reduced-space fit exactly.
        w = pweights(range(0.5, 1.5; length = T))
        function roundtrip(re)
            x1, Vp, mu, sigma = PO.prep_dim_red_reg(re, F)
            beta = PO._regression(re, y, mu, sigma, x1, Vp)
            pred = beta[1] .+ F * beta[2:end]
            pred_pc = StatsAPI.predict(StatsAPI.fit(re.retgt, x1, y))
            return maximum(abs, pred .- pred_pc)
        end
        @test roundtrip(DimensionReductionRegression()) < 1e-14
        @test roundtrip(factory(DimensionReductionRegression(), w)) < 1e-14

        # Standardise unweighted and fit weighted -- issue #398's shape -- and the two paths
        # part. The scale is what binds them, not the fit.
        mismatch = DimensionReductionRegression(; ve = SimpleVariance(),
                                                retgt = LinearModel(;
                                                                    kwargs = (;
                                                                              weights = w)))
        @test isapprox(roundtrip(mismatch) / maximum(abs, y), 2.1e-3; atol = 5e-5)
    end

    @testset "The recovered coefficients" begin
        # Equations 4.18 to 4.20 computed by hand against `_regression`.
        re = DimensionReductionRegression()
        x1, Vp, mu, sigma = PO.prep_dim_red_reg(re, F)
        beta = PO._regression(re, y, mu, sigma, x1, Vp)
        beta_pc = StatsAPI.coef(StatsAPI.fit(re.retgt, x1, y))[2:end]
        @test isapprox(beta[2:end], Vp * beta_pc ./ sigma)
        @test isapprox(beta[1], mean(y) - dot(beta[2:end], mu))

        # The reduced-space intercept is discarded, and Equation 4.20 says it costs nothing:
        # unweighted, it already equals the response mean.
        @test isapprox(StatsAPI.coef(StatsAPI.fit(re.retgt, x1, y))[1], mean(y))
    end

    @testset "The Regression of a dimension reduction fit" begin
        reg = regression(DimensionReductionRegression(), X, F)

        # `L` undoes the rescaling and the projection in turn, so it returns the reduced-space
        # coefficients the per-asset fits produced.
        x1, Vp, _, _ = PO.prep_dim_red_reg(DimensionReductionRegression(), F)
        betapc = hcat([StatsAPI.coef(StatsAPI.fit(LinearModel(), x1, X[:, i]))[2:end]
                       for i in axes(X, 2)]...)
        @test isapprox(reg.L, transpose(betapc); atol = 1e-14)
        @test size(reg.L) == (size(X, 2), N)
        @test size(reg.M) == (size(X, 2), N)

        # Every asset keeps every factor, unlike a stepwise fit.
        @test !any(iszero, reg.M)

        # `size(L, 2)` is the retained width, and `maxoutdim` moves it while `M` stays full.
        for d in (2, 3)
            r = regression(DimensionReductionRegression(;
                                                        drtgt = PCA(;
                                                                    kwargs = (;
                                                                              maxoutdim = d))),
                           X, F)
            @test size(r.L) == (size(X, 2), d)
            @test size(r.M) == (size(X, 2), N)
        end

        # The reduction is fitted ONCE, on `F` alone, so adding a second asset cannot move the
        # first asset's loadings.
        reg1 = regression(DimensionReductionRegression(), reshape(y, :, 1), F)
        @test isapprox(reg.M[1, :], reg1.M[1, :])
        @test isapprox(reg.b[1], reg1.b[1])
    end

    @testset "factory leaves a dimension reduction target alone" begin
        w = pweights(range(0.5, 1.5; length = T))
        for tg in (PCA(), PPCA(), PCA(; kwargs = (; maxoutdim = 2)))
            @test factory(tg, w) === tg
            @test factory(tg) === tg
        end

        # It reaches the target through the estimator too, and the weights land in `ve` and
        # `retgt` instead.
        re = factory(DimensionReductionRegression(), w)
        @test re.drtgt === PCA()
        @test re.ve.w === w
        @test re.retgt.kwargs.weights === w
    end
end
