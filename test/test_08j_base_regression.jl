#=
Check `src/05_Moments/20_Base_Regression.jl` against the mathematics its docstrings state.
Issue #463 of child map 3 (#417) under the map of maps (#404).

#404's condition 2 is "checked with real numbers. Not read -- run". Every testset below pins
a CLAIM the file's documentation makes, computed a second way.

FOUR FACTS SHAPE THE PROBES.

1. `k` MEANS TWO DIFFERENT THINGS IN ONE FILE. `:aic`, `:aicc` and `:bic` take `k` from
   `StatsAPI.dof`, which counts the regression coefficients, the intercept AND the residual
   variance: `dof == 7` for five predictors. `:adjr2` takes `k` as the predictors alone,
   `k == 5` for the same model. `The information criteria` and `The R^2 criteria` pin both
   counts against the same fitted model, so a reader who confuses them sees a red test.

2. A GENERALISED LINEAR MODEL HAS NO CLASSICAL R^2, so it reads a NAMED pseudo-R^2 variant.
   `StatsAPI.r2` accepts all four members of `PSEUDO_R2_VARIANTS`, and `StatsAPI.adjr2`
   accepts only the two of `ADJUSTED_PSEUDO_R2_VARIANTS`. The difference is the whole reason
   the library carries two tuples rather than one, and `The two variant tuples` holds it in
   both directions.

3. `:devianceratio` IS THE DEFAULT BECAUSE IT IS CONTINUOUS WITH THE LINEAR PATH. On a
   Normal-family generalised linear model the deviance is the residual sum of squares, so
   `r2(model, :devianceratio)` equals the classical `r2` of the linear model fitted to the
   same data. `The default variant` computes both and compares them. No other member of
   `PSEUDO_R2_VARIANTS` has that property: `:McFadden` and `:Nagelkerke` are far from it on
   the same fit.

4. AN UNSET `L` READS BACK AS `M`. A `@forward_properties` `swap(L, M)` rule makes `re.L`
   return `re.M` when `L` was never given, so `isnothing(re.L)` is NEVER true and only
   `getfield(re, :L)` tells the two cases apart. `An unset L` pins the rule, and
   `port_opt_view keeps an unset L unset` pins the consumer that has to know.

The weight type lives at top level because `@testset` expands to a function body, which
cannot hold a `struct`.
=#

# Resolves to flat weights of the right length on both arities. `StatsAPI.fit` must call
# `get_observation_weights` on it before handing the weights to GLM.
struct RegressionDynWeights <: PortfolioOptimisers.DynamicAbstractWeights end
function PortfolioOptimisers.get_observation_weights(::RegressionDynWeights,
                                                     X::PortfolioOptimisers.VecNum;
                                                     kwargs...)
    return aweights(range(0.5, 1.5; length = length(X)))
end
function PortfolioOptimisers.get_observation_weights(::RegressionDynWeights,
                                                     X::PortfolioOptimisers.MatNum;
                                                     dims::Int = 1, kwargs...)
    return aweights(range(0.5, 1.5; length = size(X, dims)))
end

const PO = PortfolioOptimisers
const GLM = PortfolioOptimisers.GLM
const StatsAPI = PortfolioOptimisers.StatsAPI

# `Accessors.@set` expands when the test set is lowered, before a `using` inside it runs.
using Accessors: Accessors
@testset "Base regression: criteria, targets and the Regression result" begin
    using Test, PortfolioOptimisers, StableRNGs, StatsBase, LinearAlgebra, Distributions

    rng = StableRNG(987654321)
    T, N = 200, 5
    F = randn(rng, T, N)
    y = 0.8 .* F[:, 1] .+ 0.3 .* F[:, 3] .+ 0.1 .* randn(rng, T)
    ovec = ones(T)
    Xd = hcat(ovec, F)
    lm = GLM.lm(Xd, y)
    rd = PO.ReturnsResult(; nx = ["y"], X = reshape(y, :, 1),
                          nf = ["f1", "f2", "f3", "f4", "f5"], F = F)

    @testset "The information criteria" begin
        # `k` is `StatsAPI.dof`: five coefficients, the intercept, and the residual
        # variance. The docstring of `STEPWISE_REGRESSION_CRITERIA` says so at three sites,
        # and every one of the three formulas below reads this same `k`.
        k = StatsAPI.dof(lm)
        @test k == N + 2 == 7

        ll = StatsAPI.loglikelihood(lm)
        @test 2k - 2ll ≈ StatsAPI.aic(lm)
        @test 2k - 2ll + 2k * (k + 1) / (T - k - 1) ≈ StatsAPI.aicc(lm)
        @test k * log(T) - 2ll ≈ StatsAPI.bic(lm)

        # `:aicc` adds a POSITIVE correction whenever the sample is longer than `k + 1`, so
        # it is the stricter of the two on any sample the search can reach.
        @test StatsAPI.aicc(lm) > StatsAPI.aic(lm)

        # `:bic`'s penalty `k*log(T)` beats `:aic`'s `2k` for `T >= 8`, which is the claim
        # the `:bic` subsection makes. Check the crossing itself, not one sample.
        @test k * log(7) < 2k
        @test k * log(8) > 2k
    end

    @testset "The R^2 criteria" begin
        r2_hand = 1 - sum(abs2, StatsAPI.residuals(lm)) / sum(abs2, y .- mean(y))
        @test r2_hand ≈ StatsAPI.r2(lm)

        # `:adjr2` reads a DIFFERENT `k`: the predictors alone, without the intercept and
        # without the residual variance. Five, against the seven of `StatsAPI.dof`.
        kp = N
        @test kp != StatsAPI.dof(lm)
        @test 1 - (1 - r2_hand) * (T - 1) / (T - kp - 1) ≈ StatsAPI.adjr2(lm)

        # `R^2` never falls when a factor is added, so it penalises no complexity at all.
        # The adjusted form can fall, which is what makes it usable as a stopping rule.
        @test StatsAPI.adjr2(lm) < StatsAPI.r2(lm)
    end

    @testset "The pseudo-R^2 variants" begin
        # A Normal-family generalised linear model on the same design.
        gm = GLM.glm(Xd, y, Distributions.Normal())
        gm0 = GLM.glm(reshape(ovec, :, 1), y, Distributions.Normal())
        L1, L0 = StatsAPI.loglikelihood(gm), StatsAPI.loglikelihood(gm0)
        D1, D0 = StatsAPI.deviance(gm), StatsAPI.deviance(gm0)
        k = StatsAPI.dof(gm)

        @test 1 - L1 / L0 ≈ StatsAPI.r2(gm, :McFadden)
        @test 1 - exp(2 * (L0 - L1) / T) ≈ StatsAPI.r2(gm, :CoxSnell)
        @test (1 - exp(2 * (L0 - L1) / T)) / (1 - exp(2 * L0 / T)) ≈
              StatsAPI.r2(gm, :Nagelkerke)
        @test 1 - D1 / D0 ≈ StatsAPI.r2(gm, :devianceratio)

        @test 1 - (L1 - k) / L0 ≈ StatsAPI.adjr2(gm, :McFadden)
        @test 1 - D1 * (T - 1) / (D0 * (T - k)) ≈ StatsAPI.adjr2(gm, :devianceratio)

        # On a Normal family the maximum likelihood carries the fitted dispersion, so
        # `(L0/L)^(2/T)` IS `D/D0` and `:CoxSnell` equals `:devianceratio` identically. The
        # `PSEUDO_R2_VARIANTS` docstring states that consequence.
        @test StatsAPI.r2(gm, :CoxSnell) ≈ StatsAPI.r2(gm, :devianceratio)
        # The same family's likelihood is a DENSITY, not a probability, so `L0` sits on
        # either side of one and the two log-likelihood forms leave `[0, 1]` in either
        # direction. Rescaling the response alone moves them from above one to below zero,
        # while the deviance ratio does not move at all.
        @test !(0 <= StatsAPI.r2(gm, :McFadden) <= 1)
        @test !(0 <= StatsAPI.r2(gm, :Nagelkerke) <= 1)
        @test exp(L0) < 1

        yb = (y .- minimum(y)) ./ (maximum(y) - minimum(y))
        gb = GLM.glm(Xd, yb, Distributions.Normal())
        gb0 = GLM.glm(reshape(ovec, :, 1), yb, Distributions.Normal())
        @test exp(StatsAPI.loglikelihood(gb0)) > 1
        @test StatsAPI.r2(gm, :McFadden) > 1
        @test StatsAPI.r2(gb, :McFadden) < 0
        @test StatsAPI.r2(gm, :Nagelkerke) > 1
        @test StatsAPI.r2(gb, :Nagelkerke) < 0
        # The deviance ratio is invariant under that rescaling, which is the property the
        # default rests on.
        @test StatsAPI.r2(gb, :devianceratio) ≈ StatsAPI.r2(gm, :devianceratio)
    end

    @testset "The two variant tuples" begin
        @test PO.PSEUDO_R2_VARIANTS == (:McFadden, :CoxSnell, :Nagelkerke, :devianceratio)
        @test PO.ADJUSTED_PSEUDO_R2_VARIANTS == (:McFadden, :devianceratio)
        # The second tuple is a STRICT subset of the first, which is why the constructor of
        # `GeneralisedLinearModel` checks against the wider one and `StepwiseRegression`
        # rejects the difference.
        @test all(v -> v in PO.PSEUDO_R2_VARIANTS, PO.ADJUSTED_PSEUDO_R2_VARIANTS)
        @test length(PO.ADJUSTED_PSEUDO_R2_VARIANTS) < length(PO.PSEUDO_R2_VARIANTS)

        gm = GLM.glm(Xd, y, Distributions.Normal())
        # `r2` takes every member of the wider tuple...
        for v in PO.PSEUDO_R2_VARIANTS
            @test isa(StatsAPI.r2(gm, v), Number)
        end
        # ...and `adjr2` takes the two of the narrower one and RAISES on the rest. That
        # difference is the whole reason two tuples exist.
        for v in PO.PSEUDO_R2_VARIANTS
            if v in PO.ADJUSTED_PSEUDO_R2_VARIANTS
                @test isa(StatsAPI.adjr2(gm, v), Number)
            else
                @test_throws ArgumentError StatsAPI.adjr2(gm, v)
            end
        end
        # Neither takes a fitted generalised linear model with NO variant at all.
        @test_throws MethodError StatsAPI.r2(gm)
        @test_throws MethodError StatsAPI.adjr2(gm)
    end

    @testset "The default variant" begin
        # `:devianceratio` for both maximisation criteria, and no method at all for a
        # minimisation criterion, which reads no variant.
        for c in PO.MAX_VAL_STEPWISE_REGRESSION_CRITERIA
            @test PO.default_regression_criterion_variant(Val(c)) === :devianceratio
        end
        for c in PO.MIN_VAL_STEPWISE_REGRESSION_CRITERIA
            @test_throws MethodError PO.default_regression_criterion_variant(Val(c))
        end

        # The default keeps the score CONTINUOUS with the `LinearModel` path: on a
        # Normal-family model the deviance is the residual sum of squares, so the deviance
        # ratio is the classical `R^2`. No other member of the tuple is close.
        gm = GLM.glm(Xd, y, Distributions.Normal())
        @test StatsAPI.r2(gm, :devianceratio) ≈ StatsAPI.r2(lm)
        @test !isapprox(StatsAPI.r2(gm, :McFadden), StatsAPI.r2(lm))
        @test !isapprox(StatsAPI.r2(gm, :Nagelkerke), StatsAPI.r2(lm))
    end

    @testset "regression_criterion_func" begin
        lmt = PO.LinearModel()
        glmt = PO.GeneralisedLinearModel()
        # A `LinearModel` target takes the `StatsAPI` function itself, in every case.
        @test PO.regression_criterion_func(Val(:aic), lmt) === StatsAPI.aic
        @test PO.regression_criterion_func(Val(:aicc), lmt) === StatsAPI.aicc
        @test PO.regression_criterion_func(Val(:bic), lmt) === StatsAPI.bic
        @test PO.regression_criterion_func(Val(:r2), lmt) === StatsAPI.r2
        @test PO.regression_criterion_func(Val(:adjr2), lmt) === StatsAPI.adjr2

        # A `GeneralisedLinearModel` target takes the same three minimisation functions...
        @test PO.regression_criterion_func(Val(:aic), glmt) === StatsAPI.aic
        @test PO.regression_criterion_func(Val(:aicc), glmt) === StatsAPI.aicc
        @test PO.regression_criterion_func(Val(:bic), glmt) === StatsAPI.bic

        # ...and a CLOSURE for the two maximisation criteria, holding the variant.
        gm = GLM.glm(Xd, y, Distributions.Normal())
        @test PO.regression_criterion_func(Val(:r2), glmt)(gm) ≈
              StatsAPI.r2(gm, :devianceratio)
        @test PO.regression_criterion_func(Val(:adjr2), glmt)(gm) ≈
              StatsAPI.adjr2(gm, :devianceratio)

        # The target's own `variant` overrides the default of the criterion.
        glmv = PO.GeneralisedLinearModel(; variant = :McFadden)
        @test PO.regression_criterion_func(Val(:r2), glmv)(gm) ≈ StatsAPI.r2(gm, :McFadden)
        @test PO.regression_criterion_func(Val(:adjr2), glmv)(gm) ≈
              StatsAPI.adjr2(gm, :McFadden)
    end

    @testset "regression_polarity" begin
        #= The one place the direction of "better" is written. `regression_threshold`, the
        two `get_*_reg_incl*!` verbs and the two `_regression` methods that seed a score
        vector all read this pairing, so a criterion cannot be minimised in one of the five
        and maximised in another. =#
        for c in PO.MIN_VAL_STEPWISE_REGRESSION_CRITERIA
            p = PO.regression_polarity(Val(c))
            @test p.best === findmin
            @test p.improves === (<)
            @test p.worst === typemax
        end
        for c in PO.MAX_VAL_STEPWISE_REGRESSION_CRITERIA
            p = PO.regression_polarity(Val(c))
            @test p.best === findmax
            @test p.improves === (>)
            @test p.worst === typemin
        end
    end

    @testset "regression_threshold" begin
        # The worst score the criterion can take, so the first addition always improves on
        # it. `Inf` under minimisation, `-Inf` under maximisation.
        for c in PO.MIN_VAL_STEPWISE_REGRESSION_CRITERIA
            @test PO.regression_threshold(Val(c)) == Inf
        end
        for c in PO.MAX_VAL_STEPWISE_REGRESSION_CRITERIA
            @test PO.regression_threshold(Val(c)) == -Inf
        end
    end

    @testset "R^2 selects every factor and removes none" begin
        # `R^2` never falls when a factor is added, so forward selection admits all five and
        # backward elimination removes none. That is a property of the criterion, not a
        # defect. The criteria that pay for size keep fewer.
        kept = Dict{Tuple{Symbol, Symbol}, Vector{Int}}()
        for c in PO.STEPWISE_REGRESSION_CRITERIA,
            a in (PO.ForwardSelection(), PO.BackwardElimination())

            rr = PO.regression(PO.StepwiseRegression(; crit = c, alg = a), rd)
            kept[(c, nameof(typeof(a)))] = findall(!iszero, vec(rr.M))
        end
        @test kept[(:r2, :ForwardSelection)] == 1:N
        @test kept[(:r2, :BackwardElimination)] == 1:N
        # Every criterion that penalises size keeps a strict subset, in both directions.
        for c in (:aic, :aicc, :bic, :adjr2), a in
                                              (:ForwardSelection, :BackwardElimination)

            @test length(kept[(c, a)]) < N
        end
        # The two factors the response was actually built from survive every criterion.
        for kv in values(kept)
            @test 1 in kv
            @test 3 in kv
        end
    end

    @testset "An unset L" begin
        M = [1.0 2.0; 3.0 4.0]
        unset = PO.Regression(; M = M)
        set = PO.Regression(; M = M, L = [9.0 8.0 7.0; 6.0 5.0 4.0])

        # `re.L` returns `re.M` when `L` was never given, so `isnothing(re.L)` is NEVER
        # true and the two cases are told apart by `getfield` alone.
        @test unset.L === unset.M
        @test !isnothing(unset.L)
        @test isnothing(getfield(unset, :L))
        @test !isnothing(getfield(set, :L))
        @test set.L == [9.0 8.0 7.0; 6.0 5.0 4.0]

        # `size(L, 2)` is the width of the basis risk is decomposed in: the original
        # factors when `L` is unset, and the reduced basis when it is set.
        @test size(unset.L, 2) == size(M, 2)
        @test size(set.L, 2) == 3

        # `Accessors.@set` rebuilds the result from its fields, so an unset `L` stays unset
        # and is never written as a copy of `M` (#1497).
        moved = Accessors.@set unset.b = [1.0, 2.0]
        @test isnothing(getfield(moved, :L))
        @test moved.b == [1.0, 2.0]
        @test moved.M === unset.M
        # The keyword constructor runs, so its guards refuse a bad patch.
        @test_throws DimensionMismatch Accessors.@set unset.b = [1.0]
        # A `CrossSectionalFactorModel` keeps an unset `L` the same way.
        csfm = PO.CrossSectionalFactorModel(; M = M, b = [0.0, 0.0])
        @test isnothing(getfield(Accessors.@set(csfm.esigma = [0.1, 0.2]), :L))
    end

    @testset "Regression validation" begin
        M = [1.0 2.0; 3.0 4.0]
        @test_throws PO.IsEmptyError PO.Regression(; M = Matrix{Float64}(undef, 0, 0))
        @test_throws PO.IsEmptyError PO.Regression(; M = M, b = Float64[])
        @test_throws DimensionMismatch PO.Regression(; M = M, b = [1.0])
        @test_throws DimensionMismatch PO.Regression(; M = M, L = [1.0 2.0])
        # An empty `L` is REJECTED, exactly as an empty `M` and an empty `b` are. The three
        # guards are one rule, and the `## Validation` block states all three.
        @test_throws PO.IsEmptyError PO.Regression(; M = M,
                                                   L = Matrix{Float64}(undef, 2, 0))
        @test isa(PO.Regression(; M = M, L = [1.0 2.0; 3.0 4.0], b = [1.0, 2.0]),
                  PO.Regression)
    end

    @testset "port_opt_view keeps an unset L unset" begin
        M = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        unset = PO.Regression(; M = M, b = [7.0, 8.0, 9.0])
        set = PO.Regression(; M = M, L = 10 .* M, b = [7.0, 8.0, 9.0])

        vu = PO.port_opt_view(unset, [1, 3])
        # The view must NOT materialise `L` as a copy of `M`: that would silently lose the
        # unset-ness the `swap` rule exists to express.
        @test isnothing(getfield(vu, :L))
        @test vu.M == M[[1, 3], :]
        @test vu.b == [7.0, 9.0]

        vs = PO.port_opt_view(set, [1, 3])
        @test getfield(vs, :L) == (10 .* M)[[1, 3], :]

        # No `b` and no `L` at all.
        vn = PO.port_opt_view(PO.Regression(; M = M), [2])
        @test isnothing(getfield(vn, :L))
        @test isnothing(getfield(vn, :b))
    end

    @testset "The regression pass-through" begin
        re = PO.Regression(; M = [1.0 2.0; 3.0 4.0])
        # A result passes through unchanged, on both arities generic code uses.
        @test PO.regression(re) === re
        @test PO.regression(re, rand(3, 2), rand(3, 2)) === re
        @test PO.regression(re, rd) === re

        # An estimator needs both matrices of the `ReturnsResult`.
        est = PO.StepwiseRegression()
        @test_throws PO.IsNothingError PO.regression(est,
                                                     PO.ReturnsResult(; nx = ["y"],
                                                                      X = reshape(y, :, 1)))
        @test_throws PO.IsNothingError PO.regression(est, PO.ReturnsResult())
        @test isa(PO.regression(est, rd), PO.Regression)
    end

    @testset "regression(re, rd) reads the source of re (#1699, #1705)" begin
        # A regression reader that receives the returns data states its own source.
        bm = 0.05 .* F[:, 2]
        rdb = PO.ReturnsResult(; nx = ["y"], X = reshape(y, :, 1), nf = rd.nf, F = F,
                               nb = ["b"], B = bm)
        excess = PO.ReturnsResult(; nx = ["y"], X = reshape(y .- bm, :, 1), nf = rd.nf,
                                  F = F)
        for x_src in (nothing, PO.DataSource())
            r = PO.regression(PO.StepwiseRegression(; x_src = x_src), rdb)
            @test r.M == PO.regression(PO.StepwiseRegression(), rd).M
        end
        r = PO.regression(PO.StepwiseRegression(; x_src = PO.ExcessSource()), rdb)
        @test r.M == PO.regression(PO.StepwiseRegression(), excess).M
        @test r.M != PO.regression(PO.StepwiseRegression(), rd).M
        # The call gives no Prior Result, and the excess source needs a benchmark.
        @test_throws ArgumentError PO.regression(PO.StepwiseRegression(;
                                                                       x_src = PO.PriorSource()),
                                                 rdb)
        @test_throws ArgumentError PO.regression(PO.StepwiseRegression(;
                                                                       x_src = PO.ExcessSource()),
                                                 rd)
    end

    @testset "GeneralisedLinearModel validation" begin
        # The constructor checks `variant` against the WIDER of the two tuples.
        for v in PO.PSEUDO_R2_VARIANTS
            @test PO.GeneralisedLinearModel(; variant = v).variant === v
        end
        @test_throws ArgumentError PO.GeneralisedLinearModel(; variant = :nonesuch)
        @test isnothing(PO.GeneralisedLinearModel().variant)
        # The default `args` is the Normal family, which reproduces ordinary least squares.
        @test isa(only(PO.GeneralisedLinearModel().args), Distributions.Normal)
    end

    @testset "factory adds the observation weights" begin
        w = pweights(range(0.5, 1.5; length = T))

        lmt = PO.factory(PO.LinearModel(; kwargs = (; dropcollinear = false)), w)
        @test lmt.kwargs.weights === w
        @test lmt.kwargs.dropcollinear === false

        glmt = PO.factory(PO.GeneralisedLinearModel(; variant = :McFadden), w)
        @test glmt.kwargs.weights === w
        # `factory` carries the other two fields across unchanged.
        @test glmt.variant === :McFadden
        @test isa(only(glmt.args), Distributions.Normal)
    end

    @testset "fit resolves a DynamicAbstractWeights" begin
        dw = RegressionDynWeights()
        resolved = PO.get_observation_weights(dw, Xd)

        # A dynamic weight type is resolved against `X` BEFORE it reaches GLM, so the fit
        # equals the one a caller gets by resolving the weights by hand.
        m_dyn = StatsAPI.fit(PO.LinearModel(; kwargs = (; weights = dw)), Xd, y)
        m_res = StatsAPI.fit(PO.LinearModel(; kwargs = (; weights = resolved)), Xd, y)
        @test GLM.coef(m_dyn) ≈ GLM.coef(m_res)
        # ...and differs from the unweighted fit, so the weights were really used.
        @test !isapprox(GLM.coef(m_dyn), GLM.coef(StatsAPI.fit(PO.LinearModel(), Xd, y)))

        g_dyn = StatsAPI.fit(PO.GeneralisedLinearModel(; kwargs = (; weights = dw)), Xd, y)
        g_res = StatsAPI.fit(PO.GeneralisedLinearModel(; kwargs = (; weights = resolved)),
                             Xd, y)
        @test GLM.coef(g_dyn) ≈ GLM.coef(g_res)

        # A plain weights vector of another kind over the same values gives the same
        # coefficients: the scale `glm_weights` applies moves no coefficient.
        pw = pweights(range(0.5, 1.5; length = T))
        m_pw = StatsAPI.fit(PO.LinearModel(; kwargs = (; weights = pw)), Xd, y)
        @test GLM.coef(m_pw) ≈ GLM.coef(m_dyn)
    end

    @testset "A weighted fit reads the Kish count of its weights (#1640)" begin
        kish(w) = sum(w)^2 / sum(abs2, w)
        # A count passes unchanged, and every other kind sums to its Kish count.
        fw = fweights(fill(2.0, T))
        @test PO.glm_weights(fw) === fw
        for w in (pweights(range(0.5, 1.5; length = T)), aweights(fill(1 / T, T)),
                  eweights(1:T, 0.05), pweights([fill(1 // 20, 20); zeros(Int, T - 20)]))
            gw = PO.glm_weights(w)
            @test isa(gw, StatsBase.FrequencyWeights)
            @test sum(gw) ≈ kish(w)
            @test gw ./ sum(gw) ≈ w ./ sum(w)
        end
        # A Rational stays Rational: the scale is read off the weights.
        @test eltype(PO.glm_weights(pweights([1 // 2, 1 // 4, 1 // 4]))) == Rational{Int}
        # Only an entry of ObsWeights is touched.
        kw = (; dropcollinear = false)
        @test PO.glm_fit_kwargs(kw, Xd) === kw
        @test PO.glm_fit_kwargs((; weights = fw), Xd).weights === fw

        # Probability weights that sum to one read T observations, so the inference of the fit
        # is that of the unweighted one. Before #1640 GLM read one observation, a negative
        # residual degree of freedom and a p-value of one for every coefficient.
        u = pweights(fill(1 / T, T))
        for tgt in (PO.LinearModel(), PO.GeneralisedLinearModel())
            m0 = StatsAPI.fit(tgt, Xd, y)
            m1 = StatsAPI.fit(PO.factory(tgt, u), Xd, y)
            @test GLM.nobs(m1) ≈ T
            @test GLM.dof_residual(m1) ≈ GLM.dof_residual(m0)
            @test GLM.stderror(m1) ≈ GLM.stderror(m0)
            @test coeftable(m1).cols[4] ≈ coeftable(m0).cols[4]
            @test GLM.aic(m1) ≈ GLM.aic(m0)
            @test GLM.bic(m1) ≈ GLM.bic(m0)
        end
        # Equal weights on 20 rows and none on the rest read 20 observations, so the fit is the
        # fit of those rows alone.
        s = pweights([fill(1 / 20, 20); zeros(T - 20)])
        ms = StatsAPI.fit(PO.factory(PO.LinearModel(), s), Xd, y)
        mr = StatsAPI.fit(PO.LinearModel(), Xd[1:20, :], y[1:20])
        @test GLM.nobs(ms) ≈ 20
        @test GLM.stderror(ms) ≈ GLM.stderror(mr)
        @test coeftable(ms).cols[4] ≈ coeftable(mr).cols[4]
    end

    @testset "The criterion tuples and their unions" begin
        @test PO.MIN_VAL_STEPWISE_REGRESSION_CRITERIA == (:aic, :aicc, :bic)
        @test PO.MAX_VAL_STEPWISE_REGRESSION_CRITERIA == (:r2, :adjr2)
        @test PO.STEPWISE_REGRESSION_CRITERIA ==
              (PO.MIN_VAL_STEPWISE_REGRESSION_CRITERIA...,
               PO.MAX_VAL_STEPWISE_REGRESSION_CRITERIA...)

        # Each union is built from its own tuple, and the third is their join.
        for c in PO.MIN_VAL_STEPWISE_REGRESSION_CRITERIA
            @test isa(Val(c), PO.MinValStepwiseRegressionCriterion)
            @test !isa(Val(c), PO.MaxValStepwiseRegressionCriterion)
        end
        for c in PO.MAX_VAL_STEPWISE_REGRESSION_CRITERIA
            @test isa(Val(c), PO.MaxValStepwiseRegressionCriterion)
            @test !isa(Val(c), PO.MinValStepwiseRegressionCriterion)
        end
        for c in PO.STEPWISE_REGRESSION_CRITERIA
            @test isa(Val(c), PO.MinMaxValStepwiseRegressionCriterion)
        end
        # `PValue` reads the coefficient p-values, not one score, so it is NOT a member.
        @test !isa(PO.PValue(), PO.MinMaxValStepwiseRegressionCriterion)
    end

    @testset "RegE_Reg matches a result and an estimator" begin
        @test isa(PO.Regression(; M = [1.0 2.0]), PO.RegE_Reg)
        @test isa(PO.StepwiseRegression(), PO.RegE_Reg)
        @test isa(PO.DimensionReductionRegression(), PO.RegE_Reg)
        @test !isa(PO.LinearModel(), PO.RegE_Reg)
    end

    # Issue #776. `esigma` is the fourth field, and it carries the same name, the same two
    # shapes and the same guards as the field `CrossSectionalFactorModel` declares, so one
    # reader answers off either block.
    @testset "esigma takes the two shapes and is checked against the asset count" begin
        M = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        @test PO.Regression(; M = M, esigma = [0.1, 0.2, 0.3]).esigma == [0.1, 0.2, 0.3]
        esm = Matrix(0.1 * I(3))
        @test PO.Regression(; M = M, esigma = esm).esigma == esm
        # A regression estimator fits loadings alone, so the default is unset.
        @test isnothing(PO.Regression(; M = M).esigma)
        # The guards are `assert_idiosyncratic_covariance`'s, run against `size(M, 1)`.
        @test_throws PO.IsEmptyError PO.Regression(; M = M, esigma = Float64[])
        @test_throws DimensionMismatch PO.Regression(; M = M, esigma = [0.1, 0.2])
        @test_throws DimensionMismatch PO.Regression(; M = M, esigma = Matrix(0.1 * I(2)))
        # A matrix must be square before its size is read.
        @test_throws DimensionMismatch PO.Regression(; M = M, esigma = zeros(3, 2))
    end

    @testset "port_opt_view cuts esigma on every axis it has" begin
        M = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        i = [1, 3]
        # A vector of variances is one axis, so it is indexed once.
        rev = PO.Regression(; M = M, b = [0.1, 0.2, 0.3], esigma = [1.0, 2.0, 3.0])
        @test PO.port_opt_view(rev, i).esigma == [1.0, 3.0]
        # A full covariance is two axes, so the selected block stays square.
        esm = [1.0 0.4 0.5; 0.4 2.0 0.6; 0.5 0.6 3.0]
        rem = PO.Regression(; M = M, esigma = esm)
        @test PO.port_opt_view(rem, i).esigma == esm[i, i]
        # An unset field stays unset through the view.
        @test isnothing(PO.port_opt_view(PO.Regression(; M = M), i).esigma)
    end

    @testset "set_idiosyncratic_covariance rewrites three fields and keeps an unset L unset" begin
        M = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        re = PO.Regression(; M = M, b = [0.1, 0.2, 0.3])
        @test isnothing(getfield(re, :L))
        re2 = PO.set_idiosyncratic_covariance(re, [1.0, 2.0, 3.0], [7, 8, 9], [9, 9, 9])
        @test re2.esigma == [1.0, 2.0, 3.0]
        @test re2.edof == [7, 8, 9]
        @test re2.ediv == [9, 9, 9]
        # The `swap(L, M)` rule makes `re.L` answer `M`, so a property-based rewrite would
        # materialise `L` as a copy of `M`. This one reads `getfield`, so the field stays
        # unset and `M` and `b` come through untouched.
        @test isnothing(getfield(re2, :L))
        @test re2.M == M
        @test re2.b == [0.1, 0.2, 0.3]
        # A set `L` survives, and `nothing` clears the field.
        reL = PO.Regression(; M = M, L = 2 * M)
        @test PO.set_idiosyncratic_covariance(reL, [1.0, 2.0, 3.0], nothing, nothing).L ==
              2 * M
        re3 = PO.set_idiosyncratic_covariance(re2, nothing, nothing, nothing)
        @test isnothing(re3.esigma)
        @test isnothing(re3.edof)
        @test isnothing(re3.ediv)
        # The guards of the constructor run again.
        @test_throws DimensionMismatch PO.set_idiosyncratic_covariance(re, [1.0, 2.0],
                                                                       nothing, nothing)
        @test_throws DimensionMismatch PO.set_idiosyncratic_covariance(re, [1.0, 2.0, 3.0],
                                                                       [1, 2], nothing)
    end
    @testset "Statements the /unslop pass corrected (#463)" begin
        # `orthogonality_weights` refuses a variance that is not finite, and not only a
        # variance that is not positive.
        M = [1.0 2.0; 3.0 4.0]
        metric = PO.InverseIdiosyncraticVarianceMetric()
        @test PO.orthogonality_weights(metric,
                                       PO.Regression(; M = M, esigma = [0.5, 0.25])) ==
              [2.0, 4.0]
        @test_throws DomainError PO.orthogonality_weights(metric,
                                                          PO.Regression(; M = M,
                                                                        esigma = [0.5, Inf]))
        @test_throws DomainError PO.orthogonality_weights(metric,
                                                          PO.Regression(; M = M,
                                                                        esigma = [0.5, 0.0]))
        # `LinearModel()` is the default of each estimator that takes a regression target,
        # and the cross-sectional linear regression takes none.
        @test isa(PO.StepwiseRegression().tgt, PO.LinearModel)
        @test isa(PO.DimensionReductionRegression().retgt, PO.LinearModel)
        @test isa(PO.CrossSectionalTargetRegression().tgt, PO.LinearModel)
        @test isa(PO.ImpliedVolatilityRegression().re, PO.LinearModel)
        @test !hasfield(typeof(PO.CrossSectionalLinearRegression()), :tgt)
        # `set_row_key` has two callers beside `FactorPrior`, and both keep an unset `L` unset.
        rr = PO.Regression(; M = [1.0 2.0; 3.0 4.0; 5.0 6.0], idx = [3, 5, 7],
                           ts = PO.Dates.Date(2020, 1, 1) .+ PO.Dates.Day.(0:2))
        rf = PO.forget_row_positions(rr)
        @test isnothing(rf.idx)
        @test rf.ts == rr.ts
        @test isnothing(getfield(rf, :L))
        rm = PO.map_row_positions(rr, [10, 11, 12, 13, 14, 15, 16])
        @test rm.idx == [12, 14, 16]
        @test isnothing(getfield(rm, :L))
    end
end

#=
Check `src/05_Moments/21_TimeSeriesRegression/01_StepwiseRegression.jl` and
`src/05_Moments/21_TimeSeriesRegression/02_DimensionReductionRegression.jl` against the mathematics and the procedures
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
   `regression` warns on the empty selection and names the asset (issue #503), so the empty
   outcome is as loud as the `PValue` path's `add_best_factor_after_pval_failure!`, and the
   testset pins the warning and pins that a non-empty search stays silent.

3. `included` IS IN INSERTION ORDER, NOT SORTED. Forward selection pushes each winner onto the
   end, so `:r2` on the sample below returns `[1, 3, 4, 5, 2]`. `regression` then writes the
   fitted coefficients into the columns that order names, and `Insertion order` checks the
   whole row against a plain least-squares fit rather than trusting the permutation.

4. THE STANDARDISATION AND THE RECOVERY MUST READ THE SAME STATISTICS. That was issue #398.
   `The round trip` builds the matched pair and the mismatched pair from the same sample and
   compares the two predictions; matched agrees to 4.4e-16 and mismatched parts by 2.1e-3.

5. `PPCA` CANNOT RETAIN EVERY COMPONENT. `MultivariateStats` caps a latent-variable model at
   one less than the input dimension. Its own fit at the full width SUCCEEDS and
   `MultivariateStats.projection` then raises a LAPACK `ArgumentError`, so the failure once
   surfaced inside `prep_dim_red_reg` with no mention of `maxoutdim`. That was issue #502.
   `StatsAPI.fit(::PPCA, ::MatNum)` now checks the width against `size(X, 1)` and raises a
   `DomainError` first. `The retained width` pins the cap, the guard, and the raw raise the
   guard stands in front of.
=#

const PO = PortfolioOptimisers
const StatsAPI = PortfolioOptimisers.StatsAPI
const MVS = PortfolioOptimisers.MultivariateStats

#=
A caller's own regression target under the time-series estimators (#1441). `CallerTestTarget`
fits through `GLM.LinearModel`, as the library's `LinearModel` does, but it is a separate type
with no `kwargs` field. Every verb an estimator reads therefore reaches it through the methods
the `# Interfaces` section of `AbstractRegressionTarget` names, never through a field that only
the library's targets carry, and its results must equal the library's.
=#
struct CallerTestTarget{W} <: PortfolioOptimisers.AbstractUnpenalisedRegressionTarget
    w::W
end
CallerTestTarget() = CallerTestTarget(nothing)
function PortfolioOptimisers.factory(::CallerTestTarget, w::StatsBase.AbstractWeights)
    return CallerTestTarget(w)
end
function PortfolioOptimisers.StatsAPI.fit(tgt::CallerTestTarget, A::AbstractMatrix,
                                          y::AbstractVector)
    if isnothing(tgt.w)
        return StatsAPI.fit(PO.GLM.LinearModel, A, y)
    end
    # GLM reads every weight vector as a count, so the caller states the count the library
    # states, the Kish count of its weights (ADR 0203), and its fit equals the library's.
    return StatsAPI.fit(PO.GLM.LinearModel, A, y; weights = PO.glm_weights(tgt.w))
end
function PortfolioOptimisers.regression_target_weights(tgt::CallerTestTarget)
    return tgt.w
end
# A target that states its fit and no other method.
struct BareTestTarget <: PortfolioOptimisers.AbstractUnpenalisedRegressionTarget end
function PortfolioOptimisers.StatsAPI.fit(::BareTestTarget, A::AbstractMatrix,
                                          y::AbstractVector)
    return StatsAPI.fit(PO.GLM.LinearModel, A, y)
end

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
        # `regression` warns and names the asset, so the empty outcome is not silent (#503).
        reg = @test_logs (:warn, r"Asset 1: the stepwise search selected no factor") regression(StepwiseRegression(;
                                                                                                                   crit = :aic,
                                                                                                                   alg = BackwardElimination()),
                                                                                                reshape(yn,
                                                                                                        :,
                                                                                                        1),
                                                                                                Fn)
        @test isapprox(reg.b[1], mean(yn))
        @test all(iszero, reg.M)

        # A search that selects a factor logs nothing. `:r2` never falls when a factor is
        # added, so forward selection keeps all four.
        reg = @test_logs regression(StepwiseRegression(; crit = :r2,
                                                       alg = ForwardSelection()),
                                    reshape(yn, :, 1), Fn)
        @test !all(iszero, reg.M)

        # One warning per empty asset, and only for the assets that emptied.
        Xn = [yn Fn[:, 1]]
        reg = @test_logs (:warn, r"Asset 1: ") match_mode = :any regression(StepwiseRegression(;
                                                                                               crit = :aic,
                                                                                               alg = BackwardElimination()),
                                                                            Xn, Fn)
        @test all(iszero, view(reg.M, 1, :))
        @test !all(iszero, view(reg.M, 2, :))

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

        # The cap is hard, and the third-party fit does not enforce it: it accepts the full
        # width, returns a model whose weights are NaN, and `projection` then raises a LAPACK
        # `ArgumentError` that names neither the cause nor the keyword. Issue #502.
        raw = MVS.fit(MVS.PPCA, Fs; maxoutdim = 5)
        @test all(isnan, raw.W)
        @test_throws ArgumentError MVS.projection(raw)
        # `prep_dim_red_reg` calls `predict` first, and the `PPCA` docstring names it.
        @test_throws ArgumentError StatsAPI.predict(raw, Fs)

        # The library checks the width first, so the caller meets a `DomainError` that names
        # `maxoutdim` and the factor count. It is raised at the fit, before `MultivariateStats`
        # is reached, so every caller of that fit inherits it.
        drtgt5 = PPCA(; kwargs = (; maxoutdim = 5))
        @test_throws DomainError StatsAPI.fit(drtgt5, Fs)
        @test_throws DomainError PO.prep_dim_red_reg(DimensionReductionRegression(;
                                                                                  drtgt = drtgt5),
                                                     Fn)
        @test_throws DomainError regression(DimensionReductionRegression(; drtgt = drtgt5),
                                            randn(StableRNG(100), 300, 2), Fn)

        # The message names the keyword, its value, and the bound it broke.
        msg = try
            StatsAPI.fit(drtgt5, Fs)
        catch e
            sprint(showerror, e)
        end
        @test occursin("kwargs.maxoutdim => 5", msg)
        @test occursin("size(X, 1) => 5", msg)

        # The guard holds the lower end too, where `MultivariateStats` raises a bare
        # `ErrorException` whose message names no argument value.
        @test_throws DomainError StatsAPI.fit(PPCA(; kwargs = (; maxoutdim = 0)), Fs)

        # A `PPCA` that sets no width at all takes no guard, because it has no `maxoutdim`
        # entry to check. The widths inside the cap are pinned above.
        @test size(MVS.projection(StatsAPI.fit(PPCA(), Fs)), 2) == 4

        # `PCA` retains every component, so it carries no such cap and no such guard.
        @test size(MVS.projection(StatsAPI.fit(PCA(; kwargs = (; maxoutdim = 5)), Fs)),
                   2) == 5
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
        # A probabilistic PCA predicts its posterior mean, `C⁻¹ Wᵀ x`, not the projection on
        # the left singular vectors of `W`. The recovery mapped back through those vectors,
        # and on this sample its prediction missed the reduced-space fit by 4.5 (#1472).
        @test roundtrip(DimensionReductionRegression(; drtgt = PPCA())) < 1e-14

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
        # coefficients the per-asset fits produced, up to the sign of each component (#1756).
        x1, Vp, _, _ = PO.prep_dim_red_reg(DimensionReductionRegression(), F)
        betapc = hcat([StatsAPI.coef(StatsAPI.fit(LinearModel(), x1, X[:, i]))[2:end]
                       for i in axes(X, 2)]...)
        s = map(c -> sum(c) < 0 ? -1 : 1, eachrow(betapc))
        @test isapprox(reg.L, transpose(betapc) .* transpose(s); atol = 1e-14)
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

    @testset "The sign of each component of a dimension reduction fit (#1756)" begin
        # `--check-bounds=yes` flipped the sign of both principal components of a fit, so a
        # factor risk budget, whose factor weights must be positive, found no long-only
        # portfolio. A flip of the components of `proj` is the same flip, and it must not
        # move `L`.
        _, Vp, _, sigma = PO.prep_dim_red_reg(DimensionReductionRegression(), F)
        P = Vp ./ sigma
        reg = regression(DimensionReductionRegression(; proj = P), X, F)
        @test all(c -> sum(c) >= 0, eachcol(reg.L))
        for s in ([-1, 1, 1, 1, 1], [1, -1, 1, -1, 1], -ones(Int, N))
            regs = regression(DimensionReductionRegression(; proj = P .* transpose(s)), X,
                              F)
            @test isapprox(regs.L, reg.L; atol = 1e-14)
            @test isapprox(regs.M, reg.M; atol = 1e-14)
        end
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

@testset "A caller's own regression target under the time-series estimators" begin
    using Test, PortfolioOptimisers, StableRNGs, StatsBase, LinearAlgebra, Statistics

    rng = StableRNG(1441)
    T, N = 120, 4
    F = randn(rng, T, N)
    X = hcat(0.7 .* F[:, 1] .- 0.4 .* F[:, 2] .+ 0.2 .* randn(rng, T),
             0.5 .* F[:, 3] .+ 0.2 .* randn(rng, T))
    w = pweights(range(0.5, 1.5; length = T))
    same(r1, r2) = r1.b == r2.b && r1.M == r2.M

    @testset "The library targets state their weights" begin
        for tgt in (LinearModel(), GeneralisedLinearModel())
            @test isnothing(PO.regression_target_weights(tgt))
            @test PO.regression_target_weights(factory(tgt, w)) === w
        end
    end

    @testset "StepwiseRegression" begin
        # Before #1441 the constructor read `tgt.kwargs`, a field only the library's targets
        # carry, and threw a `FieldError` on a caller's target.
        @test isa(StepwiseRegression(; tgt = BareTestTarget()), StepwiseRegression)

        # `PValue` reads `StatsAPI.coeftable`, and each other criterion reads its score verb.
        # `:r2` and `:adjr2` had no method for a caller's target before #1441.
        for crit in (PValue(), :aic, :aicc, :bic, :r2, :adjr2),
            alg in (ForwardSelection(), BackwardElimination())

            lib = StepwiseRegression(; crit = crit, alg = alg)
            for tgt in (CallerTestTarget(), BareTestTarget())
                @test same(regression(StepwiseRegression(; crit = crit, alg = alg,
                                                         tgt = tgt), X, F),
                           regression(lib, X, F))
            end
            # The weights reach the caller's fit through its own `factory` method.
            @test same(regression(factory(StepwiseRegression(; crit = crit, alg = alg,
                                                             tgt = CallerTestTarget()), w),
                                  X, F), regression(factory(lib, w), X, F))
        end
    end

    @testset "DimensionReductionRegression" begin
        lib = DimensionReductionRegression()
        re = DimensionReductionRegression(; retgt = CallerTestTarget())
        @test same(regression(re, X, F), regression(lib, X, F))
        # The weighted mean of the response reads the caller's weights, so the weighted
        # intercepts equal the library's too.
        @test same(regression(factory(re, w), X, F), regression(factory(lib, w), X, F))

        # A target that does not state its weights is refused at construction, because the
        # intercept would ignore the weights `factory` stored in it.
        @test_throws ArgumentError DimensionReductionRegression(; retgt = BareTestTarget())
        msg = sprint(showerror,
                     try
                         DimensionReductionRegression(; retgt = BareTestTarget())
                     catch e
                         e
                     end)
        @test occursin("regression_target_weights(::BareTestTarget)", msg)

        # The weights a caller's target states are checked like the library's.
        @test_throws ArgumentError DimensionReductionRegression(;
                                                                retgt = CallerTestTarget(1.0))
        @test_throws PO.IsEmptyError DimensionReductionRegression(;
                                                                  retgt = CallerTestTarget(pweights(Float64[])))
    end
end

#=
Check `src/05_Moments/32_CrossSectionalFactorModel/01_CrossSectionalRegression.jl` against the mathematics its docstrings
state, and against the stored oracle of map #643. Issue #679.

THREE FACTS SHAPE THE PROBES.

1. THE WEIGHT IS THE ONE CONTRACT. A pair with a zero cross-sectional weight leaves the fit
   and may carry a missing return or a missing exposure. A pair with a positive weight may
   not. `The eligibility contract` drives both directions, because a mask read off the data
   rather than off the weights would silently admit a pair the caller meant to exclude.

2. THE FOUR SOLVE MEMBERS AGREE ON A FULL-RANK DESIGN AND SPLIT ON A SQUARE SINGULAR ONE.
   Julia's `\` answers a deficient design two ways, and MEASUREMENT, not the shape of the
   API, says which. A NON-SQUARE design reaches a column-pivoted `QR` whose solve completes
   the orthogonal factorisation, so it returns the MINIMUM-NORM answer and agrees with
   `pinv`; issue #648's ground truth 3 called that a basic solution, and it is not one. A
   SQUARE singular design reaches an `LU` factorisation instead and throws
   `SingularException`. That square case is where `UncheckedSolve` really parts from the
   other three, and `A rank-deficient design` pins both halves.

3. THE TWO MEMBERS SOLVE ONE PROBLEM TWO WAYS. `CrossSectionalLinearRegression` runs the
   closed form on the weighted design, and `CrossSectionalTargetRegression` hands the
   unweighted pair to an external target with the weights as observation weights. They must
   land on the same coefficients, and `The two members agree` computes both. They differ in
   one place only: an observation with no eligible asset, which the closed form answers with
   zero factor returns and the target member refuses.

The reference fit at the top of the file is written out per observation with `pinv`, so no
testset compares the file against itself.
=#

# The weighted least squares of every observation, written out one observation at a time.
# `pinv` is the minimum-norm solution, which is what `PseudoInverseFallback` returns on a
# deficient design and what the closed form returns on a full-rank one.
function reference_cross_sectional_fit(Z, X, W; intercept = false)
    T, N, K = size(Z)
    f = zeros(T, K)
    b = zeros(T)
    n = zeros(Int, T)
    for t in 1:T
        idx = findall(W[t, :] .> 0)
        n[t] = length(idx)
        if isempty(idx)
            continue
        end
        w = W[t, idx]
        A = Z[t, idx, :]
        y = X[t, idx]
        ybar = 0.0
        xbar = zeros(K)
        if intercept
            sw = sum(w)
            ybar = dot(w, y) / sw
            xbar = (A' * w) ./ sw
            y = y .- ybar
            A = A .- xbar'
        end
        sq = sqrt.(w)
        coef = pinv(A .* sq) * (y .* sq)
        f[t, :] = coef
        b[t] = ybar - dot(coef, xbar)
    end
    return f, b, n
end

# A panel whose returns are an exact factor model plus a small noise, with a blanked pair, a
# zero-weight pair and a missing return, so the eligibility contract is exercised throughout.
function cross_sectional_panel(; T = 6, N = 25, K = 4, seed = 987)
    rng = StableRNG(seed)
    Z = randn(rng, T, N, K)
    beta = randn(rng, T, K)
    X = permutedims(reduce(hcat, Z[t, :, :] * beta[t, :] for t in 1:T)) .+
        0.02 * randn(rng, T, N)
    W = 1.0 .+ rand(rng, T, N)
    W[1, 1:3] .= 0.0
    Z[1, 1, 1] = NaN
    X[1, 2] = NaN
    return Z, X, W, beta
end

# A regression target the library does not know, for `cross_sectional_least_squares`.
struct LeverageTestTarget <: PortfolioOptimisers.AbstractUnpenalisedRegressionTarget end

# A caller's own weighted least-squares target, which states no basis invariance, so the fit
# hands it the design itself (#1719). Its solve is a pseudo-inverse.
struct PinvTestTarget{W} <: PortfolioOptimisers.AbstractUnpenalisedRegressionTarget
    w::W
end
struct PinvTestFit{C}
    c::C
end
function PortfolioOptimisers.factory(::PinvTestTarget, w::PortfolioOptimisers.ObsWeights)
    return PinvTestTarget(w)
end
function PortfolioOptimisers.StatsAPI.fit(tgt::PinvTestTarget, X::AbstractMatrix,
                                          y::AbstractVector)
    sq = sqrt.(tgt.w)
    return PinvTestFit(LinearAlgebra.pinv(X .* sq) * (y .* sq))
end
PortfolioOptimisers.StatsAPI.coef(m::PinvTestFit) = m.c

# A cross-sectional estimator the library does not know, which answers the public verb alone.
struct BlockTestRegression <: PortfolioOptimisers.AbstractCrossSectionalRegressionEstimator end
function PortfolioOptimisers.cross_sectional_regression(::BlockTestRegression, Z, X, W)
    return cross_sectional_regression(CrossSectionalLinearRegression(), Z, X, W)
end
# A cross-sectional estimator the library does not know, which answers no verb.
struct BareTestRegression <: PortfolioOptimisers.AbstractCrossSectionalRegressionEstimator end

@testset "Cross-sectional regression" begin
    Z, X, W, beta = cross_sectional_panel()
    # Rows 2 onwards carry no missing entry, so a probe that needs a clean panel takes them.
    Zc, Xc, Wc = Z[2:end, :, :], X[2:end, :], W[2:end, :]

    @testset "The closed form recovers the factor returns" begin
        csr = cross_sectional_regression(CrossSectionalLinearRegression(), Z, X, W)
        f, _, n = reference_cross_sectional_fit(Z, X, W)
        @test csr.f ≈ f
        @test csr.n == n
        @test isnothing(csr.b)
        # The noise is 2% of a unit-variance factor, so the fitted returns sit near the ones
        # the panel was built from.
        @test csr.f≈beta rtol=1e-2
        # The first observation lost exactly the three assets its zero weights name. The
        # missing exposure and the missing return sit on two of those three, so they cost
        # the fit nothing beyond the weights that already excluded them.
        @test csr.n[1] == size(X, 2) - 3
        @test all(==(size(X, 2)), csr.n[2:end])
    end

    @testset "The intercept is recovered from the weighted centroid" begin
        cre = CrossSectionalLinearRegression(; intercept = true)
        csr = cross_sectional_regression(cre, Z, X, W)
        f, b, _ = reference_cross_sectional_fit(Z, X, W; intercept = true)
        @test csr.f ≈ f
        @test csr.b ≈ b
        # A design that already carries a constant column makes the intercept redundant, so
        # the fitted values match the ones the no-intercept fit produces on that design.
        Zi = cat(Zc, ones(size(Zc, 1), size(Zc, 2), 1); dims = 3)
        with_int = cross_sectional_regression(cre, Zi, Xc, Wc)
        no_int = cross_sectional_regression(CrossSectionalLinearRegression(), Zi, Xc, Wc)
        @test predict(with_int, Zi) ≈ predict(no_int, Zi)
    end

    @testset "The residuals and the systematic part partition the returns" begin
        csr = cross_sectional_regression(CrossSectionalLinearRegression(), Z, X, W)
        Xh = predict(csr, Z)
        @test isequal(csr.eps, X - Xh)
        # A missing return leaves a missing residual, which is the honest answer for a pair
        # the fit never saw.
        @test isnan(csr.eps[1, 2])
        @test isnan(csr.eps[1, 1])
        @test all(isfinite, csr.eps[2:end, :])
        # An intercept shifts every fitted value of its observation by the same amount.
        ci = cross_sectional_regression(CrossSectionalLinearRegression(; intercept = true),
                                        Zc, Xc, Wc)
        shift = predict(ci, Zc) -
                PortfolioOptimisers.cross_sectional_systematic(ci.f, nothing, Zc)
        for t in axes(shift, 1)
            @test all(x -> isapprox(x, ci.b[t]), shift[t, :])
        end
    end

    @testset "The eligibility contract" begin
        # A positive weight on a pair with a missing exposure is refused, and so is one on a
        # pair with a missing return.
        Wz = copy(W)
        Wz[1, 1] = 1.0
        @test_throws PortfolioOptimisers.IsNonFiniteError cross_sectional_regression(CrossSectionalLinearRegression(),
                                                                                     Z, X,
                                                                                     Wz)
        Wy = copy(W)
        Wy[1, 2] = 1.0
        @test_throws PortfolioOptimisers.IsNonFiniteError cross_sectional_regression(CrossSectionalLinearRegression(),
                                                                                     Z, X,
                                                                                     Wy)
        # A zero weight admits both, and the pair leaves the fit rather than poisoning it.
        csr = cross_sectional_regression(CrossSectionalLinearRegression(), Z, X, W)
        @test all(isfinite, csr.f)
        # A negative weight and a non-finite weight are refused before the mask is built.
        Wn = copy(W)
        Wn[1, 1] = -1.0
        @test_throws DomainError cross_sectional_regression(CrossSectionalLinearRegression(),
                                                            Z, X, Wn)
        Wf = copy(W)
        Wf[1, 1] = NaN
        @test_throws PortfolioOptimisers.IsNonFiniteError cross_sectional_regression(CrossSectionalLinearRegression(),
                                                                                     Z, X,
                                                                                     Wf)
        # The three arrays must agree on the observation and the asset axes.
        @test_throws DimensionMismatch cross_sectional_regression(CrossSectionalLinearRegression(),
                                                                  Z, X, W[:, 1:(end - 1)])
        @test_throws DimensionMismatch cross_sectional_regression(CrossSectionalLinearRegression(),
                                                                  Z, X[:, 1:(end - 1)],
                                                                  W[:, 1:(end - 1)])
        # None of the three may be empty.
        @test_throws PortfolioOptimisers.IsEmptyError PortfolioOptimisers.cross_sectional_design_mask(zeros(0,
                                                                                                            0,
                                                                                                            0),
                                                                                                      X,
                                                                                                      W)
        @test_throws PortfolioOptimisers.IsEmptyError PortfolioOptimisers.cross_sectional_design_mask(Z,
                                                                                                      zeros(0,
                                                                                                            0),
                                                                                                      W)
        @test_throws PortfolioOptimisers.IsEmptyError PortfolioOptimisers.cross_sectional_design_mask(Z,
                                                                                                      X,
                                                                                                      zeros(0,
                                                                                                            0))
    end

    @testset "The four solve members agree on a full-rank design" begin
        base = cross_sectional_regression(CrossSectionalLinearRegression(), Zc, Xc, Wc)
        for alg in (PseudoInverseFallback(), RankDeficiencyRefusal(), UncheckedSolve(),
                    MinimumNormSolve())
            csr = cross_sectional_regression(CrossSectionalLinearRegression(; alg = alg),
                                             Zc, Xc, Wc)
            @test csr.f ≈ base.f
        end
    end

    @testset "A rank-deficient design" begin
        Zd = copy(Zc)
        # The second factor is twice the first, so the numerical rank falls by one and the
        # two coefficients are no longer separately identified.
        Zd[:, :, 2] = 2 .* Zd[:, :, 1]
        @test PortfolioOptimisers.cross_sectional_rank(Zd[1, :, :]) == size(Zd, 3) - 1
        @test PortfolioOptimisers.cross_sectional_rank(Zc[1, :, :]) == size(Zc, 3)
        # A design with no row has rank zero and reads no pivot.
        @test PortfolioOptimisers.cross_sectional_rank(zeros(0, 3)) == 0
        fallback = cross_sectional_regression(CrossSectionalLinearRegression(), Zd, Xc, Wc)
        minnorm = cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                            alg = MinimumNormSolve()),
                                             Zd, Xc, Wc)
        unchecked = cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                              alg = UncheckedSolve()),
                                               Zd, Xc, Wc)
        # The fallback pseudo-inverts, so it lands on the minimum-norm answer. The design is
        # not square, so `\` reaches a column-pivoted QR whose solve completes the
        # orthogonal factorisation and lands there too. All three agree.
        @test fallback.f ≈ minnorm.f
        @test unchecked.f ≈ minnorm.f
        @test predict(unchecked, Zd) ≈ predict(minnorm, Zd)
        # The two part on a SQUARE design, which `\` sends to an LU factorisation instead.
        # An exactly singular one throws, and the pseudo-inverse answers it.
        Zs = Zd[:, 1:size(Zd, 3), :]
        Xs = Xc[:, 1:size(Zd, 3)]
        Ws = Wc[:, 1:size(Zd, 3)]
        @test_throws LinearAlgebra.SingularException cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                                                               alg = UncheckedSolve()),
                                                                                Zs, Xs, Ws)
        square = cross_sectional_regression(CrossSectionalLinearRegression(), Zs, Xs, Ws)
        @test all(isfinite, square.f)
        @test square.f ≈ cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                    alg = MinimumNormSolve()),
                                     Zs, Xs, Ws).f
        # The refusal names the observation and the rank it measured.
        cre = CrossSectionalLinearRegression(; alg = RankDeficiencyRefusal())
        @test_throws ArgumentError cross_sectional_regression(cre, Zd, Xc, Wc)
        msg = try
            cross_sectional_regression(cre, Zd, Xc, Wc)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("observation 1", msg)
        @test occursin("rank $(size(Zd, 3) - 1)", msg)
        @test occursin("$(size(Zd, 2)) eligible assets", msg)
        # A target refuses before it fits, with the message of the linear member.
        tmsg = try
            cross_sectional_regression(CrossSectionalTargetRegression(;
                                                                      alg = RankDeficiencyRefusal()),
                                       Zd, Xc, Wc)
            ""
        catch e
            sprint(showerror, e)
        end
        @test tmsg == msg
        # A member means the same thing under both estimators, so a least-squares target
        # gives the factor returns of the linear member under each (#1625). Under the default
        # `PseudoInverseFallback()` the target fits the columns that the rank test keeps, and
        # the projection onto the row space gives the answer of least norm.
        # `DependentColumnDrop()` gives the dropped column zero: the pivot keeps the doubled
        # column, whose norm is larger (#1620). Every member gives the fitted values of the
        # linear member, and its intercept. `UncheckedSolve()` hands the full design to `GLM`,
        # whose own pivot drops a column.
        for intercept in (false, true)
            linear = cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                               intercept = intercept),
                                                Zd, Xc, Wc)
            for alg in (PseudoInverseFallback(), DependentColumnDrop(), UncheckedSolve())
                target = cross_sectional_regression(CrossSectionalTargetRegression(;
                                                                                   alg = alg,
                                                                                   intercept = intercept),
                                                    Zd, Xc, Wc)
                @test target.eps ≈ linear.eps
                @test intercept ? target.b ≈ linear.b : isnothing(target.b)
                if !isa(alg, UncheckedSolve)
                    own = cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                                    alg = alg,
                                                                                    intercept = intercept),
                                                     Zd, Xc, Wc)
                    @test target.f ≈ own.f
                end
            end
            drop = cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                             alg = DependentColumnDrop(),
                                                                             intercept = intercept),
                                              Zd, Xc, Wc)
            @test all(iszero, drop.f[:, 1])
            @test all(!iszero, drop.f[:, 2])
            @test drop.f[:, 3:end] ≈ linear.f[:, 3:end]
            @test !isapprox(drop.f, linear.f)
            @test drop.eps ≈ linear.eps
        end
        # A design of rank zero leaves nothing to fit, so every factor returns zero.
        for cre in (CrossSectionalTargetRegression(),
                    CrossSectionalTargetRegression(; alg = DependentColumnDrop()),
                    CrossSectionalLinearRegression(; alg = DependentColumnDrop()))
            zero_design = cross_sectional_regression(cre, zero(Zc), Xc, Wc)
            @test all(iszero, zero_design.f)
            @test zero_design.eps == Xc
        end
        # The projection on its own keeps the linear predictor, and lands on the answer of
        # the pseudo-inverse. At rank zero it returns its argument.
        D = Zd[1, :, :] .* sqrt.(Wc[1, :])
        r = PortfolioOptimisers.cross_sectional_rank(D)
        f = randn(StableRNG(1625), size(D, 2))
        p = PortfolioOptimisers.cross_sectional_row_space(f, D, r)
        @test D * p ≈ D * f
        @test p ≈ LinearAlgebra.pinv(D) * (D * f)
        @test norm(p) < norm(f)
        @test PortfolioOptimisers.cross_sectional_row_space(f, D, 0) === f
    end

    @testset "The solve algorithm of a target (#1625)" begin
        # A target has no pseudo-inverse, so `MinimumNormSolve()` is refused, on both doors.
        @test_throws "MinimumNormSolve() has no meaning" CrossSectionalTargetRegression(;
                                                                                        alg = MinimumNormSolve())
        @test_throws ArgumentError CrossSectionalTargetRegression(LinearModel(),
                                                                  MinimumNormSolve(), false,
                                                                  PortfolioOptimisers.FLoops.SequentialEx())
        # The projection keeps the answer of a target only when the target reads the design
        # through the linear predictor alone, which `is_basis_invariant` states. A target
        # that keeps its default answer is refused under the default, and named.
        @test !PortfolioOptimisers.is_basis_invariant(LeverageTestTarget())
        @test_throws "fit of LeverageTestTarget" CrossSectionalTargetRegression(;
                                                                                tgt = LeverageTestTarget())
        for alg in (UncheckedSolve(), DependentColumnDrop(), RankDeficiencyRefusal())
            cre = CrossSectionalTargetRegression(; tgt = LeverageTestTarget(), alg = alg)
            @test cre.alg === alg
        end
        @test CrossSectionalTargetRegression().alg === PseudoInverseFallback()
    end

    @testset "A nearly dependent factor set: one rank test at the scale of the design (#1719)" begin
        PO = PortfolioOptimisers
        seq = PO.FLoops.SequentialEx()
        T, N = 3, 40
        Z0 = randn(StableRNG(1), T, N, 3)
        Xn = randn(StableRNG(3), T, N) ./ 100
        Wn = ones(T, N)
        noise = randn(StableRNG(2), T, N)
        # Between a design condition number of about 1e8 and 1e14 / K the rank test of the fit
        # keeps every column, and the Gram matrix of `GLM` squares the condition number. A
        # basis-invariant target fits the design whose weighted Gram matrix is the identity,
        # so it gives the factor returns of the linear member at the precision of the design.
        # Before the fix it dropped the third column from 2e9: `[-1.36e-3, 5.02e-4, 0]`.
        for d in (1e-5, 1e-7, 1e-9, 1e-11)
            Zn = copy(Z0)
            Zn[:, :, 3] = Z0[:, :, 1] .+ d .* noise
            D = sqrt.(Wn[1, :]) .* Zn[1, :, :]
            @test PO.cross_sectional_rank(D) == 3
            tol = 100 * LinearAlgebra.cond(D) * eps()
            for alg in
                (PseudoInverseFallback(), DependentColumnDrop(), RankDeficiencyRefusal())
                linear = cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                                   alg = alg,
                                                                                   ex = seq),
                                                    Zn, Xn, Wn)
                @test isa(alg, DependentColumnDrop) ? !any(linear.dc) : isnothing(linear.dc)
                for tgt in (LinearModel(), GeneralisedLinearModel())
                    target = cross_sectional_regression(CrossSectionalTargetRegression(;
                                                                                       tgt = tgt,
                                                                                       alg = alg,
                                                                                       ex = seq),
                                                        Zn, Xn, Wn)
                    # The residuals subtract `Z * f` from returns a factor of `f` smaller,
                    # so they agree at the same precision. Measured worst: 2e-6 at 1e-11.
                    @test isapprox(target.f, linear.f; rtol = tol)
                    @test isapprox(target.eps, linear.eps; rtol = tol)
                end
            end
        end
        # The whitened fit of a full-rank design equals the plain fit to rounding.
        A = Zc[1, :, :]
        y = Xc[1, :]
        w = Wc[1, :]
        tgt = PO.factory(LinearModel(), StatsBase.aweights(w))
        plain = PO.StatsAPI.coef(PO.StatsAPI.fit(tgt, A, y))
        @test PO.cross_sectional_target_coef(tgt, A, y, A .* sqrt.(w)) ≈ plain
        # A target that is not basis-invariant fits the design itself.
        own = PO.factory(PinvTestTarget(nothing), StatsBase.aweights(w))
        @test PO.cross_sectional_target_coef(own, A, y, A .* sqrt.(w)) ==
              PO.StatsAPI.coef(PO.StatsAPI.fit(own, A, y))
        # On an exactly dependent design every member of a GLM target means what it means
        # for the linear member, and so does a caller's own target.
        Zd = copy(Zc)
        Zd[:, :, 2] = 2 .* Zd[:, :, 1]
        for intercept in (false, true)
            linear = cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                               intercept = intercept),
                                                Zd, Xc, Wc)
            drop = cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                             alg = DependentColumnDrop(),
                                                                             intercept = intercept),
                                              Zd, Xc, Wc)
            # The mask marks the column whose return the rule states, at every observation:
            # the pivot keeps the doubled column, whose norm is larger.
            dropped = falses(size(drop.f))
            dropped[:, 1] .= true
            @test drop.dc == dropped
            @test all(iszero, drop.f[drop.dc])
            for tgt in (GeneralisedLinearModel(), PinvTestTarget(nothing))
                gdrop = cross_sectional_regression(CrossSectionalTargetRegression(;
                                                                                  tgt = tgt,
                                                                                  alg = DependentColumnDrop(),
                                                                                  intercept = intercept),
                                                   Zd, Xc, Wc)
                @test gdrop.f ≈ drop.f
                @test gdrop.dc == drop.dc
                # The pseudo-inverse projection of a basis-invariant target is the one of the
                # linear member (R16 of #1630). The rule refuses the pseudo-inverse target.
                if tgt isa GeneralisedLinearModel
                    pif = cross_sectional_regression(CrossSectionalTargetRegression(;
                                                                                    tgt = tgt,
                                                                                    alg = PseudoInverseFallback(),
                                                                                    intercept = intercept),
                                                     Zd, Xc, Wc)
                    @test pif.eps ≈ linear.eps
                    @test pif.f ≈
                          cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                                    alg = PseudoInverseFallback(),
                                                                                    intercept = intercept),
                                                     Zd, Xc, Wc).f
                end
                @test_throws "has rank $(size(Zd, 3) - 1)" cross_sectional_regression(CrossSectionalTargetRegression(;
                                                                                                                     tgt = tgt,
                                                                                                                     alg = RankDeficiencyRefusal(),
                                                                                                                     intercept = intercept),
                                                                                      Zd,
                                                                                      Xc,
                                                                                      Wc)
                # `UncheckedSolve()` hands the full design to the target, whose own
                # factorisation decides the factor returns, and the fitted values stay those
                # of the linear member.
                unchecked = cross_sectional_regression(CrossSectionalTargetRegression(;
                                                                                      tgt = tgt,
                                                                                      alg = UncheckedSolve(),
                                                                                      intercept = intercept),
                                                       Zd, Xc, Wc)
                @test unchecked.eps ≈ linear.eps
                @test isnothing(unchecked.dc)
                # A pseudo-inverse target lands on the answer of least norm.
                if tgt isa PinvTestTarget
                    @test unchecked.f ≈ linear.f
                end
            end
            # A full-rank design passes the refusal of every target, with the intercept.
            refused = cross_sectional_regression(CrossSectionalTargetRegression(;
                                                                                alg = RankDeficiencyRefusal(),
                                                                                intercept = intercept),
                                                 Zc, Xc, Wc)
            plain = cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                              intercept = intercept),
                                               Zc, Xc, Wc)
            @test refused.f ≈ plain.f
            @test intercept ? refused.b ≈ plain.b : isnothing(refused.b)
        end
        # The mask has the shape of the factor returns, and a view keeps it.
        @test_throws DimensionMismatch CrossSectionalRegression(; f = ones(2, 2),
                                                                eps = ones(2, 3),
                                                                n = [3, 3],
                                                                dc = falses(2, 3))
        csr = CrossSectionalRegression(; f = ones(2, 2), eps = ones(2, 3), n = [3, 3],
                                       dc = BitMatrix([0 1; 0 0]))
        @test PO.port_opt_view(csr, [1, 3]).dc === csr.dc
        @test PO.cross_sectional_kept_columns(zeros(4, 2), 0) == Int[]
    end

    @testset "The two members agree" begin
        for intercept in (false, true)
            closed = cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                               intercept = intercept),
                                                Zc, Xc, Wc)
            target = cross_sectional_regression(CrossSectionalTargetRegression(;
                                                                               intercept = intercept),
                                                Zc, Xc, Wc)
            @test closed.f ≈ target.f
            @test closed.n == target.n
            if intercept
                @test closed.b ≈ target.b
            else
                @test isnothing(closed.b)
                @test isnothing(target.b)
            end
        end
        # A generalised linear model of the Normal family is the same least squares, so the
        # target field really is the seam it claims to be.
        glm = cross_sectional_regression(CrossSectionalTargetRegression(;
                                                                        tgt = GeneralisedLinearModel()),
                                         Zc, Xc, Wc)
        @test glm.f ≈
              cross_sectional_regression(CrossSectionalLinearRegression(), Zc, Xc, Wc).f
    end

    @testset "An observation with no eligible asset" begin
        We = copy(W)
        We[2, :] .= 0.0
        csr = cross_sectional_regression(CrossSectionalLinearRegression(), Z, X, We)
        @test csr.n[2] == 0
        @test all(iszero, csr.f[2, :])
        # The intercept of an empty cross-section is zero, because it has no centroid.
        ci = cross_sectional_regression(CrossSectionalLinearRegression(; intercept = true),
                                        Z, X, We)
        @test iszero(ci.b[2])
        @test all(iszero, ci.f[2, :])
        # An external target has no cross-section to fit, so it refuses and names the
        # observation.
        @test_throws ArgumentError cross_sectional_regression(CrossSectionalTargetRegression(),
                                                              Z, X, We)
        msg = try
            cross_sectional_regression(CrossSectionalTargetRegression(), Z, X, We)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("observation 2", msg)
        # The refusal of `RankDeficiencyRefusal` also covers the empty cross-section, whose
        # rank is zero.
        @test_throws ArgumentError cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                                             alg = RankDeficiencyRefusal()),
                                                              Z, X, We)
    end

    @testset "A block names its observations after the fitted ones (#1629)" begin
        PO = PortfolioOptimisers
        function message(f)
            try
                f()
                return ""
            catch e
                return sprint(showerror, e)
            end
        end
        Zd = copy(Zc)
        Zd[:, :, 2] = 2 .* Zd[:, :, 1]
        We = copy(W)
        We[2, :] .= 0.0
        for cre in (CrossSectionalLinearRegression(; alg = RankDeficiencyRefusal()),
                    CrossSectionalTargetRegression(; alg = RankDeficiencyRefusal()))
            m0 = message(() -> cross_sectional_regression(cre, Zd, Xc, Wc))
            @test message(() -> PO.cross_sectional_block_regression(cre, Zd, Xc, Wc, 0)) ==
                  m0
            m7 = message(() -> PO.cross_sectional_block_regression(cre, Zd, Xc, Wc, 7))
            @test m7 == replace(m0, "observation 1 " => "observation 8 ")
            lv = trues(size(Zd, 3))
            @test message(() -> PO.cross_sectional_live_regression(cre, Zd, Xc, Wc, lv, 7)) ==
                  m7
            # The offset changes the name in a refusal alone, never the fit.
            fit = PO.cross_sectional_block_regression(CrossSectionalLinearRegression(), Zc,
                                                      Xc, Wc, 7)
            @test fit.f ==
                  cross_sectional_regression(CrossSectionalLinearRegression(), Zc, Xc, Wc).f
        end
        # The refusal of an empty cross-section under a target names the same observation.
        cre = CrossSectionalTargetRegression()
        @test occursin("observation 12 has no asset",
                       message(() -> PO.cross_sectional_block_regression(cre, Z, X, We, 10)))
        # An estimator the library does not know answers the public verb, and numbers the
        # observations of the block itself. One that answers no verb is refused by name.
        csr = PO.cross_sectional_block_regression(BlockTestRegression(), Zc, Xc, Wc, 7)
        @test csr.f ==
              cross_sectional_regression(CrossSectionalLinearRegression(), Zc, Xc, Wc).f
        @test PO.cross_sectional_live_regression(BlockTestRegression(), Zc, Xc, Wc,
                                                 trues(size(Zc, 3)), 7).csr.f == csr.f
        @test_throws MethodError PO.cross_sectional_block_regression(BareTestRegression(),
                                                                     Zc, Xc, Wc, 7)
    end

    @testset "The coefficient of determination" begin
        csr = cross_sectional_regression(CrossSectionalLinearRegression(), Z, X, W)
        r2 = cross_sectional_r2(csr, Z, X, W)
        @test length(r2) == size(X, 1)
        @test all(x -> 0 <= x <= 1, r2)
        @test mean_cross_sectional_r2(csr, Z, X, W) ≈ sum(r2) / length(r2)
        # An exact fit scores one. Four assets and four factors leave no residual.
        Ze = Zc[:, 1:size(Zc, 3), :]
        Xe = Xc[:, 1:size(Zc, 3)]
        We = Wc[:, 1:size(Zc, 3)]
        exact = cross_sectional_regression(CrossSectionalLinearRegression(), Ze, Xe, We)
        @test cross_sectional_r2(exact, Ze, Xe, We) ≈ ones(size(Ze, 1))
        # An observation with no eligible asset has no ratio, and the mean skips it.
        Wn = copy(W)
        Wn[2, :] .= 0.0
        csn = cross_sectional_regression(CrossSectionalLinearRegression(), Z, X, Wn)
        r2n = cross_sectional_r2(csn, Z, X, Wn)
        @test isnan(r2n[2])
        @test mean_cross_sectional_r2(csn, Z, X, Wn) ≈
              sum(filter(isfinite, r2n)) / count(isfinite, r2n)
        # A cross-section whose returns are all equal has a zero total sum of squares, so
        # its ratio is undefined too, and a panel of such observations has no mean at all.
        Zf = ones(2, 3, 1)
        Xf = ones(2, 3)
        Wf = ones(2, 3)
        csf = cross_sectional_regression(CrossSectionalLinearRegression(), Zf, Xf, Wf)
        @test all(isnan, cross_sectional_r2(csf, Zf, Xf, Wf))
        @test isnan(mean_cross_sectional_r2(csf, Zf, Xf, Wf))
    end

    @testset "The systematic part on a tensor the fit did not see" begin
        csr = cross_sectional_regression(CrossSectionalLinearRegression(), Zc, Xc, Wc)
        # The asset axis may differ, and the observation and factor axes may not.
        @test size(predict(csr, Zc[:, 1:3, :])) == (size(Zc, 1), 3)
        @test_throws DimensionMismatch predict(csr, Zc[1:2, :, :])
        @test_throws DimensionMismatch predict(csr, Zc[:, :, 1:2])
    end

    @testset "The result and its view" begin
        csr = cross_sectional_regression(CrossSectionalLinearRegression(; intercept = true),
                                         Zc, Xc, Wc)
        v = PortfolioOptimisers.port_opt_view(csr, [1, 3, 5])
        # The asset axis of a cross-sectional result is the second one, so only `eps` moves.
        @test size(v.eps) == (size(Zc, 1), 3)
        @test v.eps == csr.eps[:, [1, 3, 5]]
        @test v.f === csr.f
        @test v.n === csr.n
        @test v.b === csr.b
        @test v.h1 == csr.h1[:, [1, 3, 5]]
        # The passthrough returns the result it was handed.
        @test cross_sectional_regression(csr, Zc, Xc, Wc) === csr
        @test cross_sectional_regression(csr) === csr
    end

    @testset "A pair of leverage one has a zero residual and a mark (#1423)" begin
        # Asset 4 is the only member of the second level, so the design gives it a direction
        # of its own: the fit reproduces its return, whatever the return is. Its residual is
        # zero in exact arithmetic, and the subtraction would leave only round-off.
        T, N = 3, 5
        Zh = zeros(T, N, 2)
        Zh[:, :, 1] .= 1.0
        Zh[:, 4, 2] .= 1.0
        Xh = [0.1 0.7 0.3 0.123456789 -0.2; 0.3 -0.1 0.25 0.987654321 0.4;
              -0.3 0.2 0.1 0.333333333 0.05]
        Wh = [1.0 2.0 1.0 3.0 1.0; 1.0 1.0 1.0 1.0 1.0; 1.0 1.0 1.0 0.0 1.0]
        mark = falses(T, N)
        mark[1:2, 4] .= true
        for cre in (CrossSectionalLinearRegression(), CrossSectionalTargetRegression(),
                    CrossSectionalTargetRegression(; tgt = GeneralisedLinearModel()),
                    CrossSectionalLinearRegression(; alg = MinimumNormSolve()))
            csr = cross_sectional_regression(cre, Zh, Xh, Wh)
            # The pair of zero weight at observation 3 left the fit, so it has no mark.
            @test csr.h1 == mark
            @test all(iszero, csr.eps[mark])
            @test all(!iszero, csr.eps[.!mark])
            # The mark reads the design alone, so the scale of the returns does not move it.
            @test cross_sectional_regression(cre, Zh, 1e-9 .* Xh, Wh).h1 == mark
        end
        # With an intercept, the level's own column still gives asset 4 its direction.
        Zi = Zh[:, :, 2:2]
        csr = cross_sectional_regression(CrossSectionalLinearRegression(; intercept = true),
                                         Zi, Xh, Wh)
        @test csr.h1 == mark
        @test all(iszero, csr.eps[mark])
        # Under a log link the fit reproduces the pair on the scale of the link, so the
        # residual on the scale of the returns keeps its value, and only the mark is set.
        ls = PortfolioOptimisers.cross_sectional_least_squares
        glmp = PortfolioOptimisers.GLM
        nrm = PortfolioOptimisers.Distributions.Normal()
        glm = GeneralisedLinearModel(; args = (nrm, glmp.LogLink()))
        @test !ls(CrossSectionalTargetRegression(; tgt = glm))
        @test ls(GeneralisedLinearModel(; args = (nrm, glmp.IdentityLink())))
        @test !ls(LeverageTestTarget())
        # The mask on its own: no row or no column marks nothing, a zero design has rank zero
        # and marks nothing, and a design with fewer rows than columns fits every row.
        lo = PortfolioOptimisers.cross_sectional_leverage_one
        @test lo(zeros(0, 2), Float64[], false) == falses(0)
        @test lo(zeros(3, 0), ones(3), false) == falses(3)
        @test lo(zeros(3, 2), ones(3), false) == falses(3)
        @test lo([1.0 2.0 3.0; 4.0 5.0 7.0], [1.0, 2.0], false) == trues(2)
        @test lo([1.0 0.0; 1.0 0.0; 1.0 1.0], [1.0, 3.0, 0.5], false) ==
              BitVector([false, false, true])
        # With the intercept column, the constant minus the level's column is the other
        # level's own direction, so a lone member of either level is marked, and a level of
        # two members on each side marks nothing.
        @test lo([0.0; 0.0; 1.0;;], [1.0, 1.0, 1.0], true) ==
              BitVector([false, false, true])
        @test lo([0.0; 1.0; 1.0;;], [1.0, 1.0, 1.0], true) ==
              BitVector([true, false, false])
        @test lo([0.0; 0.0; 1.0; 1.0;;], [1.0, 2.0, 1.0, 3.0], true) == falses(4)
        # A result a caller builds by hand states no mark, and the constructor checks one.
        f = [1.0 2.0; 3.0 4.0]
        e = [0.1 0.2 0.3; 0.4 0.5 0.6]
        @test isnothing(CrossSectionalRegression(; f = f, eps = e, n = [3, 3]).h1)
        @test_throws DimensionMismatch CrossSectionalRegression(; f = f, eps = e,
                                                                n = [3, 3],
                                                                h1 = falses(2, 2))
    end

    @testset "The result refuses a malformed construction" begin
        f = [1.0 2.0; 3.0 4.0]
        eps = [0.1 0.2 0.3; 0.4 0.5 0.6]
        n = [3, 3]
        @test CrossSectionalRegression(; f = f, eps = eps, n = n) isa
              CrossSectionalRegression
        @test_throws PortfolioOptimisers.IsEmptyError CrossSectionalRegression(;
                                                                               f = zeros(0,
                                                                                         0),
                                                                               eps = eps,
                                                                               n = n)
        @test_throws PortfolioOptimisers.IsEmptyError CrossSectionalRegression(; f = f,
                                                                               eps = zeros(0,
                                                                                           0),
                                                                               n = n)
        @test_throws PortfolioOptimisers.IsEmptyError CrossSectionalRegression(; f = f,
                                                                               eps = eps,
                                                                               n = Int[])
        @test_throws DimensionMismatch CrossSectionalRegression(; f = f, eps = eps,
                                                                n = [3, 3, 3])
        @test_throws DomainError CrossSectionalRegression(; f = f, eps = eps, n = [3, -1])
        @test_throws PortfolioOptimisers.IsEmptyError CrossSectionalRegression(; f = f,
                                                                               eps = eps,
                                                                               n = n,
                                                                               b = Float64[])
        @test_throws DimensionMismatch CrossSectionalRegression(; f = f, eps = eps, n = n,
                                                                b = [1.0, 2.0, 3.0])
    end

    @testset "The cache of a refit: the live mask, the active mask and the forecast history (#1751)" begin
        f = [1.0 0.0; 3.0 0.0]
        eps = [0.1 0.2 0.3; 0.4 0.5 0.6]
        n = [3, 3]
        # A regression records which factors it estimated, one entry per column of `f`.
        csr = CrossSectionalRegression(; f = f, eps = eps, n = n, lv = BitVector([1, 0]))
        @test csr.lv == [true, false]
        @test isnothing(CrossSectionalRegression(; f = f, eps = eps, n = n).lv)
        @test_throws DimensionMismatch CrossSectionalRegression(; f = f, eps = eps, n = n,
                                                                lv = trues(3))
        # The mask indexes factors, so an asset view keeps it.
        @test PortfolioOptimisers.port_opt_view(csr, [1, 3]).lv === csr.lv
        # The block stores the two histories of the cache on the rows of the fit.
        M = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        amr = BitMatrix([1 1 0; 1 1 1])
        hist = [0.1 0.2 0.3; 0.4 0.5 0.6]
        rw = ones(2, 3)
        blk = CrossSectionalFactorModel(; M = M, b = zeros(3), csr = csr, rw = rw,
                                        amr = amr, hist = hist)
        @test blk.amr === amr
        @test blk.hist === hist
        @test isnothing(CrossSectionalFactorModel(; M = M, b = zeros(3)).amr)
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = zeros(3),
                                                                 amr = trues(2, 2))
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = zeros(3),
                                                                 hist = zeros(2, 4))
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = zeros(3),
                                                                 rw = rw, amr = trues(3, 3))
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = zeros(3),
                                                                 rw = rw,
                                                                 hist = zeros(3, 3))
        # An asset view cuts both histories on their asset axis.
        v = PortfolioOptimisers.port_opt_view(blk, [1, 3])
        @test v.amr == amr[:, [1, 3]]
        @test v.hist == hist[:, [1, 3]]
        # The forecast block keeps the cache, so every refit after the first reads it.
        sp = (; rf = nothing, g = nothing, ap = nothing, hist = nothing)
        rb = PortfolioOptimisers.cross_sectional_forecast_block(blk, sp, 1.0, 1.0)
        @test rb.amr === amr
        @test rb.hist === hist
        @test isnothing(getfield(rb, :L))
    end

    @testset "The type tree" begin
        # Both roots are umbrellas. An estimator child names the geometry it fits, and
        # a result child names what a member carries.
        @test StepwiseRegression <:
              PortfolioOptimisers.AbstractTimeSeriesRegressionEstimator
        @test DimensionReductionRegression <:
              PortfolioOptimisers.AbstractTimeSeriesRegressionEstimator
        @test Regression <: PortfolioOptimisers.AbstractLoadingsRegressionResult
        @test CrossSectionalLinearRegression <:
              PortfolioOptimisers.AbstractCrossSectionalRegressionEstimator
        @test CrossSectionalTargetRegression <:
              PortfolioOptimisers.AbstractCrossSectionalRegressionEstimator
        @test CrossSectionalRegression <:
              PortfolioOptimisers.AbstractCrossSectionalRegressionResult
        @test PortfolioOptimisers.AbstractTimeSeriesRegressionEstimator <:
              PortfolioOptimisers.AbstractRegressionEstimator
        @test PortfolioOptimisers.AbstractCrossSectionalRegressionResult <:
              PortfolioOptimisers.AbstractRegressionResult
        @test PortfolioOptimisers.AbstractLoadingsRegressionResult <:
              PortfolioOptimisers.AbstractRegressionResult
        # The loadings criterion partitions the two result children, so neither is a
        # subtype of the other.
        @test !(PortfolioOptimisers.AbstractCrossSectionalRegressionResult <:
                PortfolioOptimisers.AbstractLoadingsRegressionResult)
        @test !(PortfolioOptimisers.AbstractLoadingsRegressionResult <:
                PortfolioOptimisers.AbstractCrossSectionalRegressionResult)
        # The loadings root replaced `AbstractTimeSeriesRegressionResult`, so a site that
        # kept the old name fails to resolve rather than binding a stale root.
        @test !isdefined(PortfolioOptimisers, :AbstractTimeSeriesRegressionResult)
        # `RegE_Reg` pairs the loadings result root with the time-series estimator root,
        # so no cross-sectional type reaches a consumer that reads the loadings matrix.
        @test StepwiseRegression <: PortfolioOptimisers.RegE_Reg
        @test Regression <: PortfolioOptimisers.RegE_Reg
        @test !(CrossSectionalLinearRegression <: PortfolioOptimisers.RegE_Reg)
        @test !(CrossSectionalRegression <: PortfolioOptimisers.RegE_Reg)
        # The four solve members share one algorithm root.
        for alg in (PseudoInverseFallback(), RankDeficiencyRefusal(), UncheckedSolve(),
                    MinimumNormSolve())
            @test isa(alg, PortfolioOptimisers.AbstractCrossSectionalSolveAlgorithm)
            @test isa(alg, PortfolioOptimisers.AbstractRegressionAlgorithm)
        end
        # None of the new abstract types is exported.
        for T in (:AbstractTimeSeriesRegressionEstimator,
                  :AbstractCrossSectionalRegressionEstimator, :AbstractLoadingsRegressionResult,
                  :AbstractCrossSectionalRegressionResult,
                  :AbstractCrossSectionalSolveAlgorithm)
            @test !Base.isexported(PortfolioOptimisers, T)
        end
    end

    @testset "The working type is the one a division of the panel lands in" begin
        Zi = ones(Int, 2, 3, 1)
        Zi[:, 2, 1] .= 2
        Zi[:, 3, 1] .= 3
        Xi = [1 2 3; 2 4 6]
        Wi = ones(Int, 2, 3)
        csr = cross_sectional_regression(CrossSectionalLinearRegression(), Zi, Xi, Wi)
        @test eltype(csr.f) == Float64
        @test csr.f ≈ [1.0; 2.0;;]
        # The division widens nothing that is already wide enough.
        csr32 = cross_sectional_regression(CrossSectionalLinearRegression(),
                                           Array{Float32}(Zi), Float32.(Xi), Float32.(Wi))
        @test eltype(csr32.f) == Float32
        @test csr32.f ≈ [1.0f0; 2.0f0;;]
        # The coefficient of determination follows the fit's type rather than the `NaN` it
        # is filled with, and its mean does too.
        r2 = cross_sectional_r2(csr32, Array{Float32}(Zi), Float32.(Xi), Float32.(Wi))
        @test eltype(r2) == Float32
        @test r2 ≈ ones(Float32, 2)
        @test mean_cross_sectional_r2(csr32, Array{Float32}(Zi), Float32.(Xi),
                                      Float32.(Wi)) isa Float32
        @test eltype(cross_sectional_r2(csr, Zi, Xi, Wi)) == Float64
        # A `Rational` panel keeps its type in `f`, but both members solve in floating
        # point, so the fit lands near the exact slope 13//14 rather than on it, #1352. The
        # float reference carries the tolerance, because `≈` of two `Rational`s has none.
        Zr = reshape(Rational{Int}[1, 2, 3], 1, 3, 1)
        for cre in (CrossSectionalLinearRegression(), CrossSectionalTargetRegression())
            csrr = cross_sectional_regression(cre, Zr, Rational{Int}[1 3 2],
                                              ones(Rational{Int}, 1, 3))
            @test eltype(csrr.f) == Rational{Int}
            @test csrr.f[1] ≈ 13 / 14
        end
    end
end

#=
Check `src/05_Moments/32_CrossSectionalFactorModel/02_CrossSectionalWeights.jl` against the mathematics its docstrings
state, and against the oracle that the map of issue #643 ports. Issue #681.

THREE FACTS SHAPE THE PROBES.

1. THE WEIGHT PAIRED WITH AN OBSERVATION READS NO RESIDUAL OF THAT OBSERVATION. The refine
   step reads row `t - 1` of the variance series, so row 1 carries no estimate at all and
   row 2 reads a fit on one observation. `The warm-up rows fall back to the cap weights`
   pins both rows. `A later residual moves no earlier weight` pins the direction twice: a
   change to the last residual moves no weight at all, and a change to an early one moves
   every later weight and no earlier one.

2. THE REALISED BLEND MUST EQUAL THE NOMINAL SHRINKAGE. Both components are normalised over
   the eligible set before they are blended, so every eligible observation sums to one and
   the share of the inverse variance component is exactly `lambda`. A blend that normalised
   over different universes would drift away from the number the caller wrote.
   `The realised blend equals the nominal shrinkage` measures both halves.

3. THE STORED WEIGHTS COME FROM THE ORACLE. `ORACLE_W1_A` and `ORACLE_W1_B` were produced by
   the oracle's own winsoriser and blend, on the inputs written beside them. The plain
   Julia `reference_blended_weights` below re-derives the same quantity a second way, so a
   testset never compares the file against itself.
=#

using Statistics
include(joinpath(@__DIR__, "parity_harness.jl"))

# The blend written out one observation at a time, from the variance series alone. It shares
# no code with the implementation: the winsorisation, the median cap and the two
# normalisations are written here in full.
function reference_blended_weights(V, W0, mask, lambda, ratio, wins)
    T, N = size(W0)
    IV = fill(NaN, T, N)
    for t in 2:T, i in 1:N
        IV[t, i] = 1 / V[t - 1, i]
    end
    for t in 1:T, i in 1:N
        if !mask[t, i]
            IV[t, i] = NaN
        end
    end
    ready = falses(T)
    for t in 1:T
        u = [IV[t, i] for i in 1:N if W0[t, i] > 0 && !isnan(IV[t, i])]
        if any(isfinite, u)
            qlo, qhi = quantile(u, wins[1]), quantile(u, wins[2])
            for i in 1:N
                IV[t, i] = clamp(IV[t, i], qlo, qhi)
            end
        else
            IV[t, :] .= NaN
        end
        ready[t] = any(isfinite, view(IV, t, :))
    end
    for t in 1:T
        if !ready[t]
            continue
        end
        cap = median([IV[t, i] for i in 1:N if !isnan(IV[t, i])]) * ratio
        IV[t, :] .= min.(view(IV, t, :), cap)
        IV[t, :] ./= sum(x for x in view(IV, t, :) if !isnan(x))
    end
    Wm = W0 ./ sum(W0; dims = 2)
    W1 = zeros(T, N)
    for t in 1:T, i in 1:N
        u = !ready[t] ? Wm[t, i] : (isnan(IV[t, i]) ? 0.0 : IV[t, i])
        W1[t, i] = mask[t, i] ? lambda * u + (1 - lambda) * Wm[t, i] : 0.0
    end
    return W1
end

# The expanding-window variance series, written out with `var` so the probe never reads the
# library's own `variance_series`.
function reference_variance_series(X)
    T, N = size(X)
    V = fill(NaN, T, N)
    for t in 2:T, i in 1:N
        V[t, i] = var(view(X, 1:t, i))
    end
    return V
end

# Four observations of three assets, every pair eligible, market caps 100, 200 and 300.
const EPS_A = [0.010 -0.020 0.030
               0.020 0.010 -0.010
               -0.030 0.020 0.010
               0.040 -0.010 0.020]
const MASK_A = trues(4, 3)
const MCAP_A = [100.0 200.0 300.0
                100.0 200.0 300.0
                100.0 200.0 300.0
                100.0 200.0 300.0]
# `MarketCapWeights(; p = 1.0)` and `BlendedInverseVarianceWeights(; p = 1.0, lambda = 0.5,
# ratio = 20.0, wins = (0.025, 0.975))`, as the oracle answers them.
const ORACLE_W1_A = [0.16666666666666666 0.3333333333333333 0.5
                     0.16666666666666666 0.3333333333333333 0.5
                     0.5055487368313388 0.21576148102689985 0.27868978214176127
                     0.20074844833880978 0.350675428988682 0.4485761226725082]

# Five observations of four assets, one pair excluded per observation.
const EPS_B = [0.010 -0.020 0.030 0.005
               0.020 0.010 -0.010 0.015
               -0.030 0.020 0.010 -0.005
               0.040 -0.010 0.020 0.025
               0.005 0.030 -0.020 0.010]
const MASK_B = [true true true false
                true true false true
                true true true true
                false true true true
                true false true true]
const MCAP_B = [400.0 100.0 900.0 1600.0
                400.0 100.0 900.0 1600.0
                400.0 100.0 900.0 1600.0
                400.0 100.0 900.0 1600.0
                400.0 100.0 900.0 1600.0]
# `BlendedInverseVarianceWeights(; p = 0.5, lambda = 1.0, ratio = 2.0, wins = (0.1, 0.9))`,
# as the oracle answers it.
const ORACLE_W1_B = [0.3333333333333333 0.16666666666666666 0.5 0.0
                     0.2857142857142857 0.14285714285714285 0.0 0.5714285714285714
                     0.45699777848302126 0.050777530942557915 0.035226912091399554 0.45699777848302126
                     0.0 0.23828125 0.25390625 0.5078125
                     0.15288220551378442 0.0 0.3258145363408521 0.5213032581453634]

@testset "Cross-sectional weights" begin
    @testset "The type tree" begin
        @test MarketCapWeights <: PortfolioOptimisers.AbstractCrossSectionalWeightsAlgorithm
        @test BlendedInverseVarianceWeights <:
              PortfolioOptimisers.AbstractCrossSectionalWeightsAlgorithm
        @test PortfolioOptimisers.AbstractCrossSectionalWeightsAlgorithm <:
              PortfolioOptimisers.AbstractAlgorithm
        # The root answers `false`, so a one-pass member needs no method of its own.
        @test !PortfolioOptimisers.needs_second_pass(MarketCapWeights())
        @test PortfolioOptimisers.needs_second_pass(BlendedInverseVarianceWeights(;
                                                                                  lambda = 0.5))
    end

    @testset "The cap weights raise a power on the eligible pairs alone" begin
        mask = [true true false
                false true true]
        mcap = [4.0 9.0 16.0
                25.0 36.0 49.0]
        @test PortfolioOptimisers.cross_sectional_cap_weights(0.5, mcap, mask) ==
              [2.0 3.0 0.0; 0.0 6.0 7.0]
        @test PortfolioOptimisers.cross_sectional_cap_weights(1.0, mcap, mask) ==
              [4.0 9.0 0.0; 0.0 36.0 49.0]
        # A power of zero needs no capitalisation matrix, and gives every eligible pair one.
        @test PortfolioOptimisers.cross_sectional_cap_weights(0.0, nothing, mask) ==
              [1.0 1.0 0.0; 0.0 1.0 1.0]
        @test PortfolioOptimisers.cross_sectional_cap_weights(0.0, mcap, mask) ==
              [1.0 1.0 0.0; 0.0 1.0 1.0]
        # Both members read the capitalisation the same way.
        @test PortfolioOptimisers.cs_weights_initial(MarketCapWeights(; p = 0.5), mcap,
                                                     mask) ==
              PortfolioOptimisers.cs_weights_initial(BlendedInverseVarianceWeights(;
                                                                                   p = 0.5,
                                                                                   lambda = 0.3),
                                                     mcap, mask)
        # The weights take the type the power lands in, and coerce nothing. An integer
        # panel raised to an integer power stays integer, a fractional power lands in a
        # float, and a `Float32` panel stays `Float32`.
        @test eltype(PortfolioOptimisers.cross_sectional_cap_weights(1, [1 2; 3 4],
                                                                     trues(2, 2))) == Int
        @test eltype(PortfolioOptimisers.cross_sectional_cap_weights(0.5, [1 2; 3 4],
                                                                     trues(2, 2))) ==
              Float64
        @test eltype(PortfolioOptimisers.cross_sectional_cap_weights(0.5f0,
                                                                     Float32[1 2; 3 4],
                                                                     trues(2, 2))) ==
              Float32
    end

    @testset "The cap weights refuse a malformed design" begin
        mask = [true true false]
        @test_throws PortfolioOptimisers.IsEmptyError PortfolioOptimisers.cross_sectional_cap_weights(0.5,
                                                                                                      nothing,
                                                                                                      falses(0,
                                                                                                             0))
        @test_throws PortfolioOptimisers.IsNothingError PortfolioOptimisers.cross_sectional_cap_weights(0.5,
                                                                                                        nothing,
                                                                                                        mask)
        @test_throws DimensionMismatch PortfolioOptimisers.cross_sectional_cap_weights(0.5,
                                                                                       [1.0 2.0],
                                                                                       mask)
        # An eligible pair must carry a finite, non-negative capitalisation. An ineligible
        # one may carry anything, because the power is never applied to it.
        @test_throws DomainError PortfolioOptimisers.cross_sectional_cap_weights(0.5,
                                                                                 [1.0 -2.0 3.0],
                                                                                 mask)
        @test_throws DomainError PortfolioOptimisers.cross_sectional_cap_weights(0.5,
                                                                                 [1.0 NaN 3.0],
                                                                                 mask)
        @test PortfolioOptimisers.cross_sectional_cap_weights(0.5, [1.0 4.0 -9.0], mask) ==
              [1.0 2.0 0.0]
    end

    @testset "The constructors refuse a malformed policy" begin
        @test_throws DomainError MarketCapWeights(; p = NaN)
        @test_throws DomainError MarketCapWeights(; p = -0.5)
        @test_throws DomainError BlendedInverseVarianceWeights(; p = Inf, lambda = 0.5)
        @test_throws DomainError BlendedInverseVarianceWeights(; lambda = -0.1)
        @test_throws DomainError BlendedInverseVarianceWeights(; lambda = 1.1)
        @test_throws DomainError BlendedInverseVarianceWeights(; lambda = 0.5, ratio = 0.0)
        @test_throws DomainError BlendedInverseVarianceWeights(; lambda = 0.5, ratio = Inf)
        @test_throws DomainError BlendedInverseVarianceWeights(; lambda = 0.5,
                                                               wins = (0.9, 0.1))
        @test_throws DomainError BlendedInverseVarianceWeights(; lambda = 0.5,
                                                               wins = (-0.1, 0.9))
        @test_throws DomainError BlendedInverseVarianceWeights(; lambda = 0.5,
                                                               wins = (0.1, 1.1))
        # The two endpoints and a zero blend are all admissible.
        @test BlendedInverseVarianceWeights(; lambda = 0.0, wins = (0.0, 1.0)) isa
              BlendedInverseVarianceWeights
        @test MarketCapWeights(; p = 0.0) isa MarketCapWeights
    end

    @testset "The lagged inverse variance reads no later observation" begin
        eps = [1.0 2.0
               3.0 6.0
               2.0 4.0]
        IV = PortfolioOptimisers.cross_sectional_lagged_inverse_variance(SimpleVariance(),
                                                                         eps, trues(3, 2))
        # Row 1 carries no estimate, and row 2 reads a fit on one observation.
        @test all(isnan, view(IV, 1, :))
        @test all(isnan, view(IV, 2, :))
        @test IV[3, :] ≈ [0.5, 0.125]
        # Row 3 reads rows 1 and 2 alone, so row 3's own residual moves nothing.
        eps2 = copy(eps)
        eps2[3, :] = [100.0, -100.0]
        IV2 = PortfolioOptimisers.cross_sectional_lagged_inverse_variance(SimpleVariance(),
                                                                          eps2, trues(3, 2))
        @test IV2[3, :] == IV[3, :]
        # A pair outside the mask carries no estimate, whatever its variance.
        IV3 = PortfolioOptimisers.cross_sectional_lagged_inverse_variance(SimpleVariance(),
                                                                          eps,
                                                                          [true false
                                                                           true true
                                                                           false true])
        @test isnan(IV3[3, 1])
        @test IV3[3, 2] ≈ 0.125
        @test_throws PortfolioOptimisers.IsEmptyError PortfolioOptimisers.cross_sectional_lagged_inverse_variance(SimpleVariance(),
                                                                                                                  zeros(0,
                                                                                                                        0),
                                                                                                                  falses(0,
                                                                                                                         0))
        @test_throws DimensionMismatch PortfolioOptimisers.cross_sectional_lagged_inverse_variance(SimpleVariance(),
                                                                                                   eps,
                                                                                                   trues(3,
                                                                                                         3))
    end

    @testset "The bounds cap an extreme inverse variance" begin
        # No winsorisation, a median of 3 and a ratio of 2 cap the row at 6.
        IV = [1.0 3.0 100.0]
        PortfolioOptimisers.cross_sectional_winsorise!(IV, ones(1, 3), (0.0, 1.0))
        @test IV == [1.0 3.0 100.0]
        ready = PortfolioOptimisers.cross_sectional_median_cap!(IV, 2.0)
        @test ready == [true]
        @test IV ≈ [0.1 0.3 0.6]
        # The winsorisation reads the estimation universe, which is the set of pairs with a
        # positive first-pass weight. The third asset leaves it, so the quantiles come from
        # the first two alone and the third is clipped to them.
        IV = [1.0 3.0 100.0]
        PortfolioOptimisers.cross_sectional_winsorise!(IV, [1.0 1.0 0.0], (0.0, 1.0))
        @test IV == [1.0 3.0 3.0]
        PortfolioOptimisers.cross_sectional_median_cap!(IV, 1e6)
        @test IV ≈ [1.0 3.0 3.0] ./ 7.0
        # An observation with no estimate at all takes an all-NaN row and answers `false`.
        IV = [NaN NaN NaN]
        PortfolioOptimisers.cross_sectional_winsorise!(IV, ones(1, 3), (0.0, 1.0))
        @test all(isnan, IV)
        ready = PortfolioOptimisers.cross_sectional_median_cap!(IV, 2.0)
        @test ready == [false]
        @test all(isnan, IV)
        @test_throws DimensionMismatch PortfolioOptimisers.cross_sectional_winsorise!([1.0 2.0],
                                                                                      ones(1,
                                                                                           3),
                                                                                      (0.0,
                                                                                       1.0))
    end

    @testset "The port reproduces the stored oracle" begin
        rwA = BlendedInverseVarianceWeights(; p = 1.0, lambda = 0.5, ratio = 20.0,
                                            wins = (0.025, 0.975))
        W0A = PortfolioOptimisers.cs_weights_initial(rwA, MCAP_A, MASK_A)
        @test W0A == MCAP_A
        W1A = PortfolioOptimisers.cs_weights_refine(rwA, W0A, EPS_A, SimpleVariance(),
                                                    MASK_A)
        # Cell by cell. Measured maxrel 2.2e-16 and 3.4e-16 on the two cases: the round-off
        # of the two normalisations, summed in another order. `1e-14` is about thirty times
        # the worst value.
        @test parity_compare(W1A, ORACLE_W1_A; rtol = 1e-14, name = "case A").ok

        rwB = BlendedInverseVarianceWeights(; p = 0.5, lambda = 1.0, ratio = 2.0,
                                            wins = (0.1, 0.9))
        W0B = PortfolioOptimisers.cs_weights_initial(rwB, MCAP_B, MASK_B)
        @test W0B == [20.0 10.0 30.0 0.0
                      20.0 10.0 0.0 40.0
                      20.0 10.0 30.0 40.0
                      0.0 10.0 30.0 40.0
                      20.0 0.0 30.0 40.0]
        W1B = PortfolioOptimisers.cs_weights_refine(rwB, W0B, EPS_B, SimpleVariance(),
                                                    MASK_B)
        @test parity_compare(W1B, ORACLE_W1_B; rtol = 1e-14, name = "case B").ok
        # An ineligible pair carries no weight at all.
        @test all(iszero, W1B[.!MASK_B])
    end

    @testset "The blend agrees with the mathematics written out" begin
        for (eps, mask, mcap, p, lambda, ratio, wins) in
            ((EPS_A, MASK_A, MCAP_A, 1.0, 0.5, 20.0, (0.025, 0.975)),
             (EPS_B, MASK_B, MCAP_B, 0.5, 1.0, 2.0, (0.1, 0.9)),
             (EPS_B, MASK_B, MCAP_B, 0.0, 0.25, 20.0, (0.05, 0.95)))
            alg = BlendedInverseVarianceWeights(; p = p, lambda = lambda, ratio = ratio,
                                                wins = wins)
            W0 = PortfolioOptimisers.cs_weights_initial(alg, iszero(p) ? nothing : mcap,
                                                        mask)
            W1 = PortfolioOptimisers.cs_weights_refine(alg, W0, eps, SimpleVariance(), mask)
            V = reference_variance_series(eps)
            # Cell by cell. Measured maxrel 1.3e-16 or less on the three cases.
            E = reference_blended_weights(V, W0, mask, lambda, ratio, wins)
            @test parity_compare(W1, E; rtol = 1e-14, name = "blend p = $(p)").ok
        end
    end

    @testset "The warm-up rows fall back to the cap weights" begin
        alg = BlendedInverseVarianceWeights(; p = 1.0, lambda = 1.0, ratio = 20.0)
        W0 = PortfolioOptimisers.cs_weights_initial(alg, MCAP_A, MASK_A)
        W1 = PortfolioOptimisers.cs_weights_refine(alg, W0, EPS_A, SimpleVariance(), MASK_A)
        # Rows 1 and 2 carry no variance estimate, so they take the normalised cap weights
        # even at a blend of one, which asks for the inverse variance component alone.
        cap = MCAP_A ./ sum(MCAP_A; dims = 2)
        @test W1[1, :] ≈ cap[1, :]
        @test W1[2, :] ≈ cap[2, :]
        @test !isapprox(W1[3, :], cap[3, :])
    end

    @testset "A later residual moves no earlier weight" begin
        alg = BlendedInverseVarianceWeights(; p = 1.0, lambda = 0.5)
        W0 = PortfolioOptimisers.cs_weights_initial(alg, MCAP_A, MASK_A)
        W1 = PortfolioOptimisers.cs_weights_refine(alg, W0, EPS_A, SimpleVariance(), MASK_A)
        # The weight of the last observation reads every earlier residual and none of its
        # own, so changing the last residual moves no weight at all.
        late = copy(EPS_A)
        late[4, :] = [5.0, -5.0, 5.0]
        @test PortfolioOptimisers.cs_weights_refine(alg, W0, late, SimpleVariance(),
                                                    MASK_A) == W1
        # Changing an earlier residual moves every weight that reads it, and no other.
        early = copy(EPS_A)
        early[2, :] = [5.0, -5.0, 5.0]
        W1b = PortfolioOptimisers.cs_weights_refine(alg, W0, early, SimpleVariance(),
                                                    MASK_A)
        @test W1b[1:2, :] == W1[1:2, :]
        @test W1b[3, :] != W1[3, :]
        @test W1b[4, :] != W1[4, :]
    end

    @testset "The realised blend equals the nominal shrinkage" begin
        rwcap = MarketCapWeights(; p = 0.5)
        W0 = PortfolioOptimisers.cs_weights_initial(rwcap, MCAP_B, MASK_B)
        cap = W0 ./ sum(W0; dims = 2)
        pure = BlendedInverseVarianceWeights(; p = 0.5, lambda = 1.0)
        Wiv = PortfolioOptimisers.cs_weights_refine(pure, W0, EPS_B, SimpleVariance(),
                                                    MASK_B)
        for lambda in (0.0, 0.25, 0.5, 0.75, 1.0)
            alg = BlendedInverseVarianceWeights(; p = 0.5, lambda = lambda)
            W1 = PortfolioOptimisers.cs_weights_refine(alg, W0, EPS_B, SimpleVariance(),
                                                       MASK_B)
            # Every eligible observation sums to one, so the two components enter in the
            # proportion the caller wrote.
            @test all(x -> isapprox(x, 1.0), sum(W1; dims = 2))
            @test W1 ≈ lambda .* Wiv .+ (1 - lambda) .* cap
        end
    end

    @testset "An observation with no eligible asset carries no weight" begin
        mask = [true true true
                false false false
                true true true
                true true true]
        alg = BlendedInverseVarianceWeights(; p = 0.5, lambda = 0.5)
        W0 = PortfolioOptimisers.cs_weights_initial(alg, MCAP_A, mask)
        @test all(iszero, W0[2, :])
        W1 = PortfolioOptimisers.cs_weights_refine(alg, W0, EPS_A, SimpleVariance(), mask)
        @test all(iszero, W1[2, :])
        @test all(isfinite, W1)
        @test isapprox(sum(W1[4, :]), 1.0)
    end

    @testset "A zero variance leaves at the median cap" begin
        # The second asset's residual never moves, so its estimated variance is zero and its
        # inverse is infinite. The median cap is what answers it.
        eps = [0.01 0.05 0.03
               0.02 0.05 -0.01
               -0.03 0.05 0.01
               0.04 0.05 0.02]
        alg = BlendedInverseVarianceWeights(; p = 0.0, lambda = 1.0, ratio = 20.0)
        W0 = PortfolioOptimisers.cs_weights_initial(alg, nothing, trues(4, 3))
        W1 = PortfolioOptimisers.cs_weights_refine(alg, W0, eps, SimpleVariance(),
                                                   trues(4, 3))
        @test all(isfinite, W1)
        @test all(x -> isapprox(x, 1.0), sum(W1; dims = 2))
        # The capped asset carries exactly `ratio` times the median of the row.
        for t in 3:4
            @test W1[t, 2] ≈ 20.0 * median(W1[t, :])
        end
    end

    @testset "The refine step refuses a malformed design" begin
        alg = BlendedInverseVarianceWeights(; lambda = 0.5)
        @test_throws DimensionMismatch PortfolioOptimisers.cs_weights_refine(alg,
                                                                             ones(4, 3),
                                                                             EPS_B,
                                                                             SimpleVariance(),
                                                                             MASK_A)
        @test_throws PortfolioOptimisers.IsNonFiniteError PortfolioOptimisers.cs_weights_refine(alg,
                                                                                                fill(NaN,
                                                                                                     4,
                                                                                                     3),
                                                                                                EPS_A,
                                                                                                SimpleVariance(),
                                                                                                MASK_A)
        @test_throws DomainError PortfolioOptimisers.cs_weights_refine(alg, -ones(4, 3),
                                                                       EPS_A,
                                                                       SimpleVariance(),
                                                                       MASK_A)
    end
end

#=
Check `src/05_Moments/32_CrossSectionalFactorModel/03_CrossSectionalFactorModel.jl` against the contract its docstrings
state, and against the stored oracle of map #643. Issue #706.

THREE FACTS SHAPE THE PROBES.

1. `csfm.L` NEVER READS `nothing`. The `swap(L, M)` rule answers `M` when the field is unset,
   so a probe of the unset case must read `getfield(csfm, :L)`. `An unset L reads back as M`
   pins both halves: the property answers `M`, and the field is still unset. `A view keeps an
   unset L unset` pins the same rule through `port_opt_view`, which is where a property read
   would silently materialise `L` as a copy of `M`.

2. THE ASSET AXIS IS NOT ONE AXIS. A loadings matrix holds one row per asset, a per-asset
   history one column per asset, an exposure history one column per asset within a slice, and
   a full idiosyncratic covariance one of each. `port_opt_view cuts every asset axis` measures
   all four against a hand-written slice, and it reorders the assets so that a probe cannot
   pass on a shape alone.

3. THE ORACLE'S OWN SELECTION TESTS SET THE RULES. They state which field
   moves under an asset selection and which passes through, that a full idiosyncratic
   covariance is cut on both axes while a diagonal one is cut once, and that a model may drop
   its exposure history, its idiosyncratic returns, its idiosyncratic variance history and its
   benchmark weights and stay a model. `A slim model keeps its loadings and drops its
   histories` is that last case.

`fcb` carries the family re-basis, and issue #724 built the [`FactorFamilyBasis`](@ref) it holds.
This file probes only the rule this result states: `L` and `fcb` are present together or
absent together.
=#
using Dates

@testset "CrossSectionalFactorModel" begin
    # One fixture, written out rather than generated: 4 observations, 3 assets, 2 factors.
    # `Ms[end, :, :] == M`, as the exposure history of a fitted model does.
    M = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    b = [0.1, 0.2, 0.3]
    Ms = zeros(4, 3, 2)
    Ms[1, :, :] = [0.5 1.0; 1.5 2.0; 2.5 3.0]
    Ms[2, :, :] = [0.6 1.1; 1.6 2.1; 2.6 3.1]
    Ms[3, :, :] = [0.7 1.2; 1.7 2.2; 2.7 3.2]
    Ms[4, :, :] = M
    csr = CrossSectionalRegression(; f = [0.01 0.02; 0.03 0.04; 0.05 0.06; 0.07 0.08],
                                   eps = [0.11 0.12 0.13; 0.14 0.15 0.16; 0.17 0.18 0.19;
                                          0.21 0.22 0.23], n = [3, 3, 3, 3],
                                   b = [0.001, 0.002, 0.003, 0.004])
    vs = [0.31 0.32 0.33; 0.34 0.35 0.36; 0.37 0.38 0.39; 0.41 0.42 0.43]
    rw = [0.51 0.52 0.53; 0.54 0.55 0.56; 0.57 0.58 0.59; 0.61 0.62 0.63]
    bw = [0.71 0.72 0.73; 0.74 0.75 0.76; 0.77 0.78 0.79; 0.81 0.82 0.83]
    esigma_diag = [0.91, 0.92, 0.93]
    esigma_full = [0.91 0.01 0.02; 0.01 0.92 0.03; 0.02 0.03 0.93]
    fam = ["style", "sector"]
    # A family re-basis drops one factor of a constrained family, so `L` is narrower than `M`.
    L = reshape([1.0, 3.0, 5.0], 3, 1)
    # The family holds both raw factors and drops the second, so the reduced axis is one
    # factor wide and every ratio is one.
    fcb = FactorFamilyBasis(; fnm = ["f"], fi = [[1, 2]], di = [2], ratios = ones(4, 1),
                            K = 2)

    full_model(; kwargs...) = CrossSectionalFactorModel(; M = M, b = b, csr = csr, Ms = Ms,
                                                        vs = vs, esigma = esigma_diag,
                                                        rw = rw, bw = bw, fam = fam,
                                                        lag = 1, kwargs...)

    @testset "An unset L reads back as M" begin
        csfm = CrossSectionalFactorModel(; M = M, b = b)
        # The property answers `M`, so no consumer of the loadings needs a `Nothing` branch.
        @test csfm.L === csfm.M
        @test !isnothing(csfm.L)
        # The field is still unset, which is what the rule exists to express.
        @test isnothing(getfield(csfm, :L))
        @test isnothing(csfm.fcb)
        @test :L in propertynames(csfm)
        # Every optional field after `b` reads back as `nothing` when it was not given.
        @test all(isnothing,
                  (csfm.csr, csfm.Ms, csfm.vs, csfm.esigma, csfm.rw, csfm.bw, csfm.fam,
                   csfm.fcb, csfm.lag, csfm.rf, csfm.fx, csfm.fr))
    end

    @testset "A re-basis makes L narrower than M" begin
        csfm = CrossSectionalFactorModel(; M = M, L = L, b = b, fcb = fcb)
        @test csfm.L === L
        @test getfield(csfm, :L) === L
        @test size(csfm.L, 2) < size(csfm.M, 2)
        @test csfm.fcb === fcb
    end

    @testset "The constructor guards every axis" begin
        @test_throws PortfolioOptimisers.IsEmptyError CrossSectionalFactorModel(;
                                                                                M = Matrix{Float64}(undef,
                                                                                                    0,
                                                                                                    0),
                                                                                b = b)
        @test_throws PortfolioOptimisers.IsEmptyError CrossSectionalFactorModel(; M = M,
                                                                                b = Float64[])
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = [0.1, 0.2])
        # `L` and `fcb` are one decision, so either alone is refused.
        @test_throws ArgumentError CrossSectionalFactorModel(; M = M, L = L, b = b)
        @test_throws ArgumentError CrossSectionalFactorModel(; M = M, b = b, fcb = fcb)
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M,
                                                                 L = reshape([1.0, 3.0], 2,
                                                                             1), b = b,
                                                                 fcb = fcb)
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b,
                                                                 fam = ["style"])
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b,
                                                                 Ms = zeros(4, 2, 2))
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b,
                                                                 Ms = zeros(4, 3, 3))
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b,
                                                                 csr = CrossSectionalRegression(;
                                                                                                f = csr.f,
                                                                                                eps = vs[:,
                                                                                                         1:2],
                                                                                                n = csr.n))
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b,
                                                                 vs = vs[:, 1:2])
        # The three per-asset histories agree on the observation axis.
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b, rw = rw,
                                                                 bw = bw[1:3, :])
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b, rw = rw,
                                                                 vs = vs[1:3, :])
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b, bw = bw,
                                                                 vs = vs[1:3, :])
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b,
                                                                 esigma = [0.91, 0.92])
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b,
                                                                 esigma = esigma_full[1:2,
                                                                                      :])
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b,
                                                                 esigma = esigma_full[1:2,
                                                                                      1:2])
        @test_throws DomainError CrossSectionalFactorModel(; M = M, b = b, lag = -1)
        # `fr` is the raw-axis history of a re-based fit: it needs the re-basis and the fit,
        # and it has the rows of the fit and the columns of `M` (#1422).
        fr = zeros(size(csr.f, 1), size(M, 2))
        @test CrossSectionalFactorModel(; M = M, L = L, b = b, csr = csr, fcb = fcb,
                                        fr = fr).fr === fr
        @test_throws ArgumentError CrossSectionalFactorModel(; M = M, b = b, csr = csr,
                                                             fr = fr)
        @test_throws ArgumentError CrossSectionalFactorModel(; M = M, L = L, b = b,
                                                             fcb = fcb, fr = fr)
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, L = L, b = b,
                                                                 csr = csr, fcb = fcb,
                                                                 fr = fr[1:3, :])
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, L = L, b = b,
                                                                 csr = csr, fcb = fcb,
                                                                 fr = fr[:, 1:1])
        @test_throws PortfolioOptimisers.IsEmptyError CrossSectionalFactorModel(; M = M,
                                                                                b = b,
                                                                                vs = Matrix{Float64}(undef,
                                                                                                     0,
                                                                                                     3))
        @test_throws PortfolioOptimisers.IsEmptyError CrossSectionalFactorModel(; M = M,
                                                                                b = b,
                                                                                esigma = Float64[])
    end

    @testset "Both shapes of esigma" begin
        diag_model = full_model()
        full_esigma_model = full_model(; esigma = esigma_full)
        @test diag_model.esigma === esigma_diag
        @test full_esigma_model.esigma === esigma_full
        # A diagonal covariance is cut once, and a full one on both axes. The oracle's
        # own selection tests state the pair.
        i = [3, 1]
        @test PortfolioOptimisers.port_opt_view(diag_model, i).esigma == esigma_diag[i]
        @test PortfolioOptimisers.port_opt_view(full_esigma_model, i).esigma ==
              esigma_full[i, i]
        @test size(PortfolioOptimisers.port_opt_view(full_esigma_model, i).esigma) == (2, 2)
        @test isnothing(PortfolioOptimisers.idiosyncratic_covariance_view(nothing, i))
    end

    @testset "port_opt_view cuts every asset axis" begin
        csfm = full_model(; L = L, fcb = fcb)
        # The selection reorders the assets, so a probe cannot pass on a shape alone.
        i = [3, 1]
        v = PortfolioOptimisers.port_opt_view(csfm, i)
        @test v.M == M[i, :]
        @test v.L == L[i, :]
        @test v.b == b[i]
        @test v.Ms == Ms[:, i, :]
        @test v.vs == vs[:, i]
        @test v.esigma == esigma_diag[i]
        @test v.rw == rw[:, i]
        @test v.bw == bw[:, i]
        @test v.csr.eps == csr.eps[:, i]
        # The factor axis and the observation axis follow no asset selection.
        @test v.csr.f === csr.f
        @test v.csr.n === csr.n
        @test v.csr.b === csr.b
        @test v.fam === fam
        @test v.fcb === fcb
        @test v.lag === 1
    end

    @testset "The row key names the observation of each fit row (#1493)" begin
        PO = PortfolioOptimisers
        ts = Dates.Date(2020, 1, 1) .+ Dates.Day.(1:4)
        @test isnothing(full_model().idx) && isnothing(full_model().ts)
        csfm = full_model(; idx = 3:6, ts = ts)
        @test csfm.idx == 3:6 && csfm.ts == ts
        # An asset view keeps the observation axis, and a time-cut cuts the key with it.
        v = PO.port_opt_view(csfm, [3, 1])
        @test v.idx === csfm.idx && v.ts === csfm.ts
        h = PO.forecast_history_block(csfm, 2)
        @test h.idx == 3:4 && h.ts == ts[1:2]
        @test PO.attribution_row_key(csfm) == (; idx = 3:6, ts = ts)
        # A key names the rows of the fit, so it needs one, with one entry for each row.
        @test_throws ArgumentError CrossSectionalFactorModel(; M = M, b = b, idx = 1:4)
        @test_throws ArgumentError CrossSectionalFactorModel(; M = M, b = b, ts = ts)
        @test_throws DimensionMismatch full_model(; idx = 1:3)
        @test_throws DimensionMismatch full_model(; ts = ts[1:3])
        # Positions are positive and increase strictly, and so do timestamps.
        @test_throws ArgumentError full_model(; idx = [1, 2, 2, 3])
        @test_throws ArgumentError full_model(; idx = 0:3)
        @test_throws ArgumentError full_model(; ts = reverse(ts))
    end

    @testset "The Return Forecast is cut on the asset axis, whatever member holds it" begin
        # The two members that fit nothing carry the two shapes the family states: a member
        # with no history, and a member with one. The view cuts `mu` on its one axis and
        # `hist` on its second, and the Descriptor weights follow no asset selection.
        hist = [0.01 0.02 0.03; 0.04 0.05 0.06; 0.07 0.08 0.09; 0.11 0.12 0.13]
        stated = CustomValueReturnForecastResult(; mu = [0.001, 0.002, 0.003])
        fitted = FixedWeightedReturnForecastResult(; mu = hist[end, :], hist = hist,
                                                   weights = [0.5, -0.5])
        i = [3, 1]
        for rf in (stated, fitted)
            v = PortfolioOptimisers.port_opt_view(full_model(; rf = rf), i)
            @test v.rf.mu == rf.mu[i]
            @test isnothing(rf.hist) ? isnothing(v.rf.hist) : v.rf.hist == rf.hist[:, i]
        end
        @test PortfolioOptimisers.port_opt_view(full_model(; rf = fitted), i).rf.weights ===
              fitted.weights
        # The guard reads the two fields every member answers, and never a member's own.
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b,
                                                                 rf = CustomValueReturnForecastResult(;
                                                                                                      mu = [0.1,
                                                                                                            0.2]))
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M[1:2, :],
                                                                 b = b[1:2], rf = fitted)
    end

    @testset "A view keeps an unset L unset" begin
        csfm = full_model()
        v = PortfolioOptimisers.port_opt_view(csfm, [1, 3])
        # A property read of `L` inside the view would have materialised a copy of `M`.
        @test isnothing(getfield(v, :L))
        @test v.L === v.M
        @test isnothing(v.fcb)
    end

    @testset "A slim model keeps its loadings and drops its histories" begin
        # The fields the oracle's own slim mode drops.
        csfm = CrossSectionalFactorModel(; M = M, b = b, esigma = esigma_full, rw = rw,
                                         L = L, fcb = fcb, lag = 1)
        @test all(isnothing, (csfm.csr, csfm.Ms, csfm.vs, csfm.bw))
        @test csfm.M === M
        @test csfm.esigma === esigma_full
        @test csfm.rw === rw
        v = PortfolioOptimisers.port_opt_view(csfm, [2, 3])
        @test all(isnothing, (v.csr, v.Ms, v.vs, v.bw))
        @test v.M == M[[2, 3], :]
        @test v.esigma == esigma_full[[2, 3], [2, 3]]
        @test v.rw == rw[:, [2, 3]]
    end

    @testset "The realised identity carries the intercept of the fit, not b" begin
        # `# Mathematical definition` states two equations, and the first one is measurable.
        # A fit of real numbers reproduces `x_t = b_t 1 + M_t f_t + eps_t`, where `b_t` is
        # the intercept of observation `t`. The per-asset `b` of the model is a term of the
        # expected return, so it never closes this identity. The docstring stated `b` in
        # that place until issue #715 measured the residue.
        rng = StableRNG(987654321)
        To, No, Ko = 6, 5, 2
        Z = randn(rng, To, No, Ko)
        Xr = randn(rng, To, No) ./ 20
        Wr = fill(1.0, To, No)
        fit = cross_sectional_regression(CrossSectionalLinearRegression(; intercept = true),
                                         Z, Xr, Wr)
        csfm = CrossSectionalFactorModel(; M = Z[end, :, :], b = randn(rng, No) ./ 100,
                                         csr = fit, Ms = Z)
        # The loadings are the last slice of the exposure history.
        @test csfm.M == csfm.Ms[end, :, :]
        for t in 1:To
            r = Xr[t, :] - csfm.Ms[t, :, :] * fit.f[t, :] - fit.eps[t, :]
            # The residue of the identity is one number repeated across the assets.
            @test all(x -> isapprox(x, fit.b[t]; atol = 1e-12), r)
            # It is the intercept of the fit, and not the factor-orthogonal expected return.
            @test !isapprox(r, csfm.b)
        end
    end

    @testset "The model is a loadings result the verbs already reach" begin
        csfm = full_model()
        @test isa(csfm, PortfolioOptimisers.AbstractLoadingsRegressionResult)
        @test isa(csfm, PortfolioOptimisers.RegE_Reg)
        # `regression` passes a result through, as it does for a `Regression`.
        @test regression(csfm) === csfm
        @test regression(csfm, csfm.M, csfm.M) === csfm
    end

    # Issue #776. One reader answers the idiosyncratic variances off either loadings block,
    # whatever shape that block stores them in, and it refuses rather than inventing a
    # weighting when the block carries none.
    @testset "idiosyncratic_variances reads either shape off either block" begin
        PO = PortfolioOptimisers
        # A vector of variances comes back unchanged, off both blocks.
        @test PO.idiosyncratic_variances(full_model()) == esigma_diag
        @test PO.idiosyncratic_variances(PO.Regression(; M = M, esigma = esigma_diag)) ==
              esigma_diag
        # A full covariance comes back as its diagonal, off both blocks.
        @test PO.idiosyncratic_variances(full_model(; esigma = esigma_full)) ==
              LinearAlgebra.diag(esigma_full)
        @test PO.idiosyncratic_variances(PO.Regression(; M = M, esigma = esigma_full)) ==
              LinearAlgebra.diag(esigma_full)
        # The diagonal of the full fixture is the diagonal fixture, so the two shapes agree.
        @test LinearAlgebra.diag(esigma_full) == esigma_diag
    end

    @testset "idiosyncratic_variances refuses a block that carries none, and names its filler" begin
        PO = PortfolioOptimisers
        # A `Regression` is filled by the prior that lifts the factor moments, so the message
        # names the switch that makes that prior add a residual block.
        reg_err = try
            PO.idiosyncratic_variances(PO.Regression(; M = M))
            nothing
        catch e
            e
        end
        @test isa(reg_err, PO.IsNothingError)
        @test occursin("rsd", reg_err.msg)
        @test occursin("Regression", reg_err.msg)
        # A cross-sectional model is filled by its own fit, so the message names the field.
        cs_err = try
            PO.idiosyncratic_variances(CrossSectionalFactorModel(; M = M, b = b))
            nothing
        catch e
            e
        end
        @test isa(cs_err, PO.IsNothingError)
        @test occursin("esigma", cs_err.msg)
        @test occursin("CrossSectionalFactorModel", cs_err.msg)
        # The two messages are not one message: each names its own block.
        @test !occursin("rsd", cs_err.msg)
    end
end

@testset "Observed factor returns ride in fx (#1367)" begin
    PO = PortfolioOptimisers
    rng = StableRNG(1_367)
    T, N = 5, 4
    Ms = randn(rng, T, N, 3)
    f = randn(rng, T, 2)
    fx = randn(rng, T, 1)
    csr = CrossSectionalRegression(; f = f, eps = randn(rng, T, N), n = fill(N, T))
    blk = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), csr = csr, Ms = Ms,
                                    nf = ["a", "b", "usd"], fam = ["s", "s", "currency"],
                                    fx = fx)
    @test cross_sectional_factor_returns(blk) == hcat(f, fx)
    @test cross_sectional_factor_returns(CrossSectionalFactorModel(; M = Ms[T, :, 1:2],
                                                                   b = zeros(N), csr = csr)) ===
          f
    # A view on the assets keeps the observed returns, which have no asset axis.
    @test PO.port_opt_view(blk, [1, 3]).fx === fx
    # The label of an observed factor is free; only the count is structural.
    @test CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), csr = csr, fx = fx,
                                    fam = ["s", "t", "anything"]) isa
          CrossSectionalFactorModel
    @test_throws PO.IsNothingError CrossSectionalFactorModel(; M = Ms[T, :, :],
                                                             b = zeros(N), fx = fx)
    @test_throws DimensionMismatch CrossSectionalFactorModel(; M = Ms[T, :, :],
                                                             b = zeros(N), csr = csr,
                                                             fx = fx[1:4, :])
    @test_throws DimensionMismatch CrossSectionalFactorModel(; M = Ms[T, :, :],
                                                             b = zeros(N), csr = csr,
                                                             fx = hcat(fx, fx))
    @test_throws PO.IsEmptyError CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N),
                                                           csr = csr, fx = zeros(T, 0))
    @test_throws PO.IsNothingError cross_sectional_factor_returns(CrossSectionalFactorModel(;
                                                                                            M = Ms[T,
                                                                                                   :,
                                                                                                   :],
                                                                                            b = zeros(N)))
end

#=
Each regression reader states its own source (#1699, ADR 0211, ADR 0212, ADR 0214).

`resolve_factor_regression` reads only `pr.rr` of a prior result, so a stub prior result that
holds a chosen `rr` isolates the table of ADR 0212, decision 1, from the prior pieces of the
migration. The stub lives at top level because `@testset` cannot hold a `struct`.
=#
struct SourceStubPrior{T} <: PortfolioOptimisers.AbstractPriorResult
    rr::T
end
@testset "Each regression reader states its own source (#1699)" begin
    PO = PortfolioOptimisers
    T, N, K = 60, 3, 2
    Fm = [sin(0.3 * t + k) + 0.05 * t / T for t in 1:T, k in 1:K]
    Xm = Fm * [0.5 1.0 -0.4; 0.8 -0.3 0.6] .+
         [0.01 * cos(1.7 * t + j) for t in 1:T, j in 1:N]
    Bv = [0.02 * sin(0.9 * t) for t in 1:T]
    rd = PO.ReturnsResult(; nx = ["A", "B", "C"], X = Xm, nf = ["f1", "f2"], F = Fm,
                          nb = ["BM"], B = Bv)
    rdn = PO.ReturnsResult(; nx = ["A", "B", "C"], X = Xm, nf = ["f1", "f2"], F = Fm)
    held = PO.Regression(; M = [1.0 2.0; 3.0 4.0; 5.0 6.0])
    pr_rr = SourceStubPrior(held)
    pr_none = SourceStubPrior(nothing)
    types = (PO.StepwiseRegression, PO.DimensionReductionRegression)

    @testset "The field, its default and its admitted set" begin
        for R in types
            @test isnothing(R().x_src)
            for x_src in PO.ALL_RETURNS_SOURCES
                re = R(; x_src = x_src)
                @test re.x_src === x_src
                @test PO.returns_source(re) === x_src
            end
            @test_throws Union{MethodError, TypeError} R(; x_src = true)
            @test_throws Union{MethodError, TypeError} R(; x_src = :data)
        end
        # The online pin rebuilds the estimator through its keyword constructor, so it
        # keeps the source.
        sw = PO.StepwiseRegression(; choice = PO.PinnedChoice(), x_src = PO.ExcessSource())
        @test PO.pin_regression_choice(sw, Xm, Fm).x_src === PO.ExcessSource()
        dr = PO.DimensionReductionRegression(; choice = PO.PinnedChoice(),
                                             x_src = PO.DataSource())
        @test PO.pin_regression_choice(dr, Xm, Fm).x_src === PO.DataSource()
        # A holder refuses an inner regression that states a source.
        @test isnothing(PO.assert_inner_reader(PO.StepwiseRegression(), :re, :FactorPrior))
        @test_throws ArgumentError PO.assert_inner_reader(PO.StepwiseRegression(;
                                                                                x_src = PO.DataSource()),
                                                          :re, :FactorPrior)
    end

    @testset "resolve_factor_regression reads the table of ADR 0212" begin
        for R in types
            re0 = R()
            raw = PO.regression(re0, Xm, Fm)
            excess = PO.regression(re0, Xm .- Bv, Fm)
            # A precomputed result needs no source.
            @test PO.resolve_factor_regression(held, rd, pr_none) === held
            # `nothing`: the loadings of the prior, else a fit on `rd.X`.
            @test PO.resolve_factor_regression(re0, rd, pr_rr) === held
            @test PO.resolve_factor_regression(re0, rd, pr_none).M ≈ raw.M
            @test PO.resolve_factor_regression(re0, rd).M ≈ raw.M
            # `DataSource()` and `ExcessSource()` always fit, over the loadings of a prior.
            red = R(; x_src = PO.DataSource())
            @test PO.resolve_factor_regression(red, rd, pr_rr).M ≈ raw.M
            ree = R(; x_src = PO.ExcessSource())
            @test PO.resolve_factor_regression(ree, rd, pr_rr).M ≈ excess.M
            @test PO.resolve_factor_regression(ree, rd).M ≈ excess.M
            @test !(PO.resolve_factor_regression(ree, rd).M ≈ raw.M)
            # `ExcessSource()` on data that carries no benchmark throws.
            @test_throws ArgumentError PO.resolve_factor_regression(ree, rdn)
            # `PriorSource()`: the loadings of the prior only.
            rep = R(; x_src = PO.PriorSource())
            @test PO.resolve_factor_regression(rep, rd, pr_rr) === held
            @test_throws ArgumentError PO.resolve_factor_regression(rep, rd, pr_none)
            @test_throws ArgumentError PO.resolve_factor_regression(rep, rd)
        end
        msg = try
            PO.resolve_factor_regression(PO.StepwiseRegression(; x_src = PO.PriorSource()),
                                         rd)
        catch e
            sprint(showerror, e)
        end
        @test occursin("`x_src = PriorSource()`", msg)
        @test occursin("StepwiseRegression", msg)
        @test occursin("`x_src = DataSource()`", msg)
    end

    @testset "ExcessSource drops the gap rows of a vector benchmark" begin
        Bg = copy(Bv)
        Bg[7] = NaN
        rdg = PO.ReturnsResult(; nx = ["A", "B", "C"], X = Xm, nf = ["f1", "f2"], F = Fm,
                               nb = ["BM"], B = Bg)
        keep = setdiff(1:T, 7)
        ree = PO.StepwiseRegression(; x_src = PO.ExcessSource())
        @test PO.resolve_factor_regression(ree, rdg).M ≈
              PO.regression(PO.StepwiseRegression(), Xm[keep, :] .- Bg[keep], Fm[keep, :]).M
        # A raw source reads no benchmark, so it keeps every row.
        red = PO.StepwiseRegression(; x_src = PO.DataSource())
        @test PO.resolve_factor_regression(red, rdg).M ≈
              PO.regression(PO.StepwiseRegression(), Xm, Fm).M
    end

    @testset "factor_space_regression reads the source of the space" begin
        raw = PO.regression(PO.StepwiseRegression(), Xm, Fm)
        @test PO.factor_space_regression(nothing, held, rd) === held
        @test isnothing(PO.factor_space_regression(nothing, nothing, rd))
        @test PO.factor_space_regression(held, nothing, nothing) === held
        @test PO.factor_space_regression(PO.StepwiseRegression(), held, nothing) === held
        red = PO.StepwiseRegression(; x_src = PO.DataSource())
        @test PO.factor_space_regression(red, held, rd).M ≈ raw.M
        @test_throws PO.IsNothingError PO.factor_space_regression(red, held, nothing)
        rep = PO.StepwiseRegression(; x_src = PO.PriorSource())
        @test PO.factor_space_regression(rep, held, nothing) === held
        @test_throws ArgumentError PO.factor_space_regression(rep, nothing, rd)
        # The keyword `brt` is gone from every constraint route.
        sets = PO.UniverseSets(; dict = Dict("nx" => ["A", "B", "C"], "nf" => ["f1", "f2"]),
                               tfkey = "nf")
        ece = PO.ExposureConstraintEstimator(;
                                             lce = PO.LinearConstraintEstimator(;
                                                                                val = "f1 <= 0.3"),
                                             space = PO.FactorSpace(; re = red))
        @test vec(PO.linear_constraints(ece, sets; rd = rd).ineq.A) ≈ raw.M[:, 1]
        @test_throws MethodError PO.linear_constraints(ece, sets; rd = rd, brt = true)
        @test_throws MethodError PO.linear_constraints(ece.lce, sets; brt = true)
        @test_throws MethodError PO.constraint_space_basis(ece.space, sets, nothing, rd;
                                                           brt = true)
    end

    @testset "The frame walk of a regression reader (ADR 0214, decision 4)" begin
        for R in types
            # At `nothing` a regression reader is raw only when it refits.
            @test PO.reader_frames(R(), "re", true).raw == ["re"]
            @test isempty(PO.reader_frames(R(), "re", false).raw)
            @test !PO.reader_frames(R(), "re", true).excess
            fe = PO.reader_frames(R(; x_src = PO.ExcessSource()), "re", false)
            @test fe.excess && isempty(fe.raw)
            for x_src in (PO.DataSource(), PO.PriorSource())
                f = PO.reader_frames(R(; x_src = x_src), "re", true)
                @test !f.excess && isempty(f.raw)
            end
        end
        # A constraint space walks its reader at its own path.
        lce = PO.LinearConstraintEstimator(; val = "f1 <= 0.3")
        ece(re) = PO.ExposureConstraintEstimator(; lce = lce,
                                                 space = PO.FactorSpace(; re = re))
        @test PO.reader_frames(ece(PO.StepwiseRegression()), "lcse", true).raw ==
              ["lcse.space.re"]
        @test PO.reader_frames([lce, ece(PO.StepwiseRegression())], "lcse", true).raw ==
              ["lcse[2].space.re"]
        @test PO.reader_frames(ece(PO.StepwiseRegression(; x_src = PO.ExcessSource())),
                               "lcse", true).excess
        # A precomputed basis and `re = nothing` hold no reader.
        for re in (nothing, held)
            f = PO.reader_frames(ece(re), "lcse", true)
            @test !f.excess && isempty(f.raw)
        end
    end
end

@testset "A dimension reduction regression names its column of ones (#1886)" begin
    using Test, PortfolioOptimisers, StableRNGs, StatsBase, Statistics, LinearAlgebra
    PO = PortfolioOptimisers
    S = PO.StatsAPI
    rng = StableRNG(21)
    T, K, N = 120, 4, 3
    F = 0.01 .* randn(rng, T, K)
    X = F * randn(rng, K, N) .+ 0.005 .* randn(rng, T, N) .+ 0.002
    aw = StatsBase.aweights(rand(rng, T))
    # The fitted values of the reduced fit, intercept included, equal the values that the
    # recovered loadings and intercept rebuild on the original factors.
    function gap(re, y)
        x1, Vp, mu, sigma = PO.prep_dim_red_reg(re, F)
        fitted = S.predict(S.fit(factory(re.retgt, InterceptColumn(1)), x1, y), x1)
        beta = PO._regression(re, y, mu, sigma, x1, Vp)
        return maximum(abs.(fitted .- (beta[1] .+ F * beta[2:end])))
    end
    weighted(tgt) = factory(DimensionReductionRegression(; retgt = tgt), aw)
    en = ElasticNet(; strength = FixedStrength(; gamma = 0.02))
    for re in
        (DimensionReductionRegression(; retgt = en), weighted(LinearModel()), weighted(en),
         weighted(ElasticNet()),
         weighted(ElasticNet(; strength = FixedStrength(; gamma = 0.02),
                             ve = SimpleVariance())))
        @test gap(re, X[:, 1]) < 1e-14
    end
    @test weighted(en).retgt.w === aw && weighted(en).ve.w === aw
    # The penalty shrinks the loadings and keeps the intercept unpenalised: at a strength
    # that zeroes every slope, the intercept is the weighted mean of each asset.
    big = ElasticNet(; strength = FixedStrength(; gamma = 1e10))
    rb = regression(weighted(big), X, F)
    @test rb.b ≈ vec(mean(X, aw; dims = 1)) && all(iszero, rb.M)
    @test regression(DimensionReductionRegression(; retgt = big), X, F).b ≈
          vec(mean(X; dims = 1))
    @test norm(regression(DimensionReductionRegression(; retgt = en), X, F).M) <
          norm(regression(DimensionReductionRegression(), X, F).M)
    # An unpenalised target ignores the call, so the default fit does not move.
    @test factory(LinearModel(), InterceptColumn(1)) === LinearModel()
end
