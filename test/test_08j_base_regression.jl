#=
Check `src/08_Moments/21_Base_Regression.jl` against the mathematics its docstrings state.
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

        # A plain weights vector passes straight through, untouched.
        pw = pweights(range(0.5, 1.5; length = T))
        m_pw = StatsAPI.fit(PO.LinearModel(; kwargs = (; weights = pw)), Xd, y)
        @test GLM.coef(m_pw) ≈ GLM.coef(m_dyn)
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
end
