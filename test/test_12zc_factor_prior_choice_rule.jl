#=
The Choice Rule on the selection regressions of a factor prior (#1472, map #1375, ADR 0193).

`StepwiseRegression` chooses a factor set per asset over every observation, and
`DimensionReductionRegression` chooses its components over every observation. On the refit route
of the online step, `BatchChoice()` chooses again at each call with no data, and
`PinnedChoice()` writes the choice of the first fit into `included` or `proj` of the regression
that the step returns. The two agree in a batch fit.

The data moves the choice on purpose: asset 1 follows factor 1 over the first 60 observations
and factor 2 after them, so the stepwise set of asset 1 over the first block differs from the
set over every observation.
=#
@testset "The Choice Rule on the selection regressions of a factor prior (#1472)" begin
    po = PortfolioOptimisers
    rng = StableRNG(1472)
    T, K, N = 200, 4, 5
    F = randn(rng, T, K) ./ 100
    X = F * randn(rng, K, N) ./ 2 .+ randn(rng, T, N) ./ 400
    X[1:60, 1] .= F[1:60, 1] .+ randn(rng, 60) ./ 400
    X[61:end, 1] .= F[61:end, 2] .+ randn(rng, T - 60) ./ 400
    edges = (0, 60, 120, 200)
    online(est; kwargs...) = po.update_online_estimator(Online(est; kwargs...))
    function stream(pe; kwargs...)
        e = online(pe; kwargs...)
        out = []
        for k in 1:3
            r = (edges[k] + 1):edges[k + 1]
            e = partial_fit!(e, view(X, r, :), view(F, r, :))
            push!(out, (; pe = e, pr = prior(e)))
        end
        return out
    end
    rows(k) = 1:edges[k + 1]
    # A read-out over a view of the buffer and a batch fit over a matrix copy of the same rows
    # differ in the last bit, at most 3e-16 relative on this data.
    same(a, b) = isapprox(a.mu, b.mu; rtol = 1e-12) &&
                 isapprox(a.sigma, b.sigma; rtol = 1e-12)
    sw = (; alg = ForwardSelection())
    swb = (; alg = BackwardElimination(), crit = :aic)
    pca = (; drtgt = PCA(; kwargs = (; maxoutdim = 2)))
    ppca = (; drtgt = PPCA(; kwargs = (; maxoutdim = 2)))

    @testset "The two rules agree in a batch fit" begin
        for (R, kw) in ((StepwiseRegression, sw), (StepwiseRegression, swb),
                        (DimensionReductionRegression, pca), (DimensionReductionRegression, ppca))
            b = R(; kw...)
            p = R(; kw..., choice = PinnedChoice())
            rb, rp = regression(b, X, F), regression(p, X, F)
            @test isequal(rb.M, rp.M) && isequal(rb.b, rp.b)
            pb = prior(FactorPrior(; re = b), X, F)
            pp = prior(FactorPrior(; re = p), X, F)
            @test isequal(pb.mu, pp.mu) && isequal(pb.sigma, pp.sigma)
        end
    end

    @testset "A pinned factor set runs no search" begin
        sets = [[2], [1, 3], nothing, Int[], [4, 1]]
        re = StepwiseRegression(; included = sets)
        rr = with_logger(NullLogger()) do
            regression(re, X, F)
        end
        for i in (1, 2, 5)
            fit = po.StatsAPI.fit(LinearModel(), [ones(T) F[:, sets[i]]], X[:, i])
            c = po.StatsAPI.coef(fit)
            @test rr.b[i] == c[1] && rr.M[i, sets[i]] == c[2:end]
            @test count(!iszero, rr.M[i, :]) == length(sets[i])
        end
        # An empty set is an intercept-only model, and it warns about no search.
        @test all(iszero, rr.M[4, :]) && rr.b[4] ≈ mean(X[:, 4])
        @test_logs regression(StepwiseRegression(; included = [Int[], [1], [1], [1], [1]]),
                              X, F)
        # The asset of `nothing` takes the search, as the batch fit does.
        b = regression(StepwiseRegression(), X, F)
        @test isequal(rr.M[3, :], b.M[3, :]) && rr.b[3] == b.b[3]
        # The rules of the field.
        @test_throws IsEmptyError StepwiseRegression(; included = Vector{Vector{Int}}())
        @test_throws DomainError StepwiseRegression(; included = [[1, 1]])
        @test_throws DomainError StepwiseRegression(; included = [[0]])
        @test_throws DimensionMismatch regression(StepwiseRegression(; included = [[1]]), X,
                                                  F)
        @test_throws DimensionMismatch regression(StepwiseRegression(;
                                                                     included = fill([5],
                                                                                     N)), X,
                                                  F)
        # A view of the regression slices the factor sets to the selected assets.
        @test po.port_opt_view(re, [2, 5]).included == [[1, 3], [4, 1]]
        @test isnothing(po.port_opt_view(StepwiseRegression(), [2]).included)
        s = sprint(show, StepwiseRegression(; choice = PinnedChoice()))
        @test occursin("choice", s) && occursin("PinnedChoice()", s)
    end

    @testset "Pinned components refit the coefficients alone" begin
        for kw in (pca, ppca)
            p = po.pin_regression_choice(DimensionReductionRegression(; kw...,
                                                                      choice = PinnedChoice()),
                                         X, F)
            @test size(p.proj) == (K, 2)
            # The components of the fit over the same rows give the batch answer.
            rb = regression(DimensionReductionRegression(; kw...), X, F)
            rp = regression(p, X, F)
            @test isapprox(rp.M, rb.M; rtol = 1e-12) && isapprox(rp.b, rb.b; rtol = 1e-12)
            @test isapprox(rp.L, rb.L; rtol = 1e-10)
            # Over other rows the pinned fit regresses on the same components.
            r = 61:200
            rr = regression(p, X[r, :], F[r, :])
            Fc = F[r, :] .- mean(F[r, :]; dims = 1)
            for i in 1:N
                c = po.StatsAPI.coef(po.StatsAPI.fit(LinearModel(),
                                                     [ones(length(r)) Fc * p.proj],
                                                     X[r, i]))
                @test isapprox(rr.M[i, :], p.proj * c[2:end]; rtol = 1e-12)
                @test isapprox(rr.L[i, :], c[2:end]; rtol = 1e-10)
            end
            # The prediction in the factors equals the prediction in the components.
            pred = rr.b' .+ F[r, :] * rr.M'
            fits = [po.StatsAPI.predict(po.StatsAPI.fit(LinearModel(),
                                                        [ones(length(r)) Fc * p.proj],
                                                        X[r, i])) for i in 1:N]
            @test maximum(abs, pred .- reduce(hcat, fits)) < 1e-14
        end
        @test_throws IsEmptyError DimensionReductionRegression(; proj = zeros(0, 0))
        @test_throws DomainError DimensionReductionRegression(; proj = [NaN 1.0])
        @test_throws DimensionMismatch regression(DimensionReductionRegression(;
                                                                               proj = ones(3,
                                                                                           2)),
                                                  X, F)
        # A pinned projection and a batch rule return unchanged.
        p = DimensionReductionRegression(; proj = ones(K, 1), choice = PinnedChoice())
        @test po.pin_regression_choice(p, X, F) === p
        d = DimensionReductionRegression()
        @test po.pin_regression_choice(d, X, F) === d
    end

    @testset "On the online step a pinned set stays and a batch set moves" begin
        pinned = StepwiseRegression(; sw..., choice = PinnedChoice())
        sb = stream(FactorPrior(; re = StepwiseRegression(; sw...)))
        sp = stream(FactorPrior(; re = pinned))
        set1 = sp[1].pe.re.included
        @test all(!isnothing, set1)
        # The pin is the set of the first fit, and later steps keep it.
        first_fit = [po._regression(pinned, X[rows(1), i], F[rows(1), :]) for i in 1:N]
        @test set1 == first_fit
        @test sp[2].pe.re.included == set1 && sp[3].pe.re.included == set1
        # A batch choice moves: asset 1 leaves factor 1 for factor 2.
        @test isnothing(sb[3].pe.re.included)
        moved = po._regression(pinned, X[:, 1], F)
        @test set1[1] == [1] && 2 in moved && moved != set1[1]
        for k in 1:3
            # Each read-out is a batch fit over the rows seen: of the search under the batch
            # rule, and of the factor sets of the first fit under the pinned rule.
            b = prior(FactorPrior(; re = StepwiseRegression(; sw...)), X[rows(k), :],
                      F[rows(k), :])
            @test same(sb[k].pr, b)
            q = prior(FactorPrior(; re = StepwiseRegression(; sw..., included = set1)),
                      X[rows(k), :], F[rows(k), :])
            @test same(sp[k].pr, q)
        end
        # At the first step the two rules give the same answer, and later they part.
        @test isequal(sp[1].pr.sigma, sb[1].pr.sigma)
        @test sp[3].pr.rr.M[1, 1] != 0 && sb[3].pr.rr.M[1, 2] != 0
        @test !isapprox(sp[3].pr.sigma, sb[3].pr.sigma; rtol = 1e-6)
    end

    @testset "On the online step pinned components stay and batch components move" begin
        for kw in (pca, ppca)
            sb = stream(FactorPrior(; re = DimensionReductionRegression(; kw...)))
            sp = stream(FactorPrior(;
                                    re = DimensionReductionRegression(; kw...,
                                                                      choice = PinnedChoice())))
            proj = sp[1].pe.re.proj
            @test sp[3].pe.re.proj === proj && isnothing(sb[3].pe.re.proj)
            first_fit = po.pin_regression_choice(DimensionReductionRegression(; kw...,
                                                                              choice = PinnedChoice()),
                                                 X[rows(1), :], F[rows(1), :]).proj
            @test isapprox(proj, first_fit; rtol = 1e-12)
            last_fit = po.pin_regression_choice(DimensionReductionRegression(; kw...,
                                                                             choice = PinnedChoice()),
                                                X, F).proj
            @test !isapprox(last_fit, proj; rtol = 1e-3)
            for k in 1:3
                b = prior(FactorPrior(; re = DimensionReductionRegression(; kw...)),
                          X[rows(k), :], F[rows(k), :])
                @test same(sb[k].pr, b)
                q = prior(FactorPrior(;
                                      re = DimensionReductionRegression(; kw...,
                                                                        proj = proj)),
                          X[rows(k), :], F[rows(k), :])
                @test same(sp[k].pr, q)
            end
            @test isapprox(sp[1].pr.sigma, sb[1].pr.sigma; rtol = 1e-12)
            @test !isapprox(sp[3].pr.sigma, sb[3].pr.sigma; rtol = 1e-8)
        end
    end

    @testset "Each asset pins at the first fit that covers it" begin
        Xg = copy(X)
        Xg[1:30, 5] .= NaN
        e = online(FactorPrior(; re = StepwiseRegression(; sw..., choice = PinnedChoice()));
                   max_history = 100)
        sets = []
        for k in 1:3
            r = (edges[k] + 1):edges[k + 1]
            e = partial_fit!(e, Xg[r, :], F[r, :])
            push!(sets, e.re.included)
            @test all(isfinite, prior(e).sigma[1:4, 1:4])
        end
        # The buffer holds the gap until the cap drops it, at the third step.
        @test isnothing(sets[1][5]) && isnothing(sets[2][5])
        @test sets[3][5] == po._regression(e.re, Xg[101:200, 5], F[101:200, :])
        @test sets[3][1:4] == sets[1][1:4]
    end

    @testset "One row pins nothing, and the matrix form pins" begin
        e = online(FactorPrior(; re = StepwiseRegression(; choice = PinnedChoice())))
        e = partial_fit!(e, X[1, :], F[1, :])
        @test isnothing(e.re.included)
        for i in 2:40
            e = partial_fit!(e, X[i, :], F[i, :])
        end
        @test e.re.included ==
              [po._regression(StepwiseRegression(), X[1:2, i], F[1:2, :]) for i in 1:N]
        d = online(FactorPrior(;
                               re = DimensionReductionRegression(; choice = PinnedChoice())))
        @test isnothing(partial_fit!(d, X[1, :], F[1, :]).re.proj)
    end

    @testset "Every prior that fits a regression pins it, and so does a prior around it" begin
        fp = FactorPrior(; re = StepwiseRegression(; sw..., choice = PinnedChoice()))
        fv = BlackLittermanViews(; P = [1.0 0 0 0], Q = [0.01])
        av = BlackLittermanViews(; P = [1.0 zeros(1, N - 1)], Q = [0.01])
        pinned(e) = e.re.included
        hosts = ((FactorBlackLittermanPrior(; views = fv, re = fp.re), pinned),
                 (AugmentedBlackLittermanPrior(; a_views = av, f_views = fv, re = fp.re),
                  pinned), (HighOrderFactorPriorEstimator(; pe = fp), e -> pinned(e.pe)),
                 (BayesianBlackLittermanPrior(; views = fv, pe = fp), e -> pinned(e.pe)),
                 (EntropyPoolingPrior(; pe = fp), e -> pinned(e.pe)),
                 (OpinionPoolingPrior(; pes = [EntropyPoolingPrior(; pe = fp)]),
                  e -> pinned(e.pes[1].pe)))
        for (est, read) in hosts
            e = online(est)
            e = partial_fit!(e, X[rows(1), :], F[rows(1), :])
            @test read(e) ==
                  [po._regression(fp.re, X[rows(1), i], F[rows(1), :]) for i in 1:N]
            e = partial_fit!(e, X[61:200, :], F[61:200, :])
            @test read(e) ==
                  [po._regression(fp.re, X[rows(1), i], F[rows(1), :]) for i in 1:N]
        end
        # A forwarding prior steps the prior it embeds, which pins its own regression.
        e = partial_fit!(BlackLittermanPrior(; views = av, pe = online(fp)), X[rows(1), :],
                         F[rows(1), :])
        @test pinned(e.pe) ==
              [po._regression(fp.re, X[rows(1), i], F[rows(1), :]) for i in 1:N]
        # A host with no regression in its tree, or with no factor returns, is unchanged.
        ep = EntropyPoolingPrior()
        @test po.pin_prior_choice(ep, X, nothing) === ep
        ho = HighOrderPriorEstimator()
        @test po.pin_prior_choice(ho, X, nothing) === ho
        op = OpinionPoolingPrior(; pes = [EntropyPoolingPrior()])
        @test po.pin_prior_choice(op, X, F) === op
    end
end
