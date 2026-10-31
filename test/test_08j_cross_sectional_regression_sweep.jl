const PO = PortfolioOptimisers

# The sweep of `src/05_Moments/32_CrossSectionalFactorModel/01_CrossSectionalRegression.jl`.
# Each testset states one formula of the file with an oracle that the library does not share:
# the normal equations of the weighted least squares with an explicit column of ones, the
# diagonal of the hat matrix, the weighted Gram matrix of the change of basis, the rank of
# `LinearAlgebra`, and the weighted coefficient of determination written out.
# `test_08j_base_regression.jl` holds the pseudo-inverse oracle per observation, the solve
# algorithms and the refusals, so this file does not repeat them.

const CSRS_Z, CSRS_X, CSRS_W = let rng = StableRNG(694)
    T, N, K = 5, 30, 3
    Z = randn(rng, T, N, K)
    X = randn(rng, T, N)
    W = 0.5 .+ rand(rng, T, N)
    W[2, 1:4] .= 0.0
    Z, X, W
end

# The weighted least squares of one observation, solved through its normal equations. The
# column of ones is in the design when `intercept` is true, so the oracle minimises over the
# intercept and the factor returns together, and holds the intercept at zero otherwise.
function csrs_normal_equations(Z, X, W, t; intercept)
    idx = findall(W[t, :] .> 0)
    G = Z[t, idx, :]
    if intercept
        G = hcat(ones(length(idx)), G)
    end
    Wt = Diagonal(W[t, idx])
    beta = (G' * Wt * G) \ (G' * Wt * X[t, idx])
    return intercept ? (beta[1], beta[2:end]) : (0.0, beta)
end

@testset "Cross-sectional regression: the sweep (#694)" begin
    Z, X, W = CSRS_Z, CSRS_X, CSRS_W
    T, N, K = size(Z)

    @testset "The fit minimises over the intercept and the factor returns together" begin
        for intercept in (false, true),
            cre in (CrossSectionalLinearRegression(; intercept = intercept),
                    CrossSectionalTargetRegression(; intercept = intercept))

            csr = cross_sectional_regression(cre, Z, X, W)
            for t in 1:T
                b, f = csrs_normal_equations(Z, X, W, t; intercept = intercept)
                @test csr.f[t, :]≈f atol=1e-12
                if intercept
                    @test csr.b[t]≈b atol=1e-12
                end
                # The residual is the return less the intercept and the systematic part.
                @test csr.eps[t, :] ≈ X[t, :] .- b .- Z[t, :, :] * f
            end
            @test intercept ? !isnothing(csr.b) : isnothing(csr.b)
            # Observation 2 lost the four assets of zero weight.
            @test csr.n == [N, N - 4, N, N, N]
        end
        # A small move of the intercept or of the factor returns raises the objective.
        csr = cross_sectional_regression(CrossSectionalLinearRegression(; intercept = true),
                                         Z, X, W)
        obj(b, f) = sum(W[1, i] * (X[1, i] - b - dot(Z[1, i, :], f))^2 for i in 1:N)
        o0 = obj(csr.b[1], csr.f[1, :])
        rng = StableRNG(6941)
        @test all(obj(csr.b[1] + 1e-3 * randn(rng), csr.f[1, :] + 1e-3 * randn(rng, K)) >
                  o0 for _ in 1:100)
    end

    @testset "The leverage mark is a diagonal entry of one of the hat matrix" begin
        # Asset 4 is the only member of the second level of a one-hot family.
        A = zeros(6, 2)
        A[:, 1] .= 1.0
        A[4, 2] = 1.0
        for w in ([1.0, 2.0, 1.0, 3.0, 1.0, 0.5], ones(6))
            D = A .* sqrt.(w)
            h = diag(D * pinv(D))
            @test h[4] ≈ 1
            @test all(<(1 - 1e-3), h[[1, 2, 3, 5, 6]])
            # A positive weight moves no row into or out of the span of the others.
            @test PO.cross_sectional_leverage_one(A, w, false) == (h .≈ 1)
        end
        # The intercept puts the column `sqrt.(w)` in front of the design.
        w = [1.0, 2.0, 1.0, 3.0, 1.0, 0.5]
        Di = hcat(sqrt.(w), A[:, 2:2] .* sqrt.(w))
        @test PO.cross_sectional_leverage_one(A[:, 2:2], w, true) ==
              (diag(Di * pinv(Di)) .≈ 1)
        # A random design has no row of leverage one.
        rng = StableRNG(6942)
        Ar = randn(rng, 12, 3)
        wr = 0.2 .+ rand(rng, 12)
        Dr = Ar .* sqrt.(wr)
        @test maximum(diag(Dr * pinv(Dr))) < 0.9
        @test !any(PO.cross_sectional_leverage_one(Ar, wr, false))
    end

    @testset "The change of basis of a target gives a weighted Gram matrix of one" begin
        rng = StableRNG(6943)
        A = randn(rng, 20, 4)
        w = 0.2 .+ rand(rng, 20)
        D = A .* sqrt.(w)
        F = qr(D, ColumnNorm())
        M = F.P / UpperTriangular(F.R)
        @test (A * M)' * Diagonal(w) * (A * M) ≈ I
        # A least-squares target fitted on the new basis and mapped back gives the
        # coefficients of the weighted least squares.
        tgt = PO.factory(LinearModel(), PO.StatsBase.aweights(w))
        y = randn(rng, 20)
        fit = PO.cross_sectional_target_coef(tgt, A, y, D)
        @test fit.f ≈ pinv(D) * (y .* sqrt.(w))
        # The change of basis keeps the span, so the fit spends the four columns.
        @test fit.edof == 20 - 4
    end

    @testset "The residual count is the trace of the residual maker of each fit" begin
        # The oracle is `n_t - tr(H_t)` with the hat matrix written out through `pinv`, with the
        # column `sqrt.(w)` in front under an intercept. A least-squares fit is a projection,
        # so the trace is the rank of the design, and a dependent column spends nothing.
        function trace_count(Z, W, t; intercept)
            idx = findall(W[t, :] .> 0)
            D = Z[t, idx, :] .* sqrt.(W[t, idx])
            D = intercept ? hcat(sqrt.(W[t, idx]), D) : D
            return length(idx) - round(Int, tr(D * pinv(D)))
        end
        Zd = copy(Z)
        Zd[:, :, 3] = Zd[:, :, 1] .- 2 .* Zd[:, :, 2]
        for Zc in (Z, Zd), intercept in (false, true),
            alg in
            (PseudoInverseFallback(), DependentColumnDrop(), RankDeficiencyRefusal(),
             UncheckedSolve(), MinimumNormSolve()),
            mk in (a -> CrossSectionalLinearRegression(; alg = a, intercept = intercept),
                   a -> CrossSectionalTargetRegression(; alg = a, intercept = intercept),
                   a -> CrossSectionalTargetRegression(; tgt = GeneralisedLinearModel(), alg = a,
                                                       intercept = intercept))

            # The target refuses `MinimumNormSolve`, and a rank-deficient design refuses under
            # `RankDeficiencyRefusal`. Unchecked, the pivot of `GLM` drops the dependent
            # column, and the fit of `GLM` counts the rank that the pivot keeps.
            if (mk(PseudoInverseFallback()) isa CrossSectionalTargetRegression &&
                alg isa MinimumNormSolve) || (Zc === Zd && alg isa RankDeficiencyRefusal)
                continue
            end
            cre = mk(alg)
            csr = cross_sectional_regression(cre, Zc, X, W)
            @test csr.edof == [trace_count(Zc, W, t; intercept = intercept) for t in 1:T]
            @test eltype(csr.edof) == eltype(csr.f)
        end
        @test cross_sectional_regression(CrossSectionalLinearRegression(), Zd, X, W).edof ==
              [N, N - 4, N, N, N] .- 2
        # A fit of `GLM` spends the rank that its pivot keeps, with a dispersion parameter of
        # its family, as a `LinearModel` has, and with none, as a Poisson fit has.
        w1 = PO.StatsBase.aweights(W[1, :])
        for tgt in (LinearModel(), GeneralisedLinearModel(),
                    GeneralisedLinearModel(; args = (PO.Distributions.Poisson(),)))
            m = PO.StatsAPI.fit(PO.factory(tgt, w1), Zd[1, :, :],
                                round.(abs.(10 .* X[1, :])))
            @test PO.residual_count(m, Zd[1, :, :]) === N - 2
        end
        # An observation with no eligible asset spends nothing, the intercept included.
        W0 = copy(W)
        W0[3, :] .= 0.0
        csr0 = cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                         intercept = true),
                                          Z, X, W0)
        @test csr0.edof[3] == 0
        @test csr0.edof[[1, 2, 4, 5]] == [N, N - 4, N, N] .- (K + 1)
        # A penalised period spends the count that its own fit states, read off a refit of
        # the centred observation, and the intercept spends one more (#1900).
        tgt = ElasticNet(; strength = FixedStrength(; gamma = 0.05))
        for intercept in (false, true), alg in (DependentColumnDrop(), UncheckedSolve())
            cre = CrossSectionalTargetRegression(; tgt = tgt, alg = alg,
                                                 intercept = intercept)
            csr = cross_sectional_regression(cre, Z, X, W)
            for t in 1:T
                idx = findall(W[t, :] .> 0)
                w, A, y = W[t, idx], Z[t, idx, :], X[t, idx]
                if intercept
                    A = A .- (A' * w / sum(w))'
                    y = y .- dot(w, y) / sum(w)
                end
                m = PO.StatsAPI.fit(PO.factory(tgt, PO.StatsBase.aweights(w)), A, y)
                @test csr.edof[t] ≈ PO.residual_count(m, A) - intercept
                @test csr.f[t, :] ≈ PO.StatsAPI.coef(m)
                # The penalty spends less than the columns and the intercept.
                @test csr.edof[t] > length(idx) - K - intercept
            end
        end
    end

    @testset "The rank test is the one of a pivoted QR" begin
        # `LinearAlgebra.rank(::QRPivoted)` exists from Julia 1.12, and the verb writes the
        # same test out for 1.11.
        if VERSION >= v"1.12"
            rng = StableRNG(6944)
            for _ in 1:200
                m, n = rand(rng, 1:12), rand(rng, 1:6)
                B = randn(rng, m, n)
                if n > 1 && rand(rng) < 0.5
                    B[:, end] = B[:, 1] .+ 10.0^(-rand(rng, 6:17)) .* randn(rng, m)
                end
                @test PO.cross_sectional_rank(B) == rank(qr(B, ColumnNorm()))
            end
        end
        # The test keeps every column up to a condition number of about 1 / (K eps): eight
        # factors keep their rank at a condition number of 1e13, and lose one at 1e17.
        rng = StableRNG(6945)
        Kc = 8
        Q1 = Matrix(qr(randn(rng, 50, Kc)).Q)[:, 1:Kc]
        Q2 = Matrix(qr(randn(rng, Kc, Kc)).Q)
        design(c) = Q1 * Diagonal([1.0; fill(1 / c, Kc - 1)]) * Q2'
        @test PO.cross_sectional_rank(design(1e13)) == Kc
        @test PO.cross_sectional_rank(design(1e17)) < Kc
    end

    @testset "The coefficient of determination is the weighted formula" begin
        csr = cross_sectional_regression(CrossSectionalLinearRegression(), Z, X, W)
        r2 = cross_sectional_r2(csr, Z, X, W)
        for t in 1:T
            idx = findall(W[t, :] .> 0)
            w = W[t, idx]
            y = X[t, idx]
            yh = Z[t, idx, :] * csr.f[t, :]
            ybar = sum(w .* y) / sum(w)
            @test r2[t] ≈ 1 - sum(w .* (y - yh) .^ 2) / sum(w .* (y .- ybar) .^ 2)
        end
        # Pure noise gives a ratio strictly between zero and one.
        @test all(x -> 0 < x < 1, r2)
        @test mean_cross_sectional_r2(csr, Z, X, W) ≈ sum(r2) / T
    end
end
