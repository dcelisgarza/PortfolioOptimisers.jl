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
    Z = permutedims(randn(rng, T, N, K), (2, 3, 1))
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
    G = Z[idx, :, t]
    if intercept
        G = hcat(ones(length(idx)), G)
    end
    Wt = Diagonal(W[t, idx])
    beta = (G' * Wt * G) \ (G' * Wt * X[t, idx])
    return intercept ? (beta[1], beta[2:end]) : (0.0, beta)
end

# A member of a caller that states only the solve of four arguments.
struct CsrsUserSolve <: PortfolioOptimisers.AbstractCrossSectionalSolveAlgorithm end
function PortfolioOptimisers.cross_sectional_solve(::CsrsUserSolve, A, y, ::Integer)
    return (; f = A \ y, edof = size(A, 1) - size(A, 2))
end
# A design of `m` rows and `k` columns whose condition number is `c`.
function csrs_conditioned(rng, m, k, c)
    U = Matrix(qr(randn(rng, m, k)).Q)[:, 1:k]
    V = Matrix(qr(randn(rng, k, k)).Q)
    return U * Diagonal(exp10.(range(0, -log10(c); length = k))) * V'
end

@testset "Cross-sectional regression: the sweep (#694)" begin
    Z, X, W = CSRS_Z, CSRS_X, CSRS_W
    N, K, T = size(Z)

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
                @test csr.eps[t, :] ≈ X[t, :] .- b .- Z[:, :, t] * f
            end
            @test intercept ? !isnothing(csr.b) : isnothing(csr.b)
            # Observation 2 lost the four assets of zero weight.
            @test csr.n == [N, N - 4, N, N, N]
        end
        # A small move of the intercept or of the factor returns raises the objective.
        csr = cross_sectional_regression(CrossSectionalLinearRegression(; intercept = true),
                                         Z, X, W)
        obj(b, f) = sum(W[1, i] * (X[1, i] - b - dot(Z[i, :, 1], f))^2 for i in 1:N)
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
        # The intercept puts the column `sqrt.(w)` in front of the design. The verb reads the
        # exposures centred on their weighted means, as the regression hands them over.
        w = [1.0, 2.0, 1.0, 3.0, 1.0, 0.5]
        Di = hcat(sqrt.(w), A[:, 2:2] .* sqrt.(w))
        Ac = A[:, 2:2] .- (w' * A[:, 2:2]) ./ sum(w)
        @test PO.cross_sectional_leverage_one(Ac, w, true) == (diag(Di * pinv(Di)) .≈ 1)
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
            D = Z[idx, :, t] .* sqrt.(W[t, idx])
            D = intercept ? hcat(sqrt.(W[t, idx]), D) : D
            return length(idx) - round(Int, tr(D * pinv(D)))
        end
        Zd = copy(Z)
        Zd[:, 3, :] = Zd[:, 1, :] .- 2 .* Zd[:, 2, :]
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
            m = PO.StatsAPI.fit(PO.factory(tgt, w1), Zd[:, :, 1],
                                round.(abs.(10 .* X[1, :])))
            @test PO.residual_count(m, Zd[:, :, 1]) === N - 2
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
                w, A, y = W[t, idx], Z[idx, :, t], X[t, idx]
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

    @testset "One factorisation gives the rank, the solve and the pivot (#1970)" begin
        # `A \ y` factorises a tall design by a pivoted QR and a square one by LU, so the
        # solve through the factorisation that the rank test took is bit-equal to it.
        rng = StableRNG(1970)
        for (m, n) in ((30, 5), (6, 6), (9, 1))
            A = randn(rng, m, n)
            y = randn(rng, m)
            F = qr(A, ColumnNorm())
            @test PO.cross_sectional_full_solve(F, A, y) == A \ y
            for alg in
                (PseudoInverseFallback(), RankDeficiencyRefusal(), DependentColumnDrop())
                @test PO.cross_sectional_solve(alg, A, y, 1).f == A \ y
            end
        end
        A = randn(rng, 20, 4)
        A[:, 4] = A[:, 1] + A[:, 2]
        y = randn(rng, 20)
        F = qr(A, ColumnNorm())
        r = PO.cross_sectional_rank(A; R = F.R)
        @test r == 3
        @test PO.cross_sectional_kept_columns(A, r; F = F) ==
              PO.cross_sectional_kept_columns(A, r)
        @test_throws ArgumentError PO.assert_cross_sectional_rank(A, 7; R = F.R)
        fit = PO.cross_sectional_solve(DependentColumnDrop(), A, y, 1)
        keep = PO.cross_sectional_kept_columns(A, r)
        @test fit.f[keep] == A[:, keep] \ y
        @test all(iszero, fit.f[setdiff(1:4, keep)])
    end

    @testset "The leverage is read off the triangular factor of the solve (#1971)" begin
        rng = StableRNG(1971)
        # A style and a one-hot family of three levels, whose second level has one member.
        lvl = [1, 1, 1, 2, 3, 3, 3, 3, 1, 3]
        A = hcat(randn(rng, 10), [lvl[i] == j ? 1.0 : 0.0 for i in 1:10, j in 1:3])
        w = 0.2 .+ rand(rng, 10)
        D = A .* sqrt.(w)
        h = diag(D * pinv(D))
        F = qr(D, ColumnNorm())
        r = PO.cross_sectional_rank(D; R = F.R)
        @test PO.cross_sectional_leverage(D, F, r)≈h atol=1e-14
        mark = PO.cross_sectional_leverage_one(A, w, false; D = D, F = F)
        @test mark == PO.cross_sectional_leverage_one(A, w, false) == (h .≈ 1)
        @test findall(mark) == [4]
        # Under an intercept, the centred design is orthogonal to `sqrt.(w)`, so the leverage
        # is the one of the centred design plus `w / sum(w)`, the hat of `[sqrt.(w) D]`.
        Ac = A .- (w' * A) ./ sum(w)
        Dc = Ac .* sqrt.(w)
        Fc = qr(Dc, ColumnNorm())
        Di = hcat(sqrt.(w), D)
        @test PO.cross_sectional_leverage(Dc, Fc, PO.cross_sectional_rank(Dc; R = Fc.R)) .+
              w ./ sum(w)≈diag(Di * pinv(Di)) atol=1e-13
        @test PO.cross_sectional_leverage_one(Ac, w, true) == (diag(Di * pinv(Di)) .≈ 1)
        # Past a condition number of eps^(-1/4), the verb forms Q from the reflectors, and
        # the leverage is the hat of the design in both branches. The oracle is the row norm
        # of the left singular vectors, because `B * pinv(B)` loses about `c * eps` at a
        # condition number of `c`.
        for c in (1e2, 1e6, 1e10)
            B = csrs_conditioned(rng, 40, 6, c)
            FB = qr(B, ColumnNorm())
            @test PO.cross_sectional_leverage(B, FB, 6)≈vec(sum(abs2, svd(B).U; dims = 2)) atol=1e-9
        end
        @test PO.cross_sectional_leverage(zeros(4, 2), qr(zeros(4, 2), ColumnNorm()), 0) ==
              zeros(4)
    end

    @testset "The buffers of a task give the bits of the allocating forms" begin
        # The loop factorises into a buffer `Q`, gathers the pivoted columns into a buffer
        # `G`, each a view of the first rows of a larger matrix as a task holds them, and
        # solves in a work vector `b` longer than the design.
        rng = StableRNG(19731)
        pif = PseudoInverseFallback()
        for T in (Float64, Float32)
            D = randn(rng, T, 9, 4)
            Q = view(fill(T(NaN), 12, 4), 1:9, :)
            G = view(fill(T(NaN), 12, 4), 1:9, :)
            F0 = qr(D, ColumnNorm())
            F1 = PO.cross_sectional_shared_qr(trues(1, 9), nothing, NormalEquationsSolve(),
                                              D, Q)
            @test F1.factors == F0.factors && F1.τ == F0.τ && F1.p == F0.p
            # With no mask, a member that reads a `QR` gets one, and one that reads none
            # gets `nothing`.
            F2 = PO.cross_sectional_shared_qr(nothing, nothing, pif, D, Q)
            @test F2.factors == F0.factors && F2.p == F0.p
            @test isnothing(PO.cross_sectional_shared_qr(nothing, nothing,
                                                         NormalEquationsSolve(), D, Q))
            r = PO.cross_sectional_rank(D; R = F0.R)
            @test PO.cross_sectional_leverage(D, F1, r; G = G) ==
                  PO.cross_sectional_leverage(D, F0, r)
            p = F0.p[1:2]
            @test PO.cross_sectional_pivot_columns(G, D, p) ==
                  PO.cross_sectional_pivot_columns(nothing, D, p) ==
                  D[:, p]
            # The work vector gives the bits of `F \ y`, tall and wide, and a work vector
            # of another element type falls back to `F \ y`.
            y = randn(rng, T, 9)
            b = fill(T(NaN), 12)
            ws = (; b = b, gram = fill(T(NaN), 4, 4), v = fill(T(NaN), 4))
            @test PO.cross_sectional_full_solve(F1, D, y, b) == F0 \ y
            @test PO.cross_sectional_solve(pif, D, y, 1, F1, ws) ==
                  PO.cross_sectional_solve(pif, D, y, 1)
            Dw, yw = randn(rng, T, 3, 4), randn(rng, T, 3)
            Fw = qr(Dw, ColumnNorm())
            @test PO.cross_sectional_full_solve(Fw, Dw, yw, b) == Fw \ yw
            @test PO.cross_sectional_full_solve(F0, D, y, zeros(BigFloat, 12)) == F0 \ y
            # The normal equations: the Gram matrix and its factor in `gram`, the right side
            # in `v`, each with the bits of `cholesky` and `C \ (A' * y)`.
            C0 = PO.cross_sectional_gram_factor(D)
            C1 = PO.cross_sectional_gram_factor(D, ws.gram)
            # The factor is the upper triangle. `cholesky` leaves the strict lower triangle of
            # its copy uninitialised and the buffer keeps the Gram matrix there, and no reader
            # of the factorisation reads it.
            @test C1.U == C0.U && C1.piv == C0.piv && C1.rank == C0.rank
            @test isnothing(PO.cross_sectional_gram_factor(zeros(T, 0, 4), ws.gram))
            @test PO.cross_sectional_normal_solve(C1, D, y, ws.v) == C0 \ (D' * y)
            nes = NormalEquationsSolve()
            @test PO.cross_sectional_solve(nes, D, y, 1, F1, ws) ==
                  PO.cross_sectional_solve(nes, D, y, 1)
            # A rank-deficient design falls back to the pseudo-inverse with the same `QR`.
            Dd = hcat(D[:, 1:3], D[:, 1])
            @test PO.cross_sectional_solve(nes, Dd, y, 1, qr(Dd, ColumnNorm()), ws) ==
                  PO.cross_sectional_solve(pif, Dd, y, 1)
            # The root drops `F` and `ws`, so a member of a caller implements four arguments.
            @test PO.cross_sectional_solve(CsrsUserSolve(), D, y, 1, F1, ws) ==
                  PO.cross_sectional_solve(CsrsUserSolve(), D, y, 1)
        end
    end

    @testset "A fit with no leverage check marks nothing and keeps every answer (#1971)" begin
        # The lone member of a level of a one-hot family has a leverage of one.
        rng = StableRNG(19712)
        Tl, Nl = 4, 12
        lvl = [i == 1 ? 2 : 1 for i in 1:Nl]
        Zl = zeros(Nl, 3, Tl)
        Zl[:, 1, :] .= permutedims(randn(rng, Tl, Nl))
        for i in 1:Nl
            Zl[i, 1 + lvl[i], :] .= 1.0
        end
        Xl = randn(rng, Tl, Nl)
        Wl = 0.5 .+ rand(rng, Tl, Nl)
        for intercept in (false, true),
            mk in (CrossSectionalLinearRegression, CrossSectionalTargetRegression)

            on = cross_sectional_regression(mk(; intercept = intercept), Zl, Xl, Wl)
            off = cross_sectional_regression(mk(; intercept = intercept, leverage = false),
                                             Zl, Xl, Wl)
            @test isnothing(off.h1)
            @test all(on.h1[:, 1])
            @test all(iszero, on.eps[on.h1])
            @test off.f == on.f
            @test off.b == on.b
            @test off.edof == on.edof
            @test off.eps[.!on.h1] == on.eps[.!on.h1]
        end
        @test !CrossSectionalLinearRegression(; leverage = false).leverage
        @test CrossSectionalTargetRegression().leverage
    end

    @testset "NormalEquationsSolve solves a well-conditioned design and no other (#1971)" begin
        rng = StableRNG(19713)
        NE, PIF = NormalEquationsSolve(), PseudoInverseFallback()
        A = randn(rng, 40, 5)
        y = randn(rng, 40)
        fit = PO.cross_sectional_solve(NE, A, y, 1)
        @test fit.f≈A \ y rtol=1e-12
        @test fit.edof == 35
        @test !isnothing(PO.cross_sectional_gram_factor(A))
        # A member that reads no QR drops the fifth argument.
        @test PO.cross_sectional_solve(NE, A, y, 1, qr(A, ColumnNorm())) == fit
        # A design whose condition number is above eps^(-1/4), a rank-deficient design and an
        # empty design take the answer of the default.
        Ad = copy(A)
        Ad[:, 5] = Ad[:, 1] .+ Ad[:, 2]
        for B in (csrs_conditioned(rng, 40, 5, 1e6), Ad, zeros(0, 5), zeros(40, 0))
            yB = randn(rng, size(B, 1))
            @test isnothing(PO.cross_sectional_gram_factor(B))
            @test PO.cross_sectional_solve(NE, B, yB, 1) ==
                  PO.cross_sectional_solve(PIF, B, yB, 1)
        end
        # The test reads the condition number of the design: 1e3 passes and 1e5 fails.
        @test !isnothing(PO.cross_sectional_gram_factor(csrs_conditioned(rng, 40, 5, 1e3)))
        @test isnothing(PO.cross_sectional_gram_factor(csrs_conditioned(rng, 40, 5, 1e5)))
        # A whole fit agrees with the default to rounding, under both members and both
        # leverage settings, and the target falls back on a rank-deficient observation.
        for intercept in (false, true), leverage in (false, true)
            ref = cross_sectional_regression(CrossSectionalLinearRegression(; intercept), Z,
                                             X, W)
            for mk in (CrossSectionalLinearRegression, CrossSectionalTargetRegression)
                csr = cross_sectional_regression(mk(; alg = NE, intercept, leverage), Z, X,
                                                 W)
                @test csr.f≈ref.f rtol=1e-10
                @test csr.edof == ref.edof
                @test isnothing(csr.h1) == !leverage
            end
        end
        Zd = copy(Z)
        Zd[:, 3, :] = Zd[:, 1, :] .+ Zd[:, 2, :]
        for mk in (CrossSectionalLinearRegression, CrossSectionalTargetRegression)
            @test cross_sectional_regression(mk(; alg = NE), Zd, X, W).f ==
                  cross_sectional_regression(mk(; alg = PIF), Zd, X, W).f
        end
    end

    @testset "A member of a caller reads no factorisation (#1971)" begin
        # The regression hands the QR of its leverage check as a fifth argument, and the
        # method of the root drops it for a member that states four.
        A = CSRS_Z[:, :, 1]
        y = CSRS_X[1, :]
        @test PO.cross_sectional_solve(CsrsUserSolve(), A, y, 1, qr(A, ColumnNorm())) ==
              PO.cross_sectional_solve(CsrsUserSolve(), A, y, 1)
        csr = cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                        alg = CsrsUserSolve()),
                                         Z, X, W)
        @test csr.f ≈ cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                    alg = UncheckedSolve()),
                                     Z, X, W).f
    end

    @testset "The coefficient of determination is the weighted formula" begin
        csr = cross_sectional_regression(CrossSectionalLinearRegression(), Z, X, W)
        r2 = cross_sectional_r2(csr, Z, X, W)
        for t in 1:T
            idx = findall(W[t, :] .> 0)
            w = W[t, idx]
            y = X[t, idx]
            yh = Z[idx, :, t] * csr.f[t, :]
            ybar = sum(w .* y) / sum(w)
            @test r2[t] ≈ 1 - sum(w .* (y - yh) .^ 2) / sum(w .* (y .- ybar) .^ 2)
        end
        # Pure noise gives a ratio strictly between zero and one.
        @test all(x -> 0 < x < 1, r2)
        @test mean_cross_sectional_r2(csr, Z, X, W) ≈ sum(r2) / T
    end
end
