#=
Check `src/08_Moments/06_SmythBrobyCovariance.jl` against the mathematics its docstrings
state, and against the source itself. Issue #455 of child map 3 (#417) under the map of maps
(#404).

Coverage was ALREADY terminal for this file -- 95 executable lines, 0 misses -- so nothing
here exists to reach a line. Every test below pins a CLAIM, computed a second way. #404's
condition 2 is "checked with real numbers. Not read -- run".

FOUR FACTS SHAPE THE PROBES.

1. THE SOURCE DEFINES EXACTLY ONE STATISTIC, and it is `SmythBroby1`. Its equation (5) is
   `sum(delta_C - delta_D) / sum(delta_C + delta_D + delta_N)`. The trailing `0` and `2` are
   the library's own reductions, on the `Gerber0` and `Gerber2` shape. The `SmythBrobyGerber*`
   prefix reads one SENTENCE of the source's conclusion -- "maintaining a count in parallel
   with the sum and then scaling the total sum contribution to match the total sum count" --
   which carries no equation. The `SmythBrobyCount*` prefix is nowhere in the source; the
   source's whole argument is that a contribution sum beats a count. So `sb_reference` below
   is written FROM the source's equations (1)-(5) and its three zones, and the count and
   Gerber prefixes are checked against the library's own stated formulas instead.

2. THE TWO ZONES READ DIFFERENT RETURNS. The confusion zone compares the RAW, UNCENTRED
   return against `c1 * sigma` and rejects only when BOTH assets fall inside it. The
   indecision zone and the outer cut-off compare the CENTRED, STANDARDISED return. That mix
   is the source's own, and `The gate seam` testset shows it is not cosmetic.

3. A MARKER NAMES TWO INDEPENDENT CHOICES. The prefix picks the score triple through
   `sb_pair_scores`; the trailing digit picks the denominator through `comovement_ratio`,
   and it means the same thing as it does in the Gerber family. Nine markers is the shape
   where an unreachable branch hides, so all nine are separated pairwise.

4. `n` IS NON-NEGATIVE, AND `Inf` IS LEGAL (ADR 0091, #496). `c1`, `c2` and `c3` pass
   through `assert_nonempty_nonneg_finite_val`, which also rejects `Inf`; `n` passes
   through `assert_nonneg`, which does not. The three thresholds are read on the scale of
   the data, where `Inf` admits no observation; `n` is an exponent whose infinite limit is
   a hard divergence gate. A NEGATIVE `n` INVERTED the severity penalty: `gamma^n` is `Inf`
   at the zero divergence an asset always has against itself, so the DIAGONAL was zero
   while the off-diagonal was not. That is what the guard now rejects.
=#
using Test, PortfolioOptimisers, Statistics, StatsBase, LinearAlgebra, StableRNGs

const PO = PortfolioOptimisers

#=
The source's own statistic, written from its equations rather than from the library.

Equations (1)-(4): `kappa = sqrt((1+|r|)(1+|r|))`, `gamma = ||r_i| - |r_j||`,
`delta = kappa / (1 + gamma^n)`, with `n = 2` the source's default.

The three zones, in the source's words: an observation is eliminated when
`|r_i| < c1 s_i AND |r_j| < c1 s_j` (confusion, on the RAW return); a co-movement in the
indecision zone `|rt_i|, |rt_j| < c2` "is not used in the decision-making process"; and data
"outside the square formed by |rt_i|, |rt_j| < 4" are ignored.

`family` and `variant` are the library's two independent choices, so the same reference
serves all nine markers.
=#
function sb_reference(X, mu, sd; c1 = 0.5, c2 = 0.5, c3 = 4, n = 2, family = :delta,
                      variant = 1)
    T, N = size(X)
    H = zeros(N, N)
    for i in 1:N, j in 1:N
        p = q = u = 0.0
        cp = cq = cu = 0
        for t in 1:T
            xi, xj = X[t, i], X[t, j]
            # The confusion zone: raw, uncentred, and AND across both assets.
            if abs(xi) < c1 * sd[i] && abs(xj) < c1 * sd[j]
                continue
            end
            ri = (xi - mu[i]) / sd[i]
            rj = (xj - mu[j]) / sd[j]
            ari, arj = abs(ri), abs(rj)
            # The outer cut-off.
            if ari > c3 || arj > c3
                continue
            end
            # The indecision zone.
            if ari < c2 && arj < c2
                continue
            end
            d = sqrt((1 + ari) * (1 + arj)) / (1 + abs(ari - arj)^n)
            if ari >= c2 && arj >= c2 && ri * rj > 0
                p += d
                cp += 1
            elseif ari >= c2 && arj >= c2 && ri * rj < 0
                q += d
                cq += 1
            else
                u += d
                cu += 1
            end
        end
        P, Q, U = if family === :delta
            (p, q, u)
        elseif family === :gerber
            (p * cp, q * cq, u * cu)
        else
            (float(cp), float(cq), float(cu))
        end
        H[i, j] = if variant == 0
            iszero(P + Q) ? 0.0 : (P - Q) / (P + Q)
        elseif variant == 1
            iszero(P + Q + U) ? 0.0 : (P - Q) / (P + Q + U)
        else
            P - Q
        end
    end
    if variant == 2
        h = max.(sqrt.(diag(H)), sqrt(eps(Float64)))
        H = H ./ (h * h')
    end
    return H
end

@testset "The Smyth-Broby sweep of #455" begin
    @testset "sb_delta reproduces the source's equations (1) to (4)" begin
        # Both arguments are ALREADY absolute, so `gamma` is the difference of the two
        # MAGNITUDES, which is what equation (3) writes as a one-dimensional Manhattan
        # distance over magnitudes.
        for (ri, rj, n) in
            ((1.7, 0.4, 2), (0.5, 0.5, 2), (3.0, 0.25, 1), (2.5, 1.0, 3), (0.0, 0.0, 2))
            kappa = sqrt((1 + ri) * (1 + rj))
            gamma = abs(ri - rj)
            @test PO.sb_delta(ri, rj, n) == kappa / (1 + gamma^n)
        end

        # It is symmetric in its two returns, and at equal magnitudes the penalty vanishes,
        # so the contribution is exactly the amplitude.
        @test PO.sb_delta(1.7, 0.4, 2) == PO.sb_delta(0.4, 1.7, 2)
        @test PO.sb_delta(2.0, 2.0, 2) == 3.0
        @test PO.sb_delta(0.0, 0.0, 2) == 1.0

        # A pair that moves far together beats a pair that moves far apart, which is the
        # source's whole reason for replacing the vote.
        @test PO.sb_delta(3.0, 3.0, 2) > PO.sb_delta(3.0, 0.5, 2)
    end

    @testset "every one of the nine markers gives a different matrix" begin
        algs = (SmythBroby0(), SmythBroby1(), SmythBroby2(), SmythBrobyGerber0(),
                SmythBrobyGerber1(), SmythBrobyGerber2(), SmythBrobyCount0(),
                SmythBrobyCount1(), SmythBrobyCount2())
        X = randn(StableRNG(20260824), 60, 5)
        rhos = [cor(SmythBrobyCovariance(; alg = a, pdm = nothing), X) for a in algs]
        for i in 1:9, j in (i + 1):9
            @test !isapprox(rhos[i], rhos[j])
        end
        # Nine markers and one default. Every correlation diagonal is unit, whichever way
        # the marker reaches it: by a per-pair denominator, or by the whole-matrix
        # normalisation of the trailing `2`.
        for r in rhos
            @test all(isapprox.(diag(r), 1))
        end
    end

    @testset "all nine markers reproduce the source-derived reference" begin
        X = randn(StableRNG(4242), 80, 6) .* [0.5 1.0 2.0 0.8 1.5 0.3] .+
            [0.1 -0.2 0.0 0.3 -0.1 0.05]
        ce0 = SmythBrobyCovariance(; pdm = nothing)
        mu = vec(mean(ce0.me, X; dims = 1))
        sd = vec(std(ce0.ve, X; dims = 1))
        fams = (:delta => (SmythBroby0(), SmythBroby1(), SmythBroby2()),
                :gerber => (SmythBrobyGerber0(), SmythBrobyGerber1(), SmythBrobyGerber2()),
                :count => (SmythBrobyCount0(), SmythBrobyCount1(), SmythBrobyCount2()))
        for (fam, algs) in fams, (v, a) in enumerate(algs)
            got = cor(SmythBrobyCovariance(; alg = a, pdm = nothing), X)
            want = sb_reference(X, mu, sd; family = fam, variant = v - 1)
            @test got == want
        end
    end

    @testset "the count prefix recovers the classic Gerber statistic" begin
        #=
        The `SmythBrobyCount*` docstrings claim that dropping `sb_delta` "recovers a Gerber
        statistic evaluated on the Smyth-Broby admission rule rather than on the Gerber
        threshold". Switch the confusion zone off, lift the outer cut-off, and centre on
        zero, and the two admission rules become the SAME rule. The two families then agree
        BIT-FOR-BIT, which is what makes the claim a fact rather than a resemblance.
        =#
        X = randn(StableRNG(31415), 120, 5)
        X .-= mean(X; dims = 1)
        me0 = CustomValueExpectedReturns(; val = 0.0)
        t = 0.5
        for (galg, salg) in
            ((Gerber0(), SmythBrobyCount0()), (Gerber1(), SmythBrobyCount1()),
             (Gerber2(), SmythBrobyCount2()))
            g = cor(GerberCovariance(; alg = galg, t = t, pdm = nothing), X)
            s = cor(SmythBrobyCovariance(; alg = salg, c1 = 0.0, c2 = t, c3 = 1e6, me = me0,
                                         pdm = nothing), X)
            @test g == s
        end

        # The delta prefix does NOT reduce to Gerber under the same settings: it sums
        # contributions where Gerber counts votes, which is the source's own point.
        d = cor(SmythBrobyCovariance(; alg = SmythBroby1(), c1 = 0.0, c2 = t, c3 = 1e6,
                                     me = me0, pdm = nothing), X)
        @test !isapprox(d,
                        cor(GerberCovariance(; alg = Gerber1(), t = t, pdm = nothing), X))
    end

    @testset "the gate seam: the confusion zone reads the UNCENTRED return" begin
        #=
        `comovement_step` gates on the raw return and zones on the centred one. The
        docstring says that mix is the source's and that centring the gate as well moves the
        statistic. On a strongly off-centre sample it moves it by more than 5 points of
        correlation, so the seam is load-bearing rather than cosmetic.
        =#
        X = randn(StableRNG(2718), 90, 4) .+ [3.0 -2.0 0.5 0.0]
        ce = SmythBrobyCovariance(; alg = SmythBroby1(), pdm = nothing)
        mu = vec(mean(ce.me, X; dims = 1))
        sd = vec(std(ce.ve, X; dims = 1))
        as_is = PO.smythbroby(ce, X, mu, sd)
        @test as_is == sb_reference(X, mu, sd)

        # The same reference with the gate CENTRED instead: `mu` is subtracted before the
        # `c1` comparison. Everything else is identical.
        centred = sb_reference(X, zeros(length(mu)), sd; c1 = 0.5)
        centred_full = let c1 = 0.5, c2 = 0.5, c3 = 4, n = 2
            T, N = size(X)
            H = zeros(N, N)
            for i in 1:N, j in 1:N
                p = q = u = 0.0
                for t in 1:T
                    ri = (X[t, i] - mu[i]) / sd[i]
                    rj = (X[t, j] - mu[j]) / sd[j]
                    (abs(ri) < c1 && abs(rj) < c1) && continue
                    ari, arj = abs(ri), abs(rj)
                    (ari > c3 || arj > c3) && continue
                    (ari < c2 && arj < c2) && continue
                    d = sqrt((1 + ari) * (1 + arj)) / (1 + abs(ari - arj)^n)
                    if ari >= c2 && arj >= c2 && ri * rj > 0
                        p += d
                    elseif ari >= c2 && arj >= c2 && ri * rj < 0
                        q += d
                    else
                        u += d
                    end
                end
                H[i, j] = iszero(p + q + u) ? 0.0 : (p - q) / (p + q + u)
            end
            H
        end
        @test !isapprox(as_is, centred_full)
        @test maximum(abs, as_is .- centred_full) > 0.05
        @test centred isa Matrix   # the third reading exists and is a third answer again
    end

    @testset "the two mutating kernels write into rho and into nothing else" begin
        X = randn(StableRNG(99), 50, 4)
        ce = SmythBrobyCovariance(; alg = SmythBroby2(), pdm = nothing)
        mu = vec(mean(ce.me, X; dims = 1))
        sd = vec(std(ce.ve, X; dims = 1))
        Xc, muc, sdc = copy(X), copy(mu), copy(sd)
        pol = PO.SmythBrobyKernel(ce.alg, mu, sd, ce.c1, ce.c2, ce.c3, ce.n)

        rho = Matrix{Float64}(undef, 4, 4)
        PO.gerber_comovement!(rho, ce.ex, X, pol)
        @test X == Xc
        @test mu == muc
        @test sd == sdc
        @test pol.alg === ce.alg

        raw = copy(rho)
        PO.standardise_comovement!(ce.alg, rho)
        @test X == Xc
        @test mu == muc
        @test sd == sdc
        @test rho != raw
        # The normalisation is what puts a unit diagonal on a matrix of net scores.
        @test all(isapprox.(diag(rho), 1))
        @test !all(isapprox.(diag(raw), 1))

        # The fall-through method is a no-op for every marker that is not a trailing `2`.
        for a in (SmythBroby0(), SmythBroby1(), SmythBrobyGerber1(), SmythBrobyCount1(),
                  Gerber1())
            r2 = copy(rho)
            PO.standardise_comovement!(a, r2)
            @test r2 == rho
        end
    end

    @testset "comovement_ratio guards each denominator" begin
        for (alg, want) in ((SmythBroby0(), (6.0 - 2.0) / (6.0 + 2.0)),
                            (SmythBrobyGerber0(), (6.0 - 2.0) / (6.0 + 2.0)),
                            (SmythBrobyCount0(), (6.0 - 2.0) / (6.0 + 2.0)),
                            (Gerber0(), (6.0 - 2.0) / (6.0 + 2.0)))
            @test PO.comovement_ratio(alg, 6.0, 2.0, 5.0, Float64) == want
            @test PO.comovement_ratio(alg, 0.0, 0.0, 5.0, Float64) === 0.0
        end
        for alg in (SmythBroby1(), SmythBrobyGerber1(), SmythBrobyCount1(), Gerber1())
            @test PO.comovement_ratio(alg, 6.0, 2.0, 5.0, Float64) ==
                  (6.0 - 2.0) / (6.0 + 2.0 + 5.0)
            @test PO.comovement_ratio(alg, 0.0, 0.0, 0.0, Float64) === 0.0
        end
        # The trailing `2` branch applies no denominator and reads `nn` at all.
        for alg in (SmythBroby2(), SmythBrobyGerber2(), SmythBrobyCount2(), Gerber2())
            @test PO.comovement_ratio(alg, 6.0, 2.0, 5.0, Float64) == 4.0
            @test PO.comovement_ratio(alg, 6.0, 2.0, 0.0, Float64) == 4.0
        end
    end

    @testset "sb_pair_scores selects by prefix and comovement_ratio by digit" begin
        acc = (pos = 7.0, neg = 3.0, nn = 2.0, cpos = 5, cneg = 4, cnn = 6)
        @test PO.sb_pair_scores(SmythBroby1(), acc) == (7.0, 3.0, 2.0)
        @test PO.sb_pair_scores(SmythBrobyGerber1(), acc) == (35.0, 12.0, 12.0)
        @test PO.sb_pair_scores(SmythBrobyCount1(), acc) == (5, 4, 6)

        # `sb_add_*` fills exactly the slot its name says, and mutates nothing.
        zero_acc = (pos = 0.0, neg = 0.0, nn = 0.0, cpos = 0, cneg = 0, cnn = 0)
        d = PO.sb_delta(1.5, 0.5, 2)
        @test PO.sb_add_pos(SmythBroby1(), zero_acc, 1.5, 0.5, 2) ==
              (pos = d, neg = 0.0, nn = 0.0, cpos = 0, cneg = 0, cnn = 0)
        @test PO.sb_add_pos(SmythBrobyGerber1(), zero_acc, 1.5, 0.5, 2) ==
              (pos = d, neg = 0.0, nn = 0.0, cpos = 1, cneg = 0, cnn = 0)
        @test PO.sb_add_pos(SmythBrobyCount1(), zero_acc, 1.5, 0.5, 2) ==
              (pos = 0.0, neg = 0.0, nn = 0.0, cpos = 1, cneg = 0, cnn = 0)
        @test PO.sb_add_neg(SmythBrobyGerber1(), zero_acc, 1.5, 0.5, 2) ==
              (pos = 0.0, neg = d, nn = 0.0, cpos = 0, cneg = 1, cnn = 0)
        @test PO.sb_add_neutral(SmythBrobyGerber1(), zero_acc, 1.5, 0.5, 2) ==
              (pos = 0.0, neg = 0.0, nn = d, cpos = 0, cneg = 0, cnn = 1)
        @test zero_acc == (pos = 0.0, neg = 0.0, nn = 0.0, cpos = 0, cneg = 0, cnn = 0)

        # Only a trailing `1` marker tracks a neutral observation. Every other marker takes
        # the fall-through, so its neutral score stays at zero and no reduction reads it.
        for a in (SmythBroby0(), SmythBroby2(), SmythBrobyGerber0(), SmythBrobyGerber2(),
                  SmythBrobyCount0(), SmythBrobyCount2())
            @test PO.sb_add_neutral(a, zero_acc, 1.5, 0.5, 2) === zero_acc
        end
    end

    @testset "the thresholds at their boundaries" begin
        X = randn(StableRNG(11), 60, 4)
        sdX = vec(std(SimpleVariance(), X; dims = 1))

        # A `c1` above every raw observation puts every pair in the confusion zone, so no
        # observation is admitted and every marker returns zero rather than a NaN. The
        # diagonal is a definition rather than a measurement, so `comovement_unit_diagonal!`
        # writes one onto it and the answer is the identity. #495, ADR 0093.
        big = maximum(abs, X) / minimum(sdX) + 1
        for a in (SmythBroby0(), SmythBroby1(), SmythBroby2(), SmythBrobyGerber1(),
                  SmythBrobyCount1())
            r = cor(SmythBrobyCovariance(; c1 = big, alg = a, pdm = nothing), X)
            @test isapprox(r, I(4))
            @test all(isfinite, r)
            @test all(iszero, r - Matrix(1.0I, 4, 4))
        end

        # A `c3` at the bottom of its range rejects on the outer cut-off instead, with the
        # same result and by a different arm.
        r = cor(SmythBrobyCovariance(; c2 = 0.0, c3 = 1e-12, alg = SmythBroby1(),
                                     pdm = nothing), X)
        @test isapprox(r, I(4))

        # At `c2 = 0` no observation is one-sided, so the neutral score is empty and the
        # trailing `0` and `1` markers coincide exactly.
        r0 = cor(SmythBrobyCovariance(; c1 = 0.0, c2 = 0.0, alg = SmythBroby0(),
                                      pdm = nothing), X)
        r1 = cor(SmythBrobyCovariance(; c1 = 0.0, c2 = 0.0, alg = SmythBroby1(),
                                      pdm = nothing), X)
        @test r0 == r1
    end

    @testset "the constructor guards, and the severity exponent (#496, ADR 0091)" begin
        for kw in
            ((; c1 = -1e-9), (; c2 = -1e-9), (; c3 = -1e-9), (; c1 = Inf), (; c2 = NaN),
             (; c3 = Inf))
            @test_throws DomainError SmythBrobyCovariance(; kw...)
        end
        # `c2 < c3` is strict, so equality throws too.
        @test_throws DomainError SmythBrobyCovariance(; c2 = 0.5, c3 = 0.5)
        @test_throws DomainError SmythBrobyCovariance(; c2 = 2.0, c3 = 1.0)
        @test_throws DomainError SmythBrobyCovariance(; c2 = 0.0, c3 = 0.0)

        #=
        `n` IS NON-NEGATIVE. A negative `n` inverted the severity penalty: `gamma^n` is
        `Inf` at the zero divergence of an asset against itself, so every diagonal
        contribution was zero and the result was not a correlation matrix. `assert_nonneg`
        rejects a negative `n` and `NaN`, and admits `Inf`, whose limit is a hard
        divergence gate rather than an empty statistic.
        =#
        X = randn(StableRNG(11), 60, 4)
        for n in (-2, -1e-9, NaN)
            @test_throws DomainError SmythBrobyCovariance(; n = n)
        end
        for n in (0, 0.5, 2, Inf)
            @test SmythBrobyCovariance(; n = n) isa SmythBrobyCovariance
        end
        # The kernel itself takes any `n`, so the inversion the guard rejects is still
        # visible one level down. This is the behaviour the constructor now keeps out.
        @test PO.sb_delta(1.5, 1.5, -2) == 0.0
        @test PO.sb_delta(1.5, 0.5, -2) > 0.9
        # Every `n` the guard admits gives a unit diagonal and a finite matrix.
        for n in (0, 0.5, 2, Inf)
            r = cor(SmythBrobyCovariance(; n = n, alg = SmythBroby1(), pdm = nothing), X)
            @test all(isapprox.(diag(r), 1))
            @test all(isfinite, r)
        end
        # `n = 2` is the default, and the guard moves no shipped number.
        @test isapprox(cor(SmythBrobyCovariance(; alg = SmythBroby1(), pdm = nothing), X),
                       cor(SmythBrobyCovariance(; n = 2, alg = SmythBroby1(),
                                                pdm = nothing), X))
    end

    @testset "cov is cor rescaled by sd, and the diagonals agree" begin
        X = randn(StableRNG(7), 70, 5) .* [0.4 1.2 2.5 0.9 1.1]
        for a in (SmythBroby0(), SmythBroby1(), SmythBroby2(), SmythBrobyGerber1(),
                  SmythBrobyGerber2(), SmythBrobyCount1(), SmythBrobyCount2())
            ce = SmythBrobyCovariance(; alg = a)
            rho = cor(ce, X)
            sigma = cov(ce, X)
            sd = vec(std(ce.ve, X; dims = 1))
            sd .= max.(sd, eps(eltype(sd)))
            @test isapprox(sigma, rho .* (sd * sd'))
            @test isapprox(diag(sigma), sd .^ 2)
            @test isapprox(diag(rho), ones(5))
            @test isapprox(StatsBase.cov2cor!(copy(sigma), sd), rho)
        end

        # A non-square input returns an `N x N` matrix, so no transposed index hides.
        Y = randn(StableRNG(5), 33, 3)
        @test size(cor(SmythBrobyCovariance(), Y)) == (3, 3)
        @test size(cov(SmythBrobyCovariance(), Y)) == (3, 3)
        # `dims = 2` reads the transpose and gives the same answer.
        @test isapprox(cor(SmythBrobyCovariance(), Y),
                       cor(SmythBrobyCovariance(), Y'; dims = 2))
        @test isapprox(cov(SmythBrobyCovariance(), Y),
                       cov(SmythBrobyCovariance(), Y'; dims = 2))
    end

    @testset "a raw Smyth-Broby matrix needs pdm" begin
        # The estimator carries `pdm` because the statistic is a matrix of pairwise scores
        # and is not positive definite in general. `pdm = nothing` returns the raw one.
        X = randn(StableRNG(2024), 40, 12)
        raw = cor(SmythBrobyCovariance(; alg = SmythBroby0(), pdm = nothing), X)
        @test minimum(eigvals(Symmetric(raw))) < 0
        fixed = cor(SmythBrobyCovariance(; alg = SmythBroby0()), X)
        @test minimum(eigvals(Symmetric(fixed))) >= -sqrt(eps(Float64))
        @test !isapprox(raw, fixed)
    end
end
