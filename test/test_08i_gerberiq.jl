#=
Check `src/08_Moments/35_GerberIQCovariance.jl` against the mathematics its docstrings
state. Issue #457 of child map 3 (#417) under the map of maps (#404).

Coverage was one line short of terminal when this file was written: the `else zro`
fall-through of `gerber_iq_weight(::PartialGerberIQ)`, which only an exactly zero return
reaches. `The weight of an exactly zero return` below reaches it. Every other testset pins a
CLAIM, computed a second way. #404's condition 2 is "checked with real numbers. Not read --
run".

FIVE FACTS SHAPE THE PROBES.

1. THE THREE TEMPLATES FORM A LADDER, and each rung is a strict generalisation of the one
   below. `BasicGerberIQ` reads the MAGNITUDE alone: one boundary `d` and three weights
   `(1, n, n^2)`. `PartialGerberIQ` adds the SIGN: four boundaries, so a concordant pair is
   judged against `dcp`/`dcn` and a discordant one against `ddp`/`ddn`, giving ten weights.
   `FullGerberIQ` adds a SECOND boundary per sign, so each axis carries a small, a moderate
   and a large class, giving twenty-one weights. `The three templates form a ladder` shows
   each reduction on the whole plane, to exactly zero.

2. THE IQ STATISTIC REDUCES TO THE CLASSIC GERBER STATISTIC. Set `n = 1` so every region
   keeps its full weight, `d = c` so the template has no body, a flat decay, and the
   per-asset scaler that `GerberCovariance` itself uses. `Gerber0` and `Gerber1` then agree
   with `GerberCovariance` bit-for-bit, and `Gerber2` agrees to one unit in the last place.
   A degenerate case that reproduces the parent is the strongest single check this file
   admits. The ulp is the associativity of the `Gerber2` denominator: this family divides by
   `sqrt(di * dj)` and the classic one by `sqrt(di) * sqrt(dj)`. The first form is what makes
   the diagonal EXACTLY one, which fact 5 needs.

3. THE SCALER FIXES THE UNITS A CO-MOVEMENT IS JUDGED IN, and the three routes do not agree.
   `AssetVolatilityGerberIQScaler` returns `(sd_i, sd_j)`, so each asset carries its OWN
   volatility. `nothing` returns the pair MEAN on both axes. A `Function` returns whatever it
   likes. Only the first is PAIR-SEPARABLE -- asset `i`'s threshold does not move when its
   partner changes -- and fact 5 turns on that.

4. THE DECAY IS CONFIGURATION AND IS NEVER WRITTEN INTO. `regenerate_decay` returns the SAME
   object when both parameters are already numbers, and a NEW one otherwise. Its argument
   still reads `nothing` afterwards.

5. THE `Gerber2` BOUND IS A GEOMETRIC-MEAN CONDITION ON THE TEMPLATE, and it is the WHOLE
   condition. Under `Gerber2` the pair entry is the net `pos - neg`, divided by the geometric
   mean of the pair's two DIAGONAL PROJECTIONS: asset `i` measured against itself in the
   units of the pair `(i, j)`. A projection is always concordant, so exactly SIX of the
   twenty-one weights sit on the diagonal -- `n11`, `n4`, `n1`, `n2`, `n5`, `n12`, one per
   magnitude class. The other FIFTEEN each join two distinct classes, one per unordered pair,
   discordant channels included. The source's internet appendix proves that
   `n_ab^2 <= n_aa n_bb` over every such channel is NECESSARY AND SUFFICIENT for
   `|rho| <= 1`. `clamp_gerber_iq_n` enforces all fifteen, every shipped default meets them
   by construction, and the projection makes the bound hold for EVERY scaler. #494 and #500
   are the two defects this replaced, and ADR 0094 records the decision.
=#
using Test, PortfolioOptimisers, Statistics, StatsBase, LinearAlgebra, StableRNGs

const PO = PortfolioOptimisers

# The weight of one co-movement, on unit scaling, so a threshold is read in raw units.
giqw(ri, rj, kind) = PO.gerber_iq_weight(ri, rj, abs(ri), abs(rj), 1.0, 1.0, kind)

#=
The six magnitude classes of `FullGerberIQ`, and a representative return in each. A pair
`(x, x)` is concordant whatever `x` is, so `giqw(x, x, kind)` names the class's DIAGONAL
weight, and `giqw(x, y, kind)` for two classes names the weight that joins them.
=#
const FULL_CLASSES = (LP = 3.5, MP = 2.5, SP = 1.5, SN = -1.5, MN = -2.5, LN = -3.5)

# `(class a, class b, the field that joins them)`, for the fifteen mixed weights.
const FULL_MIXED = [(:n11, :n4, :n14), (:n11, :n1, :n15), (:n11, :n2, :n18),
                    (:n11, :n5, :n19), (:n11, :n12, :n13), (:n4, :n1, :n7), (:n4, :n2, :n9),
                    (:n4, :n5, :n6), (:n4, :n12, :n20), (:n1, :n2, :n3), (:n1, :n5, :n10),
                    (:n1, :n12, :n21), (:n2, :n5, :n8), (:n2, :n12, :n16),
                    (:n5, :n12, :n17)]

# The template with every one of the fifteen mixed weights lowered onto its flanking bound.
function clamp_every_mixed(k::FullGerberIQ)
    v = Dict(f => getfield(k, f) for f in fieldnames(FullGerberIQ))
    for (a, b, m) in FULL_MIXED
        v[m] = min(v[m], sqrt(v[a] * v[b]))
    end
    return FullGerberIQ(; (f => v[f] for f in fieldnames(FullGerberIQ))...)
end

# The message of the error `f` raises, or `nothing` when it raises none.
function raised_message(f)
    try
        f()
        return nothing
    catch err
        return sprint(showerror, err)
    end
end

@testset "GerberIQCovariance" begin
    rng = StableRNG(987654321)
    X = randn(rng, 60, 5) .* 0.02
    sd = vec(std(SimpleVariance(), X; dims = 1))

    @testset "The three templates form a ladder" begin
        #=
        Each shipped default carries a unit diagonal. `BasicGerberIQ` is a different
        statistic from the other two, and THE OTHER TWO AGREE. `FullGerberIQ()` sets
        `dp1 == dp2` and `dn1 == dn2`, so its moderate bands are empty and its six classes
        collapse to the four `PartialGerberIQ` carries, and `PartialGerberIQ()` sets
        `ddp == dcp` and `ddn == dcn`, so its discordant boundaries collapse onto its
        concordant ones. Every default weight is now the geometric mean of the two diagonal
        weights of its channel in BOTH templates, so the two collapse onto the same template.
        Before ADR 0094 an inconsistent `n15` was the only thing that separated them.
        =#
        rhos = map(k -> cor(GerberIQCovariance(; kind = k, pdm = nothing), X),
                   (BasicGerberIQ(), PartialGerberIQ(), FullGerberIQ()))
        for r in rhos
            @test isapprox(diag(r), ones(size(X, 2)))
        end
        @test !isapprox(rhos[1], rhos[2])
        @test !isapprox(rhos[1], rhos[3])
        @test rhos[2] == rhos[3]

        # `PartialGerberIQ` reduces to `BasicGerberIQ` when its four boundaries agree and its
        # ten weights collapse onto the magnitude triple. `FullGerberIQ` reduces to the same
        # template when both moderate bands are empty. The grid skips an exactly zero return,
        # which `BasicGerberIQ` alone gives a weight to; `The weight of an exactly zero
        # return` covers that seam.
        n, d = 0.4, 2.5
        basic = BasicGerberIQ(; d = d, n = n)
        partial = PartialGerberIQ(; dcp = d, dcn = d, ddp = d, ddn = d, n1 = n, n2 = n,
                                  n3 = n, n4 = 1.0, n5 = 1.0, n6 = 1.0, n7 = n^2, n8 = n^2,
                                  n9 = n^2, n10 = n^2)
        full = FullGerberIQ(; dp1 = d, dp2 = d, dn1 = d, dn2 = d, n1 = n, n2 = n, n3 = n,
                            n4 = 1.0, n5 = 1.0, n6 = 1.0, n7 = n^2, n8 = n^2, n9 = n^2,
                            n10 = n^2, n11 = 1.0, n12 = 1.0, n13 = 1.0, n14 = 1.0,
                            n15 = n^2, n16 = n^2, n17 = 1.0, n18 = n^2, n19 = 1.0,
                            n20 = 1.0, n21 = n^2)
        grid = range(-4, 4; length = 81)
        for x in grid, y in grid
            if iszero(x) || iszero(y)
                continue
            end
            @test giqw(x, y, partial) == giqw(x, y, basic)
            @test giqw(x, y, full) == giqw(x, y, basic)
        end
    end

    @testset "The IQ statistic reduces to the classic Gerber statistic" begin
        # `n = 1` gives every region weight one, `d = c` empties the body of the template, a
        # zero decay rate and delay give every observation weight one, and
        # `AssetVolatilityGerberIQScaler` scales each threshold by that asset's own
        # volatility. That is the classic statistic, on all three markers.
        for c in (0.5, 1.5), alg in (Gerber0(), Gerber1(), Gerber2())
            ceq = GerberIQCovariance(; c = c, kind = BasicGerberIQ(; d = c, n = 1.0),
                                     sc = AssetVolatilityGerberIQScaler(),
                                     decay = ExpGerberIQDecay(; e = 0.0, y = 0.0),
                                     alg = alg, pdm = nothing)
            cep = GerberCovariance(; t = c, alg = alg, pdm = nothing)
            #=
            `Gerber0` and `Gerber1` divide by a sum this family builds the same way, so they
            agree bit-for-bit. `Gerber2` divides by `sqrt(di * dj)` where the classic family
            divides by `sqrt(di) * sqrt(dj)`, and the two group the same three operations
            differently. The gap is one unit in the last place, and it buys the exactly
            unit diagonal that `posdef!` reads with `isone`.
            =#
            if alg isa Gerber2
                @test maximum(abs, cor(ceq, X) - cor(cep, X)) <= eps(1.0)
            else
                @test cor(ceq, X) == cor(cep, X)
            end
        end
    end

    @testset "The threshold scaler" begin
        # The estimator hands each asset its own volatility, and the equivalent function
        # agrees with it exactly.
        @test PO.gerber_iq_scaling(AssetVolatilityGerberIQScaler(), sd[1], sd[2]) ==
              (sd[1], sd[2])
        @test PO.gerber_iq_scaling((x, y) -> (x, y), sd[1], sd[2]) == (sd[1], sd[2])
        @test cor(GerberIQCovariance(; sc = AssetVolatilityGerberIQScaler(), pdm = nothing),
                  X) == cor(GerberIQCovariance(; sc = (x, y) -> (x, y), pdm = nothing), X)

        # `nothing` is the pair mean on both axes, so it is a different statistic.
        m = (sd[1] + sd[2]) / 2
        @test PO.gerber_iq_scaling(nothing, sd[1], sd[2]) == (m, m)
        @test !isapprox(cor(GerberIQCovariance(; sc = AssetVolatilityGerberIQScaler(),
                                               pdm = nothing), X),
                        cor(GerberIQCovariance(; sc = nothing, pdm = nothing), X))
    end

    @testset "The temporal decay" begin
        T = size(X, 1)
        decay = ExpGerberIQDecay(; e = 3.0, y = 0.1)
        @test all(decay(T, k) == exp(-0.1 * max(0, T - k - 3.0)) for k in 1:T)

        # A fully numeric decay resolves to itself, and the resolver never writes into its
        # argument.
        @test PO.regenerate_decay(decay, X) === decay
        unset = ExpGerberIQDecay()
        resolved = PO.regenerate_decay(unset, X)
        @test isnothing(unset.e) && isnothing(unset.y)
        @test resolved.e == round(Int, T - T / size(X, 2))
        @test resolved.y == log(2) / size(X, 2)

        # A window of one observation divides by nothing: both defaults read the number of
        # ASSETS, not the number of observations.
        X1 = reshape([0.01, -0.02, 0.015], 1, 3)
        d1 = PO.regenerate_decay(ExpGerberIQDecay(), X1)
        @test isfinite(d1.e) && isfinite(d1.y)
        @test d1(1, 1) == 1
        @test isapprox(cor(GerberIQCovariance(; pdm = nothing), X1), I(3))

        # A single asset is the other degenerate window, and it divides by nothing either.
        d1c = PO.regenerate_decay(ExpGerberIQDecay(), reshape([0.01], 1, 1))
        @test isfinite(d1c.e) && isfinite(d1c.y)
    end

    @testset "gerber_iq_eps and gerber_iq_gamma" begin
        T, N = size(X)
        # A number passes through, a function is called on `X`, and `nothing` takes the
        # source's own default. The three routes reach the same number when they are asked
        # for it.
        @test PO.gerber_iq_eps(3.0, X) == 3.0
        @test PO.gerber_iq_eps(x -> round(Int, size(x, 1) - size(x, 1) / size(x, 2)), X) ==
              PO.gerber_iq_eps(nothing, X) ==
              round(Int, T - T / N)
        @test PO.gerber_iq_gamma(0.1, X) == 0.1
        @test PO.gerber_iq_gamma(x -> log(2) / size(x, 2), X) ==
              PO.gerber_iq_gamma(nothing, X) ==
              log(2) / N
    end

    @testset "The clamp guard" begin
        # A template already on the bound is left alone, and one past it is brought back to
        # the bound exactly.
        bound = sqrt(0.2 * 0.8)
        @test PO.clamp_gerber_iq_n(PartialGerberIQ(; n1 = 0.2, n4 = 0.8, n7 = bound),
                                   Gerber2()).n7 == bound
        @test PO.clamp_gerber_iq_n(PartialGerberIQ(; n1 = 0.2, n4 = 0.8, n7 = 1.0),
                                   Gerber2()).n7 == bound
        @test PO.clamp_gerber_iq_n(FullGerberIQ(; n4 = 0.2, n11 = 0.8, n14 = 1.0),
                                   Gerber2()).n14 == bound

        # The other two markers divide by a sum of the same weights, so they cannot leave
        # `[-1, 1]` and the clamp is a no-op for them.
        for alg in (Gerber0(), Gerber1())
            @test PO.clamp_gerber_iq_n(PartialGerberIQ(; n1 = 0.2, n4 = 0.8, n7 = 1.0),
                                       alg).n7 == 1.0
        end
    end

    @testset "The c against d guard" begin
        # `c` inside the innermost boundary passes.
        @test isnothing(PO.gerber_iq_assert_c_d(0.5, BasicGerberIQ(; d = 0.5)))
        @test isnothing(PO.gerber_iq_assert_c_d(0.5, PartialGerberIQ(; dcp = 0.5)))
        @test isnothing(PO.gerber_iq_assert_c_d(0.5, FullGerberIQ(; dp1 = 0.5)))

        # Past it the raise names the boundary that failed, and both values.
        @test raised_message(() -> PO.gerber_iq_assert_c_d(0.6, BasicGerberIQ(; d = 0.5))) ==
              "DomainError with c must be <= kind.d, got c = 0.6, kind.d = 0.5:\n"
        # Each boundary of the two multi-boundary templates is checked on its own, so raise
        # each one in turn by keeping the others clear of `c`.
        for (f, kind) in
            (("dcp", PartialGerberIQ(; dcp = 1.0, dcn = 3.0, ddp = 3.0, ddn = 3.0)),
             ("dcn", PartialGerberIQ(; dcp = 3.0, dcn = 1.0, ddp = 3.0, ddn = 3.0)),
             ("ddp", PartialGerberIQ(; dcp = 3.0, dcn = 3.0, ddp = 1.0, ddn = 3.0)),
             ("ddn", PartialGerberIQ(; dcp = 3.0, dcn = 3.0, ddp = 3.0, ddn = 1.0)))
            @test raised_message(() -> PO.gerber_iq_assert_c_d(2.0, kind)) ==
                  "DomainError with c (2.0) must be <= kind.$f (1.0):\n"
        end
        # `FullGerberIQ` sorts each pair of boundaries, so `dp2 <= dp1` and `dn2 <= dn1`
        # always hold and only the inner one of each pair can raise.
        for (f, kind) in
            (("dp2", FullGerberIQ(; dp1 = 3.0, dp2 = 1.0, dn1 = 3.0, dn2 = 3.0)),
             ("dn2", FullGerberIQ(; dp1 = 3.0, dp2 = 3.0, dn1 = 3.0, dn2 = 1.0)))
            @test raised_message(() -> PO.gerber_iq_assert_c_d(2.0, kind)) ==
                  "DomainError with c (2.0) must be <= kind.$f (1.0):\n"
        end
        # The estimator runs the guard, so a wrong pair never reaches the kernel.
        @test_throws DomainError GerberIQCovariance(; c = 3.0,
                                                    kind = BasicGerberIQ(; d = 2.0))
    end

    @testset "gerber_IQ_delta" begin
        # The delta is the region weight times the decay weight, and nothing else.
        decay = ExpGerberIQDecay(; e = 1.0, y = 0.25)
        kind = BasicGerberIQ(; d = 2.0, n = 0.5)
        # One return large, one small, so the template selects `n^2`.
        @test PO.gerber_IQ_delta(3.0, -1.0, 3.0, 1.0, decay, 10, 4, 1.0, 1.0, kind) ==
              0.5^2 * exp(-0.25 * max(0, 10 - 4 - 1.0))
        # Both large, so the template selects one and the delta is the decay weight alone.
        @test PO.gerber_IQ_delta(3.0, -3.0, 3.0, 3.0, decay, 10, 4, 1.0, 1.0, kind) ==
              exp(-0.25 * max(0, 10 - 4 - 1.0))
    end

    @testset "The weight of an exactly zero return" begin
        # `PartialGerberIQ` and `FullGerberIQ` read the SIGN of a return, and zero has none,
        # so a zero return falls through every channel and carries weight zero. This is the
        # `else zro` arm of both methods.
        for kind in (PartialGerberIQ(), FullGerberIQ())
            for other in (-3.0, -1.0, 0.0, 1.0, 3.0)
                @test iszero(giqw(0.0, other, kind))
                @test iszero(giqw(other, 0.0, kind))
            end
        end
        # `BasicGerberIQ` reads the magnitude alone, so a zero return is small and carries a
        # weight. The two families differ here, and the difference is deliberate.
        @test giqw(0.0, 0.0, BasicGerberIQ(; d = 2.0, n = 0.5)) == 0.5
    end

    @testset "cor and cov" begin
        ce = GerberIQCovariance(; pdm = nothing)
        rho = cor(ce, X)
        sigma = cov(ce, X)
        @test isapprox(diag(rho), ones(size(X, 2)))
        # The covariance is the correlation rescaled by the same volatilities that scaled the
        # thresholds, so its diagonal is exactly the variance.
        @test sigma == rho .* (sd * transpose(sd))
        @test isapprox(diag(sigma), sd .^ 2)
        # `dims = 2` reads the transpose and answers the same matrix.
        @test cor(ce, transpose(X); dims = 2) == rho
        @test_throws DomainError cor(ce, X; dims = 3)
    end

    @testset "The Gerber2 flanking map" begin
        # Sentinel weights `n_k = k / 100` name every channel. Six of the twenty-one weights
        # sit on the diagonal, one per magnitude class, and the other fifteen each join two
        # distinct classes -- one per unordered pair, discordant channels included.
        full = FullGerberIQ(; dp1 = 3.0, dp2 = 2.0, dn1 = 3.0, dn2 = 2.0,
                            Dict(Symbol("n$k") => k / 100 for k in 1:21)...)
        diagonal = (LP = :n11, MP = :n4, SP = :n1, SN = :n2, MN = :n5, LN = :n12)
        for (cls, f) in pairs(diagonal)
            x = FULL_CLASSES[cls]
            @test giqw(x, x, full) == getfield(full, f)
        end
        seen = Symbol[]
        for (a, b, m) in FULL_MIXED
            ca = findfirst(==(a), diagonal)
            cb = findfirst(==(b), diagonal)
            xa, xb = FULL_CLASSES[ca], FULL_CLASSES[cb]
            @test giqw(xa, xb, full) == getfield(full, m)
            @test giqw(xb, xa, full) == getfield(full, m)
            push!(seen, m)
        end
        # The six diagonal weights and the fifteen mixed ones are the whole template.
        @test sort(vcat(collect(values(diagonal)), seen)) ==
              sort([Symbol("n$k") for k in 1:21])
    end

    @testset "The Gerber2 bound holds (#494, #500)" begin
        #=
        The source's internet appendix proves `n_ab^2 <= n_aa n_bb` NECESSARY AND SUFFICIENT
        for `|rho| <= 1` under a geometric-mean denominator, over every channel that joins
        two distinct magnitude classes. `clamp_gerber_iq_n` enforces all fifteen of them for
        `FullGerberIQ` and all six for `PartialGerberIQ`, and every shipped default is
        written as the geometric mean of the two diagonal weights of the classes its channel
        joins. So no default owes the clamp anything, and the clamp returns the template it
        was handed.
        =#
        fd = FullGerberIQ()
        dg = (n11 = fd.n11, n4 = fd.n4, n1 = fd.n1, n2 = fd.n2, n5 = fd.n5, n12 = fd.n12)
        breaks = [m
                  for (a, b, m) in FULL_MIXED
                  if getfield(fd, m) > sqrt(dg[a] * dg[b]) * (1 + 1e-12)]
        @test isempty(breaks)
        @test PO.clamp_gerber_iq_n(fd, Gerber2()) == fd
        @test PO.clamp_gerber_iq_n(PartialGerberIQ(), Gerber2()) == PartialGerberIQ()

        # Two assets, one always large positive and one always small positive, then both
        # negative. Every co-movement is concordant, so the pair entry is `n15 + n16` and the
        # two projections are built from `n11`, `n12` and `n1`, `n2`. Under the earlier
        # defaults this sample returned `1.0299`, which is #494.
        Xb = Float64[4.0 1.5; 4.0 1.5; -4.0 -1.5; -4.0 -1.5]
        kind = FullGerberIQ(; dp1 = 3.0, dp2 = 2.0, dn1 = 3.0, dn2 = 2.0)
        mk(k) = GerberIQCovariance(; c = 1.0, kind = k, alg = Gerber2(),
                                   sc = (x, y) -> (1.0, 1.0),
                                   decay = ExpGerberIQDecay(; e = 0.0, y = 0.0),
                                   pdm = nothing)
        hand = 2 * (kind.n15 + kind.n16) /
               sqrt((2 * kind.n11 + 2 * kind.n12) * (2 * kind.n1 + 2 * kind.n2))
        @test cor(mk(kind), Xb)[1, 2] == hand
        @test hand == 1
        # The clamp is what holds it there. A template that breaks the bound reaches the
        # estimator through the clamp, and comes out on the bound.
        kraw = FullGerberIQ(; dp1 = 3.0, dp2 = 2.0, dn1 = 3.0, dn2 = 2.0, n15 = 1.0,
                            n16 = 1.0)
        @test mk(kraw).kind == clamp_every_mixed(kraw)
        @test cor(mk(kraw), Xb)[1, 2] == 1

        # #494's own reproduction, on the discordant channel `n13`. Its flanking classes are
        # large positive and large negative, and BOTH sit on the diagonal, so the
        # geometric-mean rule reads for a discordant channel exactly as it does for a
        # concordant one. It returned `-10` before the clamp reached `n13`.
        Xn = Float64[4.0 -4.0; 4.0 -4.0; -4.0 4.0; -4.0 4.0]
        kbad = FullGerberIQ(; dp1 = 3.0, dp2 = 2.0, dn1 = 3.0, dn2 = 2.0, n11 = 0.1,
                            n12 = 0.1, n13 = 1.0)
        @test mk(kbad).kind.n13 == sqrt(kbad.n11 * kbad.n12)
        @test cor(mk(kbad), Xn)[1, 2] == -1

        #=
        A PAIR-DEPENDENT scaler cannot break the bound, because the denominator is read in
        the pair's own units. `min` hands the pair the SMALLER of the two volatilities, so
        the volatile asset clears a threshold off the diagonal that it never clears against
        itself alone. The projection follows it there, so the class it takes in the numerator
        is the class it takes in the denominator. This sample returned `sqrt(2)` while the
        denominator was read from the assembled diagonal, which is #500.
        =#
        Xm = Float64[10.0 1.0; 10.0 1.0; -10.0 -1.0; -10.0 -1.0]
        mks(sc) = GerberIQCovariance(; kind = PartialGerberIQ(), alg = Gerber2(), sc = sc,
                                     decay = ExpGerberIQDecay(; e = 0.0, y = 0.0),
                                     pdm = nothing)
        sep = cor(mks(AssetVolatilityGerberIQScaler()), Xm)
        dep = cor(mks((x, y) -> (min(x, y), min(x, y))), Xm)
        @test sep == dep
        @test dep[1, 2] == 1

        #=
        The diagonal is EXACTLY one, and not one to within a rounding. The pair `(i, i)`
        makes the numerator and both projections the same sum, so the ratio is `x / sqrt(x
        * x)`. `posdef!` reads the diagonal with an exact `isone` test to choose between a
        correlation matrix and a covariance matrix, so an exact one decides that branch by
        construction. #500's second finding is what this closes.
        =#
        for sc in
            (nothing, AssetVolatilityGerberIQScaler(), (x, y) -> (min(x, y), min(x, y)))
            r = cor(GerberIQCovariance(; kind = FullGerberIQ(), alg = Gerber2(), sc = sc,
                                       pdm = nothing), X)
            @test all(isone, diag(r))
            @test maximum(abs, r) <= 1
        end

        #=
        The two conditions the bound needs are the template's, and nothing else. Random
        diagonal weights, random boundaries, random data with a twenty-fold volatility
        spread, and the three scalers in turn: the clamped template never leaves `[-1, 1]`.
        =#
        rngb = StableRNG(24680)
        worst = 0.0
        for trial in 1:60
            Xr = randn(rngb, 60, 4) .* transpose(1 .+ 19 .* rand(rngb, 4))
            d2 = 0.3 + 1.2 * rand(rngb)
            k = FullGerberIQ(; dp1 = d2 + 1.5 * rand(rngb), dp2 = d2,
                             dn1 = d2 + 1.5 * rand(rngb), dn2 = d2,
                             Dict(Symbol("n$i") => rand(rngb) for i in 1:21)...)
            sc = if trial % 3 == 0
                nothing
            else
                (if trial % 3 == 1
                     (x, y) -> (min(x, y), min(x, y))
                 else
                     (x, y) -> (max(x, y), max(x, y))
                 end)
            end
            ce = GerberIQCovariance(; c = d2 * rand(rngb), kind = k, alg = Gerber2(),
                                    sc = sc,
                                    decay = ExpGerberIQDecay(; e = 0.0,
                                                             y = 0.05 * rand(rngb)),
                                    pdm = nothing)
            worst = max(worst, maximum(abs, cor(ce, Xr)))
        end
        @test worst <= 1
    end
end
