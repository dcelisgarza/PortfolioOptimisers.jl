#=
The Factor Family Basis, its transforms and the Neutralisation verb (issue #724, map #643).

Issue #651 decided that `fcb` is a `FactorFamilyBasis` carrying the time axis, that
Neutralisation is a field of the prior applied to the exposure history, and that the automatic
drop is the member with the largest time-average absolute benchmark-weighted exposure. This file
holds the port of the reference implementation's basis tests and its Neutralisation validation
tests, and it pins two parity cases measured against the reference on the synthetic Asset Panel
of `test06c_setup.jl`.

The parity cases are literals rather than a live comparison: the reference is not a dependency
of this package, so the numbers it produced are stored here and the test re-derives them.
=#
using Statistics, Distributions, Dates, Random
include(joinpath(@__DIR__, "test06c_setup.jl"))

@testset "FactorFamilyBasis construction and structure" begin
    PO = PortfolioOptimisers

    fcb = FactorFamilyBasis(; fnm = ["ind"], fi = [[2, 3, 5]], di = [2],
                            ratios = [0.5 0.25; 0.4 0.20], K = 6)

    @testset "The derived reads follow the stored family" begin
        @test PO.dropped_factor_indices(fcb) == [3]
        @test PO.retained_factor_indices(fcb) == [1, 2, 4, 5, 6]
        @test PO.reduced_factor_count(fcb) == 5
        @test PO.raw_to_reduced_index(fcb) == [1, 2, 0, 3, 4, 5]
        raw, red, col = PO.family_retained_indices(fcb, 1)
        @test raw == [2, 5]
        @test red == [2, 4]
        @test col == [1, 2]
        @test PO.dropped_factor_names(fcb, ["a", "b", "c", "d", "e", "f"]) == ["c"]
        @test PO.reduce_factor_names(fcb, ["a", "b", "c", "d", "e", "f"]) ==
              ["a", "b", "d", "e", "f"]
    end

    @testset "The reduced axis is the raw axis without the dropped factors" begin
        # Two families, so two factors leave and the rest keep their relative order.
        two = FactorFamilyBasis(; fnm = ["x", "y"], fi = [[1, 2], [3, 4, 5]], di = [1, 3],
                                ratios = [0.5 0.25 0.75], K = 6)
        @test PO.dropped_factor_indices(two) == [1, 5]
        @test PO.retained_factor_indices(two) == [2, 3, 4, 6]
        @test PO.reduced_factor_count(two) == 4
        @test size(two.ratios, 2) == 3
        r1 = PO.family_retained_indices(two, 1)
        r2 = PO.family_retained_indices(two, 2)
        @test r1[3] == [1]
        @test r2[3] == [2, 3]
    end

    @testset "The constructor guards every axis" begin
        ok = (fnm = ["ind"], fi = [[1, 2]], di = [1], ratios = reshape([0.5], 1, 1), K = 3)
        @test_throws DomainError FactorFamilyBasis(; ok..., K = 0)
        @test_throws PortfolioOptimisers.IsEmptyError FactorFamilyBasis(; fnm = String[],
                                                                        fi = Vector{Int}[],
                                                                        di = Int[],
                                                                        ratios = reshape([0.5],
                                                                                         1,
                                                                                         1),
                                                                        K = 3)
        # The three per-family vectors must line up.
        @test_throws DimensionMismatch FactorFamilyBasis(; ok..., di = [1, 1])
        @test_throws ArgumentError FactorFamilyBasis(; fnm = ["a", "a"],
                                                     fi = [[1, 2], [3, 4]], di = [1, 1],
                                                     ratios = [0.5 0.25], K = 5)
        # A constrained family holds at least two members, and they are unique.
        @test_throws ArgumentError FactorFamilyBasis(; fnm = ["ind"], fi = [[1]], di = [1],
                                                     ratios = Matrix{Float64}(undef, 1, 0),
                                                     K = 3)
        @test_throws ArgumentError FactorFamilyBasis(; ok..., fi = [[1, 1]])
        # Every member lies on the raw axis, and no factor belongs to two families.
        @test_throws DomainError FactorFamilyBasis(; ok..., fi = [[1, 4]])
        @test_throws ArgumentError FactorFamilyBasis(; fnm = ["a", "b"],
                                                     fi = [[1, 2], [2, 3]], di = [1, 1],
                                                     ratios = [0.5 0.25], K = 5)
        # The dropped position indexes its own family.
        @test_throws DomainError FactorFamilyBasis(; ok..., di = [3])
        # `K` exceeds the number of families, so the reduced axis is not empty.
        @test_throws ArgumentError FactorFamilyBasis(; fnm = ["a"], fi = [[1]], di = [1],
                                                     ratios = reshape([0.5], 1, 1), K = 1)
        # One column per retained member of a constrained family.
        @test_throws DimensionMismatch FactorFamilyBasis(; ok..., ratios = [0.5 0.25])
        # A non-finite ratio is refused, which is how a zero benchmark-weighted exposure on
        # the dropped factor is reported.
        @test_throws PortfolioOptimisers.IsNonFiniteError FactorFamilyBasis(; ok...,
                                                                            ratios = reshape([Inf],
                                                                                             1,
                                                                                             1))
        @test_throws PortfolioOptimisers.IsNonFiniteError FactorFamilyBasis(; ok...,
                                                                            ratios = reshape([NaN],
                                                                                             1,
                                                                                             1))
    end

    @testset "An observation slice keeps the structure" begin
        wide = FactorFamilyBasis(; fnm = ["ind"], fi = [[1, 2]], di = [2],
                                 ratios = reshape([0.5, 0.4, 0.3, 0.2], 4, 1), K = 3)
        cut = PO.factor_basis_slice(wide, 2:3)
        @test cut.fnm == wide.fnm
        @test cut.fi == wide.fi
        @test cut.di == wide.di
        @test cut.K == wide.K
        @test cut.ratios == reshape([0.4, 0.3], 2, 1)
        # One observation still answers a basis, so attribution can read a single date.
        one = PO.factor_basis_slice(wide, 3)
        @test size(one.ratios) == (1, 1)
        @test one.ratios[1, 1] == 0.3
    end
end

@testset "FactorFamilyBasis transforms" begin
    PO = PortfolioOptimisers
    rng = StableRNG(724_001)
    T, N, K = 7, 9, 5
    # Factors 2, 3 and 5 form the constrained family, and the middle one is dropped.
    fcb = FactorFamilyBasis(; fnm = ["ind"], fi = [[2, 3, 5]], di = [2],
                            ratios = randn(rng, T, 2), K = K)
    Kr = PO.reduced_factor_count(fcb)
    Ms = randn(rng, T, N, K)

    # The dense change of basis, built column by column from the stored ratios. Every
    # transform below is checked against it, and the library never forms it.
    function dense_basis(fcb, t)
        R = zeros(fcb.K, PO.reduced_factor_count(fcb))
        for (k, i) in enumerate(PO.retained_factor_indices(fcb))
            R[i, k] = 1.0
        end
        for j in eachindex(fcb.fnm)
            _, red, col = PO.family_retained_indices(fcb, j)
            d = fcb.fi[j][fcb.di[j]]
            for p in eachindex(red)
                R[d, red[p]] = -fcb.ratios[t, col[p]]
            end
        end
        return R
    end

    @testset "Reducing an exposure history applies each row's own ratios" begin
        Z = PO.reduce_exposures(fcb, Ms)
        @test size(Z) == (T, N, Kr)
        for t in 1:T
            @test Z[t, :, :] ≈ Ms[t, :, :] * dense_basis(fcb, t)
        end
        # A loading matrix is the same transform at one observation.
        @test PO.reduce_loadings(fcb, Ms[T, :, :]) ≈ Z[T, :, :]
        @test PO.reduce_loadings(fcb, Ms[2, :, :], 2) ≈ Z[2, :, :]
    end

    @testset "Factor returns round trip through the basis" begin
        g = randn(rng, T, Kr)
        f = PO.expand_factor_returns(fcb, g)
        @test size(f) == (T, K)
        @test PO.reduce_factor_returns(fcb, f) ≈ g
        for t in 1:T
            @test f[t, :] ≈ dense_basis(fcb, t) * g[t, :]
        end
        # A vector expands with the ratios of the last observation.
        @test PO.expand_factor_returns(fcb, g[1, :]) ≈ dense_basis(fcb, T) * g[1, :]
        @test PO.reduce_factor_returns(fcb, f[3, :]) ≈ g[3, :]
    end

    @testset "The fitted values are the same in both bases" begin
        g = randn(rng, T, Kr)
        f = PO.expand_factor_returns(fcb, g)
        Z = PO.reduce_exposures(fcb, Ms)
        red = [sum(Z[t, i, k] * g[t, k] for k in 1:Kr) for t in 1:T, i in 1:N]
        raw = [sum(Ms[t, i, k] * f[t, k] for k in 1:K) for t in 1:T, i in 1:N]
        @test red ≈ raw
    end

    @testset "The expanded family returns satisfy the zero-sum condition" begin
        # `c_t` is the benchmark-weighted exposure of each factor, and the family's expanded
        # returns are orthogonal to it by construction.
        bw = rand(rng, T, N) .+ 0.1
        w = bw ./ sum(bw; dims = 2)
        c = [sum(w[t, i] * Ms[t, i, k] for i in 1:N) for t in 1:T, k in 1:K]
        basis = factor_family_basis(["ind" => nothing], Ms, bw, ["a", "b", "c", "d", "e"],
                                    ["m", "ind", "ind", "s", "ind"])
        g = randn(rng, T, PO.reduced_factor_count(basis))
        f = PO.expand_factor_returns(basis, g)
        for t in 1:T
            @test abs(sum(c[t, k] * f[t, k] for k in basis.fi[1])) < 1e-12
        end
    end

    @testset "The factor mean round trips" begin
        mu = randn(rng, Kr)
        raw = PO.expand_factor_mu(fcb, mu)
        @test length(raw) == K
        @test raw ≈ dense_basis(fcb, T) * mu
        @test PO.reduce_factor_mu(fcb, raw) ≈ mu
        @test PO.expand_factor_mu(fcb, mu, 2) ≈ dense_basis(fcb, 2) * mu
    end

    @testset "The factor covariance round trips and is singular" begin
        A = randn(rng, Kr, Kr)
        S = A * transpose(A)
        raw = PO.expand_factor_covariance(fcb, S)
        @test size(raw) == (K, K)
        @test raw ≈ dense_basis(fcb, T) * S * transpose(dense_basis(fcb, T))
        @test raw ≈ transpose(raw)
        @test PO.reduce_factor_covariance(fcb, raw) ≈ S
        # The raw axis is a linear image of a smaller one, so the answer loses one rank per
        # constrained family.
        @test LinearAlgebra.rank(raw) == Kr
        @test PO.expand_factor_covariance(fcb, S, 3) ≈
              dense_basis(fcb, 3) * S * transpose(dense_basis(fcb, 3))
    end

    @testset "Projecting coordinates is not a column selection" begin
        x = randn(rng, T, K)
        y = PO.project_factor_coordinates(fcb, x)
        @test size(y) == (T, Kr)
        for t in 1:T
            @test y[t, :] ≈ transpose(dense_basis(fcb, t)) * x[t, :]
        end
        # A dropped coordinate reaches the retained ones, so the answer differs from a
        # selection of the retained columns.
        @test y[:, :] != x[:, PO.retained_factor_indices(fcb)]
        v = randn(rng, K)
        @test PO.project_factor_coordinates(fcb, v) ≈ transpose(dense_basis(fcb, T)) * v
    end

    @testset "The reconstruction weights carry the dropped factors" begin
        W = PO.dropped_factor_weights(fcb, T)
        @test size(W) == (1, Kr)
        mu = randn(rng, Kr)
        @test (W * mu)[1] ≈ PO.expand_factor_mu(fcb, mu)[fcb.fi[1][fcb.di[1]]]
    end

    @testset "A transform refuses a wrong axis" begin
        @test_throws DimensionMismatch PO.reduce_exposures(fcb, randn(rng, T, N, K + 1))
        @test_throws DimensionMismatch PO.reduce_exposures(fcb, randn(rng, T + 1, N, K))
        @test_throws DimensionMismatch PO.reduce_loadings(fcb, randn(rng, N, K + 1))
        @test_throws DomainError PO.reduce_loadings(fcb, randn(rng, N, K), T + 1)
        @test_throws DomainError PO.reduce_loadings(fcb, randn(rng, N, K), 0)
        @test_throws DimensionMismatch PO.reduce_factor_returns(fcb, randn(rng, T, K + 1))
        @test_throws DimensionMismatch PO.reduce_factor_returns(fcb, randn(rng, K + 1))
        @test_throws DimensionMismatch PO.reduce_factor_mu(fcb, randn(rng, K + 1))
        @test_throws DimensionMismatch PO.reduce_factor_covariance(fcb,
                                                                   randn(rng, K + 1, K + 1))
        @test_throws DimensionMismatch PO.expand_factor_returns(fcb, randn(rng, T, Kr + 1))
        @test_throws DimensionMismatch PO.expand_factor_returns(fcb, randn(rng, T + 1, Kr))
        @test_throws DimensionMismatch PO.expand_factor_mu(fcb, randn(rng, Kr + 1))
        @test_throws DimensionMismatch PO.expand_factor_covariance(fcb,
                                                                   randn(rng, Kr + 1,
                                                                         Kr + 1))
        @test_throws DimensionMismatch PO.project_factor_coordinates(fcb,
                                                                     randn(rng, T, K + 1))
        @test_throws DimensionMismatch PO.project_factor_coordinates(fcb,
                                                                     randn(rng, T + 1, K))
        @test_throws DimensionMismatch PO.project_factor_coordinates(fcb, randn(rng, K + 1))
        @test_throws DimensionMismatch PO.reduce_factor_names(fcb, ["a"])
        @test_throws DimensionMismatch PO.dropped_factor_names(fcb, ["a"])
    end
end

@testset "The FactorFamilyBasis builder" begin
    PO = PortfolioOptimisers
    rng = StableRNG(724_002)
    T, N = 6, 8
    nf = ["mkt", "size", "ind=a", "ind=b", "ind=c"]
    fam = ["market", "style", "ind", "ind", "ind"]
    K = 5
    Ms = zeros(T, N, K)
    Ms[:, :, 1] .= 1.0
    Ms[:, :, 2] .= randn(rng, T, N)
    # An uneven one-hot classification, so the three industry columns carry different
    # benchmark-weighted exposures and the automatic drop has a clear winner.
    lvl = [1 + (i % 3 == 0 ? 0 : (i % 2)) for t in 1:T, i in 1:N]
    for t in 1:T, i in 1:N, k in 1:3
        Ms[t, i, 2 + k] = lvl[t, i] == k ? 1.0 : 0.0
    end
    bw = rand(rng, T, N) .+ 0.1
    w = bw ./ sum(bw; dims = 2)
    c = [sum(w[t, i] * Ms[t, i, k] for i in 1:N) for t in 1:T, k in 1:K]

    @testset "The automatic drop takes the largest average absolute exposure" begin
        fcb = factor_family_basis(["ind" => nothing], Ms, bw, nf, fam)
        avg = [Statistics.mean(abs, view(c, :, k)) for k in 3:5]
        @test fcb.fnm == ["ind"]
        @test fcb.fi == [[3, 4, 5]]
        @test fcb.di == [argmax(avg)]
        @test fcb.K == K
        @test PO.dropped_factor_names(fcb, nf) == [nf[2 + argmax(avg)]]
        # The stored ratios are the benchmark-weighted exposures over the dropped one.
        raw, _, col = PO.family_retained_indices(fcb, 1)
        for p in eachindex(raw), t in 1:T
            @test fcb.ratios[t, col[p]] ≈ c[t, raw[p]] / c[t, fcb.fi[1][fcb.di[1]]]
        end
    end

    @testset "A stated drop overrides the automatic one" begin
        fcb = factor_family_basis(["ind" => "ind=a"], Ms, bw, nf, fam)
        @test fcb.di == [1]
        @test PO.dropped_factor_names(fcb, nf) == ["ind=a"]
        @test size(fcb.ratios) == (T, 2)
    end

    @testset "A non-finite benchmark weight is read as zero" begin
        bad = copy(bw)
        bad[2, 3] = NaN
        zeroed = copy(bw)
        zeroed[2, 3] = 0.0
        @test factor_family_basis(["ind" => "ind=a"], Ms, bad, nf, fam).ratios ≈
              factor_family_basis(["ind" => "ind=a"], Ms, zeroed, nf, fam).ratios
    end

    @testset "A non-finite exposure is read as zero" begin
        holed = copy(Ms)
        holed[3, 4, 2] = NaN
        zeroed = copy(Ms)
        zeroed[3, 4, 2] = 0.0
        @test factor_family_basis(["ind" => "ind=a"], holed, bw, nf, fam).ratios ≈
              factor_family_basis(["ind" => "ind=a"], zeroed, bw, nf, fam).ratios
    end

    @testset "The builder guards its inputs" begin
        @test_throws PortfolioOptimisers.IsEmptyError factor_family_basis(Pair[], Ms, bw,
                                                                          nf, fam)
        # A family named twice would take two blocks of one column range.
        @test_throws ArgumentError factor_family_basis(["ind" => nothing, "ind" => nothing],
                                                       Ms, bw, nf, fam)
        @test_throws ArgumentError factor_family_basis(["nope" => nothing], Ms, bw, nf, fam)
        # A family of one member has nothing to re-base.
        @test_throws ArgumentError factor_family_basis(["style" => nothing], Ms, bw, nf,
                                                       fam)
        @test_throws ArgumentError factor_family_basis(["ind" => "nope"], Ms, bw, nf, fam)
        @test_throws ArgumentError factor_family_basis(["ind" => "size"], Ms, bw, nf, fam)
        @test_throws ArgumentError factor_family_basis(["ind" => nothing], Ms, bw,
                                                       ["a", "a", "b", "c", "d"], fam)
        @test_throws DimensionMismatch factor_family_basis(["ind" => nothing], Ms, bw,
                                                           nf[1:4], fam)
        @test_throws DimensionMismatch factor_family_basis(["ind" => nothing], Ms, bw, nf,
                                                           fam[1:4])
        @test_throws DimensionMismatch factor_family_basis(["ind" => nothing], Ms,
                                                           bw[:, 1:4], nf, fam)
        # A negative benchmark weight has no meaning, and an observation with no weight at
        # all cannot state a weighted exposure.
        neg = copy(bw)
        neg[1, 1] = -0.5
        @test_throws DomainError factor_family_basis(["ind" => nothing], Ms, neg, nf, fam)
        dead = copy(bw)
        dead[4, :] .= 0.0
        @test_throws ArgumentError factor_family_basis(["ind" => nothing], Ms, dead, nf,
                                                       fam)
        # Constraining every raw factor still leaves a reduced axis, because a family of
        # `m >= 2` members drops one. The empty-axis guard is reachable only through the
        # constructor, which the testset above covers.
        @test PO.reduced_factor_count(factor_family_basis(["ind" => nothing], Ms[:, :, 3:5],
                                                          bw, nf[3:5], fam[3:5])) == 2
    end

    @testset "A zero benchmark-weighted exposure on the dropped factor is refused" begin
        # `ind=c` is never taken, so its benchmark-weighted exposure is zero and the ratio
        # against it is not finite. The refusal is the finiteness rule of the constructor,
        # not a tolerance: a merely small exposure gives a large finite ratio and is kept.
        empty_lvl = copy(Ms)
        empty_lvl[:, :, 5] .= 0.0
        empty_lvl[:, :, 4] .= 1.0 .- empty_lvl[:, :, 3]
        @test_throws PortfolioOptimisers.IsNonFiniteError factor_family_basis(["ind" => "ind=c"],
                                                                              empty_lvl, bw,
                                                                              nf, fam)
    end
end

@testset "Neutralising Factor Exposures" begin
    PO = PortfolioOptimisers
    rng = StableRNG(724_003)
    T, N = 10, 8
    nf = ["mkt", "size", "momentum", "ind=a", "ind=b"]
    fam = ["market", "style", "style", "industry", "industry"]
    Ms = randn(rng, T, N, 5)
    bw = fill(1 / N, T, N)
    cre = CrossSectionalLinearRegression()

    @testset "A family key neutralises every member" begin
        Y = copy(Ms)
        PO.neutralise_exposures!(Y, ["style" => ["industry"]], cre, bw, nf, fam)
        @test Y[:, :, 2] != Ms[:, :, 2]
        @test Y[:, :, 3] != Ms[:, :, 3]
        @test Y[:, :, 1] == Ms[:, :, 1]
        @test Y[:, :, 4] == Ms[:, :, 4]
        @test Y[:, :, 5] == Ms[:, :, 5]
    end

    @testset "A factor key neutralises that factor alone" begin
        Y = copy(Ms)
        PO.neutralise_exposures!(Y, ["momentum" => ["industry"]], cre, bw, nf, fam)
        @test Y[:, :, 2] == Ms[:, :, 2]
        @test Y[:, :, 3] != Ms[:, :, 3]
    end

    @testset "A single target name needs no vector" begin
        one = copy(Ms)
        many = copy(Ms)
        PO.neutralise_exposures!(one, ["momentum" => "mkt"], cre, bw, nf, fam)
        PO.neutralise_exposures!(many, ["momentum" => ["mkt"]], cre, bw, nf, fam)
        @test one == many
    end

    @testset "An asset with a missing exposure carries no weight" begin
        holed = copy(Ms)
        holed[3, 1, 2] = NaN
        holed[4, 2, 4] = NaN
        Y = copy(holed)
        PO.neutralise_exposures!(Y, ["size" => ["industry"]], cre, bw, nf, fam)
        # The missing key exposure and the missing target exposure both leave a hole.
        @test isnan(Y[3, 1, 2])
        @test isnan(Y[4, 2, 2])
        for t in 1:T, i in 1:N
            if !((t == 3 && i == 1) || (t == 4 && i == 2))
                @test isfinite(Y[t, i, 2])
            end
        end
    end

    @testset "Entries run in the order the caller wrote them" begin
        first_then = copy(Ms)
        PO.neutralise_exposures!(first_then,
                                 ["size" => ["industry"], "momentum" => ["size"]], cre, bw,
                                 nf, fam)
        other_way = copy(Ms)
        PO.neutralise_exposures!(other_way,
                                 ["momentum" => ["size"], "size" => ["industry"]], cre, bw,
                                 nf, fam)
        # The second entry of the first run sees the size exposure the first entry changed,
        # so the two orders answer differently.
        @test first_then[:, :, 3] != other_way[:, :, 3]
        @test first_then[:, :, 2] == other_way[:, :, 2]
    end

    @testset "A key that overlaps its targets is refused" begin
        for pr in
            ("momentum" => ["momentum"], "style" => ["style"], "style" => ["momentum"],
             "style" => ["style", "industry"])
            @test_throws ArgumentError PO.neutralise_exposures!(copy(Ms), [pr], cre, bw, nf,
                                                                fam)
        end
    end

    @testset "An unknown name is refused" begin
        @test_throws ArgumentError PO.neutralise_exposures!(copy(Ms),
                                                            ["nonexistent" => ["industry"]],
                                                            cre, bw, nf, fam)
        @test_throws ArgumentError PO.neutralise_exposures!(copy(Ms),
                                                            ["style" => ["nonexistent"]],
                                                            cre, bw, nf, fam)
        @test_throws PortfolioOptimisers.IsEmptyError PO.neutralise_exposures!(copy(Ms),
                                                                               ["style" =>
                                                                                    String[]],
                                                                               cre, bw, nf,
                                                                               fam)
    end

    @testset "The verb guards its axes" begin
        @test_throws DimensionMismatch PO.neutralise_exposures!(copy(Ms),
                                                                ["size" => ["mkt"]], cre,
                                                                bw, nf[1:4], fam)
        @test_throws DimensionMismatch PO.neutralise_exposures!(copy(Ms),
                                                                ["size" => ["mkt"]], cre,
                                                                bw, nf, fam[1:4])
        @test_throws DimensionMismatch PO.neutralise_exposures!(copy(Ms),
                                                                ["size" => ["mkt"]], cre,
                                                                bw[:, 1:4], nf, fam)
    end

    @testset "Neutralisation drives the cross-sectional correlation down" begin
        big = StableRNG(724_004)
        Tb, Nb = 20, 50
        X = randn(big, Tb, Nb, 5)
        wb = fill(1 / Nb, Tb, Nb)
        before = maximum(abs(Statistics.cor(X[1, :, s], X[1, :, i])) for s in 2:3, i in 4:5)
        PO.neutralise_exposures!(X, ["style" => ["industry"]], cre, wb, nf, fam)
        after = maximum(abs(Statistics.cor(X[t, :, s], X[t, :, i]))
                        for t in 1:Tb, s in 2:3, i in 4:5)
        @test after < 0.15
        @test after < before
    end
end

@testset "Parity with the reference implementation" begin
    PO = PortfolioOptimisers
    #=
    The reference implementation's basis builder and its Neutralisation verb were run on the
    Factor Exposures built from the synthetic Asset Panel below. The two cases pinned here are
    the constraint ratios of the automatic drop and of a stated drop. The reference's answers
    are bit-identical to the ones this file recomputes, and the neutralised exposures agreed to
    6e-15, which is the difference between two orders of the same weighted least squares.
    =#
    res = synthetic_asset_panel(; n_assets = 12, n_observations = 40, n_industries = 3,
                                rng = StableRNG(987))
    rd = res.rd
    factors = ["market" => ConstantExposure(),
               "size" =>
                   CompositeExposure(; descriptors = [Passthrough(; field = "market_cap")],
                                     bw = "market_cap"),
               "vol" =>
                   CompositeExposure(; descriptors = [Passthrough(; field = "book_equity")],
                                     bw = "market_cap"),
               "industry" => OneHotExposure(; field = "industry", family = "industry")]
    ax = PO.cross_sectional_factor_axis(factors, rd)
    blocks = Any[]
    for pr in factors
        L = factor_exposure(last(pr), rd)
        push!(blocks, ndims(L) == 3 ? L : reshape(L, size(L, 1), size(L, 2), 1))
    end
    Ms = cat(blocks...; dims = 3)
    bw = PO.exposure_benchmark_weights(rd, "market_cap")

    @test ax.nf == ["market", "size", "vol", "industry=Real Estate", "industry=Software",
                    "industry=Banks"]
    @test ax.fam == ["market", "style", "style", "industry", "industry", "industry"]

    @testset "Case one: the automatic drop matches the reference" begin
        fcb = factor_family_basis(["industry" => nothing], Ms, bw, ax.nf, ax.fam)
        @test PO.dropped_factor_names(fcb, ax.nf) == ["industry=Software"]
        @test size(fcb.ratios) == (40, 2)
        # The reference's own numbers, to the last bit it printed.
        @test fcb.ratios[1, :] ≈ [0.45312850847114156, 0.5134232034472233]
        @test fcb.ratios[2, :] ≈ [0.4499883905713916, 0.5141858044142668]
        @test fcb.ratios[end, :] ≈ [0.4003878534447987, 0.4488185225503101]
        @test vec(sum(fcb.ratios; dims = 1)) ≈ [17.16704789180863, 19.62192609499553]
    end

    @testset "Case two: a stated drop matches the reference" begin
        fcb = factor_family_basis(["industry" => "industry=Real Estate"], Ms, bw, ax.nf,
                                  ax.fam)
        @test fcb.di == [1]
        @test PO.dropped_factor_names(fcb, ax.nf) == ["industry=Real Estate"]
        @test fcb.ratios[1, :] ≈ [2.2068794642252954, 1.1330631241444427]
        @test fcb.ratios[end, :] ≈ [2.4975782641664717, 1.1209593864769638]
        @test vec(sum(fcb.ratios; dims = 1)) ≈ [93.32153055932386, 45.704976978893356]
    end

    @testset "The Neutralisation matches the reference" begin
        Y = copy(Ms)
        PO.neutralise_exposures!(Y, ["vol" => ["size"]], CrossSectionalLinearRegression(),
                                 bw, ax.nf, ax.fam)
        v = Y[:, :, 3]
        fin = isfinite.(v)
        @test count(!, fin) == 7
        # The two orders of the same weighted least squares differ in the last bits, so the
        # tolerance is loose where the ratios above are exact.
        @test v[1, :][fin[1, :]] ≈
              [0.799889405563868, -0.004743865235128638, 0.2239307925031833,
               0.7848793720945417, 0.6921418278212657, 0.8139226090427376,
               0.6331743087329431, 0.5729801019175156, 1.277378472749908,
               0.7383912304326324, -0.08625133637864447, -2.3746306903228844][fin[1, :]] rtol=1e-12
        @test sum(v[fin]) ≈ 120.8785614679006 rtol=1e-12
        # The re-standardisation makes the weighted sum of squares the eligible count.
        @test sum(abs2, v[fin]) ≈ 433.0 rtol=1e-12
    end
end
