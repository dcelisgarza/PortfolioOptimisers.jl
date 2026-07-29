using PortfolioOptimisers, Test, Clustering, TimeSeries, Dates, StableRNGs, StatsBase,
      LinearAlgebra, Clarabel

const PO = PortfolioOptimisers

#=
This file owns the collapse of a feature matrix onto the synthetic assets a meta-optimiser
builds for its outer problem. Before it, `NestedClustered` and `Stacking` dropped `Z` and
an outer `FeatureDistance` threw; now the outer problem is measured on features aggregated
from its members'.

The failure mode is the same quiet one the rest of the feature suite guards: a wrong
collapse still produces a finite, symmetric, plausible distance matrix over `k` synthetic
assets, and `HierarchicalRiskParity` consumes the dendrogram's *leaf ordering* rather than
its merges, so it can return byte-identical weights from a wrong matrix. Every end-to-end
test below therefore asserts on the matrix that reached the kernel, through the same
`RecordingDistance` instrument `test_13c_feature_views.jl` uses, and not on the weights.
=#
mutable struct RecordingDistance{T} <: PO.AbstractDistanceEstimator
    de::T
    seen::Vector{Any}
    lock::ReentrantLock
end
RecordingDistance(de) = RecordingDistance(de, Any[], ReentrantLock())
function record!(de::RecordingDistance, Z)
    lock(de.lock) do
        return push!(de.seen, Z)
    end
    return nothing
end
function PO.distance(de::RecordingDistance, ce, X; Z = nothing, kwargs...)
    record!(de, Z)
    return PO.distance(de.de, ce, X; Z = Z, kwargs...)
end
function PO.cor_and_dist(de::RecordingDistance, ce, X; Z = nothing, kwargs...)
    record!(de, Z)
    return PO.cor_and_dist(de.de, ce, X; Z = Z, kwargs...)
end
PO.distance(de::RecordingDistance, Z; kwargs...) = PO.distance(de.de, Z; kwargs...)
PO.cor_and_dist(de::RecordingDistance, Z; kwargs...) = PO.cor_and_dist(de.de, Z; kwargs...)

@testset "Collapsing the feature matrix onto synthetic assets" begin
    collapse = PO.collapse_feature_matrix
    saw = PO.synthetic_asset_weights
    features_are_assets = PO.features_are_assets
    seq = PO.FLoops.SequentialEx()
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false))
    hopt(de) = HierarchicalOptimiser(; cle = ClustersEstimator(; de = de), slv = slv)
    plain_hrp() = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; slv = slv))

    T, N, K, k = 250, 8, 3, 3
    rng = StableRNG(20260729)
    F = randn(rng, T, K) ./ 50
    Bl = randn(rng, N, K)
    X = F * transpose(Bl) .+ randn(rng, T, N) ./ 500
    nx = ["A$i" for i in 1:N]
    nf = ["F$i" for i in 1:K]
    ts = collect(Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(T - 1)))

    # A rectangular feature matrix, strictly positive so the non-negative-domain metrics
    # stay in play, and a square block-structured one -- the shape a phylogeny or adjacency
    # source produces, and the only one whose feature axis is the asset axis.
    Zr = abs.(randn(StableRNG(11), N, K)) .+ 0.1
    function block_features(rng, N, blocks)
        Z = 0.05 .* abs.(randn(rng, N, N))
        for b in blocks, i in b, j in b
            Z[i, j] += 1.0
        end
        Z .= (Z .+ transpose(Z)) ./ 2
        return Z
    end
    blocks = ([1, 2, 3], [4, 5, 6], [7, 8])
    Zsq = block_features(StableRNG(707), N, blocks)
    # Z3[t, i, j] names its own position, so a mis-indexed observation axis shows up in the
    # values and not merely in the shape.
    Z3 = reshape(Float64.(1:(T * N * K)), T, N, K) ./ 1000

    # Long-only weights that sum to one per synthetic asset, and a leveraged/shorting set
    # whose columns do not -- the case an un-normalised collapse inflates.
    Wl = abs.(randn(StableRNG(3), N, k))
    Wl ./= sum(Wl; dims = 1)
    Wlev = randn(StableRNG(5), N, k)
    Wlev[:, 1] .*= 3
    Wlev[1, 2] = -4.0

    @testset "The three collapse shapes" begin
        # Rectangular: only the asset axis contracts, so the feature axis survives intact.
        Cr = collapse(Zr, false, Wl)
        @test size(Cr) == (k, K)
        @test Cr ≈ transpose(saw(Wl)) * Zr
        # A long-only, unit-sum weight matrix is already convex, so normalising is exactly
        # the identity here -- which is what makes the leveraged case below the real test.
        @test Cr ≈ transpose(Wl) * Zr

        # Square: two-sided, so the feature axis lands on the synthetic universe too.
        Cs = collapse(Zsq, true, Wl)
        @test size(Cs) == (k, k)
        @test Cs ≈ transpose(Cs)
        @test Cs ≈ transpose(saw(Wl)) * Zsq * saw(Wl)
        # The square case is *detected* by the names, not re-derived from the shape, and
        # naming the collapsed feature axis after the synthetic assets is what keeps the
        # predicate true one level up.
        @test features_are_assets(["_$i" for i in 1:k], ["_$i" for i in 1:k])
        @test !features_are_assets(nf, ["_$i" for i in 1:k])

        # Time-varying: the same arithmetic per observation, observation axis leading.
        C3 = collapse(Z3, false, Wl)
        @test size(C3) == (T, k, K)
        @test all(t -> C3[t, :, :] ≈ transpose(saw(Wl)) * Z3[t, :, :], 1:T)
        Z3sq = Array{Float64}(undef, T, N, N)
        for t in 1:T
            Z3sq[t, :, :] = Zsq
        end
        C3s = collapse(Z3sq, true, Wl)
        @test size(C3s) == (T, k, k)
        @test all(t -> C3s[t, :, :] ≈ Cs, 1:T)

        # The vector arity collapses onto a single synthetic asset. It takes no `sq`
        # argument at all: the second contraction of the square case needs every synthetic
        # asset's weights at once, and only one vector exists inside a fold.
        w = Wlev[:, 1]
        @test collapse(Zr, w) ≈ transpose(Zr) * saw(w)
        @test length(collapse(Zr, w)) == K
        @test size(collapse(Z3, w)) == (T, K)
        @test all(t -> collapse(Z3, w)[t, :] ≈ transpose(Z3[t, :, :]) * saw(w), 1:T)
        # A square source therefore keeps the *real* assets as its feature axis here.
        @test length(collapse(Zsq, w)) == N

        # An absent feature matrix stays absent at both arities.
        @test isnothing(collapse(nothing, false, Wl))
        @test isnothing(collapse(nothing, w))
    end

    @testset "The collapse is convex, not an un-normalised sum" begin
        #=
        `iv` and `ivpa` are intensive and collapse as convex combinations; features are
        treated the same way. The distinction is invisible for long-only unit-sum weights
        and decisive under leverage or shorting, where a plain weighted sum scales each
        synthetic asset's feature vector by its gross exposure.
        =#
        s = vec(sum(abs.(Wlev); dims = 1))
        @test !all(x -> isapprox(x, 1), s)
        Craw = transpose(abs.(Wlev)) * Zr
        Cvex = collapse(Zr, false, Wlev)
        @test Cvex ≈ Craw ./ s
        # Bounded: every entry of a convex combination lies within the range of the column
        # it combines, whatever the gross exposure. The un-normalised sum does not.
        for j in 1:K
            @test all(minimum(Zr[:, j]) - 1e-12 .<=
                      Cvex[:, j] .<=
                      maximum(Zr[:, j]) + 1e-12)
        end
        @test any(Craw .> maximum(Zr))

        # Under the default metric the normalisation is a mathematical no-op in the
        # rectangular case: it scales one row of the result, and `AngularDist` is invariant
        # to positive row scaling. It is not a no-op in the square case, where the two-sided
        # product rescales the feature columns as well.
        fde = FeatureDistance()
        @test PO.distance(fde, Cvex) ≈ PO.distance(fde, Craw)
        Ssq = collapse(Zsq, true, Wlev)
        Sraw = transpose(abs.(Wlev)) * Zsq * abs.(Wlev)
        @test !isapprox(PO.distance(fde, Ssq), PO.distance(fde, Sraw))
    end

    @testset "A degenerate synthetic asset gets a zero row, not an error" begin
        # The same all-zero weight column already yields a zero returns column, zero `iv`
        # and zero `ivpa` with nothing throwing; the feature matrix must not be the one
        # quantity that hard-fails mid-meta-optimisation.
        Wz = copy(Wl)
        Wz[:, 2] .= 0
        Cz = collapse(Zr, false, Wz)
        @test all(iszero, Cz[2, :])
        @test Cz[[1, 3], :] ≈ collapse(Zr, false, Wl)[[1, 3], :]
        @test all(iszero, collapse(Zr, zeros(N)))

        # It then lands on the zero-feature-vector convention the kernel already
        # implements: a finite distance matrix rather than a `NaN` one.
        D = PO.distance(FeatureDistance(), Cz)
        @test all(isfinite, D)
        @test isapprox(D, transpose(D))
        @test iszero(diag(D))
        # Two degenerate assets are declared mutually identical, which is the same
        # convention a zero row gets anywhere else.
        Wzz = copy(Wz)
        Wzz[:, 3] .= 0
        Dzz = PO.distance(FeatureDistance(), collapse(Zr, false, Wzz))
        @test iszero(Dzz[2, 3])
    end

    rd_r = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, ts = ts,
                         nz = ["z$i" for i in 1:K], Z = Zr)
    rd_sq = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, ts = ts, nz = nx, Z = Zsq)
    rd_3d = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, ts = ts, nz = nf, Z = Z3)

    @testset "prepare_outer_rd collapses, and its arity break is loud" begin
        wi = Wlev
        nb, B, iv, ivpa, nz, Z, Xb = PO.prepare_outer_rd(rd_r, wi)
        @test nz == rd_r.nz
        @test Z ≈ collapse(Zr, false, wi)
        @test size(Xb) == (T, k)

        # The square carrier renames its feature axis after the synthetic assets, which is
        # what keeps `features_are_assets` true for the result built from it.
        _, _, _, _, nz_s, Z_s, _ = PO.prepare_outer_rd(rd_sq, wi)
        @test nz_s == ["_$i" for i in 1:k]
        @test features_are_assets(nz_s, ["_$i" for i in 1:k])
        @test Z_s ≈ collapse(Zsq, true, wi)

        _, _, _, _, nz_3, Z_3, _ = PO.prepare_outer_rd(rd_3d, wi)
        @test nz_3 == nf
        @test size(Z_3) == (T, k, K)

        #=
        `predict_outer_*` is a documented overload point, so extending the tuple is a
        deliberate break. `nz`/`Z` are returned *before* `X` precisely so the break bites:
        Julia's destructuring discards trailing values without complaint, so appending them
        would have let a stale overload keep building a feature-less result in silence.
        =#
        stale_nb, stale_B, stale_iv, stale_ivpa, stale_X = PO.prepare_outer_rd(rd_r, wi)
        @test stale_X === nz
        @test !isa(stale_X, AbstractMatrix)
        @test_throws MethodError stale_X[1] = 1.0
    end

    @testset "The outer problem is measured on collapsed features" begin
        # NestedClustered's synthetic assets are its clusters; Stacking's are its inner
        # portfolios. Both reach the outer optimiser through `prepare_outer_rd`.
        mk_st(de) = Stacking(;
                             opti = [plain_hrp(),
                                     HierarchicalEqualRiskContribution(;
                                                                       opt = HierarchicalOptimiser(;
                                                                                                   slv = slv))],
                             opto = HierarchicalRiskParity(; opt = hopt(de)), ex = seq)
        mk_nco(de) = NestedClustered(; cle = ClustersEstimator(; de = FeatureDistance()),
                                     opti = plain_hrp(),
                                     opto = HierarchicalRiskParity(; opt = hopt(de)),
                                     ex = seq)
        # `wi` is the matrix whose columns are the inner solutions laid out on the full
        # universe -- exactly what the outer result was built from. Stacking's inner
        # optimisers all see the whole universe; NestedClustered's see one cluster each.
        stacked_wi(res) = reduce(hcat, [r.w for r in res.resi])
        function clustered_wi(res)
            idx = assignments(res.clr)
            cls = [findall(==(i), idx) for i in 1:(res.clr.k)]
            wi = zeros(N, length(cls))
            for (i, cl) in pairs(cls)
                wi[cl, i] = res.resi[i].w
            end
            return wi
        end
        for (mk, weights_of) in ((mk_st, stacked_wi), (mk_nco, clustered_wi))
            for (rd, Zsrc, sq) in
                ((rd_r, Zr, false), (rd_sq, Zsq, true), (rd_3d, Z3, false))
                ro = RecordingDistance(FeatureDistance())
                res = optimise(mk(ro), rd)
                @test isapprox(sum(res.w), 1)
                @test length(ro.seen) == 1
                Zo = ro.seen[1]
                wi = weights_of(res)
                @test Zo ≈ collapse(Zsrc, sq, wi)
                @test size(Zo, ndims(Zo) - 1) == size(wi, 2)
                @test all(isfinite, Zo)
            end
        end
    end

    @testset "The cross-validation path carries a time-varying collapse" begin
        #=
        Site three is `reconstruct_rd`, inside the fold. Each fold collapses with its own
        weights, so a *static* source becomes genuinely time-varying: constant within a
        fold and different in the next. The row count matches `X` exactly because both are
        built by the same stacking, and the outer default `LastObservation` then reduces it
        to the most recent fold.
        =#
        n_folds = 3
        cvopt = OptimisationCrossValidation(; cv = KFold(; n = n_folds))
        mk_st(de) = Stacking(;
                             opti = [plain_hrp(),
                                     HierarchicalEqualRiskContribution(;
                                                                       opt = HierarchicalOptimiser(;
                                                                                                   slv = slv))],
                             opto = HierarchicalRiskParity(; opt = hopt(de)), cv = cvopt,
                             ex = seq)
        mk_nco(de) = NestedClustered(; cle = ClustersEstimator(; de = FeatureDistance()),
                                     opti = plain_hrp(),
                                     opto = HierarchicalRiskParity(; opt = hopt(de)),
                                     cv = cvopt, ex = seq)
        for mk in (mk_st, mk_nco)
            for (rd, nfeat) in ((rd_r, K), (rd_3d, K))
                ro = RecordingDistance(FeatureDistance())
                res = optimise(mk(ro), rd)
                @test isapprox(sum(res.w), 1)
                Zo = ro.seen[1]
                @test ndims(Zo) == 3
                @test size(Zo, 1) == T
                @test size(Zo, 3) == nfeat
                @test all(isfinite, Zo)
            end

            # A static source is constant within each fold and changes at the boundaries:
            # one distinct feature row per fold, per synthetic asset.
            ro = RecordingDistance(FeatureDistance())
            optimise(mk(ro), rd_r)
            Zo = ro.seen[1]
            for i in axes(Zo, 2)
                @test length(unique(eachrow(Zo[:, i, :]))) == n_folds
            end
        end
    end

    @testset "A square carrier through a fold keeps the real assets as its features" begin
        #=
        Only one weight vector is in scope inside a fold, so the second contraction of the
        square case is unavailable -- and doing it one-sidedly with that single vector
        would leave one number per synthetic asset, a feature space in which every asset is
        trivially identical. The fold path therefore leaves the feature axis alone, and the
        collapsed matrix reads as the synthetic asset's weighted-average neighbourhood.
        =#
        cvopt = OptimisationCrossValidation(; cv = KFold(; n = 3))
        ro = RecordingDistance(FeatureDistance())
        st = Stacking(;
                      opti = [plain_hrp(),
                              HierarchicalEqualRiskContribution(;
                                                                opt = HierarchicalOptimiser(;
                                                                                            slv = slv))],
                      opto = HierarchicalRiskParity(; opt = hopt(ro)), cv = cvopt, ex = seq)
        res = optimise(st, rd_sq)
        @test isapprox(sum(res.w), 1)
        @test size(ro.seen[1]) == (T, 2, N)

        #=
        `NestedClustered` is the one combination this cannot serve: its folds see
        cluster-sliced returns, so a square carrier comes back with a *different* feature
        axis per cluster and there is nothing to stack them against. The matrix is dropped,
        and an outer estimator asking for features gets the same error it would get had
        none been supplied -- never a matrix assembled from mismatched axes.
        =#
        nco = NestedClustered(; cle = ClustersEstimator(; de = FeatureDistance()),
                              opti = plain_hrp(),
                              opto = HierarchicalRiskParity(;
                                                            opt = hopt(FeatureDistance())),
                              cv = cvopt, ex = seq)
        e = try
            optimise(nco, rd_sq)
        catch err
            err
        end
        @test isa(e, PO.IsNothingError)
        @test occursin("neither the returns result nor the prior result", e.msg)
    end

    @testset "The transport carrier keeps its shapes honest" begin
        # `PredictionReturnsResult` never has its feature matrix read -- it only has to
        # survive the stack -- so it validates the pair rule, the feature axis and the
        # observation axis, and nothing else.
        Xf = collect(1.0:5.0)
        Zf = randn(StableRNG(2), 5, 2)
        rdp = PredictionReturnsResult(; nx = ["_1"], X = Xf, nz = ["a", "b"], Z = Zf)
        @test rdp.Z === Zf
        @test rdp.nz == ["a", "b"]
        @test isnothing(PredictionReturnsResult(; nx = ["_1"], X = Xf).Z)
        @test_throws PO.IsNothingError PredictionReturnsResult(; nx = ["_1"], X = Xf,
                                                               nz = ["a"])
        @test_throws DimensionMismatch PredictionReturnsResult(; nx = ["_1"], X = Xf,
                                                               nz = ["a"], Z = Zf)
        @test_throws DimensionMismatch PredictionReturnsResult(; nx = ["_1"], X = Xf,
                                                               nz = ["a", "b"],
                                                               Z = Zf[1:4, :])
        # The vector shape needs one matrix per portfolio-returns entry.
        @test_throws DimensionMismatch PredictionReturnsResult(; nx = ["_1"], X = Xf,
                                                               nz = ["a", "b"], Z = [Zf])
    end
end
