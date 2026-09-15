#=
Issue #1063. A Prior Result lives on the full asset universe, and an asset outside its
Investable Mask carries `NaN` in its moments. Inside an optimiser the result arrives reduced
(ADR 0115). Standalone, the prior arm of the `ucs` triple reduces to the mask, fits, and
expands (ADR 0111), so a set fitted standalone on a point-in-time prior is over the same
assets the prior is, and a view of it at the mask recovers the reduced fit. #1062 pinned
that for the orthogonal set in `test_10b`; this file pins it for the three returns-data
families under `pe = nothing`, and for their returns-data route, which shares the tail.
=#
include(joinpath(@__DIR__, "test06c_setup.jl"))
@testset "Uncertainty sets on a point-in-time prior" begin
    using PortfolioOptimisers, Test, StableRNGs, Random, Statistics, LinearAlgebra
    PO = PortfolioOptimisers
    # This panel leaves assets outside the Investable Mask at the latest observation, which
    # is the case this file is about; the assertions below guard the fixture.
    rdp = synthetic_asset_panel(; n_assets = 40, n_observations = 200, n_industries = 3,
                                rng = StableRNG(725_001)).rd
    prp = prior(EmpiricalPrior(), rdp)
    msk = PO.investable_mask(prp)
    @test !isnothing(msk)
    @test count(msk) < length(msk)
    idx = findall(msk)
    N = length(msk)
    pidx = PO.fourth_moment_index_generator(N, idx)
    prr = PO.port_opt_view(prp, idx)
    # Two sets are the same set when every numeric array field is, `NaN` for `NaN`.
    function numeric_fields(s)
        return [n => getproperty(s, n)
                for n in propertynames(s)
                if getproperty(s, n) isa AbstractArray &&
            eltype(getproperty(s, n)) <: Number]
    end
    function view_recovers(full, red)
        v = PO.port_opt_view(full, idx)
        fv, rv = numeric_fields(v), numeric_fields(red)
        return length(fv) == length(rv) && all(zip(fv, rv)) do ((na, a), (nb, b))
        na == nb && size(a) == size(b) && isequal(Array(a), Array(b))
    end
    end
    # The frame outside the mask: a box carries the moment's own `NaN`, a shape matrix or a
    # map carries zero, and the centre is the prior's own moment with its `NaN` frame.
    function frame_is_right(set::BoxUncertaintySet{<:AbstractVector, <:AbstractVector})
        return size(set.lb, 1) == N &&
               all(isnan, view(set.lb, .!msk)) &&
               all(isnan, view(set.ub, .!msk)) &&
               set.val === prp.mu
    end
    function frame_is_right(set::BoxUncertaintySet{<:AbstractMatrix, <:AbstractMatrix})
        return size(set.lb) == (N, N) &&
               all(isnan, view(set.lb, .!msk, :)) &&
               all(isnan, view(set.ub, :, .!msk)) &&
               set.val === prp.sigma
    end
    function frame_is_right(set::EllipsoidalUncertaintySet{<:Any, <:Any,
                                                           <:MuUncertaintySetClass})
        return size(set.sigma) == (N, N) &&
               all(iszero, view(set.sigma, .!msk, :)) &&
               all(iszero, view(set.sigma, :, .!msk)) &&
               set.val === prp.mu
    end
    function frame_is_right(set::EllipsoidalUncertaintySet{<:Any, <:Any,
                                                           <:SigmaUncertaintySetClass})
        off = setdiff(1:(N ^ 2), pidx)
        return size(set.sigma) == (N^2, N^2) &&
               all(iszero, view(set.sigma, off, :)) &&
               all(iszero, view(set.sigma, :, off)) &&
               set.val === prp.sigma
    end
    function frame_is_right(set::NormBallUncertaintySet{<:Any, <:Any, <:Any,
                                                        <:MuUncertaintySetClass})
        return size(set.L, 1) == N &&
               all(iszero, view(set.L, .!msk, :)) &&
               all(isfinite, set.L) &&
               set.val === prp.mu
    end
    function frame_is_right(set::NormBallUncertaintySet{<:Any, <:Any, <:Any,
                                                        <:SigmaUncertaintySetClass})
        off = setdiff(1:(N ^ 2), pidx)
        return size(set.L, 1) == N^2 &&
               all(iszero, view(set.L, off, :)) &&
               all(isfinite, set.L) &&
               set.val === prp.sigma
    end
    # One builder per family and algorithm, taking the prior estimator, so the same
    # configuration is fitted on the prior route (`pe = nothing`) and on the returns route.
    # `seed` pins the draws, so the two fits on the same reduced inputs are identical.
    families = ["Delta" => (pe -> DeltaUncertaintySet(; pe = pe)),
                "Normal box" => (pe -> NormalUncertaintySet(; pe = pe,
                                                            alg = BoxUncertaintySetAlgorithm(),
                                                            n_sim = 100, seed = 1063)),
                "Normal ellipsoid" => (pe -> NormalUncertaintySet(; pe = pe,
                                                                  alg = EllipsoidalUncertaintySetAlgorithm(),
                                                                  n_sim = 100, seed = 1063)),
                "Normal ellipsoid, sampled radius" =>
                    (pe -> NormalUncertaintySet(; pe = pe,
                                                alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                                         method = NormalKUncertaintyAlgorithm()),
                                                n_sim = 100, seed = 1063)),
                "Normal norm ball" => (pe -> NormalUncertaintySet(; pe = pe,
                                                                  alg = NormBallUncertaintySetAlgorithm(),
                                                                  n_sim = 100, seed = 1063)),
                "ARCH box" =>
                    (pe -> ARCHUncertaintySet(; pe = pe, alg = BoxUncertaintySetAlgorithm(),
                                              n_sim = 100, seed = 1063)),
                "ARCH ellipsoid" => (pe -> ARCHUncertaintySet(; pe = pe,
                                                              alg = EllipsoidalUncertaintySetAlgorithm(),
                                                              n_sim = 100, seed = 1063)),
                "ARCH norm ball" => (pe -> ARCHUncertaintySet(; pe = pe,
                                                              alg = NormBallUncertaintySetAlgorithm(),
                                                              n_sim = 100, seed = 1063))]
    for (name, build) in families
        @testset "$(name)" begin
            ue = build(nothing)
            # The standalone fit answers the full universe, with the right frame outside
            # the mask.
            mu_f, sg_f = ucs(ue, prp)
            @test frame_is_right(mu_f)
            @test frame_is_right(sg_f)
            # A view at the mask recovers the fit on the reduced prior, which is the fit an
            # optimiser makes, on both axes and through each single-axis verb.
            mu_r, sg_r = ucs(ue, prr)
            @test view_recovers(mu_f, mu_r)
            @test view_recovers(sg_f, sg_r)
            @test view_recovers(mu_ucs(ue, prp), mu_ucs(ue, prr))
            @test view_recovers(sigma_ucs(ue, prp), sigma_ucs(ue, prr))
            # The returns route fits its own prior on the same panel and forwards to the
            # same tail, so it answers the same sets.
            mu_x, sg_x = ucs(build(EmpiricalPrior()), rdp)
            @test view_recovers(mu_x, mu_r)
            @test view_recovers(sg_x, sg_r)
            @test size(mu_x.val, 1) == N
            @test isequal(mu_x.val, prp.mu)
        end
    end
    @testset "A reduced prior is a passthrough" begin
        # On a result whose mask is `nothing`, the reduction and the expansion touch
        # nothing: the set fitted on the reduced prior is the reduced fit itself.
        ue = DeltaUncertaintySet(; pe = nothing)
        mu_r, sg_r = ucs(ue, prr)
        @test isnothing(PO.investable_mask(prr))
        @test mu_r.val === prr.mu
        @test sg_r.val === prr.sigma
        @test size(mu_r.lb, 1) == count(msk)
        @test all(isfinite, mu_r.ub)
    end
end
