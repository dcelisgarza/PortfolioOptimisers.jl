#=
`BlendDistance` (#2207, decided by #2196): a blend of the distance matrices of its parts. This
file owns the kernel against the formula of each rule, the scale rule, the weights, the chord
identity, every refusal of the constructor, the forwarders (`feature_readable_mask`,
`plain_graph_reader`, `reads_sample_values`, `factory`) and the reach of the blend into HRP,
HERC, Schur, NCO, a fold, a network and preselection.
=#
using PortfolioOptimisers, Test, StableRNGs, StatsBase, LinearAlgebra, Clarabel
const PO = PortfolioOptimisers

# Asset `d` of `drop_rd` delists at row 21, and a Coverage Policy that expires after 100 rows
# keeps it in the investable universe. `FeatureDistance` reads the last row by default, where `d`
# is inactive, so a part of that form cannot read `d`. The fixture of test_13e, with one change:
# no gap in the returns, so a returns part reads every asset.
function blend_drop_rd()
    rng = StableRNG(14542)
    T, N = 40, 5
    X = 0.01 .* randn(rng, T, N)
    A = trues(T, N)
    A[21:end, 4] .= false
    pnl = AssetPanel(; pf = [NumericPanelField(; name = "f", vals = randn(rng, T, N))],
                     amsk = A, emsk = A)
    return ReturnsResult(; nx = ["a", "b", "c", "d", "e"], X = X, pnl = pnl)
end

@testset "BlendDistance" begin
    rng = StableRNG(2207)
    T, N = 120, 10
    # Two blocks of correlated assets, and a taxonomy that cuts across the blocks, so the two
    # parts disagree about the structure.
    F0 = randn(rng, T, 2)
    X = 0.01 .* (F0[:, repeat([1, 2]; inner = 5)] .+ 0.8 .* randn(rng, T, N))
    nx = ["A$i" for i in 1:N]
    sets = UniverseSets(; xkey = "nx",
                        dict = Dict("nx" => nx, "Sector" => repeat(["Tech", "Energy"], 5)))
    rd = ReturnsResult(; nx = nx, X = X, pnl = asset_panel(panel_input(sets, ["Sector"])))
    rd_bare = ReturnsResult(; nx = nx, X = X)
    ce = PortfolioOptimisersCovariance()
    rde = Distance(; alg = CanonicalDistance())
    fde = FeatureDistance()
    DR = distance(rde, ce, X)
    DF = distance(fde, ce, X; rd = rd)
    rho = cor_and_dist(rde, ce, X)[1]
    cF = cor_and_dist(fde, ce, X; rd = rd)[1]
    seq = PO.FLoops.SequentialEx()

    @testset "The blend rules against their formulas" begin
        w = [0.3, 0.7]
        bd(alg) = distance(BlendDistance(; des = [rde, fde], w = w, alg = alg), ce, X;
                           rd = rd)
        @test bd(QuadraticBlend()) == sqrt.(0.3 .* DR .^ 2 .+ 0.7 .* DF .^ 2)
        @test bd(ArithmeticBlend()) == 0.3 .* DR .+ 0.7 .* DF
        @test bd(PowerBlend(; p = 3)) ≈ (0.3 .* DR .^ 3 .+ 0.7 .* DF .^ 3) .^ (1 / 3) rtol = 1e-14
        @test bd(PowerBlend(; p = Inf)) == max.(DR, DF)
        # Under the maximum, the weights set only which parts enter.
        @test distance(BlendDistance(; des = [rde, fde], w = [0.9, 0.1],
                                     alg = PowerBlend(; p = Inf)), ce, X; rd = rd) ==
              max.(DR, DF)
        # `cor_and_dist` returns the same distance matrix, and `sim` of it.
        for alg in (QuadraticBlend(), ArithmeticBlend(), PowerBlend(; p = 1.5))
            b = BlendDistance(; des = [rde, fde], w = w, alg = alg)
            S, D = cor_and_dist(b, ce, X; rd = rd)
            @test D == bd(alg)
            @test S == PO.distance_to_similarity(b.sim; D = D)
        end
    end

    @testset "The weights: nothing, normalisation, zero" begin
        b0 = BlendDistance(; des = [rde, fde])
        @test isnothing(b0.w)
        @test distance(b0, ce, X; rd = rd) ==
              distance(BlendDistance(; des = [rde, fde], w = [0.5, 0.5]), ce, X; rd = rd)
        # A given `w` is divided by its sum, and the field keeps the form of the caller.
        b2 = BlendDistance(; des = [rde, fde], w = [1, 3])
        @test b2.w == [1, 3]
        @test distance(b2, ce, X; rd = rd) ≈
              distance(BlendDistance(; des = [rde, fde], w = [0.25, 0.75]), ce, X; rd = rd) rtol = 1e-15
        # A part with weight zero computes nothing: the feature part would refuse returns data
        # that hold no panel, and the blend never calls it.
        bz = BlendDistance(; des = [rde, fde], w = [1.0, 0.0])
        @test distance(bz, ce, X; rd = rd_bare) == DR
        @test_throws IsNothingError distance(BlendDistance(; des = [rde, fde]), ce, X;
                                             rd = rd_bare)
        @test PO.positive_parts(bz) == [1]
        @test PO.positive_parts(b0) == 1:2
    end

    @testset "The scale rule, one rule or one entry for each part" begin
        # `LogDistance` is unbounded, so it outweighs a chord part.
        lde = Distance(; alg = LogDistance())
        DL = distance(lde, ce, X)
        @test maximum(DL) > 1
        sc(D) = D ./ maximum(D)
        b1 = BlendDistance(; des = [lde, fde], w = [0.5, 0.5], scale = MaximumScale())
        @test distance(b1, ce, X; rd = rd) ≈ sqrt.(0.5 .* sc(DL) .^ 2 .+ 0.5 .* sc(DF) .^ 2) rtol = 1e-15
        bv = BlendDistance(; des = [lde, fde], w = [0.5, 0.5],
                           scale = [MaximumScale(), nothing])
        @test distance(bv, ce, X; rd = rd) ≈ sqrt.(0.5 .* sc(DL) .^ 2 .+ 0.5 .* DF .^ 2) rtol = 1e-15
        # A zero matrix stays as it is, and so does the same object.
        Z = zeros(3, 3)
        @test PO.scale_distance(MaximumScale(), Z) === Z
        @test PO.scale_distance(nothing, DL) === DL
        # The maximum reads the finite entries alone: a `NaN` row and an infinite entry stay.
        Dn = [0.0 2.0 NaN Inf; 2.0 0.0 NaN 1.0; NaN NaN NaN NaN; Inf 1.0 NaN 0.0]
        Ds = PO.scale_distance(MaximumScale(), Dn)
        @test Ds[1, 2] == 1.0 && Ds[2, 4] == 0.5
        @test all(isnan, Ds[3, :]) && isinf(Ds[1, 4])
        # The `NaN` row of a part reaches the blend, so a fit can still drop the asset.
        @test all(isnan,
                  PO.blend_distances(QuadraticBlend(), [0.5, 0.5], [Dn, zeros(4, 4)])[3, :])
        @test all(isnan,
                  PO.blend_distances(PowerBlend(; p = Inf), [0.5, 0.5], [Dn, zeros(4, 4)])[3,
                                                                                           :])
    end

    @testset "The chord identity under QuadraticBlend" begin
        for w in ([0.25, 0.75], [0.5, 0.5], [0.8, 0.2])
            b = BlendDistance(; des = [rde, fde], w = w)
            @test b.sim == ChordSimilarity()
            S, D = cor_and_dist(b, ce, X; rd = rd)
            cbar = w[1] .* rho .+ w[2] .* cF
            @test D ≈ sqrt.(clamp.((1 .- cbar) ./ 2, 0, 1)) rtol = 1e-14
            @test S ≈ cbar atol = 1e-14
        end
        # The cosine of the stacked unit-scaled feature vectors is the weighted mean.
        rngz = StableRNG(7)
        Z1, Z2 = randn(rngz, N, 3), randn(rngz, N, 4)
        u(Z) = Z ./ norm.(eachrow(Z))
        w = [0.3, 0.7]
        Zs = hcat(sqrt(w[1]) .* u(Z1), sqrt(w[2]) .* u(Z2))
        chord_sim(D) = PO.distance_to_similarity(ChordSimilarity(); D = D)
        Db = PO.blend_distances(QuadraticBlend(), w, [distance(fde, Z1), distance(fde, Z2)])
        @test Db ≈ distance(fde, Zs) rtol = 1e-12
        @test chord_sim(Db) ≈
              w[1] .* chord_sim(distance(fde, Z1)) .+ w[2] .* chord_sim(distance(fde, Z2)) atol = 1e-14
    end

    @testset "The constructor and its refusals" begin
        @test_throws IsEmptyError BlendDistance(; des = PO.AbstractDistanceEstimator[])
        @test_throws IsEmptyError BlendDistance(; des = [rde], w = Float64[])
        @test_throws DimensionMismatch BlendDistance(; des = [rde], w = [1.0, 2.0])
        @test_throws DomainError BlendDistance(; des = [rde, fde], w = [1.0, -2.0])
        @test_throws DomainError BlendDistance(; des = [rde, fde], w = [1.0, Inf])
        @test_throws DomainError BlendDistance(; des = [rde, fde], w = [1.0, NaN])
        @test_throws "no positive entry" BlendDistance(; des = [rde, fde], w = [0, 0])
        @test_throws DimensionMismatch BlendDistance(; des = [rde, fde],
                                                     scale = [MaximumScale()])
        @test_throws UndefKeywordError BlendDistance()
        @test_throws UndefKeywordError PowerBlend()
        @test_throws DomainError PowerBlend(; p = 0.5)
        @test_throws DomainError PowerBlend(; p = NaN)
        @test_throws "ArithmeticBlend()" PowerBlend(; p = 1)
        @test_throws "QuadraticBlend()" PowerBlend(; p = 2.0)
        @test PowerBlend(; p = Inf).p == Inf
        # The default similarity follows the rule, and a stated one is kept.
        @test BlendDistance(; des = [rde]).sim == ChordSimilarity()
        @test BlendDistance(; des = [rde], alg = ArithmeticBlend()).sim ==
              ComplementSimilarity()
        @test BlendDistance(; des = [rde], alg = PowerBlend(; p = 3)).sim ==
              ComplementSimilarity()
        @test BlendDistance(; des = [rde], sim = ExponentialSimilarity()).sim ==
              ExponentialSimilarity()
        # `factory` reaches each part.
        pr = prior(EmpiricalPrior(), rd)
        @test PO.factory(BlendDistance(; des = [rde, fde]), pr) isa BlendDistance
    end

    @testset "A blend of one part, and a nested blend" begin
        @test distance(BlendDistance(; des = [rde]), ce, X) == DR
        @test distance(BlendDistance(; des = [fde]), ce, X; rd = rd) == DF
        @test distance(BlendDistance(; des = [rde], alg = PowerBlend(; p = Inf)), ce, X) ==
              DR
        # The matrix route of `logo!`, which passes no data on an incremental fit.
        @test distance(BlendDistance(; des = [rde]), rho, nothing) == distance(rde, rho)
        inner = BlendDistance(; des = [rde, fde], w = [0.5, 0.5], alg = ArithmeticBlend())
        outer = BlendDistance(; des = [inner, rde], w = [0.4, 0.6])
        Di = 0.5 .* DR .+ 0.5 .* DF
        @test distance(outer, ce, X; rd = rd) ≈ sqrt.(0.4 .* Di .^ 2 .+ 0.6 .* DR .^ 2) rtol = 1e-15
        # The forwarders recurse into the nested blend, and skip a part with weight zero.
        @test PO.reads_returns_matrix(outer)
        @test !PO.reads_returns_matrix(BlendDistance(; des = [fde]))
        @test PO.reads_returns_matrix(BlendDistance(;
                                                    des = [BlendDistance(; des = [fde]),
                                                           BlendDistance(; des = [rde])]))
        @test !PO.reads_returns_matrix(BlendDistance(; des = [fde, rde], w = [1, 0]))
        @test PO.plain_graph_reader(ClustersEstimator(; de = outer)) ==
              PO.plain_graph_reader(ClustersEstimator(; de = rde))
        @test !PO.plain_graph_reader(ClustersEstimator(; de = BlendDistance(; des = [fde])))
        @test !PO.plain_graph_reader(NetworkEstimator(; de = BlendDistance(; des = [fde])))
        @test !PO.plain_graph_reader(SpectralSeriation(; de = BlendDistance(; des = [fde])))
        @test !PO.reads_sample_values(BlendDistance(; des = [rde]))
        @test PO.reads_sample_values(BlendDistance(; des = [rde, fde]))
        @test !PO.reads_sample_values(BlendDistance(; des = [rde, fde], w = [1, 0]))
    end

    @testset "A blend reaches the clustering, network and seriation consumers" begin
        b = BlendDistance(; des = [rde, fde])
        Db = distance(b, ce, X; rd = rd)
        clr = clusterise(ClustersEstimator(; de = b), rd)
        @test clr.D ≈ Db rtol = 1e-15
        @test clr.D != clusterise(ClustersEstimator(; de = rde), rd).D
        @test clr.D != clusterise(ClustersEstimator(; de = fde), rd).D
        # The tree and network branches of `calc_adjacency`.
        # The PMFG of the similarity branch needs at least nine assets.
        for alg in (KruskalTree(), MaximumDistanceSimilarity())
            @test size(phylogeny_matrix(NetworkEstimator(; de = b, alg = alg), rd).X) ==
                  (N, N)
        end
        @test length(seriate(SpectralSeriation(; de = b), rd).order) == N
    end

    @testset "HRP, HERC, Schur and NCO, alone and in a fold" begin
        b = BlendDistance(; des = [rde, fde])
        cle = ClustersEstimator(; de = b)
        ser = HierarchicalSeriation(; cle = cle)
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("verbose" => false))
        opts = ("HierarchicalRiskParity" => HierarchicalRiskParity(; ser = ser),
                "HierarchicalEqualRiskContribution" =>
                    HierarchicalEqualRiskContribution(; cle = cle),
                "SchurComplementHierarchicalRiskParity" =>
                    SchurComplementHierarchicalRiskParity(; ser = ser),
                "NestedClustered" => NestedClustered(; cle = cle,
                                                     opti = MeanRisk(; opt = JuMPOptimiser(; slv = slv)),
                                                     opto = MeanRisk(; opt = JuMPOptimiser(; slv = slv))))
        cv = IndexWalkForward(60, 30)
        for (name, opt) in opts
            @testset "$(name)" begin
                res = optimise(opt, rd)
                @test all(isfinite, res.w)
                @test sum(res.w) ≈ 1 rtol = 1e-6
                pred = cross_val_predict(opt, rd, cv; ex = seq)
                @test all(isfinite, pred.mrd.X)
            end
        end
        # A blend of one feature part gives the weights of that part: the same `D` and the
        # same `S`, because the quadratic blend of one part is the part.
        hb = optimise(HierarchicalRiskParity(;
                                             ser = HierarchicalSeriation(;
                                                                         cle = ClustersEstimator(;
                                                                                                 de = BlendDistance(;
                                                                                                                    des = [fde])))),
                      rd)
        hf = optimise(HierarchicalRiskParity(;
                                             ser = HierarchicalSeriation(;
                                                                         cle = ClustersEstimator(;
                                                                                                 de = fde))),
                      rd)
        @test hb.w == hf.w
    end

    @testset "Preselection reads the blend" begin
        sel(de) = RedundancySelector(;
                                     alg = ClusterGroups(;
                                                         cle = ClustersEstimator(; de = de)),
                                     score = SCM())
        kept = fit_preprocessing(sel(BlendDistance(; des = [rde, fde])), rd).nx
        @test !isempty(kept) && length(kept) <= N
        @test fit_preprocessing(sel(BlendDistance(; des = [fde])), rd).nx ==
              fit_preprocessing(sel(fde), rd).nx
    end

    @testset "An asset that one feature part cannot read leaves the fit" begin
        rdd = blend_drop_rd()
        cvg = CoveragePolicy(; alg = ExpireCoverage(; after = 100))
        pe = EmpiricalPrior(; me = SimpleExpectedReturns(; cvg = cvg),
                            ce = PortfolioOptimisersCovariance(;
                                                               ce = Covariance(; cvg = cvg)))
        @test isnothing(PO.investable_mask(prior(pe, rdd)))
        fdl = FeatureDistance(; sel = ["f"])
        b = BlendDistance(; des = [rde, fdl])
        cle = ClustersEstimator(; de = b)
        hrp = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; pe = pe),
                                     ser = HierarchicalSeriation(; cle = cle))
        msg = (:info, r"1 asset\(s\) left the investable universe .*\[\"d\"\]")
        res = @test_logs msg match_mode = :any optimise(hrp, rdd)
        @test iszero(res.w[4])
        @test sum(res.w) ≈ 1 rtol = 1e-6
        # A direct call has no entry of a fit, so the feature part refuses the asset.
        @test_throws "\"d\"" clusterise(cle, rdd)
        # The mask composes the parts in turn, and a part with weight zero drops no asset.
        @test PO.feature_readable_mask(b, nothing, rdd) == BitVector([1, 1, 1, 0, 1])
        @test PO.feature_readable_mask(BlendDistance(; des = [rde, fdl], w = [1, 0]),
                                       nothing, rdd) === nothing
        @test PO.feature_readable_mask(BlendDistance(;
                                                     des = [BlendDistance(; des = [fdl])]),
                                       nothing, rdd) == BitVector([1, 1, 1, 0, 1])
    end
end
