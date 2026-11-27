@testset "The seriation slot of HRP and Schur (#2174)" begin
    using PortfolioOptimisers, Test, StableRNGs, LinearAlgebra, Clustering
    PO = PortfolioOptimisers
    rng = StableRNG(2174)
    X = randn(rng, 200, 8) * randn(rng, 8, 8) / 100
    rd = ReturnsResult(; nx = ["A$i" for i in 1:8], X = X)
    clr = clusterise(ClustersEstimator(), X)

    @testset "The constructors refuse what they cannot hold" begin
        @test_throws ArgumentError FixedSeriationResult(; order = [1, 1, 2])
        @test_throws ArgumentError SpectralSeriationResult(; order = [1, 1],
                                                           coordinates = [0.0, 1.0],
                                                           multiplicity = 1, gap = NaN)
        @test_throws DimensionMismatch SpectralSeriationResult(; order = [1, 2],
                                                               coordinates = [0.0],
                                                               multiplicity = 1, gap = NaN)
        @test_throws DomainError SpectralSeriationResult(; order = [1], coordinates = [0.0],
                                                         multiplicity = -1, gap = NaN)
        @test_throws ArgumentError SpectralSeriationState(; coordinates = [0.0, 1.0],
                                                          order = [3])
        @test_throws ArgumentError SpectralSeriationState(; coordinates = [0.0, NaN],
                                                          order = [1, 2])
        @test_throws ArgumentError SpectralSeriationState(; coordinates = [0.0, 1.0],
                                                          order = [1])
        @test_throws TypeError HierarchicalRiskParity(; ser = ClustersEstimator())
        @test_throws TypeError SchurComplementHierarchicalRiskParity(; ser = clr)
        hres = HierarchicalSeriationResult(; clr = clr)
        @test hres.order === clr.res.order
    end

    @testset "A seriation result is used as it is, over the universe that it states" begin
        h0 = optimise(HierarchicalRiskParity(), rd)
        hres = HierarchicalSeriationResult(; clr = clr)
        @test seriate(hres, X) === hres
        @test optimise(HierarchicalRiskParity(; ser = hres), rd).w ≈ h0.w
        @test optimise(HierarchicalRiskParity(; ser = HierarchicalSeriation(; cle = clr)),
                       rd).w ≈ h0.w
        @test optimise(HierarchicalRiskParity(;
                                              ser = FixedSeriationResult(;
                                                                         order = h0.ser.order)),
                       rd).w ≈ h0.w
        @test_throws DimensionMismatch PO.assert_seriation_universe(FixedSeriationResult(;
                                                                                         order = [2,
                                                                                                  1]),
                                                                    3)
        @test_throws DimensionMismatch PO.assert_seriation_universe(hres, 3)
    end

    @testset "A view keeps the assets of the view in the stated order" begin
        f = FixedSeriationResult(; order = [3, 1, 4, 2])
        @test PO.port_opt_view(f, [2, 4]).order == [2, 1]
        st = SpectralSeriationState(; coordinates = [0.1, NaN, -0.2, 0.3],
                                    order = [3, 1, 4])
        v = PO.port_opt_view(st, [1, 2, 4])
        @test v.order == [1, 3]
        @test isequal(v.coordinates, [0.1, NaN, 0.3])
        @test copy(st).order == st.order
        @test_throws ArgumentError PO.merge_states(st, st)
        sr = SpectralSeriationResult(; order = [3, 1, 2], coordinates = [0.0, 0.5, -0.5],
                                     multiplicity = 1, gap = 0.2)
        vr = PO.port_opt_view(sr, [1, 2])
        @test vr.order == [1, 2]
        @test vr.coordinates == [0.0, 0.5]
        @test vr.gap == 0.2
        rev = FixedSeriationResult(; order = collect(8:-1:1))
        @test PO.port_opt_view(HierarchicalRiskParity(; ser = rev), [2, 5, 7], X).ser.order ==
              [3, 2, 1]
        @test PO.port_opt_view(SchurComplementHierarchicalRiskParity(; ser = rev),
                               [2, 5, 7], X).ser.order == [3, 2, 1]
    end

    @testset "NestedClustered refuses a tree that a view cannot slice" begin
        @test_throws ArgumentError PO.assert_internal_optimiser(HierarchicalRiskParity(;
                                                                                       ser = HierarchicalSeriationResult(;
                                                                                                                         clr = clr)))
        @test_throws ArgumentError PO.assert_internal_optimiser(SchurComplementHierarchicalRiskParity(;
                                                                                                      ser = HierarchicalSeriation(;
                                                                                                                                  cle = clr)))
        @test_throws ArgumentError PO.assert_internal_optimiser(HierarchicalEqualRiskContribution(;
                                                                                                  cle = clr))
        @test isnothing(PO.assert_internal_optimiser(HierarchicalRiskParity(;
                                                                            ser = FixedSeriationResult(;
                                                                                                       order = collect(1:8)))))
        @test isnothing(PO.assert_internal_optimiser(HierarchicalRiskParity(;
                                                                            ser = SpectralSeriation())))
        # A stated order over the universe orders the assets of each cluster.
        nco = NestedClustered(;
                              opti = HierarchicalRiskParity(;
                                                            ser = FixedSeriationResult(;
                                                                                       order = collect(1:8))),
                              opto = HierarchicalRiskParity())
        @test sum(optimise(nco, rd).w) ≈ 1
    end

    @testset "The readers of a seriation" begin
        @test PO.plain_graph_reader(HierarchicalSeriation()) ==
              PO.plain_graph_reader(ClustersEstimator())
        @test PO.plain_graph_reader(SpectralSeriation()) ==
              PO.plain_graph_reader(ClustersEstimator())
        @test !PO.plain_graph_reader(SpectralSeriation(; de = FeatureDistance()))
        @test PO.returns_source(SpectralSeriation(; x_src = PriorSource())) isa PriorSource
        @test PO.returns_source(HierarchicalSeriation(;
                                                      cle = ClustersEstimator(;
                                                                              x_src = PriorSource()))) isa
              PriorSource
        @test isnothing(PO.feature_readable_mask(SpectralSeriation(), nothing, rd))
        @test isnothing(PO.feature_readable_mask(HierarchicalSeriation(), nothing, rd))
        @test PO.group_frames(HierarchicalRiskParity()).raw == ["opt.pe", "ser.cle"]
        @test PO.group_frames(SchurComplementHierarchicalRiskParity(;
                                                                    ser = SpectralSeriation())).raw ==
              ["opt.pe", "ser"]
        @test PO.group_frames(HierarchicalEqualRiskContribution()).raw == ["opt.pe", "cle"]
        pr = prior(EmpiricalPrior(), rd)
        @test seriate(SpectralSeriation(), pr; rd = rd).order ==
              seriate(SpectralSeriation(), X).order
        @test_throws MethodError seriate(SpectralSeriation(), pr; rd = rd, branchorder = :r)
    end

    @testset "A Pipeline routes a clustering into the seriation" begin
        hrp = PO.pipe_route(HierarchicalRiskParity(), Val(:cle), clr)
        @test hrp.ser.cle === clr
        @test PO.pipe_accepts(HierarchicalRiskParity(), Val(:cle))
        @test !PO.pipe_accepts(HierarchicalRiskParity(; ser = SpectralSeriation()),
                               Val(:cle))
        @test PO.pipe_route(HierarchicalRiskParity(; ser = SpectralSeriation()), Val(:cle),
                            clr).ser isa SpectralSeriation
        sh = PO.pipe_route(SchurComplementHierarchicalRiskParity(), Val(:cle), clr)
        @test sh.ser.cle === clr
        @test PO.pipe_accepts(SchurComplementHierarchicalRiskParity(), Val(:cle))
        @test PO.pipe_route(HierarchicalEqualRiskContribution(), Val(:cle), clr).cle === clr
        # The frame walk leaves the replaced slot out of the group of the optimiser.
        @test PO.pipe_target_paths(HierarchicalRiskParity(), Val(:cle)) == ["ser.cle"]
        @test PO.pipe_target_paths(SchurComplementHierarchicalRiskParity(), Val(:cle)) ==
              ["ser.cle"]
        @test isempty(PO.pipe_target_paths(HierarchicalRiskParity(;
                                                                  ser = SpectralSeriation()),
                                           Val(:cle)))
        @test PO.pipe_target_paths(HierarchicalEqualRiskContribution(), Val(:cle)) ==
              ["cle"]
    end

    @testset "The memory of a spectral fit" begin
        s1 = partial_fit!(SpectralSeriation(), X)
        @test s1.cache isa SpectralSeriationState
        @test seriate(s1, X).coordinates ≈ s1.cache.coordinates
        @test seriate(s1, X).order == s1.cache.order
        # The generic hook leaves an estimator with no memory unchanged.
        hrp = HierarchicalRiskParity()
        @test PO.carry_fold_state(hrp, nothing) === hrp
        sp = HierarchicalRiskParity(; ser = SpectralSeriation())
        @test PO.carry_fold_state(sp, nothing) === sp
        # A view slices the memory to the assets of the view.
        @test length(PO.port_opt_view(s1, [1, 3], X).cache.coordinates) == 2
        # One asset gives the coordinate zero, and none gives an empty order.
        r1 = PO.spectral_seriation(ones(1, 1))
        @test r1.order == [1]
        @test r1.coordinates == [0.0]
        @test isnan(r1.gap)
        @test isempty(PO.spectral_seriation(zeros(0, 0)).order)
    end

    @testset "A Stacking carries the memory to its inner and outer optimisers (#2396)" begin
        sp = HierarchicalRiskParity(; ser = SpectralSeriation())
        st = Stacking(; pe = Online(EmpiricalPrior()), opti = [sp, InverseVolatility()],
                      opto = sp)
        p = cross_val_predict(st, rd, OnlineIndexWalkForward(100, 50))
        @test length(p.pred) == 2
        c = PO.carry_fold_state(st, p.pred[1])
        @test c.opti[1].ser.cache.order == p.pred[1].res.resi[1].ser.order
        @test c.opto.ser.cache.order == p.pred[1].res.reso.ser.order
        # An optimiser with no memory keeps its identity, and so does a Stacking of them.
        @test c.opti[2] === st.opti[2]
        iv = Stacking(; opti = [InverseVolatility(), EqualWeighted()],
                      opto = InverseVolatility())
        pv = cross_val_predict(iv, rd, IndexWalkForward(100, 50))
        @test PO.carry_fold_state(iv, pv.pred[1]) === iv
        # A result that is not a StackingResult, such as a fallback, takes the generic method.
        @test PO.carry_fold_state(st, nothing) === st
    end

    @testset "A Float32 sample stays in Float32" begin
        rd32 = ReturnsResult(; nx = ["A$i" for i in 1:8], X = Float32.(X))
        @test eltype(optimise(HierarchicalRiskParity(; ser = SpectralSeriation()), rd32).w) ==
              Float32
        @test eltype(optimise(SchurComplementHierarchicalRiskParity(;
                                                                    ser = SpectralSeriation()),
                              rd32).w) == Float32
    end
end
