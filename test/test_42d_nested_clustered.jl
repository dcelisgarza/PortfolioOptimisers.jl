@testset "Nested Clustered Optimisation states its mathematics" begin
    using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Statistics, Clarabel

    #=
    The sweep of `src/17_Optimisation/06_Meta/02_NestedClustered.jl` (#791). Each testset
    asserts one claim that the docstrings of that file state, with numbers:

    - The weights are `W v`: an inverse-volatility oracle written from the definition gives
      the inner and the outer weights, and the synthetic returns `X W` carry the reduced
      moments `W' mu` and `W' Sigma W`.
    - The outer optimiser names a cluster by its synthetic asset, through either shape of
      outer asset sets, and the caller's sets do not change.
    - The weight bounds of the meta-optimiser bind the final weights, and an infeasible
      bound names the finalisation stage.
    - `set_retcode` keeps the expanded weights of a result whose universe dropped an asset.
    - Every refusal reads a precomputed result directly, in a vector and in a
      `TimeDependent` schedule. A schedule of precomputed constraints reached a cluster
      and threw `DimensionMismatch` before the constructor read schedules.
    =#

    PO = PortfolioOptimisers
    rng = StableRNG(987654321)
    T, N = 300, 8
    X = randn(rng, T, N) ./ 100 .+ 0.0005
    nx = string.('A':'H')
    rd = ReturnsResult(; nx = nx, X = X)
    seq = PO.FLoops.SequentialEx()
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false),
                 check_sol = (; allow_local = true, allow_almost = true))
    nco = NestedClustered(; opti = InverseVolatility(), opto = InverseVolatility(),
                          ex = seq)
    r = optimise(nco, rd)
    K = r.clr.k
    cls = [findall(==(k), PO.assignments(r.clr)) for k in 1:K]

    @testset "the weights are W v, and the synthetic returns carry the reduced moments" begin
        @test K >= 2
        @test sort(reduce(vcat, cls)) == 1:N
        W = zeros(N, K)
        for k in 1:K
            W[cls[k], k] = r.resi[k].w
        end
        # Inverse volatility inside each cluster, written from the definition.
        sd = vec(std(X; dims = 1))
        Wo = zeros(N, K)
        for k in 1:K
            v = inv.(sd[cls[k]])
            Wo[cls[k], k] = v / sum(v)
        end
        @test W ≈ Wo
        Xo = X * W
        @test r.reso.pr.X ≈ Xo
        vo = inv.(vec(std(Xo; dims = 1)))
        @test r.reso.w ≈ vo / sum(vo)
        @test r.w ≈ W * r.reso.w
        @test sum(r.w) ≈ 1
        # The sample moments of X W are the reduced moments of the clusters.
        @test vec(mean(Xo; dims = 1)) ≈ transpose(W) * vec(mean(X; dims = 1))
        @test cov(Xo) ≈ transpose(W) * cov(X) * W
        @test r.retcode isa OptimisationSuccess
    end

    @testset "the outer optimiser names a cluster by its synthetic asset" begin
        osets = UniverseSets(; dict = Dict("nx" => ["X", "Y"]))
        # An outer optimiser whose `opt` holds the sets.
        opto = MeanRisk(;
                        opt = JuMPOptimiser(; slv = slv, sets = osets,
                                            lcse = LinearConstraintEstimator(;
                                                                             val = ["_2 >= 0.8"])))
        r1 = optimise(NestedClustered(; opti = InverseVolatility(), opto = opto, ex = seq),
                      rd)
        @test r1.retcode isa OptimisationSuccess
        @test r1.reso.w[2] >= 0.8 - 1e-6
        @test isapprox(r1.reso.w[2], 0.8; atol = 1e-4)
        # An outer optimiser that holds the sets itself.
        opto = InverseVolatility(; sets = osets,
                                 wb = WeightBoundsEstimator(; lb = ["_1" => 0.7],
                                                            ub = nothing))
        r2 = optimise(NestedClustered(; opti = InverseVolatility(), opto = opto, ex = seq),
                      rd)
        @test r2.retcode isa OptimisationSuccess
        @test r2.reso.w[1] ≈ 0.7
        W2 = zeros(N, K)
        for k in 1:K
            W2[cls[k], k] = r2.resi[k].w
        end
        @test r2.w ≈ W2 * r2.reso.w
        # The caller's sets do not change.
        @test osets.dict["nx"] == ["X", "Y"]
        # Sets that already hold the vector of the synthetic names are kept as they are.
        rdo = ReturnsResult(; nx = ["_1", "_2"], X = X[:, 1:2])
        same = UniverseSets(; dict = Dict("nx" => rdo.nx))
        n3 = NestedClustered(; opti = InverseVolatility(),
                             opto = InverseVolatility(; sets = same))
        @test PO._update_asset_sets(n3, rdo) === n3
        @test PO._update_asset_sets(nco, rdo) === nco
    end

    @testset "the weight bounds of the meta-optimiser bind the final weights" begin
        rb = optimise(NestedClustered(; opti = InverseVolatility(),
                                      opto = InverseVolatility(),
                                      wb = WeightBounds(; lb = 0.0, ub = 0.128), ex = seq),
                      rd)
        @test maximum(r.w) > 0.128
        @test rb.retcode isa OptimisationSuccess
        @test maximum(rb.w) ≈ 0.128
        @test sum(rb.w) ≈ 1
        # Eight assets at most 0.12 each cannot sum to one.
        rf = optimise(NestedClustered(; opti = InverseVolatility(),
                                      opto = InverseVolatility(),
                                      wb = WeightBounds(; lb = 0.0, ub = 0.12), ex = seq),
                      rd)
        @test rf.retcode isa OptimisationFailure
        @test rf.retcode.res.msg == "weight bounds finalisation failed.\n"
    end

    @testset "set_retcode keeps the expanded weights" begin
        Xn = copy(X)
        Xn[:, 3] .= NaN
        rn = optimise(nco, ReturnsResult(; nx = nx, X = Xn))
        @test rn.imsk == [true, true, false, true, true, true, true, true]
        @test length(rn.w) == N
        @test iszero(rn.w[3])
        @test sum(rn.w) ≈ 1
        rs = PO.set_retcode(rn, OptimisationFailure())
        @test rs.retcode isa OptimisationFailure
        @test rs.w === rn.w
        @test rs.imsk === rn.imsk
        @test PO.result_investable_mask(rs) === rn.imsk
    end

    @testset "holds_precomputed reads a slot in each shape" begin
        lc = linear_constraints(LinearConstraintEstimator(; val = ["A >= 0.05"]),
                                UniverseSets(; dict = Dict("nx" => nx)))
        lce = LinearConstraintEstimator(; val = ["A >= 0.05"])
        LC = LinearConstraint
        @test PO.holds_precomputed(lc, LC)
        @test !PO.holds_precomputed(lce, LC)
        @test !PO.holds_precomputed(nothing, LC)
        @test PO.holds_precomputed([lce, lc], LC)
        @test !PO.holds_precomputed([lce, lce], LC)
        @test PO.holds_precomputed(TimeDependent([lce, lc]), LC)
        @test PO.holds_precomputed(TimeDependent([lce, lce]; default = lc), LC)
        @test PO.holds_precomputed(TimeDependent([lce, [lce, lc]]), LC)
        @test !PO.holds_precomputed(TimeDependent([lce, lce]), LC)
        # A callable schedule has no value until it runs, so only its default is read.
        @test !PO.holds_precomputed(TimeDependent(ctx -> lc), LC)
        @test PO.holds_precomputed(TimeDependent(ctx -> lce; default = lc), LC)
    end

    @testset "the refusals read a schedule" begin
        lc = linear_constraints(LinearConstraintEstimator(; val = ["A >= 0.05"]),
                                UniverseSets(; dict = Dict("nx" => nx)))
        inner(lcse) = MeanRisk(; opt = JuMPOptimiser(; slv = slv, lcse = lcse))
        @test_throws ArgumentError NestedClustered(; opti = inner(lc),
                                                   opto = InverseVolatility())
        # A schedule of precomputed constraints is refused at construction too.
        @test_throws ArgumentError NestedClustered(; opti = inner(TimeDependent([lc, lc])),
                                                   opto = InverseVolatility())
        @test_throws ArgumentError NestedClustered(;
                                                   opti = inner(TimeDependent([lc, lc];
                                                                              default = lc)),
                                                   opto = InverseVolatility())
        # So is a schedule of clustering results in a nested clustering optimiser.
        hrp = HierarchicalRiskParity(;
                                     opt = HierarchicalOptimiser(;
                                                                 cle = TimeDependent([r.clr,
                                                                                      r.clr])))
        @test_throws ArgumentError NestedClustered(; opti = hrp, opto = InverseVolatility())
        @test_throws ArgumentError NestedClustered(; opti = InverseVolatility(),
                                                   opto = NestedClustered(;
                                                                          cle = TimeDependent([r.clr]),
                                                                          opti = InverseVolatility(),
                                                                          opto = InverseVolatility()))
        # The same optimisers pass with estimators in those slots.
        @test NestedClustered(; opti = HierarchicalRiskParity(),
                              opto = InverseVolatility()) isa NestedClustered
        cv = OptimisationCrossValidation(; cv = KFold(; n = 2))
        @test NestedClustered(; opti = InverseVolatility(),
                              opto = NestedClustered(; opti = InverseVolatility(),
                                                     opto = InverseVolatility(), cv = cv)) isa
              NestedClustered
        # A precomputed risk contribution constraint of a variance in a vector of measures.
        mr(rs) = MeanRisk(; r = rs, opt = JuMPOptimiser(; slv = slv))
        @test NestedClustered(; opti = mr([Variance(), ConditionalValueatRisk()]),
                              opto = InverseVolatility()) isa NestedClustered
        @test_throws ArgumentError NestedClustered(;
                                                   opti = mr([Variance(; rc = lc),
                                                              ConditionalValueatRisk()]),
                                                   opto = InverseVolatility())
        # The constructor refuses a stated basis in a schedule of the outer lcse slot.
        F = X[:, 1:2] * [1.0 0.5; 0.2 1.0] .+ randn(StableRNG(5), T, 2) ./ 1000
        rdf = ReturnsResult(; nx = nx, X = X, nf = ["f1", "f2"], F = F)
        rr = regression(StepwiseRegression(), rdf)
        @test PO.stated_constraint_space_basis(TimeDependent([FactorSpace(; re = rr)]))
        @test !PO.stated_constraint_space_basis(TimeDependent([FactorSpace()]))
        # A precomputed regression in a factor risk budget of the outer optimiser.
        rb(rba) = RiskBudgeting(; opt = JuMPOptimiser(; slv = slv), rba = rba)
        @test_throws ArgumentError NestedClustered(; opti = InverseVolatility(),
                                                   opto = rb(FactorRiskBudgeting(; re = rr)))
        @test_throws ArgumentError NestedClustered(; opti = InverseVolatility(),
                                                   opto = rb(TimeDependent([FactorRiskBudgeting(;
                                                                                                re = rr)])))
        @test NestedClustered(; opti = InverseVolatility(),
                              opto = rb(FactorRiskBudgeting())) isa NestedClustered
    end

    @testset "an estimator in wb or fees needs sets" begin
        @test_throws IsNothingError NestedClustered(; opti = InverseVolatility(),
                                                    opto = InverseVolatility(),
                                                    wb = WeightBoundsEstimator(;
                                                                               lb = ["A" =>
                                                                                         0.1],
                                                                               ub = nothing))
        @test_throws IsNothingError NestedClustered(; opti = InverseVolatility(),
                                                    opto = InverseVolatility(),
                                                    fees = FeesEstimator(;
                                                                         l = ["A" => 0.1]))
    end

    @testset "a static optimiser passes through the fold resolution" begin
        ctx = TimeDependentContext(; i = 2, n = 2, rd = rd, train_idx = [1:20, 1:40],
                                   test_idx = [21:40, 41:60])
        @test !PO.is_time_dependent(nco)
        @test PO.update_time_dependent_estimator(nco, ctx) === nco
    end
end
