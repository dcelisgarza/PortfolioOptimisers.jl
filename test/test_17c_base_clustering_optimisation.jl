@testset "The base of the clustering optimisers states its mathematics" begin
    using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Dates
    PO = PortfolioOptimisers

    #=
    The sweep of `src/17_Optimisation/04_Hierarchical/01_Base_ClusteringOptimisation.jl`
    (#879). Each testset asserts one claim that the docstrings of that file state, with
    numbers:

    - `unitary_expected_risks` is ``R(e_i)``: the diagonal of the covariance for a
      `Variance`, the measure of column `i` for a measure that reads the returns, and the
      net series under fees. The in-place form agrees and leaves its scratch vector zero.
      The risks take the type of the risk, so an integer `X` gives `Float64` risks.
    - `assert_clustering_universe` returns `nothing` on a match and raises
      `DimensionMismatch` on a mismatch.
    - The keyword constructor of `HierarchicalResult` expands `w` onto the mask, and the
      positional one stores it as given.
    - A leaf result forwards every property of its core.
    - `HierarchicalOptimiser` carries `cache` unchanged through `factory` and slices it
      through `port_opt_view`; its schedulable defaults are its keyword defaults.
    - The time-dependent methods of the family recurse into the inner optimiser.
    =#
    rng = StableRNG(879)
    T, N = 60, 5
    X = randn(rng, T, N) / 100
    pr = prior(EmpiricalPrior(), X)
    unit(i) = [j == i ? 1.0 : 0.0 for j in 1:N]

    @testset "unitary_expected_risks is the risk of each unit weight vector" begin
        rv = factory(Variance(), pr)
        @test PO.unitary_expected_risks(rv, pr.X) == diag(pr.sigma)

        rc = factory(ConditionalValueatRisk(), pr)
        rk = PO.unitary_expected_risks(rc, pr.X)
        @test rk == [rc(pr.X[:, i]) for i in 1:N]

        # A fee is charged on the unit portfolio, so each entry is the measure of the net
        # series. CVaR is translation invariant, so a long fee moves every entry by itself.
        fees = Fees(; l = 1e-3)
        rkf = PO.unitary_expected_risks(rc, pr.X, fees)
        @test rkf == [rc(PO.calc_net_returns(unit(i), pr.X, fees)) for i in 1:N]
        @test rkf ≈ rk .+ 1e-3

        wk = zeros(N)
        rk2 = fill(NaN, N)
        @test isnothing(PO.unitary_expected_risks!(wk, rk2, rc, pr.X, fees))
        @test rk2 == rkf
        @test all(iszero, wk)

        # The risks take the type of the risk, not the element type of `X`.
        pr32 = prior(EmpiricalPrior(), Float32.(X))
        rk32 = PO.unitary_expected_risks(factory(Variance(), pr32), pr32.X)
        @test eltype(rk32) === Float32
        @test rk32 == diag(pr32.sigma)

        # An integer returns matrix has fractional risks. Before the sweep the risk vector
        # took `eltype(X)` and raised `InexactError`.
        Xi = rand(StableRNG(5), -20:20, T, N)
        pri = prior(EmpiricalPrior(), Xi)
        prf = prior(EmpiricalPrior(), float(Xi))
        for r in (Variance(), ConditionalValueatRisk())
            rki = PO.unitary_expected_risks(factory(r, pri), pri.X)
            @test eltype(rki) === Float64
            @test rki ≈ PO.unitary_expected_risks(factory(r, prf), prf.X)
        end
        wki = zeros(N)
        rki = zeros(N)
        PO.unitary_expected_risks!(wki, rki, factory(Variance(), pri), pri.X)
        @test rki ≈ diag(prf.sigma)
    end

    @testset "assert_clustering_universe refuses a clustering of another universe" begin
        clr = clusterise(ClustersEstimator(), pr)
        @test isnothing(PO.assert_clustering_universe(clr, N))
        @test_throws DimensionMismatch PO.assert_clustering_universe(clr, N - 1)
    end

    imsk = BitVector([true, false, true, true, false])
    wr = [0.2, 0.3, 0.5]
    hk = HierarchicalResult(; pr = nothing, clr = nothing, wb = nothing, fees = nothing,
                            retcode = OptimisationSuccess(), w = wr, imsk = imsk)

    @testset "HierarchicalResult expands in the keyword constructor alone" begin
        @test hk.w == [0.2, 0.0, 0.3, 0.5, 0.0]
        hp = HierarchicalResult(nothing, nothing, nothing, nothing, OptimisationSuccess(),
                                wr, imsk)
        @test hp.w === wr
        hn = HierarchicalResult(; pr = nothing, clr = nothing, wb = nothing, fees = nothing,
                                retcode = OptimisationSuccess(), w = wr)
        @test hn.w === wr
    end

    @testset "A leaf result reads every property of its core" begin
        for res in
            (HierarchicalRiskParityResult(; hr = hk, r = Variance(), sca = SumScalariser(),
                                          fb = nothing),
             HierarchicalEqualRiskContributionResult(; hr = hk, ri = Variance(),
                                                     ro = Variance(),
                                                     scai = SumScalariser(),
                                                     scao = SumScalariser(), fb = nothing))
            for p in (:pr, :clr, :wb, :fees, :retcode, :w, :imsk)
                @test getproperty(res, p) === getproperty(hk, p)
            end
            @test PO.result_investable_mask(res) === imsk
        end
    end

    @testset "HierarchicalOptimiser carries its cache and states its defaults once" begin
        rd = ReturnsResult(; nx = ["A", "B", "C", "D", "E"], X = X,
                           ts = collect(Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(T - 1))))
        st = partial_fit!(PO.ReturnsBufferState(), rd)
        ho = HierarchicalOptimiser(; cache = st)
        @test factory(ho, fill(0.2, N)).cache === st
        @test PO.port_opt_view(ho, [1, 3]).cache.nx == ["A", "C"]

        h0 = HierarchicalOptimiser()
        d = PO.hierarchical_optimiser_td_defaults()
        @test keys(d) == (:pe, :cle, :wb, :wf)
        for k in keys(d)
            @test typeof(getfield(h0, k)) == typeof(d[k])
        end
        @test PO.time_dependent_field_defaults(h0) === d

        @test !PO.needs_previous_weights(h0)
        tn = Turnover(; w = fill(0.2, N), val = 0.01)
        @test PO.needs_previous_weights(HierarchicalOptimiser(; fees = Fees(; tn = tn)))
    end

    @testset "The time-dependent methods recurse into the inner optimiser" begin
        rd = ReturnsResult(; nx = ["A", "B", "C", "D", "E"], X = X,
                           ts = collect(Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(T - 1))))
        wbs = [WeightBounds(; lb = 0.0, ub = 0.5), WeightBounds(; lb = 0.0, ub = 0.8)]
        hrp = HierarchicalRiskParity(;
                                     opt = HierarchicalOptimiser(; wb = TimeDependent(wbs)))
        @test PO.is_time_dependent(hrp)
        @test !PO.is_time_dependent(HierarchicalRiskParity())

        ctx = TimeDependentContext(; i = 2, n = 2, rd = rd, train_idx = [1:20, 1:40],
                                   test_idx = [21:40, 41:60])
        u = PO.update_time_dependent_estimator(hrp, ctx)
        @test u.opt.wb.ub == 0.8
        @test !PO.is_time_dependent(u)
        @test PO.reset_time_dependent_estimator(hrp).opt.wb.ub == 1.0

        # A schedule of two folds refuses a loop of three.
        @test_throws DimensionMismatch PO.assert_time_dependent_fold_count(hrp, 3)
    end
end
