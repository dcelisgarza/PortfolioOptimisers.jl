@testset "Factor exposure constraints" begin
    using PortfolioOptimisers, Test, Logging, LinearAlgebra, Clarabel, StableRNGs
    # Hand-built loadings and universes throughout: the point of these tests is that the row
    # that comes out is `M * a`, so anything estimated would obscure the arithmetic.
    #
    #        MTUM  VLUE  QUAL
    #   A     1.0   0.0   0.2
    #   B     0.5   0.5   0.0
    #   C     0.0   1.0   0.7
    M = [1.0 0.0 0.2; 0.5 0.5 0.0; 0.0 1.0 0.7]
    rr = Regression(; M = M)
    sets = UniverseSets(;
                        dict = Dict("nx" => ["A", "B", "C"],
                                    "nf" => ["MTUM", "VLUE", "QUAL"],
                                    "defensive" => ["VLUE", "QUAL"]))
    fs = FactorSpace()
    @testset "Type structure" begin
        @test FactorSpace <: PortfolioOptimisers.AbstractConstraintSpace
        @test PortfolioOptimisers.AbstractConstraintSpace <:
              PortfolioOptimisers.AbstractAlgorithm
        # ADR 0047: the asset frame is the *absence* of a re-basis, so there is no
        # `AssetSpace` to write down. Anything else in the family is a real second basis.
        @test !isdefined(PortfolioOptimisers, :AssetSpace)
        # `space` is required: a re-basis is asked for, never inferred.
        @test_throws UndefKeywordError ExposureConstraintEstimator(;
                                                                   lce = LinearConstraintEstimator(;
                                                                                                   val = "MTUM <= 0.3"))
        @test_throws UndefKeywordError ExposureConstraintEstimator(; space = fs)
        # The bound is exactly what `lcse` accepts, so no shape reaches it un-re-based.
        @test_throws TypeError ExposureConstraintEstimator(; lce = "MTUM <= 0.3",
                                                           space = fs)
        @test_throws TypeError ExposureConstraintEstimator(;
                                                           lce = LinearConstraintEstimator(;
                                                                                           val = "MTUM <= 0.3"),
                                                           space = "factor")
        @test_throws PortfolioOptimisers.IsEmptyError ExposureConstraintEstimator(;
                                                                                  lce = LinearConstraintEstimator[],
                                                                                  space = fs)
        # `LinearConstraintEstimator` is left exactly as it was — being unmarked is what
        # keeps it the only thing `gcarde`/`sgcarde` admit.
        @test fieldnames(LinearConstraintEstimator) == (:val, :key)
        ece = ExposureConstraintEstimator(;
                                          lce = LinearConstraintEstimator(;
                                                                          val = "MTUM <= 0.3"),
                                          space = fs)
        @test fieldnames(typeof(ece)) == (:lce, :space)
        @test isa(ece.space, FactorSpace)
    end
    @testset "Projection: the row is M * a" begin
        # A single factor: the row is that factor's loadings column.
        lcr = linear_constraints(ExposureConstraintEstimator(;
                                                             lce = LinearConstraintEstimator(;
                                                                                             val = "MTUM <= 0.3"),
                                                             space = fs), sets; rr = rr)
        @test isa(lcr, LinearConstraint)
        @test isnothing(lcr.eq)
        @test lcr.ineq.A == transpose(reshape(M[:, 1], :, 1))
        @test lcr.ineq.B == [0.3]
        # The whole point, stated as the identity it is: for any weights, the projected row
        # evaluates to the same number as the factor row applied to `M' * w`.
        w = [0.2, 0.5, 0.3]
        a = [1.0, 0.0, 0.0]
        @test only(lcr.ineq.A * w) ≈ dot(a, transpose(M) * w)
        # Equality, and a multi-term row with coefficients.
        lcr = linear_constraints(ExposureConstraintEstimator(;
                                                             lce = LinearConstraintEstimator(;
                                                                                             val = "2*MTUM - VLUE == 0.1"),
                                                             space = fs), sets; rr = rr)
        @test isnothing(lcr.ineq)
        a = [2.0, -1.0, 0.0]
        @test vec(lcr.eq.A) ≈ M * a
        @test lcr.eq.B == [0.1]
        # `>=` flips sign on both sides, exactly as on the asset axis.
        lcr = linear_constraints(ExposureConstraintEstimator(;
                                                             lce = LinearConstraintEstimator(;
                                                                                             val = "QUAL >= 0.25"),
                                                             space = fs), sets; rr = rr)
        @test vec(lcr.ineq.A) ≈ -M[:, 3]
        @test lcr.ineq.B == [-0.25]
        # Both kinds in one estimator.
        lcr = linear_constraints(ExposureConstraintEstimator(;
                                                             lce = LinearConstraintEstimator(;
                                                                                             val = ["MTUM <= 0.3",
                                                                                                    "VLUE == 0.0"]),
                                                             space = fs), sets; rr = rr)
        @test vec(lcr.ineq.A) ≈ M[:, 1]
        @test vec(lcr.eq.A) ≈ M[:, 2]
    end
    @testset "Group expansion is axis-blind" begin
        # Characterisation: `replace_group_by_assets` expands any name found in `dict` and
        # never asks which axis it is on, so a factor group needs no machinery of its own.
        lcr = linear_constraints(ExposureConstraintEstimator(;
                                                             lce = LinearConstraintEstimator(;
                                                                                             val = "defensive >= 0.2"),
                                                             space = fs), sets; rr = rr)
        @test vec(lcr.ineq.A) ≈ -(M[:, 2] + M[:, 3])
        @test lcr.ineq.B == [-0.2]
        # Same answer written out longhand.
        lng = linear_constraints(ExposureConstraintEstimator(;
                                                             lce = LinearConstraintEstimator(;
                                                                                             val = "VLUE + QUAL >= 0.2"),
                                                             space = fs), sets; rr = rr)
        @test lng.ineq.A == lcr.ineq.A
    end
    @testset "Duplicate factor names sum their columns" begin
        # The projection sums `view(M, :, Ai)` rather than indexing `findfirst`, so a factor
        # universe carrying a name twice contributes both columns — matching how the asset
        # path treats a duplicated asset name.
        setsd = UniverseSets(; dict = Dict("nx" => ["A", "B"], "nf" => ["MTUM", "MTUM"]))
        Md = [1.0 2.0; 3.0 4.0]
        lcr = linear_constraints(ExposureConstraintEstimator(;
                                                             lce = LinearConstraintEstimator(;
                                                                                             val = "MTUM == 1"),
                                                             space = fs), setsd;
                                 rr = Regression(; M = Md))
        @test vec(lcr.eq.A) == [3.0, 7.0]
        @test lcr.eq.B == [1.0]
    end
    @testset "Precomputed LinearConstraint projects wholesale" begin
        # The one place `linear_constraints(::LinearConstraint, …)` stops being a
        # pass-through: a precomputed constraint written in the wrapped basis must still be
        # re-based.
        pc = LinearConstraint(;
                              ineq = PartialLinearConstraint(;
                                                             A = [1.0 0.0 0.0;
                                                                  0.0 1.0 1.0],
                                                             B = [0.3, 0.5]),
                              eq = PartialLinearConstraint(; A = [0.0 0.0 1.0], B = [0.1]))
        lcr = linear_constraints(ExposureConstraintEstimator(; lce = pc, space = fs), sets;
                                 rr = rr)
        @test lcr.ineq.A ≈ pc.ineq.A * transpose(M)
        @test lcr.ineq.B == pc.ineq.B          # the bound is untouched by a change of basis
        @test lcr.eq.A ≈ pc.eq.A * transpose(M)
        @test lcr.eq.B == pc.eq.B
        # Nothing else would catch an asset-space matrix handed to a factor-space wrapper,
        # because a precomputed constraint carries no names.
        bad = LinearConstraint(; ineq = PartialLinearConstraint(; A = [1.0 0.0], B = [0.3]))
        @test_throws DimensionMismatch linear_constraints(ExposureConstraintEstimator(;
                                                                                      lce = bad,
                                                                                      space = fs),
                                                          sets; rr = rr)
        # An asset-space precomputed constraint reaching `lcse` alongside loadings is still
        # a pass-through: only the decorator asks for a re-basis.
        @test linear_constraints(pc, sets; rr = rr) === pc
    end
    @testset "Vectors, and mixing bases" begin
        vece = ExposureConstraintEstimator(;
                                           lce = [LinearConstraintEstimator(;
                                                                            val = "MTUM <= 0.3"),
                                                  LinearConstraintEstimator(;
                                                                            val = "VLUE == 0.1")],
                                           space = fs)
        v = linear_constraints(vece, sets; rr = rr)
        @test length(v) == 2
        @test vec(v[1].ineq.A) ≈ M[:, 1]
        @test vec(v[2].eq.A) ≈ M[:, 2]
        # A vector may mix a re-based constraint with an asset-space one; each resolves
        # against its own axis.
        ece = ExposureConstraintEstimator(;
                                          lce = LinearConstraintEstimator(;
                                                                          val = "MTUM <= 0.3"),
                                          space = fs)
        mixed = PortfolioOptimisers.EcE_LcE_Lc[LinearConstraintEstimator(;
                                                                         val = "A >= 0.1"),
                                               ece]
        mx = linear_constraints(mixed, sets; rr = rr)
        @test length(mx) == 2
        @test vec(mx[1].ineq.A) == [-1.0, 0.0, 0.0]   # "A" is an asset, not a factor
        @test vec(mx[2].ineq.A) ≈ M[:, 1]
        # A bare estimator accepts the loadings and drops them: the space is what the user
        # wrote, never what the prior happened to carry.
        bare = LinearConstraintEstimator(; val = "A >= 0.1")
        @test linear_constraints(bare, sets; rr = rr).ineq.A ==
              linear_constraints(bare, sets).ineq.A
        # Homogeneous vectors of bare estimators still take the narrower method, which now
        # accepts and forwards the loadings its elements go on to drop.
        vb = linear_constraints([bare, bare], sets; rr = rr)
        @test length(vb) == 2
        @test all(x -> x.ineq.A == linear_constraints(bare, sets).ineq.A, vb)
    end
    @testset "The wrapped key overrides the factor axis" begin
        # `key === nothing` resolves to `sets.fkey` rather than `sets.xkey`; a key written on
        # the wrapped estimator still wins, so an alternative factor partition can be named.
        setsk = UniverseSets(;
                             dict = Dict("nx" => ["A", "B", "C"],
                                         "nf" => ["MTUM", "VLUE", "QUAL"],
                                         "nf_alias" => ["mom", "val", "qual"]))
        lcr = linear_constraints(ExposureConstraintEstimator(;
                                                             lce = LinearConstraintEstimator(;
                                                                                             val = "val <= 0.3",
                                                                                             key = "nf_alias"),
                                                             space = fs), setsk; rr = rr)
        @test vec(lcr.ineq.A) ≈ M[:, 2]
    end
    @testset "Failure modes" begin
        ece = ExposureConstraintEstimator(;
                                          lce = LinearConstraintEstimator(;
                                                                          val = "MTUM <= 0.3"),
                                          space = fs)
        # Missing loadings throw *regardless of `strict`*. `strict` governs unknown names — a
        # per-row, recoverable condition. A missing regression makes every row unbuildable,
        # and dropping them silently yields a portfolio carrying none of the exposure asked
        # for.
        @test_throws PortfolioOptimisers.IsNothingError linear_constraints(ece, sets;
                                                                           strict = false)
        @test_throws PortfolioOptimisers.IsNothingError linear_constraints(ece, sets;
                                                                           strict = true)
        @test_throws PortfolioOptimisers.IsNothingError linear_constraints(ece, sets;
                                                                           rr = nothing)
        # The diagnosis is the library's shared one, so a user sees the same remedy here as
        # from an estimator that needed a factor block.
        msg = try
            linear_constraints(ece, sets)
        catch e
            sprint(showerror, e)
        end
        @test occursin(PortfolioOptimisers.prior_regression_remedy, msg)
        # The factor axis is optional on `UniverseSets`; it is not optional for a constraint
        # written against it.
        setsx = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"]))
        @test_throws KeyError linear_constraints(ece, setsx; rr = rr)
        # The loadings and the declared axis must agree on how many factors there are.
        sets2 = UniverseSets(;
                             dict = Dict("nx" => ["A", "B", "C"], "nf" => ["MTUM", "VLUE"]))
        @test_throws DimensionMismatch linear_constraints(ece, sets2; rr = rr)
    end
    @testset "Diagnostics name the factor axis" begin
        # An unknown *name* keeps `strict` semantics, but the message must name the universe
        # the user was writing in — reporting the asset universe would send them looking in
        # the wrong place.
        ecu = ExposureConstraintEstimator(;
                                          lce = LinearConstraintEstimator(;
                                                                          val = "MTUMM <= 0.3"),
                                          space = fs)
        msg = try
            linear_constraints(ecu, sets; rr = rr, strict = true)
        catch e
            sprint(showerror, e)
        end
        @test occursin("not in factor universe", msg)
        @test occursin("3 factors under key `nf`", msg)
        @test occursin("did you mean `MTUM`?", msg)
        # Non-strict warns and drops, then reports the row as empty against the factor axis.
        logs, _ = Test.collect_test_logs() do
            return linear_constraints(ExposureConstraintEstimator(;
                                                                  lce = LinearConstraintEstimator(;
                                                                                                  val = "ZZZ <= 0.3"),
                                                                  space = fs), sets;
                                      rr = rr, strict = false)
        end
        msgs = [l.message for l in logs]
        @test any(m -> occursin("not in factor universe", m), msgs)
        @test any(m -> occursin("matched no factors in the universe", m), msgs)
        # A row whose names *did* resolve but whose loadings annihilate it is a different
        # failure, and says so: reporting "matched no factors" would send a user hunting for
        # a typo that is not there.
        rrz = Regression(; M = [0.0 0.0 0.2; 0.0 0.5 0.0; 0.0 1.0 0.7])
        msg = try
            linear_constraints(ExposureConstraintEstimator(;
                                                           lce = LinearConstraintEstimator(;
                                                                                           val = "MTUM <= 0.3"),
                                                           space = fs), sets; rr = rrz,
                               strict = true)
        catch e
            sprint(showerror, e)
        end
        @test occursin("resolved against the factor universe", msg)
        @test occursin("every matched factor has zero loadings", msg)
        # The asset path's messages are untouched.
        msg = try
            linear_constraints(LinearConstraintEstimator(; val = "ZZZ <= 0.3"), sets;
                               strict = true)
        catch e
            sprint(showerror, e)
        end
        @test occursin("not in asset universe", msg)
        @test occursin("3 assets under key `nx`", msg)
    end
    @testset "Reduced loadings are not what a constraint reads" begin
        # `M` is the loadings over the *named original* factors; `L` is the same information
        # in the reduced basis, whose columns are principal components and cannot be named in
        # an equation. A constraint must be written, so it reads `M`. Risk decomposition
        # reads `L` and is right to — ADR 0047 says not to harmonise them.
        rrl = Regression(; M = M, L = [1.0 0.0; 0.0 1.0; 1.0 1.0])
        lcr = linear_constraints(ExposureConstraintEstimator(;
                                                             lce = LinearConstraintEstimator(;
                                                                                             val = "MTUM <= 0.3"),
                                                             space = fs), sets; rr = rrl)
        @test vec(lcr.ineq.A) ≈ M[:, 1]
        # The axis check counts `M`'s columns, not `L`'s: `L` has two here and the constraint
        # still builds against a three-factor axis.
        @test size(rrl.L, 2) != length(sets.dict[sets.fkey])
    end
    # A tiny factor market for the optimiser tests: the assets are combinations of two
    # factors plus noise, so a fitted `M` is close to `Mo` without being it. Every assertion
    # reads the *fitted* loadings off the result — the claim is that the realised exposure
    # is the one the row was written against, not that the regression recovers `Mo`.
    orng = StableRNG(987654321)
    Fo = randn(orng, 400, 2) * 0.01
    Mo = [1.0 0.0; 0.5 0.5; 0.0 1.0]
    Xo = Fo * transpose(Mo) + randn(orng, 400, 3) * 0.001
    rdo = ReturnsResult(; nx = ["A", "B", "C"], X = Xo, nf = ["MTUM", "VLUE"], F = Fo)
    osets = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"], "nf" => ["MTUM", "VLUE"]))
    oslv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = "verbose" => false)
    oece = ExposureConstraintEstimator(;
                                       lce = LinearConstraintEstimator(;
                                                                       val = "MTUM <= 0.3"),
                                       space = fs)
    @testset "`lcse` is the only slot that admits a re-basis" begin
        @test JuMPOptimiser(; slv = oslv, sets = osets, lcse = oece) isa JuMPOptimiser
        # Cardinality rows index the binary held-indicators rather than `w`, so a projected
        # row is neither integral nor an index into them. ADR 0047 makes that
        # *unrepresentable* rather than validated: the narrow bound rejects it outright.
        @test_throws TypeError JuMPOptimiser(; slv = oslv, sets = osets, gcarde = oece)
        @test_throws TypeError JuMPOptimiser(; slv = oslv, sets = osets, sgcarde = oece)
        # A vector may mix bases. It needs the explicit element type for the same reason
        # `VecLcE_Lc` does — a heterogeneous literal promotes to the abstract supertype.
        mixed = PortfolioOptimisers.EcE_LcE_Lc[oece,
                                               LinearConstraintEstimator(;
                                                                         val = "A >= 0.05")]
        @test JuMPOptimiser(; slv = oslv, sets = osets, lcse = mixed) isa JuMPOptimiser
        # The names resolve against `sets`, so a re-basis without one fails at construction
        # like every other estimator-typed field. This holds for a wrapped *precomputed*
        # constraint too: `constraint_space_basis` needs the sets to find the factor axis.
        @test_throws PortfolioOptimisers.IsNothingError JuMPOptimiser(; slv = oslv,
                                                                      lcse = oece)
        @test_throws PortfolioOptimisers.IsNothingError JuMPOptimiser(; slv = oslv,
                                                                      lcse = mixed)
    end
    @testset "End to end: the realised exposure is the requested one" begin
        res = optimise(MeanRisk(;
                                opt = JuMPOptimiser(; pe = FactorPrior(), slv = oslv,
                                                    sets = osets, lcse = oece)), rdo)
        Mf = res.pa.pr.rr.M
        # The mandate binds: momentum exposure sits on the cap it was written against.
        @test isapprox(dot(Mf[:, 1], res.w), 0.3; atol = 1e-6)
        # What reached the model is an ordinary asset-space constraint — one row, one column
        # per asset. There is no factor-length row anywhere and no second pathway.
        @test isa(res.lcsr, LinearConstraint)
        @test size(res.lcsr.ineq.A) == (1, length(rdo.nx))
        @test vec(res.lcsr.ineq.A) ≈ Mf[:, 1]
        # Handing the optimiser the projected row directly is the same problem, which is the
        # operational content of "re-basing is a change of coordinates, not a feature".
        hand = LinearConstraint(;
                                ineq = PartialLinearConstraint(; A = transpose(Mf[:, 1:1]),
                                                               B = [0.3]))
        resh = optimise(MeanRisk(;
                                 opt = JuMPOptimiser(; pe = FactorPrior(), slv = oslv,
                                                     sets = osets, lcse = hand)), rdo)
        @test isapprox(res.w, resh.w; rtol = 5e-5)
    end
    @testset "Every JuMP optimiser inherits it" begin
        # They all share `JuMPOptimiser`, so the wiring is one edit rather than one per
        # optimiser.
        for est in (MeanRisk, RiskBudgeting, RelaxedRiskBudgeting)
            res = optimise(est(;
                               opt = JuMPOptimiser(; pe = FactorPrior(), slv = oslv,
                                                   sets = osets, lcse = oece)), rdo)
            @test dot(res.pa.pr.rr.M[:, 1], res.w) <= 0.3 + 1e-6
        end
        # Near Optimal Centering round-trips `lcsr` back into an `lcse` slot through
        # `jump_optimiser_from_attributes`, where the pass-through method for a precomputed
        # `LinearConstraint` keeps it from being projected a second time. Its *constrained*
        # algorithm is the one that carries the linear constraints into the centering model
        # — the default `UnconstrainedNearOptimalCentering` deliberately drops them there,
        # and does so for an asset-space constraint just the same.
        noc = optimise(NearOptimalCentering(; alg = ConstrainedNearOptimalCentering(),
                                            opt = JuMPOptimiser(; pe = FactorPrior(),
                                                                slv = oslv, sets = osets,
                                                                lcse = oece)), rdo)
        @test isapprox(dot(noc.pa.pr.rr.M[:, 1], noc.w), 0.3; atol = 1e-6)
    end
    @testset "Missing loadings throw at the optimiser, ignoring `strict`" begin
        # `strict` governs recoverable per-row name failures. Dropping every row of a factor
        # mandate would yield a feasible, plausible portfolio carrying none of the requested
        # exposure, so this is not that.
        for strict in (false, true)
            mre = MeanRisk(;
                           opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = oslv,
                                               sets = osets, lcse = oece, strict = strict))
            @test_throws PortfolioOptimisers.IsNothingError optimise(mre, rdo)
        end
    end
    @testset "Axis order is checked against the data" begin
        fmr(s) = MeanRisk(;
                          opt = JuMPOptimiser(; pe = FactorPrior(), slv = oslv, sets = s,
                                              lcse = oece))
        # Same factors, wrong order: every row would be attached to the wrong loadings
        # column, and the problem would still solve.
        badf = UniverseSets(;
                            dict = Dict("nx" => ["A", "B", "C"], "nf" => ["VLUE", "MTUM"]))
        @test_throws ArgumentError optimise(fmr(badf), rdo)
        # Same assets, wrong order.
        badx = UniverseSets(;
                            dict = Dict("nx" => ["A", "C", "B"], "nf" => ["MTUM", "VLUE"]))
        @test_throws ArgumentError optimise(fmr(badx), rdo)
        # Different universes entirely is a different diagnosis, and says so.
        shortx = UniverseSets(; dict = Dict("nx" => ["A", "B"], "nf" => ["MTUM", "VLUE"]))
        @test_throws DimensionMismatch optimise(fmr(shortx), rdo)
        # The asset check is new behaviour on a path with nothing to do with factors: it
        # fires with no factor axis declared and no re-basis in sight.
        amr = MeanRisk(;
                       opt = JuMPOptimiser(; slv = oslv,
                                           sets = UniverseSets(;
                                                               dict = Dict("nx" =>
                                                                               ["A", "C",
                                                                                "B"])),
                                           lcse = LinearConstraintEstimator(;
                                                                            val = "A >= 0.05")))
        @test_throws ArgumentError optimise(amr, ReturnsResult(; nx = rdo.nx, X = rdo.X))
        # Only where both sides exist. The factor axis is optional on `UniverseSets`, and an
        # undeclared one is silence, not disagreement.
        nofsets = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"]))
        @test optimise(MeanRisk(;
                                opt = JuMPOptimiser(; pe = FactorPrior(), slv = oslv,
                                                    sets = nofsets)), rdo) isa
              MeanRiskResult
        msg = try
            optimise(fmr(badf), rdo)
        catch e
            sprint(showerror, e)
        end
        @test occursin("factor universe declared under key `nf`", msg)
        @test occursin("first at position 1: `VLUE` vs `MTUM`", msg)
        # Info-leak-safe, as the other constraint messages are: sizes and the first
        # differing pair only, never a universe in full.
        msg = try
            optimise(fmr(badx), rdo)
        catch e
            sprint(showerror, e)
        end
        @test occursin("first at position 2: `C` vs `B`", msg)
        @test !occursin("\"A\"", msg)
    end
    @testset "A schedule of exposure constraints" begin
        # ADR 0030: constraints are problem definition, so `lcse` is `TD_Option` and the
        # widened bound widens with it. Fold two tightens the momentum cap.
        cv = IndexWalkForward(200, 100)
        tight = ExposureConstraintEstimator(;
                                            lce = LinearConstraintEstimator(;
                                                                            val = "MTUM <= 0.05"),
                                            space = fs)
        mre = MeanRisk(;
                       opt = JuMPOptimiser(; pe = FactorPrior(), slv = oslv, sets = osets,
                                           lcse = TimeDependent([oece, tight])))
        preds = cross_val_predict(mre, rdo, cv)
        @test length(preds.pred) == 2
        for (k, cap) in enumerate((0.3, 0.05))
            r = preds.pred[k].res
            @test dot(r.pa.pr.rr.M[:, 1], r.w) <= cap + 1e-6
        end
        # The loadings are refit per fold, which is the whole reason the constraint is
        # re-based at generation time rather than written out once by the caller.
        @test preds.pred[1].res.pa.pr.rr.M != preds.pred[2].res.pa.pr.rr.M
    end
    # A space carries the *source* of its basis (ADR 0047, amended). `FactorSpace` stopped
    # being a singleton and gained `re`, whose precedence is `resolve_factor_regression`'s —
    # the one every other factor consumer already uses.
    stated = Regression(; M = [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0])
    lcem = LinearConstraintEstimator(; val = "MTUM <= 0.3")
    sece(space) = ExposureConstraintEstimator(; lce = lcem, space = space)
    @testset "The space carries its basis source" begin
        @test fieldnames(FactorSpace) == (:re,)
        # `nothing` is the default, and it is what every constraint written before the
        # field did: read the prior's loadings.
        @test isnothing(FactorSpace().re)
        @test isnothing(fs.re)
        # The bound is `RegE_Reg` — a fitted result or an estimator. A bare matrix is not a
        # basis source, it is a basis, and the space deliberately does not hold one.
        @test_throws TypeError FactorSpace(; re = M)
        @test_throws TypeError FactorSpace(; re = "MTUM")
        @test isa(FactorSpace(; re = rr).re, Regression)
        @test isa(FactorSpace(; re = StepwiseRegression()).re, StepwiseRegression)
    end
    @testset "The precedence is resolve_factor_regression's" begin
        # 1. A precomputed result wins outright: the caller has stated the answer.
        @test vec(linear_constraints(sece(FactorSpace(; re = stated)), sets; rr = rr).ineq.A) ==
              stated.M[:, 1]
        # 2. The prior's loadings are next — and a stated *estimator* loses to them, which
        #    is the documented asymmetry, not an oversight. Pass a fitted `Regression` to
        #    override a factor prior.
        @test vec(linear_constraints(sece(FactorSpace(; re = StepwiseRegression())), sets;
                                     rr = rr).ineq.A) == M[:, 1]
        # 3. `re === nothing` is bit-identical to the singleton it replaced.
        @test linear_constraints(sece(FactorSpace()), sets; rr = rr).ineq.A ==
              linear_constraints(sece(fs), sets; rr = rr).ineq.A
    end
    @testset "The standalone route cannot refit" begin
        # `linear_constraints(ece, sets; rr = ...)` receives the loadings and never the data
        # behind them, so the third arm is unreachable here and says so rather than
        # silently doing nothing.
        ece_fit = sece(FactorSpace(; re = StepwiseRegression()))
        @test_throws PortfolioOptimisers.IsNothingError linear_constraints(ece_fit, sets)
        msg = try
            linear_constraints(ece_fit, sets)
        catch e
            sprint(showerror, e)
        end
        # The message names the estimator that could not be fitted and both fixes.
        @test occursin("StepwiseRegression", msg)
        @test occursin("Regression(; M = ...)", msg)
        @test occursin("JuMPOptimiser", msg)
        # A space that names no source at all is the older diagnosis, and it now points at
        # the field as a third remedy.
        msg = try
            linear_constraints(sece(fs), sets)
        catch e
            sprint(showerror, e)
        end
        @test occursin("space.re === nothing", msg)
        @test occursin("FactorSpace(; re = StepwiseRegression())", msg)
    end
    @testset "An estimator refits: a factor mandate with no factor prior" begin
        # The capability the field buys. `EmpiricalPrior` carries no regression, so before
        # `re` this combination threw — it is the case the older test above asserts.
        ece_fit = ExposureConstraintEstimator(; lce = lcem,
                                              space = FactorSpace(;
                                                                  re = StepwiseRegression()))
        res = optimise(MeanRisk(;
                                opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = oslv,
                                                    sets = osets, lcse = ece_fit)), rdo)
        # The prior really carries none: the basis came from the refit and nowhere else.
        @test isnothing(res.pa.pr.rr)
        fitted = regression(StepwiseRegression(), rdo)
        @test vec(res.lcsr.ineq.A) ≈ fitted.M[:, 1]
        # And it binds, measured against the loadings the refit produced.
        @test dot(fitted.M[:, 1], res.w) <= 0.3 + 1e-6
        # A stated result at the optimiser is the pinned case: the row is exactly `M * a`
        # for the loadings handed in, whatever the prior would have fitted.
        pin = Regression(; M = [1.0 0.0; 0.0 1.0; 1.0 1.0])
        resp = optimise(MeanRisk(;
                                 opt = JuMPOptimiser(; pe = FactorPrior(), slv = oslv,
                                                     sets = osets,
                                                     lcse = ExposureConstraintEstimator(;
                                                                                        lce = lcem,
                                                                                        space = FactorSpace(;
                                                                                                            re = pin)))),
                        rdo)
        @test vec(resp.lcsr.ineq.A) == pin.M[:, 1]
        @test vec(resp.lcsr.ineq.A) != resp.pa.pr.rr.M[:, 1]
    end
    @testset "A stated basis is viewed where the universe is sliced" begin
        # The loadings are assets-major, so a subset of assets is a subset of rows and the
        # projection over the subset is exact. This is why an inner solve views rather than
        # refuses.
        ece_st = ExposureConstraintEstimator(; lce = lcem,
                                             space = FactorSpace(;
                                                                 re = Regression(; M = Mo)))
        i = [1, 3]
        v = PortfolioOptimisers.port_opt_view(ece_st, i)
        @test v.space.re.M == Mo[i, :]
        # A stated basis is usually just `M`. Viewing one used to crash on `view(nothing,
        # i)` for `b`, and to materialise `L` as a copy of `M` because the `swap(L, M)`
        # property rule makes `re.L` never `nothing` — both fixed, and both only reachable
        # once a constraint space could hold a hand-written `Regression`.
        @test isnothing(getfield(v.space.re, :L))
        @test isnothing(getfield(v.space.re, :b))
        rfull = Regression(; M = Mo, L = Mo .* 2, b = [1.0, 2.0, 3.0])
        @test PortfolioOptimisers.port_opt_view(rfull, i).L == (Mo .* 2)[i, :]
        @test PortfolioOptimisers.port_opt_view(rfull, i).b == [1.0, 3.0]
        # The wrapped shape is written in the space's names, so an asset index means
        # nothing to it and it is passed through.
        @test v.lce === ece_st.lce
        # A space that states no source, and a space holding an estimator, come back with
        # nothing to slice.
        @test isnothing(PortfolioOptimisers.port_opt_view(oece, i).space.re)
        @test isa(PortfolioOptimisers.port_opt_view(sece(FactorSpace(;
                                                                     re = StepwiseRegression())),
                                                    i).space.re, StepwiseRegression)
        # A vector maps elementwise. Without the method the universal fallback would `view`
        # the vector itself, slicing a list of constraints by asset indices.
        vv = PortfolioOptimisers.port_opt_view(PortfolioOptimisers.EcE_LcE_Lc[ece_st, lcem],
                                               i)
        @test length(vv) == 2
        @test vv[1].space.re.M == Mo[i, :]
        @test vv[2] === lcem
        # A precomputed `LinearConstraint` in `lcse` is passed through unchanged. The
        # identity is what the slot already did; slicing `A` here would change a path this
        # work does not touch. See ADR 0047's 2026-08-16 basis-source amendment.
        lc = LinearConstraint(;
                              ineq = PartialLinearConstraint(; A = transpose(Mo[:, 1:1]),
                                                             B = [0.3]))
        @test PortfolioOptimisers.port_opt_view(lc, i) === lc
    end
    @testset "A stated basis is refused where the universe is replaced" begin
        # An NCO outer solve writes cluster names over `sets.dict[xkey]` rather than
        # slicing it, so no view of asset-major loadings can follow it.
        pinned = ExposureConstraintEstimator(; lce = lcem,
                                             space = FactorSpace(;
                                                                 re = Regression(; M = Mo)))
        fitting = ExposureConstraintEstimator(; lce = lcem,
                                              space = FactorSpace(;
                                                                  re = StepwiseRegression()))
        jopt(l) = MeanRisk(;
                           opt = JuMPOptimiser(; pe = FactorPrior(), slv = oslv,
                                               sets = osets, lcse = l))
        @test_throws ArgumentError PortfolioOptimisers.assert_external_optimiser(jopt(pinned))
        # The inner path views instead, so it accepts what the outer refuses.
        @test isnothing(PortfolioOptimisers.assert_internal_optimiser(jopt(pinned)))
        # An estimator is refused nowhere: it refits against whatever universe it is
        # handed, which is exactly the remedy the message names.
        @test isnothing(PortfolioOptimisers.assert_external_optimiser(jopt(fitting)))
        # And a space that states no source is unaffected — no configuration that was legal
        # before this field became refusable.
        @test isnothing(PortfolioOptimisers.assert_external_optimiser(jopt(oece)))
        # A vector is walked, not ignored.
        @test_throws ArgumentError PortfolioOptimisers.assert_external_optimiser(jopt(PortfolioOptimisers.EcE_LcE_Lc[lcem,
                                                                                                                     pinned]))
        # The refusal is wired at construction, in the shape of the `opt.re` and
        # `opt.rba.re` refusals it copies.
        hclust = ClustersEstimator()
        @test_throws ArgumentError NestedClustered(; pe = FactorPrior(), cle = hclust,
                                                   opti = jopt(oece), opto = jopt(pinned))
        @test NestedClustered(; pe = FactorPrior(), cle = hclust, opti = jopt(pinned),
                              opto = jopt(oece)) isa NestedClustered
        # `Stacking` routes its outer optimiser through the same assertion, and its outer
        # universe is the synthetic sub-portfolios, so the refusal is right there too.
        @test_throws ArgumentError Stacking(; opti = [jopt(oece)], opto = jopt(pinned))
        # `SubsetResampling` asserts the *internal* contract, because it slices. A stated
        # basis is therefore legal under it, and correct: the view follows the subset.
        @test SubsetResampling(; opt = jopt(pinned)) isa SubsetResampling
    end
end
