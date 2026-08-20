# Solver-free tests for the shared JuMP model-assembly pipeline (ADR 0008).
#
# These build a model through the per-optimiser head and `assemble_jump_model!`, then
# assert on the *constructed* model — without ever solving it. The interface under test is
# the assembler's keyword arguments (`r`, `l2c`/`l1`/… settings); the assertions check which
# Model-State keys the middle registers, so they exercise the routing rather than numerics.
@testset "JuMP model assembly (solver-free)" begin
    using Test, PortfolioOptimisers, CSV, TimeSeries, Clarabel, JuMP
    PO = PortfolioOptimisers
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)

    # Run the head + assembler for `mr`, stop before solving, return the set of registered
    # Model-State key names. `r`/`b1`/`obj`/… default to the optimiser's own values but can
    # be overridden to probe a single branch in isolation.
    function assemble_keys(mr; r = mr.r, b1 = nothing, obj = mr.obj,
                           sdp_asset_phylogeny = true)
        attrs = PO.processed_jump_optimiser_attributes(mr.opt, rd)
        model = JuMP.Model()
        PO.set_model_scales!(model, mr.opt.sc, mr.opt.so)
        # Reach the head's own seams, not a copy of them: `set_maximum_ratio_factor_variables!`
        # takes exactly one objective, and the budget group travels as `mr.opt`. A hand-copied
        # call used to pass three arguments here and to name `bgt`/`sbgt` alone, so the
        # `MaximumRatio` branch was never built and `gbgt` never reached the model.
        PO.set_maximum_ratio_factor_variables!(model, obj)
        PO.set_w!(model, attrs.pr.X, mr.wi)
        PO.set_weight_constraints!(model, attrs.wb, mr.opt)
        PO.assemble_jump_model!(model, mr, mr.opt, attrs, rd, r, obj, b1,
                                sdp_asset_phylogeny)
        return Set(string.(keys(JuMP.object_dictionary(model))))
    end

    base = MeanRisk(; r = Variance(), opt = JuMPOptimiser(; slv = slv))

    @testset "core state is established" begin
        ks = assemble_keys(base)
        @test "w" in ks
        @test "k" in ks
        @test "ret" in ks
    end

    # `k` has two spellings and the objective picks between them. A wrong-arity call to the
    # producer used to be absorbed by a variadic fallback, which registered the constant
    # under every objective — so this branch built but was never the one under test.
    @testset "the objective picks the spelling of `k`" begin
        model = JuMP.Model()
        PO.set_maximum_ratio_factor_variables!(model, MinimumRisk())
        @test model[:k] == 1
        @test !isa(model[:k], JuMP.VariableRef)

        rmodel = JuMP.Model()
        PO.set_maximum_ratio_factor_variables!(rmodel, MaximumRatio())
        @test isa(rmodel[:k], JuMP.VariableRef)
        @test JuMP.lower_bound(rmodel[:k]) == 0

        # The producer takes exactly one objective; a wrong-arity call must not be absorbed.
        @test_throws MethodError PO.set_maximum_ratio_factor_variables!(JuMP.Model(),
                                                                        nothing,
                                                                        MaximumRatio())
    end

    # The budget group travels as one object, so a head cannot read part of it. `gbgt` binds
    # only once the long/short decomposition exists, which a negative weight bound forces.
    @testset "the gross budget reaches the model through the optimiser" begin
        gopt = JuMPOptimiser(; slv = slv, wb = WeightBounds(; lb = -1.0, ub = 1.0),
                             bgt = nothing, sbgt = nothing, gbgt = 2.0)
        ks = assemble_keys(MeanRisk(; r = Variance(), opt = gopt))
        @test "gbgt" in ks
        @test "gbgt" ∉ assemble_keys(base)
    end

    @testset "risk is routed by the `r` kwarg" begin
        ks = assemble_keys(base)
        @test "variance_risk_1" in ks
        @test "variance_risk_2" ∉ ks

        # A vector of risk measures registers one indexed expression each.
        ksv = assemble_keys(MeanRisk(; r = [Variance(), Variance()],
                                     opt = JuMPOptimiser(; slv = slv)))
        @test "variance_risk_1" in ksv
        @test "variance_risk_2" in ksv

        # r = nothing → the risk + scalarise step is a no-op (the RelaxedRiskBudgeting path),
        # but the head and the return constraints are still applied.
        ksn = assemble_keys(base; r = nothing)
        @test "variance_risk_1" ∉ ksn
        @test "risk" ∉ ksn
        @test "w" in ksn
        @test "ret" in ksn
    end

    @testset "optional constraints toggle on their settings" begin
        @test "l2c" ∉ assemble_keys(base)
        @test "l2c" in assemble_keys(MeanRisk(; r = Variance(),
                                              opt = JuMPOptimiser(; slv = slv, l2c = 0.5)))

        @test "l1" ∉ assemble_keys(base)
        @test "l1" in assemble_keys(MeanRisk(; r = Variance(),
                                             opt = JuMPOptimiser(; slv = slv, l1 = 0.1)))
    end

    @testset "integer phylogeny gates on the held indicator, not a fixed key" begin
        # `set_iplg_constraints!` re-read `model[:ib]`, which only the long-only builder
        # registers. Anything that routes the model to the long-short builder — a short
        # threshold here, but fixed fees or `xbgt` do it too — registers `ilb`/`isb`/`i_mip`
        # and no `:ib`, so the phylogeny constraint threw `KeyError(:ib)` at assembly time.
        # The builder now registers its bundle in Model State (`set_mip_indicators!`) and the
        # late emitter reads it back with `mip_indicators` + `held_bin` — no key involved.
        mr = MeanRisk(; r = Variance(),
                      opt = JuMPOptimiser(; slv = slv, bgt = 1, sbgt = 1,
                                          wb = WeightBounds(; lb = -1, ub = 1),
                                          ple = IntegerPhylogenyEstimator(;
                                                                          pl = NetworkEstimator(),
                                                                          B = 1),
                                          st = Threshold(; val = 0.05)))
        ks = assemble_keys(mr)
        @test "card_plg_1" in ks       # the phylogeny constraint was emitted at all
        @test "i_mip" in ks            # ... gated on the long-short builder's held indicator
        @test "ib" ∉ ks                # ... which is not `:ib`; that key is never registered

        # The long-only builder still routes through the same call and keeps `:ib`.
        mrl = MeanRisk(; r = Variance(),
                       opt = JuMPOptimiser(; slv = slv, bgt = 1, sbgt = 1,
                                           wb = WeightBounds(; lb = -1, ub = 1),
                                           ple = IntegerPhylogenyEstimator(;
                                                                           pl = NetworkEstimator(),
                                                                           B = 1)))
        ksl = assemble_keys(mrl)
        @test "card_plg_1" in ksl
        @test "ib" in ksl
    end

    @testset "lp penalty and lpc constraint register distinct keys" begin
        # The p-norm weight constraint and the Lp regularisation penalty both build
        # PowerCone epigraphs. They must not register under the same Model-State names,
        # or whichever runs second silently clobbers the other's handle in the object
        # dictionary (JuMP's `model[k] = v` overwrites without complaint).
        ks = assemble_keys(MeanRisk(; r = Variance(),
                                    opt = JuMPOptimiser(; slv = slv,
                                                        lp = LpRegularisation(; p = 3,
                                                                              val = 0.1),
                                                        lpc = LpRegularisation(; p = 3,
                                                                               val = 0.5))))
        for k in ("clp_1", "cslp_1", "t_lp_1", "r_lp_1")       # Lp penalty
            @test k in ks
        end
        for k in ("clpc_1", "cslpc_1", "t_lpc_1", "r_lpc_1")   # p-norm constraint
            @test k in ks
        end
    end
end
