# Solver-free tests for the shared JuMP model-assembly pipeline (ADR 0008).
#
# These build a model through the per-optimiser head and `assemble_jump_model!`, then
# assert on the *constructed* model — without ever solving it. The interface under test is
# the assembler's keyword arguments (`r`, `wn2`/`l1`/… settings); the assertions check which
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
        nt = PO.processed_jump_optimiser_attributes(mr.opt, rd)
        model = JuMP.Model()
        PO.set_model_scales!(model, mr.opt.sc, mr.opt.so)
        PO.set_maximum_ratio_factor_variables!(model, nt.pr.mu, mr.obj)
        PO.set_w!(model, nt.pr.X, mr.wi)
        PO.set_weight_constraints!(model, nt.wb, mr.opt.bgt, mr.opt.sbgt)
        attrs = PO.ProcessedJuMPOptimiserAttributes(; pr = nt.pr, wb = nt.wb, lt = nt.lt,
                                                    st = nt.st, lcsr = nt.lcsr,
                                                    ctr = nt.ctr, gcardr = nt.gcardr,
                                                    sgcardr = nt.sgcardr, smtx = nt.smtx,
                                                    sgmtx = nt.sgmtx, slt = nt.slt,
                                                    sst = nt.sst, sglt = nt.sglt,
                                                    sgst = nt.sgst, tn = nt.tn,
                                                    fees = nt.fees, plr = nt.plr,
                                                    ret = nt.ret)
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
        @test "wn2" ∉ assemble_keys(base)
        @test "wn2" in assemble_keys(MeanRisk(; r = Variance(),
                                              opt = JuMPOptimiser(; slv = slv, wn2 = 5)))

        @test "l1" ∉ assemble_keys(base)
        @test "l1" in assemble_keys(MeanRisk(; r = Variance(),
                                             opt = JuMPOptimiser(; slv = slv, l1 = 0.1)))
    end

    @testset "integer phylogeny gates on the held indicator, not a fixed key" begin
        # `set_iplg_constraints!` re-read `model[:ib]`, which only the long-only builder
        # registers. Anything that routes the model to the long-short builder — a short
        # threshold here, but fixed fees or `xbgt` do it too — registers `ilb`/`isb`/`i_mip`
        # and no `:ib`, so the phylogeny constraint threw `KeyError(:ib)` at assembly time.
        # The held indicator is already a local in `set_mip_constraints!`; pass it down.
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

    @testset "lp penalty and wnp constraint register distinct keys" begin
        # The p-norm weight constraint and the Lp regularisation penalty both build
        # PowerCone epigraphs. They must not register under the same Model-State names,
        # or whichever runs second silently clobbers the other's handle in the object
        # dictionary (JuMP's `model[k] = v` overwrites without complaint).
        ks = assemble_keys(MeanRisk(; r = Variance(),
                                    opt = JuMPOptimiser(; slv = slv,
                                                        lp = LpRegularisation(; p = 3,
                                                                              val = 0.1),
                                                        wnp = LpRegularisation(; p = 3,
                                                                               val = 6))))
        for k in ("clp_1", "cslp_1", "t_lp_1", "r_lp_1")       # Lp penalty
            @test k in ks
        end
        for k in ("cwnp_1", "cswnp_1", "t_wnp_1", "r_wnp_1")   # p-norm constraint
            @test k in ks
        end
    end
end
