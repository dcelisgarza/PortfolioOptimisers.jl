#=
The Asset Panel reaches every consumer that needs it.

`#803` decided that the panel is the one carrier of feature data: it rides `rd.pnl`, it names its
own columns, and no prior result carries one. `#804` decided how it reaches the kernel: both
carriers travel as the keywords `pr` and `rd`, and one verb, `asset_panel(ape, pr, rd, X)`,
resolves the source by dispatch. That rests on one claim: every consumer that needs the panel is
passed a carrier that holds it.

`Pr_RR` is `Union{<:AbstractPriorResult, <:ReturnsResult}`, so a `ReturnsResult` is admitted in
the `pr` slot as well as in `rd`. `clusterise(cle, rd)` is the shortest public call and the one
every `Pipeline` step makes, and it puts a full carrier there with no `rd` beside it. The
resolution therefore reads **either** slot, and a name selector resolves against the column names
the panel derives. This census pins that invariant, and pins the interface around it so a new
entry point cannot forget.

Three neighbouring classes need no gate, and are recorded here so the next reader need not
re-derive them:

  - `set_risk_constraints!` and `set_return_constraints!` forward `kwargs...` transparently, so
    an `rd` a caller passed rides through untouched.
  - `gcarde`, `sgcarde` and `Variance.rc` are bounded by `LcE_Lc`, which excludes
    `ExposureConstraintEstimator`. Nothing that reads returns data can reach those slots, so
    their `linear_constraints` calls need no `rd` — the type bound is the guard, as the design
    rules ask.
  - `PredictionReturnsResult` deliberately carries no `pnl`: `reconstruct_rd` collapses the
    asset axis onto one synthetic asset, and a panel over the old assets would be wrong.
=#
@testset "Asset Panel carrier census: the panel reaches every consumer" begin
    PO = PortfolioOptimisers

    rng = StableRNG(20260902)
    na, nk = 6, 4
    X = randn(rng, 60, na)
    # Deliberately unrelated to the returns, so a distance derived from it cannot coincide
    # with a correlation distance by accident.
    Z = abs.(randn(rng, na, nk))
    nz = ["z$i" for i in 1:nk]
    nx = ["A$i" for i in 1:na]
    pnl = asset_panel([NumericPanelInput(; name = nz[k], vals = Z[:, k]) for k in 1:nk])
    rd = ReturnsResult(; nx = nx, X = X, pnl = pnl)
    w = fill(inv(na), na)

    # ---------------------------------------- 1. the invariant this census exists to pin

    @testset "The panel resolves off either carrier slot" begin
        #=
        A `ReturnsResult` in the `pr` slot with no `rd` beside it, and the same carrier in
        `rd`. One verb answers both, and the columns it derives carry the caller's own names,
        so a name selector resolves in either shape.
        =#
        @test PO.asset_panel(nothing, rd, nothing, X) === pnl
        @test PO.asset_panel(nothing, nothing, rd, X) === pnl
        # `rd` wins when both slots hold a carrier, because the data carrier is where a panel
        # is data rather than a by-product.
        rd2 = ReturnsResult(; nx = nx, X = X,
                            pnl = asset_panel([NumericPanelInput(; name = "other",
                                                                 vals = Z[:, 1])]))
        @test PO.asset_panel(nothing, rd, rd2, X) === rd2.pnl

        # The names come off the panel itself, so they cannot disagree with the values.
        @test PO.panel_feature_names(pnl) == nz
        @test PO.panel_feature_matrix(pnl)[2] == Z

        # The `X` picker is unchanged: `pr.X` is right whichever carrier sits in the slot.
        @test PO.returns_matrix_picker(rd, nothing, :prior) === rd.X
        @test PO.returns_matrix_picker(rd, rd, :data) === rd.X
    end

    @testset "A prior result alone carries no panel, and the refusal names the way out" begin
        #=
        The documented limit. No prior result carries feature data at all: the panel is on the
        data carrier, or a producer builds one at the point of use. A call that reaches the
        kernel with a prior alone therefore raises, and the message names both routes.
        =#
        pr = prior(EmpiricalPrior(), rd)
        @test !hasproperty(pr, :pnl)
        res = @test_throws PO.IsNothingError PO.asset_panel(nothing, pr, nothing, X)
        @test occursin("ReturnsResult", res.value.msg)
        @test occursin("RegressionPanel", res.value.msg)

        # A carrier that holds no panel raises too, and says how to build one.
        rd_no = ReturnsResult(; nx = nx, X = X)
        res = @test_throws PO.IsNothingError PO.asset_panel(nothing, nothing, rd_no, X)
        @test occursin("asset_panel", res.value.msg)
    end

    # ---------------------------------------- 2. the interface: every carrier method takes `rd`

    #=
    Closed polarity, as ADR 0037's rules and the censuses of ADR 0058 have it: the rule names
    no verb, so a carrier method written next year is covered the day it is written. A method
    that takes a `Pr_RR` either declares an `rd` keyword, or its verb is excused below with the
    reason it reads no feature matrix.
    =#
    @testset "Every `Pr_RR` method declares `rd`, or is excused by name" begin
        excused = Dict(:returns_matrix_picker => "takes `rd` positionally; it *is* the picker",
                       :expected_risk => "reads `pr.X` and the moments; no feature matrix",
                       :calc_net_returns => "reads a returns matrix and fees; no feature matrix")

        carrier_methods = Tuple{Symbol, Method}[]
        for n in names(PO; all = true)
            startswith(string(n), "#") && continue
            isdefined(PO, n) || continue
            f = getfield(PO, n)
            isa(f, Function) || continue
            for m in methods(f)
                m.module === PO || continue
                ps = Base.unwrap_unionall(m.sig).parameters
                if any(p -> isa(p, Type) && p == PO.Pr_RR, ps)
                    push!(carrier_methods, (n, m))
                end
            end
        end

        # A walk that answered nothing would make every check below vacuously green.
        @test length(carrier_methods) >= 11

        offenders = String[]
        for (n, m) in carrier_methods
            (:rd in Base.kwarg_decl(m) || haskey(excused, n)) && continue
            push!(offenders, "$(n)  [$(m.file):$(m.line)]")
        end
        @test ("`Pr_RR` methods that neither take `rd` nor are excused", offenders) ==
              ("`Pr_RR` methods that neither take `rd` nor are excused", String[])

        # The excuse is not stale: each name is still a `Pr_RR` verb, and still takes no `rd`.
        found = Set(first.(carrier_methods))
        for (n, _) in excused
            @test n in found
            @test all(m -> !(:rd in Base.kwarg_decl(m)),
                      [m for (v, m) in carrier_methods if v == n])
        end
    end

    # ---------------------------------------- 3. the behaviour: every entry point resolves a name

    #=
    The end-to-end half. A name selector under `strict = true` throws when the names do not
    arrive, so a green run here *is* the proof that the panel reached the consumer. Every
    public entry point that takes a carrier is driven, each with the bare `ReturnsResult` that
    `#666` found losing its names.
    =#
    @testset "Every entry point resolves a *name* selector off a bare carrier" begin
        fde = FeatureDistance(; sel = ["z1", "z3"], strict = true)
        nte = NetworkEstimator(; de = fde, alg = KruskalTree())
        cle = ClustersEstimator(; de = fde)
        cte = CentralityEstimator(; pl = nte)

        @test clusterise(cle, rd) isa PortfolioOptimisers.AbstractClusteringResult
        @test phylogeny_matrix(nte, rd) isa PhylogenyResult
        @test phylogeny_constraints(SemiDefinitePhylogenyEstimator(; pl = nte), rd) isa
              SemiDefinitePhylogeny
        @test centrality_vector(cte, rd) isa PhylogenyResult
        @test centrality_vector(nte, DegreeCentrality(), rd) isa PhylogenyResult
        @test average_centrality(cte, w, rd) isa Number
        @test average_centrality(nte, DegreeCentrality(), w, rd) isa Number
        @test asset_phylogeny(nte, w, rd) isa Number
        @test centrality_constraints(CentralityConstraint(; A = cte, B = 1.0, comp = <=),
                                     rd) isa LinearConstraint

        #=
        Three floors on the half above. Without them a selector that had stopped selecting, or
        a `strict` that had stopped throwing, would leave every assertion green.
        =#
        # The selector does real work: two of four columns is not four of four.
        @test clusterise(cle, rd).D !=
              clusterise(ClustersEstimator(; de = FeatureDistance()), rd).D
        # A name the carrier does not hold is refused, so a green run above is a resolution.
        @test_throws ArgumentError clusterise(ClustersEstimator(;
                                                                de = FeatureDistance(;
                                                                                     sel = ["nope"],
                                                                                     strict = true)),
                                              rd)
        # And a call that reaches the kernel with no carrier at all is the refusal this census
        # closes: there is nothing to resolve the panel from.
        @test_throws PortfolioOptimisers.IsNothingError clusterise(cle, rd.X)
    end

    # ---------------------------------------- 4. the four `Pipeline` sites

    #=
    The pipeline is where the audit found the defect in `src/`: `run_step` and
    `constraint_step_value` pass `ctx.returns` in the `pr` slot and never fill `rd`, because
    the context holds one carrier and there is no second one to pass. Each of the four sites
    threw before `#666`.
    =#
    @testset "The four `Pipeline` sites carry the panel" begin
        fde = FeatureDistance(; sel = ["z1", "z3"], strict = true)
        nte = NetworkEstimator(; de = fde, alg = KruskalTree())
        cle = ClustersEstimator(; de = fde)
        cte = CentralityEstimator(; pl = nte)
        ctx = PO.PipelineContext(; returns = rd)

        @test PO.run_step(cle, ctx)[1] isa PortfolioOptimisers.AbstractClusteringResult
        @test PO.run_step(nte, ctx)[1] isa PhylogenyResult
        @test PO.constraint_step_value(SemiDefinitePhylogenyEstimator(; pl = nte), ctx) isa
              SemiDefinitePhylogeny
        @test PO.constraint_step_value(CentralityConstraint(; A = cte, B = 1.0, comp = <=),
                                       ctx) isa LinearConstraint
    end
end
