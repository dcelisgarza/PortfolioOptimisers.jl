#=
The Asset Panel reaches every consumer that needs it.

`#646` decided that the panel rides `panel_feature_matrix(rd.pnl)[2]`, `panel_feature_matrix(rd.pnl)[1]` and `rd.pnl`, stops at `ReturnsResult` and
never travels onto `LowOrderPrior`. That decision rests on one claim: every consumer that needs
the panel is passed the `rd` that carries it. `#666` audited the claim across the 83 call sites
of the eleven verbs that can carry an `rd`, and found one place where it fails.

`Pr_RR` is `Union{<:AbstractPriorResult, <:ReturnsResult}`, so a `ReturnsResult` is admitted in
the `pr` slot as well as in `rd`. `clusterise(cle, rd)` is the shortest public call and the one
every `Pipeline` step makes, and it puts a full carrier there with no `rd` beside it.
`feature_matrix_picker` then read `Z` off that carrier and `nz` off `rd`, which was `nothing`,
so the names of a carrier that holds them were dropped. A `FeatureDistance` column selector
written as a name could not resolve, and the refusal named `LowOrderPrior` and `z_src = :prior`
— neither of which the caller had used.

`carrier_feature_names` closes it: the names come off **the carrier that supplied `Z`**. This
census pins that invariant, and pins the interface around it so a new entry point cannot forget.

The audit's other three classes need no gate, and are recorded here so the next reader need not
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
@testset "Asset Panel carrier census: the names travel with the matrix" begin
    PO = PortfolioOptimisers

    rng = StableRNG(20260902)
    na, nk = 6, 4
    X = randn(rng, 60, na)
    # Deliberately unrelated to the returns, so a distance derived from it cannot coincide
    # with a correlation distance by accident.
    Z = abs.(randn(rng, na, nk))
    nz = ["z$i" for i in 1:nk]
    nx = ["A$i" for i in 1:na]
    rd = ReturnsResult(; nx = nx, X = X, pnl = feature_matrix_panel(nz, Z))
    w = fill(inv(na), na)

    # ---------------------------------------- 1. the invariant this census exists to pin

    @testset "`Z` and `nz` come off the same carrier" begin
        #=
        A `ReturnsResult` in the `pr` slot with no `rd` beside it. Both selectors resolve to
        that one carrier, so both must return its names. The `:data` arm returned `nothing`
        before `#666`, which is the defect.
        =#
        for z_src in (:prior, :data)
            Zs, nzs, z_diag = PO.feature_matrix_picker(rd, nothing, z_src)
            @test Zs == panel_feature_matrix(rd.pnl)[2]
            @test nzs == nz
            @test z_diag === z_src
        end

        # A separate `rd` wins under `:data` and is inert under `:prior`. The pairing holds
        # either way, because both carriers here are the same one.
        for z_src in (:prior, :data)
            Zs, nzs, _ = PO.feature_matrix_picker(rd, rd, z_src)
            @test Zs == panel_feature_matrix(rd.pnl)[2]
            @test nzs == nz
        end

        # The `X` picker was never at fault, and stays unchanged: `pr.X` is right whichever
        # carrier sits in the slot.
        @test PO.returns_matrix_picker(rd, nothing, :prior) === rd.X
        @test PO.returns_matrix_picker(rd, rd, :data) === rd.X
    end

    @testset "A prior result supplies `Z` and names it positionally" begin
        #=
        The documented limit. A producer runs inside `prior(pe, X, F; …)` with raw matrices,
        so the names a *caller* knows are structurally unavailable to it. The panel it builds
        therefore names its columns positionally, which is what a nameless `Z` offered a
        selector before: an integer resolves, and a caller's own name does not.
        =#
        pnz = ["_z$i" for i in 1:nk]
        pr = LowOrderPrior(; X = X, mu = vec(mean(X; dims = 1)), sigma = cov(X),
                           pnl = feature_matrix_panel(pnz, Z))
        @test PO.carrier_asset_panel(pr) === pr.pnl
        Zs, nzs, z_diag = PO.feature_matrix_picker(pr, nothing, :prior)
        @test Zs == Z
        @test nzs == pnz
        @test z_diag === :prior

        # With both carriers populated the selector picks between them, and the names follow
        # the pick rather than the argument position.
        @test PO.feature_matrix_picker(pr, rd, :prior)[2] == pnz
        @test PO.feature_matrix_picker(pr, rd, :data)[2] == nz

        # A prior result that carries no `Z` at all still diagnoses `:neither`.
        pr_noz = prior(EmpiricalPrior(), ReturnsResult(; nx = nx, X = X))
        @test PO.feature_matrix_picker(pr_noz, nothing, :data)[3] === :neither
    end

    # ---------------------------------------- 2. the interface: every carrier method takes `rd`

    #=
    Closed polarity, as ADR 0037's rules and the censuses of ADR 0058 have it: the rule names
    no verb, so a carrier method written next year is covered the day it is written. A method
    that takes a `Pr_RR` either declares an `rd` keyword, or its verb is excused below with the
    reason it reads no feature matrix.
    =#
    @testset "Every `Pr_RR` method declares `rd`, or is excused by name" begin
        excused = Dict(:feature_matrix_picker => "takes `rd` positionally; it *is* the picker",
                       :returns_matrix_picker => "takes `rd` positionally; it *is* the picker",
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
        # And the pre-`#666` state is the refusal this census closes: hand the inner method
        # the `nothing` the picker used to produce, and the name cannot resolve.
        @test_throws PortfolioOptimisers.IsNothingError clusterise(cle, rd.X;
                                                                   Z = panel_feature_matrix(rd.pnl)[2],
                                                                   nz = nothing,
                                                                   z_src = :data)
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
