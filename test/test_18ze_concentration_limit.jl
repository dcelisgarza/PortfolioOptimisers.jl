@testset "Concentration limit: the UCITS 5/10/40 rule (#2018)" begin
    using PortfolioOptimisers, Test, Clarabel, HiGHS, Pajarito, StableRNGs, JuMP,
          LinearAlgebra
    hslv = Solver(; name = :highs, solver = HiGHS.Optimizer,
                  settings = "log_to_console" => false,
                  check_sol = (; allow_local = true, allow_almost = true))
    N = 20
    nx = ["a$i" for i in 1:N]
    # A drift that falls with the index, so a maximum return book wants the first assets.
    X = 0.01 .* randn(StableRNG(42), 200, N) .+ range(0.002, 0.0002; length = N)'
    rd = ReturnsResult(; nx = nx, X = X)
    tol = 1e-6
    over(x, t = 0.05) = sum(x[x .> t + tol]; init = 0.0)
    issuer_sum(w, issuer) = [sum(w[issuer .== u]) for u in unique(issuer)]
    function solve(; obj = MaximumReturn(), kwargs...)
        opt = JuMPOptimiser(; slv = hslv, wb = WeightBounds(; lb = 0.0, ub = 0.1),
                            kwargs...)
        return optimise(MeanRisk(; r = ConditionalValueatRisk(), obj = obj, opt = opt), rd)
    end
    success(res) = isa(res.retcode, OptimisationSuccess)

    @testset "The rule is active and holds on a long-only book" begin
        # Without the rule, ten positions at 10 % sum to 100 %.
        r0 = solve()
        @test success(r0)
        @test over(r0.w) > 0.4 + tol
        # With the rule, the positions over 5 % sum to at most 40 %, and none is over 10 %.
        r1 = solve(; conc = ConcentrationLimit())
        @test success(r1)
        @test over(r1.w) <= 0.4 + tol
        @test maximum(r1.w) <= 0.1 + tol
        @test isapprox(sum(r1.w), 1; atol = tol)
        cm = concentration_margins(ConcentrationLimit(), r1.w)
        @test all(cm.cap .<= tol)
        @test cm.total <= tol
        # The keys of the rule are registered under the index of the rule.
        @test haskey(r1.model, :conc_z_1)
        @test haskey(r1.model, :conc_sum_1)
        # Under a constant budget the binary is the gate, so the model holds no gate variable.
        @test !haskey(r1.model, :conc_g_1)
        # A ratio objective solves for scaled weights, and the rule holds after the scale.
        rr = solve(; obj = MaximumRatio(; rf = 0.0), conc = ConcentrationLimit())
        @test success(rr)
        @test haskey(rr.model, :conc_g_1)
        @test over(rr.w) <= 0.4 + tol
        @test maximum(rr.w) <= 0.1 + tol
    end

    @testset "Two share classes of one issuer are one position" begin
        # The first eight assets are four issuers of two share classes each. Sixteen issuers
        # can reach 100 % under 5/10/40: four at 10 % and twelve at 5 %.
        issuer = [i <= 8 ? "i$(cld(i, 2))" : "s$i" for i in 1:N]
        sets = UniverseSets(; dict = Dict("nx" => nx, "nx_issuer" => issuer))
        cl = ConcentrationLimitEstimator(;
                                         set = AssetSetsMatrixEstimator(;
                                                                        val = "nx_issuer"))
        res = solve(; sets = sets, conc = cl)
        @test success(res)
        si = issuer_sum(res.w, issuer)
        @test maximum(si) <= 0.1 + tol
        @test over(si) <= 0.4 + tol
        # The rule over the assets lets an issuer hold 20 %, so the set changes the book.
        ra = solve(; conc = ConcentrationLimit())
        @test maximum(issuer_sum(ra.w, issuer)) > 0.1 + tol
        cm = concentration_margins(cl, res.w; sets = sets)
        @test length(cm.x) == 16
        @test cm.total <= tol
    end

    @testset "The readings of a long-short book" begin
        lb = fill(0.0, N)
        ub = fill(0.2, N)
        lb[1] = ub[1] = 0.08
        lb[2] = ub[2] = -0.08
        issuer = ["A", "A", ["i$j" for j in 3:N]...]
        sets = UniverseSets(; dict = Dict("nx" => nx, "nx_issuer" => issuer))
        iset = AssetSetsMatrixEstimator(; val = "nx_issuer")
        function solve_ls(pos, c)
            cl = ConcentrationLimitEstimator(; c = c, s = 0.9, set = iset, pos = pos)
            opt = JuMPOptimiser(; slv = hslv, sets = sets,
                                wb = WeightBounds(; lb = lb, ub = ub), sbgt = 0.1,
                                conc = cl)
            return optimise(MeanRisk(; r = ConditionalValueatRisk(), opt = opt), rd)
        end
        # Issuer A holds +8 % and -8 %: 16 % gross, 0 % net and 8 % long.
        @test !success(solve_ls(GrossPosition(), 0.15))
        @test success(solve_ls(NetPosition(), 0.15))
        @test success(solve_ls(LongPosition(), 0.15))
        @test !success(solve_ls(LongPosition(), 0.07))
        @test success(solve_ls(NetPosition(), 0.07))
        w = zeros(N)
        w[1] = 0.08
        w[2] = -0.08
        x = [concentration_margins(ConcentrationLimitEstimator(; set = iset, pos = p), w;
                                   sets = sets).x[1]
             for p in (GrossPosition(), NetPosition(), LongPosition())]
        @test x ≈ [0.16, 0.0, 0.08]
        # The budgets pin the long part at 1.3, but the bounds let the book hold 1.1 long at
        # most. With `xbgt` off, the long reading sees the pinned parts, and 19 positions at
        # 6 % cannot hold them.
        wbp = WeightBounds(; lb = [-0.1; zeros(N - 1)], ub = [0.0; fill(1.0, N - 1)])
        function solve_pin(pos, xbgt, sbgt)
            opt = JuMPOptimiser(; slv = hslv, wb = wbp, sbgt = sbgt, xbgt = xbgt,
                                conc = ConcentrationLimit(; t = 0.02, c = 0.06, s = 1.2,
                                                          pos = pos))
            return optimise(MeanRisk(; r = ConditionalValueatRisk(), opt = opt), rd)
        end
        @test success(solve_pin(NetPosition(), false, 0.3))
        @test !success(solve_pin(LongPosition(), false, 0.3))
        # A range lets the parts fall to the parts of the weights.
        rl = solve_pin(LongPosition(), false, BudgetRange(; lb = 0.0, ub = 0.3))
        @test success(rl)
        @test maximum(rl.w) <= 0.06 + tol
        # A short of -30 % counts as 30 % under the gross reading, and the other two
        # readings do not see it.
        w = [-0.3; zeros(N - 1)]
        cms = [concentration_margins(ConcentrationLimit(; pos = p), w)
               for p in (GrossPosition(), NetPosition(), LongPosition())]
        @test cms[1].x[1] ≈ 0.3
        @test cms[1].cap[1] > 0
        @test cms[2].x[1] ≈ -0.3
        @test cms[3].x[1] == 0
        @test all(cm.total < 0 for cm in cms[2:3])
    end

    @testset "The linear form is safe and stricter" begin
        res = solve(; conc = ConcentrationLimit(; alg = SumLargestConcentration()))
        @test success(res)
        @test over(res.w) <= 0.4 + tol
        @test maximum(res.w) <= 0.1 + tol
        @test sum(sort(res.w; rev = true)[1:8]) <= 0.4 + tol
        @test PortfolioOptimisers.sum_largest_count(0.05, 0.4) == 8
        @test PortfolioOptimisers.sum_largest_count(0.05, 0.42) == 9
        # `0.14 / 0.02` rounds to `7.000000000000001`, and `ceil` gives 8.
        @test ceil(Int, 0.14 / 0.02) == 8
        @test PortfolioOptimisers.sum_largest_count(0.02, 0.14) == 7
        # A book that holds eight positions at 9 % is infeasible.
        lb = zeros(N)
        lb[1:8] .= 0.09
        opt = JuMPOptimiser(; slv = hslv, wb = WeightBounds(; lb = lb, ub = 0.1),
                            conc = ConcentrationLimit(; alg = SumLargestConcentration()))
        @test !success(optimise(MeanRisk(; r = ConditionalValueatRisk(), opt = opt), rd))
        # Four positions at 10 % and six at 5 % meet the rule, and the linear form
        # refuses them.
        w = [fill(0.1, 4); fill(0.05, 12); zeros(N - 16)]
        @test concentration_margins(ConcentrationLimit(; alg = SumLargestConcentration()),
                                    w).total <= tol
        @test sum(sort(w; rev = true)[1:8]) > 0.4 + tol
    end

    @testset "The `on` filter selects the issuers of a rule" begin
        nxs = ["C1", "C2", "K1", "G1"]
        sets = UniverseSets(;
                            dict = Dict("nx" => nxs,
                                        "nx_issuer" => ["Acme", "Acme", "BankX", "DE"],
                                        "ucits_standard" => ["C1", "C2"],
                                        "ucits_covered" => ["K1"]))
        A1 = asset_sets_matrix(AssetSetsMatrixEstimator(; val = "nx_issuer",
                                                        on = "ucits_standard"), sets)
        @test A1 == Bool[1 1 0 0]
        A2 = asset_sets_matrix(AssetSetsMatrixEstimator(; val = "nx_issuer",
                                                        on = "ucits_covered"), sets)
        @test A2 == Bool[0 0 1 0]
        A3 = asset_sets_matrix(AssetSetsMatrixEstimator(; val = "nx_issuer",
                                                        on = ["K1", "G1"]), sets)
        @test A3 == Bool[0 0 1 0; 0 0 0 1]
        bad = AssetSetsMatrixEstimator(; val = "nx_issuer", on = "ucits_standrad")
        @test_throws ArgumentError asset_sets_matrix(bad, sets; strict = true)
        @test (@test_logs (:warn,) asset_sets_matrix(bad, sets)) == falses(0, 4)
        # The method for a vector gives `strict` to each entry.
        @test_throws ArgumentError asset_sets_matrix([bad], sets; strict = true)
        @test asset_sets_matrix([AssetSetsMatrixEstimator(; val = "nx_issuer", on = "K1"),
                                 AssetSetsMatrixEstimator(; val = "nx_issuer", on = "G1")],
                                sets) == [Bool[0 0 1 0], Bool[0 0 0 1]]
        # Stage 1 of the block gives `strict` to the rule.
        cbe = ConstraintBlock(; sets = sets,
                              conc = ConcentrationLimitEstimator(; set = bad))
        @test_throws ArgumentError PortfolioOptimisers.resolve_constraint_block_names(cbe,
                                                                                      true,
                                                                                      Float64)
        @test_throws IsEmptyError AssetSetsMatrixEstimator(; val = "nx_issuer", on = "")
        @test_throws IsEmptyError AssetSetsMatrixEstimator(; val = "nx_issuer",
                                                           on = String[])
        @test_throws IsEmptyError AssetSetsMatrixEstimator(; val = "nx_issuer",
                                                           on = ["K1", ""])
        # A name on the Non-Investable Axis drops with no message.
        setsni = UniverseSets(;
                              dict = Dict("nx" => ["C1", "C2"],
                                          "nx_issuer" => ["Acme", "BankX"], "ni" => ["K1"]))
        A4 = @test_logs asset_sets_matrix(AssetSetsMatrixEstimator(; val = "nx_issuer",
                                                                   on = ["C2", "K1"]),
                                          setsni; strict = true)
        @test A4 == Bool[0 1]
    end

    @testset "Two rules in one block" begin
        issuer = ["i$i" for i in 1:N]
        cov = ifelse.(1:N .<= 10, "standard", "covered")
        sets = UniverseSets(;
                            dict = Dict("nx" => nx, "nx_issuer" => issuer,
                                        "ucits_standard" => nx[1:10],
                                        "ucits_covered" => nx[11:N]))
        conc = [ConcentrationLimitEstimator(;
                                            set = AssetSetsMatrixEstimator(;
                                                                           val = "nx_issuer",
                                                                           on = "ucits_standard")),
                ConcentrationLimitEstimator(; c = 0.25, s = 0.8,
                                            set = AssetSetsMatrixEstimator(;
                                                                           val = "nx_issuer",
                                                                           on = "ucits_covered"))]
        opt = JuMPOptimiser(; slv = hslv, sets = sets,
                            wb = WeightBounds(; lb = 0.0, ub = 0.3), conc = conc)
        res = optimise(MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumReturn(),
                                opt = opt), rd)
        @test success(res)
        ws = res.w[cov .== "standard"]
        wc = res.w[cov .== "covered"]
        @test maximum(ws) <= 0.1 + tol
        @test over(ws) <= 0.4 + tol
        @test maximum(wc) <= 0.25 + tol
        @test over(wc) <= 0.8 + tol
        @test haskey(res.model, :conc_z_1) && haskey(res.model, :conc_z_2)
        cms = concentration_margins(conc, res.w; sets = sets)
        @test length(cms) == 2
        @test all(cm.total <= tol && all(cm.cap .<= tol) for cm in cms)
    end

    @testset "The rule holds on the weights of a risk budgeting book" begin
        #=
        A `RiskBudgeting` head solves for `k w`, with `k` about 22 here, and divides by `k`
        after the solve. A rule that multiplies its numbers by 1 binds the wrong scale.
        =#
        Xr = 0.01 .* randn(StableRNG(42), 200, N) .* range(0.5, 2.0; length = N)' .+ 0.0005
        rdr = ReturnsResult(; nx = nx, X = Xr)
        cslv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                      settings = "verbose" => false,
                      check_sol = (; allow_local = true, allow_almost = true))
        pslv = Solver(; name = :pajarito,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" =>
                                                             optimizer_with_attributes(HiGHS.Optimizer,
                                                                                       MOI.Silent() =>
                                                                                           true),
                                                         "conic_solver" =>
                                                             optimizer_with_attributes(Clarabel.Optimizer,
                                                                                       "verbose" =>
                                                                                           false)),
                      check_sol = (; allow_local = true, allow_almost = true))
        function solve_rb(s, alg; kwargs...)
            opt = JuMPOptimiser(; slv = s, wb = WeightBounds(; lb = 0.0, ub = 1.0),
                                kwargs...)
            return optimise(RiskBudgeting(; r = StandardDeviation(), opt = opt,
                                          rba = AssetRiskBudgeting(; alg = alg)), rdr)
        end
        base = solve_rb(cslv, LogRiskBudgeting())
        @test success(base)
        # The linear form binds at its limit, on the weights that the result reports.
        cl = ConcentrationLimit(; t = 0.055, c = 0.8 * maximum(base.w), s = 0.65,
                                alg = SumLargestConcentration())
        for (s, alg) in ((cslv, LogRiskBudgeting()), (pslv, MixedIntegerRiskBudgeting()))
            res = solve_rb(s, alg; conc = cl)
            @test success(res)
            @test value(res.model[:k]) > 2
            @test isapprox(sum(sort(res.w; rev = true)[1:12]), 0.65; rtol = 1e-4)
            @test maximum(res.w) <= cl.c + tol
            @test concentration_margins(cl, res.w).total <= tol
        end
        cle = ConcentrationLimit(; t = 0.055, c = 0.08, s = 0.5)
        res = solve_rb(pslv, MixedIntegerRiskBudgeting(); conc = cle)
        @test success(res)
        @test isapprox(maximum(res.w), 0.08; rtol = 1e-4)
        @test concentration_margins(cle, res.w).total <= tol
    end

    @testset "The check reads a book of any optimiser" begin
        hres = optimise(HierarchicalRiskParity(), rd)
        cm = concentration_margins(ConcentrationLimit(), hres.w)
        @test length(cm.x) == N
        @test cm.x == hres.w
        @test cm.cap == hres.w .- 0.1
        @test cm.total ≈ over(hres.w) - 0.4
        jres = solve(; conc = ConcentrationLimit())
        @test concentration_margins(ConcentrationLimit(), jres.w).total <= tol
    end

    @testset "Validation and views" begin
        @test_throws DomainError ConcentrationLimit(; t = 0.1, c = 0.1)
        @test_throws DomainError ConcentrationLimit(; t = 0.0)
        @test_throws DomainError ConcentrationLimitEstimator(; s = -0.4)
        @test_throws DomainError ConcentrationLimit(; c = Inf)
        @test_throws DimensionMismatch ConcentrationMargins(; x = [0.1], cap = [0.1, 0.2],
                                                            total = 0.0)
        cl = ConcentrationLimit(; set = [1 1 0; 0 0 1])
        @test PortfolioOptimisers.port_opt_view(cl, [1, 3]).set == [1 0; 0 1]
        cle = ConcentrationLimitEstimator(; set = AssetSetsMatrixEstimator(; val = "k"))
        @test PortfolioOptimisers.port_opt_view(cle, [1, 3]) === cle
        @test concentration_limit_constraints(cl) === cl
        @test isnothing(concentration_limit_constraints(nothing))
        @test PortfolioOptimisers.name_keyed(cle)
        @test !PortfolioOptimisers.name_keyed(ConcentrationLimitEstimator())
        @test_throws IsNothingError ConstraintBlock(; conc = cle)
        # The check of a book states that the rule needs `sets`, also in a vector.
        @test_throws IsNothingError concentration_margins(cle, fill(0.5, 2))
        @test_throws IsNothingError concentration_margins([cle], fill(0.5, 2))
        @test_throws IsNothingError concentration_limit_constraints(cle, nothing)
        @test_throws IsEmptyError ConstraintBlock(; conc = ConcentrationLimit[])
        # A mixed vector narrows, and the view keeps one rule for each entry.
        cbe = ConstraintBlock(;
                              conc = [ConcentrationLimit(), ConcentrationLimitEstimator()])
        @test length(PortfolioOptimisers.port_opt_view(cbe, 1:2).conc) == 2
        # The slot is a property of the owner.
        opt = JuMPOptimiser(; slv = hslv, conc = cl)
        @test opt.conc === cl
    end

    @testset "The tolerance of the check" begin
        # A solver can put a position over `t` by its tolerance. The sum is a step, so a
        # strict check then counts the whole position.
        w = [fill(0.1, 4); fill(0.05 + 5e-5, 12)]
        cl = ConcentrationLimit()
        @test concentration_margins(cl, w).total ≈ 12 * (0.05 + 5e-5)
        @test abs(concentration_margins(cl, w; tol = 1e-4).total) <= tol
        # The estimator and the vector give `tol` to each rule.
        cms = concentration_margins([ConcentrationLimitEstimator(),
                                     ConcentrationLimitEstimator(; c = 0.2)], w; tol = 1e-4)
        @test all(cm -> abs(cm.total) <= tol, cms)
        # An integer and a rational type do not round, so their tolerance is zero.
        @test PortfolioOptimisers.weights_bound_tolerance(Int) == 0
        @test concentration_margins(cl, [1, 0]).total ≈ 0.6
        clr = ConcentrationLimit(; t = 1 // 20, c = 1 // 10, s = 2 // 5)
        @test concentration_margins(clr, [1 // 2, 1 // 2]).total === 3 // 5
        #=
        HiGHS takes a binary of 4e-7 as zero in this model. The big-M constant of 1000 lets
        its position rise to 5.0024 %. A smaller constant closes the leak.
        =#
        Xs = 0.01 .* randn(StableRNG(18), 120, 30) .+ range(0.002, 0.0002; length = 30)'
        rds = ReturnsResult(; nx = ["a$i" for i in 1:30], X = Xs)
        res = optimise(MeanRisk(; r = ConditionalValueatRisk(),
                                obj = MaximumRatio(; rf = 0.0),
                                opt = JuMPOptimiser(; slv = hslv, ss = 10, conc = cl)), rds)
        @test success(res)
        @test concentration_margins(cl, res.w).total <= tol
        @test concentration_margins(cl, res.w; tol = 1e-5).total <= tol
    end

    @testset "A rule with no position adds no limit" begin
        # Each name of `on` can be on the Non-Investable Axis, so the matrix has no row.
        for alg in (ExactConcentration(), SumLargestConcentration()),
            obj in (MaximumReturn(), MaximumRatio(; rf = 0.0))

            res = solve(; obj = obj,
                        conc = ConcentrationLimit(; set = falses(0, N), alg = alg))
            @test success(res)
            @test isapprox(res.w, solve(; obj = obj).w; atol = 1e-5)
        end
        @test concentration_margins(ConcentrationLimit(; set = falses(0, N)),
                                    fill(1 / N, N)).total == -0.4
        # The width of the matrix must be the count of the assets.
        @test_throws DimensionMismatch solve(;
                                             conc = ConcentrationLimit(;
                                                                       set = ones(1, N - 1)))
    end

    @testset "A block of an allocation set" begin
        cslv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                      settings = "verbose" => false,
                      check_sol = (; allow_local = true, allow_almost = true))
        # The three largest positions sum to at most 0.6.
        cl = ConcentrationLimit(; t = 0.2, c = 0.5, s = 0.6,
                                alg = SumLargestConcentration())
        top3(x) = sum(sort(x; rev = true)[1:3])
        function resolve(set)
            return PortfolioOptimisers.resolve_allocation_set(set, 6, false, Float64)
        end
        q = [0.5, 0.3, 0.1, 0.1, 0.0, 0.0]
        wh = fill(1 / 6, 6)
        set = ProgrammeAllocationSet(; slv = cslv)
        setc = ProgrammeAllocationSet(; slv = cslv, conc = cl)
        p0 = PortfolioOptimisers.project(EuclideanProjection(), resolve(set), q, wh)
        @test isapprox(p0, q; atol = 1e-5)
        p1 = PortfolioOptimisers.project(EuclideanProjection(), resolve(setc), q, wh)
        @test isapprox(sum(p1), 1; atol = tol)
        @test isapprox(top3(p1), 0.6; atol = 1e-5)
        @test concentration_margins(cl, p1; tol = 1e-5).total <= 1e-5
        # The block of the leader and the block of the set each state a rule in one model.
        rd6 = ReturnsResult(; nx = nx[1:6], X = X[:, 1:6])
        lcl = ConcentrationLimit(; t = 0.2, c = 0.45, s = 0.9,
                                 alg = SumLargestConcentration())
        lead = MeanRisk(; r = StandardDeviation(), obj = MaximumReturn(),
                        opt = JuMPOptimiser(; slv = cslv, conc = lcl))
        function ftl(s)
            alg = FollowTheLeader(; opt = lead)
            return optimise(PortfolioOptimisers.OnlinePortfolioSelection(; alg = alg,
                                                                         set = s), rd6).w
        end
        wl = ftl(set)
        @test isapprox(maximum(wl), 0.45; atol = 1e-5)
        @test top3(wl) > 0.65
        wb = ftl(setc)
        @test isapprox(top3(wb), 0.6; atol = 1e-5)
        @test sum(sort(wb; rev = true)[1:5]) <= 0.9 + 1e-5
    end

    @testset "NCO views the matrix of a rule at each cluster" begin
        issuer = ["i$(cld(i, 2))" for i in 1:N]
        sets = UniverseSets(; dict = Dict("nx" => nx, "nx_issuer" => issuer))
        est = AssetSetsMatrixEstimator(; val = "nx_issuer")
        A = Matrix(asset_sets_matrix(est, sets))
        function nco(conc, s = nothing)
            opti = MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumReturn(),
                            opt = JuMPOptimiser(; slv = hslv, sets = s, conc = conc))
            return optimise(NestedClustered(; opti = opti, opto = EqualWeighted()), rd)
        end
        numbers = (; t = 0.2, c = 0.6, s = 0.6)
        re = nco(ConcentrationLimitEstimator(; numbers..., set = est), sets)
        @test success(re)
        # The view keeps the columns of the cluster, so each form gives the book of the
        # estimator.
        @test isapprox(nco(ConcentrationLimit(; numbers..., set = A)).w, re.w; atol = tol)
        @test isapprox(nco([ConcentrationLimitEstimator(; numbers..., set = A)]).w, re.w;
                       atol = tol)
        # The rule binds.
        @test !isapprox(nco(nothing).w, re.w; atol = 1e-2)
    end
end
