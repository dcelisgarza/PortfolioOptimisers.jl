@testset "Base cross validation" begin
    using Test, PortfolioOptimisers, Clarabel, Dates, StableRNGs
    naive(w; retcode = OptimisationSuccess()) = NaiveOptimisationResult(; pr = nothing,
                                                                        wb = nothing,
                                                                        retcode = retcode,
                                                                        w = w, fb = nothing)
    fold(X) = PredictionResult(; res = naive([0.5, 0.5]),
                               rd = PredictionReturnsResult(; nx = ["p"], X = X))
    X2 = [[0.01, 0.02, 0.03], [0.0, 0.01, -0.01]]
    @testset "A split result splits to itself" begin
        rd = ReturnsResult(; nx = ["a", "b"], X = randn(StableRNG(1), 12, 2) / 100)
        res = split(KFold(; n = 3), rd)
        @test split(res, rd) === res
        @test n_splits(res) == n_splits(res, rd) == 3
    end
    @testset "PredictionReturnsResult checks a population's benchmark" begin
        @test PredictionReturnsResult(; X = X2, B = [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]]).B ==
              [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]]
        @test_throws DimensionMismatch PredictionReturnsResult(; X = X2,
                                                               B = [[0.1, 0.2],
                                                                    [0.0, 0.0, 0.0]])
        @test_throws ArgumentError PredictionReturnsResult(; X = X2, B = [0.1, 0.2, 0.3])
        @test_throws DimensionMismatch PredictionReturnsResult(; nf = ["f1", "f2"],
                                                               F = [1.0 2.0; 3.0 4.0],
                                                               B = [[0.1, 0.2, 0.3]],
                                                               ts = [Date(2020, 1, 1),
                                                                     Date(2020, 1, 2)])
        @test_throws DimensionMismatch PredictionReturnsResult(; X = X2,
                                                               B = [[0.1, 0.2], [0.0, 0.0]],
                                                               ts = [Date(2020, 1, i)
                                                                     for i in 1:3])
    end
    @testset "PredictionReturnsResult admits a population's iv without ivpa" begin
        iv = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        prr = PredictionReturnsResult(; X = X2, iv = iv)
        @test prr.iv == iv
        @test isnothing(prr.ivpa)
        @test_throws DimensionMismatch PredictionReturnsResult(; X = X2, iv = iv,
                                                               ivpa = [1.0])
        @test_throws DomainError PredictionReturnsResult(; X = X2, iv = iv,
                                                         ivpa = [1.0, 0.0])
        @test_throws IsNothingError PredictionReturnsResult(; iv = [0.1, 0.2, 0.3])
        @test_throws IsNothingError PredictionReturnsResult(; iv = iv, ivpa = [1.0, 1.0])
    end
    @testset "A frontier predicts over implied volatilities without a premium" begin
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = "verbose" => false)
        rng = StableRNG(1)
        X = randn(rng, 60, 4) / 100 .+ 0.001
        iv = abs.(randn(rng, 60, 4)) / 10 .+ 0.1
        rd = ReturnsResult(; nx = ["a", "b", "c", "d"], X = X, iv = iv)
        mr = MeanRisk(;
                      opt = JuMPOptimiser(; slv = slv,
                                          ret = ArithmeticReturn(;
                                                                 settings = JuMPReturnsSettings(;
                                                                                                lb = Frontier(;
                                                                                                              N = 3)))))
        res = optimise(mr, rd)
        pred = predict(res, rd)
        @test isnothing(pred.rd.ivpa)
        @test length(pred.rd.iv) == 3
        for (ivi, wi) in zip(pred.rd.iv, res.w)
            @test isapprox(ivi, iv * (abs.(wi) / sum(abs, wi)))
        end
    end
    @testset "Population measures and the order of folds" begin
        mp = MultiPeriodPredictionResult(; pred = [fold([0.01, -0.02]), fold([0.03])])
        r = ConditionalValueatRisk()
        @test expected_risk(r, [mp, mp]) == fill(expected_risk(r, mp), 2)
        rd = ReturnsResult(; nx = ["a", "b"], X = randn(StableRNG(2), 12, 2) / 100)
        res = split(KFold(; n = 3), rd)
        preds = [fold([Float64(i)]) for i in 1:3]
        perm = sortperm(res.test_idx; by = first)
        @test PortfolioOptimisers.sort_predictions!(res, preds) == preds[perm]
        shuffled = [[5, 6], [1, 2], [3, 4]]
        @test [p.rd.X[1] for p in PortfolioOptimisers.sort_predictions!(shuffled, preds)] ==
              [2.0, 3.0, 1.0]
        @test [p.rd.X[1] for p in preds] == [1.0, 2.0, 3.0]
    end
    @testset "The fold-loop predicates" begin
        @test PortfolioOptimisers.folds_are_time_ordered(nothing) === true
        @test PortfolioOptimisers.folds_are_time_ordered(IndexWalkForward(10, 5)) === true
        @test PortfolioOptimisers.folds_are_time_ordered(KFold()) === false
        @test PortfolioOptimisers.folds_are_stepped(nothing) === false
        @test PortfolioOptimisers.folds_are_stepped(KFold()) === false
    end
    @testset "A population of single folds ranks to single folds" begin
        low = fold([0.01, -0.02, 0.03])
        high = fold([0.05, -0.20, 0.0])
        pop = PopulationPredictionResult(; pred = [high, low])
        @test pop.pred[1] === high
        r = ConditionalValueatRisk()
        sorted = sort_by_measure(pop, r)
        @test all(x -> isa(x, PredictionResult), sorted)
        @test sorted[1] === low
        @test sorted[2] === high
        @test PortfolioOptimisers.quantile_by_measure(pop, r, 1.0) === high
        @test PortfolioOptimisers.quantile_by_measure(pop, r, 0.0) === low
    end
    @testset "A failed member keeps its weights with no previous weights" begin
        s = OptimisationSuccess()
        f = OptimisationFailure(; res = nothing)
        W = [[0.5, 0.5], [NaN, NaN], [0.1, 0.9]]
        hsw = PortfolioOptimisers.held_start_weights
        @test isequal(hsw([s, f, s], W, nothing), W)
        @test hsw([s, f, s], W, [0.3, 0.7]) == [[0.5, 0.5], [0.3, 0.7], [0.1, 0.9]]
        @test hsw(f, W, [0.3, 0.7]) == fill([0.3, 0.7], 3)
        @test_throws DimensionMismatch hsw([s, f, s], W, [[1.0, 0.0]])
    end
end
