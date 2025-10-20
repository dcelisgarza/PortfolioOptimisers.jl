#=
@setup_workload begin
    rng = Xoshiro(42)
    X = randn(rng, 252, 15) * 0.01
    F = randn(rng, 252, 5) * 0.01
    ew = eweights(1:(252), inv(252))
    rf = 4.2 / 100 / 252
    @compile_workload begin
        rd = ReturnsResult(; nx = ["_$i" for i in 1:15], X = X, nf = ["_$i" for i in 1:5],
                           F = F)
        sets = AssetSets(; dict = Dict("nx" => rd.nx, "group1" => ["_$i" for i in 1:3:15]))
        fsets = AssetSets(; dict = Dict("nx" => rd.nf))
        pes = [HighOrderPriorEstimator(; pe = EmpiricalPrior()),
               HighOrderPriorEstimator(;
                                       kte = Cokurtosis(; alg = Semi(),
                                                        mp = DefaultMatrixProcessing()),
                                       ske = Coskewness(; alg = Semi()),
                                       pe = EmpiricalPrior(;
                                                           me = SimpleExpectedReturns(;
                                                                                      w = ew),
                                                           ce = PortfolioOptimisersCovariance(;
                                                                                              ce = Covariance(;
                                                                                                              me = SimpleExpectedReturns(;
                                                                                                                                         w = ew),
                                                                                                              ce = GeneralCovariance(;
                                                                                                                                             ce = SimpleCovariance(;
                                                                                                                                                                   corrected = false),
                                                                                                                                             w = ew),
                                                                                                              alg = Semi()),
                                                                                              mp = DefaultMatrixProcessing(;
                                                                                                                           denoise = Denoise(),
                                                                                                                           detone = Detone(),
                                                                                                                           alg = LoGo())),
                                                           horizon = 252)),
               FactorPrior(; re = StepwiseRegression(; crit = PValue(; threshold = 0.8)),
                           pe = EmpiricalPrior(;
                                               ce = PortfolioOptimisersCovariance(;
                                                                                  ce = GerberCovariance(;
                                                                                                        ve = SimpleVariance(;
                                                                                                                            me = ShrunkExpectedReturns())),
                                                                                  mp = DefaultMatrixProcessing(;
                                                                                                               denoise = Denoise(;
                                                                                                                                 alg = SpectralDenoise()))))),
               FactorPrior(;
                           pe = EmpiricalPrior(;
                                               ce = PortfolioOptimisersCovariance(;
                                                                                  ce = GerberCovariance(;
                                                                                                        alg = Gerber0(),
                                                                                                        ve = SimpleVariance(;
                                                                                                                            me = ShrunkExpectedReturns(;
                                                                                                                                                       alg = BayesStein(;
                                                                                                                                                                        target = VolatilityWeighted())))),
                                                                                  mp = DefaultMatrixProcessing(;
                                                                                                               denoise = Denoise(;
                                                                                                                                 alg = FixedDenoise())))),
                           re = StepwiseRegression(; target = GeneralisedLinearModel(),
                                                   crit = AIC(), alg = Backward()),
                           mp = DefaultMatrixProcessing()),
               FactorPrior(;
                           pe = EmpiricalPrior(;
                                               ce = PortfolioOptimisersCovariance(;
                                                                                  ce = GerberCovariance(;
                                                                                                        alg = Gerber2(),
                                                                                                        ve = SimpleVariance(;
                                                                                                                            me = ShrunkExpectedReturns(;
                                                                                                                                                       alg = BodnarOkhrinParolya(;
                                                                                                                                                                                 target = MeanSquaredError())))))),
                           re = StepwiseRegression(; crit = AICC()), rsd = false),
               FactorPrior(;
                           pe = EmpiricalPrior(;
                                               ce = PortfolioOptimisersCovariance(;
                                                                                  ce = GerberCovariance(;
                                                                                                        alg = StandardisedGerber0()))),
                           re = StepwiseRegression(; crit = BIC()),
                           ve = SimpleVariance(; corrected = false)),
               FactorPrior(;
                           pe = EmpiricalPrior(;
                                               ce = PortfolioOptimisersCovariance(;
                                                                                  ce = GerberCovariance(;
                                                                                                        alg = StandardisedGerber1()))),
                           re = StepwiseRegression(; crit = RSquared())),
               FactorPrior(;
                           pe = EmpiricalPrior(;
                                               ce = PortfolioOptimisersCovariance(;
                                                                                  ce = GerberCovariance(;
                                                                                                        alg = StandardisedGerber2()))),
                           re = StepwiseRegression(; crit = AdjustedRSquared())),
               FactorPrior(;
                           pe = EmpiricalPrior(; me = EquilibriumExpectedReturns(),
                                               ce = PortfolioOptimisersCovariance(;
                                                                                  ce = SmythBrobyCovariance())),
                           re = DimensionReductionRegression()),
               FactorPrior(;
                           pe = EmpiricalPrior(; me = EquilibriumExpectedReturns(),
                                               ce = PortfolioOptimisersCovariance(;
                                                                                  ce = SmythBrobyCovariance(;
                                                                                                            alg = SmythBroby0()))),
                           re = DimensionReductionRegression(; drtgt = PPCA(),
                                                             retgt = GeneralisedLinearModel())),
               BlackLittermanPrior(;
                                   views = LinearConstraintEstimator(;
                                                                     val = Union{String,
                                                                                 Expr}["_1==0.008",
                                                                                       :(group1 ==
                                                                                         0.0091)]),
                                   sets = sets,
                                   pe = EmpiricalPrior(;
                                                       me = ExcessExpectedReturns(;
                                                                                  rf = rf),
                                                       ce = PortfolioOptimisersCovariance(;
                                                                                          ce = SmythBrobyCovariance(;
                                                                                                                    alg = SmythBroby1())))),
               BayesianBlackLittermanPrior(;
                                           views = LinearConstraintEstimator(;
                                                                             val = Union{String,
                                                                                         Expr}["_1==0.008"]),
                                           sets = fsets, views_conf = [0.25],
                                           pe = FactorPrior(;
                                                            re = StepwiseRegression(;
                                                                                    crit = AIC()),
                                                            pe = EmpiricalPrior(;
                                                                                ce = PortfolioOptimisersCovariance(;
                                                                                                                   ce = SmythBrobyCovariance(;
                                                                                                                                             alg = SmythBroby2()))))),
               FactorBlackLittermanPrior(; re = StepwiseRegression(; crit = RSquared()),
                                         views = LinearConstraintEstimator(;
                                                                           val = ["_1==0.008"]),
                                         sets = fsets, views_conf = 0.25,
                                         pe = EmpiricalPrior(;
                                                             ce = PortfolioOptimisersCovariance(;
                                                                                                ce = SmythBrobyCovariance(;
                                                                                                                          alg = StandardisedSmythBroby0()))))]
        # Missing StandardisedSmythBroby1, StandardisedSmythBroby2, SmythBrobyGerber0, SmythBrobyGerber2, StandardisedSmythBrobyGerber0, StandardisedSmythBrobyGerber1, StandardisedSmythBrobyGerber2
        for pe in pes
            pr = prior(pe, rd)
        end
    end
end
=#
