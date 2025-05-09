@safetestset "DBHT" begin
    using PortfolioOptimisers, DataFrames, CSV, Random, StableRNGs, Test, StatsBase,
          Statistics, SparseArrays

    function find_tol(a1, a2; name1 = :a1, name2 = :a2)
        for rtol ∈
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; rtol = rtol)
                println("isapprox($name1, $name2, rtol = $(rtol))")
                break
            end
        end
    end
    @testset "DBHT Clustering tests" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        ce = PortfolioOptimisersCovariance()
        de = Distance(; alg = SimpleDistance())
        rho = cor(ce, X)
        dist = distance(de, rho, X)

        @test isapprox(dbht_similarity(GeneralExponentialSimilarity(); D = rho),
                       dbht_similarity(ExponentialSimilarity(); D = rho))

        sim = MaximumDistanceSimilarity()
        S = dbht_similarity(sim; S = rho, D = dist)

        root = UniqueRoot()
        T8, Rpm, Adjv, Dpm, Mv, Z1, dbht = DBHTs(dist, S; branchorder = :default,
                                                 root = root)
        m1 = Z1[:, 1] .< 0
        m2 = Z1[:, 2] .< 0
        Z1[.!m1, 1] .+= size(Z1, 1) + 1
        Z1[.!m2, 2] .+= size(Z1, 1) + 1
        Z1[m1, 1] .= -Z1[m1, 1]
        Z1[m2, 2] .= -Z1[m2, 2]
        Z1[:, 1:2] .-= 1
        Z1_t = reshape([4.0, 9.0, 5.0, 16.0, 7.0, 6.0, 0.0, 3.0, 1.0, 18.0, 14.0, 24.0,
                        21.0, 29.0, 20.0, 26.0, 23.0, 32.0, 35.0, 12.0, 11.0, 8.0, 22.0,
                        17.0, 13.0, 2.0, 10.0, 19.0, 28.0, 15.0, 27.0, 25.0, 31.0, 30.0,
                        33.0, 34.0, 36.0, 37.0, 0.05263157894736842, 0.05555555555555555,
                        0.058823529411764705, 0.0625, 0.06666666666666667,
                        0.07142857142857142, 0.07692307692307693, 0.08333333333333333,
                        0.09090909090909091, 0.1, 0.1111111111111111, 0.125,
                        0.14285714285714285, 0.16666666666666666, 0.2, 0.25,
                        0.3333333333333333, 0.5, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0,
                        2.0, 2.0, 3.0, 2.0, 4.0, 4.0, 7.0, 4.0, 9.0, 7.0, 11.0, 20.0], :, 4)
        @test isapprox(Z1, Z1_t)

        A1, tri1, separators1, cliques1, cliqueTree1 = PortfolioOptimisers.PMFG_T2s(S, 5)

        A1_t = sparse([2, 3, 11, 1, 3, 8, 11, 18, 19, 20, 1, 2, 11, 18, 5, 8, 10, 11, 12,
                       13, 14, 15, 18, 4, 6, 8, 9, 11, 13, 15, 17, 5, 9, 11, 15, 17, 10, 11,
                       14, 2, 4, 5, 11, 18, 19, 20, 5, 6, 17, 4, 7, 11, 12, 14, 15, 16, 1,
                       2, 3, 4, 5, 6, 7, 8, 10, 14, 15, 16, 18, 19, 4, 10, 14, 4, 5, 15, 4,
                       7, 10, 11, 12, 4, 5, 6, 10, 11, 13, 16, 17, 10, 11, 15, 5, 6, 9, 15,
                       2, 3, 4, 8, 11, 2, 8, 11, 20, 2, 8, 19],
                      [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                       5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8,
                       9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
                       11, 11, 11, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 14, 14,
                       15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 17, 17, 17, 17, 18, 18,
                       18, 18, 18, 19, 19, 19, 19, 20, 20, 20],
                      [0.50318866710514, 0.507434024660484, 0.5018071219443814,
                       0.50318866710514, 0.5252367663619791, 0.518627322701924,
                       0.5070522680394457, 0.5271846677129006, 0.5128869332152788,
                       0.518925287022916, 0.507434024660484, 0.5252367663619791,
                       0.5156605204548081, 0.49795785351422994, 0.5359616871182638,
                       0.5194872306284464, 0.5178183019012799, 0.5364885144854562,
                       0.5102027495773904, 0.5023076443069009, 0.5030007393514963,
                       0.5045902552652581, 0.5188563953076473, 0.5359616871182638,
                       0.4897742884134034, 0.5031837562210253, 0.5101603180831713,
                       0.510580170676277, 0.5017118392395952, 0.5449102305001545,
                       0.5113209057501814, 0.4897742884134034, 0.5157226027305121,
                       0.5306243687224272, 0.5322057813235671, 0.5087642439245168,
                       0.5081666514428258, 0.5016871337678717, 0.5126305468474727,
                       0.518627322701924, 0.5194872306284464, 0.5031837562210253,
                       0.5122392148577684, 0.5344154186732986, 0.5152197352334124,
                       0.5097041122747309, 0.5101603180831713, 0.5157226027305121,
                       0.501701674296825, 0.5178183019012799, 0.5081666514428258,
                       0.5196487318349611, 0.5241234543691717, 0.5182620453992988,
                       0.5304905692697387, 0.49615244465353336, 0.5018071219443814,
                       0.5070522680394457, 0.5156605204548081, 0.5364885144854562,
                       0.510580170676277, 0.5306243687224272, 0.5016871337678717,
                       0.5122392148577684, 0.5196487318349611, 0.5195075036072444,
                       0.5122738402711995, 0.5136428469691559, 0.4949709331031463,
                       0.5083232047006303, 0.5102027495773904, 0.5241234543691717,
                       0.5191955897948763, 0.5023076443069009, 0.5017118392395952,
                       0.5142915044271688, 0.5030007393514963, 0.5126305468474727,
                       0.5182620453992988, 0.5195075036072444, 0.5191955897948763,
                       0.5045902552652581, 0.5449102305001545, 0.5322057813235671,
                       0.5304905692697387, 0.5122738402711995, 0.5142915044271688,
                       0.5072568684325893, 0.5284158676704074, 0.49615244465353336,
                       0.5136428469691559, 0.5072568684325893, 0.5113209057501814,
                       0.5087642439245168, 0.501701674296825, 0.5284158676704074,
                       0.5271846677129006, 0.49795785351422994, 0.5188563953076473,
                       0.5344154186732986, 0.4949709331031463, 0.5128869332152788,
                       0.5152197352334124, 0.5083232047006303, 0.5078617977059765,
                       0.518925287022916, 0.5097041122747309, 0.5078617977059765], 20, 20)
        tri1_t = reshape([15, 5, 15, 15, 15, 4, 15, 5, 15, 5, 4, 11, 4, 10, 5, 4, 4, 11, 11,
                          8, 11, 18, 11, 8, 8, 2, 5, 6, 11, 10, 15, 5, 15, 11, 11, 2, 5, 4,
                          5, 4, 11, 11, 11, 11, 6, 6, 10, 10, 14, 14, 11, 11, 8, 8, 18, 18,
                          2, 2, 2, 2, 19, 19, 17, 17, 14, 14, 4, 4, 10, 10, 3, 3, 13, 8, 17,
                          10, 16, 14, 6, 6, 17, 9, 12, 7, 12, 12, 8, 18, 18, 19, 3, 2, 1, 3,
                          19, 20, 20, 20, 9, 9, 7, 7, 13, 13, 16, 16, 1, 1], :, 3)
        separators1_t = reshape([15, 15, 15, 4, 4, 5, 4, 11, 11, 11, 8, 5, 11, 15, 15, 11,
                                 4, 5, 5, 11, 10, 4, 11, 8, 18, 8, 2, 6, 10, 5, 11, 2, 11,
                                 11, 6, 10, 14, 11, 8, 18, 2, 2, 19, 17, 14, 4, 10, 3], :,
                                3)
        cliques1_t = reshape([15, 15, 15, 15, 4, 4, 5, 4, 11, 11, 11, 8, 5, 11, 15, 15, 11,
                              5, 4, 5, 5, 11, 10, 4, 11, 8, 18, 8, 2, 6, 10, 5, 11, 2, 4,
                              11, 11, 6, 10, 14, 11, 8, 18, 2, 2, 19, 17, 14, 4, 10, 3, 11,
                              10, 6, 17, 14, 12, 8, 18, 2, 3, 19, 20, 9, 7, 13, 16, 1], :,
                             4)
        cliqueTree1_t = sparse([4, 5, 8, 16, 1, 3, 6, 7, 8, 14, 15, 1, 2, 7, 13, 15, 16, 1,
                                3, 15, 2, 7, 8, 16, 5, 14, 1, 2, 3, 5, 9, 11, 15, 2, 5, 7,
                                10, 11, 8, 12, 17, 9, 11, 8, 9, 10, 17, 11, 4, 5, 6, 16, 2,
                                3, 4, 7, 2, 3, 5, 14, 10, 11],
                               [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4,
                                5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9,
                                9, 10, 10, 11, 11, 11, 11, 12, 13, 14, 14, 14, 15, 15, 15,
                                15, 16, 16, 16, 16, 17, 17],
                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1], 17, 17)
        @test isapprox(A1, A1_t)
        @test isapprox(tri1, tri1_t)
        @test isapprox(separators1, separators1_t)
        @test isapprox(cliques1, cliques1_t)
        @test isapprox(cliqueTree1, cliqueTree1_t)

        rng = StableRNG(1357268)
        X = randn(rng, 1000, 20)
        ce = PortfolioOptimisersCovariance()
        de = Distance(; alg = SimpleDistance())
        rho = cor(ce, X)
        dist = distance(de, rho, X)

        sim = ExponentialSimilarity()
        S = dbht_similarity(sim; S = rho, D = dist)

        root = EqualRoot()
        T8, Rpm, Adjv, Dpm, Mv, Z2, dbht = DBHTs(dist, S; branchorder = :optimal,
                                                 root = root)
        m1 = Z2[:, 1] .< 0
        m2 = Z2[:, 2] .< 0
        Z2[.!m1, 1] .+= size(Z2, 1) + 1
        Z2[.!m2, 2] .+= size(Z2, 1) + 1
        Z2[m1, 1] .= -Z2[m1, 1]
        Z2[m2, 2] .= -Z2[m2, 2]
        Z2[:, 1:2] .-= 1
        Z2_t = reshape([18.0, 2.0, 0.0, 5.0, 22.0, 1.0, 20.0, 25.0, 3.0, 6.0, 13.0, 4.0,
                        11.0, 10.0, 8.0, 32.0, 30.0, 31.0, 27.0, 19.0, 15.0, 7.0, 16.0,
                        23.0, 21.0, 24.0, 26.0, 9.0, 17.0, 29.0, 12.0, 14.0, 28.0, 33.0,
                        34.0, 35.0, 36.0, 37.0, 0.125, 0.14285714285714285,
                        0.16666666666666666, 0.2, 0.25, 0.3333333333333333, 0.5, 1.0, 0.1,
                        0.1111111111111111, 0.125, 0.14285714285714285, 0.16666666666666666,
                        0.2, 0.25, 0.3333333333333333, 0.5, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                        4.0, 3.0, 6.0, 9.0, 2.0, 2.0, 3.0, 2.0, 2.0, 3.0, 4.0, 6.0, 9.0,
                        11.0, 20.0], :, 4)
        @test isapprox(Z2, Z2_t)

        A2, tri2, separators2, cliques2, cliqueTree2 = PMFG_T2s(S, 5)
        A2_t = sparse([2, 3, 4, 8, 16, 1, 3, 4, 1, 2, 4, 16, 1, 2, 3, 5, 6, 8, 9, 10, 11,
                       12, 15, 16, 17, 18, 19, 20, 4, 10, 12, 13, 4, 8, 16, 17, 11, 14, 18,
                       1, 4, 6, 10, 11, 16, 17, 4, 11, 18, 4, 5, 8, 11, 12, 13, 14, 18, 4,
                       7, 8, 9, 10, 14, 17, 18, 20, 4, 5, 10, 13, 15, 18, 5, 10, 12, 7, 10,
                       11, 18, 4, 12, 18, 1, 3, 4, 6, 8, 4, 6, 8, 11, 19, 20, 4, 7, 9, 10,
                       11, 12, 14, 15, 4, 17, 20, 4, 11, 17, 19],
                      [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                       4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8,
                       9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11,
                       11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 14, 14, 14, 14, 15, 15,
                       15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18,
                       18, 18, 18, 19, 19, 19, 20, 20, 20, 20],
                      [0.4861216051823635, 0.4964847416158843, 0.49564295678779974,
                       0.5051717567105267, 0.49560748950891576, 0.4861216051823635,
                       0.4914737679774494, 0.49115813252112195, 0.4964847416158843,
                       0.4914737679774494, 0.49910216476196656, 0.5049714854419631,
                       0.49564295678779974, 0.49115813252112195, 0.49910216476196656,
                       0.48860867116211915, 0.4944997037713434, 0.4944385041612001,
                       0.4915748812354363, 0.5011281795557058, 0.49935030296091015,
                       0.494566228074772, 0.4948549249227664, 0.49407930735033495,
                       0.49621921654307166, 0.4913944495315472, 0.49763836261865485,
                       0.49877971984786884, 0.48860867116211915, 0.4927824556031762,
                       0.5044654093121019, 0.5018265195209648, 0.4944997037713434,
                       0.5011783096130855, 0.4996271448164821, 0.5005524984723763,
                       0.490898804330398, 0.4973516480039865, 0.5001784900067453,
                       0.5051717567105267, 0.4944385041612001, 0.5011783096130855,
                       0.4986671156736144, 0.49887171590469714, 0.4985024025332727,
                       0.49959404533694635, 0.4915748812354363, 0.49465531151178377,
                       0.5029835299236661, 0.5011281795557058, 0.4927824556031762,
                       0.4986671156736144, 0.5009628686287566, 0.5015829169124495,
                       0.4945685108549255, 0.5086820179705279, 0.5015814889580512,
                       0.49935030296091015, 0.490898804330398, 0.49887171590469714,
                       0.49465531151178377, 0.5009628686287566, 0.49223372846174424,
                       0.49821424132101594, 0.508404707228152, 0.5043222257853511,
                       0.494566228074772, 0.5044654093121019, 0.5015829169124495,
                       0.48959262403618975, 0.5042104154882951, 0.49342143601570715,
                       0.5018265195209648, 0.4945685108549255, 0.48959262403618975,
                       0.4973516480039865, 0.5086820179705279, 0.49223372846174424,
                       0.49462726239155747, 0.4948549249227664, 0.5042104154882951,
                       0.5001675389401475, 0.49560748950891576, 0.5049714854419631,
                       0.49407930735033495, 0.4996271448164821, 0.4985024025332727,
                       0.49621921654307166, 0.5005524984723763, 0.49959404533694635,
                       0.49821424132101594, 0.4983777389244161, 0.49490127907455145,
                       0.4913944495315472, 0.5001784900067453, 0.5029835299236661,
                       0.5015814889580512, 0.508404707228152, 0.49342143601570715,
                       0.49462726239155747, 0.5001675389401475, 0.49763836261865485,
                       0.4983777389244161, 0.5039779324575323, 0.49877971984786884,
                       0.5043222257853511, 0.49490127907455145, 0.5039779324575323], 20, 20)
        tri2_t = reshape([1, 2, 1, 1, 1, 3, 1, 4, 4, 16, 4, 8, 4, 8, 4, 8, 4, 11, 4, 17, 4,
                          17, 11, 10, 4, 10, 4, 18, 4, 11, 11, 18, 4, 10, 10, 12, 2, 3, 2,
                          3, 4, 4, 16, 16, 8, 8, 6, 6, 17, 17, 11, 11, 10, 10, 11, 11, 20,
                          20, 18, 18, 18, 18, 12, 12, 18, 18, 14, 14, 12, 12, 5, 5, 3, 4, 4,
                          16, 8, 16, 8, 6, 10, 6, 17, 17, 19, 11, 9, 10, 5, 14, 20, 20, 19,
                          19, 7, 14, 15, 12, 15, 15, 9, 9, 7, 7, 5, 13, 13, 13], :, 3)
        separators2_t = reshape([1, 1, 4, 4, 4, 4, 4, 4, 4, 11, 4, 4, 4, 11, 4, 10, 3, 4,
                                 16, 8, 8, 8, 11, 17, 17, 10, 10, 18, 11, 18, 10, 12, 4, 16,
                                 8, 6, 17, 11, 10, 11, 20, 18, 18, 12, 18, 14, 12, 5], :, 3)
        cliques2_t = reshape([1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 11, 4, 4, 4, 11, 4, 10, 2, 3, 4,
                              16, 8, 8, 8, 11, 17, 17, 10, 10, 18, 11, 18, 10, 12, 3, 4, 16,
                              8, 6, 17, 11, 10, 11, 20, 18, 18, 12, 18, 14, 12, 5, 4, 16, 8,
                              6, 17, 11, 10, 18, 20, 19, 14, 12, 15, 9, 7, 5, 13], :, 4)
        cliqueTree2_t = sparse([3, 1, 4, 2, 5, 6, 7, 3, 6, 7, 4, 7, 9, 10, 4, 5, 8, 10, 14,
                                4, 5, 6, 9, 11, 12, 14, 16, 7, 9, 13, 15, 16, 6, 7, 8, 14,
                                6, 9, 8, 12, 14, 8, 11, 14, 17, 12, 14, 16, 7, 8, 9, 11, 12,
                                13, 15, 11, 14, 8, 12, 13, 16],
                               [1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7,
                                7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 11,
                                11, 11, 12, 12, 12, 12, 13, 13, 13, 14, 14, 14, 14, 14, 14,
                                14, 15, 15, 16, 16, 16, 17],
                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1], 17, 17)
        @test isapprox(A2, A2_t)
        @test isapprox(tri2, tri2_t)
        @test isapprox(separators2, separators2_t)
        @test isapprox(cliques2, cliques2_t)
        @test isapprox(cliqueTree2, cliqueTree2_t)

        T8, Rpm, Adjv, Dpm, Mv, Z3, dbht = DBHTs(dist, S; branchorder = :r, root = root)
        m1 = Z3[:, 1] .< 0
        m2 = Z3[:, 2] .< 0
        Z3[.!m1, 1] .+= size(Z3, 1) + 1
        Z3[.!m2, 2] .+= size(Z3, 1) + 1
        Z3[m1, 1] .= -Z3[m1, 1]
        Z3[m2, 2] .= -Z3[m2, 2]
        Z3[:, 1:2] .-= 1
        Z3_t = reshape([18.0, 2.0, 0.0, 5.0, 22.0, 1.0, 20.0, 25.0, 3.0, 6.0, 13.0, 4.0,
                        11.0, 10.0, 8.0, 32.0, 30.0, 31.0, 27.0, 19.0, 15.0, 7.0, 16.0,
                        23.0, 21.0, 24.0, 26.0, 9.0, 17.0, 29.0, 12.0, 14.0, 28.0, 33.0,
                        34.0, 35.0, 36.0, 37.0, 0.125, 0.14285714285714285,
                        0.16666666666666666, 0.2, 0.25, 0.3333333333333333, 0.5, 1.0, 0.1,
                        0.1111111111111111, 0.125, 0.14285714285714285, 0.16666666666666666,
                        0.2, 0.25, 0.3333333333333333, 0.5, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                        4.0, 3.0, 6.0, 9.0, 2.0, 2.0, 3.0, 2.0, 2.0, 3.0, 4.0, 6.0, 9.0,
                        11.0, 20.0], :, 4)
        @test isapprox(Z3, Z3_t)
    end
    @testset "LoGo" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        ce = PortfolioOptimisersCovariance()
        sigma = cov(ce, X)

        logo_t = CSV.read(joinpath(@__DIR__, "./assets/LoGo-MaximumDistanceSimilarity.csv"),
                          DataFrame)

        des = [Distance(; alg = CanonicalDistance()),
               DistanceDistance(; alg = CanonicalDistance()),
               Distance(; alg = SimpleDistance()),
               DistanceDistance(; alg = SimpleDistance()),
               Distance(; alg = SimpleAbsoluteDistance()),
               DistanceDistance(; alg = SimpleAbsoluteDistance()),
               Distance(; alg = CorrelationDistance()),
               DistanceDistance(; alg = CorrelationDistance()),
               Distance(; alg = LogDistance()), DistanceDistance(; alg = LogDistance()),
               Distance(; alg = VariationInfoDistance()),
               DistanceDistance(; alg = VariationInfoDistance())]

        for i ∈ 1:ncol(logo_t)
            sigma1 = copy(sigma)
            PortfolioOptimisers.matrix_processing_algorithm!(PortfolioOptimisers.LoGo(;
                                                                                      dist = des[i]),
                                                             PosDefEstimator(), sigma1, X)
            MN = size(sigma1)
            res1 = isapprox(sigma1, reshape(logo_t[!, i], MN))
            if !res1
                println("Fails on LoGo MaxDist sim iteration $i")
                find_tol(sigma1, reshape(logo_t[!, i], MN); name1 = :sigma, name2 = :logo_t)
            end
            @test res1
        end

        des = [GeneralDistance(; alg = CanonicalDistance()),
               GeneralDistanceDistance(; alg = CanonicalDistance()),
               GeneralDistance(; alg = SimpleDistance()),
               GeneralDistanceDistance(; alg = SimpleDistance()),
               GeneralDistance(; alg = SimpleAbsoluteDistance()),
               GeneralDistanceDistance(; alg = SimpleAbsoluteDistance()),
               GeneralDistance(; alg = CorrelationDistance()),
               GeneralDistanceDistance(; alg = CorrelationDistance()),
               GeneralDistance(; alg = LogDistance()),
               GeneralDistanceDistance(; alg = LogDistance()),
               GeneralDistance(; alg = VariationInfoDistance()),
               GeneralDistanceDistance(; alg = VariationInfoDistance())]

        for i ∈ 1:ncol(logo_t)
            sigma1 = copy(sigma)
            PortfolioOptimisers.matrix_processing_algorithm!(PortfolioOptimisers.LoGo(;
                                                                                      dist = des[i]),
                                                             PosDefEstimator(), sigma1, X)
            MN = size(sigma1)
            res1 = isapprox(sigma1, reshape(logo_t[!, i], MN))
            if !res1
                println("Fails on LoGo General distance MaxDist sim iteration $i")
                find_tol(sigma1, reshape(logo_t[!, i], MN); name1 = :sigma, name2 = :logo_t)
            end
            @test res1
        end

        logo_t = CSV.read(joinpath(@__DIR__, "./assets/LoGo-ExponentialSimilarity.csv"),
                          DataFrame)

        des = [Distance(; alg = CanonicalDistance()),
               DistanceDistance(; alg = CanonicalDistance()),
               Distance(; alg = SimpleDistance()),
               DistanceDistance(; alg = SimpleDistance()),
               Distance(; alg = SimpleAbsoluteDistance()),
               DistanceDistance(; alg = SimpleAbsoluteDistance()),
               Distance(; alg = CorrelationDistance()),
               DistanceDistance(; alg = CorrelationDistance()),
               Distance(; alg = LogDistance()), DistanceDistance(; alg = LogDistance()),
               Distance(; alg = VariationInfoDistance()),
               DistanceDistance(; alg = VariationInfoDistance())]

        for i ∈ 1:ncol(logo_t)
            sigma1 = copy(sigma)
            PortfolioOptimisers.matrix_processing_algorithm!(PortfolioOptimisers.LoGo(;
                                                                                      dist = des[i],
                                                                                      sim = ExponentialSimilarity()),
                                                             PosDefEstimator(), sigma1, X)
            MN = size(sigma1)
            res1 = isapprox(sigma1, reshape(logo_t[!, i], MN))
            if !res1
                println("Fails on LoGo ExpDist sim iteration $i")
                find_tol(sigma1, reshape(logo_t[!, i], MN); name1 = :sigma, name2 = :logo_t)
            end
            @test res1
        end

        des = [GeneralDistance(; alg = CanonicalDistance()),
               GeneralDistanceDistance(; alg = CanonicalDistance()),
               GeneralDistance(; alg = SimpleDistance()),
               GeneralDistanceDistance(; alg = SimpleDistance()),
               GeneralDistance(; alg = SimpleAbsoluteDistance()),
               GeneralDistanceDistance(; alg = SimpleAbsoluteDistance()),
               GeneralDistance(; alg = CorrelationDistance()),
               GeneralDistanceDistance(; alg = CorrelationDistance()),
               GeneralDistance(; alg = LogDistance()),
               GeneralDistanceDistance(; alg = LogDistance()),
               GeneralDistance(; alg = VariationInfoDistance()),
               GeneralDistanceDistance(; alg = VariationInfoDistance())]

        for i ∈ 1:ncol(logo_t)
            sigma1 = copy(sigma)
            PortfolioOptimisers.matrix_processing_algorithm!(PortfolioOptimisers.LoGo(;
                                                                                      dist = des[i],
                                                                                      sim = ExponentialSimilarity()),
                                                             PosDefEstimator(), sigma1, X)
            MN = size(sigma1)
            res1 = isapprox(sigma1, reshape(logo_t[!, i], MN))
            if !res1
                println("Fails on LoGo General distance ExpDist sim iteration $i")
                find_tol(sigma1, reshape(logo_t[!, i], MN); name1 = :sigma, name2 = :logo_t)
            end
            @test res1
        end

        @test isnothing(PortfolioOptimisers.logo!(nothing))
    end
end
