include(joinpath(@__DIR__, "test16_setup.jl"))

@testset "MIP Risk Budgeting" begin
    r = StandardDeviation()
    r = factory(r, pr, mip_slv)

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv)
    rb = RiskBudgeting(; r = r, opt = opt,
                       rba = AssetRiskBudgeting(; alg = MixedIntegerRiskBudgeting()))
    res = optimise(rb, rd)
    rkc = risk_contribution(r, res.w, pr.X)
    v1 = minimum(rkc)
    v2 = maximum(rkc)
    success = isapprox(v2 / v1, 1; rtol = 5e-4)
    if !success
        find_tol(v2 / v1, 1)
    end
    @test success
    @test isapprox(res.w,
                   [0.033127220559750765, 0.022769806963934144, 0.03980006959829777,
                    0.03244615241501114, 0.05173109281388823, 0.03970743751722664,
                    0.041819441368669594, 0.08221586980652329, 0.042607653818805415,
                    0.06254237111913212, 0.053763355951943745, 0.08026745642402928,
                    0.03515784043192167, 0.06306076653811903, 0.05566770018572234,
                    0.06352751462163037, 0.029346131719786624, 0.05375126385057467,
                    0.06560682427264843, 0.05108403002238485], rtol = 1e-4)

    rb = RiskBudgeting(; r = r, opt = opt,
                       rba = AssetRiskBudgeting(; alg = MixedIntegerRiskBudgeting(),
                                                rkb = RiskBudget(; val = 1:20)))
    res = optimise(rb, rd)
    rkc = risk_contribution(r, res.w, pr.X)
    v1, m1 = findmin(rkc)
    v2, m2 = findmax(rkc)
    success = isapprox(v2 / v1, 20; rtol = 1e-4)
    if !success
        find_tol(v2 / v1, 20)
    end
    @test success
    @test m1 == 1
    @test m2 == 20
    @test isapprox(res.w,
                   [0.003291670890764884, 0.0048310396898406495, 0.012062367227361308,
                    0.013372410078120452, 0.02313267450027435, 0.02417219040979293,
                    0.028449321750264458, 0.05828809852572409, 0.0382423448854881,
                    0.05607985937522164, 0.0523858066047695, 0.08241709539390478,
                    0.04482404375427151, 0.07812672108647016, 0.07296166947749294,
                    0.08910512294473034, 0.04281741172335355, 0.0840529450637892,
                    0.10412048295611062, 0.08726672366225435], rtol = 5e-5)

    rd2 = PortfolioOptimisers.returns_result_view(rd, 1:2:20)
    pr2 = prior(EmpiricalPrior(), rd2)
    r = factory(StandardDeviation(), pr2, mip_slv)

    opt = JuMPOptimiser(; pe = pr2, slv = mip_slv, sbgt = nothing, bgt = nothing,
                        wb = nothing)
    rb = RiskBudgeting(; r = r, opt = opt,
                       rba = AssetRiskBudgeting(; alg = MixedIntegerRiskBudgeting()))
    res = optimise(rb, rd2)
    rkc = risk_contribution(r, res.w, pr2.X)
    v1 = minimum(rkc)
    v2 = maximum(rkc)
    @test isapprox(v2 / v1, 1; rtol = 5e-4)
    @test isapprox(res.w,
                   [-11.9670204785192, -20.27855656466867, -9.031638570761155,
                    -12.404198792206339, 23.38560846510856, -11.411103386313773,
                    15.532421847999915, 11.074304614043934, 5.235656319779921,
                    10.963933763939947], rtol = 5e-4)

    rb = RiskBudgeting(; r = r, opt = opt,
                       rba = AssetRiskBudgeting(;
                                                alg = LogRiskBudgeting(z = [-1, -1, -1, -1,
                                                                            1, -1, 1, 1, 1,
                                                                            1])))
    res1 = optimise(rb, rd2)
    idx = isone.(rb.rba.alg.z)
    @test all(res1.w[idx] .>= 0)
    @test all(res1.w[.!idx] .< 0)
    v1 = minimum(rkc)
    v2 = maximum(rkc)
    @test isapprox(v2 / v1, 1; rtol = 5e-4)
    @test isapprox(res.w/sum(res.w), res1.w/sum(res1.w), rtol = 1e-3)

    rb = RiskBudgeting(; r = r, opt = opt,
                       rba = AssetRiskBudgeting(;
                                                alg = LogRiskBudgeting(z = [-1, 1, -1, 1,
                                                                            -1, 1, -1, 1,
                                                                            -1, 1])))
    res1 = optimise(rb, rd2)
    idx = isone.(rb.rba.alg.z)
    @test all(res1.w[idx] .>= 0)
    @test all(res1.w[.!idx] .< 0)
    v1 = minimum(rkc)
    v2 = maximum(rkc)
    @test isapprox(v2 / v1, 1; rtol = 5e-4)

    rb = RiskBudgeting(; r = r, opt = opt,
                       rba = AssetRiskBudgeting(; alg = MixedIntegerRiskBudgeting(),
                                                rkb = RiskBudget(; val = 1:10)))
    res = optimise(rb, rd2)
    rkc = risk_contribution(r, res.w, pr2.X)
    v1, m1 = findmin(rkc)
    v2, m2 = findmax(rkc)
    @test isapprox(v2 / v1, 10; rtol = 5e-4)
    @test m1 == 1
    @test m2 == 10
    @test isapprox(res.w,
                   [-2.4295346851425466, -4.782006626613654, -2.4181860407479356,
                    -3.5475446827000723, 5.25869258317521, -3.557616841873538,
                    3.737346772646009, 3.54749774747547, 1.626855992937381,
                    3.665078207675533], rtol = 5e-4)

    rb = RiskBudgeting(; r = r, opt = opt,
                       rba = AssetRiskBudgeting(; rkb = RiskBudget(; val = 1:10),
                                                alg = LogRiskBudgeting(z = [-1, -1, -1, -1,
                                                                            1, -1, 1, 1, 1,
                                                                            1])))
    res1 = optimise(rb, rd2)
    @test isapprox(res.w/sum(res.w), res1.w/sum(res1.w), rtol = 5e-4)

    opt = JuMPOptimiser(; pe = pr2, slv = mip_slv, sbgt = nothing, bgt = nothing,
                        wb = nothing,
                        lt = Threshold(; val = [1000, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                        st = Threshold(; val = [0, 0, 0, 0, 1000, 0, 0, 0, 0, 0]))
    rb = RiskBudgeting(; r = r, opt = opt,
                       rba = AssetRiskBudgeting(; alg = MixedIntegerRiskBudgeting()))
    res = optimise(rb, rd2)
    rkc = risk_contribution(r, res.w, pr2.X)
    v1 = minimum(rkc)
    v2 = maximum(rkc)
    @test isapprox(v2 / v1, 1; rtol = 5e-4)
    @test isapprox(res.w,
                   [0.005049243860991792, 0.00855665106568725, 0.003810322142750114,
                    0.005233835537719221, -0.009867552804947562, 0.0048147567001987045,
                    -0.006553717442072589, -0.004672617364069549, -0.002209020574956733,
                    -0.0046259448171769154], rtol = 5e-4)

    rb = RiskBudgeting(; r = r, opt = opt,
                       rba = AssetRiskBudgeting(; alg = MixedIntegerRiskBudgeting(),
                                                rkb = RiskBudget(; val = 1:10)))
    res = optimise(rb, rd2)
    rkc = risk_contribution(r, res.w, pr2.X)
    v1, m1 = findmin(rkc)
    v2, m2 = findmax(rkc)
    @test isapprox(v2 / v1, 10; rtol = 5e-4)
    @test m1 == 1
    @test m2 == 10
    @test isapprox(res.w,
                   [0.0038783685946485852, 0.0076341311932820125, 0.0038604291393048358,
                    0.005663308711158661, -0.0083951759687286, 0.0056794244876683265,
                    -0.005966226762046662, -0.005663164898664754, -0.0025971591991454266,
                    -0.005850802364782971], rtol = 5e-4)
end
