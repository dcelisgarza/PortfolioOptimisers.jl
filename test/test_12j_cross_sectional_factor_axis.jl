#=
The two pre-fit cross-sectional factor-axis verbs (issue #724, map #643).

Issue #651 decision 5 settled that the factor names are derived rather than carried, and that a
caller reads them before any fit: one verb answers the axis and its Factor Family labels from the
`factors` Pairs and the Asset Panel's field index, and a second widens a `UniverseSets` with the
axis under `cfkey` and one plain group per family. The block stores the same answer in `nf` and
`fam`, so there is one source of truth.
=#
using Statistics, Distributions, Dates, Random
include(joinpath(@__DIR__, "test06c_setup.jl"))

@testset "The cross-sectional factor axis before any fit" begin
    PO = PortfolioOptimisers
    res = synthetic_asset_panel(; n_assets = 10, n_observations = 30, n_industries = 3,
                                rng = StableRNG(724_101))
    rd = res.rd
    factors = ["market" => ConstantExposure(),
               "size" =>
                   CompositeExposure(; descriptors = [Passthrough(; field = "market_cap")],
                                     bw = "market_cap"),
               "industry" => OneHotExposure(; field = "industry", family = "industry")]

    @testset "A one-hot member expands to one name per level" begin
        ax = PO.cross_sectional_factor_axis(factors, rd)
        @test ax.nf == ["market", "size", "industry=Real Estate", "industry=Software",
                        "industry=Banks"]
        @test ax.fam == ["market", "style", "industry", "industry", "industry"]
        # The axis is the Panel Field's level order, so it is the same in every fold.
        @test ax.nf[3:end] == PO.one_hot_exposure_names(last(factors[3]), rd)
        @test length(ax.nf) == length(ax.fam)
    end

    @testset "Every other member takes the name the caller wrote" begin
        one = PO.cross_sectional_factor_axis(["mkt" => ConstantExposure(; family = "m")],
                                             rd)
        @test one.nf == ["mkt"]
        @test one.fam == ["m"]
        two = PO.cross_sectional_factor_axis(["beta" => DerivedExposure(; source = "size",
                                                                        f = abs)], rd)
        @test two.nf == ["beta"]
        @test two.fam == ["style"]
    end

    @testset "The axis refuses an empty list and a repeated name" begin
        @test_throws PortfolioOptimisers.IsEmptyError PO.cross_sectional_factor_axis(Pair[],
                                                                                     rd)
        @test_throws ArgumentError PO.cross_sectional_factor_axis(["a" =>
                                                                       ConstantExposure(),
                                                                   "a" =>
                                                                       ConstantExposure()],
                                                                  rd)
    end

    @testset "The sets verb declares the axis and one group per family" begin
        sets = PO.cross_sectional_factor_sets(factors, rd)
        ax = PO.cross_sectional_factor_axis(factors, rd)
        @test sets.dict[sets.cfkey] == ax.nf
        @test sets.dict["industry"] ==
              ["industry=Real Estate", "industry=Software", "industry=Banks"]
        @test sets.dict["style"] == ["size"]
        @test sets.dict[sets.xkey] == rd.nx
        # A consumer never names the key by hand: the loadings result it holds picks it.
        csfm = CrossSectionalFactorModel(; M = ones(10, 5), b = zeros(10), nf = ax.nf,
                                         fam = ax.fam)
        @test PO.factor_axis_key(sets, csfm) == sets.cfkey
        @test sets.dict[PO.factor_axis_key(sets, csfm)] == csfm.nf
    end

    @testset "A family label that names a factor is allowed only when it is that factor" begin
        # `market` is both the name of the single factor and the label of its family, which
        # is the case the reference implementation permits.
        sets = PO.cross_sectional_factor_sets(factors, rd)
        @test sets.dict["market"] == ["market"]
        # A label shared with a factor of another family would answer two different lists.
        clash = ["industry" => ConstantExposure(; family = "size"),
                 "size" => CompositeExposure(;
                                             descriptors = [Passthrough(; field = "market_cap")],
                                             bw = "market_cap", family = "size")]
        @test_throws ArgumentError PO.cross_sectional_factor_sets(clash, rd)
    end

    @testset "The verb widens a declared universe" begin
        s0 = UniverseSets(; xkey = "assets",
                          dict = Dict{String, Any}("assets" => rd.nx, "nf" => ["F1", "F2"],
                                                   "sector" => ["a"]))
        s1 = PO.cross_sectional_factor_sets(factors, rd, s0)
        # Every axis the caller declared survives, and the key prefixes are the caller's.
        @test s1.xkey == "assets"
        @test s1.dict["assets"] == rd.nx
        @test s1.dict[s1.tfkey] == ["F1", "F2"]
        @test s1.dict["sector"] == ["a"]
        @test s1.dict[s1.cfkey] == PO.cross_sectional_factor_axis(factors, rd).nf
        @test s1.dict["industry"] ==
              ["industry=Real Estate", "industry=Software", "industry=Banks"]
        # The original is untouched, so the widening is a new declaration.
        @test !haskey(s0.dict, s0.cfkey)
    end

    @testset "A new sets needs the asset names" begin
        # The asset axis is the one mandatory axis of a `UniverseSets`, and `ReturnsResult`
        # refuses every carrier that holds data without `nx`, so an empty one is what
        # reaches the refusal.
        bare = ReturnsResult()
        @test isnothing(bare.nx)
        @test_throws ArgumentError PO.cross_sectional_sets_dict(bare, nothing)
        @test_throws ArgumentError PO.cross_sectional_factor_sets(["mkt" =>
                                                                       ConstantExposure()],
                                                                  bare)
    end
end
