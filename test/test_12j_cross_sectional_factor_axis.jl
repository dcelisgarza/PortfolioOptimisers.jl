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

    @testset "The verb refuses to replace a group or an axis the caller declared" begin
        # A Factor Family label is a plain group name, so a caller's own group can share
        # it. Replacing it in silence would re-point every constraint written against it.
        s0 = UniverseSets(; dict = Dict{String, Any}("nx" => rd.nx, "style" => rd.nx[1:2]))
        @test_throws ArgumentError PO.cross_sectional_factor_sets(factors, rd, s0)
        msg = try
            PO.cross_sectional_factor_sets(factors, rd, s0)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("under style", msg)
        @test s0.dict["style"] == rd.nx[1:2]
        # The same holds for the axis itself under `cfkey`.
        ax = PO.cross_sectional_factor_axis(factors, rd).nf
        s2 = UniverseSets(; dict = Dict{String, Any}("nx" => rd.nx, "ncf" => reverse(ax)))
        @test_throws ArgumentError PO.cross_sectional_factor_sets(factors, rd, s2)
        # The same list is not a replacement, so it is accepted.
        s3 = UniverseSets(;
                          dict = Dict{String, Any}("nx" => rd.nx, "ncf" => ax,
                                                   "style" => ["size"]))
        s4 = PO.cross_sectional_factor_sets(factors, rd, s3)
        @test s4.dict["ncf"] == ax
        @test s4.dict["style"] == ["size"]
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

    @testset "Both verbs read the axis off the estimator, and both are exported" begin
        # Issue #1059: a caller who writes a factor mandate in a pipeline step holds the
        # estimator, not its Pairs, so the estimator method forwards `pe.factors`.
        pe = CrossSectionalFactorPrior(; factors = factors)
        # The names resolve unqualified, so the pipeline example can call them.
        @test :cross_sectional_factor_axis in names(PortfolioOptimisers)
        @test :cross_sectional_factor_sets in names(PortfolioOptimisers)
        ax = cross_sectional_factor_axis(pe, rd)
        @test ax == PO.cross_sectional_factor_axis(factors, rd)
        @test ax.nf == ["market", "size", "industry=Real Estate", "industry=Software",
                        "industry=Banks"]
        sets = cross_sectional_factor_sets(pe, rd)
        ref = PO.cross_sectional_factor_sets(factors, rd)
        @test sets.dict == ref.dict
        @test sets.cfkey == ref.cfkey
        @test sets.dict[sets.cfkey] == ax.nf
        # The `sets` argument is forwarded, so the estimator method widens too.
        s0 = UniverseSets(; xkey = "assets",
                          dict = Dict{String, Any}("assets" => rd.nx, "sector" => ["a"]))
        s1 = cross_sectional_factor_sets(pe, rd, s0)
        @test s1.dict == PO.cross_sectional_factor_sets(factors, rd, s0).dict
        @test s1.xkey == "assets"
        @test s1.dict["sector"] == ["a"]
        @test s1.dict[s1.cfkey] == ax.nf
    end

    @testset "The docstrings of 09_CrossSectionalFactorAxis against numbers" begin
        # `exposure_axis_names`: one name per level for a one-hot member, and the caller's
        # name for every other member. The one-hot method ignores the caller's name.
        oh = last(factors[3])
        lv = PO.one_hot_exposure_names(oh, rd)
        @test PO.exposure_axis_names("mkt", ConstantExposure(), rd) == (["mkt"], ["market"])
        @test PO.exposure_axis_names("size", last(factors[2]), rd) == (["size"], ["style"])
        @test PO.exposure_axis_names("ignored", oh, rd) ==
              (lv, fill("industry", length(lv)))
        # `cross_sectional_factor_axis`: one name per member that is not one-hot, and one
        # per level, so 1 + 1 + 3 names here.
        ax = PO.cross_sectional_factor_axis(factors, rd)
        @test length(ax.nf) == 2 + length(lv) == 5
        # The levels are the ones the Panel Field declares, so a fold over fewer
        # observations and fewer assets reads the same axis.
        @test PO.cross_sectional_factor_axis(factors, PO.port_opt_view(rd, 1:10, :)) == ax
        @test PO.cross_sectional_factor_axis(factors, PO.port_opt_view(rd, 11:30, 1:4)) ==
              ax
        # `cross_sectional_factor_sets`: a new sets takes the default key prefixes of
        # `UniverseSets`, and it equals the widening of the bare default universe.
        dflt = UniverseSets(; dict = Dict{String, Any}("nx" => rd.nx))
        ns = PO.cross_sectional_factor_sets(factors, rd)
        for k in (:xkey, :uxkey, :tfkey, :utfkey, :cfkey, :ucfkey, :nikey)
            @test getfield(ns, k) == getfield(dflt, k)
        end
        @test ns.dict == PO.cross_sectional_factor_sets(factors, rd, dflt).dict
        @test sort!(collect(keys(ns.dict))) == ["industry", "market", "ncf", "nx", "style"]
        # A widened universe keeps all seven of its key prefixes.
        own = UniverseSets(; xkey = "assets", uxkey = "uassets", tfkey = "tsf",
                           utfkey = "utsf", cfkey = "csf", ucfkey = "ucsf", nikey = "gone",
                           dict = Dict{String, Any}("assets" => rd.nx))
        wid = PO.cross_sectional_factor_sets(factors, rd, own)
        for k in (:xkey, :uxkey, :tfkey, :utfkey, :cfkey, :ucfkey, :nikey)
            @test getfield(wid, k) == getfield(own, k)
        end
        @test wid.dict["csf"] == ax.nf
        @test !haskey(wid.dict, "ncf")
        # A Factor Family label that starts with a key prefix of the universe, or equals
        # its `nikey`, would be read as an axis, so it is refused. Before the guard,
        # UniverseSets took `"ni"` as the Non-Investable Axis, and a family `"ncfx"` that
        # held every factor as a partition of the cross-sectional axis.
        fam_of(lab) = ["market" => ConstantExposure(),
                       "a" => ConstantExposure(; family = lab)]
        for lab in ("nxt", "uxx", "nfam", "uf", "ncfx", "ucfam", "ni")
            @test_throws ArgumentError PO.cross_sectional_factor_sets(fam_of(lab), rd)
        end
        @test_throws ArgumentError PO.cross_sectional_factor_sets(["a" =>
                                                                       ConstantExposure(;
                                                                                        family = "ncfx"),
                                                                   "b" =>
                                                                       ConstantExposure(;
                                                                                        family = "ncfx")],
                                                                  rd)
        msg = try
            PO.cross_sectional_factor_sets(fam_of("ni"), rd)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("Factor Family ni", msg)
        # A label that only starts with `nikey` is a plain group.
        @test PO.cross_sectional_factor_sets(fam_of("nikkei"), rd).dict["nikkei"] == ["a"]
        # The prefixes are the universe's own: a label the default refuses is a plain group
        # of a universe with other prefixes, and a label that starts with one of those is
        # refused.
        @test PO.cross_sectional_factor_sets(fam_of("ncfx"), rd, own).dict["ncfx"] == ["a"]
        @test_throws ArgumentError PO.cross_sectional_factor_sets(fam_of("csfx"), rd, own)
        @test_throws ArgumentError PO.cross_sectional_factor_sets(fam_of("gone"), rd, own)
        # `cross_sectional_sets_write!`: a new key and the same list are written, and a
        # different list is refused with its key named.
        d = Dict{String, Any}("style" => ["a"])
        @test isnothing(PO.cross_sectional_sets_write!(d, "style", ["a"]))
        @test isnothing(PO.cross_sectional_sets_write!(d, "industry", ["b", "c"]))
        @test d == Dict{String, Any}("style" => ["a"], "industry" => ["b", "c"])
        msg = try
            PO.cross_sectional_sets_write!(d, "style", ["b"])
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("under style", msg)
        @test d["style"] == ["a"]
        # `cross_sectional_sets_dict`: the method for `nothing` declares the asset axis
        # alone under `"nx"`, and the method for a universe copies its dictionary.
        @test PO.cross_sectional_sets_dict(rd, nothing) == Dict{String, Any}("nx" => rd.nx)
        cpd = PO.cross_sectional_sets_dict(rd, own)
        @test cpd == own.dict
        @test cpd !== own.dict
    end
end
