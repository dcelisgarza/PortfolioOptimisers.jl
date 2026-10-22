@testset "show_fields docstring census: the Returns section names the tuple the method returns (#1800)" begin
    using Test

    #=
    Issue #1800. The `# Returns` section of a `show_fields` docstring names the field names
    that the method returns. Two docstrings left `:x_src` out of the tuple after the method
    gained it, and nothing compared the two, so the omission passed every gate.

    For each docstring of a `show_fields` method, the census reads the `# Returns` section
    and finds the rule it states:

      - one backticked tuple of symbols, `(:a, :b)`: the method returns that tuple;
      - two tuples, as in "`(…)` with no policy and `(…)` with one": the method returns the
        first for an instance that leaves the optional field at `nothing`, and the second
        for an instance that sets it;
      - "every field name but `:cache`", or "except `cache`": the method returns every field
        name of the type but `:cache`, in declaration order.

    The census builds each instance with the keyword constructor of the type. `KW` holds the
    keywords a constructor needs, and `ALT` holds the one that sets the optional field of a
    two-tuple docstring. The default for `ALT` is a coverage policy `cvg`. A docstring of a
    `Union` signature is checked for each member of the union.

    A docstring that states its rule in prose and states no tuple in a `# Returns` section is
    named in `PROSE`, with the reason. A new docstring that states no rule the census reads is a red, so the census
    cannot pass by skipping it.
    =#

    PO = PortfolioOptimisers
    fv = BlackLittermanViews(; P = [1.0 0 0], Q = [0.01])
    KW = Dict{Symbol, NamedTuple}(:EWBeta => (; decay = 0.5, min_obs = 2),
                                  :EWMean => (; decay = 0.5, min_obs = 2),
                                  :EWVolumeRatio =>
                                      (; num = nothing, den = "adj_volume", decay = 0.5,
                                       min_obs = 2), :RollingLogReturn => (; window = 2),
                                  :ImpliedVolatility => (; ppy = 252),
                                  :BayesianBlackLittermanPrior => (; views = fv),
                                  :FactorBlackLittermanPrior => (; views = fv),
                                  :AugmentedBlackLittermanPrior =>
                                      (; a_views = fv, f_views = fv),
                                  :CrossSectionalFactorPrior =>
                                      (; factors = ["market" => ConstantExposure()]),
                                  :ChangeInIntensity =>
                                      (; field = "a", scale = "b", lag = 1),
                                  :ChangeToScale => (; field = "a", scale = "b", lag = 1),
                                  :GrowthRate => (; field = "a", lag = 1),
                                  :OpinionPoolingPrior => (; pes = [EntropyPoolingPrior()]),
                                  :JuMPOptimiser => (; slv = Solver(; solver = nothing)),
                                  :NestedClustered => (; opti = InverseVolatility(),
                                                       opto = InverseVolatility()),
                                  :Stacking => (; opti = [InverseVolatility()],
                                                opto = InverseVolatility()),
                                  :SubsetResampling => (; opt = InverseVolatility()))
    ALT = Dict{Symbol, NamedTuple}(:EmpiricalPrior => (; max_scenarios = 80))
    # The docstrings that state their rule in prose, by the bare name of a type in the
    # signature.
    PROSE = Dict{Symbol, String}(:Any => "the fallback, which returns every declared field",
                                 :Pipeline => "the `cache` is shown only when it holds a state",
                                 :Fees => "the `# Algorithm` section states the two tuples",
                                 :PricesToReturns => "the `cache` is shown only when it holds a state")

    md = Base.Docs.meta(PO)[Base.Docs.Binding(PO, :show_fields)]
    tuple_re = r"`(\((?::\w+,\s*)*:\w+,?\))`"
    nocache_re = r"[Ee]very field(?: name)? (?:but|except) `:?cache`"
    n_checked = 0
    prose_seen = Set{Symbol}()
    for (sig, ds) in md.docs
        T = sig.parameters[1]
        Us = T === Any ? Any[Any] : Base.uniontypes(T)
        names = [T === Any ? :Any : nameof(Base.typename(U).wrapper) for U in Us]
        txt = join(filter(x -> x isa AbstractString, collect(ds.text)))
        m = match(r"\n# Returns\n(.*?)(?:\n# |\z)"s, txt)
        ret = isnothing(m) ? "" : m.captures[1]
        tups = [Tuple(Symbol.(getindex.(eachmatch(r":(\w+)", t.captures[1]), 1)))
                for t in eachmatch(tuple_re, ret)]
        nocache = occursin(nocache_re, ret)
        if isempty(tups) && !nocache
            # No rule the census reads: the docstring must be a named prose exemption.
            @test any(n -> haskey(PROSE, n), names)
            union!(prose_seen, filter(n -> haskey(PROSE, n), names))
            continue
        end
        @test length(tups) <= 2
        @test !(nocache && !isempty(tups))
        for U in Us
            W = Base.typename(U).wrapper
            n = nameof(W)
            kw = get(KW, n, (;))
            x = W(; kw...)
            got = Tuple(PO.show_fields(x))
            want = if nocache
                filter(!=(:cache), fieldnames(typeof(x)))
            else
                tups[1]
            end
            @test (n, got) == (n, want)
            if length(tups) == 2
                y = W(; kw..., get(ALT, n, (; cvg = CoveragePolicy()))...)
                @test (n, Tuple(PO.show_fields(y))) == (n, tups[2])
            end
            n_checked += 1
        end
    end
    # 27 docstrings state one or two tuples, and eight state the rule without `:cache`. Their
    # signatures name 55 types. The floor keeps the census from passing on an empty read.
    @test n_checked >= 55
    # An exemption whose docstring now states a tuple, or is gone, leaves the list.
    @test Set(keys(PROSE)) == prose_seen
end
