@testset "Range tails census: a range declares its tails, or is on the fused list" begin
    using PortfolioOptimisers, Test, InteractiveUtils

    #=
    ADR 0057: a range risk measure is its base measure applied twice, once on the returns
    and once on their negation. The method of `set_risk_constraints!` for a
    `RangeRiskMeasure` reads the pair from `range_tails`, builds each tail, and registers
    only their sum under `range_risk_key(r)`.

    The throwing fallback of `range_tails` on `RangeRiskMeasure` enforces ONE direction of
    that rule. A non-fused range that forgets `range_tails` throws the
    first time it is used. It cannot enforce the other direction: a FUSED range that
    wrongly declared `range_tails` would be decomposed and double-counted, and nothing
    would object. Nor can it see the settings a declared tail carries — a tail that kept
    `rke = true` would reach the objective on its own, beside the composite.

    So the invariant had no test at all: before this file, `grep -rl range_tails test/`
    returned nothing. This census closes both gaps. Every range measure must appear on one
    of the two lists below, so adding one is a deliberate edit to this file: its author
    picks a side and says which.

    The split is per FORMULATION, not per type. `ValueatRiskRange` decomposes under
    `MIPValueatRisk` and fuses under `DistributionValueatRisk`; `OrderedWeightsArrayRange`
    decomposes under `ApproxOrderedWeightsArray` and fuses under
    `ExactOrderedWeightsArray`. A census keyed on the type alone would be blind to exactly
    the case that is easiest to get wrong, so both lists hold instances.
    =#

    # ------------------------------------------------------------------ the two lists

    #=
    Each entry decomposes: `range_tails` answers a `RangeTails` pair of point measures, and
    the method of `set_risk_constraints!` for a `RangeRiskMeasure` sums them under the key
    that `range_risk_key` answers.
    =#
    declaring = ["ConditionalValueatRiskRange" => ConditionalValueatRiskRange(),
                 "DistributionallyRobustConditionalValueatRiskRange" =>
                     DistributionallyRobustConditionalValueatRiskRange(),
                 "EntropicValueatRiskRange" => EntropicValueatRiskRange(),
                 "GenericValueatRiskRange" => GenericValueatRiskRange(),
                 "PowerNormValueatRiskRange" => PowerNormValueatRiskRange(),
                 "RelativisticValueatRiskRange" => RelativisticValueatRiskRange(),
                 "ValueatRiskRange/MIPValueatRisk" =>
                     ValueatRiskRange(; alg = MIPValueatRisk()),
                 "OrderedWeightsArrayRange/ApproxOrderedWeightsArray" =>
                     OrderedWeightsArrayRange(; alg = ApproxOrderedWeightsArray())]

    #=
    Each entry fuses its two tails into one formulation and must therefore declare NO
    `range_tails`. The reason is written per entry, because "it throws" is not a reason.

      - `Range` writes `wr_risk - br_risk` directly in
        `09_RiskMeasureConstraints/14_RangeConstraints.jl`. Its own method of
        `set_risk_constraints!` is more specific, so there is no pair to state.
      - `ValueatRiskRange` under `DistributionValueatRisk` is parametric: the two tails are
        two quantiles of one fitted distribution, not two separate measures.
      - `OrderedWeightsArrayRange` under `ExactOrderedWeightsArray` writes one exact OWA
        block over the combined weight vector.
      - `OrderedWeightsArrayRange` under `SortingNetworkOrderedWeightsArray` writes one sorting
        network over the combined weight vector.
    =#
    fused = ["Range" => Range(),
             "ValueatRiskRange/DistributionValueatRisk" =>
                 ValueatRiskRange(; alg = DistributionValueatRisk()),
             "OrderedWeightsArrayRange/ExactOrderedWeightsArray" =>
                 OrderedWeightsArrayRange(; alg = ExactOrderedWeightsArray()),
             "OrderedWeightsArrayRange/SortingNetworkOrderedWeightsArray" =>
                 OrderedWeightsArrayRange(; alg = SortingNetworkOrderedWeightsArray())]

    # ------------------------------------------------- the population is fully covered

    #=
    A list is only a gate if nothing can be added outside it. Walk `RangeRiskMeasure` for
    every concrete range, and require that the two lists name exactly that set. A new range
    measure fails here until its author lists it. A type whose name ends in `Range` and that
    does not subtype `RangeRiskMeasure` fails too, because the build of its tails would
    never reach it.

    `parentmodule(S) === PortfolioOptimisers` matters: test files share a worker, so
    `subtypes` otherwise finds a measure another file declared.
    =#
    function concrete_leaves(T, acc = Set{Type}())
        for S in subtypes(T)
            parentmodule(S) === PortfolioOptimisers || continue
            isabstracttype(S) ? concrete_leaves(S, acc) : push!(acc, S)
        end
        return acc
    end

    RangeRiskMeasure = PortfolioOptimisers.RangeRiskMeasure
    range_types = Set(nameof(T) for T in concrete_leaves(RangeRiskMeasure))
    named_ranges = Set(nameof(T)
                       for T in
                           concrete_leaves(PortfolioOptimisers.AbstractBaseRiskMeasure)
                       if endswith(string(nameof(T)), "Range"))
    listed = Set(nameof(typeof(r)) for (_, r) in vcat(declaring, fused))
    @test range_types == listed
    @test named_ranges == range_types
    @test all(r -> isa(r, RangeRiskMeasure), last.(vcat(declaring, fused)))

    # A floor, so a predicate that stopped matching cannot pass vacuously.
    @test length(range_types) >= 9
    @test length(declaring) >= 8
    @test !isempty(fused)

    # ------------------------------------------------------- the declaring half is sound

    #=
    The key of the sum follows `<base>_range_risk_`, and no two formulations share one, so a
    reader of the model finds each range by the name of its base.
    =#
    keys_of = [PortfolioOptimisers.range_risk_key(r) for (_, r) in declaring]
    @test all(k -> isa(k, Symbol) && endswith(string(k), "_range_risk_"), keys_of)
    @test allunique(keys_of)

    for (name, r) in declaring
        tails = PortfolioOptimisers.range_tails(r)

        # The shape the consumer destructures: the public type of the contract (#2113).
        @test isa(tails, PortfolioOptimisers.RangeTails)
        @test issetequal(propertynames(tails), (:loss, :gain))

        for tail in (tails.loss, tails.gain)
            #=
            "The sum of TWO POINT measures". A tail that were itself a range would nest a
            decomposition inside a decomposition, and `nested_index` would be carrying a
            structure the ADR does not describe.
            =#
            @test_throws MethodError PortfolioOptimisers.range_tails(tail)

            #=
            `rke = false` is what keeps the tail out of the objective. Only the composite
            contributes a risk expression; a tail that kept `rke = true` would be counted
            twice, once on its own and once inside the sum. This is the direction the
            throwing fallback structurally cannot see, and it is the whole reason this
            census exists.
            =#
            @test tail.settings.rke === false

            #=
            The upper bound is the same argument for the bound rather than the objective,
            with ONE documented exception. `GenericValueatRiskRange` is the range whose
            tails are GIVEN rather than derived: the caller states both measures, and the
            constructor keeps their `ub` on purpose, "so a caller can bound one tail"
            (the comment on `range_tails` in `18_GenericValueatRiskRange.jl`). Its default
            tails carry no bound, so the
            assertion below still bites on the shipped value; what it must not do is forbid
            a bound the design admits.
            =#
            if !isa(r, GenericValueatRiskRange)
                @test isnothing(tail.settings.ub)
            end
        end

        #=
        "Its base measure applied twice" — so the two tails are the SAME measure, and the
        type name is what says so. They differ in a type parameter, never in the measure:
        `OrderedWeightsArrayRange`'s gain tail composes `reverse` into the OWA function,
        which changes the parameter and not the name.

        `GenericValueatRiskRange` is exempt because its tails are given rather than
        derived: a caller may pair a `ValueatRisk` loss with a `ConditionalValueatRisk`
        gain, and the type is what admits it (`loss::ValueatRiskRMs`).
        =#
        if !isa(r, GenericValueatRiskRange)
            @test nameof(typeof(tails.loss)) == nameof(typeof(tails.gain))
        end
    end

    #=
    The two tails are read at the two levels the range carries, and in that order: the loss
    tail at `alpha`, the gain tail at `beta`. A range that passed `alpha` to both would
    measure one level twice and would still satisfy every check above, because the shape
    and the settings would be right and only the number would be wrong.

    Asserted at asymmetric levels, so `alpha == beta` cannot hide a swap. The six are the
    declaring ranges that carry a level pair; `GenericValueatRiskRange` states its tails
    outright and `OrderedWeightsArrayRange` carries weight vectors instead.
    =#
    for r in (ConditionalValueatRiskRange(; alpha = 0.03, beta = 0.07),
              DistributionallyRobustConditionalValueatRiskRange(; alpha = 0.03, beta = 0.07),
              EntropicValueatRiskRange(; alpha = 0.03, beta = 0.07),
              PowerNormValueatRiskRange(; alpha = 0.03, beta = 0.07),
              RelativisticValueatRiskRange(; alpha = 0.03, beta = 0.07),
              ValueatRiskRange(; alpha = 0.03, beta = 0.07, alg = MIPValueatRisk()))
        tails = PortfolioOptimisers.range_tails(r)
        @test tails.loss.alpha == r.alpha
        @test tails.gain.alpha == r.beta
    end

    #=
    `GenericValueatRiskRange` keeps a caller's `ub` and still forces `rke = false`, through
    `no_risk_expr_risk_measure` in its inner constructor. Both halves are asserted here
    rather than left to the loop above, because the exemption is the interesting case.
    =#
    let bounded = ConditionalValueatRisk(;
                                         settings = RiskMeasureSettings(; rke = true,
                                                                        ub = 0.5))
        g = GenericValueatRiskRange(; loss = bounded, gain = bounded)
        tails = PortfolioOptimisers.range_tails(g)
        @test tails.loss.settings.rke === false      # stripped, always
        @test tails.gain.settings.rke === false
        @test tails.loss.settings.ub == 0.5          # kept, by decision
        @test tails.gain.settings.ub == 0.5
    end

    # ----------------------------------------------------------- the fused half is silent

    for (name, r) in fused
        #=
        The fallback's message names the type and states the rule, so a reader who hits it
        learns which side of the split they are on. Assert both the throw and the naming:
        a bare `ArgumentError` would pass the first check while telling the reader nothing.
        =#
        @test_throws ArgumentError PortfolioOptimisers.range_tails(r)
        err = try
            PortfolioOptimisers.range_tails(r)
            nothing
        catch e
            e
        end
        @test occursin(string(nameof(typeof(r))), sprint(showerror, err))
        @test occursin("range_tails", sprint(showerror, err))
        # A fused range states its key in its own builder, so it has no key method.
        @test !hasmethod(PortfolioOptimisers.range_risk_key, Tuple{typeof(r)})
    end

    #=
    A point measure is not a range, so `range_tails` has no method for it. That is what
    makes the "tails are point measures" check above meaningful rather than circular.
    =#
    @test_throws MethodError PortfolioOptimisers.range_tails(ConditionalValueatRisk())
    @test_throws MethodError PortfolioOptimisers.range_tails(Variance())

    #=
    `RangeTails` is the contract that a range of your own builds (#2113). Its constructor
    holds the invariant of the census above, so a tail that would add to the risk
    expression fails where it is made. A tail of either side is enough to refuse.
    =#
    let off = RiskMeasureSettings(; rke = false),
        RangeTails = PortfolioOptimisers.RangeTails

        tails = RangeTails(; loss = ConditionalValueatRisk(; settings = off, alpha = 0.05),
                           gain = ConditionalValueatRisk(; settings = off, alpha = 0.1))
        # The builder of a `RangeRiskMeasure` destructures it by name.
        (; loss, gain) = tails
        @test (loss.alpha, gain.alpha) == (0.05, 0.1)
        @test RangeTails(loss, gain).gain === gain
        # The gain tail measures the negated portfolio. A measure that cannot be built on that
        # side is refused there, and it stays a valid loss tail (#2328).
        for bad in (MaximumDrawdown(; settings = off), UlcerIndex(; settings = off),
                    ConditionalDrawdownatRisk(; settings = off), LowOrderMoment(; settings = off),
                    LowOrderMoment(; settings = off, alg = MeanAbsoluteDeviation(), mu = 0.0),
                    LowOrderMoment(; settings = off, alg = SecondMoment(; alg1 = SemiMoment())),
                    LowOrderMoment(; settings = off, alg = EvenMoment(; alg = SemiMoment())),
                    NegativeSkewness(; settings = off),
                    TurnoverRiskMeasure(; settings = off, w = [0.5, 0.5]))
            @test !PortfolioOptimisers.supports_gain_tail(bad)
            @test_throws ArgumentError RangeTails(loss, bad)
            @test RangeTails(bad, gain).loss === bad
        end
        # The refusal names the type and the declaration that a measure of a user's own makes.
        @test_throws "supports_gain_tail(::MaximumDrawdown) = true" RangeTails(loss,
                                                                               MaximumDrawdown(;
                                                                                               settings = off))
        # A full moment about the mean has one value on both sides, so it is a gain tail.
        for alg in (MeanAbsoluteDeviation(), SecondMoment(), EvenMoment()),
            mu in (nothing, [0.1, 0.2])

            @test RangeTails(loss, LowOrderMoment(; settings = off, alg = alg, mu = mu)).gain.mu ===
                  mu
        end
        # A range answers for its loss tail, because that tail moves to the gain side with it.
        for (name, r) in vcat(declaring, fused)
            @test PortfolioOptimisers.supports_gain_tail(r)
        end
        @test_throws ArgumentError RangeTails(ConditionalValueatRisk(), gain)
        @test_throws ArgumentError RangeTails(loss, ConditionalValueatRisk())
        # The typed keyword constructor is the door that refuses another type. `@concrete`
        # appends a generic positional constructor, so the positional route does not.
        @test_throws TypeError RangeTails(; loss = loss, gain = nothing)
        # A public part of a configuration shows as the library shows its estimators: the
        # wrapper name and one line for each field, not the long parametric header.
        txt = sprint(show, MIME"text/plain"(), tails)
        # A name carries its module when `Main` of the process does not load the package.
        @test occursin(r"^(PortfolioOptimisers\.)?RangeTails\n", txt)
        @test occursin(r"loss ┼ (PortfolioOptimisers\.)?ConditionalValueatRisk\n", txt)
        @test occursin(r"gain ┼ (PortfolioOptimisers\.)?ConditionalValueatRisk\n", txt)
        @test !occursin('{', first(split(txt, '\n')))
    end
end
