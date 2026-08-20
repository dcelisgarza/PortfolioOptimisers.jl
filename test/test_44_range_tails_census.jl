@testset "Range tails census: a range declares its tails, or is on the fused list" begin
    using PortfolioOptimisers, Test, InteractiveUtils

    #=
    ADR 0057: a range risk measure is its base measure applied twice, once on the returns
    and once on their negation. `set_range_risk_constraints!` reads the pair from
    `range_tails`, builds each tail, and registers only their sum.

    The throwing fallback at `src/19_RiskMeasures/01_Base_RiskMeasures.jl:227` enforces
    ONE direction of that rule. A non-fused range that forgets `range_tails` throws the
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
    Each entry decomposes: `range_tails` answers a `(; loss, gain)` pair of point measures,
    and `set_range_risk_constraints!` sums them.
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
        `20_RiskMeasureConstraints/16_RangeConstraints.jl:47`. It never reaches
        `set_range_risk_constraints!`, so there is no pair to state.
      - `ValueatRiskRange` under `DistributionValueatRisk` is parametric: the two tails are
        two quantiles of one fitted distribution, not two separate measures.
      - `OrderedWeightsArrayRange` under `ExactOrderedWeightsArray` writes one exact OWA
        block over the combined weight vector.
    =#
    fused = ["Range" => Range(),
             "ValueatRiskRange/DistributionValueatRisk" =>
                 ValueatRiskRange(; alg = DistributionValueatRisk()),
             "OrderedWeightsArrayRange/ExactOrderedWeightsArray" =>
                 OrderedWeightsArrayRange(; alg = ExactOrderedWeightsArray())]

    # ------------------------------------------------- the population is fully covered

    #=
    A list is only a gate if nothing can be added outside it. Walk the risk-measure tree
    for every concrete type whose name ends in `Range`, and require that the two lists
    name exactly that set. A new range measure fails here until its author lists it.

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

    range_types = Set(nameof(T)
                      for T in concrete_leaves(PortfolioOptimisers.AbstractBaseRiskMeasure)
                      if endswith(string(nameof(T)), "Range"))
    listed = Set(nameof(typeof(r)) for (_, r) in vcat(declaring, fused))
    @test range_types == listed

    # A floor, so a predicate that stopped matching cannot pass vacuously.
    @test length(range_types) >= 9
    @test length(declaring) >= 8
    @test !isempty(fused)

    # ------------------------------------------------------- the declaring half is sound

    for (name, r) in declaring
        tails = PortfolioOptimisers.range_tails(r)

        # The shape the consumer destructures.
        @test isa(tails, NamedTuple)
        @test issetequal(propertynames(tails), (:loss, :gain))

        for (side, tail) in pairs(tails)
            #=
            "The sum of TWO POINT measures". A tail that were itself a range would nest a
            decomposition inside a decomposition, and `nested_index` would be carrying a
            structure the ADR does not describe.
            =#
            @test_throws ArgumentError PortfolioOptimisers.range_tails(tail)

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
            (`21_GenericValueatRiskRange.jl:151`). Its default tails carry no bound, so the
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
    end

    #=
    The fallback is reachable from outside the range family too — it is the answer for
    every point measure, which is what makes the "tails are point measures" check above
    meaningful rather than circular.
    =#
    @test_throws ArgumentError PortfolioOptimisers.range_tails(ConditionalValueatRisk())
    @test_throws ArgumentError PortfolioOptimisers.range_tails(Variance())
end
