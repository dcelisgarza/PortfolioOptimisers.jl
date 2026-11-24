@testset "Fallback-shortcut census: no `optimise` method goes around the driver" begin
    using PortfolioOptimisers, Test

    # An estimator with no fallback holds `fb = nothing`. The driver `optimise` in
    # `01_Base_Optimisation/05_InvestableUniverse.jl` returns the first answer before it builds
    # the fallback chain, so for `fb::Nothing` the compiler folds the test of the fallback, and
    # the driver infers the type of the hook `_optimise` alone (#2101, #2103).
    #
    # Many optimisers once carried a shortcut method of the form
    #
    #     optimise(o::Opt{<:Any, …, Nothing}, args…; kwargs…) = _optimise(o, args…; kwargs…)
    #
    # that went around the driver. The `Nothing` was positional, so each site re-encoded
    # "`fb` is type parameter k" by hand. `02_DiscreteFiniteAllocation.jl` miscounted: its
    # `Nothing` sat on `wf`, and the method could never match. A field inserted before `fb`
    # gives the dangerous polarity: `Nothing` slides onto that field, and the shortcut matches
    # an estimator that DOES have a fallback and skips its chain in silence.
    #
    # #2101 and #2103 measured each shortcut against the driver and removed all of them. A
    # method of the verb that goes around the driver is a wrapper, by
    # `.github/instructions/julia-source-code.instructions.md` § *A wrapper is for convenience
    # or for a dispatch problem*, item 4. So this census asserts that no `optimise` method of
    # the package pins a type parameter of its estimator to `Nothing`, whatever its position.
    # It names no optimiser, so a shortcut added in future fails the day it is written.

    # Strip the `UnionAll` wrappers off a method signature and off the first argument. A first
    # argument spelled `::E` with `where {E <: Opt{…, Nothing}}` is a `TypeVar`, so its upper
    # bound is the type that the method pins.
    function base_type(m::Method)
        sig = m.sig
        while isa(sig, UnionAll)
            sig = sig.body
        end
        T = sig.parameters[2]
        while isa(T, UnionAll) || isa(T, TypeVar)
            T = isa(T, TypeVar) ? T.ub : T.body
        end
        return T
    end

    # A parameter is "pinned to `Nothing`" whether it is spelled `Nothing` or `<:Nothing`. The
    # two are equivalent here because `Nothing` is a singleton.
    is_nothing_param(p) = p === Nothing || (isa(p, TypeVar) && p.ub === Nothing)

    # Each method of `ms` whose first argument pins a type parameter to `Nothing`, with the
    # positions of the pins and the position of the field `fb`.
    function nothing_pins(ms)
        out = Tuple{Method, DataType, Vector{Int}, Union{Int, Nothing}}[]
        for m in ms
            T = base_type(m)
            isa(T, DataType) || continue
            isempty(T.parameters) && continue
            idxs = findall(is_nothing_param, collect(T.parameters))
            isempty(idxs) && continue
            push!(out, (m, T, idxs, findfirst(==(:fb), fieldnames(T.name.wrapper))))
        end
        return out
    end

    # The detector must find a shortcut, or the empty census below passes in silence. The
    # probe is a local function with the exact shape of the removed shortcuts, and a generic
    # method beside it.
    fb_slot(fb) = DiscreteAllocation{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                     fb}
    probe_optimise(::fb_slot(Nothing), ::Any) = :shortcut
    probe_optimise(::DiscreteAllocation, ::Any) = :driver
    probe = nothing_pins(methods(probe_optimise))
    @test length(probe) == 1
    @test probe[1][3] == [probe[1][4]] == [9]
    # The same shape with the estimator type in a `where` clause.
    probe_where(::E, ::Any) where {E <: fb_slot(Nothing)} = :shortcut
    probe_where(::DiscreteAllocation, ::Any) = :driver
    probe = nothing_pins(methods(probe_where))
    @test length(probe) == 1
    @test probe[1][3] == [probe[1][4]] == [9]

    # The methods of the package and of its extensions, whose root module is the extension.
    # The list must hold the driver, or a filter that keeps nothing passes in silence.
    of_package(m) = startswith(string(nameof(Base.moduleroot(m.module))),
                               "PortfolioOptimisers")
    pkg_methods = filter(of_package, collect(methods(PortfolioOptimisers.optimise)))
    driver = which(PortfolioOptimisers.optimise,
                   Tuple{PortfolioOptimisers.OptimisationEstimator})
    @test driver in pkg_methods
    shortcuts = nothing_pins(pkg_methods)
    sites = [string(basename(string(m.file)), ":", m.line, " (", T.name.name, ", pins ",
                    idxs, ", fb at ", fbpos, ")") for (m, T, idxs, fbpos) in shortcuts]
    @test isempty(sites)

    # The site that miscounted, pinned by name. `fb` is the ninth field, and an estimator
    # without a fallback reaches the same driver as an estimator with one.
    @test fieldnames(DiscreteAllocation) ==
          (:slv, :sc, :so, :wf, :unit, :sets, :idx, :strict, :fb)
    da_type(fb) = DiscreteAllocation{Any, Float64, Float64,
                                     PortfolioOptimisers.JuMPWeightFinaliserFormulation,
                                     LotUnit{Int}, Nothing, Nothing, Bool, fb}
    no_fb = which(PortfolioOptimisers.optimise,
                  Tuple{da_type(Nothing), FiniteAllocationInput})
    with_fb = which(PortfolioOptimisers.optimise,
                    Tuple{da_type(GreedyAllocation), FiniteAllocationInput})
    @test no_fb === with_fb
    @test basename(string(no_fb.file)) == "05_InvestableUniverse.jl"
end
