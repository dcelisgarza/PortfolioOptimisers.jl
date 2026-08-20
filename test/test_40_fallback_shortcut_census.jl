@testset "Fallback-shortcut census: `Nothing` always lands on `fb`" begin
    using PortfolioOptimisers, Test

    # Sixteen optimisers carry a shortcut method of the form
    #
    #     optimise(o::Opt{<:Any, …, Nothing}, args…; kwargs…) = _optimise(o, args…; kwargs…)
    #
    # which skips the fallback chain in the generic `optimise`
    # (`01_Base_Optimisation.jl`) when the estimator has no fallback. The `Nothing` is
    # positional, so each site re-encodes "`fb` is type parameter k" by hand, and the
    # count is invisible to the reader.
    #
    # `22_DiscreteFiniteAllocation.jl` miscounted: it wrote four parameters for a
    # five-field estimator, so `Nothing` sat on `wf` instead of `fb`. Because
    # `Nothing <: JuMPWeightFinaliserFormulation` is false, the method could never match
    # and the miscount was silent.
    #
    # The dangerous polarity is the other one. Insert a field before `fb` in any of the
    # other fifteen and `Nothing` slides onto that field while `fb` goes free — the
    # shortcut then matches estimators that DO have a fallback and calls `_optimise`
    # directly, silently skipping the fallback chain.
    #
    # This census names no optimiser, so a shortcut added in future is covered the day it
    # is written. It asserts two things about every shortcut it finds:
    #
    #   1. Exactly one type parameter is constrained to `Nothing`.
    #   2. That parameter is the position of the field `fb`.

    # Strip the `UnionAll` wrappers off a method signature and off the first argument.
    function base_type(m::Method)
        sig = m.sig
        while isa(sig, UnionAll)
            sig = sig.body
        end
        T = sig.parameters[2]
        while isa(T, UnionAll)
            T = T.body
        end
        return T
    end

    # A parameter is "pinned to `Nothing`" whether it is spelled `Nothing` or `<:Nothing`
    # (`05_HierarchicalRiskParity.jl` uses the second spelling; the two are equivalent
    # here because `Nothing` is a singleton).
    is_nothing_param(p) = p === Nothing || (isa(p, TypeVar) && p.ub === Nothing)

    shortcuts = Tuple{Method, DataType, Vector{Int}, Union{Int, Nothing}}[]
    for m in methods(PortfolioOptimisers.optimise)
        m.module === PortfolioOptimisers || continue
        T = base_type(m)
        isa(T, DataType) || continue
        isempty(T.parameters) && continue
        idxs = findall(is_nothing_param, collect(T.parameters))
        isempty(idxs) && continue
        fields = fieldnames(T.name.wrapper)
        push!(shortcuts, (m, T, idxs, findfirst(==(:fb), fields)))
    end

    # A shortcut that stops being found is a shape that was deleted, not a passing test.
    @test !isempty(shortcuts)

    for (m, T, idxs, fbpos) in shortcuts
        site = string(basename(string(m.file)), ":", m.line, " (", T.name.name, ")")
        @test (site, fbpos !== nothing) == (site, true)
        @test (site, idxs) == (site, [fbpos])
    end

    # The site that miscounted, pinned by name so the regression cannot come back quietly.
    # `fb` is the fifth field, and an estimator without a fallback reaches the shortcut
    # while an estimator with one still reaches the generic `optimise`.
    @test fieldnames(DiscreteAllocation) == (:slv, :sc, :so, :wf, :fb)
    da_type(fb) = DiscreteAllocation{Any, Float64, Float64,
                                     PortfolioOptimisers.JuMPWeightFinaliserFormulation, fb}
    no_fb = which(PortfolioOptimisers.optimise,
                  Tuple{da_type(Nothing), FiniteAllocationInput})
    with_fb = which(PortfolioOptimisers.optimise,
                    Tuple{da_type(GreedyAllocation), FiniteAllocationInput})
    @test basename(string(no_fb.file)) == "22_DiscreteFiniteAllocation.jl"
    @test basename(string(with_fb.file)) == "01_Base_Optimisation.jl"
end
