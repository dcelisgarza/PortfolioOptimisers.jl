#=
The partial-fit state interface: every state pays it, and a view reaches every cache.

ADR 0106 and ADR 0107 state an interface for every `AbstractPartialFitState`:

  - `merge_states(a::S, b::S)`: the fold, or a refusal naming the reason.
  - `Base.copy(x::S)`: a fresh state aliasing no array; `partial_fit` rests on it.
  - `port_opt_view(x::S, i, args...)`: an exact slice by index copy, for every state a view of
    its estimator can reach. A state with no exact slice refuses.

And they state the host side: the `cache` field of every propagatable estimator of the seam
carries `@fprop @vprop`, so that `port_opt_view` slices the state rather than carrying it. A
host that forgets the tag carries a state fitted on the whole universe into a view of a few
assets, which is the silent defect ADR 0107 removed, and the universal `port_opt_view`
fallback is what makes a state with no method of its own equally silent.

Issue #1054 found that nothing measured any of it. Each family's test file drives its own
methods, and `test_48` drives the root refusal on a probe type, but no walk of the family
asked whether the seventeenth state pays what the first sixteen paid by hand. This census is
that walk, in the closed polarity of ADR 0037: the rule names no state and no host, so a
state or a host added later is measured the day it is written, and every exemption names its
reason and is asserted to still hold.

The host side reads dispatch and lowered code rather than the tags, because the tag is one
mechanism and not the rule. Five hosts write `port_opt_view` by hand -- the three
meta-optimisers, `JuMPOptimiser` and `Pipeline` -- and the rule is the same for them: the
cache passes through `port_opt_view`, or the view refuses. `cache_travels` reads the method's
lowered body for a `port_opt_view` call whose argument is the host's `cache` field, which is
what the `@vprop` tag emits and what a hand-written method spells.

`test_60_partial_fit_cache_narrowing_census.jl` is the sibling: it measures the `cache`
narrowing of every `partial_fit!` method, which is how a fold reaches its own state. This
file measures what the state answers once it is reached.
=#
using Test, PortfolioOptimisers, StatsBase, InteractiveUtils

const po = PortfolioOptimisers

# ------------------------------------------------------------------- the two populations

# Every concrete subtype below `T`, through the abstract ones.
function concrete_leaves(T::Type)
    out = Type[]
    for S in subtypes(T)
        isabstracttype(S) ? append!(out, concrete_leaves(S)) : push!(out, S)
    end
    return out
end

# Every partial-fit state the library declares. `subtypes` sees every module in the process,
# so the filter keeps a probe type another test file declares out of this walk: `test_48`
# declares three that deliberately pay nothing.
const STATES = filter(S -> parentmodule(S) === po, subtypes(po.AbstractPartialFitState))

# Every estimator the library declares that carries a `cache` field. `StatsBase`'s surface
# joins the walk because the covariance estimators subtype it rather than the library's root.
const HOSTS = filter(
                     unique(vcat(concrete_leaves(po.AbstractEstimator),
                                 concrete_leaves(StatsBase.CovarianceEstimator)))) do T
    B = Base.unwrap_unionall(T)
    return parentmodule(T) === po &&
           hasfield(B, :cache) &&
           !(B <: po.AbstractPartialFitState)
end

# ----------------------------------------------------------------------------- the probes

# The method `f` dispatches to for `Tuple{S, rest...}`, or `nothing` when none applies. A
# `@concrete` state is a `UnionAll`, and `which` on one is exact here because every state
# method is declared on the bare name; a method that narrowed a type parameter would read as
# the fallback, which `test_08l` explains.
function dispatched(f, S::Type, rest::Type...)
    return try
        which(f, Tuple{S, rest...})
    catch
        nothing
    end
end

# The type the dispatched method declares its first argument at.
first_parameter(m::Method) = Base.unwrap_unionall(m.sig).parameters[2]

# `true` when `S` owns the method: it is the library's, and declared on `S` rather than on a
# supertype or on `Any`.
function owns(f, S::Type, rest::Type...)
    m = dispatched(f, S, rest...)
    if isnothing(m)
        return false
    end
    P = first_parameter(m)
    return m.module === po && Base.unwrap_unionall(P) === Base.unwrap_unionall(S)
end

# The `port_opt_view` method a view of a host `T` runs, or `nothing` when the host takes the
# identity a surface declares -- the case where no channel reaches its `cache` at all.
function view_method(T::Type)
    m = dispatched(po.port_opt_view, T, Vector{Int}, Matrix{Float64})
    if isnothing(m)
        return nothing
    end
    P = Base.unwrap_unionall(first_parameter(m))
    return (m.module === po && P.name === Base.unwrap_unionall(T).name) ? m : nothing
end

# What a host's `port_opt_view` does with its `cache`, read off the method's lowered body:
#
#   - `:slices`  -- some `port_opt_view` call takes the host's `cache` field as an argument.
#                   `@vprop` emits `port_opt_view(x.cache, i, args...)`, which lowers to an
#                   `_apply_iterate` over a tuple holding `getproperty(x, :cache)`; a
#                   hand-written method spells `port_opt_view(opt.cache, i)`, a plain call.
#   - `:refuses` -- the body throws and never reads the cache.
#   - `:carries` -- neither: the cache passes through unchanged, which is the defect.
#
# A method with keyword arguments is a thin sorter whose body lives in the keyword body
# function, so that is the one read. `Base.uncompressed_ir` and `Base.bodyfunction` are the
# reflection every lowered-code reader in `Base` itself uses.
function cache_travels(m::Method)
    if !isempty(Base.kwarg_decl(m))
        m = only(methods(Base.bodyfunction(m)))
    end
    code = Base.uncompressed_ir(m).code
    stmt(x) = x isa Core.SSAValue ? code[x.id] : x
    is_call(x) = x isa Expr && x.head === :call
    # `port_opt_view` as a `GlobalRef`, or as `getproperty(PortfolioOptimisers, :port_opt_view)`.
    function names_view(x)
        x = stmt(x)
        return (x isa GlobalRef && x.name === :port_opt_view) || (is_call(x) &&
                                                                  length(x.args) == 3 &&
                                                                  x.args[3] === QuoteNode(:port_opt_view))
    end
    # `getproperty(x, :cache)`, or a tuple that holds one.
    function from_cache(x)
        x = stmt(x)
        if !(is_call(x))
            return false
        end
        f = stmt(x.args[1])
        return (length(x.args) == 3 && x.args[3] === QuoteNode(:cache)) ||
               (f isa GlobalRef && f.name === :tuple && any(from_cache, x.args[2:end]))
    end
    throws = false
    for st in code
        if st isa Expr && st.head === :(=)
            st = st.args[2]
        end
        if !(is_call(st))
            continue
        end
        f = stmt(st.args[1])
        throws |= f isa GlobalRef && f.name === :throw
        if names_view(f) && any(from_cache, st.args[2:end])
            return :slices
        end
        if f isa GlobalRef &&
           f.name === :_apply_iterate &&
           names_view(st.args[3]) &&
           any(from_cache, st.args[4:end])
            return :slices
        end
    end
    return throws ? :refuses : :carries
end

# ------------------------------------------------------------------------ the exemptions

#=
Hosts whose `port_opt_view` is the identity a surface declares, so no view reaches their
`cache`. Each is a `@concrete` struct rather than a `@propagatable` one, and each holds its
own family state, which a view therefore never meets. The day one of them becomes
`@propagatable` this list reds, and its state owes `port_opt_view` -- the exact slice for the
exponentially weighted moments and the preprocessing steps, and the `nothing` refusal ADR 0107
names for the two regime families, whose regime state reads the standardised innovation of
every asset in the universe and so has no exact slice.
=#
const NO_VIEW_HOSTS = (ExpWeightedExpectedReturns, ExpWeightedVariance,
                       ExpWeightedCovariance, RegimeAdjustedExpWeightedVariance,
                       RegimeAdjustedExpWeightedCovariance, PricesToReturns, PriceGapFill,
                       MissingDataFilter)

# Hosts whose `port_opt_view` refuses outright, so a cache never travels a view of them. A
# `Pipeline`'s asset universe is fitted state, which ADR 0028 says a view cannot select.
const VIEW_REFUSING_HOSTS = (Pipeline,)

#=
States no view reaches, each paired with the host that holds it. A state is here because its
host is on `NO_VIEW_HOSTS`, and for no other reason: the pairing is what the census asserts,
so a state whose host gains a view channel is found by the host row and by this one. A state
that owns `port_opt_view` while it is listed has paid the exemption, and the row is deleted.
=#
const UNVIEWED_STATES = (po.ExpWeightedExpectedReturnsState => ExpWeightedExpectedReturns,
                         po.ExpWeightedVarianceState => ExpWeightedVariance,
                         po.ExpWeightedCovarianceState => ExpWeightedCovariance,
                         po.RegimeAdjustedVarianceState =>
                             RegimeAdjustedExpWeightedVariance,
                         po.RegimeAdjustedCovarianceState =>
                             RegimeAdjustedExpWeightedCovariance,
                         po.PricesToReturnsState => PricesToReturns,
                         po.PriceGapFillState => PriceGapFill,
                         po.MissingDataFilterState => MissingDataFilter)

# ------------------------------------------------------------------------------ the census

@testset "Partial-fit state census: every state pays the interface" begin
    # A walk that answered nothing would make every check below vacuously green.
    @test !isempty(STATES)
    @test !isempty(HOSTS)

    # ------------------------------------------- 1. merge_states and copy, below the root
    @testset "$(nameof(S)) answers merge_states and copy" for S in STATES
        # The root method is the refusal that names the method a family still owes, so a
        # state whose pair reaches it has not paid. The method is declared on `S` itself.
        @test owns(po.merge_states, S, S)
        @test owns(Base.copy, S)
    end

    # ---------------------------------------------- 2. a view of every host reaches its cache
    @testset "a view of $(nameof(T)) reaches its cache" for T in HOSTS
        m = view_method(T)
        if isnothing(m)
            # No channel reaches the cache, and the host says so on the list.
            @test T in NO_VIEW_HOSTS
        elseif T in VIEW_REFUSING_HOSTS
            @test cache_travels(m) === :refuses
        else
            # `:carries` is the defect: a state fitted on the whole universe travels into a
            # view of a few assets. Tag the field `@fprop @vprop`, or slice it by hand.
            @test cache_travels(m) === :slices
        end
    end

    # ---------------------------------- 3. every state a view can reach owns port_opt_view
    unviewed = Dict(UNVIEWED_STATES)
    @testset "$(nameof(S)) answers port_opt_view or is unreachable" for S in STATES
        if haskey(unviewed, S)
            host = unviewed[S]
            # The reason still holds: the host carries the state, and takes the identity.
            @test hasfield(Base.unwrap_unionall(host), :cache)
            @test host in NO_VIEW_HOSTS
            @test isnothing(view_method(host))
            # And the exemption is unpaid; a paid one is deleted, not kept.
            @test !owns(po.port_opt_view, S, Vector{Int}, Matrix{Float64})
        else
            @test owns(po.port_opt_view, S, Vector{Int}, Matrix{Float64})
        end
    end

    # ------------------------------------------------- 4. the exemptions name what exists
    # A row that names nothing the walk found is stale, and a host on the identity list that
    # has gained a view channel has paid its exemption.
    @testset "the exemption lists name what the walk found" begin
        for T in NO_VIEW_HOSTS
            @test T in HOSTS
            @test isnothing(view_method(T))
        end
        for T in VIEW_REFUSING_HOSTS
            @test T in HOSTS
        end
        for (S, host) in UNVIEWED_STATES
            @test S in STATES
            @test host in NO_VIEW_HOSTS
        end
    end
end
