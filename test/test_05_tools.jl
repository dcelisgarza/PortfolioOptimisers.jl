@testset "Tools" begin
    using PortfolioOptimisers, Statistics, StatsBase, Test, LinearAlgebra
    A = [0, 1, 1, 2, 3, 4, 5, 8]
    B = [2, 3, 3, 5, 8, 10, 15, 21]
    C = 13
    W = fweights([3, 5, 9, 8, 12, 3, 5, 9])
    @test PortfolioOptimisers.:⊙(C, A) == C * A
    @test PortfolioOptimisers.:⊘(A, C) == A / C
    @test PortfolioOptimisers.:⊘(C, C) == C / C
    @test PortfolioOptimisers.:⊕(A, B) == A + B
    @test PortfolioOptimisers.:⊕(C, B) == C .+ B
    @test PortfolioOptimisers.dot_scalar(C, A) == C * sum(A)
    @test PortfolioOptimisers.dot_scalar(A, C) == C * sum(A)
    @test PortfolioOptimisers.vec_to_real_measure(StdValue(), A) == std(A)
    @test PortfolioOptimisers.vec_to_real_measure(VarValue(), A) == var(A)
    @test PortfolioOptimisers.vec_to_real_measure(SumValue(), A) == sum(A)
    @test PortfolioOptimisers.vec_to_real_measure(ProdValue(), A) == prod(A)
    @test PortfolioOptimisers.vec_to_real_measure(ModeValue(), A) == mode(A)
    @test PortfolioOptimisers.vec_to_real_measure(MedianValue(), A) == median(A)
    @test PortfolioOptimisers.vec_to_real_measure(StandardisedValue(), A) ==
          mean(A) / std(A)
    # A single value has no corrected standard deviation. The denominator is one, so the
    # reduction gives the mean itself instead of a `NaN`.
    @test PortfolioOptimisers.vec_to_real_measure(StandardisedValue(), [0.37]) == 0.37
    @test PortfolioOptimisers.vec_to_real_measure(StandardisedValue(), (0.37,)) == 0.37
    # An exact-zero standard deviation keeps its own guard.
    @test PortfolioOptimisers.vec_to_real_measure(StandardisedValue(), [2.0, 2.0, 2.0]) ==
          2 / sqrt(eps(Float64))
    msv = StandardisedValue()
    @test factory(msv) === msv
    msv = factory(StandardisedValue(), W)
    @test msv.mv.w === W
    @test msv.sv.w === W
    vv = factory(VarValue(), W)
    @test vv.w === W
    vv = factory(VarValue(), W)
    @test vv.w === W
    mdv = factory(MedianValue(), W)
    @test mdv.w === W
end
@testset "@propagatable contract" begin
    using PortfolioOptimisers, Test
    PO = PortfolioOptimisers
    # The shipped guarantee: every `@propagatable` declaration satisfies the contract behind
    # the generated `factory`/`port_opt_view` methods. This is what precompilation asserts.
    @test !isempty(PO.PROPAGATABLE_CONTRACTS)
    @test isnothing(PO.check_propagatable_contracts())
    # A type declared outside this package must join the factory propagation chain (ADR 0002,
    # decision 4 and its 2026-08-17 amendment). Every name the expansion emits is escaped into
    # the caller, so a bare one resolves in the *declaring* module. This probe imports the
    # macro and nothing else, which is the least a caller can write, and it catches a bare name
    # of either kind: a private one throws an `UndefVarError`, and an exported one is worse --
    # `using` binds it implicitly, so the method definition declares a function of the caller's
    # own and the declaration then compiles and registers while carrying no method at all.
    n_registered = length(PO.PROPAGATABLE_CONTRACTS)
    foreign = Module(:PropagatableForeignProbe)
    # The import lands in its own `eval`: a block is macro-expanded as a unit, so a
    # `@propagatable` written beside its own `using` is expanded before that `using` runs.
    Base.eval(foreign, :(using PortfolioOptimisers))
    Base.eval(foreign, :(using PortfolioOptimisers: @propagatable))
    Base.eval(foreign, quote
                  @propagatable struct ForeignProbe{T1}
                      inner::T1
                  end
                  ForeignProbe(; inner = nothing) = ForeignProbe(inner)
              end)
    @test length(PO.PROPAGATABLE_CONTRACTS) == n_registered + 1
    @test PO.PROPAGATABLE_CONTRACTS[end] == (foreign.ForeignProbe, ())
    # `factory` in the declaring module is still this package's, not a new one of the caller's.
    # This is the assertion a bare exported name fails, and it fails silently: the two lines
    # above pass either way, because the declaration compiles and registers regardless.
    @test getglobal(foreign, :factory) === PO.factory
    @test parentmodule(getglobal(foreign, :factory)) === PO
    # The generated method is owned by the declaring module and lives on this package's
    # function, so the new type dispatches through the shipped chain.
    @test count(m -> m.module === foreign, methods(PO.factory)) == 1
    foreign_probe = foreign.ForeignProbe(; inner = 7)
    @test PO.factory(foreign_probe) === foreign_probe
    # The declaration was throwaway, so drop it again and leave the checked registry as found.
    pop!(PO.PROPAGATABLE_CONTRACTS)
    @test length(PO.PROPAGATABLE_CONTRACTS) == n_registered
    pool = PO.prior_result_property_pool()
    # `HighOrderPrior` forwards the whole of its `pr`, so the low-order names are in the
    # pool without being its fields.
    @test issubset((:mu, :sigma, :w, :kt, :sk), pool)
    @test issubset(fieldnames(PO.LowOrderPrior), pool)
    @test issubset(fieldnames(PO.HighOrderPrior), pool)
    # Declared in a throwaway module so the checked registry stays clean.
    m = Module(:PropagatableContractProbe)
    Base.eval(m, quote
                  struct KwMismatch{T1, T2}
                      inner::T1
                      typo::T2
                  end
                  KwMismatch(; inner = nothing, typoo = 1) = KwMismatch(inner, typoo)
                  struct Slurped{T1}
                      inner::T1
                  end
                  Slurped(; kwargs...) = Slurped(get(kwargs, :inner, nothing))
                  struct Named{T1}
                      inner::T1
                  end
                  Named(; inner = nothing) = Named(inner)
              end)
    # A field the outer keyword constructor does not name, with its near-miss suggestion.
    kwmsgs = PO.propagatable_contract_violations(m.KwMismatch, (), pool)
    @test length(kwmsgs) == 1
    @test occursin("field `typo` is not a keyword", kwmsgs[1])
    @test occursin("did you mean `typoo`?", kwmsgs[1])
    # A `kwargs...` slurp accepts the keyword and discards it, so it must not satisfy the
    # contract.
    @test PO.propagatable_keywords(m.Slurped) == Symbol[]
    @test length(PO.propagatable_contract_violations(m.Slurped, (), pool)) == 1
    # A named keyword does satisfy it.
    @test PO.propagatable_keywords(m.Named) == [:inner]
    @test isempty(PO.propagatable_contract_violations(m.Named, (), pool))
    # An `@pprop` field absent from every prior result carrier.
    prmsgs = PO.propagatable_contract_violations(m.Named, (:sgima,), pool)
    @test length(prmsgs) == 1
    @test occursin("`@pprop` field `sgima` is not a property of a prior result", prmsgs[1])
    @test occursin("did you mean `sigma`?", prmsgs[1])
    # Every violation is reported, not just the first.
    allmsgs = PO.propagatable_contract_violations(m.KwMismatch, (:sgima, :muu), pool)
    @test length(allmsgs) == 3
end
@testset "propagation tag table" begin
    using PortfolioOptimisers, Test
    PO = PortfolioOptimisers
    # The shipped guarantee: every row of the tag table has a stub macro, a field transform
    # and a channel, so a tag cannot parse and then never propagate (ADR 0061).
    @test isnothing(PO.check_prop_tag_macros())
    @test PO.PROP_TAG_MACRO_NAMES == map(tag -> Symbol("@", tag), PO.PROP_TAG_NAMES)
    # A tag is looked up in the table, in both spellings a `:macrocall` head takes: a bare
    # `Symbol` in a hand-written struct body, and a `GlobalRef` once another macro expanded
    # around it.
    for tag in PO.PROP_TAG_NAMES
        @test PO.prop_tag(Symbol("@", tag)) === tag
        @test PO.prop_tag(GlobalRef(PO, Symbol("@", tag))) === tag
        @test PO.is_prop_tag_call(Expr(:macrocall, Symbol("@", tag), nothing, :a))
    end
    # A macro outside the table is not a tag. It used to fall through to `@wprop`, which
    # substituted observation weights into the field with no diagnostic.
    @test isnothing(PO.prop_tag(Symbol("@sixth")))
    @test isnothing(PO.prop_tag(:a))
    @test isnothing(PO.prop_tag(1))
    @test !PO.is_prop_tag_call(Expr(:macrocall, Symbol("@sixth"), nothing, :a))
    sixth = Expr(:macrocall, Symbol("@sixth"), nothing, :a)
    @test PO.peel_prop_tags(sixth) == (Set{Symbol}(), sixth)
    # Stacked tags peel in either order, and the whole stack is recorded.
    stacked = Expr(:macrocall, Symbol("@pprop"), nothing,
                   Expr(:macrocall, Symbol("@wprop"), nothing, :(w::T1)))
    @test PO.peel_prop_tags(stacked) == (Set([:pprop, :wprop]), :(w::T1))
    # The parser answers one entry per table row, and every declared field.
    tagged, all_fields, new_body = PO.propagatable_parse_body(quote
                                                                  @fprop @vprop a
                                                                  @pprop @wprop w
                                                                  @cprop slv
                                                                  plain
                                                              end)
    @test issetequal(keys(tagged), PO.PROP_TAG_NAMES)
    @test tagged[:fprop] == [:a]
    @test tagged[:vprop] == [:a]
    @test tagged[:pprop] == [:w]
    @test tagged[:wprop] == [:w]
    @test tagged[:cprop] == [:slv]
    @test all_fields == [:a, :w, :slv, :plain]
    @test !any(PO.is_prop_tag_call, new_body.args)
    # The gate decides whether a channel emits a method at all.
    @test PO.prop_channel_active(:factory, tagged)
    @test PO.prop_channel_active(:view, tagged)
    @test PO.prop_channel_active(:prior, tagged)
    untagged = Dict{Symbol, Vector{Symbol}}(tag => Symbol[] for tag in PO.PROP_TAG_NAMES)
    @test !PO.prop_channel_active(:factory, untagged)
    @test !PO.prop_channel_active(:view, untagged)
    @test !PO.prop_channel_active(:prior, untagged)
    @test PO.prop_channel_active(:obs, tagged)
    @test !PO.prop_channel_active(:obs, untagged)
    # `@wprop` alone opens the factory and observation channels, and neither of the others.
    wonly = merge(untagged, Dict(:wprop => [:w]))
    @test PO.prop_channel_active(:factory, wonly)
    @test PO.prop_channel_active(:obs, wonly)
    @test !PO.prop_channel_active(:view, wonly)
    @test !PO.prop_channel_active(:prior, wonly)
    # `@fprop` is in the observation channel's precedence but not its gate, so a struct with
    # composed children and no weights of its own emits no `obs_weights_view` method and
    # falls through to the identity.
    fonly = merge(untagged, Dict(:fprop => [:a]))
    @test PO.prop_channel_active(:factory, fonly)
    @test !PO.prop_channel_active(:obs, fonly)
    # The precedence is data, and the prior channel puts `@pprop` above `@fprop` (ADR 0012).
    pairs = PO.prop_channel_pairs(:prior, tagged, all_fields, :xr, PO, (:pr,))
    @test [p.args[1] for p in pairs] == all_fields
    @test pairs[2].args[2] == :($(PO).sel(xr.w, getproperty(pr, :w)))
    @test pairs[3].args[2] == :($(PO).sel(xr.slv, $(PO)._ctx(args...)))
    @test pairs[4].args[2] == :(xr.plain)
    both = merge(untagged, Dict(:pprop => [:a], :fprop => [:a]))
    prior_pair = only(PO.prop_channel_pairs(:prior, both, [:a], :xr, PO, (:pr,)))
    @test prior_pair.args[2] == :($(PO).sel(xr.a, getproperty(pr, :a)))
    # The factory channel does not know `@pprop`, so the same field recurses there instead.
    factory_pair = only(PO.prop_channel_pairs(:factory, both, [:a], :x, PO, ()))
    @test factory_pair.args[2] == :($(PO).factory_child(x.a, args...; kwargs...))
    #=
    A tag means what its CHANNEL says it means. `@wprop` names the same field in the factory
    channel and in the observation channel, and the two transforms differ: `factory`
    REPLACES the field with an incoming `ObsWeights`, while `obs_weights_view` INDEXES the
    value already there. This is why `prop_tag_expr` takes the channel as well as the tag,
    and it is what lets a weights field join the observation-axis view with no second tag.
    =#
    obs_pairs = PO.prop_channel_pairs(:obs, tagged, all_fields, :x, PO, (:i,))
    @test [p.args[1] for p in obs_pairs] == all_fields
    # `@fprop` recurses, `@wprop` indexes, everything else is carried through.
    @test obs_pairs[1].args[2] == :($(PO).obs_weights_view(x.a, i))
    @test obs_pairs[2].args[2] == :($(PO).nothing_scalar_array_getindex(x.w, i))
    @test obs_pairs[4].args[2] == :(x.plain)
    # The same field, the same tag, a different channel, a different transform.
    factory_w = PO.prop_channel_pairs(:factory, wonly, [:w], :x, PO, ())[1].args[2]
    obs_w = PO.prop_channel_pairs(:obs, wonly, [:w], :x, PO, (:i,))[1].args[2]
    @test factory_w == :($(PO)._wprop(x.w, args...; kwargs...))
    @test obs_w == :($(PO).nothing_scalar_array_getindex(x.w, i))
    @test factory_w != obs_w
    # Indexing, not viewing: a `view` of an `AbstractWeights` is a `SubArray`, which the
    # weighted `Statistics.std` methods do not accept.
    aw = StatsBase.AnalyticWeights(collect(1.0:10.0))
    @test isa(PO.nothing_scalar_array_getindex(aw, 3:5), StatsBase.AnalyticWeights)
    # A table row with no field transform errors rather than taking another tag's transform.
    @test_throws ErrorException PO.prop_tag_expr(:factory, :sixth, :a, :(x.a), PO, ())
    # `@vprop` has no transform in the observation channel, and asking for one errors rather
    # than silently taking the view channel's.
    @test_throws ErrorException PO.prop_tag_expr(:obs, :vprop, :a, :(x.a), PO, (:i,))
    # Every row is complete in every channel that names it.
    @test isnothing(PO.check_prop_tag_macros())
end
@testset "VectorToScalarMeasure: each weighted branch carries its own denominator" begin
    using PortfolioOptimisers, Test, Statistics, StatsBase
    PO = PortfolioOptimisers
    #=
    Condition 2 of the sweep of `src/02_Tools.jl`, issue #441. Every number below is computed
    from the closed form the docstring states and compared with what the code returns. The
    weighted branch of every measure was uncovered before this testset, so the whole weighted
    half of the family shipped with no number pinned to it.

    The vector `[1.0, 2.0, 3.0, 4.0]` with the weights `[0.1, 0.2, 0.3, 0.4]` is the case the
    `MedianValue` docstring works, so the same pair drives every measure here.
    =#
    v = [1.0, 2.0, 3.0, 4.0]
    t = (1.0, 2.0, 3.0, 4.0)
    wv = [0.1, 0.2, 0.3, 0.4]
    # --- MeanValue: the weighted mean divides by the weight total, not by the length. ---
    wm = PO.vec_to_real_measure(MeanValue(; w = pweights(wv)), v)
    @test wm == sum(wv .* v) / sum(wv)
    @test wm == 3.0
    # The unweighted branch is the plain mean, so the weights move the answer by 0.5.
    @test PO.vec_to_real_measure(MeanValue(), v) == 2.5
    # The tuple method collects first, so it answers with the same number.
    @test PO.vec_to_real_measure(MeanValue(; w = pweights(wv)), t) == wm
    #=
    --- MedianValue: the weighted case INTERPOLATES. ---

    `Statistics.median(val, w)` is the weighted 0.5-quantile, not an order statistic, so the
    answer need not be an entry of the input. A test that assumes an order statistic passes on
    a symmetric input and fails on a real one, which is why the input here is asymmetric.
    =#
    wmed = PO.vec_to_real_measure(MedianValue(; w = weights(wv)), v)
    @test wmed == median(v, weights(wv))
    @test isapprox(wmed, 17 / 6; atol = 1e-15)
    @test wmed ∉ v
    # The order statistic the naive reading expects: the first value whose cumulative weight
    # reaches one half. It is 3.0, and it is wrong by 1/6.
    @test v[findfirst(>=(0.5), cumsum(wv))] == 3.0
    @test !isapprox(wmed, 3.0; atol = 1e-3)
    @test PO.vec_to_real_measure(MedianValue(), v) == 2.5
    @test PO.vec_to_real_measure(MedianValue(; w = weights(wv)), t) == wmed
    #=
    --- StdValue and VarValue: `corrected` AND the weights TYPE pick the denominator. ---

    The uncorrected weighted variance is the weighted sum of the squared deviations from the
    weighted mean, divided by the weight total. Each corrected weights type then divides that
    same numerator by a different denominator, so the four answers below are four numbers.
    =#
    ss = sum(wv .* (v .- wm) .^ 2)
    @test ss == 1.0
    @test PO.vec_to_real_measure(VarValue(; w = weights(wv), corrected = false), v) == ss
    @test PO.vec_to_real_measure(StdValue(; w = weights(wv), corrected = false), v) ==
          sqrt(ss)
    # `AnalyticWeights`: the reliability correction, 1 - sum of the squared normalised weights.
    @test PO.vec_to_real_measure(VarValue(; w = aweights(wv), corrected = true), v) ==
          ss / (1 - sum(abs2, wv))
    @test PO.vec_to_real_measure(VarValue(; w = aweights(wv), corrected = true), v) ==
          10 / 7
    # `FrequencyWeights`: the counts are repetitions, so the denominator is their sum less one.
    fw = [1.0, 2.0, 3.0, 4.0]
    @test PO.vec_to_real_measure(VarValue(; w = fweights([1, 2, 3, 4]), corrected = true),
                                 v) == sum(fw .* (v .- wm) .^ 2) / (sum(fw) - 1)
    @test PO.vec_to_real_measure(VarValue(; w = fweights([1, 2, 3, 4]), corrected = true),
                                 v) == 10 / 9
    # `ProbabilityWeights`: n / ((n - 1) * weight total), with n the count of non-zero weights.
    @test PO.vec_to_real_measure(VarValue(; w = pweights(wv), corrected = true), v) ==
          ss * 4 / (3 * sum(wv))
    @test PO.vec_to_real_measure(VarValue(; w = pweights(wv), corrected = true), v) == 4 / 3
    # The four denominators are four different numbers, so the weights type is half the choice.
    @test length(unique([10 / 7, 10 / 9, 4 / 3, ss])) == 4
    # The tuple methods of both measures collect first and answer with the same numbers.
    @test PO.vec_to_real_measure(VarValue(; w = aweights(wv), corrected = true), t) ==
          10 / 7
    @test PO.vec_to_real_measure(StdValue(; w = aweights(wv), corrected = true), t) ==
          sqrt(10 / 7)
    #=
    Issue #444, settled by ADR 0087. A plain `Weights` with the default `corrected = true`
    raises at the reduction, not at the constructor, and `factory` reaches the same state with
    no `StdValue` in the caller's hand: it replaces the `w` of the default `sv = StdValue()`
    and leaves `corrected` at `true`. This is the behaviour that ships. The library writes one
    bias-correction default rather than deferring to the callee's, because upstream carries no
    single default to defer to.
    =#
    @test_throws ArgumentError PO.vec_to_real_measure(StdValue(; w = weights(wv)), v)
    @test_throws ArgumentError PO.vec_to_real_measure(VarValue(; w = weights(wv)), v)
    @test_throws ArgumentError PO.vec_to_real_measure(factory(StandardisedValue(),
                                                              weights(wv)), v)
    # The same call with a weights type that does support the correction answers a number.
    @test PO.vec_to_real_measure(factory(StandardisedValue(), aweights(wv)), v) ==
          wm / sqrt(10 / 7)
    #=
    ADR 0087, the consistency the written default buys. The variance path and the covariance
    path answer the same number over the same data, on both branches, and they refuse a plain
    `Weights` together. `GeneralCovariance` reaches that agreement only because it writes
    `StatsBase.SimpleCovariance(; corrected = true)`: that estimator's own default is `false`,
    and under it the unweighted covariance would leave the unweighted variance.
    =#
    Xm = hcat(v, reverse(v))
    aw = aweights(wv)
    @test isapprox(var(SimpleVariance(), Xm)[1], cov(GeneralCovariance(), Xm)[1, 1])
    @test isapprox(var(factory(SimpleVariance(), aw), Xm)[1],
                   cov(factory(GeneralCovariance(), aw), Xm)[1, 1])
    @test_throws ArgumentError cov(factory(GeneralCovariance(), weights(wv)), Xm)
    @test_throws ArgumentError var(factory(SimpleVariance(), weights(wv)), Xm)
    #=
    The comparison holds only when the mean carries the same weights. `factory` propagates the
    incoming weights into `me` as well as into `w`, and the covariance always centres on the
    weighted mean. A `SimpleVariance` built by hand keeps whatever `me` the caller gave it, so
    `SimpleVariance(; w = aw)` weights the deviations about the UNWEIGHTED mean and leaves the
    covariance path. That is the `me` field doing its job, not a bias-correction difference.
    =#
    @test isapprox(var(SimpleVariance(; me = SimpleExpectedReturns(; w = aw), w = aw), Xm)[1],
                   cov(GeneralCovariance(; w = aw), Xm)[1, 1])
    @test !isapprox(var(SimpleVariance(; w = aw), Xm)[1],
                    cov(GeneralCovariance(; w = aw), Xm)[1, 1])
    #=
    The estimator's own default is the one the library declines to inherit. Under it the
    unweighted covariance leaves the unweighted variance, which is why the written `true` in
    `GeneralCovariance` is load-bearing rather than redundant.
    =#
    @test StatsBase.SimpleCovariance().corrected == false
    @test !isapprox(cov(GeneralCovariance(; ce = StatsBase.SimpleCovariance()), Xm)[1, 1],
                    var(SimpleVariance(), Xm)[1])
    # --- The `Function` branch reduces with the function it is given. ---
    @test PO.vec_to_real_measure(sum, v) == 10.0
    @test PO.vec_to_real_measure(sum, t) == 10.0
    @test PO.vec_to_real_measure(maximum, v) == 4.0
end
@testset "The view and index seam selects the axis its verb names" begin
    using PortfolioOptimisers, Test, StatsBase, LinearAlgebra
    PO = PortfolioOptimisers
    #=
    Condition 2 of the sweep of `src/02_Tools.jl`, issue #441. The seam has two verbs of two
    shapes each: a VIEW verb and a GETINDEX verb, each with a square form and an odd-order
    form. The square form reads the SAME index on both axes, because its subject is a
    per-asset square matrix; the odd-order form reads two different indices, because its
    subject is an `N x N^k` co-moment matrix. Reading one for the other transposes the
    selection in silence, so both are pinned here against hand-written answers.
    =#
    S = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    vs = VecScalar([1.0, 2.0, 3.0], 4.2)
    # --- the passthrough Union: a value that carries no asset axis is returned itself ---
    for x in (nothing, 3.0, :a => 1, [:a => 1, :b => 2], Dict(:a => 1), UniformValues(),
              MeanValue(), MinValue())
        @test PO.nothing_scalar_array_view(x, 1:2) === x
    end
    #=
    The tenth member of that Union is `StatsBase.CovarianceEstimator`, and it is the reason
    the Union is longer than the getindex verb's. `SimpleVariance` is a
    `StatsBase.CovarianceEstimator` and it is NOT an `AbstractEstimator`, so it reaches the
    view verb through that member alone. The docstring of `nothing_scalar_array_view` omitted
    the member until #441 measured it.
    =#
    sve = SimpleVariance()
    @test sve isa StatsBase.CovarianceEstimator
    @test !(sve isa PO.AbstractEstimator)
    @test PO.nothing_scalar_array_view(sve, 1:2) === sve
    # The getindex verb's passthrough list is shorter, so the same value has no method there.
    @test_throws MethodError PO.nothing_scalar_array_getindex(sve, 1:2)
    # --- the view verb over each remaining shape ---
    @test PO.nothing_scalar_array_view([1.0, 2.0, 3.0], 2:3) == [2.0, 3.0]
    @test isa(PO.nothing_scalar_array_view([1.0, 2.0, 3.0], 2:3), SubArray)
    # A `VecScalar` keeps its scalar part, which carries no asset axis, and views its vector.
    vsv = PO.nothing_scalar_array_view(vs, 2:3)
    @test isa(vsv, VecScalar)
    @test vsv.v == [2.0, 3.0]
    @test vsv.s === 4.2
    # A matrix takes the SAME index on both axes.
    @test PO.nothing_scalar_array_view(S, [1, 3]) == [1.0 3.0; 7.0 9.0]
    # A vector of vectors, matrices or `VecScalar`s applies the rule to each element, and the
    # outer vector keeps its own length: the index reaches the elements, never the vector.
    nested = PO.nothing_scalar_array_view([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 2:3)
    @test length(nested) == 2
    @test nested[1] == [2.0, 3.0]
    @test nested[2] == [5.0, 6.0]
    nested_vs = PO.nothing_scalar_array_view([vs, vs], 1:2)
    @test length(nested_vs) == 2
    @test nested_vs[1].v == [1.0, 2.0]
    @test nested_vs[1].s === 4.2
    # --- the odd-order view verb reads TWO indices ---
    @test isnothing(PO.nothing_scalar_array_view_odd_order(nothing, 1:2, 1:2))
    @test PO.nothing_scalar_array_view_odd_order(S, [1, 3], [2]) ==
          reshape([2.0, 8.0], 2, 1)
    # The two indices are not interchangeable, so the square verb answers differently.
    @test PO.nothing_scalar_array_view_odd_order(S, [1, 3], [1, 3]) ==
          PO.nothing_scalar_array_view(S, [1, 3])
    @test PO.nothing_scalar_array_view_odd_order(S, [1, 3], [2, 3]) !=
          PO.nothing_scalar_array_view(S, [1, 3])
    # --- the getindex verb: the same rules, and a copy rather than a view ---
    @test PO.nothing_scalar_array_getindex(nothing, 1:2) === nothing
    @test PO.nothing_scalar_array_getindex([1.0, 2.0, 3.0], 2:3) == [2.0, 3.0]
    @test !isa(PO.nothing_scalar_array_getindex([1.0, 2.0, 3.0], 2:3), SubArray)
    gvs = PO.nothing_scalar_array_getindex(vs, [1, 3])
    @test isa(gvs, VecScalar)
    @test gvs.v == [1.0, 3.0]
    @test gvs.s === 4.2
    @test PO.nothing_scalar_array_getindex(S, [1, 3]) == [1.0 3.0; 7.0 9.0]
    gnested = PO.nothing_scalar_array_getindex([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 1:2)
    @test length(gnested) == 2
    @test gnested[1] == [1.0, 2.0]
    @test gnested[2] == [4.0, 5.0]
    gmat = PO.nothing_scalar_array_getindex([S, S], [1, 3])
    @test length(gmat) == 2
    @test gmat[1] == [1.0 3.0; 7.0 9.0]
    #=
    The element type is what selects the nested method, and it must be a subtype of the
    method's `Union`. A vector holding a vector AND a matrix has the element type
    `Array{Float64}`, which is a subtype of neither `AbstractVector` nor `AbstractMatrix`, so
    such a vector falls through to the plain vector method and the index selects ELEMENTS of
    the outer vector instead of assets. The library builds no such vector; this pins the
    boundary so a later change does not cross it without a test saying so.
    =#
    mixed = [[1.0, 2.0, 3.0], S]
    @test eltype(mixed) == Array{Float64}
    @test !(eltype(mixed) <: Union{<:AbstractVector, <:AbstractMatrix, <:VecScalar})
    @test PO.nothing_scalar_array_getindex(mixed, 1:2) == mixed
    @test isnothing(PO.nothing_scalar_array_getindex_odd_order(nothing, 1:2, 1:2))
    @test PO.nothing_scalar_array_getindex_odd_order(S, [1, 3], [2]) ==
          reshape([2.0, 8.0], 2, 1)
    #=
    Indexing rather than viewing is the whole point of the getindex verb in the observation
    channel: it preserves the `AbstractWeights` subtype, which a `view` does not, and the
    weighted `Statistics.std` methods dispatch on that subtype.
    =#
    aw = aweights(collect(1.0:10.0))
    @test isa(PO.nothing_scalar_array_getindex(aw, 3:5), AnalyticWeights)
    @test PO.nothing_scalar_array_getindex(aw, 3:5) == [3.0, 4.0, 5.0]
    # --- port_opt_view: the leaf and the `VecScalar` method agree ---
    @test isnothing(PO.port_opt_view(nothing, 1))
    @test isnothing(PO.port_opt_view(nothing, 1, S))
    @test PO.port_opt_view([1.0, 2.0, 3.0], 2:3) == [2.0, 3.0]
    # The `VecScalar` method exists so a threaded tail resolves there; both routes agree.
    tailed = PO.port_opt_view(vs, 2:3, S)
    @test tailed.v == [2.0, 3.0]
    @test tailed.s === 4.2
    @test PO.port_opt_view(vs, 2:3).v == tailed.v
    # The vector method views each element and keeps the outer length.
    pov = PO.port_opt_view([MeanValue(), nothing], 1:2)
    @test length(pov) == 2
    @test pov[1] == MeanValue()
    @test isnothing(pov[2])
    # --- obs_weights_view: the universal fallback, and the vector method ---
    @test PO.obs_weights_view(MinValue(), 1:2) == MinValue()
    owv = PO.obs_weights_view([MeanValue(; w = aw), MedianValue(; w = aw)], 3:5)
    @test length(owv) == 2
    @test owv[1].w == [3.0, 4.0, 5.0]
    @test owv[2].w == [3.0, 4.0, 5.0]
    @test isa(owv[1].w, AnalyticWeights)
    # --- get_window: the window is a row selection, and `dims` is read ---
    X = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0; 10.0 11.0 12.0]
    @test PO.get_window(nothing, X) === Colon()
    @test PO.get_window(Colon(), X) === Colon()
    # A 4 x 3 matrix separates the two axes: the last two rows are 3:4, the last two columns 2:3.
    @test PO.get_window(2, X, 1) == 3:4
    @test PO.get_window(2, X, 2) == 2:3
    @test PO.get_window(2, [1.0, 2.0, 3.0]) == 2:3
    # A window longer than the axis is clamped to the whole axis rather than running past it.
    @test PO.get_window(99, X, 1) == 1:4
    # An explicit index vector is returned unchanged.
    @test PO.get_window([1, 3], X) == [1, 3]
    #=
    --- fourth_moment_index_generator ---

    `(c - 1) * N + r` is the column-major linear index of the pair `(r, c)`, and that pair is
    one axis of the SQUARE cokurtosis matrix, which is `N^2 x N^2`. For `N = 3` and the assets
    `[1, 3]` the four pairs are (1,1), (3,1), (1,3) and (3,3), whose linear indices are 1, 3,
    7 and 9.
    =#
    N = 3
    idx = PO.fourth_moment_index_generator(N, [1, 3])
    @test idx == [1, 3, 7, 9]
    @test idx == [(c - 1) * N + r for c in [1, 3] for r in [1, 3]]
    @test length(PO.fourth_moment_index_generator(N, [1, 2, 3])) == 9
    # --- traverse_concrete_subtypes ---
    #=
    A `subtypes` census run in a REPL shared with other sessions is contaminated by the types
    those sessions defined, so the result is filtered on the declaring module before it is
    counted.
    =#
    cts = PO.traverse_concrete_subtypes(PO.VectorToScalarMeasure)
    own = filter(T -> parentmodule(T) === PO, cts)
    @test !isempty(own)
    @test all(T -> T <: PO.VectorToScalarMeasure, own)
    @test MinValue ∈ own
    @test MaxValue ∈ own
    @test length(own) == length(unique(own))
    #=
    What it collects is a STRUCT type, which is not the same as a concrete type. A parametric
    struct is reported as its `UnionAll`, and a `UnionAll` is not concrete. Of the ten
    measures, the five that carry no field are concrete and the five that carry one are not,
    so a caller that treats the answer as a list of concrete types is wrong for half of them.
    =#
    @test all(isstructtype, own)
    @test !all(isconcretetype, own)
    @test isconcretetype(MinValue)
    @test !isconcretetype(MeanValue)
    @test MeanValue ∈ own
    # The recursion opens an abstract type at any depth and never collects it.
    deep = PO.traverse_concrete_subtypes(PO.AbstractEstimator)
    @test all(isstructtype, filter(T -> parentmodule(T) === PO, deep))
    # --- concrete_typed_array ---
    @test eltype(concrete_typed_array(Any[1, 2.0])) == Union{Float64, Int64}
    @test concrete_typed_array(Any[1, 2.0]) == [1, 2.0]
    @test size(concrete_typed_array(Any[1 2.0; 3 4.0])) == (2, 2)
    # The narrowing is opt-in and it is skipped when the element type is already concrete.
    already = [1.0, 2.0]
    @test PO.concrete_typed_array_if_abstract(already) === already
    @test eltype(PO.concrete_typed_array_if_abstract(Any[1, 2.0])) == Union{Float64, Int64}
    # --- factory and factory_child over a vector ---
    @test factory([MeanValue(), nothing]) == [MeanValue(), nothing]
    fc = PO.factory_child([MeanValue(), MedianValue()], aw)
    @test length(fc) == 2
    @test fc[1].w === aw
    @test fc[2].w === aw
    # --- resolve_deferred_quantities: the universal fallback returns its argument ---
    @test PO.resolve_deferred_quantities(7, nothing) === 7
    @test PO.resolve_deferred_quantities(MinValue(), nothing) == MinValue()
end
@testset "The propagation machinery reports its own broken rows" begin
    using PortfolioOptimisers, Test
    PO = PortfolioOptimisers
    #=
    Condition 2 of the sweep of `src/02_Tools.jl`, issue #441. The two module self-checks and
    the AST helpers behind `@propagatable` and `@forward_properties` state what they reject.
    Before this testset every rejecting branch was uncovered, so each claim was read and never
    run. The three tables of `check_prop_tag_macros` and the registry of
    `check_propagatable_contracts` are arguments for exactly this reason: the shipped tables
    are complete, so a violation can only be driven by passing one that is not.
    =#
    # --- extract_field_name: a bare name, a typed field, and anything else ---
    @test PO.extract_field_name(:a) === :a
    @test PO.extract_field_name(:(a::Int)) === :a
    @test_throws ErrorException PO.extract_field_name(1)
    # --- propagatable_bare_name: a name, a parametric name, a subtyped name ---
    @test PO.propagatable_bare_name(:A) === :A
    @test PO.propagatable_bare_name(:(A{T1})) === :A
    @test PO.propagatable_bare_name(:(A <: B)) === :A
    @test PO.propagatable_bare_name(:(A{T1} <: B)) === :A
    @test_throws ErrorException PO.propagatable_bare_name(1)
    # --- propagatable_find_struct: unwrap a macro chain, and reject anything else ---
    sd = :(struct Q
               a
           end)
    node, rebuild = PO.propagatable_find_struct(sd)
    @test node === sd
    @test rebuild(node) === sd
    # A macro chain is unwrapped to the struct and rebuilt around a replacement, which is what
    # lets `@propagatable` sit outside `@concrete`.
    wrapped = Expr(:macrocall, Symbol("@m"), nothing, sd)
    node2, rebuild2 = PO.propagatable_find_struct(wrapped)
    @test node2 === sd
    @test rebuild2(sd) == wrapped
    @test_throws ErrorException PO.propagatable_find_struct(1)
    @test_throws ErrorException PO.propagatable_find_struct(:(f(x)))
    #=
    --- propagatable_parse_body over a DOCUMENTED field ---

    A docstring in a struct body parses as a `Core.@doc` macrocall wrapping the field, so the
    tag sits one level in. The parser must reach through the doc node to record the tag, and
    it must rebuild the doc node with the tag stripped, or the struct keeps a macro call no
    macro defines. A documented field carrying no tag is carried through unchanged and still
    counted as a field.
    =#
    body = quote
        """
        the documented tagged field
        """
        @fprop a::T1
        """
        the documented plain field
        """
        plain::T2
        @vprop c
    end
    tagged, all_fields, new_body = PO.propagatable_parse_body(body)
    @test tagged[:fprop] == [:a]
    @test tagged[:vprop] == [:c]
    @test all_fields == [:a, :plain, :c]
    # The rebuilt body holds no tag call at any depth, and the doc nodes survive.
    @test !any(PO.is_prop_tag_call, new_body.args)
    docnodes = filter(x -> x isa Expr && x.head == :macrocall && PO.is_doc_macro(x.args[1]),
                      new_body.args)
    @test length(docnodes) == 2
    @test docnodes[1].args[end] == :(a::T1)
    @test docnodes[2].args[end] == :(plain::T2)
    #=
    --- check_prop_tag_macros: each of its three clauses ---

    The shipped tables report nothing, which is the guarantee the module's own call asserts.
    =#
    @test isnothing(PO.check_prop_tag_macros())
    # A tag with no stub macro, which no channel names either: two clauses at once.
    e1 = try
        PO.check_prop_tag_macros((:sixth,), (Symbol("@sixth"),))
        nothing
    catch err
        err
    end
    @test isa(e1, ArgumentError)
    @test occursin("`:sixth` declares no `@sixth` stub macro.", e1.msg)
    @test occursin("`:sixth` appears in no channel of `PROP_TAG_CHANNELS`.", e1.msg)
    # A tag a channel DOES name, and for which `prop_tag_expr` has no field transform. This is
    # the clause that catches a tag added to a second channel without a transform there.
    bad_channels = (factory = (gate = (:sixth,), precedence = (:sixth,)),)
    e2 = try
        PO.check_prop_tag_macros((:sixth,), (Symbol("@sixth"),), bad_channels)
        nothing
    catch err
        err
    end
    @test isa(e2, ArgumentError)
    @test occursin("`:sixth` has no field transform in channel `:factory`.", e2.msg)
    @test !occursin("appears in no channel", e2.msg)
    # Every violation is reported together, so one run names them all.
    @test count("`:sixth`", e1.msg) == 2
    #=
    --- check_propagatable_contracts: a broken registry is reported, not passed over ---

    The registry is an argument, so a broken contract is driven without registering a broken
    type in the shipped registry.
    =#
    @test isnothing(PO.check_propagatable_contracts())
    mc = Module(:ContractThrowProbe)
    Base.eval(mc, quote
                  struct KwGap{T1, T2}
                      inner::T1
                      typo::T2
                  end
                  KwGap(; inner = nothing, typoo = 1) = KwGap(inner, typoo)
              end)
    pool = PO.prior_result_property_pool()
    e3 = try
        PO.check_propagatable_contracts([(mc.KwGap, ())], pool)
        nothing
    catch err
        err
    end
    @test isa(e3, ArgumentError)
    @test occursin("1 broken contract(s)", e3.msg)
    @test occursin("field `typo` is not a keyword", e3.msg)
    # The count in the message is the count of messages, not the count of types.
    e4 = try
        PO.check_propagatable_contracts([(mc.KwGap, (:sgima,))], pool)
        nothing
    catch err
        err
    end
    @test occursin("2 broken contract(s)", e4.msg)
    #=
    --- the prior channel of @propagatable ---

    A struct with an `@pprop` field gains a `factory(x, pr::AbstractPriorResult, ...)` method,
    and no other channel emits one. `sel` keeps a value the caller already set and takes the
    prior's value when the field is unset, so the two cases below are the whole rule.
    =#
    n0 = length(PO.PROPAGATABLE_CONTRACTS)
    mp = Module(:PpropChannelProbe)
    Base.eval(mp, :(using PortfolioOptimisers))
    Base.eval(mp, :(using PortfolioOptimisers: @propagatable, @pprop))
    Base.eval(mp, quote
                  @propagatable struct PpropProbe{T1}
                      @pprop mu::T1
                  end
                  PpropProbe(; mu = nothing) = PpropProbe(mu)
              end)
    @test length(PO.PROPAGATABLE_CONTRACTS) == n0 + 1
    @test PO.PROPAGATABLE_CONTRACTS[end] == (mp.PpropProbe, (:mu,))
    pr = PO.LowOrderPrior(; X = [1.0 2.0; 3.0 4.0; 5.0 6.0], mu = [0.1, 0.2],
                          sigma = [1.0 0.0; 0.0 1.0])
    @test factory(mp.PpropProbe(; mu = nothing), pr).mu == [0.1, 0.2]
    @test factory(mp.PpropProbe(; mu = [9.9, 8.8]), pr).mu == [9.9, 8.8]
    # The declaration was throwaway, so leave the checked registry as it was found.
    pop!(PO.PROPAGATABLE_CONTRACTS)
    @test length(PO.PROPAGATABLE_CONTRACTS) == n0
    # --- forward_flatten_path: a bare name, a dotted path, and the two rejections ---
    @test PO.forward_flatten_path(:a) == [:a]
    @test PO.forward_flatten_path(:(a.b)) == [:a, :b]
    @test PO.forward_flatten_path(:(a.b.c)) == [:a, :b, :c]
    @test_throws ErrorException PO.forward_flatten_path(Expr(:., :a, 1))
    @test_throws ErrorException PO.forward_flatten_path(:(f(x)))
    #=
    --- the getproperty fallback of @forward_properties ---

    The generated `getproperty` tries the `swap` rules, then the struct's own fields, then the
    computed rules, and falls through to `getfield` for anything else. The fall-through is the
    branch that gives an unknown property the ordinary `FieldError` rather than a silent
    `nothing`.
    =#
    mf = Module(:ForwardFallbackProbe)
    Base.eval(mf, :(using PortfolioOptimisers))
    Base.eval(mf, :(using PortfolioOptimisers: @forward_properties))
    Base.eval(mf, quote
                  struct FwdProbe
                      a::Int
                      b::Int
                  end
                  @forward_properties FwdProbe begin
                      swap(a, b)
                  end
              end)
    fx = mf.FwdProbe(1, 2)
    # The `swap` rule wins over the field of the same name, and `getfield` still reaches it.
    @test fx.a == 2
    @test fx.b == 2
    @test getfield(fx, :a) == 1
    @test propertynames(fx) == (:a, :b)
    # The fall-through: a property that is neither a rule nor a field.
    @test_throws FieldError fx.zzz
    # The macro rejects a body that is not a block, and a rule that is not a call.
    @test_throws LoadError Base.eval(mf, :(@forward_properties FwdProbe 1))
    @test_throws LoadError Base.eval(mf, :(@forward_properties FwdProbe begin
                                               1
                                           end))
end
@testset "@forward_properties: every rule, and every rule it rejects" begin
    using PortfolioOptimisers, Test
    PO = PortfolioOptimisers
    #=
    Condition 2 of the sweep of `src/02_Tools.jl`, issue #441. The body of this macro runs at
    macro-expansion time, so the only way to reach it from a test is to write the rule. Every
    shipped call sits in `src/`, where the expansion happens during precompilation and no test
    process ever compiles the body. The rules below are therefore the first exercise this
    macro's own code gets, and each rejection states the message it gives.
    =#
    mb = Module(:ForwardRulesProbe)
    Base.eval(mb, :(using PortfolioOptimisers))
    Base.eval(mb, :(using PortfolioOptimisers: @forward_properties))
    Base.eval(mb, quote
                  struct Inner
                      w::Any
                      z::Any
                  end
                  struct OuterF
                      pa::Inner
                      n::Int
                  end
                  struct OuterFN
                      pa::Inner
                      n::Int
                  end
                  struct OuterA
                      sol::Inner
                      n::Int
                  end
                  struct OuterC
                      sol::Inner
                      n::Int
                  end
                  struct OuterS
                      sol::Inner
                      n::Int
                  end
                  @forward_properties OuterF begin
                      forward(pa)
                  end
                  @forward_properties OuterFN begin
                      forward(pa, w)
                  end
                  @forward_properties OuterA begin
                      alias(zz, sol.z)
                  end
                  @forward_properties OuterC begin
                      compute(cw, sol.w; broadcast)
                      compute(fn, x -> 42)
                  end
                  @forward_properties OuterS begin
                      swap(n, x -> 7)
                  end
              end)
    inn = mb.Inner([1.0, 2.0], "zed")
    #=
    `forward` with a locator alone is the catch-all: every property the receiver does not
    declare is read off the named child, and `propertynames` unions the child's names in.
    =#
    f = mb.OuterF(inn, 3)
    @test f.w == [1.0, 2.0]
    @test f.z == "zed"
    @test f.n == 3
    @test propertynames(f) == (:pa, :n, :w, :z)
    #=
    `forward` with names forwards those names ALONE, so a name outside the list keeps the
    ordinary `FieldError` rather than reaching the child.
    =#
    fn = mb.OuterFN(inn, 3)
    @test fn.w == [1.0, 2.0]
    @test_throws FieldError fn.z
    @test propertynames(fn) == (:pa, :n, :w)
    # `alias` exposes one name under another, through a dotted path.
    a = mb.OuterA(inn, 3)
    @test a.zz == "zed"
    @test propertynames(a) == (:sol, :n, :zz)
    # `compute` takes a dotted path or an anonymous function; `broadcast` is its only option.
    c = mb.OuterC(inn, 3)
    @test c.cw == [1.0, 2.0]
    @test c.fn == 42
    # `swap` in its function form replaces the field of the same name, and `getfield` still
    # reaches the value the struct really holds.
    sw = mb.OuterS(inn, 3)
    @test sw.n == 7
    @test getfield(sw, :n) == 3
    #=
    Every rejection. The macro returns an `error` at expansion time, so `Base.eval` of the
    declaration raises a `LoadError` that wraps it. The message is asserted, because a rule
    that is rejected with the wrong message sends its author to the wrong line.
    =#
    function rule_error(rule)
        return try
            Base.eval(mb, :(@forward_properties OuterF begin
                                $rule
                            end))
            ""
        catch err
            sprint(showerror, err isa LoadError ? err.error : err)
        end
    end
    @test occursin("unknown option", rule_error(:(forward(pa; bogus))))
    @test occursin("`forward` needs a locator", rule_error(:(forward())))
    @test occursin("`forward` names must be bare identifiers",
                   rule_error(:(forward(pa, 1))))
    @test occursin("`alias` takes `(exposed, locator)`", rule_error(:(alias(zz))))
    @test occursin("`alias` exposed name must be a bare identifier",
                   rule_error(:(alias(1, sol.z))))
    @test occursin("`compute` takes `(exposed, locator|fn)`", rule_error(:(compute(cw))))
    @test occursin("`compute` exposed name must be a bare identifier",
                   rule_error(:(compute(1, sol.w))))
    @test occursin("`broadcast` does not apply to the function form of `compute`",
                   rule_error(:(compute(cw, x -> x; broadcast))))
    @test occursin("`compute` source must be a dotted path", rule_error(:(compute(cw, 1))))
    @test occursin("`swap` takes `(field, locator|fn)`", rule_error(:(swap(n))))
    @test occursin("`swap` field name must be a bare identifier", rule_error(:(swap(1, n))))
    @test occursin("`broadcast` does not apply to the function form of `swap`",
                   rule_error(:(swap(n, x -> x; broadcast))))
    @test occursin("`swap` source must be a bare name", rule_error(:(swap(n, 1))))
    @test occursin("unknown rule `elide`", rule_error(:(elide(n))))
    #=
    `forward_walk_expr` is the half that builds the property walk. A path of depth two or more
    guards each step with `forward_nonnothing`, so an unset intermediate names the whole
    locator rather than raising a bare `FieldError` from the middle of it.
    =#
    walk = PO.forward_walk_expr([:sol, :w], :OuterA, false)
    @test walk isa Expr
    @test walk.head === :let
    @test occursin("forward_nonnothing", string(walk))
    # A path of depth one needs no guard, because there is no intermediate to be unset.
    @test !occursin("forward_nonnothing",
                    string(PO.forward_walk_expr([:sol], :OuterA, false)))
    # The broadcast form is built only at the last step of the path.
    @test occursin("AbstractVector",
                   string(PO.forward_walk_expr([:sol, :w], :OuterA, true)))
    # `forward_nonnothing` names the whole locator and the node that was unset.
    @test PO.forward_nonnothing(3, Int, "sol.w", "sol") === 3
    err = try
        PO.forward_nonnothing(nothing, Int, "sol.w", "sol")
        nothing
    catch e
        e
    end
    @test isa(err, PO.PropertyPathError)
    msg = sprint(showerror, err)
    @test occursin("cannot descend path `sol.w`", msg)
    @test occursin("intermediate `sol` is `nothing`", msg)
end
