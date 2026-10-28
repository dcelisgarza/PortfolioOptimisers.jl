#=
The condition 2 and 3 sweep of `src/01_Base/` (#439, child map 1 of #404).

Every claim below is computed and compared, never read. The file's other units are checked
where they are used: the equation and resource caps, the preference channel and the
suggestion threshold in `test_02_equation_parsing.jl`, and `failed_solve_msg` in
`test_01_structs.jl`. What is left is the compact-show contract, the `ScopedConfig` restore
path, the message builders, the iteration protocol, the norm seam and the partial-fit
state seam.
=#
using PortfolioOptimisers, Test, Clustering, JuMP, StableRNGs, StatsBase
using PortfolioOptimisers: @define_pretty_show, compact_show_budget,
                           pretty_show_vector_summary, pretty_show_vector_element,
                           pretty_show_vector_body, pretty_show_fields
import PortfolioOptimisers: has_pretty_show_method, show_fields

# `@define_pretty_show` escapes its whole body, so a caller outside the module must bring the
# six names the body uses into scope, and must `import` the one it adds a method to. These
# probes exercise the branches no shipped type reaches: an empty field list, a vector field of
# renderable elements, and a `DataType` field.
struct PSFieldless end
struct PSLeaf
    a::Int
end
struct PSProbe
    leaf::PSLeaf
    rs::Vector{PSLeaf}
    dt::DataType
    n::Nothing
end
@define_pretty_show(PSFieldless)
@define_pretty_show(PSLeaf)
@define_pretty_show(PSProbe)

# The `nothing`-field switch. `PSAllNothing` has fields and every one holds `nothing`, so the
# shipped default renders it as fieldless. `PSParent` holds it as its last field, so the
# parent's connector reads the child's RESOLVED field list, not its declared one. `PSHook`
# overloads `show_fields` and always hides `drop`, whatever `drop` holds.
struct PSAllNothing
    a::Nothing
    b::Nothing
end
struct PSParent
    child::PSAllNothing
end
struct PSHook
    keep::Int
    drop::Int
end
@define_pretty_show(PSAllNothing)
@define_pretty_show(PSParent)
@define_pretty_show(PSHook)
show_fields(::PSHook) = (:keep,)

# A `DynamicAbstractWeights` with no `get_observation_weights` method of its own. The whole
# point of the type is that the unimplemented shape raises rather than computing unweighted.
struct NoWeightsProbe <: PortfolioOptimisers.DynamicAbstractWeights end

# Three partial-fit state probes. `PFSBare` implements nothing, so it reaches the generic
# `merge_states` that names the method its family owes. `PFSOther` is a second struct, so a pair
# drawn from both is a type mismatch. `PFSFull` implements the interface the way a shipped state
# will: it calls `assert_mergeable_states` first, refuses the mismatch its own family adds, and
# folds the triple with `chan_merge`.
struct PFSBare <: PortfolioOptimisers.AbstractPartialFitState
    n::Int
    mu::Vector{Float64}
    M::Vector{Float64}
end
struct PFSOther <: PortfolioOptimisers.AbstractPartialFitState
    n::Int
end
struct PFSFull <: PortfolioOptimisers.AbstractPartialFitState
    decay::Float64
    n::Int
    mu::Vector{Float64}
    M::Matrix{Float64}
end
function PortfolioOptimisers.merge_states(a::PFSFull, b::PFSFull)
    PortfolioOptimisers.assert_mergeable_states(a, b)
    if a.decay != b.decay
        throw(ArgumentError("two states of different decay cannot be merged, but `a` has $(a.decay) and `b` has $(b.decay)."))
    end
    n, mu, M = PortfolioOptimisers.chan_merge(a.n, a.mu, a.M, b.n, b.mu, b.M)
    return PFSFull(a.decay, n, mu, M)
end

# The batch state of `X`, so a merged state is compared against the state the whole block gives.
# `M` is the accumulator, not the covariance, so nothing is divided by a count.
function batch_state(X)
    n = size(X, 1)
    mu = vec(sum(X; dims = 1)) / n
    D = X .- transpose(mu)
    return n, mu, transpose(D) * D
end

render(f) = (io = IOBuffer(); f(io); String(take!(io)))

# A type declared in a test sandbox renders module-qualified, so the expected text is built
# from the names themselves. The layout, the connectors and the padding are still exact.
const LEAF = string(PSLeaf)
const PROBE = string(PSProbe)

@testset "@define_pretty_show renders every branch" begin
    pe = PortfolioOptimisers
    probe = PSProbe(PSLeaf(1), [PSLeaf(2), PSLeaf(3)], Float64, nothing)
    # An empty field list prints `T()` and returns, so no connector is drawn.
    @test render(io -> show(io, PSFieldless())) == "$(string(PSFieldless))()\n"
    # `:compact` and `:multiline` each print the type name alone.
    @test render(io -> show(IOContext(io, :compact => true), PSLeaf(1))) == "$(LEAF)\n"
    @test render(io -> show(IOContext(io, :multiline => true), PSLeaf(1))) == "$(LEAF)\n"
    # The full rendering, one line per field, each field through its own branch. The `n`
    # field holds `nothing`, so the shipped default hides it and the `┴` marker lands on the
    # last RENDERED field, `dt`. The documentation setting renders `n` as well.
    @test render(io -> show(io, probe)) ==
          join(["$(PROBE)", "  leaf ┼ $(LEAF)", "       │   a ┴ Int64: 1",
                "    rs ┼ 2-element Vector{$(LEAF)}", "       │ $(LEAF) ⋯",
                "       │ $(LEAF) ⋯", "    dt ┴ DataType: Float64", ""], '\n')
    full = pe.with_show_nothing_fields(true) do
        return render(io -> show(io, probe))
    end
    @test full == join(["$(PROBE)", "  leaf ┼ $(LEAF)", "       │   a ┴ Int64: 1",
                        "    rs ┼ 2-element Vector{$(LEAF)}", "       │ $(LEAF) ⋯",
                        "       │ $(LEAF) ⋯", "    dt ┼ DataType: Float64", "     n ┴ nothing", ""],
                       '\n')
    # A `DataType` field prints its type and the wrapper name of the value, so a parametrised
    # type reports the wrapper it instantiates.
    @test occursin("dt ┴ DataType: Array",
                   render(io -> show(io,
                                     PSProbe(PSLeaf(1), [PSLeaf(2)], Vector{Float64},
                                             nothing))))
    # `:po_compact` reaches the nested buffer, so the budget applies at every depth. At a
    # budget of one line the nested leaf collapses to `Name ⋯` and the vector loses its tail.
    @test render(io -> show(IOContext(io, :po_compact => 1), probe)) ==
          join(["$(PROBE)", "  leaf ┼ $(LEAF) ⋯", "    rs ┼ 2-element Vector{$(LEAF)}",
                "       │ $(LEAF) ⋯", "       │ ⋮", "    dt ┴ DataType: Float64", ""], '\n')
    # A budget the rendering fits under changes nothing.
    @test render(io -> show(IOContext(io, :po_compact => 3), probe)) ==
          render(io -> show(io, probe))
    # The macro reads the value with `getproperty`, so it is the flattened surface of
    # `@forward_properties` that stays out of `show`, not the swapped value of a real field.
    @test length(fieldnames(PSProbe)) == 4
    @test all(f -> hasfield(PSProbe, f), fieldnames(PSProbe))
end
@testset "The nothing-field switch and the show_fields hook" begin
    pe = PortfolioOptimisers
    probe = PSProbe(PSLeaf(1), [PSLeaf(2)], Float64, nothing)
    ALL = string(PSAllNothing)
    PARENT = string(PSParent)
    HOOK = string(PSHook)
    shown(x) = render(io -> show(io, x))
    has_n(x) = occursin("n ┴ nothing", shown(x))
    # The shipped default hides a `nothing` field, and the global setter renders it.
    @test pe.SHOW_NOTHING_FIELDS[].default === false
    @test isempty(pe.SHOW_NOTHING_FIELDS[].by_type)
    @test !has_n(probe)
    @test pe.set_show_nothing_fields!(true).default === true
    @test has_n(probe)
    pe.set_show_nothing_fields!(false)
    @test !has_n(probe)
    # A per-name entry beats the global switch, in both directions, and `nothing` removes it.
    pe.set_show_nothing_fields!(:PSProbe, true)
    @test has_n(probe)
    pe.set_show_nothing_fields!(true)
    pe.set_show_nothing_fields!(:PSProbe, false)
    @test !has_n(probe)
    @test pe.set_show_nothing_fields!(:PSProbe, nothing).by_type == Dict{Symbol, Bool}()
    @test has_n(probe)
    pe.set_show_nothing_fields!(false)
    @test !has_n(probe)
    # `with_show_nothing_fields` is scoped: the override holds inside, a task spawned inside
    # inherits it, the global default is untouched, and the previous value returns on exit,
    # including through an error.
    pe.with_show_nothing_fields(true) do
        @test has_n(probe)
        @test fetch(Threads.@spawn has_n(probe))
        @test (@atomic pe.SHOW_NOTHING_FIELDS.default).default === false
    end
    @test !has_n(probe)
    pe.with_show_nothing_fields(:PSProbe, true) do
        @test has_n(probe)
        # A nested override inherits the enclosing per-name entry, so the global `false`
        # inside does not hide `n`.
        pe.with_show_nothing_fields(false) do
            @test has_n(probe)
        end
        pe.with_show_nothing_fields(:PSProbe, nothing) do
            @test !has_n(probe)
        end
    end
    @test_throws ErrorException pe.with_show_nothing_fields(true) do
        @test has_n(probe)
        error("restore")
    end
    @test !has_n(probe)
    # A type whose every field is `nothing` still renders its name, with no empty body. As
    # the last field of a parent it prints on one line, so the parent's connector is `┴`.
    @test shown(PSAllNothing(nothing, nothing)) == "$(ALL)()\n"
    @test shown(PSParent(PSAllNothing(nothing, nothing))) ==
          join(["$(PARENT)", "  child ┴ $(ALL)()", ""], '\n')
    pe.with_show_nothing_fields(true) do
        @test shown(PSAllNothing(nothing, nothing)) ==
              join(["$(ALL)", "  a ┼ nothing", "  b ┴ nothing", ""], '\n')
        @test shown(PSParent(PSAllNothing(nothing, nothing))) ==
              join(["$(PARENT)", "  child ┼ $(ALL)", "        │   a ┼ nothing",
                    "        │   b ┴ nothing", ""], '\n')
    end
    # A `show_fields` overload hides `drop` whatever it holds and whatever the global switch
    # says. A per-name `true` entry overrides the overload and renders every declared field,
    # and a per-name `false` entry keeps the overload's list.
    hook = PSHook(1, 2)
    @test pretty_show_fields(hook) == [:keep]
    @test shown(hook) == join(["$(HOOK)", "  keep ┴ Int64: 1", ""], '\n')
    pe.with_show_nothing_fields(true) do
        @test pretty_show_fields(hook) == (:keep,)
        @test shown(hook) == join(["$(HOOK)", "  keep ┴ Int64: 1", ""], '\n')
    end
    pe.with_show_nothing_fields(:PSHook, true) do
        @test pretty_show_fields(hook) == (:keep, :drop)
        @test shown(hook) ==
              join(["$(HOOK)", "  keep ┼ Int64: 1", "  drop ┴ Int64: 2", ""], '\n')
    end
    pe.with_show_nothing_fields(:PSHook, false) do
        @test pretty_show_fields(hook) == [:keep]
    end
    # The resolver returns the declared tuple when the switch is on, and a filtered vector
    # when it is off, whether or not a field was dropped.
    @test pretty_show_fields(PSLeaf(1)) == [:a]
    @test pretty_show_fields(probe) == [:leaf, :rs, :dt]
    pe.with_show_nothing_fields(true) do
        @test pretty_show_fields(PSLeaf(1)) == (:a,)
        @test pretty_show_fields(probe) == (:leaf, :rs, :dt, :n)
    end
    @test pe.SHOW_NOTHING_FIELDS[].default === false
    @test isempty(pe.SHOW_NOTHING_FIELDS[].by_type)
end
@testset "The vector rendering helpers" begin
    # The summary names the element type when every element shares a wrapper, and falls back
    # to the vector's own element type otherwise.
    @test pretty_show_vector_summary([PSLeaf(1), PSLeaf(2)]) == "2-element Vector{$(LEAF)}"
    @test pretty_show_vector_summary(Union{PSLeaf, PSFieldless}[PSLeaf(1), PSFieldless()]) ==
          "2-element Vector{Union{$(string(PSFieldless)), $(LEAF)}}"
    @test pretty_show_vector_summary([PSLeaf(1), PSFieldless()]) == "2-element Vector{Any}"
    # An element with fields is elided; a fieldless one has nothing to elide.
    @test pretty_show_vector_element(PSLeaf(1)) == "$(LEAF) ⋯"
    @test pretty_show_vector_element(PSFieldless()) == string(PSFieldless)
    # No budget, or a budget the vector fits under, returns the lines unchanged.
    lines = ["a", "b", "c", "d", "e"]
    @test pretty_show_vector_body(IOBuffer(), lines) === lines
    @test pretty_show_vector_body(IOContext(IOBuffer(), :po_compact => 5), lines) === lines
    # Over budget, the head keeps `cld(budget, 2)` lines and the tail keeps the rest, so an
    # odd budget spends its extra line on the head.
    @test pretty_show_vector_body(IOContext(IOBuffer(), :po_compact => 4), lines) ==
          ["a", "b", "⋮", "d", "e"]
    @test pretty_show_vector_body(IOContext(IOBuffer(), :po_compact => 3), lines) ==
          ["a", "b", "⋮", "e"]
    @test pretty_show_vector_body(IOContext(IOBuffer(), :po_compact => 2), lines) ==
          ["a", "⋮", "e"]
    @test pretty_show_vector_body(IOContext(IOBuffer(), :po_compact => 1), lines) ==
          ["a", "⋮"]
end
@testset "has_pretty_show_method covers the three foreign types" begin
    # A parent finds that a field renders through the macro by this predicate. The three
    # foreign types print their own way, and everything else answers `false`.
    @test has_pretty_show_method(JuMP.Model())
    @test has_pretty_show_method(hclust([0.0 1.0; 1.0 0.0]))
    @test has_pretty_show_method(kmeans(randn(StableRNG(123456789), 3, 20), 2))
    @test has_pretty_show_method(PSLeaf(1))
    @test !has_pretty_show_method(1)
    @test !has_pretty_show_method("a")
end
@testset "compact_show_budget resolves its five branches" begin
    pe = PortfolioOptimisers
    # Without `:po_compact`, collapsing applies only to height-limited output, so a plain
    # buffer, a `string` or a file write expands fully.
    @test isnothing(compact_show_budget(IOBuffer()))
    # Height-limited output reads the global setting. The automatic budget is four lines
    # under the terminal height, with a floor of eight.
    @test compact_show_budget(IOContext(IOBuffer(), :limit => true)) == 20
    @test displaysize(IOContext(IOBuffer(), :limit => true)) == (24, 80)
    @test compact_show_budget(IOContext(IOBuffer(), :limit => true,
                                        :displaysize => (30, 80))) == 26
    @test compact_show_budget(IOContext(IOBuffer(), :limit => true,
                                        :displaysize => (10, 80))) == 8
    # A per-call `:po_compact` overrides both, and skips the `:limit` test entirely.
    @test isnothing(compact_show_budget(IOContext(IOBuffer(), :po_compact => false)))
    @test compact_show_budget(IOContext(IOBuffer(), :po_compact => 7)) == 7
    @test compact_show_budget(IOContext(IOBuffer(), :po_compact => true)) == 20
    # The global setting takes the same three values, and the integer form is a fixed budget.
    pe.with_compact_show(false) do
        return @test isnothing(compact_show_budget(IOContext(IOBuffer(), :limit => true)))
    end
    pe.with_compact_show(5) do
        return @test compact_show_budget(IOContext(IOBuffer(), :limit => true)) == 5
    end
    @test pe.COMPACT_SHOW[] === true
end
@testset "ScopedConfig restores the previous value on a throw" begin
    pe = PortfolioOptimisers
    # The outer constructor takes the element type from the value; the inner one converts.
    cfg = pe.ScopedConfig(3)
    @test cfg isa pe.ScopedConfig{Int}
    @test cfg[] === 3
    @test pe.ScopedConfig{Float64}(3)[] === 3.0
    @test_throws InexactError pe.ScopedConfig{Int}(1.5)
    # `set_default!` converts before it stores and returns the stored value.
    @test pe.set_default!(cfg, 5) === 5
    @test cfg[] === 5
    @test pe.set_default!(pe.ScopedConfig{Float64}(0.0), 2) === 2.0
    # A scoped override is restored when the block returns and when it raises.
    @test pe.with_config(cfg, 7) do
        return cfg[]
    end == 7
    @test cfg[] === 5
    @test_throws ErrorException pe.with_config(cfg, 7) do
        @test cfg[] === 7
        return error("boom")
    end
    @test cfg[] === 5
    # The same on each of the four shipped verbs, which is the property a caller relies on
    # when it wraps an untrusted batch: a raise inside the block must not leak the override.
    @test_throws ErrorException pe.with_string_distance(; min_score = 1.5) do
        @test pe.STRING_DISTANCE[].min_score == 1.5
        return error("boom")
    end
    @test pe.STRING_DISTANCE[].min_score == 0.7
    @test_throws ErrorException pe.with_equation_limits(; max_length = 8) do
        @test pe.EQUATION_LIMITS[].max_length == 8
        return error("boom")
    end
    @test pe.EQUATION_LIMITS[].max_length == 4096
    @test_throws ErrorException pe.with_resource_limits(; max_bins = 3) do
        @test pe.RESOURCE_LIMITS[].max_bins == 3
        return error("boom")
    end
    @test pe.RESOURCE_LIMITS[].max_bins == 10_000
    @test_throws ErrorException pe.with_compact_show(4) do
        @test pe.COMPACT_SHOW[] === 4
        return error("boom")
    end
    @test pe.COMPACT_SHOW[] === true
end
@testset "A `with_*` inherits from the active value, a `set_*!` from the global" begin
    pe = PortfolioOptimisers
    # This is the fact each pair owes its reader. A nested `with_*` defaults an omitted
    # keyword from the enclosing override, so overrides compose; a `set_*!` defaults it from
    # the global default, so it cannot be used to amend an override.
    try
        pe.with_string_distance(; min_score = 0.9) do
            pe.with_string_distance(; dist = pe.StringDistances.DamerauLevenshtein()) do
                @test pe.STRING_DISTANCE[].min_score == 0.9
                return @test pe.STRING_DISTANCE[].dist isa
                             pe.StringDistances.DamerauLevenshtein
            end
            pe.set_string_distance!(; dist = pe.StringDistances.DamerauLevenshtein())
            # The global default now carries the shipped 0.7, not the active 0.9.
            return @test (@atomic pe.STRING_DISTANCE.default).min_score == 0.7
        end
    finally
        pe.set_string_distance!(; dist = pe.StringDistances.Levenshtein(), min_score = 0.7)
    end
    @test pe.STRING_DISTANCE[].min_score == 0.7
    @test pe.STRING_DISTANCE[].dist isa pe.StringDistances.Levenshtein
end
@testset "The suggestion admits by score and breaks a tie by position" begin
    pe = PortfolioOptimisers
    SD = pe.StringDistances
    # The normalised similarity is what the threshold gates. Both candidates score the same
    # against `APL`, so the collection's own order decides which one the message names.
    @test SD.compare("APL", "AAPL", SD.Levenshtein()) == 0.75
    @test SD.compare("APL", "APPL", SD.Levenshtein()) == 0.75
    @test SD.compare("APL", "MSFT", SD.Levenshtein()) == 0.0
    @test pe.did_you_mean("APL", ["MSFT", "APPL", "AAPL"]) == " (did you mean `APPL`?)"
    @test pe.did_you_mean("APL", ["AAPL", "APPL", "MSFT"]) == " (did you mean `AAPL`?)"
    # `suggest_declared_key` exists because the strict global default is dead code over short
    # keys. The docstring's own example: a transposition scores 0.5 under Levenshtein, below
    # the 0.7 threshold, and 0.75 under Damerau-Levenshtein, above the looser 0.5.
    @test SD.compare("nuon", "noun", SD.Levenshtein()) == 0.5
    @test SD.compare("nuon", "noun", SD.DamerauLevenshtein()) == 0.75
    @test pe.did_you_mean("nuon", ["noun", "axis"]) == ""
    @test pe.suggest_declared_key(:nuon, (:noun, :axis)) == " (did you mean `noun`?)"
    # The looser configuration is scoped, so it leaves the global default alone.
    @test pe.STRING_DISTANCE[].min_score == 0.7
    @test pe.STRING_DISTANCE[].dist isa SD.Levenshtein
    # Nothing close enough still draws no suggestion, at either threshold.
    @test pe.suggest_declared_key("zzzz", (:noun, :axis)) == ""
    @test pe.did_you_mean("APL", String[]) == ""
end
@testset "The message builders name a size, never a universe" begin
    pe = PortfolioOptimisers
    nx = ["AAPL", "MSFT", "GOOG"]
    # `axis` names the universe the variable was written against, and it inflects the noun.
    m = pe.unknown_variable_msg("APL", nx, "nx"; axis = "factor")
    @test occursin("not in factor universe (3 factors under key `nx`)", m)
    @test occursin("did you mean `AAPL`?", m)
    # A wider candidate pool can name a mistyped group, and the reported size stays the
    # universe's, not the pool's.
    mg = pe.unknown_variable_msg("techh", nx, "nx"; candidates = vcat(nx, "tech"))
    @test occursin("(3 assets under key `nx`)", mg)
    @test occursin("did you mean `tech`?", mg)
    # A re-based row that resolved but projected to zero is a different failure from a row
    # whose names missed, so it gets its own text and names no typo.
    mp = pe.empty_projected_row_msg("f1 >= 0", ["f1", "f2"], "nf", 5)
    @test occursin("resolved against the factor universe (2 factors under key `nf`)", mp)
    @test occursin("projected to an all-zero row over 5 assets", mp)
    @test !occursin("did you mean", mp)
    @test occursin("view `",
                   pe.empty_projected_row_msg("f1 == 0", ["f1"], "nf", 2; noun = "view"))
    # A group that resolved but whose members did not: the member names are caller input and
    # do reach the text; the universe does not.
    mm = pe.missing_group_assets_msg("tech", ["APL"], nx, "nx")
    @test occursin("group `tech`: 1 member(s) not in asset universe", mm)
    @test occursin("(3 assets under key `nx`)", mm)
    @test occursin("[\"APL\"]", mm)
    @test occursin("did you mean `AAPL`?", mm)
    @test !occursin("GOOG", mm)
    # The gross budget message names the bound size and the failed predicate, never a bound
    # value. Scalar or absent bounds have no size, so the scope is named without a count.
    ms = pe.gross_budget_bounds_msg(nothing, nothing)
    @test occursin("Got weight bounds with no negative element in lb or ub.", ms)
    mv = pe.gross_budget_bounds_msg(zeros(3), ones(3))
    @test occursin("Got weight bounds over 3 assets with no negative element in lb or ub.",
                   mv)
    @test !occursin("0.0", mv)
    @test !occursin("1.0", mv)
    # The greater of the two lengths binds, so a scalar paired with a vector still reports.
    @test occursin("over 4 assets", pe.gross_budget_bounds_msg(0.0, ones(4)))
    @test occursin("over 4 assets", pe.gross_budget_bounds_msg(zeros(4), 1.0))
    # A misaligned universe has two shapes, and the message tells them apart. A length
    # mismatch names both counts; an order mismatch names the first position that differs.
    ml = pe.misaligned_axis_msg(["A", "B"], ["A", "B", "C"], "asset", "nx", :nx)
    @test occursin("2 assets are declared, but the data has 3", ml)
    @test occursin("Set `sets.dict[\"nx\"]` to `rd.nx`", ml)
    mo = pe.misaligned_axis_msg(["A", "B"], ["A", "C"], "asset", "nx", :nx)
    @test occursin("both have 2 assets but the order differs, first at position 2: `B` vs `C`",
                   mo)
    mf = pe.misaligned_axis_msg(["f1"], ["f1", "f2"], "factor", "nf", :nf)
    @test occursin("the factor universe declared under key `nf`", mf)
    @test occursin("1 factors are declared, but the data has 2", mf)
end
@testset "strict_diagnostic branches on strict" begin
    pe = PortfolioOptimisers
    # One function owns the policy: the same text throws under `strict` and warns otherwise,
    # and in both cases the offending term is dropped.
    # A macro resolves its module when the testset expands, before `pe` is bound.
    id = PortfolioOptimisers.@log_id()
    @test_throws ArgumentError pe.strict_diagnostic("term dropped", true; id = id)
    err = try
        pe.strict_diagnostic("term dropped", true; id = id)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test err.msg == "term dropped"
    @test_logs (:warn, "term dropped") pe.strict_diagnostic("term dropped", false; id = id)
    @test isnothing(@test_logs (:warn, "term dropped") pe.strict_diagnostic("term dropped",
                                                                            false; id = id))
    # The caller states the `_id`: the default would be the line inside the helper, which
    # every caller shares, so a cap would hide the messages of every other caller.
    @test_throws UndefKeywordError pe.strict_diagnostic("term dropped", false)
    logs, = Test.collect_test_logs() do
        return pe.strict_diagnostic("term dropped", false; id = id)
    end
    @test only(logs).id == id
end
@testset "Every log site of the library passes maxlog and _id (#1690)" begin
    #=
    A cross-validation, a walk-forward, an online selection and a meta-optimiser call the
    same verb many times, and a log site repeats its message on each call. Each site passes
    `maxlog = log_limit(level)`, so that the `max_warn`, `max_info` and `max_debug` caps of
    `LOG_LIMITS` reach it, and an `_id` from `@log_id` over the stable parts of its message,
    so that the cap counts each configuration on its own. The census reads the parsed
    source, because a new site copies its neighbour.
    =#
    root = normpath(joinpath(@__DIR__, ".."))
    levels = (Symbol("@warn"), Symbol("@info"), Symbol("@debug"), Symbol("@error"),
              Symbol("@logmsg"))
    sites = Tuple{String, Int, Bool, Bool}[]
    function walk!(x::Expr, path::AbstractString, line::Ref{Int})
        if x.head === :macrocall && x.args[1] in levels
            if x.args[2] isa LineNumberNode
                line[] = x.args[2].line
            end
            kws = [a.args[1] for a in x.args[3:end] if a isa Expr && a.head in (:(=), :kw)]
            push!(sites, (relpath(path, root), line[], :maxlog in kws, :_id in kws))
        end
        for a in x.args
            walk!(a, path, line)
        end
        return nothing
    end
    walk!(x::LineNumberNode, ::AbstractString, line::Ref{Int}) = (line[] = x.line; nothing)
    walk!(::Any, ::AbstractString, ::Ref{Int}) = nothing
    for dir in (joinpath(root, "src"), joinpath(root, "ext")), (dp, _, fns) in walkdir(dir)
        for fn in fns
            if endswith(fn, ".jl")
                path = joinpath(dp, fn)
                walk!(Meta.parseall(read(path, String); filename = path), path, Ref(0))
            end
        end
    end
    @test !isempty(sites)
    @test filter(s -> !(s[3] && s[4]), sites) == Tuple{String, Int, Bool, Bool}[]
end
@testset "first_error_line truncates the content, not the line" begin
    pe = PortfolioOptimisers
    # The cap counts the characters of the error, and the ellipsis is added on top, so a
    # truncated line is one character longer than the cap.
    long = pe.first_error_line(ErrorException("x"^500), 200)
    @test length(long) == 201
    @test long == "x"^200 * "…"
    # A line of exactly the cap is left alone; one character more is cut.
    @test pe.first_error_line(ErrorException("abcde"), 5) == "abcde"
    @test pe.first_error_line(ErrorException("abcde"), 4) == "abcd…"
    # Only the first line survives, so a multi-line payload cannot reach a log.
    @test pe.first_error_line(ErrorException("boom\nsecond line"), 200) == "boom"
    # A value that is not an exception is shown with `repr`.
    @test pe.first_error_line(:boom, 200) == ":boom"
    @test pe.first_error_line(42, 200) == "42"
    # A trial that recorded no stage is reported under the single stage `:trial`.
    @test pe.failed_solve_msg(Dict("s" => ErrorException("bang"))) ==
          "Model could not be solved satisfactorily (1 solver trial(s)).\n  s: trial → bang"
    # The keyword reaches `first_error_line`.
    @test occursin("s: optimize! → xxxx…",
                   pe.failed_solve_msg(Dict("s" =>
                                                Dict(:optimize! => ErrorException("x"^20)));
                                       max_line_length = 4))
end
@testset "An estimator, an algorithm and a result are one-element iterables" begin
    pe = PortfolioOptimisers
    # The protocol lets a caller write one estimator where the API takes a collection, so a
    # scalar and a one-element vector are the same input.
    for obj in (L1Norm(), pe.NoDefault(), pe.VecScalar(; v = [1.0, 2.0], s = 3.0))
        @test length(obj) == 1
        @test only(collect(obj)) === obj
        @test first(obj) === obj
        @test obj[1] === obj
        @test iterate(obj) === (obj, 2)
        @test isnothing(iterate(obj, 2))
        @test_throws BoundsError obj[2]
        @test_throws BoundsError obj[0]
    end
end
@testset "assert_gt0 over each of its five shapes" begin
    pe = PortfolioOptimisers
    # The guard is one verb over five containers, and each states the predicate it failed.
    @test isnothing(pe.assert_gt0(Dict(:a => 1.0, :b => 2.0)))
    @test_throws DomainError pe.assert_gt0(Dict(:a => 1.0, :b => 0.0))
    @test isnothing(pe.assert_gt0([:a => 1.0, :b => 2.0]))
    @test_throws DomainError pe.assert_gt0([:a => 1.0, :b => -1.0])
    @test isnothing(pe.assert_gt0(:a => 1.0))
    @test_throws DomainError pe.assert_gt0(:a => 0.0)
    @test isnothing(pe.assert_gt0([1.0, 2.0]))
    @test_throws DomainError pe.assert_gt0([1.0, 0.0])
    @test isnothing(pe.assert_gt0(1.0))
    @test_throws DomainError pe.assert_gt0(0.0)
    # The message names the symbol the caller passed, so the report points at the keyword.
    derr = try
        pe.assert_gt0(Dict(:a => 0.0), :n_sim)
        nothing
    catch e
        e
    end
    @test derr isa DomainError
    # The guard builds `DomainError(val, text)`, so the container is `.val` and the text
    # names the symbol and counts the entries that failed.
    @test derr.val == Dict(:a => 0.0)
    @test occursin("all(x -> 0 < x, values(n_sim)) must hold", derr.msg)
    @test occursin("count(x -> !(0 < x), values(n_sim)) => 1", derr.msg)
    perr = try
        pe.assert_gt0(:a => 0.0, :alpha)
        nothing
    catch e
        e
    end
    @test perr isa DomainError
    @test perr.val == 0.0
    @test occursin("0 < alpha[2] must hold", perr.msg)
    # The composed guards take a value of any of the five shapes, and their varargs method
    # accepts everything else without a check, which is how an absent value passes through.
    @test isnothing(pe.assert_nonempty_gt0_finite_val([1.0, 2.0]))
    @test_throws DomainError pe.assert_nonempty_gt0_finite_val([1.0, 0.0])
    @test isnothing(pe.assert_nonempty_gt0_finite_val(nothing))
    @test isnothing(pe.assert_nonempty_nonneg_finite_val(nothing))
    @test isnothing(pe.assert_nonempty_finite_val(nothing))
end
@testset "assert_finite refuses a container that carries one non-finite entry" begin
    pe = PortfolioOptimisers
    # The guard is named for what it demands of the whole container, so one bad cell fails
    # it. It used to be `any`, which let a matrix that is all but one cell `NaN` satisfy a
    # check named *finite*, and left an `Inf` admitted by every composite that calls it.
    @test isnothing(pe.assert_finite(Dict(:a => 1.0, :b => 2.0)))
    @test_throws DomainError pe.assert_finite(Dict(:a => 1.0, :b => NaN))
    @test_throws DomainError pe.assert_finite(Dict(:a => 1.0, :b => Inf))
    @test isnothing(pe.assert_finite([:a => 1.0, :b => 2.0]))
    @test_throws DomainError pe.assert_finite([:a => 1.0, :b => NaN])
    @test_throws DomainError pe.assert_finite([:a => 1.0, :b => Inf])
    @test isnothing(pe.assert_finite([1.0 2.0; 3.0 4.0]))
    @test_throws DomainError pe.assert_finite([1.0 2.0; NaN 4.0])
    @test_throws DomainError pe.assert_finite([1.0 2.0; Inf 4.0])
    # The scalar shapes were already strict, and stay so.
    @test isnothing(pe.assert_finite(:a => 1.0))
    @test_throws DomainError pe.assert_finite(:a => NaN)
    @test isnothing(pe.assert_finite(1.0))
    @test_throws DomainError pe.assert_finite(Inf)
    # The message names the predicate, so a refusal reports finiteness rather than the
    # non-negativity check that used to fire first on a `NaN` and never fired on an `Inf`.
    err = try
        pe.assert_nonempty_nonneg_finite_val([0.2 0.3; Inf 0.31], :iv)
        nothing
    catch e
        e
    end
    @test err isa DomainError
    @test err.val == [0.2 0.3; Inf 0.31]
    @test occursin("all(isfinite, iv) must hold", err.msg)
    @test occursin("count(!isfinite, iv) => 1", err.msg)
    err = try
        pe.assert_nonempty_nonneg_finite_val([0.2 0.3; NaN 0.31], :iv)
        nothing
    catch e
        e
    end
    @test err isa DomainError
    @test isequal(err.val, [0.2 0.3; NaN 0.31])
    @test occursin("all(isfinite, iv) must hold", err.msg)
    # `assert_all_finite` checks the same predicate over an array; it differs in the error
    # type it raises and in reporting where the breach is without quoting the data.
    @test isnothing(pe.assert_all_finite([1.0 2.0; 3.0 4.0]))
    @test_throws pe.IsNonFiniteError pe.assert_all_finite([1.0 2.0; NaN 4.0])
end
@testset "The two unit-interval guards differ only at the ends" begin
    pe = PortfolioOptimisers
    # The open guard refuses both ends; the closed one takes them. That is the whole of the
    # difference, and it is why a compression weight needs the second and a probability the
    # first.
    @test isnothing(pe.assert_unit_interval(0.5))
    @test isnothing(pe.assert_closed_unit_interval(0.5))
    @test_throws DomainError pe.assert_unit_interval(0.0)
    @test_throws DomainError pe.assert_unit_interval(1.0)
    @test isnothing(pe.assert_closed_unit_interval(0.0))
    @test isnothing(pe.assert_closed_unit_interval(1.0))
    @test_throws DomainError pe.assert_closed_unit_interval(-eps())
    @test_throws DomainError pe.assert_closed_unit_interval(1.0 + eps())
    # Each message states the predicate that failed and names the symbol the caller passed.
    oerr = try
        pe.assert_unit_interval(1.0, :alpha)
        nothing
    catch e
        e
    end
    cerr = try
        pe.assert_closed_unit_interval(1.5, :n7)
        nothing
    catch e
        e
    end
    @test oerr isa DomainError
    @test cerr isa DomainError
    # The guards build `DomainError(val, text)`, so the number is `.val` and the text is
    # `.msg`, and `showerror` prints one after the other.
    @test oerr.val == 1.0
    @test oerr.msg == "0 < alpha < 1 must hold. Got\nalpha => 1.0"
    @test cerr.val == 1.5
    @test cerr.msg == "0 <= n7 <= 1 must hold. Got\nn7 => 1.5"
    @test sprint(showerror, oerr) ==
          "DomainError with 1.0:\n0 < alpha < 1 must hold. Got\nalpha => 1.0"
    # Both carry the varargs method that checks nothing, which is how a slot holding a
    # Calibration Rule passes a guard written for a number.
    rule = ScenarioCount(; n = 5)
    @test isnothing(pe.assert_unit_interval(nothing))
    @test isnothing(pe.assert_closed_unit_interval(nothing))
    @test isnothing(pe.assert_closed_unit_interval(rule, :n))
end
@testset "norm_factor and norm_error carry the units of the alg" begin
    pe = PortfolioOptimisers
    a, b = [0.5, 0.5], [0.2, 0.9]
    # With no observation count there is nothing to normalise by, whatever the alg.
    @test pe.norm_factor(nothing, nothing) == 1
    @test pe.norm_factor(L2Norm(), nothing) == 1
    @test pe.norm_factor(SquaredL2Norm(), nothing) == 1
    @test pe.norm_factor(L1Norm(), nothing) == 1
    @test pe.norm_factor(LpNorm(), nothing) == 1
    @test pe.norm_factor(LInfNorm(), nothing) == 1
    # `a - b` has norm 0.5 exactly, which separates the squaring alg from the plain one.
    @test pe.norm_error(L2Norm(), a, b) == 0.5
    @test pe.norm_error(SquaredL2Norm(), a, b) == 0.25
    # The single-argument arity is the same measure over `a` alone.
    @test pe.norm_error(SquaredL2Norm(), a - b) == 0.25
    @test pe.norm_error(L2Norm(), a - b) == 0.5
    @test pe.norm_error(nothing, a - b) == 0.5
    # The observation count divides, and each alg divides by its own factor. `L2Norm` takes
    # the square root and `SquaredL2Norm` does not, so the second is the square of the first.
    @test pe.norm_factor(L2Norm(; ddof = 1), 5) == 2.0
    @test pe.norm_factor(SquaredL2Norm(; ddof = 1), 5) == 4
    @test pe.norm_error(L2Norm(), a, b, 5) == 0.25
    @test pe.norm_error(SquaredL2Norm(), a, b, 5) == 0.0625
    @test pe.norm_error(SquaredL2Norm(), a - b, 5) == 0.0625
    @test pe.norm_error(SquaredL2Norm(), a, b, 5) == pe.norm_error(L2Norm(), a, b, 5)^2
    # The norm of order `p` divides by `(T - d)^(1/p)` (#1271). `L1Norm` takes `ddof` and
    # keeps the source's `T` at its default, `LpNorm` defaults to the `ddof` of `L2Norm`, so
    # the two agree at `p = 2`, and `LInfNorm` divides by the limit `1` at every `T`.
    @test pe.norm_factor(L1Norm(), 5) == 5
    @test pe.norm_factor(L1Norm(; ddof = 1), 5) == 4
    @test pe.norm_error(L1Norm(; ddof = 1), a, b, 5) ≈ 0.175
    @test pe.norm_factor(LpNorm(; p = 2), 5) == pe.norm_factor(L2Norm(), 5)
    @test pe.norm_error(LpNorm(; p = 2), a, b, 5) ≈ pe.norm_error(L2Norm(), a, b, 5)
    @test pe.norm_factor(LpNorm(; p = 3), 9) ≈ 2
    @test pe.norm_factor(LInfNorm(), 5) == 1
    @test pe.norm_factor(LInfNorm(), 252) == 1
    @test pe.norm_error(LInfNorm(), a, b, 252) == pe.norm_error(LInfNorm(), a, b)
    @test pe.norm_error(LInfNorm(), a, b, 252) ≈ 0.4
    @test_throws MethodError LInfNorm(; ddof = 0)
    @test_throws DomainError L1Norm(; ddof = -1)
end
@testset "An unimplemented observation-weight shape names the shape" begin
    pe = PortfolioOptimisers
    # The refusal names the arity that is missing, so the reader knows which method to write.
    # With no data argument at all there is no shape to name, and the text says so.
    err = try
        pe.get_observation_weights(NoWeightsProbe())
        nothing
    catch e
        e
    end
    @test err isa pe.ObservationWeightsError
    @test occursin("no `get_observation_weights` method for the given input", err.msg)
    @test occursin("NoWeightsProbe is a DynamicAbstractWeights", err.msg)
    merr = try
        pe.get_observation_weights(NoWeightsProbe(), ones(3, 10))
        nothing
    catch e
        e
    end
    @test merr isa pe.ObservationWeightsError
    @test occursin("for a 2-dimensional input of size (3, 10)", merr.msg)
    # The two shapes that need no method: nothing computes unweighted, and a plain vector is
    # already the weights.
    @test isnothing(pe.get_observation_weights(nothing, ones(3, 10)))
    @test pe.get_observation_weights([1.0, 2.0]) == [1.0, 2.0]
end

@testset "A partial-fit state is a Result under its own root" begin
    pe = PortfolioOptimisers
    # The root buys the length-1 iteration protocol and the pretty show, so it sits under
    # `AbstractResult` and neither `Union` of the protocol moves.
    @test pe.AbstractPartialFitState <: pe.AbstractResult
    @test supertype(pe.RegimeAdjustedVarianceState) === pe.AbstractPartialFitState
    s = PFSOther(3)
    @test length(s) == 1
    @test s[1] === s
    @test collect(s) == [s]
    # The root is not exported. The seam's whole exported surface is the two verbs, so nothing
    # here reaches a caller by its bare name. The root, `merge_states`, and the two helpers
    # that a new state's merge calls are `public` (#1775), which `names` also lists, so those
    # read `Base.isexported`.
    @test :partial_fit! in names(PortfolioOptimisers)
    @test :partial_fit in names(PortfolioOptimisers)
    @test !Base.isexported(PortfolioOptimisers, :AbstractPartialFitState)
    @test !Base.isexported(PortfolioOptimisers, :merge_states)
    @test :chan_merge in names(PortfolioOptimisers)
    @test :assert_mergeable_states in names(PortfolioOptimisers)
    @test !Base.isexported(PortfolioOptimisers, :chan_merge)
    @test !Base.isexported(PortfolioOptimisers, :assert_mergeable_states)
end
@testset "A view drops a state on the observation axis, whatever the family" begin
    pe = PortfolioOptimisers
    # One root method is the drop, so a family that adds a state owes no method of its own.
    # `@fprop` routes the field through the verb, and the method returns `nothing`.
    # ADR 0107.
    for state in
        (PFSOther(3), PFSBare(1, [1.0], [0.0]), PFSFull(0.97, 2, [1.0], fill(0.5, 1, 1)))
        @test isnothing(pe.obs_weights_view(state, 1:5))
        @test isnothing(pe.obs_weights_view(state, [2, 4]))
        @test isnothing(pe.obs_weights_view(state, Colon()))
    end
    # The universal fallback still carries every value that is not a state.
    @test pe.obs_weights_view(3, 1:5) == 3
end
@testset "assert_mergeable_states names the mismatch it refuses" begin
    pe = PortfolioOptimisers
    a = PFSBare(2, [1.0, 2.0], [0.5, 0.5])
    # A pair of one struct over one asset count passes, and returns nothing.
    @test isnothing(pe.assert_mergeable_states(a, PFSBare(3, [2.0, 3.0], [1.0, 1.0])))
    # Two different structs. The message names both types, so the reader sees which pair was
    # handed over rather than only that a pair was refused.
    terr = try
        pe.assert_mergeable_states(a, PFSOther(1))
        nothing
    catch e
        e
    end
    @test terr isa ArgumentError
    @test occursin("different types cannot be merged", terr.msg)
    @test occursin("PFSBare", terr.msg)
    @test occursin("PFSOther", terr.msg)
    # One struct, two asset counts. The message names the field that disagrees and both shapes.
    derr = try
        pe.assert_mergeable_states(a, PFSBare(3, [1.0, 2.0, 3.0], [1.0, 1.0, 1.0]))
        nothing
    catch e
        e
    end
    @test derr isa DimensionMismatch
    @test occursin("different numbers of assets", derr.msg)
    @test occursin("`mu`", derr.msg)
    @test occursin("(2,)", derr.msg)
    @test occursin("(3,)", derr.msg)
end
@testset "chan_merge reproduces the batch state, and is associative" begin
    pe = PortfolioOptimisers
    rng = StableRNG(987654321)
    X = randn(rng, 30, 4)
    # The three shapes the merge takes. `M` is a matrix for a co-moment, a vector for a per-asset
    # moment, and a scalar for one series, and the outer product degenerates to a square each time.
    nA, muA, MA = batch_state(view(X, 1:12, :))
    nB, muB, MB = batch_state(view(X, 13:30, :))
    n, mu, M = pe.chan_merge(nA, muA, MA, nB, muB, MB)
    nX, muX, MX = batch_state(X)
    @test n == nX
    @test isapprox(mu, muX; rtol = 1e-12)
    @test isapprox(M, MX; rtol = 1e-12)
    # A per-asset accumulator is the diagonal of the co-moment one, so the vector method and the
    # matrix method answer the same numbers over the same blocks.
    nv, muv, Mv = pe.chan_merge(nA, muA, [MA[i, i] for i in axes(MA, 1)], nB, muB,
                                [MB[i, i] for i in axes(MB, 1)])
    @test nv == nX
    @test isapprox(muv, muX; rtol = 1e-12)
    @test isapprox(Mv, [MX[i, i] for i in axes(MX, 1)]; rtol = 1e-12)
    # One series. Every operand is a scalar and the merge stays scalar.
    x = view(X, :, 1)
    ns, mus, Ms = pe.chan_merge(12, StatsBase.mean(view(x, 1:12)),
                                sum(abs2, view(x, 1:12) .- StatsBase.mean(view(x, 1:12))),
                                18, StatsBase.mean(view(x, 13:30)),
                                sum(abs2, view(x, 13:30) .- StatsBase.mean(view(x, 13:30))))
    @test ns == 30
    @test isapprox(mus, StatsBase.mean(x); rtol = 1e-12)
    @test isapprox(Ms, sum(abs2, x .- StatsBase.mean(x)); rtol = 1e-12)
    # Associativity is the property that makes a parallel fit legal: three disjoint blocks give
    # one state whatever order they are folded in.
    s1 = batch_state(view(X, 1:7, :))
    s2 = batch_state(view(X, 8:19, :))
    s3 = batch_state(view(X, 20:30, :))
    l = pe.chan_merge(pe.chan_merge(s1..., s2...)..., s3...)
    r = pe.chan_merge(s1..., pe.chan_merge(s2..., s3...)...)
    @test l[1] == r[1]
    @test isapprox(l[2], r[2]; rtol = 1e-12)
    @test isapprox(l[3], r[3]; rtol = 1e-12)
    # The merge divides by the total count, so two empty blocks are refused rather than answered
    # with NaN. This is a domain check on a count, not a tolerance.
    @test_throws DomainError pe.chan_merge(0, [0.0], [0.0], 0, [0.0], [0.0])
    @test_throws DomainError pe.chan_merge(0, [0.0], zeros(1, 1), 0, [0.0], zeros(1, 1))
end
@testset "merge_states names the method a family still owes" begin
    pe = PortfolioOptimisers
    # The generic method is reached only by a pair no family answers. It refuses the pair that
    # cannot merge at all first, so a type mismatch reads as a type mismatch.
    terr = try
        pe.merge_states(PFSBare(1, [1.0], [0.0]), PFSOther(1))
        nothing
    catch e
        e
    end
    @test terr isa ArgumentError
    @test occursin("different types cannot be merged", terr.msg)
    # A mergeable pair whose family wrote no method names the method to write.
    merr = try
        pe.merge_states(PFSBare(1, [1.0], [0.0]), PFSBare(2, [2.0], [1.0]))
        nothing
    catch e
        e
    end
    @test merr isa ArgumentError
    @test occursin("PFSBare is an AbstractPartialFitState with no `merge_states` method",
                   merr.msg)
    @test occursin("merge_states(a::PFSBare, b::PFSBare)", merr.msg)
    # A family that implements the interface reaches its own method, and the generic one never
    # runs. The route is the one a shipped state takes: guard, family guard, fold.
    rng = StableRNG(24680)
    X = randn(rng, 20, 3)
    nA, muA, MA = batch_state(view(X, 1:8, :))
    nB, muB, MB = batch_state(view(X, 9:20, :))
    a = PFSFull(0.94, nA, muA, MA)
    b = PFSFull(0.94, nB, muB, MB)
    s = pe.merge_states(a, b)
    nX, muX, MX = batch_state(X)
    @test s isa PFSFull
    @test s.n == nX
    @test isapprox(s.mu, muX; rtol = 1e-12)
    @test isapprox(s.M, MX; rtol = 1e-12)
    # The decay refusal the family adds runs after the generic guard, so a decay mismatch over
    # one asset count is the message the family wrote.
    derr = try
        pe.merge_states(a, PFSFull(0.97, nB, muB, MB))
        nothing
    catch e
        e
    end
    @test derr isa ArgumentError
    @test occursin("different decay cannot be merged", derr.msg)
end
@testset "chan_merge is the merge in sums of Equation 2.1b, and is symmetric" begin
    pe = PortfolioOptimisers
    rng = StableRNG(13579)
    x = randn(rng, 25) .+ 3.0
    # The merge in sums of the Stanford report: S = S_A + S_B + n_A / (n_B n) (n_B / n_A T_A -
    # T_B)^2, with the block sums T. `chan_merge` writes the same term in means.
    nA, nB = 9, 16
    xa, xb = view(x, 1:nA), view(x, (nA + 1):(nA + nB))
    TA, TB = sum(xa), sum(xb)
    SA = sum(abs2, xa .- TA / nA)
    SB = sum(abs2, xb .- TB / nB)
    n, mu, M = pe.chan_merge(nA, TA / nA, SA, nB, TB / nB, SB)
    @test isapprox(M, SA + SB + nA / (nB * n) * (nB / nA * TA - TB)^2; rtol = 1e-14)
    @test isapprox(mu, (TA + TB) / n; rtol = 1e-14)
    # The two blocks enter the formula alike, so the order of the operands does not change the
    # state.
    X = randn(rng, 20, 3)
    a = batch_state(view(X, 1:6, :))
    b = batch_state(view(X, 7:20, :))
    ab = pe.chan_merge(a..., b...)
    ba = pe.chan_merge(b..., a...)
    @test ab[1] == ba[1]
    @test isapprox(ab[2], ba[2]; rtol = 1e-14)
    @test isapprox(ab[3], ba[3]; rtol = 1e-14)
    # One entry of the co-moment merge is Equation 3.1 of the Sandia report, the pairwise update
    # of the co-moment of two variables u and v.
    du = b[2][1] - a[2][1]
    dv = b[2][3] - a[2][3]
    @test isapprox(ab[3][1, 3], a[3][1, 3] + b[3][1, 3] + a[1] * b[1] / ab[1] * du * dv;
                   rtol = 1e-14)
end
@testset "A Welford fold stays inside the updating bound of Table 1" begin
    using Statistics
    pe = PortfolioOptimisers
    # The fixture of the partial-fit tests, shifted to a mean of 1000 and unit spread, so the
    # condition number of each column is about 1000.
    rng = StableRNG(987654321)
    X = randn(rng, 50, 4) .+ 1000.0
    N = size(X, 1)
    u = eps(Float64) / 2
    S = vec(sum(abs2, big.(X) .- mean(big.(X); dims = 1); dims = 1))
    exact = Float64.(S / (N - 1))
    # The condition number of Equation 2.1 of the American Statistician survey.
    kappa = Float64.(sqrt.(1 .+ vec(mean(big.(X); dims = 1)) .^ 2 .* N ./ S))
    @test all(>(500), kappa)
    fold(est) = foldl(partial_fit!, eachrow(X); init = est)
    relerr(v) = abs.(v .- exact) ./ exact
    @test all(relerr(var(fold(SimpleVariance()))) .<= N .* kappa .* u)
    C = cov(fold(Covariance()))
    @test all(relerr([C[i, i] for i in axes(C, 1)]) .<= N .* kappa .* u)
end
@testset "assert_partial_fit_state refuses each quantity that cannot describe a sample" begin
    pe = PortfolioOptimisers
    @test isnothing(pe.assert_partial_fit_state(0, [1.0, 2.0]))
    @test isnothing(pe.assert_partial_fit_state(0, [1.0, 2.0], zeros(2)))
    @test isnothing(pe.assert_partial_fit_state(3, [1.0, 2.0], zeros(2, 2)))
    @test_throws DomainError pe.assert_partial_fit_state(-1, [1.0])
    @test_throws IsEmptyError pe.assert_partial_fit_state(1, Float64[])
    @test_throws IsNonFiniteError pe.assert_partial_fit_state(1, [NaN])
    @test_throws IsNonFiniteError pe.assert_partial_fit_state(1, [1.0], [Inf])
    @test_throws DimensionMismatch pe.assert_partial_fit_state(1, [1.0, 2.0], zeros(3))
    @test_throws DimensionMismatch pe.assert_partial_fit_state(1, [1.0, 2.0], zeros(2, 3))
end
@testset "observation_count reads the folded rows, and names a state that has no count" begin
    using Statistics
    pe = PortfolioOptimisers
    rng = StableRNG(1)
    X = randn(rng, 10, 3)
    X[4, 2] = NaN
    # A row that holds a NaN is still a row of the sample, as `size(X, 1)` counts it.
    @test pe.observation_count(partial_fit!(Covariance(), X)) == 10
    @test pe.observation_count(partial_fit!(SimpleVariance(), X)) == 10
    @test pe.observation_count(partial_fit!(SimpleExpectedReturns(), X)) == 10
    @test_throws ArgumentError pe.observation_count(Covariance())
    # The two exponentially weighted covariance states count the rows in `rows`, because
    # `obs_count` counts each asset alone and a reset sets it to zero (#1851).
    @test pe.observation_count(partial_fit!(ExpWeightedCovariance(), X)) == 10
    @test pe.observation_count(partial_fit!(RegimeAdjustedExpWeightedCovariance(), X)) == 10
    act = trues(10, 3)
    act[5:6, 1] .= false
    ew = partial_fit!(ExpWeightedCovariance(), X; active_mask = act)
    @test pe.observation_count(ew) == 10
    # A fold of the copy does not count in the original.
    st = copy(ew.cache)
    @test st.rows !== ew.cache.rows
    ew2 = partial_fit!(ExpWeightedCovariance(; cache = st), X)
    @test pe.observation_count(ew2) == 20
    @test pe.observation_count(ew) == 10
    ra = partial_fit!(RegimeAdjustedExpWeightedCovariance(), X)
    @test copy(ra.cache).rows !== ra.cache.rows
    # A state that records no count still meets the named refusal.
    err = try
        pe.observation_count(partial_fit!(ExpWeightedVariance(), X))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("`ExpWeightedVarianceState` records no count", err.msg)
end
@testset "A folded PortfolioOptimisersCovariance reads the count of its inner state (#1851)" begin
    using Statistics
    pe = PortfolioOptimisers
    X = randn(StableRNG(1), 120, 3)
    dn = MatrixProcessing(; dn = Denoise())
    # The default `mp` repairs alone and reads no count. A denoiser reads `T = size(X, 1)` on
    # the batch route, and the fold reads the same rows.
    for ce in (ExpWeightedCovariance(), RegimeAdjustedExpWeightedCovariance(),
               GeodesicShrinkageCovariance(; ce = ExpWeightedCovariance())),
        mp in (MatrixProcessing(), dn)

        pcb = PortfolioOptimisersCovariance(; ce = ce, mp = mp)
        pc = partial_fit!(pcb, X)
        @test pe.observation_count(pc) == 120
        @test cov(pc) ≈ cov(pcb, X)
        @test cor(pc) ≈ cor(pcb, X)
    end
    # A seed window folds the rows that the window kept. The default `mp` reads no count, so
    # the fold equals the batch fit.
    sw = WindowedCovariance(; window = 50, rule = SeedWindow(), ce = Covariance())
    pc = partial_fit!(PortfolioOptimisersCovariance(; ce = sw), X)
    @test pe.observation_count(pc) == 50
    @test cov(pc) ≈ cov(PortfolioOptimisersCovariance(; ce = sw), X)
    @test cor(pc) ≈ cor(PortfolioOptimisersCovariance(; ce = sw), X)
    # `cor` and `std` with no data read the inner estimator too, as `cov` and `var` do.
    @test cor(pc.ce) ≈ cor(sw, X)
    wv = WindowedVariance(; window = 50, rule = SeedWindow())
    @test std(partial_fit!(wv, X)) ≈ vec(std(wv, X))
    pc = partial_fit!(pc, X[1:7, :])
    @test pe.observation_count(pc) == 57
    # The shape route reads no count where no step reads one, and a denoiser refuses a
    # missing count by name.
    @test !pe.matrix_processing_reads_count(MatrixProcessing())
    @test pe.matrix_processing_reads_count(dn)
    sigma = cov(X)
    @test pe.matrix_processing_block!(MatrixProcessing(), copy(sigma), nothing, 3) ≈
          pe.matrix_processing_block!(MatrixProcessing(), copy(sigma), 120, 3)
    @test_throws ArgumentError pe.matrix_processing_block!(dn, copy(sigma), nothing, 3)
end
@testset "A denoiser of a weighted moment reads the effective sample size (#1892)" begin
    using Statistics
    pe = PortfolioOptimisers
    rng = StableRNG(7)
    X = randn(rng, 200, 3) * randn(rng, 3, 25) .* 0.01 .+ randn(rng, 200, 25) .* 0.01
    T, N = size(X)
    dn = MatrixProcessing(; dn = Denoise())
    w = StatsBase.pweights(collect(range(0.2, 1.0; length = T)))
    kish = sum(w)^2 / sum(abs2, w)
    ew = ExpWeightedCovariance(; decay = 0.97)
    # The Kish count of the weights `0.97^(T - t)`. The estimated centring spends one
    # observation, as `exp_weighted_variance_count` states.
    lw = [0.97^k for k in 0:(T - 2)]
    Tew = sum(lw)^2 / sum(abs2, lw) + 1
    # The counts. An unweighted estimator keeps `size(X, 1)`.
    @test pe.effective_observation_count(Covariance(), X) === T
    @test pe.effective_observation_count(ew, X) ≈ Tew
    @test pe.effective_observation_count(Covariance(; w = w), X) ≈ kish
    @test pe.effective_observation_count(Covariance(; ce = GeneralCovariance(; w = w)), X) ≈
          kish
    @test pe.effective_observation_count(Covariance(; w = StatsBase.fweights(fill(2, T))),
                                         X) == 2T
    @test pe.effective_observation_count(DistanceCovariance(; w = w), X) ≈ kish
    @test pe.effective_observation_count(RegimeAdjustedExpWeightedCovariance(;
                                                                             decay = 0.97),
                                         X) ≈ Tew
    # A separate correlation decay reads the pre-centred law of the correlation.
    lc = [0.99^k for k in 0:(T - 1)]
    @test pe.effective_observation_count(RegimeAdjustedExpWeightedCovariance(; decay = 0.97,
                                                                             cor_decay = 0.99),
                                         X) ≈ sum(lc)^2 / sum(abs2, lc)
    wc = WindowedCovariance(; ce = ew, window = 50)
    @test pe.effective_observation_count(wc, X) ≈
          pe.effective_observation_count(ew, X[151:end, :])
    for ce in
        (PortfolioOptimisersCovariance(; ce = PortfolioOptimisersCovariance(; ce = ew)),
         CorrelationCovariance(; ce = ew), GeodesicShrinkageCovariance(; ce = ew),
         ImpliedVolatility(; ce = ew, ppy = 252))
        @test pe.effective_observation_count(ce, X) ≈ Tew
    end
    # The batch route denoises at the ratio of the count, and a weighted moment moves.
    S = Matrix(cov(ew, X))
    ref = pe.denoise!(Denoise(), pe.posdef!(Posdef(), copy(S)), Tew / N)
    pc = PortfolioOptimisersCovariance(; ce = ew, mp = dn)
    @test cov(pc, X) ≈ ref
    @test !isapprox(cov(pc, X),
                    pe.denoise!(Denoise(), pe.posdef!(Posdef(), copy(S)), T / N))
    @test cor(pc, X) ≈
          pe.denoise!(Denoise(), pe.posdef!(Posdef(), Matrix(cor(ew, X))), Tew / N)
    @test cov(pc, X, nothing) ≈ ref
    # The fold reads the same count as the batch route.
    for ce in (ew, RegimeAdjustedExpWeightedCovariance(; decay = 0.97, cor_decay = 0.99),
               WindowedCovariance(; ce = ew, window = 50, rule = SeedWindow()),
               pe.update_online_estimator(Online(wc)))
        pcb = PortfolioOptimisersCovariance(; ce = ce, mp = dn)
        pcf = partial_fit!(pcb, X)
        @test pe.effective_observation_count(pcf) ≈ pe.processing_observation_count(pcb, X)
        @test cov(pcf) ≈ cov(pcb, X)
        @test cor(pcf) ≈ cor(pcb, X)
    end
    # A weighted cokurtosis and a weighted coskewness denoise at the count of their weights.
    X6 = X[:, 1:6]
    Y = X6 .- mean(X6, w; dims = 1)
    o = transpose(ones(6))
    z = kron(o, Y) .* kron(Y, o)
    kt = transpose(w .* z) * z / sum(w)
    @test cokurtosis(Cokurtosis(; w = w, mp = dn), X6) ≈
          pe.denoise!(Denoise(), pe.posdef!(Posdef(), kt), kish / 6)
    cs, V = coskewness(Coskewness(; w = w, mp = dn), X6)
    @test V ≈ pe.negative_spectral_coskewness(cs, X6, dn, kish)
    # A matrix processing estimator of the caller states only the sample form, so it gets `X`.
    sigma = cov(X)
    @test pe.sample_matrix_processing!(nothing, copy(sigma), X, kish) == sigma
    @test pe.matrix_processing_block!(dn, copy(sigma), X) ≈
          matrix_processing!(dn, copy(sigma), X)
    @test pe.matrix_processing_block!(dn, copy(sigma), X, kish) ≈
          pe.matrix_processing_block!(dn, copy(sigma), kish, N)
end
@testset "A rolling window has no fold, and a seed window folds" begin
    pe = PortfolioOptimisers
    rng = StableRNG(2)
    X = randn(rng, 12, 2)
    @test_throws ArgumentError partial_fit!(WindowedVariance(; window = 5,
                                                             rule = RollingWindow()), X)
    sw = partial_fit!(WindowedVariance(; window = 5, rule = SeedWindow()), X)
    @test sw isa WindowedVariance
    # The declaration of each windowed type sets the inner estimator as the first field.
    @test isnothing(pe.obs_weights_view(getfield(sw, 1).cache, 1:3))
end
