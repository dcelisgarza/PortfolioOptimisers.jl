#=
The condition 2 and 3 sweep of `src/01_Base/` (#439, child map 1 of #404).

Every claim below is computed and compared, never read. The file's other units are checked
where they are used: the equation and resource caps, the preference channel and the
suggestion threshold in `test_02_equation_parsing.jl`, and `failed_solve_msg` in
`test_01_structs.jl`. What is left is the compact-show contract, the `ScopedConfig` restore
path, the message builders, the iteration protocol and the norm seam.
=#
using PortfolioOptimisers, Test, Clustering, JuMP, StableRNGs, StatsBase
using PortfolioOptimisers: @define_pretty_show, compact_show_budget,
                           pretty_show_vector_summary, pretty_show_vector_element,
                           pretty_show_vector_body
import PortfolioOptimisers: has_pretty_show_method

# `@define_pretty_show` escapes its whole body, so a caller outside the module must bring the
# five names the body uses into scope, and must `import` the one it adds a method to. These
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

# A `DynamicAbstractWeights` with no `get_observation_weights` method of its own. The whole
# point of the type is that the unimplemented shape raises rather than computing unweighted.
struct NoWeightsProbe <: PortfolioOptimisers.DynamicAbstractWeights end

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
    # The full rendering, one line per field, each field through its own branch.
    @test render(io -> show(io, probe)) ==
          join(["$(PROBE)", "  leaf ┼ $(LEAF)", "       │   a ┴ Int64: 1",
                "    rs ┼ 2-element Vector{$(LEAF)}", "       │ $(LEAF) ⋯",
                "       │ $(LEAF) ⋯", "    dt ┼ DataType: Float64", "     n ┴ nothing", ""],
               '\n')
    # A `DataType` field prints its type and the wrapper name of the value, so a parametrised
    # type reports the wrapper it instantiates.
    @test occursin("dt ┼ DataType: Array",
                   render(io -> show(io,
                                     PSProbe(PSLeaf(1), [PSLeaf(2)], Vector{Float64},
                                             nothing))))
    # `:po_compact` reaches the nested buffer, so the budget applies at every depth. At a
    # budget of one line the nested leaf collapses to `Name ⋯` and the vector loses its tail.
    @test render(io -> show(IOContext(io, :po_compact => 1), probe)) ==
          join(["$(PROBE)", "  leaf ┼ $(LEAF) ⋯", "    rs ┼ 2-element Vector{$(LEAF)}",
                "       │ $(LEAF) ⋯", "       │ ⋮", "    dt ┼ DataType: Float64",
                "     n ┴ nothing", ""], '\n')
    # A budget the rendering fits under changes nothing.
    @test render(io -> show(IOContext(io, :po_compact => 3), probe)) ==
          render(io -> show(io, probe))
    # The macro reads the value with `getproperty`, so it is the flattened surface of
    # `@forward_properties` that stays out of `show`, not the swapped value of a real field.
    @test length(fieldnames(PSProbe)) == 4
    @test all(f -> hasfield(PSProbe, f), fieldnames(PSProbe))
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
    @test_throws ArgumentError pe.strict_diagnostic("term dropped", true)
    err = try
        pe.strict_diagnostic("term dropped", true)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test err.msg == "term dropped"
    @test_logs (:warn, "term dropped") pe.strict_diagnostic("term dropped", false)
    @test isnothing(@test_logs (:warn, "term dropped") pe.strict_diagnostic("term dropped",
                                                                            false))
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
    # The guard builds `DomainError(text)`, so the text is the `val` field, not the `msg`.
    @test derr.msg == ""
    @test occursin("all(x -> 0 < x, values(n_sim)) must hold", derr.val)
    perr = try
        pe.assert_gt0(:a => 0.0, :alpha)
        nothing
    catch e
        e
    end
    @test perr isa DomainError
    @test occursin("0 < alpha[2] must hold", perr.val)
    # The composed guards take a value of any of the five shapes, and their varargs method
    # accepts everything else without a check, which is how an absent value passes through.
    @test isnothing(pe.assert_nonempty_gt0_finite_val([1.0, 2.0]))
    @test_throws DomainError pe.assert_nonempty_gt0_finite_val([1.0, 0.0])
    @test isnothing(pe.assert_nonempty_gt0_finite_val(nothing))
    @test isnothing(pe.assert_nonempty_nonneg_finite_val(nothing))
    @test isnothing(pe.assert_nonempty_finite_val(nothing))
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
    # The guards build `DomainError(text)`, so the text is the `val` field, not the `msg`.
    @test oerr.val == "0 < alpha < 1 must hold. Got\nalpha => 1.0"
    @test cerr.val == "0 <= n7 <= 1 must hold. Got\nn7 => 1.5"
    # Both carry the varargs method that checks nothing, which is how a slot holding a
    # Calibration Role passes a guard written for a number.
    role = SignificanceTailCalibration(; alg = ScenarioCount(; n = 5))
    @test isnothing(pe.assert_unit_interval(nothing))
    @test isnothing(pe.assert_closed_unit_interval(nothing))
    @test isnothing(pe.assert_closed_unit_interval(role, :n))
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
