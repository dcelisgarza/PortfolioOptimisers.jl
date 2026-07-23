@testset "Equation tests" begin
    using PortfolioOptimisers, Test, Logging
    # Scope warning suppression to this task. `Logging.disable_logging` mutates a
    # process-global level that never restores and leaks into other test files sharing
    # a reused worker (see ParallelTestRunner worker pool), randomly emptying their
    # `@test_logs` captures. `with_logger` is task-local and restores on exit.
    with_logger(SimpleLogger(stderr, Logging.Error)) do
        res = parse_equation("2*sqrt(prior(a ,   1b1)) /2*  5 + cbrt(3)^3*f >= 5/5 + d-69/3*c")
        idx = sortperm(res.vars)
        @test res.vars[idx] == ["d", "f", "c", "sqrt(prior(a, 1b1))"][idx]
        @test res.coef[idx] == [-1.0, 3.0, 23.0, 5.0][idx]
        @test res.op == ">="
        @test res.rhs == 1.0
        res = parse_equation("2*sqrt(prior(a ,   b)) /2*  5 - - - -6 + cbrt(3)^3*f  -69/3*c <= - d")
        idx = sortperm(res.vars)
        @test res.vars[idx] == ["sqrt(prior(a, b))", "f", "c", "d"][idx]
        @test res.coef[idx] == [5.0, 3.0, -23.0, 1.0][idx]
        @test res.op == "<="
        @test res.rhs == -6.0
        res = parse_equation("0 == 2*sqrt(prior(a ,   b)) /2*  5 - 6 + 7 + -  - cbrt(3)^3 *f  - - 69/ 3*c-d+(3*2)/(3*(1+1))*d")
        idx = sortperm(res.vars)
        @test res.vars[idx] == ["sqrt(prior(a, b))", "f", "c", "d"][idx]
        @test res.coef[idx] == [-5.0, -3.0, -23.0, 0.0][idx]
        @test res.op == "=="
        @test res.rhs == 1.0
        # An empty side fails closed: assuming zero would silently create a constraint
        # the author never wrote (e.g. a truncated config/spreadsheet row).
        @test_throws Meta.ParseError parse_equation("w_A >= ")
        @test_throws Meta.ParseError parse_equation(" == 2*w_A + 1")
        @test_throws Meta.ParseError parse_equation("2*sqrt(prior(a ,   b)) /2*  5 + cbrt(3)^3*f >= 5/")
        @test_throws Meta.ParseError parse_equation("2*sqrt(prior(a ,   b))   5 + cbrt(3)^3*f + 1/>= 5/5 ")
        @test_throws Meta.ParseError parse_equation("2*sqrt(prior(a ,   b))   5 + cbrt(3)^3*f + 1/ = 5/5 ")
        @test_throws Meta.ParseError parse_equation("2*sqrt(prior(a ,   b)) /2*  5 ++ cbrt(3)^3*f >= 5/5 + d-69/3*c")
        @test_throws Meta.ParseError parse_equation(:(2 * sqrt(prior(a, b)) / 2 * 5 ++
                                                      cbrt(3)^3 * f >=
                                                      5 / 5 + d - 69 / 3 * c))
        @test_throws Meta.ParseError parse_equation("2*sqrt(prior(a ,   b)) /2*  5 + cbrt(3)^3*f >= >= 5/5 + d-69/3*c")
        @test_throws Meta.ParseError parse_equation(:(2 * sqrt(prior(a, b)) / 2 * 5 >=
                                                      cbrt(3)^3 * f >=
                                                      5 / 5 + d - 69 / 3 * c))
        @test_throws Meta.ParseError parse_equation(:(2 * sqrt(prior(a, b)) / 2 * 5 +
                                                      cbrt(3)^3 * f +
                                                      5 / 5 +
                                                      d - 69 / 3 * c))
        res = parse_equation("1/3*x/((2^z)*y)+5z==1-5")
        idx = sortperm(res.vars)
        @test res.vars[idx] == ["z", "(0.3333333333333333x) / (2 ^ z * y)"][idx]
        @test res.coef[idx] == [5.0, 1.0][idx]
        @test res.op == "=="
        @test res.rhs == -4.0
        # @test res.eqn == "5.0*z + (0.3333333333333333x) / (2 ^ z * y) == -4.0"
        res = parse_equation("-x>=-1.0y+2")
        idx = sortperm(res.vars)
        @test res.vars[idx] == ["x", "y"][idx]
        @test res.coef[idx] == [-1.0, 1.0][idx]
        @test res.op == ">="
        @test res.rhs == 2.0
        # @test res.eqn == "-x + y >= 2.0"
        res = parse_equation("-_1*_3<=-1.1y+2")
        idx = sortperm(res.vars)
        @test res.vars[idx] == ["-_1 * _3", "y"][idx]
        @test res.coef[idx] == [1.0, 1.1][idx]
        @test res.op == "<="
        @test res.rhs == 2.0
        # @test res.eqn == "-_1 * _3 + 1.1*y <= 2.0"
        res = parse_equation("Inf*a<=Inf")
        idx = sortperm(res.vars)
        @test res.vars[idx] == ["a"][idx]
        @test res.coef[idx] == [Inf][idx]
        @test res.op == "<="
        @test res.rhs == Inf
        # @test res.eqn == "Inf*a <= Inf"
        # F3: numeric literals evaluate in the target float type, not machine Int64, so an
        # overflowing exponent yields the correct float instead of silently wrapping, and a
        # negative integer exponent no longer raises a raw DomainError.
        res = parse_equation("w_A <= 2^64")
        @test res.vars == ["w_A"]
        @test res.coef == [1.0]
        @test res.rhs == 1.8446744073709552e19
        res = parse_equation("w_A <= 10^19")
        @test res.rhs == 1.0e19
        res = parse_equation("w_A <= 2^-1")
        @test res.rhs == 0.5
        # F3: coercion respects a non-default datatype.
        res = parse_equation("w_A <= 2^64"; datatype = Float32)
        @test res.rhs == Float32(2)^64
        # F2: only the enumerated math functions may be evaluated. `prior(...)` reaching numeric
        # evaluation (all-numeric args) is a typed parse error, not a raw UndefVarError, and a
        # name absent from the table fails closed rather than resolving against Base.
        @test_throws Meta.ParseError parse_equation("w_A <= prior(2)")
        @test_throws Meta.ParseError parse_equation("w_A <= run(2)")
        @test_throws Meta.ParseError parse_equation("w_A <= gensym(2)")
        # F2: `prior(name)` and `sqrt(prior(name))` still pass through structurally.
        res = parse_equation("w_A <= prior(b)")
        @test "prior(b)" in res.vars
    end
end
@testset "Equation parser recursion caps" begin
    using PortfolioOptimisers, Test, Logging
    with_logger(SimpleLogger(stderr, Logging.Error)) do
        pe = PortfolioOptimisers
        # Trust boundary: an over-long / deeply nested untrusted string fails closed with a
        # typed Meta.ParseError before Meta.parse and the recursive walks can exhaust the stack.
        deep = "(" ^ 20000 * "w_A" * ")" ^ 20000 * " <= 1"
        @test length(deep) > pe.EQUATION_LIMITS[].max_length
        @test_throws Meta.ParseError parse_equation(deep)
        # The Expr form has no length cap; the depth guard rejects an over-deep pre-built AST.
        ex = :w_A
        for _ in 1:(pe.EQUATION_LIMITS[].max_depth + 10)
            ex = Expr(:call, :-, ex)
        end
        @test_throws Meta.ParseError parse_equation(Expr(:call, :(<=), ex, 1))
        # The global default is runtime-settable via the ScopedConfig setter, and validated.
        @test_throws ArgumentError pe.set_equation_limits!(max_length = 0)
        @test_throws ArgumentError pe.set_equation_limits!(max_depth = -1)
        pe.set_equation_limits!(max_length = 8)
        @test_throws Meta.ParseError parse_equation("w_A <= 1.0")   # now under the tightened cap
        pe.set_equation_limits!(max_length = 4096, max_depth = 256)  # restore defaults
        # Task-scoped override: tightened inside the block (inherited by child tasks),
        # restored automatically on exit, global default untouched.
        pe.with_equation_limits(; max_length = 8) do
            @test pe.EQUATION_LIMITS[].max_length == 8
            @test_throws Meta.ParseError parse_equation("w_A <= 1.0")
            @test fetch(@async pe.EQUATION_LIMITS[].max_length) == 8
        end
        @test pe.EQUATION_LIMITS[].max_length == 4096
        # A legitimate constraint still parses under the default caps.
        res = parse_equation("w_A <= 1.0")
        @test res.vars == ["w_A"]
        @test res.rhs == 1.0
    end
end
@testset "Asset name suggestions" begin
    using PortfolioOptimisers, Test
    pe = PortfolioOptimisers
    nx = ["AAPL", "MSFT", "GOOG"]
    # Close typo -> suggestion; far-off name (meta-opt legit-absent) -> none; empty -> none.
    @test occursin("did you mean `AAPL`?", pe.did_you_mean("APL", nx))
    @test pe.did_you_mean("ZZZZ", nx) == ""
    @test pe.did_you_mean("APL", String[]) == ""
    # Message builders: name + count + key only (no full universe), suggestion appended to msg1.
    m1 = pe.unknown_variable_msg("APL", nx, "nx")
    @test occursin("3 assets under key `nx`", m1)
    @test occursin("did you mean `AAPL`?", m1)
    @test !occursin("GOOG", m1)              # universe not dumped
    # All-zero-row message: no suggestion, noun switches for views.
    @test occursin("constraint `APL >= 0.05` matched no assets",
                   pe.empty_row_msg("APL >= 0.05", nx, "nx"))
    @test occursin("view `", pe.empty_row_msg("APL == 0.02", nx, "nx"; noun = "view"))
    @test !occursin("did you mean", pe.empty_row_msg("APL >= 0.05", nx, "nx"))
    # Global default: threshold gates, metric is swappable (ScopedConfig setter).
    pe.set_string_distance!(min_score = 1.1)
    @test pe.did_you_mean("APL", nx) == ""   # suggestions disabled
    pe.set_string_distance!(dist = pe.StringDistances.DamerauLevenshtein(), min_score = 0.7)
    @test occursin("did you mean `MSFT`?", pe.did_you_mean("MSTF", nx))  # transposition
    pe.set_string_distance!(dist = pe.StringDistances.Levenshtein(), min_score = 0.7)  # restore defaults
    # Task-scoped override (e.g. silencing suggestions inside a meta-optimiser loop):
    # active inside the block, restored on exit, global default untouched.
    pe.with_string_distance(; min_score = 1.1) do
        @test pe.did_you_mean("APL", nx) == ""
    end
    @test occursin("did you mean `AAPL`?", pe.did_you_mean("APL", nx))
end
@testset "ScopedConfig preferences fail closed" begin
    using PortfolioOptimisers, Test
    pe = PortfolioOptimisers
    # Valid preference values route through the set_*! setters (same validation as runtime).
    pe.apply_preferences!(Dict{String, Any}("equation_max_length" => 512,
                                            "suggestion_distance" => "damerau_levenshtein",
                                            "suggestion_min_score" => 0.8,
                                            "compact_show" => 4))
    @test pe.EQUATION_LIMITS[].max_length == 512
    @test pe.EQUATION_LIMITS[].max_depth == 256          # unset key keeps its default
    @test pe.STRING_DISTANCE[].dist isa pe.StringDistances.DamerauLevenshtein
    @test pe.STRING_DISTANCE[].min_score == 0.8
    @test pe.COMPACT_SHOW[] == 4
    # Unset preferences (nothing) are skipped entirely.
    pe.apply_preferences!(Dict{String, Any}())
    @test pe.EQUATION_LIMITS[].max_length == 512
    # Invalid values throw a typed error naming the key — the package refuses to load
    # rather than silently running with a weaker cap than the project requested.
    @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}("equation_max_length" =>
                                                                           -5))
    @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}("equation_max_length" => "512"))
    @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}("equation_max_depth" =>
                                                                           true))
    @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}("suggestion_min_score" =>
                                                                           true))
    @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}("compact_show" => "yes"))
    # Unknown distance name fails closed against the enumerated allowlist, with a suggestion.
    err = try
        pe.apply_preferences!(Dict{String, Any}("suggestion_distance" => "levenstein"))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("did you mean `levenshtein`?", err.msg)
    # Scoped display override, then restore all shipped defaults.
    pe.with_compact_show(false) do
        @test pe.COMPACT_SHOW[] === false
    end
    pe.set_equation_limits!(max_length = 4096, max_depth = 256)
    pe.set_string_distance!(dist = pe.StringDistances.Levenshtein(), min_score = 0.7)
    pe.set_compact_show!(true)
end
@testset "Suggestion threshold rejects a non-positive min_score" begin
    using PortfolioOptimisers, Test
    pe = PortfolioOptimisers
    # A zero or negative threshold would make did_you_mean echo a real asset name for any
    # probe, turning the fuzzy hint into a universe-name oracle. The inner constructor is
    # the single choke point, so every route in fails closed.
    @test_throws ArgumentError pe.StringDistanceConfig(pe.StringDistances.Levenshtein(),
                                                       -0.5)
    @test_throws ArgumentError pe.set_string_distance!(min_score = -0.5)
    @test_throws ArgumentError pe.with_string_distance(() -> nothing; min_score = -1)
    @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}("suggestion_min_score" =>
                                                                           -0.5))
    # NaN is rejected too (NaN >= 0 is false).
    @test_throws ArgumentError pe.StringDistanceConfig(pe.StringDistances.Levenshtein(),
                                                       NaN)
    # 0 is out, but the documented ">1 disables suggestions" sentinel stays legal.
    @test_throws ArgumentError pe.StringDistanceConfig(pe.StringDistances.Levenshtein(), 0)
    @test pe.StringDistanceConfig(pe.StringDistances.Levenshtein(), 1.5).min_score == 1.5
    nx = ["AAPL", "MSFT", "GOOG"]
    pe.with_string_distance(; min_score = 1.5) do
        @test pe.did_you_mean("APL", nx) == ""
    end
    @test pe.STRING_DISTANCE[].min_score == 0.7  # scoped override restored
end
@testset "Resource caps fail closed" begin
    using PortfolioOptimisers, Test
    pe = PortfolioOptimisers
    # One cap per sink (ADR 0041); the keyword constructor is the safe path since the four
    # fields are same-typed and two share the value 100_000.
    @test pe.RESOURCE_LIMITS[] == pe.ResourceLimits(1_000_000, 100_000, 100_000, 10_000)
    @test pe.ResourceLimits() == pe.ResourceLimits(1_000_000, 100_000, 100_000, 10_000)
    @test pe.ResourceLimits(; max_bins = 500).max_bins == 500
    # Every field must be positive, like EquationLimits.
    @test_throws ArgumentError pe.ResourceLimits(; max_n_sim = 0)
    @test_throws ArgumentError pe.ResourceLimits(; max_n_subsets = -1)
    @test_throws ArgumentError pe.ResourceLimits(; max_frontier = 0)
    @test_throws ArgumentError pe.ResourceLimits(; max_bins = -1)
    @test_throws ArgumentError pe.set_resource_limits!(max_n_sim = 0)
    @test_throws ArgumentError pe.set_resource_limits!(max_n_subsets = -1)
    @test_throws ArgumentError pe.set_resource_limits!(max_frontier = 0)
    @test_throws ArgumentError pe.set_resource_limits!(max_bins = -1)
    # n_sim sizes an N^2 * n_sim array in both uncertainty-set estimators; the ceiling
    # converts an OOM kill into a typed DomainError naming the knob that raises it.
    @test_throws DomainError NormalUncertaintySet(; n_sim = 10_000_000_000)
    @test_throws DomainError ARCHUncertaintySet(; n_sim = 10_000_000_000)
    err = try
        NormalUncertaintySet(; n_sim = 10_000_000_000)
        nothing
    catch e
        e
    end
    @test err isa DomainError
    @test occursin("max_n_sim", err.msg)
    @test occursin("set_resource_limits!", err.msg)
    # The shipped default is far below the cap and still constructs.
    @test NormalUncertaintySet(; n_sim = 3_000).n_sim == 3_000
    # n_subsets is resolved at optimise time (it may be a schedule or callable), so the
    # cap lives at that resolution seam and covers both the integer and callable forms.
    rd = ReturnsResult(; nx = ["A", "B", "C"], X = randn(60, 3))
    @test_throws DomainError pe.get_n_subsets(10_000_000)
    @test_throws DomainError pe.get_n_subsets(x -> 10_000_000, rd)
    @test pe.get_n_subsets(50) == 50
    @test pe.get_n_subsets(x -> 7, rd) == 7
    # Frontier.N drives one full inner solve per point: a compute sink capped like n_subsets.
    @test_throws DomainError Frontier(; N = 10_000_000)
    ferr = try
        Frontier(; N = 10_000_000)
        nothing
    catch e
        e
    end
    @test ferr isa DomainError
    @test occursin("max_frontier", ferr.msg)
    @test Frontier(; N = 20).N == 20
    # bins drives a bins x bins joint histogram: a quadratic memory sink with its own cap,
    # shared by MutualInfoCovariance and VariationInfoDistance. Auto-selecting bin
    # algorithms are data-bounded and uncapped.
    @test_throws DomainError MutualInfoCovariance(; bins = 10_000_000)
    @test_throws DomainError pe.VariationInfoDistance(; bins = 10_000_000)
    berr = try
        MutualInfoCovariance(; bins = 10_000_000)
        nothing
    catch e
        e
    end
    @test berr isa DomainError
    @test occursin("max_bins", berr.msg)
    @test MutualInfoCovariance(; bins = 20).bins == 20
    @test MutualInfoCovariance().bins isa pe.AbstractBins  # HacineGharbiRavier(), uncapped
    # A raised ceiling is honoured, and scoped overrides restore on exit.
    pe.with_resource_limits(; max_n_sim = 20_000_000_000) do
        @test NormalUncertaintySet(; n_sim = 10_000_000_000).n_sim == 10_000_000_000
    end
    pe.with_resource_limits(; max_frontier = 20_000_000) do
        @test Frontier(; N = 10_000_000).N == 10_000_000
    end
    pe.with_resource_limits(; max_bins = 20_000_000) do
        @test MutualInfoCovariance(; bins = 10_000_000).bins == 10_000_000
    end
    @test pe.RESOURCE_LIMITS[].max_n_sim == 1_000_000
    # Preferences fail closed at load, like the equation caps.
    pe.apply_preferences!(Dict{String, Any}("max_n_subsets" => 1_234, "max_bins" => 321))
    @test pe.RESOURCE_LIMITS[].max_n_subsets == 1_234
    @test pe.RESOURCE_LIMITS[].max_bins == 321
    @test pe.RESOURCE_LIMITS[].max_n_sim == 1_000_000  # unset key keeps its default
    @test pe.RESOURCE_LIMITS[].max_frontier == 100_000  # unset key keeps its default
    @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}("max_n_sim" => -5))
    @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}("max_n_subsets" =>
                                                                           true))
    @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}("max_frontier" => "1000"))
    @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}("max_bins" => 1.5))
    pe.set_resource_limits!(max_n_sim = 1_000_000, max_n_subsets = 100_000,
                            max_frontier = 100_000, max_bins = 10_000)  # restore
end
