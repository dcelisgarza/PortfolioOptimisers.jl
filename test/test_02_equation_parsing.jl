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
