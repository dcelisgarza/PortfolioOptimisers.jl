@testset "Equation tests" begin
    using PortfolioOptimisers, Test, Logging
    # Scope warning suppression to this task. `Logging.disable_logging` mutates a
    # process-global level that never restores and leaks into other test files sharing
    # a reused worker (see ParallelTestRunner worker pool), randomly emptying their
    # `@test_logs` captures. `with_logger` is task-local and restores on exit.
    with_logger(SimpleLogger(stderr, Logging.Error)) do
        # `parse_equation` collects its terms in a `Dict`, so the order of `vars` and of
        # `coef` is the dictionary's and is no part of the answer. Each comparison below
        # sorts BOTH sides by their own names: `idx` orders the result and `jdx` orders
        # the literal. One permutation applied to both sides compares two different
        # orderings, and passes only while the dictionary happens to iterate in the order
        # the literal was written, which it did until Julia 1.13 rehashed it.
        res = parse_equation("2*sqrt(prior(a ,   1b1)) /2*  5 + cbrt(3)^3*f >= 5/5 + d-69/3*c")
        idx = sortperm(res.vars)
        jdx = sortperm(["d", "f", "c", "sqrt(prior(a, 1b1))"])
        @test res.vars[idx] == ["d", "f", "c", "sqrt(prior(a, 1b1))"][jdx]
        @test res.coef[idx] == [-1.0, 3.0, 23.0, 5.0][jdx]
        @test res.op == ">="
        @test res.rhs == 1.0
        res = parse_equation("2*sqrt(prior(a ,   b)) /2*  5 - - - -6 + cbrt(3)^3*f  -69/3*c <= - d")
        idx = sortperm(res.vars)
        jdx = sortperm(["sqrt(prior(a, b))", "f", "c", "d"])
        @test res.vars[idx] == ["sqrt(prior(a, b))", "f", "c", "d"][jdx]
        @test res.coef[idx] == [5.0, 3.0, -23.0, 1.0][jdx]
        @test res.op == "<="
        @test res.rhs == -6.0
        res = parse_equation("0 == 2*sqrt(prior(a ,   b)) /2*  5 - 6 + 7 + -  - cbrt(3)^3 *f  - - 69/ 3*c-d+(3*2)/(3*(1+1))*d")
        idx = sortperm(res.vars)
        jdx = sortperm(["sqrt(prior(a, b))", "f", "c", "d"])
        @test res.vars[idx] == ["sqrt(prior(a, b))", "f", "c", "d"][jdx]
        @test res.coef[idx] == [-5.0, -3.0, -23.0, 0.0][jdx]
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
        jdx = sortperm(["z", "(0.3333333333333333x) / (2 ^ z * y)"])
        @test res.vars[idx] == ["z", "(0.3333333333333333x) / (2 ^ z * y)"][jdx]
        @test res.coef[idx] == [5.0, 1.0][jdx]
        @test res.op == "=="
        @test res.rhs == -4.0
        # @test res.eqn == "5.0*z + (0.3333333333333333x) / (2 ^ z * y) == -4.0"
        res = parse_equation("-x>=-1.0y+2")
        idx = sortperm(res.vars)
        jdx = sortperm(["x", "y"])
        @test res.vars[idx] == ["x", "y"][jdx]
        @test res.coef[idx] == [-1.0, 1.0][jdx]
        @test res.op == ">="
        @test res.rhs == 2.0
        # @test res.eqn == "-x + y >= 2.0"
        res = parse_equation("-_1*_3<=-1.1y+2")
        idx = sortperm(res.vars)
        jdx = sortperm(["-_1 * _3", "y"])
        @test res.vars[idx] == ["-_1 * _3", "y"][jdx]
        @test res.coef[idx] == [1.0, 1.1][jdx]
        @test res.op == "<="
        @test res.rhs == 2.0
        # @test res.eqn == "-_1 * _3 + 1.1*y <= 2.0"
        res = parse_equation("Inf*a<=Inf")
        idx = sortperm(res.vars)
        jdx = sortperm(["a"])
        @test res.vars[idx] == ["a"][jdx]
        @test res.coef[idx] == [Inf][jdx]
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
        # The string form is held to the same depth cap: a string that clears the length
        # cap can still carry a tree deeper than max_depth, so the cap is read off the
        # parsed tree rather than inferred from the character count.
        over = pe.EQUATION_LIMITS[].max_depth + 10
        deep_str = "-("^over * "w_A" * ")"^over * " <= 1"
        @test length(deep_str) <= pe.EQUATION_LIMITS[].max_length
        @test_throws "too deeply nested" parse_equation(deep_str)
        # A tree under the cap still parses, so the bound is the stated number.
        under = pe.EQUATION_LIMITS[].max_depth - 2
        @test parse_equation("-("^under * "w_A" * ")"^under * " <= 1") isa
              PortfolioOptimisers.ParsingResult
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
    # All-zero-row message: no suggestion, noun switches for views. The row reached this
    # branch with every name resolved -- one that did not took the row with it -- so the
    # text says the row cancelled, never that a name missed the universe (#943).
    @test occursin("constraint `APL >= 0.05` resolved every name against the universe",
                   pe.empty_row_msg("APL >= 0.05", nx, "nx"))
    @test occursin("still summed to zero, so it constrains nothing; row dropped",
                   pe.empty_row_msg("APL >= 0.05", nx, "nx"))
    @test !occursin("matched no", pe.empty_row_msg("APL >= 0.05", nx, "nx"))
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
                                            "compact_show" => 4,
                                            "show_nothing_fields" => true,
                                            "show_nothing_fields_by_type" =>
                                                Dict{String, Any}("SimpleVariance" => false)))
    @test pe.EQUATION_LIMITS[].max_length == 512
    @test pe.EQUATION_LIMITS[].max_depth == 256          # unset key keeps its default
    @test pe.STRING_DISTANCE[].dist isa pe.StringDistances.DamerauLevenshtein
    @test pe.STRING_DISTANCE[].min_score == 0.8
    @test pe.COMPACT_SHOW[] == 4
    @test pe.SHOW_NOTHING_FIELDS[].default === true
    @test pe.SHOW_NOTHING_FIELDS[].by_type == Dict(:SimpleVariance => false)
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
    # The two show-nothing keys fail closed on a wrong value: a non-boolean switch, a table
    # that is not a table, and a table entry that is not a boolean. A name that matches no
    # type is accepted, because a type outside the package can render through the macro.
    @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}("show_nothing_fields" =>
                                                                           1))
    @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}("show_nothing_fields_by_type" =>
                                                                           true))
    @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}("show_nothing_fields_by_type" =>
                                                                           Dict{String,
                                                                                Any}("SimpleVariance" => "no")))
    @test isnothing(pe.apply_preferences!(Dict{String, Any}("show_nothing_fields_by_type" =>
                                                                Dict{String, Any}("NoSuchType" =>
                                                                                      true))))
    @test pe.SHOW_NOTHING_FIELDS[].by_type[:NoSuchType] === true
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
    pe.set_show_nothing_fields!(false)
    pe.set_show_nothing_fields!(:SimpleVariance, nothing)
    pe.set_show_nothing_fields!(:NoSuchType, nothing)
    @test pe.SHOW_NOTHING_FIELDS[].default === false
    @test isempty(pe.SHOW_NOTHING_FIELDS[].by_type)
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
    # One cap per sink (ADR 0041); the keyword constructor is the safe path since the
    # seven fields are same-typed and four share the value 100_000.
    @test pe.RESOURCE_LIMITS[] ==
          pe.ResourceLimits(1_000_000, 100_000, 100_000, 10_000, 100_000, 100_000, 10_000)
    @test pe.ResourceLimits() ==
          pe.ResourceLimits(1_000_000, 100_000, 100_000, 10_000, 100_000, 100_000, 10_000)
    @test pe.ResourceLimits(; max_bins = 500).max_bins == 500
    @test pe.ResourceLimits(; max_hop_count = 7).max_hop_count == 7
    @test pe.ResourceLimits(; max_search_grid = 9).max_search_grid == 9
    @test pe.ResourceLimits(; max_ep_grid = 21).max_ep_grid == 21
    # Every field must be positive, like EquationLimits.
    @test_throws ArgumentError pe.ResourceLimits(; max_n_sim = 0)
    @test_throws ArgumentError pe.ResourceLimits(; max_n_subsets = -1)
    @test_throws ArgumentError pe.ResourceLimits(; max_frontier = 0)
    @test_throws ArgumentError pe.ResourceLimits(; max_bins = -1)
    @test_throws ArgumentError pe.ResourceLimits(; max_hop_count = 0)
    @test_throws ArgumentError pe.ResourceLimits(; max_search_grid = -1)
    @test_throws ArgumentError pe.ResourceLimits(; max_ep_grid = 0)
    @test_throws ArgumentError pe.set_resource_limits!(max_n_sim = 0)
    @test_throws ArgumentError pe.set_resource_limits!(max_n_subsets = -1)
    @test_throws ArgumentError pe.set_resource_limits!(max_frontier = 0)
    @test_throws ArgumentError pe.set_resource_limits!(max_bins = -1)
    @test_throws ArgumentError pe.set_resource_limits!(max_hop_count = 0)
    @test_throws ArgumentError pe.set_resource_limits!(max_search_grid = -1)
    @test_throws ArgumentError pe.set_resource_limits!(max_ep_grid = -1)
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
    # HopCount.n indexes a sum of matrix powers `0:n`, so the compute cost is linear in n.
    # The constructor is also where resolve_separation sends a rule's answer, so the same
    # cap covers a stated hop count and a computed one.
    @test_throws DomainError HopCount(; n = 10^9)
    herr = try
        HopCount(; n = 10^9)
        nothing
    catch e
        e
    end
    @test herr isa DomainError
    @test occursin("max_hop_count", herr.msg)
    @test HopCount(; n = 3).n == 3
    @test HopCount(; n = pe.HopCountQuantile()).n isa pe.HopCountQuantile  # a rule, uncapped
    let nte = NetworkEstimator(), X = randn(50, 4)
        @test_throws DomainError pe.resolve_separation(HopCount(;
                                                                n = (args...; kwargs...) -> 10^9),
                                                       nte, X)
        @test pe.resolve_separation(HopCount(; n = (args...; kwargs...) -> 2), nte, X).n ==
              2
    end
    # A search grid is an Iterators.product: k parameters of N values are N^k candidates
    # and N^k full cross-validated fits, so the cap is on the product, not on any one N.
    @test_throws DomainError pe.lens_val_grid(["a" => collect(1:100), "b" => collect(1:100),
                                               "c" => collect(1:100)])
    gerr = try
        pe.lens_val_grid(["a" => collect(1:100), "b" => collect(1:100),
                          "c" => collect(1:100)])
        nothing
    catch e
        e
    end
    @test gerr isa DomainError
    @test occursin("max_search_grid", gerr.msg)
    @test occursin("a = 100", gerr.msg)  # the message names the guilty dimensions
    @test_throws DomainError pe.lens_val_grid(Dict("a" => collect(1:100),
                                                   "b" => collect(1:100),
                                                   "c" => collect(1:100)))
    @test length(pe.lens_val_grid(["a" => collect(1:10), "b" => collect(1:10)])[2]) == 100
    # Concatenated parameter sets are a sum of products, capped on the way out too.
    pe.with_resource_limits(; max_search_grid = 150) do
        @test length(pe.lens_val_grid([["a" => collect(1:10), "b" => collect(1:10)],
                                       ["a" => collect(1:5)]])[2]) == 105
        @test_throws DomainError pe.lens_val_grid([["a" => collect(1:10),
                                                    "b" => collect(1:10)],
                                                   ["a" => collect(1:10),
                                                    "b" => collect(1:10)]])
    end
    # K sizes a binary block of the mixed-integer program an upper-bound or equality
    # tail view builds, so the grid formulations take a cap of their own. The parity
    # check alone admitted a grid of ten million binaries from one view.
    @test_throws DomainError GridEntropicValueatRiskView(; K = 10_000_001)
    @test_throws DomainError GridRelativisticValueatRiskView(; K = 10_000_001)
    kerr = try
        GridRelativisticValueatRiskView(; K = 10_000_001)
        nothing
    catch e
        e
    end
    @test kerr isa DomainError
    @test occursin("max_ep_grid", kerr.msg)
    @test GridEntropicValueatRiskView(; K = 21).K == 21
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
    pe.with_resource_limits(; max_hop_count = 20_000_000) do
        @test HopCount(; n = 10_000_000).n == 10_000_000
    end
    pe.with_resource_limits(; max_search_grid = 10^7) do
        @test length(pe.lens_val_grid(["a" => collect(1:100), "b" => collect(1:100)])[2]) ==
              10_000
    end
    pe.with_resource_limits(; max_ep_grid = 20_000_000) do
        @test GridRelativisticValueatRiskView(; K = 10_000_001).K == 10_000_001
    end
    @test pe.RESOURCE_LIMITS[].max_n_sim == 1_000_000
    # Preferences fail closed at load, like the equation caps.
    pe.apply_preferences!(Dict{String, Any}("max_n_subsets" => 1_234, "max_bins" => 321))
    @test pe.RESOURCE_LIMITS[].max_n_subsets == 1_234
    @test pe.RESOURCE_LIMITS[].max_bins == 321
    @test pe.RESOURCE_LIMITS[].max_n_sim == 1_000_000  # unset key keeps its default
    @test pe.RESOURCE_LIMITS[].max_frontier == 100_000  # unset key keeps its default
    pe.apply_preferences!(Dict{String, Any}("max_hop_count" => 12, "max_search_grid" => 34,
                                            "max_ep_grid" => 56))
    @test pe.RESOURCE_LIMITS[].max_hop_count == 12
    @test pe.RESOURCE_LIMITS[].max_search_grid == 34
    @test pe.RESOURCE_LIMITS[].max_ep_grid == 56
    @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}("max_ep_grid" => 0))
    @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}("max_hop_count" => 0))
    @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}("max_search_grid" =>
                                                                           2.5))
    @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}("max_n_sim" => -5))
    @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}("max_n_subsets" =>
                                                                           true))
    @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}("max_frontier" => "1000"))
    @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}("max_bins" => 1.5))
    pe.set_resource_limits!(max_n_sim = 1_000_000, max_n_subsets = 100_000,
                            max_frontier = 100_000, max_bins = 10_000,
                            max_hop_count = 100_000, max_search_grid = 100_000,
                            max_ep_grid = 10_000)  # restore
end
@testset "A preference that widens a guard is announced" begin
    using PortfolioOptimisers, Test
    pe = PortfolioOptimisers
    # A LocalPreferences.toml is data: it travels with a cloned project and applies
    # before any user code runs. A widening value keeps its full range (ADR 0041
    # amendment) but is announced, so nothing distinguishes a chosen wide cap from an
    # inherited one.
    logs, = Test.collect_test_logs() do
        return pe.apply_preferences!(Dict{String, Any}("max_bins" => 50_000,
                                                       "max_n_sim" => 500,
                                                       "equation_max_depth" => 4096,
                                                       "suggestion_min_score" => 1.0e-9))
    end
    @test length(logs) == 1                    # one message for the whole load
    @test first(logs).level == Logging.Warn
    msg = first(logs).message
    @test occursin("max_bins: 10000 → 50000", msg)
    @test occursin("equation_max_depth: 256 → 4096", msg)
    @test occursin("suggestion_min_score: 0.7 → 1.0e-9", msg)  # lower admits more
    @test !occursin("max_n_sim", msg)          # 500 tightens the cap, so it stays silent
    @test occursin("LocalPreferences.toml", msg)
    # Tightening alone is silent: a project that hardens itself must not be trained to
    # ignore a warning at every `using`. An unchanged value is silent too.
    logs, = Test.collect_test_logs() do
        return pe.apply_preferences!(Dict{String, Any}("max_bins" => 100,
                                                       "suggestion_min_score" => 0.9,
                                                       "equation_max_length" => 4096))
    end
    @test isempty(logs)
    @test pe.RESOURCE_LIMITS[].max_bins == 100  # the value is applied either way
    pe.set_resource_limits!(max_n_sim = 1_000_000, max_n_subsets = 100_000,
                            max_frontier = 100_000, max_bins = 10_000,
                            max_hop_count = 100_000, max_search_grid = 100_000,
                            max_ep_grid = 10_000)  # restore
    pe.set_equation_limits!(max_length = 4096, max_depth = 256)
    pe.set_string_distance!(dist = pe.StringDistances.Levenshtein(), min_score = 0.7)
end
@testset "Log limits cap the repeats of one log site (#1690)" begin
    using PortfolioOptimisers, Test
    pe = PortfolioOptimisers
    # The shipped default sets no cap, so each site keeps the limit it states.
    @test pe.LOG_LIMITS[] == pe.LogLimits(nothing, nothing, nothing)
    @test pe.LogLimits() == pe.LogLimits(nothing, nothing, nothing)
    @test pe.LogLimits(; max_info = 3).max_info == 3
    @test pe.log_limit(:warn) === nothing
    @test pe.log_limit(:warn, 1) == 1
    # A cap is at least one, so it never hides the first message of a site.
    @test_throws ArgumentError pe.LogLimits(; max_warn = 0)
    @test_throws ArgumentError pe.LogLimits(; max_debug = -1)
    @test_throws ArgumentError pe.LogLimits(; max_info = true)
    @test_throws ArgumentError pe.set_log_limits!(max_warn = 0)
    @test_throws ArgumentError pe.with_log_limits(() -> nothing; max_info = 0)
    @test_throws ArgumentError pe.log_limit(:error)
    # A set cap replaces the limit of every site of its level, a site limit included.
    pe.with_log_limits(; max_warn = 2, max_info = 3, max_debug = 4) do
        @test pe.log_limit(:warn) == 2
        @test pe.log_limit(:warn, 1) == 2
        @test pe.log_limit(:info) == 3
        @test pe.log_limit(:debug, 1) == 4
        # Nested overrides inherit the keywords they omit.
        pe.with_log_limits(; max_warn = nothing) do
            @test pe.log_limit(:warn, 1) == 1
            @test pe.log_limit(:info) == 3
        end
    end
    @test pe.LOG_LIMITS[] == pe.LogLimits()  # the scoped override is restored
    pe.set_log_limits!(max_info = 5)
    @test pe.LOG_LIMITS[] == pe.LogLimits(; max_info = 5)
    pe.set_log_limits!(max_info = nothing)
    @test pe.LOG_LIMITS[] == pe.LogLimits()
    # The `_id` of `@log_id` keys on the site and on the parts, so a different part or a
    # different site counts on its own, and the same part at the same site repeats.
    # A macro resolves its module when the testset expands, before `pe` is bound.
    id(k) = PortfolioOptimisers.@log_id(k)
    @test id("a") == id("a")
    @test id("a") != id("b")
    @test PortfolioOptimisers.@log_id("a") != id("a")
    # A site under the cap logs the cap and no more, once per `_id`.
    diag(k) = pe.strict_diagnostic("unknown name `$k` at $(rand())", false; id = id(k))
    logs, = Test.collect_test_logs() do
        return pe.with_log_limits(; max_warn = 2) do
            for _ in 1:4, k in ("a", "b")
                diag(k)
            end
        end
    end
    @test length(logs) == 4
    @test count(r -> occursin("`a`", r.message), logs) == 2
    logs, = Test.collect_test_logs() do
        for _ in 1:4
            diag("a")
        end
    end
    @test length(logs) == 4  # no cap, no limit
    # The preferences seed the global default. A valid value is silent: a cap widens no
    # guard, so it records nothing for the announcement.
    logs, = Test.collect_test_logs() do
        return pe.apply_preferences!(Dict{String, Any}("max_warn" => 1, "max_debug" => 7))
    end
    @test isempty(logs)
    @test pe.LOG_LIMITS[] == pe.LogLimits(; max_warn = 1, max_debug = 7)
    pe.apply_preferences!(Dict{String, Any}("max_info" => 2))
    @test pe.LOG_LIMITS[] == pe.LogLimits(1, 2, 7)  # an unset key keeps its value
    # The announcement of a widened guard runs under the cap the same load set. Each pass
    # restores the shipped cap first, so each pass widens the guard again.
    logs, = Test.collect_test_logs() do
        for _ in 1:3
            pe.set_resource_limits!(max_bins = 10_000)
            pe.apply_preferences!(Dict{String, Any}("max_bins" => 50_000))
        end
    end
    @test length(logs) == 1
    pe.with_log_limits(; max_warn = nothing) do
        logs, = Test.collect_test_logs() do
            for _ in 1:3
                pe.set_resource_limits!(max_bins = 10_000)
                pe.apply_preferences!(Dict{String, Any}("max_bins" => 50_000))
            end
        end
        @test length(logs) == 3
    end
    for (key, val) in
        (("max_warn", 0), ("max_info", -1), ("max_debug", true), ("max_warn", "1"),
         ("max_info", 1.5))
        @test_throws ArgumentError pe.apply_preferences!(Dict{String, Any}(key => val))
    end
    pe.set_log_limits!(max_warn = nothing, max_info = nothing, max_debug = nothing)
    pe.set_resource_limits!(max_bins = 10_000)  # restore
    @test pe.LOG_LIMITS[] == pe.LogLimits()
end
@testset "Canonicalisation invariants (issue #513)" begin
    using PortfolioOptimisers, Test, Logging
    with_logger(SimpleLogger(stderr, Logging.Error)) do
        # Repeated names accumulate into one coefficient rather than two entries.
        res = parse_equation("2*A + 3*A <= 1")
        @test res.vars == ["A"]
        @test res.coef == [5.0]

        # `collect_terms!` handles `:-` by negating the last argument, and a unary minus
        # holds no argument but the last, so its one operand is the negated one.
        res = parse_equation("-A <= 1")
        @test res.coef == [-1.0]
        res = parse_equation("A - B - C <= 1")
        idx = sortperm(res.vars)
        @test res.vars[idx] == ["A", "B", "C"]
        @test res.coef[idx] == [1.0, -1.0, -1.0]
        res = parse_equation("-(A - B) <= 1")
        idx = sortperm(res.vars)
        @test res.vars[idx] == ["A", "B"]
        @test res.coef[idx] == [-1.0, 1.0]

        # A product or a quotient of two non-numeric sides is opaque: it becomes one
        # variable named by its own text, which resolves against the universe like any
        # other name and is dropped when it matches none.
        @test parse_equation("A/B <= 1").vars == ["A / B"]
        @test parse_equation("A*B <= 1").vars == ["A * B"]
        # A node whose head is not `:call` is rebuilt from its folded arguments, so a
        # tuple survives the fold and becomes one opaque name.
        @test parse_equation("(A, B) <= 1").vars == ["(A, B)"]
        sets = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"]))
        # One warning, not two: an unresolved name takes its whole row (ADR 0125), so the
        # empty-row report it used to reach a line later cannot fire.
        @test isnothing(@test_logs (:warn,) linear_constraints("A/B <= 1", sets))
        @test_throws ArgumentError linear_constraints("A/B <= 1", sets; strict = true)

        # `datatype` names the numeric domain of the coefficients as well as of the
        # right-hand side. The walk starts from `one(datatype)`, so `coef` carries it.
        res = parse_equation("2*A + 1 <= 3"; datatype = Float32)
        @test res.coef isa Vector{Float32}
        @test res.rhs isa Float32
        @test res.coef == Float32[2.0]
        @test res.rhs == Float32(2)

        # One constraint prints one way whether or not a group expanded: both renderings
        # go through `format_term`, which elides a unit coefficient.
        gsets = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"], "g" => ["A", "B"]))
        res = replace_group_by_assets(parse_equation("g + 2*C <= 1"), gsets)
        @test res.eqn == "2.0*C + A + B <= 1.0"

        # A `>=` row is the same constraint as the negated `<=` row, in both halves.
        ge = linear_constraints("A >= 0.1", sets)
        le = linear_constraints("-A <= -0.1", sets)
        @test collect(ge.ineq.A) == collect(le.ineq.A)
        @test ge.ineq.B == le.ineq.B
        @test collect(ge.ineq.A) == [-1.0 0.0 0.0]
        @test ge.ineq.B == [-0.1]

        # A group carries the coefficient onto every member, and the Black-Litterman
        # expansion divides it by the member count instead: a sum against a mean.
        s3 = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"], "g3" => ["A", "B", "C"]))
        eq6 = parse_equation("6*g3 <= 1")
        @test replace_group_by_assets(eq6, s3, false).coef == [6.0, 6.0, 6.0]
        @test replace_group_by_assets(eq6, s3, true).coef == [2.0, 2.0, 2.0]
    end
end
@testset "One comparison-operator table (issue #520)" begin
    using PortfolioOptimisers, Test
    # The (sign, is_inequality) table had five encodings and one caller each, so one
    # change to it forced five edits. Both spellings of an operator — the function a
    # constraint estimator carries and the string a `ParsingResult` carries — now go
    # through `comparison_sign_ineq_flag`, and the two spellings must give the same row.
    csif = PortfolioOptimisers.comparison_sign_ineq_flag
    @test csif(==) == csif("==") == (1, false)
    @test csif(<=) == csif("<=") == (1, true)
    @test csif(>=) == csif(">=") == (-1, true)
    @test all(r -> isa(r[1], Int) && isa(r[2], Bool),
              (csif("=="), csif("<="), csif(">="), csif(==), csif(<=), csif(>=)))
    # An operator outside the three used to fall through to the equality branch in
    # silence. `ParsingResult` is exported and takes `op` positionally, so a hand-built
    # result can carry any string and reach the table.
    @test_throws ArgumentError csif("!=")
    @test_throws ArgumentError csif("<")
    sets = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"]))
    @test_throws ArgumentError PortfolioOptimisers.get_linear_constraints([ParsingResult(["A"],
                                                                                         [1.0],
                                                                                         "!=",
                                                                                         1.0,
                                                                                         "A != 1.0")],
                                                                          sets)
    # Each of the three operators still lands in the half, and with the sign, the table
    # names: `==` in the equality half, `<=` and `>=` in the inequality half written in
    # the `<=` sense.
    lc = PortfolioOptimisers.get_linear_constraints([ParsingResult(["A"], [1.0], "==", 1.0,
                                                                   "A == 1.0")], sets)
    @test isnothing(lc.ineq)
    @test collect(lc.eq.A) == [1.0 0.0 0.0]
    @test lc.eq.B == [1.0]
    lc = PortfolioOptimisers.get_linear_constraints([ParsingResult(["A"], [1.0], "<=", 1.0,
                                                                   "A <= 1.0")], sets)
    @test isnothing(lc.eq)
    @test collect(lc.ineq.A) == [1.0 0.0 0.0]
    @test lc.ineq.B == [1.0]
    lc = PortfolioOptimisers.get_linear_constraints([ParsingResult(["A"], [1.0], ">=", 1.0,
                                                                   "A >= 1.0")], sets)
    @test isnothing(lc.eq)
    @test collect(lc.ineq.A) == [-1.0 0.0 0.0]
    @test lc.ineq.B == [-1.0]
end
@testset "The docstrings of 02_LinearConstraintGeneration against numbers" begin
    using PortfolioOptimisers, Test, Logging
    PO = PortfolioOptimisers
    sorted(r) = (p = sortperm(r.vars); (r.vars[p], r.coef[p], r.rhs))
    # The parser writes a product of three factors as one call, `*(2, 3, x)`. Every
    # numeric factor joins the coefficient, so no factor is lost.
    @test sorted(parse_equation("2*3*x <= 1")) == (["x"], [6.0], 1.0)
    @test sorted(parse_equation("2*x*3 <= 1")) == (["x"], [6.0], 1.0)
    @test sorted(parse_equation("0.5*2*x + y <= 1")) == (["x", "y"], [1.0, 1.0], 1.0)
    # More than one factor that is not a number makes one opaque term, and the numeric
    # factors stay in its coefficient.
    @test sorted(parse_equation("2*x*y + z <= 1")) == (["x * y", "z"], [2.0, 1.0], 1.0)
    @test sorted(parse_equation("x/y + 2*sqrt(z) <= 1")) ==
          (["sqrt(z)", "x / y"], [2.0, 1.0], 1.0)
    # A string literal is a variable named by its text, and it is no longer dropped.
    @test sorted(parse_equation("\"w_A\" + w_B <= 1")) == (["w_A", "w_B"], [1.0, 1.0], 1.0)
    # Julia reads `2f1` as the Float32 literal 20 and `2e1` as the Float64 literal 20, so
    # a name that starts like an exponent needs an explicit `*`.
    @test sorted(parse_equation("2e1 + 2f1 <= 1")) == (String[], Float64[], -39.0)
    @test sorted(parse_equation("2*f1 <= 1")) == (["f1"], [2.0], 1.0)
    @test isnothing(PO.collect_terms!(:x, 1.0, []))
    @test PO._collect_terms(:(2 * 3 * 4)) == Any[(24.0, nothing)]
    @test PO._collect_terms(:(x + 1)) isa Vector{Any}
    # The right-hand side is accumulated in `datatype`.
    @test PO._parse_equation(:x, "<=", 1, Rational{Int}).rhs === 1 // 1
    # The uniform value is computed in `datatype`, so a Rational is exact.
    sets = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"]))
    u = PO.estimator_to_val(UniformValues(), sets; datatype = Rational{Int})
    @test all(==(1 // 3), u)
    @test sum(u) == 1
    @test PO.estimator_to_val(UniformValues(), sets) == fill(inv(3.0), 3)
    @test eltype(PO.estimator_to_val(UniformValues(), sets; datatype = BigFloat)) ==
          BigFloat
    # An empty group stays empty when the counterpart axis is not empty.
    @test PO.shed_departed_members(String[], ["D"], nothing, "e", "e == 1") == String[]
    @test PO.shed_departed_members(String[], String[], ["D"], nothing, "(e, e)", "x") ==
          (String[], String[])
    # A group that sheds every member keeps its first member, and the row that then names
    # a departed asset is dropped in silence, with one entry in the ledger.
    sets2 = UniverseSets(;
                         dict = Dict("nx" => ["A", "B", "C"], "h" => ["D", "E"],
                                     "g" => ["A", "B", "D"], "e" => String[],
                                     "ni" => ["D", "E"]))
    ledger = String[]
    res = replace_group_by_assets(parse_equation("h + A <= 1"), sets2; ledger = ledger)
    @test sorted(res) == (["A", "D"], [1.0, 1.0], 1.0)
    @test ledger == ["the row `h + A <= 1.0`, whose group `h` lost every member"]
    @test isnothing(PO.get_linear_constraints(res, sets2))
    # A group that holds no member expands to nothing: the sum over it is zero.
    @test sorted(replace_group_by_assets(parse_equation("e + A <= 1"), sets2)) ==
          (["A"], [1.0], 1.0)
    # The sum expansion repeats the coefficient, and the mean expansion divides it by the
    # surviving count, here two of three members.
    @test sorted(replace_group_by_assets(parse_equation("g == 0.3"), sets2)) ==
          (["A", "B"], [1.0, 1.0], 0.3)
    @test sorted(replace_group_by_assets(parse_equation("g == 0.3"), sets2, true)) ==
          (["A", "B"], [0.5, 0.5], 0.3)
    led = String[]
    PO.record_group_shed!(led, "g", 1, 2, "g <= 1")
    PO.record_group_shed!(led, "g", 0, 2, "g <= 1")
    @test led == ["1 departed member(s) of the group `g` in the row `g <= 1`"]
    # A `>=` row is negated into the `<=` sense of the inequality half.
    lc = linear_constraints(["A + B == 1", "A >= 0.1", "C <= 0.5"], sets)
    @test collect(lc.ineq.A) == [-1.0 0.0 0.0; 0.0 0.0 1.0]
    @test lc.ineq.B == [-0.1, 0.5]
    @test collect(lc.eq.A) == [1.0 1.0 0.0]
    @test lc.eq.B == [1.0]
    # Merging the halves of two constraints equals one call over all their equations.
    l1 = linear_constraints(["A + B == 1", "A >= 0.1"], sets)
    l2 = linear_constraints(["C <= 0.5", "B - C == 0"], sets)
    la = linear_constraints(["A + B == 1", "A >= 0.1", "C <= 0.5", "B - C == 0"], sets)
    mg = PO.merge_linear_constraints([l1, l2])
    @test collect(mg.ineq.A) == collect(la.ineq.A)
    @test mg.ineq.B == la.ineq.B
    @test collect(mg.eq.A) == collect(la.eq.A)
    @test mg.eq.B == la.eq.B
    # A row written in factor names is re-based as (M a)' w over the assets.
    M = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    rr = PO.Regression(; M = M, b = zeros(3))
    fs = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"], "nf" => ["f1", "f2"]))
    lr = PO.get_linear_constraints(parse_equation("2*f1 - f2 <= 0.1"), fs, "nf"; rr = rr)
    @test vec(collect(lr.ineq.A)) == M * [2.0, -1.0]
    @test PO.factor_axis_key(fs, rr) == "nf"
    # The view slices the asset axis, rebuilds a unique-entry group from the sliced
    # partition, drops the exact `nikey` entry, and keeps the seven key prefixes.
    vs = UniverseSets(;
                      dict = Dict("nx" => ["A", "B", "C"], "nx_s" => ["x", "y", "x"],
                                  "ux_s" => ["x", "y"], "ni" => ["D"], "nikkei" => ["A"],
                                  "nf" => ["f1"]))
    v = PO.port_opt_view(vs, [1, 3])
    @test v.dict["nx"] == ["A", "C"]
    @test v.dict["ux_s"] == ["x"]
    @test !haskey(v.dict, "ni")
    @test v.dict["nikkei"] == ["A"]
    @test v.dict["nf"] == ["f1"]
    @test (v.xkey, v.uxkey, v.tfkey, v.utfkey, v.cfkey, v.ucfkey, v.nikey) ==
          (vs.xkey, vs.uxkey, vs.tfkey, vs.utfkey, vs.cfkey, vs.ucfkey, vs.nikey)
    # The depth counts the levels of `Expr` along the deepest path.
    @test [PO._expr_depth_exceeds(:(a + b), l) for l in 0:2] == [true, false, false]
    @test PO._expr_depth_exceeds(:(a + (b + c)), 1)
    @test !PO._expr_depth_exceeds(:(a + (b + c)), 2)
    # A precomputed constraint wider than the investable universe is refused.
    @test_throws DimensionMismatch PO.assert_investable_constraint_width(lc, 2, "lcse")
    @test_throws PO.IsEmptyError LinearConstraintEstimator(; val = "A <= 1", key = "")
end

@testset "Factor exposure constraints" begin
    using PortfolioOptimisers, Test, Logging, LinearAlgebra, Clarabel, StableRNGs
    # Hand-built loadings and universes throughout: the point of these tests is that the row
    # that comes out is `M * a`, so anything estimated would obscure the arithmetic.
    #
    #        MTUM  VLUE  QUAL
    #   A     1.0   0.0   0.2
    #   B     0.5   0.5   0.0
    #   C     0.0   1.0   0.7
    M = [1.0 0.0 0.2; 0.5 0.5 0.0; 0.0 1.0 0.7]
    rr = Regression(; M = M)
    sets = UniverseSets(;
                        dict = Dict("nx" => ["A", "B", "C"],
                                    "nf" => ["MTUM", "VLUE", "QUAL"],
                                    "defensive" => ["VLUE", "QUAL"]))
    fs = FactorSpace()
    @testset "Type structure" begin
        @test FactorSpace <: PortfolioOptimisers.AbstractConstraintSpace
        @test PortfolioOptimisers.AbstractConstraintSpace <:
              PortfolioOptimisers.AbstractAlgorithm
        # ADR 0047: the asset frame is the *absence* of a re-basis, so there is no
        # `AssetSpace` to write down. Anything else in the family is a real second basis.
        @test !isdefined(PortfolioOptimisers, :AssetSpace)
        # `space` is required: a re-basis is asked for, never inferred.
        @test_throws UndefKeywordError ExposureConstraintEstimator(;
                                                                   lce = LinearConstraintEstimator(;
                                                                                                   val = "MTUM <= 0.3"))
        @test_throws UndefKeywordError ExposureConstraintEstimator(; space = fs)
        # The bound is exactly what `lcse` accepts, so no shape reaches it un-re-based.
        @test_throws TypeError ExposureConstraintEstimator(; lce = "MTUM <= 0.3",
                                                           space = fs)
        @test_throws TypeError ExposureConstraintEstimator(;
                                                           lce = LinearConstraintEstimator(;
                                                                                           val = "MTUM <= 0.3"),
                                                           space = "factor")
        @test_throws PortfolioOptimisers.IsEmptyError ExposureConstraintEstimator(;
                                                                                  lce = LinearConstraintEstimator[],
                                                                                  space = fs)
        # `LinearConstraintEstimator` is left exactly as it was — being unmarked is what
        # keeps it the only thing `gcarde`/`sgcarde` admit.
        @test fieldnames(LinearConstraintEstimator) == (:val, :key)
        ece = ExposureConstraintEstimator(;
                                          lce = LinearConstraintEstimator(;
                                                                          val = "MTUM <= 0.3"),
                                          space = fs)
        @test fieldnames(typeof(ece)) == (:lce, :space)
        @test isa(ece.space, FactorSpace)
    end
    @testset "Projection: the row is M * a" begin
        # A single factor: the row is that factor's loadings column.
        lcr = linear_constraints(ExposureConstraintEstimator(;
                                                             lce = LinearConstraintEstimator(;
                                                                                             val = "MTUM <= 0.3"),
                                                             space = fs), sets; rr = rr)
        @test isa(lcr, LinearConstraint)
        @test isnothing(lcr.eq)
        @test lcr.ineq.A == transpose(reshape(M[:, 1], :, 1))
        @test lcr.ineq.B == [0.3]
        # The whole point, stated as the identity it is: for any weights, the projected row
        # evaluates to the same number as the factor row applied to `M' * w`.
        w = [0.2, 0.5, 0.3]
        a = [1.0, 0.0, 0.0]
        @test only(lcr.ineq.A * w) ≈ dot(a, transpose(M) * w)
        # Equality, and a multi-term row with coefficients.
        lcr = linear_constraints(ExposureConstraintEstimator(;
                                                             lce = LinearConstraintEstimator(;
                                                                                             val = "2*MTUM - VLUE == 0.1"),
                                                             space = fs), sets; rr = rr)
        @test isnothing(lcr.ineq)
        a = [2.0, -1.0, 0.0]
        @test vec(lcr.eq.A) ≈ M * a
        @test lcr.eq.B == [0.1]
        # `>=` flips sign on both sides, exactly as on the asset axis.
        lcr = linear_constraints(ExposureConstraintEstimator(;
                                                             lce = LinearConstraintEstimator(;
                                                                                             val = "QUAL >= 0.25"),
                                                             space = fs), sets; rr = rr)
        @test vec(lcr.ineq.A) ≈ -M[:, 3]
        @test lcr.ineq.B == [-0.25]
        # Both kinds in one estimator.
        lcr = linear_constraints(ExposureConstraintEstimator(;
                                                             lce = LinearConstraintEstimator(;
                                                                                             val = ["MTUM <= 0.3",
                                                                                                    "VLUE == 0.0"]),
                                                             space = fs), sets; rr = rr)
        @test vec(lcr.ineq.A) ≈ M[:, 1]
        @test vec(lcr.eq.A) ≈ M[:, 2]
    end
    @testset "Group expansion is axis-blind" begin
        # Characterisation: `replace_group_by_assets` expands any name found in `dict` and
        # never asks which axis it is on, so a factor group needs no machinery of its own.
        lcr = linear_constraints(ExposureConstraintEstimator(;
                                                             lce = LinearConstraintEstimator(;
                                                                                             val = "defensive >= 0.2"),
                                                             space = fs), sets; rr = rr)
        @test vec(lcr.ineq.A) ≈ -(M[:, 2] + M[:, 3])
        @test lcr.ineq.B == [-0.2]
        # Same answer written out longhand.
        lng = linear_constraints(ExposureConstraintEstimator(;
                                                             lce = LinearConstraintEstimator(;
                                                                                             val = "VLUE + QUAL >= 0.2"),
                                                             space = fs), sets; rr = rr)
        @test lng.ineq.A == lcr.ineq.A
    end
    @testset "Duplicate factor names sum their columns" begin
        # The projection sums `view(M, :, Ai)` rather than indexing `findfirst`, so a factor
        # universe carrying a name twice contributes both columns — matching how the asset
        # path treats a duplicated asset name.
        setsd = UniverseSets(; dict = Dict("nx" => ["A", "B"], "nf" => ["MTUM", "MTUM"]))
        Md = [1.0 2.0; 3.0 4.0]
        lcr = linear_constraints(ExposureConstraintEstimator(;
                                                             lce = LinearConstraintEstimator(;
                                                                                             val = "MTUM == 1"),
                                                             space = fs), setsd;
                                 rr = Regression(; M = Md))
        @test vec(lcr.eq.A) == [3.0, 7.0]
        @test lcr.eq.B == [1.0]
    end
    @testset "Precomputed LinearConstraint projects wholesale" begin
        # The one place `linear_constraints(::LinearConstraint, …)` stops being a
        # pass-through: a precomputed constraint written in the wrapped basis must still be
        # re-based.
        pc = LinearConstraint(;
                              ineq = PartialLinearConstraint(;
                                                             A = [1.0 0.0 0.0;
                                                                  0.0 1.0 1.0],
                                                             B = [0.3, 0.5]),
                              eq = PartialLinearConstraint(; A = [0.0 0.0 1.0], B = [0.1]))
        lcr = linear_constraints(ExposureConstraintEstimator(; lce = pc, space = fs), sets;
                                 rr = rr)
        @test lcr.ineq.A ≈ pc.ineq.A * transpose(M)
        @test lcr.ineq.B == pc.ineq.B          # the bound is untouched by a change of basis
        @test lcr.eq.A ≈ pc.eq.A * transpose(M)
        @test lcr.eq.B == pc.eq.B
        # Nothing else would catch an asset-space matrix handed to a factor-space wrapper,
        # because a precomputed constraint carries no names.
        bad = LinearConstraint(; ineq = PartialLinearConstraint(; A = [1.0 0.0], B = [0.3]))
        @test_throws DimensionMismatch linear_constraints(ExposureConstraintEstimator(;
                                                                                      lce = bad,
                                                                                      space = fs),
                                                          sets; rr = rr)
        # An asset-space precomputed constraint reaching `lcse` alongside loadings is still
        # a pass-through: only the decorator asks for a re-basis.
        @test linear_constraints(pc, sets; rr = rr) === pc
    end
    @testset "Vectors, and mixing bases" begin
        vece = ExposureConstraintEstimator(;
                                           lce = [LinearConstraintEstimator(;
                                                                            val = "MTUM <= 0.3"),
                                                  LinearConstraintEstimator(;
                                                                            val = "VLUE == 0.1")],
                                           space = fs)
        v = linear_constraints(vece, sets; rr = rr)
        @test length(v) == 2
        @test vec(v[1].ineq.A) ≈ M[:, 1]
        @test vec(v[2].eq.A) ≈ M[:, 2]
        # A vector may mix a re-based constraint with an asset-space one; each resolves
        # against its own axis.
        ece = ExposureConstraintEstimator(;
                                          lce = LinearConstraintEstimator(;
                                                                          val = "MTUM <= 0.3"),
                                          space = fs)
        mixed = PortfolioOptimisers.EcE_LcE_Lc[LinearConstraintEstimator(;
                                                                         val = "A >= 0.1"),
                                               ece]
        mx = linear_constraints(mixed, sets; rr = rr)
        @test length(mx) == 2
        @test vec(mx[1].ineq.A) == [-1.0, 0.0, 0.0]   # "A" is an asset, not a factor
        @test vec(mx[2].ineq.A) ≈ M[:, 1]
        # A bare estimator accepts the loadings and drops them: the space is what the user
        # wrote, never what the prior happened to carry.
        bare = LinearConstraintEstimator(; val = "A >= 0.1")
        @test linear_constraints(bare, sets; rr = rr).ineq.A ==
              linear_constraints(bare, sets).ineq.A
        # Homogeneous vectors of bare estimators still take the narrower method, which now
        # accepts and forwards the loadings its elements go on to drop.
        vb = linear_constraints([bare, bare], sets; rr = rr)
        @test length(vb) == 2
        @test all(x -> x.ineq.A == linear_constraints(bare, sets).ineq.A, vb)
    end
    @testset "The wrapped key overrides the factor axis" begin
        # `key === nothing` resolves to `sets.tfkey` rather than `sets.xkey`; a key written on
        # the wrapped estimator still wins, so an alternative factor partition can be named.
        setsk = UniverseSets(;
                             dict = Dict("nx" => ["A", "B", "C"],
                                         "nf" => ["MTUM", "VLUE", "QUAL"],
                                         "nf_alias" => ["mom", "val", "qual"]))
        lcr = linear_constraints(ExposureConstraintEstimator(;
                                                             lce = LinearConstraintEstimator(;
                                                                                             val = "val <= 0.3",
                                                                                             key = "nf_alias"),
                                                             space = fs), setsk; rr = rr)
        @test vec(lcr.ineq.A) ≈ M[:, 2]
    end
    @testset "Failure modes" begin
        ece = ExposureConstraintEstimator(;
                                          lce = LinearConstraintEstimator(;
                                                                          val = "MTUM <= 0.3"),
                                          space = fs)
        # Missing loadings throw *regardless of `strict`*. `strict` governs unknown names — a
        # per-row, recoverable condition. A missing regression makes every row unbuildable,
        # and dropping them silently yields a portfolio carrying none of the exposure asked
        # for.
        @test_throws PortfolioOptimisers.IsNothingError linear_constraints(ece, sets;
                                                                           strict = false)
        @test_throws PortfolioOptimisers.IsNothingError linear_constraints(ece, sets;
                                                                           strict = true)
        @test_throws PortfolioOptimisers.IsNothingError linear_constraints(ece, sets;
                                                                           rr = nothing)
        # The diagnosis is the library's shared one, so a user sees the same remedy here as
        # from an estimator that needed a factor block.
        msg = try
            linear_constraints(ece, sets)
        catch e
            sprint(showerror, e)
        end
        @test occursin(PortfolioOptimisers.prior_regression_remedy, msg)
        # The factor axis is optional on `UniverseSets`; it is not optional for a constraint
        # written against it.
        setsx = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"]))
        @test_throws KeyError linear_constraints(ece, setsx; rr = rr)
        # The loadings and the declared axis must agree on how many factors there are.
        sets2 = UniverseSets(;
                             dict = Dict("nx" => ["A", "B", "C"], "nf" => ["MTUM", "VLUE"]))
        @test_throws DimensionMismatch linear_constraints(ece, sets2; rr = rr)
    end
    @testset "Diagnostics name the factor axis" begin
        # An unknown *name* keeps `strict` semantics, but the message must name the universe
        # the user was writing in — reporting the asset universe would send them looking in
        # the wrong place.
        ecu = ExposureConstraintEstimator(;
                                          lce = LinearConstraintEstimator(;
                                                                          val = "MTUMM <= 0.3"),
                                          space = fs)
        msg = try
            linear_constraints(ecu, sets; rr = rr, strict = true)
        catch e
            sprint(showerror, e)
        end
        @test occursin("not in factor universe", msg)
        @test occursin("3 factors under key `nf`", msg)
        @test occursin("did you mean `MTUM`?", msg)
        # Non-strict warns once and drops the row whole (ADR 0125). It does *not* also
        # report the row as empty: the row went at the name it could not resolve, so the
        # second message would have been the same news twice.
        logs, _ = Test.collect_test_logs() do
            return linear_constraints(ExposureConstraintEstimator(;
                                                                  lce = LinearConstraintEstimator(;
                                                                                                  val = "ZZZ <= 0.3"),
                                                                  space = fs), sets;
                                      rr = rr, strict = false)
        end
        msgs = [l.message for l in logs]
        @test any(m -> occursin("not in factor universe", m), msgs)
        @test !any(m -> occursin("summed to zero", m), msgs)
        @test length(msgs) == 1
        # A row whose names *did* resolve but whose loadings annihilate it is a different
        # failure, and says so: reporting the row's own arithmetic for it would send a user
        # auditing coefficients that are fine.
        rrz = Regression(; M = [0.0 0.0 0.2; 0.0 0.5 0.0; 0.0 1.0 0.7])
        msg = try
            linear_constraints(ExposureConstraintEstimator(;
                                                           lce = LinearConstraintEstimator(;
                                                                                           val = "MTUM <= 0.3"),
                                                           space = fs), sets; rr = rrz,
                               strict = true)
        catch e
            sprint(showerror, e)
        end
        @test occursin("resolved against the factor universe", msg)
        @test occursin("every matched factor has zero loadings", msg)
        # The asset path's messages are untouched.
        msg = try
            linear_constraints(LinearConstraintEstimator(; val = "ZZZ <= 0.3"), sets;
                               strict = true)
        catch e
            sprint(showerror, e)
        end
        @test occursin("not in asset universe", msg)
        @test occursin("3 assets under key `nx`", msg)
    end
    @testset "Reduced loadings are not what a constraint reads" begin
        # `M` is the loadings over the *named original* factors; `L` is the same information
        # in the reduced basis, whose columns are principal components and cannot be named in
        # an equation. A constraint must be written, so it reads `M`. Risk decomposition
        # reads `L` and is right to — ADR 0047 says not to harmonise them.
        rrl = Regression(; M = M, L = [1.0 0.0; 0.0 1.0; 1.0 1.0])
        lcr = linear_constraints(ExposureConstraintEstimator(;
                                                             lce = LinearConstraintEstimator(;
                                                                                             val = "MTUM <= 0.3"),
                                                             space = fs), sets; rr = rrl)
        @test vec(lcr.ineq.A) ≈ M[:, 1]
        # The axis check counts `M`'s columns, not `L`'s: `L` has two here and the constraint
        # still builds against a three-factor axis.
        @test size(rrl.L, 2) != length(sets.dict[sets.tfkey])
    end
    # A tiny factor market for the optimiser tests: the assets are combinations of two
    # factors plus noise, so a fitted `M` is close to `Mo` without being it. Every assertion
    # reads the *fitted* loadings off the result — the claim is that the realised exposure
    # is the one the row was written against, not that the regression recovers `Mo`.
    orng = StableRNG(987654321)
    Fo = randn(orng, 400, 2) * 0.01
    Mo = [1.0 0.0; 0.5 0.5; 0.0 1.0]
    Xo = Fo * transpose(Mo) + randn(orng, 400, 3) * 0.001
    rdo = ReturnsResult(; nx = ["A", "B", "C"], X = Xo, nf = ["MTUM", "VLUE"], F = Fo)
    osets = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"], "nf" => ["MTUM", "VLUE"]))
    oslv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = "verbose" => false)
    oece = ExposureConstraintEstimator(;
                                       lce = LinearConstraintEstimator(;
                                                                       val = "MTUM <= 0.3"),
                                       space = fs)
    @testset "`lcse` is the only slot that admits a re-basis" begin
        @test JuMPOptimiser(; slv = oslv, sets = osets, lcse = oece) isa JuMPOptimiser
        # Cardinality rows index the binary held-indicators rather than `w`, so a projected
        # row is neither integral nor an index into them. ADR 0047 makes that
        # *unrepresentable* rather than validated: the narrow bound rejects it outright.
        @test_throws TypeError JuMPOptimiser(; slv = oslv, sets = osets, gcarde = oece)
        @test_throws TypeError JuMPOptimiser(; slv = oslv, sets = osets, sgcarde = oece)
        # A vector may mix bases. It needs the explicit element type for the same reason
        # `VecLcE_Lc` does — a heterogeneous literal promotes to the abstract supertype.
        mixed = PortfolioOptimisers.EcE_LcE_Lc[oece,
                                               LinearConstraintEstimator(;
                                                                         val = "A >= 0.05")]
        @test JuMPOptimiser(; slv = oslv, sets = osets, lcse = mixed) isa JuMPOptimiser
        # The names resolve against `sets`, so a re-basis without one fails at construction
        # like every other estimator-typed field. This holds for a wrapped *precomputed*
        # constraint too: `constraint_space_basis` needs the sets to find the factor axis.
        @test_throws PortfolioOptimisers.IsNothingError JuMPOptimiser(; slv = oslv,
                                                                      lcse = oece)
        @test_throws PortfolioOptimisers.IsNothingError JuMPOptimiser(; slv = oslv,
                                                                      lcse = mixed)
    end
    @testset "End to end: the realised exposure is the requested one" begin
        res = optimise(MeanRisk(;
                                opt = JuMPOptimiser(; pe = FactorPrior(), slv = oslv,
                                                    sets = osets, lcse = oece)), rdo)
        Mf = res.pa.pr.rr.M
        # The mandate binds: momentum exposure sits on the cap it was written against.
        @test isapprox(dot(Mf[:, 1], res.w), 0.3; atol = 1e-6)
        # What reached the model is an ordinary asset-space constraint — one row, one column
        # per asset. There is no factor-length row anywhere and no second pathway.
        @test isa(res.lcsr, LinearConstraint)
        @test size(res.lcsr.ineq.A) == (1, length(rdo.nx))
        @test vec(res.lcsr.ineq.A) ≈ Mf[:, 1]
        # Handing the optimiser the projected row directly is the same problem, which is the
        # operational content of "re-basing is a change of coordinates, not a feature".
        hand = LinearConstraint(;
                                ineq = PartialLinearConstraint(; A = transpose(Mf[:, 1:1]),
                                                               B = [0.3]))
        resh = optimise(MeanRisk(;
                                 opt = JuMPOptimiser(; pe = FactorPrior(), slv = oslv,
                                                     sets = osets, lcse = hand)), rdo)
        @test isapprox(res.w, resh.w; rtol = 5e-5)
    end
    @testset "A re-based row survives the ratio's scale floor (#924)" begin
        # A `FactorSpace` row is dense by construction — one column per asset, all of them
        # non-zero — so it is the row that carried the largest violation when `MaximumRatio`
        # answered on the degenerate homogenised ray. The floor `MaximumRatio` now writes is
        # what keeps it binding; `kmin` set below the collapse reproduces the defect exactly.
        # See the scale floor section of `MaximumRatio` for the mechanism.
        ue = EllipsoidalUncertaintySet(; sigma = Matrix(0.1I, 3, 3), k = 1e3,
                                       class = MuUncertaintySetClass())
        cap = ExposureConstraintEstimator(;
                                          lce = LinearConstraintEstimator(;
                                                                          val = "MTUM >= 0.9"),
                                          space = fs)
        ratio(obj) = optimise(MeanRisk(; obj = obj,
                                       opt = JuMPOptimiser(; pe = FactorPrior(), slv = oslv,
                                                           sets = osets, bgt = 1.0,
                                                           wb = WeightBounds(; lb = 0.0,
                                                                             ub = 1.0),
                                                           lcse = cap,
                                                           ret = ArithmeticReturn(;
                                                                                  ucs = ue))),
                              rdo)
        held = ratio(MaximumRatio())
        @test isa(held.retcode, PortfolioOptimisers.OptimisationSuccess)
        @test dot(held.pa.pr.rr.M[:, 1], held.w) >= 0.9 - 1e-5
        # Which the collapsed scale did not: a floor below the ray reports the same success
        # and hands back weights that break the mandate the caller wrote.
        broken = ratio(MaximumRatio(; kmin = 1e-12))
        @test isa(broken.retcode, PortfolioOptimisers.OptimisationSuccess)
        @test dot(broken.pa.pr.rr.M[:, 1], broken.w) < 0.9 - 1e-3
    end

    @testset "Every JuMP optimiser inherits it" begin
        # They all share `JuMPOptimiser`, so the wiring is one edit rather than one per
        # optimiser.
        for est in (MeanRisk, RiskBudgeting, RelaxedRiskBudgeting)
            res = optimise(est(;
                               opt = JuMPOptimiser(; pe = FactorPrior(), slv = oslv,
                                                   sets = osets, lcse = oece)), rdo)
            @test dot(res.pa.pr.rr.M[:, 1], res.w) <= 0.3 + 1e-6
        end
        # Near Optimal Centering round-trips `lcsr` back into an `lcse` slot through
        # `jump_optimiser_from_attributes`, where the pass-through method for a precomputed
        # `LinearConstraint` keeps it from being projected a second time. Its *constrained*
        # algorithm is the one that carries the linear constraints into the centering model
        # — the default `UnconstrainedNearOptimalCentering` deliberately drops them there,
        # and does so for an asset-space constraint just the same.
        noc = optimise(NearOptimalCentering(; alg = ConstrainedNearOptimalCentering(),
                                            opt = JuMPOptimiser(; pe = FactorPrior(),
                                                                slv = oslv, sets = osets,
                                                                lcse = oece)), rdo)
        @test isapprox(dot(noc.pa.pr.rr.M[:, 1], noc.w), 0.3; atol = 1e-6)
    end
    @testset "Missing loadings throw at the optimiser, ignoring `strict`" begin
        # `strict` governs recoverable per-row name failures. Dropping every row of a factor
        # mandate would yield a feasible, plausible portfolio carrying none of the requested
        # exposure, so this is not that.
        for strict in (false, true)
            mre = MeanRisk(;
                           opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = oslv,
                                               sets = osets, lcse = oece, strict = strict))
            @test_throws PortfolioOptimisers.IsNothingError optimise(mre, rdo)
        end
    end
    @testset "Axis order is checked against the data" begin
        fmr(s) = MeanRisk(;
                          opt = JuMPOptimiser(; pe = FactorPrior(), slv = oslv, sets = s,
                                              lcse = oece))
        # Same factors, wrong order: every row would be attached to the wrong loadings
        # column, and the problem would still solve.
        badf = UniverseSets(;
                            dict = Dict("nx" => ["A", "B", "C"], "nf" => ["VLUE", "MTUM"]))
        @test_throws ArgumentError optimise(fmr(badf), rdo)
        # Same assets, wrong order.
        badx = UniverseSets(;
                            dict = Dict("nx" => ["A", "C", "B"], "nf" => ["MTUM", "VLUE"]))
        @test_throws ArgumentError optimise(fmr(badx), rdo)
        # Different universes entirely is a different diagnosis, and says so.
        shortx = UniverseSets(; dict = Dict("nx" => ["A", "B"], "nf" => ["MTUM", "VLUE"]))
        @test_throws DimensionMismatch optimise(fmr(shortx), rdo)
        # The asset check is new behaviour on a path with nothing to do with factors: it
        # fires with no factor axis declared and no re-basis in sight.
        amr = MeanRisk(;
                       opt = JuMPOptimiser(; slv = oslv,
                                           sets = UniverseSets(;
                                                               dict = Dict("nx" =>
                                                                               ["A", "C",
                                                                                "B"])),
                                           lcse = LinearConstraintEstimator(;
                                                                            val = "A >= 0.05")))
        @test_throws ArgumentError optimise(amr, ReturnsResult(; nx = rdo.nx, X = rdo.X))
        # Only where both sides exist. The factor axis is optional on `UniverseSets`, and an
        # undeclared one is silence, not disagreement.
        nofsets = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"]))
        @test optimise(MeanRisk(;
                                opt = JuMPOptimiser(; pe = FactorPrior(), slv = oslv,
                                                    sets = nofsets)), rdo) isa
              MeanRiskResult
        msg = try
            optimise(fmr(badf), rdo)
        catch e
            sprint(showerror, e)
        end
        @test occursin("factor universe declared under key `nf`", msg)
        @test occursin("first at position 1: `VLUE` vs `MTUM`", msg)
        # Info-leak-safe, as the other constraint messages are: sizes and the first
        # differing pair only, never a universe in full.
        msg = try
            optimise(fmr(badx), rdo)
        catch e
            sprint(showerror, e)
        end
        @test occursin("first at position 2: `C` vs `B`", msg)
        @test !occursin("\"A\"", msg)
    end
    @testset "A schedule of exposure constraints" begin
        # ADR 0030: constraints are problem definition, so `lcse` is `TD_Option` and the
        # widened bound widens with it. Fold two tightens the momentum cap.
        cv = IndexWalkForward(200, 100)
        tight = ExposureConstraintEstimator(;
                                            lce = LinearConstraintEstimator(;
                                                                            val = "MTUM <= 0.05"),
                                            space = fs)
        mre = MeanRisk(;
                       opt = JuMPOptimiser(; pe = FactorPrior(), slv = oslv, sets = osets,
                                           lcse = TimeDependent([oece, tight])))
        preds = cross_val_predict(mre, rdo, cv)
        @test length(preds.pred) == 2
        for (k, cap) in enumerate((0.3, 0.05))
            r = preds.pred[k].res
            @test dot(r.pa.pr.rr.M[:, 1], r.w) <= cap + 1e-6
        end
        # The loadings are refit per fold, which is the whole reason the constraint is
        # re-based at generation time rather than written out once by the caller.
        @test preds.pred[1].res.pa.pr.rr.M != preds.pred[2].res.pa.pr.rr.M
    end
    # A space carries the *source* of its basis (ADR 0047, amended). `FactorSpace` stopped
    # being a singleton and gained `re`, whose precedence is `resolve_factor_regression`'s —
    # the one every other factor consumer already uses.
    stated = Regression(; M = [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0])
    lcem = LinearConstraintEstimator(; val = "MTUM <= 0.3")
    sece(space) = ExposureConstraintEstimator(; lce = lcem, space = space)
    @testset "The space carries its basis source" begin
        @test fieldnames(FactorSpace) == (:re,)
        # `nothing` is the default, and it is what every constraint written before the
        # field did: read the prior's loadings.
        @test isnothing(FactorSpace().re)
        @test isnothing(fs.re)
        # The bound is `RegE_Reg` — a fitted result or an estimator. A bare matrix is not a
        # basis source, it is a basis, and the space deliberately does not hold one.
        @test_throws TypeError FactorSpace(; re = M)
        @test_throws TypeError FactorSpace(; re = "MTUM")
        @test isa(FactorSpace(; re = rr).re, Regression)
        @test isa(FactorSpace(; re = StepwiseRegression()).re, StepwiseRegression)
    end
    @testset "The precedence is resolve_factor_regression's" begin
        # 1. A precomputed result wins outright: the caller has stated the answer.
        @test vec(linear_constraints(sece(FactorSpace(; re = stated)), sets; rr = rr).ineq.A) ==
              stated.M[:, 1]
        # 2. The prior's loadings are next — and a stated *estimator* loses to them, which
        #    is the documented asymmetry, not an oversight. Pass a fitted `Regression` to
        #    override a factor prior.
        @test vec(linear_constraints(sece(FactorSpace(; re = StepwiseRegression())), sets;
                                     rr = rr).ineq.A) == M[:, 1]
        # 3. `re === nothing` is bit-identical to the singleton it replaced.
        @test linear_constraints(sece(FactorSpace()), sets; rr = rr).ineq.A ==
              linear_constraints(sece(fs), sets; rr = rr).ineq.A
    end
    @testset "The standalone route cannot refit" begin
        # `linear_constraints(ece, sets; rr = ...)` receives the loadings and never the data
        # behind them, so the third arm is unreachable here and says so rather than
        # silently doing nothing.
        ece_fit = sece(FactorSpace(; re = StepwiseRegression()))
        @test_throws PortfolioOptimisers.IsNothingError linear_constraints(ece_fit, sets)
        msg = try
            linear_constraints(ece_fit, sets)
        catch e
            sprint(showerror, e)
        end
        # The message names the estimator that could not be fitted and both fixes.
        @test occursin("StepwiseRegression", msg)
        @test occursin("Regression(; M = ...)", msg)
        @test occursin("JuMPOptimiser", msg)
        # A space that names no source at all is the older diagnosis, and it now points at
        # the field as a third remedy.
        msg = try
            linear_constraints(sece(fs), sets)
        catch e
            sprint(showerror, e)
        end
        @test occursin("space.re === nothing", msg)
        @test occursin("FactorSpace(; re = StepwiseRegression())", msg)
    end
    @testset "An estimator refits: a factor mandate with no factor prior" begin
        # The capability the field buys. `EmpiricalPrior` carries no regression, so before
        # `re` this combination threw — it is the case the older test above asserts.
        ece_fit = ExposureConstraintEstimator(; lce = lcem,
                                              space = FactorSpace(;
                                                                  re = StepwiseRegression()))
        res = optimise(MeanRisk(;
                                opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = oslv,
                                                    sets = osets, lcse = ece_fit)), rdo)
        # The prior really carries none: the basis came from the refit and nowhere else.
        @test isnothing(res.pa.pr.rr)
        fitted = regression(StepwiseRegression(), rdo)
        @test vec(res.lcsr.ineq.A) ≈ fitted.M[:, 1]
        # And it binds, measured against the loadings the refit produced.
        @test dot(fitted.M[:, 1], res.w) <= 0.3 + 1e-6
        # A stated result at the optimiser is the pinned case: the row is exactly `M * a`
        # for the loadings handed in, whatever the prior would have fitted.
        pin = Regression(; M = [1.0 0.0; 0.0 1.0; 1.0 1.0])
        resp = optimise(MeanRisk(;
                                 opt = JuMPOptimiser(; pe = FactorPrior(), slv = oslv,
                                                     sets = osets,
                                                     lcse = ExposureConstraintEstimator(;
                                                                                        lce = lcem,
                                                                                        space = FactorSpace(;
                                                                                                            re = pin)))),
                        rdo)
        @test vec(resp.lcsr.ineq.A) == pin.M[:, 1]
        @test vec(resp.lcsr.ineq.A) != resp.pa.pr.rr.M[:, 1]
    end
    @testset "A stated basis is viewed where the universe is sliced" begin
        # The loadings are assets-major, so a subset of assets is a subset of rows and the
        # projection over the subset is exact. This is why an inner solve views rather than
        # refuses.
        ece_st = ExposureConstraintEstimator(; lce = lcem,
                                             space = FactorSpace(;
                                                                 re = Regression(; M = Mo)))
        i = [1, 3]
        v = PortfolioOptimisers.port_opt_view(ece_st, i)
        @test v.space.re.M == Mo[i, :]
        # A stated basis is usually just `M`. Viewing one used to crash on `view(nothing,
        # i)` for `b`, and to materialise `L` as a copy of `M` because the `swap(L, M)`
        # property rule makes `re.L` never `nothing` — both fixed, and both only reachable
        # once a constraint space could hold a hand-written `Regression`.
        @test isnothing(getfield(v.space.re, :L))
        @test isnothing(getfield(v.space.re, :b))
        rfull = Regression(; M = Mo, L = Mo .* 2, b = [1.0, 2.0, 3.0])
        @test PortfolioOptimisers.port_opt_view(rfull, i).L == (Mo .* 2)[i, :]
        @test PortfolioOptimisers.port_opt_view(rfull, i).b == [1.0, 3.0]
        # The wrapped shape is written in the space's names, so an asset index means
        # nothing to it and it is passed through.
        @test v.lce === ece_st.lce
        # A space that states no source, and a space holding an estimator, come back with
        # nothing to slice.
        @test isnothing(PortfolioOptimisers.port_opt_view(oece, i).space.re)
        @test isa(PortfolioOptimisers.port_opt_view(sece(FactorSpace(;
                                                                     re = StepwiseRegression())),
                                                    i).space.re, StepwiseRegression)
        # A vector maps elementwise. Without the method the universal fallback would `view`
        # the vector itself, slicing a list of constraints by asset indices.
        vv = PortfolioOptimisers.port_opt_view(PortfolioOptimisers.EcE_LcE_Lc[ece_st, lcem],
                                               i)
        @test length(vv) == 2
        @test vv[1].space.re.M == Mo[i, :]
        @test vv[2] === lcem
        # A precomputed `LinearConstraint` in `lcse` is passed through unchanged. The
        # identity is what the slot already did; slicing `A` here would change a path this
        # work does not touch. See ADR 0047's 2026-08-16 basis-source amendment.
        lc = LinearConstraint(;
                              ineq = PartialLinearConstraint(; A = transpose(Mo[:, 1:1]),
                                                             B = [0.3]))
        @test PortfolioOptimisers.port_opt_view(lc, i) === lc
    end
    @testset "A stated basis is refused where the universe is replaced" begin
        # An NCO outer solve writes cluster names over `sets.dict[xkey]` rather than
        # slicing it, so no view of asset-major loadings can follow it.
        pinned = ExposureConstraintEstimator(; lce = lcem,
                                             space = FactorSpace(;
                                                                 re = Regression(; M = Mo)))
        fitting = ExposureConstraintEstimator(; lce = lcem,
                                              space = FactorSpace(;
                                                                  re = StepwiseRegression()))
        jopt(l) = MeanRisk(;
                           opt = JuMPOptimiser(; pe = FactorPrior(), slv = oslv,
                                               sets = osets, lcse = l))
        @test_throws ArgumentError PortfolioOptimisers.assert_external_optimiser(jopt(pinned))
        # The inner path views instead, so it accepts what the outer refuses.
        @test isnothing(PortfolioOptimisers.assert_internal_optimiser(jopt(pinned)))
        # An estimator is refused nowhere: it refits against whatever universe it is
        # handed, which is exactly the remedy the message names.
        @test isnothing(PortfolioOptimisers.assert_external_optimiser(jopt(fitting)))
        # And a space that states no source is unaffected — no configuration that was legal
        # before this field became refusable.
        @test isnothing(PortfolioOptimisers.assert_external_optimiser(jopt(oece)))
        # A vector is walked, not ignored.
        @test_throws ArgumentError PortfolioOptimisers.assert_external_optimiser(jopt(PortfolioOptimisers.EcE_LcE_Lc[lcem,
                                                                                                                     pinned]))
        # The refusal is wired at construction, in the shape of the `opt.re` and
        # `opt.rba.re` refusals it copies.
        hclust = ClustersEstimator()
        @test_throws ArgumentError NestedClustered(; pe = FactorPrior(), cle = hclust,
                                                   opti = jopt(oece), opto = jopt(pinned))
        @test NestedClustered(; pe = FactorPrior(), cle = hclust, opti = jopt(pinned),
                              opto = jopt(oece)) isa NestedClustered
        # `Stacking` routes its outer optimiser through the same assertion, and its outer
        # universe is the synthetic sub-portfolios, so the refusal is right there too.
        @test_throws ArgumentError Stacking(; opti = [jopt(oece)], opto = jopt(pinned))
        # `SubsetResampling` asserts the *internal* contract, because it slices. A stated
        # basis is therefore legal under it, and correct: the view follows the subset.
        @test SubsetResampling(; subset_size = 0.8, opt = jopt(pinned)) isa SubsetResampling
    end
end

@testset "The two re-basis routes agree (issue #513)" begin
    using PortfolioOptimisers, Test, Logging, LinearAlgebra
    #        F1    F2
    #   A    0.5   0.2
    #   B    0.1   0.9
    #   C    0.3   0.4
    M = [0.5 0.2; 0.1 0.9; 0.3 0.4]
    rr = Regression(; M = M)
    sets = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"], "nf" => ["F1", "F2"]))
    # `constraint_row_term` projects each term as the row is assembled;
    # `project_linear_constraint` projects the assembled factor-width row wholesale. The
    # identity `aᵀ wf = (M a)ᵀ wa` says the two must land on the same row.
    eqn = "F1 + 2*F2 <= 1"
    stepwise = linear_constraints(eqn, sets, "nf"; rr = rr)
    wholesale = PortfolioOptimisers.project_linear_constraint(linear_constraints(eqn, sets,
                                                                                 "nf"), M)
    @test collect(stepwise.ineq.A) ≈ collect(wholesale.ineq.A)
    @test stepwise.ineq.B == wholesale.ineq.B
    @test vec(collect(stepwise.ineq.A)) ≈ M[:, 1] + 2 * M[:, 2]

    # A repeated factor name contributes every column bearing it, in both routes, which is
    # how the asset path treats a duplicated asset name.
    dsets = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"], "nf" => ["F1", "F1"]))
    dstep = linear_constraints("F1 <= 1", dsets, "nf"; rr = rr)
    dwhole = PortfolioOptimisers.project_linear_constraint(linear_constraints("F1 <= 1",
                                                                              dsets, "nf"),
                                                           M)
    @test collect(dstep.ineq.A) ≈ collect(dwhole.ineq.A)
    @test vec(collect(dstep.ineq.A)) ≈ M[:, 1] + M[:, 2]

    # The re-based row is asset-length whatever the row was written in.
    @test PortfolioOptimisers.constraint_row_length(nothing, ["F1", "F2"]) == 2
    @test PortfolioOptimisers.constraint_row_length(rr, ["F1", "F2"]) == 3
end
