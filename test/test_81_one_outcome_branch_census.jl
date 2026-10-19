#=
`code_health/CodeHealth.jl` holds the one parser every census reads the source text with. The load
sits OUTSIDE the `@testset` on purpose: `include` defines methods, and a method defined inside one
top-level statement is not visible to a call in that same statement. The module wrapper keeps that
module's own names out of the worker module. `test_79_type_decided_early_return_census.jl` loads it the same way.
=#
module OneOutcomeBranchCensusHealth
include(joinpath(@__DIR__, "..", "code_health", "CodeHealth.jl"))
end

@testset "One outcome branch census: branches that share an outcome are one branch (#1753)" begin
    using Test

    #=
    Issue #1753. `mean_test_answers` held three `if` blocks in a row, and each returned `false`.
    `.github/instructions/julia-source-code.instructions.md` § *Branches that share an outcome
    are one branch* is the Authority.

    `||` and `&&` evaluate from left to right and stop at the first operand that decides the
    result. So a merged condition tests the same conditions in the same order, and it costs the
    same at run time. The merge removes a branch from the source, not from the compiled code.

    The census reads four shapes:

    - two exits in a row with the same body, which ends in a jump: `if c … end` with no `else`,
      `c && …` or `c || …`, and a jump is `return`, `continue`, `break`, `throw` or `error`;
    - two adjacent arms of an `if` chain with the same body;
    - an `if` with no `else` whose body is only an `if` with no `else`;
    - an `else` whose body is only an `if`, a ternary included.

    It does not read two shapes, because the merge makes the text harder to read:

    - an exit whose condition calls a function whose name ends in `!`, after another exit. In a
      merged `a || f!(x)`, the change of `x` happens only when `a` is false, and the reader of
      the condition does not see that;
    - an `else` whose `if` tests the same condition as the last `if` of the first arm. The two
      arms are then a table of two rows and two columns, and the nesting shows the table.

    The commits of #1753 merged every site that existed, so every other site fails the census.
    =#
    CH = OneOutcomeBranchCensusHealth.CodeHealth
    root = normpath(joinpath(@__DIR__, ".."))

    strip_lnn(x) = x
    function strip_lnn(x::Expr)
        return Expr(x.head, (strip_lnn(a) for a in x.args if !(a isa LineNumberNode))...)
    end
    function stmts(b)
        return if b isa Expr && b.head === :block
            [s for s in b.args if !(s isa LineNumberNode)]
        else
            Any[b]
        end
    end
    # The parser wraps the condition of an `elseif` in a block that holds its line number.
    condition(c) = (s = stmts(c); length(s) == 1 ? s[1] : c)

    callee(f::Symbol) = f
    function callee(f::Expr)
        return f.head === :. && f.args[end] isa QuoteNode ? f.args[end].value : nothing
    end
    callee(f) = nothing

    isjump(x) = false
    function isjump(x::Expr)
        return x.head in (:return, :continue, :break) ||
               (x.head === :call && callee(x.args[1]) in (:throw, :error, :rethrow))
    end
    jumps(b) = (s = stmts(b); !isempty(s) && isjump(s[end]))

    # The negation `!` is an operator, not a function that changes its argument.
    mutates(x) = false
    function mutates(x::Expr)
        if x.head === :call
            f = callee(x.args[1])
            if !isnothing(f) && f !== :! && endswith(String(f), "!")
                return true
            end
        end
        return any(mutates, x.args)
    end

    # The condition under which a statement exits, and the body of the exit, or `nothing`. The
    # form `c || return x` exits when `c` is false, so its condition is `!c`, and two exits
    # merge into one whose condition joins theirs with `||`.
    exit_of(st) = nothing
    function exit_of(st::Expr)
        if st.head === :if && length(st.args) == 2 && jumps(st.args[2])
            return (st.args[1], strip_lnn(st.args[2]))
        end
        if !(st.head in (:&&, :||)) || !(st.args[end] isa Expr)
            return nothing
        end
        c = st.head === :&& ? st.args[1] : Expr(:call, :!, st.args[1])
        tail = st.args[end]
        if isjump(tail)
            return (c, strip_lnn(Expr(:block, tail)))
        end
        # The parser nests `a && b && return x` as `a && (b && return x)`.
        e = tail.head === st.head ? exit_of(tail) : nothing
        return isnothing(e) ? nothing : (Expr(:&&, c, e[1]), e[2])
    end

    isif(x) = x isa Expr && x.head === :if
    noelse(x) = isif(x) && length(x.args) == 2

    function sites!(sites, ex::Expr, name, rel)
        site(shape, a, b) = push!(sites,
                                  string(rel, ": ", name, ": ", shape, ": ",
                                         strip_lnn(condition(a)), " / ",
                                         strip_lnn(condition(b))))
        if ex.head === :block
            prev = nothing
            for st in stmts(ex)
                e = exit_of(st)
                if !isnothing(e) && !isnothing(prev) && e[2] == prev[2] && !mutates(e[1])
                    site("exits", prev[1], e[1])
                end
                prev = e
            end
        end
        if ex.head in (:if, :elseif) && length(ex.args) == 3
            alt = ex.args[3]
            if alt isa Expr &&
               alt.head === :elseif &&
               strip_lnn(ex.args[2]) == strip_lnn(alt.args[2])
                site("arms", ex.args[1], alt.args[1])
            end
            inner = stmts(alt)
            if alt isa Expr && alt.head === :block && length(inner) == 1 && isif(inner[1])
                first_arm = stmts(ex.args[2])
                last = isempty(first_arm) ? nothing : first_arm[end]
                if !(isif(last) &&
                     strip_lnn(condition(last.args[1])) == strip_lnn(inner[1].args[1]))
                    site("else-if", ex.args[1], inner[1].args[1])
                end
            end
        end
        if ex.head in (:if, :elseif) && length(ex.args) == 2
            inner = stmts(ex.args[2])
            if length(inner) == 1 && noelse(inner[1])
                site("nested", ex.args[1], inner[1].args[1])
            end
        end
        return sites
    end

    function isdefinition(ex::Expr)
        return ex.head === :function && length(ex.args) == 2 ||
               (ex.head === :(=) && CH.is_signature(ex.args[1]))
    end

    census!(sites, x, name, rel) = sites
    function census!(sites, ex::Expr, name, rel)
        if isdefinition(ex)
            name = CH.defname(ex.args[1])
        end
        sites!(sites, ex, name, rel)
        for a in ex.args
            census!(sites, a, name, rel)
        end
        return sites
    end
    census(ast, rel) = census!(String[], ast, "<toplevel>", rel)

    # A witness for each shape, and for each shape the census does not read.
    witness(src) = census(Meta.parseall(src), "w")
    @test witness("function f(x)\n    if x < 0\n        return 0\n    end\n    if isnan(x)\n        return 0\n    end\n    return x\nend") ==
          ["w: f: exits: x < 0 / isnan(x)"]
    @test witness("function f(x)\n    x < 0 && return 0\n    isnan(x) || throw(x)\n    x > 1 || throw(x)\n    return x\nend") ==
          ["w: f: exits: !(isnan(x)) / !(x > 1)"]
    @test witness("function f(x)\n    for i in x\n        i < 0 && continue\n        iszero(i) && continue\n        g(i)\n    end\nend") ==
          ["w: f: exits: i < 0 / iszero(i)"]
    @test witness("f(i, j) = if i == j\n    0\nelseif i == -j\n    0\nelse\n    1\nend") ==
          ["w: f: arms: i == j / i == -j"]
    @test witness("function f(x)\n    if x > 0\n        if x < 1\n            g(x)\n        end\n    end\n    return x\nend") ==
          ["w: f: nested: x > 0 / x < 1"]
    @test witness("f(x) = if isempty(x)\n    0\nelse\n    length(x) == 1 ? x[1] : sum(x)\nend") ==
          ["w: f: else-if: isempty(x) / length(x) == 1"]
    # A different body, a body that does not jump, a change of state in a later condition, an
    # outer `else`, and a table of two by two.
    @test isempty(witness("function f(x)\n    x < 0 && return 0\n    isnan(x) && return 1\n    return x\nend"))
    @test isempty(witness("function f(x)\n    if x < 0\n        g(x)\n    end\n    if isnan(x)\n        g(x)\n    end\n    return x\nend"))
    @test isempty(witness("function f(x)\n    ok(x) && return x\n    repair!(x) && return x\n    return h(x)\nend"))
    @test isempty(witness("function f(x)\n    if x > 0\n        if x < 1\n            g(x)\n        end\n    else\n        h(x)\n    end\nend"))
    @test isempty(witness("function f(x, p)\n    if isnothing(x)\n        if isnothing(p)\n            g()\n        else\n            g(p)\n        end\n    else\n        if isnothing(p)\n            h(x)\n        else\n            h(x, p)\n        end\n    end\nend"))

    #=
    A site whose merge makes the text harder to read for a reason that the two exceptions above
    do not state. Add it here with its reason.
    =#
    KEEP = String[]

    found = String[]
    nfiles = 0
    for d in ("src", "ext"), (r, _, fs) in walkdir(joinpath(root, d)), f in fs
        endswith(f, ".jl") || continue
        nfiles += 1
        append!(found,
                census(CH.parse_file(joinpath(r, f); root), relpath(joinpath(r, f), root)))
    end

    # A census that reads no file passes on an empty list.
    @test nfiles > 0

    # A new entry here is a set of branches that share an outcome. Merge them into one branch.
    # If the merge makes the text harder to read, add the site to `KEEP` with its reason.
    @test sort!(setdiff(found, KEEP)) == String[]

    # A stale entry names a site that is gone. Remove it.
    @test sort!(setdiff(KEEP, found)) == String[]
end
