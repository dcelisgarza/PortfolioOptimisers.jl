@testset "Constructor docstrings do not drift from the signatures they copy" begin
    using Test

    # `field_dict` in `01_Base.jl` centralises the PROSE of a field, but the keyword
    # SIGNATURE of a constructor is copied by hand into each type's `# Constructors`
    # docstring block. The copy and the code drift silently, and on 2026-08-17 twenty
    # blocks had drifted: seven carried a stale default (`SubsetResampling` advertised
    # `subset_size = 0.5`, `n_subsets = 100` and `max_comb = 1000` against a code that has
    # held `0.8`, `2` and `1_000_000_000` since `67c236315`), and thirteen more carried a
    # stale type annotation, a stale keyword order, or had lost a keyword entirely.
    #
    # Generating the block instead would be the stronger fix, but it is blocked:
    # `DocStringExtensions.TYPEDSIGNATURES`, used 578 times in `src/`, renders the
    # signature WITHOUT default values, and the defaults are the part that drifts.
    #
    # So the block stays hand-written and this census makes the drift loud. It reads the
    # source TEXT only -- it parses `src/**/*.jl` with `Meta.parseall` and never reflects
    # on a loaded binding -- so it costs no compilation and covers a type the day its
    # docstring is written.
    #
    # Four differences are cosmetic and are normalised away, because a census that fails
    # on them is a census that gets deleted:
    #
    #   1. module qualification    `FLoops.ThreadedEx()`  ==  `ThreadedEx()`
    #   2. the pi spelling         `pi`                   ==  `π`
    #   3. numeric literal width   `1`                    ==  `1.0`
    #   4. operator spacing        `2/3`                  ==  `2 / 3`
    #
    # (1) and (2) are rewritten in `normex`; (3) sends every literal through `Float64`;
    # (4) falls out of `string`, which re-prints a parsed expression canonically.

    srcdir = normpath(joinpath(@__DIR__, "..", "src"))

    # ------------------------------------------------------------- normalisation

    function normex(x)
        if isa(x, Expr)
            if x.head === :. && length(x.args) == 2 && isa(x.args[2], QuoteNode)
                return normex(x.args[2].value)   # drop the module qualification
            end
            return Expr(x.head, map(normex, x.args)...)
        elseif isa(x, QuoteNode)
            return QuoteNode(normex(x.value))
        elseif isa(x, Symbol)
            return x === :pi ? :π : x
        elseif isa(x, Bool)
            return x                              # `Bool <: Integer`; keep it a Bool
        elseif isa(x, Integer) || isa(x, AbstractFloat)
            return Float64(x)
        else
            return x
        end
    end
    normstr(x) = string(normex(x))

    # One keyword, as `name = default` or as a bare `name::T`.
    function kwstring(p)
        return if isa(p, Expr) && p.head === :kw
            string(normstr(p.args[1]), " = ", normstr(p.args[2]))
        else
            normstr(p)
        end
    end

    # The callee of `A.B.f(…)` or `f(…)`, as a bare `Symbol`.
    function callname(c)
        n = c.args[1]
        while isa(n, Expr)
            n = n.args[1]
        end
        return isa(n, Symbol) ? n : nothing
    end

    # `(name, keywords)` of a call that takes keywords, or `nothing` if it takes none.
    function kwspec(c)
        isa(c, Expr) && c.head === :call || return nothing
        params = nothing
        for a in c.args[2:end]
            if isa(a, Expr) && a.head === :parameters
                params = a
            end
        end
        params === nothing && return nothing
        name = callname(c)
        name === nothing && return nothing
        return (name, String[kwstring(p) for p in params.args])
    end

    # ------------------------------------------------------------- the doc side

    # Every `# Constructors` block: the blank lines after the heading are skipped, then
    # the indented body is taken up to the first line that is neither blank nor indented.
    function doc_blocks(text)
        lines = split(text, '\n')
        out = Tuple{Int, String}[]
        i = 1
        while i <= length(lines)
            if strip(lines[i]) == "# Constructors"
                j = i + 1
                while j <= length(lines) && isempty(strip(lines[j]))
                    j += 1
                end
                start = j
                buf = String[]
                while j <= length(lines) &&
                      (isempty(strip(lines[j])) || startswith(lines[j], "    "))
                    push!(buf, String(lines[j]))
                    j += 1
                end
                while !isempty(buf) && isempty(strip(buf[end]))
                    pop!(buf)
                end
                isempty(buf) || push!(out, (start, join(buf, "\n")))
                i = j
            else
                i += 1
            end
        end
        return out
    end

    function dedent(block)
        return join([startswith(l, "    ") ? l[5:end] : l for l in split(block, '\n')],
                    "\n")
    end

    # The block lives inside a string literal, so `"` and `$` reach the file escaped.
    unescape(block) = replace(block, "\\\"" => "\"", "\\\$" => "\$")

    # Strip `where {…}` and a `::T` return annotation off a definition's signature.
    function unwrap_sig(sig)
        while isa(sig, Expr) && (sig.head === :where || sig.head === :(::))
            sig = sig.args[1]
        end
        return sig
    end

    # A block reads `T(; kw…) -> T`, so the parse is an `->` whose left side is the call.
    # A block that declares several constructors parses as a `:block` of them.
    function documented(text)
        out = Tuple{Int, Symbol, Vector{String}}[]
        bad = Tuple{Int, String}[]
        for (ln, block) in doc_blocks(text)
            src = dedent(unescape(block))
            parsed = try
                Meta.parseall(src)
            catch e
                push!(bad, (ln, sprint(showerror, e)))
                continue
            end
            found = false
            for e in parsed.args
                isa(e, Expr) || continue
                c = e
                if c.head === :->
                    c = c.args[1]
                end
                if isa(c, Expr) && c.head === :block
                    for cc in c.args
                        isa(cc, Expr) || continue
                        k = kwspec(cc.head === :-> ? cc.args[1] : cc)
                        if k !== nothing
                            push!(out, (ln, k[1], k[2]))
                            found = true
                        end
                    end
                    continue
                end
                k = kwspec(c)
                if k !== nothing
                    push!(out, (ln, k[1], k[2]))
                    found = true
                end
            end
            found || push!(bad, (ln, "no keyword call found"))
        end
        return out, bad
    end

    # ------------------------------------------------------------- the code side

    function collect_defs!(out, ex)
        isa(ex, Expr) || return nothing
        if ex.head === :function || ex.head === :(=)
            c = unwrap_sig(ex.args[1])
            k = kwspec(c)
            k === nothing || push!(out, k)
        end
        for a in ex.args
            collect_defs!(out, a)
        end
        return nothing
    end

    # ------------------------------------------------------------- the census

    files = String[]
    for (root, _, fs) in walkdir(srcdir), f in fs
        endswith(f, ".jl") && push!(files, joinpath(root, f))
    end
    sort!(files)

    real = Dict{Symbol, Vector{Vector{String}}}()
    docs = Tuple{String, Int, Symbol, Vector{String}}[]
    parse_errors = Tuple{String, Int, String}[]
    positional = Tuple{String, Int}[]
    for f in files
        text = read(f, String)
        defs = Tuple{Symbol, Vector{String}}[]
        collect_defs!(defs, Meta.parseall(text))
        for (n, spec) in defs
            push!(get!(real, n, Vector{Vector{String}}()), spec)
        end
        d, b = documented(text)
        rel = relpath(f, srcdir)
        for (ln, n, spec) in d
            push!(docs, (rel, ln, n, spec))
        end
        for (ln, msg) in b
            # A block that declares only positional constructors is out of scope; a block
            # that fails to PARSE is a defect in the block (a missing comma, say).
            if msg == "no keyword call found"
                push!(positional, (rel, ln))
            else
                push!(parse_errors, (rel, ln, msg))
            end
        end
    end

    # An extractor that silently stops finding anything must not pass. The figures are
    # from 2026-08-17: 197 files, 1028 keyword-taking definitions, 321 documented keyword
    # signatures. The floors are loose enough to survive ordinary growth.
    @test length(files) >= 190
    @test sum(length, values(real); init = 0) >= 1000
    @test length(docs) >= 300

    # A `# Constructors` block must be valid Julia. `OptimEntropyPooling` was not: its
    # last keyword had lost its trailing comma.
    @test isempty(parse_errors)
    if !isempty(parse_errors)
        println("Unparsable `# Constructors` blocks:")
        for (f, ln, msg) in parse_errors
            println("  ", f, ":", ln, "  ", msg)
        end
    end

    # Every documented keyword signature must equal one real definition of that name.
    # Several definitions can share a name (an outer constructor and a `@concrete` inner
    # one), so matching ANY of them is enough.
    mismatched = Tuple{String, Int, Symbol, Vector{String}, Vector{Vector{String}}}[]
    unmatched = Tuple{String, Int, Symbol}[]
    for (f, ln, n, spec) in docs
        cands = get(real, n, nothing)
        if cands === nothing
            push!(unmatched, (f, ln, n))
        elseif !any(==(spec), cands)
            push!(mismatched, (f, ln, n, spec, cands))
        end
    end

    @test isempty(unmatched)
    if !isempty(unmatched)
        println("Documented constructors with no definition in `src/`:")
        for (f, ln, n) in unmatched
            println("  ", f, ":", ln, "  ", n)
        end
    end

    @test isempty(mismatched)
    if !isempty(mismatched)
        println("`# Constructors` blocks that disagree with the code. Correct the block ",
                "to match the signature, or correct both if the signature is wrong:")
        for (f, ln, n, spec, cands) in mismatched
            println("  ", f, ":", ln, "  ", n)
            println("     documented: ", spec)
            for c in cands
                println("     code      : ", c)
            end
        end
    end
end
