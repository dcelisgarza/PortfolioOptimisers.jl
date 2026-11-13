#=
`code_health/forward_survey.jl` reads the methods that the wrapper standard reads, and gives each
one the verdict that the rulings of map #2072 give it (#2084, #2086). This file is the fixture
adapter of the survey: each detector is driven over a small tree in a temporary directory, on a
shape that it must report and on a near shape that it must leave alone or judge another way.

The load sits OUTSIDE the `@testset`, for the reason `test_49_coverage_attribution_census.jl`
gives: `include` defines methods, and a method defined inside one top-level statement is not
visible to a call in that same statement. The module wrapper keeps the names of the script out of
the worker module.
=#
module ForwardSurvey
include(joinpath(@__DIR__, "..", "code_health", "forward_survey.jl"))
end

using Test

@testset "Forward survey: each detector gives the verdict of its ruling (#2086)" begin
    FS = ForwardSurvey

    # A docstring with an `# Interfaces` section that names the hook `hook_fn`.
    q3 = "\"\"\""
    hook_doc = join([q3, "    hook_fn(x)", "", "A hook.", "", "# Interfaces", "",
                     "## `hook_fn`", "", "A subtype implements it.", q3], '\n')
    source = """
             export verb, verb2, pubf, pubk, mk, kwv
             struct A end
             struct B end
             struct R
                 a
                 b
                 c
             end
             R(; a, b, c) = R(a, b, c)
             struct S
                 a
                 b
                 c
             end
             struct T2
                 a
             end
             struct T3
                 a
                 b
             end

             # interface: `helper` is shared by `f` and `k`.
             helper(x) = x + 1
             f(x::A) = helper(x)
             f(x::B) = helper(x)
             k(x::A) = helper(x)

             # per-form: only `w` calls each callee.
             w(x::A) = only_a(x)
             w(x::B) = only_b(x)
             only_a(x) = x + 1
             only_b(x) = x + 2

             # nothing pair: case 1, and case 2 or 3.
             p(::Nothing) = nothing
             p(x::A) = q(x)
             q(::Nothing) = nothing
             q(x::A) = x + 1
             p2(::Nothing) = nothing
             p2(x::A) = q2(x)
             q2(x::A) = x + 1

             # default
             trait(x) = false
             ident(x) = x
             noop(x) = nothing
             step_td(x) = false
             step_td(x::A) = owner_td(x)
             owner_td(x::A) = true

             # step functions in front of an owner (#2109)
             walk_path(x::A) = walk_path(x.a)
             walk_path(v::B) = elem_path(walk_path, v)
             walk_path(::Any) = nothing
             elem_path(f, v::B) = 1
             step_q(x::A) = owner_q(x)
             step_q(::Any) = nothing
             owner_q(x::A) = 1
             step_j(x::A) = owner_j!(x)
             step_j(::Any) = nothing
             owner_j!(x::A) = 1
             rewrap(w::T3, e) = T3(e, w.b)
             rewrap(::Any, e) = e
             step_i(x::A) = owner_i(x)
             step_i(x) = x
             owner_i(x::A) = 1
             factory_child(v::A) = factory(v)
             factory_child(v) = v
             factory(v::A) = 1

             # untyped fallbacks with no owner (#2108)
             setter!(args...) = nothing
             setter!(m::A, x::B) = 1
             assert_pos(args...) = nothing
             assert_pos(x::A, s::Symbol = :x) = 1
             assert_role(::Any) = nothing
             assert_role(x::A) = throw(ArgumentError("a role"))
             refuse(::Any, ::Any) = nothing
             refuse(x::A, y::B) = throw(ArgumentError("refused"))
             fill_slot!(::Any) = nothing
             fill_slot!(x::A) = 1
             pair_op(::Any, ::Any) = nothing
             pair_op(x::A, y::A) = 1
             leaf_pass(x, ::Any) = x
             leaf_pass(x::A, y::B) = 1
             timestamp_positions(::Any, i) = i
             timestamp_positions(::Any, i::A) = 1

             # rebuild
             set_c(r::R, v) = R(r.a, r.b, v)
             set_s(s::S, v) = S(s.a, s.b, v)
             set_u(r, v) = R(r.a, r.b, v)
             two(x::T3, v) = T3(x.a, v)

             # self-forward
             sf(x::A) = 1
             sf(x::A, ::B) = sf(x)
             pubf(x::A, y::B) = pubf(x, string(y))
             privf(x::A, y::B) = privf(x, string(y))
             pubk(x::A; s = 1) = pubk(s, x)
             fill1(x::A) = fill1(x, 1)
             walk(x::A) = walk(x.child)

             # worker: a private worker with one caller, and a method around a driver.
             verb(x::A) = verb_worker(x)
             verb_worker(x) = x + 1
             $hook_doc
             hook_fn(x::A) = 1
             function verb2(x::B)
                 y = hook_fn(x)
                 return y + 1
             end
             verb2(x::A) = hook_fn(x)

             # preset
             mk(x::A) = T2(x)

             # in front of a constructor: one private method, and two (#2112, rule 3)
             mk_priv(x::T3, v) = T3(; a = x.a, b = v)
             mk_two(x::A) = T2(x)
             mk_two(x::B) = T2(x)

             # keyword form of a public verb
             kwv(x::A; s = 1) = kwv_worker(x, s)
             kwv_worker(x, s) = x + s
             """
    census = """
             kept = Dict{Symbol, String}(:mk => "public preset")
             kept_forwards = Dict{Tuple{Symbol, Symbol}, String}((:w, :only_a) => "(b) a fixture reason")
             interface_functions = Dict{Symbol, String}()
             pending_forwards = Set{Tuple{Symbol, Symbol}}([(:w, :only_b)])
             """
    found, out, status = mktempdir() do dir
        mkpath(joinpath(dir, "src"))
        mkpath(joinpath(dir, "test"))
        write(joinpath(dir, "src", "a.jl"), source)
        write(joinpath(dir, "test", "census.jl"), census)
        c = joinpath(dir, "test", "census.jl")
        found = FS.survey(; root = dir, files = ["src/a.jl"], census = c)
        io = IOBuffer()
        status = FS.main(["src/a.jl"]; io, root = dir, files = ["src/a.jl"])
        return found, String(take!(io)), status
    end
    # The findings of `name`, as `(shape, callee, verdict)`.
    of(name) = [(x.shape, x.callee, x.verdict) for x in found if x.name === name]
    first_of(name) = only(x for x in found if x.name === name)

    @testset "interface and per-form (#2078, rule 3)" begin
        @test of(:f) == [("interface", :helper, "keep (interface)"),
                         ("interface", :helper, "keep (interface)")]
        @test of(:w) == [("per-form", :only_a, "go"), ("per-form", :only_b, "go")]
        # `k` has one method, and its callee is shared: a second name.
        @test of(:k) == [("interface", :helper, "go")]
    end

    @testset "nothing pair (#2077)" begin
        @test of(:p) == [("nothing pair", :q, "go")]
        @test of(:p2) == [("nothing pair", :q2, "review")]
    end

    @testset "default (#2074, rule 2)" begin
        @test of(:trait) == [("default", nothing, "keep (default)")]
        @test of(:ident) == [("default", nothing, "keep (default)")]
        # One slot, and no `!`: a query owns `nothing`, the answer "none" (#2108).
        @test of(:noop) == [("default", nothing, "keep (default)")]
        # A step function in front of an owner: the owner takes `false`.
        @test ("default", :owner_td, "go") in of(:step_td)
    end

    @testset "step functions in front of an owner (#2109)" begin
        # A walk owns its leaf answer: its callee is not its owner.
        @test ("default", :elem_path, "keep (default)") in of(:walk_path)
        @test occursin("a walk owns its leaf",
                       only(x
                            for x in found if x.name === :walk_path && x.shape == "default").why)
        # `nothing` in front of an owner: a query or a check owns it, so the owner takes it.
        @test ("default", :owner_q, "go") in of(:step_q)
        # An owner that does a job owns no default: a person rules it.
        @test ("default", :owner_j!, "review") in of(:step_j)
        # The input in front of a constructor stays as (b): the constructor owns no default.
        @test ("default", :T3, "keep (b)") in of(:rewrap)
        # The input in front of a function: a person rules it, and `RULED_DEFAULTS` holds it.
        @test ("default", :owner_i, "review") in of(:step_i)
        @test ("default", :factory, "go") in of(:factory_child)
        @test only(x for x in found if x.name === :factory_child && x.shape == "default").why ==
              FS.RULED_DEFAULTS[:factory_child][2]
    end

    @testset "untyped fallbacks with no owner (#2108)" begin
        # An `args...` fallback goes, for a contract verb and for a check alike.
        @test of(:setter!) == [("default", nothing, "go")]
        @test of(:assert_pos) == [("default", nothing, "go")]
        @test occursin("every arity", first_of(:assert_pos).why)
        # A check keeps its open method: by its name, or because another method throws.
        @test of(:assert_role) == [("default", nothing, "keep (default)")]
        @test of(:refuse) == [("default", nothing, "keep (default)")]
        @test occursin("no violation", first_of(:refuse).why)
        # A verb that does a job owns no default.
        @test of(:fill_slot!) == [("default", nothing, "go")]
        # Two slots that return `nothing`: a typed method on the set that reaches it.
        @test of(:pair_op) == [("default", nothing, "go")]
        # An identity on two slots: a person rules it, and `RULED_DEFAULTS` holds the ruling.
        @test of(:leaf_pass) == [("default", nothing, "review")]
        @test of(:timestamp_positions) == [("default", nothing, "keep (default)")]
        @test first_of(:timestamp_positions).why ==
              FS.RULED_DEFAULTS[:timestamp_positions][2]
    end

    @testset "rebuild (#2082, rule 6)" begin
        @test of(:set_c) == [("rebuild", :R, "go")]
        # No keyword constructor for `rebuild_struct` to call.
        @test of(:set_s) == [("rebuild", :S, "review")]
        # The root has no declared type, so it can be another type.
        @test of(:set_u) == [("rebuild", :R, "review")]
        # One copy is no rebuild.
        @test all(x -> x[1] != "rebuild", of(:two))
    end

    @testset "self-forward (#2081, rule 5)" begin
        @test of(:sf) == [("self-forward", :sf, "go")]
        @test of(:pubf) == [("self-forward", :pubf, "keep (a)")]
        @test of(:privf) == [("self-forward", :privf, "go")]
        @test of(:pubk) == [("self-forward", :pubk, "keep (a)")]
        @test of(:fill1) == [("self-forward", :fill1, "go")]
        @test occursin("defaulted positional", first_of(:fill1).why)
        # A recursion on a field is a traversal.
        @test isempty(of(:walk))
    end

    @testset "worker and preset (#2080, #2079)" begin
        @test of(:verb) == [("worker", :verb_worker, "go")]
        @test of(:verb2) == [("worker", :hook_fn, "go")]
        @test occursin("around the driver", first_of(:verb2).why)
        @test of(:mk) == [("preset", :T2, "keep (a)")]
        # One private method that builds a type is a second name for the constructor: its
        # form goes to the constructor as a method (#2112).
        @test of(:mk_priv) == [("interface", :T3, "go")]
        @test occursin("#2112", first_of(:mk_priv).why)
        # Two methods that share the constructor are interface methods (rule 3).
        @test of(:mk_two) == [("interface", :T2, "keep (interface)"),
                              ("interface", :T2, "keep (interface)")]
        # A public verb that passes a keyword as a slot is a keyword form, not a worker.
        @test of(:kwv) == [("keyword form", :kwv_worker, "keep (a)")]
    end

    @testset "the rulings of the census" begin
        w = filter(x -> x.name === :w, found)
        ruled = only(x for x in w if x.callee === :only_a)
        pending = only(x for x in w if x.callee === :only_b)
        @test startswith(ruled.ruling, "ruled by `kept_forwards`")
        @test !FS.is_candidate(ruled)
        @test pending.ruling == "pending"
        @test FS.is_candidate(pending)
        @test startswith(first_of(:mk).ruling, "ruled by `kept`")
        @test !FS.is_candidate(first_of(:trait))
        @test !FS.is_candidate(first_of(:noop))
        @test FS.is_candidate(first_of(:leaf_pass))
    end

    @testset "the command line" begin
        @test status == 1
        @test occursin("src/a.jl:", out)
        @test occursin(r"\d+ candidate\(s\) without a ruling\.", out)
        x = first_of(:verb)
        @test FS.in_scope(x, ["src/a.jl"])
        @test FS.in_scope(x, ["src"])
        @test FS.in_scope(x, String[])
        @test !FS.in_scope(x, ["src/b.jl"])
        @test !FS.in_scope(x, ["sr"])
    end

    @testset "the survey reads the lists of test_70 from its parse" begin
        r = FS.read_rulings(joinpath(@__DIR__, "test_70_trivial_unit_census.jl"))
        @test haskey(r.kept, :OWA_GMD)
        @test !isempty(r.kept_forwards)
        # Every pair that the census holds is a finding of the survey: the two read forwards
        # with the one reader, `CodeHealth.forward_methods`.
        found_real = FS.survey()
        real = Set((x.name, x.callee) for x in found_real)
        @test [p for p in union(r.pending, keys(r.kept_forwards)) if !(p in real)] ==
              Tuple{Symbol, Symbol}[]
        # An entry of `RULED_DEFAULTS` whose name is no longer a `default` finding is stale:
        # the build ticket that typed or removed the fallback deletes it (#2108).
        defaults = Set(x.name for x in found_real if x.shape == "default")
        @test ("stale entries of RULED_DEFAULTS",
               sort!([n for n in keys(FS.RULED_DEFAULTS) if !(n in defaults)])) ==
              ("stale entries of RULED_DEFAULTS", Symbol[])
    end
end
