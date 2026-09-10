@testset "Partial-fit cache census: an estimator method narrows the cache it reads" begin
    using Test

    #=
    Issue #997. `Online` seeds a `SampleBufferState` into the `cache` field of the estimator
    it wraps, and the state's type is the whole route: a buffer means the batch verb over
    the buffer's rows, and a family state means that family's exact fold. Dispatch is what
    carries the route, so every `partial_fit!` method whose first argument is an estimator
    must narrow the `cache` type parameter of that estimator to the state it reads. A method
    that does not narrow wins over the generic buffering one and reads a buffer as its own
    state, which is the `MethodError` #997 reported.

    The narrowing is positional -- `cache` is the last type parameter of every family, and a
    signature names it by counting `<:Any` up to it. Julia leaves trailing parameters free
    on a partial instantiation, so a field added ahead of `cache` shifts every position and
    the narrowing silently binds the new field instead, leaving `cache` unbound. Nothing
    about that is a conflict a merge would surface: the field lives in the struct's file and
    the signature in the fold's. This census is what makes it a red build.

    It reads dispatch rather than source text, because the breach is a position rather than
    a spelling, and only the resolved signature knows which position `cache` is at.
    =#

    po = PortfolioOptimisers

    # Every `partial_fit!` method whose first argument is a concrete estimator carrying a
    # `cache` field. A state-first method carries no `cache`, and the generic buffering
    # method's first argument is a `Union` rather than a struct, so both fall out here.
    checked = Tuple{Method, Any, Int}[]
    for m in methods(partial_fit!)
        sig = Base.unwrap_unionall(m.sig)
        length(sig.parameters) >= 2 || continue
        T = sig.parameters[2]
        B = Base.unwrap_unionall(T)
        isa(B, DataType) && isstructtype(B) || continue
        fns = fieldnames(B)
        :cache in fns || continue
        i = findfirst(==(:cache), fns)
        # A `@concrete` struct records one type parameter per field, in field order, so the
        # `cache` field's position is the `cache` parameter's position. A struct that broke
        # that would make every narrowing below meaningless, so it is asserted, not assumed.
        @test length(B.parameters) == fieldcount(B)
        push!(checked, (m, B, i))
    end

    # A census that matches nothing passes vacuously. Eleven families declare two methods
    # each, and three of them declare a refusal beside the fold.
    @test length(checked) >= 22

    @testset "$(Base.typename(B).name) at $(basename(String(m.file))):$(m.line)" for (m, B,
                                                                                      i) in
                                                                                     checked

        p = B.parameters[i]
        bound = isa(p, TypeVar) ? p.ub : p
        # The method reads a family state, so it must admit one, and `nothing` beside it:
        # a fold seeds its own state on the first call.
        @test bound <: po.Option{<:po.AbstractPartialFitState}
        # And it must refuse a buffer, so a wrapped estimator falls through to the generic
        # buffering method rather than reading a buffer as its own state.
        @test !(po.SampleBufferState <: bound)
    end
end
