@safetestset "Docs completeness" begin
    using PortfolioOptimisers, Test
    all_symbols = names(PortfolioOptimisers; all = true)
    filter!(x -> !contains(string(x), r"#|^eval$|^include$"), all_symbols)
    no_docs = Symbol[]
    for sym in all_symbols
        docstr = string(Base.Docs.doc(getfield(PortfolioOptimisers, sym)))
        if isempty(docstr) || contains(docstr,
                                       r"No documentation found for (?:public|private) binding \`PortfolioOptimisers\.")
            push!(no_docs, sym)
        end
    end
    @test length(no_docs) == 378
end
