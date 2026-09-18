# The public/private classification of a documented binding (ADR 0128 § Decision,
# `docs/adr/0128-the-docs-split-into-mirrored-public-and-private-api-trees-and-a-census-gates-placement.md`).
#
# The source declares what is public — `export` and `public` in `src/` and `ext/` — and the
# docs must only ever derive that boundary, never restate it. A name `PortfolioOptimisers`
# owns classifies by `Base.isexported`/`Base.ispublic`; a name naming a method the package
# adds to a `Base` or `StatsAPI` generic it does not own classifies always-public,
# unconditionally, because extending a foreign generic on a library type is, by
# construction, part of that type's public interface. A page may spell such a name qualified
# (`StatsAPI.fit`) or bare (`fit`); both classify identically.
#
# This is the one function the mirror-page generator (once the per-directory migration
# lands) and the placement census both call, so the rule is written once. It assumes
# `PortfolioOptimisers` is already loaded into the caller, because the classification reads
# the live module.

# The foreign-owned class ADR 0128 names, keyed by the bare method name: a page may spell the
# name qualified or bare, and both resolve to the same classification, so the set holds only
# the bare form.
const FOREIGN_PUBLIC_BINDINGS = Set([:iterate, :getproperty, :getindex, :propertynames,
                                     :showerror, :split, :fit, :predict])

"""
    classify_binding(name) -> Symbol

Whether `name` — a binding `PortfolioOptimisers` documents, bare or module-qualified, with
any call signature already stripped (`docs_block_names` in `docs/page_metadata.jl` does
this) — is `:public` or `:private` (ADR 0128 § Decision). A qualification is dropped before
classification, so `Base.iterate` and `iterate` classify identically.
"""
function classify_binding(name::AbstractString)
    bare = Symbol(last(split(name, '.')))
    if bare in FOREIGN_PUBLIC_BINDINGS
        return :public
    end
    if !(isdefined(PortfolioOptimisers, bare))
        return :private
    end
    is_public = Base.isexported(PortfolioOptimisers, bare) ||
                Base.ispublic(PortfolioOptimisers, bare)
    return is_public ? :public : :private
end
