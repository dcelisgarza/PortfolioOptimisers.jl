---
agent: agent
description: Check API docs completeness.
---

Check API documentation completeness for PortfolioOptimisers.jl.

Two checks are already gated. Run them, and report what they fail on:

- `test/test_26_docs.jl` — every public and private name carries a docstring.
- `test/test_65_docs_public_private_placement_census.jl` — every `@docs` entry sits on the side of the mirrored API tree that its classification puts it on.

No gate reads the rest. For each symbol, types, functions, macros and aliases, public and private:

1. Check that the docstring is complete, accurate, and current against the code it documents.
2. Check that the mirror page lists the symbol in an `@docs` block. `./src/SomeFeature.jl` maps to `./docs/src/public_api/SomeFeature.md` and `./docs/src/private_api/SomeFeature.md`, and `./src/SomeFolder/AnotherFeature.jl` maps to `./docs/src/public_api/SomeFolder/AnotherFeature.md` and `./docs/src/private_api/SomeFolder/AnotherFeature.md`. The page holds the entry, not a copy of the docstring.
3. Check that every `@docs` entry names a symbol the codebase still defines.

List results by file and symbol, with links to the relevant source and documentation locations.
