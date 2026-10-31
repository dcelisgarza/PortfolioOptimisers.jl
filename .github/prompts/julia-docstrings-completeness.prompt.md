---
agent: agent
description: Check API docs completeness.
---

Check API documentation completeness for PortfolioOptimisers.jl.

Two gates already hold part of the work. Run them, and report what they fail on:

- `test/test_26_docs.jl` checks that every public and private name carries a docstring.
- `test/test_65_docs_public_private_placement_census.jl` checks that every `@docs` entry is on the side of the mirrored API tree that its classification gives it.

No gate reads the rest. Do these checks for each public and private symbol, whether it is a type, a function, a macro or an alias:

1. Check that the docstring is complete, accurate, and current against the code it documents.
2. Check that the mirror page lists the symbol in an `@docs` block. `./src/SomeFeature.jl` maps to `./docs/src/public_api/SomeFeature.md` and `./docs/src/private_api/SomeFeature.md`, and `./src/SomeFolder/AnotherFeature.jl` maps to `./docs/src/public_api/SomeFolder/AnotherFeature.md` and `./docs/src/private_api/SomeFolder/AnotherFeature.md`. The page holds the entry, not a copy of the docstring.
3. Check that every `@docs` entry names a symbol the codebase still defines.

List results by file and symbol, with links to the relevant source and documentation locations.
