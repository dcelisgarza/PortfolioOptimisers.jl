---
mode: ask
description: Check API docs completeness.
---

Check API documentation completeness for PortfolioOptimisers.jl:

 1. For every public and private symbol with no exception (types, functions, macros, aliases) exported or defined in the codebase:
    
      + Verify that a Julia docstring is present for the symbol in the source code.
      + Verify that the docsstring is complete, accurate, and up to date.
      + Also verify the corresponding markdown file in `docs/src/api/` references the symbol.

 2. For each symbol with a docstring:
    
      + Check that the symbol is documented in the corresponding markdown file in `docs/src/api/` (e.g., `src/SomeFeature.jl` has corresponding `docs/src/api/SomeFeature.md`, `src/SomeFolder/AnotherFeature.jl` has corresponding `docs/src/api/SomeFolder/AnotherFeature.md`).
      + Confirm that the docstring content (or a summary of it) is included in the markdown file.
 3. Report:
    
      + Any symbols missing a docstring.
      + Any symbols with incomplete, inaccurate, or outdated docstrings.
      + Any symbols with a docstring that are not referenced in the corresponding markdown file.
      + Any markdown files that reference symbols not present in the codebase.

Use the established conventions and file structure of PortfolioOptimisers.jl. List results by file and symbol, with links to the relevant source and documentation locations.
