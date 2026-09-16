```@meta
Description = "The API reference of PortfolioOptimisers.jl: every documented name of the library, one page per source file, mirroring the src tree."
```

# API introduction

This is the API reference of `PortfolioOptimisers.jl`: every documented name of the library, on
one page per source file, so the pages are organised exactly as the `src` folder itself and a
documentation page corresponds one to one with a source file[^1]. Each source file's names are
split across a public page and a private page, the split being read off the source declarations
at build time (ADR 0128, `docs/adr/`). The public page holds the names the package exports or
declares `public`, which are the API a release keeps; the private page holds the internals,
which can change between releases. To browse by the job you want done rather than by the file
the code lives in, start from the [capability catalogue](@ref capability-catalogue).

[^1]: Except for a few cases, most of which are convenience function overloads. This means some links do not go to the exact method definition. Other than hard-coding links to specific lines of code, which is fragile, I haven't found an easy solution.

## Features

The inventory of everything the package can do — grouped by the job each thing
does rather than by the file it lives in — is the
[capability catalogue](@ref capability-catalogue). It is generated from the live
package, so it cannot fall behind the code.
