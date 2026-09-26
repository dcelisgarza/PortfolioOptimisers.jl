```@meta
Description = "The API reference of PortfolioOptimisers.jl, with a public page and a private page for each source file of the library."
```

# API introduction

This is the API reference of `PortfolioOptimisers.jl`. It documents every name of the library that
has a docstring, and its pages follow the layout of the `src` folder, one source file to one pair
of pages[^1]. The public page of a source file lists the names that the package exports or declares
`public`, and they change only at a `v0.X.0` release. The private page lists the internal names,
which can change at any release. The docs build reads the split from the declarations in the source.

[^1]: A few names are exceptions, most of them convenience methods of a function. A link to one of these methods can go to the function rather than to the exact method definition. I have found no fix other than a link to a fixed line of code, and such a link breaks when the code moves.

## Capability catalogue

The [capability catalogue](@ref capability-catalogue) lists everything the package can do. It
groups the parts by the job each one does, not by the file that holds its code. That grouping is
written by hand. An entry with no label of its own takes its one-line description from the first
sentence of its docstring. The docs build and a test both fail when the package adds a type you
can choose that the catalogue does not list.
