```@meta
Description = "Report writers, public API of PortfolioOptimisers.jl: write_report, Base.show."
```

# Report writers

```@docs
write_report
Base.show(io::IO, ::MIME"text/markdown", b::ReportText)
Base.show(io::IO, ::MIME"text/html", b::ReportText)
```
