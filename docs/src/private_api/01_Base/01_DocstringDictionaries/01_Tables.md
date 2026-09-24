```@meta
Description = "Docstring dictionaries, private API of PortfolioOptimisers.jl: arg_dict, val_dict, ret_dict, field_dict, math_dict, err_name_dict, ref_dict, …"
```

# Docstring dictionaries: private API

The files under [`src/01_Base/`](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/src/01_Base) hold the code that the rest of `PortfolioOptimisers.jl` depends on. Each file has its own page in this section, and this page documents the first.

The docstrings of the library take the text of an argument, a field, a validation rule, a return value, a formula, an error name or a citation from the dictionaries below. A name that many functions share therefore reads the same on every page that documents it.

This file declares the seven tables and the guard that fills them. Each later file of [`src/01_Base/01_DocstringDictionaries/`](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/src/01_Base/01_DocstringDictionaries) fills one table for one subject, and its page names the table and the subject.

```@docs
arg_dict
val_dict
ret_dict
field_dict
math_dict
err_name_dict
ref_dict
unique_key_dict!
```
