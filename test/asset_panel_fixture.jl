#=
One helper, shared by every test that needs an `AssetPanel` built from a bare matrix.

`#803` made the Asset Panel the one carrier of feature data, and `#807` deleted
`feature_matrix_panel`, the verb that used to put a bare matrix back on a carrier: the collapse
returns a panel and a producer builds its tensor field directly, so the inverse lost its last
reader. A *test* still wants the shorthand — it is checking something else, and a panel of one
numeric Panel Field per column is the shortest panel whose derived Feature Matrix is the matrix
it was handed.

This is deliberately not in `src/`. A caller with a bare matrix authors the Panel Fields they
mean, named after what the columns are; a test that only needs *a* panel says so here.
=#

"""
    matrix_panel(nz, Z::AbstractMatrix) -> AssetPanel
    matrix_panel(nz, Z::AbstractArray{<:Any, 3}) -> AssetPanel

Build the `AssetPanel` whose derived Feature Matrix is `Z`, with `nz` naming its columns: one
`NumericPanelField` per column, under that column's own name. A static `assets × features` matrix
builds a static panel, and a time-varying `observations × assets × features` array builds a
time-varying one with all-`true` universe masks.
"""
function matrix_panel(nz, Z::AbstractMatrix)
    return asset_panel([NumericPanelInput(; name = String(n), vals = Z[:, k])
                        for (k, n) in pairs(nz)])
end
function matrix_panel(nz, Z::AbstractArray{<:Any, 3})
    return asset_panel([NumericPanelInput(; name = String(n), vals = Z[:, :, k])
                        for (k, n) in pairs(nz)])
end
