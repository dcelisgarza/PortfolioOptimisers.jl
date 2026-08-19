using PortfolioOptimisers, StableRNGs, LinearAlgebra, JET
BLAS.set_num_threads(1)
const PO = PortfolioOptimisers
const LBL = get(ENV, "PO_LBL", "?");
const SCHEME = get(ENV, "PO_CV", "Combinatorial")
rd = ReturnsResult(; nx = ["A$i" for i in 1:20],
                   X = randn(StableRNG(987654321), 1008, 20) ./ 100)
cv = if SCHEME == "KFold"
    KFold(; n = 10)
elseif SCHEME == "WalkForward"
    IndexWalkForward(127, 171)
elseif SCHEME == "Combinatorial"
    CombinatorialCrossValidation(; n_folds = 6, n_test_folds = 2, purged_size = 5,
                                 embargo_size = 3)
else
    MultipleRandomised(IndexWalkForward(127, 171); subset_size = 5, n_subsets = 3,
                       rng = StableRNG(666), seed = 69)
end
o = EqualWeighted()
rep = JET.report_opt(cross_val_predict, Tuple{typeof(o), typeof(rd), typeof(cv)};
                     target_modules = (PO,))
for r in JET.get_reports(rep)
    msg = sprint(io -> JET.print_report_message(IOContext(io, :color => false), r))
    kind = occursin("runtime dispatch", msg) ? "D" : occursin("recursion", msg) ? "R" : "O"
    sig = replace(string(r.sig), r"%\d+" => "%", r"\s+" => " ", r"#\d+" => "#")
    # callee = the first meaningful token of the signature
    m = match(r"Any\[:?\(?([A-Za-z_][\w\.!]*)", sig)
    callee = m === nothing ? "?" : m.captures[1]
    frame = isempty(r.vst) ? "?" : string(last(r.vst).linfo)
    fname = replace(replace(split(frame, "(")[1], "MethodInstance for " => ""),
                    r"#\d+" => "#")
    println(LBL, "|", SCHEME, "|", kind, "|", callee, "|", fname)
end
