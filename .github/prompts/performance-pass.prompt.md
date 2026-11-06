---
agent: agent
description: Make a hot path of PortfolioOptimisers.jl faster with bit-identical results, one measured part at a time.
---

A **performance pass** makes one hot path faster and keeps every output bit-identical. The rules
it applies are in [`.github/instructions/julia-source-code.instructions.md`](../instructions/julia-source-code.instructions.md)
§ *A hot path is measured, and its results stay bit-identical*. This file is the order of the
work. Issue #1973 is the worked case: in five parts, the default fit of a
`CrossSectionalFactorPrior` went from 2.44 s to 1.26 s and its fastest configuration from 1.91 s
to 0.85 s. Each part kept 211 probe cases bit-identical.

Run every Julia process single-threaded, with `julia -t 1` and `BLAS.set_num_threads(1)`, in a
fresh process. `CLAUDE.md` § *Running Julia* owns the rest of how to run Julia.

## Step 1. Fix the case, the base and the probe

1. Write the **case**: the sizes and the configuration that the issue names, for example
   `T = 500, N = 1600`, one fit. Write the configurations that the hot path branches on, for
   example each solve algorithm of the regression.
2. Record the **base**: the commit before the pass. Keep a detached worktree of it, so that the
   base and each part can run in the same session.
3. Write the **probe**: a script that runs every configuration that the change can reach on small
   data, and serializes every output. Add the cases that mix element types (`Float32` data with
   `Float64` weights, and the inverse) and the cases that refuse, recorded as their error text.
   The probe has two modes:

   ```julia
   # julia -t 1 --project=<worktree> probe.jl save|check <file>
   out = Dict(name => attempt(() -> summary(prior(pe, rd))) for (name, pe, rd) in cases)
   mode == "save" ? serialize(path, out) :
   println(isempty([k for (k, v) in deserialize(path) if !isequal(v, out[k])]) ?
           "BIT-IDENTICAL $(length(out))" : "DIFF")
   ```

   `isequal` separates `0.0` from `-0.0` and compares `NaN` as equal, which is the bit-level
   comparison that the rule needs. When a part changes the shape of an output, the probe
   permutes the new output back to the old shape before it compares.

The step is done when `probe.jl save` on the base has written the file, and `probe.jl check` on
the base prints `BIT-IDENTICAL`.

## Step 2. Measure the base

1. Write the **timing driver**: it warms up on small data, then runs each configuration five times
   on the case, with `GC.gc(true)` before each run, and prints the median of `@timed`.
2. Measure the allocations of one fit with `@allocated`.
3. If the issue compares with another library, time that library in the same session, on the
   same case.

The step is done when one table holds the median time and the allocations of each configuration
at the base, and the time of the other library when the issue compares with one.

## Step 3. Profile, then plan the parts

1. Profile the time and the allocations of one fit, with `Profile` and `Profile.Allocs` at a
   sample rate that keeps the run under a minute.
2. Rank the stages by their share of the time and of the allocated bytes.
3. Write one **part** for each change. A part changes one mechanism: one stage stops copying, one
   loop takes its buffers once for each task, two histories share a backing, a block is computed
   once, or an array changes its layout. A layout change of an array that users build is a
   breaking change, so `src/CLAUDE.md` decides how the part names it.

Issue #1973 found its savings in this order, which is a fair prior for a similar fit. The times come
from several sessions on a shared machine, so read the order and the size of each step, not the
exact values:

| Part | Mechanism | Time of the default fit |
| --- | --- | --- |
| base | | 2.72 s |
| A | the prior stage reads views of the exposure history, not copies | 1.89 s |
| B | the regression loop gathers into buffers of each task | 1.76 s |
| C | the exposure history and the observed exposures share one backing | 1.56 s |
| D | the lift states each block once | 1.37 s |
| E | the arrays of rank three of the factor model hold their observations last (ADR 0230) | 1.26 s |

The step is done when each part names its mechanism, the stage that the profile ranks for it, and
the files that it changes.

## Step 4. Do each part

Do the parts one at a time, in their own commit:

1. Make the change. Give a fast case its own method, so the complexity gate stays green.
2. Run `probe.jl check`. When it reports a difference, find the cases, the largest relative change and
   its cause before you go on. The bit-identical rule of the instructions decides whether the
   cause is accepted and where the part states it.
3. Run the timing driver on the base and on the part in one session, the base first and last, and
   compare the medians.
4. Run the targeted tests with coverage, as `CLAUDE.md` § *Running Julia* states.
5. Commit the part, with its timing and the result of the probe in the message.

The step is done when every part is committed, the probe prints `BIT-IDENTICAL` or a stated
bound on each one, and each part has a measured gain. Drop a part that gains nothing.

## Step 5. Land and report

1. Land the branch as the `sweep-conform` skill § *Land the branch* states.
2. Post the table of the pass on the issue: the median time of each configuration at the base
   and after the pass, the change in percent, and the allocations. A time of another library
   stays in the report of the session, not on the tracker.
3. File an issue for each layout or hot path that the pass found and left, with its measurement.

The step is done when the issue holds the table and every finding that the pass left has an issue.

## A map of performance passes

A wayfinder map applies this pass to the library after its files are swept. The map lists the hot
paths of one subject, ranked by a measurement of the case that each one serves. Each child ticket
is one performance pass on one hot path, with its case, its base and its probe written in the
ticket before the work starts. A ticket whose profile shows no stage above a few percent closes
with its measurement and no change.
