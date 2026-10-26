# Toy Scheduler Simulator: Requirements

Oct 22, 2026 · @Jacob · Revised Oct 25, 2026: Universal Scalability Law speedup; capacity and footprint resources (see Design changes)

## Purpose and scope

This is a toy simulator for practicing the RL scheduler before it runs against DuckDB. It keeps the shape of the real problem. Jobs arrive as DAGs of tasks, the way queries arrive as DAGs of pipelines. Tasks compete for a fixed set of execution slots, and slow each other down when they compete for shared resources. There are no real threads, no SQL, and no database.

At fixed intervals, a learned policy decides what share of the slots each eligible task gets. A fixed algorithm turns the shares into an assignment of tasks to slots. Between decisions, the simulator consumes task progress, gives freed slots out by the policy's refill rule, and advances time.

The work is in two phases:

1. Phase 1: the environment, the FIFO policy, and the other baselines. The implementing agent builds this first, and nothing from Phase 2.
2. Phase 2: an encoder adapted from PQueryFormer and a Dirichlet PPO agent, trained and compared against the baselines.

All decisions are made. Deferred items are listed under Future work.

## Concepts

- Job: a DAG of tasks, with an insertion time (time\_inserted). It plays the role of a query. A job is in the system from its insertion time until its last task finishes.
- Task: a node in a job's DAG, and the unit that gets scheduled. It plays the role of a pipeline. A task can run on several slots at once.
- Slot: runs one task at a time. It stands in for a worker thread. The number of slots, S, is fixed.
- Capacity resource: something running tasks consume at a rate, such as memory bandwidth. There are N capacity resources, each with one global capacity shared by all slots. Load adds once per slot.
- Footprint resource: something running tasks occupy, such as a cache. There are K footprint resources. Load adds once per task, plus a smaller part per slot. When one fills up, the accesses it can no longer serve overflow into a capacity resource.
- Resource names such as memory bandwidth, storage I/O and cache are labels in the configuration. The simulator treats every capacity resource alike, and every footprint resource alike.
- Eligible task: a task whose dependencies have all finished and which has not finished itself. Running tasks are eligible.
- Allocation: a slot count n\_j for each eligible task, at most its maximum parallelism, summing to at most S. The simulator works only in allocations.
- Policy: decides allocations. It supplies two rules: allocate, called at each decision, and refill, called when slots free up between decisions.
- Placement algorithm: turns an allocation into a task on each slot, changing as few slots as possible.
- Share vector: the RL policy's output, a share of the slots for each eligible task. An adapter turns it into allocations. Only the RL policy uses it.

## Notation

Every symbol used in this document. Knobs are configurable; state values change during a run.

| Symbol | Meaning | Default |
| --- | --- | --- |
| S | Number of slots | 8 |
| N | Number of capacity resources | 2: memory bandwidth (r = 1) and storage I/O (r = 2) |
| C\_r | Capacity of capacity resource r, its maximum total rate of use | 1 |
| K | Number of footprint resources | 1: cache (c = 1) |
| B\_c | Capacity of footprint resource c, the total footprint it holds | 1 |
| o(c) | Capacity resource that footprint resource c overflows into, or none | Memory bandwidth (r = 1) |
| η\_c | Miss penalty of c: how many times longer an access takes when it misses | 3 |
| J | Jobs per episode | 100 |
| M | Maximum number of jobs visible to the policy | 20 |
| Δ | Decision interval, in time units | 2.5 |
| λ\_a | Arrival rate in streaming mode, in jobs per time unit | 0.1 |
| L | Number of levels in a job's DAG | Normal(μ\_L, σ\_L), rounded, clipped to \[L\_min, L\_max\] |
| μ\_L, σ\_L, L\_min, L\_max | Parameters of the height draw | 3, 1, 1, 6 |
| l | Level index, 1 at the top |  |
| T | Number of tasks in a job | Between L and 2^(L−1) |
| μ\_T, σ\_T | Mean and standard deviation of T, as fractions of the range \[L, 2^(L−1)\] | 0.5, 0.25 |
| λ\_L | Mean of the level draw, as a fraction of L | 0.3 |
| p\_e, γ | Base probability and distance decay for extra edges | 0.4, 1.5 |
| j, σ, r, c, k | Indices: task, slot, capacity resource, footprint resource, slot count |  |
| f\_{j,0} | Compute fraction of task j's time, never slowed by contention | 1 − Σ\_c f\_{j,c} − Σ\_r f\_{j,r} |
| f\_{j,c} | Fraction of task j's uncontended time spent on accesses served by footprint resource c | Dirichlet mixture |
| f\_{j,r} | Fraction of task j's uncontended time bound by capacity resource r | Dirichlet mixture |
| π | Probability that a task's breakdown comes from the heavy Dirichlet | 0.1 |
| β^light | Light Dirichlet parameters (compute, cache, memory bandwidth, storage I/O) | (5, 2, 2, 0.5) |
| β^heavy | Heavy Dirichlet parameters (compute, cache, memory bandwidth, storage I/O) | (2, 1, 1, 6) |
| ρ\_{j,r} | Rate at which one slot of task j uses r while bound by it, as a fraction of capacity | Lognormal around ρ\_r, clipped at 1 |
| ρ\_r | Median of ρ\_{j,r} | Memory 0.3, I/O 0.6 |
| σ\_ρ | Standard deviation of ln ρ\_{j,r} | 0.3 |
| a\_{j,r} | Average fraction of r's capacity task j uses on one slot at full speed | f\_{j,r} × ρ\_{j,r} |
| w^sh\_{j,c} | Shared footprint of task j on c, held once however many slots run it, as a fraction of B\_c | Lognormal with median w̃^sh |
| w^sl\_{j,c} | Per-slot footprint of task j on c, held once per slot, as a fraction of B\_c | Lognormal with median w̃^sl |
| w̃^sh, w̃^sl, σ\_w | Medians of w^sh and w^sl; standard deviation of their logarithms | 0.25, 0.02, 1 |
| F\_{j,c}(k) | Footprint of task j on c when it runs on k slots | w^sh\_{j,c} + k × w^sl\_{j,c} |
| W\_j | Work time of task j: time on one slot at full speed, without contention | Lognormal(μ\_W, σ\_W) |
| μ\_W, σ\_W | Mean and standard deviation of ln W\_j | ln 10, 0.75 |
| s\_j\[k\] | Per-slot speed of task j when it runs on k slots | Universal Scalability Law, from α\_j and κ\_j |
| α\_j | Serial fraction of task j (USL's σ, renamed because σ is the slot index) | Lognormal with median α̃, clipped to \[0, 1\] |
| α̃, σ\_α | Median of α\_j; standard deviation of ln α\_j | 0.02, 0.75 |
| κ\_j | Coherency cost of task j | (1 − α\_j) / (k^peak\_j)² |
| k^peak\_j | Slot count, treated as continuous, at which task j's total speed peaks | Lognormal with median m\_peak × S |
| m\_peak, σ\_peak | Median of k^peak\_j as a multiple of S; standard deviation of ln k^peak\_j | 4, 1 |
| k\*\_j | Maximum parallelism: the k that maximizes k × s\_j\[k\] | Derived |
| k\_j | Slots currently running task j | State |
| n\_j | Slots given to task j by the latest decision | State |
| g\_r | Rate factor of capacity resource r under contention | State |
| v\_σ | Current speed of slot σ | State |
| U\_r | Current load on capacity resource r | State |
| V\_c | Current load on footprint resource c: the total footprint of running tasks | State |
| h\_c | Hit fraction of c: the share of accesses to c that it serves | State, min(1, B\_c / V\_c) |
| f^hit\_j, f′\_{j,r}, a′\_{j,r} | Task j's effective breakdown and usage after cache misses (see Footprint resources) | State |
| R\_j | Progress rate of task j | State |
| Q | Number of jobs in the system | State |
| t, t\_dec | Current time; time of the next decision | State |
| p\_j | Share of the slots the policy gives task j | Action |
| r\_d | Reward for decision d | Phase 2 |

## Phase 1: Environment and baselines

### Job generation

Each job's DAG is built in four steps. Every distribution parameter is a configurable knob. The values given are defaults.

1. Height. Draw the number of levels L from a normal distribution with mean 3 and standard deviation 1. Round to the nearest integer and clip to \[1, 6\].
2. Task count. The number of tasks T must lie in \[L, 2^(L−1)\]. Let T\_min = L and T\_max = 2^(L−1). Draw T from a normal distribution with mean T\_min + μ\_T (T\_max − T\_min) and standard deviation σ\_T (T\_max − T\_min). Defaults are μ\_T = 0.5 and σ\_T = 0.25. Round to the nearest integer and clip to \[T\_min, T\_max\]. For L = 1 and L = 2 the range holds one value, so T = L.
3. Levels. Levels are numbered from the top, starting at 1. Level l holds at most 3^(l−1) tasks, so level 1 holds one task, the root. First assign one task to each level. Then, for each remaining task, draw x from an exponential distribution with mean λ\_L × L (λ\_L is a knob), and assign the task to level L − ⌊x⌋, counting up from the bottom. If that level is below 1 or already full, draw again. Extra tasks therefore land mostly near the leaves, where the caps are largest, so the DAG widens downward like a query plan. A free level always exists, since total capacity (3^L − 1)/2 is at least 2^(L−1).
4. Edges. An edge runs from a parent on a higher level (smaller l) to a child on a lower level (larger l). A parent depends on its children. Edges are added in two passes, adapted from your pseudocode:
   1. Connectivity. Give every task below level 1 a parent chosen uniformly from the level directly above. Then give every task above level L that still has no child a child chosen uniformly from all lower levels.
   2. Extra edges. For every pair of tasks u on level x and v on level y > x that are not already connected, add u → v with probability p\_e / (y − x)^γ. Defaults are p\_e = 0.4 and γ = 1.5.

No edge is added twice. Every task below level 1 has a parent one level up, so each DAG has a single root, is connected, and has depth exactly L. The connectivity pass gives every task above level L a child, so all leaves are on level L.

Reference version of the pseudocode, with levels\[0\] as level 1:

```python
import random

# levels = [[0], [1, 2], [3, 4, 5], ...]   task ids per level, top first
L = len(levels)
edges = set()
child_count = {t: 0 for lvl in levels for t in lvl}

def add_edge(parent, child):
    if (parent, child) not in edges:
        edges.add((parent, child))
        child_count[parent] += 1

# Pass 1: connectivity
for i in range(1, L):
    for v in levels[i]:
        add_edge(random.choice(levels[i - 1]), v)   # parent from the level directly above
for i in range(L - 1):
    lower = [t for lvl in levels[i + 1:] for t in lvl]
    for u in levels[i]:
        if child_count[u] == 0:
            add_edge(u, random.choice(lower))

# Pass 2: distance-decayed extra edges
p_e, gamma = 0.4, 1.5
for x in range(L):
    for y in range(x + 1, L):
        prob = p_e / (y - x) ** gamma
        for u in levels[x]:
            for v in levels[y]:
                if (u, v) not in edges and random.random() < prob:
                    add_edge(u, v)
```

After the shape is built, each task's attributes are drawn (next section), and the job's time\_inserted is set by the workload mode.

### Tasks

Each task j has six attributes, drawn independently for each task.

1. Time breakdown f\_{j,0}; f\_{j,c} for c = 1..K; and f\_{j,r} for r = 1..N, summing to 1. f\_{j,r} is the fraction of the task's uncontended running time that is bound by capacity resource r. f\_{j,c} is the fraction spent on accesses served by footprint resource c, at full speed when they hit (see Footprint resources). f\_{j,0} is the compute fraction: time bound only by the CPU, which contention never slows, since the slots are the cores. The breakdown is drawn from a mixture of two Dirichlet distributions over 1 + K + N components, ordered compute, footprint resources, capacity resources. With probability π it comes from a heavy Dirichlet(β^heavy), and otherwise from a light Dirichlet(β^light). By default the light one is mostly compute, cache and memory bandwidth, with I/O near 0, and the heavy one is I/O-bound, which gives the occasional polarizing task.
2. Capacity usage a\_{j,r} = f\_{j,r} × ρ\_{j,r}, for r ≥ 1. ρ\_{j,r} is the rate at which one slot uses capacity resource r while it is bound by r, including while it serves misses from a footprint resource that overflows into r, as a fraction of the global capacity. It is drawn per task, with ln ρ\_{j,r} \~ Normal(ln ρ\_r, σ\_ρ), and clipped to at most 1. a\_{j,r} is then the task's average use of r on one slot at full speed, as a fraction of capacity.
3. Work time W\_j, in time units: the time the task takes on one slot at full speed with no contention. It is drawn from a lognormal distribution, ln W\_j \~ Normal(μ\_W, σ\_W). This is always positive and has a long right tail, so most tasks are short and a few are very long.
4. Speedup array s\_j. Entry s\_j\[k\] is the speed of each slot when the task runs on k slots, relative to one slot. The task's total speed on k slots, before contention, is k × s\_j\[k\]. The array follows the Universal Scalability Law (below), from two per-task parameters: the serial fraction α\_j and the coherency cost κ\_j.
5. Maximum parallelism k\*\_j: the k that maximizes k × s\_j\[k\], taking the smallest k on ties. Past k\*\_j, another slot lowers the task's total speed. The scheduler never gives a task more than k\*\_j slots. Per-slot speed never reaches 0, but past k\*\_j total speed falls. With no coherency cost, total speed never falls and k\*\_j = S.
6. Footprint w^sh\_{j,c} and w^sl\_{j,c} on each footprint resource c, each drawn from a lognormal distribution with median w̃^sh or w̃^sl and log standard deviation σ\_w (see Footprint resources).

### Why usage is derived from the time breakdown

A task's average use of a resource is the fraction of its time spent on that resource, times the rate at which it uses the resource during that time. So a = f × ρ. This builds in the correlation we want: a task that uses a resource heavily spends much of its time on it, and so depends more on it.

Because ρ varies from task to task, the correlation is strong but not exact. This matches real pipelines. A streaming scan pulls memory bandwidth at a high rate and stalls on memory often. A hash probe pulls less bandwidth, but still stalls often, because it waits on memory latency.

The reverse, drawing a first and setting f = a / ρ, was rejected. One core can only use part of the machine's capacity, roughly a quarter of socket bandwidth, so a must stay below ρ. Heavy draws would then need clipping, which distorts exactly the polarizing tasks.

Calibration: with all S slots busy and no cache misses, U\_r ≈ S × E\[f\_r\] × E\[ρ\_{j,r}\]. The defaults give E\[f\_cache\] ≈ 0.20, E\[f\_mem\] ≈ 0.20 and E\[f\_io\] ≈ 0.11. The memory share of 0.40 under the original defaults is split evenly between cache-served and streaming accesses. With ρ\_mem = 0.3 and ρ\_io = 0.6, memory bandwidth sits near half capacity (U ≈ 0.5) while everything hits, and storage I/O sits near half capacity too (U ≈ 0.5), except when heavy I/O tasks run together. Cache misses push memory bandwidth up from there (see Footprint resources).

Each task also tracks its state: remaining work (starting at W\_j), status (waiting, eligible, running, or finished), its current slots, and the time it became eligible.

### Speedup: Universal Scalability Law

Revised Oct 25, 2026. This replaces the five speedup families (see Design changes).

Each task's per-slot speed follows the Universal Scalability Law (USL), which is Amdahl's law plus a coherency term:

```latex
s_j[k] = \frac{1}{1 + \alpha_j (k - 1) + \kappa_j\, k (k - 1)}, \qquad X_j(k) = k\, s_j[k]
```

X\_j(k) is the task's total speed on k slots, before contention.

- α\_j is the serial fraction: the part of the task that does not parallelize, such as a single-threaded finalize. With κ\_j = 0 this is exactly Amdahl's law, X(k) = 1 / (α + (1 − α)/k). USL usually calls this σ, but σ is the slot index here.
- κ\_j is the coherency cost. Every pair of slots pays a little to stay in sync, for example on shared hash tables or when merging partial results. The cost grows like k², so total speed eventually peaks and then falls.
- The curve covers only these synchronization losses. Memory bandwidth and storage I/O saturation are charged only by the resource models (Capacity resources and contention, Footprint resources), so the same loss is not counted twice.

Peak and maximum parallelism. Treating k as continuous, X′(k) has the sign of 1 − α − κk², so total speed peaks at k^peak = √((1 − α)/κ). k\*\_j is still the integer k ≤ S that maximizes k × s\_j\[k\]: whichever of ⌊k^peak⌋ and ⌈k^peak⌉ gives more, clipped to \[1, S\]. With κ = 0 there is no peak and k\* = S.

At an interior peak the per-slot speed is about one half. Substituting κ (k^peak)² = 1 − α makes the denominator about 2. A smooth curve can only peak where the last slot adds nothing, so by then the earlier slots are already well below full speed. Tasks that peak before S therefore run inefficiently on k\* slots, and the distribution of k^peak decides how many such tasks there are. The coherency term also costs speed well before the peak: at half the peak, per-slot speed is already down by about 20%.

Sampling. Each task draws α\_j from a lognormal distribution with median α̃ = 0.02 and log standard deviation σ\_α = 0.75, clipped to \[0, 1\]. It then draws k^peak\_j from a lognormal distribution with median m\_peak × S = 4S and log standard deviation σ\_peak = 1, and sets κ\_j = (1 − α\_j) / (k^peak\_j)². The peak is drawn instead of κ because it is easier to reason about: it is where adding a slot stops helping. With the defaults, about 8% of tasks peak before S (Φ(−ln 4)), so a few tasks top out early, and the median task uses all S slots at a per-slot speed near 0.8. Setting α̃ = 0 gives α\_j = 0, and setting m\_peak to none gives κ\_j = 0. Both together give perfect scaling, which the sanity check uses. All of these are knobs.

Examples, on 8 slots: α = 0.03 and κ = 0 give s\[8\] = 1/1.21 ≈ 0.83, a total speedup of 6.6×, and k\* = 8. α = 0.03 and κ = 0.02 peak at k^peak ≈ 6.96, so k\* = 7 and s\[7\] ≈ 0.50.

### Dependencies

A task becomes eligible once all of its children have finished. Leaves become eligible as soon as their job is inserted. The root depends, directly or indirectly, on every other task, so a job finishes when its root finishes.

### Capacity resources and contention

There are N capacity resources, by default two: memory bandwidth (r = 1) and storage I/O (r = 2). Each has one global capacity C\_r = 1, shared by all S slots. There are no resource containers, because the policy cannot direct tasks to particular containers. Usage is derived from each task's time breakdown. With all slots busy and no cache misses, the defaults put both near half capacity. Cache misses raise memory bandwidth from there (see Tasks and Footprint resources).

The contention computation runs every time the assignment of tasks to slots changes. It follows Amdahl's law. A slot spends fraction f\_{j,r} of its time bound by resource r, so when r runs slower, only that fraction of its time stretches.

Let g\_r ∈ (0, 1\] be the rate factor of resource r: the fraction of its normal rate that each task gets from it. A slot σ running task j on k\_j slots has speed:

```latex
v_\sigma = \frac{s_j[k_j]}{f_{j,0} + f^{hit}_j + \sum_{r=1}^{N} f'_{j,r} / g_r}
```

Its use of capacity resource r is a′\_{j,r} v\_σ, so the load on r is:

```latex
U_r = \sum_{\sigma \,\text{occupied}} a'_{j(\sigma),r}\, v_\sigma
```

f^hit\_j, f′\_{j,r} and a′\_{j,r} are the task's effective breakdown and usage after the footprint step (see Footprint resources). With no footprint resources, or when every running task's footprint fits, f^hit\_j = Σ\_c f\_{j,c}, f′ = f and a′ = a. The formulas then reduce to time bound by r stretching by 1/g\_r, and everything else running at full speed.

The rate factors are the largest values with U\_r ≤ C\_r, with equality wherever g\_r < 1. They are found by fixed-point iteration:

1. Set g\_r = 1 for every resource.
2. Compute every v\_σ and U\_r from the current factors.
3. Update g\_r ← min(1, g\_r × C\_r / U\_r) for every resource.
4. Repeat steps 2 and 3 until no g\_r changes by more than a tolerance, or an iteration cap is reached.

Each task then progresses at R\_j = Σ v\_σ over its slots, which equals k\_j v\_σ, since all its slots run at the same speed.

Why iterate: a slot only slows on the fraction of its time spent on r, so its use of r falls by less than g\_r. One pass of g\_r = C\_r / U\_r would leave the resource oversubscribed. Lowering one resource's factor also lowers the load on the others, and the update raises a factor again when its resource has room, so all resources are solved together.

Consequences:

- With no contention, every g\_r = 1 and v\_σ = s\_j\[k\_j\], so a task takes W\_j time units on one slot, as defined.
- Each task is slowed by how much of its time depends on the oversubscribed resource. Example: with memory at half rate (g\_mem = 0.5), a task with f\_mem = 0.8 runs at 1 / (0.2 + 1.6) = 0.56 of its speed, and a task with f\_mem = 0.2 runs at 1 / (0.8 + 0.4) = 0.83.
- A pure-compute task (f\_{j,0} = 1) is never slowed.
- Using a′\_{j,r} v\_σ as the load is consistent with the time breakdown. Per unit of progress, a task needs f\_{j,r} ρ\_{j,r} of resource r however fast it runs, so its rate of use is a\_{j,r} times its speed.

### Footprint resources

Added Oct 25, 2026 (see Design changes).

A footprint resource is something running tasks occupy rather than consume at a rate, such as a shared cache. There are K of them, by default one, labelled cache. Each has a capacity B\_c = 1. What matters is how much of it the running tasks hold, not how fast they run, since a slower task holds the same footprint. So a footprint resource cannot be throttled through a rate factor the way a capacity resource is. When it fills up, it instead changes what the tasks do.

Footprint. Task j running on k slots holds F\_{j,c}(k) = w^sh\_{j,c} + k × w^sl\_{j,c} of resource c.

- The shared part w^sh is held once per task, however many slots run it. An example is one hash table probed by every thread of a pipeline.
- The per-slot part w^sl is held once per slot. Examples are each thread's local aggregation table or sort buffer. It keeps concentrating slots on one task from being free.

Both are drawn per task, as fractions of B\_c, from lognormal distributions with medians w̃^sh = 0.25 and w̃^sl = 0.02 and log standard deviation σ\_w = 1. A task holds no footprint while it has no slots: its data is assumed evicted when it is preempted.

Load and hit fraction. The load on c is V\_c = Σ F\_{j,c}(k\_j) over running tasks. If it fits, every access to c hits. If not, each task keeps a share of c in proportion to its footprint, so every task hits on the same fraction of its accesses, h\_c = min(1, B\_c / V\_c).

Overflow. Each footprint resource overflows into one capacity resource o(c), by default memory bandwidth. A task spends fraction f\_{j,c} of its uncontended time on accesses served by c.

- Hits run at full speed, and contention never slows them, as with compute.
- Misses take η\_c times as long (the miss penalty, default 3). They are bound by o(c): they wait on it, and they use it at the task's own rate ρ\_{j,o(c)}.
- o(c) may be none. Misses then only cost the η\_c latency and add no load anywhere.

Effective breakdown. The footprint step rewrites each running task's breakdown and usage:

```latex
f^{hit}_j = \sum_{c=1}^{K} f_{j,c}\, h_c, \qquad
f'_{j,r} = f_{j,r} + \sum_{c \,:\, o(c) = r} \eta_c\, f_{j,c} (1 - h_c), \qquad
a'_{j,r} = f'_{j,r}\, \rho_{j,r}
```

The contention computation then runs exactly as in Capacity resources and contention, with these in place of f and a. When misses happen, f′ no longer sums to 1. The task's uncontended time per unit of work grows by Σ\_c (η\_c − 1) f\_{j,c} (1 − h\_c), and the missed accesses add load to o(c).

Order. h\_c depends only on which tasks run and on how many slots each has, not on their speeds. It is computed once each time the assignment changes, before the fixed point, and the fixed point itself does not change.

Why this gives the behavior we want. Concentrating slots on few tasks keeps V\_c small, so tasks hit and memory bandwidth has headroom. Mixing many tasks overflows the cache, so misses add memory traffic and bandwidth saturates. Mixing tasks with complementary capacity profiles still helps. The policy therefore faces a real trade-off between spreading capacity load and keeping working sets in the cache. Without footprint resources, mixing had the benefit and no cost.

Calibration. These values are first guesses. With median footprints:

- One task on all 8 slots holds 0.25 + 8 × 0.02 = 0.41, and fits.
- Four tasks on 2 slots each hold about 1.16, so h ≈ 0.86.
- Eight tasks on one slot each hold about 2.2, so h ≈ 0.46. With mean rather than median footprints they hold about 3.5, so h ≈ 0.3, close to the 0.34 measured for equal share in batch mode.

With all slots busy, memory bandwidth sits near U ≈ 0.5 while everything hits. Each 10-point drop in h adds about 0.15 (S × η × E\[f\_cache\] × E\[ρ\_mem\] × (1 − h) ≈ 1.5 (1 − h)), so memory bandwidth saturates once h falls below about 2/3.

Simplifications, acceptable for a toy:

- Proportional sharing ignores reuse: a streaming scan with little reuse still takes its share of the cache and pushes out others' data.
- The miss fraction grows linearly with overflow. Real miss rates fall off smoothly with cache size.

### Scheduling decisions

Decisions happen at fixed intervals of Δ time units, starting at t = 0. Events never trigger a decision. Δ is a knob.

The simulator works in allocations: a slot count n\_j for each eligible task, with n\_j ≤ k\*\_j and the counts summing to at most S. Every policy, learned or baseline, supplies two rules:

- allocate: called at each decision. It returns an allocation, which placement then applies.
- refill: called whenever slots are free between decisions, after a task finishes or when a job arrives while slots are idle. It decides which eligible tasks get the free slots, up to their maximum parallelism. Refill only assigns free slots, so it never moves or preempts a running task.

A task that loses all its slots at a decision is preempted. It keeps its remaining work and continues when it gets slots again. Moving a task costs nothing for now. A warm-up penalty is future work.

### RL adapter

The RL policy outputs a share vector p over the eligible tasks, non-negative and summing to 1. An adapter turns it into the two rules. The share vector exists only here: the simulator and the baselines never see it.

allocate:

1. Counts. Multiply each share by S and round down. Then hand out the remaining slots one at a time, to the tasks with the largest fractional parts (largest-remainder rounding).
2. Caps. Cut any task above its maximum parallelism k\* down to k\*. Give the freed slots to uncapped tasks with a positive share, in proportion to their shares, again by largest-remainder rounding. Repeat until no task is over its cap, or no uncapped task with a positive share remains.

refill:

1. Recompute target counts with allocate's two steps, using p restricted to the tasks that were eligible at the last decision and are still unfinished, renormalized over those tasks. Give the free slots to tasks below their target, largest shortfall first, breaking ties by larger share. Tasks above their target keep their slots. This keeps the policy's intent in force between decisions.
2. FIFO fallback. Once every remaining task from the last decision is at its target, or all of them have finished, refill reverts to FIFO. Any slots still free go to eligible tasks below their maximum parallelism, in FIFO order: by the job's time\_inserted, then job id, then the time the task became eligible, then task id. This covers tasks that became eligible after the last decision. It happens often when Δ is large, because the tasks from the last decision are likely to finish well before the next one.

Gap logging, for the RL policy only: at each decision, log the L1 distance between p and n/S. Over each interval, log the fraction of slot-time assigned by allocate, by refill step 1, and by the FIFO fallback.

### Placement

Given the previous assignment and the new counts n\_j:

1. Each task j keeps min(n\_j, its previous count) of the slots it already holds.
2. All other slots are released.
3. Released and empty slots go to tasks that still need slots, n\_j minus what they kept.

This changes the fewest slots possible. Each task that grows must receive n\_j minus its previous count in new slots, so the total number of changes, the sum of max(0, n\_j − previous count) over all tasks, is a lower bound, and this algorithm meets it.

Without containers, slots are interchangeable. So which slot a task gets only affects results if moving costs something, which is future work. The environment still logs the number of slot changes at each decision.

### Simulation loop

Time jumps from event to event. Speeds only change when the slot assignment changes, so between events every task progresses at a constant rate and the next event time is exact. There are three kinds of event: a task finishing, at t + remaining\_j / R\_j; a job arriving; and the next decision, at t\_dec. Events within a small tolerance of each other count as simultaneous. At one instant, completions are handled first, then arrivals, then the decision.

1. Find the earliest event time t′.
2. Advance. Reduce each running task's remaining work by R\_j (t′ − t). Add Q (t′ − t) to the latency integral, where Q is the number of jobs in the system. Set t = t′.
3. Handle every event at t′:
   - Task finished: mark it finished and free its slots. Any parent whose children have now all finished becomes eligible. If the task was a root, its job finishes; record the completion time.
   - Job arrived: insert it. Its leaves become eligible.
   - Decision: call the policy's allocate rule (for the RL policy, this is the Gymnasium step boundary), apply the allocation through placement, and set t\_dec = t\_dec + Δ.
4. If a slot is free and an eligible task is below its maximum parallelism, call the policy's refill rule.
5. If the assignment changed, recompute the footprint loads and hit fractions, then rerun the contention computation.

### Workload

Each job's time\_inserted is its arrival time. There are two modes:

- Batch: every job has time\_inserted = 0.
- Streaming: jobs arrive as a Poisson process, with exponential inter-arrival times at rate λ\_a (a knob).

Batch is built and tested first. An episode is a fixed number J of jobs (a knob) in both modes, and ends when all J have finished.

### Gymnasium interface

The environment is a Gymnasium environment. reset(seed) generates the workload from a seeded random generator and returns the observation at t = 0. step(action) applies the share vector at the current decision, simulates until the next decision or the end of the episode, and returns the next observation, the reward, and whether the episode is over.

Observation. Every feature is visible and exact. This is unrealistic, but the point here is to see what the scheduling decisions look like.

- Per task: time breakdown, capacity usage, footprint (shared and per-slot) on each footprint resource, work time, remaining work, speedup array, serial fraction α\_j, coherency cost κ\_j, maximum parallelism, level, status, current slot count, and current per-slot speed.
- Per job: time\_inserted, time in the system, tasks finished, and the DAG's edges.
- Global: current time, free slots, the current load U\_r on each capacity resource, and the current load V\_c and hit fraction h\_c of each footprint resource.

Variable sizes are padded to fixed maxima, with masks. Under the default height cap, a job has at most 2^5 = 32 tasks. At most M jobs (a knob) are visible at once. If more are in the system, the M oldest are visible. The rest wait, and still count in Q.

Action. A vector with one entry per padded task position, each in \[0, 1\]. The environment ignores entries for padding and for tasks that are not eligible, and renormalizes the rest into the share vector p. The RL adapter turns p into allocate and refill rules. If every eligible entry is 0, the shares are taken as equal. The Gymnasium environment wraps the simulator together with the RL adapter. The baselines call the simulator directly.

Reward. Defined under Phase 2. Phase 1 must compute the quantities it needs: the integral of Q over each interval, and each job's completion time.

Each episode logs makespan, mean and 90th percentile job latency (completion time minus time\_inserted), slot utilization, the fraction of time each capacity resource is oversubscribed, the time-average hit fraction of each footprint resource and the fraction of time it overflows, slot changes per decision, and, for the RL policy, the gap between its shares and its allocations (see RL adapter).

### Baselines

Each baseline supplies allocate and refill directly, in slot counts. None uses a share vector.

| Baseline | allocate | refill | Data it needs |
| --- | --- | --- | --- |
| FIFO | Sort eligible tasks by the job's time\_inserted, then job id, then the time the task became eligible, then task id. Give each task slots up to k\*, in that order, until slots run out. | Same order, applied to the free slots. A freed slot takes the next task in line, like DuckDB's default scheduler. | time\_inserted |
| Equal share | Split S evenly over the eligible tasks with largest-remainder rounding, capped at k\*, with the excess spread over uncapped tasks. | Split the free slots evenly over eligible tasks below their maximum parallelism. | Number of eligible tasks |
| Shortest job first (SJF) | As FIFO, ordered by the job's total work at insertion (the sum of its W\_j), smallest first. | Same order. | Each job's total work |
| Shortest remaining processing time (SRPT) | As FIFO, ordered by the job's remaining work (the sum of remaining work over its tasks), smallest first. | Same order, re-sorted at each refill. | Each task's remaining work |

The order-based baselines take newly eligible tasks into their refill immediately. The RL policy only sees them at its next decision. This slightly favors the baselines, which keeps any gain the policy shows conservative. A smaller Δ shrinks the difference.

Sanity check: use height 1 (single-task jobs), perfect scaling (α̃ = 0 and m\_peak set to none, so α\_j = κ\_j = 0 and k\* = S), N = 0 capacity resources, and K = 0 footprint resources, so there is no contention. Then the machine behaves like one server of speed S, and SRPT minimizes mean job latency. In batch mode, SRPT's refill keeps it exact at every completion. In streaming mode, a short job that arrives mid-interval can only preempt at the next decision, so use a small Δ there. In Phase 2, the trained policy should approach SRPT in this configuration.

### Configuration

Every knob in the Notation table lives in one configuration object, with the defaults listed there. For example, a dataclass that can also be loaded from a YAML file. Any knob can be overridden for a single run without changing code. Each run saves its full configuration and seed with its results, so it can be reproduced.

## Phase 2: Encoder and RL agent

### Encoder

The encoder is adapted from PQueryFormer, the two-level QueryFormer variant built for the DuckDB scheduler. You will link the code. Tasks have no internal operators, so PQueryFormer's first level, which encodes the operators inside a pipeline, is removed. Only the second level remains. It runs over each job's task DAG, with every task feature as the node input: time breakdown, capacity usage, footprint, work time, remaining work, speedup array, maximum parallelism, level, status, slot count and speed. Task level serves as the height encoding, and DAG distance sets the attention bias. As in the DuckDB scheduler, cross-job attention and a global node follow. Adding subtasks, and with them the first level, is future work.

### Policy

The policy samples the share vector from a Dirichlet distribution over the eligible tasks, and is trained with PPO.

### Reward

What the two papers do:

- LSched penalizes latency at each scheduling decision. For decision d, latency is approximated as Q\_d (t\_d − t\_{d−1}), where Q\_d is the number of queries in the system. Summed over an episode, this is the total time queries spent in the system, so it targets average latency. A second term uses the 90th percentile of these values over the episode to penalize the tail. The two terms are weighted 0.5 each.
- BQSched minimizes the makespan of a batch. The reward is minus the makespan, given once at the end of the episode, with no per-step rewards. To offset that sparse signal, its IQ-PPO algorithm adds an auxiliary head that predicts the finish time of the next query to finish (MSE loss). It trains that head every 10 PPO iterations, with a KL term that keeps the policy from drifting.

Proposal for the toy model. Decisions now come every Δ, and jobs arrive and finish inside an interval, so Q is not constant between decisions. Use the exact integral, which the event loop already computes:

```latex
r_d = -\frac{1}{\Delta} \int_{t_d}^{t_d + \Delta} Q(t)\, dt
```

This is the average number of jobs in the system over the interval. Summed over an episode, Δ times the return equals minus the total job latency, so maximizing return minimizes mean job latency. It is LSched's average term, made exact for fixed intervals. Dividing by Δ keeps the reward's scale independent of the interval.

Options to add later, each a small change:

- LSched's tail term, for 90th percentile latency.
- A makespan reward for batch workloads, as in BQSched. Spread over intervals, it is minus the fraction of each interval in which any job remained. It sums to minus the makespan divided by Δ, but every action gets the same reward until the end, so it gives weak credit assignment.
- The IQ-PPO auxiliary head, predicting the time until the next job finishes.

### Evaluation

Run every policy on the same seeded workloads. Report mean and 90th percentile job latency and makespan over several seeds, in batch and streaming modes. Run the SRPT sanity check first. This will be revisited.

## Design changes

### Oct 25, 2026: speedup families replaced by the Universal Scalability Law

What changed. The five speedup families (linear, logarithmic, exponential cliff, power, ideal) and their knobs (w, z, z\_min, z\_max, b, q) are removed. Each task now has a serial fraction α\_j and a coherency cost κ\_j, and its speedup array follows the USL (see Speedup: Universal Scalability Law). The ideal family is replaced by α = κ = 0. The observation gains α\_j and κ\_j.

Why. The Phase 1 baselines showed the old curves dominating the results:

- The curves were steep. On 2 slots, average per-slot speed was already 0.85. 43% of tasks had k\* ≤ 3, and per-slot speed at k\* averaged 0.67.
- Order-based baselines give each task up to k\* slots, so they paid for this in full. With no resources, FIFO and SRPT did 0.70 units of work per busy slot-time, against 0.96 for equal share. Equal share then beat SRPT on mean latency even in streaming mode, where ordering should matter most.
- The curves double-counted memory bandwidth. Much of a real pipeline's lost scaling is bandwidth saturation, which the contention model already charges. The curve now covers only synchronization: the serial part and coherency.

Effect, streaming mode, mean job latency over seeds 0 to 4 (FIFO / equal share / SRPT):

| Setup | Old families | USL |
| --- | --- | --- |
| No resources | 49.5 / 32.4 / 35.2 | 30.2 / 20.4 / 20.1 |
| With contention (default) | 70.7 / 35.4 / 44.0 | 81.5 / 31.4 / 44.4 |

Without resources, SRPT now matches equal share. With contention on, equal share still leads. The order-based baselines put a single task on all S slots, which copies that task's resource profile onto every slot. That is the contention model working as designed, but its calibration (memory sits near capacity whenever all slots are busy) deserves a separate look.

Default peak. A median peak of 2S was tried first. It left the median task at a per-slot speed of 0.70 at k\*, because a USL curve has already lost about 20% per slot at half its peak. A median of 4S, with σ\_peak = 1, gives 0.80, and about 8% of tasks peak before S.

### Oct 25, 2026: footprint resources, and resources split into two types

Status: implemented Oct 25, 2026. In the configuration, the knobs are capacity\_resources, capacity\_limits, footprint\_resources, footprint\_limits, footprint\_overflow, miss\_penalty, footprint\_shared\_median, footprint\_slot\_median and footprint\_log\_std.

What changed.

- The original resources are now called capacity resources. Nothing about how they work changes.
- A second type is added, footprint resources (see Footprint resources). Their load counts once per task, plus a per-slot part. When one fills up, the missed accesses take η\_c times longer and become load on a capacity resource it overflows into. By default there is one, labelled cache, overflowing into memory bandwidth.
- The time breakdown gains one component per footprint resource. The Dirichlet parameters go from (compute, memory, I/O) = (5, 4, 0.5) and (2, 2, 6) to (compute, cache, memory, I/O) = (5, 2, 2, 0.5) and (2, 1, 1, 6). This splits the old memory share evenly between cache-served and streaming accesses.
- Notation: the Dirichlet parameters are renamed from c^light and c^heavy to β^light and β^heavy, so that c can index footprint resources.
- Resource names are labels in the configuration. The simulator does not hard-code memory, cache or I/O.

Why. After the USL change, equal share still beat SRPT by a wide margin in streaming mode with contention on (31.4 vs 44.4 mean latency). Without resources the two were even (20.4 vs 20.1). The order-based baselines put one task on all S slots, which copies that task's capacity profile onto every slot. The model rewarded mixing tasks with complementary profiles but charged nothing for it. Real hardware does: threads of one pipeline share a working set in cache, while different pipelines compete for it. DuckDB's FIFO concentrates pipelines, which the original model gave no credit for.

Alternative rejected. Adding a penalty to U\_r that grows with the number of distinct running tasks would give similar behavior. It has no physical mechanism behind it, though, and could not be calibrated against real cache sizes or footprints.

Effect. Mean job latency over seeds 0 to 4 (FIFO / equal share / SJF / SRPT):

| Mode | Without footprint resources (USL only) | With footprint resources |
| --- | --- | --- |
| Batch | 529.0 / 375.6 / 336.3 / 336.8 | 541.7 / 514.2 / 344.8 / 345.5 |
| Streaming | 81.5 / 31.4 / 47.9 / 44.4 | 66.3 / 62.4 / 39.6 / 35.4 |

- Ordering now matters as expected. SRPT and SJF lead on latency in both modes, and equal share falls to the level of FIFO. Equal share still has the shortest batch makespan (1008 against 1081 to 1091), but by much less than before (740 against about 1070).
- The order-based baselines keep their working sets in cache: their time-average hit fraction is 0.92 to 0.93, and memory bandwidth is oversubscribed only 6% of the time. Equal share mixes many tasks: its hit fraction is 0.34 in batch and 0.60 in streaming, and memory bandwidth is oversubscribed 75% and 45% of the time.
- The best policy is now somewhere in between. As a check (not a baseline), SRPT with each task capped at 4 slots reached 294.8 in batch and 26.5 in streaming, better than every baseline. A cap of 2 overflows the cache (hit fraction 0.71 in streaming) and is worse again in streaming (37.7). The RL policy has a real trade-off to learn: spread capacity load, but keep working sets in cache.

Calibration note. Measured hit fractions are lower than the worked examples in Footprint resources suggest, because those examples use median footprints. Footprints are lognormal with σ\_w = 1, so their mean is e^0.5 ≈ 1.65 times the median, and eight one-slot tasks typically hold about 3.5 rather than 2.2.

### Oct 25, 2026: clarifications made while implementing Phase 1

- The limit of M visible jobs applies only to the RL environment. Under the RL adapter, tasks of invisible jobs never get slots, including from the FIFO fallback. Baselines see every job.
- Decisions with no eligible visible task, which happen only when the system is empty in streaming mode, are applied automatically and are not environment steps. Q is 0 over those intervals. Q-time from before the first step is reported in reset's info.
- Refill step 1 gives free slots out one at a time to the task with the largest remaining shortfall, ties to the larger share.
- Order-based baselines break ties in their sort key with the FIFO order.
- A capacity resource counts as oversubscribed when g\_r < 1, that is, when demand on it would exceed its capacity.
- Makespan is the last completion time minus the earliest insertion time. Slot utilization is busy slot-time divided by S × makespan.
- In streaming mode, the first job arrives after one exponential gap, not at t = 0.
- Until Phase 2 defines the reward, the environment returns 0 and reports the integral of Q over each step in the step's info.

## Future work

- Memory capacity as a footprint resource. Overflow means spilling to disk, so it would overflow into storage I/O. It probably needs a sharper penalty than the proportional miss model gives.
- Buffer pool as a footprint resource overflowing into storage I/O. This is less simple, since it depends on reuse across tasks and on the eviction policy, and less impactful for now.
- Better cache models: miss-rate curves instead of a linear miss fraction, cache pollution by streaming tasks with little reuse, and a warm-up cost after preemption while a task refetches its working set.
- Migration warm-up. A slot that switches tasks runs at reduced speed for a short warm-up. This gives the minimum-change placement an effect on results.
- Adaptive decision interval. Adjust Δ during training instead of fixing it.
- Subtasks. Give tasks internal operators, called subtasks, and restore PQueryFormer's first level to encode them.
- Speedup tied to the task's kind. Since contention now carries the bandwidth loss, the old idea of steeper curves for memory-bound tasks mostly falls away. α\_j or κ\_j could still depend on what the task does, for example a larger κ for tasks that build shared hash tables.
- Serial part as its own phase. The per-slot Amdahl form slows all k slots for the whole task. In DuckDB the serial part is a separate phase, and idle threads move on to other pipelines. Modelling it as a phase with parallelism 1 would free those slots.
- Calibration against DuckDB. Measure real pipelines in isolation at 1 to S threads, and fit α, κ and resource usage from the measurements.
- Reward additions: LSched's tail term, a makespan reward, and the IQ-PPO auxiliary head (see Reward).
- Evaluation, to be revisited.
